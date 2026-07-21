#!/usr/bin/env python3
"""Prepare data streams, LOSO annotations, and config paths for the paper datasets.

This single entry point supports:

- WEAR
- RealWorld-HAR waist accelerometer (``rwhar``)
- WISDM smartwatch accelerometer (``wisdm_watch``)

The script deliberately preserves model/training hyperparameters in existing YAML
files.  It only prepares/normalizes data paths, LOSO annotation lists, sampling
rates, window lengths, input dimensions, class counts, and log subdirectories.

Examples
--------
WEAR (prepared native 50 Hz ``sbj_*.csv`` files)::

    python utils/tools/prepare_dataset.py \
      --dataset wear \
      --source /path/to/wear/native_50hz \
      --repo-root . \
      --steps all \
      --overwrite

RealWorld-HAR (raw ``proband1..proband15`` tree or already processed
``probandX_activity.csv`` files)::

    python utils/tools/prepare_dataset.py \
      --dataset rwhar \
      --source /path/to/realworld/raw_root \
      --repo-root . \
      --steps all \
      --overwrite

WISDM-watch (prepared ``sbj_*.csv`` files or raw CSVs containing
``subject_id, activity, timestamp, acc_x, acc_y, acc_z``)::

    python utils/tools/prepare_dataset.py \
      --dataset wisdm_watch \
      --source /path/to/wisdm_watch_csvs \
      --repo-root . \
      --steps all \
      --overwrite

All three in one invocation::

    python utils/tools/prepare_dataset.py \
      --dataset all \
      --wear-source /path/to/wear/native_50hz \
      --rwhar-source /path/to/realworld \
      --wisdm-source /path/to/wisdm_watch \
      --repo-root . \
      --steps all \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    config_name: str
    native_rate: int
    rates: Tuple[int, ...]
    sensor_root: str
    annotation_root: str
    has_null: bool
    expected_subjects: int
    default_label_candidates: Tuple[str, ...]


SPECS: Mapping[str, DatasetSpec] = {
    "wear": DatasetSpec(
        key="wear",
        config_name="wear",
        native_rate=50,
        rates=(50, 25, 12, 6),
        sensor_root="data/wear",
        annotation_root="data/wear/annotations",
        has_null=True,
        expected_subjects=22,
        default_label_candidates=("label", "in/out", "locomotion", "coarse"),
    ),
    "rwhar": DatasetSpec(
        key="rwhar",
        config_name="rwhar",
        native_rate=50,
        rates=(50, 25, 12, 6),
        sensor_root="data/rwhar",
        # Kept compatible with the paper-release configs.
        annotation_root="data/realworld/annotations/50hz/waist_accel",
        has_null=False,
        expected_subjects=15,
        default_label_candidates=("label",),
    ),
    "wisdm_watch": DatasetSpec(
        key="wisdm_watch",
        config_name="wisdm_watch",
        native_rate=20,
        rates=(20, 10, 5),
        sensor_root="data/wisdm_watch",
        annotation_root="data/wisdm/annotations/Multirate/watch",
        has_null=False,
        expected_subjects=51,
        default_label_candidates=("label", "activity"),
    ),
}

RWHAR_ACTIVITIES: Tuple[str, ...] = (
    "walking",
    "running",
    "sitting",
    "standing",
    "lying",
    "jumping",
    "climbingup",
    "climbingdown",
)

RWHAR_FILE_RE = re.compile(r"^proband(\d+)_([A-Za-z0-9_]+)\.csv$")
SBJ_MAIN_RE = re.compile(r"^sbj_(\d+)$")
SBJ_ANY_RE = re.compile(r"^sbj_(\d+)(?:_.+)?$")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _resolve(repo_root: Path, path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (repo_root / p).resolve()


def _repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _safe_remove(path: Path, overwrite: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not overwrite:
        raise FileExistsError(f"Destination already exists: {path}. Use --overwrite.")
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _link_or_copy_dir(source: Path, destination: Path, mode: str, overwrite: bool) -> None:
    if source.resolve() == destination.resolve():
        return

    _safe_remove(destination, overwrite=overwrite)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if mode == "symlink":
        destination.symlink_to(source.resolve(), target_is_directory=True)
    elif mode == "copy":
        shutil.copytree(source, destination)
    else:
        raise ValueError(f"Unsupported native mode: {mode}")


def _csv_files(folder: Path) -> List[Path]:
    return sorted(p for p in folder.glob("*.csv") if p.is_file())


def _read_columns(csv_path: Path) -> List[str]:
    return list(pd.read_csv(csv_path, nrows=0).columns)


def _detect_label_column(csv_path: Path, candidates: Sequence[str], explicit: str) -> str:
    columns = _read_columns(csv_path)
    if explicit != "auto":
        if explicit not in columns:
            raise ValueError(
                f"Requested label column '{explicit}' not found in {csv_path}. "
                f"Columns: {columns}"
            )
        return explicit

    for candidate in candidates:
        if candidate in columns:
            return candidate

    if columns and str(columns[-1]).lower() in {"label", "activity", "class", "target"}:
        return str(columns[-1])

    raise ValueError(
        f"Could not detect a label column in {csv_path}. Columns: {columns}. "
        "Pass --label-col explicitly."
    )


def _validate_training_csv(csv_path: Path, label_col: str) -> int:
    columns = _read_columns(csv_path)
    if label_col not in columns:
        raise ValueError(f"'{label_col}' not found in {csv_path}")
    if len(columns) < 3:
        raise ValueError(
            f"Training CSV must contain an ID column, at least one feature column, "
            f"and the label column: {csv_path} has {columns}"
        )
    if columns[-1] != label_col:
        raise ValueError(
            f"The trainer expects the label to be the final CSV column. "
            f"In {csv_path}, label column '{label_col}' is at position "
            f"{columns.index(label_col)} of {len(columns)}."
        )
    return len(columns) - 2


def _downsample_csv(source: Path, destination: Path, factor: int, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        return
    df = pd.read_csv(source, low_memory=False)
    out = df.iloc[::factor].reset_index(drop=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(destination, index=False)


def _downsample_folder(source: Path, destination: Path, factor: int, overwrite: bool) -> None:
    files = _csv_files(source)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {source}")

    if destination.exists() or destination.is_symlink():
        if overwrite:
            _safe_remove(destination, overwrite=True)
        else:
            existing = _csv_files(destination) if destination.is_dir() else []
            if existing:
                print(f"[SKIP] {destination}: already contains {len(existing)} CSV files")
                return
            raise FileExistsError(f"Destination exists but is unusable: {destination}")

    destination.mkdir(parents=True, exist_ok=True)
    for index, src in enumerate(files, start=1):
        dst = destination / src.name
        _downsample_csv(src, dst, factor=factor, overwrite=True)
        if index == 1 or index % 20 == 0 or index == len(files):
            print(f"  [{index:3d}/{len(files):3d}] {src.name}")


def _rate_folder(spec: DatasetSpec, repo_root: Path, rate: int) -> Path:
    return _resolve(repo_root, f"{spec.sensor_root}/{rate}hz")


def _multirate_folder(spec: DatasetSpec, repo_root: Path) -> Path:
    return _resolve(repo_root, f"{spec.sensor_root}/multirate")


def _annotation_folder(spec: DatasetSpec, repo_root: Path, rate_key: str) -> Path:
    if spec.key == "wear":
        name = "Multirate" if rate_key == "multirate" else f"{int(rate_key)}Hz"
        return _resolve(repo_root, f"{spec.annotation_root}/{name}")
    return _resolve(repo_root, spec.annotation_root)


def _nominal_factor(native_rate: int, target_rate: int) -> int:
    ratio = native_rate / target_rate
    factor = int(round(ratio))
    if factor < 1:
        raise ValueError(f"Cannot downsample {native_rate} Hz to {target_rate} Hz")
    effective = native_rate / factor
    if abs(effective - target_rate) > 0.75:
        raise ValueError(
            f"Target rate {target_rate} Hz cannot be represented by integer decimation "
            f"from {native_rate} Hz (nearest effective rate {effective:.3f} Hz)."
        )
    if abs(effective - target_rate) > 1e-6:
        print(
            f"[INFO] Nominal {target_rate} Hz uses stride {factor}; "
            f"effective rate is {effective:.3f} Hz."
        )
    return factor


# ---------------------------------------------------------------------------
# RealWorld-HAR preprocessing
# ---------------------------------------------------------------------------


def _standardize_rwhar_acc(df: pd.DataFrame) -> pd.DataFrame:
    required = {"attr_time", "attr_x", "attr_y", "attr_z"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"RealWorld-HAR accelerometer CSV is missing {sorted(missing)}")
    return df.rename(
        columns={"attr_time": "t", "attr_x": "ax", "attr_y": "ay", "attr_z": "az"}
    )[["t", "ax", "ay", "az"]]


def _rwhar_segment_token(activity: str, path: str) -> str:
    base = os.path.basename(path)
    file_match = re.match(
        rf"^acc_{re.escape(activity)}(?:_(\d+))?_waist\.csv$", base, re.IGNORECASE
    )
    if file_match and file_match.group(1):
        return file_match.group(1)

    zip_match = re.match(
        rf"^acc_{re.escape(activity)}(?:_(\d+))?_csv\.zip$", base, re.IGNORECASE
    )
    if zip_match and zip_match.group(1):
        return zip_match.group(1)

    for part in reversed(path.replace("\\", "/").split("/")):
        directory_match = re.match(
            rf"acc_{re.escape(activity)}_(\d+)_csv", part, re.IGNORECASE
        )
        if directory_match:
            return directory_match.group(1)
    return ""


def _read_rwhar_waist_segments(data_dir: Path, activity: str) -> Dict[str, pd.DataFrame]:
    root = data_dir / f"acc_{activity}_csv"
    fallback_zip = data_dir / f"acc_{activity}_csv.zip"
    if not root.is_dir() and not fallback_zip.is_file():
        raise FileNotFoundError(f"No accelerometer data for {activity} under {data_dir}")

    scan_root = root if root.is_dir() else data_dir
    file_re = re.compile(
        rf"^acc_{re.escape(activity)}(?:_(\d+))?_waist\.csv$", re.IGNORECASE
    )
    zip_re = re.compile(
        rf"^acc_{re.escape(activity)}(?:_(\d+))?_csv\.zip$", re.IGNORECASE
    )
    collected: Dict[str, List[pd.DataFrame]] = {}

    for dirpath, _, filenames in os.walk(scan_root):
        for filename in filenames:
            if file_re.match(filename):
                path = Path(dirpath) / filename
                token = _rwhar_segment_token(activity, str(path))
                collected.setdefault(token, []).append(pd.read_csv(path))

    for dirpath, _, filenames in os.walk(scan_root):
        for filename in filenames:
            if not zip_re.match(filename):
                continue
            zip_path = Path(dirpath) / filename
            zip_token = _rwhar_segment_token(activity, str(zip_path))
            with zipfile.ZipFile(zip_path, "r") as archive:
                for member in archive.namelist():
                    base = os.path.basename(member)
                    if file_re.match(base):
                        token = _rwhar_segment_token(activity, base) or zip_token
                        with archive.open(member, "r") as handle:
                            collected.setdefault(token, []).append(pd.read_csv(handle))

    if not collected:
        raise FileNotFoundError(f"No waist accelerometer CSVs found for {activity}")

    return {
        token: pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        for token, parts in collected.items()
    }


def _preprocess_rwhar_raw(raw_root: Path, destination: Path, overwrite: bool) -> Path:
    proband_dirs = sorted(
        (p for p in raw_root.iterdir() if p.is_dir() and re.fullmatch(r"proband\d+", p.name)),
        key=lambda p: int(p.name.replace("proband", "")),
    )
    if not proband_dirs:
        raise FileNotFoundError(f"No proband1..proband15 folders found under {raw_root}")

    if destination.exists() or destination.is_symlink():
        if overwrite:
            _safe_remove(destination, overwrite=True)
        elif _csv_files(destination):
            print(f"[SKIP] Using existing processed RealWorld-HAR folder: {destination}")
            return destination
        else:
            raise FileExistsError(f"Destination exists but is empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    for proband in proband_dirs:
        pid = int(proband.name.replace("proband", ""))
        data_dir = proband / "data"
        if not data_dir.is_dir():
            print(f"[WARN] {proband.name}: missing data/ folder")
            continue

        for activity in RWHAR_ACTIVITIES:
            output_path = destination / f"proband{pid}_{activity}.csv"
            try:
                segments = _read_rwhar_waist_segments(data_dir, activity)
            except FileNotFoundError as exc:
                print(f"[WARN] {proband.name} {activity}: {exc}")
                continue

            ordered_parts: List[pd.DataFrame] = []
            for token in sorted(segments, key=lambda value: (value == "", value)):
                ordered_parts.append(_standardize_rwhar_acc(segments[token]).sort_values("t"))
            acc = pd.concat(ordered_parts, ignore_index=True)

            out = pd.DataFrame(
                {
                    "id": pid,
                    "acc_x": acc["ax"].astype("float32"),
                    "acc_y": acc["ay"].astype("float32"),
                    "acc_z": acc["az"].astype("float32"),
                    "label": activity,
                }
            )
            out.to_csv(output_path, index=False)
            print(f"[OK] {output_path} rows={len(out)}")

    return destination


# ---------------------------------------------------------------------------
# WISDM preprocessing
# ---------------------------------------------------------------------------


def _looks_like_prepared_subject_folder(source: Path) -> bool:
    return any(SBJ_ANY_RE.fullmatch(p.stem) for p in _csv_files(source))


def _prepare_wisdm_raw(source: Path, destination: Path, overwrite: bool) -> Path:
    files = _csv_files(source)
    if not files:
        raise FileNotFoundError(f"No CSV files found in {source}")

    required = {"subject_id", "activity", "timestamp", "acc_x", "acc_y", "acc_z"}
    subject_frames: Dict[int, List[pd.DataFrame]] = {}

    for file in files:
        columns = set(_read_columns(file))
        if not required.issubset(columns):
            raise ValueError(
                f"Raw WISDM CSV {file} does not contain required columns "
                f"{sorted(required)}. Found: {sorted(columns)}"
            )
        frame = pd.read_csv(file, low_memory=False)
        for subject_id, group in frame.groupby("subject_id", sort=False):
            subject_frames.setdefault(int(subject_id), []).append(group.copy())

    if destination.exists() or destination.is_symlink():
        if overwrite:
            _safe_remove(destination, overwrite=True)
        elif _csv_files(destination):
            print(f"[SKIP] Using existing prepared WISDM folder: {destination}")
            return destination
        else:
            raise FileExistsError(f"Destination exists but is empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "source": source.resolve().as_posix(),
        "subject_mapping": {},
        "files": [],
    }

    for index, subject_id in enumerate(sorted(subject_frames)):
        frame = pd.concat(subject_frames[subject_id], ignore_index=True)
        frame = frame.sort_values("timestamp", kind="mergesort")
        out = pd.DataFrame(
            {
                "subject": int(subject_id),
                "acc_x": frame["acc_x"].astype(float),
                "acc_y": frame["acc_y"].astype(float),
                "acc_z": frame["acc_z"].astype(float),
                "label": frame["activity"].astype(str),
            }
        )
        output_path = destination / f"sbj_{index}.csv"
        out.to_csv(output_path, index=False)
        manifest["subject_mapping"][str(subject_id)] = f"sbj_{index}"
        manifest["files"].append(
            {
                "subject_id": int(subject_id),
                "output": output_path.name,
                "rows": int(len(out)),
            }
        )
        print(f"[OK] {output_path} subject_id={subject_id} rows={len(out)}")

    with open(destination / "manifest_wisdm_watch.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return destination


# ---------------------------------------------------------------------------
# Stream preparation
# ---------------------------------------------------------------------------


def _prepare_native_source(
    spec: DatasetSpec,
    source: Path,
    repo_root: Path,
    native_mode: str,
    overwrite: bool,
) -> Path:
    native_destination = _rate_folder(spec, repo_root, spec.native_rate)

    if spec.key == "rwhar" and any(
        p.is_dir() and re.fullmatch(r"proband\d+", p.name) for p in source.iterdir()
    ):
        # Raw RealWorld-HAR needs conversion; output is necessarily a real folder.
        return _preprocess_rwhar_raw(source, native_destination, overwrite=overwrite)

    if spec.key == "wisdm_watch" and not _looks_like_prepared_subject_folder(source):
        return _prepare_wisdm_raw(source, native_destination, overwrite=overwrite)

    files = _csv_files(source)
    if not files:
        raise FileNotFoundError(f"No CSV files found in source folder: {source}")

    _link_or_copy_dir(source, native_destination, mode=native_mode, overwrite=overwrite)
    return native_destination


def prepare_streams(
    spec: DatasetSpec,
    source: Path,
    repo_root: Path,
    native_mode: str,
    overwrite: bool,
) -> Path:
    print(f"\n=== Preparing {spec.key} data streams ===")
    if not source.is_dir():
        raise FileNotFoundError(f"Source folder does not exist: {source}")

    native_folder = _prepare_native_source(
        spec=spec,
        source=source,
        repo_root=repo_root,
        native_mode=native_mode,
        overwrite=overwrite,
    )

    native_files = _csv_files(native_folder)
    if not native_files:
        raise FileNotFoundError(f"No native CSV files found in {native_folder}")

    label_col = _detect_label_column(
        native_files[0], spec.default_label_candidates, explicit="auto"
    )
    input_dim = _validate_training_csv(native_files[0], label_col=label_col)
    print(
        f"Native folder: {native_folder} | files={len(native_files)} | "
        f"label={label_col} | input_dim={input_dim}"
    )

    for rate in spec.rates:
        if rate == spec.native_rate:
            continue
        factor = _nominal_factor(spec.native_rate, rate)
        destination = _rate_folder(spec, repo_root, rate)
        print(f"Downsampling {spec.native_rate} Hz -> {rate} Hz (stride={factor})")
        _downsample_folder(native_folder, destination, factor=factor, overwrite=overwrite)

    multirate = _multirate_folder(spec, repo_root)
    _link_or_copy_dir(native_folder, multirate, mode="symlink", overwrite=overwrite)
    print(f"Multirate source: {multirate} -> {native_folder}")
    return native_folder


# ---------------------------------------------------------------------------
# LOSO annotation generation
# ---------------------------------------------------------------------------


def _collect_label_order(
    csv_paths: Sequence[Path], label_col: str, has_null: bool, null_label: str
) -> List[str]:
    labels: set[str] = set()
    for path in csv_paths:
        frame = pd.read_csv(path, usecols=[label_col], low_memory=False)
        labels.update(str(value) for value in frame[label_col].dropna().unique())

    ordered = sorted(labels)
    if has_null:
        ordered = [label for label in ordered if label.lower() != null_label.lower()]
    return ordered


def _segments_from_labels(
    csv_path: Path,
    label_col: str,
    label_to_id: Mapping[str, int],
    sampling_rate: float,
    null_label: str,
) -> List[Dict[str, object]]:
    frame = pd.read_csv(csv_path, usecols=[label_col], low_memory=False)
    values = frame[label_col].tolist()
    if not values:
        return []

    annotations: List[Dict[str, object]] = []
    current = values[0]
    start = 0

    for index in range(1, len(values) + 1):
        boundary = index == len(values) or values[index] != current
        if not boundary:
            continue

        if not pd.isna(current):
            label = str(current)
            # Null/background is not an action segment. It is inserted as class 0
            # by scripts/run_loso.py when has_null=True.
            if label.lower() != null_label.lower() and label in label_to_id:
                annotations.append(
                    {
                        "segment": [float(start / sampling_rate), float(index / sampling_rate)],
                        "label": label,
                        "label_id": int(label_to_id[label]),
                    }
                )

        if index < len(values):
            current = values[index]
            start = index

    return annotations


def _subject_stems(folder: Path) -> Tuple[List[str], List[str]]:
    stems = sorted(path.stem for path in _csv_files(folder) if SBJ_ANY_RE.fullmatch(path.stem))
    main = sorted(
        (stem for stem in stems if SBJ_MAIN_RE.fullmatch(stem)),
        key=lambda stem: int(SBJ_MAIN_RE.fullmatch(stem).group(1)),  # type: ignore[union-attr]
    )
    extras = [stem for stem in stems if stem not in main]
    return main, extras


def _generate_subject_annotations(
    spec: DatasetSpec,
    sensor_folder: Path,
    output_folder: Path,
    sampling_rate: int,
    label_col: str,
    overwrite: bool,
    null_label: str,
) -> Tuple[List[Path], List[str], int]:
    main_subjects, extras = _subject_stems(sensor_folder)
    if not main_subjects:
        raise ValueError(f"No main sbj_<integer>.csv files found in {sensor_folder}")

    csv_paths = [sensor_folder / f"{stem}.csv" for stem in main_subjects + extras]
    labels = _collect_label_order(
        csv_paths, label_col=label_col, has_null=spec.has_null, null_label=null_label
    )
    label_to_id = {label: index for index, label in enumerate(labels)}

    if output_folder.exists() and overwrite:
        for old in output_folder.glob("loso_sbj_*.json"):
            old.unlink()
    output_folder.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []
    for holdout in main_subjects:
        database: Dict[str, Dict[str, object]] = {}
        holdout_index = int(SBJ_MAIN_RE.fullmatch(holdout).group(1))  # type: ignore[union-attr]

        for stem in main_subjects + extras:
            match = SBJ_ANY_RE.fullmatch(stem)
            if match is None:
                continue
            base_index = int(match.group(1))
            subset = "Validation" if base_index == holdout_index else "Training"
            database[stem] = {
                "subset": subset,
                "annotations": _segments_from_labels(
                    sensor_folder / f"{stem}.csv",
                    label_col=label_col,
                    label_to_id=label_to_id,
                    sampling_rate=float(sampling_rate),
                    null_label=null_label,
                ),
            }

        output = {
            "version": "1.0",
            "dataset": spec.key,
            "sampling_rate_hz": float(sampling_rate),
            "label_dict": labels,
            "database": database,
        }
        output_path = output_folder / f"loso_sbj_{holdout_index}.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
        outputs.append(output_path)

    return outputs, labels, len(main_subjects)


def _generate_rwhar_annotations(
    spec: DatasetSpec,
    sensor_folder: Path,
    output_folder: Path,
    sampling_rate: int,
    overwrite: bool,
) -> Tuple[List[Path], List[str], int]:
    items: List[Tuple[int, str, str, Path]] = []
    for path in _csv_files(sensor_folder):
        match = RWHAR_FILE_RE.fullmatch(path.name)
        if match:
            items.append((int(match.group(1)), match.group(2).lower(), path.stem, path))
    if not items:
        raise ValueError(
            f"No proband<ID>_<activity>.csv files found in {sensor_folder}. "
            "Use scripts/preprocess_realworld_waist_accel.py or pass the raw proband root."
        )

    found_labels = {activity for _, activity, _, _ in items}
    labels = [activity for activity in RWHAR_ACTIVITIES if activity in found_labels]
    labels.extend(sorted(found_labels.difference(labels)))
    label_to_id = {label: index for index, label in enumerate(labels)}
    subjects = sorted({pid for pid, _, _, _ in items})

    if output_folder.exists() and overwrite:
        for old in output_folder.glob("loso_sbj_*.json"):
            old.unlink()
    output_folder.mkdir(parents=True, exist_ok=True)

    durations: Dict[str, float] = {}
    for _, _, stem, path in items:
        rows = max(0, sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1)
        durations[stem] = float(rows / sampling_rate)

    outputs: List[Path] = []
    for split_index, holdout_pid in enumerate(subjects):
        database: Dict[str, Dict[str, object]] = {}
        for pid, activity, stem, _ in items:
            database[stem] = {
                "subset": "Validation" if pid == holdout_pid else "Training",
                "annotations": [
                    {
                        "segment": [0.0, durations[stem]],
                        "label": activity,
                        "label_id": int(label_to_id[activity]),
                    }
                ],
            }

        output = {
            "version": "1.0",
            "dataset": "realworld_waist_accel",
            "sampling_rate_hz": float(sampling_rate),
            "label_dict": labels,
            "database": database,
        }
        output_path = output_folder / f"loso_sbj_{split_index}.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
        outputs.append(output_path)

    return outputs, labels, len(subjects)


def prepare_annotations(
    spec: DatasetSpec,
    repo_root: Path,
    label_col_arg: str,
    null_label: str,
    overwrite: bool,
) -> Tuple[Dict[str, List[Path]], List[str], int, int]:
    print(f"\n=== Generating {spec.key} LOSO annotations ===")
    annotation_sets: Dict[str, List[Path]] = {}
    labels: List[str] = []
    subject_count = 0
    input_dim = 0

    if spec.key == "wear":
        # WEAR keeps per-rate annotation directories, matching existing configs.
        keys: List[str] = [str(rate) for rate in spec.rates] + ["multirate"]
    else:
        # RWHAR and WISDM reuse one LOSO membership set for all rates.
        keys = ["multirate"]

    for key in keys:
        rate = spec.native_rate if key == "multirate" else int(key)
        sensor_folder = (
            _multirate_folder(spec, repo_root)
            if key == "multirate"
            else _rate_folder(spec, repo_root, rate)
        )
        csvs = _csv_files(sensor_folder)
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {sensor_folder}")

        label_col = _detect_label_column(
            csvs[0], spec.default_label_candidates, explicit=label_col_arg
        )
        current_input_dim = _validate_training_csv(csvs[0], label_col=label_col)
        if input_dim and input_dim != current_input_dim:
            raise ValueError(
                f"Input dimension differs across streams: {input_dim} vs {current_input_dim}"
            )
        input_dim = current_input_dim

        output_folder = _annotation_folder(spec, repo_root, key)
        if spec.key == "rwhar":
            generated, current_labels, current_subjects = _generate_rwhar_annotations(
                spec,
                sensor_folder=sensor_folder,
                output_folder=output_folder,
                sampling_rate=rate,
                overwrite=overwrite,
            )
        else:
            generated, current_labels, current_subjects = _generate_subject_annotations(
                spec,
                sensor_folder=sensor_folder,
                output_folder=output_folder,
                sampling_rate=rate,
                label_col=label_col,
                overwrite=overwrite,
                null_label=null_label,
            )

        if labels and labels != current_labels:
            raise ValueError(
                f"Label order differs between generated annotation sets for {spec.key}. "
                f"Expected {labels}, got {current_labels}"
            )
        labels = current_labels
        subject_count = current_subjects
        annotation_sets[key] = generated
        print(
            f"[OK] {key}: subjects={current_subjects}, labels={len(labels)}, "
            f"annotations={len(generated)}, output={output_folder}"
        )

    if subject_count != spec.expected_subjects:
        print(
            f"[WARN] {spec.key}: expected {spec.expected_subjects} LOSO subjects, "
            f"found {subject_count}."
        )

    # Alias all non-WEAR rates to the shared annotation set.
    if spec.key != "wear":
        shared = annotation_sets["multirate"]
        for rate in spec.rates:
            annotation_sets[str(rate)] = shared

    return annotation_sets, labels, subject_count, input_dim


# ---------------------------------------------------------------------------
# Config refresh/validation
# ---------------------------------------------------------------------------


def _config_rate_key(config_path: Path, cfg: Mapping[str, object], spec: DatasetSpec) -> str:
    if "multirate" in config_path.stem.lower():
        return "multirate"
    dataset_cfg = cfg.get("dataset", {})
    if not isinstance(dataset_cfg, Mapping):
        raise ValueError(f"Invalid dataset mapping in {config_path}")
    rate = int(round(float(dataset_cfg.get("sampling_rate", spec.native_rate))))
    if rate not in spec.rates:
        raise ValueError(f"Unexpected sampling rate {rate} in {config_path}")
    return str(rate)


def _architecture_name(config_path: Path) -> str:
    parts = list(config_path.parts)
    if "configs" in parts:
        index = parts.index("configs")
        if index + 1 < len(parts):
            return parts[index + 1].lower()
    return config_path.parent.parent.name.lower()


def refresh_configs(
    spec: DatasetSpec,
    repo_root: Path,
    annotation_sets: Mapping[str, Sequence[Path]],
    labels: Sequence[str],
    input_dim: int,
    overwrite: bool,
) -> List[Path]:
    del overwrite  # Existing YAMLs are updated in place after .bak creation.
    print(f"\n=== Refreshing {spec.key} YAML configs ===")
    config_root = repo_root / "configs"
    if not config_root.is_dir():
        raise FileNotFoundError(f"Config root not found: {config_root}")

    updated: List[Path] = []
    for path in sorted(config_root.rglob("*.yaml")):
        with open(path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        if not isinstance(cfg, MutableMapping):
            raise ValueError(f"Config is empty or invalid: {path}")
        if str(cfg.get("dataset_name", "")).lower() != spec.config_name:
            continue

        key = _config_rate_key(path, cfg, spec)
        rate = spec.native_rate if key == "multirate" else int(key)
        annotation_paths = annotation_sets.get(key)
        if not annotation_paths:
            raise KeyError(f"No annotations available for {spec.key} key={key}")

        dataset_cfg = cfg.setdefault("dataset", {})
        model_cfg = cfg.setdefault("model", {})
        train_cfg = cfg.setdefault("train_cfg", {})
        if not isinstance(dataset_cfg, MutableMapping):
            raise ValueError(f"dataset section is not a mapping: {path}")
        if not isinstance(model_cfg, MutableMapping):
            raise ValueError(f"model section is not a mapping: {path}")
        if not isinstance(train_cfg, MutableMapping):
            raise ValueError(f"train_cfg section is not a mapping: {path}")

        sensor_folder = (
            _multirate_folder(spec, repo_root)
            if key == "multirate"
            else _rate_folder(spec, repo_root, rate)
        )
        cfg["anno_json"] = [_repo_relative(repo_root, item) for item in annotation_paths]
        dataset_cfg["sens_folder"] = _repo_relative(repo_root, sensor_folder)
        dataset_cfg["sampling_rate"] = int(rate)
        dataset_cfg["window_size"] = int(rate)  # one-second windows
        dataset_cfg["input_dim"] = int(input_dim)
        dataset_cfg["num_classes"] = int(len(labels))

        if key == "multirate" or bool(model_cfg.get("multirate_training", False)):
            model_cfg["supported_sample_rates"] = list(spec.rates[::-1])

        architecture = _architecture_name(path)
        train_cfg["log_subdir"] = f"{architecture}/{spec.key}/{key if key == 'multirate' else str(rate) + 'hz'}"

        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        updated.append(path)
        print(f"[OK] {path.relative_to(repo_root)}")

    if not updated:
        raise RuntimeError(f"No YAML configs matched dataset_name={spec.config_name}")
    return updated


def validate_prepared_dataset(
    spec: DatasetSpec,
    repo_root: Path,
    labels: Optional[Sequence[str]] = None,
) -> None:
    print(f"\n=== Validating {spec.key} prepared data/configs ===")
    problems: List[str] = []

    for rate in spec.rates:
        folder = _rate_folder(spec, repo_root, rate)
        files = _csv_files(folder)
        if not files:
            problems.append(f"Missing/empty sensor folder: {folder}")
            continue
        try:
            label_col = _detect_label_column(
                files[0], spec.default_label_candidates, explicit="auto"
            )
            _validate_training_csv(files[0], label_col=label_col)
        except Exception as exc:  # noqa: BLE001 - report every validation problem
            problems.append(str(exc))

    multirate = _multirate_folder(spec, repo_root)
    if not _csv_files(multirate):
        problems.append(f"Missing/empty multirate folder: {multirate}")

    config_root = repo_root / "configs"
    matched = 0
    for path in sorted(config_root.rglob("*.yaml")):
        with open(path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        if not isinstance(cfg, Mapping) or str(cfg.get("dataset_name", "")).lower() != spec.config_name:
            continue
        matched += 1
        dataset_cfg = cfg.get("dataset", {})
        if not isinstance(dataset_cfg, Mapping):
            problems.append(f"Invalid dataset mapping: {path}")
            continue
        sensor = _resolve(repo_root, str(dataset_cfg.get("sens_folder", "")))
        if not _csv_files(sensor):
            problems.append(f"Config points to missing/empty sensor folder: {path} -> {sensor}")
        annos = cfg.get("anno_json", [])
        if not isinstance(annos, list) or not annos:
            problems.append(f"Config has no anno_json list: {path}")
        else:
            for annotation in annos:
                annotation_path = _resolve(repo_root, str(annotation))
                if not annotation_path.is_file():
                    problems.append(f"Missing annotation: {path} -> {annotation_path}")
        if labels is not None and int(dataset_cfg.get("num_classes", -1)) != len(labels):
            problems.append(
                f"num_classes mismatch: {path} has {dataset_cfg.get('num_classes')} "
                f"but annotations contain {len(labels)} labels"
            )

    if matched == 0:
        problems.append(f"No configs matched dataset_name={spec.config_name}")

    if problems:
        print("Validation failed:")
        for problem in problems:
            print(f"  - {problem}")
        raise RuntimeError(f"{spec.key}: {len(problems)} validation problem(s)")

    print(f"[OK] {spec.key}: streams, annotations, and {matched} configs are consistent")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_steps(value: str) -> Tuple[str, ...]:
    if value.strip().lower() == "all":
        return ("streams", "annotations", "configs", "validate")
    allowed = {"streams", "annotations", "configs", "validate"}
    steps = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    unknown = set(steps).difference(allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown steps: {sorted(unknown)}")
    if not steps:
        raise argparse.ArgumentTypeError("At least one step is required")
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare streams, LOSO annotations, and YAML config paths for all paper datasets."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("wear", "rwhar", "wisdm_watch", "all"),
    )
    parser.add_argument("--source", help="Source folder when preparing one dataset.")
    parser.add_argument("--wear-source", help="WEAR native 50 Hz prepared CSV folder.")
    parser.add_argument(
        "--rwhar-source",
        help="RealWorld-HAR raw proband root or processed probandX_activity.csv folder.",
    )
    parser.add_argument(
        "--wisdm-source",
        help="WISDM-watch prepared sbj_*.csv folder or raw CSV folder.",
    )
    parser.add_argument("--repo-root", default=".", help="Repository root (default: current folder).")
    parser.add_argument(
        "--steps",
        type=_parse_steps,
        default=_parse_steps("all"),
        help="all or comma-separated: streams,annotations,configs,validate",
    )
    parser.add_argument(
        "--native-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to expose already prepared native-rate data (default: symlink).",
    )
    parser.add_argument("--label-col", default="auto", help="Label column, or auto.")
    parser.add_argument("--null-label", default="null", help="WEAR background label (default: null).")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _source_map(args: argparse.Namespace, repo_root: Path) -> Dict[str, Optional[Path]]:
    if args.dataset != "all":
        source = args.source
        if source is None and "streams" in args.steps:
            raise SystemExit("--source is required when --steps includes streams")
        return {args.dataset: _resolve(repo_root, source) if source else None}

    result: Dict[str, Optional[Path]] = {
        "wear": _resolve(repo_root, args.wear_source) if args.wear_source else None,
        "rwhar": _resolve(repo_root, args.rwhar_source) if args.rwhar_source else None,
        "wisdm_watch": _resolve(repo_root, args.wisdm_source) if args.wisdm_source else None,
    }
    if "streams" in args.steps:
        missing = [key for key, value in result.items() if value is None]
        if missing:
            raise SystemExit(
                f"--dataset all with streams requires dataset-specific sources; missing {missing}"
            )
    return result


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not (repo_root / "configs").is_dir():
        raise SystemExit(f"Not a repository root (configs/ missing): {repo_root}")

    selected = list(SPECS) if args.dataset == "all" else [args.dataset]
    sources = _source_map(args, repo_root)

    summaries: List[Dict[str, object]] = []
    for dataset_key in selected:
        spec = SPECS[dataset_key]
        source = sources.get(dataset_key)

        if "streams" in args.steps:
            if source is None:
                raise SystemExit(f"No source supplied for {dataset_key}")
            prepare_streams(
                spec,
                source=source,
                repo_root=repo_root,
                native_mode=args.native_mode,
                overwrite=args.overwrite,
            )

        annotation_sets: Dict[str, List[Path]] = {}
        labels: List[str] = []
        subjects = 0
        input_dim = 0

        if "annotations" in args.steps:
            annotation_sets, labels, subjects, input_dim = prepare_annotations(
                spec,
                repo_root=repo_root,
                label_col_arg=args.label_col,
                null_label=args.null_label,
                overwrite=args.overwrite,
            )

        if "configs" in args.steps:
            if not annotation_sets:
                # Load already generated annotation paths when configs are refreshed separately.
                if spec.key == "wear":
                    keys = [str(rate) for rate in spec.rates] + ["multirate"]
                else:
                    keys = ["multirate"]
                for key in keys:
                    folder = _annotation_folder(spec, repo_root, key)
                    files = sorted(folder.glob("loso_sbj_*.json"))
                    if not files:
                        raise FileNotFoundError(
                            f"No generated annotations in {folder}; run annotations first."
                        )
                    annotation_sets[key] = files
                if spec.key != "wear":
                    for rate in spec.rates:
                        annotation_sets[str(rate)] = annotation_sets["multirate"]

                with open(annotation_sets["multirate"][0], "r", encoding="utf-8") as handle:
                    first_annotation = json.load(handle)
                raw_labels = first_annotation.get("label_dict", [])
                labels = list(raw_labels) if isinstance(raw_labels, list) else list(raw_labels)

                native_csvs = _csv_files(_rate_folder(spec, repo_root, spec.native_rate))
                if not native_csvs:
                    raise FileNotFoundError("Native sensor folder is empty")
                detected_label = _detect_label_column(
                    native_csvs[0], spec.default_label_candidates, explicit=args.label_col
                )
                input_dim = _validate_training_csv(native_csvs[0], detected_label)

            refresh_configs(
                spec,
                repo_root=repo_root,
                annotation_sets=annotation_sets,
                labels=labels,
                input_dim=input_dim,
                overwrite=args.overwrite,
            )

        if "validate" in args.steps:
            validate_prepared_dataset(spec, repo_root=repo_root, labels=labels or None)

        summaries.append(
            {
                "dataset": spec.key,
                "subjects": subjects or None,
                "classes": len(labels) if labels else None,
                "input_dim": input_dim or None,
            }
        )

    print("\n=== Preparation complete ===")
    for summary in summaries:
        print(
            f"{summary['dataset']}: subjects={summary['subjects']} "
            f"classes={summary['classes']} input_dim={summary['input_dim']}"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
