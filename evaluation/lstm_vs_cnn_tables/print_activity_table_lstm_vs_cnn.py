from pathlib import Path
import argparse
import re
import pandas as pd
import numpy as np

ROOT = Path("/home/g263412/GAMMA")

MODEL_ORDER = ["pure_cnn", "deepconvlstm"]

DATASET_RATES = {
    "rwhar": [6, 12, 25, 50],
    "wisdm": [5, 10, 20],
    "wisdm_watch": [5, 10, 20],
}

BAD_FILE_PATTERNS = [
    "summary",
    "by_split",
    "latex",
    "collected",
    "activity_table",
]

BAD_ROWS = {
    "",
    "nan",
    "accuracy",
    "macro avg",
    "weighted avg",
    "micro avg",
    "samples avg",
}


def norm(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def detect_rate(path, df):
    # Prefer explicit column.
    for c in df.columns:
        nc = norm(c)
        if nc in {"rate", "rate_hz", "hz", "freq", "frequency", "sampling_rate"}:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals):
                return int(round(float(vals.iloc[0])))

    # Then path.
    s = str(path).lower()
    for r in [50, 25, 20, 15, 12, 10, 6, 5]:
        pats = [
            rf"(^|[/_\-]){r}hz($|[/_\-])",
            rf"(^|[/_\-]){r}_hz($|[/_\-])",
            rf"(^|[/_\-])rate_{r}($|[/_\-])",
            rf"(^|[/_\-]){r}($|[/_\-])",
        ]
        if any(re.search(p, s) for p in pats):
            return r

    return None


def detect_split(path):
    s = str(path)
    m = re.search(r"loso_sbj_\d+", s)
    if m:
        return m.group(0)

    m = re.search(r"(?:sbj|subject|subj)[_\-]?(\d+)", s, flags=re.I)
    if m:
        return "sbj_" + m.group(1)

    return Path(path).parent.name


def detect_columns(df):
    cols = list(df.columns)
    normed = {c: norm(c) for c in cols}

    f1_col = None
    for c, nc in normed.items():
        if nc in {"f1", "f1_score", "f1score"} or "f1" in nc:
            f1_col = c
            break

    if f1_col is None:
        return None, None

    activity_col = None
    for c, nc in normed.items():
        if nc in {"activity", "class", "label", "class_name", "activity_name"}:
            activity_col = c
            break

    if activity_col is None:
        unnamed = [c for c in cols if str(c).lower().startswith("unnamed")]
        activity_col = unnamed[0] if unnamed else cols[0]

    return activity_col, f1_col


def should_skip_file(path):
    name = path.name.lower()
    full = str(path).lower()

    if not name.endswith(".csv"):
        return True

    if any(p in name for p in BAD_FILE_PATTERNS):
        return True

    # keep likely classification report / per-class metric CSVs
    # but do not force exact names because your checkpoint folders may differ
    return False


def read_csv_file(path, dataset, model):
    if should_skip_file(path):
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if df.empty:
        return None

    activity_col, f1_col = detect_columns(df)
    if activity_col is None or f1_col is None:
        return None

    rate = detect_rate(path, df)
    if rate is None:
        return None

    split = detect_split(path)

    out = pd.DataFrame({
        "model": model,
        "rate_hz": rate,
        "split": split,
        "activity": df[activity_col].astype(str).str.strip(),
        "f1": pd.to_numeric(df[f1_col], errors="coerce"),
        "source": str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path),
    })

    WISDM_CLASS_MAP = {
        "class_0": "A",
        "class_1": "B",
        "class_2": "C",
        "class_3": "D",
        "class_4": "E",
        "class_5": "F",
        "class_6": "G",
        "class_7": "H",
        "class_8": "I",
        "class_9": "J",
        "class_10": "K",
        "class_11": "L",
        "class_12": "M",
        "class_13": "O",
        "class_14": "P",
        "class_15": "Q",
        "class_16": "R",
        "class_17": "S",
    }

    if dataset in {"wisdm", "wisdm_watch"}:
        out["activity"] = out["activity"].replace(WISDM_CLASS_MAP)

    # Remove aggregate rows and the bad split rows that polluted your previous table.
    out = out.dropna(subset=["f1"])
    out = out[~out["activity"].str.lower().isin(BAD_ROWS)]
    out = out[~out["activity"].str.match(r"^loso_sbj_\d+$", na=False)]

    if out.empty:
        return None

    # Percent to fraction if needed.
    if out["f1"].median() > 1.5:
        out["f1"] = out["f1"] / 100.0

    out = out[(out["f1"] >= 0) & (out["f1"] <= 1)]

    if out.empty:
        return None

    return out


def collect(base, dataset):
    base = Path(base)
    rates = DATASET_RATES[dataset]
    rows = []

    for model in MODEL_ORDER:
        model_dir = base / model

        if not model_dir.exists():
            print(f"WARNING missing model dir: {model_dir}")
            continue

        for path in sorted(model_dir.rglob("*.csv")):
            part = read_csv_file(path, dataset, model)
            if part is not None and not part.empty:
                rows.append(part)

    if not rows:
        raise SystemExit("No valid per-activity CSV rows found.")

    raw = pd.concat(rows, ignore_index=True)
    raw = raw[raw["rate_hz"].isin(rates)]

    # One row per split/activity/model/rate/source. Then mean over splits later.
    raw = raw.drop_duplicates(["model", "rate_hz", "split", "activity", "source"])

    summary = (
        raw.groupby(["model", "rate_hz", "activity"], as_index=False)
           .agg(
               mean=("f1", "mean"),
               n_subjects=("split", "nunique"),
               std=("f1", "std"),
           )
    )

    return raw, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["rwhar", "wisdm", "wisdm_watch"])
    ap.add_argument("--base", required=True)
    args = ap.parse_args()

    rates = DATASET_RATES[args.dataset]
    raw, summary = collect(args.base, args.dataset)

    print("\nSOURCE FILES USED:")
    for s in sorted(raw["source"].unique()):
        print(s)

    print("\nSUBJECT/FOLD COUNT CHECK:")
    count_rows = []
    for model in MODEL_ORDER:
        for rate in rates:
            sub = summary[(summary["model"] == model) & (summary["rate_hz"] == rate)]
            if len(sub):
                count_rows.append({
                    "model": model,
                    "rate": rate,
                    "min_n_subjects": int(sub["n_subjects"].min()),
                    "max_n_subjects": int(sub["n_subjects"].max()),
                    "activities": int(sub["activity"].nunique()),
                })
    print(pd.DataFrame(count_rows).to_string(index=False))

    activities = sorted(summary["activity"].unique())

    table = pd.DataFrame({"Activity": activities})

    for model in MODEL_ORDER:
        prefix = "PureCNN" if model == "pure_cnn" else "DeepConvLSTM"

        for rate in rates:
            vals = {}
            sub = summary[(summary["model"] == model) & (summary["rate_hz"] == rate)]

            for _, row in sub.iterrows():
                vals[row["activity"]] = row["mean"]

            table[f"{prefix}_{rate}"] = [
                "--" if a not in vals else f"{vals[a]:.3f}"
                for a in activities
            ]

    print("\nNORMAL TABLE:")
    print(table.to_string(index=False))

    out_dir = ROOT / "results" / "lstm_vs_cnn_activity_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / f"{args.dataset}_raw_clean_activity_rows.csv"
    summary_path = out_dir / f"{args.dataset}_summary_clean_activity_mean.csv"
    table_path = out_dir / f"{args.dataset}_normal_activity_table.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    table.to_csv(table_path, index=False)

    print("\nSAVED:")
    print(raw_path)
    print(summary_path)
    print(table_path)


if __name__ == "__main__":
    main()
