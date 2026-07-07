from pathlib import Path
import argparse
import re
import pandas as pd
import numpy as np

ROOT = Path("/home/g263412/GAMMA")
OUT = ROOT / "results/lstm_vs_cnn_per_activity_tables"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PRETTY = {
    "pure_cnn": "Pure CNN",
    "deepconvlstm": "DeepConvLSTM",
}

MODEL_ORDER = ["pure_cnn", "deepconvlstm"]

DATASET_PRETTY = {
    "rwhar": "RealWorld-HAR",
    "wisdm": "WISDM-watch",
    "wisdm_watch": "WISDM-watch",
}

DATASET_RATES = {
    "rwhar": [6, 12, 25, 50],
    "wisdm": [5, 10, 20],
    "wisdm_watch": [5, 10, 20],
}

TOL = 0.02


def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def detect_rate(path, df):
    # First try explicit CSV columns.
    for c in df.columns:
        cn = norm_col(c)
        if cn in {"rate", "rate_hz", "hz", "freq", "frequency", "sampling_rate"}:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals):
                return int(round(float(vals.iloc[0])))

    # Then try path tokens, but prefer rate-like directory names.
    parts = [p.lower() for p in Path(path).parts]
    for token in reversed(parts):
        m = re.search(r"(?:^|[_\-])(?:rate|hz|freq)?[_\-]?(50|25|20|15|12|10|6|5)\s*hz?(?:$|[_\-])", token)
        if m:
            return int(m.group(1))
        m = re.search(r"(50|25|20|15|12|10|6|5)\s*hz", token)
        if m:
            return int(m.group(1))

    # Last fallback: any standalone known rate in path.
    s = str(path).lower()
    candidates = []
    for r in [50, 25, 20, 15, 12, 10, 6, 5]:
        if re.search(rf"(^|[/_\-]){r}(hz)?($|[/_\-])", s):
            candidates.append(r)
    if candidates:
        return candidates[-1]

    return None


def detect_split(path, df):
    # First try CSV columns.
    for c in df.columns:
        cn = norm_col(c)
        if cn in {"split", "fold", "subject", "sbj", "loso", "held_out_subject"}:
            val = str(df[c].iloc[0])
            if val and val.lower() != "nan":
                return val

    s = str(path)
    m = re.search(r"loso_sbj_\d+", s)
    if m:
        return m.group(0)

    m = re.search(r"(?:sbj|subject|subj)[_\-]?(\d+)", s, flags=re.I)
    if m:
        return "sbj_" + m.group(1)

    return Path(path).parent.name


def detect_columns(df):
    cols_norm = {c: norm_col(c) for c in df.columns}

    # F1 column candidates.
    f1_candidates = []
    for c, cn in cols_norm.items():
        if cn in {"f1", "f1_score", "f1score", "f1_macro", "f1_score_mean", "mean_f1"}:
            f1_candidates.append(c)
        elif "f1" in cn and "macro" not in cn and "weighted" not in cn:
            f1_candidates.append(c)
        elif cn == "f1_score":
            f1_candidates.append(c)

    if not f1_candidates:
        return None, None

    f1_col = f1_candidates[0]

    # Activity/class column candidates.
    activity_candidates = []
    for c, cn in cols_norm.items():
        if cn in {"activity", "class", "label", "target", "class_name", "activity_name", "name"}:
            activity_candidates.append(c)
        elif "activity" in cn or "class" in cn or cn == "label":
            activity_candidates.append(c)

    if activity_candidates:
        activity_col = activity_candidates[0]
    else:
        # Classification report CSVs often store class names in first unnamed column.
        unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
        if unnamed:
            activity_col = unnamed[0]
        else:
            activity_col = df.columns[0]

    return activity_col, f1_col


def read_per_activity_csv(path, dataset, model):
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

    split = detect_split(path, df)

    tmp = pd.DataFrame({
        "dataset": dataset,
        "model": model,
        "rate_hz": rate,
        "split": split,
        "activity": df[activity_col].astype(str),
        "f1": pd.to_numeric(df[f1_col], errors="coerce"),
        "source_file": str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path),
        "mtime": path.stat().st_mtime,
    })

    # Remove sklearn report summary rows and invalid rows.
    bad = {
        "accuracy",
        "macro avg",
        "weighted avg",
        "micro avg",
        "samples avg",
        "nan",
        "",
    }
    tmp["activity_clean"] = tmp["activity"].str.strip().str.lower()
    tmp = tmp[~tmp["activity_clean"].isin(bad)]
    tmp = tmp.drop(columns=["activity_clean"])
    tmp = tmp.dropna(subset=["f1"])

    if tmp.empty:
        return None

    # Convert percent to fraction if needed.
    if tmp["f1"].median() > 1.5:
        tmp["f1"] = tmp["f1"] / 100.0

    # Guard against prediction CSVs or nonsense.
    tmp = tmp[(tmp["f1"] >= 0.0) & (tmp["f1"] <= 1.0)]

    if tmp.empty:
        return None

    return tmp


def collect_dataset(dataset, base):
    base = Path(base)

    if not base.exists():
        raise SystemExit(f"Base path does not exist: {base}")

    all_rows = []

    for model in MODEL_ORDER:
        model_dir = base / model
        if not model_dir.exists():
            print(f"WARNING: missing model directory: {model_dir}")
            continue

        for csv_path in sorted(model_dir.rglob("*.csv")):
            part = read_per_activity_csv(csv_path, dataset, model)
            if part is not None and not part.empty:
                all_rows.append(part)

    if not all_rows:
        raise SystemExit(
            f"No usable per-activity CSVs found under {base}. "
            "Check that the CSVs contain an activity/class column and an F1/f1-score column."
        )

    df = pd.concat(all_rows, ignore_index=True)

    rates = DATASET_RATES[dataset]
    df = df[df["rate_hz"].isin(rates)]

    # If multiple CSVs exist for the same split/model/rate/activity, keep the newest file.
    df = df.sort_values(["dataset", "model", "rate_hz", "split", "activity", "mtime"])
    df = df.drop_duplicates(["dataset", "model", "rate_hz", "split", "activity"], keep="last")

    raw_path = OUT / f"{dataset}_per_activity_by_split_raw.csv"
    df.drop(columns=["mtime"]).to_csv(raw_path, index=False)

    summary = (
        df.groupby(["dataset", "model", "rate_hz", "activity"], as_index=False)
          .agg(
              folds=("split", "nunique"),
              mean=("f1", "mean"),
              std=("f1", "std"),
              min=("f1", "min"),
              max=("f1", "max"),
          )
    )

    summary_path = OUT / f"{dataset}_per_activity_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\nSaved raw split-level file: {raw_path}")
    print(f"Saved per-activity summary: {summary_path}")

    print("\nDetected source files:")
    for src in sorted(df["source_file"].unique()):
        print("  " + src)

    return summary


def latex_cell(v, best, underline):
    if pd.isna(v):
        return "--"

    s = f"{v:.3f}"

    if v >= best - TOL:
        if underline:
            return r"\underline{\textcolor{green}{" + s + "}}"
        return r"\textcolor{green}{" + s + "}"

    return s


def make_latex_table(summary, dataset):
    pretty_dataset = DATASET_PRETTY.get(dataset, dataset)
    rates = DATASET_RATES[dataset]
    n_rates = len(rates)

    sub = summary.copy()

    activities = sorted(sub["activity"].unique())

    colspec = (
        r"@{}>{\raggedright\arraybackslash}X "
        + "c" * n_rates
        + r" @{\hspace{1.5em}} "
        + "c" * n_rates
        + "@{}"
    )

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering\scriptsize")
    lines.append(
        rf"  \caption{{Per-activity performance along the sampling rates, on the {pretty_dataset} dataset, using two standard models, a pure CNN and the DeepConvLSTM model. Values are mean, per-class $F_1$ scores from LOSO cross validation. For each model, scores within a 0.02 absolute $F_1$ range of the best score are shown in \textcolor{{green}}{{green}}, and the lowest-rate score within this range is \underline{{\textcolor{{green}}{{underlined}}}}.}}"
    )
    lines.append(rf"  \label{{tab:freq_perclass_{dataset}_combined}}")
    lines.append(rf"  \begin{{tabularx}}{{\linewidth}}{{{colspec}}}")
    lines.append(r"    \toprule")
    lines.append(
        rf"    & \multicolumn{{{n_rates}}}{{c}}{{Pure CNN}} & \multicolumn{{{n_rates}}}{{c}}{{DeepConvLSTM}} \\"
    )
    rate_header = " & ".join(str(r) for r in rates)
    lines.append(rf"    Activity & {rate_header} & {rate_header} \\")
    lines.append(r"    \midrule")

    selected_values = {m: [] for m in MODEL_ORDER}

    for activity in activities:
        row = [f"    {activity}"]

        for model in MODEL_ORDER:
            vals = {}
            for rate in rates:
                hit = sub[
                    (sub["model"] == model)
                    & (sub["activity"] == activity)
                    & (sub["rate_hz"] == rate)
                ]
                vals[rate] = float(hit["mean"].iloc[0]) if len(hit) else np.nan

            valid = [v for v in vals.values() if not pd.isna(v)]
            if not valid:
                for _ in rates:
                    row.append("--")
                continue

            best = max(valid)
            close_rates = [
                rate for rate in rates
                if not pd.isna(vals[rate]) and vals[rate] >= best - TOL
            ]

            # This matches your table logic: underline the lowest sampling rate
            # that is within 0.02 of the best, not necessarily the numerical maximum.
            underline_rate = close_rates[0] if close_rates else None

            if underline_rate is not None:
                selected_values[model].append(vals[underline_rate])

            for rate in rates:
                row.append(latex_cell(vals[rate], best, rate == underline_rate))

        lines.append(" & ".join(row) + r" \\")

    lines.append(r"    \midrule")

    # Macro mean row.
    row = ["    Macro mean"]
    for model in MODEL_ORDER:
        vals = {}
        for rate in rates:
            hit = sub[(sub["model"] == model) & (sub["rate_hz"] == rate)]
            vals[rate] = float(hit["mean"].mean()) if len(hit) else np.nan

        valid = [v for v in vals.values() if not pd.isna(v)]
        best = max(valid)
        close_rates = [
            rate for rate in rates
            if not pd.isna(vals[rate]) and vals[rate] >= best - TOL
        ]
        underline_rate = close_rates[0] if close_rates else None

        for rate in rates:
            row.append(latex_cell(vals[rate], best, rate == underline_rate))

    lines.append(" & ".join(row) + r" \\")

    # Macro F1 optimal per-activity row.
    opt_vals = []
    for model in MODEL_ORDER:
        vals = selected_values[model]
        opt_vals.append(np.mean(vals) if vals else np.nan)

    opt_pure = "--" if pd.isna(opt_vals[0]) else f"{opt_vals[0]:.3f}"
    opt_lstm = "--" if pd.isna(opt_vals[1]) else f"{opt_vals[1]:.3f}"

    lines.append(
        rf"    Macro $F_1$ optimal per-activity & \multicolumn{{{n_rates}}}{{c}}{{{opt_pure}}} & \multicolumn{{{n_rates}}}{{c}}{{{opt_lstm}}} \\"
    )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabularx}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["rwhar", "wisdm", "wisdm_watch"])
    parser.add_argument("--base", required=True)
    args = parser.parse_args()

    summary = collect_dataset(args.dataset, args.base)
    table = make_latex_table(summary, args.dataset)

    tex_path = OUT / f"{args.dataset}_purecnn_vs_deepconvlstm_per_activity_table.tex"
    tex_path.write_text(table, encoding="utf-8")

    print(f"\nSaved LaTeX table: {tex_path}")
    print("\nLaTeX table:\n")
    print(table)


if __name__ == "__main__":
    main()
