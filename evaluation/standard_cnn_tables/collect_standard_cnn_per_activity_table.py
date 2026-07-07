import os
import glob
import re
import numpy as np
import pandas as pd

OUT_DIR = "evaluation/standard_cnn_tables"
os.makedirs(OUT_DIR, exist_ok=True)

ROOTS = {
    "6":  "logs/deepconvlstm/standard_cnn/6hz",
    "12": "logs/deepconvlstm/standard_cnn/12hz",
    "25": "logs/deepconvlstm/standard_cnn/25hz",
    "50": "logs/deepconvlstm/standard_cnn/50hz",
}

ACTIVITY_ORDER = [
    "null",
    "bench-dips",
    "burpees",
    "jogging",
    "jogging (butt-kicks)",
    "jogging (rotating arms)",
    "jogging (sidesteps)",
    "jogging (skipping)",
    "lunges",
    "lunges (complex)",
    "push-ups",
    "push-ups (complex)",
    "sit-ups",
    "sit-ups (complex)",
    "stretching (hamstrings)",
    "stretching (lumbar rotation)",
    "stretching (lunging)",
    "stretching (shoulders)",
    "stretching (triceps)",
]

ID_TO_ACTIVITY = {i: name for i, name in enumerate(ACTIVITY_ORDER)}

# Bold best score and values within 0.02 absolute F1 of the best.
BOLD_WITHIN = 0.02


def fold_key(path):
    m = re.search(r"loso_sbj_(\d+)", path)
    return int(m.group(1)) if m else 10**9


def detect_col(df, candidates, contains=None):
    lower = {str(c).lower().strip(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]

    if contains is not None:
        for c in df.columns:
            lc = str(c).lower().strip()
            if contains in lc:
                return c

    return None


def read_per_class_csv(path):
    df = pd.read_csv(path, keep_default_na=False)

    f1_col = detect_col(
        df,
        candidates=[
            "f1",
            "F1",
            "f1_score",
            "f1-score",
            "f1score",
            "F1-score",
            "F1_score",
        ],
        contains="f1",
    )

    if f1_col is None:
        raise RuntimeError(f"Could not detect F1 column in {path}. Columns: {list(df.columns)}")

    name_col = detect_col(
        df,
        candidates=[
            "class_name",
            "activity",
            "label",
            "label_name",
            "name",
            "class",
        ],
    )

    id_col = detect_col(
        df,
        candidates=[
            "class_id",
            "label_id",
            "id",
            "target",
        ],
    )

    out = pd.DataFrame()
    out["f1"] = pd.to_numeric(df[f1_col], errors="coerce")

    if name_col is not None:
        out["activity"] = df[name_col].astype(str).str.strip()
    elif id_col is not None:
        ids = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
        out["activity"] = ids.map(ID_TO_ACTIVITY)
    else:
        if len(df) == len(ACTIVITY_ORDER):
            out["activity"] = ACTIVITY_ORDER
        else:
            raise RuntimeError(
                f"Could not detect activity/class column in {path}. Columns: {list(df.columns)}"
            )

    out = out.dropna(subset=["f1"])
    out = out[out["activity"].isin(ACTIVITY_ORDER)].copy()

    # Some files may store F1 as percent. Convert to 0--1 if needed.
    if out["f1"].mean() > 1.5:
        out["f1"] = out["f1"] / 100.0

    return out[["activity", "f1"]]


def collect():
    all_rows = []

    for rate, root in ROOTS.items():
        fold_dirs = sorted(glob.glob(os.path.join(root, "loso_sbj_*")), key=fold_key)

        if not fold_dirs:
            print(f"WARNING: no fold dirs found for {rate} Hz under {root}")
            continue

        for fold_dir in fold_dirs:
            split = os.path.basename(fold_dir)

            # Standard CNN separate-rate run usually has exactly one per_class_metrics file per fold.
            files = sorted(glob.glob(os.path.join(fold_dir, "per_class_metrics_*.csv")))

            # Avoid accidentally reading multirate suffix files if they exist.
            # For standard_cnn/rate-specific folders, we want the base fold file.
            base_candidates = [
                p for p in files
                if os.path.basename(p) == f"per_class_metrics_{split}.csv"
            ]

            if base_candidates:
                csv_path = base_candidates[0]
            elif files:
                csv_path = files[0]
                print(f"WARNING: using first per-class file for {split}, {rate} Hz: {csv_path}")
            else:
                print(f"Missing per-class CSV: {fold_dir}")
                continue

            df = read_per_class_csv(csv_path)
            df["rate"] = rate
            df["split"] = split
            df["csv_path"] = csv_path
            all_rows.append(df)

    if not all_rows:
        raise RuntimeError("No per-class CSVs found.")

    raw = pd.concat(all_rows, ignore_index=True)

    print("Found LOSO folds per rate:")
    print(raw.groupby("rate")["split"].nunique().reindex(["6", "12", "25", "50"]))
    print()

    mean_table = (
        raw.groupby(["activity", "rate"], as_index=False)["f1"]
        .mean()
        .pivot(index="activity", columns="rate", values="f1")
    )

    mean_table = mean_table.reindex(ACTIVITY_ORDER)

    for rate in ["6", "12", "25", "50"]:
        if rate not in mean_table.columns:
            mean_table[rate] = np.nan

    mean_table = mean_table[["6", "12", "25", "50"]]

    best_rate = mean_table.idxmax(axis=1)
    best_f1 = mean_table.max(axis=1)
    tradeoff = best_f1 - mean_table["50"]

    result = mean_table.copy()
    result["Trade-off"] = tradeoff
    result["Opt. rate"] = best_rate + " Hz"

    return raw, result


def fmt_num(x, bold=False):
    if pd.isna(x):
        s = "--"
    else:
        s = f"{x:.3f}"

    if bold and s != "--":
        return r"\textbf{" + s + "}"
    return s


def make_latex(result):
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Per-activity sensitivity to the sampling rate on WEAR using the CNN-only baseline. Values are mean per-class $F_1$ across 22 LOSO subjects. ``$F_1$ trade-off'' denotes the gain of the activity-optimal sampling rate over $50$\,Hz, and is zero when $50$\,Hz is optimal.}"
    )
    lines.append(r"  \label{tab:freq_perclass_wear_standard_cnn}")
    lines.append(r"  \begin{tabular}{lcccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Activity & $F_1$@6 & $F_1$@12 & $F_1$@25 & $F_1$@50 & Trade-off & Opt.\ rate \\")
    lines.append(r"    \midrule")

    for activity in ACTIVITY_ORDER:
        row = result.loc[activity]
        vals = row[["6", "12", "25", "50"]].astype(float)
        best = vals.max()
        bold = vals >= (best - BOLD_WITHIN)

        line = (
            f"    {activity:<30} & "
            f"{fmt_num(vals['6'], bold['6'])} & "
            f"{fmt_num(vals['12'], bold['12'])} & "
            f"{fmt_num(vals['25'], bold['25'])} & "
            f"{fmt_num(vals['50'], bold['50'])} & "
            f"{fmt_num(row['Trade-off'])} & "
            f"{row['Opt. rate']} \\\\"
        )
        lines.append(line)

    macro = result[["6", "12", "25", "50"]].mean(axis=0)
    best_macro = macro.max()
    bold_macro = macro >= (best_macro - BOLD_WITHIN)

    lines.append(r"    \midrule")
    lines.append(
        f"    Macro mean                     & "
        f"{fmt_num(macro['6'], bold_macro['6'])} & "
        f"{fmt_num(macro['12'], bold_macro['12'])} & "
        f"{fmt_num(macro['25'], bold_macro['25'])} & "
        f"{fmt_num(macro['50'], bold_macro['50'])} & "
        f"{fmt_num(best_macro - macro['50'])} & "
        f"{macro.idxmax()} Hz \\\\"
    )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    raw, result = collect()

    raw_path = os.path.join(OUT_DIR, "standard_cnn_per_activity_f1_raw_all_folds.csv")
    csv_path = os.path.join(OUT_DIR, "standard_cnn_per_activity_f1.csv")
    tex_path = os.path.join(OUT_DIR, "standard_cnn_per_activity_f1_table.tex")

    raw.to_csv(raw_path, index=False, float_format="%.6f")
    result.to_csv(csv_path, float_format="%.6f")

    latex = make_latex(result)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex + "\n")

    print("Saved:")
    print(f"  {raw_path}")
    print(f"  {csv_path}")
    print(f"  {tex_path}")
    print()
    print("LaTeX table:")
    print(latex)


if __name__ == "__main__":
    main()
