import os
import glob
import math
import torch
import pandas as pd
from collections import defaultdict

ROOT = "logs/deepconvlstm/standard_cnn_nolstm"
OUTDIR = "evaluation/standard_cnn_nolstm_tables"

RATES = [6, 12, 25, 50]
RATE_DIRS = {6: "6hz", 12: "12hz", 25: "25hz", 50: "50hz"}

os.makedirs(OUTDIR, exist_ok=True)


def load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_f1_value(x):
    x = float(x)
    if math.isnan(x):
        return x
    # Logs often store F1 as percent, e.g. 78.9. Paper table should use 0.789.
    if x > 1.5:
        return x / 100.0
    return x


def detect_col(df, exact_names=None, contains=None, numeric_fallback=False):
    exact_names = exact_names or []
    contains = contains or []

    lower_map = {str(c).lower().strip(): c for c in df.columns}

    for name in exact_names:
        key = name.lower().strip()
        if key in lower_map:
            return lower_map[key]

    for c in df.columns:
        lc = str(c).lower().strip()
        if any(s in lc for s in contains):
            return c

    if numeric_fallback:
        numeric_cols = []
        for c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().any():
                numeric_cols.append(c)
        if numeric_cols:
            return numeric_cols[-1]

    return None


def detect_activity_col(df):
    candidates = [
        "activity", "class", "label", "class_name", "activity_name",
        "name", "target", "gesture"
    ]
    lower_map = {str(c).lower().strip(): c for c in df.columns}

    for name in candidates:
        if name in lower_map:
            return lower_map[name]

    # Prefer first non-numeric column.
    for c in df.columns:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.isna().any():
            return c

    # Fallback: first column.
    return df.columns[0]


def latex_escape(s):
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def bold_within_best(values, tol=0.02):
    clean = [v for v in values if pd.notna(v)]
    if not clean:
        return ["" for _ in values]

    best = max(clean)
    out = []
    for v in values:
        if pd.isna(v):
            out.append("--")
        elif v >= best - tol - 1e-12:
            out.append(r"\textbf{" + f"{v:.3f}" + "}")
        else:
            out.append(f"{v:.3f}")
    return out


def lower_bound_opt_rate(values_by_rate, tol=0.02):
    clean = {r: v for r, v in values_by_rate.items() if pd.notna(v)}
    if not clean:
        return "--"
    best = max(clean.values())
    eligible = [r for r, v in clean.items() if v >= best - tol - 1e-12]
    return min(eligible)


# ---------------------------------------------------------------------
# 1. Completion and checkpoint/model-size audit
# ---------------------------------------------------------------------
model_rows = []
completion_rows = []

for rate in RATES:
    rate_dir = RATE_DIRS[rate]
    base = os.path.join(ROOT, rate_dir)

    ckpts = sorted(glob.glob(os.path.join(base, "loso_sbj_*", "best_loso_sbj_*.pth.tar")))
    macro_files = sorted(glob.glob(os.path.join(base, "loso_sbj_*", "best_macro_metrics_*.csv")))
    perclass_files = sorted(glob.glob(os.path.join(base, "loso_sbj_*", "per_class_metrics_*.csv")))

    completion_rows.append({
        "rate_hz": rate,
        "checkpoints": len(ckpts),
        "macro_csvs": len(macro_files),
        "per_class_csvs": len(perclass_files),
        "complete_22_folds": len(ckpts) == 22 and len(macro_files) == 22 and len(perclass_files) == 22,
    })

    if ckpts:
        p = ckpts[0]
        ck = load_torch(p)
        sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        keys = list(sd.keys())

        params = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
        lstm_keys = [k for k in keys if "lstm" in k.lower()]
        classifier_keys = [k for k in keys if k.startswith("classifier.")]
        cnn_classifier_keys = [k for k in keys if k.startswith("cnn_classifier.")]
        branch_keys = [k for k in keys if k.startswith("branches.")]
        gamma_keys = [
            k for k in keys
            if any(t in k.lower() for t in ["gamma", "quant", "quantizer", "ste"])
        ]

        model_rows.append({
            "rate_hz": rate,
            "example_checkpoint": p,
            "params": params,
            "fp32_mb": params * 4 / 1e6,
            "four_bit_mb_est": params * 0.5 / 1e6,
            "checkpoint_file_mb": os.path.getsize(p) / 1e6,
            "has_lstm_keys": bool(lstm_keys),
            "has_classifier_keys": bool(classifier_keys),
            "has_cnn_classifier_keys": bool(cnn_classifier_keys),
            "has_branch_keys": bool(branch_keys),
            "has_gamma_or_quant_keys": bool(gamma_keys),
            "num_lstm_keys": len(lstm_keys),
            "num_branch_keys": len(branch_keys),
            "num_gamma_or_quant_keys": len(gamma_keys),
        })

completion_df = pd.DataFrame(completion_rows)
model_df = pd.DataFrame(model_rows)

completion_df.to_csv(os.path.join(OUTDIR, "standard_cnn_nolstm_completion_summary.csv"), index=False)
model_df.to_csv(os.path.join(OUTDIR, "standard_cnn_nolstm_model_size_summary.csv"), index=False)


# ---------------------------------------------------------------------
# 2. Macro summary from best_macro_metrics_*.csv
# ---------------------------------------------------------------------
macro_raw = []

for rate in RATES:
    rate_dir = RATE_DIRS[rate]
    files = sorted(glob.glob(os.path.join(ROOT, rate_dir, "loso_sbj_*", "best_macro_metrics_*.csv")))

    for p in files:
        split = os.path.basename(os.path.dirname(p))

        df = pd.read_csv(p)
        if df.empty:
            continue

        row = df.iloc[-1]

        f1_col = detect_col(
            df,
            exact_names=["f1", "F1", "macro_f1", "Macro F1", "val_f1", "best_f1"],
            contains=["f1"],
            numeric_fallback=True,
        )
        acc_col = detect_col(
            df,
            exact_names=["acc", "accuracy", "Acc", "Accuracy"],
            contains=["acc"],
            numeric_fallback=False,
        )
        prec_col = detect_col(
            df,
            exact_names=["prec", "precision", "Prec", "Precision"],
            contains=["prec"],
            numeric_fallback=False,
        )
        rec_col = detect_col(
            df,
            exact_names=["rec", "recall", "Rec", "Recall"],
            contains=["rec"],
            numeric_fallback=False,
        )

        if f1_col is None:
            raise RuntimeError(f"Could not detect F1 column in {p}. Columns: {list(df.columns)}")

        out = {
            "rate_hz": rate,
            "split": split,
            "file": p,
            "f1": normalize_f1_value(row[f1_col]),
        }

        if acc_col is not None:
            out["acc"] = normalize_f1_value(row[acc_col])
        if prec_col is not None:
            out["precision"] = normalize_f1_value(row[prec_col])
        if rec_col is not None:
            out["recall"] = normalize_f1_value(row[rec_col])

        macro_raw.append(out)

macro_raw_df = pd.DataFrame(macro_raw)
macro_raw_df.to_csv(os.path.join(OUTDIR, "standard_cnn_nolstm_macro_raw_all_folds.csv"), index=False)

macro_rows = []
for rate in RATES:
    sub = macro_raw_df[macro_raw_df["rate_hz"] == rate]
    vals = sub["f1"].dropna().astype(float)

    row = {
        "rate_hz": rate,
        "folds": len(vals),
        "macro_f1_mean": vals.mean() if len(vals) else float("nan"),
        "macro_f1_std": vals.std(ddof=1) if len(vals) > 1 else 0.0,
        "macro_f1_min": vals.min() if len(vals) else float("nan"),
        "macro_f1_max": vals.max() if len(vals) else float("nan"),
        "macro_f1_mean_percent": vals.mean() * 100 if len(vals) else float("nan"),
        "macro_f1_std_percent": vals.std(ddof=1) * 100 if len(vals) > 1 else 0.0,
    }

    if "acc" in sub.columns:
        acc_vals = sub["acc"].dropna().astype(float)
        row["acc_mean"] = acc_vals.mean() if len(acc_vals) else float("nan")
        row["acc_mean_percent"] = acc_vals.mean() * 100 if len(acc_vals) else float("nan")

    macro_rows.append(row)

macro_df = pd.DataFrame(macro_rows)

f50 = macro_df.loc[macro_df["rate_hz"] == 50, "macro_f1_mean"]
if not f50.empty and pd.notna(f50.iloc[0]):
    base50 = float(f50.iloc[0])
    macro_df["delta_vs_50"] = macro_df["macro_f1_mean"] - base50
    macro_df["delta_vs_50_percent_points"] = macro_df["delta_vs_50"] * 100

macro_df.to_csv(os.path.join(OUTDIR, "standard_cnn_nolstm_macro_summary.csv"), index=False)


# ---------------------------------------------------------------------
# 3. Per-activity table
# ---------------------------------------------------------------------
per_rows = []

for rate in RATES:
    rate_dir = RATE_DIRS[rate]
    files = sorted(glob.glob(os.path.join(ROOT, rate_dir, "loso_sbj_*", "per_class_metrics_*.csv")))

    for p in files:
        split = os.path.basename(os.path.dirname(p))

        df = pd.read_csv(p)
        if df.empty:
            continue

        act_col = detect_activity_col(df)
        f1_col = detect_col(
            df,
            exact_names=["f1", "F1", "f1_score", "f1-score", "F1-score"],
            contains=["f1"],
            numeric_fallback=True,
        )

        if f1_col is None:
            raise RuntimeError(f"Could not detect per-class F1 column in {p}. Columns: {list(df.columns)}")

        for _, row in df.iterrows():
            activity = str(row[act_col])
            f1 = normalize_f1_value(row[f1_col])

            per_rows.append({
                "rate_hz": rate,
                "split": split,
                "activity": activity,
                "f1": f1,
                "file": p,
            })

per_raw_df = pd.DataFrame(per_rows)
per_raw_df.to_csv(os.path.join(OUTDIR, "standard_cnn_nolstm_per_activity_f1_raw_all_folds.csv"), index=False)

per_mean = (
    per_raw_df
    .groupby(["activity", "rate_hz"], as_index=False)["f1"]
    .mean()
)

wide = per_mean.pivot(index="activity", columns="rate_hz", values="f1").reset_index()

for rate in RATES:
    if rate not in wide.columns:
        wide[rate] = float("nan")

wide = wide[["activity"] + RATES]

# Add lower-bound optimal rate among values within 0.02 of best.
wide["opt_rate_lower_bound"] = wide.apply(
    lambda r: lower_bound_opt_rate({rate: r[rate] for rate in RATES}, tol=0.02),
    axis=1,
)

wide.to_csv(os.path.join(OUTDIR, "standard_cnn_nolstm_per_activity_f1.csv"), index=False)


# ---------------------------------------------------------------------
# 4. LaTeX tables
# ---------------------------------------------------------------------
# Per-activity LaTeX table.
latex_lines = []
latex_lines.append(r"\begin{table}[t]")
latex_lines.append(r"  \centering")
latex_lines.append(r"  \small")
latex_lines.append(
    r"  \caption{Per-activity sensitivity to the sampling rate on WEAR using the pure CNN baseline "
    r"(mean per-class $F_1$ across 22 LOSO subjects). The best scores per activity, as well as scores "
    r"within 0.02 absolute $F_1$ of the best, are shown in bold. The optimal rate reports the lowest "
    r"sampling rate within this tolerance.}"
)
latex_lines.append(r"  \label{tab:standard_cnn_nolstm_perclass_wear}")
latex_lines.append(r"  \begin{tabularx}{\linewidth}{lccccc}")
latex_lines.append(r"    \toprule")
latex_lines.append(r"    Activity & $F_1$@6 & $F_1$@12 & $F_1$@25 & $F_1$@50 & Opt.\ rate \\")
latex_lines.append(r"    \midrule")

for _, row in wide.sort_values("activity").iterrows():
    vals = [row[6], row[12], row[25], row[50]]
    bvals = bold_within_best(vals, tol=0.02)
    opt = row["opt_rate_lower_bound"]
    opt_s = "--" if opt == "--" else f"{int(opt)} Hz"

    latex_lines.append(
        "    "
        + latex_escape(row["activity"])
        + " & "
        + " & ".join(bvals)
        + " & "
        + opt_s
        + r" \\"
    )

macro_vals_by_rate = {}
for rate in RATES:
    macro_vals_by_rate[rate] = wide[rate].mean(skipna=True)

macro_vals = [macro_vals_by_rate[6], macro_vals_by_rate[12], macro_vals_by_rate[25], macro_vals_by_rate[50]]
macro_bold = bold_within_best(macro_vals, tol=0.02)
macro_opt = lower_bound_opt_rate(macro_vals_by_rate, tol=0.02)
macro_opt_s = "--" if macro_opt == "--" else f"{int(macro_opt)} Hz"

latex_lines.append(r"    \midrule")
latex_lines.append(
    r"    Macro mean & "
    + " & ".join(macro_bold)
    + " & "
    + macro_opt_s
    + r" \\"
)
latex_lines.append(r"    \bottomrule")
latex_lines.append(r"  \end{tabularx}")
latex_lines.append(r"\end{table}")

with open(os.path.join(OUTDIR, "standard_cnn_nolstm_per_activity_f1_table.tex"), "w") as f:
    f.write("\n".join(latex_lines) + "\n")


# Macro summary LaTeX table.
macro_latex = []
macro_latex.append(r"\begin{table}[t]")
macro_latex.append(r"  \centering")
macro_latex.append(r"  \small")
macro_latex.append(
    r"  \caption{Macro-level performance of the pure CNN baseline across sampling rates on WEAR. "
    r"Values are averaged over 22 LOSO folds.}"
)
macro_latex.append(r"  \label{tab:standard_cnn_nolstm_macro_wear}")
macro_latex.append(r"  \begin{tabular}{lrrrr}")
macro_latex.append(r"    \toprule")
macro_latex.append(r"    Rate & Folds & Mean $F_1$ & Std. & $\Delta$ vs. 50 Hz \\")
macro_latex.append(r"    \midrule")

for _, row in macro_df.sort_values("rate_hz").iterrows():
    rate = int(row["rate_hz"])
    folds = int(row["folds"])
    mean_f1 = row["macro_f1_mean"]
    std_f1 = row["macro_f1_std"]
    delta = row.get("delta_vs_50", float("nan"))

    macro_latex.append(
        f"    {rate} Hz & {folds} & {mean_f1:.3f} & {std_f1:.3f} & {delta:.3f} \\\\"
    )

macro_latex.append(r"    \bottomrule")
macro_latex.append(r"  \end{tabular}")
macro_latex.append(r"\end{table}")

with open(os.path.join(OUTDIR, "standard_cnn_nolstm_macro_summary_table.tex"), "w") as f:
    f.write("\n".join(macro_latex) + "\n")


# Model size LaTeX table.
size_latex = []
size_latex.append(r"\begin{table}[t]")
size_latex.append(r"  \centering")
size_latex.append(r"  \small")
size_latex.append(
    r"  \caption{Model size of the pure CNN baseline. Parameter counts are taken directly from "
    r"the saved checkpoints. The 4-bit size is an estimated weight-only storage cost.}"
)
size_latex.append(r"  \label{tab:standard_cnn_nolstm_size_wear}")
size_latex.append(r"  \begin{tabular}{lrrr}")
size_latex.append(r"    \toprule")
size_latex.append(r"    Rate & Parameters & FP32 MB & 4-bit MB \\")
size_latex.append(r"    \midrule")

for _, row in model_df.sort_values("rate_hz").iterrows():
    size_latex.append(
        f"    {int(row['rate_hz'])} Hz & "
        f"{int(row['params']):,} & "
        f"{row['fp32_mb']:.3f} & "
        f"{row['four_bit_mb_est']:.3f} \\\\"
    )

size_latex.append(r"    \bottomrule")
size_latex.append(r"  \end{tabular}")
size_latex.append(r"\end{table}")

with open(os.path.join(OUTDIR, "standard_cnn_nolstm_model_size_table.tex"), "w") as f:
    f.write("\n".join(size_latex) + "\n")


# ---------------------------------------------------------------------
# 5. Console summary
# ---------------------------------------------------------------------
print("=" * 100)
print("COMPLETION")
print("=" * 100)
print(completion_df.to_string(index=False))

print()
print("=" * 100)
print("MODEL SIZE / CHECKPOINT AUDIT")
print("=" * 100)
cols = [
    "rate_hz", "params", "fp32_mb", "four_bit_mb_est",
    "checkpoint_file_mb", "has_lstm_keys", "has_branch_keys",
    "has_gamma_or_quant_keys"
]
print(model_df[cols].sort_values("rate_hz").to_string(index=False))

print()
print("=" * 100)
print("MACRO F1 SUMMARY")
print("=" * 100)
show_cols = [
    "rate_hz", "folds", "macro_f1_mean", "macro_f1_std",
    "macro_f1_min", "macro_f1_max"
]
if "delta_vs_50" in macro_df.columns:
    show_cols.append("delta_vs_50")
print(macro_df[show_cols].sort_values("rate_hz").to_string(index=False))

print()
print("=" * 100)
print("WRITTEN FILES")
print("=" * 100)
for name in [
    "standard_cnn_nolstm_completion_summary.csv",
    "standard_cnn_nolstm_model_size_summary.csv",
    "standard_cnn_nolstm_macro_raw_all_folds.csv",
    "standard_cnn_nolstm_macro_summary.csv",
    "standard_cnn_nolstm_per_activity_f1_raw_all_folds.csv",
    "standard_cnn_nolstm_per_activity_f1.csv",
    "standard_cnn_nolstm_per_activity_f1_table.tex",
    "standard_cnn_nolstm_macro_summary_table.tex",
    "standard_cnn_nolstm_model_size_table.tex",
]:
    print(os.path.join(OUTDIR, name))
