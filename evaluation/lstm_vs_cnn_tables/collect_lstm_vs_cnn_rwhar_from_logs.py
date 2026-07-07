from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT = Path("/home/g263412/GAMMA")
LOG_DIR = ROOT / "slurm_logs"
OUT = ROOT / "results/lstm_vs_cnn_rwhar"
OUT.mkdir(parents=True, exist_ok=True)

records = []

for path in sorted(LOG_DIR.glob("lstm_vs_cnn_*118*.out")):
    txt = path.read_text(errors="ignore")

    m_dataset = re.search(r"Dataset:\s*(\S+)", txt)
    m_model = re.search(r"Model:\s*(\S+)", txt)
    m_rate = re.search(r"Rate:\s*(\d+)", txt)

    if not (m_dataset and m_model and m_rate):
        continue

    dataset = m_dataset.group(1)
    model = m_model.group(1)
    rate = int(m_rate.group(1))

    if dataset != "rwhar":
        continue

    if model not in {"deepconvlstm", "pure_cnn"}:
        continue

    # Prefer the explicit training line.
    m_split = re.search(r"Split\s+1\s*/\s*1\s+\(\d+hz\)\s*->\s*(loso_sbj_\d+)", txt)

    # Fallback: use any loso_sbj_X mention.
    if m_split:
        split = m_split.group(1)
    else:
        matches = re.findall(r"loso_sbj_\d+", txt)
        split = matches[-1] if matches else None

    if split is None:
        continue

    f1s = []

    # Best checkpoint lines.
    for m in re.finditer(r"New best selection F1\s+([0-9]+(?:\.[0-9]+)?)%", txt):
        f1s.append(float(m.group(1)) / 100.0)

    # Fallback validation lines.
    if not f1s:
        for m in re.finditer(r"F1\s+([0-9]+(?:\.[0-9]+)?)\s*\(%\)", txt):
            f1s.append(float(m.group(1)) / 100.0)

    best_f1 = max(f1s) if f1s else np.nan

    finished = "Number of splits trained: 1" in txt or "Finished LOSO training" in txt

    records.append({
        "dataset": dataset,
        "model": model,
        "rate_hz": rate,
        "split": split,
        "best_f1": best_f1,
        "finished": finished,
        "log_file": str(path.relative_to(ROOT)),
    })

df = pd.DataFrame(records)

if df.empty:
    raise SystemExit("No matching RealWorld-HAR logs found.")

# Keep one result per dataset/model/rate/split.
# If a split was run more than once, keep the completed run with the highest F1.
df = df.sort_values(
    ["dataset", "model", "rate_hz", "split", "finished", "best_f1"],
    ascending=[True, True, True, True, True, True],
)

df = df.drop_duplicates(
    ["dataset", "model", "rate_hz", "split"],
    keep="last",
)

df.to_csv(OUT / "rwhar_lstm_vs_cnn_by_split.csv", index=False)

summary = (
    df.groupby(["dataset", "model", "rate_hz"])
      .agg(
          folds=("split", "nunique"),
          finished=("finished", "sum"),
          mean=("best_f1", "mean"),
          std=("best_f1", "std"),
          min=("best_f1", "min"),
          max=("best_f1", "max"),
          missing_f1=("best_f1", lambda x: int(pd.isna(x).sum())),
      )
      .reset_index()
)

summary["rate_hz"] = summary["rate_hz"].astype(int)

model_order = {"deepconvlstm": 0, "pure_cnn": 1}
rate_order = {50: 0, 25: 1, 12: 2, 6: 3}

summary["model_order"] = summary["model"].map(model_order)
summary["rate_order"] = summary["rate_hz"].map(rate_order)
summary = summary.sort_values(["model_order", "rate_order"]).drop(columns=["model_order", "rate_order"])

summary.to_csv(OUT / "rwhar_lstm_vs_cnn_summary.csv", index=False)

print("Saved:", OUT)
print()
print("Summary:")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# Paper rows: 50, 25, 12, 6
rates = [50, 25, 12, 6]
models = ["deepconvlstm", "pure_cnn"]

vals = {}
stds = {}

for _, row in summary.iterrows():
    vals[(row["model"], int(row["rate_hz"]))] = float(row["mean"])
    stds[(row["model"], int(row["rate_hz"]))] = float(row["std"])

latex_path = OUT / "rwhar_lstm_vs_cnn_latex_rows.txt"

with open(latex_path, "w", encoding="utf-8") as f:
    f.write("\\multirow{3}{*}{RealWorld-HAR}\n")

    for model in models:
        pretty = "DeepConvLSTM" if model == "deepconvlstm" else "Pure CNN"
        row_vals = []
        for r in rates:
            v = vals.get((model, r), np.nan)
            row_vals.append("--" if pd.isna(v) else f"{v:.3f}")
        f.write(f" & {pretty} & " + " & ".join(row_vals) + " \\\\\n")

    delta_vals = []
    for r in rates:
        v_cnn = vals.get(("pure_cnn", r), np.nan)
        v_lstm = vals.get(("deepconvlstm", r), np.nan)
        if pd.isna(v_cnn) or pd.isna(v_lstm):
            delta_vals.append("--")
        else:
            delta_vals.append(f"${v_cnn - v_lstm:+.3f}$")

    f.write(" & $\\Delta F_1$ & " + " & ".join(delta_vals) + " \\\\\n")

print()
print("LaTeX rows:")
print(latex_path.read_text())

print()
print("Completion check:")
for model in models:
    for rate in rates:
        sub = df[(df["model"] == model) & (df["rate_hz"] == rate)]
        print(
            f"{model:13s} {rate:2d}Hz "
            f"folds={sub['split'].nunique():2d} "
            f"finished={int(sub['finished'].sum()):2d} "
            f"missing_f1={int(sub['best_f1'].isna().sum()):2d}"
        )
