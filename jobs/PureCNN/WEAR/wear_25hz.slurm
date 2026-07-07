#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=purecnn_wear_25hz
#SBATCH --array=1-22
#SBATCH --output=slurm_logs/purecnn_wear_25hz_%A_%a.out
#SBATCH --error=slurm_logs/purecnn_wear_25hz_%A_%a.err

# ------------------------------------------------------------------
# EDIT ME: adjust these two lines for your cluster / environment.
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
CONDA_ENV="${CONDA_ENV:-gamma}"
# ------------------------------------------------------------------

set -eo pipefail
cd "$PROJECT_ROOT"
mkdir -p slurm_logs

source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then module load miniconda3/4.9.2 || true; fi
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV"; fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

SPLIT=${SLURM_ARRAY_TASK_ID}

python scripts/run_loso.py \
  --config "configs/PureCNN/WEAR/wear_25hz.yaml" \
  --seed 1 \
  --start_split ${SPLIT} \
  --end_split ${SPLIT}
