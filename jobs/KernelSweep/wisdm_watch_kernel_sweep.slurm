#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --job-name=sweep_wisdm_watch
#SBATCH --array=1-51
#SBATCH --output=slurm_logs/sweep_wisdm_watch_%A_%a.out
#SBATCH --error=slurm_logs/sweep_wisdm_watch_%A_%a.err

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
SPLIT=${SLURM_ARRAY_TASK_ID}

# Discrete temporal kernel sweep (k_t = 1..11, odd) over the per-rate DeepConvLSTM configs.
python scripts/run_kernel_sweep_from_configs.py \
  --configs "configs/DeepConvLSTM/WISDM_WATCH/wisdm_watch_20hz.yaml,configs/DeepConvLSTM/WISDM_WATCH/wisdm_watch_10hz.yaml,configs/DeepConvLSTM/WISDM_WATCH/wisdm_watch_5hz.yaml" \
  --exp_tag "kernel_sweep_wisdm_watch" \
  --kernel_min 1 --kernel_max 11 --kernel_step 2 --odd_only \
  --seed 1 \
  --start_split ${SPLIT} \
  --end_split ${SPLIT}
