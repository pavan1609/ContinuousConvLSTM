# SLURM jobs

```text
jobs/<Architecture>/<DATASET>/<config-name>.slurm   # LOSO array job (1 task per held-out subject)
jobs/KernelSweep/<dataset>_kernel_sweep.slurm       # discrete k_t sweep for Fig. 4
jobs/Ablations/SingleBranch/*.slurm                 # Appendix C single-branch ablation
```

Each file has a two-line `EDIT ME` block (`PROJECT_ROOT`, `CONDA_ENV`); 
everything else is portable. Array sizes match the LOSO fold counts (WEAR 22, RealWorld-HAR 15, WISDM-watch 51). 
Submit with `sbatch <file>`; single folds run locally via `python scripts/run_loso.py --config <cfg> --start_split N --end_split N`.
