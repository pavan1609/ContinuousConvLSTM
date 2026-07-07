## Folder structure

```text
configs/
  DeepConvLSTM/
    WEAR/
    WISDM_WATCH/
    RWHAR/
  ContinuousConvLSTM/
    WEAR/
    WISDM_WATCH/
    RWHAR/
  PureCNN/
    WEAR/
    WISDM_WATCH/
    RWHAR/
  Ablations/
    SingleBranch/
```

## Rates included

- WEAR: `6hz`, `12hz`, `25hz`, `50hz`, `multirate`
- WISDM_WATCH: `5hz`, `10hz`, `20hz`, `multirate`
- RWHAR: `6hz`, `12hz`, `25hz`, `50hz`, `multirate`

## Internal path normalization

Every YAML config uses the same clean internal hierarchy.

```yaml
dataset:
  sens_folder: data/<dataset>/<rate-or-multirate>

train_cfg:
  log_subdir: <architecture>/<dataset>/<rate-or-multirate>
```

Example:

```yaml
dataset:
  sens_folder: data/wisdm_watch/10hz
train_cfg:
  log_subdir: deepconvlstm/wisdm_watch/10hz
```

## Ablations

`Ablations/SingleBranch/` trains a single-branch ContinuousConvLSTM (`conv_type: continuous_single`) at one fixed rate (Appendix C).

## Notes

- WISDM-watch configs use the 3-axis accelerometer only (`input_dim: 3`).
- Continuous configs derive the temporal support as `conv_kernel_size / sampling_rate` (0.18 s at 50 Hz; 0.45 s at 20 Hz).
