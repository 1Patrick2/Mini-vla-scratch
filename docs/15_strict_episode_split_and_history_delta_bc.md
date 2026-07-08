# Stage 7: Strict Episode Split + History/Delta BC

## Motivation

Stage 6 proved normalized PushT pipeline works. Stage 7 moves from
"pipeline works" to "reproducible benchmark":

- Strict episode-level train/eval split
- Train-only normalization statistics
- SingleFrame vs History vs Delta comparison
- Reproducible report with split metadata

## Strict Split Protocol

- Dataset: `lerobot/pusht` (1024 samples)
- Split: episode-level, 80/20, seed 42
- Train episodes: ~819 samples
- Eval episodes: ~205 samples (strictly unseen during training)
- Normalization stats: computed from **train split only**

## Models

| Model | Input | Target | Notes |
|-------|-------|--------|-------|
| SingleFrame | image + state + text | action | Stage 6 baseline |
| History | image + state + prev_state + prev_action + text | action | temporal context |
| DeltaAction | image + state + prev_state + prev_action + text | action - prev_action | residual prediction |

## Results

All metrics in **raw action space** on held-out episodes (strict split).

| Method | Raw MAE | Raw RMSE | Beats Zero | Beats Mean | Beats Previous | Notes |
|--------|-------:|--------:|:----------:|:----------:|:--------------:|-------|
| Zero Action | 249.50 | — | — | — | — | All-zero prediction |
| Mean Action | 92.32 | — | — | — | — | Train split mean |
| Previous Action | 6.67 | — | — | — | — | Strong temporal baseline |
| SingleFrame | 19.51 | 24.23 | ✅ | ✅ | ❌ | Single-step BC |
| History | 11.02 | 15.82 | ✅ | ✅ | ❌ | +prev state/action |
| DeltaAction | — | — | — | — | — | Pending Stage 7-E |

### Interpretation

History BC reduces raw MAE by **43.5%** compared to SingleFrame
(from 19.51 to 11.02), demonstrating that previous state/action
provides useful temporal context.

However, History still does not outperform the previous-action baseline
(6.67), indicating that PushT action continuity remains a strong prior.
This motivates Delta / residual action prediction in Stage 7-E, where
the model learns corrections to the previous action rather than
predicting the absolute action.

## Commands

### Create split

```bash
python scripts/create_episode_split.py --dataset-name pusht --repo-id lerobot/pusht --loader lerobot --max-samples 1024 --train-ratio 0.8 --seed 42 --output outputs/splits/pusht_seed42_80_20_split.json
```

### Compute train-only stats

```bash
python scripts/compute_dataset_stats.py --dataset-name pusht --repo-id lerobot/pusht --loader lerobot --max-samples 1024 --split-path outputs/splits/pusht_seed42_80_20_split.json --split train --output outputs/dataset_reports/pusht_strict_train_stats.json
```

### Train SingleFrame

```bash
python scripts/train.py --config configs/train/pusht_strict_single_frame.yaml
```

### Evaluate SingleFrame

```bash
python scripts/evaluate_robot_dataset.py --config configs/train/pusht_strict_single_frame.yaml --ckpt outputs/runs/pusht_strict_single_frame_seed42/checkpoints/best.pt --dataset-name pusht --repo-id lerobot/pusht --loader lerobot --split-path outputs/splits/pusht_seed42_80_20_split.json --split eval --output outputs/eval/pusht_strict_single_frame_report.json
```

### Train History

```bash
python scripts/train.py --config configs/train/pusht_strict_history.yaml
```

### Evaluate History

```bash
python scripts/evaluate_robot_dataset.py --config configs/train/pusht_strict_history.yaml --ckpt outputs/runs/pusht_strict_history_seed42/checkpoints/best.pt --dataset-name pusht --repo-id lerobot/pusht --loader lerobot --split-path outputs/splits/pusht_seed42_80_20_split.json --split eval --output outputs/eval/pusht_strict_history_report.json
```

## Limitations

- Previous-action baseline is very strong for PushT; History improves but
  does not surpass it. Delta/residual prediction (Stage 7-E) is expected
  to address this.
- Single experiment per method (seed=42); multiple seeds would strengthen
  reproducibility claims.
- PushT is a single task; generalization to ALOHA (action_dim=14) or
  LIBERO (multi-task) is deferred.

## Next: Stage 7-E Delta / Residual Action BC

Predict delta_action = action_t - prev_action_t instead of action_t directly,
with the goal of challenging the previous-action baseline.
