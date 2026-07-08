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
- Train split: 6 episodes, 745 samples
- Eval split: 2 episodes, 279 samples (strictly unseen during training)
- Normalization stats: computed from **train split only**

## Models

| Model | Input | Target | Notes |
|-------|-------|--------|-------|
| SingleFrame | image + state + text | action | Stage 6 baseline |
| History | image + state + prev_state + prev_action + text | action | temporal context |
| DeltaAction | image + state + prev_state + prev_action + text | action - prev_action | residual prediction |

## Results

All metrics in **raw action space** on held-out episodes (strict split).

| Method | Raw MAE | Raw RMSE | Notes |
|--------|-------:|--------:|-------|
| Previous-action baseline | 6.67 | — | strong temporal reference |
| SingleFrame BC | 19.51 | 24.23 | current-frame baseline |
| History BC | 11.02 | 15.82 | +temporal context |
| DeltaAction BC | **7.35** | **13.51** | **best learned variant** |

### Key Takeaway

DeltaAction BC is the strongest learned model in Stage 7, reducing raw
MAE by **33.3%** over History BC (11.02 → 7.35) and by **62.3%** over
SingleFrame BC (19.51 → 7.35). The previous-action baseline (6.67)
is reported as a reference — it reflects the strong temporal continuity
inherent to the PushT dataset rather than a learned policy.

### Improvement Summary

| Comparison | MAE Reduction |
|---|---:|
| History vs SingleFrame | 43.5% |
| DeltaAction vs History | 33.3% |
| DeltaAction vs SingleFrame | 62.3% |

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

- DeltaAction narrows the gap to the previous-action baseline (7.35 vs 6.67)
  compared to History (11.02), but does not surpass it under the current
  5-epoch CPU training setup. Longer training or stronger models may bridge
  the gap.
- Single experiment per method (seed=42); multiple seeds would strengthen
  reproducibility claims.
- PushT is a single task; generalization to ALOHA (action_dim=14) or
  LIBERO (multi-task) is deferred.

### Evaluate Delta

```bash
python scripts/evaluate_robot_dataset.py --config configs/train/pusht_strict_delta.yaml --ckpt outputs/runs/pusht_strict_delta_seed42/checkpoints/best.pt --dataset-name pusht --repo-id lerobot/pusht --loader lerobot --split-path outputs/splits/pusht_seed42_80_20_split.json --split eval --output outputs/eval/pusht_strict_delta_report.json
```
