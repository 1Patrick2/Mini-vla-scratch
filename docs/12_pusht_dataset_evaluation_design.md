# PushT Dataset Evaluation Design

## Why Stage 5 Changed From FakeRobot to PushT

Stage 5 was originally planned as "Fake Robot Rollout" — a continuous action
prediction loop with a simulated robot state update.  However, the user's
goal is not just to "run a loop" but to **prove MiniVLA works on real
benchmark data**.

PushT (`lerobot/pusht`) is a standard LeRobot robotics dataset with:

- 206 episodes / 25,650 frames
- Image: 96×96×3 (resized to 64×64 for MiniVLA)
- State: float32, shape [2]
- Action: float32, shape [2]

These dimensions match Toy2D's state=2 / action=2 almost exactly,
making it the ideal stepping stone from synthetic Toy2D to a real
robot-learning benchmark.

## PushT Dataset Fields

From the Hugging Face dataset card:

| Field | Shape / Type | Description |
|-------|-------------|-------------|
| `observation.image` | [96, 96, 3] uint8 | Top-down view of the PushT workspace |
| `observation.state` | [2] float32 | T-block (x, y) position |
| `action` | [2] float32 | Expert push action (dx, dy) |
| `episode_index` | int | Which episode this frame belongs to |
| `frame_index` | int | Frame within the episode |
| `timestamp` | float | Timestamp in seconds |
| `next.reward` | float | Reward (1.0 on success, 0.0 otherwise) |
| `next.done` | bool | True when episode ends |
| `next.success` | bool | True when task succeeded |
| `task_index` | int | Task ID (single-task: always 0) |

## PushT vs Toy2D Correspondence

| Aspect | Toy2D | PushT |
|--------|-------|-------|
| Image size | 64×64 RGB | 96×96 RGB → resize to 64×64 |
| State | object [x, y] | T-block [x, y] |
| Action | step [dx, dy] | push [dx, dy] |
| Instruction | "move red object…" | "push the T block to the target" (fixed) |
| Episodes | configurable | 206 |
| Frames | configurable | ~25,650 |

## Offline Evaluation: What It Is and Why

**Offline evaluation** means we load a trained model, run it over a held-out
set of dataset samples, and compare its predictions to the ground-truth
expert actions — **without** executing the actions in a simulator or on a
real robot.

This is a standard first step in robot learning (LeRobot itself evaluates
checkpoints this way).  It separates model quality assessment from the
infrastructure of environment rollout.

**Limitations:**
- Offline evaluation cannot measure closed-loop success (the model's action
  affects the next observation in a real deployment).
- It can only measure one-step prediction accuracy.

**What it does prove:**
- The model's action predictions are closer to expert actions than simple
  baselines (zero, mean, previous-action).
- The model's output is finite, non-NaN, and in a reasonable range.
- Per-dimension error patterns are visible.

## Episode-Level Split

PushT has only a training split, but it includes `episode_index`.
We split by episode to avoid information leakage between consecutive frames:

```
train_episodes: 0–164  (80% of episodes)
test_episodes:  165–205 (20% of episodes)
```

## Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| MAE / L1 | `mean(|pred - gt|)` | Primary accuracy metric |
| MSE | `mean((pred - gt)²)` | Squared error, penalises large outliers |
| RMSE | `sqrt(MSE)` | MSE in original units |
| Per-dim MAE | MAE per action dimension | Which axis is harder to predict |
| Cosine similarity | `cos(pred, gt)` | Directional agreement (0–1) |
| Finite ratio | `all(isfinite(pred))` | NaN/Inf guard |
| Clip rate | `mean(pred != clipped)` | How often clipping kicks in |

## Baselines

| Baseline | Formula | Why |
|----------|---------|-----|
| ZeroAction | `[0, 0]` | Minimal sanity — model must beat always-zero |
| MeanAction | `mean(train_actions)` | Model must beat always-average |
| PreviousAction | `action_{t-1}` | Strong baseline for continuous trajectory |

The PreviousAction baseline is particularly strong for PushT because
consecutive frames have highly correlated actions.  MiniVLA may not beat it
initially, but it **must** beat Zero and Mean.

## Outputs

```
outputs/eval/pusht_report.json          # Aggregate metrics
outputs/eval/pusht_predictions.jsonl    # Per-sample predictions
outputs/eval/pusht_prediction_grid.png  # Optional sample visualizations
```

## Out of Scope

- Reinforcement Learning (no reward-based optimisation)
- Real robot control
- ACT / Diffusion Policy / SmolVLA
- Online PushT environment rollout (simulation)
- Open Kaka robot adapter

## References

- [lerobot/pusht on Hugging Face](https://huggingface.co/datasets/lerobot/pusht)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot/index)

---

## Final Stage 5 Validation Results

### Default tests (no external dependencies)

```bash
pytest tests/test_pusht_adapter.py tests/test_pusht_inspection.py \
  tests/test_pusht_model_forward.py tests/test_pusht_training_smoke.py \
  tests/test_action_metrics.py tests/test_evaluation_baselines.py \
  tests/test_pusht_evaluator.py tests/test_evaluate_pusht_cli.py
```

```
76 passed, 1 skipped  (realdata marker, requires RUN_REAL_PUSHT=1)
```

### Real PushT vision inspect

```bash
python scripts/inspect_pusht_dataset.py \
  --repo-id lerobot/pusht --loader lerobot \
  --max-samples 8 --output outputs/dataset_reports/pusht_lerobot_report.json
```

```
has_image:         true
matched_image_key: observation.image
image_shape:       [3, 96, 96]   (CHW torch.Tensor)
state_shape:       [2]
action_shape:      [2]
```

### Real PushT debug training

```bash
python scripts/train.py --config configs/train/pusht_debug.yaml
```

```
Epoch 1/1  loss=13985.03  mae=86.85
checkpoints saved to outputs/checkpoints/best.pt and last.pt
```

Note: loss and MAE appear large because PushT action values are in pixel/coordinate
scale (e.g. 0-255), unlike Toy2D's small normalized actions (~0.03).  Action
normalisation is a planned enhancement (Stage 6).

### Real PushT offline evaluation

```bash
python scripts/evaluate_pusht.py \
  --config configs/train/pusht_debug.yaml \
  --ckpt outputs/checkpoints/best.pt \
  --repo-id lerobot/pusht --loader lerobot \
  --max-samples 128 --heldout-ratio 0.2 \
  --output outputs/eval/pusht_real_report.json \
  --predictions-output outputs/eval/pusht_real_predictions.jsonl \
  --no-clip-action
```

```
num_samples=25 (held-out)
model MAE:          42.28
zero_action MAE:   205.46  →  model beats zero baseline ✓
mean_action MAE:    93.35  →  model beats mean baseline ✓
previous_action MAE: 9.10  →  model does not beat (expected)
```

PreviousActionBaseline is a strong baseline for continuous trajectories
because consecutive frames have highly correlated actions.  The model
exceeding zero and mean baselines confirms it has learned meaningful
action prediction beyond trivial strategies.

### Real PushT pytest (requires `$env:RUN_REAL_PUSHT=1`)

```powershell
$env:RUN_REAL_PUSHT="1"
python -m pytest tests/test_evaluate_pusht_cli.py -m realdata
```

```
1 passed, 4 deselected
```

### What Stage 5 covers

- Real PushT vision data loading via LeRobotDataset
- PushT → MiniVLA sample adapter (image resize, tokenization)
- Behavior cloning training on real PushT data
- Held-out episode evaluation (no frame-level leakage)
- Offline evaluation with zero, mean, previous-action baselines
- MAE / MSE / RMSE / per-dim MAE / cosine similarity / finite ratio
- Mock-data CLI test (no external deps)
- Real-data CLI test (optional, `RUN_REAL_PUSHT=1`)
- Evaluation report saved as JSON (model + baselines + per-dim metrics)

### What Stage 5 does not cover

| Feature | Reason |
|---------|--------|
| Action normalisation | PushT actions are in pixel coordinates; normalisation deferred to Stage 6 |
| Longer training (>1 epoch) | Debug config intentionally minimal for smoke test |
| Temporal context | MiniVLA is a single-step model; temporal context is a V1+ direction |
| Stronger vision encoder | SmallCNN is sufficient for smoke; ResNet/TinyBERT deferred |
| Prediction visualization | PushT arrow visualization shares Toy2D visualizer; enhancement deferred |
| Online (simulation) rollout | Requires PushT environment; deferred to Stage 6+ |
| RL / ACT / Diffusion Policy | Out of scope for Stage 5 — offline action prediction only |
