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
