# Logging and Observability Design

## 1. Logging Goal

Logging is used to make training, inference, visualization, and future robot rollout behavior **observable, debuggable, and reproducible**.

Without a logging convention, common project issues become hard to track:

- Which config / checkpoint / data_root was used for a run?
- Was action clipping applied?
- Did a forward pass produce NaN?
- Which stage of the pipeline is slow?
- In a rollout, was the failure caused by the model, the state update, or the success check?

This document establishes a logging convention **before** implementing concrete logging code.

---

## 2. Log Levels

Four levels, matching standard practice:

| Level | Purpose | Examples |
|-------|---------|----------|
| **DEBUG** | Detailed debug info, enabled only when needed | Tensor shapes, dtypes, value ranges, per-layer timing |
| **INFO** | Normal workflow milestones | Module start/end, checkpoint loaded/saved, inference completed |
| **WARNING** | Non-fatal anomalies | Action clipped, output near safety boundary, unusual timing |
| **ERROR** | Fatal failures that abort the run | Checkpoint missing, data root missing, config mismatch, NaN output |

---

## 3. Module Startup Parameter Logs

Every CLI entry point should log its resolved startup parameters at INFO level.

### Training (`scripts/train.py`)

```
stage=train
config_path
data_root
output_root
device
epochs
batch_size
lr
model_name
action_dim
```

### Inference (`scripts/infer_one.py`)

```
stage=inference
config_path
checkpoint_path
data_root
sample_index
device
clip_action
action_limit
arrow_scale
output_path
```

### Future Rollout (`scripts/rollout_fake_robot.py`)

```
stage=rollout
checkpoint_path
num_episodes
max_steps
success_threshold
action_limit
device
```

These startup logs make it possible to reproduce a run from its log alone.

---

## 4. Input/Output Shape and Format Logs

Record tensor shapes, dtypes, and value ranges at each pipeline boundary. This catches silent dimension/dtype/value mismatches early.

### Inference

```
image:          Tensor[3, 64, 64], float32, range=[0,1]
input_ids:      Tensor[16],       int64
attention_mask: Tensor[16],       int64
state:          Tensor[2],        float32
pred_action:    Tensor[2],        float32
gt_action:      Tensor[2],        float32
```

### Training

```
image:          Tensor[B, 3, 64, 64], float32
input_ids:      Tensor[B, T],         int64
attention_mask: Tensor[B, T],         int64
state:          Tensor[B, 2],         float32
action:         Tensor[B, 2],         float32
action_pred:    Tensor[B, 2],         float32
loss:           scalar,               float32
```

Common bugs caught by these logs:

- Shape mismatch (e.g. state is `[B]` instead of `[B, 2]`)
- dtype mismatch (e.g. int64 vs float32)
- Value range overflow (e.g. image outside `[0, 1]`)
- Missing batch dimension

---

## 5. Timing Logs

Record elapsed time per stage using `time.perf_counter()` (standard library).

### Training

| Event | Description |
|-------|-------------|
| `config_load_time` | Loading and merging YAML config |
| `dataset_load_time` | Toy2DDataset initialization |
| `model_build_time` | build_model() + .to(device) |
| `forward_time` | One batch forward pass (epoch average) |
| `backward_time` | One batch backward pass (epoch average) |
| `optimizer_step_time` | One optimizer.step (epoch average) |
| `checkpoint_save_time` | save_checkpoint per epoch |
| `epoch_time` | Total time per epoch |

### Inference

| Event | Description |
|-------|-------------|
| `config_load_time` | Loading config |
| `checkpoint_load_time` | load_checkpoint |
| `dataset_load_time` | Toy2DDataset init |
| `sample_load_time` | ds[sample_index] |
| `forward_time` | Predictor.predict |
| `visualization_save_time` | save_prediction_visualization |
| `total_inference_time` | Script total runtime |

### Future Rollout

| Event | Description |
|-------|-------------|
| `policy_inference_time` | Predictor.select_action |
| `robot_step_time` | FakeRobot.step |
| `state_update_time` | Environment state update |
| `success_check_time` | Success criteria evaluation |
| `episode_time` | Total time per rollout episode |

---

## 6. Timeout and Exception Logs

Every error path should produce a clear ERROR-level message.

### Existing errors (Stage 4)

- `checkpoint not found` — specified checkpoint path does not exist
- `data root not found` — data directory missing
- `sample index out of range` — dataset index bounds exceeded

### Planned / future errors

| Error | Context |
|-------|---------|
| `checkpoint load failed` | corrupt or incompatible checkpoint |
| `config-model mismatch` | checkpoint dimensions vs model config |
| `dataset empty` | no episodes found |
| `image file missing` | episode frame PNG not found |
| `forward output contains NaN` | model produced NaN action |
| `action exceeds safety limit` (without clipping) | raw output out of safe range |
| `visualization save failed` | cannot write output PNG |
| `robot command timeout` | robot interface did not respond in time |
| `state update timeout` | environment state did not update |
| `emergency stop triggered` | safety wrapper fired |

---

## 7. Critical State Change Logs

Log state transitions at INFO level.

### Training

```
model mode changed: train -> eval
checkpoint loaded
checkpoint saved
best checkpoint updated
```

### Inference

```
checkpoint loaded
action clipped       (WARNING if clipping was applied)
prediction completed
visualization saved
```

### Future Rollout

```
episode started
state updated
distance_to_target changed
success reached
max_steps reached
rollout ended
```

### Future Robot / Adapter

```
robot adapter initialized
safety mode enabled
command sent
command rejected
emergency stop
connection lost
```

---

## 8. Recommended Log Formats

Two output formats, serving different purposes.

### Console Log (human-readable)

```
[INFO] inference.start  config=configs/train/debug.yaml  ckpt=outputs/checkpoints/best.pt  device=cpu
[INFO] predictor.input  image=[3,64,64]  state=[2]  input_ids=[16]
[INFO] predictor.output  pred_action=[0.021,-0.013]  clipped=false
[INFO] visualization.saved  path=outputs/predictions/prediction.png  time_ms=12.4
```

### JSONL Log (machine-parseable)

```jsonl
{"level":"INFO","event":"inference.start","config":"configs/train/debug.yaml","device":"cpu"}
{"level":"INFO","event":"predictor.input","image_shape":[3,64,64],"state_shape":[2]}
{"level":"INFO","event":"predictor.output","pred_action":[0.021,-0.013],"clipped":false}
{"level":"INFO","event":"visualization.saved","path":"outputs/predictions/prediction.png","time_ms":12.4}
```

JSONL is useful for post-hoc analysis:

- Aggregate inference latency across runs
- Count how often actions were clipped
- Detect NaN output frequency
- Correlate rollout success rate with state/action statistics
- Reproduce a specific run from its logged parameters

---

## 9. Future Implementation Plan

### Phase 1: Lightweight Logger (Stage 4-L2)

Add `mini_vla/utils/logging.py` with:

```python
def get_logger(name: str) -> Logger
def log_event(logger, event: str, level="INFO", **fields)
class Timer:  # context manager for timing blocks
```

No third-party dependencies.  The logger writes to both stdout (colored, human) and a JSONL file (machine).

### Phase 2: Integration

Start by integrating the logger into one entry point — `scripts/infer_one.py` — to gain experience:

- Replace direct `print()` calls with structured log events
- Add Timer blocks around forward / visualization
- Write a `.jsonl` sidecar file alongside the prediction PNG

Then extend to `scripts/train.py` and future `scripts/rollout_fake_robot.py`.

### Phase 3: Observability Dashboard (V1+)

If the project accumulates enough JSONL logs, a small analysis script or
notebook can:
- Plot inference latency distributions
- Show action clip rates over runs
- Compare rollout success rates across checkpoints
- Flag runs with NaN or out-of-range outputs

---

## References

- Python `time.perf_counter()` — [docs.python.org](https://docs.python.org/3/library/time.html#time.perf_counter)
- JSON Lines format — [jsonlines.org](https://jsonlines.org/)
