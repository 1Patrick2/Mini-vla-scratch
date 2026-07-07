# Dataset Zoo Feasibility Report

## Summary

| Dataset | Repo ID | Inspect | Train support | Eval support | Notes |
|---------|---------|---------|---------------|--------------|-------|
| PushT | `lerobot/pusht` | ✅ | ✅ | ✅ | Full pipeline (raw + normalized) |
| ALOHA sim transfer cube | `lerobot/aloha_sim_transfer_cube_scripted` | ✅ | inspect only | inspect only | `action_dim=14`, needs config override |
| LIBERO | `lerobot/libero` | ❌ remote SSL error | feasibility only | feasibility only | Remote download blocked in current env |

---

## PushT

| Field | Value |
|-------|-------|
| Supported | ✅ Train + Eval |
| matched_image_key | `observation.image` |
| matched_state_key | `observation.state` |
| matched_action_key | `action` |
| image_shape | `[3, 96, 96]` (CHW torch.Tensor) |
| state_shape | `[2]` |
| action_shape | `[2]` |
| Normalization | ✅ Stats computed, normalized train/eval pipeline works |
| Config | `configs/train/pusht_debug.yaml` (raw), `configs/train/pusht_normalized.yaml` (norm) |
| Inference | `scripts/evaluate_pusht.py`, `scripts/evaluate_robot_dataset.py` |

### PushT normalized validation results

```text
Stats (512 samples):
  state_mean: [206.49, 276.25]
  state_std:  [99.80, 93.63]
  action_mean: [206.53, 278.75]
  action_std:  [99.51, 92.39]

Training (3 epochs, normalized space):
  Epoch 1/3  loss=0.216  mae=0.309
  Epoch 2/3  loss=0.052  mae=0.176
  Epoch 3/3  loss=0.044  mae=0.159

Evaluation (128 samples, heldout=0.2, raw action space):
  raw_action MAE:  6.891
  zero_action MAE: 205.460  (Y model beats baseline)
  mean_action MAE: 17.848   (Y model beats baseline)
  previous MAE:    205.820  (Y model beats baseline)
  CosSim:  0.9998
  Finite:  1.0000
```

### PushT raw pipeline (Stage 5)

```bash
python scripts/train.py --config configs/train/pusht_debug.yaml
python scripts/evaluate_pusht.py --config configs/train/pusht_debug.yaml --ckpt outputs/checkpoints/best.pt
```

### PushT normalized pipeline (Stage 6)

```bash
python scripts/compute_dataset_stats.py \
  --dataset-name pusht --repo-id lerobot/pusht --loader lerobot \
  --max-samples 512 --output outputs/dataset_reports/pusht_stats.json

python scripts/train.py --config configs/train/pusht_normalized.yaml

python scripts/evaluate_robot_dataset.py \
  --config configs/train/pusht_normalized.yaml \
  --ckpt outputs/checkpoints/best.pt \
  --dataset-name pusht --repo-id lerobot/pusht --loader lerobot \
  --max-samples 128 --heldout-ratio 0.2 \
  --output outputs/eval/pusht_normalized_report.json \
  --predictions-output outputs/eval/pusht_normalized_predictions.jsonl
```

---

## ALOHA sim transfer cube

| Field | Value |
|-------|-------|
| Status | ✅ Inspect passed |
| Repo ID | `lerobot/aloha_sim_transfer_cube_scripted` |
| Train support | ❌ (Stage 6 inspect only — `action_dim=14` != MiniVLA default) |
| Eval support | ❌ (Stage 6 inspect only) |
| matched_image_key | `observation.images.top` |
| matched_state_key | `observation.state` |
| matched_action_key | `action` |
| image_shape | `[3, 480, 640]` |
| state_shape | `[14]` |
| action_shape | `[14]` |
| has_language | ✅ (`task` key available) |

### Known challenges

- `action_dim=14`, `state_dim=14` — MiniVLA defaults to `action_dim=2`.
  Config override needed for training.
- Image is `[3, 480, 640]` (much larger than PushT's 96×96).
  Resize to 64×64 handled by adapter, but crops information.
- Multi-camera: only `observation.images.top` was matched; camera
  selection policy needed for multi-camera datasets.
- Full training deferred to Stage 7+.

---

## LIBERO

| Field | Value |
|-------|-------|
| Status | ❌ Remote inspect failed (SSL error) |
| Repo ID | `lerobot/libero` |
| Train support | ❌ (feasibility only) |
| Eval support | ❌ (feasibility only) |
| Mock inspect | ✅ Schema report from mock data available |

### Known challenges

- Remote download failed due to SSL error (`UNEXPECTED_EOF_WHILE_READING`).
  May be intermittent or require VPN/proxy.
- Large dataset (130 tasks × 4 suites) — download size is significant.
- Multi-camera fields expected: `observation.images.agentview`,
  `observation.images.eye_in_hand`.
- Multi-task language instructions may be complex.
- Full training deferred to Stage 7+.

---

## Known Limitations

- **Action dim variation**: Some datasets have `action_dim > 2`; MiniVLA currently
  builds output head from `config.model.action_dim`.  For multi-dataset training,
  a config override or adaptive head is needed.
- **Multi-camera**: The `BaseRobotDatasetAdapter` selects the first available image
  key; it does not fuse multiple cameras.  A camera selection/fusion strategy is
  deferred to Stage 7+.
- **Language fields**: Instruction key names vary across datasets.  The
  `language_keys` candidate list + `default_instruction` fallback works for
  single-task data.  Multi-task data may need new logic.
- **Full training**: ALOHA / LIBERO full training is deferred to Stage 7+.
  Stage 6 covers inspect, schema report, and adapter feasibility only.
- **Video decoding**: LeRobotDataset uses `pyav` or `torchcodec` for video frame
  decoding.  On Windows, `torchcodec` may not be available (falls back to `pyav`).
  This is benign but may produce warnings.
