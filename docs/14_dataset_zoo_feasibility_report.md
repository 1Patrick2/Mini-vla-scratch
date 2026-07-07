# Dataset Zoo Feasibility Report

## Summary

| Dataset | Repo ID | Inspect | Train support | Eval support | Notes |
|---------|---------|---------|---------------|--------------|-------|
| PushT | `lerobot/pusht` | ✅ | ✅ | ✅ | Full pipeline (raw + normalized) |
| ALOHA sim transfer cube | `lerobot/aloha_sim_transfer_cube_scripted` | ⏳ | inspect only | inspect only | Pending real-data inspect |
| LIBERO | `lerobot/libero` | ⏳ | feasibility only | feasibility only | Pending real-data inspect |

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
| Status | ⏳ Pending inspect |
| Repo ID | `lerobot/aloha_sim_transfer_cube_scripted` |
| Train support | ❌ (Stage 6 inspect only) |
| Eval support | ❌ (Stage 6 inspect only) |

### Known challenges (estimated)

- Action dim may differ from MiniVLA default `action_dim=2`
- State keys may include `observation.qpos` (joint positions)
- Multi-camera fields (`observation.images.top`, `observation.images.cam_high`) require camera selection policy
- Default `max_text_len=16` may be insufficient; ALOHA instructions may be longer
- `strict: false` recommended for initial inspect

---

## LIBERO

| Field | Value |
|-------|-------|
| Status | ⏳ Pending inspect |
| Repo ID | `lerobot/libero` |
| Train support | ❌ (Stage 6 feasibility only) |
| Eval support | ❌ (Stage 6 feasibility only) |

### Known challenges (estimated)

- LIBERO has 4 task suites, 130 tasks — much larger than PushT
- Multi-task language instructions may be complex
- Multi-camera fields: `observation.images.agentview`, `observation.images.eye_in_hand`
- Higher state/action dimensionality possible
- `max_text_len=32` recommended (config already set)
- Very large download size — expect slower first load
- Requires `lerobot` optional dependency

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
