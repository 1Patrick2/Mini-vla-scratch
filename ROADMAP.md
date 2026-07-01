# Roadmap

## Stage 0: Project Skeleton

**Goal:** Set up project structure, config system, environment files, and basic tests.

| Component | Status |
|-----------|--------|
| Root files (README, SETUP, PRINCIPLES, etc.) | ✅ Complete |
| pyproject.toml with dev tool config | ✅ Complete |
| Config system (loader + schema) | ✅ Complete |
| Training CLI dry-run | ✅ Complete |
| Action clipping utility | ✅ Complete |
| Safety wrapper | ⏳ Placeholder, planned for later stage |
| Basic unit tests (4 passing) | ✅ Complete |
| Environment scripts (conda/venv/WSL/Windows) | ✅ Complete |

**Command:** `pytest` — 4 passed
**Command:** `python scripts/train.py --config configs/train/debug.yaml --dry-run`

---

## Stage 1: Toy 2D Data Pipeline

**Goal:** Generate synthetic 2D manipulation episodes, implement dataset loader and collation.

| Component | Status |
|-----------|--------|
| Toy data generator | ✅ Complete |
| Per-episode `episode.json` output | ✅ Complete |
| RGB frame rendering (red object, green target) | ✅ Complete |
| `Toy2DDataset` loader | ✅ Complete |
| Minimal instruction tokenizer | ✅ Complete |
| `collate_toy_2d` DataLoader collation | ✅ Complete |
| Dataset and collation tests | ✅ Complete |

**Command:** `python scripts/generate_toy_data.py --config configs/data/toy_2d.yaml --num-episodes 5`
**Command:** `pytest tests/test_dataset.py`
**Tests:** 20 passed

---

## Stage 2: MiniVLA Model Forward

**Goal:** Implement `image + instruction + state → action_pred` forward pass with LLM-ready text backbone.

| Component | Status |
|-----------|--------|
| Stage 2-0: TinyVLA / SmolVLA research notes | ✅ Complete |
| Stage 2-A: SmallCNNVisionEncoder | ✅ Complete |
| Stage 2-B: LLM-ready TextBackbone + attention_mask | ✅ Complete |
| Stage 2-C: StateEncoder + FusionMLP + ActionHead | ✅ Complete |
| Stage 2-D: MiniVLA end-to-end forward | ⏳ |
| Stage 2-E: Docs, configs, changelog cleanup | ⏳ |

**Architecture:**
```
image → SmallCNNVisionEncoder → image_feat    Tensor[B, 128]
input_ids + attention_mask → LLM-ready TextBackbone → text_feat    Tensor[B, 128]
state → StateEncoder → state_feat    Tensor[B, 128]
concat(image_feat, text_feat, state_feat) → FusionMLP → fused_feat
fused_feat → ActionHead → action_pred    Tensor[B, 2]
```

**Key design decisions:**
- Text backbone is LLM-ready (`BaseTextEncoder` → `MockLLMTextEncoder` / `LLMTextEncoder`).
- MockLLMTextEncoder does NOT require transformers or real LLM downloads.
- Dataset returns `attention_mask` alongside `input_ids`.
- Action is continuous MLP regression (no tokenization, no diffusion, no flow matching).

**Acceptance:** All component forward shape tests pass + MiniVLA can consume DataLoader batch.

---

## Stage 3: Behavior Cloning Training Loop

**Goal:** Run end-to-end training: dataset → model → loss → backward → checkpoint.

**Main files:**
- `mini_vla/training/losses.py`
- `mini_vla/training/metrics.py`
- `mini_vla/training/optimizer.py`
- `mini_vla/training/checkpoint.py`
- `mini_vla/training/trainer.py` (full implementation)
- `tests/test_training_step.py`

**Command:**
```bash
python scripts/train.py --config configs/train/debug.yaml
```

**Acceptance:** Training loss decreases over epochs, checkpoint saved to `outputs/checkpoints/best.pt`.

---

## Stage 4: Inference and Visualization

**Goal:** Load a trained model, predict action for a single sample, and visualize.

**Main files:**
- `mini_vla/inference/predictor.py`
- `mini_vla/inference/visualizer.py`
- `scripts/infer_one.py`

**Command:**
```bash
python scripts/infer_one.py --ckpt outputs/checkpoints/best.pt --config configs/train/debug.yaml
```

**Acceptance:** Generates `outputs/predictions/prediction.png` with direction arrow overlay.

---

## Stage 5: Fake Robot Rollout

**Goal:** Continuous action prediction loop with fake robot state update and success evaluation.

**Main files:**
- `mini_vla/inference/rollout.py`
- `mini_vla/robot_interface/fake_robot.py`
- `scripts/rollout_fake_robot.py`

**Command:**
```bash
python scripts/rollout_fake_robot.py --ckpt outputs/checkpoints/best.pt --num-episodes 50
```

**Acceptance:** Reports success rate, average steps, and average final distance to target.

---

## Stage 6: Open Kaka Adapter Skeleton

**Goal:** Design and implement the Open Kaka robot adapter interface (no real robot).

**Main files:**
- `mini_vla/robot_interface/open_kaka_adapter.py`
- `docs/05_open_kaka_integration.md`

**Acceptance:** Adapter can be imported and called without error.

---

## Future Directions (V1+)

| Direction | Goal |
|-----------|------|
| Action chunk | Predict future N-step action sequence |
| Episode robot dataset | Support real robot episode format |
| Pretrained encoder | Replace small CNN/GRU with ResNet/TinyBERT |
| Real robot rollout | Open Kaka virtual arm + safety loop |
