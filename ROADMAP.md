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
| Stage 2-D: MiniVLA end-to-end forward | ✅ Complete |
| Stage 2-E: Config-driven builder + DataLoader smoke | ✅ Complete |
| Stage 2-F: Docs, configs, changelog cleanup | ✅ Complete |

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

## Stage 3: Behavior Cloning Training Loop ✅ Complete

**Goal:** Run end-to-end training: dataset → model → loss → backward → checkpoint.

**Status:** MSE loss, optimizer with frozen-parameter filtering, Trainer with fit()/train_one_epoch(), checkpoint save/load (last.pt + best.pt), and training CLI all implemented.

**Key design decisions:**
- Text backbone is **frozen** by default (``freeze: true``).
- Trainable: vision_encoder, state_encoder, fusion, action_head.
- Loss: MSE(action_pred, action_gt).

**Main files:**
- `mini_vla/training/losses.py`
- `mini_vla/training/metrics.py`
- `mini_vla/training/optimizer.py`
- `mini_vla/training/checkpoint.py`
- `mini_vla/training/trainer.py`
- `tests/test_training.py`

**Command:**
```bash
python scripts/train.py --config configs/train/debug.yaml
```

**Acceptance:** Training loop runs end-to-end, finite loss/mae are reported, trainable parameters update, frozen text encoder stays unchanged, and last.pt/best.pt are saved.

---

## Stage 4: Policy Inference and Visualization ✅ Complete

**Goal:** Load a trained model, predict action for a single sample, visualize, and CLI.

**Status:** Predictor with policy-style select_action, action clipping, PIL-based visualizer
(pred in blue, GT in green), and infer_one CLI all implemented.

**Key files:**
- `mini_vla/inference/predictor.py` — Predictor with predict/select_action
- `mini_vla/inference/visualizer.py` — PIL arrow drawing (blue pred, green gt)
- `scripts/infer_one.py` — CLI with --config, --ckpt, --sample-index, --output
- `docs/10_policy_inference_and_robot_learning_notes.md` — ML/DL/IL/BC/RL notes
- `tests/test_inference_predictor.py` — 8 tests
- `tests/test_inference_visualizer.py` — 8 tests
- `tests/test_infer_one_cli.py` — 6 tests (subprocess)

**Command:**
```bash
python scripts/infer_one.py --ckpt outputs/checkpoints/best.pt --config configs/train/debug.yaml
```

**Acceptance:** Predictor loads checkpoint, predict returns Tensor[2], select_action works, output is finite and clippable, visualizer saves prediction.png, CLI prints pred/gt/L1 error. No Stage 5 rollout, Transformer, RL, or LeRobot dependency introduced.

---

## Stage 5: PushT Dataset Evaluation ⏳

**Goal:** Use the `lerobot/pusht` dataset to verify MiniVLA action prediction
on a real robot-learning benchmark, with offline evaluation and baselines.

**Key files:**
- `docs/12_pusht_dataset_evaluation_design.md` — design doc
- `configs/data/pusht.yaml` — PushT data config
- `configs/train/pusht_debug.yaml` — PushT training config
- `mini_vla/datasets/pusht_adapter.py` — PushT → MiniVLA sample adapter
- `mini_vla/datasets/pusht_inspection.py` — dataset schema inspection
- `mini_vla/datasets/factory.py` — dataset factory
- `mini_vla/evaluation/action_metrics.py` — MAE, MSE, RMSE, cosine similarity
- `mini_vla/evaluation/baselines.py` — zero/mean/previous-action baselines
- `mini_vla/evaluation/evaluator.py` — offline evaluation
- `scripts/inspect_pusht_dataset.py` — inspect PushT schema
- `scripts/evaluate_pusht.py` — run evaluation and produce report

**Command:**
```bash
python scripts/inspect_pusht_dataset.py --repo-id lerobot/pusht --max-samples 128
python scripts/train.py --config configs/train/pusht_debug.yaml
python scripts/evaluate_pusht.py --config configs/train/pusht_debug.yaml --ckpt outputs/checkpoints/best.pt
```

**Acceptance:**
- inspect prints pushT schema
- adapt PushT sample to MiniVLA format (image [3,64,64], state [2], action [2])
- train/evaluate on PushT subset
- held-out episode evaluation
- MiniVLA beats zero-action and mean-action baselines
- report.json saved with metrics and baseline comparisons

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
