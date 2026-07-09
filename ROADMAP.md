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

## Stage 5: PushT Dataset Evaluation ✅ Complete

**Goal:** Use the `lerobot/pusht` dataset to verify MiniVLA action prediction
on a real robot-learning benchmark, with offline evaluation and baselines.

**Status:** Real PushT vision data loading, adapter, training, held-out episode
evaluation, baseline comparison (zero/mean/previous-action), and evaluation
report all implemented and validated on real PushT data via LeRobotDataset.
Model beats zero-action and mean-action baselines on held-out episodes.

**Key files:**
- `docs/12_pusht_dataset_evaluation_design.md` — design doc + validation results
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

## Stage 6: Dataset Zoo + Action Normalization ✅

**Goal:** Extend MiniVLA from single-dataset (PushT) to a multi-dataset
framework with DatasetSpec, registry, generic adapter, action
normalization, and ALOHA/LIBERO feasibility inspect.

**Status:** DatasetSpec/Registry, key_utils, generic loader,
BaseRobotDatasetAdapter, factory robot_dataset, normalization utilities,
normalized PushT train/eval all implemented and validated.
ALOHA sim inspect passed (action_dim=14). LIBERO inspect blocked by SSL.

**Key files:**
- `docs/13_dataset_zoo_and_normalization_plan.md` — design doc
- `mini_vla/datasets/spec.py` — DatasetSpec dataclass
- `mini_vla/datasets/registry.py` — dataset registry
- `mini_vla/datasets/normalization.py` — action/state normalizer
- `scripts/inspect_robot_dataset.py` — generic dataset inspect CLI
- `scripts/compute_dataset_stats.py` — normalisation stats CLI
- `configs/train/pusht_normalized.yaml` — normalised training config
- `requirements-robot.txt` — optional LeRobot dependencies

**Acceptance:**
- Stage 5 raw PushT pipeline still works
- PushT normalized stats compute works
- Generic inspect_robot_dataset.py works with mock & real PushT
- ALOHA sim inspect passes
- LIBERO inspect fails with clear diagnostic (SSL error)
- Default pytest does not require network

---

## Stage 7: Strict Episode Split + History/Delta BC ✅ Complete

**Goal:** Build a reproducible PushT BC benchmark with strict episode-level
split, train-only normalization, and history/delta action policy variants.

**Key files:**
- `mini_vla/datasets/splits.py` — EpisodeSplit manifest
- `mini_vla/datasets/transforms.py` — HistoryDatasetWrapper, DeltaActionTargetWrapper
- `scripts/create_episode_split.py` — split manifest CLI
- `scripts/compute_dataset_stats.py` — split-aware stats CLI
- `configs/train/pusht_strict_single_frame.yaml` — strict baseline
- `configs/train/pusht_strict_history.yaml` — history model
- `configs/train/pusht_strict_delta.yaml` — delta action
- `docs/15_strict_episode_split_and_history_delta_bc.md` — design doc + results
- `scripts/compare_eval_reports.py` — cross-method comparison
- `scripts/audit_stage7_outputs.py` — output integrity checks

**Results (raw action space, strict unseen-episode eval):**

| Model | Raw MAE | Takeaway |
|---|---|---:|
| SingleFrame BC | 19.51 | current-frame baseline |
| History BC | 11.02 | temporal context |
| DeltaAction BC | **7.35** | **best learned variant** |

**Acceptance:**
- Reproducible episode-level train/eval split (6 train / 2 eval episodes)
- Train-only normalization statistics
- SingleFrame, History, DeltaAction BC all train and evaluate on strict split
- Raw action evaluation with baseline reports
- DeltaAction improves over History on strict unseen episodes
- Data-integrity audit passes (9/9 checks)
- All unit tests pass, ruff clean
- Stage 5/6 pipelines not broken

---

## Stage 8: Policy × Dataset Matrix ⏳

**Goal:** Upgrade MiniVLA from a PushT-only strict benchmark into a lightweight
policy × dataset framework.

**Delivered:**
- Policy registry and ``BasePolicy`` unified interface
- ``shape_meta`` utilities (dimension resolution, no more hard-coded 2)
- SingleFrame / History / DeltaAction migrated to policy interface
- ActionChunk BC smoke (chunk regression with episode-safe chunking)
- PushT full benchmark retained
- ALOHA sim training smoke configs
- LIBERO inspect/language feasibility retained
- Matrix runner and Stage 8 audit

**Acceptance:**
- Policy abstraction complete: ``build_policy(config)`` works
- All four policy types registered and tested
- shape_meta resolves state/action dims from config
- ActionChunk target wrapper respects episode boundaries
- PushT reports and Stage 7 audit still pass
- ALOHA smoke configs exist (training not guaranteed offline)
- pytest / ruff pass
- audit_stage8_matrix.py pass

---

| Direction | Goal |
|-----------|------|
| Action chunk | Predict future N-step action sequence |
| Episode robot dataset | Support real robot episode format |
| Pretrained encoder | Replace small CNN/GRU with ResNet/TinyBERT |
| Real robot rollout | Open Kaka virtual arm + safety loop |
