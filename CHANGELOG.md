# Changelog

## 0.7.0 — Stage 4 Policy Inference and Visualization

### Added
- Predictor with policy-style predict/select_action API
- Checkpoint-based single-sample inference (build_model + load_checkpoint + model.eval)
- Action postprocessing with configurable clip_action/action_limit
- Manual loaded model consistency test (clip_action=False matches manual forward)
- Prediction visualizer (PIL-based, blue pred arrow, optional green gt arrow)
- infer_one CLI with --config, --ckpt, --data-root, --sample-index, --output,
  --device, --arrow-scale, --no-clip-action, --action-limit
- CLI subprocess tests (end-to-end, --no-clip-action, error checks)
- Policy inference and robot learning notes (docs/10_...)

### Changed
- ROADMAP.md: Stage 4 marked complete
- README.md: updated to Stage 4 status, added Predictor/visualizer/CLI to implemented

## 0.6.0 — Stage 3 Behavior Cloning Training Loop

### Added
- Training design document (docs/09_training_design.md)
- MSE action loss (mse_action_loss)
- Training metrics (mse, l1)
- Optimizer factory with frozen parameter filtering (create_optimizer)
- Full end-to-end training step test (DataLoader → model → loss → backward → step)
- Trainer class with fit() and train_one_epoch()
- Checkpoint save/load (save_checkpoint, load_checkpoint) with last.pt / best.pt tracking
- Checkpoint unit tests (save, load, epoch/metrics restoration, optimizer state)
- Training CLI (scripts/train.py) with dry-run mode, data validation, and checkpoint output paths
- Training correctness tests: zero-action baseline comparison, tiny dataset overfit, evaluate_loss() helper, checkpoint reload consistency

### Changed
- ROADMAP.md: deduplicated Stage 3 section
- PLAN.md: updated to LLM-ready architecture, Stage 2/3 status marked complete
- config/schema.py: removed state_dim requirement (build_model uses component-level config)

### Fixed
- Config schema now requires model.name and action_dim only; state_dim is optional
- Test config dimension mismatch in test_training.py (fusion.output_dim defaulted to 128 instead of 16)

## 0.5.0 — Stage 2 MiniVLA Forward and Builder

### Added
- MiniVLA end-to-end forward (assembles all components into a single model)
- Config-driven `build_model()` with strict validation:
  - `model.name` check
  - Component type checks (vision_encoder, text_encoder, fusion)
  - Dimension consistency checks (fusion.input_dim, action_head.input_dim)
  - `text_encoder.freeze` support
- DataLoader-to-model smoke test
- `build_model` exported from `mini_vla.models`

### Changed
- Stage 2 docs and roadmap aligned with LLM-ready TextBackbone
- `action_dim` controlled by `model.action_dim` (not action_head.output_dim)

## 0.4.0 — Stage 2 Model Components

### Added
- Lightweight VLA research notes (`docs/07_lightweight_vla_research.md`)
- SmallCNNVisionEncoder — `[B,3,64,64]` → `[B,128]`
- LLM-ready TextBackbone (`BaseTextEncoder`, `MockLLMTextEncoder`, `LLMTextEncoder`)
- `attention_mask` support in Toy2DDataset and collate_toy_2d
- `build_attention_mask()` helper in transforms
- StateEncoder, FusionMLP, ActionHead
- Component tests (test_vision_encoder, test_text_encoder, test_model_components)

## 0.3.0 — Stage 1-C DataLoader Collation

### Added
- `mini_vla/datasets/collate.py` — `collate_toy_2d()` for batching samples
- Collation tests verifying batch tensor shapes

## 0.2.0 — Stage 1-B Toy2DDataset

### Added
- `mini_vla/datasets/toy_2d_dataset.py` — `Toy2DDataset` loading per-episode data
- `mini_vla/datasets/transforms.py` — minimal tokenizer (no transformers dependency)
- `tests/test_dataset.py` — dataset shape, keys, dtype tests (5 tests)

## 0.1.1 — Stage 1-A Toy 2D Data Generator

### Added
- `scripts/generate_toy_data.py` fully implemented (was placeholder)
  - Synthetic per-episode data generation
  - `episode.json` with `frame/state/action/target` per step
  - RGB frame rendering (red object, green target)
  - CLI with `--config` and `--num-episodes`
- `tests/test_generate_toy_data.py` — structure and determinism tests

### Notes
- Action clipping utility implemented; safety wrapper is a placeholder reserved for later stages
- Dataset loader, MiniVLA model, training loop, inference, and rollout remain in planned stages

## 0.1.0 — Stage 0 Skeleton (2026-06-30)

### Added
- Project skeleton with package structure (`mini_vla/`)
- Config system: YAML loading, base merge, schema validation
- Training CLI with dry-run mode
- Action clipping utility
- Basic unit tests (4 passing)
- All Stage 0 root files
- 6 design documents in `docs/` (00–05)
- Comprehensive long-term plan (`PLAN.md`)

### Changed
- README aligned with actual Stage 0 skeleton state
- ROADMAP rewritten as Stage 0–6 with commands and acceptance criteria
- SETUP expanded with verification commands, Stage 1 preview, Common Problems
