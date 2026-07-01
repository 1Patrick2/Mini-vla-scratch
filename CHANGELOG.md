# Changelog

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
