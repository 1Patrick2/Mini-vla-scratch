# Mini VLA from Scratch

A minimal Vision-Language-Action framework from scratch for learning robot action prediction.

## Goal

```
image + instruction + state -> action
```

## Current Stage

**V0: Toy 2D Mini VLA — Stage 3 behavior cloning training loop completed; preparing Stage 4 inference and visualization.**

The full training pipeline is complete: Toy 2D data pipeline, SmallCNN vision encoder, LLM-ready text backbone,
attention_mask support, StateEncoder, FusionMLP, ActionHead, MiniVLA end-to-end forward, config-driven
model builder with strict validation, MSE behavior cloning loss, optimizer with frozen parameter filtering,
full Trainer loop, checkpoint save/load (last.pt + best.pt), and training CLI.

## Implemented

- Project skeleton and package structure
- Config loading and schema validation
- Toy 2D episode data generation (`episode.json` + RGB frames)
- Toy2DDataset with per-episode loading
- Minimal instruction tokenizer
- `attention_mask` support in Dataset and collation
- DataLoader collation (`collate_toy_2d`)
- Lightweight VLA research notes (TinyVLA, SmolVLA)
- SmallCNNVisionEncoder
- LLM-ready TextBackbone (`MockLLMTextEncoder` for tests, `LLMTextEncoder` for HuggingFace)
- StateEncoder, FusionMLP, ActionHead
- MiniVLA end-to-end forward
- Config-driven model builder with strict type/dimension validation
- DataLoader-to-model smoke test
- MSE action loss (mse_action_loss)
- Training metrics (MSE, L1)
- Optimizer factory with frozen parameter filtering (create_optimizer)
- Full Trainer class (fit, train_one_epoch)
- Checkpoint save/load (last.pt + best.pt tracking)
- Training CLI (scripts/train.py)
- Action clipping utility
- Component, dataset, model-forward, training, and checkpoint tests (83 passing)

## Planned

- Single-sample inference and visualization
- Fake robot rollout
- Future Open Kaka robot adapter

## Setup

See [SETUP.md](SETUP.md).

## Project Principles

See [PROJECT_PRINCIPLES.md](PROJECT_PRINCIPLES.md).

## Verification

```bash
pip install -e .
python -c "import mini_vla; print('OK')"
pytest
python scripts/train.py --config configs/train/debug.yaml --dry-run
python scripts/generate_toy_data.py --config configs/data/toy_2d.yaml --num-episodes 5
```

## Roadmap

See [ROADMAP.md](ROADMAP.md).
