# Mini VLA from Scratch

A minimal Vision-Language-Action framework from scratch for learning robot action prediction.

## Goal

```
image + instruction + state -> action
```

## Current Stage

**V0: Toy 2D Mini VLA — Stage 2 model components implemented; preparing Stage 2-D end-to-end MiniVLA forward.**

All model components are implemented: vision encoder, LLM-ready text backbone, state encoder, fusion MLP, and action head. The full MiniVLA forward pass is the next step.

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
- Action clipping utility
- Component and dataset tests (39 passing)

## Planned

- MiniVLA end-to-end forward with LLM-ready TextBackbone
- Behavior cloning training loop
- Checkpoint saving and evaluation
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
