# Mini VLA from Scratch

A minimal Vision-Language-Action framework from scratch for learning robot action prediction.

## Goal

```
image + instruction + state -> action
```

## Current Stage

**V0: Toy 2D Mini VLA — Stage 1 data pipeline completed; preparing Stage 2 model forward.**

The data pipeline is complete: synthetic episode generation, per-episode dataset loading,
minimal tokenizer, and DataLoader collation.

## Implemented

- Project skeleton and package structure
- Config loading and schema validation
- Toy 2D episode data generation (`episode.json` + RGB frames)
- Toy2DDataset with per-episode loading
- Minimal instruction tokenizer
- DataLoader collation (`collate_toy_2d`)
- Action clipping utility
- Basic tests (20 passing)

## Planned

- MiniVLA model forward (CNN + GRU + MLP fusion)
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
