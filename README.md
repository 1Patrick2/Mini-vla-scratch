# Mini VLA from Scratch

A minimal Vision-Language-Action framework from scratch for learning robot action prediction.

## Goal

```
image + instruction + state -> action
```

## Current Stage

**V0: Toy 2D manipulation behavior cloning — Stage 0 skeleton.**

The project skeleton is in place: package structure, config system, training CLI dry-run, basic tests.
Dataset generation, model implementation, training loop, inference, and rollout are planned for subsequent stages.

## Implemented

- Project skeleton and package structure
- Config loading and base config merge
- Minimal config schema validation
- Training CLI dry-run
- Action clipping utility
- Basic unit tests (4 passing)

## Planned

- Toy 2D episode dataset generation
- Dataset loader and collation
- CNN / GRU / MLP MiniVLA model
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
```

## Roadmap

See [ROADMAP.md](ROADMAP.md).
