# Mini VLA from Scratch

A minimal Vision-Language-Action framework from scratch for learning robot action prediction.

## Goal

This project implements a small VLA pipeline:

```text
image + instruction + state -> action
```

## Current Stage

V0: Toy 2D manipulation behavior cloning.

## Features

- Config-driven experiments
- Toy 2D episode dataset
- Image / language / state encoders
- Multimodal fusion
- Continuous action prediction
- Behavior cloning training
- Inference and fake robot rollout
- Future Open Kaka robot adapter

## Setup

See [SETUP.md](SETUP.md).

## Project Principles

See [PROJECT_PRINCIPLES.md](PROJECT_PRINCIPLES.md).

## Quick Start

```bash
python scripts/train.py --help
pytest
```

## Roadmap

See [ROADMAP.md](ROADMAP.md).
