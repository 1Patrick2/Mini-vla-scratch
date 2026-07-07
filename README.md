# Mini VLA from Scratch

A minimal Vision-Language-Action framework from scratch for learning robot action prediction.

## Goal

```
image + instruction + state -> action
```

## Current Stage

**V0: Toy 2D Mini VLA — Stage 5 PushT real dataset evaluation completed; preparing Stage 6 Dataset Zoo + Action Normalization.**

The full pipeline is complete: Toy 2D data pipeline, MiniVLA model forward, config-driven builder,
behavior cloning training loop, checkpoint save/load, policy-style Predictor with action clipping,
PIL-based prediction visualizer, infer_one CLI, **real PushT vision data loading via LeRobotDataset**,
**held-out episode evaluation**, **baseline comparison (zero/mean/previous-action)**,
**action metrics (MAE/MSE/RMSE/cosine similarity/finite ratio)**,
and **realdata pytest with `RUN_REAL_PUSHT=1`**.

## Quickstart

### Toy 2D

```bash
pip install -e .
python scripts/generate_toy_data.py --config configs/data/toy_2d.yaml --num-episodes 10
python scripts/train.py --config configs/train/debug.yaml
python scripts/evaluate_pusht.py --config configs/train/pusht_debug.yaml --ckpt outputs/checkpoints/best.pt --mock-data
```

### PushT (real data, optional LeRobot dependency)

```bash
pip install -r requirements-robot.txt
python scripts/train.py --config configs/train/pusht_debug.yaml
python scripts/evaluate_pusht.py --config configs/train/pusht_debug.yaml --ckpt outputs/checkpoints/best.pt
```

```powershell
$env:RUN_REAL_PUSHT="1"
python -m pytest tests/test_evaluate_pusht_cli.py -m realdata
```

## Implemented

- Project skeleton and package structure
- Config loading and schema validation
- Toy 2D episode data generation (`episode.json` + RGB frames)
- Toy2DDataset with per-episode loading
- Minimal instruction tokenizer with `attention_mask` support
- DataLoader collation (`collate_toy_2d`)
- SmallCNNVisionEncoder, LLM-ready TextBackbone, StateEncoder, FusionMLP, ActionHead
- MiniVLA end-to-end forward with config-driven builder
- MSE behavior cloning loss, optimizer with frozen parameter filtering
- Full Trainer class and training CLI
- Checkpoint save/load (last.pt + best.pt tracking)
- Predictor with policy-style select_action API and action clipping
- PIL-based prediction visualizer and infer_one CLI
- **Real PushT vision data loading via LeRobotDataset** with `loader: lerobot`
- **Held-out episode evaluation** (no frame-level leakage)
- **Offline evaluation with baselines**: zero, mean, previous-action
- **Action metrics**: MAE, MSE, RMSE, per-dim MAE, cosine similarity, finite ratio
- **Realdata pytest** with `RUN_REAL_PUSHT=1` marker
- **DatasetSpec + Registry** for multi-dataset support
- **Generic robot dataset inspect** CLI
- **Action/state normalization** utilities
- Component, dataset, model-forward, training, checkpoint, inference, and PushT tests (185+ passing)

## Planned

- Normalized PushT training/evaluation pipeline
- ALOHA sim transfer cube inspection and adapter smoke test
- LIBERO feasibility inspection
- Generic `BaseRobotDatasetAdapter` for multi-dataset training

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
