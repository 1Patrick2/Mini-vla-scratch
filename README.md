# Mini VLA from Scratch

A minimal Vision-Language-Action framework from scratch for learning robot action prediction.

## Goal

```
image + instruction + state -> action
```

## Current Stage

**Stage 7: Strict Episode Split + History/Delta BC — Complete.**

A strict offline Behavior Cloning benchmark on real LeRobot PushT data with episode-level train/eval split, train-only normalization, and three BC variants.

## Stage 7: Robot Learning Benchmark

| Model | Raw MAE | Takeaway |
|-------|--------:|----------|
| SingleFrame BC | 19.51 | current-frame baseline |
| History BC | 11.02 | +temporal context |
| DeltaAction BC | **7.35** | **best learned variant** |

**Protocol:** episode-level split (6 train / 2 eval episodes, 1024 samples), train-only normalization, raw action space metrics, baseline reports.

DeltaAction BC reduces raw MAE by **33.3%** over History BC and **62.3%** over SingleFrame BC on strictly unseen episodes.

### Key components

- `create_episode_split.py` — reproducible episode-level split manifest
- `compute_dataset_stats.py` — train-only normalization statistics
- Three configs: `pusht_strict_single_frame.yaml`, `pusht_strict_history.yaml`, `pusht_strict_delta.yaml`
- `evaluate_robot_dataset.py` — raw/normalized action evaluation with baselines
- `compare_eval_reports.py` — cross-method comparison table
- `audit_stage7_outputs.py` — data-integrity audit

See [docs/15](docs/15_strict_episode_split_and_history_delta_bc.md) for full protocol and results.

## Quickstart

### Toy 2D

```bash
pip install -e .
python scripts/generate_toy_data.py --config configs/data/toy_2d.yaml --num-episodes 10
python scripts/train.py --config configs/train/debug.yaml
python scripts/infer_one.py --config configs/train/debug.yaml --ckpt outputs/checkpoints/best.pt
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
- **Strict episode-level split** with reproducible manifest
- **Train-only normalization** statistics
- **SingleFrame, History, DeltaAction BC** comparison
- **Raw action evaluation** with baseline reports
- **DeltaAction** is the best learned variant (raw MAE=7.35)
- **Data-integrity audit** for reproducibility

## Planned

- Action chunk / ACT-lite temporal policy
- Multi-seed / larger subset benchmark
- Stronger visual encoder or pretrained backbone

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
