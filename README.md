# Mini VLA from Scratch

A minimal Vision-Language-Action framework from scratch for learning robot action prediction.

## Goal

```
image + instruction + state -> action
```

## Current Stage

**Stage 8: Policy × Dataset Matrix — Complete.**

MiniVLA now supports a policy abstraction layer and a Policy × Dataset Matrix, including PushT full benchmark results and ALOHA sim training smoke support.

## Policy × Dataset Matrix

| Dataset | Status | Policies |
|---------|--------|----------|
| PushT | full benchmark | SingleFrame / History / DeltaAction / ActionChunk smoke |
| ALOHA sim | training smoke | SingleFrame / History / ActionChunk |
| LIBERO | inspect feasibility | language/multitask candidate |

| Policy | Output | Purpose |
|--------|--------|---------|
| SingleFrame BC | action | current-frame baseline |
| History BC | action | temporal context |
| DeltaAction BC | delta action | residual action prediction |
| ActionChunk BC | future action chunk | chunked policy smoke |

See [docs/16](docs/16_policy_dataset_matrix.md) for full design and [docs/15](docs/15_strict_episode_split_and_history_delta_bc.md) for PushT results.

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
- **Policy abstraction** with registry (SingleFrame / History / DeltaAction / ActionChunk)
- **shape_meta** for multi-dataset dimension resolution
- **ActionChunk BC** smoke (PushT train/eval verified) with episode-safe chunking
- **Policy × Dataset Matrix** runner and audit
- **ALOHA sim training smoke** configs
- Comprehensive unit and smoke tests

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
