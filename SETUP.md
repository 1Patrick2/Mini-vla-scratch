# Setup Guide

This document explains how to set up the development environment for `mini-vla-from-scratch`.

## 1. Basic Install

### Requirements

- Python 3.10 or later
- PyTorch (CPU or CUDA)
- pip / conda

### Steps

```bash
pip install -e .
pip install -r requirements-dev.txt  # dev tools (ruff, pytest, ...)
```

Verify:

```bash
pytest
python -c "import mini_vla; print('OK')"
python scripts/train.py --config configs/train/debug.yaml --dry-run
```

## 2. Conda Environment

```bash
conda env create -f environment.yml
conda activate mini-vla
pip install -e .
```

## 3. One-Command Setup

Linux / WSL:

```bash
bash setup_env.sh           # basic development setup
bash setup_env.sh --robot   # includes LeRobot + datasets
```

Windows PowerShell:

```powershell
.\setup_env.ps1             # basic development setup
.\setup_env.ps1 -Robot      # includes LeRobot + datasets
```

## 4. Optional Robot Dataset Dependencies

Stage 5 and 6 use `lerobot/pusht` and other LeRobot datasets.
These dependencies are **optional** — default `pytest` does not require them.

```bash
pip install -r requirements-robot.txt
```

This installs: `lerobot`, `datasets`, `huggingface_hub`, `av`, `jsonlines`.

### Windows Notes

- `lerobot` uses `pyav` for video decoding.  If you see
  `torchcodec not available, falling back to pyav`, this is normal.
- If HF Hub symlink warnings appear, set:
  ```powershell
  $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
  ```

## 5. Realdata Tests

Real PushT tests require LeRobot and network access.
They are **not** run by default.

```powershell
$env:RUN_REAL_PUSHT = "1"
python -m pytest tests/test_evaluate_pusht_cli.py -m realdata
```

Future dataset zoo tests (Stage 6+):

```powershell
$env:RUN_REAL_DATASET_ZOO = "1"
python -m pytest tests/test_inspect_robot_dataset_realdata.py -m realdata
```

## 6. Quick Start

### Toy 2D

```bash
python scripts/generate_toy_data.py --config configs/data/toy_2d.yaml --num-episodes 10
python scripts/train.py --config configs/train/debug.yaml
python scripts/infer_one.py --config configs/train/debug.yaml --ckpt outputs/checkpoints/best.pt
```

### PushT (real data)

```bash
pip install -r requirements-robot.txt
python scripts/train.py --config configs/train/pusht_debug.yaml
python scripts/evaluate_pusht.py --config configs/train/pusht_debug.yaml --ckpt outputs/checkpoints/best.pt
```

### Inspect a robot dataset

```bash
python scripts/inspect_robot_dataset.py --dataset-name pusht --mock-data
python scripts/inspect_robot_dataset.py --dataset-name pusht --repo-id lerobot/pusht --loader lerobot --max-samples 8
```

## 7. Project Outputs

```text
outputs/
├── checkpoints/     # last.pt, best.pt
├── dataset_reports/ # dataset schema / stats JSON
├── eval/            # evaluation reports
├── predictions/     # prediction PNGs
└── visualizations/

data/
├── toy_2d/          # generated Toy2D episodes
└── episodes/
```

Large files (checkpoints, datasets) are gitignored.

## 8. Common Problems

### PyTorch not installed

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### `mini_vla` import error

```bash
pip install -e .
```

### CUDA out of memory

Use `configs/train/debug.yaml` which uses smaller batch size (4).

### LeRobot import path

If you see `ModuleNotFoundError: No module named 'lerobot.common'`,
your lerobot version uses:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
```

**Not** `from lerobot.common...`.  The code handles this automatically.

### PushT data has no observation.image

If `datasets.load_dataset("lerobot/pusht")` does not return image frames,
use `loader: lerobot` (LeRobotDataset) instead of the default HF datasets
loader.  Set `loader: lerobot` in your config or pass `--loader lerobot`
to the CLI.

### HF Hub rate limits

```bash
export HF_TOKEN=your_token  # Linux
$env:HF_TOKEN = "your_token"  # PowerShell
```

### torchcodec warning

```
'torchcodec' is not available in your platform, falling back to 'pyav'
```

This is benign.  The decoder works with pyav as fallback.
