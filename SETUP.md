# Setup Guide

This document explains how to set up the development environment for `mini-vla-from-scratch`.

## 1. Recommended Environment

- Python 3.10 or 3.11
- Conda or venv
- PyTorch
- CUDA is optional
- Linux / WSL2 is recommended
- Windows PowerShell is supported for basic usage

## 2. Create Environment with Conda

```bash
conda env create -f environment.yml
conda activate mini-vla
```

If the environment already exists:

```bash
conda activate mini-vla
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

## 3. Create Environment with venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

## 4. One-Command Setup

Linux / WSL:

```bash
bash setup_env.sh
```

Windows PowerShell:

```powershell
.\setup_env.ps1
```

## 5. Verify Installation

```bash
# Run unit tests
pytest

# Check package import
python -c "import mini_vla; print('mini_vla import OK')"

# Dry-run training (Stage 0 skeleton)
python scripts/train.py --config configs/train/debug.yaml --dry-run
```

## 6. Stage 1 Preview (after Stage 1 is implemented)

Generate toy 2D data and load it:

```bash
python scripts/generate_toy_data.py --config configs/data/toy_2d.yaml --num-episodes 100

python -c "
from mini_vla.datasets.toy_2d_dataset import Toy2DDataset
ds = Toy2DDataset('data/toy_2d')
sample = ds[0]
print('Keys:', list(sample.keys()))
print('Image shape:', sample['image'].shape)
"
```

> **Note:** These commands are placeholders in Stage 0. Toy data generation will be implemented in Stage 1.

## 7. Project Outputs

Generated files are stored under:

```text
outputs/
├── checkpoints/
├── logs/
├── predictions/
└── visualizations/
```

Dataset files are stored under:

```text
data/
├── toy_2d/
└── episodes/
```

Large files should not be committed to Git.

## 8. Common Problems

### PyTorch is not installed correctly

Install PyTorch according to your CUDA version from the official PyTorch website.
For CPU-only usage:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Import error: `mini_vla` not found

Run `pip install -e .` from the project root. This installs the package in editable mode.

### Dataset not found

Generate toy data first once Stage 1 is implemented:

```bash
python scripts/generate_toy_data.py --config configs/data/toy_2d.yaml --num-episodes 100
```

### CUDA out of memory

Use the debug config which uses a smaller batch size:

```bash
python scripts/train.py --config configs/train/debug.yaml --dry-run
```

Or reduce `batch_size` in your config file.

### Windows PowerShell execution policy

If you get an execution policy error when activating `.venv\Scripts\Activate.ps1`, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
