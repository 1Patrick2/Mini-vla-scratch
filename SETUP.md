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

## 4. Verify Installation

Run unit tests:

```bash
pytest
```

Check package import:

```bash
python -c "import mini_vla; print('mini_vla import OK')"
```

Run a dry training command:

```bash
python scripts/train.py --config configs/train/debug.yaml
```

## 5. Project Outputs

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
