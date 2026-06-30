# Stage 0 Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the initial Mini VLA project skeleton defined in `interact.md`.

**Architecture:** The skeleton keeps docs, configs, package modules, scripts, data, outputs, and tests separated by responsibility. Stage 0 includes only minimal executable code for config loading, a trainer shell, action clipping, and CLI parsing.

**Tech Stack:** Python 3.10+, PyYAML, pytest, optional PyTorch dependencies for later stages.

---

### Task 1: Project Baseline Files

**Files:**
- Create: `README.md`
- Create: `PROJECT_PRINCIPLES.md`
- Create: `SETUP.md`
- Create: `ROADMAP.md`
- Create: `CHANGELOG.md`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `environment.yml`
- Create: `setup_env.sh`
- Create: `setup_env.ps1`
- Create: `.env.example`
- Create: `.gitignore`

- [x] **Step 1: Write baseline files from `interact.md`**

- [x] **Step 2: Keep Stage 0 docs concise and responsibility-focused**

### Task 2: Config, Package, Scripts, and Tests

**Files:**
- Create: `configs/base.yaml`
- Create: `configs/model/mini_vla_cnn_gru.yaml`
- Create: `configs/model/mini_vla_resnet_gru.yaml`
- Create: `configs/data/toy_2d.yaml`
- Create: `configs/data/episode_robot.yaml`
- Create: `configs/train/debug.yaml`
- Create: `configs/train/default.yaml`
- Create: `mini_vla/config/loader.py`
- Create: `mini_vla/config/schema.py`
- Create: `mini_vla/training/trainer.py`
- Create: `mini_vla/robot_interface/action_adapter.py`
- Create: `scripts/train.py`
- Create: `tests/test_config_loader.py`
- Create: `tests/test_trainer.py`
- Create: `tests/test_action_adapter.py`

- [x] **Step 1: Add failing tests for config loading, trainer summary, and action clipping**

- [x] **Step 2: Implement the minimal code needed for those tests**

- [x] **Step 3: Add CLI entry points as thin wrappers**

### Task 3: Validate Stage 0

**Files:**
- Use: `scripts/train.py`
- Use: `tests/`

- [x] **Step 1: Run help command**

Run: `python scripts/train.py --help`

Expected: argparse help output with `--config` and `--dry-run`.

- [x] **Step 2: Run tests**

Run: `pytest`

Expected: all Stage 0 tests pass. Current local fallback: `python -m unittest discover -s tests` passes; `python -m pytest` requires pytest to be installed.
