# Changelog

## 0.1.0 — Stage 0 Skeleton (2026-06-30)

### Added
- Project skeleton with package structure (`mini_vla/`)
- Config system: YAML loading, base merge, schema validation
- Training CLI with dry-run mode
- Action clipping / safety wrapper utilities
- Basic unit tests (4 passing)
- All Stage 0 root files:
  - `pyproject.toml` with black/ruff/pytest/mypy config
  - `requirements.txt` and `requirements-dev.txt`
  - `environment.yml` for Conda
  - `setup_env.sh` (Linux/WSL) and `setup_env.ps1` (Windows)
  - `.env.example` and `.gitignore`
- 6 design documents in `docs/` (00–05)
- Comprehensive long-term plan (`PLAN.md`)

### Changed
- README aligned with actual Stage 0 skeleton state
  - Separated `Implemented` / `Planned` sections
  - Removed overstated feature claims
- ROADMAP rewritten as Stage 0–6 with commands and acceptance criteria
- SETUP expanded with verification commands, Stage 1 preview, Common Problems

### Not Yet Implemented (Planned for upcoming stages)
- Toy 2D dataset generation (Stage 1)
- Dataset loader (Stage 1)
- MiniVLA model forward (Stage 2)
- Training loop (Stage 3)
- Inference and visualization (Stage 4)
- Fake robot rollout (Stage 5)
- Open Kaka adapter (Stage 6)
