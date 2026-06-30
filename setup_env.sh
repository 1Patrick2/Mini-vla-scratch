#!/usr/bin/env bash
set -e

echo "[MiniVLA] Setting up Python environment..."

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .

echo "[MiniVLA] Environment setup completed."
echo "Run: source .venv/bin/activate"
echo "Verify: pytest"
