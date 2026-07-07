#!/usr/bin/env bash
set -e

echo "[MiniVLA] Setting up Python environment..."

ROBOT=false
if [ "$1" = "--robot" ]; then
  ROBOT=true
fi

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .

if [ "$ROBOT" = true ]; then
  echo "[MiniVLA] Installing robot dataset dependencies..."
  pip install -r requirements-robot.txt
  echo "[MiniVLA] Robot dependencies installed."
fi

echo "[MiniVLA] Environment setup completed."
echo "Run: source .venv/bin/activate"
echo "Verify: pytest"
