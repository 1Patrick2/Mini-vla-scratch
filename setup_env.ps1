Write-Host "[MiniVLA] Setting up Python environment..."

if (!(Test-Path ".venv")) {
    python -m venv .venv
}

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .

Write-Host "[MiniVLA] Environment setup completed."
Write-Host "Run: .\.venv\Scripts\Activate.ps1"
Write-Host "Verify: pytest"
