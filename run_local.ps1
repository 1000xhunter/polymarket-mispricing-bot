param(
    [switch]$SkipInstall,
    [switch]$SetupOnly
)

$ErrorActionPreference = "Stop"

$venvPython = ".\.venv\Scripts\python.exe"

Write-Host "[1/5] Checking Python 3.11 launcher..."
try {
    py -3.11 --version | Out-Host
} catch {
    Write-Host "Python 3.11 not found." -ForegroundColor Red
    Write-Host "Install it with: winget install -e --id Python.Python.3.11" -ForegroundColor Yellow
    throw
}

if (-not (Test-Path $venvPython)) {
    Write-Host "[2/5] Creating virtual environment..."
    py -3.11 -m venv .venv
} else {
    Write-Host "[2/5] Virtual environment already exists."
}

if (-not $SkipInstall) {
    Write-Host "[3/5] Installing/updating dependencies..."
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r requirements.txt
} else {
    Write-Host "[3/5] Skipping dependency install (--SkipInstall)."
}

if (-not (Test-Path ".env")) {
    Write-Host "[4/5] Creating .env from .env.example"
    Copy-Item .env.example .env
    Write-Host "Open .env and fill your monitoring/alert settings as needed." -ForegroundColor Yellow
} else {
    Write-Host "[4/5] .env already exists."
}

if ($SetupOnly) {
    Write-Host "[5/5] Setup complete (--SetupOnly)."
    exit 0
}

Write-Host "[5/5] Starting bot... (Ctrl+C to stop)"
& $venvPython main.py
