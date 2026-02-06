# ============================================
# Quiz Vulnerability Scanner - Windows Setup
# ============================================
# This script sets up the complete environment for the Quiz Vulnerability Scanner.
# Run with: powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Quiz Vulnerability Scanner - Windows Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Get project directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

Write-Host "Working directory: $ProjectDir"
Write-Host ""

# ============================================
# 1. Check Python
# ============================================
Write-Host "[1/7] Checking Python installation..." -ForegroundColor Yellow

try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ $pythonVersion found" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found" -ForegroundColor Red
    Write-Host "Please install Python 3.9+ from https://www.python.org/downloads/"
    Write-Host "Make sure to check 'Add Python to PATH' during installation"
    exit 1
}

# ============================================
# 2. Create Virtual Environment
# ============================================
Write-Host ""
Write-Host "[2/7] Setting up virtual environment..." -ForegroundColor Yellow

if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# ============================================
# 3. Install Python Dependencies
# ============================================
Write-Host ""
Write-Host "[3/7] Installing Python dependencies..." -ForegroundColor Yellow

pip install --upgrade pip -q
pip install -r requirements.txt -q
Write-Host "✓ Python packages installed" -ForegroundColor Green

# ============================================
# 4. Install Playwright Browsers
# ============================================
Write-Host ""
Write-Host "[4/7] Installing Playwright browsers..." -ForegroundColor Yellow

playwright install chromium
Write-Host "✓ Playwright chromium installed" -ForegroundColor Green

# ============================================
# 5. Check Ollama Installation
# ============================================
Write-Host ""
Write-Host "[5/7] Checking Ollama installation..." -ForegroundColor Yellow

$ollamaPath = Get-Command ollama -ErrorAction SilentlyContinue

if ($ollamaPath) {
    Write-Host "✓ Ollama is installed" -ForegroundColor Green
} else {
    Write-Host "Ollama not found." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please install Ollama manually:" -ForegroundColor Cyan
    Write-Host "  1. Download from: https://ollama.ai/download/windows"
    Write-Host "  2. Run the installer"
    Write-Host "  3. Re-run this setup script"
    Write-Host ""
    $response = Read-Host "Press Enter to open the download page, or 'S' to skip"
    if ($response -ne 'S') {
        Start-Process "https://ollama.ai/download/windows"
        Write-Host "After installing Ollama, please re-run this setup script."
        exit 0
    }
}

# ============================================
# 6. Download AI Models (if Ollama available)
# ============================================
Write-Host ""
Write-Host "[6/7] Setting up AI models..." -ForegroundColor Yellow

if ($ollamaPath) {
    Write-Host "Downloading llama3:8b (main model, ~4.7GB)..."
    ollama pull llama3:8b

    Write-Host "Downloading llava (vision model, ~4.5GB)..."
    ollama pull llava

    Write-Host "✓ AI models ready" -ForegroundColor Green
} else {
    Write-Host "Skipping model download (Ollama not installed)" -ForegroundColor Yellow
}

# ============================================
# 7. Create Output Directories
# ============================================
Write-Host ""
Write-Host "[7/7] Creating output directories..." -ForegroundColor Yellow

New-Item -ItemType Directory -Force -Path "output\raw_attempts" | Out-Null
New-Item -ItemType Directory -Force -Path "output\reports" | Out-Null
New-Item -ItemType Directory -Force -Path "output\dashboards" | Out-Null
New-Item -ItemType Directory -Force -Path "temp_screenshots" | Out-Null
Write-Host "✓ Directories created" -ForegroundColor Green

# ============================================
# Setup Complete
# ============================================
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the scanner:" -ForegroundColor White
Write-Host ""
Write-Host "  1. Start Chrome with debugging (in a NEW PowerShell window):" -ForegroundColor White
Write-Host '     & "C:\Program Files\Google\Chrome\Application\chrome.exe" `' -ForegroundColor Gray
Write-Host '       --remote-debugging-port=9222 `' -ForegroundColor Gray
Write-Host '       --user-data-dir=C:\temp\chrome-debug' -ForegroundColor Gray
Write-Host ""
Write-Host "  2. In the Chrome window, navigate to your quiz and start an attempt" -ForegroundColor White
Write-Host ""
Write-Host "  3. Run the scanner (in THIS PowerShell window):" -ForegroundColor White
Write-Host "     .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "     python -m streamlit run App.py" -ForegroundColor Gray
Write-Host ""
Write-Host "The scanner will open at: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
