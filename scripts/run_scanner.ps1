# ============================================
# Quiz Vulnerability Scanner - Launch Script (Windows)
# ============================================
# Run with: powershell -ExecutionPolicy Bypass -File scripts\run_scanner.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

# Activate virtual environment if it exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    & ".\venv\Scripts\Activate.ps1"
}

# Check if Chrome debug port is available
$connection = Test-NetConnection -ComputerName localhost -Port 9222 -WarningAction SilentlyContinue -ErrorAction SilentlyContinue

if (-not $connection.TcpTestSucceeded) {
    Write-Host ""
    Write-Host "⚠️  Chrome is not running with remote debugging enabled." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please start Chrome in a separate PowerShell window with:" -ForegroundColor White
    Write-Host ""
    Write-Host '  & "C:\Program Files\Google\Chrome\Application\chrome.exe" `' -ForegroundColor Gray
    Write-Host '    --remote-debugging-port=9222 `' -ForegroundColor Gray
    Write-Host '    --user-data-dir=C:\temp\chrome-debug' -ForegroundColor Gray
    Write-Host ""
    Write-Host "Then navigate to your quiz and start an attempt." -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter when Chrome is ready..."
}

Write-Host ""
Write-Host "Starting Quiz Vulnerability Scanner..." -ForegroundColor Cyan
Write-Host "Opening browser at http://localhost:8501" -ForegroundColor Cyan
Write-Host ""

python -m streamlit run App.py
