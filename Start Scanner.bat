@echo off
REM ============================================
REM Quiz Vulnerability Scanner - One-Click Launch (Windows)
REM ============================================
REM Double-click this file to start everything!

cd /d "%~dp0"

echo.
echo ============================================
echo     Quiz Vulnerability Scanner
echo     Starting up...
echo ============================================
echo.

REM Check if first-time setup is needed
if not exist "venv\Scripts\activate.bat" (
    echo First-time setup required!
    echo.
    echo Please run "First Time Setup.bat" first.
    echo.
    pause
    exit /b 1
)

REM 1. Start Chrome with debugging (if not already running)
netstat -an | find "9222" >nul
if errorlevel 1 (
    echo Starting Chrome with remote debugging...
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%TEMP%\chrome-quiz-scanner" --new-window about:blank
    timeout /t 3 >nul
)
echo [OK] Chrome ready

REM 2. Start Ollama if not running
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if errorlevel 1 (
    echo Starting Ollama...
    start /B ollama serve >nul 2>&1
    timeout /t 3 >nul
)
echo [OK] Ollama ready

REM 3. Check for required models (in background)
start /B cmd /c "ollama list | find "llama3:8b" >nul || (echo Downloading llama3:8b... & ollama pull llama3:8b)"
start /B cmd /c "ollama list | find "llava" >nul || (echo Downloading llava... & ollama pull llava)"

REM 4. Activate virtual environment
call venv\Scripts\activate.bat
echo [OK] Environment ready

REM 5. Start the scanner
echo.
echo Opening Scanner in your browser...
echo (Keep this window open while using the scanner)
echo.

python -m streamlit run App.py --server.headless=true --browser.gatherUsageStats=false

echo.
echo Scanner closed.
pause
