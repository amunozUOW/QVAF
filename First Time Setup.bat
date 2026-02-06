@echo off
REM ============================================
REM Quiz Vulnerability Scanner - First Time Setup (Windows)
REM ============================================
REM Double-click this file for guided setup!

cd /d "%~dp0"

echo.
echo ========================================================
echo.
echo     Quiz Vulnerability Scanner
echo     First Time Setup (Windows)
echo.
echo ========================================================
echo.
echo This will install everything you need. It may take 10-15 minutes.
echo.
pause

REM ============================================
REM Step 1: Check Python
REM ============================================
echo.
echo --------------------------------------------------------
echo  Step 1 of 5: Checking Python
echo --------------------------------------------------------

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python first:
    echo   1. Go to https://www.python.org/downloads/
    echo   2. Download Python 3.9 or later
    echo   3. IMPORTANT: Check "Add Python to PATH" during install
    echo   4. Run this setup again
    echo.
    echo Opening download page...
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo [OK] %PYVER% installed

REM ============================================
REM Step 2: Create Virtual Environment
REM ============================================
echo.
echo --------------------------------------------------------
echo  Step 2 of 5: Setting up Python environment
echo --------------------------------------------------------

if exist "venv\Scripts\activate.bat" (
    echo [OK] Virtual environment already exists
) else (
    echo Creating virtual environment...
    python -m venv venv

    if not exist "venv\Scripts\activate.bat" (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing Python packages (this may take a few minutes)...
echo.

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install requirements with visible output so user can see progress
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Package installation failed!
    echo.
    echo Try running manually:
    echo   python -m pip install streamlit playwright ollama chromadb
    echo.
    pause
    exit /b 1
)

REM Verify key packages installed
echo.
echo Verifying installation...
python -c "import streamlit; print('  streamlit:', streamlit.__version__)" 2>nul
if errorlevel 1 (
    echo [ERROR] Streamlit not installed!
    echo.
    echo Installing streamlit directly...
    python -m pip install streamlit
)

python -c "import playwright; print('  playwright: OK')" 2>nul
if errorlevel 1 (
    echo [ERROR] Playwright not installed!
    python -m pip install playwright
)

python -c "import chromadb; print('  chromadb: OK')" 2>nul
if errorlevel 1 (
    echo [ERROR] ChromaDB not installed!
    python -m pip install chromadb
)

echo.
echo [OK] Python packages installed

REM ============================================
REM Step 3: Browser Automation
REM ============================================
echo.
echo --------------------------------------------------------
echo  Step 3 of 5: Installing browser automation
echo --------------------------------------------------------

python -m playwright install chromium
if errorlevel 1 (
    echo [WARNING] Playwright installation may have issues
    echo Try running: python -m playwright install chromium
) else (
    echo [OK] Browser automation ready
)

REM ============================================
REM Step 4: Check Ollama
REM ============================================
echo.
echo --------------------------------------------------------
echo  Step 4 of 5: Setting up Ollama (AI engine)
echo --------------------------------------------------------

ollama --version >nul 2>&1
if errorlevel 1 (
    echo Ollama not found.
    echo.
    echo Please install Ollama:
    echo   1. Go to https://ollama.ai/download/windows
    echo   2. Download and run the installer
    echo   3. After installation, run this setup again
    echo.
    echo Opening download page...
    start https://ollama.ai/download/windows
    echo.
    echo After installing Ollama, please run this setup again.
    pause
    exit /b 0
) else (
    echo [OK] Ollama is installed
)

REM Start Ollama if not running
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if errorlevel 1 (
    echo Starting Ollama service...
    start /B ollama serve >nul 2>&1
    timeout /t 5 >nul
)

REM ============================================
REM Step 5: Download AI Models
REM ============================================
echo.
echo --------------------------------------------------------
echo  Step 5 of 5: Downloading AI models
echo --------------------------------------------------------
echo.
echo This will download ~9GB of AI models. This is a one-time download.
echo.

echo Downloading Llama 3 (main model, ~4.7GB)...
ollama pull llama3:8b

echo.
echo Downloading LLaVA (vision model, ~4.5GB)...
ollama pull llava

echo [OK] AI models ready

REM ============================================
REM Create directories
REM ============================================
if not exist "output\raw_attempts" mkdir "output\raw_attempts"
if not exist "output\reports" mkdir "output\reports"
if not exist "output\dashboards" mkdir "output\dashboards"
if not exist "temp_screenshots" mkdir "temp_screenshots"

REM ============================================
REM Complete!
REM ============================================
echo.
echo.
echo ========================================================
echo.
echo   [OK] Setup Complete!
echo.
echo ========================================================
echo.
echo To use the scanner:
echo.
echo   1. Double-click 'Start Scanner.bat'
echo.
echo   2. In the Chrome window that opens:
echo      - Go to your Moodle/LMS
echo      - Log in and navigate to your quiz
echo      - Start a quiz attempt
echo.
echo   3. Follow the steps in the Scanner web interface
echo.
pause
