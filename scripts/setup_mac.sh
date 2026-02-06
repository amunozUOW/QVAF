#!/bin/bash
# ============================================
# Quiz Vulnerability Scanner - macOS Setup
# ============================================
# This script sets up the complete environment for the Quiz Vulnerability Scanner.
# Run with: bash scripts/setup_mac.sh

set -e  # Exit on any error

echo "============================================"
echo "Quiz Vulnerability Scanner - macOS Setup"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"

cd "$PROJECT_DIR"
echo "Working directory: $PROJECT_DIR"
echo ""

# ============================================
# 1. Check Python
# ============================================
echo -e "${YELLOW}[1/7] Checking Python installation...${NC}"

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
else
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo "Please install Python 3.9+ from https://www.python.org/downloads/"
    exit 1
fi

# ============================================
# 2. Create Virtual Environment
# ============================================
echo ""
echo -e "${YELLOW}[2/7] Setting up virtual environment...${NC}"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# ============================================
# 3. Install Python Dependencies
# ============================================
echo ""
echo -e "${YELLOW}[3/7] Installing Python dependencies...${NC}"

pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Python packages installed${NC}"

# ============================================
# 4. Install Playwright Browsers
# ============================================
echo ""
echo -e "${YELLOW}[4/7] Installing Playwright browsers...${NC}"

playwright install chromium
echo -e "${GREEN}✓ Playwright chromium installed${NC}"

# ============================================
# 5. Check/Install Ollama
# ============================================
echo ""
echo -e "${YELLOW}[5/7] Checking Ollama installation...${NC}"

if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama is installed${NC}"
else
    echo "Ollama not found. Installing via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install ollama
        echo -e "${GREEN}✓ Ollama installed${NC}"
    else
        echo -e "${RED}Homebrew not found. Please install Ollama manually:${NC}"
        echo "  brew install ollama"
        echo "  OR download from: https://ollama.ai/download"
        exit 1
    fi
fi

# ============================================
# 6. Start Ollama and Download Models
# ============================================
echo ""
echo -e "${YELLOW}[6/7] Setting up AI models...${NC}"

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve &
    sleep 3  # Wait for service to start
fi

echo "Downloading llama3:8b (main model, ~4.7GB)..."
ollama pull llama3:8b

echo "Downloading llava (vision model, ~4.5GB)..."
ollama pull llava

echo -e "${GREEN}✓ AI models ready${NC}"

# ============================================
# 7. Create Output Directories
# ============================================
echo ""
echo -e "${YELLOW}[7/7] Creating output directories...${NC}"

mkdir -p output/raw_attempts output/reports output/dashboards temp_screenshots
echo -e "${GREEN}✓ Directories created${NC}"

# ============================================
# Setup Complete
# ============================================
echo ""
echo "============================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "============================================"
echo ""
echo "To run the scanner:"
echo ""
echo "  1. Start Chrome with debugging (in a NEW terminal):"
echo "     /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome \\"
echo "       --remote-debugging-port=9222 \\"
echo "       --user-data-dir=/tmp/chrome-debug"
echo ""
echo "  2. In the Chrome window, navigate to your quiz and start an attempt"
echo ""
echo "  3. Run the scanner (in THIS terminal):"
echo "     source venv/bin/activate"
echo "     python -m streamlit run App.py"
echo ""
echo "The scanner will open at: http://localhost:8501"
echo ""
