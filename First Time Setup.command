#!/bin/bash
# ============================================
# Quiz Vulnerability Scanner - First Time Setup (macOS)
# ============================================
# Double-click this file for guided setup!

cd "$(dirname "$0")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

clear
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║     🔍 Quiz Vulnerability Scanner                      ║"
echo "║        First Time Setup                                ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "This will install everything you need. It may take 10-15 minutes."
echo ""
read -p "Press Enter to begin setup..."

# ============================================
# Step 1: Xcode Command Line Tools
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 1 of 6: Checking Xcode Command Line Tools"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if xcode CLI tools are installed
if ! xcode-select -p &>/dev/null; then
    echo -e "${YELLOW}Xcode Command Line Tools not found.${NC}"
    echo "Installing now... (this may take several minutes)"
    echo ""

    # Trigger the install
    xcode-select --install 2>/dev/null

    echo "A dialog should appear asking to install Command Line Tools."
    echo "Please click 'Install' and wait for it to complete."
    echo ""

    # Wait for installation to complete
    echo "Waiting for installation to complete..."
    until xcode-select -p &>/dev/null; do
        sleep 5
        echo "  Still waiting for Xcode CLI tools..."
    done

    echo -e "${GREEN}✓ Xcode Command Line Tools installed${NC}"
else
    echo -e "${GREEN}✓ Xcode Command Line Tools already installed${NC}"
fi

# ============================================
# Step 2: Check Python
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 2 of 6: Checking Python"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo -e "${GREEN}✓ $PYTHON_VERSION installed${NC}"
else
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo ""
    echo "Please install Python first:"
    echo "  1. Go to https://www.python.org/downloads/"
    echo "  2. Download and install Python 3.9 or later"
    echo "  3. Run this setup again"
    echo ""
    read -p "Press Enter to open the download page..."
    open "https://www.python.org/downloads/"
    exit 1
fi

# ============================================
# Step 3: Python Environment
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 3 of 6: Setting up Python environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Remove broken venv if it exists but is incomplete
if [ -d "venv" ] && [ ! -f "venv/bin/activate" ]; then
    echo "Removing incomplete virtual environment..."
    rm -rf venv
fi

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv

    # Verify it was created successfully
    if [ ! -f "venv/bin/activate" ]; then
        echo -e "${RED}✗ Failed to create virtual environment${NC}"
        echo ""
        echo "This might be a permissions issue. Try running:"
        echo "  python3 -m venv venv"
        echo ""
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Verify activation worked
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}✗ Failed to activate virtual environment${NC}"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Installing Python packages..."

# Use python3 -m pip as a more reliable method
python3 -m pip install --upgrade pip -q 2>/dev/null || python -m pip install --upgrade pip -q

if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt -q 2>/dev/null || python -m pip install -r requirements.txt -q

    # Verify key packages installed
    if python3 -c "import streamlit" 2>/dev/null; then
        echo -e "${GREEN}✓ Python packages installed${NC}"
    else
        echo -e "${RED}✗ Package installation may have failed${NC}"
        echo "Trying again with verbose output..."
        python3 -m pip install -r requirements.txt
    fi
else
    echo -e "${RED}✗ requirements.txt not found${NC}"
    read -p "Press Enter to exit..."
    exit 1
fi

# ============================================
# Step 4: Browser Automation
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 4 of 6: Installing browser automation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Use python -m playwright to ensure we use the venv version
python3 -m playwright install chromium 2>/dev/null || python -m playwright install chromium

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Browser automation ready${NC}"
else
    echo -e "${YELLOW}⚠ Browser automation may need manual setup${NC}"
    echo "Try running: python -m playwright install chromium"
fi

# ============================================
# Step 5: Ollama
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 5 of 6: Setting up Ollama (AI engine)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if Ollama is installed (either as command or as app)
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama already installed${NC}"
elif [ -d "/Applications/Ollama.app" ]; then
    echo -e "${GREEN}✓ Ollama.app found${NC}"
else
    echo "Ollama not found."
    echo ""

    # Check for Homebrew
    if command -v brew &> /dev/null; then
        echo "Installing Ollama via Homebrew..."
        brew install ollama
    else
        # On macOS without Homebrew, download the .app directly
        echo "Downloading Ollama for macOS..."
        echo ""

        # Download the macOS zip file
        OLLAMA_DMG="/tmp/Ollama.zip"
        curl -L -o "$OLLAMA_DMG" "https://ollama.com/download/Ollama-darwin.zip"

        if [ -f "$OLLAMA_DMG" ]; then
            echo "Extracting Ollama..."
            unzip -q -o "$OLLAMA_DMG" -d /tmp/

            if [ -d "/tmp/Ollama.app" ]; then
                echo "Installing Ollama to Applications..."
                mv /tmp/Ollama.app /Applications/
                rm -f "$OLLAMA_DMG"
                echo -e "${GREEN}✓ Ollama installed${NC}"
            else
                echo -e "${YELLOW}Extraction failed.${NC}"
            fi
        fi

        # Verify installation worked
        if [ ! -d "/Applications/Ollama.app" ] && ! command -v ollama &> /dev/null; then
            echo -e "${YELLOW}Automatic install didn't work.${NC}"
            echo ""
            echo "Please install Ollama manually:"
            echo "  1. Go to https://ollama.ai/download"
            echo "  2. Download and install the macOS app"
            echo "  3. Run this setup again"
            echo ""
            read -p "Press Enter to open the download page..."
            open "https://ollama.ai/download"
            exit 1
        fi
    fi
fi

# Start Ollama service
echo "Starting Ollama service..."

# Try to start via the app first (if installed as app)
if [ -d "/Applications/Ollama.app" ]; then
    open -a Ollama
    sleep 5
elif command -v ollama &> /dev/null; then
    ollama serve &>/dev/null &
    sleep 5
fi

# Wait for Ollama to be ready (with timeout)
echo "Waiting for Ollama to start..."
OLLAMA_READY=false
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        OLLAMA_READY=true
        break
    fi
    sleep 1
done

if [ "$OLLAMA_READY" = true ]; then
    echo -e "${GREEN}✓ Ollama service running${NC}"
else
    echo -e "${YELLOW}⚠ Ollama service may not be running${NC}"
    echo "Trying to start it again..."
    ollama serve &>/dev/null &
    sleep 5
fi

# ============================================
# Step 6: Download AI Models
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 6 of 6: Downloading AI models"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This will download ~9GB of AI models. This is a one-time download."
echo ""

echo "Downloading Llama 3 (main model, ~4.7GB)..."
ollama pull llama3:8b

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠ Failed to download llama3:8b${NC}"
    echo "You can try manually later: ollama pull llama3:8b"
fi

echo ""
echo "Downloading LLaVA (vision model, ~4.5GB)..."
ollama pull llava

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠ Failed to download llava${NC}"
    echo "You can try manually later: ollama pull llava"
fi

echo -e "${GREEN}✓ AI models ready${NC}"

# ============================================
# Create directories
# ============================================
mkdir -p output/raw_attempts output/reports output/dashboards temp_screenshots

# ============================================
# Complete!
# ============================================
echo ""
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║   ✓ Setup Complete!                                    ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo -e "To use the scanner:"
echo ""
echo -e "  1. Double-click '${CYAN}Start Scanner.command${NC}'"
echo ""
echo "  2. In the Chrome window that opens:"
echo "     • Go to your Moodle/LMS"
echo "     • Log in and navigate to your quiz"
echo "     • Start a quiz attempt"
echo ""
echo "  3. Follow the steps in the Scanner web interface"
echo ""
read -p "Press Enter to close this window..."
