#!/bin/bash
# ============================================
# Quiz Vulnerability Scanner - One-Click Launch (macOS)
# ============================================
# Double-click this file to start everything!

cd "$(dirname "$0")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

clear
echo ""
echo "╔════════════════════════════════════════════╗"
echo "║     Quiz Vulnerability Scanner             ║"
echo "║     Starting up...                         ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# 1. Check if Chrome is already running with debug port
if nc -z localhost 9222 2>/dev/null; then
    echo -e "${GREEN}✓ Chrome debug port already active${NC}"
else
    echo -e "${YELLOW}Starting Chrome with remote debugging...${NC}"

    # Start Chrome in background
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
        --remote-debugging-port=9222 \
        --user-data-dir="/tmp/chrome-quiz-scanner" \
        --new-window \
        "about:blank" &

    sleep 2
    echo -e "${GREEN}✓ Chrome started${NC}"
fi

# 2. Check/start Ollama
if ! pgrep -x "ollama" > /dev/null && ! pgrep -f "Ollama" > /dev/null; then
    echo -e "${YELLOW}Starting Ollama...${NC}"
    # Try to start Ollama app first, fallback to CLI
    if [ -d "/Applications/Ollama.app" ]; then
        open -a Ollama
    else
        ollama serve &>/dev/null &
    fi
    sleep 3
fi
echo -e "${GREEN}✓ Ollama running${NC}"

# 3. Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 4. Find an available port (8501-8510)
PORT=8501
while nc -z localhost $PORT 2>/dev/null; do
    PORT=$((PORT + 1))
    if [ $PORT -gt 8510 ]; then
        echo -e "${RED}Error: No available ports (8501-8510). Close other Streamlit apps and try again.${NC}"
        echo "Press Enter to exit..."
        read
        exit 1
    fi
done

echo -e "${GREEN}✓ Using port $PORT${NC}"

# 5. Start the scanner
echo ""
echo -e "${CYAN}Opening Scanner in your browser...${NC}"
echo -e "${CYAN}URL: http://localhost:$PORT${NC}"
echo -e "${CYAN}(Keep this window open while using the scanner)${NC}"
echo ""

# Open browser after a short delay (streamlit takes a moment to start)
(sleep 4 && open "http://localhost:$PORT") &

# Start streamlit on the specific port
python3 -m streamlit run App.py \
    --server.port=$PORT \
    --server.headless=true \
    --browser.gatherUsageStats=false

# When streamlit exits, offer to close Chrome
echo ""
echo "Scanner closed. Press Enter to exit..."
read
