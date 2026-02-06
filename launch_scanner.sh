#!/bin/bash
#
# Quiz Vulnerability Scanner Launcher
# ====================================
# Double-click to start the scanner with all dependencies
#

# Configuration - UPDATE THESE PATHS
SCANNER_DIR="$HOME/quiz-scanner"  # Where your scanner files are located
CHROME_APP="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
STREAMLIT_PORT=8501

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "  Quiz Vulnerability Scanner Launcher"
echo "========================================"
echo ""

# Check if Chrome debug session is already running
if lsof -i :9222 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Chrome debug session already running on port 9222${NC}"
else
    echo -e "${YELLOW}Starting Chrome in debug mode...${NC}"
    
    # Create temp profile directory if it doesn't exist
    mkdir -p /tmp/chrome-debug-profile
    
    # Launch Chrome with remote debugging
    "$CHROME_APP" \
        --remote-debugging-port=9222 \
        --user-data-dir=/tmp/chrome-debug-profile \
        --no-first-run \
        --no-default-browser-check \
        > /dev/null 2>&1 &
    
    # Wait for Chrome to start
    echo "Waiting for Chrome to initialize..."
    sleep 3
    
    if lsof -i :9222 > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Chrome started successfully${NC}"
    else
        echo -e "${RED}✗ Failed to start Chrome. Please start it manually.${NC}"
    fi
fi

echo ""

# Check if Streamlit is already running
if lsof -i :$STREAMLIT_PORT > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Streamlit already running on port $STREAMLIT_PORT${NC}"
    echo "Opening browser..."
    open "http://localhost:$STREAMLIT_PORT"
else
    echo -e "${YELLOW}Starting Streamlit app...${NC}"
    
    # Change to scanner directory
    if [ -d "$SCANNER_DIR" ]; then
        cd "$SCANNER_DIR"
    else
        echo -e "${RED}Scanner directory not found: $SCANNER_DIR${NC}"
        echo "Please update SCANNER_DIR in this script."
        echo ""
        echo "Press any key to exit..."
        read -n 1
        exit 1
    fi
    
    # Check for app file
    if [ -f "app_v2.py" ]; then
        APP_FILE="app_v2.py"
    elif [ -f "app.py" ]; then
        APP_FILE="app.py"
    else
        echo -e "${RED}No app.py or app_v2.py found in $SCANNER_DIR${NC}"
        echo "Press any key to exit..."
        read -n 1
        exit 1
    fi
    
    echo "Starting $APP_FILE..."
    
    # Start Streamlit in background and capture PID
    python3 -m streamlit run "$APP_FILE" \
        --server.port=$STREAMLIT_PORT \
        --server.headless=true \
        --browser.gatherUsageStats=false \
        > /tmp/streamlit-scanner.log 2>&1 &
    
    STREAMLIT_PID=$!
    
    # Wait for Streamlit to start
    echo "Waiting for Streamlit to initialize..."
    for i in {1..10}; do
        if lsof -i :$STREAMLIT_PORT > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    
    if lsof -i :$STREAMLIT_PORT > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Streamlit started successfully${NC}"
        
        # Open in default browser
        sleep 1
        open "http://localhost:$STREAMLIT_PORT"
    else
        echo -e "${RED}✗ Failed to start Streamlit${NC}"
        echo "Check log: /tmp/streamlit-scanner.log"
        cat /tmp/streamlit-scanner.log
        echo ""
        echo "Press any key to exit..."
        read -n 1
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}Scanner is ready!${NC}"
echo ""
echo "Next steps:"
echo "  1. In the Chrome window, log into Moodle"
echo "  2. Navigate to your quiz and start an attempt"
echo "  3. Use the Streamlit app to run scans"
echo ""
echo "To stop the scanner, close this terminal window."
echo ""

# Keep the script running so user can see output
# This also keeps Streamlit running
wait
