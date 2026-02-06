#!/bin/bash
# ============================================
# Quiz Vulnerability Scanner - Launch Script (macOS/Linux)
# ============================================
# Run with: bash scripts/run_scanner.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"

cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve &
    sleep 2
fi

# Check if Chrome debug port is available
if ! nc -z localhost 9222 2>/dev/null; then
    echo ""
    echo "⚠️  Chrome is not running with remote debugging enabled."
    echo ""
    echo "Please start Chrome in a separate terminal with:"
    echo ""
    echo "  /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome \\"
    echo "    --remote-debugging-port=9222 \\"
    echo "    --user-data-dir=/tmp/chrome-debug"
    echo ""
    echo "Then navigate to your quiz and start an attempt."
    echo ""
    read -p "Press Enter when Chrome is ready..."
fi

echo ""
echo "Starting Quiz Vulnerability Scanner..."
echo "Opening browser at http://localhost:8501"
echo ""

python -m streamlit run App.py
