#!/bin/bash
#
# Quiz Vulnerability Scanner - Installer
# =======================================
# Run this script to set up the scanner and create a desktop app
#
# Usage: bash install_scanner.sh
#

set -e

# Configuration
INSTALL_DIR="$HOME/quiz-scanner"
DESKTOP_DIR="$HOME/Desktop"

echo "========================================"
echo "  Quiz Vulnerability Scanner Installer"
echo "========================================"
echo ""

# Step 1: Create installation directory
echo "Step 1: Creating installation directory..."
mkdir -p "$INSTALL_DIR"
echo "  → Created $INSTALL_DIR"

# Step 2: Check if source files exist in current directory
echo ""
echo "Step 2: Checking for source files..."

REQUIRED_FILES=(
    "app_v2.py"
)

OPTIONAL_FILES=(
    "multi_sample_llm.py"
    "quiz_browser_enhanced.py"
    "reform_agent.py"
    "analysis_agent.py"
    "evolution_agent.py"
)

missing_required=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ Found $file"
        cp "$file" "$INSTALL_DIR/"
    else
        echo "  ✗ Missing required file: $file"
        missing_required=1
    fi
done

for file in "${OPTIONAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ Found $file"
        cp "$file" "$INSTALL_DIR/"
    else
        echo "  - Optional file not found: $file (skipping)"
    fi
done

if [ $missing_required -eq 1 ]; then
    echo ""
    echo "ERROR: Missing required files. Please run this script from the"
    echo "directory containing your scanner files."
    exit 1
fi

# Step 3: Copy chroma_db if it exists
echo ""
echo "Step 3: Checking for RAG database..."
if [ -d "chroma_db" ]; then
    echo "  → Copying chroma_db..."
    cp -r chroma_db "$INSTALL_DIR/"
    echo "  ✓ RAG database copied"
else
    echo "  - No chroma_db found (RAG will not be available)"
fi

# Step 4: Create the launcher script
echo ""
echo "Step 4: Creating launcher script..."

cat > "$INSTALL_DIR/launch.sh" << 'LAUNCHER'
#!/bin/bash
SCANNER_DIR="$(dirname "$0")"
cd "$SCANNER_DIR"

# Start Chrome if not already running with debug port
if ! lsof -i :9222 > /dev/null 2>&1; then
    echo "Starting Chrome in debug mode..."
    mkdir -p /tmp/chrome-debug-profile
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
        --remote-debugging-port=9222 \
        --user-data-dir=/tmp/chrome-debug-profile \
        --no-first-run \
        --no-default-browser-check &
    sleep 3
fi

# Start Streamlit if not already running
if ! lsof -i :8501 > /dev/null 2>&1; then
    echo "Starting Streamlit..."
    python3 -m streamlit run app_v2.py \
        --server.port=8501 \
        --server.headless=true &
    sleep 4
fi

# Open in browser
open "http://localhost:8501"

echo ""
echo "Quiz Scanner is running!"
echo "Close this terminal to stop the scanner."
wait
LAUNCHER

chmod +x "$INSTALL_DIR/launch.sh"
echo "  ✓ Launcher script created"

# Step 5: Create the macOS app bundle
echo ""
echo "Step 5: Creating macOS app bundle..."

APP_NAME="Quiz Scanner"
APP_DIR="$DESKTOP_DIR/$APP_NAME.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

# Remove old app if exists
rm -rf "$APP_DIR"

# Create app structure
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Create the executable
cat > "$MACOS_DIR/QuizScanner" << EXECUTABLE
#!/bin/bash
open -a Terminal "$INSTALL_DIR/launch.sh"
EXECUTABLE

chmod +x "$MACOS_DIR/QuizScanner"

# Create Info.plist
cat > "$CONTENTS_DIR/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>QuizScanner</string>
    <key>CFBundleIdentifier</key>
    <string>com.quizscanner.app</string>
    <key>CFBundleName</key>
    <string>Quiz Scanner</string>
    <key>CFBundleDisplayName</key>
    <string>Quiz Scanner</string>
    <key>CFBundleVersion</key>
    <string>2.0</string>
    <key>CFBundleShortVersionString</key>
    <string>2.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

# Create a simple icon (colored square as placeholder)
# In a real scenario, you'd include a proper .icns file

echo "  ✓ App bundle created at $APP_DIR"

# Step 6: Remove quarantine attribute (macOS security)
echo ""
echo "Step 6: Configuring security settings..."
xattr -cr "$APP_DIR" 2>/dev/null || true
echo "  ✓ Security attributes configured"

# Step 7: Verify Python dependencies
echo ""
echo "Step 7: Checking Python dependencies..."

check_package() {
    python3 -c "import $1" 2>/dev/null && echo "  ✓ $1" || echo "  ✗ $1 (install with: pip3 install $1)"
}

check_package streamlit
check_package playwright
check_package chromadb
check_package ollama

# Done!
echo ""
echo "========================================"
echo "  Installation Complete!"
echo "========================================"
echo ""
echo "A 'Quiz Scanner' app has been created on your Desktop."
echo ""
echo "To use:"
echo "  1. Double-click 'Quiz Scanner' on your Desktop"
echo "  2. A terminal window will open and start the services"
echo "  3. Chrome will open - log into Moodle and start a quiz"
echo "  4. The scanner web app will open in your browser"
echo ""
echo "Files installed to: $INSTALL_DIR"
echo ""

# Offer to open the app now
read -p "Would you like to launch the scanner now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    open "$APP_DIR"
fi
