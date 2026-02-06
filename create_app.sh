#!/bin/bash
#
# Run this from your quiz-vulnerability folder
#

SCANNER_DIR="$(pwd)"
APP_PATH="$HOME/Desktop/Quiz Scanner.app"

# Check for App.py
if [ ! -f "App.py" ]; then
    echo "ERROR: App.py not found. Run this from your quiz-vulnerability folder."
    exit 1
fi

rm -rf "$APP_PATH"
mkdir -p "$APP_PATH/Contents/MacOS"

cat > "$APP_PATH/Contents/MacOS/launch" << EOF
#!/bin/bash
osascript -e 'tell application "Terminal"
    activate
    do script "cd \"$SCANNER_DIR\" && \"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome\" --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug --no-first-run & sleep 3 && python3 -m streamlit run App.py"
end tell'
EOF

chmod +x "$APP_PATH/Contents/MacOS/launch"

cat > "$APP_PATH/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>launch</string>
    <key>CFBundleName</key>
    <string>Quiz Scanner</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>
EOF

xattr -cr "$APP_PATH" 2>/dev/null

echo "Done! Quiz Scanner app is on your Desktop."