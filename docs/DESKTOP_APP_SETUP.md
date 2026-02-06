# Creating a Clickable Desktop App for Quiz Scanner

This guide explains how to create a macOS application that launches the Quiz Vulnerability Scanner with a single click.

---

## Option 1: Shell Script Launcher (Simplest)

### Setup

1. **Create the scanner directory:**
   ```bash
   mkdir -p ~/quiz-scanner
   ```

2. **Copy scanner files to the directory:**
   ```bash
   cp App.py ~/quiz-scanner/
   cp multi_sample_llm.py ~/quiz-scanner/
   cp quiz_browser_enhanced.py ~/quiz-scanner/
   cp reform_agent.py ~/quiz-scanner/
   cp analysis_agent.py ~/quiz-scanner/
   # Copy any other required files
   ```

3. **Copy and configure the launcher:**
   ```bash
   cp launch_scanner.sh ~/quiz-scanner/
   chmod +x ~/quiz-scanner/launch_scanner.sh
   ```

4. **Edit the launcher to set your path** (if different from `~/quiz-scanner`):
   ```bash
   nano ~/quiz-scanner/launch_scanner.sh
   # Update SCANNER_DIR if needed
   ```

5. **Create a Desktop alias:**
   - Open Finder
   - Navigate to `~/quiz-scanner/`
   - Right-click `launch_scanner.sh`
   - Select "Make Alias"
   - Drag the alias to your Desktop
   - Rename to "Quiz Scanner"

6. **Double-click the alias to run!**

---

## Option 2: AppleScript Application (Recommended)

This creates a proper macOS `.app` that appears in Finder with an icon.

### Setup

1. **Open Script Editor:**
   - Press `Cmd + Space`, type "Script Editor", press Enter

2. **Create new script:**
   - File → New

3. **Paste the AppleScript:**
   - Open `QuizScanner.applescript` in a text editor
   - Copy all contents
   - Paste into Script Editor

4. **Update the scanner path:**
   - Find this line near the top:
     ```applescript
     set scannerDirectory to (POSIX path of (path to home folder)) & "quiz-scanner"
     ```
   - Change `quiz-scanner` to your actual folder name if different

5. **Export as Application:**
   - File → Export...
   - Set "File Format" to **Application**
   - Save as "Quiz Scanner" to Desktop
   - Check "Run-only" (optional, for cleaner app)

6. **Add a custom icon (optional):**
   - Find an icon you like (`.icns` or `.png`)
   - Right-click "Quiz Scanner.app" → Get Info
   - Drag your icon onto the icon in the top-left of the Info window

7. **Double-click to run!**

### First Run Security

On first run, macOS may block the app:
1. Go to System Preferences → Security & Privacy → General
2. Click "Open Anyway" next to the Quiz Scanner message
3. Or: Right-click the app → Open → Open (bypasses Gatekeeper once)

---

## Option 3: Automator Application

### Setup

1. **Open Automator:**
   - Press `Cmd + Space`, type "Automator", press Enter

2. **Create new Application:**
   - Choose "Application" as document type

3. **Add "Run Shell Script" action:**
   - Search for "Run Shell Script" in the left panel
   - Drag it to the workflow area

4. **Configure the script:**
   - Set "Shell" to `/bin/bash`
   - Set "Pass input" to "as arguments"
   - Paste this script:

   ```bash
   #!/bin/bash
   
   SCANNER_DIR="$HOME/quiz-scanner"
   
   # Start Chrome if not running
   if ! lsof -i :9222 > /dev/null 2>&1; then
       mkdir -p /tmp/chrome-debug-profile
       "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
           --remote-debugging-port=9222 \
           --user-data-dir=/tmp/chrome-debug-profile \
           --no-first-run &
       sleep 3
   fi
   
   # Start Streamlit if not running
   if ! lsof -i :8501 > /dev/null 2>&1; then
       cd "$SCANNER_DIR"
       python3 -m streamlit run App.py --server.port=8501 --server.headless=true &
       sleep 5
   fi
   
   # Open browser
   open "http://localhost:8501"
   ```

5. **Save as Application:**
   - File → Save
   - Save as "Quiz Scanner" to Desktop
   - Format: Application

---

## Stopping the Scanner

To fully stop the scanner:

1. **Close the Terminal window** running Streamlit

2. **Or use Terminal:**
   ```bash
   # Stop Streamlit
   pkill -f streamlit
   
   # Stop Chrome debug session (optional - closes Chrome)
   pkill -f "remote-debugging-port=9222"
   ```

---

## Troubleshooting

### "App is damaged and can't be opened"

```bash
xattr -cr ~/Desktop/Quiz\ Scanner.app
```

### Chrome won't start

Make sure Chrome is installed at `/Applications/Google Chrome.app`

### Streamlit won't start

Check that Python and Streamlit are installed:
```bash
python3 --version
python3 -m streamlit --version
```

If not installed:
```bash
pip3 install streamlit
```

### Port already in use

```bash
# Find what's using port 8501
lsof -i :8501

# Kill it
kill -9 <PID>
```

---

## Adding to Dock

1. Drag "Quiz Scanner.app" to the Dock
2. It will stay there for quick access

---

## File Structure

Your scanner directory should contain:

```
~/quiz-scanner/
├── App.py                 # Main Streamlit app
├── multi_sample_llm.py    # Multi-sample testing module
├── quiz_browser_enhanced.py
├── reform_agent.py
├── analysis_agent.py
├── evolution_agent.py
├── launch_scanner.sh      # Shell launcher (Option 1)
└── chroma_db/             # RAG database (if configured)
```
