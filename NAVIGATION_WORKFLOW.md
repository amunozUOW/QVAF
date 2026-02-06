# Quiz Vulnerability Scanner - Navigation Workflow

## First Page (System Check / Landing Page)

**What the user sees when they first open the app:**

1. **Title and Caption**
- "Quiz Vulnerability Assessment Framework"
    -Two clumn layout showing
    Left column System check panel
    Right column showing instructions
   

2. **System Check Panel Left Column**
   - Two-column layout showing:
        - AI Models status (Ready ✓ or Missing ✗)
        - If missing, shows: "Run in terminal: `ollama pull llama3:8b`"
        - **If AI Models are ready:**
        - Button: "Start" (primary, full width)
        - Pressing it sets `onboarding_complete = True` and shows the Instructions tab
        - **If AI Models are missing:**
        - Warning message: "Please install AI models before proceeding."
        - Button: "Check Again" (allows user to retry after installing)
3. **Right Column - Instructions and Workflow Starting Point**
    -"Instructions"

## Overview
The app now uses a dynamic tab system that shows only relevant tabs based on the user's chosen workflow.

## Workflow Paths

### Path 1: Test a Single Question
**Use case:** Test of a single question without connecting to Moodle

**Time:** 2-3 minutes

**Tabs Always Visible:**
- Instructions
- Test Question

**Steps:**
1. From Instructions, expand "Testing a single question, no setup needed (2-3 minutes)"
2. Click "Go to Test a Single Question"
3. Type or paste a quiz question with multiple choice options
4. Mark the correct answer
5. Click "Test Single Question" to run the AI against that question
6. Review the result and see if the AI answered it correctly and how confident it was

**Notes:**
- Individual questions are tested with your configured LLM
- No Moodle connection needed
- Takes 30 seconds to 2 minutes depending on model speed
- Basic analysis is provided

---

### Path 2: Baseline Scan (Quiz without Course Materials)
**Use case:** Test quiz vulnerability without additional course materials

**Time:** 5-15 minutes

**Tabs Shown:**
- Instructions
- Connect
- Scan
- Results
- Test a Single Question

**Steps:**
1. From Instructions, expand "Baseline Scan: Quiz without Course Materials (5-15 minutes)"
2. Click "Start Baseline Scan"
   - `use_rag_mode = False`
   - Tabs reorder: Connect tab appears
3. Open your Moodle quiz in Chrome and start an attempt (or start preview)
4. Make sure the first question is visible
5. Go to Connect tab and click "Connect to Browser"
6. System Status shows AI models and browser connection
7. Go to Scan tab and click "Start Scan"
8. The scanner will:
   - Read each question from your quiz
   - Ask the AI to answer each question using only general knowledge
   - Automatically submit answers to Moodle
9. When complete:
   - Submit the answers and navigate to the results page in Moodle
   - Click the "collect results" button
   - A basic dashboard should appear showing basic metrics
   - For a complete analysis, click on the "Generate Report" button

**What happens:**
- We test your quiz with AI using only general knowledge (no course materials)
- This gives you a baseline vulnerability measurement
- Shows which questions are easiest/hardest for AI to answer
- Takes 5-15 minutes depending on quiz length and model speed

---

### Path 3: Complete Assessment (With and Without Course Materials)
**Use case:** Comprehensive test showing how course materials affect AI performance

**Time:** 10-30 minutes

**Tabs Shown:**
- Instructions
- Connect
- First Scan
- Second Scan
- Results
- Test a Single Question

**Flexible Workflow - Choose Your Path:**

#### Option A: Upload Materials First (Recommended)
1. From Instructions, expand "Complete Assessment: With and Without Course Materials (10-30 minutes)"
2. Click "Start Complete Assessment"
   - `use_rag_mode = True`
   - Tabs reorder: Connect appears
3. Open your Moodle quiz in Chrome and start an attempt (or start preview)
4. Make sure the first question is visible
5. Go to Connect tab and click "Connect to Browser"
6. **UPLOAD COURSE MATERIALS (Before First Scan):**
   - Click on the "Upload Course Materials" link in the First Scan tab
   - Upload your course materials (lecture slides, notes, readings, etc.)
7. **FIRST SCAN (Baseline - without course materials):**
   - Go to First Scan tab and click "Start First Scan"
   - The scanner will:
     - Read each question from your quiz
     - Ask the AI to answer each question using only general knowledge
     - Automatically submit answers to Moodle
   - When complete:
     - Submit the answers and navigate to the results page in Moodle
     - Click the "collect results" button
8. **SECOND SCAN (With course materials):**
   - Start a new quiz attempt/preview in Moodle
   - Go to Second Scan tab and click "Start Second Scan"
   - Tests your quiz with AI having access to your uploaded course materials
   - When complete:
     - Submit the answers and navigate to the results page in Moodle
     - Click the "collect results" button
9. **VIEW RESULTS:**
   - When both scans are complete, go to Results tab
   - Click "Generate Report" for detailed analysis

#### Option B: Do First Scan First, Then Upload Materials
1. From Instructions, expand "Complete Assessment: With and Without Course Materials (10-30 minutes)"
2. Click "Start Complete Assessment"
   - `use_rag_mode = True`
   - Tabs reorder: Connect appears
3. Open your Moodle quiz in Chrome and start an attempt (or start preview)
4. Make sure the first question is visible
5. Go to Connect tab and click "Connect to Browser"
6. **FIRST SCAN (Baseline - without course materials):**
   - Go to First Scan tab and click "Start First Scan"
   - The scanner will:
     - Read each question from your quiz
     - Ask the AI to answer each question using only general knowledge
     - Automatically submit answers to Moodle
   - When complete:
     - Submit the answers and navigate to the results page in Moodle
     - Click the "collect results" button
7. **UPLOAD COURSE MATERIALS (After First Scan):**
   - Click on the "Upload Course Materials" link in the Second Scan tab
   - Upload your course materials (lecture slides, notes, readings, etc.)
8. **SECOND SCAN (With course materials):**
   - Start a new quiz attempt/preview in Moodle
   - Go to Second Scan tab and click "Start Second Scan"
   - Tests your quiz with AI having access to your uploaded course materials
   - When complete:
     - Submit the answers and navigate to the results page in Moodle
     - Click the "collect results" button
9. **VIEW RESULTS:**
   - When both scans are complete, go to Results tab
   - Click "Generate Report" for detailed analysis

**What happens:**
- First scan: AI answers without any course materials (baseline vulnerability)
- Second scan: AI answers with full access to your course materials
- Detailed comparison showing which questions become easier when AI has materials
- Identifies material-specific vulnerabilities vs general knowledge vulnerabilities
- Takes 10-30 minutes depending on quiz length and course materials

---

## Breadcrumb/Status Banner

Located at the top of the page, shows:
- **Left:** Current workflow name (e.g., "Baseline Scan (general knowledge only)")
- **Right:** Current status (e.g., "🔄 Scanning...", "✓ Scan complete", etc.)

Updates in real-time as user progresses through workflow.

---

## Tab Visibility Rules

### Always Visible
- **Instructions:** Choose workflow or get quick access to Test a Single Question
- **Test a Single Question:** Single question testing, no dependencies

### Conditionally Visible
- **Connect:** Appears for Baseline Scan or Complete Assessment (both require browser)
- **Scan / First Scan:** Appears for Baseline Scan or Complete Assessment
- **Second Scan:** Appears only for Complete Assessment
- **Results:** Appears for Baseline Scan or Complete Assessment

### Course Materials Upload
- **Before First Scan:** Available via "Upload Course Materials" link in the First Scan tab
- **Before Second Scan:** Available via "Upload Course Materials" link in the Second Scan tab
- No separate Settings tab needed — materials can be uploaded at two flexible points in the workflow

---

## Navigation Buttons

### Instruction Buttons (from Instructions Tab)
- "Go to Test Question tab" → Sets `navigate_to = 'test_question'`
- "Start Baseline Scan" → Sets `navigate_to = 'first_scan'`, `use_rag_mode = False`
- "Start Complete Assessment" → Sets `navigate_to = 'first_scan'`, `use_rag_mode = True`

### Contextual Next-Step Buttons (Inline)
After a scan completes, inline buttons guide to the next step:
- Basic Scan after Scan complete → "Generate Report"
- Full Assessment after First Scan → "Continue to Second Scan"
- Full Assessment after Second Scan → "Generate Report"

When clicked, these buttons auto-navigate by setting the `navigate_to` flag and calling `st.rerun()`.

---

## System Status Panel (Connect Tab)

Shows in the left column under troubleshooting:
- AI models ready/missing
- Browser detected/not connected
- "Refresh System Status" button

Automatically updated on Connect tab load and when user clicks refresh.

---

## Key Implementation Details

1. **Dynamic Labels:** Tab labels are computed based on `use_rag_mode`:
   - `None` → Only Instructions + Test a Single Question
   - `False` → Instructions + Connect + Scan + Results +Test a Single Question
   - `True` → Instructions + Connect + First Scan + Second Scan + Results + Test a Single Question

2. **Tab Reordering:** When `navigate_to` is set, the target tab label is moved to the front of the list so Streamlit selects it by default.

3. **Auto-Navigation:** After scan completes, buttons appear with clear next-step labels that trigger `navigate_to` and `st.rerun()`.

4. **Breadcrumb Logic:** Status text updates based on `is_scanning`, `is_testing`, and completion flags (`no_rag_score`, `with_rag_score`).

---

## Future Improvements (Optional)

- Add progress indicator (e.g., "Step 2 of 4")
- Show summary of results collected so far (e.g., "Baseline: 72%, need Second Scan for comparison")
