# QVAF Plans and To Dos

This document captures implementation plans and priorities for the Quiz Vulnerability Assessment Framework. Each plan is written so it can be handed directly to a coding agent for implementation.

---

## Implementation Sequence

| Order | What | Effort | Payoff | Status |
|-------|------|--------|--------|--------|
| 1 | Split App.py into modules | 1 session | Makes everything else easier + better vibe coding | NOT STARTED |
| 2 | Extract parsing into shared utils | 1 session | Required for both testing and standalone mode | DONE |
| 3 | Automated tests (Priority 1-2) | 1 session | Protects existing code before we add features | DONE (127 tests) |
| 4 | Standalone manual entry mode | 2-3 sessions | Biggest adoption unlock | NOT STARTED |
| 5 | Text paste parser | 1 session | Speed upgrade for manual mode | NOT STARTED |
| 6 | CSV/spreadsheet upload | 1 session | Familiar format for educators | NOT STARTED |

---

## Plan 1: Standalone Question Input Mode

### Problem

The only way to get a quiz into QVAF is through live browser automation with a Moodle page open. Educators need Chrome in debug mode, a Moodle instance, and the quiz actively loaded. Most educators who would benefit from this tool will not get past that setup.

### Vision

A "Paste & Scan" mode where educators can input questions without needing a browser connection at all. The tool analyses each question the same way it does during a live scan, but the input comes from the educator directly.

### Input Methods (ranked by ease-of-use for non-technical educators)

#### Method 1: Manual entry form (simplest, build first)

- Expand the existing "Test Question" tab into a multi-question workflow
- Educator types/pastes questions one at a time, entering the question text, options (A-E), and correct answer via form fields
- After entering all questions, click "Analyse Quiz" to run the full pipeline
- This already half-exists in the Test Question tab which handles single questions. The gaps are:
  - It does not accumulate questions into a quiz
  - It does not run the reform agent analysis
  - It does not generate a dashboard
  - RAG collection is hardcoded to "unit_materials" instead of using the selected collection

#### Method 2: Simple text paste (medium effort, high value)

- A large text box where educators paste their quiz in a natural format:

```
1. What is the capital of France?
A. London
B. Paris *
C. Berlin
D. Madrid

2. Which of these is a mammal?
A. Shark
B. Eagle
C. Dolphin *
D. Lizard
```

- The `*` marks the correct answer (or they could use bold, or a separate "correct answers" field like `1B, 2C, 3A`)
- A parser extracts questions, options, and correct answers from this free-form text
- Much faster than one-at-a-time entry for a 20-question quiz

#### Method 3: Spreadsheet upload (low effort, familiar format)

- Accept CSV or Excel upload with columns: `Question | Option A | Option B | Option C | Option D | Correct Answer`
- Most educators are comfortable with spreadsheets
- Easy to export from existing quiz banks
- Pandas already in the dependency list

#### Method 4: Copy-paste from Word/Google Docs (stretch goal)

- Many educators write quizzes in Word. A smart parser that handles numbered lists with lettered sub-items
- Harder due to formatting variability but could reuse the text paste parser with fuzzy matching

#### What NOT to support (for now)

- Moodle XML / QTI exports: complex XML schemas that non-technical users will not know how to generate, and parsing them correctly is large effort for edge cases
- API-based imports from LMS platforms: requires institutional access and OAuth setup

### Architecture

The existing pipeline already separates "getting the data" from "analysing the data." The browser automation (`quiz_browser_enhanced.py`) produces a JSON file. The reform agent (`reform_agent.py`) consumes that JSON. The standalone mode just needs to produce the same JSON structure from manual input, and the rest of the pipeline works unchanged.

```
Manual Input (new)      --> Same JSON format --> reform_agent.py --> analysis_agent.py
Browser Scan (existing) --> Same JSON format --> reform_agent.py --> analysis_agent.py
```

### Implementation Steps

#### Step 1: Create quiz_input.py (new module)

Create a new input parser module with functions:

- `parse_text_quiz(text: str) -> list[dict]`: parses the free-form text format (Method 2)
- `parse_csv_quiz(filepath: str) -> list[dict]`: parses CSV/Excel (Method 3)
- `build_manual_quiz(questions: list[dict]) -> dict`: assembles the standard JSON structure from form inputs (Method 1)
- All return the same format that `quiz_browser_enhanced.py` produces

#### Step 2: Expand the Test Question tab

- Rename tab from "Test Question" to "Test Questions" (plural)
- Add three sub-modes: "Enter one-by-one", "Paste quiz text", "Upload spreadsheet"
- Accumulated questions shown in an editable table with add/edit/remove controls
- "Analyse Quiz" button triggers the scan (calls Ollama for each question), then reform agent, then dashboard
- Fix the hardcoded RAG collection name ("unit_materials") to use the selected collection from Settings

#### Step 3: Run LLM scan without browser automation

- Reuse the prompt-building logic from `quiz_browser_enhanced.py` but without Playwright
- Extract `build_prompt()` into a shared utility so both browser and manual modes use the same prompt
- Support multi-sample mode and RAG (if a collection is selected)

#### Step 4: Feed results into existing pipeline

- Save results as the same JSON format to `output/raw_attempts/`
- Run `reform_agent.py` analysis
- Generate dashboard via `analysis_agent.py`
- Display in Results tab identically to browser-scanned quizzes

### What This Unlocks

- Educators can try QVAF with zero setup beyond Ollama
- Workshop/demo friendly: paste a quiz and get results in minutes
- Works for any LMS (or no LMS) since input is platform-agnostic
- The correct answer is provided by the educator, so no Moodle scraping needed

---

## Plan 2: Automated Testing

### What This Is (in plain terms)

Automated tests are small programs that check your code still works correctly. You write them once, then run them any time you make a change. If something breaks, the test fails immediately and tells you exactly what went wrong, instead of discovering it later when an educator runs a scan and gets garbage results.

Every time we edit the code that reads AI responses, we are trusting it still works. A test verifies it works, every time, in seconds.

### What We Would Test (and Why)

#### Priority 1: Response Parsing (most fragile part of the codebase)

The LLM returns free-form text, and regex patterns extract the answer, confidence, and reasoning. Small model variations can break this.

Tests would cover:
- Standard format: `ANSWER: B\nCONFIDENCE: 85\nREASONING: Because...`
- No space: `ANSWER:B`
- With period: `ANSWER: B.`
- Lowercase: `answer: b`
- Missing marker entirely (should trigger fallback)
- Confidence over 100 (should clamp to 100)
- Confidence missing (should default to 0)
- Multi-line reasoning (should capture all of it, not just first line)
- Garbage response (should return "?" not crash)

#### Priority 2: Correctness Checking

The logic that determines if the AI got the right answer.

Tests would cover:
- Letter matches: "A" vs "A" -> True
- Case insensitive: "a" vs "A" -> True
- With text: "A. Paris" vs "A" -> True
- Unknown correct answer -> None (not False)
- Empty answer -> None (not crash)
- Correctness pattern logic with None values (edge cases around what "unknown" means)

#### Priority 3: Question Matching (merge step)

The merge step that pairs baseline and RAG results.

Tests would cover:
- Exact text match
- Questions with minor whitespace differences
- Questions with identical first 50 characters but different endings
- Missing question in one set (should handle gracefully)
- Empty question text

#### Priority 4: Option Label Stripping

Parsing option labels like "A.", "1)", "iv."

Tests would cover:
- All formats: `A.`, `a)`, `1.`, `i.`, `IV.`
- Content that looks like a label: "IV therapy is..." (should not strip)
- No label (should return text unchanged)

### How It Works Technically

We use pytest (a standard Python testing tool). One command:

```bash
pytest
```

It either says "all 47 tests passed" (green) or tells you exactly which test failed and why. Takes about 2 seconds.

### Implementation Steps

#### Step 1: Install pytest

```bash
pip install pytest
```

#### Step 2: Extract parsing functions into shared module

Currently the response parsing regex is inline inside App.py (lines 2338-2344), which means you cannot test it without running the whole Streamlit app. Extract into `parsing_utils.py`:

- `extract_answer(text: str) -> str`: extracts answer letter from LLM response
- `extract_confidence(text: str) -> int`: extracts and clamps confidence (0-100)
- `extract_reasoning(text: str) -> str`: extracts reasoning text
- `parse_llm_response(text: str) -> dict`: combines all three

Both App.py and quiz_browser_enhanced.py would import from this module, eliminating the current duplication where the same regex exists in 3+ places with slight variations.

#### Step 3: Create test files

```
tests/
  __init__.py
  test_response_parsing.py    (~15 tests for answer/confidence/reasoning extraction)
  test_correctness.py         (~10 tests for check_correct and pattern logic)
  test_question_matching.py   (~8 tests for merge_attempts matching)
  test_option_parsing.py      (~8 tests for label stripping)
  test_quiz_input.py          (~6 tests for standalone input parsers, once built)
```

#### Step 4: Add pytest config

Add a `pytest.ini` or `[tool.pytest.ini_options]` section in a `pyproject.toml` so tests run with a single command from the project root.

### What You Do Day-to-Day

- Before pushing a change: run `pytest` (takes about 2 seconds)
- If it passes: confident the change did not break anything
- If it fails: it tells you exactly what broke and where
- When we add new features: write tests alongside the code

### What This Protects Against

- Editing a regex in the response parser and accidentally breaking confidence extraction
- Changing the merge logic and silently mismatching questions
- An LLM model update producing slightly different output format that breaks parsing
- Future contributors introducing bugs in code they do not fully understand

### Known Fragility Points to Fix During Test Implementation

These are bugs/risks discovered during code review that the tests would both document and fix:

| Issue | Location | Risk |
|-------|----------|------|
| Confidence not clamped to 0-100 | App.py:2343 | UI shows >100% progress bars |
| Multi-sample tie-breaking undefined | App.py:2418 | `max()` picks arbitrarily on ties |
| Question matching on 50-char prefix only | merge_attempts.py:57 | Different questions can falsely match |
| RAG silent failure with bare except | App.py:2283 | No logging when RAG fails |
| Correctness pattern treats None as False | reform_agent.py:171 | "Unknown" conflated with "incorrect" |
| Only 1 fallback regex but loop suggests multiple | App.py:2383 | Incomplete fallback coverage |

---

## Prerequisite: Split App.py into Modules

### Why

App.py is 2,930 lines. When making changes:
- The entire file must be read to understand context
- Edits to scanning logic risk colliding with unrelated UI code
- Multiple features are entangled in one file

Splitting it directly improves the vibe coding workflow by allowing targeted reads and edits.

### Proposed Split

| New File | Contents | Approximate Lines |
|----------|----------|--------------------|
| `app_ui.py` | Streamlit layout, tabs, styling, welcome/instructions | ~600 |
| `app_scanning.py` | Scan orchestration, progress tracking, Chrome connection | ~800 |
| `app_settings.py` | RAG management, configuration UI, collection CRUD | ~400 |
| `app_results.py` | Results display, dashboard embedding, file downloads | ~500 |
| `App.py` | Main entry point, imports, session state init, tab routing | ~300 |
| `parsing_utils.py` | Shared response parsing (extracted from App.py and quiz_browser_enhanced.py) | ~100 |

### Approach

- Extract functions into new modules with no logic changes
- App.py becomes a thin orchestrator that imports and calls module functions
- Session state management stays in App.py (single source of truth)
- Each module receives `st` (Streamlit) as needed via function parameters or direct import
- All existing functionality preserved; no user-facing changes

---

## Decisions Made (for reference)

These items were discussed and intentionally deprioritised or rejected:

| Item | Decision | Reason |
|------|----------|--------|
| Cloud-hosted LLM (OpenAI/Anthropic API) | Rejected | Cost is non-negotiable; tool must remain free |
| Evolution agent | Dormant | Too risky without human-in-loop; educator makes single targeted adjustments per semester instead |
| Multi-model comparative benchmarking | Rejected | "Tests the AI, not the test" -- misaligns with project purpose |
| Moodle XML / QTI file import | Rejected for now | Non-technical users cannot generate these; complex to parse |
| LMS API integrations | Deferred | Requires institutional access to develop against |
| Broader LMS support (Canvas, Blackboard, Google, MS Forms) | Future priority | Needs access to live instances; browser automation approach may generalise |
