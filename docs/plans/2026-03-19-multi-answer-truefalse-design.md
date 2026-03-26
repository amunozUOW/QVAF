# Plan: Add Multi-Answer MCQ and True/False Question Types

## Context

The quiz-vulnerability framework currently only supports single-answer multiple choice questions (radio buttons). Moodle's `multichoice` CSS class is shared by both single-answer (radios) and multi-answer (checkboxes) questions, meaning multi-answer questions are currently either silently broken or mishandled. True/False questions (`truefalse` CSS class) are not detected at all during browser scraping, though the Test Question tab has partial T/F UI support.

**Goal:** Add full pipeline support for multi-answer MCQ (checkboxes) and True/False questions — from browser scraping through LLM answering, result merging, and the Test Question UI — with robust validation, format compliance checks, and graceful handling of malformed questions.

## Internal Type Taxonomy

Introduce three explicit format types (replacing the current single `'multichoice'`):
- `multichoice_single` — single-answer MCQ (radio buttons) — existing behavior
- `multichoice_multi` — multi-answer MCQ (checkboxes) — new
- `truefalse` — True/False (2 radio buttons) — new

**Fallback:** Unrecognized question types with radio buttons fall back to `multichoice_single` (best-guess). Unrecognized types without any inputs are skipped with a log warning.

## Implementation Steps

### Step 1: Multi-answer parsing (`parsing_utils.py`)

Add `extract_answers_multi(text)` alongside existing `extract_answer()`:
- Pattern: `ANSWER:\s*([A-H](?:\s*,\s*[A-H])*)` for "ANSWER: A, C, D"
- Fallback patterns for "A and C" and "ACD" (no separators)
- Returns sorted list of unique uppercase letters, e.g. `["A", "C", "D"]`
- Falls back to `extract_answer()` wrapped in a list

Add `parse_llm_response_multi(text)` that returns answer as comma-separated string `"A, C, D"`.

**Existing functions unchanged** — zero risk to current single-answer path.

### Step 2: Parsing tests (`tests/test_response_parsing.py`)

Add `TestExtractAnswersMulti` class:
- `"ANSWER: A, C, D"` → `["A", "C", "D"]`
- `"ANSWER: B"` → `["B"]`
- `"ANSWER: A and C"` → `["A", "C"]`
- `"ANSWER: D, A, C"` → `["A", "C", "D"]` (sorted)
- `"ANSWER: A, A, C"` → `["A", "C"]` (deduped)
- `"no answer"` → `["?"]`

Run full test suite to confirm no regressions.

### Step 3: LLM format compliance validation (`parsing_utils.py`)

Add `validate_multi_answer(answers, valid_options)` function:
- Checks that every letter in the parsed answer list is a valid option key
- Filters out invalid letters (e.g. "F" when only A-D exist)
- Logs warning if any letters were filtered
- Returns `["?"]` if all letters were invalid
- Called after `extract_answers_multi()` in the answering pipeline

**Tests for format compliance** (`tests/test_response_parsing.py`):
- `validate_multi_answer(["A", "C", "F"], {"A","B","C","D"})` → `["A", "C"]` + warning
- `validate_multi_answer(["X", "Z"], {"A","B","C","D"})` → `["?"]`
- `validate_multi_answer(["A", "C"], {"A","B","C","D"})` → `["A", "C"]` (no change)
- `validate_multi_answer([], {"A","B"})` → `["?"]`

### Step 4: Type detection with ambiguity handling (`quiz_browser_enhanced.py:565-566`)

Replace the single-line type detection with robust three-way logic:

1. If CSS classes contain `truefalse` → `q_type = 'truefalse'`
2. If CSS classes contain `multichoice`:
   - Check `answer_container` for `input[type="checkbox"]`
   - If checkboxes found → `q_type = 'multichoice_multi'`
   - Else → `q_type = 'multichoice_single'`
3. **Fallback (best-guess):** If CSS class is unrecognized but question has radio buttons → treat as `multichoice_single`. If no inputs at all → skip with log warning.

**Secondary signal:** Also check for "Select one or more:" prompt text (`.prompt` element inside `.ablock`) as confirmation for `multichoice_multi`. If checkbox detection and prompt text disagree, log a warning and trust the input type.

**Detection ambiguity tests** (`tests/test_type_detection.py` — new file):
- Test `multichoice` class + radio inputs → `multichoice_single`
- Test `multichoice` class + checkbox inputs → `multichoice_multi`
- Test `truefalse` class → `truefalse`
- Test unknown class + radio inputs → `multichoice_single` (best-guess fallback)
- Test unknown class + no inputs → `unknown` (skip)
- Test `multichoice` class + "Select one or more:" prompt → `multichoice_multi`
- Test `multichoice` class + checkboxes but no prompt text → still `multichoice_multi` (trust input type)

These tests use mock HTML snippets (not live Moodle), testing the detection function in isolation.

### Step 5: Malformed question validation (`quiz_browser_enhanced.py`)

Add `validate_question(q_type, options, checkbox_buttons, radio_buttons)` function called after option extraction, before prompting the LLM:

**Checks:**
1. **Missing/empty options:** If any option text is empty or whitespace-only, log warning and skip that option (don't include in prompt). If all options empty, skip the question entirely.
2. **Only one checkbox:** If `multichoice_multi` but only 1 checkbox found, log warning "Multi-answer question has only 1 option — treating as single-answer" and downgrade to `multichoice_single`.
3. **Duplicate option text:** If two or more options have identical text after stripping, log warning "Duplicate options detected" and keep only the first occurrence (remove duplicates from `options` dict and corresponding button dicts).

**Tests for malformed questions** (`tests/test_malformed_questions.py` — new file):
- Empty option text → filtered out, warning logged
- All options empty → question skipped
- Single checkbox → downgraded to `multichoice_single`
- Duplicate options `{"A": "True", "B": "True", "C": "False"}` → deduped to `{"A": "True", "C": "False"}`
- Normal well-formed question → no changes, no warnings

### Step 6: Checkbox extraction (`quiz_browser_enhanced.py:571-601`)

In the answer extraction loop, handle both input types:
- Query `input[type="checkbox"]` in addition to `input[type="radio"]`
- Store checkbox elements in `checkbox_buttons` dict (parallel to `radio_buttons`)
- Label-stripping regex applies identically to all types
- For `truefalse`: existing radio extraction works as-is (always 2 options)

### Step 7: Type-aware prompts (`quiz_browser_enhanced.py:build_prompt()` and `answer_question()`)

Add `q_type` parameter to `build_prompt()` (default `'multichoice_single'`):
- **`multichoice_single`** (default): No change to existing prompt
- **`multichoice_multi`**: Change instruction to "Select ALL correct options." Format: `ANSWER: X, Y, Z` (comma-separated letters). Include explicit example in prompt: "Example: ANSWER: A, C, D"
- **`truefalse`**: Simplified prompt — "Answer this true/false question." Format: `ANSWER: A` (True) or `ANSWER: B` (False)

Pass `q_type` through `answer_question()` → `build_prompt()`.

For multi-answer + multi-sample: use `extract_answers_multi()` per sample. Consistency = how often the exact same *set* of answers appears (compare sorted sets, not strings).

### Step 8: Multi-checkbox clicking (`quiz_browser_enhanced.py:649-657`)

Branch on type when clicking answers:
- `multichoice_single` / `truefalse`: existing single `radio_buttons[answer].click()`
- `multichoice_multi`: parse answer string `"A, C, D"` into letters, validate each against `checkbox_buttons` keys, iterate and click each valid checkbox. Log warning for any letter not found in buttons.

### Step 9: Result data format (`quiz_browser_enhanced.py:660-674`)

No structural changes to the result dict. The `type` field gets the new values and `llm_answer` contains either a single letter or comma-separated letters.

### Step 10: Merge pipeline (`merge_attempts.py`)

- Add `format_type` field to merged question dict (line 174), sourced from raw question `type`
- Correctness: primary check uses Moodle's CSS `correct`/`incorrect` classes (works for all types, no change needed)
- `extract_correct_answer_letter()`: for multi-answer, Moodle's `.rightanswer` may list multiple answers. Add companion logic to handle this case.

### Step 11: Test Question tab UI (`app_test_question.py`)

Extend the question type radio (line 240) to three options:
- "Multiple Choice (Single)" — existing behavior
- "Multiple Choice (Multi)" — new: use `st.multiselect` for correct answers instead of single selectbox
- "True/False" — existing, just wire up `q_type='truefalse'`

Update the prompt construction (lines 49-97) to use the shared type-aware `build_prompt()` from `quiz_browser_enhanced.py` (or replicate the type-aware logic).

### Step 12: Correctness tests (`tests/test_correctness.py`)

Add multi-answer correctness tests:
- `"A, C, D"` vs `"A, C, D"` → correct
- `"D, A, C"` vs `"A, C, D"` → correct (order-independent)
- `"A, C"` vs `"A, C, D"` → incorrect (missing selection)
- `"A, B, C, D"` vs `"A, C, D"` → incorrect (extra selection)

## Files Modified

| File | Changes |
|------|---------|
| `parsing_utils.py` | Add `extract_answers_multi()`, `parse_llm_response_multi()`, `validate_multi_answer()` |
| `quiz_browser_enhanced.py` | Type detection, checkbox extraction, `validate_question()`, type-aware prompts, multi-click |
| `app_test_question.py` | Multi-answer UI option, correct answer multiselect |
| `merge_attempts.py` | `format_type` field, multi-answer correct answer extraction |
| `tests/test_response_parsing.py` | Multi-answer parsing tests + format compliance validation tests |
| `tests/test_type_detection.py` | **New** — type detection tests with mock HTML snippets |
| `tests/test_malformed_questions.py` | **New** — malformed question validation tests |
| `tests/test_correctness.py` | Multi-answer correctness tests |

## Reuse

- **`strip_label()` in `merge_attempts.py:63`** — reuse for label stripping (already used in scraper)
- **`extract_answer()` in `parsing_utils.py:38`** — reuse as fallback within `extract_answers_multi()`
- **Moodle CSS correctness (`scrape_results()` line 705-712)** — authoritative check works for all types, no changes needed

## Verification

1. **Unit tests:** `python -m pytest tests/ -v` — all existing tests pass + new tests pass
2. **Test Question tab:** Launch `streamlit run App.py`, go to Test Question tab, test each type:
   - Single MCQ: existing behavior unchanged
   - Multi-answer MCQ: enter question with multiple correct answers, verify comma-separated response
   - True/False: enter T/F question, verify correct detection
3. **Browser scan (if Moodle available):** Run a scan against a quiz containing multi-answer and T/F questions, verify:
   - Correct type detection in console output
   - Checkboxes clicked for multi-answer
   - Results JSON contains correct `type` values
   - Merged report includes `format_type`
4. **Malformed question tests:** Verify graceful handling:
   - Enter question with duplicate options in Test Question tab
   - Enter question with blank options
   - Verify warnings appear in logs, no crashes

## Risks

- **LLM format compliance:** Small models may not reliably produce comma-separated answers. Mitigated by multiple fallback patterns in parser + `validate_multi_answer()` filtering invalid letters.
- **Detection ambiguity:** If Moodle renders multichoice without visible inputs (e.g., review mode), detection could fail. Mitigated by secondary "Select one or more:" prompt text check + best-guess fallback to `multichoice_single` for unknown types with radios.
- **Malformed questions:** Instructor errors (empty options, duplicates, single-checkbox multi) could cause unexpected behavior. Mitigated by `validate_question()` with logging and graceful degradation (downgrade, dedup, skip).
