# Forced Alternative Enumeration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add forced alternative enumeration (ALTERNATIVE + ALT_RATIONALE fields) and move PROBABILITY to last output line (reasoning-before-number ordering) in the quiz LLM prompt and parsing pipeline.

**Architecture:** Two new LLM response fields are woven into the existing prompt analysis steps and output format. New extraction functions in `parsing_utils.py` feed into new `_full()` parser variants that return dicts. Existing 3-tuple parsers stay unchanged for backward compatibility. `quiz_browser_enhanced.py` switches to `_full()` parsers and passes new fields through in the response dict.

**Tech Stack:** Python, regex parsing, pytest

---

### Task 1: Add `extract_alternative()` tests

**Files:**
- Modify: `tests/test_response_parsing.py`

**Step 1: Write the failing tests**

Add a new test class after the existing `TestExtractReasoning` class (after line 314 in `tests/test_response_parsing.py`). First, update the import on line 38-41 to include the new functions:

```python
from parsing_utils import (
    extract_answer, extract_confidence, extract_reasoning, parse_llm_response,
    extract_answers_multi, validate_multi_answer, parse_llm_response_multi,
    extract_alternative, extract_alt_rationale,
    parse_llm_response_full, parse_llm_response_multi_full,
)
```

Then add the test class:

```python
# ============================================
# ALTERNATIVE EXTRACTION TESTS
# ============================================
# These test extract_alternative() which pulls the runner-up answer
# from the ALTERNATIVE: field in the LLM response.

class TestExtractAlternative:
    """Tests for extracting the alternative answer from LLM responses."""

    def test_standard_letter(self):
        """Standard format: ALTERNATIVE: C"""
        assert extract_alternative("ALTERNATIVE: C") == "C"

    def test_lowercase_letter(self):
        """Lowercase letter is uppercased."""
        assert extract_alternative("ALTERNATIVE: c") == "C"

    def test_none_value(self):
        """Model reports no plausible alternative."""
        assert extract_alternative("ALTERNATIVE: none") == "none"

    def test_none_uppercase(self):
        """NONE in various cases."""
        assert extract_alternative("ALTERNATIVE: None") == "none"
        assert extract_alternative("ALTERNATIVE: NONE") == "none"

    def test_no_space_after_colon(self):
        """No space: ALTERNATIVE:C"""
        assert extract_alternative("ALTERNATIVE:C") == "C"

    def test_missing_field(self):
        """No ALTERNATIVE marker returns empty string."""
        assert extract_alternative("ANSWER: B\nPROBABILITY: 7") == ""

    def test_empty_string(self):
        """Empty input returns empty string."""
        assert extract_alternative("") == ""

    def test_none_input(self):
        """None input returns empty string."""
        assert extract_alternative(None) == ""

    def test_in_full_response(self):
        """Extracted correctly from a complete response."""
        text = "ANSWER: B\nALTERNATIVE: D\nALT_RATIONALE: D is close but...\nREASONING: B is correct.\nDOUBT: Could be D.\nPROBABILITY: 7"
        assert extract_alternative(text) == "D"

    def test_extended_options(self):
        """Supports letters up to H."""
        assert extract_alternative("ALTERNATIVE: F") == "F"
        assert extract_alternative("ALTERNATIVE: H") == "H"

    def test_with_trailing_text(self):
        """Letter with trailing text (model adds extra words)."""
        assert extract_alternative("ALTERNATIVE: C (close second)") == "C"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_response_parsing.py::TestExtractAlternative -v`
Expected: FAIL with `ImportError: cannot import name 'extract_alternative'`

**Step 3: Commit**

```bash
git add tests/test_response_parsing.py
git commit -m "test: add failing tests for extract_alternative()"
```

---

### Task 2: Add `extract_alt_rationale()` tests

**Files:**
- Modify: `tests/test_response_parsing.py`

**Step 1: Write the failing tests**

Add after the `TestExtractAlternative` class:

```python
# ============================================
# ALT_RATIONALE EXTRACTION TESTS
# ============================================

class TestExtractAltRationale:
    """Tests for extracting the alternative rationale from LLM responses."""

    def test_standard_format(self):
        """Standard single-line rationale."""
        text = "ALT_RATIONALE: D is close but focuses on the wrong aspect."
        assert extract_alt_rationale(text) == "D is close but focuses on the wrong aspect."

    def test_no_space_after_colon(self):
        """No space: ALT_RATIONALE:reason"""
        text = "ALT_RATIONALE:D could apply but is less specific."
        assert extract_alt_rationale(text) == "D could apply but is less specific."

    def test_missing_field(self):
        """No ALT_RATIONALE marker returns empty string."""
        assert extract_alt_rationale("ANSWER: B\nPROBABILITY: 7") == ""

    def test_empty_string(self):
        """Empty input returns empty string."""
        assert extract_alt_rationale("") == ""

    def test_none_input(self):
        """None input returns empty string."""
        assert extract_alt_rationale(None) == ""

    def test_stops_at_next_field(self):
        """Rationale stops at the next field marker (REASONING:)."""
        text = "ALT_RATIONALE: D is plausible but weaker.\nREASONING: B is the best fit."
        assert extract_alt_rationale(text) == "D is plausible but weaker."

    def test_in_full_response(self):
        """Extracted correctly from a complete response."""
        text = "ANSWER: B\nALTERNATIVE: D\nALT_RATIONALE: D relates to supply chain not operations.\nREASONING: B is correct because...\nDOUBT: Could be D.\nPROBABILITY: 7"
        assert extract_alt_rationale(text) == "D relates to supply chain not operations."
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_response_parsing.py::TestExtractAltRationale -v`
Expected: FAIL with `ImportError: cannot import name 'extract_alt_rationale'`

**Step 3: Commit**

```bash
git add tests/test_response_parsing.py
git commit -m "test: add failing tests for extract_alt_rationale()"
```

---

### Task 3: Implement `extract_alternative()` and `extract_alt_rationale()`

**Files:**
- Modify: `parsing_utils.py`

**Step 1: Add `extract_alternative()` after `extract_reasoning()` (after line 192)**

```python
def extract_alternative(text):
    """
    Extract the alternative answer from an LLM response.

    The ALTERNATIVE field names the next most plausible answer option,
    or "none" if no other option is plausible. This supports the forced
    alternative enumeration mechanism for confidence calibration.

    Parameters
    ----------
    text : str
        The raw text response from the LLM.

    Returns
    -------
    str
        An uppercase letter (A-H), "none", or "" if field not found.

    Examples
    --------
    >>> extract_alternative("ALTERNATIVE: C")
    'C'
    >>> extract_alternative("ALTERNATIVE: none")
    'none'
    >>> extract_alternative("No alternative here")
    ''
    """
    if not text:
        return ""

    match = re.search(r'ALTERNATIVE:\s*([A-Ha-h])\b', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r'ALTERNATIVE:\s*(none)\b', text, re.IGNORECASE)
    if match:
        return "none"

    return ""


def extract_alt_rationale(text):
    """
    Extract the alternative rationale from an LLM response.

    Captures everything after "ALT_RATIONALE:" up to the next known
    field marker (REASONING:, DOUBT:, PROBABILITY:, CONFIDENCE:) or
    end of text.

    Parameters
    ----------
    text : str
        The raw text response from the LLM.

    Returns
    -------
    str
        The rationale text, stripped of whitespace. Empty if not found.

    Examples
    --------
    >>> extract_alt_rationale("ALT_RATIONALE: D is close but wrong.")
    'D is close but wrong.'
    """
    if not text:
        return ""

    match = re.search(
        r'ALT_RATIONALE:\s*(.+?)(?=\n(?:REASONING|DOUBT|PROBABILITY|CONFIDENCE|ANSWER):|\n\n|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    return ""
```

**Step 2: Run the tests from Tasks 1-2 to verify they pass**

Run: `pytest tests/test_response_parsing.py::TestExtractAlternative tests/test_response_parsing.py::TestExtractAltRationale -v`
Expected: ALL PASS

**Step 3: Run the full test suite to check no regressions**

Run: `pytest tests/test_response_parsing.py -v`
Expected: ALL PASS (existing tests unaffected)

**Step 4: Commit**

```bash
git add parsing_utils.py
git commit -m "feat: add extract_alternative() and extract_alt_rationale() parsers"
```

---

### Task 4: Add `parse_llm_response_full()` tests

**Files:**
- Modify: `tests/test_response_parsing.py`

**Step 1: Write the failing tests**

Add after `TestExtractAltRationale`:

```python
# ============================================
# FULL RESPONSE PARSING TESTS (dict format)
# ============================================

class TestParseLlmResponseFull:
    """Tests for parse_llm_response_full() which returns all fields as a dict."""

    def test_complete_response(self):
        """All six fields parsed correctly."""
        text = "ANSWER: B\nALTERNATIVE: D\nALT_RATIONALE: D is about supply chain.\nREASONING: B matches operations definition.\nDOUBT: Could be D if scope is broader.\nPROBABILITY: 7"
        result = parse_llm_response_full(text)
        assert result['answer'] == "B"
        assert result['confidence'] == 78
        assert result['reasoning'] == "B matches operations definition."
        assert result['alternative'] == "D"
        assert result['alt_rationale'] == "D is about supply chain."

    def test_alternative_none(self):
        """Alternative is 'none' for unambiguous questions."""
        text = "ANSWER: B\nALTERNATIVE: none\nALT_RATIONALE: No other option fits.\nREASONING: Clearly B.\nDOUBT: None.\nPROBABILITY: 9"
        result = parse_llm_response_full(text)
        assert result['alternative'] == "none"
        assert result['alt_rationale'] == "No other option fits."

    def test_missing_alternative_fields(self):
        """Graceful fallback when alternative fields are missing (legacy response)."""
        text = "ANSWER: B\nPROBABILITY: 7\nREASONING: Because.\nDOUBT: Maybe not."
        result = parse_llm_response_full(text)
        assert result['answer'] == "B"
        assert result['confidence'] == 78
        assert result['alternative'] == ""
        assert result['alt_rationale'] == ""

    def test_garbage_input(self):
        """Unparseable response returns safe defaults."""
        result = parse_llm_response_full("asdfghjkl")
        assert result['answer'] == "?"
        assert result['confidence'] == 0
        assert result['reasoning'] == ""
        assert result['alternative'] == ""
        assert result['alt_rationale'] == ""

    def test_none_input(self):
        """None input returns safe defaults."""
        result = parse_llm_response_full(None)
        assert result['answer'] == "?"
        assert result['confidence'] == 0

    def test_backward_compat_tuple_still_works(self):
        """Existing parse_llm_response() 3-tuple is unaffected."""
        text = "ANSWER: B\nALTERNATIVE: D\nALT_RATIONALE: Close.\nREASONING: Correct.\nDOUBT: Maybe.\nPROBABILITY: 7"
        answer, confidence, reasoning = parse_llm_response(text)
        assert answer == "B"
        assert confidence == 78
        assert reasoning == "Correct."


class TestParseLlmResponseMultiFull:
    """Tests for parse_llm_response_multi_full() dict format."""

    def test_complete_multi_response(self):
        """Multi-answer response with all fields."""
        text = "ANSWER: A, C\nALTERNATIVE: D\nALT_RATIONALE: D almost included.\nREASONING: A and C are both correct.\nDOUBT: Maybe D too.\nPROBABILITY: 6"
        result = parse_llm_response_multi_full(text)
        assert result['answer'] == "A, C"
        assert result['confidence'] == 67
        assert result['alternative'] == "D"

    def test_missing_alternative_fields(self):
        """Graceful fallback for multi-answer legacy response."""
        text = "ANSWER: A, C\nPROBABILITY: 7\nREASONING: Both correct."
        result = parse_llm_response_multi_full(text)
        assert result['answer'] == "A, C"
        assert result['alternative'] == ""
        assert result['alt_rationale'] == ""
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_response_parsing.py::TestParseLlmResponseFull tests/test_response_parsing.py::TestParseLlmResponseMultiFull -v`
Expected: FAIL with `ImportError: cannot import name 'parse_llm_response_full'`

**Step 3: Commit**

```bash
git add tests/test_response_parsing.py
git commit -m "test: add failing tests for parse_llm_response_full() and multi variant"
```

---

### Task 5: Implement `parse_llm_response_full()` and `parse_llm_response_multi_full()`

**Files:**
- Modify: `parsing_utils.py`

**Step 1: Add `parse_llm_response_full()` after `parse_llm_response()` (after line 223)**

```python
def parse_llm_response_full(text):
    """
    Parse a complete LLM response into a dict with all fields.

    Returns all fields including the forced alternative enumeration
    fields (alternative, alt_rationale). Use this when you need
    the full response data. For backward-compatible 3-tuple output,
    use parse_llm_response() instead.

    Parameters
    ----------
    text : str
        The raw text response from the LLM.

    Returns
    -------
    dict
        Keys: answer, confidence, reasoning, alternative, alt_rationale
    """
    return {
        'answer': extract_answer(text),
        'confidence': extract_confidence(text),
        'reasoning': extract_reasoning(text),
        'alternative': extract_alternative(text),
        'alt_rationale': extract_alt_rationale(text),
    }
```

**Step 2: Add `parse_llm_response_multi_full()` after `parse_llm_response_multi()` (after line 341)**

```python
def parse_llm_response_multi_full(text):
    """
    Parse a multi-answer LLM response into a dict with all fields.

    Like parse_llm_response_multi() but returns a dict including
    alternative and alt_rationale fields.

    Returns
    -------
    dict
        Keys: answer, confidence, reasoning, alternative, alt_rationale
    """
    answers = extract_answers_multi(text)
    return {
        'answer': ", ".join(answers),
        'confidence': extract_confidence(text),
        'reasoning': extract_reasoning(text),
        'alternative': extract_alternative(text),
        'alt_rationale': extract_alt_rationale(text),
    }
```

**Step 3: Run the tests from Task 4 to verify they pass**

Run: `pytest tests/test_response_parsing.py::TestParseLlmResponseFull tests/test_response_parsing.py::TestParseLlmResponseMultiFull -v`
Expected: ALL PASS

**Step 4: Run the full test suite**

Run: `pytest tests/test_response_parsing.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add parsing_utils.py
git commit -m "feat: add parse_llm_response_full() and parse_llm_response_multi_full()"
```

---

### Task 6: Update `build_prompt()` in `quiz_browser_enhanced.py`

**Files:**
- Modify: `quiz_browser_enhanced.py:418-447`

**Step 1: Replace the prompt template**

Replace lines 418-447 (the `return f"""...` block in `build_prompt()`) with:

```python
    # Type-specific alternative guidance
    if q_type == 'multichoice_multi':
        alt_guidance = "After selecting your answers, identify which other option you were least sure about including or excluding. State it as ALTERNATIVE: [letter] or ALTERNATIVE: none if your selections are clearly correct. Briefly explain why you included or excluded it."
    else:
        alt_guidance = "After selecting your answer, identify which other option was the strongest runner-up. State it as ALTERNATIVE: [letter] or ALTERNATIVE: none if no other option is plausible. Briefly explain why you rejected it."

    return f"""{instruction}

QUESTION: {question}

OPTIONS:
{options_text}

{context_block}

First, evaluate each option briefly. {selection_guidance}

{alt_guidance}

Then, considering the strength of that alternative, rate the probability your primary answer is correct.

Format your response EXACTLY like this:
{answer_format}
ALTERNATIVE: Y
ALT_RATIONALE: Why Y was considered and rejected
REASONING: Your explanation here
DOUBT: What could make your answer wrong
PROBABILITY: N

{answer_description}, Y is the next most plausible option letter (or "none"), and N is a single digit 0-9:
  0 = guessing randomly
  1 = very unlikely correct
  2 = unlikely, major problems with reasoning
  3 = somewhat unlikely, notable gaps
  4 = slightly below even odds
  5 = about even odds, coin flip
  6 = slightly more likely correct than not
  7 = probably correct, but alternatives are plausible
  8 = likely correct, only minor reservations
  9 = near certain, would be very surprised if wrong

Your response:"""
```

**Step 2: Run existing tests to check nothing broke**

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add quiz_browser_enhanced.py
git commit -m "feat: add ALTERNATIVE/ALT_RATIONALE fields and reasoning-before-probability to build_prompt()"
```

---

### Task 7: Update `optimized_prompt_v4()` in `optimized_prompt.py`

**Files:**
- Modify: `optimized_prompt.py:134-182`

**Step 1: Replace the prompt template**

Replace the return string in `optimized_prompt_v4()` (lines 134-182) with:

```python
    return f"""TASK: Answer this multiple choice question correctly.

QUESTION: {question}

OPTIONS:
{options_text}
{context}
ANALYSIS STEPS:

1. QUESTION TYPE: Is this testing recall, conceptual understanding, calculation, or application?

2. KEY INSIGHT: What core concept or distinction is being tested here?

3. EVALUATE OPTIONS:
   A: [KEEP/ELIMINATE] - why?
   B: [KEEP/ELIMINATE] - why?
   C: [KEEP/ELIMINATE] - why?
   D: [KEEP/ELIMINATE] - why?
   E: [KEEP/ELIMINATE] - why?

4. CALCULATION (if needed): Show your working.

5. FINAL SELECTION: From options marked KEEP, select the single best answer.

6. ALTERNATIVE: Which other option was the strongest runner-up? Name it (e.g. "C"), or say "none" if no other option is remotely plausible. Explain briefly why you rejected it.

7. DOUBT CHECK: Considering the alternative and any other uncertainties, what is the strongest argument AGAINST your chosen answer?

=== REQUIRED OUTPUT FORMAT ===
After your analysis, you MUST write these six lines:

ANSWER: [write ONE letter: A, B, C, D, or E]
ALTERNATIVE: [write the runner-up letter, or "none"]
ALT_RATIONALE: [write one sentence: why you rejected the alternative]
REASONING: [write one sentence explaining why your answer is correct]
DOUBT: [write one sentence about what could make your answer wrong]
PROBABILITY: [write a single digit 0-9 using the scale below]

PROBABILITY SCALE — rate the probability your answer is correct:
  0 = I am guessing randomly, I have no basis for this answer
  1 = Very unlikely correct, almost certainly wrong
  2 = Unlikely correct, I see major problems with my reasoning
  3 = Somewhat unlikely, notable gaps in my reasoning
  4 = Slightly below even odds, could easily be wrong
  5 = About even odds, roughly a coin flip between options
  6 = Slightly more likely correct than not
  7 = Probably correct, but alternative answers are plausible
  8 = Likely correct with only minor reservations
  9 = Near certain, I would be very surprised if wrong

Do not write anything after the PROBABILITY line.

Begin your analysis:"""
```

**Step 2: Commit**

```bash
git add optimized_prompt.py
git commit -m "feat: add ALTERNATIVE/ALT_RATIONALE to optimized_prompt_v4 and move PROBABILITY last"
```

---

### Task 8: Wire `_full()` parsers into `answer_question()`

**Files:**
- Modify: `quiz_browser_enhanced.py:28` (import) and `quiz_browser_enhanced.py:525-591` (`answer_question()`)

**Step 1: Update the import on line 28**

Change:
```python
from parsing_utils import parse_llm_response, parse_llm_response_multi, extract_answers_multi, validate_multi_answer
```
To:
```python
from parsing_utils import (
    parse_llm_response, parse_llm_response_multi,
    parse_llm_response_full, parse_llm_response_multi_full,
    extract_answers_multi, validate_multi_answer,
)
```

**Step 2: Update `answer_question()` single-sample path (lines 535-553)**

Replace the single-sample block:

```python
    if num_samples == 1:
        text = call_llm_single(prompt, model)
        if is_multi:
            answer, confidence, reasoning = parse_llm_response_multi(text)
            # Validate against actual options
            answer_letters = validate_multi_answer(
                [l.strip() for l in answer.split(',')], options
            )
            answer = ", ".join(answer_letters)
        else:
            answer, confidence, reasoning = parse_llm_response(text)
        return {
            'answer': answer,
            'confidence': confidence,
            'reasoning': reasoning,
            'consistency': '1/1',
            'consistency_pct': 100,
            'raw_response': text,
        }
```

With:

```python
    if num_samples == 1:
        text = call_llm_single(prompt, model)
        if is_multi:
            parsed = parse_llm_response_multi_full(text)
            # Validate against actual options
            answer_letters = validate_multi_answer(
                [l.strip() for l in parsed['answer'].split(',')], options
            )
            parsed['answer'] = ", ".join(answer_letters)
        else:
            parsed = parse_llm_response_full(text)
        return {
            'answer': parsed['answer'],
            'confidence': parsed['confidence'],
            'reasoning': parsed['reasoning'],
            'alternative': parsed['alternative'],
            'alt_rationale': parsed['alt_rationale'],
            'consistency': '1/1',
            'consistency_pct': 100,
            'raw_response': text,
        }
```

Note: The multi-sample paths (`call_llm_multi_sample` and the multi-answer multi-sample loop) still use the 3-tuple parsers. This is intentional — in multi-sample mode, alternative/rationale from individual runs are less meaningful since we're taking a plurality vote. The alternative fields will only be populated for single-sample runs initially.

**Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add quiz_browser_enhanced.py
git commit -m "feat: wire parse_llm_response_full() into answer_question() for alternative fields"
```

---

### Task 9: Update `LLMInstructions.md`

**Files:**
- Modify: `LLMInstructions.md:27-39` (section 1, LLM response contract)
- Modify: `LLMInstructions.md:103-110` (section 4, LLM response format)
- Modify: `LLMInstructions.md:129-139` (section 5, prompt and parsing contract)

**Step 1: Update the LLM response contract in section 1 (lines 27-39)**

Replace:

```
- LLM response contract (must be parseable by `parse_llm_response()` or `parse_llm_response_multi()`):

  Single-answer (multichoice_single, truefalse):
  ANSWER: X
  PROBABILITY: N
  REASONING: ...

  Multi-answer (multichoice_multi):
  ANSWER: X, Y, Z
  PROBABILITY: N
  REASONING: ...

  Where X is letter A–H (single) or comma-separated letters (multi), and N is a digit 0–9 (converted to 0–100%). Keep this exact header format or update both the prompt template and parsing functions. The parser `parse_llm_response()` handles single answers; `parse_llm_response_multi()` handles comma-separated multi-answers. Both live in `parsing_utils.py`.
```

With:

```
- LLM response contract (must be parseable by `parse_llm_response_full()` or `parse_llm_response_multi_full()`):

  Single-answer (multichoice_single, truefalse):
  ANSWER: X
  ALTERNATIVE: Y
  ALT_RATIONALE: ...
  REASONING: ...
  DOUBT: ...
  PROBABILITY: N

  Multi-answer (multichoice_multi):
  ANSWER: X, Y, Z
  ALTERNATIVE: Y
  ALT_RATIONALE: ...
  REASONING: ...
  DOUBT: ...
  PROBABILITY: N

  Where X is letter A–H (single) or comma-separated letters (multi), Y is the next most plausible option letter or "none", and N is a digit 0–9 (converted to 0–100%). PROBABILITY is intentionally last to enforce reasoning-before-number ordering. The `_full()` parsers return dicts; the legacy 3-tuple parsers (`parse_llm_response()`, `parse_llm_response_multi()`) remain for backward compatibility. All live in `parsing_utils.py`.
```

**Step 2: Update the LLM response format in section 4 (lines 103-110)**

Replace:

```
- LLM response format: prompts are constructed in `quiz_browser_enhanced.build_prompt(q_type=...)` and the code expects answers in type-specific formats:
  - **Single-answer** (`multichoice_single`, `truefalse`): `ANSWER: X` (one letter) → parsed by `parse_llm_response()`
  - **Multi-answer** (`multichoice_multi`): `ANSWER: X, Y, Z` (comma-separated) → parsed by `parse_llm_response_multi()`
  - Both formats include `PROBABILITY: N` (0–9 scale) and `REASONING: ...`
  - Multi-answer responses are validated by `validate_multi_answer()` which filters invalid letters against the actual option keys
  - `build_prompt()` accepts `q_type` parameter and adapts instructions accordingly

  Keep these formats stable or update prompts, parsers (`parsing_utils.py`), and UI simultaneously.
```

With:

```
- LLM response format: prompts are constructed in `quiz_browser_enhanced.build_prompt(q_type=...)` and the code expects answers in type-specific formats:
  - **Single-answer** (`multichoice_single`, `truefalse`): `ANSWER: X` (one letter) → parsed by `parse_llm_response_full()`
  - **Multi-answer** (`multichoice_multi`): `ANSWER: X, Y, Z` (comma-separated) → parsed by `parse_llm_response_multi_full()`
  - Both formats include `ALTERNATIVE: Y` (runner-up letter or "none"), `ALT_RATIONALE: ...`, `REASONING: ...`, `DOUBT: ...`, and `PROBABILITY: N` (0–9 scale, intentionally last)
  - Multi-answer responses are validated by `validate_multi_answer()` which filters invalid letters against the actual option keys
  - `build_prompt()` accepts `q_type` parameter and adapts instructions accordingly
  - Legacy 3-tuple parsers (`parse_llm_response()`, `parse_llm_response_multi()`) still work for callers that only need answer/confidence/reasoning

  Keep these formats stable or update prompts, parsers (`parsing_utils.py`), and UI simultaneously.
```

**Step 3: Update the prompt and parsing contract in section 5 (lines 129-139)**

Replace:

```
- Prompt builder: `quiz_browser_enhanced.build_prompt(q_type=...)` composes `QUESTION`, `OPTIONS`, and optional context blocks (`COURSE MATERIALS`, `IMAGE CONTENT`, `LINKED CONTENT`). The prompt adapts based on `q_type`:
  - `multichoice_single` (default): asks for single best answer
  - `multichoice_multi`: asks to select ALL correct options, expects comma-separated letters
  - `truefalse`: simplified True/False prompt
  When editing the prompt, preserve the ending instructions that require EXACT output formatting.
- Parsers (all in `parsing_utils.py`):
  - `parse_llm_response(text)` extracts single `ANSWER`, `PROBABILITY`/`CONFIDENCE`, `REASONING` — used for single-answer and true/false
  - `parse_llm_response_multi(text)` extracts comma-separated `ANSWER`, `PROBABILITY`, `REASONING` — used for multi-answer
  - `validate_multi_answer(answers, valid_options)` filters invalid letters against actual option keys
  If you change the expected format, update both the prompt and parser simultaneously and add a unit test (see `tests/test_response_parsing.py`).
```

With:

```
- Prompt builder: `quiz_browser_enhanced.build_prompt(q_type=...)` composes `QUESTION`, `OPTIONS`, and optional context blocks (`COURSE MATERIALS`, `IMAGE CONTENT`, `LINKED CONTENT`). The prompt adapts based on `q_type`:
  - `multichoice_single` (default): asks for single best answer
  - `multichoice_multi`: asks to select ALL correct options, expects comma-separated letters
  - `truefalse`: simplified True/False prompt
  The prompt includes forced alternative enumeration (ALTERNATIVE + ALT_RATIONALE fields) and places PROBABILITY last to enforce reasoning-before-number ordering.
  When editing the prompt, preserve the ending instructions that require EXACT output formatting.
- Parsers (all in `parsing_utils.py`):
  - `parse_llm_response_full(text)` returns dict with `answer`, `confidence`, `reasoning`, `alternative`, `alt_rationale` — primary parser for single-answer and true/false
  - `parse_llm_response_multi_full(text)` returns dict with same keys — primary parser for multi-answer
  - `parse_llm_response(text)` legacy 3-tuple (`answer`, `confidence`, `reasoning`) — backward compat
  - `parse_llm_response_multi(text)` legacy 3-tuple — backward compat
  - `validate_multi_answer(answers, valid_options)` filters invalid letters against actual option keys
  - `extract_alternative(text)` extracts ALTERNATIVE field (letter, "none", or "")
  - `extract_alt_rationale(text)` extracts ALT_RATIONALE field
  If you change the expected format, update both the prompt and parser simultaneously and add a unit test (see `tests/test_response_parsing.py`).
```

**Step 4: Commit**

```bash
git add LLMInstructions.md
git commit -m "docs: update LLMInstructions.md with ALTERNATIVE/ALT_RATIONALE response contract"
```

---

### Task 10: Update `ARCHITECTURE.md`

**Files:**
- Modify: `ARCHITECTURE.md:402-448` (Quiz Attempt JSON example)
- Modify: `ARCHITECTURE.md:190-196` (Type-aware prompting description)

**Step 1: Update the Quiz Attempt JSON example (lines 402-448)**

Replace the `"response"` block in the first question example (lines 419-424):

```json
      "response": {
        "answer": "B",
        "confidence": 85,
        "reasoning": "Based on the definition...",
        "consistency": "9/10"
      },
```

With:

```json
      "response": {
        "answer": "B",
        "confidence": 85,
        "reasoning": "Based on the definition...",
        "alternative": "C",
        "alt_rationale": "C is related but focuses on transformed resources.",
        "consistency": "9/10"
      },
```

Replace the second question example response (lines 437-441):

```json
      "response": {
        "answer": "A, C, D",
        "confidence": 78,
        "reasoning": "Options A, C and D are correct because..."
      },
```

With:

```json
      "response": {
        "answer": "A, C, D",
        "confidence": 78,
        "reasoning": "Options A, C and D are correct because...",
        "alternative": "B",
        "alt_rationale": "B was nearly included but is less directly relevant."
      },
```

**Step 2: Update the Type-aware prompting section (around line 190-196)**

After the line about true/false prompting (line 195), add:

```
All question types include forced alternative enumeration: the LLM must name a runner-up option (`ALTERNATIVE`) with rationale (`ALT_RATIONALE`) before assigning its probability score (`PROBABILITY` is the last output field).
```

**Step 3: Commit**

```bash
git add ARCHITECTURE.md
git commit -m "docs: update ARCHITECTURE.md with alternative fields in JSON examples and prompt description"
```

---

### Task 11: Update module docstring in `parsing_utils.py`

**Files:**
- Modify: `parsing_utils.py:1-33`

**Step 1: Update the module docstring**

Replace lines 16-30:

```python
HOW LLM RESPONSES WORK:
    The AI is asked to answer quiz questions and format its response like:
        ANSWER: B
        CONFIDENCE: 85
        REASONING: Paris is the capital of France because...

    This module extracts those three fields from the free-form text the AI
    returns. The AI does not always follow the format perfectly, so fallback
    patterns handle common variations.

USAGE:
    from parsing_utils import parse_llm_response

    answer, confidence, reasoning = parse_llm_response(llm_output_text)
    # answer: "B" (uppercase letter) or "?" if unparseable
    # confidence: 85 (integer 0-100, clamped)
    # reasoning: "Paris is the capital..." (string, may be empty)
```

With:

```python
HOW LLM RESPONSES WORK:
    The AI is asked to answer quiz questions and format its response like:
        ANSWER: B
        ALTERNATIVE: C
        ALT_RATIONALE: C is close but focuses on the wrong aspect.
        REASONING: Paris is the capital of France because...
        DOUBT: Could be wrong if the question refers to a different era.
        PROBABILITY: 7

    This module extracts those fields from the free-form text the AI
    returns. The AI does not always follow the format perfectly, so fallback
    patterns handle common variations.

USAGE:
    from parsing_utils import parse_llm_response_full

    result = parse_llm_response_full(llm_output_text)
    # result['answer']: "B" (uppercase letter) or "?" if unparseable
    # result['confidence']: 78 (integer 0-100, from 0-9 probability scale)
    # result['reasoning']: "Paris is the capital..." (string, may be empty)
    # result['alternative']: "C" (letter), "none", or "" if missing
    # result['alt_rationale']: "C is close but..." (string, may be empty)

    # Legacy 3-tuple interface (backward compatible):
    from parsing_utils import parse_llm_response
    answer, confidence, reasoning = parse_llm_response(llm_output_text)
```

**Step 2: Run all tests one final time**

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add parsing_utils.py
git commit -m "docs: update parsing_utils module docstring for new response format"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Tests for `extract_alternative()` | `tests/test_response_parsing.py` |
| 2 | Tests for `extract_alt_rationale()` | `tests/test_response_parsing.py` |
| 3 | Implement extractors | `parsing_utils.py` |
| 4 | Tests for `_full()` parsers | `tests/test_response_parsing.py` |
| 5 | Implement `_full()` parsers | `parsing_utils.py` |
| 6 | Update `build_prompt()` | `quiz_browser_enhanced.py` |
| 7 | Update `optimized_prompt_v4()` | `optimized_prompt.py` |
| 8 | Wire `_full()` into `answer_question()` | `quiz_browser_enhanced.py` |
| 9 | Update LLMInstructions.md | `LLMInstructions.md` |
| 10 | Update ARCHITECTURE.md | `ARCHITECTURE.md` |
| 11 | Update parsing_utils docstring | `parsing_utils.py` |
