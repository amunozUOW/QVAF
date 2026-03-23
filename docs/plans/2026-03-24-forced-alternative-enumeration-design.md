# Design: Forced Alternative Enumeration + Reasoning-Before-Probability

**Date:** 2026-03-24
**Status:** Approved

## Motivation

The current confidence elicitation prompt asks the model for a DOUBT field (free-form) but does not force it to name a specific competing answer letter. Inspired by the evidence-coding confidence algorithm used in a parallel classification project, this design adds two mechanisms:

1. **Forced alternative enumeration** — the model must name the next most plausible answer option (or "none") and explain why it was rejected, before assigning probability.
2. **Reasoning-before-number ordering** — PROBABILITY moves to the final output line, ensuring the model has completed all analytical reasoning before committing to a numeric confidence score.

These changes produce structured data on where the model is genuinely uncertain vs. confidently wrong, which is directly actionable for identifying vulnerable questions.

## Changes

### 1. Prompt: `build_prompt()` in `quiz_browser_enhanced.py`

The narrative instruction block changes from:

```
First, evaluate each option briefly. {selection_guidance} Before rating your
probability, consider what is the strongest argument AGAINST your chosen answer.
Then rate the probability...
```

To a structured multi-step flow:

```
First, evaluate each option briefly. {selection_guidance}

After selecting your answer, identify which other option was the strongest
runner-up. State it as ALTERNATIVE: [letter] or ALTERNATIVE: none if no other
option is plausible. Briefly explain why you rejected it.

Then, considering the strength of that alternative, rate the probability your
primary answer is correct.
```

Output format reordered from 4 lines to 6:

```
ANSWER: X
ALTERNATIVE: Y
ALT_RATIONALE: Why Y was considered and rejected
REASONING: Your explanation here
DOUBT: What could make your answer wrong
PROBABILITY: N
```

For `multichoice_multi`: ALTERNATIVE is the option the model was least sure about including or excluding.

### 2. Prompt: `optimized_prompt_v4` in `optimized_prompt.py`

Steps 5-6 become steps 5-7:

```
5. FINAL SELECTION: From options marked KEEP, select the single best answer.

6. ALTERNATIVE: Which other option was the strongest runner-up? Name it, or
   say "none" if no other option is remotely plausible. Explain briefly why
   you rejected it.

7. DOUBT CHECK: Considering the alternative and any other uncertainties, what
   is the strongest argument AGAINST your chosen answer?
```

Output format reordered identically: ANSWER → ALTERNATIVE → ALT_RATIONALE → REASONING → DOUBT → PROBABILITY (last).

### 3. Parsing: `parsing_utils.py`

Two new extraction functions:

```python
def extract_alternative(text) -> str:
    """Extract ALTERNATIVE: field. Returns letter A-H, 'none', or ''."""

def extract_alt_rationale(text) -> str:
    """Extract ALT_RATIONALE: field. Returns string, may be empty."""
```

New combined parser (existing `parse_llm_response()` 3-tuple unchanged for backward compat):

```python
def parse_llm_response_full(text) -> dict:
    """Parse all fields including alternative. Returns dict with keys:
    answer, confidence, reasoning, alternative, alt_rationale"""
```

Similarly for multi-answer:

```python
def parse_llm_response_multi_full(text) -> dict:
    """Parse all fields for multi-answer. Returns dict with keys:
    answer, confidence, reasoning, alternative, alt_rationale"""
```

### 4. Data storage (JSON output)

Response dict gains two fields:

```json
"response": {
    "answer": "B",
    "confidence": 78,
    "reasoning": "...",
    "alternative": "C",
    "alt_rationale": "C could apply if...",
    "consistency": "9/10"
}
```

Downstream consumers (merge_attempts, analysis_agent, reform_agent) pass these through without modification — they are additive fields.

### 5. Documentation

- **ARCHITECTURE.md:** Update Quiz Attempt JSON example, update `build_prompt()` description.
- **LLMInstructions.md:** Update section 1 (Critical Invariants) LLM response contract. Update section 5 (Prompt and parsing contract).

### 6. Tests (`tests/test_response_parsing.py`)

New test classes:
- `TestExtractAlternative` — letter, "none", missing, case variations
- `TestExtractAltRationale` — standard, multi-line, missing
- `TestParseLlmResponseFull` — full dict parse with all fields
- Verify existing `parse_llm_response()` 3-tuple still works unchanged

## What doesn't change

- `extract_answer()`, `extract_confidence()`, `extract_reasoning()` — untouched
- `parse_llm_response()` return signature — stays as 3-tuple
- `parse_llm_response_multi()` — same 3-tuple
- Analysis/dashboard code — new fields stored but not yet visualized
- `num_predict` stays at 300 tokens initially (monitor for truncation)

## Risk

Token budget: the two new output fields add ~20-40 tokens. With `num_predict: 300`, this is tight. If truncation occurs, bump to 350-400.
