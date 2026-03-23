#!/usr/bin/env python3
"""
Parsing Utilities
=================

Shared functions for extracting structured data from LLM responses.

This module exists because the same parsing logic was previously duplicated
across App.py, quiz_browser_enhanced.py, and multi_sample_llm.py with slight
variations. Centralising it here means:

1. Fixes (like clamping confidence to 0-100) only need to happen once
2. Automated tests can verify parsing without running Streamlit or Ollama
3. All callers get consistent behaviour

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
"""

import re


def extract_answer(text):
    """
    Extract the answer letter from an LLM response.

    Tries several patterns in order of reliability:
    1. Explicit "ANSWER: B" format (what we ask the LLM to produce)
    2. Natural language like "the answer is B" or "select B"
    3. Phrases like "B is correct" or "B is the best answer"
    4. Last resort: "FINAL ... B" pattern

    Parameters
    ----------
    text : str
        The raw text response from the LLM.

    Returns
    -------
    str
        A single uppercase letter (A-E) or "?" if no answer could be found.

    Examples
    --------
    >>> extract_answer("ANSWER: B\\nCONFIDENCE: 90")
    'B'
    >>> extract_answer("I think the answer is C.")
    'C'
    >>> extract_answer("Some random text with no answer")
    '?'
    """
    if not text:
        return "?"

    # Pattern 1: Explicit ANSWER: X format (primary, most reliable)
    #   Matches: "ANSWER: B", "ANSWER:B", "ANSWER: b", "answer: b", "ANSWER: B."
    match = re.search(r'ANSWER:\s*([A-Za-z])', text, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        if letter in 'ABCDEFGH':  # Support up to H for >5 option questions
            return letter

    # Pattern 2: Natural language - "the answer is B", "I select C", "choose A"
    #   Allows words between the keyword and the letter (e.g., "answer is C")
    match = re.search(r'(?:answer|select|choose)(?:\s+\w+)*?\s+([A-Ea-e])\b', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 3: "[Letter] is correct" or "[Letter] is the best answer"
    match = re.search(r'\b([A-E])\s*(?:is correct|is the (?:best|correct))', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern 4: "FINAL ... [Letter]" pattern
    match = re.search(r'FINAL.*?([A-E])\b', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return "?"


def extract_confidence(text):
    """
    Extract and validate the confidence score from an LLM response.

    Supports two scales:
    - PROBABILITY: 0-9 scale (preferred; see Yang et al., 2024 — best
      calibration for Llama3-8B). Converted to 0-100 for display.
    - CONFIDENCE: 0-100 scale (legacy fallback)

    The 0-9 scale reduces the clustering at multiples of five observed
    with 0-100 scales (Xiong et al., 2024) and produces more varied,
    better-calibrated scores on small models (Yang et al., 2024).

    Parameters
    ----------
    text : str
        The raw text response from the LLM.

    Returns
    -------
    int
        Confidence score between 0 and 100 (inclusive).

    Examples
    --------
    >>> extract_confidence("PROBABILITY: 7")
    78
    >>> extract_confidence("PROBABILITY: 9")
    100
    >>> extract_confidence("CONFIDENCE: 85")
    85
    >>> extract_confidence("CONFIDENCE: 150")
    100
    >>> extract_confidence("No confidence here")
    0
    """
    if not text:
        return 0

    # Primary: PROBABILITY on 0-9 scale (Yang et al., 2024)
    prob_match = re.search(r'PROBABILITY:\s*(\d+)', text, re.IGNORECASE)
    if prob_match:
        value = int(prob_match.group(1))
        if value <= 9:
            # Convert 0-9 scale to 0-100 for consistent display
            # 0→0, 1→11, 2→22, 3→33, 4→44, 5→56, 6→67, 7→78, 8→89, 9→100
            return round(value * 100 / 9)
        else:
            # Model wrote a larger number despite instructions — treat as 0-100
            return max(0, min(100, value))

    # Fallback: CONFIDENCE on 0-100 scale (legacy prompts)
    conf_match = re.search(r'CONFIDENCE:\s*(\d+)', text, re.IGNORECASE)
    if conf_match:
        value = int(conf_match.group(1))
        return max(0, min(100, value))

    return 0


def extract_reasoning(text):
    """
    Extract the reasoning text from an LLM response.

    Captures everything after "REASONING:" up to either a double newline
    or the end of the text. This handles both single-line and multi-line
    reasoning.

    Parameters
    ----------
    text : str
        The raw text response from the LLM.

    Returns
    -------
    str
        The reasoning text, stripped of leading/trailing whitespace.
        Returns empty string if no reasoning marker is found.

    Examples
    --------
    >>> extract_reasoning("REASONING: Paris is the capital of France.")
    'Paris is the capital of France.'
    >>> extract_reasoning("No reasoning marker here")
    ''
    """
    if not text:
        return ""

    # Match everything after REASONING: until double newline or end of string
    # Using re.DOTALL so . matches newlines (captures multi-line reasoning)
    match = re.search(r'REASONING:\s*(.+?)(?=\n\n|\Z)', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return ""


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


def parse_llm_response(text):
    """
    Parse a complete LLM response into answer, confidence, and reasoning.

    This is the main function most callers should use. It combines
    extract_answer(), extract_confidence(), and extract_reasoning()
    into a single call.

    Parameters
    ----------
    text : str
        The raw text response from the LLM.

    Returns
    -------
    tuple of (str, int, str)
        - answer: uppercase letter A-E or "?" if unparseable
        - confidence: integer 0-100 (clamped)
        - reasoning: string (may be empty)

    Examples
    --------
    >>> parse_llm_response("ANSWER: B\\nCONFIDENCE: 85\\nREASONING: Because...")
    ('B', 85, 'Because...')
    """
    answer = extract_answer(text)
    confidence = extract_confidence(text)
    reasoning = extract_reasoning(text)
    return answer, confidence, reasoning


# ============================================
# MULTI-ANSWER PARSING (for checkbox questions)
# ============================================

def extract_answers_multi(text):
    """
    Extract multiple answer letters from an LLM response.

    For multi-answer MCQ questions where the LLM is asked to select
    ALL correct options. Tries several patterns:
    1. "ANSWER: A, C, D" (comma-separated)
    2. "ANSWER: A and C and D" (natural language)
    3. "ANSWER: ACD" (no separators)
    4. Falls back to extract_answer() wrapped in a list

    Returns
    -------
    list of str
        Sorted list of unique uppercase letters, e.g. ["A", "C", "D"].
        Returns ["?"] if no answer could be found.

    Examples
    --------
    >>> extract_answers_multi("ANSWER: A, C, D")
    ['A', 'C', 'D']
    >>> extract_answers_multi("ANSWER: B")
    ['B']
    """
    if not text:
        return ["?"]

    # Pattern 1: "ANSWER: A, C, D" or "ANSWER: A,C,D"
    match = re.search(r'ANSWER:\s*([A-Ha-h](?:\s*,\s*[A-Ha-h])+)', text, re.IGNORECASE)
    if match:
        letters = [l.strip().upper() for l in match.group(1).split(',')]
        letters = sorted(set(l for l in letters if l in 'ABCDEFGH'))
        if letters:
            return letters

    # Pattern 2: "ANSWER: A and C and D"
    match = re.search(r'ANSWER:\s*([A-Ha-h](?:\s+and\s+[A-Ha-h])+)', text, re.IGNORECASE)
    if match:
        letters = [l.strip().upper() for l in re.split(r'\s+and\s+', match.group(1))]
        letters = sorted(set(l for l in letters if l in 'ABCDEFGH'))
        if letters:
            return letters

    # Pattern 3: "ANSWER: ACD" (consecutive letters, no separators, 2+ letters)
    match = re.search(r'ANSWER:\s*([A-Ha-h]{2,})\b', text, re.IGNORECASE)
    if match:
        letters = sorted(set(l.upper() for l in match.group(1) if l.upper() in 'ABCDEFGH'))
        if letters:
            return letters

    # Fallback: try single-answer extraction
    single = extract_answer(text)
    if single != "?":
        return [single]

    return ["?"]


def validate_multi_answer(answers, valid_options):
    """
    Validate that parsed multi-answer letters are valid option keys.

    Filters out any letters not present in valid_options.

    Parameters
    ----------
    answers : list of str
        Parsed answer letters from extract_answers_multi().
    valid_options : set or dict
        The valid option keys (e.g. {"A", "B", "C", "D"} or options dict).

    Returns
    -------
    list of str
        Filtered list of valid letters, or ["?"] if none are valid.
    """
    if not answers or answers == ["?"]:
        return ["?"]

    valid_keys = set(valid_options) if isinstance(valid_options, dict) else set(valid_options)
    filtered = sorted(l for l in answers if l in valid_keys)

    if not filtered:
        print(f"  ⚠️  All answer letters {answers} are invalid for options {sorted(valid_keys)}", flush=True)
        return ["?"]

    removed = set(answers) - set(filtered) - {"?"}
    if removed:
        print(f"  ⚠️  Filtered invalid answer letters: {sorted(removed)} (valid: {sorted(valid_keys)})", flush=True)

    return filtered


def parse_llm_response_multi(text):
    """
    Parse a multi-answer LLM response into answers string, confidence, reasoning.

    Like parse_llm_response() but returns a comma-separated answer string
    for multi-answer questions.

    Returns
    -------
    tuple of (str, int, str)
        - answer: comma-separated letters like "A, C, D" or "?" if unparseable
        - confidence: integer 0-100
        - reasoning: string
    """
    answers = extract_answers_multi(text)
    confidence = extract_confidence(text)
    reasoning = extract_reasoning(text)
    answer_str = ", ".join(answers)
    return answer_str, confidence, reasoning


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
