"""
Tests for LLM Response Parsing
================================

WHAT THESE TESTS DO:
    When the AI answers a quiz question, it returns free-form text that looks
    something like:

        ANSWER: B
        CONFIDENCE: 85
        REASONING: Paris is the capital of France because...

    But the AI doesn't always follow this format perfectly. Sometimes it writes
    "ANSWER:B" (no space), "answer: b" (lowercase), "CONFIDENCE: 150" (invalid),
    or skips the markers entirely.

    These tests verify that our parsing code handles all these variations
    correctly, so that quiz results are accurate regardless of how the AI
    formats its response.

WHY THESE TESTS MATTER:
    This is the most fragile part of the codebase. The parsing regex is the
    bridge between unpredictable AI output and structured data. If it breaks,
    every quiz scan produces wrong results. These tests catch regressions
    before they reach educators.

HOW TO RUN:
    pytest tests/test_response_parsing.py -v
"""

import sys
import os

# Add the project root to the Python path so we can import our modules.
# This is needed because tests/ is a subfolder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parsing_utils import extract_answer, extract_confidence, extract_reasoning, parse_llm_response


# ============================================
# ANSWER EXTRACTION TESTS
# ============================================
# These test the extract_answer() function which pulls the answer letter
# out of the AI's response text.

class TestExtractAnswer:
    """Tests for extracting the answer letter from LLM responses."""

    def test_standard_format(self):
        """The ideal case: AI follows the exact format we asked for."""
        assert extract_answer("ANSWER: B\nCONFIDENCE: 85") == "B"

    def test_no_space_after_colon(self):
        """AI writes 'ANSWER:B' without a space."""
        assert extract_answer("ANSWER:B\nCONFIDENCE: 85") == "B"

    def test_lowercase_answer(self):
        """AI writes the letter in lowercase - should still work."""
        assert extract_answer("ANSWER: b\nCONFIDENCE: 85") == "B"

    def test_lowercase_marker(self):
        """AI writes 'answer: b' all lowercase."""
        # Our primary regex is case-sensitive on "ANSWER:" but the fallback
        # patterns handle lowercase. This tests the full chain.
        result = extract_answer("answer: b\nconfidence: 85")
        # The primary pattern r'ANSWER:\s*([A-Za-z])' does match lowercase ANSWER
        # because re.search is case-sensitive on 'ANSWER' but [A-Za-z] matches both
        assert result == "B"

    def test_answer_with_period(self):
        """AI writes 'ANSWER: B.' with a trailing period."""
        assert extract_answer("ANSWER: B.\nCONFIDENCE: 85") == "B"

    def test_answer_with_parenthetical(self):
        """AI writes 'ANSWER: B (Paris)' with extra text."""
        assert extract_answer("ANSWER: B (Paris)\nCONFIDENCE: 85") == "B"

    def test_all_five_letters(self):
        """Each valid answer letter A through E works."""
        for letter in "ABCDE":
            assert extract_answer(f"ANSWER: {letter}") == letter

    def test_extended_options_f_g_h(self):
        """Questions with more than 5 options (F, G, H) are supported."""
        assert extract_answer("ANSWER: F") == "F"
        assert extract_answer("ANSWER: G") == "G"
        assert extract_answer("ANSWER: H") == "H"

    def test_fallback_natural_language(self):
        """AI writes 'the answer is C' without using the ANSWER: format."""
        assert extract_answer("I think the answer is C because it's correct.") == "C"

    def test_fallback_select_language(self):
        """AI writes 'I select B' or 'I would select B'."""
        assert extract_answer("After analysis, I select B as my choice.") == "B"

    def test_fallback_is_correct(self):
        """AI writes 'B is correct' or 'B is the best answer'."""
        assert extract_answer("Based on analysis, B is correct.") == "B"
        assert extract_answer("Option D is the best answer here.") == "D"

    def test_fallback_final_pattern(self):
        """AI writes 'FINAL ANSWER: A' or 'My final choice is A'."""
        assert extract_answer("FINAL ANSWER: A") == "A"

    def test_no_answer_found(self):
        """If the AI's response contains no recognisable answer, return '?'."""
        assert extract_answer("I'm not sure about this question.") == "?"

    def test_empty_string(self):
        """Empty input returns '?'."""
        assert extract_answer("") == "?"

    def test_none_input(self):
        """None input returns '?'."""
        assert extract_answer(None) == "?"

    def test_answer_in_long_response(self):
        """Answer marker buried in a long multi-paragraph response."""
        text = """Let me analyze this question step by step.

First, I'll consider each option:
A. London - This is the capital of the UK
B. Paris - This is the capital of France
C. Berlin - This is the capital of Germany
D. Madrid - This is the capital of Spain

Based on the question asking about France:

ANSWER: B
CONFIDENCE: 95
REASONING: Paris is the capital of France."""
        assert extract_answer(text) == "B"

    def test_multiple_answer_markers_takes_first(self):
        """If the AI writes ANSWER: twice, take the first one."""
        text = "ANSWER: A\nWait, let me reconsider.\nANSWER: B"
        # re.search finds the first match
        assert extract_answer(text) == "A"


# ============================================
# CONFIDENCE EXTRACTION TESTS
# ============================================
# These test the extract_confidence() function which pulls the confidence
# score and clamps it to a valid 0-100 range.

class TestExtractConfidence:
    """Tests for extracting and validating confidence scores."""

    def test_standard_format(self):
        """Normal confidence value."""
        assert extract_confidence("CONFIDENCE: 85") == 85

    def test_zero_confidence(self):
        """AI reports 0% confidence."""
        assert extract_confidence("CONFIDENCE: 0") == 0

    def test_hundred_confidence(self):
        """AI reports exactly 100% confidence."""
        assert extract_confidence("CONFIDENCE: 100") == 100

    def test_over_100_clamped(self):
        """
        AI writes confidence > 100 (a known issue).
        This should be clamped to 100 instead of being displayed as 150%.
        """
        assert extract_confidence("CONFIDENCE: 150") == 100

    def test_way_over_100_clamped(self):
        """Extreme values like 999 are clamped."""
        assert extract_confidence("CONFIDENCE: 999") == 100

    def test_no_confidence_found(self):
        """If no CONFIDENCE marker exists, return 0."""
        assert extract_confidence("Just some text without confidence") == 0

    def test_empty_string(self):
        """Empty input returns 0."""
        assert extract_confidence("") == 0

    def test_none_input(self):
        """None input returns 0."""
        assert extract_confidence(None) == 0

    def test_confidence_no_space(self):
        """AI writes 'CONFIDENCE:85' without a space."""
        assert extract_confidence("CONFIDENCE:85") == 85

    def test_confidence_in_full_response(self):
        """Confidence extracted correctly from a complete response."""
        text = "ANSWER: B\nCONFIDENCE: 72\nREASONING: Because..."
        assert extract_confidence(text) == 72

    def test_confidence_with_percent_sign(self):
        """AI writes 'CONFIDENCE: 85%' - the % is ignored by \\d+ regex."""
        assert extract_confidence("CONFIDENCE: 85%") == 85

    def test_low_confidence(self):
        """Very low but valid confidence."""
        assert extract_confidence("CONFIDENCE: 5") == 5


# ============================================
# PROBABILITY SCALE TESTS (0-9 → 0-100)
# ============================================
# The 0-9 probability scale (Yang et al., 2024) produces better-calibrated
# confidence scores for small models like Llama3-8B. These tests verify the
# PROBABILITY: format is parsed correctly and converted to 0-100.

class TestProbabilityScale:
    """Tests for the 0-9 probability scale (preferred format)."""

    def test_probability_zero(self):
        """PROBABILITY: 0 means random guess → 0%."""
        assert extract_confidence("PROBABILITY: 0") == 0

    def test_probability_five(self):
        """PROBABILITY: 5 means coin flip → 56%."""
        assert extract_confidence("PROBABILITY: 5") == 56

    def test_probability_nine(self):
        """PROBABILITY: 9 means near certain → 100%."""
        assert extract_confidence("PROBABILITY: 9") == 100

    def test_probability_seven(self):
        """PROBABILITY: 7 means probably correct → 78%."""
        assert extract_confidence("PROBABILITY: 7") == 78

    def test_probability_three(self):
        """PROBABILITY: 3 means somewhat unlikely → 33%."""
        assert extract_confidence("PROBABILITY: 3") == 33

    def test_probability_one(self):
        """PROBABILITY: 1 means very unlikely → 11%."""
        assert extract_confidence("PROBABILITY: 1") == 11

    def test_probability_in_full_response(self):
        """PROBABILITY extracted from a complete response with new format."""
        text = "ANSWER: B\nPROBABILITY: 7\nREASONING: Because...\nDOUBT: Could be wrong if..."
        assert extract_confidence(text) == 78

    def test_probability_no_space(self):
        """PROBABILITY:7 without space."""
        assert extract_confidence("PROBABILITY:7") == 78

    def test_probability_takes_precedence_over_confidence(self):
        """If both PROBABILITY and CONFIDENCE are present, PROBABILITY wins."""
        text = "PROBABILITY: 6\nCONFIDENCE: 90"
        assert extract_confidence(text) == 67

    def test_probability_large_number_fallback(self):
        """If model writes PROBABILITY: 85 (ignoring instructions), treat as 0-100."""
        assert extract_confidence("PROBABILITY: 85") == 85

    def test_probability_over_100_clamped(self):
        """PROBABILITY: 150 (model error) clamped to 100."""
        assert extract_confidence("PROBABILITY: 150") == 100

    def test_all_scale_values(self):
        """Verify all 10 values of the 0-9 scale produce expected percentages."""
        expected = {0: 0, 1: 11, 2: 22, 3: 33, 4: 44, 5: 56, 6: 67, 7: 78, 8: 89, 9: 100}
        for val, pct in expected.items():
            assert extract_confidence(f"PROBABILITY: {val}") == pct, \
                f"PROBABILITY: {val} should map to {pct}%"


# ============================================
# REASONING EXTRACTION TESTS
# ============================================
# These test the extract_reasoning() function which captures the AI's
# explanation for its answer.

class TestExtractReasoning:
    """Tests for extracting reasoning text."""

    def test_standard_single_line(self):
        """Normal single-line reasoning."""
        text = "ANSWER: B\nCONFIDENCE: 85\nREASONING: Paris is the capital of France."
        assert extract_reasoning(text) == "Paris is the capital of France."

    def test_multi_line_reasoning(self):
        """Reasoning that spans multiple lines (should be captured fully)."""
        text = "ANSWER: B\nCONFIDENCE: 85\nREASONING: Paris is the capital.\nIt has been since 508 AD."
        result = extract_reasoning(text)
        assert "Paris is the capital." in result
        assert "508 AD" in result

    def test_reasoning_with_trailing_whitespace(self):
        """Reasoning with extra whitespace is stripped."""
        text = "REASONING:   Paris is the capital.   "
        assert extract_reasoning(text) == "Paris is the capital."

    def test_no_reasoning_found(self):
        """If no REASONING marker exists, return empty string."""
        assert extract_reasoning("ANSWER: B\nCONFIDENCE: 85") == ""

    def test_empty_string(self):
        """Empty input returns empty string."""
        assert extract_reasoning("") == ""

    def test_none_input(self):
        """None input returns empty string."""
        assert extract_reasoning(None) == ""

    def test_reasoning_stops_at_double_newline(self):
        """Reasoning ends at double newline (start of next section)."""
        text = "REASONING: Paris is the capital.\n\nSome other section here."
        assert extract_reasoning(text) == "Paris is the capital."


# ============================================
# FULL RESPONSE PARSING TESTS
# ============================================
# These test parse_llm_response() which combines all three extractions.

class TestParseLlmResponse:
    """Tests for the combined parse_llm_response() function."""

    def test_perfect_format(self):
        """AI follows the exact requested format."""
        text = "ANSWER: B\nCONFIDENCE: 85\nREASONING: Paris is the capital of France."
        answer, confidence, reasoning = parse_llm_response(text)
        assert answer == "B"
        assert confidence == 85
        assert reasoning == "Paris is the capital of France."

    def test_all_defaults_on_garbage(self):
        """Completely unparseable response returns safe defaults."""
        answer, confidence, reasoning = parse_llm_response("asdfghjkl")
        assert answer == "?"
        assert confidence == 0
        assert reasoning == ""

    def test_empty_string(self):
        """Empty response returns safe defaults."""
        answer, confidence, reasoning = parse_llm_response("")
        assert answer == "?"
        assert confidence == 0
        assert reasoning == ""

    def test_none_input(self):
        """None input returns safe defaults."""
        answer, confidence, reasoning = parse_llm_response(None)
        assert answer == "?"
        assert confidence == 0
        assert reasoning == ""

    def test_partial_response_answer_only(self):
        """Response has answer but no confidence or reasoning."""
        answer, confidence, reasoning = parse_llm_response("ANSWER: C")
        assert answer == "C"
        assert confidence == 0
        assert reasoning == ""

    def test_confidence_clamped_in_full_parse(self):
        """Confidence clamping works through the full parse pipeline."""
        text = "ANSWER: A\nCONFIDENCE: 200\nREASONING: Very sure"
        answer, confidence, reasoning = parse_llm_response(text)
        assert confidence == 100

    def test_realistic_long_response(self):
        """A realistic multi-paragraph AI response is parsed correctly."""
        text = """Let me analyze this step by step.

The question asks about the capital of France.

Option A (London) is the capital of the UK.
Option B (Paris) is the capital of France.
Option C (Berlin) is the capital of Germany.
Option D (Madrid) is the capital of Spain.

ANSWER: B
CONFIDENCE: 95
REASONING: Paris has been the capital of France since the late 10th century, making it the clear correct choice."""
        answer, confidence, reasoning = parse_llm_response(text)
        assert answer == "B"
        assert confidence == 95
        assert "Paris" in reasoning

    def test_new_probability_format_full_parse(self):
        """Full parse of the new probability format with DOUBT line."""
        text = """QUESTION TYPE: Recall

KEY INSIGHT: Testing knowledge of European capitals.

EVALUATE OPTIONS:
A: [ELIMINATE] - London is the UK capital
B: [KEEP] - Paris is the capital of France
C: [ELIMINATE] - Berlin is Germany's capital
D: [ELIMINATE] - Madrid is Spain's capital

DOUBT CHECK: All major geography sources confirm Paris. No reasonable doubt.

ANSWER: B
PROBABILITY: 8
REASONING: Paris has been the capital of France since the late 10th century.
DOUBT: Could be wrong if the question refers to a historical period before Paris became capital."""
        answer, confidence, reasoning = parse_llm_response(text)
        assert answer == "B"
        assert confidence == 89  # 8 on 0-9 scale → 89%
        assert "Paris" in reasoning

    def test_probability_uncertain_response(self):
        """Low probability score indicates genuine uncertainty."""
        text = "ANSWER: C\nPROBABILITY: 4\nREASONING: Not sure about this one.\nDOUBT: Options B and D are also plausible."
        answer, confidence, reasoning = parse_llm_response(text)
        assert answer == "C"
        assert confidence == 44  # 4 on 0-9 scale → 44%
        assert "Not sure" in reasoning
