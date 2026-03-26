"""
Tests for Correctness Checking
================================

WHAT THESE TESTS DO:
    After the AI answers a question and we know the correct answer, the code
    needs to determine: did the AI get it right?

    This sounds simple, but there are edge cases:
    - The AI might answer "A" and the correct answer is "A. Paris" (should match)
    - The correct answer might be "UNKNOWN" (should return None, not False)
    - Both values might be None (should not crash)

    There's also the "correctness pattern" logic that categorises the AI's
    performance across two conditions (with and without course materials):
    - CORRECT_BOTH: AI got it right both times
    - CORRECT_RAG_ONLY: AI only got it right with course materials
    - CORRECT_BASELINE_ONLY: AI only got it right without course materials
    - INCORRECT_BOTH: AI got it wrong both times
    - UNKNOWN: We can't determine correctness

    Additionally, there is a "confidence flag" that highlights cases where the
    AI was very confident (>=90%) but got the answer wrong. These are
    particularly concerning for educators because students using AI would
    receive a confident-sounding wrong answer.

WHY THESE TESTS MATTER:
    If correctness checking is wrong, the entire vulnerability analysis is
    wrong. A question might be flagged as "AI got this right" when it actually
    got it wrong, leading the educator to worry about the wrong questions.

HOW TO RUN:
    pytest tests/test_correctness.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reform_agent import check_correct, get_correctness_pattern, get_confidence_flag


# ============================================
# check_correct() TESTS
# ============================================
# This function compares the AI's answer to the correct answer.
# It's used as a fallback when Moodle's own correctness data isn't available.

class TestCheckCorrect:
    """Tests for the check_correct() function in reform_agent.py."""

    def test_exact_letter_match(self):
        """Simple case: both are just letters."""
        assert check_correct("A", "A") == True

    def test_different_letters(self):
        """AI answered wrong."""
        assert check_correct("A", "B") == False

    def test_case_insensitive(self):
        """Lowercase and uppercase should match."""
        assert check_correct("a", "A") == True
        assert check_correct("A", "a") == True

    def test_answer_with_text(self):
        """AI answer is 'A. Paris' and correct is 'A' - should match on first letter."""
        assert check_correct("A. Paris", "A") == True

    def test_correct_with_text(self):
        """Correct answer is 'B. London' and AI answered 'B' - should match."""
        assert check_correct("B", "B. London") == True

    def test_both_with_text(self):
        """Both have text, same first letter."""
        assert check_correct("C. Berlin", "C. Berlin is correct") == True

    def test_both_with_text_different(self):
        """Both have text, different first letters."""
        assert check_correct("A. Paris", "B. London") == False

    def test_empty_answer(self):
        """Empty answer should return None (unknown), not False."""
        assert check_correct("", "A") is None

    def test_empty_correct_answer(self):
        """Empty correct answer should return None."""
        assert check_correct("A", "") is None

    def test_both_empty(self):
        """Both empty should return None."""
        assert check_correct("", "") is None

    def test_none_answer(self):
        """None answer should return None."""
        assert check_correct(None, "A") is None

    def test_none_correct(self):
        """None correct answer should return None."""
        assert check_correct("A", None) is None

    def test_unknown_correct_answer(self):
        """When correct answer is 'UNKNOWN', return None (we don't know)."""
        assert check_correct("A", "UNKNOWN") is None

    def test_whitespace_handling(self):
        """Answers with extra whitespace should still match."""
        assert check_correct("  A  ", "A") == True
        assert check_correct("A", "  A  ") == True

    # --- Multi-answer correctness ---

    def test_multi_answer_exact_match(self):
        """Multi-answer: exact same letters in same order."""
        assert check_correct("A, C, D", "A, C, D") == True

    def test_multi_answer_different_order(self):
        """Multi-answer: same letters in different order → correct."""
        assert check_correct("D, A, C", "A, C, D") == True

    def test_multi_answer_missing_selection(self):
        """Multi-answer: missing a correct answer → incorrect."""
        assert check_correct("A, C", "A, C, D") == False

    def test_multi_answer_extra_selection(self):
        """Multi-answer: extra incorrect answer → incorrect."""
        assert check_correct("A, B, C, D", "A, C, D") == False

    def test_multi_answer_completely_wrong(self):
        """Multi-answer: completely different selections → incorrect."""
        assert check_correct("B, E", "A, C, D") == False

    def test_multi_answer_single_vs_multi(self):
        """Single answer vs multi-answer correct → incorrect."""
        assert check_correct("A", "A, C, D") == False

    def test_multi_answer_case_insensitive(self):
        """Multi-answer: case insensitive comparison."""
        assert check_correct("a, c, d", "A, C, D") == True

    def test_multi_answer_with_spaces(self):
        """Multi-answer: extra spaces around letters."""
        assert check_correct("A ,  C , D", "A, C, D") == True


# ============================================
# get_correctness_pattern() TESTS
# ============================================
# This function takes two booleans (correct without RAG, correct with RAG)
# and returns a descriptive pattern name.

class TestGetCorrectnessPattern:
    """Tests for correctness pattern classification."""

    def test_correct_both(self):
        """AI correct in both conditions."""
        assert get_correctness_pattern(True, True) == "CORRECT_BOTH"

    def test_correct_rag_only(self):
        """AI correct only with course materials."""
        assert get_correctness_pattern(False, True) == "CORRECT_RAG_ONLY"

    def test_correct_baseline_only(self):
        """AI correct only without course materials (anomalous)."""
        assert get_correctness_pattern(True, False) == "CORRECT_BASELINE_ONLY"

    def test_incorrect_both(self):
        """AI wrong in both conditions."""
        assert get_correctness_pattern(False, False) == "INCORRECT_BOTH"

    def test_both_none(self):
        """Both unknown - should return UNKNOWN."""
        assert get_correctness_pattern(None, None) == "UNKNOWN"

    def test_basic_mode_correct(self):
        """Basic mode (no RAG scan): correct_with is None, correct_without is True."""
        assert get_correctness_pattern(True, None) == "CORRECT_BOTH"

    def test_basic_mode_incorrect(self):
        """Basic mode: correct_with is None, correct_without is False."""
        assert get_correctness_pattern(False, None) == "INCORRECT_BOTH"

    def test_only_rag_data_correct(self):
        """Edge case: only RAG data available and correct."""
        assert get_correctness_pattern(None, True) == "CORRECT_RAG_ONLY"

    def test_only_rag_data_incorrect(self):
        """Edge case: only RAG data available and incorrect."""
        assert get_correctness_pattern(None, False) == "INCORRECT_BOTH"

    def test_basic_mode_none_without(self):
        """Basic mode: correct_without is None (not False), correct_with is None."""
        # Both None should return UNKNOWN
        assert get_correctness_pattern(None, None) == "UNKNOWN"


# ============================================
# get_confidence_flag() TESTS
# ============================================
# This function flags cases where the AI was highly confident (>=90%)
# but got the answer wrong. These are especially concerning.

class TestGetConfidenceFlag:
    """Tests for high-confidence incorrect answer flagging."""

    def test_high_confidence_wrong_baseline(self):
        """AI was 95% confident but wrong (baseline) - should flag."""
        result = get_confidence_flag(False, 95, True, 80)
        assert result is not None
        assert "baseline" in result.lower()

    def test_high_confidence_wrong_rag(self):
        """AI was 92% confident but wrong (RAG) - should flag."""
        result = get_confidence_flag(True, 80, False, 92)
        assert result is not None
        assert "rag" in result.lower()

    def test_high_confidence_wrong_both(self):
        """AI was highly confident and wrong in both conditions."""
        result = get_confidence_flag(False, 95, False, 91)
        assert result is not None
        assert "baseline" in result.lower()
        assert "rag" in result.lower()

    def test_low_confidence_wrong(self):
        """AI was wrong but with low confidence - should NOT flag."""
        result = get_confidence_flag(False, 60, False, 70)
        assert result is None

    def test_high_confidence_correct(self):
        """AI was highly confident AND correct - should NOT flag."""
        result = get_confidence_flag(True, 95, True, 98)
        assert result is None

    def test_exactly_90_wrong(self):
        """Boundary: exactly 90% confidence and wrong - should flag."""
        result = get_confidence_flag(False, 90, True, 80)
        assert result is not None

    def test_89_wrong(self):
        """Boundary: 89% confidence and wrong - should NOT flag (below threshold)."""
        result = get_confidence_flag(False, 89, True, 80)
        assert result is None

    def test_none_correctness(self):
        """When correctness is None (unknown), should NOT flag."""
        result = get_confidence_flag(None, 95, None, 95)
        assert result is None

    def test_zero_confidence(self):
        """Zero confidence and wrong - should NOT flag."""
        result = get_confidence_flag(False, 0, False, 0)
        assert result is None
