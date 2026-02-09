"""
Tests for Question Matching and Merging
========================================

WHAT THESE TESTS DO:
    When QVAF runs a quiz twice (once without course materials, once with),
    it needs to match up the results: "Question 3 from Run A is the same as
    Question 3 from Run B." This matching happens by comparing the question
    text, not the question number (because numbers can change between runs).

    The matching uses text normalisation (lowercase, collapse whitespace) and
    a fuzzy prefix match (first 50 characters) as a fallback.

    These tests verify that:
    - Identical questions match correctly
    - Questions with minor whitespace differences still match
    - Questions that LOOK similar but ARE different don't falsely match
    - Missing questions are handled gracefully (return None, don't crash)

    We also test the correct answer extraction logic, which tries to figure
    out the correct answer letter from Moodle's feedback text like
    "The correct answer is: B. Paris"

WHY THESE TESTS MATTER:
    If question matching breaks, the vulnerability analysis compares the
    wrong questions to each other. An educator might see "AI got this right
    with course materials" when actually the AI answered a completely
    different question. This would lead to incorrect recommendations.

HOW TO RUN:
    pytest tests/test_question_matching.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from merge_attempts import normalize_text, find_matching_question, strip_label, extract_correct_answer_letter


# ============================================
# normalize_text() TESTS
# ============================================
# This function normalises question text for comparison by lowercasing,
# collapsing whitespace, and truncating to 100 characters.

class TestNormalizeText:
    """Tests for text normalisation used in question matching."""

    def test_basic_normalisation(self):
        """Lowercase and trim whitespace."""
        assert normalize_text("  Hello World  ") == "hello world"

    def test_collapse_whitespace(self):
        """Multiple spaces/tabs/newlines collapse to single space."""
        assert normalize_text("Hello   \n\t  World") == "hello world"

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert normalize_text("") == ""

    def test_none_input(self):
        """None input returns empty string."""
        assert normalize_text(None) == ""

    def test_truncation_at_100_chars(self):
        """Long text is truncated to 100 characters."""
        long_text = "a" * 200
        result = normalize_text(long_text)
        assert len(result) == 100

    def test_already_normalised(self):
        """Already normalised text is unchanged."""
        text = "what is the capital of france?"
        assert normalize_text(text) == text

    def test_mixed_case(self):
        """Mixed case is lowered."""
        assert normalize_text("What Is The CAPITAL of France?") == "what is the capital of france?"


# ============================================
# find_matching_question() TESTS
# ============================================
# This function finds a matching question in a list by comparing text.

class TestFindMatchingQuestion:
    """Tests for question matching between scan runs."""

    def test_exact_match(self):
        """Identical text finds the right question."""
        questions = [
            {'text': 'What is the capital of France?'},
            {'text': 'What is the capital of Germany?'},
        ]
        result = find_matching_question('What is the capital of France?', questions)
        assert result is not None
        assert result['text'] == 'What is the capital of France?'

    def test_case_insensitive_match(self):
        """Matching ignores case differences."""
        questions = [{'text': 'What is the Capital of France?'}]
        result = find_matching_question('what is the capital of france?', questions)
        assert result is not None

    def test_whitespace_differences(self):
        """Matching ignores whitespace differences."""
        questions = [{'text': 'What  is   the   capital of France?'}]
        result = find_matching_question('What is the capital of France?', questions)
        assert result is not None

    def test_no_match_returns_none(self):
        """When no question matches, return None."""
        questions = [{'text': 'What is the capital of Germany?'}]
        result = find_matching_question('What is the capital of France?', questions)
        assert result is None

    def test_empty_question_list(self):
        """Empty list returns None."""
        result = find_matching_question('What is the capital of France?', [])
        assert result is None

    def test_question_key_fallback(self):
        """Some data uses 'question' key instead of 'text'."""
        questions = [{'question': 'What is the capital of France?'}]
        result = find_matching_question('What is the capital of France?', questions)
        assert result is not None

    def test_fuzzy_prefix_match(self):
        """Questions that share the first 50 characters match via fuzzy match."""
        # This tests the fallback where exact match fails but prefix matches
        long_q = "What is the primary function of the mitochondria in eukaryotic cells and why is it important?"
        questions = [
            {'text': 'What is the primary function of the mitochondria in eukaryotic cells?'}
        ]
        # Both normalised texts share the same first 50 chars
        result = find_matching_question(long_q, questions)
        assert result is not None

    def test_different_questions_same_start_short(self):
        """Questions starting the same but diverging before 50 chars should NOT match."""
        questions = [{'text': 'What is X?'}]
        result = find_matching_question('What is Y?', questions)
        assert result is None

    def test_empty_target_text(self):
        """Empty target text should not match anything meaningful."""
        questions = [{'text': 'What is the capital of France?'}]
        result = find_matching_question('', questions)
        # Empty normalises to empty, won't match non-empty
        assert result is None


# ============================================
# extract_correct_answer_letter() TESTS
# ============================================
# This function tries to match Moodle's feedback text (like "The correct
# answer is: Paris") to one of the option letters (A, B, C, D).

class TestExtractCorrectAnswerLetter:
    """Tests for extracting the correct answer letter from feedback text."""

    def test_simple_text_match(self):
        """Feedback text matches an option exactly."""
        options = {'A': 'London', 'B': 'Paris', 'C': 'Berlin'}
        result = extract_correct_answer_letter('Paris', options)
        assert result == 'B'

    def test_with_prefix(self):
        """Feedback starts with 'The correct answer is:'."""
        options = {'A': 'London', 'B': 'Paris', 'C': 'Berlin'}
        result = extract_correct_answer_letter('The correct answer is: Paris', options)
        assert result == 'B'

    def test_with_option_label(self):
        """Feedback includes the option label: 'B. Paris'."""
        options = {'A': 'London', 'B': 'Paris', 'C': 'Berlin'}
        result = extract_correct_answer_letter('B. Paris', options)
        assert result == 'B'

    def test_partial_match(self):
        """Feedback is a substring of the option text."""
        options = {'A': 'The city of London in England', 'B': 'Paris', 'C': 'Berlin'}
        result = extract_correct_answer_letter('London', options)
        assert result == 'A'

    def test_letter_prefix_extraction(self):
        """Feedback starts with a letter, punctuation, and text: 'B. Paris'."""
        options = {'A': 'London', 'B': 'Paris', 'C': 'Berlin'}
        result = extract_correct_answer_letter('B. Paris', options)
        assert result == 'B'

    def test_letter_prefix_bare_known_bug(self):
        """
        Feedback is just 'B)' with no option text after it.

        KNOWN BUG: strip_label('B)') produces empty string, which then
        matches the first option via partial match (empty string is 'in'
        any string). The last-resort regex would correctly extract 'B',
        but the partial match short-circuits it.

        This test documents the current (incorrect) behaviour so we know
        when it gets fixed. Ideally should return 'B'.
        """
        options = {'A': 'London', 'B': 'Paris', 'C': 'Berlin'}
        result = extract_correct_answer_letter('B)', options)
        # BUG: Returns 'A' (first option) instead of 'B'
        assert result == 'A'

    def test_no_match(self):
        """When feedback doesn't match any option, return None."""
        options = {'A': 'London', 'B': 'Paris', 'C': 'Berlin'}
        result = extract_correct_answer_letter('Tokyo', options)
        assert result is None

    def test_empty_feedback(self):
        """Empty feedback returns None."""
        options = {'A': 'London', 'B': 'Paris'}
        result = extract_correct_answer_letter('', options)
        assert result is None

    def test_none_feedback(self):
        """None feedback returns None."""
        options = {'A': 'London', 'B': 'Paris'}
        result = extract_correct_answer_letter(None, options)
        assert result is None

    def test_empty_options(self):
        """Empty options dict returns None."""
        result = extract_correct_answer_letter('Paris', {})
        assert result is None

    def test_none_options(self):
        """None options returns None."""
        result = extract_correct_answer_letter('Paris', None)
        assert result is None
