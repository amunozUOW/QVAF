"""
Tests for Option Label Stripping
==================================

WHAT THESE TESTS DO:
    Quiz options in Moodle come with various label prefixes:
    - "A. Paris" (letter with period)
    - "1) Paris" (number with parenthesis)
    - "iii. Paris" (roman numeral)
    - "B] Paris" (letter with bracket)

    When comparing the AI's answer to the correct answer, we need to strip
    these labels off to get just the option text. The strip_label() function
    handles this with a regex pattern.

    These tests verify that:
    - All common label formats are stripped correctly
    - Actual content that LOOKS like a label is NOT stripped
      (e.g., "IV therapy is..." should NOT become "therapy is...")
    - Text with no label is returned unchanged

WHY THESE TESTS MATTER:
    If label stripping is wrong, the correct answer extraction fails.
    For example, if "IV. The Roman Empire" has "IV. " stripped, that's
    correct. But if "IV therapy" has "IV" stripped, the answer matching
    breaks because the text no longer matches the option.

HOW TO RUN:
    pytest tests/test_option_parsing.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from merge_attempts import strip_label


# ============================================
# LETTER LABEL TESTS
# ============================================

class TestStripLetterLabels:
    """Tests for stripping letter-based option labels (A., B), C] etc.)."""

    def test_letter_period(self):
        """'A. Paris' -> 'Paris'."""
        assert strip_label("A. Paris") == "Paris"

    def test_letter_parenthesis(self):
        """'B) London' -> 'London'."""
        assert strip_label("B) London") == "London"

    def test_letter_bracket(self):
        """'C] Berlin' -> 'Berlin'."""
        assert strip_label("C] Berlin") == "Berlin"

    def test_lowercase_letter(self):
        """'a. Paris' -> 'Paris'."""
        assert strip_label("a. Paris") == "Paris"

    def test_letter_with_extra_space(self):
        """'A.  Paris' (extra space) -> 'Paris'."""
        assert strip_label("A.  Paris") == "Paris"

    def test_all_common_letters(self):
        """Letters A through E all work."""
        for letter in "ABCDE":
            assert strip_label(f"{letter}. Test") == "Test"


# ============================================
# NUMBER LABEL TESTS
# ============================================

class TestStripNumberLabels:
    """Tests for stripping number-based option labels (1., 2), etc.)."""

    def test_single_digit_period(self):
        """'1. Paris' -> 'Paris'."""
        assert strip_label("1. Paris") == "Paris"

    def test_single_digit_parenthesis(self):
        """'2) London' -> 'London'."""
        assert strip_label("2) London") == "London"

    def test_double_digit(self):
        """'12. Answer text' -> 'Answer text'."""
        assert strip_label("12. Answer text") == "Answer text"


# ============================================
# ROMAN NUMERAL LABEL TESTS
# ============================================

class TestStripRomanNumeralLabels:
    """Tests for stripping roman numeral labels (I., ii., IV., etc.)."""

    def test_uppercase_roman_i(self):
        """'I. First option' -> 'First option'."""
        assert strip_label("I. First option") == "First option"

    def test_uppercase_roman_ii(self):
        """'II. Second option' -> 'Second option'."""
        assert strip_label("II. Second option") == "Second option"

    def test_uppercase_roman_iii(self):
        """'III. Third option' -> 'Third option'."""
        assert strip_label("III. Third option") == "Third option"

    def test_uppercase_roman_iv(self):
        """'IV. Fourth option' -> 'Fourth option'."""
        assert strip_label("IV. Fourth option") == "Fourth option"

    def test_lowercase_roman(self):
        """'iv. fourth option' -> 'fourth option'."""
        assert strip_label("iv. fourth option") == "fourth option"

    def test_roman_v(self):
        """'V. Fifth option' -> 'Fifth option'."""
        assert strip_label("V. Fifth option") == "Fifth option"


# ============================================
# EDGE CASE TESTS
# ============================================

class TestStripLabelEdgeCases:
    """Tests for edge cases in label stripping."""

    def test_no_label(self):
        """Text with no label is returned unchanged."""
        assert strip_label("Paris is the capital") == "Paris is the capital"

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert strip_label("") == ""

    def test_just_a_letter(self):
        """Single letter without punctuation should NOT be stripped."""
        # "A" alone doesn't match the pattern (needs . ) or ] after it)
        assert strip_label("A") == "A"

    def test_content_starting_with_i(self):
        """
        'I think this is correct' should NOT strip the 'I'.
        The 'I' here is a word, not a roman numeral label. The regex
        requires punctuation (. ) ]) after the letter, so this should
        be safe.
        """
        assert strip_label("I think this is correct") == "I think this is correct"

    def test_whitespace_only(self):
        """Whitespace-only input returns empty string after strip."""
        assert strip_label("   ") == ""

    def test_newline_prefix_stripped(self):
        """Leading newlines after label removal are cleaned."""
        result = strip_label("A.\nParis")
        assert result == "Paris"

    def test_label_with_no_content(self):
        """Label with nothing after it."""
        result = strip_label("A. ")
        assert result == ""

    def test_iv_therapy_not_stripped(self):
        """
        CRITICAL: 'IV therapy is important' should NOT be treated as a
        roman numeral label. 'IV' here means 'intravenous', not 'four'.

        The regex requires punctuation after the numeral, so 'IV therapy'
        (no . or )) should be safe.
        """
        assert strip_label("IV therapy is important") == "IV therapy is important"

    def test_iv_with_period_is_stripped(self):
        """
        'IV. The Roman Empire' - here IV IS a label and should be stripped.
        The period after IV distinguishes this from 'IV therapy'.
        """
        assert strip_label("IV. The Roman Empire") == "The Roman Empire"
