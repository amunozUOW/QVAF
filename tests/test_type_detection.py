"""
Tests for Moodle Question Type Detection
==========================================

These tests verify that detect_question_type() correctly identifies
question types from CSS classes and input element types, including
edge cases and fallback behavior.

Uses mock data (no live Moodle needed).

HOW TO RUN:
    pytest tests/test_type_detection.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quiz_browser_enhanced import detect_question_type


class TestDetectQuestionType:
    """Tests for detect_question_type() function."""

    # --- Standard detection ---

    def test_multichoice_with_radios(self):
        """multichoice class + radio buttons → multichoice_single"""
        result = detect_question_type("que multichoice deferredfeedback", has_radios=True)
        assert result == "multichoice_single"

    def test_multichoice_with_checkboxes(self):
        """multichoice class + checkbox inputs → multichoice_multi"""
        result = detect_question_type("que multichoice deferredfeedback", has_checkboxes=True)
        assert result == "multichoice_multi"

    def test_truefalse(self):
        """truefalse class → truefalse"""
        result = detect_question_type("que truefalse deferredfeedback", has_radios=True)
        assert result == "truefalse"

    def test_truefalse_without_radios(self):
        """truefalse class even without input info → truefalse"""
        result = detect_question_type("que truefalse notyetanswered")
        assert result == "truefalse"

    # --- Fallback behavior ---

    def test_unknown_class_with_radios_best_guess(self):
        """Unknown CSS class but has radio buttons → best-guess as multichoice_single"""
        result = detect_question_type("que shortanswer deferredfeedback", has_radios=True)
        assert result == "multichoice_single"

    def test_unknown_class_no_inputs(self):
        """Unknown CSS class with no inputs → unknown"""
        result = detect_question_type("que shortanswer deferredfeedback")
        assert result == "unknown"

    def test_empty_classes(self):
        """Empty CSS classes with no inputs → unknown"""
        result = detect_question_type("")
        assert result == "unknown"

    def test_none_classes(self):
        """None CSS classes → unknown"""
        result = detect_question_type(None)
        assert result == "unknown"

    # --- Secondary signal: prompt text ---

    def test_select_one_or_more_with_checkboxes(self):
        """'Select one or more:' prompt + checkboxes → multichoice_multi (even without multichoice class)"""
        result = detect_question_type(
            "que somethingelse",
            has_checkboxes=True,
            prompt_text="Select one or more:"
        )
        assert result == "multichoice_multi"

    def test_select_one_or_more_without_checkboxes(self):
        """'Select one or more:' prompt but no checkboxes → falls through to other detection"""
        result = detect_question_type(
            "que somethingelse",
            has_radios=True,
            prompt_text="Select one or more:"
        )
        # Has radios → best-guess as multichoice_single
        assert result == "multichoice_single"

    # --- Multichoice class takes precedence over prompt text ---

    def test_multichoice_class_checkboxes_no_prompt(self):
        """multichoice class + checkboxes but no prompt text → still multichoice_multi"""
        result = detect_question_type("que multichoice", has_checkboxes=True, prompt_text="")
        assert result == "multichoice_multi"

    def test_multichoice_class_radios_with_multi_prompt(self):
        """multichoice class + radios + 'Select one or more:' → trust input type (single)"""
        result = detect_question_type(
            "que multichoice",
            has_radios=True,
            prompt_text="Select one or more:"
        )
        # CSS class multichoice + radios → multichoice_single (trust input type)
        assert result == "multichoice_single"

    # --- Both input types present ---

    def test_both_checkboxes_and_radios(self):
        """If both checkbox and radio detected, checkboxes win for multichoice."""
        result = detect_question_type("que multichoice", has_checkboxes=True, has_radios=True)
        assert result == "multichoice_multi"

    # --- Case insensitivity ---

    def test_uppercase_classes(self):
        """CSS classes with mixed case are handled."""
        result = detect_question_type("QUE Multichoice DeferredFeedback", has_radios=True)
        assert result == "multichoice_single"

    def test_truefalse_mixed_case(self):
        """TrueFalse in mixed case."""
        result = detect_question_type("que TrueFalse", has_radios=True)
        assert result == "truefalse"
