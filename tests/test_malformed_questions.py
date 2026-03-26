"""
Tests for Malformed Question Validation
==========================================

These tests verify that validate_question() gracefully handles
poorly-formatted questions created by instructors:
- Empty/blank options
- Duplicate option text
- Multi-answer questions with only one checkbox

HOW TO RUN:
    pytest tests/test_malformed_questions.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quiz_browser_enhanced import validate_question


class TestValidateQuestion:
    """Tests for validate_question() function."""

    # --- Well-formed questions (no changes) ---

    def test_normal_question_unchanged(self):
        """A well-formed question passes through without changes."""
        options = {"A": "Paris", "B": "London", "C": "Berlin", "D": "Madrid"}
        radios = {"A": "r1", "B": "r2", "C": "r3", "D": "r4"}
        q_type, opts, cb, rb, warnings = validate_question(
            "multichoice_single", options, radio_buttons=radios
        )
        assert q_type == "multichoice_single"
        assert opts == {"A": "Paris", "B": "London", "C": "Berlin", "D": "Madrid"}
        assert warnings == []

    # --- Empty/whitespace options ---

    def test_empty_option_removed(self):
        """Empty string options are removed."""
        options = {"A": "Paris", "B": "", "C": "Berlin"}
        radios = {"A": "r1", "B": "r2", "C": "r3"}
        q_type, opts, cb, rb, warnings = validate_question(
            "multichoice_single", options, radio_buttons=radios
        )
        assert "B" not in opts
        assert "B" not in rb
        assert len(warnings) == 1
        assert "empty" in warnings[0].lower() or "Removed" in warnings[0]

    def test_whitespace_only_option_removed(self):
        """Whitespace-only options are removed."""
        options = {"A": "Paris", "B": "   ", "C": "Berlin"}
        radios = {"A": "r1", "B": "r2", "C": "r3"}
        q_type, opts, cb, rb, warnings = validate_question(
            "multichoice_single", options, radio_buttons=radios
        )
        assert "B" not in opts
        assert len(warnings) == 1

    def test_all_options_empty(self):
        """All options empty → all removed (caller should skip question)."""
        options = {"A": "", "B": "  "}
        radios = {"A": "r1", "B": "r2"}
        q_type, opts, cb, rb, warnings = validate_question(
            "multichoice_single", options, radio_buttons=radios
        )
        assert len(opts) == 0
        assert len(warnings) == 1

    # --- Duplicate option text ---

    def test_duplicate_options_deduped(self):
        """Duplicate option text keeps first occurrence."""
        options = {"A": "True", "B": "True", "C": "False"}
        radios = {"A": "r1", "B": "r2", "C": "r3"}
        q_type, opts, cb, rb, warnings = validate_question(
            "multichoice_single", options, radio_buttons=radios
        )
        assert "A" in opts  # First "True" kept
        assert "B" not in opts  # Duplicate removed
        assert "C" in opts
        assert any("Duplicate" in w for w in warnings)

    def test_duplicate_case_insensitive(self):
        """Duplicates detected case-insensitively."""
        options = {"A": "True", "B": "TRUE", "C": "False"}
        radios = {"A": "r1", "B": "r2", "C": "r3"}
        q_type, opts, cb, rb, warnings = validate_question(
            "multichoice_single", options, radio_buttons=radios
        )
        assert "A" in opts
        assert "B" not in opts
        assert any("Duplicate" in w for w in warnings)

    def test_no_duplicates_when_different(self):
        """Distinct options produce no duplicate warnings."""
        options = {"A": "Paris", "B": "London", "C": "Berlin"}
        q_type, opts, cb, rb, warnings = validate_question(
            "multichoice_single", options
        )
        assert len(opts) == 3
        assert not any("Duplicate" in w for w in warnings)

    # --- Single checkbox → downgrade ---

    def test_single_checkbox_downgraded(self):
        """Multi-answer with only 1 checkbox downgrades to single-answer."""
        options = {"A": "Only option"}
        checkboxes = {"A": "cb1"}
        q_type, opts, cb, rb, warnings = validate_question(
            "multichoice_multi", options, checkbox_buttons=checkboxes
        )
        assert q_type == "multichoice_single"
        assert "A" in rb  # Moved to radio_buttons
        assert len(cb) == 0
        assert any("only 1 option" in w.lower() for w in warnings)

    def test_zero_checkboxes_downgraded(self):
        """Multi-answer with 0 checkboxes downgrades."""
        options = {"A": "Something"}
        q_type, opts, cb, rb, warnings = validate_question(
            "multichoice_multi", options, checkbox_buttons={}
        )
        assert q_type == "multichoice_single"

    def test_multiple_checkboxes_not_downgraded(self):
        """Multi-answer with 2+ checkboxes stays as multi."""
        options = {"A": "Option 1", "B": "Option 2", "C": "Option 3"}
        checkboxes = {"A": "cb1", "B": "cb2", "C": "cb3"}
        q_type, opts, cb, rb, warnings = validate_question(
            "multichoice_multi", options, checkbox_buttons=checkboxes
        )
        assert q_type == "multichoice_multi"
        assert not any("only 1 option" in w.lower() for w in warnings)

    # --- Combined issues ---

    def test_empty_and_duplicate_combined(self):
        """Question with both empty and duplicate options."""
        options = {"A": "Paris", "B": "", "C": "Paris", "D": "London"}
        radios = {"A": "r1", "B": "r2", "C": "r3", "D": "r4"}
        q_type, opts, cb, rb, warnings = validate_question(
            "multichoice_single", options, radio_buttons=radios
        )
        assert "B" not in opts  # Empty removed
        assert "C" not in opts  # Duplicate removed
        assert "A" in opts  # Original kept
        assert "D" in opts
        assert len(warnings) == 2  # One for empty, one for duplicate

    # --- True/False ---

    def test_truefalse_valid(self):
        """Valid True/False question passes through."""
        options = {"A": "True", "B": "False"}
        radios = {"A": "r1", "B": "r2"}
        q_type, opts, cb, rb, warnings = validate_question(
            "truefalse", options, radio_buttons=radios
        )
        assert q_type == "truefalse"
        assert len(opts) == 2
        assert warnings == []
