"""
Tests for Rule-Based Stem Classifier
=====================================

WHAT THESE TESTS DO:
    Before the LLM classifies a question, the rule-based pre-classifier
    checks the question stem for obvious linguistic patterns that map to
    specific cognitive demand categories. These tests verify that:

    1. Clear-cut stems ARE caught by the correct category's patterns
    2. Ambiguous stems are NOT caught (they should go to the LLM)
    3. The priority order (highest category first) works correctly
    4. Edge cases (long stems, unusual formatting) are handled safely

WHY THESE TESTS MATTER:
    A false positive (catching a question that should go to the LLM) is
    worse than a false negative (missing a question that could be caught).
    False positives bypass the LLM entirely and produce a potentially
    wrong classification with no opportunity for correction.

HOW TO RUN:
    pytest tests/test_stem_classifier.py -v
"""

import sys
import os

# Add the project root to the Python path so we can import our modules.
# This is needed because tests/ is a subfolder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reform_agent import (
    _prepare_stem,
    _is_recall_by_stem,
    _is_routine_application_by_stem,
    _is_conceptual_understanding_by_stem,
    _is_analytical_reasoning_by_stem,
    _is_strategic_integration_by_stem,
    MAX_RULE_STEM_LENGTH,
)


# ============================================
# STEM PREPARATION TESTS
# ============================================
# These test the _prepare_stem() helper which normalises question text
# before pattern matching.

class TestStemPreparation:
    """Tests for _prepare_stem() normalisation helper."""

    def test_lowercase(self):
        assert _prepare_stem("WHAT IS INVENTORY?") == "what is inventory?"

    def test_strip_whitespace(self):
        assert _prepare_stem("  What is X?  ") == "what is x?"

    def test_strip_question_number_colon(self):
        assert _prepare_stem("Question 5: What is X?") == "what is x?"

    def test_strip_question_number_dot(self):
        assert _prepare_stem("Question 3. What is X?") == "what is x?"

    def test_strip_q_number_dot(self):
        assert _prepare_stem("Q3. What is X?") == "what is x?"

    def test_strip_q_number_colon(self):
        assert _prepare_stem("Q12: What is X?") == "what is x?"

    def test_no_number_prefix(self):
        assert _prepare_stem("What is X?") == "what is x?"

    def test_empty_string(self):
        assert _prepare_stem("") == ""

    def test_preserves_internal_whitespace(self):
        assert _prepare_stem("What is the relationship between A and B?") == \
            "what is the relationship between a and b?"


# ============================================
# RECALL STEM TESTS
# ============================================
# Tests for _is_recall_by_stem() — protects existing behaviour and
# verifies correct classification of definition/identification stems.

class TestRecallByStem:
    """Tests for _is_recall_by_stem() patterns."""

    # --- TRUE POSITIVES: should match RECALL ---

    def test_what_is_term(self):
        assert _is_recall_by_stem("What is inventory?") is True

    def test_what_are_terms(self):
        assert _is_recall_by_stem("What are the four Ps of marketing?") is True

    def test_define_term(self):
        assert _is_recall_by_stem("Define supply chain management") is True

    def test_which_defines(self):
        assert _is_recall_by_stem("Which of the following defines throughput?") is True

    def test_which_best_defines(self):
        assert _is_recall_by_stem("Which of the following best defines lean manufacturing?") is True

    def test_which_statements_best_defines(self):
        """Filler words like 'statements' between 'following' and 'defines'."""
        assert _is_recall_by_stem("Which of the following statements best defines inventory?") is True

    def test_which_options_defines(self):
        """Filler word 'options' between 'following' and 'defines'."""
        assert _is_recall_by_stem("Which of the following options defines throughput?") is True

    def test_what_does_stand_for(self):
        assert _is_recall_by_stem("What does BOM stand for?") is True

    def test_what_does_mean(self):
        assert _is_recall_by_stem("What does throughput mean?") is True

    def test_term_is_defined_as(self):
        assert _is_recall_by_stem("Inventory is defined as:") is True

    def test_term_refers_to(self):
        assert _is_recall_by_stem("The concept of hierarchy in operations management refers to:") is True

    def test_definition_of(self):
        assert _is_recall_by_stem("The definition of lead time is:") is True

    def test_with_question_number(self):
        assert _is_recall_by_stem("Question 3: What is a bottleneck?") is True

    # --- TRUE NEGATIVES: should NOT match RECALL ---

    def test_scenario_question(self):
        """Long scenario question should not match RECALL."""
        assert _is_recall_by_stem(
            "An ice cream production operation requires a minimum amount of "
            "ice cream to be stored on site to ensure smooth running of the "
            "logistics processes. Additional finished ice cream held on site, "
            "and not sold or allocated to a particular customer represents "
            "which type of waste?"
        ) is False

    def test_explain_why(self):
        assert _is_recall_by_stem("Explain why lean manufacturing reduces waste") is False

    def test_calculate(self):
        assert _is_recall_by_stem("Calculate the throughput using Little's Law") is False

    def test_how_does(self):
        assert _is_recall_by_stem("How does lead time affect customer satisfaction?") is False

    def test_evaluate(self):
        assert _is_recall_by_stem("Evaluate the effectiveness of JIT") is False


# ============================================
# ROUTINE APPLICATION STEM TESTS
# ============================================

class TestRoutineApplicationByStem:
    """Tests for _is_routine_application_by_stem() patterns."""

    # --- TRUE POSITIVES: should match ROUTINE APPLICATION ---

    def test_calculate_using(self):
        assert _is_routine_application_by_stem(
            "Calculate the reorder point using the EOQ formula"
        ) is True

    def test_compute_the(self):
        assert _is_routine_application_by_stem(
            "Compute the net present value"
        ) is True

    def test_calculate_the(self):
        assert _is_routine_application_by_stem(
            "Calculate the labor productivity"
        ) is True

    def test_what_will_code_output(self):
        assert _is_routine_application_by_stem(
            "What will this code output?"
        ) is True

    def test_what_does_code_return(self):
        assert _is_routine_application_by_stem(
            "What does this function return?"
        ) is True

    def test_solve_for(self):
        assert _is_routine_application_by_stem(
            "Solve for x in the equation 3x + 7 = 22"
        ) is True

    def test_solve_the(self):
        assert _is_routine_application_by_stem(
            "Solve the following equation"
        ) is True

    def test_classify_this(self):
        assert _is_routine_application_by_stem(
            "Classify this scenario as a push or pull system"
        ) is True

    def test_classify_the_following(self):
        assert _is_routine_application_by_stem(
            "Classify the following example as batch or flow production"
        ) is True

    def test_using_formula_calculate(self):
        assert _is_routine_application_by_stem(
            "Using the EOQ formula, calculate the optimal order quantity"
        ) is True

    # --- TRUE NEGATIVES: should NOT match ROUTINE APPLICATION ---

    def test_what_is_term(self):
        """Definition question should NOT match Routine Application."""
        assert _is_routine_application_by_stem("What is inventory?") is False

    def test_explain_why(self):
        """Conceptual question should NOT match Routine Application."""
        assert _is_routine_application_by_stem(
            "Explain why batch production increases inventory"
        ) is False

    def test_given_determine_ambiguous(self):
        """Ambiguous 'given...determine' should go to LLM."""
        assert _is_routine_application_by_stem(
            "Given the following financial data, determine which investment "
            "provides the best risk-adjusted return"
        ) is False

    def test_long_scenario_with_calculate(self):
        """Long narrative with 'calculate' buried in it should go to LLM."""
        long_stem = (
            "A manufacturing company has been operating at 85% capacity for "
            "three years. Last quarter, they invested $2M in new equipment "
            "and hired 50 additional workers. The production manager wants "
            "to know the new utilization rate. Calculate the utilization "
            "considering all the factors described above and explain your "
            "reasoning for each step of the analysis."
        )
        # This exceeds MAX_RULE_STEM_LENGTH, so it should go to LLM
        assert len(long_stem.lower()) > MAX_RULE_STEM_LENGTH
        assert _is_routine_application_by_stem(long_stem) is False

    def test_evaluate_not_routine(self):
        """Analytical question should NOT match Routine Application."""
        assert _is_routine_application_by_stem(
            "Evaluate the effectiveness of just-in-time inventory"
        ) is False


# ============================================
# CONCEPTUAL UNDERSTANDING STEM TESTS
# ============================================

class TestConceptualUnderstandingByStem:
    """Tests for _is_conceptual_understanding_by_stem() patterns."""

    # --- TRUE POSITIVES: should match CONCEPTUAL UNDERSTANDING ---

    def test_explain_why(self):
        assert _is_conceptual_understanding_by_stem(
            "Explain why lean manufacturing reduces waste"
        ) is True

    def test_why_does(self):
        assert _is_conceptual_understanding_by_stem(
            "Why does increasing batch size increase inventory levels?"
        ) is True

    def test_why_is(self):
        assert _is_conceptual_understanding_by_stem(
            "Why is quality important in supply chain management?"
        ) is True

    def test_why_are(self):
        assert _is_conceptual_understanding_by_stem(
            "Why are safety stocks necessary in supply chains?"
        ) is True

    def test_how_does_affect(self):
        assert _is_conceptual_understanding_by_stem(
            "How does lead time reduction affect customer satisfaction?"
        ) is True

    def test_how_does_work(self):
        assert _is_conceptual_understanding_by_stem(
            "How does a kanban system work?"
        ) is True

    def test_compare_and_contrast(self):
        assert _is_conceptual_understanding_by_stem(
            "Compare and contrast push and pull production systems"
        ) is True

    def test_what_distinguishes(self):
        assert _is_conceptual_understanding_by_stem(
            "What distinguishes a process layout from a product layout?"
        ) is True

    def test_what_is_relationship(self):
        assert _is_conceptual_understanding_by_stem(
            "What is the relationship between quality and productivity?"
        ) is True

    def test_what_is_difference(self):
        assert _is_conceptual_understanding_by_stem(
            "What is the difference between efficiency and effectiveness?"
        ) is True

    def test_what_is_distinction(self):
        assert _is_conceptual_understanding_by_stem(
            "What is the distinction between strategy and tactics?"
        ) is True

    def test_explain_how(self):
        assert _is_conceptual_understanding_by_stem(
            "Explain how Total Quality Management improves processes"
        ) is True

    # --- TRUE NEGATIVES: should NOT match CONCEPTUAL UNDERSTANDING ---

    def test_how_many(self):
        """'How many' is computational, not conceptual."""
        assert _is_conceptual_understanding_by_stem(
            "How many units did the factory produce last quarter?"
        ) is False

    def test_how_much(self):
        """'How much' is computational, not conceptual."""
        assert _is_conceptual_understanding_by_stem(
            "How much inventory should the company hold?"
        ) is False

    def test_what_is_term(self):
        """Definition question should NOT match Conceptual Understanding."""
        assert _is_conceptual_understanding_by_stem(
            "What is inventory?"
        ) is False

    def test_calculate(self):
        """Computational question should NOT match Conceptual Understanding."""
        assert _is_conceptual_understanding_by_stem(
            "Calculate the throughput using Little's Law"
        ) is False

    def test_what_is_without_relationship(self):
        """Plain 'what is' (not 'what is the relationship') should NOT match."""
        assert _is_conceptual_understanding_by_stem(
            "What is the bullwhip effect?"
        ) is False

    def test_long_scenario(self):
        """Long scenario should go to LLM even with conceptual verb."""
        long_stem = (
            "A global supply chain company has been experiencing delays in "
            "multiple regions due to weather events, port congestion, and "
            "political instability. Their customers are increasingly "
            "dissatisfied. The board has asked for an explanation. "
            "Explain why the company's supply chain is vulnerable to "
            "these disruptions and how the bullwhip effect amplifies "
            "the impact across their distribution network."
        )
        assert len(long_stem.lower()) > MAX_RULE_STEM_LENGTH
        assert _is_conceptual_understanding_by_stem(long_stem) is False


# ============================================
# ANALYTICAL REASONING STEM TESTS
# ============================================

class TestAnalyticalReasoningByStem:
    """Tests for _is_analytical_reasoning_by_stem() patterns."""

    # --- TRUE POSITIVES: should match ANALYTICAL REASONING ---

    def test_evaluate_effectiveness(self):
        assert _is_analytical_reasoning_by_stem(
            "Evaluate the effectiveness of just-in-time inventory management"
        ) is True

    def test_evaluate_validity(self):
        assert _is_analytical_reasoning_by_stem(
            "Evaluate the validity of the hypothesis"
        ) is True

    def test_evaluate_trade_offs(self):
        assert _is_analytical_reasoning_by_stem(
            "Evaluate the trade-offs between cost and quality"
        ) is True

    def test_evaluate_argument(self):
        assert _is_analytical_reasoning_by_stem(
            "Evaluate the argument that automation always reduces costs"
        ) is True

    def test_assess_validity(self):
        assert _is_analytical_reasoning_by_stem(
            "Assess the validity of the study's conclusions"
        ) is True

    def test_assess_effectiveness(self):
        assert _is_analytical_reasoning_by_stem(
            "Assess the effectiveness of the new quality control process"
        ) is True

    def test_critique_argument(self):
        assert _is_analytical_reasoning_by_stem(
            "Critique the following argument about outsourcing"
        ) is True

    def test_critique_approach(self):
        assert _is_analytical_reasoning_by_stem(
            "Critique the approach taken to reduce lead times"
        ) is True

    def test_what_conclusions(self):
        assert _is_analytical_reasoning_by_stem(
            "What conclusions can you draw from the productivity data?"
        ) is True

    def test_what_conclusion_singular(self):
        assert _is_analytical_reasoning_by_stem(
            "What conclusion can be drawn from this evidence?"
        ) is True

    def test_judge_which(self):
        assert _is_analytical_reasoning_by_stem(
            "Judge which strategy is most appropriate for this situation"
        ) is True

    def test_analyse_to_determine(self):
        assert _is_analytical_reasoning_by_stem(
            "Analyse the sales data to determine the seasonal trends"
        ) is True

    def test_analyze_american_spelling(self):
        assert _is_analytical_reasoning_by_stem(
            "Analyze the production data to determine the bottleneck"
        ) is True

    # --- TRUE NEGATIVES: should NOT match ANALYTICAL REASONING ---

    def test_evaluate_mathematical(self):
        """Mathematical 'evaluate' should NOT match."""
        assert _is_analytical_reasoning_by_stem(
            "Evaluate: 3x + 7 = 22. What is x?"
        ) is False

    def test_evaluate_without_analytical_noun(self):
        """'Evaluate' without analytical noun should NOT match."""
        assert _is_analytical_reasoning_by_stem(
            "Evaluate the following expression: 2 + 3 * 4"
        ) is False

    def test_assess_simple_classification(self):
        """Simple 'assess which' should NOT match."""
        assert _is_analytical_reasoning_by_stem(
            "Assess which of these is a renewable resource"
        ) is False

    def test_analyse_without_to_determine(self):
        """'Analyse' without 'to determine' should NOT match."""
        assert _is_analytical_reasoning_by_stem(
            "Analyse the word 'photosynthesis' into its Greek roots"
        ) is False

    def test_what_is_term(self):
        """Definition question should NOT match Analytical Reasoning."""
        assert _is_analytical_reasoning_by_stem(
            "What is critical path analysis?"
        ) is False

    def test_explain_why(self):
        """Conceptual question should NOT match Analytical Reasoning."""
        assert _is_analytical_reasoning_by_stem(
            "Explain why lean manufacturing reduces waste"
        ) is False

    def test_long_scenario(self):
        """Long scenario should go to LLM."""
        long_stem = (
            "A manufacturing company has been operating at 85% capacity for "
            "three years. Last quarter, they invested $2M in new equipment "
            "and hired 50 additional workers. The production manager reports "
            "that defect rates have increased by 15% while throughput has "
            "only improved by 5%. The CEO is concerned about the return on "
            "investment. Evaluate the effectiveness of the expansion."
        )
        assert len(long_stem.lower()) > MAX_RULE_STEM_LENGTH
        assert _is_analytical_reasoning_by_stem(long_stem) is False


# ============================================
# STRATEGIC INTEGRATION STEM TESTS
# ============================================

class TestStrategicIntegrationByStem:
    """Tests for _is_strategic_integration_by_stem() patterns."""

    # --- TRUE POSITIVES: should match STRATEGIC INTEGRATION ---

    def test_design_that_integrates(self):
        assert _is_strategic_integration_by_stem(
            "Design a supply chain strategy that integrates lean principles "
            "and sustainability requirements"
        ) is True

    def test_design_that_combines(self):
        assert _is_strategic_integration_by_stem(
            "Design a production system that combines JIT and Six Sigma"
        ) is True

    def test_design_that_addresses(self):
        assert _is_strategic_integration_by_stem(
            "Design a framework that addresses cost, quality, and delivery"
        ) is True

    def test_propose_comprehensive(self):
        assert _is_strategic_integration_by_stem(
            "Propose a comprehensive approach to improving supply chain resilience"
        ) is True

    def test_propose_integrated(self):
        assert _is_strategic_integration_by_stem(
            "Propose an integrated strategy for quality and efficiency"
        ) is True

    def test_develop_considering_competing(self):
        assert _is_strategic_integration_by_stem(
            "Develop a plan for the company considering competing "
            "stakeholder requirements"
        ) is True

    def test_develop_considering_multiple(self):
        assert _is_strategic_integration_by_stem(
            "Develop a strategy considering multiple constraints from "
            "different departments"
        ) is True

    def test_synthesise_to_create(self):
        assert _is_strategic_integration_by_stem(
            "Synthesise lean and agile principles to create a hybrid "
            "production framework"
        ) is True

    def test_synthesize_american(self):
        assert _is_strategic_integration_by_stem(
            "Synthesize findings from marketing and operations to design "
            "an optimal product launch strategy"
        ) is True

    # --- TRUE NEGATIVES: should NOT match STRATEGIC INTEGRATION ---

    def test_design_simple_experiment(self):
        """Single-domain design should NOT match."""
        assert _is_strategic_integration_by_stem(
            "Design an experiment to test whether temperature affects yield"
        ) is False

    def test_create_chart(self):
        """Simple creation task should NOT match."""
        assert _is_strategic_integration_by_stem(
            "Create a bar chart showing quarterly sales"
        ) is False

    def test_propose_without_comprehensive(self):
        """Plain 'propose' without integration language should NOT match."""
        assert _is_strategic_integration_by_stem(
            "Propose a hypothesis for why defect rates increased"
        ) is False

    def test_develop_without_constraints(self):
        """'Develop' without multi-constraint language should NOT match."""
        assert _is_strategic_integration_by_stem(
            "Develop a hypothesis about inventory levels"
        ) is False

    def test_what_is_term(self):
        """Definition question should NOT match Strategic Integration."""
        assert _is_strategic_integration_by_stem(
            "What is strategic integration?"
        ) is False

    def test_evaluate(self):
        """Analytical question should NOT match Strategic Integration."""
        assert _is_strategic_integration_by_stem(
            "Evaluate the effectiveness of the integration strategy"
        ) is False

    def test_explain_why(self):
        """Conceptual question should NOT match Strategic Integration."""
        assert _is_strategic_integration_by_stem(
            "Explain why cross-functional integration improves performance"
        ) is False


# ============================================
# PRIORITY ORDER TESTS
# ============================================
# These verify that the checking order (highest first) produces the
# correct classification. Only uses stems that ARE caught by the rule-based
# classifier (no LLM dependency).

class TestPriorityOrder:
    """Tests verifying that higher categories take precedence."""

    def test_recall_not_caught_by_higher(self):
        """'What is X?' should only match RECALL, not higher categories."""
        assert _is_recall_by_stem("What is inventory?") is True
        assert _is_routine_application_by_stem("What is inventory?") is False
        assert _is_conceptual_understanding_by_stem("What is inventory?") is False
        assert _is_analytical_reasoning_by_stem("What is inventory?") is False
        assert _is_strategic_integration_by_stem("What is inventory?") is False

    def test_conceptual_not_caught_by_recall(self):
        """'Explain why X' should match Conceptual, not Recall."""
        stem = "Explain why lean manufacturing reduces waste"
        assert _is_conceptual_understanding_by_stem(stem) is True
        assert _is_recall_by_stem(stem) is False
        assert _is_routine_application_by_stem(stem) is False

    def test_analytical_not_caught_by_lower(self):
        """'Evaluate the effectiveness' should match Analytical, not lower."""
        stem = "Evaluate the effectiveness of JIT inventory management"
        assert _is_analytical_reasoning_by_stem(stem) is True
        assert _is_recall_by_stem(stem) is False
        assert _is_routine_application_by_stem(stem) is False
        assert _is_conceptual_understanding_by_stem(stem) is False

    def test_routine_not_caught_by_recall(self):
        """'Calculate X using Y' should match Routine, not Recall."""
        stem = "Calculate the reorder point using the EOQ formula"
        assert _is_routine_application_by_stem(stem) is True
        assert _is_recall_by_stem(stem) is False

    def test_strategic_not_caught_by_lower(self):
        """Strategic Integration stems should not match any lower category."""
        stem = (
            "Design a supply chain strategy that integrates lean principles "
            "and sustainability requirements"
        )
        assert _is_strategic_integration_by_stem(stem) is True
        assert _is_analytical_reasoning_by_stem(stem) is False
        assert _is_conceptual_understanding_by_stem(stem) is False
        assert _is_routine_application_by_stem(stem) is False
        assert _is_recall_by_stem(stem) is False


# ============================================
# MAX STEM LENGTH TESTS
# ============================================
# Verify that the long-stem escape hatch works correctly for non-RECALL
# categories. RECALL is exempt because its patterns are self-limiting.

class TestMaxStemLength:
    """Tests for MAX_RULE_STEM_LENGTH escape hatch."""

    def test_max_length_constant_exists(self):
        """Verify the constant is set to a reasonable value."""
        assert MAX_RULE_STEM_LENGTH == 250

    def test_short_stem_matches(self):
        """Short stems should still be matched by rules."""
        assert _is_conceptual_understanding_by_stem("Why does X cause Y?") is True

    def test_long_routine_stem_escapes(self):
        """Long Routine Application stem should escape to LLM."""
        long_stem = "Calculate the " + "very " * 50 + "important metric"
        assert len(long_stem) > MAX_RULE_STEM_LENGTH
        assert _is_routine_application_by_stem(long_stem) is False

    def test_long_conceptual_stem_escapes(self):
        """Long Conceptual Understanding stem should escape to LLM."""
        long_stem = "Explain why the " + "extremely " * 40 + "complex phenomenon occurs"
        assert len(long_stem) > MAX_RULE_STEM_LENGTH
        assert _is_conceptual_understanding_by_stem(long_stem) is False

    def test_long_analytical_stem_escapes(self):
        """Long Analytical Reasoning stem should escape to LLM."""
        long_stem = "Evaluate the effectiveness of the " + "comprehensive " * 30 + "strategy"
        assert len(long_stem) > MAX_RULE_STEM_LENGTH
        assert _is_analytical_reasoning_by_stem(long_stem) is False

    def test_long_strategic_stem_escapes(self):
        """Long Strategic Integration stem should escape to LLM."""
        long_stem = "Design a " + "multi-layered " * 30 + "strategy that integrates multiple domains"
        assert len(long_stem) > MAX_RULE_STEM_LENGTH
        assert _is_strategic_integration_by_stem(long_stem) is False

    def test_recall_exempt_from_length_check(self):
        """RECALL patterns are anchored and self-limiting, no length check needed."""
        # "What is X?" is short by nature; the patterns themselves limit length
        assert _is_recall_by_stem("What is inventory?") is True
