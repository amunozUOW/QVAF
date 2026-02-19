#!/usr/bin/env python3
"""
Reform Agent - AI Performance Analysis
=======================================
REVISED: Removed categorical vulnerability labels (HIGH/MODERATE/LOW).
Now provides OBJECTIVE metrics and DESCRIPTIVE patterns only.
SME interprets metrics and makes all classification decisions.

This module:
- Classifies question cognitive demand
- Calculates objective metrics (correctness, consistency)
- Records correctness patterns (descriptive, not judgmental)
- Generates analysis for SME review
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import ollama
except ImportError:
    print("ERROR: ollama not installed. Run: pip install ollama")
    sys.exit(1)

# Import config for output paths and default model
try:
    from config import REPORTS_DIR, DEFAULT_MODEL
except ImportError:
    REPORTS_DIR = Path(".")
    DEFAULT_MODEL = "llama3:8b"

# Global model variable - can be set via command line
ANALYSIS_MODEL = DEFAULT_MODEL


# ============================================
# TEQSA GUIDANCE CONTEXT
# ============================================

teqsa_guidance = """
TEQSA 2025 ASSESSMENT REFORM GUIDANCE

Key principle: "Design assessments where gen AI use is irrelevant to the 
demonstration of learning outcomes. This means designing questions that require higher-order thinking, application, and synthesis, rather than simple recall or pattern recognition. This also means suggesting rewrites to questions that either lower LLM confidence in the answer or consistently drive the LLM to select an incorrect answer. It is imperative that the suggested rewrites do not lead to questions that are nonsensical or intentionally select incorrect answers." 

This analysis helps educators understand AI performance patterns on their 
questions. The SME interprets these patterns and decides what action to take.
"""


# ============================================
# COGNITIVE DEMAND TAXONOMY
# ============================================

question_type_system = """
COGNITIVE DEMAND CLASSIFICATION SYSTEM

Classify the question into ONE of these categories based on what type of
thinking is PRIMARILY required. Choose the LOWEST category that fully
describes the cognitive demand — do NOT over-classify.

This taxonomy synthesises Bloom's Revised Taxonomy (Anderson & Krathwohl,
2001), Webb's Depth of Knowledge (Webb, 1997), and SOLO Taxonomy (Biggs &
Collis, 1982).

==========================================================================
1. RECALL
==========================================================================
Definition: Direct retrieval of memorised facts, definitions, terms, or
previously learned information from long-term memory. The response requires
recognition or reproduction of known content with no transformation,
interpretation, or application of the knowledge. The student need only
access a single, discrete piece of information.

Taxonomy alignment:
  Bloom's: Remember (recognise, recall) on Factual Knowledge
  Webb's DOK: Level 1 — Recall & Reproduction
  SOLO: Unistructural

Key verbs: define, identify, list, name, state, label, recognise, recall,
match, select, locate, enumerate

Critical test: If the question can be answered by retrieving a SINGLE fact,
name, date, definition, or category label from memory — with no scenario
interpretation required — it is RECALL. Focus on the QUESTION STEM, not
the answer options. Complex or plausible distractors among the options do
NOT increase the cognitive demand of the question itself. A question that
asks "What is [term]?" or "Which of the following defines [concept]?" is
RECALL even if the options include nuanced or overlapping definitions.

Examples:
  "What is the definition of [term]?"
  "What is [concept]?" (selecting the correct definition from options)
  "In what year did [event] occur?"
  "What does [acronym] stand for?"
  "Which [item] has the highest [property]?" (when answerable from a
   single memorised fact)
  "Name the [components/stages/parts] of [concept]."

==========================================================================
2. ROUTINE APPLICATION
==========================================================================
Definition: Applying a known procedure, definition, or classification
scheme to a specific scenario or instance. The student must recognise
which established concept, category, or formula applies to a described
situation and carry out the matching or computation. The procedure is
standard and well-rehearsed; the primary cognitive demand is selecting and
executing the correct known approach, not judging between approaches or
reasoning about why.

Taxonomy alignment:
  Bloom's: Understand (classify, exemplify) / Apply (execute, implement)
           on Factual and Procedural Knowledge
  Webb's DOK: Level 2 — Skills & Concepts
  SOLO: Multistructural

Key verbs: apply, calculate, classify, categorise, compute, demonstrate,
determine, execute, implement, solve (using known method), use, organise,
interpret (simple data), sort

Boundary — what makes it Level 2, not Level 1:
  The question presents a scenario or context that must be interpreted and
  matched to a concept. Level 1 asks about the concept directly.
  IMPORTANT: If the question stem asks "What is [term]?" or "Define [term]"
  or "Which of the following is [concept]?", it is asking about the concept
  directly and is therefore Level 1 (RECALL), regardless of how complex or
  nuanced the answer options are. Level 2 requires a described SITUATION
  that the student must interpret — not simply choosing a definition.

Boundary — what makes it Level 2, not Level 3:
  The student does NOT need to explain WHY the concept works or how
  concepts relate. The question asks WHAT (category/label/result), not
  WHY or HOW concepts interrelate.

Critical test: If the student can answer correctly by (a) recognising a
pattern they have seen before and (b) applying a known formula or
classification, without needing to explain underlying reasoning or compare
alternatives, it is ROUTINE APPLICATION.

Examples:
  "This scenario describes which type of [category]?"
  "Calculate [X] using [formula/method]."
  "Given these symptoms, which condition is this consistent with?"
  "What will this code snippet output?" (tracing a known procedure)
  "Classify this example as [category A] or [category B]."

==========================================================================
3. CONCEPTUAL UNDERSTANDING
==========================================================================
Definition: Demonstrating understanding of the relationships, mechanisms,
and principles that connect concepts. The student must explain WHY
something works, HOW concepts relate to each other, or WHAT distinguishes
related ideas. This requires going beyond applying labels or procedures to
articulating the underlying logic, comparing and contrasting related ideas,
or explaining cause-and-effect relationships. The student constructs
meaning rather than merely recognising patterns.

Taxonomy alignment:
  Bloom's: Understand (compare, explain, infer) / Analyse (differentiate)
           on Conceptual Knowledge
  Webb's DOK: Level 2-3 boundary
  SOLO: Relational

Key verbs: explain, compare, contrast, distinguish, differentiate, relate,
describe (how/why), infer, interpret (relationships), summarise, connect,
predict (based on principles), generalise

Boundary — what makes it Level 3, not Level 2:
  The question asks WHY or HOW, not just WHAT category. The student must
  explain a mechanism, relationship, or principle — not just apply one.
  Understanding could not be demonstrated by executing a procedure alone.

Boundary — what makes it Level 3, not Level 4:
  The concepts being related are ESTABLISHED and taught. There is generally
  one defensible answer based on established theory. The student is not
  evaluating competing evidence or weighing trade-offs.

Critical test: If the question can be answered by someone who genuinely
understands the concept but has never seen this specific scenario, AND it
CANNOT be answered by someone who merely memorised definitions without
understanding relationships, it is CONCEPTUAL UNDERSTANDING.

Examples:
  "Explain why [X] causes [Y]."
  "What distinguishes [concept A] from [concept B]?"
  "How does an increase in [variable] affect [outcome] and why?"
  "Compare and contrast [idea A] and [idea B]."
  "Why does [phenomenon] occur?"

==========================================================================
4. ANALYTICAL REASONING
==========================================================================
Definition: Breaking down complex, multi-faceted information to draw
conclusions by evaluating competing evidence, weighing trade-offs, or
making judgments based on multiple criteria simultaneously. The student
must go beyond understanding established relationships to performing
novel analysis — integrating data from multiple sources, evaluating
arguments, or making justified decisions where multiple considerations
must be balanced. Stating reasoning and providing supporting evidence
are essential.

Taxonomy alignment:
  Bloom's: Analyse (differentiate, organise, attribute) / Evaluate
           (check, critique, judge) on Conceptual and Procedural Knowledge
  Webb's DOK: Level 3 — Strategic Thinking
  SOLO: Relational to Extended Abstract transition

Key verbs: analyse, evaluate, assess, critique, judge, justify, determine
(given evidence), prioritise, investigate, deduce, conclude (from data),
appraise, defend, argue

Boundary — what makes it Level 4, not Level 3:
  The question presents NOVEL data or a scenario not directly covered in
  the curriculum. Multiple factors must be weighed simultaneously. The
  student must EVALUATE, not just explain. There may be more than one
  defensible answer.

Boundary — what makes it Level 4, not Level 5:
  The analysis operates within a SINGLE domain or problem space. The
  student analyses given information rather than creating something new.

Critical test: If the question provides data or evidence and asks the
student to reach a conclusion that requires weighing multiple factors and
CANNOT be answered by applying a single known principle, it is ANALYTICAL
REASONING.

Examples:
  "Given this data, determine which [option] is most effective and why."
  "Evaluate the trade-offs between [approach A] and [approach B]."
  "What conclusions can you draw from [this evidence]?"
  "Critique the following argument: [argument]."
  "A study reports [finding] but has [limitation]. Assess its validity."

==========================================================================
5. STRATEGIC INTEGRATION
==========================================================================
Definition: Synthesising knowledge from multiple domains, frameworks, or
sources to create, design, or propose a novel solution to a complex,
non-routine problem. The student must integrate conceptual understanding
across different areas, apply knowledge to genuinely new situations that
have no single correct procedure, and produce something that requires
planning, development, and original thought.

Taxonomy alignment:
  Bloom's: Create (generate, plan, produce) / Evaluate at highest level
           on Conceptual, Procedural, and Metacognitive Knowledge
  Webb's DOK: Level 4 — Extended Thinking
  SOLO: Extended Abstract

Key verbs: design, create, synthesise, propose, develop, formulate,
construct, integrate, plan, devise, generate, hypothesise, compose

Boundary — what makes it Level 5, not Level 4:
  The task requires synthesis ACROSS multiple domains or frameworks. The
  student must CREATE or DESIGN something new, not just evaluate given
  options. The problem is genuinely novel and non-routine. Note: this
  level is very difficult to assess in multiple-choice format.

Critical test: If the question cannot be answered by mastering any single
concept or framework, and instead requires pulling together ideas from
multiple areas to produce something that did not previously exist, it is
STRATEGIC INTEGRATION.

Examples:
  "Design a [system/strategy/plan] that addresses [multiple constraints
   from different domains]."
  "Propose a comprehensive approach integrating [framework A] and
   [framework B] to solve [novel problem]."
  "Develop a [plan] for [situation] considering [competing requirements
   across disciplines]."

==========================================================================
BOUNDARY DECISION FLOWCHART — apply these sequential tests:
==========================================================================

1. Can it be answered by retrieving a single fact or definition?
   -> YES = RECALL

2. Does it require matching a scenario to a known category, or executing
   a known procedure?
   -> YES = ROUTINE APPLICATION

3. Does it require explaining WHY or HOW concepts relate, or comparing
   and contrasting ideas?
   -> YES = CONCEPTUAL UNDERSTANDING

4. Does it require evaluating novel evidence, weighing trade-offs, or
   making a justified judgment across multiple criteria?
   -> YES = ANALYTICAL REASONING

5. Does it require synthesising across multiple domains or frameworks to
   create something new?
   -> YES = STRATEGIC INTEGRATION

Always apply the LOWEST level that fully describes the cognitive demand.
"""


# ============================================
# DATA LOADING
# ============================================

def load_quiz_data(filepath):
    """Load merged quiz data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


# ============================================
# CORRECTNESS CHECKING
# ============================================

def check_correct(answer, correct_answer):
    """
    Check if an answer matches the correct answer (supplementary fallback).
    Used only when Moodle's is_correct is not available.
    Handles various formats (letter only, "A. text", etc.)
    """
    if not answer or not correct_answer:
        return None
    if correct_answer == 'UNKNOWN':
        return None

    # Extract just the letter
    answer_letter = answer.strip()[0].upper()
    correct_letter = correct_answer.strip()[0].upper()

    # Direct letter comparison
    if answer_letter and correct_letter:
        return answer_letter == correct_letter

    # Fallback to full string matching
    return answer.upper().strip() == correct_answer.upper().strip()


# ============================================
# CORRECTNESS PATTERNS (DESCRIPTIVE ONLY)
# ============================================

def get_correctness_pattern(correct_without: bool, correct_with: bool) -> str:
    """
    Determine the DESCRIPTIVE correctness pattern based on AI performance.

    IMPORTANT: These are DESCRIPTIONS, not vulnerability judgments.
    The SME interprets what these patterns mean for their context.

    Returns one of:
    - "CORRECT_BOTH": AI correct in both conditions (or just correct in basic mode)
    - "CORRECT_RAG_ONLY": AI correct only with course materials
    - "CORRECT_BASELINE_ONLY": AI correct only without course materials (anomalous)
    - "INCORRECT_BOTH": AI incorrect in both conditions (or just incorrect in basic mode)
    - "UNKNOWN": Unable to determine (both values are None)
    """
    if correct_without is None and correct_with is None:
        return "UNKNOWN"

    if correct_with is None:
        # Basic mode — only baseline data available
        if correct_without:
            return "CORRECT_BOTH"  # Treated as "correct" for basic mode
        elif correct_without is False:
            return "INCORRECT_BOTH"  # Treated as "incorrect" for basic mode
        else:
            return "UNKNOWN"

    if correct_without is None:
        # Edge case — only RAG data available
        if correct_with:
            return "CORRECT_RAG_ONLY"
        else:
            return "INCORRECT_BOTH"

    if correct_without and correct_with:
        return "CORRECT_BOTH"
    elif not correct_without and correct_with:
        return "CORRECT_RAG_ONLY"
    elif correct_without and not correct_with:
        return "CORRECT_BASELINE_ONLY"
    else:
        return "INCORRECT_BOTH"


def get_pattern_description(pattern: str) -> str:
    """
    Return a human-readable description of the correctness pattern.
    These are OBJECTIVE descriptions, not judgments.
    """
    descriptions = {
        "CORRECT_BOTH": "AI answered correctly in both conditions (with and without course materials)",
        "CORRECT_RAG_ONLY": "AI answered correctly only when provided with course materials",
        "CORRECT_BASELINE_ONLY": "AI answered correctly only without course materials",
        "INCORRECT_BOTH": "AI answered incorrectly in both conditions",
        "UNKNOWN": "Unable to determine correctness pattern"
    }
    return descriptions.get(pattern, "Pattern not recognized")


def get_confidence_flag(correct_without, conf_without, correct_with, conf_with):
    """Flag high-confidence incorrect answers (≥90%) for SME attention"""
    flags = []
    
    if correct_without == False and conf_without and conf_without >= 90:
        flags.append(f"High confidence incorrect (baseline): {conf_without}%")
    
    if correct_with == False and conf_with and conf_with >= 90:
        flags.append(f"High confidence incorrect (RAG): {conf_with}%")
    
    return "; ".join(flags) if flags else None


# ============================================
# QUESTION TYPE CLASSIFICATION (via LLM)
# ============================================

def classify_question_type(question_text, options=None):
    """Use LLM to classify question into cognitive demand category"""
    global ANALYSIS_MODEL

    options_text = ""
    if options:
        options_text = "\n\nOPTIONS:\n" + "\n".join([f"{k}. {v}" for k, v in options.items()])

    prompt = f"""{question_type_system}

QUESTION TO CLASSIFY:
{question_text}{options_text}

IMPORTANT: Respond in EXACTLY this format. The category MUST be one of: RECALL, ROUTINE APPLICATION, CONCEPTUAL UNDERSTANDING, ANALYTICAL REASONING, or STRATEGIC INTEGRATION.

CATEGORY: [write the full category name here]
RATIONALE: [one sentence explanation]
KEY COGNITIVE DEMAND: [what the question primarily requires]

Example response:
CATEGORY: CONCEPTUAL UNDERSTANDING
RATIONALE: The question requires explaining the relationship between two concepts.
KEY COGNITIVE DEMAND: Understanding relationships between ideas
"""

    try:
        response = ollama.chat(
            model=ANALYSIS_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0, 'num_predict': 256}
        )
        return response['message']['content']
    except Exception as e:
        return f"CATEGORY: UNKNOWN\nRATIONALE: Classification failed: {e}"


def extract_question_category(classification_text):
    """Extract category name from classification response"""
    text_upper = classification_text.upper()

    # First try to find explicit CATEGORY: format
    match = re.search(r'CATEGORY:\s*(\w+(?:\s+\w+)*)', classification_text, re.IGNORECASE)
    if match:
        category = match.group(1).upper().strip()

        # Reject single letters (likely answer letters like A, B, C, D)
        if len(category) <= 2:
            # Continue to fallback matching below
            pass
        else:
            # Normalize category names
            if 'RECALL' in category:
                return 'RECALL'
            elif 'ROUTINE' in category or 'APPLICATION' in category:
                return 'ROUTINE APPLICATION'
            elif 'CONCEPTUAL' in category:
                return 'CONCEPTUAL UNDERSTANDING'
            elif 'ANALYTICAL' in category or 'ANALYSIS' in category:
                return 'ANALYTICAL REASONING'
            elif 'STRATEGIC' in category or 'INTEGRATION' in category:
                return 'STRATEGIC INTEGRATION'
            return category

    # Fallback: Look for category keywords anywhere in the response
    # Priority order matters - check most specific first
    if 'STRATEGIC INTEGRATION' in text_upper or ('STRATEGIC' in text_upper and 'INTEGRATION' in text_upper):
        return 'STRATEGIC INTEGRATION'
    elif 'ANALYTICAL REASONING' in text_upper or ('ANALYTICAL' in text_upper and 'REASONING' in text_upper):
        return 'ANALYTICAL REASONING'
    elif 'CONCEPTUAL UNDERSTANDING' in text_upper or ('CONCEPTUAL' in text_upper and 'UNDERSTANDING' in text_upper):
        return 'CONCEPTUAL UNDERSTANDING'
    elif 'ROUTINE APPLICATION' in text_upper or ('ROUTINE' in text_upper and 'APPLICATION' in text_upper):
        return 'ROUTINE APPLICATION'
    elif 'RECALL' in text_upper and 'RECALL' not in text_upper.split('RATIONALE')[0] if 'RATIONALE' in text_upper else 'RECALL' in text_upper:
        # Check if RECALL appears meaningfully (not just in rationale context)
        lines = classification_text.split('\n')
        for line in lines[:5]:  # Check first few lines
            if 'RECALL' in line.upper() and not line.upper().startswith('RATIONALE'):
                return 'RECALL'

    # Last resort: check for abbreviated/partial matches
    if 'STRATEGIC' in text_upper:
        return 'STRATEGIC INTEGRATION'
    elif 'ANALYTICAL' in text_upper or 'ANALYSIS' in text_upper:
        return 'ANALYTICAL REASONING'
    elif 'CONCEPTUAL' in text_upper:
        return 'CONCEPTUAL UNDERSTANDING'
    elif 'ROUTINE' in text_upper or ('APPLICATION' in text_upper and 'ROUTINE' not in text_upper):
        return 'ROUTINE APPLICATION'
    elif 'RECALL' in text_upper:
        return 'RECALL'

    return 'UNKNOWN'


# ============================================
# ANALYSIS GENERATION (for SME review)
# ============================================

def analyse_question(question_text, options, correct_answer,
                     answer_without, conf_without, reasoning_without,
                     answer_with, conf_with, reasoning_with,
                     correctness_pattern, question_type, confidence_flag,
                     is_basic_mode=False):
    """
    Generate per-question vulnerability analysis for the educator.

    Acts as an objective third-party advisor: identifies what made the question
    easy or difficult for AI, and recommends concrete changes to reduce AI
    confidence or correctness. The educator decides what action to take.

    is_basic_mode: When True, only baseline (no RAG) data is shown.
    """

    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()]) if options else "N/A"
    pattern_desc = get_pattern_description(correctness_pattern)

    # System message establishing the agent's role and perspective
    system_message = """You are an objective, third-party assessment vulnerability advisor working on behalf of an educator. Your role is to help the educator understand how and why an AI was able (or unable) to answer their quiz questions correctly.

YOUR PERSPECTIVE:
- You are NOT coaching the AI or helping it improve. You are advising the EDUCATOR.
- Your goal is to help the educator identify what made a question easy or difficult for the AI.
- Your recommendations should focus on how to REDUCE AI confidence and REDUCE the likelihood that AI will arrive at the correct answer, so that the assessment better measures genuine student understanding.
- You are unbiased and evidence-driven. You do not advocate for the AI or soften findings about AI capabilities.
- When the AI got a question right, explain what structural or content features allowed it to succeed, and suggest how those features could be altered.
- When the AI got a question wrong, explain what made the question resistant to AI and what properties the educator should preserve or replicate in other questions."""

    if is_basic_mode:
        # Basic mode prompt - no RAG/course materials section
        prompt = f"""{teqsa_guidance}

QUESTION TYPE: {question_type}

QUESTION:
{question_text}

OPTIONS:
{options_text}

CORRECT ANSWER: {correct_answer}

AI PERFORMANCE:
- Answer: {answer_without}
- Confidence: {conf_without}%
- Reasoning: {reasoning_without}

AI RESULT: {"CORRECT" if correctness_pattern in ['CORRECT_BOTH', 'CORRECT_BASELINE_ONLY'] else "INCORRECT"}
{f"ATTENTION: {confidence_flag}" if confidence_flag else ""}

Analyse this question from the educator's perspective. Your job is to help them understand what made this question easy or difficult for the AI, and what they could change to reduce AI performance on it.

1. VULNERABILITY ANALYSIS (2-3 sentences):
   - What specific features of this question (structure, content, option design, cognitive demand) made it easy or difficult for the AI?
   - Did the AI's reasoning reveal genuine understanding, or did it exploit surface-level patterns, common phrasing, or process-of-elimination to arrive at the answer?
   - How does the question type ({question_type}) contribute to or protect against AI success?

2. RECOMMENDATIONS FOR EDUCATOR:
   - What concrete changes could the educator make to this question to reduce AI confidence or reduce the likelihood of AI selecting the correct answer?
   - If the AI already failed on this question, what properties of the question should the educator preserve or replicate in other questions?
   - Focus on actionable, specific suggestions (e.g., requiring application of local/unpublished context, adding plausible distractors, requiring multi-step reasoning, embedding scenario-specific details).

Format your response as:
VULNERABILITY ANALYSIS:
[your analysis]

RECOMMENDATIONS FOR EDUCATOR:
[your recommendations]
"""
    else:
        # Full mode prompt - includes RAG comparison
        prompt = f"""{teqsa_guidance}

QUESTION TYPE: {question_type}

QUESTION:
{question_text}

OPTIONS:
{options_text}

CORRECT ANSWER: {correct_answer}

AI PERFORMANCE (Baseline - without course materials):
- Answer: {answer_without}
- Confidence: {conf_without}%
- Reasoning: {reasoning_without}

AI PERFORMANCE (RAG - with course materials):
- Answer: {answer_with}
- Confidence: {conf_with}%
- Reasoning: {reasoning_with}

CORRECTNESS PATTERN: {correctness_pattern}
({pattern_desc})
{f"ATTENTION: {confidence_flag}" if confidence_flag else ""}

Analyse this question from the educator's perspective. Your job is to help them understand what made this question easy or difficult for the AI, and what they could change to reduce AI performance on it.

1. VULNERABILITY ANALYSIS (2-3 sentences):
   - What specific features of this question (structure, content, option design, cognitive demand) made it easy or difficult for the AI?
   - Did the AI's reasoning reveal genuine understanding, or did it exploit surface-level patterns, common phrasing, or process-of-elimination to arrive at the answer?
   - What does the difference (or lack of difference) between baseline and RAG performance tell the educator about this question's reliance on general knowledge vs. course-specific content?
   - How does the question type ({question_type}) contribute to or protect against AI success?

2. RECOMMENDATIONS FOR EDUCATOR:
   - What concrete changes could the educator make to this question to reduce AI confidence or reduce the likelihood of AI selecting the correct answer?
   - If the AI already failed on this question, what properties of the question should the educator preserve or replicate in other questions?
   - Focus on actionable, specific suggestions (e.g., requiring application of local/unpublished context, adding plausible distractors, requiring multi-step reasoning, embedding scenario-specific details).

Format your response as:
VULNERABILITY ANALYSIS:
[your analysis]

RECOMMENDATIONS FOR EDUCATOR:
[your recommendations]
"""
    
    try:
        global ANALYSIS_MODEL
        response = ollama.chat(
            model=ANALYSIS_MODEL,
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': prompt}
            ],
            options={'temperature': 0, 'num_predict': 1024}
        )
        return response['message']['content']
    except Exception as e:
        return f"Analysis failed: {e}"


# ============================================
# MAIN PROCESSING
# ============================================

def process_quiz(filepath):
    """Process quiz data and generate analysis report"""
    
    print(f"Loading {filepath}...")
    data = load_quiz_data(filepath)
    
    questions = data.get('questions', [])
    print(f"Analysing {len(questions)} questions...")
    
    # Extract actual scores if present (from web app)
    actual_scores = {
        'no_rag_score': data.get('no_rag_score'),
        'with_rag_score': data.get('with_rag_score')
    }

    # Detect basic mode (no RAG scan performed)
    is_basic_mode = actual_scores.get('with_rag_score') is None
    scan_mode = 'basic' if is_basic_mode else 'full'
    print(f"Scan mode: {scan_mode}")

    quantitative_summary = {
        'total_questions': len(questions),
        'correct_without_rag': 0,
        'correct_with_rag': 0,
        'pattern_counts': {
            'CORRECT_BOTH': 0,
            'CORRECT_RAG_ONLY': 0,
            'CORRECT_BASELINE_ONLY': 0,
            'INCORRECT_BOTH': 0,
            'UNKNOWN': 0
        }
    }
    
    results = []
    qualitative_analyses = []
    
    for i, q in enumerate(questions):
        print(f"\n--- Question {i+1}/{len(questions)} ---")
        
        question_text = q.get('question', q.get('text', ''))
        options = q.get('options', {})
        correct_answer = q.get('correct_answer', 'UNKNOWN')
        
        # Get AI answers from the nested response structure
        response_without = q.get('response_without_rag', {})
        response_with = q.get('response_with_rag', {})

        answer_without = response_without.get('answer') if isinstance(response_without, dict) else q.get('answer_without_rag')
        answer_with = response_with.get('answer') if isinstance(response_with, dict) else q.get('answer_with_rag')

        # Get confidence/consistency
        conf_without = response_without.get('confidence', 0) if isinstance(response_without, dict) else q.get('confidence_without_rag', 0)
        conf_with = response_with.get('confidence', 0) if isinstance(response_with, dict) else q.get('confidence_with_rag', 0)

        # Get reasoning
        reasoning_without = response_without.get('reasoning', '') if isinstance(response_without, dict) else q.get('reasoning_without_rag', '')
        reasoning_with = response_with.get('reasoning', '') if isinstance(response_with, dict) else q.get('reasoning_with_rag', '')

        
        # Check correctness — prefer Moodle's authoritative is_correct
        correct_without = q.get('is_correct_without_rag')
        correct_with = q.get('is_correct_with_rag')

        # Fallback to letter comparison only when is_correct unavailable
        if correct_without is None and answer_without and correct_answer != 'UNKNOWN':
            correct_without = check_correct(answer_without, correct_answer)
        if correct_with is None and answer_with and correct_answer != 'UNKNOWN':
            correct_with = check_correct(answer_with, correct_answer)
        
        # Get correctness pattern (DESCRIPTIVE, not judgmental)
        pattern = get_correctness_pattern(correct_without, correct_with)
        quantitative_summary['pattern_counts'][pattern] = \
            quantitative_summary['pattern_counts'].get(pattern, 0) + 1
        
        # Update counters
        if correct_without:
            quantitative_summary['correct_without_rag'] += 1
        if correct_with:
            quantitative_summary['correct_with_rag'] += 1
        
        # Classify question type
        print(f"  Classifying cognitive demand...")
        classification = classify_question_type(question_text, options)
        question_type = extract_question_category(classification)
        
        # Check for confidence flags
        confidence_flag = get_confidence_flag(
            correct_without, conf_without,
            correct_with, conf_with
        )
        
        # Generate analysis for SME
        print(f"  Generating analysis for SME review...")
        analysis = analyse_question(
            question_text, options, correct_answer,
            answer_without, conf_without, reasoning_without,
            answer_with, conf_with, reasoning_with,
            pattern, question_type, confidence_flag,
            is_basic_mode=is_basic_mode
        )
        
        # Store results
        result = {
            'id': q.get('number', i + 1),
            'question': question_text[:100] + '...' if len(question_text) > 100 else question_text,
            'correct_answer': correct_answer,
            'answer_without_rag': answer_without,
            'answer_with_rag': answer_with,
            'correct_without_rag': correct_without,
            'correct_with_rag': correct_with,
            'confidence_without_rag': conf_without,
            'confidence_with_rag': conf_with,
            'consistency_without_rag': q.get('consistency_without_rag'),
            'consistency_with_rag': q.get('consistency_with_rag'),
            'question_type': question_type,
            'correctness_pattern': pattern,  # Renamed from vulnerability_category
            'confidence_flag': confidence_flag
        }
        results.append(result)
        
        qualitative_analyses.append({
            'id': result['id'],
            'question_type': question_type,
            'correctness_pattern': pattern,  # Renamed from vulnerability_category
            'confidence_flag': confidence_flag,
            'analysis': analysis
        })
        
        # Print summary
        pattern_icon = {
            'CORRECT_BOTH': '⚠',      # SME may want to review
            'CORRECT_RAG_ONLY': '◐',   # Course materials helped AI
            'INCORRECT_BOTH': '✓',     # AI struggled
            'CORRECT_BASELINE_ONLY': '?',  # Anomalous
            'UNKNOWN': '-'
        }
        icon = pattern_icon.get(pattern, '-')
        print(f"  Type: {question_type}")
        print(f"  Pattern: {pattern} {icon}")
        if confidence_flag:
            print(f"  Attention: {confidence_flag}")
    
    # Build report
    report = {
        'metadata': {
            'source_file': filepath,
            'generated_at': datetime.now().isoformat(),
            'framework_version': '2.0',
            'note': 'Metrics are OBJECTIVE; SME interprets and makes decisions'
        },
        'scan_mode': scan_mode,  # 'basic' or 'full'
        'actual_scores': actual_scores,
        'quantitative_summary': quantitative_summary,
        'question_results': results,
        'qualitative_analyses': qualitative_analyses
    }
    
    # Save report - keep in same directory as input file
    input_path = Path(filepath)
    output_file = input_path.parent / (input_path.stem + '_analysis_report.json')
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Scan mode: {scan_mode}")
    print(f"Total questions: {quantitative_summary['total_questions']}")
    print(f"\nAI Performance:")
    print(f"  AI accuracy: {quantitative_summary['correct_without_rag']}/{quantitative_summary['total_questions']}")
    if not is_basic_mode:
        print(f"  AI accuracy (with course materials): {quantitative_summary['correct_with_rag']}/{quantitative_summary['total_questions']}")
    print(f"\nCorrectness Results:")
    if is_basic_mode:
        # In basic mode, just show correct/incorrect counts
        correct = quantitative_summary['pattern_counts'].get('CORRECT_BOTH', 0) + quantitative_summary['pattern_counts'].get('CORRECT_BASELINE_ONLY', 0)
        incorrect = quantitative_summary['pattern_counts'].get('INCORRECT_BOTH', 0) + quantitative_summary['pattern_counts'].get('CORRECT_RAG_ONLY', 0)
        print(f"  AI Correct: {correct}")
        print(f"  AI Incorrect: {incorrect}")
    else:
        for pattern, count in quantitative_summary['pattern_counts'].items():
            if count > 0:
                print(f"  {pattern}: {count}")
    print(f"\nReport saved to: {output_file}")
    
    return report


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze quiz data and generate objective metrics for SME review.")
    parser.add_argument('filepath', help='Path to merged quiz data JSON file')
    parser.add_argument('--model', default=DEFAULT_MODEL, help=f'LLM model to use for analysis (default: {DEFAULT_MODEL})')

    args = parser.parse_args()

    # Set the global model
    ANALYSIS_MODEL = args.model
    print(f"Using model: {ANALYSIS_MODEL}")

    process_quiz(args.filepath)