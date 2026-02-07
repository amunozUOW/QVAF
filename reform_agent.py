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
thinking is PRIMARILY required:

1. RECALL
   - Direct retrieval of memorised facts, definitions, or procedures
   - Recognition of previously learned information
   - No transformation of knowledge required
   - Examples: "Define X", "What is the name of Y", "List the components of Z"

2. ROUTINE APPLICATION  
   - Applying a known procedure where method selection is obvious
   - Following established steps to reach a solution
   - Minimal judgment required about which approach to use
   - Examples: "Calculate X using formula Y", "Apply the standard procedure to..."

3. CONCEPTUAL UNDERSTANDING
   - Demonstrating understanding of relationships between concepts
   - Explaining mechanisms, causes, or effects
   - Comparing or contrasting related ideas
   - Examples: "Explain why X causes Y", "How does A relate to B"

4. ANALYTICAL REASONING
   - Breaking down complex information into components
   - Evaluating evidence to draw conclusions
   - Making judgments based on criteria
   - Examples: "Analyze the data to determine...", "Evaluate the argument for..."

5. STRATEGIC INTEGRATION
   - Synthesizing information from multiple sources
   - Applying knowledge to novel, non-routine situations
   - Creating solutions that require integration of multiple concepts
   - Examples: "Design a solution for...", "How would you address this new scenario..."
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
    Check if an answer matches the correct answer.
    Handles various formats (letter only, "A. text", etc.)
    """
    if not answer or not correct_answer:
        return None
    
    # Extract just the letter
    answer_letter = answer.strip()[0].upper() if answer else ""
    correct_letter = correct_answer.strip()[0].upper() if correct_answer else ""
    
    # Direct letter comparison
    if answer_letter and correct_letter:
        return answer_letter == correct_letter
    
    # Fallback to full string matching
    if not answer or not correct_answer or correct_answer == 'UNKNOWN':
        return None
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
    - "CORRECT_BOTH": AI correct in both conditions
    - "CORRECT_RAG_ONLY": AI correct only with course materials
    - "CORRECT_BASELINE_ONLY": AI correct only without course materials (anomalous)
    - "INCORRECT_BOTH": AI incorrect in both conditions
    - "UNKNOWN": Unable to determine
    """
    if correct_without is None or correct_with is None:
        return "UNKNOWN"
    
    if correct_without and correct_with:
        return "CORRECT_BOTH"
    elif not correct_without and correct_with:
        return "CORRECT_RAG_ONLY"
    elif not correct_without and not correct_with:
        return "INCORRECT_BOTH"
    elif correct_without and not correct_with:
        return "CORRECT_BASELINE_ONLY"
    
    return "UNKNOWN"


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
        response = ollama.chat(model=ANALYSIS_MODEL, messages=[
            {'role': 'user', 'content': prompt}
        ])
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
    Generate analysis for SME review.

    NOTE: This generates INFORMATIONAL analysis, not prescriptive recommendations.
    The SME decides what action to take based on this information.

    is_basic_mode: When True, only baseline (no RAG) data is shown.
    """

    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()]) if options else "N/A"
    pattern_desc = get_pattern_description(correctness_pattern)

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

Based on this information, provide:

1. PERFORMANCE ANALYSIS (2-3 sentences):
   - What does the AI performance suggest about this question?
   - Was the reasoning used by the AI sufficient to arrive at the correct answer, or did it rely on patterns that may not reflect true understanding?
   - How might the question type ({question_type}) relate to AI performance?

2. CONSIDERATIONS FOR SME:
   - What factors should the educator consider when reviewing this question?
   - Note: Do NOT make prescriptive recommendations. Present options neutrally.

Format your response as:
PERFORMANCE ANALYSIS:
[your analysis]

CONSIDERATIONS FOR SME:
[factors to consider]
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

Based on this information, provide:

1. PERFORMANCE ANALYSIS (2-3 sentences):
   - What does the AI performance pattern suggest about this question?
   - Was the reasoning used by the AI sufficient to arrive at the correct answer, or did it rely on patterns that may not reflect true understanding?
   - How might the question type ({question_type}) relate to AI performance?

2. CONSIDERATIONS FOR SME:
   - What factors should the educator consider when reviewing this question?
   - Note: Do NOT make prescriptive recommendations. Present options neutrally.

Format your response as:
PERFORMANCE ANALYSIS:
[your analysis]

CONSIDERATIONS FOR SME:
[factors to consider]
"""
    
    try:
        global ANALYSIS_MODEL
        response = ollama.chat(model=ANALYSIS_MODEL, messages=[
            {'role': 'user', 'content': prompt}
        ])
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

        
        # Check correctness
        correct_without = check_correct(answer_without, correct_answer)
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