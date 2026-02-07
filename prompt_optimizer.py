#!/usr/bin/env python3
"""
Prompt Optimizer for Quiz Vulnerability Scanner
================================================

This script helps you systematically test and optimize prompts for MCQ answering.
The goal is to maximize AI performance to ensure thorough vulnerability testing.

REVISED: Removed hardcoded domain-specific questions. Test questions should be
provided via JSON file or configured by the user.

Usage:
    python3 prompt_optimizer.py --questions questions.json      # Test with custom questions
    python3 prompt_optimizer.py --prompt baseline --questions q.json
    python3 prompt_optimizer.py --compare --questions q.json

Question JSON format:
[
    {
        "id": 1,
        "question": "Your question text here",
        "options": {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D", "E": "Option E"},
        "correct": "B",
        "domain": "your_domain",
        "cognitive_level": "recall|comprehension|application|analysis"
    },
    ...
]
"""

import argparse
import json
import re
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    import ollama
except ImportError:
    print("ERROR: ollama not installed. Run: pip install ollama")
    sys.exit(1)

try:
    from config import DEFAULT_MODEL
except ImportError:
    DEFAULT_MODEL = "llama3:8b"


# ============================================
# EXAMPLE QUESTIONS (Discipline-Agnostic)
# ============================================
# These are generic examples showing the expected format.
# For actual testing, provide your own questions via --questions flag.

EXAMPLE_QUESTIONS = [
    {
        "id": 1,
        "question": "This is a placeholder question. Please provide your own test questions via --questions flag.",
        "options": {
            "A": "Option A",
            "B": "Option B (correct for this example)",
            "C": "Option C",
            "D": "Option D",
            "E": "Option E"
        },
        "correct": "B",
        "domain": "example",
        "cognitive_level": "recall",
        "explanation": "This is just an example format. Provide real questions for testing."
    }
]


def load_questions_from_file(filepath: str) -> List[Dict]:
    """Load test questions from a JSON file"""
    try:
        with open(filepath, 'r') as f:
            questions = json.load(f)
        
        # Validate format
        required_fields = ['id', 'question', 'options', 'correct']
        for i, q in enumerate(questions):
            for field in required_fields:
                if field not in q:
                    raise ValueError(f"Question {i+1} missing required field: {field}")
            
            if q['correct'] not in q['options']:
                raise ValueError(f"Question {i+1}: correct answer '{q['correct']}' not in options")
        
        print(f"Loaded {len(questions)} questions from {filepath}")
        return questions
    
    except FileNotFoundError:
        print(f"ERROR: Questions file not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {filepath}: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: Invalid question format: {e}")
        sys.exit(1)


# ============================================
# PROMPT STRATEGIES
# ============================================

def get_prompt_strategies() -> Dict[str, callable]:
    """Return all available prompt strategies"""
    return {
        'baseline': prompt_baseline,
        'chain_of_thought': prompt_chain_of_thought,
        'elimination': prompt_elimination,
        'expert_role': prompt_expert_role,
        'structured_analysis': prompt_structured_analysis,
        'domain_primed': prompt_domain_primed,
        'confidence_calibrated': prompt_confidence_calibrated,
        'step_by_step': prompt_step_by_step,
    }


def prompt_baseline(question: str, options: Dict[str, str], context: str = "") -> str:
    """Current baseline prompt - minimal structure"""
    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    
    return f"""Answer this multiple choice question.

QUESTION: {question}

OPTIONS:
{options_text}
{context}
First, state your answer as a single letter. Then explain briefly.

Format your response EXACTLY like this:
ANSWER: X
CONFIDENCE: N
REASONING: Your explanation here

where X is the letter (A, B, C, D, or E) and N is a number from 0 to 100.

Your response:"""


def prompt_chain_of_thought(question: str, options: Dict[str, str], context: str = "") -> str:
    """Chain of thought - reason before answering"""
    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    
    return f"""You are answering a multiple choice question. Think through this step by step.

QUESTION: {question}

OPTIONS:
{options_text}
{context}
Let's work through this carefully:

1. First, identify what the question is asking
2. Consider what you know about this topic
3. Evaluate each option against your knowledge
4. Select the best answer

After your reasoning, provide your final answer in this format:
ANSWER: [single letter]
CONFIDENCE: [0-100]
REASONING: [brief summary of why]"""


def prompt_elimination(question: str, options: Dict[str, str], context: str = "") -> str:
    """Process of elimination approach"""
    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    
    return f"""Answer this multiple choice question using process of elimination.

QUESTION: {question}

OPTIONS:
{options_text}
{context}
For each option, briefly state whether it's likely correct or can be eliminated and why:

A: [evaluate]
B: [evaluate]
C: [evaluate]
D: [evaluate]
E: [evaluate]

Based on this analysis:
ANSWER: [single letter]
CONFIDENCE: [0-100]
REASONING: [why this is the best choice]"""


def prompt_expert_role(question: str, options: Dict[str, str], context: str = "") -> str:
    """Expert role-play prompt - DISCIPLINE-AGNOSTIC VERSION"""
    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    
    # NOTE: Changed from domain-specific "operations management professor" to generic expert
    return f"""You are a knowledgeable expert in this subject area with extensive experience. A student has asked you to help with this exam question.

QUESTION: {question}

OPTIONS:
{options_text}
{context}
As an expert in this field, analyze this question and provide the correct answer. Draw on your deep knowledge of the relevant concepts, frameworks, and best practices.

ANSWER: [single letter A-E]
CONFIDENCE: [0-100 based on your certainty]
REASONING: [expert explanation]"""


def prompt_structured_analysis(question: str, options: Dict[str, str], context: str = "") -> str:
    """Highly structured analysis prompt"""
    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    
    return f"""Analyze this multiple choice question systematically.

QUESTION: {question}

OPTIONS:
{options_text}
{context}
ANALYSIS FRAMEWORK:

1. QUESTION TYPE: [recall/comprehension/application/analysis]

2. KEY CONCEPT: What topic/concept is being tested?

3. OPTION ANALYSIS:
   A: [Why correct or incorrect]
   B: [Why correct or incorrect]
   C: [Why correct or incorrect]
   D: [Why correct or incorrect]
   E: [Why correct or incorrect]

4. DECISION: Based on analysis, select the best answer.

ANSWER: [letter]
CONFIDENCE: [0-100]
REASONING: [summary]"""


def prompt_domain_primed(question: str, options: Dict[str, str], context: str = "") -> str:
    """Domain-primed prompt - now discipline-agnostic"""
    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    
    # NOTE: Removed domain-specific priming; now generic
    return f"""You are answering an academic multiple choice question from a university course.

QUESTION: {question}

OPTIONS:
{options_text}
{context}
Apply your knowledge of relevant concepts, theories, and principles to answer this question correctly.

Think carefully about:
- What specific concept is being tested
- How each option relates to established knowledge
- Which answer best reflects academic understanding

ANSWER: [letter A-E]
CONFIDENCE: [0-100]
REASONING: [explanation based on relevant concepts]"""


def prompt_confidence_calibrated(question: str, options: Dict[str, str], context: str = "") -> str:
    """Prompt encouraging calibrated confidence"""
    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    
    return f"""Answer this question and provide a well-calibrated confidence score.

QUESTION: {question}

OPTIONS:
{options_text}
{context}
CONFIDENCE GUIDELINES:
- 90-100: You are certain this is correct (would bet on it)
- 70-89: Highly likely correct, but some uncertainty
- 50-69: More likely correct than not, but significant uncertainty
- 30-49: Uncertain, making an educated guess
- 0-29: Very uncertain, essentially guessing

Provide your answer with an honest assessment of your confidence:

ANSWER: [letter]
CONFIDENCE: [0-100, calibrated honestly]
REASONING: [why you chose this answer and this confidence level]"""


def prompt_step_by_step(question: str, options: Dict[str, str], context: str = "") -> str:
    """Explicit step-by-step prompt"""
    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    
    return f"""Answer this question by following these steps exactly.

QUESTION: {question}

OPTIONS:
{options_text}
{context}
STEP 1 - UNDERSTAND: What is the question asking? (one sentence)

STEP 2 - RECALL: What relevant knowledge do I have? (key facts)

STEP 3 - EVALUATE: Which options match my knowledge?

STEP 4 - DECIDE: Which is the BEST answer?

ANSWER: [letter]
CONFIDENCE: [0-100]
REASONING: [brief explanation]"""


# ============================================
# TESTING FUNCTIONS
# ============================================

def call_llm(prompt: str, model: str = None) -> Tuple[str, int, str]:
    """Call LLM and parse response"""
    model = model or DEFAULT_MODEL
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': 0,
                'num_predict': 512,
            }
        )
        
        text = response['message']['content']
        
        # Parse response
        answer_match = re.search(r'ANSWER:\s*([A-Za-z])', text)
        confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', text)
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\n\n|\Z)', text, re.DOTALL)
        
        answer = answer_match.group(1).upper() if answer_match else "?"
        confidence = int(confidence_match.group(1)) if confidence_match else 50
        reasoning = reasoning_match.group(1).strip() if reasoning_match else text[:200]
        
        return answer, confidence, reasoning
        
    except Exception as e:
        return "?", 0, f"Error: {e}"


def test_prompt_strategy(
    strategy_name: str,
    strategy_fn: callable,
    questions: List[Dict],
    model: str = None,
    verbose: bool = True
) -> Dict:
    """Test a prompt strategy against all questions"""
    model = model or DEFAULT_MODEL

    results = {
        'strategy': strategy_name,
        'model': model,
        'timestamp': datetime.now().isoformat(),
        'total': len(questions),
        'correct': 0,
        'incorrect': 0,
        'questions': []
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing: {strategy_name}")
        print(f"Model: {model}")
        print(f"{'='*60}")
    
    for q in questions:
        prompt = strategy_fn(q['question'], q['options'])
        answer, confidence, reasoning = call_llm(prompt, model)
        
        is_correct = answer == q['correct']
        if is_correct:
            results['correct'] += 1
        else:
            results['incorrect'] += 1
        
        result = {
            'id': q['id'],
            'question': q['question'][:50] + '...',
            'correct_answer': q['correct'],
            'llm_answer': answer,
            'is_correct': is_correct,
            'confidence': confidence,
            'reasoning': reasoning[:100] + '...' if len(reasoning) > 100 else reasoning
        }
        results['questions'].append(result)
        
        if verbose:
            status = "✓" if is_correct else "✗"
            print(f"\nQ{q['id']}: {status} (answered {answer}, correct {q['correct']}, conf {confidence}%)")
            if not is_correct:
                print(f"   Expected: {q['correct']}. {q['options'][q['correct']]}")
                print(f"   Got: {answer}. {q['options'].get(answer, 'Invalid')}")
    
    results['accuracy'] = round(results['correct'] / results['total'] * 100, 1)
    
    if verbose:
        print(f"\n{'-'*40}")
        print(f"ACCURACY: {results['correct']}/{results['total']} ({results['accuracy']}%)")
    
    return results


def compare_all_strategies(
    questions: List[Dict],
    model: str = None,
    strategies: Optional[List[str]] = None
) -> List[Dict]:
    """Compare all prompt strategies"""
    model = model or DEFAULT_MODEL
    
    all_strategies = get_prompt_strategies()
    
    if strategies:
        all_strategies = {k: v for k, v in all_strategies.items() if k in strategies}
    
    results = []
    
    print("\n" + "="*70)
    print("PROMPT STRATEGY COMPARISON")
    print("="*70)
    
    for name, fn in all_strategies.items():
        result = test_prompt_strategy(name, fn, questions, model, verbose=False)
        results.append(result)
        print(f"{name:<25} {result['correct']}/{result['total']} ({result['accuracy']}%)")
    
    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\n" + "-"*70)
    print("RANKING (Best to Worst):")
    print("-"*70)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['strategy']:<25} {r['accuracy']}%")
    
    return results


def detailed_comparison(questions: List[Dict], model: str = None):
    """Show detailed side-by-side comparison for each question"""
    model = model or DEFAULT_MODEL

    strategies = get_prompt_strategies()
    
    print("\n" + "="*70)
    print("DETAILED QUESTION-BY-QUESTION COMPARISON")
    print("="*70)
    
    for q in questions:
        print(f"\n{'='*70}")
        print(f"Q{q['id']}: {q['question'][:60]}...")
        print(f"Correct Answer: {q['correct']}. {q['options'][q['correct']]}")
        print("-"*70)
        
        for name, fn in strategies.items():
            prompt = fn(q['question'], q['options'])
            answer, confidence, _ = call_llm(prompt, model)
            
            status = "✓" if answer == q['correct'] else "✗"
            print(f"  {name:<25} {status} {answer} (conf: {confidence}%)")


def create_sample_questions_file():
    """Create a sample questions JSON file as a template"""
    sample = [
        {
            "id": 1,
            "question": "Replace this with your actual question text.",
            "options": {
                "A": "First option",
                "B": "Second option (mark as correct if this is right)",
                "C": "Third option",
                "D": "Fourth option",
                "E": "Fifth option"
            },
            "correct": "B",
            "domain": "your_subject_area",
            "cognitive_level": "recall"
        },
        {
            "id": 2,
            "question": "Add more questions following this format.",
            "options": {
                "A": "Option A",
                "B": "Option B",
                "C": "Option C",
                "D": "Option D",
                "E": "Option E"
            },
            "correct": "C",
            "domain": "your_subject_area",
            "cognitive_level": "application"
        }
    ]
    
    filepath = "sample_questions.json"
    with open(filepath, 'w') as f:
        json.dump(sample, f, indent=2)
    
    print(f"Created sample questions file: {filepath}")
    print("Edit this file with your own questions, then run:")
    print(f"  python3 prompt_optimizer.py --questions {filepath}")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test and optimize prompts for MCQ answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 prompt_optimizer.py --questions myquestions.json   # Test with your questions
  python3 prompt_optimizer.py --prompt baseline --questions q.json
  python3 prompt_optimizer.py --detailed --questions q.json
  python3 prompt_optimizer.py --create-sample               # Create template file

NOTE: You must provide your own test questions via --questions flag.
The tool no longer includes hardcoded domain-specific questions.
        """
    )
    
    parser.add_argument('--questions', type=str, 
                       help='JSON file containing test questions (REQUIRED for testing)')
    parser.add_argument('--prompt', type=str, 
                       help='Test specific prompt strategy')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                       help=f'Ollama model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed question-by-question comparison')
    parser.add_argument('--list', action='store_true',
                       help='List available prompt strategies')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create a sample questions JSON file as template')
    
    args = parser.parse_args()
    
    strategies = get_prompt_strategies()
    
    if args.list:
        print("\nAvailable prompt strategies:")
        for name in strategies.keys():
            print(f"  - {name}")
        sys.exit(0)
    
    if args.create_sample:
        create_sample_questions_file()
        sys.exit(0)
    
    # Check if questions file provided
    if not args.questions:
        print("\nERROR: You must provide a questions file via --questions flag.")
        print("\nTo create a sample template, run:")
        print("  python3 prompt_optimizer.py --create-sample")
        print("\nThen edit the sample file and run:")
        print("  python3 prompt_optimizer.py --questions sample_questions.json")
        sys.exit(1)
    
    # Load questions
    questions = load_questions_from_file(args.questions)
    
    if args.prompt:
        if args.prompt not in strategies:
            print(f"Unknown strategy: {args.prompt}")
            print(f"Available: {', '.join(strategies.keys())}")
            sys.exit(1)
        
        test_prompt_strategy(
            args.prompt,
            strategies[args.prompt],
            questions,
            args.model,
            verbose=True
        )
    
    elif args.detailed:
        detailed_comparison(questions, args.model)
    
    else:
        results = compare_all_strategies(questions, args.model)
        
        # Save results
        output_file = f"prompt_comparison_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")