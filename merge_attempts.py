#!/usr/bin/env python3
"""
Merge Quiz Attempts
===================

Combines two quiz attempts (one without RAG, one with RAG) into a single
JSON file formatted for the reform agent and analysis pipeline.

Questions are matched by their TEXT CONTENT (not by number) to handle
cases where question numbers weren't captured correctly.

USAGE:
------
python3 merge_attempts.py <no_rag_attempt.json> <with_rag_attempt.json>

OUTPUT:
-------
quiz_attempt_YYYYMMDD_HHMMSS.json (compatible with reform_agent.py)
"""

import json
import sys
import re
from datetime import datetime


def load_attempt(filepath):
    """Load a quiz attempt JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def normalize_text(text):
    """Normalize text for comparison (lowercase, remove extra whitespace)"""
    if not text:
        return ""
    # Remove extra whitespace, lowercase
    text = re.sub(r'\s+', ' ', text.lower().strip())
    # Take first 100 chars for comparison (enough to be unique)
    return text[:100]


def find_matching_question(target_text, questions_list):
    """Find a question in the list that matches the target text"""
    target_norm = normalize_text(target_text)
    
    for q in questions_list:
        q_text = q.get('text') or q.get('question') or ''
        if normalize_text(q_text) == target_norm:
            return q
    
    # Fuzzy match - check if target starts with same words
    for q in questions_list:
        q_text = q.get('text') or q.get('question') or ''
        q_norm = normalize_text(q_text)
        # Check if first 50 chars match
        if target_norm[:50] == q_norm[:50]:
            return q
    
    return None


def extract_correct_answer_letter(correct_answer_text, options):
    """
    Try to match the correct answer text to an option letter.
    """
    if not correct_answer_text or not options:
        return None
    
    # Clean up the correct answer text
    correct_text = correct_answer_text
    
    # Remove common prefixes like "The correct answer is:"
    correct_text = re.sub(r'^The correct answer is:?\s*', '', correct_text, flags=re.IGNORECASE)
    correct_text = correct_text.strip()
    
    correct_lower = correct_text.lower()
    
    # Try exact match
    for letter, option_text in options.items():
        if option_text.lower().strip() == correct_lower:
            return letter
    
    # Try partial match (option contains correct answer or vice versa)
    for letter, option_text in options.items():
        opt_lower = option_text.lower().strip()
        # Check if one contains the other (for multi-line answers)
        if correct_lower in opt_lower or opt_lower in correct_lower:
            return letter
        # Check first 50 chars
        if correct_lower[:50] == opt_lower[:50]:
            return letter
    
    # Check if correct_answer_text starts with a letter
    letter_match = re.match(r'^([A-Za-z])[\.\)\:]', correct_answer_text)
    if letter_match:
        return letter_match.group(1).upper()
    
    return None


def merge_attempts(no_rag_data, with_rag_data):
    """
    Merge two attempts by matching questions based on text content.
    """
    
    no_rag_questions = no_rag_data.get('questions', [])
    with_rag_questions = with_rag_data.get('questions', [])
    
    no_rag_results = no_rag_data.get('results', [])
    with_rag_results = with_rag_data.get('results', [])
    
    print(f"No-RAG attempt: {len(no_rag_questions)} questions, {len(no_rag_results)} results")
    print(f"With-RAG attempt: {len(with_rag_questions)} questions, {len(with_rag_results)} results")
    
    merged_questions = []
    question_id = 0
    
    # Use no_rag questions as the base (they typically have better structure)
    for no_rag_q in no_rag_questions:
        q_text = no_rag_q.get('text') or no_rag_q.get('question') or ''
        options = no_rag_q.get('options', {})
        
        # Skip info blocks (questions with no options)
        if not options:
            print(f"  Skipping info block: {q_text[:50]}...")
            continue
        
        question_id += 1
        print(f"\nProcessing Q{question_id}: {q_text[:50]}...")
        
        # Find matching with_rag question by text
        with_rag_q = find_matching_question(q_text, with_rag_questions)
        
        if with_rag_q:
            print(f"  ✓ Found matching with-RAG question")
        else:
            print(f"  ✗ No matching with-RAG question found")
        
        # Find matching results by text
        no_rag_result = find_matching_question(q_text, no_rag_results)
        with_rag_result = find_matching_question(q_text, with_rag_results)
        
        # Determine correct answer
        correct_answer = "UNKNOWN"
        
        if no_rag_result and no_rag_result.get('correct_answer'):
            correct_answer = extract_correct_answer_letter(
                no_rag_result['correct_answer'], options
            ) or "UNKNOWN"
        
        if correct_answer == "UNKNOWN" and with_rag_result and with_rag_result.get('correct_answer'):
            correct_answer = extract_correct_answer_letter(
                with_rag_result['correct_answer'], options
            ) or "UNKNOWN"
        
        print(f"  Correct answer: {correct_answer}")
        
        # Build merged question
        merged_q = {
            "id": question_id,
            "question": q_text,
            "options": options,
            "correct_answer": correct_answer,
            "response_without_rag": {
                "answer": no_rag_q.get('llm_answer', ''),
                "confidence": no_rag_q.get('llm_confidence', 0),
                "consistency": no_rag_q.get('llm_consistency'),
                "consistency_count": no_rag_q.get('llm_consistency_count'),
                "consistency_pct": no_rag_q.get('llm_consistency_pct'),
                "distribution": no_rag_q.get('llm_distribution'),
                "reasoning": no_rag_q.get('llm_reasoning', ''),
                "search_terms": "",
                "retrieved_context": ""
            },
            "response_with_rag": {
                "answer": with_rag_q.get('llm_answer', '') if with_rag_q else '',
                "confidence": with_rag_q.get('llm_confidence', 0) if with_rag_q else 0,
                "consistency": with_rag_q.get('llm_consistency') if with_rag_q else None,
                "consistency_count": with_rag_q.get('llm_consistency_count') if with_rag_q else None,
                "consistency_pct": with_rag_q.get('llm_consistency_pct') if with_rag_q else None,
                "distribution": with_rag_q.get('llm_distribution') if with_rag_q else None,
                "reasoning": with_rag_q.get('llm_reasoning', '') if with_rag_q else '',
                "search_terms": "",
                "retrieved_context": ""
            }
        }
        
        print(f"  No-RAG answer: {merged_q['response_without_rag']['answer']}")
        print(f"  With-RAG answer: {merged_q['response_with_rag']['answer']}")
        
        merged_questions.append(merged_q)
    
    # Build final merged data
    merged = {
        "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
        "source": "moodle_browser_automation",
        "no_rag_score": no_rag_data.get('score', {}),
        "with_rag_score": with_rag_data.get('score', {}),
        "num_samples": no_rag_data.get('num_samples', 1),
        "avg_consistency_no_rag": no_rag_data.get('avg_consistency'),
        "avg_consistency_with_rag": with_rag_data.get('avg_consistency'),
        "questions": merged_questions
    }
    
    return merged


def print_summary(merged_data):
    """Print a summary of the merged data"""
    print("\n" + "="*60)
    print("MERGE SUMMARY")
    print("="*60)
    
    questions = merged_data.get('questions', [])
    print(f"\nTotal questions merged: {len(questions)}")
    
    # Score comparison
    no_rag_score = merged_data.get('no_rag_score', {})
    with_rag_score = merged_data.get('with_rag_score', {})
    
    if no_rag_score:
        print(f"\nWithout RAG: {no_rag_score.get('correct', '?')}/{no_rag_score.get('total', '?')} ({no_rag_score.get('percentage', '?')}%)")
    if with_rag_score:
        print(f"With RAG:    {with_rag_score.get('correct', '?')}/{with_rag_score.get('total', '?')} ({with_rag_score.get('percentage', '?')}%)")
    
    # Preview vulnerability categories
    print("\nQuestion Summary:")
    for q in questions:
        no_rag_ans = q['response_without_rag']['answer']
        with_rag_ans = q['response_with_rag']['answer']
        correct = q['correct_answer']
        
        no_rag_correct = no_rag_ans.upper() == correct.upper() if correct != "UNKNOWN" else "?"
        with_rag_correct = with_rag_ans.upper() == correct.upper() if correct != "UNKNOWN" else "?"
        
        print(f"  Q{q['id']}: No-RAG={no_rag_ans} ({'✓' if no_rag_correct == True else '✗' if no_rag_correct == False else '?'}), "
              f"With-RAG={with_rag_ans} ({'✓' if with_rag_correct == True else '✗' if with_rag_correct == False else '?'}), "
              f"Correct={correct}")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 merge_attempts.py <no_rag_attempt.json> <with_rag_attempt.json>")
        print("\nExample:")
        print("  python3 merge_attempts.py quiz_attempt_20251227_100000_no_rag.json quiz_attempt_20251227_100500_with_rag.json")
        sys.exit(1)
    
    no_rag_file = sys.argv[1]
    with_rag_file = sys.argv[2]
    
    print(f"Loading {no_rag_file}...")
    no_rag_data = load_attempt(no_rag_file)
    
    print(f"Loading {with_rag_file}...")
    with_rag_data = load_attempt(with_rag_file)
    
    print("\n" + "-"*60)
    print("MERGING ATTEMPTS")
    print("-"*60)
    
    merged = merge_attempts(no_rag_data, with_rag_data)
    
    # Print summary
    print_summary(merged)
    
    # Save output
    output_file = f"quiz_attempt_{merged['timestamp']}.json"
    
    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Merged file saved to: {output_file}")
    print(f"\nNext step:")
    print(f"  python3 reform_agent.py {output_file}")
    print("="*60)