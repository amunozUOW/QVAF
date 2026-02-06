#!/usr/bin/env python3
"""
Multi-Sample LLM Answering Module
=================================

Implements multi-sample consistency for measuring AI confidence.
Instead of asking the AI "how confident are you?", we run the same
question multiple times and measure how consistently it gives the same answer.

Usage:
    from multi_sample_llm import ask_question_multi_sample
    
    result = ask_question_multi_sample(
        question="What is the capital of France?",
        options={'A': 'London', 'B': 'Paris', 'C': 'Berlin', 'D': 'Madrid'},
        model='llama3:8b',
        num_samples=10,
        rag_context=""  # Optional course materials
    )
    
    print(result['answer'])        # 'B'
    print(result['consistency'])   # '9/10'
    print(result['distribution'])  # {'B': 9, 'C': 1}
"""

import re
from collections import Counter
from typing import Optional, Callable

try:
    import ollama
except ImportError:
    ollama = None


def ask_question_single(
    question: str,
    options_text: str,
    model: str,
    rag_context: str = "",
    temperature: float = 0.7
) -> tuple:
    """
    Ask a single question once and return (answer, reasoning).
    """
    prompt = f"""Answer this multiple choice question.

QUESTION: {question}

OPTIONS:
{options_text}
{rag_context}
Think through this step-by-step:
1. What is this question testing?
2. Evaluate each option.
3. Select the best answer.

After your analysis, write your final answer as:
ANSWER: [ONE letter: A, B, C, D, or E]
REASONING: [One sentence explaining why]"""

    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': temperature,
                'num_predict': 400,
            }
        )
        
        text = response['message']['content']
        
        # Parse answer
        answer_match = re.search(r'ANSWER:\s*([A-Ea-e])', text)
        if answer_match:
            answer = answer_match.group(1).upper()
        else:
            # Fallback parsing
            alt_patterns = [
                r'(?:answer|select|choose)[:\s]*([A-Ea-e])\b',
                r'\b([A-E])\s*(?:is correct|is the (?:best|correct))',
                r'FINAL.*?([A-E])\b',
            ]
            answer = None
            for pattern in alt_patterns:
                alt_match = re.search(pattern, text, re.IGNORECASE)
                if alt_match:
                    answer = alt_match.group(1).upper()
                    break
            if not answer:
                answer = "?"
        
        # Parse reasoning
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        return answer, reasoning
        
    except Exception as e:
        return "ERROR", str(e)


def ask_question_multi_sample(
    question: str,
    options: dict,
    model: str,
    num_samples: int = 10,
    rag_context: str = "",
    progress_callback: Optional[Callable] = None
) -> dict:
    """
    Ask a question multiple times and measure consistency.
    
    Args:
        question: The question text
        options: Dict mapping letter to option text (e.g., {'A': 'Paris', 'B': 'London'})
        model: Ollama model name
        num_samples: Number of times to run the question (default: 10)
        rag_context: Optional RAG context to include
        progress_callback: Optional function called with (current, total) for progress updates
    
    Returns:
        dict with:
            - answer: Most common answer
            - consistency: "X/N" string
            - consistency_count: Integer count of most common
            - consistency_pct: Percentage (0-100)
            - distribution: Full count of all answers
            - reasoning: Reasoning from first valid response
            - all_answers: List of all answers in order
    """
    if ollama is None:
        return {'error': 'Ollama not installed'}
    
    # Build options text
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
    
    # Add RAG context if provided
    if rag_context:
        rag_section = f"\n\nCOURSE MATERIALS:\n{rag_context}"
    else:
        rag_section = ""
    
    # Collect samples
    answers = []
    reasonings = []
    
    for i in range(num_samples):
        if progress_callback:
            progress_callback(i + 1, num_samples)
        
        answer, reasoning = ask_question_single(
            question=question,
            options_text=options_text,
            model=model,
            rag_context=rag_section,
            temperature=0.7  # Non-zero for variation
        )
        
        answers.append(answer)
        
        # Capture first valid reasoning
        if reasoning and not reasonings:
            reasonings.append(reasoning)
    
    # Calculate distribution
    distribution = Counter(answers)
    
    # Find most common valid answer
    valid_answers = {k: v for k, v in distribution.items() if k in 'ABCDE'}
    
    if valid_answers:
        most_common = max(valid_answers.items(), key=lambda x: x[1])
        answer = most_common[0]
        consistency_count = most_common[1]
    else:
        answer = "?"
        consistency_count = 0
    
    consistency_pct = round(consistency_count / num_samples * 100, 1)
    
    return {
        'answer': answer,
        'consistency': f"{consistency_count}/{num_samples}",
        'consistency_count': consistency_count,
        'consistency_pct': consistency_pct,
        'distribution': dict(distribution),
        'reasoning': reasonings[0] if reasonings else "",
        'all_answers': answers,
        'num_samples': num_samples
    }


def get_rag_context(question: str, collection, n_results: int = 3) -> str:
    """
    Query RAG collection for relevant context.
    
    Args:
        question: Question text to search for
        collection: ChromaDB collection
        n_results: Number of results to retrieve
    
    Returns:
        Concatenated context string
    """
    try:
        results = collection.query(query_texts=[question], n_results=n_results)
        if results and results['documents']:
            return "\n\n---\n\n".join(results['documents'][0])
    except Exception as e:
        print(f"RAG query error: {e}")
    return ""


# ============================================
# INTEGRATION WITH QUIZ SCANNER
# ============================================

def answer_question_for_quiz(
    question_text: str,
    options: dict,
    model: str,
    use_rag: bool = False,
    rag_collection = None,
    num_samples: int = 10,
    image_context: str = "",
    link_context: str = ""
) -> dict:
    """
    Full question answering for quiz scanner integration.
    
    Combines:
    - Multi-sample consistency
    - RAG context (if enabled)
    - Image interpretation context
    - Link scraping context
    
    Returns dict compatible with quiz scanner format.
    """
    # Build combined context
    context_parts = []
    
    if use_rag and rag_collection:
        rag_text = get_rag_context(question_text, rag_collection)
        if rag_text:
            context_parts.append(f"COURSE MATERIALS:\n{rag_text}")
    
    if image_context:
        context_parts.append(f"IMAGE CONTENT:\n{image_context}")
    
    if link_context:
        context_parts.append(f"LINKED CONTENT:\n{link_context}")
    
    full_context = "\n\n".join(context_parts)
    
    # Run multi-sample
    result = ask_question_multi_sample(
        question=question_text,
        options=options,
        model=model,
        num_samples=num_samples,
        rag_context=full_context
    )
    
    # Format for quiz scanner compatibility
    return {
        'llm_answer': result['answer'],
        'llm_confidence': result['consistency_pct'],  # Backwards compatible
        'llm_consistency': result['consistency'],
        'llm_consistency_count': result['consistency_count'],
        'llm_distribution': result['distribution'],
        'llm_reasoning': result['reasoning'],
        'llm_num_samples': result['num_samples']
    }


# ============================================
# CLI FOR TESTING
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test multi-sample question answering")
    parser.add_argument("--model", default="llama3:8b", help="Ollama model")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples")
    args = parser.parse_args()
    
    # Example question
    test_q = "What is the primary purpose of safety stock in inventory management?"
    test_opts = {
        'A': 'To reduce ordering costs',
        'B': 'To buffer against demand uncertainty',
        'C': 'To increase warehouse utilization',
        'D': 'To minimize holding costs',
        'E': 'To improve supplier relationships'
    }
    
    print(f"Testing with {args.samples} samples using {args.model}...")
    print(f"\nQuestion: {test_q}")
    print(f"Options: {test_opts}")
    print()
    
    def show_progress(current, total):
        print(f"  Sample {current}/{total}...", end='\r')
    
    result = ask_question_multi_sample(
        question=test_q,
        options=test_opts,
        model=args.model,
        num_samples=args.samples,
        progress_callback=show_progress
    )
    
    print(f"\n\nResults:")
    print(f"  Answer: {result['answer']}")
    print(f"  Consistency: {result['consistency']}")
    print(f"  Distribution: {result['distribution']}")
    print(f"  Reasoning: {result['reasoning'][:100]}...")
