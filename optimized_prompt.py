#!/usr/bin/env python3
"""
Optimized Prompt for Quiz Vulnerability Scanner
================================================

Based on empirical testing across prompt strategies, this prompt combines:
- Elimination approach (best for conceptual questions)
- Structured calculation support (best for numerical questions)
- Strict format enforcement (prevents "?" failures)
- Domain-agnostic design (works across subjects)

Test results that informed this design:
- elimination: 80% on llama3:8b (best for Q1 transforming resources)
- structured_analysis: 80% on llama3:8b (best for Q4 calculation)
- step_by_step: 0% (format failures - model knew answers but didn't format)
- expert_role: 20% (verbose preamble broke format)

"""

def optimized_prompt_v1(question: str, options: dict, context: str = "") -> str:
    """
    Optimized prompt combining elimination + structured analysis.
    Domain-agnostic, strict format enforcement.
    """
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
    
    return f"""You must answer this multiple choice question. Follow the format exactly.

QUESTION: {question}

OPTIONS:
{options_text}
{context}
INSTRUCTIONS:
1. Read the question carefully. Identify what is being asked.
2. If this involves a calculation, show your working.
3. Evaluate EACH option - state if it could be correct or should be eliminated:
   A: [likely/unlikely] because [brief reason]
   B: [likely/unlikely] because [brief reason]
   C: [likely/unlikely] because [brief reason]
   D: [likely/unlikely] because [brief reason]
   E: [likely/unlikely] because [brief reason]
4. Select the BEST answer from the options that were not eliminated.

IMPORTANT: You MUST end your response with exactly this format:
ANSWER: [single letter A, B, C, D, or E]
CONFIDENCE: [number from 0 to 100]
REASONING: [one sentence explaining your choice]

Confidence guide: 90-100 = certain, 70-89 = confident, 50-69 = educated guess, below 50 = uncertain

Begin your analysis:"""


def optimized_prompt_v2(question: str, options: dict, context: str = "") -> str:
    """
    Streamlined version - less verbose, same core approach.
    """
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
    
    return f"""Answer this multiple choice question by evaluating each option.

QUESTION: {question}

OPTIONS:
{options_text}
{context}
For each option, briefly state if it's likely correct or can be eliminated:
A: [evaluate]
B: [evaluate]
C: [evaluate]
D: [evaluate]
E: [evaluate]

If the question requires calculation, show your working.

Then provide your final answer in EXACTLY this format:
ANSWER: [letter]
CONFIDENCE: [0-100]
REASONING: [brief explanation]"""


def optimized_prompt_v3(question: str, options: dict, context: str = "") -> str:
    """
    Maximum structure version - most explicit instructions.
    Use this if v1/v2 still have format issues.
    """
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
    
    return f"""TASK: Select the correct answer for this multiple choice question.

QUESTION: {question}

OPTIONS:
{options_text}
{context}
STEP 1 - IDENTIFY QUESTION TYPE:
What is this question testing? (factual recall / conceptual understanding / calculation / application)

STEP 2 - WORK THROUGH THE PROBLEM:
If calculation needed, show the math.
If conceptual, identify the key concept being tested.

STEP 3 - EVALUATE EACH OPTION:
A: [KEEP or ELIMINATE] - [reason]
B: [KEEP or ELIMINATE] - [reason]
C: [KEEP or ELIMINATE] - [reason]
D: [KEEP or ELIMINATE] - [reason]
E: [KEEP or ELIMINATE] - [reason]

STEP 4 - SELECT BEST ANSWER:
From the options marked KEEP, which is the best match?

REQUIRED OUTPUT FORMAT (must appear at end of response):
ANSWER: X
CONFIDENCE: N
REASONING: Your explanation

Where X is exactly one letter (A, B, C, D, or E) and N is a number 0-100.

Begin:"""


def optimized_prompt_v4(question: str, options: dict, context: str = "") -> str:
    """
    Best of both worlds:
    - v3's structured analysis (gets conceptual Q1 right)
    - Stronger format enforcement (prevents ? failures)
    - Explicit reminder about format at the end
    """
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
    
    return f"""TASK: Answer this multiple choice question correctly.

QUESTION: {question}

OPTIONS:
{options_text}
{context}
ANALYSIS STEPS:

1. QUESTION TYPE: Is this testing recall, conceptual understanding, calculation, or application?

2. KEY INSIGHT: What core concept or distinction is being tested here?

3. EVALUATE OPTIONS:
   A: [KEEP/ELIMINATE] - why?
   B: [KEEP/ELIMINATE] - why?
   C: [KEEP/ELIMINATE] - why?
   D: [KEEP/ELIMINATE] - why?
   E: [KEEP/ELIMINATE] - why?

4. CALCULATION (if needed): Show your working.

5. FINAL SELECTION: From options marked KEEP, select the single best answer.

=== REQUIRED OUTPUT FORMAT ===
After your analysis, you MUST write these three lines:

ANSWER: [write ONE letter: A, B, C, D, or E]
CONFIDENCE: [write a number from 0 to 100]
REASONING: [write one sentence explaining why]

Do not write anything after the REASONING line.

Begin your analysis:"""


# For testing
if __name__ == "__main__":
    import json
    import re
    import sys
    
    try:
        import ollama
    except ImportError:
        print("pip install ollama")
        sys.exit(1)

    try:
        from config import DEFAULT_MODEL
    except ImportError:
        DEFAULT_MODEL = "llama3:8b"
    
    # Test questions
    TEST_QUESTIONS = [
        {
            "id": 1,
            "question": "In a typical operation, which of the following are classifiable as transforming resources (Inputs)?",
            "options": {
                "A": "Ingredients in food recipes",
                "B": "Employees operating manufacturing equipment",
                "C": "Flour in a bakery",
                "D": "Bandages in a hospital",
                "E": "Water in a brewing process"
            },
            "correct": "B"
        },
        {
            "id": 2,
            "question": "The concept of hierarchy in operations management refers to:",
            "options": {
                "A": "The chain of command in an organization",
                "B": "The levels of decision-making from strategic to operational",
                "C": "The ranking of employees by seniority",
                "D": "The order in which tasks are completed",
                "E": "The structure of the supply chain"
            },
            "correct": "B"
        },
        {
            "id": 3,
            "question": "Which of the following is NOT a qualitative approach to forecasting?",
            "options": {
                "A": "Delphi method",
                "B": "Market research",
                "C": "Moving average",
                "D": "Expert opinion",
                "E": "Sales force composite"
            },
            "correct": "C"
        },
        {
            "id": 4,
            "question": "A company produces 1000 units per day with 10 workers working 8 hours each. What is the labor productivity?",
            "options": {
                "A": "10 units per labor hour",
                "B": "12.5 units per labor hour",
                "C": "100 units per worker",
                "D": "125 units per worker per day",
                "E": "80 units per labor hour"
            },
            "correct": "B"
        },
        {
            "id": 5,
            "question": "In the context of supply chain management, the 'bullwhip effect' refers to:",
            "options": {
                "A": "The tendency for product quality to decline along the supply chain",
                "B": "The amplification of demand variability as you move upstream in the supply chain",
                "C": "The speed at which products move through the supply chain",
                "D": "The impact of transportation costs on final product pricing",
                "E": "The effect of supplier relationships on production scheduling"
            },
            "correct": "B"
        }
    ]
    
    def test_prompt(prompt_fn, model=DEFAULT_MODEL):
        print(f"\nTesting {prompt_fn.__name__} with {model}")
        print("="*60)
        
        correct = 0
        for q in TEST_QUESTIONS:
            prompt = prompt_fn(q['question'], q['options'])
            
            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0, 'num_predict': 512}
            )
            
            text = response['message']['content']
            match = re.search(r'ANSWER:\s*([A-Za-z])', text)
            answer = match.group(1).upper() if match else "?"
            
            is_correct = answer == q['correct']
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"Q{q['id']}: {status} (got {answer}, expected {q['correct']})")
        
        print(f"\nAccuracy: {correct}/5 ({correct/5*100}%)")
        return correct
    
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    
    print("\n" + "="*60)
    print("OPTIMIZED PROMPT COMPARISON")
    print("="*60)
    
    results = {}
    for fn in [optimized_prompt_v1, optimized_prompt_v2, optimized_prompt_v3, optimized_prompt_v4]:
        results[fn.__name__] = test_prompt(fn, model)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, score in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name}: {score}/5 ({score/5*100}%)")