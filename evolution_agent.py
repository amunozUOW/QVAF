#!/usr/bin/env python3
"""
Evolution Agent with Validation Framework
==========================================

Iteratively evolves quiz questions toward AI-resistance while maintaining
educational validity. Implements multi-objective validation to prevent
Goodhart's Law degenerate solutions.

VALIDATION GAUNTLET:
- Content Validity: Does it still test the learning objective?
- Construct Validity: Is the marked answer actually correct?
- Fairness: Can a knowledgeable student answer this?
- Clarity: Is it unambiguous?
- Semantic Anchoring: Is it still about the same concept?

EVOLUTION LOOP:
Questions → Test → Analyze → Rewrite → Validate → Accept/Reject → Repeat

Usage:
  python3 evolution_agent.py <vulnerability_report.json> [options]
  
Options:
  --iterations N     Maximum evolution iterations (default: 3)
  --target N         Target AI failure rate % (default: 70)
  --model MODEL      Ollama model to use (default: llama3:8b)
  --focus CATEGORY   Only evolve HIGH, MODERATE, or both (default: both)
"""

import json
import sys
import argparse
import re
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple
from enum import Enum

try:
    import ollama
except ImportError:
    print("pip install ollama")
    sys.exit(1)


# ============================================
# CONFIGURATION
# ============================================

try:
    from config import DEFAULT_MODEL
except ImportError:
    DEFAULT_MODEL = "llama3:8b"

EVOLUTION_MODEL = DEFAULT_MODEL  # Model for rewriting
VALIDATION_MODEL = DEFAULT_MODEL  # Model for validation checks

# Validation thresholds
MIN_VALIDITY_SCORE = 4  # Out of 5
MAX_SEMANTIC_DRIFT = 0.4  # 0-1 scale, higher = more drift allowed
MIN_EXPERT_ACCURACY = 0.8  # Expert should get it right 80%+ of time


# ============================================
# DATA STRUCTURES
# ============================================

class ValidationResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"


@dataclass
class ValidationCheck:
    """Result of a single validation check"""
    name: str
    result: ValidationResult
    score: Optional[float] = None
    explanation: str = ""
    details: dict = field(default_factory=dict)


@dataclass 
class ValidationReport:
    """Complete validation report for a rewritten question"""
    question_id: int
    original_question: str
    rewritten_question: str
    checks: List[ValidationCheck] = field(default_factory=list)
    overall_result: ValidationResult = ValidationResult.FAIL
    recommendation: str = ""
    
    def passed(self) -> bool:
        return self.overall_result == ValidationResult.PASS
    
    def to_dict(self) -> dict:
        return {
            'question_id': self.question_id,
            'original_question': self.original_question,
            'rewritten_question': self.rewritten_question,
            'checks': [
                {
                    'name': c.name,
                    'result': c.result.value,
                    'score': c.score,
                    'explanation': c.explanation,
                    'details': c.details
                }
                for c in self.checks
            ],
            'overall_result': self.overall_result.value,
            'recommendation': self.recommendation
        }


@dataclass
class EvolutionAttempt:
    """Record of one evolution attempt"""
    iteration: int
    original_question: str
    original_options: dict
    rewritten_question: str
    rewritten_options: dict
    correct_answer: str
    validation_report: ValidationReport
    ai_test_result: dict = field(default_factory=dict)
    accepted: bool = False


@dataclass
class QuestionEvolution:
    """Complete evolution history for one question"""
    question_id: int
    initial_vulnerability: str
    initial_question: str
    initial_options: dict
    correct_answer: str
    learning_objective: str
    attempts: List[EvolutionAttempt] = field(default_factory=list)
    final_vulnerability: str = "UNKNOWN"
    final_question: str = ""
    final_options: dict = field(default_factory=dict)
    converged: bool = False
    conclusion: str = ""


# ============================================
# VALIDATION FRAMEWORK
# ============================================

class ValidationFramework:
    """
    Multi-objective validation to prevent Goodhart's Law degenerate solutions.
    Every rewrite must pass ALL validation checks.
    """
    
    def __init__(self, model: str = VALIDATION_MODEL):
        self.model = model
    
    def validate_rewrite(
        self,
        original_question: str,
        original_options: dict,
        rewritten_question: str,
        rewritten_options: dict,
        correct_answer: str,
        learning_objective: str = ""
    ) -> ValidationReport:
        """
        Run complete validation gauntlet on a rewritten question.
        Returns ValidationReport with all check results.
        """
        report = ValidationReport(
            question_id=0,  # Set by caller
            original_question=original_question,
            rewritten_question=rewritten_question
        )
        
        # Run all validation checks
        checks = [
            self._check_answer_verification(rewritten_question, rewritten_options, correct_answer),
            self._check_content_validity(original_question, rewritten_question, learning_objective),
            self._check_expert_can_answer(rewritten_question, rewritten_options, correct_answer),
            self._check_clarity(rewritten_question, rewritten_options),
            self._check_semantic_drift(original_question, rewritten_question),
        ]
        
        report.checks = checks
        
        # Determine overall result
        failures = [c for c in checks if c.result == ValidationResult.FAIL]
        warnings = [c for c in checks if c.result == ValidationResult.WARN]
        
        if failures:
            report.overall_result = ValidationResult.FAIL
            report.recommendation = f"REJECTED: Failed {len(failures)} validation check(s): " + \
                                   ", ".join([f.name for f in failures])
        elif warnings:
            report.overall_result = ValidationResult.WARN
            report.recommendation = f"CAUTION: {len(warnings)} warning(s). Manual review recommended."
        else:
            report.overall_result = ValidationResult.PASS
            report.recommendation = "ACCEPTED: Passed all validation checks."
        
        return report
    
    def _check_answer_verification(
        self,
        question: str,
        options: dict,
        correct_answer: str
    ) -> ValidationCheck:
        """
        Verify the marked correct answer is actually defensible.
        Ask LLM to explain why it's correct - if it can't, question is broken.
        """
        options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
        correct_text = options.get(correct_answer, correct_answer)
        
        prompt = f"""You are validating a quiz question. The marked correct answer is {correct_answer}: "{correct_text}"

QUESTION: {question}

OPTIONS:
{options_text}

TASK: Explain why answer {correct_answer} is correct. Be specific and reference the question content.

If you CANNOT justify why {correct_answer} is correct (because another answer seems better or the question is flawed), say "CANNOT JUSTIFY" and explain why.

Your analysis:"""

        response = self._call_llm(prompt)
        
        # Check if justification succeeded
        cannot_justify = "CANNOT JUSTIFY" in response.upper() or \
                        "CANNOT BE JUSTIFIED" in response.upper() or \
                        "NOT CORRECT" in response.upper()
        
        if cannot_justify:
            return ValidationCheck(
                name="Answer Verification",
                result=ValidationResult.FAIL,
                explanation="Could not justify the marked correct answer",
                details={'llm_response': response}
            )
        
        return ValidationCheck(
            name="Answer Verification",
            result=ValidationResult.PASS,
            explanation="Correct answer can be justified",
            details={'justification': response[:500]}
        )
    
    def _check_content_validity(
        self,
        original_question: str,
        rewritten_question: str,
        learning_objective: str
    ) -> ValidationCheck:
        """
        Check if rewritten question still tests the intended learning objective.
        """
        lo_text = f"\nLearning Objective: {learning_objective}" if learning_objective else ""
        
        prompt = f"""You are validating that a rewritten question still tests the same concept.

ORIGINAL QUESTION:
{original_question}
{lo_text}

REWRITTEN QUESTION:
{rewritten_question}

TASK: Rate how well the rewritten question tests the SAME core concept as the original.

Score 1-5:
1 = Completely different topic/concept
2 = Related but tests different knowledge
3 = Same general area but different focus
4 = Same concept with minor shift in emphasis
5 = Tests exactly the same knowledge/skill

Respond in this format:
SCORE: [1-5]
RATIONALE: [explanation]"""

        response = self._call_llm(prompt)
        
        # Extract score
        score_match = re.search(r'SCORE:\s*(\d)', response)
        score = int(score_match.group(1)) if score_match else 3
        
        if score < MIN_VALIDITY_SCORE:
            return ValidationCheck(
                name="Content Validity",
                result=ValidationResult.FAIL,
                score=score,
                explanation=f"Score {score}/5 - Question drifted from original concept",
                details={'llm_response': response}
            )
        
        return ValidationCheck(
            name="Content Validity",
            result=ValidationResult.PASS,
            score=score,
            explanation=f"Score {score}/5 - Question tests same concept",
            details={'llm_response': response}
        )
    
    def _check_expert_can_answer(
        self,
        question: str,
        options: dict,
        correct_answer: str
    ) -> ValidationCheck:
        """
        Test if a domain expert can answer correctly (fairness check).
        If even an expert can't answer, the question may be flawed.
        """
        options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
        
        prompt = f"""You are an EXPERT in this field with comprehensive knowledge. 
Answer this question using your deep domain expertise.

QUESTION: {question}

OPTIONS:
{options_text}

Think through this carefully using your expert knowledge, then provide your answer.

ANSWER: [single letter A-E]
CONFIDENCE: [0-100]
REASONING: [brief explanation]"""

        response = self._call_llm(prompt)
        
        # Extract answer
        answer_match = re.search(r'ANSWER:\s*([A-Ea-e])', response)
        expert_answer = answer_match.group(1).upper() if answer_match else "?"
        
        expert_correct = expert_answer == correct_answer.upper()
        
        if not expert_correct:
            return ValidationCheck(
                name="Expert Fairness",
                result=ValidationResult.WARN,  # Warning, not failure
                score=0,
                explanation=f"Expert answered {expert_answer}, correct is {correct_answer}. May need review.",
                details={'expert_answer': expert_answer, 'correct_answer': correct_answer, 'reasoning': response}
            )
        
        return ValidationCheck(
            name="Expert Fairness",
            result=ValidationResult.PASS,
            score=1,
            explanation="Expert can answer correctly - question is fair",
            details={'expert_answer': expert_answer}
        )
    
    def _check_clarity(
        self,
        question: str,
        options: dict
    ) -> ValidationCheck:
        """
        Check if the question is clear and unambiguous.
        """
        options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
        
        prompt = f"""Evaluate this quiz question for clarity and potential ambiguity.

QUESTION: {question}

OPTIONS:
{options_text}

Check for:
1. Grammatical correctness
2. Clear, unambiguous wording
3. Options that are clearly distinct
4. No trick wording or double negatives
5. Sufficient information to answer

Score 1-5:
1 = Seriously flawed, confusing, or ambiguous
2 = Multiple issues with clarity
3 = Minor clarity issues
4 = Mostly clear with trivial issues
5 = Crystal clear and well-written

Respond:
SCORE: [1-5]
ISSUES: [list any problems found, or "None"]"""

        response = self._call_llm(prompt)
        
        score_match = re.search(r'SCORE:\s*(\d)', response)
        score = int(score_match.group(1)) if score_match else 3
        
        if score < 3:
            return ValidationCheck(
                name="Clarity",
                result=ValidationResult.FAIL,
                score=score,
                explanation=f"Score {score}/5 - Question has clarity issues",
                details={'llm_response': response}
            )
        elif score < 4:
            return ValidationCheck(
                name="Clarity",
                result=ValidationResult.WARN,
                score=score,
                explanation=f"Score {score}/5 - Minor clarity issues",
                details={'llm_response': response}
            )
        
        return ValidationCheck(
            name="Clarity",
            result=ValidationResult.PASS,
            score=score,
            explanation=f"Score {score}/5 - Question is clear",
            details={'llm_response': response}
        )
    
    def _check_semantic_drift(
        self,
        original: str,
        rewritten: str
    ) -> ValidationCheck:
        """
        Check if the rewritten question has drifted too far semantically.
        Uses LLM to assess similarity since we don't have embeddings.
        """
        prompt = f"""Compare these two questions and rate their semantic similarity.

ORIGINAL: {original}

REWRITTEN: {rewritten}

Rate similarity 0-100:
0 = Completely different meaning/topic
50 = Same general topic but different question
80 = Very similar, testing same specific knowledge
100 = Essentially identical meaning (just rephrased)

Respond:
SIMILARITY: [0-100]
EXPLANATION: [brief comparison]"""

        response = self._call_llm(prompt)
        
        sim_match = re.search(r'SIMILARITY:\s*(\d+)', response)
        similarity = int(sim_match.group(1)) if sim_match else 50
        
        # Convert to drift score (0 = no drift, 1 = complete drift)
        drift = (100 - similarity) / 100
        
        if drift > MAX_SEMANTIC_DRIFT:
            return ValidationCheck(
                name="Semantic Anchoring",
                result=ValidationResult.FAIL,
                score=similarity,
                explanation=f"Similarity {similarity}% - Too much drift from original",
                details={'similarity': similarity, 'drift': drift}
            )
        
        return ValidationCheck(
            name="Semantic Anchoring",
            result=ValidationResult.PASS,
            score=similarity,
            explanation=f"Similarity {similarity}% - Acceptable semantic anchoring",
            details={'similarity': similarity, 'drift': drift}
        )
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with error handling"""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0, 'num_predict': 512}
            )
            return response['message']['content']
        except Exception as e:
            return f"[LLM Error: {e}]"


# ============================================
# REWRITE AGENT
# ============================================

class RewriteAgent:
    """
    Generates candidate rewrites for vulnerable questions.
    Uses strategies based on question type and vulnerability analysis.
    """
    
    def __init__(self, model: str = EVOLUTION_MODEL):
        self.model = model
    
    def generate_rewrite(
        self,
        question: str,
        options: dict,
        correct_answer: str,
        question_type: str,
        vulnerability: str,
        analysis: str = "",
        previous_attempts: List[str] = None
    ) -> Tuple[str, dict]:
        """
        Generate a rewritten question that aims to be more AI-resistant.
        Returns (rewritten_question, rewritten_options).
        """
        options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
        
        # Build context about previous attempts
        prev_context = ""
        if previous_attempts:
            prev_context = f"""
PREVIOUS REWRITE ATTEMPTS (do something different):
{chr(10).join(['- ' + a[:100] + '...' for a in previous_attempts[-3:]])}
"""
        
        prompt = f"""You are an expert assessment designer tasked with making this question more resistant to AI assistance while maintaining educational validity.

ORIGINAL QUESTION:
{question}

OPTIONS:
{options_text}

CORRECT ANSWER: {correct_answer}

QUESTION TYPE: {question_type}
CURRENT VULNERABILITY: {vulnerability}

{f"ANALYSIS: {analysis}" if analysis else ""}
{prev_context}

REWRITE STRATEGIES (choose based on question type):
1. For RECALL questions: Add context-dependent details, require integration of multiple facts
2. For ROUTINE APPLICATION: Use non-standard numbers, require intermediate reasoning steps
3. For COMPREHENSION: Add novel scenarios, require transfer to new contexts
4. For ANALYSIS: Include local/specific details AI wouldn't know, add realistic constraints
5. For all types: Add authentic workplace/scenario details, require judgment calls

REQUIREMENTS:
- Keep the SAME correct answer ({correct_answer})
- Test the SAME core concept
- Make it harder for AI by requiring domain-specific knowledge or reasoning
- Keep it fair for knowledgeable students
- Maintain clarity

OUTPUT FORMAT:
REWRITTEN QUESTION:
[your improved question]

REWRITTEN OPTIONS:
A. [option A]
B. [option B]
C. [option C]
D. [option D]
E. [option E]

STRATEGY USED:
[brief explanation of what you changed and why]"""

        response = self._call_llm(prompt)
        
        # Parse response
        rewritten_q, rewritten_opts = self._parse_rewrite(response, options)
        
        return rewritten_q, rewritten_opts
    
    def _parse_rewrite(self, response: str, original_options: dict) -> Tuple[str, dict]:
        """Parse LLM response to extract rewritten question and options"""
        
        # Extract question
        q_match = re.search(r'REWRITTEN QUESTION:\s*\n(.+?)(?=REWRITTEN OPTIONS:|$)', response, re.DOTALL)
        rewritten_q = q_match.group(1).strip() if q_match else ""
        
        # Extract options
        rewritten_opts = {}
        for letter in ['A', 'B', 'C', 'D', 'E']:
            pattern = rf'{letter}[\.\)]\s*(.+?)(?=[A-E][\.\)]|\nSTRATEGY|$)'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                rewritten_opts[letter] = match.group(1).strip()
        
        # Fall back to original options if parsing failed
        if not rewritten_opts:
            rewritten_opts = original_options.copy()
        
        # If we got partial options, fill in from original
        for letter in original_options:
            if letter not in rewritten_opts:
                rewritten_opts[letter] = original_options[letter]
        
        return rewritten_q, rewritten_opts
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with error handling"""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.7, 'num_predict': 1024}  # Slightly higher temp for creativity
            )
            return response['message']['content']
        except Exception as e:
            return f"[LLM Error: {e}]"


# ============================================
# ADVERSARIAL TESTER
# ============================================

class AdversarialTester:
    """
    Tests questions against AI to measure vulnerability.
    Uses the optimized prompt from prompt_optimizer.py.
    """
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
    
    def test_question(
        self,
        question: str,
        options: dict,
        correct_answer: str,
        use_rag: bool = False,
        rag_context: str = ""
    ) -> dict:
        """
        Test a question against AI and return results.
        """
        options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
        
        context_block = ""
        if use_rag and rag_context:
            context_block = f"""
COURSE MATERIALS:
{rag_context}
"""
        
        # Use optimized v4 prompt
        prompt = f"""TASK: Answer this multiple choice question correctly.

QUESTION: {question}

OPTIONS:
{options_text}
{context_block}
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

        response = self._call_llm(prompt)
        
        # Parse response
        answer_match = re.search(r'ANSWER:\s*([A-Ea-e])', response)
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)', response)
        
        ai_answer = answer_match.group(1).upper() if answer_match else "?"
        confidence = int(conf_match.group(1)) if conf_match else 0
        
        is_correct = ai_answer == correct_answer.upper()
        
        return {
            'ai_answer': ai_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'confidence': confidence,
            'raw_response': response[:500],
            'use_rag': use_rag
        }
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with error handling"""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0, 'num_predict': 512}
            )
            return response['message']['content']
        except Exception as e:
            return f"[LLM Error: {e}]"


# ============================================
# EVOLUTION CONTROLLER
# ============================================

class EvolutionController:
    """
    Main controller for the evolution process.
    Manages iterations, tracks metrics, determines convergence.
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_iterations: int = 3,
        target_failure_rate: float = 70.0
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.target_failure_rate = target_failure_rate
        
        self.validator = ValidationFramework(model)
        self.rewriter = RewriteAgent(model)
        self.tester = AdversarialTester(model)
    
    def evolve_question(
        self,
        question_id: int,
        question: str,
        options: dict,
        correct_answer: str,
        question_type: str,
        initial_vulnerability: str,
        analysis: str = "",
        learning_objective: str = ""
    ) -> QuestionEvolution:
        """
        Evolve a single question through the validation loop.
        """
        evolution = QuestionEvolution(
            question_id=question_id,
            initial_vulnerability=initial_vulnerability,
            initial_question=question,
            initial_options=options,
            correct_answer=correct_answer,
            learning_objective=learning_objective
        )
        
        current_question = question
        current_options = options.copy()
        previous_attempts = []
        
        print(f"\n{'='*60}")
        print(f"EVOLVING QUESTION {question_id}")
        print(f"Initial vulnerability: {initial_vulnerability}")
        print(f"{'='*60}")
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")
            
            # Generate rewrite
            print("  Generating rewrite...")
            rewritten_q, rewritten_opts = self.rewriter.generate_rewrite(
                question=current_question,
                options=current_options,
                correct_answer=correct_answer,
                question_type=question_type,
                vulnerability=initial_vulnerability,
                analysis=analysis,
                previous_attempts=previous_attempts
            )
            
            if not rewritten_q:
                print("  ⚠ Rewrite generation failed")
                continue
            
            print(f"  Rewritten: {rewritten_q[:80]}...")
            previous_attempts.append(rewritten_q)
            
            # Validate rewrite
            print("  Running validation gauntlet...")
            validation = self.validator.validate_rewrite(
                original_question=question,  # Always compare to original
                original_options=options,
                rewritten_question=rewritten_q,
                rewritten_options=rewritten_opts,
                correct_answer=correct_answer,
                learning_objective=learning_objective
            )
            validation.question_id = question_id
            
            # Print validation results
            for check in validation.checks:
                icon = "✓" if check.result == ValidationResult.PASS else \
                       "⚠" if check.result == ValidationResult.WARN else "✗"
                print(f"    {icon} {check.name}: {check.explanation}")
            
            # Test against AI if validation passed
            ai_test = {'tested': False}
            if validation.passed() or validation.overall_result == ValidationResult.WARN:
                print("  Testing against AI...")
                
                # Test without RAG
                no_rag_result = self.tester.test_question(
                    rewritten_q, rewritten_opts, correct_answer, use_rag=False
                )
                
                # Test with RAG (simulated - no context for now)
                with_rag_result = self.tester.test_question(
                    rewritten_q, rewritten_opts, correct_answer, use_rag=True, rag_context=""
                )
                
                ai_test = {
                    'tested': True,
                    'no_rag': no_rag_result,
                    'with_rag': with_rag_result
                }
                
                no_rag_icon = "✗" if no_rag_result['is_correct'] else "✓"
                with_rag_icon = "✗" if with_rag_result['is_correct'] else "✓"
                print(f"    AI No-RAG: {no_rag_icon} (answered {no_rag_result['ai_answer']})")
                print(f"    AI With-RAG: {with_rag_icon} (answered {with_rag_result['ai_answer']})")
            
            # Record attempt
            attempt = EvolutionAttempt(
                iteration=iteration,
                original_question=current_question,
                original_options=current_options,
                rewritten_question=rewritten_q,
                rewritten_options=rewritten_opts,
                correct_answer=correct_answer,
                validation_report=validation,
                ai_test_result=ai_test
            )
            
            # Determine if we should accept this rewrite
            accept = False
            if validation.passed():
                if ai_test.get('tested'):
                    # Accept if AI now fails both tests
                    ai_fails_both = not ai_test['no_rag']['is_correct'] and \
                                   not ai_test['with_rag']['is_correct']
                    if ai_fails_both:
                        accept = True
                        print("  ✓ ACCEPTED: Validation passed + AI fails both tests")
                    else:
                        print("  ⚠ Validation passed but AI still succeeds")
                else:
                    accept = True
                    print("  ✓ ACCEPTED: Validation passed")
            else:
                print(f"  ✗ REJECTED: {validation.recommendation}")
            
            attempt.accepted = accept
            evolution.attempts.append(attempt)
            
            # Check convergence
            if accept:
                # Determine new vulnerability category
                if ai_test.get('tested'):
                    no_rag_correct = ai_test['no_rag']['is_correct']
                    with_rag_correct = ai_test['with_rag']['is_correct']
                    
                    if not no_rag_correct and not with_rag_correct:
                        new_vuln = "LOW"
                    elif no_rag_correct and with_rag_correct:
                        new_vuln = "HIGH"
                    elif not no_rag_correct and with_rag_correct:
                        new_vuln = "MODERATE"
                    else:
                        new_vuln = "ANOMALY"
                else:
                    new_vuln = "UNKNOWN"
                
                current_question = rewritten_q
                current_options = rewritten_opts
                
                if new_vuln == "LOW":
                    evolution.converged = True
                    evolution.final_vulnerability = new_vuln
                    evolution.final_question = rewritten_q
                    evolution.final_options = rewritten_opts
                    evolution.conclusion = f"SUCCESS: Converged to LOW vulnerability after {iteration} iteration(s)"
                    print(f"\n🎉 CONVERGED to LOW vulnerability!")
                    break
        
        # Final status if not converged
        if not evolution.converged:
            # Check if any iteration was accepted
            accepted = [a for a in evolution.attempts if a.accepted]
            if accepted:
                last_accepted = accepted[-1]
                evolution.final_question = last_accepted.rewritten_question
                evolution.final_options = last_accepted.rewritten_options
                
                # Determine final vulnerability from last accepted
                if last_accepted.ai_test_result.get('tested'):
                    no_rag = last_accepted.ai_test_result['no_rag']['is_correct']
                    with_rag = last_accepted.ai_test_result['with_rag']['is_correct']
                    if not no_rag and not with_rag:
                        evolution.final_vulnerability = "LOW"
                    elif no_rag and with_rag:
                        evolution.final_vulnerability = "HIGH"
                    elif not no_rag and with_rag:
                        evolution.final_vulnerability = "MODERATE"
                    else:
                        evolution.final_vulnerability = "ANOMALY"
                
                evolution.conclusion = f"PARTIAL: Improved but not fully AI-resistant after {self.max_iterations} iterations"
            else:
                evolution.final_question = question
                evolution.final_options = options
                evolution.final_vulnerability = initial_vulnerability
                evolution.conclusion = f"CANNOT_EVOLVE: This question may not be improvable as MCQ. Consider alternative assessment format."
            
            print(f"\n⚠ Did not converge: {evolution.conclusion}")
        
        return evolution


# ============================================
# REPORT GENERATION
# ============================================

def generate_evolution_report(evolutions: List[QuestionEvolution], filepath: str) -> str:
    """Generate markdown report of evolution results"""
    
    total = len(evolutions)
    converged = sum(1 for e in evolutions if e.converged)
    partial = sum(1 for e in evolutions if not e.converged and e.final_vulnerability != e.initial_vulnerability)
    unchanged = total - converged - partial
    
    report = f"""# Quiz Evolution Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source: {filepath}

## Summary

| Metric | Count |
|--------|-------|
| Total Questions Evolved | {total} |
| Converged to LOW | {converged} ({round(converged/total*100, 1) if total else 0}%) |
| Partial Improvement | {partial} ({round(partial/total*100, 1) if total else 0}%) |
| Cannot Improve | {unchanged} ({round(unchanged/total*100, 1) if total else 0}%) |

## Question-by-Question Results

"""
    
    for ev in evolutions:
        status_icon = "✅" if ev.converged else "⚠️" if ev.final_vulnerability != ev.initial_vulnerability else "❌"
        
        report += f"""### Question {ev.question_id} {status_icon}

**Status:** {ev.conclusion}

**Vulnerability:** {ev.initial_vulnerability} → {ev.final_vulnerability}

**Original:**
> {ev.initial_question[:200]}{'...' if len(ev.initial_question) > 200 else ''}

**Final:**
> {ev.final_question[:200]}{'...' if len(ev.final_question) > 200 else ''}

**Evolution Attempts:** {len(ev.attempts)}

"""
        
        for att in ev.attempts:
            accept_icon = "✓" if att.accepted else "✗"
            report += f"""
#### Iteration {att.iteration} [{accept_icon}]

Validation: {att.validation_report.overall_result.value}
"""
            for check in att.validation_report.checks:
                check_icon = "✓" if check.result == ValidationResult.PASS else "⚠" if check.result == ValidationResult.WARN else "✗"
                report += f"- {check_icon} {check.name}: {check.explanation}\n"
            
            if att.ai_test_result.get('tested'):
                report += f"\nAI Test: No-RAG={att.ai_test_result['no_rag']['ai_answer']} ({'✓' if not att.ai_test_result['no_rag']['is_correct'] else '✗'}), "
                report += f"With-RAG={att.ai_test_result['with_rag']['ai_answer']} ({'✓' if not att.ai_test_result['with_rag']['is_correct'] else '✗'})\n"
        
        report += "\n---\n\n"
    
    # Add section for questions that can't be improved
    cannot_improve = [e for e in evolutions if e.conclusion.startswith("CANNOT_EVOLVE")]
    if cannot_improve:
        report += """## Questions Requiring Alternative Assessment

The following questions could not be made AI-resistant in MCQ format. 
Consider alternative assessment approaches:

"""
        for ev in cannot_improve:
            report += f"- **Q{ev.question_id}:** {ev.initial_question[:100]}...\n"
    
    return report


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Evolve quiz questions toward AI-resistance with validation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input_file', help='Vulnerability report JSON from reform_agent.py')
    parser.add_argument('--iterations', type=int, default=3, help='Max evolution iterations (default: 3)')
    parser.add_argument('--target', type=float, default=70.0, help='Target AI failure rate %% (default: 70)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help=f'Ollama model (default: {DEFAULT_MODEL})')
    parser.add_argument('--focus', choices=['HIGH', 'MODERATE', 'both'], default='both',
                       help='Which vulnerability categories to evolve (default: both)')
    
    args = parser.parse_args()
    
    # Load vulnerability report
    print(f"Loading {args.input_file}...")
    with open(args.input_file, 'r') as f:
        report = json.load(f)
    
    # Get questions to evolve
    questions = report.get('question_results', [])
    analyses = report.get('qualitative_analyses', [])
    
    # Create analysis lookup
    analysis_by_id = {a['id']: a for a in analyses}
    
    # Filter by focus
    if args.focus == 'both':
        targets = ['HIGH', 'MODERATE']
    else:
        targets = [args.focus]
    
    to_evolve = [q for q in questions if q.get('vulnerability_category') in targets]
    
    print(f"\nFound {len(to_evolve)} questions to evolve (focus: {args.focus})")
    print(f"Model: {args.model}")
    print(f"Max iterations: {args.iterations}")
    
    # Initialize controller
    controller = EvolutionController(
        model=args.model,
        max_iterations=args.iterations,
        target_failure_rate=args.target
    )
    
    # Evolve questions
    evolutions = []
    
    for q in to_evolve:
        q_id = q.get('id')
        
        # Get original question text from analyses
        analysis = analysis_by_id.get(q_id, {})
        q_text = analysis.get('question', q.get('question', ''))
        
        # If question text not in analysis, try to find from other sources
        if not q_text:
            print(f"  ⚠ No question text found for Q{q_id}, skipping")
            continue
        
        # Get options - need to reconstruct from data
        # This may need adjustment based on actual data structure
        options = q.get('options', {})
        if not options:
            # Try to get from analysis
            options = analysis.get('options', {'A': '?', 'B': '?', 'C': '?', 'D': '?', 'E': '?'})
        
        ev = controller.evolve_question(
            question_id=q_id,
            question=q_text,
            options=options,
            correct_answer=q.get('correct_answer', 'A'),
            question_type=q.get('question_type', 'UNKNOWN'),
            initial_vulnerability=q.get('vulnerability_category', 'UNKNOWN'),
            analysis=analysis.get('analysis', ''),
            learning_objective=""
        )
        
        evolutions.append(ev)
    
    # Generate report
    print("\n" + "="*60)
    print("GENERATING REPORTS")
    print("="*60)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Markdown report
    md_report = generate_evolution_report(evolutions, args.input_file)
    md_file = f"evolution_{timestamp}_report.md"
    with open(md_file, 'w') as f:
        f.write(md_report)
    print(f"Markdown report: {md_file}")
    
    # JSON data
    json_data = {
        'timestamp': timestamp,
        'source_file': args.input_file,
        'model': args.model,
        'max_iterations': args.iterations,
        'target_failure_rate': args.target,
        'summary': {
            'total': len(evolutions),
            'converged': sum(1 for e in evolutions if e.converged),
            'partial': sum(1 for e in evolutions if not e.converged and e.final_vulnerability != e.initial_vulnerability),
            'unchanged': sum(1 for e in evolutions if not e.converged and e.final_vulnerability == e.initial_vulnerability)
        },
        'evolutions': [
            {
                'question_id': e.question_id,
                'initial_vulnerability': e.initial_vulnerability,
                'final_vulnerability': e.final_vulnerability,
                'converged': e.converged,
                'conclusion': e.conclusion,
                'initial_question': e.initial_question,
                'final_question': e.final_question,
                'final_options': e.final_options,
                'attempts_count': len(e.attempts),
                'attempts': [
                    {
                        'iteration': a.iteration,
                        'accepted': a.accepted,
                        'rewritten_question': a.rewritten_question,
                        'validation_result': a.validation_report.overall_result.value,
                        'ai_tested': a.ai_test_result.get('tested', False)
                    }
                    for a in e.attempts
                ]
            }
            for e in evolutions
        ]
    }
    
    json_file = f"evolution_{timestamp}_data.json"
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON data: {json_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVOLUTION COMPLETE")
    print("="*60)
    print(f"Total questions: {len(evolutions)}")
    print(f"Converged to LOW: {json_data['summary']['converged']}")
    print(f"Partial improvement: {json_data['summary']['partial']}")
    print(f"Cannot improve: {json_data['summary']['unchanged']}")
    
    if json_data['summary']['unchanged'] > 0:
        print(f"\n⚠ {json_data['summary']['unchanged']} question(s) could not be made AI-resistant.")
        print("  Consider alternative assessment formats for these.")


if __name__ == "__main__":
    main()
