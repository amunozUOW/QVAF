"""
Test single question functionality and UI tab.

Extracted from App.py — no logic changes.
"""

import streamlit as st
import pandas as pd
from collections import Counter

from parsing_utils import parse_llm_response, extract_answer, extract_confidence, extract_reasoning
from config import AVAILABLE_MODELS, DEFAULT_MODEL
from app_rag import check_rag_available


def test_single_question(question: str, options: dict, correct_answer: str, model: str,
                         use_rag: bool = False, num_samples: int = 1):
    """
    Test a single question against the AI without browser automation.

    Multi-sample behavior (num_samples > 1):
    - Runs the AI multiple times on the same question
    - Selects the MOST COMMON answer (by count, not confidence)
    - Reports consistency as "X/N" (e.g., "7/10" means 7 of 10 runs chose the same answer)
    """
    try:
        import ollama
    except ImportError:
        return {'error': 'Ollama not installed'}

    # Build options text
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])

    # RAG context (if available and requested)
    rag_context = ""
    if use_rag and check_rag_available():
        try:
            import chromadb
            client = chromadb.PersistentClient(path="./chroma_db")
            collection = client.get_collection("unit_materials")
            results = collection.query(query_texts=[question], n_results=3)
            if results and results['documents']:
                rag_context = "\n\nCOURSE MATERIALS:\n" + "\n\n---\n\n".join(results['documents'][0])
        except:
            pass

    # Use the optimized v4 prompt with calibrated confidence elicitation
    # (0-9 probability scale per Yang et al., 2024; "consider the opposite" per Chhikara et al., 2025)
    prompt = f"""TASK: Answer this question correctly.

QUESTION: {question}

OPTIONS:
{options_text}
{rag_context}
ANALYSIS STEPS:

1. QUESTION TYPE: Which of the following categories best describes this type of question: content recall, routine application, conceptual understanding, analytical reasoning, or strategic integration? Choose one category only.

2. KEY INSIGHT: What core concept or distinction is being tested here?

3. EVALUATE OPTIONS (if more than five options are present, evaluate them all):
   A: [KEEP/ELIMINATE] - why?
   B: [KEEP/ELIMINATE] - why?
   C: [KEEP/ELIMINATE] - why?
   D: [KEEP/ELIMINATE] - why?
   E: [KEEP/ELIMINATE] - why?

4. CALCULATION (if needed): Show your working.

5. FINAL SELECTION: From options marked KEEP, select the single best answer.

6. DOUBT CHECK: Before rating your probability, consider: what is the strongest argument AGAINST your chosen answer? What would make an alternative option correct instead?

=== REQUIRED OUTPUT FORMAT ===
After your analysis, you MUST write these four lines:

ANSWER: [write ONE letter: A, B, C, D, or E] (if there are more than five options choose the correct answer from all the options)
PROBABILITY: [write a single digit 0-9 using the scale below]
REASONING: [write 1-2 sentences explaining why]
DOUBT: [write one sentence about what could make your answer wrong]

PROBABILITY SCALE — rate the probability your answer is correct:
  0 = I am guessing randomly, I have no basis for this answer
  1 = Very unlikely correct, almost certainly wrong
  2 = Unlikely correct, I see major problems with my reasoning
  3 = Somewhat unlikely, notable gaps in my reasoning
  4 = Slightly below even odds, could easily be wrong
  5 = About even odds, roughly a coin flip between options
  6 = Slightly more likely correct than not
  7 = Probably correct, but alternative answers are plausible
  8 = Likely correct with only minor reservations
  9 = Near certain, I would be very surprised if wrong

Do not write anything after the DOUBT line.

Begin your analysis:"""

    try:
        if num_samples == 1:
            # Single sample - deterministic
            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0, 'num_predict': 512}
            )

            text = response['message']['content']

            # Parse response using shared parsing utilities
            ai_answer, confidence, reasoning = parse_llm_response(text)

            is_correct = ai_answer == correct_answer.upper() if correct_answer else None

            return {
                'ai_answer': ai_answer,
                'confidence': confidence,
                'reasoning': reasoning,
                'is_correct': is_correct,
                'correct_answer': correct_answer,
                'full_response': text,
                'used_rag': use_rag and bool(rag_context),
                'samples': 1,
                'consistency': '1/1'
            }

        else:
            # Multi-sample - run multiple times, select MOST COMMON answer
            answers = []
            confidences = []
            reasonings = []
            sample_details = []  # Store details for each sample

            for i in range(num_samples):
                # Update progress in session state for UI to display
                st.session_state.test_sample_progress = f"Sample {i+1}/{num_samples}"

                response = ollama.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0, 'num_predict': 300}  # temp=0 for deterministic
                )

                text = response['message']['content']

                # Parse using shared parsing utilities
                answer = extract_answer(text)
                conf = extract_confidence(text)

                answers.append(answer)
                confidences.append(conf)

                # Store sample details for display
                sample_details.append({
                    'sample_num': i + 1,
                    'answer': answer,
                    'confidence': conf
                })

                if not reasonings:
                    r_text = extract_reasoning(text)
                    if r_text:
                        reasonings.append(r_text[:200])

            # Clear progress
            st.session_state.test_sample_progress = None

            # Select MOST COMMON answer (not highest confidence)
            distribution = dict(Counter(answers))
            valid = {k: v for k, v in distribution.items() if k in 'ABCDE'}

            if valid:
                most_common = max(valid.items(), key=lambda x: x[1])
                final_answer = most_common[0]
                consistency_count = most_common[1]
            else:
                final_answer = "?"
                consistency_count = 0

            is_correct = final_answer == correct_answer.upper() if correct_answer and final_answer != "?" else None

            # Calculate average confidence
            valid_confs = [c for c in confidences if c > 0]
            avg_confidence = sum(valid_confs) / len(valid_confs) if valid_confs else 0

            return {
                'ai_answer': final_answer,
                'confidence': None,  # No single confidence for multi-sample
                'avg_confidence': round(avg_confidence, 1),  # Average across samples
                'reasoning': reasonings[0] if reasonings else "",
                'is_correct': is_correct,
                'correct_answer': correct_answer,
                'full_response': f"Distribution: {distribution}",
                'used_rag': use_rag and bool(rag_context),
                'samples': num_samples,
                'consistency': f"{consistency_count}/{num_samples}",
                'distribution': distribution,
                'sample_details': sample_details  # Per-sample breakdown
            }

    except Exception as e:
        return {'error': str(e)}


def render_test_question_tab(tab_obj):
    """Render the Test Question tab UI."""
    if tab_obj is None:
        return

    with tab_obj:
        st.subheader("Test a Single Question")

        # Show instructions at the top (matching the workflow description)
        with st.expander("Instructions", expanded=True):
            st.markdown("""
            **Steps:**
            1. Type or paste a quiz question with multiple choice options
            2. Mark the correct answer
            3. Click "Test Question" to run the AI against that question
            4. Review the result and see if the AI answered it correctly (and how confident it was)

            **Notes:**
            - Individual questions are tested with your configured LLM
            - No Moodle connection needed
            - Takes 30 seconds to 2 minutes depending on model speed (and number of samples)
            - **Multi-sample mode:** Runs AI multiple times and selects the **most common answer**
            - Consistency shows how often AI chose the same answer across runs (e.g., "7/10" = 70% agreement)
            """)

        # Initialize session state for this tab
        if 'test_question_result' not in st.session_state:
            st.session_state.test_question_result = None

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("### Enter Question")

            # Question type selector
            q_type = st.radio("Question Type", ["Multiple Choice", "True/False"], horizontal=True, key="q_type_select")

            question_text = st.text_area(
                "Question text",
                placeholder="Enter the question here...",
                height=100,
                key="test_q_text"
            )

            if q_type == "Multiple Choice":
                st.markdown("### Options")

                opt_cols = st.columns(2)
                with opt_cols[0]:
                    opt_a = st.text_input("A.", key="opt_a")
                    opt_b = st.text_input("B.", key="opt_b")
                    opt_c = st.text_input("C.", key="opt_c")
                with opt_cols[1]:
                    opt_d = st.text_input("D.", key="opt_d")
                    opt_e = st.text_input("E.", key="opt_e")

                correct = st.selectbox("Correct answer", ["A", "B", "C", "D", "E"], key="correct_ans")
            else:
                # True/False
                opt_a = "True"
                opt_b = "False"
                opt_c = opt_d = opt_e = ""
                correct = st.selectbox("Correct answer", ["A (True)", "B (False)"], key="correct_tf")
                correct = "A" if "True" in correct else "B"

            st.divider()

            # Test options
            test_cols = st.columns(4)
            with test_cols[0]:
                # Get installed models for test dropdown
                test_model_options = list(AVAILABLE_MODELS.keys())
                try:
                    import ollama as ollama_check
                    models_resp = ollama_check.list()
                    installed = []
                    for m in models_resp.get('models', []):
                        name = m.get('name', '') or m.get('model', '')
                        if name:
                            installed.append(name.lower())
                    # Filter to only installed models
                    test_model_options = [m for m in test_model_options if any(m.split(':')[0].lower() in inst or m.lower() in inst for inst in installed)]
                    if not test_model_options:
                        test_model_options = [DEFAULT_MODEL]  # Fallback
                except:
                    pass

                test_model = st.selectbox(
                    "Model",
                    test_model_options,
                    key="test_model"
                )
            with test_cols[1]:
                test_with_rag = st.checkbox("Use RAG", key="test_rag", help="Include course materials")
            with test_cols[2]:
                num_samples = st.number_input("Samples", min_value=1, max_value=10, value=1, key="num_samples",
                                              help="1=fast single test, 5-10=multi-sample consistency check")
            with test_cols[3]:
                st.write("")  # Spacing
                testing_disabled = st.session_state.is_testing
                test_btn_label = "Testing..." if testing_disabled else "Test Question"
                test_button = st.button(test_btn_label, type="primary", use_container_width=True, disabled=testing_disabled)

            if test_button and question_text:
                options = {}
                if opt_a: options['A'] = opt_a
                if opt_b: options['B'] = opt_b
                if opt_c: options['C'] = opt_c
                if opt_d: options['D'] = opt_d
                if opt_e: options['E'] = opt_e

                if len(options) < 2:
                    st.error("Please provide at least 2 options")
                else:
                    st.session_state.is_testing = True
                    st.session_state.test_sample_progress = None

                    # Show spinner with sample info if multi-sample
                    if num_samples > 1:
                        spinner_text = f"Testing with {test_model} (Sample 1/{num_samples})..."
                    else:
                        spinner_text = f"Testing with {test_model}..."

                    with st.spinner(spinner_text):
                        try:
                            result = test_single_question(
                                question=question_text,
                                options=options,
                                correct_answer=correct,
                                model=test_model,
                                use_rag=test_with_rag,
                                num_samples=num_samples
                            )
                            st.session_state.test_question_result = result
                        finally:
                            st.session_state.is_testing = False
                            st.session_state.test_sample_progress = None

        with col2:
            st.markdown("### Results")

            result = st.session_state.test_question_result

            if result is None:
                st.caption("Enter a question and click Test to see results.")
            elif 'error' in result:
                st.error(f"Error: {result['error']}")
            else:
                # Show result
                is_correct = result.get('is_correct')
                ai_answer = result.get('ai_answer', '?')
                confidence = result.get('confidence')
                consistency = result.get('consistency', '1/1')
                samples = result.get('samples', 1)

                # Main result
                if is_correct:
                    st.error(f"### AI Correct: {ai_answer}")
                    st.caption("This question is VULNERABLE")
                elif is_correct is False:
                    st.success(f"### AI Wrong: {ai_answer}")
                    st.caption("AI answered incorrectly - good resistance!")
                else:
                    st.warning(f"### AI answered: {ai_answer}")
                    st.caption("Could not determine correctness")

                # Multi-sample info
                if samples > 1:
                    st.markdown(f"**Consistency:** {consistency}")
                    consistency_pct = int(consistency.split('/')[0]) / int(consistency.split('/')[1]) * 100
                    st.progress(consistency_pct / 100)
                    st.caption("How often AI chose the same answer across runs")

                    # Show average confidence if available
                    avg_conf = result.get('avg_confidence')
                    if avg_conf:
                        st.markdown(f"**Avg Confidence:** {avg_conf}%")

                    # Distribution histogram
                    if result.get('distribution'):
                        st.markdown("**Answer Distribution:**")
                        dist = result['distribution']
                        max_count = max(dist.values()) if dist else 1

                        # Simple text-based histogram
                        for ans in sorted(dist.keys()):
                            count = dist[ans]
                            bar_len = int((count / max_count) * 10)  # Max 10 chars
                            bar = "\u2588" * bar_len
                            st.text(f"  {ans}: {bar} ({count})")

                    # Per-sample details - proper table
                    if result.get('sample_details'):
                        st.markdown("**Sample Details:**")
                        df = pd.DataFrame(result['sample_details'])
                        df.columns = ['Sample', 'Answer', 'Confidence (%)']
                        st.dataframe(df, use_container_width=True, hide_index=True)

                    # Reasoning (collapsed for multi-sample)
                    if result.get('reasoning'):
                        with st.expander("AI Reasoning (first sample)"):
                            st.write(result['reasoning'])

                # Confidence (single sample only)
                elif confidence is not None:
                    st.markdown(f"**Confidence:** {confidence}%")
                    st.progress(confidence / 100)

                    if confidence >= 80 and not is_correct:
                        st.warning("High confidence but wrong - AI is confidently incorrect!")
                    elif confidence >= 80 and is_correct:
                        st.error("High confidence AND correct - very vulnerable!")

                    # Reasoning and full response only for single sample
                    if result.get('reasoning'):
                        with st.expander("AI Reasoning"):
                            st.write(result['reasoning'])

                    with st.expander("Full AI Response"):
                        st.code(result.get('full_response', 'N/A'))

                # RAG indicator
                if result.get('used_rag'):
                    st.caption("Tested with course materials")
                else:
                    st.caption("Tested with general knowledge only")
