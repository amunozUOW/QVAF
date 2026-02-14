"""
Onboarding / welcome wizard UI.

3-step wizard for first-time users. Currently auto-skipped
but preserved for future use.

Extracted from App.py — no logic changes.
"""

import os
import streamlit as st

from config import CHROMA_DB_PATH, DEFAULT_COLLECTION_NAME, get_rag_collection_name
from app_checks import check_chrome, check_ollama
from app_test_question import test_single_question


def show_onboarding():
    """Show welcome screen for new users - multi-step onboarding"""

    step = st.session_state.onboarding_step

    # ========== STEP 1: Welcome & System Check ==========
    if step == 1:
        st.markdown("# Welcome to Quiz Vulnerability Assessment Framework")
        st.markdown("#### Test how resistant your quizzes are to AI-assisted cheating")

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 1. Connect")
            st.markdown("Open your quiz in Chrome and connect the scanner to read the questions.")

        with col2:
            st.markdown("### 2. Test")
            st.markdown("Run AI against your quiz to see how it performs.")

        with col3:
            st.markdown("### 3. Analyze")
            st.markdown("Get a detailed report showing which questions are vulnerable.")

        st.markdown("---")

        # System checks
        st.markdown("### System Status")

        col1, col2 = st.columns(2)

        with col1:
            text_model_ok, vision_ok = check_ollama()
            if text_model_ok:
                st.success("\u2713 AI Models Ready")
            else:
                st.error("\u2717 AI Models Missing")
                st.caption("Run: `ollama pull llama3:8b`")

        with col2:
            chrome_ok, _ = check_chrome()
            if chrome_ok:
                st.success("\u2713 Browser Connected")
                st.session_state.chrome_ok = True
            else:
                st.info("\u25cb Waiting for browser...")
                if st.button("Check Now", key="check_browser_onboarding"):
                    st.rerun()

        # Add refresh button if models are missing
        if not text_model_ok:
            st.markdown("---")
            st.markdown("### Installation")
            st.markdown("""
            **To install AI models**, open Terminal and run:
            ```bash
            ollama pull llama3:8b
            ```
            Then click refresh:
            """)
            if st.button("Refresh Status", use_container_width=False):
                check_ollama.clear()
                st.rerun()

        # ===== Test a Single Question Section (available immediately if AI is ready) =====
        if text_model_ok:
            st.markdown("---")
            st.markdown("### Testing a Single Question")
            st.markdown("Try testing a question right now \u2014 no browser setup needed.")

            with st.expander("Test a Question", expanded=False):
                # Simplified test question interface
                test_q_text = st.text_area(
                    "Paste a question",
                    placeholder="Enter a multiple choice question...",
                    height=80,
                    key="onboard_test_q"
                )

                opt_cols = st.columns(2)
                with opt_cols[0]:
                    onboard_opt_a = st.text_input("A.", key="onboard_opt_a")
                    onboard_opt_b = st.text_input("B.", key="onboard_opt_b")
                with opt_cols[1]:
                    onboard_opt_c = st.text_input("C.", key="onboard_opt_c")
                    onboard_opt_d = st.text_input("D.", key="onboard_opt_d")

                test_col1, test_col2 = st.columns([1, 1])
                with test_col1:
                    onboard_correct = st.selectbox("Correct answer", ["A", "B", "C", "D"], key="onboard_correct")
                with test_col2:
                    onboard_samples = st.number_input("Samples", min_value=1, max_value=5, value=1, key="onboard_samples")

                if st.button("Test Question", type="primary", key="onboard_test_btn"):
                    if test_q_text and onboard_opt_a and onboard_opt_b:
                        options = {'A': onboard_opt_a, 'B': onboard_opt_b}
                        if onboard_opt_c: options['C'] = onboard_opt_c
                        if onboard_opt_d: options['D'] = onboard_opt_d

                        with st.spinner(f"Testing ({onboard_samples} sample{'s' if onboard_samples > 1 else ''})..."):
                            result = test_single_question(
                                question=test_q_text,
                                options=options,
                                correct_answer=onboard_correct,
                                model=st.session_state.model,
                                use_rag=False,
                                num_samples=onboard_samples
                            )

                        # Show result
                        if 'error' in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            is_correct = result.get('is_correct')
                            ai_answer = result.get('ai_answer', '?')

                            if is_correct:
                                st.error(f"AI answered correctly: **{ai_answer}** \u2014 This question may be vulnerable!")
                            else:
                                st.success(f"AI answered incorrectly: **{ai_answer}** \u2014 Good resistance!")

                            if onboard_samples > 1:
                                st.caption(f"Consistency: {result.get('consistency', 'N/A')} | Avg confidence: {result.get('avg_confidence', 'N/A')}%")
                            elif result.get('confidence'):
                                st.caption(f"Confidence: {result['confidence']}%")
                    else:
                        st.warning("Please enter a question and at least options A and B.")

        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if text_model_ok:
                if st.button("Continue to Full Scanner \u2192", type="primary", use_container_width=True):
                    st.session_state.onboarding_step = 2
                    st.rerun()
            else:
                st.warning("Please install AI models before continuing.")

    # ========== STEP 2: Scan Mode Selection ==========
    elif step == 2:
        st.markdown("# Choose Your Scan Type")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Baseline Scan: Quiz without Course Materials")
            st.markdown("""
            Test your quiz against AI using only general knowledge.

            - Baseline vulnerability measurement
            - AI uses general knowledge only
            - Results show which questions are vulnerable
            """)

            st.markdown("")
            if st.button("Start Baseline Scan", use_container_width=True, key="basic_scan"):
                st.session_state.use_rag_mode = False
                st.session_state.onboarding_complete = True
                st.rerun()

        with col2:
            st.markdown("### Full Scan (Recommended)")
            st.markdown("""
            **Comprehensive vulnerability test**

            Tests your quiz twice:
            1. AI with general knowledge only
            2. AI with your course materials

            - Shows if uploading notes helps cheating
            - More detailed comparison analysis
            """)

            st.markdown("")
            if st.button("Set Up Course Materials", use_container_width=True, key="full_scan"):
                st.session_state.onboarding_step = 3
                st.rerun()

            st.caption("You'll upload lecture notes, slides, or textbook excerpts that students might use with AI.")

        st.markdown("---")

        # Back button
        if st.button("\u2190 Back"):
            st.session_state.onboarding_step = 1
            st.rerun()

    # ========== STEP 3: RAG Setup ==========
    elif step == 3:
        st.markdown("# Add Course Materials")
        st.markdown("#### Upload the materials students might use to cheat")

        st.markdown("---")

        st.info("""
        **What to upload:** Lecture slides, notes, textbook chapters, study guides \u2014 anything
        a student might upload to an AI chatbot to get help with your quiz.
        """)

        # File uploader
        uploaded_files = st.file_uploader(
            "Drop files here (PDF, TXT, or MD)",
            type=['txt', 'md', 'pdf'],
            accept_multiple_files=True,
            key="onboarding_rag_upload"
        )

        # Show current status
        rag_count = 0
        try:
            import chromadb
            if os.path.exists(str(CHROMA_DB_PATH)):
                client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
                internal_name = get_rag_collection_name(DEFAULT_COLLECTION_NAME)
                try:
                    coll = client.get_collection(internal_name)
                    rag_count = coll.count()
                except:
                    pass
        except:
            pass

        if rag_count > 0:
            st.success(f"\u2713 {rag_count} text chunks loaded from course materials")

        if uploaded_files:
            if st.button("Process Files", type="primary", use_container_width=True):
                with st.spinner("Processing files..."):
                    try:
                        import chromadb
                        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
                        internal_name = get_rag_collection_name(DEFAULT_COLLECTION_NAME)
                        coll = client.get_or_create_collection(name=internal_name)

                        total_chunks = 0
                        for uploaded_file in uploaded_files:
                            content = ""
                            if uploaded_file.name.endswith('.pdf'):
                                try:
                                    from pypdf import PdfReader
                                    import io
                                    reader = PdfReader(io.BytesIO(uploaded_file.read()))
                                    content = "\n".join([page.extract_text() for page in reader.pages])
                                except ImportError:
                                    st.warning(f"Skipped {uploaded_file.name} - PDF support requires: pip install pypdf")
                                    continue
                            else:
                                content = uploaded_file.read().decode('utf-8', errors='ignore')

                            if not content.strip():
                                continue

                            # Chunk content
                            chunk_size, overlap = 1000, 200
                            chunks = []
                            start = 0
                            while start < len(content):
                                chunk = content[start:start + chunk_size]
                                if chunk.strip():
                                    chunks.append(chunk)
                                start += chunk_size - overlap

                            if chunks:
                                base_id = f"onboard_{uploaded_file.name}".replace(" ", "_")[:50]
                                coll.add(
                                    documents=chunks,
                                    ids=[f"{base_id}_chunk_{i}" for i in range(len(chunks))],
                                    metadatas=[{"source": uploaded_file.name, "chunk": i} for i in range(len(chunks))]
                                )
                                total_chunks += len(chunks)

                        if total_chunks > 0:
                            st.success(f"\u2713 Added {total_chunks} chunks from {len(uploaded_files)} file(s)!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error processing files: {e}")

        st.markdown("---")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("\u2190 Back"):
                st.session_state.onboarding_step = 2
                st.rerun()

        with col2:
            if st.button("Skip for Now", use_container_width=True):
                st.session_state.use_rag_mode = False
                st.session_state.onboarding_complete = True
                st.rerun()

        with col3:
            if rag_count > 0:
                if st.button("Start Full Scan \u2192", type="primary", use_container_width=True):
                    st.session_state.use_rag_mode = True
                    st.session_state.onboarding_complete = True
                    st.rerun()
            else:
                st.button("Start Full Scan \u2192", type="primary", use_container_width=True, disabled=True)
                st.caption("Add materials first")
