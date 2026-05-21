"""
Sidebar UI rendering.

System status, settings (model, samples, course materials),
progress steps, help links.

Extracted from App.py — no logic changes.
"""

import streamlit as st

from config import (
    AVAILABLE_MODELS, CHROMA_DB_PATH,
    get_rag_collection_name, get_display_name
)
from app_checks import (
    check_chrome, check_ollama, check_stack, get_installed_models,
    launch_chrome_debug, is_debug_port_open, VISION_MODEL_KEYWORDS
)
from app_rag import get_all_courses, get_course_files, process_uploaded_files


def render_sidebar():
    """Render the full sidebar UI."""
    with st.sidebar:

        # Compact spacing for the entire sidebar
        st.markdown("""<style>
        section[data-testid="stSidebar"] .block-container { padding-top: 0.5rem; }
        section[data-testid="stSidebar"] .stMarkdown p { margin-bottom: 0; line-height: 1.3; }
        section[data-testid="stSidebar"] .stAlert { padding: 0.4rem 0.75rem; margin-bottom: 0.25rem; }
        section[data-testid="stSidebar"] .stCaption { margin-bottom: 0.1rem; }
        </style>""", unsafe_allow_html=True)

        # ── SYSTEM STATUS ──
        st.markdown("**SYSTEM STATUS**")

        text_model_ok, vision_ok, installed_models = check_ollama()

        # Auto-select the first available text model when current default isn't installed
        if installed_models and st.session_state.model not in [m.lower() for m in installed_models]:
            text_names = [
                m for m in installed_models
                if not any(kw in m.split(':')[0].lower() for kw in VISION_MODEL_KEYWORDS)
            ]
            if text_names:
                st.session_state.model = text_names[0]

        # 1. AI status
        if text_model_ok:
            st.success(f"AI Ready: {st.session_state.model}")
        else:
            st.error("AI Models Missing")
            st.caption("Run: `ollama pull llama3.2:3b`")
            if st.button("Refresh", key="sidebar_refresh_ai"):
                get_installed_models.clear()
                st.rerun()

        # 2. Workflow status
        workflow = st.session_state.use_rag_mode
        if st.session_state.get('test_question_mode', False) and workflow is None:
            st.info("Workflow: Test a Single Question")
        elif workflow is None:
            st.info("Workflow: None selected")
        elif workflow:
            st.info("Workflow: Complete Assessment")
        else:
            st.info("Workflow: Baseline Scan")

        # 3. Browser status
        if st.session_state.chrome_ok:
            st.success("Browser: Connected")
        else:
            st.warning("Browser: Not connected")
            if workflow is not None:
                col_launch, col_connect = st.columns(2)
                with col_launch:
                    if st.button("Launch Browser", use_container_width=True, key="sidebar_launch_browser"):
                        with st.spinner("Starting Chrome..."):
                            ok, msg = launch_chrome_debug()
                        if ok:
                            st.success(msg)
                            with st.spinner("Connecting..."):
                                conn_ok, url = check_chrome()
                            if conn_ok:
                                st.session_state.chrome_ok = True
                                st.rerun()
                            else:
                                st.info("Chrome running — navigate to your quiz, then click Connect.")
                        else:
                            st.error(msg)
                with col_connect:
                    if st.button("Connect", use_container_width=True, key="sidebar_connect_browser"):
                        with st.spinner("Connecting..."):
                            ok, url = check_chrome()
                        if ok:
                            st.session_state.chrome_ok = True
                            st.rerun()
                        else:
                            st.error("No Moodle quiz page found. Open a quiz attempt first.")

        # Progress steps — only shown after a workflow is selected
        if workflow is not None:
            if workflow:
                # Full scan mode - 3 steps
                steps = [
                    ("First Scan", st.session_state.no_rag_score is not None),
                    ("Second Scan", st.session_state.with_rag_score is not None),
                    ("Report", st.session_state.report_file is not None),
                ]
            else:
                # Basic scan mode - 2 steps
                steps = [
                    ("Scan", st.session_state.no_rag_score is not None),
                    ("Report", st.session_state.report_file is not None),
                ]

            # Find current step (first incomplete)
            current_step = len(steps)  # All done
            for i, (label, done) in enumerate(steps):
                if not done:
                    current_step = i
                    break

            # Display progress with current step highlighted
            for i, (label, done) in enumerate(steps):
                if done:
                    st.markdown(f"~~{label}~~")
                elif i == current_step:
                    st.markdown(f"**\u2192 {label}**")
                else:
                    st.markdown(f"\u25cb {label}")

            # Show completion message
            if current_step == len(steps):
                st.success("All steps complete!")

        # Stack diagnostics (collapsed by default)
        with st.expander("System Diagnostics", expanded=False):
            stack = check_stack()
            all_ok = all(item['ok'] for item in stack)
            for item in stack:
                icon = "+" if item['ok'] else "-"
                st.markdown(f"`[{icon}]` **{item['component']}** — {item['detail']}")
            if all_ok:
                st.caption("All components healthy.")
            else:
                if st.button("Re-check", key="sidebar_recheck_stack"):
                    check_stack.clear()
                    get_installed_models.clear()
                    st.rerun()

        st.divider()

        # ── SETTINGS ──
        st.markdown("**SETTINGS**")

        # AI Model expander
        can_change_model = st.session_state.no_rag_file is None

        with st.expander("AI Model", expanded=False):
            if can_change_model:
                # Build model list from what Ollama actually has installed
                all_installed = get_installed_models()
                text_installed = [
                    m for m in all_installed
                    if not any(kw in m.split(':')[0].lower() for kw in VISION_MODEL_KEYWORDS)
                ]

                sidebar_models = {}
                for name in text_installed:
                    label = AVAILABLE_MODELS.get(name)
                    if label is None:
                        base = name.split(':')[0]
                        tag = name.split(':')[1] if ':' in name else 'latest'
                        label = f"{base.title()} ({tag})"
                    sidebar_models[name] = label

                if not sidebar_models:
                    sidebar_models = {'llama3.2:3b': 'Llama 3.2 3B (not installed)'}

                selected = st.selectbox(
                    "Select model",
                    options=list(sidebar_models.keys()),
                    format_func=lambda x: sidebar_models.get(x, x),
                    index=list(sidebar_models.keys()).index(st.session_state.model) if st.session_state.model in sidebar_models else 0,
                    help="For consistent results, use the same model for both scans",
                    label_visibility="collapsed"
                )
                st.session_state.model = selected

                # Samples selection
                samples_selected = st.slider(
                    "Samples per question",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.num_samples,
                    help="Multiple samples measure AI consistency"
                )
                st.session_state.num_samples = samples_selected

                if samples_selected == 1:
                    st.caption("Fast mode \u2014 1 sample per question")
                else:
                    st.caption(f"Consistency mode \u2014 {samples_selected} samples per question")
            else:
                st.caption(f"Model: **{st.session_state.model}**")
                if st.session_state.num_samples > 1:
                    st.caption(f"Samples: {st.session_state.num_samples}x")
                st.caption("Settings locked during scan session.")

        # Course Materials expander
        with st.expander("Course Materials", expanded=False):
            all_courses = get_all_courses()

            if all_courses:
                # Course selector
                course_options = [c['display_name'] for c in all_courses]
                current_idx = 0
                for i, name in enumerate(course_options):
                    if name == st.session_state.selected_rag_collection:
                        current_idx = i
                        break
                new_selection = st.selectbox(
                    "Active course:",
                    course_options,
                    index=current_idx,
                    key="sidebar_course_selector"
                )
                if new_selection != st.session_state.selected_rag_collection:
                    st.session_state.selected_rag_collection = new_selection
                    st.rerun()

                # Show file count for selected course
                selected_coll = next((c for c in all_courses if c['display_name'] == new_selection), None)
                if selected_coll:
                    st.caption(f"{selected_coll['file_count']} file(s), {selected_coll['segment_count']} segments")
            else:
                st.caption("No courses created yet.")

            # New course creation
            new_name = st.text_input("New course name:", placeholder="e.g., PSYC101", key="sidebar_new_course_name")
            if st.button("Create Course", key="sidebar_create_course", disabled=not new_name, use_container_width=True):
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
                    internal_name = get_rag_collection_name(new_name)
                    client.get_or_create_collection(name=internal_name)
                    st.session_state.selected_rag_collection = new_name
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

            # Upload files
            sidebar_uploaded = st.file_uploader(
                "Upload materials",
                type=['txt', 'md', 'pdf'],
                accept_multiple_files=True,
                key="sidebar_rag_upload"
            )
            if sidebar_uploaded:
                target_course = st.session_state.selected_rag_collection
                target_internal = get_rag_collection_name(target_course)
                if st.button(f"Upload to {target_course}", type="primary", key="sidebar_upload_btn", use_container_width=True):
                    progress_area = st.empty()
                    with progress_area.container():
                        progress_text = st.empty()
                        total = process_uploaded_files(sidebar_uploaded, target_internal, progress_text)
                        if total > 0:
                            progress_text.success(f"Added {len(sidebar_uploaded)} file(s)")
                            st.rerun()

        st.divider()

        # ── START OVER ──
        if st.button("Start Over", use_container_width=True, key="sidebar_start_over"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        st.divider()

        # ── HELP ──
        st.markdown("**HELP**")
        st.markdown(
            '<a href="https://github.com/amunozUOW/QVAF#readme" target="_blank">Documentation</a>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<a href="https://github.com/amunozUOW/QVAF#troubleshooting" target="_blank">Troubleshooting</a>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<a href="https://github.com/amunozUOW/QVAF" target="_blank">GitHub Repository</a>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<a href="https://github.com/amunozUOW/QVAF/issues" target="_blank">Report an Issue</a>',
            unsafe_allow_html=True
        )
