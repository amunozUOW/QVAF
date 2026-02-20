"""
Main content tab rendering functions.

Tab 0: Home/Instructions
Tab 2: First Scan / Baseline Scan
Tab 3: Second Scan (RAG-enhanced)
Tab 4: Results

Extracted from App.py — no logic changes.
"""

import json
import os
import re
import subprocess
import streamlit as st
from pathlib import Path

from config import DASHBOARDS_DIR, get_rag_collection_name
from app_checks import log, clear_log
from app_scanning import run_quiz, scrape_results, save_results, merge_attempts
from app_rag import get_course_files


def render_home_tab(tab_obj):
    """Render Tab 0: Instructions / Home."""
    if tab_obj is None:
        return

    with tab_obj:
        st.subheader("How to Use the Quiz Vulnerability Assessment Framework")

        st.markdown("""
        This tool tests how well your quiz resists AI-assisted cheating. Choose one of the workflows below:
        """)

        # Test Question Tab
        with st.expander("Testing a single question, no setup needed (2-3 minutes)", expanded=False):
            st.markdown("""

            **Steps:**
            1. Type or paste a quiz question with multiple choice options
            2. Mark the correct answer
            3. Click "Test Single Question" to run the AI against that question
            4. Review the result and see if the AI answered it correctly (and how confident it was in providing the answer)

            **Notes:**
            - Individual questions are tested with your configured LLM
            - No Moodle connection needed
            - Takes 30 seconds to 2 minutes depending on model speed (and number of samples requested)
            - Basic analysis is provided
            """)

            if st.button("Click here to test a single question \u2192", use_container_width=True, key="nav_test_q", type="primary"):
                st.session_state.test_question_mode = True
                st.session_state.navigate_to = 'test_question'
                st.session_state._nav_pending = False  # Reset so two-phase nav starts fresh
                st.rerun()

        # Baseline Scan Tab
        with st.expander("Baseline Scan: Quiz without Course Materials (5-15 minutes)", expanded=False):
            st.markdown("""

            **Steps:**
            1. Open your Moodle quiz in Chrome and start an attempt (or start preview)
            2. Make sure the first question is visible
            3. Click the **Connect** button and make sure browser is connected
            4. Click the **Scan** button
            5. The scanner will:
               - Read each question from your quiz
               - Ask the AI to answer each question using only general knowledge
               - Automatically submit answers to Moodle
            6. When complete, submit the answers and navigate to the results page in Moodle.
            7. Click the collect results button, a basic dashboard should appear showing some basic metrics. For a full report, click on the **Generate Report** button.

            **What happens:**
            - We test your quiz with AI using only general knowledge (no course materials)
            - This gives you a baseline vulnerability measurement
            - Shows which questions are easiest/hardest for AI to answer
            - Takes 5-15 minutes depending on quiz length and model speed
            """)

            if st.button("Click here to start Baseline Scan \u2192", use_container_width=True, key="nav_baseline", type="primary"):
                st.session_state.use_rag_mode = False
                st.session_state.test_question_mode = False
                st.session_state.navigate_to = 'scan'
                st.session_state._nav_pending = False  # Reset so two-phase nav starts fresh
                st.rerun()

        # Full Scan Tab
        with st.expander("Complete Assessment: With and Without Course Materials (10-30 minutes)", expanded=False):
            st.markdown("""

            This runs two scans to show how course materials affect AI performance:

            **Steps:**
            1. Open your Moodle quiz in Chrome and start an attempt (or start preview)
            2. Make sure the first question is visible
            3. Click the **Connect** button and make sure browser is connected
            4. Go to the **First Scan** tab and click "Start First Scan"
            5. The scanner will:
               - Read each question from your quiz
               - Ask the AI to answer each question using only general knowledge
               - Automatically submit answers to Moodle
            6. When complete, submit the answers and navigate to the results page in Moodle.
            7. Click the collect results button, then start a new quiz attempt/preview in Moodle.

            **Upload Course Materials** (Choose when it fits your workflow):
            - **Option A (Recommended):** Upload materials before First Scan in the **First Scan** tab
            - **Option B:** Do First Scan, then upload before Second Scan in the **Second Scan** tab

            6. Go to the **Second Scan** tab and click "Start Second Scan"
               - Tests your quiz with AI having access to your uploaded course materials
            8. When both scans are complete, go to the **Results** tab and click **Generate Report** for detailed analysis

            **What happens:**
            - First scan: AI answers without any course materials (baseline vulnerability)
            - Second scan: AI answers with full access to your course materials
            - Detailed comparison showing which questions become easier when AI has materials
            - Identifies material-specific vulnerabilities vs general knowledge vulnerabilities
            - Takes 10-30 minutes depending on quiz length and course materials
            """)

            if st.button("Click here to start Complete Assessment \u2192", use_container_width=True, key="nav_full", type="primary"):
                st.session_state.use_rag_mode = True
                st.session_state.test_question_mode = False
                st.session_state.navigate_to = 'first_scan'
                st.session_state._nav_pending = False  # Reset so two-phase nav starts fresh
                st.rerun()

            # Settings & Configuration as sub-expander
            st.markdown("---")
            st.markdown("**Preparing Course Materials**")
            st.markdown("""
            Upload materials that students might share with an AI to help with your quiz:
            - Lecture slides or notes
            - Textbook chapters or excerpts
            - Study guides
            - Any required readings

            You can upload these materials at two flexible points:
            - **Before First Scan** in the **First Scan** tab
            - **Before Second Scan** in the **Second Scan** tab
            """)


def render_first_scan_tab(tab_obj):
    """Render Tab 2: First Scan / Baseline Scan."""
    if tab_obj is None:
        return

    with tab_obj:
        left, right = st.columns([3, 2])

        with left:
            # Dynamic title based on scan mode
            if st.session_state.use_rag_mode:
                st.subheader("First Scan: Baseline AI")
                scan_description = "This scan uses only general AI knowledge\u2014no course materials."

                # Instructions for Full Assessment mode
                with st.expander("Instructions", expanded=True):
                    st.markdown("""
                    **Steps:**
                    1. Open your Moodle quiz in Chrome and start an attempt (or start preview)
                    2. Make sure the first question is visible
                    3. Click **Connect to Browser** in the sidebar (if not already connected)
                    4. Click **Start First Scan** below
                    5. When complete, submit the quiz in Moodle and navigate to the results page
                    6. Click **Collect Results** below
                    7. Then proceed to **Second Scan** to test with course materials

                    **What happens:** AI answers using only general knowledge (no course materials)
                    """)
            else:
                st.subheader("Scan: AI Vulnerability Test")
                scan_description = "This scan tests how well AI can answer your quiz using general knowledge."

                # Instructions for Baseline Scan mode
                with st.expander("Instructions", expanded=True):
                    st.markdown("""
                    **Steps:**
                    1. Open your Moodle quiz in Chrome and start an attempt (or start preview)
                    2. Make sure the first question is visible
                    3. Click **Connect to Browser** in the sidebar (if not already connected)
                    4. Click **Start Scan** below
                    5. When complete, submit the quiz in Moodle and navigate to the results page
                    6. Click **Collect Results** below
                    7. Click **Generate Report** for detailed analysis

                    **What happens:** AI answers using only general knowledge\u2014this gives you a baseline vulnerability measurement
                    """)

            # STATE 1: Not connected
            if not st.session_state.chrome_ok:
                st.warning("**Connect to Chrome first** to start scanning.")
                st.caption("Use the **Connect to Browser** button in the sidebar.")

            # STATE 2: Scan complete with results
            elif st.session_state.no_rag_score:
                score = st.session_state.no_rag_score

                st.success("\u2713 Scan complete")

                col1, col2, col3 = st.columns(3)
                col1.metric("AI Score", f"{score['percentage']}%")
                col2.metric("Correct", f"{score['correct']}/{score['total']}")
                avg_conf = score.get('avg_confidence', 0)
                col3.metric("Avg Confidence", f"{avg_conf}%")

                if score['percentage'] >= 50:
                    st.warning("AI can pass with general knowledge alone.")
                else:
                    st.success("\u2713 AI struggles without course materials.")

                st.markdown("---")

                # Auto-navigate based on workflow
                if st.session_state.use_rag_mode:
                    # Full assessment mode: guide to Second Scan
                    if not st.session_state.with_rag_score:
                        st.info("Next: Go to **Second Scan** tab to test with course materials.")
                        st.markdown("")
                        if st.button("Continue to Second Scan", type="primary", use_container_width=True):
                            st.session_state.navigate_to = 'second_scan'
                            st.session_state._nav_pending = False
                            st.rerun()
                    else:
                        # Both scans done
                        st.success("Both scans complete! Go to **Results** for full analysis.")
                        st.markdown("")
                        if st.button("Generate Report", key="gen_report_first_scan_full", type="primary", use_container_width=True):
                            st.session_state.navigate_to = 'results'
                            st.session_state._nav_pending = False
                            st.rerun()
                else:
                    # Basic scan mode: go straight to Results
                    st.success("Ready to generate your analysis report.")
                    st.markdown("")
                    if st.button("View Results", type="primary", use_container_width=True):
                        st.session_state.navigate_to = 'results'
                        st.session_state._nav_pending = False
                        st.rerun()

            # STATE 3: Answers filled, waiting for submission
            elif st.session_state.no_rag_file:
                st.success("\u2713 AI has filled in the answers")

                st.markdown("**Now in Chrome:**")
                st.markdown("1. Review the answers if you'd like\n2. Click **\"Finish attempt\"**\n3. Click **\"Submit all and finish\"**")

                st.markdown("---")

                st.markdown("**When you see the results page in Chrome:**")

                if st.button("Collect Results", key="get1", type="primary", use_container_width=True):
                    with st.spinner("Reading results from Chrome..."):
                        results = scrape_results()
                        score = save_results(st.session_state.no_rag_file, results)
                        st.session_state.no_rag_score = score
                    st.rerun()

            # STATE 4: Ready to start scan
            else:
                st.markdown(f"**{scan_description}**")

                st.markdown("---")

                st.markdown("**Before you start:**\n- Make sure your quiz is open in Chrome\n- Navigate to the **first question**\n- The scanner will fill in answers automatically")

                st.markdown("")

                if st.button("Start Scan", type="primary", use_container_width=True):
                    st.session_state.is_scanning = True
                    clear_log()
                    st.session_state._scan_trigger = 'no_rag'
                    st.rerun()

        with right:
            if st.session_state.get('_scan_trigger') == 'no_rag':
                st.subheader("Scan Progress")
                try:
                    with st.status("Starting scan...", expanded=True) as status:
                        output, q_count = run_quiz(use_rag=False, status_container=status)
                        status.update(label=f"Scan complete \u2014 {q_count} questions answered", state="complete", expanded=False)
                    st.session_state.no_rag_file = output
                except Exception as e:
                    st.error(f"Scan failed: {str(e)}")
                finally:
                    st.session_state.is_scanning = False
                    st.session_state._scan_trigger = None
                st.rerun()


def render_second_scan_tab(tab_obj):
    """Render Tab 3: Second Scan (with RAG)."""
    if tab_obj is None:
        return

    with tab_obj:
        left, right = st.columns([3, 2])

        with left:
            st.subheader("Second Scan: AI + Course Materials")

            # Instructions for Second Scan
            with st.expander("Instructions", expanded=True):
                st.markdown("""
                **Steps:**
                1. Start a **new quiz attempt/preview** in Moodle (make sure the first question is visible)
                2. Upload course materials below if you haven't already
                3. Click **Start Second Scan**
                4. When complete, submit the quiz in Moodle and navigate to the results page
                5. Click **Collect Results** below
                6. Go to **Results** tab and click **Generate Report** for detailed analysis

                **What this tests:** Can someone pass by uploading your lecture notes to an AI?
                This scan gives the AI access to your course materials.
                """)

            # Course materials status
            selected_course = st.session_state.selected_rag_collection
            selected_internal = get_rag_collection_name(selected_course)
            course_files = get_course_files(selected_internal)

            if course_files:
                total_segments = sum(course_files.values())
                st.success(f"Course: **{selected_course}** \u2014 {len(course_files)} file(s), {total_segments} segments")
            else:
                st.warning(f"No files in **{selected_course}**. Add materials via **Course Materials** in the sidebar.")

            # STATE 1: First scan not done yet
            if not st.session_state.no_rag_score:
                st.warning("**Complete the first scan** before running this one.")
                st.caption("Go to the First Scan tab to test baseline AI performance.")

            # STATE 2: Second scan complete with results
            elif st.session_state.with_rag_score:
                score = st.session_state.with_rag_score
                baseline = st.session_state.no_rag_score
                change = score['percentage'] - baseline['percentage']

                st.success("\u2713 Second scan complete")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("AI Score", f"{score['percentage']}%")
                col2.metric("Correct", f"{score['correct']}/{score['total']}")
                avg_conf = score.get('avg_confidence', 0)
                col3.metric("Avg Confidence", f"{avg_conf}%")
                col4.metric("Change vs Baseline", f"{change:+.0f}%")

                if change > 10:
                    st.warning("Course materials significantly boost AI performance.")
                elif change < 0:
                    st.success("\u2713 Course materials actually confused the AI!")
                else:
                    st.info("\u2192 Course materials had minimal impact.")

                st.markdown("---")

                # Auto-navigate to Results
                st.success("Both scans complete! Ready to generate full analysis report.")
                st.markdown("")
                if st.button("Generate Report", key="gen_report_second_scan", type="primary", use_container_width=True):
                    st.session_state.navigate_to = 'results'
                    st.session_state._nav_pending = False
                    st.rerun()

            # STATE 3: Answers filled, waiting for submission
            elif st.session_state.with_rag_file:
                st.success("\u2713 AI has filled in the answers (with course materials)")

                st.markdown("**Now in Chrome:**")
                st.markdown("1. Review the answers if you'd like\n2. Click **\"Finish attempt\"**\n3. Click **\"Submit all and finish\"**")

                st.markdown("---")

                st.markdown("**When you see the results page in Chrome:**")

                if st.button("Collect Results", key="get2", type="primary", use_container_width=True):
                    with st.spinner("Reading results from Chrome..."):
                        results = scrape_results()
                        score = save_results(st.session_state.with_rag_file, results)
                        st.session_state.with_rag_score = score
                    st.rerun()

            # STATE 4: Ready to start second scan
            else:
                st.markdown("---")

                st.markdown("**Before you start:**\n- Start a **new quiz attempt** in Moodle\n- Navigate to the **first question**\n- The scanner will use course materials to answer")

                st.markdown("")

                if st.button("Start Second Scan", type="primary", use_container_width=True):
                    st.session_state.is_scanning = True
                    clear_log()
                    st.session_state._scan_trigger = 'with_rag'
                    st.rerun()

        with right:
            if st.session_state.get('_scan_trigger') == 'with_rag':
                st.subheader("Scan Progress")
                try:
                    with st.status("Starting scan with course materials...", expanded=True) as status:
                        output, q_count = run_quiz(use_rag=True, status_container=status)
                        status.update(label=f"Scan complete \u2014 {q_count} questions answered with course materials", state="complete", expanded=False)
                    st.session_state.with_rag_file = output
                except Exception as e:
                    st.error(f"Scan failed: {str(e)}")
                finally:
                    st.session_state.is_scanning = False
                    st.session_state._scan_trigger = None
                st.rerun()


def render_results_tab(tab_obj):
    """Render Tab 4: Results."""
    if tab_obj is None:
        return

    with tab_obj:
        # Determine what results we have based on scan mode
        is_full_scan_mode = st.session_state.use_rag_mode
        has_baseline = st.session_state.no_rag_score is not None
        has_enhanced = st.session_state.with_rag_score is not None

        # Check if we have enough data to show results
        if is_full_scan_mode and not has_enhanced:
            st.subheader("Results")
            if not has_baseline:
                st.info("Complete both scans to see your results here.")
                st.caption("Start with the **First Scan** tab.")
            else:
                st.info("Complete the second scan to see your full comparison results.")
                st.caption("Go to the **Second Scan** tab to continue.")
        elif not is_full_scan_mode and not has_baseline:
            st.subheader("Results")
            st.info("Complete the scan to see your results here.")
            st.caption("Go to the **Scan** tab to test your quiz.")
        else:
            # We have results to show
            baseline = st.session_state.no_rag_score

            # Summary metrics - different display for single vs full scan
            st.subheader("Summary")

            if is_full_scan_mode and has_enhanced:
                # Full scan mode - show comparison
                enhanced = st.session_state.with_rag_score
                best = max(baseline['percentage'], enhanced['percentage'])

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Baseline Score", f"{baseline['percentage']}%", help="AI with general knowledge only")
                col2.metric("Enhanced Score", f"{enhanced['percentage']}%", help="AI with course materials")
                col3.metric("Materials Effect", f"{enhanced['percentage'] - baseline['percentage']:+.0f}%")

                if best < 40:
                    col4.metric("Risk Level", "LOW", help="AI struggles with this quiz")
                elif best < 60:
                    col4.metric("Risk Level", "MEDIUM", help="AI can nearly pass")
                else:
                    col4.metric("Risk Level", "HIGH", help="AI can pass this quiz")
            else:
                # Basic scan mode - show single result
                score = baseline['percentage']

                col1, col2, col3 = st.columns(3)
                col1.metric("AI Score", f"{score}%", help="AI with general knowledge only")
                col2.metric("Correct Answers", f"{baseline['correct']}/{baseline['total']}")

                if score < 40:
                    col3.metric("Risk Level", "LOW", help="AI struggles with this quiz")
                elif score < 60:
                    col3.metric("Risk Level", "MEDIUM", help="AI can nearly pass")
                else:
                    col3.metric("Risk Level", "HIGH", help="AI can pass this quiz")

                st.info("""
                **Basic Scan Complete!** This shows how well AI performs with general knowledge only.

                For a more comprehensive analysis that shows how course materials affect AI performance,
                go to the **Instructions** tab and choose **Complete Assessment**.
                """)

            st.divider()

            # Report generation or display
            if st.session_state.report_file:
                # Dashboard is in DASHBOARDS_DIR with base name
                report_path = Path(st.session_state.report_file)
                base_name = report_path.stem.replace('_analysis_report', '').replace('_vulnerability_report', '')

                # Try multiple possible locations for the dashboard
                possible_paths = [
                    DASHBOARDS_DIR / f"{base_name}_dashboard.html",
                    Path("./output/dashboards") / f"{base_name}_dashboard.html",
                    report_path.parent / f"{base_name}_dashboard.html",
                    Path(".") / f"{base_name}_dashboard.html",
                ]

                dashboard = None
                for path in possible_paths:
                    if path.exists():
                        dashboard = path
                        break

                if dashboard and dashboard.exists():
                    st.subheader("Vulnerability Assessment Report")

                    with open(dashboard) as f:
                        html = f.read()

                    st.components.v1.html(html, height=650, scrolling=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "Download Dashboard (HTML)",
                            html,
                            "quiz_vulnerability_dashboard.html",
                            "text/html",
                            use_container_width=True
                        )
                    with col2:
                        with open(st.session_state.report_file) as f:
                            report_data = f.read()
                        st.download_button(
                            "Download Report (JSON)",
                            report_data,
                            "vulnerability_report.json",
                            "application/json",
                            use_container_width=True
                        )
                else:
                    st.error("Dashboard file not found. Try generating the report again.")
                    with st.expander("Debug info"):
                        st.write(f"Report file: {st.session_state.report_file}")
                        st.write(f"Base name: {base_name}")
                        st.write("Searched paths:")
                        for path in possible_paths:
                            st.write(f"  - {path} (exists: {path.exists()})")

            else:
                left, right = st.columns([3, 2])

                with left:
                    st.subheader("Generate Analysis Report")

                    st.markdown("**Your scan is complete.** Generate a detailed report to:")

                    if is_full_scan_mode:
                        st.markdown("""
                        - Classify each question by cognitive type
                        - Compare baseline vs enhanced AI performance
                        - Identify the most vulnerable questions
                        - Create an interactive dashboard
                        """)
                    else:
                        st.markdown("""
                        - Classify each question by cognitive type
                        - Identify the most vulnerable questions
                        - Create an interactive dashboard
                        """)

                    st.markdown("")

                    if st.button("Generate Report", key="gen_report_results", type="primary", use_container_width=True):
                        st.session_state._scan_trigger = 'report'
                        st.rerun()

                with right:
                    if st.session_state.get('_scan_trigger') == 'report':
                        st.subheader("Report Progress")
                        try:
                            with st.status("Preparing scan results...", expanded=True) as status:
                                # Handle single-scan vs full-scan mode
                                if is_full_scan_mode and st.session_state.with_rag_file:
                                    merged = merge_attempts(
                                        st.session_state.no_rag_file,
                                        st.session_state.with_rag_file,
                                        no_rag_score=st.session_state.no_rag_score,
                                        with_rag_score=st.session_state.with_rag_score
                                    )
                                else:
                                    merged = merge_attempts(
                                        st.session_state.no_rag_file,
                                        None,
                                        no_rag_score=st.session_state.no_rag_score,
                                        with_rag_score=None
                                    )
                                st.session_state.merged_file = merged

                                with open(merged) as f:
                                    merged_data = json.load(f)
                                total_q = len(merged_data.get('questions', []))
                                status.write(f"- Merged {total_q} questions")

                                # Phase 1: Reform agent
                                status.update(label="Phase 1: Classifying questions...")
                                status.write("- Phase 1: Classifying question types...")

                                result1 = subprocess.run(
                                    ['python3', 'reform_agent.py', merged, '--model', st.session_state.model],
                                    capture_output=True,
                                    text=True
                                )

                                for line in result1.stdout.split('\n'):
                                    line = line.strip()
                                    if "Classifying Question" in line:
                                        match = re.search(r'Question (\d+)', line)
                                        if match:
                                            status.write(f"- Classifying Q{match.group(1)}...")
                                    elif "Type:" in line:
                                        qtype = line.split("Type:")[1].strip()
                                        status.write(f"  Type: {qtype}")
                                    elif "Vulnerability:" in line:
                                        vuln = line.split("Vulnerability:")[1].strip()
                                        status.write(f"  Vulnerability: {vuln}")

                                report = merged.replace('.json', '_analysis_report.json')
                                if not os.path.exists(report):
                                    alt_report = merged.replace('.json', '_vulnerability_report.json')
                                    if os.path.exists(alt_report):
                                        report = alt_report
                                    else:
                                        st.error(f"Reform agent failed.\n\nStderr: {result1.stderr[:500] if result1.stderr else 'None'}")
                                        st.stop()

                                status.write("- Phase 1 complete")

                                # Phase 2: Analysis agent
                                status.update(label="Phase 2: Generating dashboard...")
                                status.write("- Phase 2: Generating dashboard...")

                                result2 = subprocess.run(
                                    ['python3', 'analysis_agent.py', report],
                                    capture_output=True,
                                    text=True
                                )

                                for line in result2.stdout.split('\n'):
                                    line = line.strip()
                                    if "Calculating" in line:
                                        status.write("- Calculating statistics...")
                                    elif "LLM interpretation" in line.lower():
                                        status.write("- Generating AI interpretation...")
                                    elif "markdown" in line.lower() and "saved" in line.lower():
                                        status.write("- Markdown summary created")
                                    elif "Dashboard saved" in line:
                                        status.write("- HTML dashboard created")

                                report_path = Path(report)
                                base_name = report_path.stem.replace('_analysis_report', '').replace('_vulnerability_report', '')
                                dashboard = DASHBOARDS_DIR / f"{base_name}_dashboard.html"

                                if dashboard.exists():
                                    status.write("- Phase 2 complete")
                                    status.update(label="Report complete", state="complete", expanded=False)
                                    st.session_state.report_file = report
                                else:
                                    status.update(label="Dashboard generation had issues", state="error", expanded=False)
                                    st.warning(f"Dashboard not generated.\n\nStderr: {result2.stderr[:500] if result2.stderr else 'None'}")
                                    st.session_state.report_file = report
                        finally:
                            st.session_state._scan_trigger = None
                        st.rerun()
