#!/usr/bin/env python3
"""
Quiz Vulnerability Assessment Framework
==========================

A clean interface for testing quiz resistance to AI assistance.

Run: python3 -m streamlit run App.py
"""

# Suppress urllib3 SSL warning on older macOS versions
import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import streamlit as st
import streamlit.components.v1 as components

# ============================================
# CONFIGURATION (imported from central config)
# ============================================

from config import DEFAULT_MODEL, DEFAULT_COLLECTION_NAME, ensure_directories

# Import extracted modules
from app_sidebar import render_sidebar
from app_onboarding import show_onboarding
from app_tabs import render_home_tab, render_first_scan_tab, render_second_scan_tab, render_results_tab
from app_test_question import render_test_question_tab

# Ensure output directories exist
ensure_directories()


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Quiz Vulnerability Assessment Framework",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal custom CSS - let Streamlit handle most styling
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] { background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; }
    /* Hide the Deploy button */
    .stDeployButton { display: none; }
    [data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE
# ============================================

defaults = {
    'no_rag_file': None,
    'with_rag_file': None,
    'merged_file': None,
    'report_file': None,
    'no_rag_score': None,
    'with_rag_score': None,
    'activity': [],
    'chrome_ok': False,
    'model': DEFAULT_MODEL,
    'num_samples': 1,
    'onboarding_complete': False,
    'test_question_result': None,
    'selected_rag_collection': DEFAULT_COLLECTION_NAME,  # Currently selected RAG collection
    'use_rag_mode': None,  # None = not decided, True = use RAG, False = skip RAG (single scan mode)
    'test_question_mode': False,  # True when user selected "Test a Single Question"
    'onboarding_step': 1,  # Track onboarding progress: 1=welcome, 2=rag decision
    'is_scanning': False,  # Track if a scan is currently running
    'is_testing': False,  # Track if a test question is running
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ============================================
# SIDEBAR
# ============================================

render_sidebar()


# ============================================
# ONBOARDING
# ============================================

# Landing page - skip the old onboarding, go straight to Instructions view
# System status will be shown in the sidebar
if not st.session_state.onboarding_complete:
    st.session_state.onboarding_complete = True
    st.rerun()


# ============================================
# MAIN CONTENT
# ============================================

st.title("Quiz Vulnerability Assessment Framework")

# Global CSS: single-spaced text inside expanders and compact list spacing
st.markdown("""<style>
/* Single-spaced text inside expandable sections */
div[data-testid="stExpander"] .stMarkdown p {
    margin-bottom: 0.25rem;
    line-height: 1.4;
}
div[data-testid="stExpander"] .stMarkdown ol,
div[data-testid="stExpander"] .stMarkdown ul {
    margin-top: 0.1rem;
    margin-bottom: 0.25rem;
    line-height: 1.4;
}
div[data-testid="stExpander"] .stMarkdown li {
    margin-bottom: 0.1rem;
    line-height: 1.4;
}

/* Fixed-height scrollable container for st.status() progress panels */
div[data-testid="stStatusWidget"] > div[data-testid="stVerticalBlock"] {
    max-height: 400px;
    overflow-y: auto;
}

/* Compact line spacing inside st.status() progress panels */
div[data-testid="stStatusWidget"] .stMarkdown p {
    margin-bottom: 0.15rem;
    line-height: 1.35;
    font-size: 0.875rem;
}
div[data-testid="stStatusWidget"] .stText {
    margin-bottom: 0.1rem;
    line-height: 1.3;
    font-size: 0.875rem;
}

/* Compact spacing for bullet-point lists in scan tabs */
div[data-testid="stVerticalBlock"] > div > .stMarkdown p {
    line-height: 1.4;
}
</style>""", unsafe_allow_html=True)

# Track workflow state - use_rag_mode determines which tabs are shown
workflow = st.session_state.use_rag_mode
if st.session_state.get('test_question_mode', False) and workflow is None:
    workflow_type = "test_question"
    workflow_name = "Test a Single Question"
elif workflow is None:
    workflow_type = "instructions"  # User hasn't chosen a workflow yet
    workflow_name = "Choose a Workflow"
elif workflow is True:
    workflow_type = "full_assessment"
    workflow_name = "Full Assessment (with and without course materials)"
else:
    workflow_type = "basic_scan"
    workflow_name = "Baseline Scan (general knowledge only)"

# Check for pending navigation - this happens AFTER use_rag_mode is set but BEFORE tabs are built
# Two-phase navigation: on the first render we inject JS and mark _nav_pending.
# On the next rerun (triggered by the JS tab click) we still honour the target,
# then clear it so subsequent reruns go back to normal.
nav_target = st.session_state.get('navigate_to', None)
_was_nav_pending = st.session_state.get('_nav_pending', False)


# Build tab list dynamically based on workflow
# Home always first, Test Question always last (for scan workflows)
if workflow_type == "instructions":
    # User hasn't chosen a workflow yet - show only Home
    labels = ["Home"]
elif workflow_type == "test_question":
    # Test a Single Question mode - Home + Test Question only
    labels = ["Home", "Test Question"]
elif workflow_type == "basic_scan":
    # Baseline Scan path: Results after scan
    labels = ["Home", "Scan", "Results", "Test Question"]
elif workflow_type == "full_assessment":
    # Complete Assessment path: Both scans with results
    labels = ["Home", "First Scan", "Second Scan", "Results", "Test Question"]
else:
    labels = ["Home"]

# Allow navigation buttons to make a tab appear selected
# nav_target was already read above after workflow state was set
target_tab_index = 0  # Default to first tab
if nav_target == 'test_question' and 'Test Question' in labels:
    target_tab_index = labels.index('Test Question')
elif nav_target == 'scan' and 'Scan' in labels:
    target_tab_index = labels.index('Scan')
elif nav_target == 'first_scan' and 'First Scan' in labels:
    target_tab_index = labels.index('First Scan')
elif nav_target == 'second_scan' and 'Second Scan' in labels:
    target_tab_index = labels.index('Second Scan')
elif nav_target == 'results' and 'Results' in labels:
    target_tab_index = labels.index('Results')

# Create tabs
tab_objs = st.tabs(labels)

# Use JavaScript to click the target tab if navigation was requested.
# The JS runs client-side inside a components.html iframe and accesses the parent
# Streamlit document to find and click the correct tab element.
# We use a two-phase approach: on the first render we inject JS and set _nav_pending,
# on the next render (after the JS-triggered rerun) we clear navigate_to.
if nav_target and target_tab_index > 0:
    # Embed a unique counter in the JS so the HTML content changes each time.
    # This prevents Streamlit from serving a cached iframe and ensures the JS re-executes.
    _nav_counter = st.session_state.get('_nav_counter', 0) + 1
    st.session_state._nav_counter = _nav_counter
    _nav_js = f"""
    <!-- nav_id={_nav_counter} -->
    <script>
        (function() {{
            var target = {target_tab_index};
            var doc = window.parent.document;

            function clickTab() {{
                var tabs = doc.querySelectorAll('[data-baseweb="tab"]');
                if (tabs && tabs.length > target) {{
                    tabs[target].click();
                    return true;
                }}
                return false;
            }}

            // Try immediately, then retry with increasing delays
            if (!clickTab()) {{
                var delays = [50, 100, 200, 400, 700, 1000, 1500, 2000];
                delays.forEach(function(d) {{
                    setTimeout(clickTab, d);
                }});
            }}
        }})();
    </script>
    """
    components.html(_nav_js, height=0)
    # Mark that we've injected JS — don't clear navigate_to yet
    st.session_state._nav_pending = True

def _get_tab_obj(name):
    try:
        return tab_objs[labels.index(name)]
    except Exception:
        return None

tab0 = _get_tab_obj('Home')
tab2 = _get_tab_obj('First Scan') or _get_tab_obj('Scan')
tab3 = _get_tab_obj('Second Scan')
tab4 = _get_tab_obj('Results')
tab5 = _get_tab_obj('Test Question')

# Two-phase clearing of navigation target:
# Phase 1 (above): JS was injected, _nav_pending is set, navigate_to is kept
# Phase 2 (this render): If _nav_pending was True at start of this render, clear both now
if _was_nav_pending:
    st.session_state._nav_pending = False
    if 'navigate_to' in st.session_state:
        st.session_state.navigate_to = None
elif 'navigate_to' in st.session_state and not nav_target:
    # No navigation was requested — ensure clean state
    st.session_state.navigate_to = None


# ============================================
# RENDER TABS
# ============================================

render_home_tab(tab0)
render_first_scan_tab(tab2)
render_second_scan_tab(tab3)
render_results_tab(tab4)
render_test_question_tab(tab5)

# Footer
st.divider()
st.caption("Quiz Vulnerability Assessment Framework")
