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
import json
import os
import re
import glob
import pandas as pd
from datetime import datetime
from pathlib import Path

# ============================================
# CONFIGURATION (imported from central config)
# ============================================

from config import (
    PROJECT_ROOT, OUTPUT_DIR, RAW_ATTEMPTS_DIR, REPORTS_DIR, DASHBOARDS_DIR,
    MOODLE_URL_PATTERNS, EXCLUDE_URL_PATTERNS, AVAILABLE_MODELS,
    DEFAULT_MODEL, CHROMA_DB_PATH, ensure_directories,
    RAG_COLLECTION_PREFIX, DEFAULT_COLLECTION_NAME,
    get_rag_collection_name, get_display_name
)

# Ensure output directories exist
ensure_directories()


def find_moodle_page(browser):
    """
    Find the Moodle quiz page from all available browser pages.
    Filters out internal Chrome pages and finds actual quiz pages.
    """
    all_pages = []
    
    # Collect ALL pages from ALL contexts
    for context in browser.contexts:
        for page in context.pages:
            try:
                url = page.url
                title = page.title() if url else ""
            except:
                url = "unknown"
                title = ""
            all_pages.append({'page': page, 'url': url, 'title': title})
    
    # Filter out internal Chrome pages
    candidate_pages = []
    for p in all_pages:
        url_lower = p['url'].lower()
        if any(excl in url_lower for excl in EXCLUDE_URL_PATTERNS):
            continue
        candidate_pages.append(p)
    
    # Look for Moodle/LMS pages
    for p in candidate_pages:
        url_lower = p['url'].lower()
        title_lower = p['title'].lower()
        
        # Check URL patterns
        if any(pattern in url_lower for pattern in MOODLE_URL_PATTERNS):
            return p['page'], p['url']
        
        # Check title for quiz-related keywords
        if any(kw in title_lower for kw in ['quiz', 'assessment', 'exam', 'test']):
            return p['page'], p['url']
    
    # No Moodle page found - return first non-internal page if available
    if candidate_pages:
        return candidate_pages[0]['page'], candidate_pages[0]['url']
    
    # Fallback to first page (will show error to user)
    if all_pages:
        return all_pages[0]['page'], all_pages[0]['url']
    
    raise Exception("No pages found in Chrome")


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
    'model': 'llama3:8b',
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
# ACTIVITY LOG
# ============================================

def log(message, icon="•"):
    """Add timestamped message to activity log"""
    st.session_state.activity.append({
        'time': datetime.now().strftime("%H:%M:%S"),
        'icon': icon,
        'text': message
    })
    st.session_state.activity = st.session_state.activity[-50:]  # Keep more entries


def clear_log():
    st.session_state.activity = []


def show_activity():
    """Display activity log in a clean format"""
    if not st.session_state.activity:
        st.caption("Waiting to start...")
        return
    
    # Show most recent first, show more items
    for item in reversed(st.session_state.activity[-20:]):
        st.text(f"{item['icon']} {item['time']}  {item['text']}")


# ============================================
# SYSTEM CHECKS
# ============================================

def check_chrome():
    """Check if Chrome is connected and find Moodle page"""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
            page, url = find_moodle_page(browser)
            return True, url
    except Exception as e:
        return False, str(e)


@st.cache_data(ttl=120)
def check_ollama():
    """Check Ollama models - returns (text_model_ok, vision_model_ok)"""
    try:
        import ollama
        text_model_ok = False
        vision_model_ok = False

        # Get list of installed models
        try:
            models_response = ollama.list()
            installed_models = []
            for m in models_response.get('models', []):
                # Handle both old and new Ollama API formats
                name = m.get('name', '') or m.get('model', '')
                if name:
                    installed_models.append(name.lower())

            # Check for any valid text model (llama3, mistral, gemma2, mixtral)
            text_model_names = ['llama3', 'mistral', 'gemma2', 'mixtral', 'llama3.1', 'llama3.2']
            for model in installed_models:
                base_name = model.split(':')[0]
                if any(txt in base_name for txt in text_model_names):
                    text_model_ok = True
                    break

            # Check for vision model
            for model in installed_models:
                if 'llava' in model.lower():
                    vision_model_ok = True
                    break

        except Exception:
            # Fallback: try direct model check
            try:
                ollama.show('llama3:8b')
                text_model_ok = True
            except: pass
            try:
                ollama.show('llava')
                vision_model_ok = True
            except: pass

        return text_model_ok, vision_model_ok
    except Exception:
        return False, False


# ============================================
# QUIZ FUNCTIONS
# ============================================

def run_quiz(use_rag=False):
    """Run quiz attempt via quiz_browser_enhanced.py"""
    import subprocess

    mode = "--with-rag" if use_rag else "--no-rag"
    model = st.session_state.model
    num_samples = st.session_state.num_samples
    label = "with course materials" if use_rag else "baseline"

    log(f"Starting {label} scan using {model}...", "🚀")
    if num_samples > 1:
        log(f"Sampling: {num_samples} samples per question", "🎲")

    # Build command
    cmd = ['python3', 'quiz_browser_enhanced.py', mode, '--no-wait',
           '--model', model, '--samples', str(num_samples)]

    # Add collection name if using RAG
    if use_rag:
        collection_name = st.session_state.selected_rag_collection
        internal_name = get_rag_collection_name(collection_name)
        cmd.extend(['--collection', internal_name])
        log(f"Using course materials: {collection_name}", "📚")

    existing = set(glob.glob(str(RAW_ATTEMPTS_DIR / "quiz_attempt_*_*.json")))

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    q_count = 0

    for line in process.stdout:
        line = line.strip()
        if not line:
            continue

        # Parse structured progress messages from quiz_browser_enhanced
        if "[PROGRESS]" in line:
            progress_msg = line.replace("[PROGRESS]", "").strip()

            if "Question" in progress_msg and ":" in progress_msg:
                # New question starting
                q_count += 1
                q_preview = progress_msg.split(":", 1)[1][:40].strip()
                log(f"Q{q_count}: {q_preview}", "❓")

            elif "AI thinking" in progress_msg:
                log(f"AI analyzing Q{q_count}...", "🤔")

            elif "Running" in progress_msg and "samples" in progress_msg:
                log(f"Running multiple samples for Q{q_count}...", "🎲")

            elif "Answer:" in progress_msg:
                # Got an answer with confidence
                match = re.search(r'Answer:\s*([A-E]).*confidence:\s*(\d+)', progress_msg)
                if match:
                    ans, conf = match.groups()
                    if int(conf) >= 80:
                        log(f"Q{q_count} → {ans} (high confidence: {conf}%)", "🎯")
                    else:
                        log(f"Q{q_count} → {ans} (confidence: {conf}%)", "🤖")
                else:
                    log(f"Q{q_count} answered", "✓")

            elif "image" in progress_msg.lower() and "analyzing" in progress_msg.lower():
                # Image found in question
                match = re.search(r'(\d+)\s+image', progress_msg)
                img_count = match.group(1) if match else "1"
                log(f"Q{q_count}: Analyzing {img_count} image(s)...", "🖼️")

            elif "link" in progress_msg.lower() and "following" in progress_msg.lower():
                # Link found in question
                match = re.search(r'(\d+)\s+link', progress_msg)
                link_count = match.group(1) if match else "1"
                log(f"Q{q_count}: Following {link_count} link(s)...", "🔗")

            elif "Found" in progress_msg and "questions" in progress_msg:
                # Starting question answering
                match = re.search(r'Found\s+(\d+)\s+questions', progress_msg)
                if match:
                    total = match.group(1)
                    log(f"Starting scan: {total} questions found", "📋")

            elif "Page complete" in progress_msg:
                # Page finished
                match = re.search(r'(\d+)\s+questions answered', progress_msg)
                if match:
                    answered = match.group(1)
                    log(f"Page complete: {answered} questions answered", "✅")

        # Also parse other useful messages
        elif "INFO BLOCK" in line:
            log("Reading scenario context...", "📋")
        elif "image" in line.lower() and "found" in line.lower():
            log("Analyzing image...", "🖼️")
        elif "link" in line.lower() and "found" in line.lower():
            log("Following link...", "🔗")
        elif "RAG" in line and ("loaded" in line.lower() or "initialized" in line.lower()):
            log("Course materials loaded", "📚")
        elif "Connected" in line or "Found Moodle" in line:
            log("Browser connected", "🌐")
    
    process.wait()

    # Find output file - check multiple possible locations
    suffix = "_with_rag_" if use_rag else "_no_rag_"

    # Check the expected output directory
    new_files = set(glob.glob(str(RAW_ATTEMPTS_DIR / "quiz_attempt_*_*.json"))) - existing
    matches = [f for f in new_files if suffix in f]

    # Also check current directory (fallback location if config import failed)
    if not matches:
        cwd_files = set(glob.glob("quiz_attempt_*_*.json"))
        matches = [f for f in cwd_files if suffix in f]
        if matches:
            # Move file to correct location
            src = sorted(matches)[-1]
            dst = str(RAW_ATTEMPTS_DIR / Path(src).name)
            import shutil
            shutil.move(src, dst)
            matches = [dst]
            log("Moved output to correct location", "📁")

    # Also check project root
    if not matches:
        root_files = set(glob.glob(str(PROJECT_ROOT / "quiz_attempt_*_*.json")))
        matches = [f for f in root_files if suffix in f]
        if matches:
            src = sorted(matches)[-1]
            dst = str(RAW_ATTEMPTS_DIR / Path(src).name)
            import shutil
            shutil.move(src, dst)
            matches = [dst]
            log("Moved output to correct location", "📁")

    if matches:
        output = sorted(matches)[-1]
        log(f"Complete! {q_count} questions answered", "✅")
        return output, q_count
    else:
        log("Error: No output file created", "❌")
        # Provide more diagnostic info
        all_json = glob.glob(str(RAW_ATTEMPTS_DIR / "*.json")) + glob.glob("*.json")
        if all_json:
            log(f"Found JSON files: {len(all_json)}", "🔍")
        raise Exception("Scan failed - no output file")


def scrape_results():
    """Get results from submitted quiz"""
    from playwright.sync_api import sync_playwright
    
    log("Reading quiz results...", "📊")
    results = []
    
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        page, _ = find_moodle_page(browser)
        
        for q in page.query_selector_all('div.que'):
            try:
                r = {}
                qno = q.query_selector('.info .qno')
                r['number'] = qno.inner_text().strip() if qno else "?"
                
                qtext = q.query_selector('.qtext')
                r['question'] = qtext.inner_text().strip() if qtext else ""
                
                classes = q.get_attribute('class') or ''
                if 'correct' in classes and 'incorrect' not in classes:
                    r['is_correct'] = True
                elif 'incorrect' in classes:
                    r['is_correct'] = False
                else:
                    r['is_correct'] = None
                
                right = q.query_selector('.rightanswer')
                r['correct_answer'] = right.inner_text().strip() if right else ""
                
                if r['question']:
                    results.append(r)
            except:
                continue
    
    correct = sum(1 for r in results if r.get('is_correct'))
    total = len([r for r in results if r.get('is_correct') is not None])
    pct = round(correct/total*100) if total else 0
    
    log(f"Score: {correct}/{total} ({pct}%)", "🏆")
    return results


def save_results(attempt_file, results):
    """Save results to attempt file"""
    with open(attempt_file, 'r') as f:
        data = json.load(f)

    data['results'] = results
    correct = sum(1 for r in results if r.get('is_correct'))
    total = len([r for r in results if r.get('is_correct') is not None])

    # Calculate average confidence from questions data
    questions = data.get('questions', [])
    confidences = [q.get('llm_confidence', 0) for q in questions if q.get('llm_confidence')]
    avg_confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0

    data['score'] = {
        'correct': correct,
        'total': total,
        'percentage': round(correct / total * 100, 1) if total else 0,
        'avg_confidence': avg_confidence
    }

    with open(attempt_file, 'w') as f:
        json.dump(data, f, indent=2)

    return data['score']


def merge_attempts(file1, file2=None, no_rag_score=None, with_rag_score=None):
    """Combine scan attempts into single analysis file.

    If file2 is None (basic scan mode), only file1 data is used.
    """
    log("Preparing scan results...", "🔄")

    with open(file1) as f:
        d1 = json.load(f)

    # Handle single-scan mode (file2 is None)
    if file2 is not None:
        with open(file2) as f:
            d2 = json.load(f)
    else:
        d2 = {'questions': [], 'results': []}

    def norm(t):
        return re.sub(r'\s+', ' ', (t or '').lower().strip())[:100]

    def find(target, items, key='text'):
        tn = norm(target)
        for i in items:
            if norm(i.get(key, '') or i.get('question', '')) == tn:
                return i
        return None

    def get_letter(text, opts):
        if not text or not opts:
            return None
        clean = re.sub(r'^The correct answer is:?\s*', '', text, flags=re.IGNORECASE).strip()
        for letter, opt in opts.items():
            if opt.lower().strip() == clean.lower():
                return letter
        return None

    questions = []
    for q in d1['questions']:
        if not q.get('options'):
            continue

        # Get question text - raw files use 'question', not 'text'
        q_text = q.get('question', '') or q.get('text', '')

        # Find matching question in with_rag data (if available)
        q2 = find(q_text, d2['questions'], 'question') if d2['questions'] else None

        correct = 'UNKNOWN'
        results_to_check = [d1.get('results', [])]
        if d2.get('results'):
            results_to_check.append(d2['results'])
        for res in results_to_check:
            m = find(q_text, res, 'question')
            if m and m.get('correct_answer'):
                letter = get_letter(m['correct_answer'], q['options'])
                if letter:
                    correct = letter
                    break

        questions.append({
            'id': len(questions) + 1,
            'question': q.get('text', '') or q.get('question', ''),
            'options': q.get('options', {}),
            'correct_answer': correct,
            'response_without_rag': {
                'answer': q.get('llm_answer', ''),
                'confidence': q.get('llm_confidence', 0),
                'reasoning': q.get('llm_reasoning', '')
            },
            'response_with_rag': {
                'answer': q2.get('llm_answer', '') if q2 else '',
                'confidence': q2.get('llm_confidence', 0) if q2 else 0,
                'reasoning': q2.get('llm_reasoning', '') if q2 else ''
            }
        })

    # Use passed scores if available, otherwise fall back to file data
    merged = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'source': 'web_app',
        'scan_mode': 'full' if file2 else 'basic',  # Track which mode was used
        'no_rag_score': no_rag_score if no_rag_score else d1.get('score', {}),
        'with_rag_score': with_rag_score if with_rag_score else (d2.get('score', {}) if file2 else None),
        'questions': questions
    }

    out = REPORTS_DIR / f"quiz_attempt_{merged['timestamp']}.json"
    with open(out, 'w') as f:
        json.dump(merged, f, indent=2)

    log(f"Prepared {len(questions)} questions", "✅")
    return str(out)


def run_analysis(merged_file):
    """Generate vulnerability report and dashboard with detailed progress"""
    import subprocess
    
    log("Starting vulnerability analysis...", "🔬")
    
    # Load merged file to get question count
    with open(merged_file) as f:
        merged_data = json.load(f)
    total_questions = len(merged_data.get('questions', []))
    log(f"Analyzing {total_questions} questions", "📋")
    
    # Run reform_agent with streaming output
    log("Phase 1: Classifying question types...", "🏷️")
    
    process = subprocess.Popen(
        ['python3', 'reform_agent.py', merged_file, '--model', st.session_state.model],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    question_count = 0
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        
        # Parse reform_agent output
        if "Classifying Question" in line:
            question_count += 1
            log(f"Classifying Q{question_count}/{total_questions}...", "🔍")
        elif "Type:" in line:
            qtype = line.split("Type:")[1].strip()
            log(f"  → Type: {qtype}", "📝")
        elif "Vulnerability:" in line:
            vuln = line.split("Vulnerability:")[1].strip()
            icon = "🔴" if vuln == "HIGH" else "🟡" if vuln == "MODERATE" else "🟢" if vuln == "LOW" else "⚪"
            log(f"  → Vulnerability: {vuln}", icon)
        elif "Generating qualitative" in line.lower():
            log("Generating detailed recommendations...", "💡")
        elif "Question" in line and "analysis" in line.lower():
            # e.g. "Question 3: Generating analysis..."
            match = re.search(r'Question (\d+)', line)
            if match:
                log(f"Writing recommendation for Q{match.group(1)}...", "✍️")
    
    process.wait()
    
    report = merged_file.replace('.json', '_analysis_report.json')
    if not os.path.exists(report):
        # Try alternate filename
        alt_report = merged_file.replace('.json', '_vulnerability_report.json')
        if os.path.exists(alt_report):
            report = alt_report
        else:
            log("Classification failed - no report generated", "❌")
            return None, None
    
    log("Phase 1 complete!", "✅")
    
    # Run analysis_agent with streaming output
    log("Phase 2: Generating dashboard...", "📊")
    
    process = subprocess.Popen(
        ['python3', 'analysis_agent.py', report],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        
        # Parse analysis_agent output
        if "Calculating" in line:
            log("Calculating statistics...", "🔢")
        elif "LLM interpretation" in line.lower():
            log("Generating AI interpretation...", "🤖")
        elif "markdown" in line.lower():
            log("Creating markdown summary...", "📄")
        elif "HTML" in line.lower() or "dashboard" in line.lower():
            log("Building HTML dashboard...", "🎨")
        elif "saved" in line.lower():
            log("Saving files...", "💾")
    
    process.wait()
    
    # Handle both naming conventions
    if '_analysis_report.json' in report:
        dashboard = report.replace('_analysis_report.json', '_dashboard.html')
    else:
        dashboard = report.replace('_vulnerability_report.json', '_dashboard.html')
    
    if os.path.exists(dashboard):
        log("Phase 2 complete!", "✅")
        log("Dashboard ready to view", "🎉")
        return report, dashboard
    
    log("Dashboard generation failed", "❌")
    return report, None


# ========================================
# Helper Functions for Course Materials
# ========================================

def get_all_courses():
    """Get all course material collections from the database."""
    courses = []
    try:
        import chromadb
        if os.path.exists(str(CHROMA_DB_PATH)):
            client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            for coll in client.list_collections():
                if coll.name.startswith(RAG_COLLECTION_PREFIX):
                    display_name = get_display_name(coll.name)
                    # Get file count (unique source files)
                    files = get_course_files(coll.name)
                    file_count = len(files)
                    segment_count = coll.count()
                    courses.append({
                        'name': coll.name,
                        'display_name': display_name,
                        'file_count': file_count,
                        'segment_count': segment_count
                    })
    except:
        pass
    return courses

# Keep old function name as alias for compatibility
get_all_rag_collections = get_all_courses

def get_course_files(collection_name):
    """Get list of source files in a course with their segment counts."""
    files = {}
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        coll = client.get_collection(collection_name)
        results = coll.get(include=['metadatas'])
        for meta in results.get('metadatas', []):
            source = meta.get('source', 'Unknown')
            files[source] = files.get(source, 0) + 1
    except:
        pass
    return files

# Keep old function name as alias for compatibility
get_collection_files = get_course_files

def delete_file_from_course(collection_name, filename):
    """Delete all segments from a specific file in a course."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        coll = client.get_collection(collection_name)
        # Get all IDs where source matches the filename
        results = coll.get(include=['metadatas'])
        ids_to_delete = []
        for i, meta in enumerate(results.get('metadatas', [])):
            if meta.get('source') == filename:
                ids_to_delete.append(results['ids'][i])
        if ids_to_delete:
            coll.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        return 0
    except Exception as e:
        return -1

def clear_course_materials(collection_name):
    """Delete all materials from a course."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        # Delete and recreate the collection to clear it
        client.delete_collection(collection_name)
        client.get_or_create_collection(name=collection_name)
        return True
    except Exception as e:
        return False

def process_uploaded_files(uploaded_files, collection_name, progress_container):
    """Process uploaded files with progress feedback. Returns total segments added."""
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    coll = client.get_or_create_collection(name=collection_name)

    total_segments = 0
    total_files = len(uploaded_files)

    for file_idx, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress_container.markdown(f"**Processing file {file_idx + 1} of {total_files}:** {uploaded_file.name}")

        content = ""
        if uploaded_file.name.endswith('.pdf'):
            try:
                from pypdf import PdfReader
                import io
                pdf_bytes = uploaded_file.read()
                reader = PdfReader(io.BytesIO(pdf_bytes))
                total_pages = len(reader.pages)

                page_texts = []
                for page_idx, page in enumerate(reader.pages):
                    progress_container.markdown(f"**Processing file {file_idx + 1} of {total_files}:** {uploaded_file.name}  \nPage {page_idx + 1} of {total_pages}")
                    page_texts.append(page.extract_text() or "")
                content = "\n".join(page_texts)
            except ImportError:
                progress_container.warning(f"Skipped {uploaded_file.name} - PDF support requires: pip install pypdf")
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
            display_name = get_display_name(collection_name)
            base_id = f"{display_name}_{uploaded_file.name}".replace(" ", "_")[:50]
            coll.add(
                documents=chunks,
                ids=[f"{base_id}_chunk_{i}" for i in range(len(chunks))],
                metadatas=[{"source": uploaded_file.name, "chunk": i} for i in range(len(chunks))]
            )
            total_segments += len(chunks)

    return total_segments


# ============================================
# SIDEBAR
# ============================================

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

    text_model_ok, vision_ok = check_ollama()

    # 1. AI status
    if text_model_ok:
        st.success(f"AI Ready: {st.session_state.model}")
    else:
        st.error("AI Models Missing")
        st.caption("Run: `ollama pull llama3:8b`")
        if st.button("Refresh", key="sidebar_refresh_ai"):
            check_ollama.clear()
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
            if st.button("Connect to Browser", use_container_width=True, key="sidebar_connect_browser"):
                with st.spinner("Connecting..."):
                    ok, url = check_chrome()
                if ok:
                    st.session_state.chrome_ok = True
                    st.rerun()
                else:
                    st.error("Could not connect.")
                    with st.expander("Troubleshooting"):
                        st.markdown("""
**If "Start Scanner" was used to launch the app**, Chrome should connect automatically. Try clicking Connect again.

**Manual setup (if Chrome wasn't started by the app):**

macOS:
```
/Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug
```

Windows:
```
"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=9222 --user-data-dir=%TEMP%\\chrome-debug
```

Then navigate to your quiz and start an attempt.
                        """)

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
                st.markdown(f"✅ ~~{label}~~")
            elif i == current_step:
                st.markdown(f"**→ {label}**")
            else:
                st.markdown(f"○ {label}")

        # Show completion message
        if current_step == len(steps):
            st.success("All steps complete!")

    st.divider()

    # ── SETTINGS ──
    st.markdown("**SETTINGS**")

    # AI Model expander
    can_change_model = st.session_state.no_rag_file is None

    with st.expander("AI Model", expanded=False):
        if can_change_model:
            # Get only installed models
            sidebar_models = dict(AVAILABLE_MODELS)
            try:
                import ollama as ollama_sidebar
                models_resp = ollama_sidebar.list()
                installed_names = []
                for m in models_resp.get('models', []):
                    name = m.get('name', '') or m.get('model', '')
                    if name:
                        installed_names.append(name.lower())

                # Filter to only installed
                sidebar_models = {
                    k: v for k, v in AVAILABLE_MODELS.items()
                    if any(k.split(':')[0].lower() in inst or k.lower() in inst for inst in installed_names)
                }
                if not sidebar_models:
                    sidebar_models = {'llama3:8b': 'Llama 3 8B (not installed)'}
            except:
                pass  # Keep all models if can't connect

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
                st.caption("Fast mode — 1 sample per question")
            else:
                st.caption(f"Consistency mode — {samples_selected} samples per question")
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


# ============================================
# ONBOARDING / WELCOME SCREEN
# ============================================

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
            st.markdown("### 1️⃣ Connect")
            st.markdown("Open your quiz in Chrome and connect the scanner to read the questions.")

        with col2:
            st.markdown("### 2️⃣ Test")
            st.markdown("Run AI against your quiz to see how it performs.")

        with col3:
            st.markdown("### 3️⃣ Analyze")
            st.markdown("Get a detailed report showing which questions are vulnerable.")

        st.markdown("---")

        # System checks
        st.markdown("### System Status")

        col1, col2 = st.columns(2)

        with col1:
            text_model_ok, vision_ok = check_ollama()
            if text_model_ok:
                st.success("✓ AI Models Ready")
            else:
                st.error("✗ AI Models Missing")
                st.caption("Run: `ollama pull llama3:8b`")

        with col2:
            chrome_ok, _ = check_chrome()
            if chrome_ok:
                st.success("✓ Browser Connected")
                st.session_state.chrome_ok = True
            else:
                st.info("○ Waiting for browser...")
                if st.button("🔄 Check Now", key="check_browser_onboarding"):
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
            if st.button("🔄 Refresh Status", use_container_width=False):
                check_ollama.clear()
                st.rerun()

        # ===== Test a Single Question Section (available immediately if AI is ready) =====
        if text_model_ok:
            st.markdown("---")
            st.markdown("### 🧪 Testing a Single Question")
            st.markdown("Try testing a question right now — no browser setup needed.")

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

                if st.button("🧪 Test Question", type="primary", key="onboard_test_btn"):
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
                                st.error(f"⚠️ AI answered correctly: **{ai_answer}** — This question may be vulnerable!")
                            else:
                                st.success(f"✅ AI answered incorrectly: **{ai_answer}** — Good resistance!")

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
                if st.button("Continue to Full Scanner →", type="primary", use_container_width=True):
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
            st.markdown("### 📊 Baseline Scan: Quiz without Course Materials")
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
            st.markdown("### 📚 Full Scan (Recommended)")
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
        if st.button("← Back"):
            st.session_state.onboarding_step = 1
            st.rerun()

    # ========== STEP 3: RAG Setup ==========
    elif step == 3:
        st.markdown("# Add Course Materials")
        st.markdown("#### Upload the materials students might use to cheat")

        st.markdown("---")

        st.info("""
        **What to upload:** Lecture slides, notes, textbook chapters, study guides — anything
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
            st.success(f"✓ {rag_count} text chunks loaded from course materials")

        if uploaded_files:
            if st.button("📥 Process Files", type="primary", use_container_width=True):
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
                            st.success(f"✓ Added {total_chunks} chunks from {len(uploaded_files)} file(s)!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error processing files: {e}")

        st.markdown("---")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("← Back"):
                st.session_state.onboarding_step = 2
                st.rerun()

        with col2:
            if st.button("Skip for Now", use_container_width=True):
                st.session_state.use_rag_mode = False
                st.session_state.onboarding_complete = True
                st.rerun()

        with col3:
            if rag_count > 0:
                if st.button("Start Full Scan →", type="primary", use_container_width=True):
                    st.session_state.use_rag_mode = True
                    st.session_state.onboarding_complete = True
                    st.rerun()
            else:
                st.button("Start Full Scan →", type="primary", use_container_width=True, disabled=True)
                st.caption("Add materials first")


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
tab6 = None  # Settings tab removed - use sidebar instead

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


# ----------------------------------------
# TAB 0: INSTRUCTIONS
# ----------------------------------------

with tab0:
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
        
        if st.button("Click here to test a single question →", use_container_width=True, key="nav_test_q", type="primary"):
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
        
        if st.button("Click here to start Baseline Scan →", use_container_width=True, key="nav_baseline", type="primary"):
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
        
        if st.button("Click here to start Complete Assessment →", use_container_width=True, key="nav_full", type="primary"):
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



    # ----------------------------------------
# ----------------------------------------
# TAB 2: FIRST SCAN
# ----------------------------------------

if tab2 is not None:
    with tab2:
        left, right = st.columns([3, 2])

        with left:
            # Dynamic title based on scan mode
            if st.session_state.use_rag_mode:
                st.subheader("First Scan: Baseline AI")
                scan_description = "This scan uses only general AI knowledge—no course materials."

                # Instructions for Full Assessment mode
                with st.expander("📋 Instructions", expanded=True):
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
                with st.expander("📋 Instructions", expanded=True):
                    st.markdown("""
                    **Steps:**
                    1. Open your Moodle quiz in Chrome and start an attempt (or start preview)
                    2. Make sure the first question is visible
                    3. Click **Connect to Browser** in the sidebar (if not already connected)
                    4. Click **Start Scan** below
                    5. When complete, submit the quiz in Moodle and navigate to the results page
                    6. Click **Collect Results** below
                    7. Click **Generate Report** for detailed analysis

                    **What happens:** AI answers using only general knowledge—this gives you a baseline vulnerability measurement
                    """)

            # STATE 1: Not connected
            if not st.session_state.chrome_ok:
                st.warning("**Connect to Chrome first** to start scanning.")
                st.caption("Use the **Connect to Browser** button in the sidebar.")

            # STATE 2: Scan complete with results
            elif st.session_state.no_rag_score:
                score = st.session_state.no_rag_score

                st.success("✓ Scan complete")

                col1, col2, col3 = st.columns(3)
                col1.metric("AI Score", f"{score['percentage']}%")
                col2.metric("Correct", f"{score['correct']}/{score['total']}")
                avg_conf = score.get('avg_confidence', 0)
                col3.metric("Avg Confidence", f"{avg_conf}%")

                if score['percentage'] >= 50:
                    st.warning("⚠️ AI can pass with general knowledge alone.")
                else:
                    st.success("✓ AI struggles without course materials.")

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
                        if st.button("Generate Report", type="primary", use_container_width=True):
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
                st.success("✓ AI has filled in the answers")

                st.markdown("""
                **Now in Chrome:**
                1. Review the answers if you'd like
                2. Click **"Finish attempt"**
                3. Click **"Submit all and finish"**
                """)

                st.markdown("---")

                st.markdown("**When you see the results page in Chrome:**")

                if st.button("📥 Collect Results", key="get1", type="primary", use_container_width=True):
                    with st.spinner("Reading results from Chrome..."):
                        results = scrape_results()
                        score = save_results(st.session_state.no_rag_file, results)
                        st.session_state.no_rag_score = score
                    st.rerun()

            # STATE 4: Ready to start scan
            else:
                st.markdown(f"**{scan_description}**")

                st.markdown("---")

                st.markdown("**Before you start:**")
                st.markdown("• Make sure your quiz is open in Chrome")
                st.markdown("• Navigate to the **first question**")
                st.markdown("• The scanner will fill in answers automatically")

                st.markdown("")

                # Disable button while scanning
                scan_disabled = st.session_state.is_scanning
                button_label = "⏳ Scanning..." if scan_disabled else "▶️ Start Scan"

                if st.button(button_label, type="primary", use_container_width=True, disabled=scan_disabled):
                    st.session_state.is_scanning = True
                    clear_log()
                    progress_placeholder = st.empty()

                    try:
                        output, q_count = run_quiz(use_rag=False)
                        st.session_state.no_rag_file = output
                        progress_placeholder.success(f"✓ Scan complete! {q_count} questions answered")
                    finally:
                        st.session_state.is_scanning = False
                    st.rerun()

        with right:
            st.subheader("Activity")
            with st.container(height=300):
                show_activity()


    # ----------------------------------------
    # TAB 3: SECOND SCAN (only shown in full scan mode)
# ----------------------------------------

if tab3 is not None:
    with tab3:
        left, right = st.columns([3, 2])

        with left:
            st.subheader("Second Scan: AI + Course Materials")

            # Instructions for Second Scan
            with st.expander("📋 Instructions", expanded=True):
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
                st.success(f"Course: **{selected_course}** — {len(course_files)} file(s), {total_segments} segments")
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

                st.success("✓ Second scan complete")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("AI Score", f"{score['percentage']}%")
                col2.metric("Correct", f"{score['correct']}/{score['total']}")
                avg_conf = score.get('avg_confidence', 0)
                col3.metric("Avg Confidence", f"{avg_conf}%")
                col4.metric("Change vs Baseline", f"{change:+.0f}%")

                if change > 10:
                    st.warning("⚠️ Course materials significantly boost AI performance.")
                elif change < 0:
                    st.success("✓ Course materials actually confused the AI!")
                else:
                    st.info("→ Course materials had minimal impact.")

                st.markdown("---")

                # Auto-navigate to Results
                st.success("Both scans complete! Ready to generate full analysis report.")
                st.markdown("")
                if st.button("Generate Report", type="primary", use_container_width=True):
                    st.session_state.navigate_to = 'results'
                    st.session_state._nav_pending = False
                    st.rerun()

            # STATE 3: Answers filled, waiting for submission
            elif st.session_state.with_rag_file:
                st.success("✓ AI has filled in the answers (with course materials)")

                st.markdown("""
                **Now in Chrome:**
                1. Review the answers if you'd like
                2. Click **"Finish attempt"**
                3. Click **"Submit all and finish"**
                """)

                st.markdown("---")

                st.markdown("**When you see the results page in Chrome:**")

                if st.button("📥 Collect Results", key="get2", type="primary", use_container_width=True):
                    with st.spinner("Reading results from Chrome..."):
                        results = scrape_results()
                        score = save_results(st.session_state.with_rag_file, results)
                        st.session_state.with_rag_score = score
                    st.rerun()

            # STATE 4: Ready to start second scan
            else:
                st.markdown("---")

                st.markdown("**Before you start:**")
                st.markdown("• Start a **new quiz attempt** in Moodle")
                st.markdown("• Navigate to the **first question**")
                st.markdown("• The scanner will use course materials to answer")

                st.markdown("")

                # Disable button while scanning
                scan_disabled = st.session_state.is_scanning
                button_label = "⏳ Scanning..." if scan_disabled else "▶️ Start Second Scan"

                if st.button(button_label, type="primary", use_container_width=True, disabled=scan_disabled):
                    st.session_state.is_scanning = True
                    clear_log()

                    # Create placeholder for progress display
                    progress_placeholder = st.empty()

                    try:
                        output, q_count = run_quiz(use_rag=True)
                        st.session_state.with_rag_file = output
                        progress_placeholder.success(f"✓ Second scan complete! {q_count} questions answered with course materials")
                    except Exception as e:
                        progress_placeholder.error(f"Scan failed: {str(e)}")
                    finally:
                        st.session_state.is_scanning = False
                    st.rerun()

        with right:
            st.subheader("Activity")
            with st.container(height=300):
                show_activity()


# ----------------------------------------
# TAB 4: RESULTS
# ----------------------------------------

if tab4 is not None:
    with tab4:
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
                    st.subheader("Detailed Analysis")
                
                    with open(dashboard) as f:
                        html = f.read()
                
                    st.components.v1.html(html, height=650, scrolling=True)
                
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Download Dashboard (HTML)",
                            html,
                            "quiz_vulnerability_dashboard.html",
                            "text/html",
                            use_container_width=True
                        )
                    with col2:
                        with open(st.session_state.report_file) as f:
                            report_data = f.read()
                        st.download_button(
                            "📥 Download Report (JSON)",
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

                    if st.button("🔬 Generate Report", type="primary", use_container_width=True):
                        clear_log()
                        import subprocess

                        log("Preparing scan results...", "📦")

                        # Handle single-scan vs full-scan mode
                        if is_full_scan_mode and st.session_state.with_rag_file:
                            # Full scan mode - merge both files
                            merged = merge_attempts(
                                st.session_state.no_rag_file,
                                st.session_state.with_rag_file,
                                no_rag_score=st.session_state.no_rag_score,
                                with_rag_score=st.session_state.with_rag_score
                            )
                        else:
                            # Basic scan mode - use single file with score info
                            merged = merge_attempts(
                                st.session_state.no_rag_file,
                                None,  # No second scan
                                no_rag_score=st.session_state.no_rag_score,
                                with_rag_score=None
                            )
                        st.session_state.merged_file = merged
                    
                        # Load to get question count
                        with open(merged) as f:
                            merged_data = json.load(f)
                        total_q = len(merged_data.get('questions', []))
                        log(f"Merged {total_q} questions", "✅")
                    
                        # Phase 1: Reform agent
                        log("Phase 1: Classifying questions...", "🏷️")
                    
                        with st.spinner(f"Classifying {total_q} questions..."):
                            result1 = subprocess.run(
                                ['python3', 'reform_agent.py', merged, '--model', st.session_state.model],
                                capture_output=True,
                                text=True
                            )
                    
                        # Parse reform_agent output for question-level logging
                        for line in result1.stdout.split('\n'):
                            line = line.strip()
                            if "Classifying Question" in line:
                                match = re.search(r'Question (\d+)', line)
                                if match:
                                    log(f"Classifying Q{match.group(1)}...", "🔍")
                            elif "Type:" in line:
                                qtype = line.split("Type:")[1].strip()
                                log(f"  Type: {qtype}", "📝")
                            elif "Vulnerability:" in line:
                                vuln = line.split("Vulnerability:")[1].strip()
                                icon = "🔴" if vuln == "HIGH" else "🟡" if vuln == "MODERATE" else "🟢" if vuln == "LOW" else "⚪"
                                log(f"  Vulnerability: {vuln}", icon)
                    
                        report = merged.replace('.json', '_analysis_report.json')
                        if not os.path.exists(report):
                            # Try alternate filename in case reform_agent uses different naming
                            alt_report = merged.replace('.json', '_vulnerability_report.json')
                            if os.path.exists(alt_report):
                                report = alt_report
                            else:
                                log("Classification failed", "❌")
                                st.error(f"Reform agent failed.\n\nStderr: {result1.stderr[:500] if result1.stderr else 'None'}")
                                st.stop()
                    
                        log("Phase 1 complete!", "✅")
                    
                        # Phase 2: Analysis agent
                        log("Phase 2: Building dashboard...", "📊")
                    
                        with st.spinner("Generating dashboard..."):
                            result2 = subprocess.run(
                                ['python3', 'analysis_agent.py', report],
                                capture_output=True,
                                text=True
                            )
                    
                        # Parse analysis_agent output
                        for line in result2.stdout.split('\n'):
                            line = line.strip()
                            if "Calculating" in line:
                                log("Calculating statistics...", "🔢")
                            elif "LLM interpretation" in line.lower():
                                log("Generating AI insights...", "🤖")
                            elif "markdown" in line.lower() and "saved" in line.lower():
                                log("Markdown summary created", "📄")
                            elif "Dashboard saved" in line:
                                log("HTML dashboard created", "🎨")
                    
                        # Dashboard is in DASHBOARDS_DIR
                        report_path = Path(report)
                        base_name = report_path.stem.replace('_analysis_report', '').replace('_vulnerability_report', '')
                        dashboard = DASHBOARDS_DIR / f"{base_name}_dashboard.html"

                        if dashboard.exists():
                            log("Phase 2 complete!", "✅")
                            log("Dashboard ready to view!", "🎉")
                            st.session_state.report_file = report
                        else:
                            log("Dashboard generation had issues", "⚠️")
                            st.warning(f"Dashboard not generated.\n\nStderr: {result2.stderr[:500] if result2.stderr else 'None'}")
                            st.session_state.report_file = report
                    
                        st.rerun()
            
                with right:
                    st.subheader("Activity")
                    with st.container(height=350):
                        show_activity()


    # ----------------------------------------
    # TAB 5: TEST QUESTIONS
# ----------------------------------------

def check_rag_available():
    """Check if RAG database is available"""
    try:
        import chromadb
        if os.path.exists('./chroma_db'):
            client = chromadb.PersistentClient(path="./chroma_db")
            collection = client.get_collection("unit_materials")
            return True
        return False
    except:
        return False


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
    
    # Use the optimized v4 prompt
    prompt = f"""TASK: Answer this multiple choice question correctly.

QUESTION: {question}

OPTIONS:
{options_text}
{rag_context}
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

    try:
        from collections import Counter
        
        if num_samples == 1:
            # Single sample - deterministic
            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0, 'num_predict': 512}
            )
            
            text = response['message']['content']
            
            # Parse response
            answer_match = re.search(r'ANSWER:\s*([A-Ea-e])', text)
            conf_match = re.search(r'CONFIDENCE:\s*(\d+)', text)
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', text, re.DOTALL)
            
            ai_answer = answer_match.group(1).upper() if answer_match else "?"
            confidence = int(conf_match.group(1)) if conf_match else 0
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            
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

                answer_match = re.search(r'ANSWER:\s*([A-Ea-e])', text)
                if answer_match:
                    answer = answer_match.group(1).upper()
                else:
                    for pattern in [r'(?:answer|select)[:\s]*([A-Ea-e])\b']:
                        m = re.search(pattern, text, re.IGNORECASE)
                        if m:
                            answer = m.group(1).upper()
                            break
                    else:
                        answer = "?"

                # Extract confidence for this sample
                conf_match = re.search(r'CONFIDENCE:\s*(\d+)', text)
                conf = int(conf_match.group(1)) if conf_match else 0

                answers.append(answer)
                confidences.append(conf)

                # Store sample details for display
                sample_details.append({
                    'sample_num': i + 1,
                    'answer': answer,
                    'confidence': conf
                })

                if not reasonings:
                    rm = re.search(r'REASONING:\s*(.+?)(?:\n|$)', text, re.DOTALL)
                    if rm:
                        reasonings.append(rm.group(1).strip()[:200])

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


if tab5 is not None:
    with tab5:
        st.subheader("🧪 Test a Single Question")

        # Show instructions at the top (matching the workflow description)
        with st.expander("📋 Instructions", expanded=True):
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
            - Consistency shows how often AI chose the same answer (e.g., "7/10" = 70% agreement)
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
                test_model_options = ['llama3:8b', 'mistral', 'gemma2:9b']
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
                        test_model_options = ['llama3:8b']  # Fallback
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
                test_btn_label = "⏳ Testing..." if testing_disabled else "🧪 Test Question"
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
                    st.error(f"### ⚠️ AI Correct: {ai_answer}")
                    st.caption("This question is VULNERABLE")
                elif is_correct is False:
                    st.success(f"### ✅ AI Wrong: {ai_answer}")
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
                            bar = "█" * bar_len
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
                        st.warning("⚠️ High confidence but wrong - AI is confidently incorrect!")
                    elif confidence >= 80 and is_correct:
                        st.error("🚨 High confidence AND correct - very vulnerable!")

                    # Reasoning and full response only for single sample
                    if result.get('reasoning'):
                        with st.expander("AI Reasoning"):
                            st.write(result['reasoning'])

                    with st.expander("Full AI Response"):
                        st.code(result.get('full_response', 'N/A'))

                # RAG indicator
                if result.get('used_rag'):
                    st.caption("📚 Tested with course materials")
                else:
                    st.caption("🧠 Tested with general knowledge only")


# ----------------------------------------
# TAB 6: SETTINGS
# ----------------------------------------

if tab6 is not None:
    with tab6:
        st.subheader("Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📚 Course Materials")

            # Get existing collections
            all_collections = get_all_rag_collections()
            collection_names = [c['display_name'] for c in all_collections]

            # ========================================
            # STEP 1: Select or Create a Collection
            # ========================================
            st.markdown("#### Step 1: Select a Course")
            st.caption("Each course has its own set of materials. Select an existing course or create a new one.")

            # Create new collection option
            with st.expander("➕ Create New Course", expanded=len(all_collections) == 0):
                new_name = st.text_input(
                    "Course name",
                    placeholder="e.g., PSYC101, Biology 200, History Fall 2024",
                    key="new_collection_name"
                )
                if st.button("Create Course", type="primary", disabled=not new_name):
                    if new_name:
                        try:
                            import chromadb
                            client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
                            internal_name = get_rag_collection_name(new_name)
                            client.get_or_create_collection(name=internal_name)
                            st.session_state.selected_rag_collection = new_name
                            st.success(f"✓ Created course: {new_name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error creating course: {e}")

            # Select existing collection
            if all_collections:
                # Find current selection index
                current_idx = 0
                for i, name in enumerate(collection_names):
                    if name == st.session_state.selected_rag_collection:
                        current_idx = i
                        break

                selected = st.selectbox(
                    "Select course",
                    collection_names,
                    index=current_idx,
                    key="collection_selector"
                )

                if selected != st.session_state.selected_rag_collection:
                    st.session_state.selected_rag_collection = selected
                    st.rerun()

                # Show selected collection info
                selected_coll = next((c for c in all_collections if c['display_name'] == selected), None)
                if selected_coll:
                    st.success(f"✓ **{selected}** — {selected_coll['count']} chunks loaded")

                    # Show files in collection
                    files = get_collection_files(selected_coll['name'])
                    if files:
                        with st.expander(f"📁 Files in {selected} ({len(files)} files)"):
                            for filename, chunk_count in files.items():
                                st.text(f"  • {filename} ({chunk_count} chunks)")
            else:
                st.info("No courses created yet. Create one above to get started.")

            st.markdown("---")

            # ========================================
            # STEP 2: Upload Materials
            # ========================================
            st.markdown("#### Step 2: Upload Materials")
            st.caption("Add lecture notes, textbook excerpts, or study guides to the selected course.")

            if not all_collections:
                st.warning("Create a course first before uploading materials.")
            else:
                uploaded_files = st.file_uploader(
                    f"Upload to: **{st.session_state.selected_rag_collection}**",
                    type=['txt', 'md', 'pdf'],
                    accept_multiple_files=True,
                    help="Supported formats: .txt, .md, .pdf",
                    key="rag_file_uploader_settings"
                )

                if uploaded_files:
                    if st.button("📥 Add to Course", type="primary", key="add_to_collection_btn"):
                        try:
                            import chromadb
                            client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
                            internal_name = get_rag_collection_name(st.session_state.selected_rag_collection)
                            collection = client.get_or_create_collection(name=internal_name)

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
                                        st.warning("PDF support requires: pip install pypdf")
                                        continue
                                else:
                                    content = uploaded_file.read().decode('utf-8', errors='ignore')

                                if not content.strip():
                                    st.warning(f"Skipped {uploaded_file.name} (empty or unreadable)")
                                    continue

                                # Chunk the content
                                chunk_size, overlap = 1000, 200
                                chunks = []
                                start = 0
                                while start < len(content):
                                    chunk = content[start:start + chunk_size]
                                    if chunk.strip():
                                        chunks.append(chunk)
                                    start += chunk_size - overlap

                                if chunks:
                                    base_id = f"{st.session_state.selected_rag_collection}_{uploaded_file.name}".replace(" ", "_")[:50]
                                    collection.add(
                                        documents=chunks,
                                        ids=[f"{base_id}_chunk_{i}" for i in range(len(chunks))],
                                        metadatas=[{"source": uploaded_file.name, "chunk": i} for i in range(len(chunks))]
                                    )
                                    total_chunks += len(chunks)
                                    st.success(f"✓ Added {len(chunks)} chunks from {uploaded_file.name}")

                            if total_chunks > 0:
                                st.balloons()
                                st.rerun()

                        except Exception as e:
                            st.error(f"Error adding files: {e}")

            st.markdown("---")

            # ========================================
            # STEP 3: Manage Collections
            # ========================================
            with st.expander("🗑️ Delete Course", expanded=False):
                st.warning("**Warning:** This permanently deletes all materials in the selected course.")
                if all_collections:
                    delete_target = st.selectbox(
                        "Course to delete",
                        collection_names,
                        key="delete_collection_selector"
                    )
                    if st.button(f"🗑️ Delete {delete_target}", type="secondary"):
                        try:
                            import chromadb
                            client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
                            internal_name = get_rag_collection_name(delete_target)
                            client.delete_collection(internal_name)
                            # Reset selection if we deleted the selected one
                            if st.session_state.selected_rag_collection == delete_target:
                                remaining = [c for c in collection_names if c != delete_target]
                                st.session_state.selected_rag_collection = remaining[0] if remaining else DEFAULT_COLLECTION_NAME
                            st.success(f"Deleted: {delete_target}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

        with col2:
            st.markdown("### 🤖 AI Models")

            # Show available models
            st.caption("Currently selected model:")
            st.info(f"**{st.session_state.model}**")

            # Model status check - only show INSTALLED models
            try:
                import ollama
                models_response = ollama.list()

                # Get installed model names (handle both old and new Ollama API formats)
                installed = []
                for m in models_response.get('models', []):
                    name = m.get('name', '') or m.get('model', '')
                    if name:
                        installed.append(name.lower())

                # Filter AVAILABLE_MODELS to only show installed ones
                installed_models = {}
                for model_name, description in AVAILABLE_MODELS.items():
                    base_name = model_name.split(':')[0].lower()
                    # Check if base name or full name matches any installed model
                    is_installed = any(base_name in m or model_name.lower() in m for m in installed)
                    if is_installed:
                        installed_models[model_name] = description

                if installed_models:
                    st.caption("Installed models:")
                    for model_name, description in installed_models.items():
                        st.success(f"✓ {model_name}")
                        st.caption(f"  {description}")
                else:
                    st.error("No compatible AI models installed")
                    st.info("Run: `ollama pull llama3:8b`")

                # Show note about adding more models
                st.caption("To add more models, see README or run `ollama pull <model>`")

            except Exception as e:
                st.error(f"Cannot connect to Ollama: {e}")
                st.info("Make sure Ollama is running: `ollama serve`")

            st.markdown("---")
            st.markdown("### 📁 Output Locations")
            st.caption("Generated files are saved to:")
            st.code(f"""
    Raw attempts: output/raw_attempts/
    Reports:      output/reports/
    Dashboards:   output/dashboards/
            """)

            st.markdown("---")
            st.markdown("### 🔄 Scan Mode")

            current_mode = "Complete Assessment (2 scans)" if st.session_state.use_rag_mode else "Baseline Scan (1 scan)"
            st.caption(f"Current mode: **{current_mode}**")

            if st.session_state.use_rag_mode:
                st.info("Complete Assessment compares AI performance with and without course materials.")
                if st.button("Switch to Baseline Scan", use_container_width=True):
                    st.session_state.use_rag_mode = False
                    st.rerun()
            else:
                st.info("Baseline Scan tests with general knowledge only.")
                if st.button("Switch to Complete Assessment", type="primary", use_container_width=True):
                    st.session_state.use_rag_mode = True
                    st.rerun()

            st.markdown("---")
            st.markdown("### 🎓 Help & Onboarding")
            if st.button("📖 Show Welcome Screen Again"):
                st.session_state.onboarding_complete = False
                st.session_state.onboarding_step = 1
                st.rerun()

# Footer
st.divider()
st.caption("Quiz Vulnerability Assessment Framework")