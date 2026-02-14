"""
Quiz scanning orchestration functions.

Handles browser automation subprocess calls, result scraping,
merging attempts, and analysis pipeline.

Extracted from App.py — no logic changes.
"""

import json
import os
import re
import glob
import subprocess
import streamlit as st
from datetime import datetime
from pathlib import Path

from config import (
    PROJECT_ROOT, RAW_ATTEMPTS_DIR, REPORTS_DIR,
    MOODLE_URL_PATTERNS, EXCLUDE_URL_PATTERNS,
    get_rag_collection_name
)
from app_checks import log


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


def run_quiz(use_rag=False, status_container=None):
    """Run quiz attempt via quiz_browser_enhanced.py

    Args:
        use_rag: Whether to use RAG/course materials
        status_container: Optional st.status() container for real-time progress
    """
    mode = "--with-rag" if use_rag else "--no-rag"
    model = st.session_state.model
    num_samples = st.session_state.num_samples
    label = "with course materials" if use_rag else "baseline"

    log(f"Starting {label} scan using {model}...")
    if num_samples > 1:
        log(f"Sampling: {num_samples} samples per question")

    # Build command
    cmd = ['python3', 'quiz_browser_enhanced.py', mode, '--no-wait',
           '--model', model, '--samples', str(num_samples)]

    # Add collection name if using RAG
    if use_rag:
        collection_name = st.session_state.selected_rag_collection
        internal_name = get_rag_collection_name(collection_name)
        cmd.extend(['--collection', internal_name])
        log(f"Using course materials: {collection_name}")

    existing = set(glob.glob(str(RAW_ATTEMPTS_DIR / "quiz_attempt_*_*.json")))

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    q_count = 0

    def update(message):
        """Log message and write to live status container if available."""
        log(message)
        if status_container:
            status_container.write(f"- {message}")

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
                update(f"Q{q_count}: {q_preview}")
                if status_container:
                    status_container.update(label=f"Answering question {q_count}")

            elif "AI thinking" in progress_msg:
                update(f"AI analyzing Q{q_count}...")

            elif "Running" in progress_msg and "samples" in progress_msg:
                update(f"Running multiple samples for Q{q_count}...")

            elif "Answer:" in progress_msg:
                # Got an answer with confidence
                match = re.search(r'Answer:\s*([A-Z]).*confidence:\s*(\d+)', progress_msg)
                if match:
                    ans, conf = match.groups()
                    update(f"Q{q_count} -> {ans} (confidence: {conf}%)")
                else:
                    update(f"Q{q_count} answered")

            elif "image" in progress_msg.lower() and "analyzing" in progress_msg.lower():
                match = re.search(r'(\d+)\s+image', progress_msg)
                img_count = match.group(1) if match else "1"
                update(f"Q{q_count}: Analyzing {img_count} image(s)...")

            elif "link" in progress_msg.lower() and "following" in progress_msg.lower():
                match = re.search(r'(\d+)\s+link', progress_msg)
                link_count = match.group(1) if match else "1"
                update(f"Q{q_count}: Following {link_count} link(s)...")

            elif "Reading quiz questions" in progress_msg:
                update("Reading quiz questions...")

            elif "Page complete" in progress_msg:
                match = re.search(r'(\d+)\s+questions answered', progress_msg)
                if match:
                    answered = match.group(1)
                    update(f"Page complete: {answered} questions answered")

        # Also parse other useful messages
        elif "INFO BLOCK" in line:
            update("Reading scenario context...")
        elif "image" in line.lower() and "found" in line.lower():
            update("Analyzing image...")
        elif "link" in line.lower() and "found" in line.lower():
            update("Following link...")
        elif "RAG" in line and ("loaded" in line.lower() or "initialized" in line.lower()):
            update("Course materials loaded")
        elif "Connected" in line or "Found Moodle" in line:
            log("Browser connected")

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
            log("Moved output to correct location")

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
            log("Moved output to correct location")

    if matches:
        output = sorted(matches)[-1]
        log(f"Complete! {q_count} questions answered")
        return output, q_count
    else:
        log("Error: No output file created")
        # Provide more diagnostic info
        all_json = glob.glob(str(RAW_ATTEMPTS_DIR / "*.json")) + glob.glob("*.json")
        if all_json:
            log(f"Found JSON files: {len(all_json)}")
        raise Exception("Scan failed - no output file")


def scrape_results():
    """Get results from submitted quiz"""
    from playwright.sync_api import sync_playwright

    log("Reading quiz results...")
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

    log(f"Score: {correct}/{total} ({pct}%)")
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
    log("Preparing scan results...")

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

    def strip_label(text):
        """Strip option label prefixes (A. 1. III. etc.) from text."""
        return re.sub(
            r'^(?:[a-zA-Z][\.)\]]\s*'
            r'|[0-9]+[\.)\]]\s*'
            r'|(?:I{1,3}|IV|V|VI{0,3}|IX|X)[\.)\]]\s*'
            r'|(?:i{1,3}|iv|v|vi{0,3}|ix|x)[\.)\]]\s*'
            r')', '', text
        ).lstrip('\n').strip()

    def get_letter(text, opts):
        """Extract correct answer letter from Moodle feedback text (supplementary)."""
        if not text or not opts:
            return None
        clean = re.sub(r'^The correct answer is:?\s*', '', text, flags=re.IGNORECASE).strip()
        clean = strip_label(clean)
        clean_lower = clean.lower()

        # Exact match
        for letter, opt in opts.items():
            if strip_label(opt).lower() == clean_lower:
                return letter

        # Partial match (contains)
        for letter, opt in opts.items():
            opt_clean = strip_label(opt).lower()
            if clean_lower in opt_clean or opt_clean in clean_lower:
                return letter

        return None

    questions = []
    d1_results = d1.get('results', [])
    d2_results = d2.get('results', [])

    for q in d1['questions']:
        if not q.get('options'):
            continue

        # Get question text - raw files use 'question', not 'text'
        q_text = q.get('question', '') or q.get('text', '')

        # Find matching question in with_rag data (if available)
        q2 = find(q_text, d2['questions'], 'question') if d2['questions'] else None

        # Match results from each scan separately
        m1 = find(q_text, d1_results, 'question')
        m2 = find(q_text, d2_results, 'question') if d2_results else None

        # Authoritative correctness from Moodle CSS classes
        is_correct_no_rag = m1.get('is_correct') if m1 else None
        is_correct_with_rag = m2.get('is_correct') if m2 else None

        # Supplementary: extract correct answer letter from feedback text
        correct = 'UNKNOWN'
        for m in [m1, m2]:
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
            'is_correct_without_rag': is_correct_no_rag,
            'is_correct_with_rag': is_correct_with_rag,
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

    log(f"Prepared {len(questions)} questions")
    return str(out)


def run_analysis(merged_file):
    """Generate vulnerability report and dashboard with detailed progress"""
    log("Starting vulnerability analysis...")

    # Load merged file to get question count
    with open(merged_file) as f:
        merged_data = json.load(f)
    total_questions = len(merged_data.get('questions', []))
    log(f"Analyzing {total_questions} questions")

    # Run reform_agent with streaming output
    log("Phase 1: Classifying question types...")

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
            log(f"Classifying Q{question_count}/{total_questions}...")
        elif "Type:" in line:
            qtype = line.split("Type:")[1].strip()
            log(f"  Type: {qtype}")
        elif "Vulnerability:" in line:
            vuln = line.split("Vulnerability:")[1].strip()
            log(f"  Vulnerability: {vuln}")
        elif "Generating qualitative" in line.lower():
            log("Generating detailed recommendations...")
        elif "Question" in line and "analysis" in line.lower():
            # e.g. "Question 3: Generating analysis..."
            match = re.search(r'Question (\d+)', line)
            if match:
                log(f"Writing recommendation for Q{match.group(1)}...")

    process.wait()

    report = merged_file.replace('.json', '_analysis_report.json')
    if not os.path.exists(report):
        # Try alternate filename
        alt_report = merged_file.replace('.json', '_vulnerability_report.json')
        if os.path.exists(alt_report):
            report = alt_report
        else:
            log("Classification failed - no report generated")
            return None, None

    log("Phase 1 complete")

    # Run analysis_agent with streaming output
    log("Phase 2: Generating dashboard...")

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
            log("Calculating statistics...")
        elif "LLM interpretation" in line.lower():
            log("Generating AI interpretation...")
        elif "markdown" in line.lower():
            log("Creating markdown summary...")
        elif "HTML" in line.lower() or "dashboard" in line.lower():
            log("Building HTML dashboard...")
        elif "saved" in line.lower():
            log("Saving files...")

    process.wait()

    # Handle both naming conventions
    if '_analysis_report.json' in report:
        dashboard = report.replace('_analysis_report.json', '_dashboard.html')
    else:
        dashboard = report.replace('_vulnerability_report.json', '_dashboard.html')

    if os.path.exists(dashboard):
        log("Phase 2 complete")
        log("Dashboard ready to view")
        return report, dashboard

    log("Dashboard generation failed")
    return report, None
