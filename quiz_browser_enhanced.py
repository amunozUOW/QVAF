#!/usr/bin/env python3
"""
Moodle Quiz Browser Automation (Enhanced)
==========================================

PATCHED: Added smart page detection to find Moodle quiz page instead of 
grabbing first available page (which might be an internal Chrome popup).

NEW: Multi-sample consistency measurement (--samples flag)
When --samples > 1, each question is run multiple times and consistency
is measured (e.g., "8/10" = AI gave same answer 8 out of 10 times).

Usage:
  python3 quiz_browser_enhanced.py --no-rag              # Fast (1 sample)
  python3 quiz_browser_enhanced.py --no-rag --samples 10 # Accurate (10 samples)
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import Counter

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("ERROR: Playwright not installed.")
    sys.exit(1)

try:
    import ollama
except ImportError:
    print("ERROR: Ollama not installed.")
    sys.exit(1)

try:
    import chromadb
except ImportError:
    chromadb = None


# ============================================
# CONFIGURATION (imported from central config)
# ============================================

try:
    from config import (
        DEFAULT_MODEL, VISION_MODEL as OLLAMA_VISION_MODEL,
        CHROMA_DB_PATH, CHROMA_COLLECTION_NAME as CHROMA_COLLECTION,
        TEMP_SCREENSHOTS_DIR as TEMP_FOLDER,
        MOODLE_URL_PATTERNS, EXCLUDE_URL_PATTERNS,
        RAW_ATTEMPTS_DIR, ensure_directories
    )
    ensure_directories()
except ImportError:
    # Fallback for standalone usage
    DEFAULT_MODEL = "llama3:8b"
    OLLAMA_VISION_MODEL = "llava"
    CHROMA_DB_PATH = "./chroma_db"
    CHROMA_COLLECTION = "unit_materials"
    TEMP_FOLDER = Path("./temp_screenshots")
    TEMP_FOLDER.mkdir(exist_ok=True)
    # Create proper output directory structure even in fallback mode
    RAW_ATTEMPTS_DIR = Path("./output/raw_attempts")
    RAW_ATTEMPTS_DIR.mkdir(parents=True, exist_ok=True)
    MOODLE_URL_PATTERNS = [
        'moodle', '/mod/quiz/', 'quiz/attempt', 'quiz/view',
        'lms.', 'learn.', 'elearning', 'blackboard', 'canvas', 'brightspace',
    ]
    EXCLUDE_URL_PATTERNS = [
        'chrome://', 'chrome-extension://', 'devtools://', 'about:', 'edge://',
    ]


# ============================================
# SMART PAGE DETECTION (NEW)
# ============================================

def find_moodle_page(browser, debug=False):
    """
    Find the Moodle quiz page from all available browser pages.
    
    Returns the best matching page, or raises an exception with
    available pages listed for debugging.
    """
    all_pages = []
    
    # Collect ALL pages from ALL contexts
    for ctx_idx, context in enumerate(browser.contexts):
        for page_idx, page in enumerate(context.pages):
            try:
                url = page.url
                title = page.title() if url else ""
            except:
                url = "unknown"
                title = ""
            
            all_pages.append({
                'page': page,
                'url': url,
                'title': title,
                'context_idx': ctx_idx,
                'page_idx': page_idx,
            })
    
    if debug:
        print(f"\n[DEBUG] Found {len(all_pages)} total pages:")
        for i, p in enumerate(all_pages):
            print(f"  {i}: {p['url'][:80]}")
    
    # Filter out internal Chrome pages
    candidate_pages = []
    for p in all_pages:
        url_lower = p['url'].lower()
        
        # Skip excluded patterns
        if any(excl in url_lower for excl in EXCLUDE_URL_PATTERNS):
            if debug:
                print(f"  [SKIP] Internal page: {p['url'][:60]}")
            continue
        
        candidate_pages.append(p)
    
    if debug:
        print(f"\n[DEBUG] {len(candidate_pages)} candidate pages after filtering")
    
    # Look for Moodle/LMS pages
    moodle_pages = []
    for p in candidate_pages:
        url_lower = p['url'].lower()
        title_lower = p['title'].lower()
        
        # Check URL patterns
        if any(pattern in url_lower for pattern in MOODLE_URL_PATTERNS):
            moodle_pages.append(p)
            if debug:
                print(f"  [MATCH] Moodle URL: {p['url'][:60]}")
            continue
        
        # Check title for quiz-related keywords
        if any(kw in title_lower for kw in ['quiz', 'assessment', 'exam', 'test']):
            moodle_pages.append(p)
            if debug:
                print(f"  [MATCH] Quiz title: {p['title'][:40]}")
            continue
    
    # Return best match
    if moodle_pages:
        # Prefer pages with 'quiz' in URL
        quiz_pages = [p for p in moodle_pages if 'quiz' in p['url'].lower()]
        if quiz_pages:
            chosen = quiz_pages[0]
        else:
            chosen = moodle_pages[0]
        
        print(f"✓ Found Moodle page: {chosen['url'][:80]}")
        return chosen['page']
    
    # No Moodle page found - show what's available
    print("\n" + "="*60)
    print("❌ ERROR: No Moodle quiz page found!")
    print("="*60)
    
    if candidate_pages:
        print("\nAvailable pages (non-internal):")
        for i, p in enumerate(candidate_pages):
            print(f"  {i+1}. {p['url'][:70]}")
            if p['title']:
                print(f"      Title: {p['title'][:50]}")
    else:
        print("\nNo usable pages found in Chrome.")
    
    print("\n" + "-"*60)
    print("TROUBLESHOOTING:")
    print("  1. Open your Moodle quiz in Chrome")
    print("  2. Make sure you're on the quiz attempt page")
    print("  3. Close unnecessary tabs (especially DevTools)")
    print("  4. Try again")
    print("-"*60 + "\n")
    
    raise Exception("No Moodle quiz page found. Please open your quiz in Chrome first.")


def list_available_pages(browser):
    """Debug helper: list all pages visible to Playwright"""
    print("\n" + "="*60)
    print("AVAILABLE CHROME PAGES")
    print("="*60)
    
    for ctx_idx, context in enumerate(browser.contexts):
        print(f"\nContext {ctx_idx}:")
        for page_idx, page in enumerate(context.pages):
            try:
                url = page.url
                title = page.title()
            except:
                url = "error"
                title = ""
            
            status = ""
            url_lower = url.lower()
            if any(excl in url_lower for excl in EXCLUDE_URL_PATTERNS):
                status = " [INTERNAL]"
            elif any(pat in url_lower for pat in MOODLE_URL_PATTERNS):
                status = " [MOODLE] ✓"
            
            print(f"  Page {page_idx}: {url[:60]}{status}")
            if title:
                print(f"           Title: {title[:50]}")
    
    print("="*60 + "\n")


# ============================================
# RAG SETUP
# ============================================

def setup_rag(collection_name=None):
    """
    Initialize RAG with a specific collection.

    Args:
        collection_name: The internal collection name (e.g., 'rag_PSYC101').
                        If None, uses the default from config.
    """
    if chromadb is None:
        return None
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        # Use provided collection name or fall back to default
        coll_name = collection_name if collection_name else CHROMA_COLLECTION
        collection = client.get_collection(name=coll_name)
        print(f"RAG initialized: {collection.count()} chunks available from '{coll_name}'")
        return collection
    except Exception as e:
        print(f"WARNING: Could not initialize RAG: {e}")
        return None


def query_rag(collection, question_text, n_results=3):
    if collection is None:
        return ""
    try:
        results = collection.query(query_texts=[question_text], n_results=n_results)
        if results and results['documents']:
            return "\n\n---\n\n".join(results['documents'][0])
        return ""
    except Exception as e:
        print(f"RAG query error: {e}")
        return ""


# ============================================
# IMAGE CAPTURE AND INTERPRETATION
# ============================================

def capture_question_images(q_elem, q_num, debug=False):
    image_paths = []
    try:
        images = q_elem.query_selector_all('img')
        for i, img in enumerate(images):
            try:
                box = img.bounding_box()
                if box and box['width'] > 50 and box['height'] > 50:
                    img_path = TEMP_FOLDER / f"q{q_num}_img{i}_{datetime.now().strftime('%H%M%S')}.png"
                    img.screenshot(path=str(img_path))
                    image_paths.append(str(img_path))
            except:
                continue
    except:
        pass
    return image_paths


def interpret_image_with_llava(image_path, question_context=""):
    try:
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        prompt = f"""Analyze this image carefully. It's required to understand the content of the quiz question (and maybe other questions that refer to this image). Be as thorough as possible, as this may be critical for answering the quiz question.
{f"Question context: {question_context}" if question_context else ""}
Extract ALL text, numbers, and data visible. If it's a table, reproduce it. If it's a chart, describe the data. If it's an image, describe it in detail. """

        response = ollama.chat(
            model=OLLAMA_VISION_MODEL,
            messages=[{'role': 'user', 'content': prompt, 'images': [image_data]}]
        )
        return response['message']['content']
    except Exception as e:
        return f"[Image interpretation failed: {e}]"


def process_question_images(q_elem, q_num, q_text, debug=False):
    image_paths = capture_question_images(q_elem, q_num, debug)
    if not image_paths:
        return "", []
    
    print(f"       🖼️  Found {len(image_paths)} image(s), interpreting...")
    interpretations = []
    for img_path in image_paths:
        interpretation = interpret_image_with_llava(img_path, q_text[:200])
        interpretations.append(interpretation)
    
    combined = "\n\n".join([f"[IMAGE {i+1}]:\n{interp}" for i, interp in enumerate(interpretations)])
    return combined, image_paths


# ============================================
# LINK DETECTION AND SCRAPING
# ============================================

def detect_question_links(q_elem, base_url):
    links = []
    try:
        for selector in ['.qtext', '.formulation']:
            elem = q_elem.query_selector(selector)
            if elem:
                anchors = elem.query_selector_all('a[href]')
                for anchor in anchors:
                    href = anchor.get_attribute('href')
                    text = anchor.inner_text().strip()
                    if href and not href.startswith(('mailto:', 'javascript:', '#')):
                        if href.startswith('/'):
                            from urllib.parse import urljoin
                            href = urljoin(base_url, href)
                        if (text, href) not in links:
                            links.append((text or "Link", href))
    except:
        pass
    return links


def scrape_linked_page(page, url, debug=False):
    scraped = {'url': url, 'text': '', 'javascript': '', 'images': []}
    try:
        new_page = page.context.new_page()
        try:
            new_page.goto(url, timeout=30000)
            new_page.wait_for_load_state('networkidle', timeout=15000)
            time.sleep(2)
            
            for selector in ['main', 'article', '.content', '#content', 'body']:
                main = new_page.query_selector(selector)
                if main:
                    scraped['text'] = main.inner_text()[:5000]
                    break
        finally:
            new_page.close()
    except Exception as e:
        scraped['error'] = str(e)
    return scraped


def process_question_links(page, q_elem, base_url, debug=False):
    links = detect_question_links(q_elem, base_url)
    if not links:
        return "", []
    
    print(f"       🔗 Found {len(links)} link(s), scraping...")
    all_scraped = []
    for link_text, url in links:
        scraped = scrape_linked_page(page, url, debug)
        all_scraped.append(scraped)
    
    parts = []
    for scraped in all_scraped:
        if scraped.get('text'):
            parts.append(f"[LINKED PAGE: {scraped['url']}]\n{scraped['text'][:2000]}")
    
    return "\n\n".join(parts), all_scraped


# ============================================
# LLM ANSWERING
# ============================================

def build_prompt(question, options, rag_context="", image_context="", link_context=""):
    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    
    context_parts = []
    if rag_context:
        context_parts.append(f"COURSE MATERIALS:\n{rag_context}")
    if image_context:
        context_parts.append(f"IMAGE CONTENT:\n{image_context}")
    if link_context:
        context_parts.append(f"LINKED CONTENT:\n{link_context}")
    
    context_block = "\n\n".join(context_parts)
    
    # Build the list of valid option letters from the actual options provided
    option_letters = sorted(options.keys())
    letters_str = ", ".join(option_letters) if option_letters else "A, B, C, D, or E"

    return f"""Answer this multiple choice question.

QUESTION: {question}

OPTIONS:
{options_text}

{context_block}

First, state your answer as a single choice. Then quantify how confident you are that the choice is the correct answer to the question. Please explain briefly your reasoning. Be concise but thorough, as the question may be tricky and require careful thought.

Format your response EXACTLY like this:
ANSWER: X
CONFIDENCE: N
REASONING: Your explanation here

where X is one of the options ({letters_str}) and N is a number from 0 to 100.

Your response:"""


def parse_llm_response(text):
    answer_match = re.search(r'ANSWER:\s*([A-Za-z])', text)
    confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', text)
    reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\n\n|\Z)', text, re.DOTALL)
    
    answer = answer_match.group(1).upper() if answer_match else None
    confidence = int(confidence_match.group(1)) if confidence_match else 50
    reasoning = reasoning_match.group(1).strip() if reasoning_match else text[:200]
    
    # Fallback answer extraction
    if not answer:
        first_letter = re.search(r'^([A-Z])\b', text.strip())
        if first_letter:
            answer = first_letter.group(1)
        else:
            answer = "A"  # Default fallback
    
    return answer, confidence, reasoning


def call_llm_single(prompt, model):
    """Single LLM call"""
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0, 'num_predict': 300}
        )
        return response['message']['content']
    except Exception as e:
        return f"ANSWER: A\nCONFIDENCE: 0\nREASONING: Error: {e}"


def call_llm_multi_sample(prompt, model, num_samples):
    """
    Multi-sample LLM calls for consistency measurement.
    Returns the most common answer and consistency stats.
    
    NOTE: Even with temperature=0, LLMs show some variability due to
    floating-point arithmetic, GPU parallelism, and other factors.
    This variability is meaningful signal about model uncertainty.
    """
    answers = []
    confidences = []
    reasonings = []
    
    for i in range(num_samples):
        try:
            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0, 'num_predict': 300}
            )
            text = response['message']['content']
            answer, conf, reasoning = parse_llm_response(text)
            answers.append(answer)
            confidences.append(conf)
            reasonings.append(reasoning)
        except Exception as e:
            answers.append("A")
            confidences.append(0)
            reasonings.append(f"Error: {e}")
        
        print(".", end="", flush=True)
    
    print()  # Newline after dots
    
    # Calculate consistency
    counter = Counter(answers)
    most_common_answer, most_common_count = counter.most_common(1)[0]
    
    # Get the reasoning from the most common answer
    best_idx = answers.index(most_common_answer)
    best_reasoning = reasonings[best_idx]
    avg_confidence = sum(confidences) / len(confidences)
    
    consistency_pct = round(most_common_count / num_samples * 100, 1)
    consistency_str = f"{most_common_count}/{num_samples}"
    
    return {
        'answer': most_common_answer,
        'confidence': avg_confidence,
        'reasoning': best_reasoning,
        'consistency': consistency_str,
        'consistency_pct': consistency_pct,
        'consistency_count': most_common_count,
        'answer_distribution': dict(counter),
        'all_answers': answers,
    }


def answer_question(question, options, use_rag, rag_collection, image_context, link_context, model, num_samples=1):
    """Answer a question with optional RAG and multi-sample"""
    
    rag_context = ""
    if use_rag and rag_collection:
        rag_context = query_rag(rag_collection, question)
    
    prompt = build_prompt(question, options, rag_context, image_context, link_context)
    
    if num_samples == 1:
        text = call_llm_single(prompt, model)
        answer, confidence, reasoning = parse_llm_response(text)
        return {
            'answer': answer,
            'confidence': confidence,
            'reasoning': reasoning,
            'consistency': '1/1',
            'consistency_pct': 100,
            'raw_response': text,
        }
    else:
        result = call_llm_multi_sample(prompt, model, num_samples)
        result['consistency_pct'] = result['consistency_pct']
        # Backwards compatibility: use consistency as confidence proxy
        result['confidence'] = result['consistency_pct']
        return result


# ============================================
# SCRAPE AND ANSWER PAGE
# ============================================

def scrape_and_answer_page(page, use_rag, rag_collection, debug=False, model=None, num_samples=1):
    if model is None:
        model = DEFAULT_MODEL
    
    questions_answered = []
    base_url = page.url

    info_block_context = {'text': '', 'images': '', 'links': ''}

    question_elements = page.query_selector_all('div.que')
    total_questions = len([q for q in question_elements if q.query_selector('.qtext')])
    print(f"[PROGRESS] Found {total_questions} questions on this page", flush=True)

    for q_elem in question_elements:
        try:
            qno_elem = q_elem.query_selector('.info .qno')
            q_num = qno_elem.inner_text().strip() if qno_elem else "?"
            
            qtext_elem = q_elem.query_selector('.qtext')
            q_text = qtext_elem.inner_text().strip() if qtext_elem else ""
            
            if not q_text:
                formulation = q_elem.query_selector('.formulation.clearfix')
                if formulation:
                    q_text = formulation.inner_text().strip()
            
            classes = q_elem.get_attribute('class') or ''
            q_type = 'multichoice' if 'multichoice' in classes else 'unknown'
            
            options = {}
            radio_buttons = {}
            
            answer_container = q_elem.query_selector('.answer')
            if answer_container:
                answer_items = answer_container.query_selector_all('div[class*="r0"], div[class*="r1"]')
                if not answer_items:
                    answer_items = answer_container.query_selector_all('div.flex-fill')
                if not answer_items:
                    answer_items = answer_container.query_selector_all('div:has(input[type="radio"])')
                
                letter_index = 0
                seen_texts = set()
                
                for item in answer_items:
                    radio = item.query_selector('input[type="radio"]')
                    if radio:
                        label = item.query_selector('label')
                        text = label.inner_text().strip() if label else item.inner_text().strip()
                        text = re.sub(r'^[a-zA-Z][\.\\)]\s*', '', text)
                        
                        if text and text not in seen_texts:
                            seen_texts.add(text)
                            letter = chr(65 + letter_index)
                            options[letter] = text
                            radio_buttons[letter] = radio
                            letter_index += 1
            
            # Info block (no options)
            if not options:
                print(f"\n  📋 INFO BLOCK: {q_text[:60]}...")
                info_block_context['text'] = q_text
                img_ctx, _ = process_question_images(q_elem, f"info_{q_num}", q_text, debug)
                if img_ctx:
                    info_block_context['images'] = img_ctx
                link_ctx, _ = process_question_links(page, q_elem, base_url, debug)
                if link_ctx:
                    info_block_context['links'] = link_ctx
                continue
            
            # Progress output for UI parsing
            q_preview = q_text[:50].replace('\n', ' ')
            print(f"\n[PROGRESS] Question {q_num}: {q_preview}...", flush=True)
            print(f"       Options: {list(options.keys())}", flush=True)

            # Gather context
            image_context, image_paths = process_question_images(q_elem, q_num, q_text, debug)
            if image_paths:
                print(f"[PROGRESS] Q{q_num} has {len(image_paths)} image(s) - analyzing", flush=True)

            link_context, scraped_links = process_question_links(page, q_elem, base_url, debug)
            if scraped_links:
                print(f"[PROGRESS] Q{q_num} has {len(scraped_links)} link(s) - following", flush=True)

            combined_image = info_block_context['images'] + ("\n\n" + image_context if image_context else "")
            combined_link = info_block_context['links'] + ("\n\n" + link_context if link_context else "")

            full_question = q_text
            if info_block_context['text']:
                full_question = f"SCENARIO:\n{info_block_context['text']}\n\nQUESTION:\n{q_text}"

            if num_samples > 1:
                print(f"[PROGRESS] Running {num_samples} samples for Q{q_num}...", flush=True)
            else:
                print(f"[PROGRESS] AI thinking about Q{q_num}...", flush=True)

            result = answer_question(
                full_question, options, use_rag, rag_collection,
                combined_image, combined_link, model, num_samples
            )

            answer = result['answer']
            confidence = result.get('confidence', 0)

            # Click the answer
            if answer in radio_buttons:
                try:
                    radio_buttons[answer].click()
                    print(f"[PROGRESS] Q{q_num} → Answer: {answer} (confidence: {confidence}%)", flush=True)
                except Exception as e:
                    print(f"       ⚠️  Click failed: {e}", flush=True)
            else:
                print(f"       ⚠️  Answer '{answer}' not in options")
            
            # Build result
            q_result = {
                'number': q_num,
                'question': q_text,
                'options': options,
                'llm_answer': answer,
                'llm_confidence': result.get('confidence'),
                'llm_reasoning': result.get('reasoning', '')[:500],
                'llm_consistency': result.get('consistency'),
                'llm_consistency_pct': result.get('consistency_pct'),
                'llm_consistency_count': result.get('consistency_count'),
                'answer_distribution': result.get('answer_distribution'),
                'type': q_type,
                'image_paths': image_paths if debug else [],
                'links_scraped': len(scraped_links),
            }
            
            questions_answered.append(q_result)

        except Exception as e:
            print(f"  ❌ Error on question: {e}", flush=True)

    if questions_answered:
        print(f"[PROGRESS] Page complete: {len(questions_answered)} questions answered", flush=True)

    return questions_answered


# ============================================
# RESULTS SCRAPING
# ============================================

def scrape_results(page, debug=False):
    results = []
    
    question_elements = page.query_selector_all('div.que')
    
    for q_elem in question_elements:
        try:
            qno_elem = q_elem.query_selector('.info .qno')
            q_num = qno_elem.inner_text().strip() if qno_elem else "?"
            
            # Check correctness
            classes = q_elem.get_attribute('class') or ''
            is_correct = 'correct' in classes and 'incorrect' not in classes
            
            # Get correct answer from feedback
            correct_answer = None
            feedback = q_elem.query_selector('.rightanswer')
            if feedback:
                text = feedback.inner_text()
                match = re.search(r'correct answer is[:\s]*([A-Z])', text, re.IGNORECASE)
                if match:
                    correct_answer = match.group(1).upper()
            
            results.append({
                'number': q_num,
                'is_correct': is_correct,
                'correct_answer': correct_answer,
            })
        except:
            continue
    
    return results


# ============================================
# MAIN RUNNER
# ============================================

def run_quiz_attempt(use_rag=False, debug=False, no_wait=False, model=None, num_samples=1, rag_collection_name=None):
    """
    Run the quiz automation.

    Args:
        use_rag: Whether to use RAG (course materials)
        debug: Enable debug mode
        no_wait: Exit after filling answers (for web app)
        model: LLM model to use
        num_samples: Number of samples per question
        rag_collection_name: Internal collection name (e.g., 'rag_PSYC101')
    """
    if model is None:
        model = DEFAULT_MODEL

    mode = "with_rag" if use_rag else "no_rag"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"quiz_attempt_{mode}_{timestamp}.json"
    output_file = str(RAW_ATTEMPTS_DIR / output_filename)

    print("\n" + "="*60)
    print(f"QUIZ VULNERABILITY SCANNER")
    print(f"Mode: {'WITH RAG' if use_rag else 'NO RAG (Baseline)'}")
    print(f"Model: {model}")
    print(f"Samples: {num_samples}")
    if use_rag and rag_collection_name:
        print(f"RAG Collection: {rag_collection_name}")
    print("="*60)

    rag_collection = None
    if use_rag:
        rag_collection = setup_rag(rag_collection_name)
    
    print("\nConnecting to Chrome...")
    
    all_questions = []
    attempt_data = None
    
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.connect_over_cdp("http://localhost:9222")
                
                # ============================================
                # PATCHED: Smart page detection
                # ============================================
                if debug:
                    list_available_pages(browser)
                
                page = find_moodle_page(browser, debug=debug)
                print(f"Connected: {page.url}")
                
            except Exception as e:
                print(f"ERROR connecting to Chrome: {e}")
                print("\n" + "-"*60)
                print("TROUBLESHOOTING:")
                print("  1. Make sure Chrome is running with --remote-debugging-port=9222")
                print("  2. Open your Moodle quiz in Chrome")
                print("  3. Close any DevTools windows")
                print("  4. Try again")
                print("-"*60 + "\n")
                
                # Create empty output file so app knows scan failed
                attempt_data = {
                    'timestamp': timestamp,
                    'mode': mode,
                    'use_rag': use_rag,
                    'model': model,
                    'error': str(e),
                    'questions': []
                }
                with open(output_file, 'w') as f:
                    json.dump(attempt_data, f, indent=2)
                return
            
            page_num = 1
            
            while True:
                print(f"\n--- Page {page_num} ---")
                
                try:
                    questions = scrape_and_answer_page(page, use_rag, rag_collection, debug, model, num_samples)
                except Exception as e:
                    print(f"Error on page {page_num}: {e}")
                    questions = []
                
                for q in questions:
                    q['page'] = page_num
                    all_questions.append(q)
                
                print(f"\n  Answered {len(questions)} questions")
                
                next_btn = page.query_selector('input[value="Next page"]')
                if next_btn:
                    try:
                        next_btn.click()
                        page.wait_for_load_state('networkidle')
                        time.sleep(1)
                        page_num += 1
                    except Exception as e:
                        print(f"Error navigating: {e}")
                        break
                else:
                    break
            
            # Calculate average consistency
            avg_consistency = None
            if num_samples > 1:
                counts = [q.get('llm_consistency_count', 0) for q in all_questions if q.get('llm_consistency_count')]
                if counts:
                    avg_consistency = sum(counts) / len(counts)
            
            attempt_data = {
                'timestamp': timestamp,
                'mode': mode,
                'use_rag': use_rag,
                'model': model,
                'num_samples': num_samples,
                'avg_consistency': avg_consistency,
                'url': page.url,
                'questions': all_questions
            }
            
            with open(output_file, 'w') as f:
                json.dump(attempt_data, f, indent=2)
            
            print(f"\n{'='*60}")
            print(f"ANSWERED {len(all_questions)} QUESTIONS")
            if avg_consistency:
                print(f"Average Consistency: {avg_consistency:.1f}/{num_samples}")
            print(f"Saved: {output_file}")
            print("="*60)
            
            if no_wait:
                return
            
            print("\n⚠️  Submit the quiz in browser, then press Enter...")
            input()
            
            print("\nScraping results...")
            results = scrape_results(page, debug)
            
            attempt_data['results'] = results
            correct = sum(1 for r in results if r.get('is_correct'))
            total = len([r for r in results if r.get('is_correct') is not None])
            
            attempt_data['score'] = {
                'correct': correct,
                'total': total,
                'percentage': round(correct / total * 100, 1) if total else 0
            }
            
            with open(output_file, 'w') as f:
                json.dump(attempt_data, f, indent=2)
            
            print(f"\nScore: {correct}/{total} ({attempt_data['score']['percentage']}%)")
            print(f"Saved: {output_file}")
    
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        # Ensure we always create an output file
        if attempt_data is None:
            attempt_data = {
                'timestamp': timestamp,
                'mode': mode,
                'use_rag': use_rag,
                'model': model,
                'error': str(e),
                'questions': all_questions
            }
        with open(output_file, 'w') as f:
            json.dump(attempt_data, f, indent=2)
        print(f"Partial results saved to: {output_file}")


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quiz automation with multi-sample consistency")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--no-rag', action='store_true', help='Baseline (no course materials)')
    mode_group.add_argument('--with-rag', action='store_true', help='With course materials')

    parser.add_argument('--model', default=DEFAULT_MODEL, help=f'Model (default: {DEFAULT_MODEL})')
    parser.add_argument('--debug', action='store_true', help='Debug mode: show all pages, save screenshots')
    parser.add_argument('--no-wait', action='store_true', help='Exit after filling (for web app)')
    parser.add_argument('--samples', type=int, default=1, help='Samples per question (1=fast, 10=accurate)')
    parser.add_argument('--collection', default=None, help='RAG collection name (e.g., rag_PSYC101)')
    parser.add_argument('--list-pages', action='store_true', help='Just list available Chrome pages and exit')

    args = parser.parse_args()

    # Debug helper: just list pages
    if args.list_pages:
        with sync_playwright() as p:
            try:
                browser = p.chromium.connect_over_cdp("http://localhost:9222")
                list_available_pages(browser)
            except Exception as e:
                print(f"Could not connect to Chrome: {e}")
                print("\nMake sure Chrome is running with --remote-debugging-port=9222")
        sys.exit(0)

    run_quiz_attempt(
        use_rag=args.with_rag,
        debug=args.debug,
        no_wait=args.no_wait,
        model=args.model,
        num_samples=args.samples,
        rag_collection_name=args.collection
    )