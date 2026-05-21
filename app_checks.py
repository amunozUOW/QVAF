"""
System checks and activity log functions.

Extracted from App.py — no logic changes.
"""

import streamlit as st
from datetime import datetime


# ============================================
# ACTIVITY LOG
# ============================================

def log(message):
    """Add timestamped message to activity log"""
    st.session_state.activity.append({
        'time': datetime.now().strftime("%H:%M:%S"),
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
        st.text(f"{item['time']}  {item['text']}")


# ============================================
# SYSTEM CHECKS
# ============================================

def check_chrome():
    """Check if Chrome is connected and find Moodle page"""
    try:
        from playwright.sync_api import sync_playwright
        from app_scanning import find_moodle_page
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
            page, url = find_moodle_page(browser)
            return True, url
    except Exception as e:
        return False, str(e)


@st.cache_data(ttl=120)
def get_installed_models():
    """Query Ollama for all installed models.

    Returns a list of model name strings (e.g. ['llama3:8b', 'llava:latest']),
    or an empty list if Ollama is unreachable.
    """
    try:
        import ollama
        models_response = ollama.list()

        raw_models = getattr(models_response, 'models', None)
        if raw_models is None:
            raw_models = models_response.get('models', []) if hasattr(models_response, 'get') else []

        installed = []
        for m in raw_models:
            name = getattr(m, 'model', None) or getattr(m, 'name', None)
            if name is None and isinstance(m, dict):
                name = m.get('name', '') or m.get('model', '')
            if name:
                installed.append(name)
        return installed
    except Exception:
        return []


VISION_MODEL_KEYWORDS = ['llava', 'llama3.2-vision', 'bakllava', 'moondream']


def check_ollama():
    """Check Ollama models - returns (text_model_ok, vision_model_ok, installed_models).

    Any model that is NOT purely a vision model counts as a text model.
    """
    installed = get_installed_models()
    text_models = []
    vision_models = []
    for name in installed:
        base = name.split(':')[0].lower()
        if any(kw in base for kw in VISION_MODEL_KEYWORDS):
            vision_models.append(name)
        else:
            text_models.append(name)
    return bool(text_models), bool(vision_models), installed


@st.cache_data(ttl=300)
def check_stack():
    """Validate the full software stack on startup.

    Returns a list of dicts: [{'component': str, 'ok': bool, 'detail': str}]
    """
    results = []

    # 1. Python packages
    _required_packages = [
        ('streamlit', 'streamlit', '1.32.0'),
        ('playwright', 'playwright', '1.44.0'),
        ('ollama', 'ollama', '0.4.0'),
        ('chromadb', 'chromadb', '0.5.0'),
        ('pypdf', 'pypdf', '4.0.0'),
    ]
    for label, module_name, min_ver in _required_packages:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, '__version__', None)
            if ver is None:
                try:
                    from importlib.metadata import version as pkg_version
                    ver = pkg_version(module_name)
                except Exception:
                    ver = 'unknown'
            results.append({'component': label, 'ok': True, 'detail': f'v{ver}'})
        except ImportError:
            results.append({'component': label, 'ok': False, 'detail': f'not installed (need >={min_ver})'})

    # 2. Playwright browser
    try:
        import subprocess
        proc = subprocess.run(
            ['python3', '-m', 'playwright', 'install', '--dry-run', 'chromium'],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0:
            results.append({'component': 'chromium browser', 'ok': True, 'detail': 'installed'})
        else:
            results.append({'component': 'chromium browser', 'ok': False,
                            'detail': 'run: python -m playwright install chromium'})
    except Exception:
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                browser.close()
            results.append({'component': 'chromium browser', 'ok': True, 'detail': 'installed'})
        except Exception:
            results.append({'component': 'chromium browser', 'ok': False,
                            'detail': 'run: python -m playwright install chromium'})

    # 3. Ollama service
    try:
        import ollama as _ol
        _ol.list()
        results.append({'component': 'ollama service', 'ok': True, 'detail': 'running'})
    except Exception:
        results.append({'component': 'ollama service', 'ok': False,
                        'detail': 'not reachable — start Ollama app or run: ollama serve'})

    # 4. Ollama models
    models = get_installed_models()
    text_names = [m for m in models
                  if not any(kw in m.split(':')[0].lower() for kw in VISION_MODEL_KEYWORDS)]
    if text_names:
        results.append({'component': 'text model', 'ok': True,
                        'detail': ', '.join(text_names[:3])})
    else:
        results.append({'component': 'text model', 'ok': False,
                        'detail': 'run: ollama pull llama3.2:3b'})

    vision_names = [m for m in models
                    if any(kw in m.split(':')[0].lower() for kw in VISION_MODEL_KEYWORDS)]
    if vision_names:
        results.append({'component': 'vision model', 'ok': True,
                        'detail': ', '.join(vision_names[:2])})
    else:
        results.append({'component': 'vision model', 'ok': False,
                        'detail': 'optional — run: ollama pull llava'})

    return results
