"""
System checks and activity log functions.

Extracted from App.py — no logic changes.
"""

import os
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


def _scan_ollama_filesystem():
    """Read installed models directly from Ollama's on-disk manifest registry.

    Works even when the Ollama API is unreachable (e.g. blocked by Cisco
    Secure Client socket filter on university machines).
    """
    from pathlib import Path
    manifests = Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library"
    if not manifests.is_dir():
        return []
    installed = []
    for model_dir in sorted(manifests.iterdir()):
        if not model_dir.is_dir():
            continue
        name = model_dir.name
        for tag_file in sorted(model_dir.iterdir()):
            if tag_file.is_file():
                installed.append(f"{name}:{tag_file.name}")
    return installed


def _ensure_ollama_host():
    """Try the default Ollama port, then the fallback port.

    If the default port (11434) is blocked (common with Cisco Secure Client
    on university machines), try the fallback port (11435) and start Ollama
    on it if needed. Sets the OLLAMA_HOST env var so the ollama Python client
    uses the working port for all subsequent calls.
    """
    if os.environ.get('_OLLAMA_HOST_RESOLVED'):
        return

    from config import OLLAMA_DEFAULT_PORT, OLLAMA_FALLBACK_PORT
    import socket

    for port in [OLLAMA_DEFAULT_PORT, OLLAMA_FALLBACK_PORT]:
        try:
            sock = socket.create_connection(('127.0.0.1', port), timeout=2)
            sock.close()
            os.environ['OLLAMA_HOST'] = f'http://127.0.0.1:{port}'
            os.environ['_OLLAMA_HOST_RESOLVED'] = '1'
            return
        except (ConnectionRefusedError, socket.timeout, OSError):
            continue

    # Neither port responded — try starting Ollama on the fallback port
    import subprocess
    try:
        env = os.environ.copy()
        env['OLLAMA_HOST'] = f'127.0.0.1:{OLLAMA_FALLBACK_PORT}'
        subprocess.Popen(
            ['/Applications/Ollama.app/Contents/Resources/ollama', 'serve'],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        import time
        time.sleep(3)
        try:
            sock = socket.create_connection(('127.0.0.1', OLLAMA_FALLBACK_PORT), timeout=2)
            sock.close()
            os.environ['OLLAMA_HOST'] = f'http://127.0.0.1:{OLLAMA_FALLBACK_PORT}'
            os.environ['_OLLAMA_HOST_RESOLVED'] = '1'
            return
        except Exception:
            pass
    except Exception:
        pass


def _list_models_via_api():
    """Call ollama.list() and return a list of model name strings."""
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


@st.cache_data(ttl=120)
def get_installed_models():
    """Query Ollama for all installed models.

    Probes the default and fallback ports, auto-starts Ollama on the
    fallback port if needed, then queries the API. Falls back to reading
    the on-disk manifest registry if the API is still unreachable.

    Returns a list of model name strings (e.g. ['qwen3:14b', 'llava:latest']),
    or an empty list if nothing is found.
    """
    _ensure_ollama_host()

    try:
        installed = _list_models_via_api()
        if installed:
            return installed
    except Exception:
        pass

    # Fallback: scan the filesystem
    return _scan_ollama_filesystem()


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
        fs_models = _scan_ollama_filesystem()
        if fs_models:
            results.append({'component': 'ollama service', 'ok': True,
                            'detail': 'API blocked (VPN/firewall) — using local model registry'})
        else:
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
