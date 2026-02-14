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
