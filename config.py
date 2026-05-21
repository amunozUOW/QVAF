#!/usr/bin/env python3
"""
Configuration Module for Quiz Vulnerability Scanner
====================================================

Centralizes all paths and settings for consistent behavior across the application.
"""

import os
from pathlib import Path

# ============================================
# BASE PATHS
# ============================================

# Project root directory (where this file lives)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
RAW_ATTEMPTS_DIR = OUTPUT_DIR / "raw_attempts"
REPORTS_DIR = OUTPUT_DIR / "reports"
DASHBOARDS_DIR = OUTPUT_DIR / "dashboards"

# Data directories
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
TEMP_SCREENSHOTS_DIR = PROJECT_ROOT / "temp_screenshots"

# ============================================
# LLM CONFIGURATION
# ============================================

DEFAULT_MODEL = "llama3.2:3b"
VISION_MODEL = "llava"
OLLAMA_HOST = "http://localhost:11434"

# Friendly labels for known models (models not listed here still appear in the
# selector — the sidebar generates a label automatically for unlisted ones).
AVAILABLE_MODELS = {
    'llama3.2:3b': 'Llama 3.2 3B (recommended — fast, 2 GB)',
    'llama3.2:1b': 'Llama 3.2 1B (ultra-light, 1.3 GB)',
    'llama3.1:8b': 'Llama 3.1 8B (higher accuracy, 4.7 GB)',
    'llama3:8b': 'Llama 3 8B (legacy)',
    'gemma2:9b': 'Gemma 2 9B (alternative, 5.4 GB)',
    'qwen3:14b': 'Qwen 3 14B (high quality, 9 GB)',
    'qwen2.5:14b': 'Qwen 2.5 14B (high quality, 9 GB)',
    'qwen2.5:7b': 'Qwen 2.5 7B (alternative, 4.7 GB)',
    'phi4:14b': 'Phi-4 14B (high quality, 9 GB)',
    'mistral': 'Mistral 7B (fast, 4.1 GB)',
    'mistral:latest': 'Mistral 7B (fast, 4.1 GB)',
    'deepseek-r1:8b': 'DeepSeek-R1 8B (reasoning, 4.9 GB)',
}

# ============================================
# BROWSER AUTOMATION
# ============================================

CHROME_DEBUG_PORT = 9222
CHROME_CDP_URL = f"http://localhost:{CHROME_DEBUG_PORT}"

# URL patterns to identify Moodle/LMS pages
MOODLE_URL_PATTERNS = [
    'moodle', '/mod/quiz/', 'quiz/attempt', 'quiz/view',
    'lms.', 'learn.', 'elearning', 'blackboard', 'canvas', 'brightspace',
]

# URLs to exclude (internal Chrome pages)
EXCLUDE_URL_PATTERNS = [
    'chrome://', 'chrome-extension://', 'devtools://', 'about:', 'edge://',
]

# ============================================
# RAG CONFIGURATION
# ============================================

# Collection naming: all user collections are prefixed with this
RAG_COLLECTION_PREFIX = "rag_"
DEFAULT_COLLECTION_NAME = "Default"
CHROMA_COLLECTION_NAME = f"{RAG_COLLECTION_PREFIX}Default"
RAG_TOP_K_RESULTS = 3

def get_rag_collection_name(user_name: str) -> str:
    """Convert user-friendly name to internal collection name."""
    # Sanitize: replace spaces and special chars
    safe_name = "".join(c if c.isalnum() else "_" for c in user_name)
    return f"{RAG_COLLECTION_PREFIX}{safe_name}"

def get_display_name(collection_name: str) -> str:
    """Convert internal collection name to user-friendly name."""
    if collection_name.startswith(RAG_COLLECTION_PREFIX):
        return collection_name[len(RAG_COLLECTION_PREFIX):].replace("_", " ")
    return collection_name

# ============================================
# DIRECTORY INITIALIZATION
# ============================================

def ensure_directories():
    """Create all required directories if they don't exist."""
    for directory in [OUTPUT_DIR, RAW_ATTEMPTS_DIR, REPORTS_DIR, DASHBOARDS_DIR, TEMP_SCREENSHOTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def get_output_path(filename: str, category: str = "raw") -> Path:
    """
    Get the appropriate output path for a file.

    Args:
        filename: The filename (with extension)
        category: One of "raw", "report", "dashboard"

    Returns:
        Full path to the output file
    """
    ensure_directories()

    if category == "raw":
        return RAW_ATTEMPTS_DIR / filename
    elif category == "report":
        return REPORTS_DIR / filename
    elif category == "dashboard":
        return DASHBOARDS_DIR / filename
    else:
        return OUTPUT_DIR / filename

# Initialize directories on import
ensure_directories()
