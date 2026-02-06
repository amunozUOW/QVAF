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

DEFAULT_MODEL = "llama3:8b"
VISION_MODEL = "llava"
OLLAMA_HOST = "http://localhost:11434"

# Available models (in order of recommendation)
AVAILABLE_MODELS = {
    'llama3:8b': 'Llama 3 8B (recommended, 100% test accuracy)',
    'mistral': 'Mistral 7B (faster, 60% test accuracy)',
    'gemma2:9b': 'Gemma 2 9B (alternative)',
    'mixtral': 'Mixtral 8x7B (best quality, needs 26GB RAM)',
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
