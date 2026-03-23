# Architecture Overview

This document provides technical documentation for the Quiz Vulnerability Assessment Framework (QVAF). It is intended for developers, contributors, and technically-inclined educators who want to understand how the system works.

---

## System Overview

QVAF is a Python-based application that:

1. Connects to a browser running a Moodle quiz via Chrome DevTools Protocol
2. Extracts quiz questions (text and images)
3. Submits each question to a local LLM in two conditions (baseline and RAG-augmented)
4. Records AI answers, confidence levels, and consistency scores
5. Compares AI answers against correct answers after quiz submission
6. Classifies questions by cognitive demand
7. Generates reports and dashboards for educator review

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│                           (Streamlit: App.py)                               │
│                                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│  │ Connect │  │ First   │  │ Second  │  │ Results │  │Settings │          │
│  │         │  │ Scan    │  │ Scan    │  │         │  │         │          │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘          │
└───────┼────────────┼────────────┼────────────┼────────────┼───────────────┘
        │            │            │            │            │
        ▼            ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CORE MODULES                                       │
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────┐                      │
│  │ quiz_browser_        │    │ reform_agent.py      │                      │
│  │ enhanced.py          │    │                      │                      │
│  │                      │    │ • Cognitive          │                      │
│  │ • Browser automation │    │   classification     │                      │
│  │ • Question scraping  │    │ • Correctness        │                      │
│  │ • LLM interaction    │    │   checking           │                      │
│  │ • Multi-sample mode  │    │ • Pattern analysis   │                      │
│  └──────────┬───────────┘    └──────────┬───────────┘                      │
│             │                           │                                   │
│             ▼                           ▼                                   │
│  ┌──────────────────────┐    ┌──────────────────────┐                      │
│  │ analysis_agent.py    │    │ merge_attempts.py    │                      │
│  │                      │    │                      │                      │
│  │ • Dashboard HTML     │    │ • Combines baseline  │                      │
│  │ • Statistics         │    │   and RAG results    │                      │
│  │ • Visualizations     │    │ • Score alignment    │                      │
│  └──────────────────────┘    └──────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
        │                                   │
        ▼                                   ▼
┌─────────────────────────┐    ┌─────────────────────────────────────────────┐
│   EXTERNAL SERVICES     │    │              DATA STORES                     │
│                         │    │                                             │
│  ┌───────────────────┐  │    │  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Ollama            │  │    │  │ ChromaDB        │  │ JSON Files      │  │
│  │ (Local LLM)       │  │    │  │ (RAG Store)     │  │ (Results)       │  │
│  │                   │  │    │  │                 │  │                 │  │
│  │ • llama3:8b       │  │    │  │ • Course        │  │ • Attempts      │  │
│  │ • llava (vision)  │  │    │  │   materials     │  │ • Reports       │  │
│  └───────────────────┘  │    │  │ • Embeddings    │  │ • Analyses      │  │
│                         │    │  └─────────────────┘  └─────────────────┘  │
│  ┌───────────────────┐  │    │                                             │
│  │ Chrome (CDP)      │  │    │  ┌─────────────────┐                        │
│  │                   │  │    │  │ HTML Dashboards │                        │
│  │ • Debug port 9222 │  │    │  │ (Generated)     │                        │
│  │ • Moodle session  │  │    │  └─────────────────┘                        │
│  └───────────────────┘  │    │                                             │
└─────────────────────────┘    └─────────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Interface** | Streamlit | User-facing application |
| **Browser Automation** | Playwright | Moodle interaction via CDP |
| **Local LLM** | Ollama | Text generation and analysis |
| **Vision Model** | LLaVA (via Ollama) | Image-based question interpretation |
| **Vector Database** | ChromaDB | RAG storage for course materials |
| **Data Format** | JSON | Intermediate and output files |
| **Reports** | HTML + CSS | Interactive dashboards |

---

## Module Reference

### App.py (Main Application)

The Streamlit web application that orchestrates the entire workflow.

**Key responsibilities:**
- Session state management
- Chrome connection handling
- Tab-based workflow navigation
- File upload and RAG management
- Report generation triggers

**Key functions:**

```python
def find_moodle_page(browser)
    """Locate Moodle quiz page from available Chrome pages."""
    
def check_chrome()
    """Verify Chrome is running with debug port."""
    
def check_ollama()
    """Verify Ollama is running and models are available."""
```

**Session state variables:**

| Variable | Type | Purpose |
|----------|------|---------|
| `no_rag_file` | str | Path to baseline scan results |
| `with_rag_file` | str | Path to RAG scan results |
| `merged_file` | str | Path to combined results |
| `no_rag_score` | dict | Actual quiz score from Moodle (baseline) |
| `with_rag_score` | dict | Actual quiz score from Moodle (RAG) |
| `selected_rag_collection` | str | Currently selected course materials |
| `use_rag_mode` | bool | Full scan (True) or basic scan (False) |

---

### quiz_browser_enhanced.py (Browser Automation)

Handles all browser interaction and LLM communication.

**Key responsibilities:**
- Connect to Chrome via CDP (port 9222)
- Navigate Moodle quiz pages
- Detect question types (single-answer MCQ, multi-answer MCQ, True/False)
- Extract question text, options, and images
- Submit questions to Ollama with type-aware prompts
- Click radio buttons (single-answer) or checkboxes (multi-answer)
- Handle multi-sample consistency measurement
- Validate malformed questions (empty options, duplicates, misconfigurations)
- Manage RAG retrieval when enabled

**Question type detection:**

```python
def detect_question_type(css_classes, has_checkboxes, has_radios, prompt_text) -> str:
    """
    Detect Moodle question type from CSS classes and input elements.

    Returns one of:
    - 'multichoice_single'  (radio buttons)
    - 'multichoice_multi'   (checkboxes)
    - 'truefalse'           (2 radio buttons)
    - 'unknown'             (unrecognized — skipped with warning)

    Fallback: unknown types with radio buttons → 'multichoice_single' (best-guess).
    """

def validate_question(q_type, options, checkbox_buttons, radio_buttons) -> tuple:
    """
    Validate and clean up malformed questions before LLM prompting.

    Handles: empty options, duplicate text, single-checkbox multi-answer.
    Returns: (q_type, options, checkbox_buttons, radio_buttons, warnings)
    """
```

**Key classes:**

```python
class QuizBrowser:
    """Main browser automation class."""

    def connect(self) -> bool
        """Connect to Chrome debug port."""

    def scrape_question(self) -> dict
        """Extract current question from page."""

    def answer_question(self, question: dict, use_rag: bool) -> dict
        """Submit question to LLM and get response."""

    def navigate_next(self) -> bool
        """Move to next question in quiz."""
```

**Type-aware prompting:**

`build_prompt()` accepts a `q_type` parameter and adapts the prompt:
- `multichoice_single` (default): "Answer this multiple choice question." → `ANSWER: X` (single letter)
- `multichoice_multi`: "Select ALL correct options." → `ANSWER: X, Y, Z` (comma-separated)
- `truefalse`: "Answer this true/false question." → `ANSWER: A` (True) or `ANSWER: B` (False)

All question types include forced alternative enumeration: the LLM must name a runner-up option (`ALTERNATIVE`) with rationale (`ALT_RATIONALE`) before assigning its probability score (`PROBABILITY` is the last output field).

**Multi-sample mode:**

When `--samples N` is specified (N > 1), each question is submitted to the LLM N times:

```python
def answer_question_multi_sample(self, question: dict, n: int) -> dict:
    """
    Run question through LLM n times.

    Returns:
        {
            'answer': str,           # Plurality vote winner
            'confidence': int,       # Average confidence
            'consistency': str,      # e.g., "8/10"
            'distribution': dict     # Answer -> count mapping
        }
    """
```

For multi-answer questions, consistency compares full answer sets (e.g., "A, C, D" treated as a single unit).

**RAG integration:**

```python
def get_rag_context(self, question_text: str, collection: str) -> str:
    """
    Retrieve relevant course materials for question.
    
    Uses ChromaDB semantic search to find top-k chunks.
    """
```

---

### reform_agent.py (Analysis Engine)

Classifies questions and calculates objective metrics.

**Key responsibilities:**
- Cognitive demand classification (via LLM)
- Correctness checking
- Pattern determination
- Report generation

**Key functions:**

```python
def classify_question_type(question_text: str, options: dict) -> str:
    """
    Classify question into cognitive demand category.
    
    Returns one of:
    - RECALL
    - ROUTINE APPLICATION
    - CONCEPTUAL UNDERSTANDING
    - ANALYTICAL REASONING
    - STRATEGIC INTEGRATION
    """

def get_correctness_pattern(correct_without: bool, correct_with: bool) -> str:
    """
    Determine descriptive correctness pattern.
    
    Returns one of:
    - CORRECT_BOTH
    - CORRECT_RAG_ONLY
    - INCORRECT_BOTH
    - CORRECT_BASELINE_ONLY
    - UNKNOWN
    """

def process_quiz(filepath: str) -> dict:
    """
    Main processing function.
    
    Loads merged quiz data, classifies all questions,
    calculates metrics, generates report.
    """
```

**Classification prompt:**

The cognitive demand classification uses a structured prompt that asks the LLM to:
1. Identify what type of thinking is primarily required
2. Match to taxonomy levels with justification
3. Return standardised category label

---

### analysis_agent.py (Dashboard Generator)

Generates HTML dashboards and summary statistics.

**Key responsibilities:**
- Calculate aggregate statistics
- Generate visualisations
- Create interactive HTML reports
- Format data for educator review

**Key functions:**

```python
def calculate_grades(results: list, actual_scores: dict) -> dict:
    """
    Calculate AI correctness percentages.
    
    Returns grades for baseline and RAG conditions,
    plus RAG effect (difference).
    """

def calculate_grades_by_type(results: list) -> dict:
    """
    Break down AI performance by cognitive demand level.
    """

def generate_dashboard(report: dict, output_path: str) -> str:
    """
    Generate interactive HTML dashboard.
    
    Includes:
    - Summary statistics
    - Question-by-question breakdown
    - Cognitive type distribution
    - Correctness pattern visualisation
    """
```

---

### merge_attempts.py (Data Combiner)

Combines baseline and RAG scan results into unified format.

**Key responsibilities:**
- Align questions across scans
- Merge AI responses
- Preserve actual scores from Moodle
- Handle basic mode (single scan)

```python
def merge_attempts(no_rag_path: str, with_rag_path: str, 
                   no_rag_score: dict, with_rag_score: dict) -> str:
    """
    Merge two quiz attempts into single analysis file.
    
    Returns path to merged JSON file.
    """
```

---

## Data Flow

### Scan Workflow

```
1. USER: Navigates to Moodle quiz, starts attempt
                    │
                    ▼
2. APP: Connects to Chrome via CDP
                    │
                    ▼
3. BROWSER: For each question:
   ├── Scrape question text and options
   ├── Capture images (if present)
   ├── Submit to LLM (baseline or RAG)
   ├── Record answer, confidence, reasoning
   ├── Fill in answer on page
   └── Navigate to next question
                    │
                    ▼
4. USER: Submits quiz in Moodle
                    │
                    ▼
5. APP: Scrapes results page for actual scores
                    │
                    ▼
6. APP: Saves attempt to JSON
```

### Analysis Workflow

```
1. MERGE: Combine baseline + RAG attempts
                    │
                    ▼
2. REFORM: For each question:
   ├── Check correctness against actual answers
   ├── Classify cognitive demand
   ├── Determine correctness pattern
   └── Generate analysis text
                    │
                    ▼
3. ANALYSIS: Calculate aggregate statistics
                    │
                    ▼
4. DASHBOARD: Generate HTML report
```

---

## File Formats

### Quiz Attempt JSON

```json
{
  "timestamp": "2026-02-05T10:30:00",
  "quiz_name": "Week 5 Quiz",
  "model": "llama3:8b",
  "mode": "no_rag",
  "questions": [
    {
      "number": 1,
      "question": "What is the primary purpose of...",
      "type": "multichoice_single",
      "options": {
        "A": "Option text A",
        "B": "Option text B",
        "C": "Option text C",
        "D": "Option text D"
      },
      "response": {
        "answer": "B",
        "confidence": 85,
        "reasoning": "Based on the definition...",
        "alternative": "C",
        "alt_rationale": "C is related but focuses on transformed resources.",
        "consistency": "9/10"
      },
      "correct_answer": "B",
      "is_correct": true
    },
    {
      "number": 2,
      "question": "Select all that apply: Which of the following are...",
      "type": "multichoice_multi",
      "options": {
        "A": "Option A",
        "B": "Option B",
        "C": "Option C",
        "D": "Option D"
      },
      "response": {
        "answer": "A, C, D",
        "confidence": 78,
        "reasoning": "Options A, C and D are correct because...",
        "alternative": "B",
        "alt_rationale": "B was nearly included but is less directly relevant."
      },
      "correct_answer": "A, C, D",
      "is_correct": true
    }
  ]
}
```

**Question type values:** `multichoice_single`, `multichoice_multi`, `truefalse`

### Merged Results JSON

```json
{
  "timestamp": "2026-02-05T10:45:00",
  "quiz_name": "Week 5 Quiz",
  "no_rag_score": {
    "correct": 8,
    "total": 10,
    "percentage": 80.0
  },
  "with_rag_score": {
    "correct": 9,
    "total": 10,
    "percentage": 90.0
  },
  "questions": [
    {
      "number": 1,
      "question": "...",
      "options": {...},
      "format_type": "multichoice_single",
      "correct_answer": "B",
      "response_without_rag": {
        "answer": "C",
        "confidence": 70,
        "reasoning": "..."
      },
      "response_with_rag": {
        "answer": "B",
        "confidence": 90,
        "reasoning": "..."
      }
    }
  ]
}
```

### Analysis Report JSON

```json
{
  "metadata": {
    "source_file": "quiz_attempt_merged.json",
    "generated_at": "2026-02-05T11:00:00",
    "framework_version": "2.0"
  },
  "scan_mode": "full",
  "actual_scores": {
    "no_rag_score": {"correct": 8, "total": 10, "percentage": 80.0},
    "with_rag_score": {"correct": 9, "total": 10, "percentage": 90.0}
  },
  "quantitative_summary": {
    "total_questions": 10,
    "correct_without_rag": 8,
    "correct_with_rag": 9,
    "pattern_counts": {
      "CORRECT_BOTH": 7,
      "CORRECT_RAG_ONLY": 2,
      "INCORRECT_BOTH": 1,
      "CORRECT_BASELINE_ONLY": 0
    }
  },
  "question_results": [
    {
      "id": 1,
      "question": "...",
      "question_type": "RECALL",
      "correctness_pattern": "CORRECT_BOTH",
      "correct_without_rag": true,
      "correct_with_rag": true,
      "confidence_without_rag": 85,
      "confidence_with_rag": 90
    }
  ]
}
```

---

## Configuration

### config.py

Central configuration file for paths and defaults:

```python
# Paths
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
RAW_ATTEMPTS_DIR = OUTPUT_DIR / "raw_attempts"
REPORTS_DIR = OUTPUT_DIR / "reports"
DASHBOARDS_DIR = OUTPUT_DIR / "dashboards"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
TEMP_SCREENSHOTS_DIR = PROJECT_ROOT / "temp_screenshots"

# Models
DEFAULT_MODEL = "llama3:8b"
VISION_MODEL = "llava"
AVAILABLE_MODELS = {
    "llama3:8b": "Recommended - best accuracy",
    "mistral": "Faster but less accurate",
    "gemma2:9b": "Alternative option"
}

# Browser detection patterns
MOODLE_URL_PATTERNS = [
    'moodle', '/mod/quiz/', 'quiz/attempt', 'quiz/view',
    'lms.', 'learn.', 'elearning', 'blackboard', 'canvas'
]
EXCLUDE_URL_PATTERNS = [
    'chrome://', 'chrome-extension://', 'devtools://', 'about:'
]

# RAG settings
RAG_COLLECTION_PREFIX = "course_"
DEFAULT_COLLECTION_NAME = "Default Course"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `CHROME_DEBUG_PORT` | `9222` | Chrome DevTools Protocol port |

---

## Extending QVAF

### Adding a New LLM Provider

1. Create adapter in `llm_adapters/` directory
2. Implement standard interface:

```python
class LLMAdapter:
    def generate(self, prompt: str, options: dict) -> str:
        """Generate text response."""
        
    def generate_with_image(self, prompt: str, image_path: str) -> str:
        """Generate response for image-based question."""
```

3. Register in `config.py` AVAILABLE_MODELS

### Adding a New LMS

1. Create detector in `lms_detectors/` directory
2. Implement page detection:

```python
class LMSDetector:
    url_patterns: list[str]
    
    def is_quiz_page(self, url: str, title: str) -> bool:
        """Check if page is a quiz."""
    
    def scrape_question(self, page) -> dict:
        """Extract question from page."""
```

3. Register in `config.py` URL patterns

### Adding New Report Formats

1. Create generator in `report_generators/` directory
2. Implement standard interface:

```python
class ReportGenerator:
    def generate(self, report_data: dict, output_path: str) -> str:
        """Generate report file, return path."""
```

---

## Testing

### Running Tests

```bash
# Unit tests
python -m pytest tests/

# Integration tests (requires Ollama running)
python -m pytest tests/integration/ --integration

# Coverage report
python -m pytest --cov=. --cov-report=html
```

### Test Structure

```
tests/
├── test_response_parsing.py      # LLM response parsing (single + multi-answer)
├── test_correctness.py           # Correctness checking (single + multi-answer)
├── test_option_parsing.py        # Option label stripping
├── test_question_matching.py     # Question text matching for merge
├── test_stem_classifier.py       # Cognitive demand classification
├── test_type_detection.py        # Moodle question type detection
└── test_malformed_questions.py   # Malformed question validation
```

---

## Performance Considerations

### LLM Inference

- Each question requires 1-2 LLM calls (more with multi-sample mode)
- Local inference with Llama 3 8B: ~2-5 seconds per question
- Multi-sample mode (n=10): ~20-50 seconds per question

### Memory Requirements

| Component | Memory |
|-----------|--------|
| Llama 3 8B model | ~5 GB |
| LLaVA vision model | ~4 GB |
| ChromaDB (typical) | ~100 MB |
| Streamlit app | ~200 MB |
| **Total recommended** | **16 GB RAM** |

### Optimisation Tips

1. Use `temperature=0` for consistent results
2. Limit `num_predict` to reduce token generation
3. Pre-load models before scan starts
4. Use SSD for ChromaDB storage

---

## Security Considerations

### Local-First Design

QVAF is designed to run entirely locally:
- No data sent to external APIs
- LLM inference via local Ollama
- Quiz content stays on educator's machine

### Credential Handling

- No credentials stored by QVAF
- Educator logs into Moodle in their browser
- QVAF reads page content via CDP (no credential access)

### Data Sensitivity

Quiz questions may be sensitive. QVAF:
- Stores data only in local output directories
- Does not transmit data externally
- Allows educator to delete all generated files

---

*Document version: 1.1*
*Framework version: QVAF 2.1*
*Last updated: March 2026*
