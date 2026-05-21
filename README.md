# Quiz Vulnerability Assessment Framework (QVAF)

A decision-support tool that helps educators evaluate how resistant their online quiz questions are to AI-assisted answering.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TEQSA Aligned](https://img.shields.io/badge/TEQSA-2025%20Aligned-green.svg)](https://www.teqsa.gov.au/)

---

## What This Tool Does

QVAF helps you understand which quiz questions are vulnerable to students simply asking an AI "what's the answer?" by:

1. **Testing your quiz** against a local AI model in two conditions:
   - **Baseline**: AI answers using only general knowledge
   - **With course materials**: AI answers with access to your lecture notes/readings, uses Retreival Augmented Generation (RAG)

2. **Measuring AI performance across multiple question types**:
   - **Single-answer multiple choice** (radio buttons) — including images, tables, charts, and 5+ options
   - **Multi-answer multiple choice** (checkboxes) — "select all that apply" questions
   - **True/False** questions
   - Which questions does AI get right?
   - How confident is it?
   - How consistently does it answer?

3. **Classifying questions** by cognitive demand level (Recall → Strategic Integration)

4. **Generating recommendations** for questions that may benefit from redesign

The tool provides **information for your professional judgment**—it doesn't make decisions for you.

---

## Why This Matters

TEQSA's 2025 guidance on assessment reform is clear:

> "Design assessments where gen AI use is irrelevant to the demonstration of learning outcomes."

Rather than trying to detect AI use after the fact, QVAF helps you **proactively identify vulnerable questions** so you can redesign them before deployment.

**This is not about "AI-proofing" your quizzes**—that's impossible. It's about understanding your vulnerability profile and making informed decisions about acceptable risk.

📚 [Read the theoretical foundations →](QVAF_Literature_Review.md)

---

## Quick Start

### Prerequisites

- **Python 3.9+**
- **Google Chrome**
- **8GB RAM minimum** (16GB recommended)
- **15GB free disk space** (for AI models)

### Installation

#### macOS (One-Click)

1. Download or clone this repository
2. Double-click `First Time Setup.command`
3. Follow the prompts (~10-15 minutes)
4. When complete, double-click `Start Scanner.command`

#### macOS (Terminal)

```bash
# Navigate to project folder
cd ~/Documents/qvaf

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python -m playwright install chromium

# Install Ollama (if not already installed)
brew install ollama

# Download AI models (~9GB)
ollama pull llama3:8b
ollama pull llava

# Create output directories
mkdir -p output/raw_attempts output/reports output/dashboards temp_screenshots
```

#### Windows (One-Click)

1. Download or clone this repository
2. Double-click `First Time Setup.bat`
3. Follow the prompts (~10-15 minutes)
4. When complete, double-click `Start Scanner.bat`

#### Windows (PowerShell)

```powershell
# Navigate to project folder
cd C:\Users\YourName\Documents\qvaf

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
python -m pip install -r requirements.txt
python -m playwright install chromium

# Install Ollama from https://ollama.com/download/windows
# Then download models:
ollama pull llama3:8b
ollama pull llava

# Create output directories
mkdir output\raw_attempts, output\reports, output\dashboards, temp_screenshots
```

#### Linux

```bash
# Navigate to project folder
cd ~/Documents/qvaf

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python -m playwright install chromium

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download AI models
ollama pull llama3:8b
ollama pull llava

# Create output directories
mkdir -p output/raw_attempts output/reports output/dashboards temp_screenshots
```

---

## Usage

### Step 1: Start Chrome with Remote Debugging

Open a terminal and run:

**macOS:**
```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=/tmp/chrome-debug
```

**Windows:**
```cmd
"C:\Program Files\Google\Chrome\Application\chrome.exe" ^
  --remote-debugging-port=9222 --user-data-dir=%TEMP%\chrome-debug
```

**Linux:**
```bash
google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug
```

### Step 2: Log into Moodle

In the Chrome window that opened:
1. Navigate to your Moodle site
2. Log in
3. Go to the quiz you want to test
4. Start a quiz attempt (get to the first question)

### Step 3: Start QVAF

Open a **new terminal** (keep Chrome open):

```bash
cd ~/Documents/qvaf
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
python -m streamlit run App.py
```

The scanner opens in your browser at `http://localhost:8501`

### Step 4: Run Your Scan

1. **Connect tab**: Verify QVAF found your Moodle quiz page
2. **First Scan tab**: Click "Start Scan" → AI fills in answers → Submit quiz in Moodle → Click "Collect Results"
3. **Second Scan tab** (optional): Start new attempt in Moodle → Select course materials → Click "Start Scan" → Submit → Collect
4. **Results tab**: View analysis and download reports

---

## Setting Up Course Materials (RAG)

The RAG (Retrieval-Augmented Generation) feature lets you test whether AI performs better when given access to your course materials.

### Why Use RAG Testing?

- A question AI can only answer **with** your materials suggests students sharing notes is the vulnerability, not general AI capability
- A question AI answers correctly **without** your materials suggests it's testing general knowledge AI already has
- This distinction helps you decide where to focus redesign efforts

### Adding Materials

1. Go to the **Settings** tab
2. Click **Create New Course** and name it (e.g., "PSYC101")
3. Upload your materials:
   - Lecture slides (PDF)
   - Textbook excerpts (PDF, TXT)
   - Study guides (MD, TXT)
4. Run your second scan with this course selected

### Tips for Effective RAG

| Do | Don't |
|----|-------|
| Upload materials students actually have access to | Upload answer keys or solutions |
| Include lecture notes, required readings | Include every resource ever created |
| Use PDF exports of slides (preserves formatting) | Use massive single files (>50MB) |
| Create separate collections per course | Mix materials from different courses |

---

## Understanding Results

### AI Correctness Patterns

| Pattern | Baseline | With RAG | What It Suggests |
|---------|----------|----------|------------------|
| **CORRECT_BOTH** | ✓ | ✓ | AI succeeds regardless—consider redesign |
| **CORRECT_RAG_ONLY** | ✗ | ✓ | Course materials help AI—material-specific vulnerability |
| **INCORRECT_BOTH** | ✗ | ✗ | AI struggles—question may be naturally resistant |
| **CORRECT_BASELINE_ONLY** | ✓ | ✗ | Unusual—RAG confused AI (investigate) |

### Cognitive Demand Levels

QVAF classifies each question using a five-level cognitive demand taxonomy that synthesises Bloom's Revised Taxonomy (Anderson & Krathwohl, 2001), Webb's Depth of Knowledge (Webb, 1997), and the SOLO Taxonomy (Biggs & Collis, 1982). Classification uses a hybrid approach: high-confidence linguistic patterns in the question stem catch obvious cases (e.g., "What is [term]?" is always Recall, "Explain why [X]" is always Conceptual Understanding), while ambiguous questions proceed to LLM-based classification using formal definitions, boundary criteria, and a sequential decision flowchart. The classifier applies the lowest level that fully describes the cognitive demand to guard against over-classification. These classifications are suggestions for your validation — you know your questions and learning outcomes better than any automated system. See the [literature review](QVAF_Literature_Review.md#52-cognitive-demand-taxonomy) for full definitions, taxonomy mappings, and references.

| Level | Name | AI Vulnerability | Example |
|-------|------|-----------------|---------|
| 1 | **Recall** | Usually high | "Define supply chain management" |
| 2 | **Routine Application** | Moderate-high | "Calculate utilisation using this formula" |
| 3 | **Conceptual Understanding** | Moderate | "Explain why X causes Y" |
| 4 | **Analytical Reasoning** | Lower | "Evaluate which approach is better for..." |
| 5 | **Strategic Integration** | Usually low | "Design a solution for this novel scenario" |

### Interpreting Confidence Scores

QVAF elicits confidence using a calibrated 0–9 probability scale, converted to a 0–100% display value. This approach addresses the well-documented problem of LLM overconfidence, where models asked to rate confidence on a 0–100 scale overwhelmingly cluster at 80–90% (Xiong et al., 2024). The 0–9 scale produces better-calibrated scores for small models (Yang et al., 2024), and a "consider the opposite" debiasing step forces the AI to identify counter-arguments before rating its probability (Chhikara et al., 2025). See the [literature review](QVAF_Literature_Review.md#31-confidence-calibration-techniques) for details.

- **High confidence (80%+) + Correct**: AI found this easy
- **High confidence + Incorrect**: AI was confidently wrong (interesting signal!)
- **Low confidence (<50%)**: AI was uncertain—question may be resistant
- **Inconsistent answers** (e.g., 6/10): AI essentially guessing

---

## Dashboard Overview

The generated HTML dashboard includes:

- **Summary Statistics**: Overall AI accuracy, RAG effect
- **Question Breakdown**: Per-question results with cognitive classification
- **Pattern Distribution**: How many questions fall into each correctness pattern
- **Recommendations**: AI-generated suggestions for vulnerable questions

All recommendations require your professional evaluation—they're starting points for discussion, not prescriptions.

---

## Supported Question Types

| Moodle Type | Detection | How AI Answers | Notes |
|-------------|-----------|---------------|-------|
| **Multiple Choice (single answer)** | `multichoice` class + radio buttons | Single letter (e.g. "B") | Original supported type |
| **Multiple Choice (multi answer)** | `multichoice` class + checkboxes | Comma-separated letters (e.g. "A, C, D") | "Select all that apply" questions |
| **True/False** | `truefalse` class + radio buttons | Single letter ("A" for True, "B" for False) | Simplified prompt |

**Malformed question handling:** The scanner gracefully handles instructor errors including empty options (removed), duplicate option text (deduped), and misconfigured multi-answer questions with only one checkbox (downgraded to single-answer). Unrecognized question types with radio buttons are treated as single-answer MCQ (best-guess fallback).

---

## Project Structure

```
qvaf/
├── App.py                      # Main Streamlit application
├── config.py                   # Centralised configuration
├── quiz_browser_enhanced.py    # Browser automation + LLM interaction
├── parsing_utils.py            # LLM response parsing (single + multi-answer)
├── reform_agent.py             # Cognitive classification + analysis
├── analysis_agent.py           # Dashboard generation
├── merge_attempts.py           # Combines baseline + RAG results
├── app_test_question.py        # Test Question tab (single question testing)
├── requirements.txt            # Python dependencies
│
├── ARCHITECTURE.md             # Technical documentation
├── QVAF_Literature_Review.md   # Theoretical foundations + literature review
├── output/                     # Generated files
│   ├── raw_attempts/          # Quiz attempt JSON files
│   ├── reports/               # Analysis reports
│   └── dashboards/            # HTML dashboards
│
├── chroma_db/                  # RAG vector database
└── temp_screenshots/           # Temporary image captures
```

---

## Troubleshooting

### "Cannot connect to Chrome"

1. Make sure Chrome is running with `--remote-debugging-port=9222`
2. Close any other Chrome instances first
3. Don't open Chrome DevTools in the debug window

```bash
# Check if port is in use
lsof -i :9222  # macOS/Linux
netstat -an | findstr 9222  # Windows
```

### "No Moodle page found"

1. Make sure you're on the quiz **attempt** page (URL contains `/mod/quiz/attempt.php`)
2. Not the quiz info/description page
3. Try refreshing the page

### "Ollama not responding"

```bash
# Check if Ollama is running
ollama list

# Restart Ollama
ollama serve

# Verify model is installed
ollama pull llama3:8b
```

### Scan is very slow

- **Expected speed**: 2-5 seconds per question (baseline), longer with multi-sample mode
- **If slower**: Check RAM usage—models need ~8GB free
- **GPU acceleration**: NVIDIA or Apple Silicon significantly speeds inference

### Results don't match what I see in Moodle

1. Make sure you clicked "Collect Results" **after** submitting the quiz
2. Ensure correct answer feedback is visible in Moodle's review page
3. Check that QVAF detected the correct number of questions

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8GB | 16GB+ |
| **Disk Space** | 15GB free | 20GB+ |
| **CPU** | 64-bit (x86_64 or ARM64) | Multi-core |
| **GPU** | Not required | NVIDIA/Apple Silicon |
| **OS** | macOS 11+, Windows 10+, Ubuntu 20.04+ | Latest |

---

## Available AI Models

| Model | Size | Accuracy* | Best For |
|-------|------|----------|----------|
| `llama3:8b` | 4.7GB | ~95% | **Recommended default** |
| `mistral` | 4.1GB | ~60% | Faster, less accurate |
| `gemma2:9b` | 5.4GB | ~80% | Alternative |
| `llava` | 4GB | N/A | Image-based questions |

*Accuracy measured on internal test set; your results may vary.

---

## Limitations

QVAF is a **decision-support tool**, not an oracle. Important limitations:

1. **Single cheating behaviour**: Only models direct question→AI→answer. Doesn't cover paraphrasing, contract cheating, or sophisticated prompt engineering.

2. **Single AI model**: Uses one representative LLM. Students may use different models with different capabilities.

3. **Point-in-time**: AI capabilities evolve. A question resistant today may be vulnerable tomorrow.

4. **No guarantees**: There is no such thing as an "AI-proof" question.

5. **Classification is suggestive**: Cognitive demand classification is automated and should be validated by you.

The goal is **informed risk reduction**, not impossible perfection.

---

## Citation

If you use QVAF in research, please cite:

```bibtex
@software{qvaf2026,
  author = {Munoz, Albert},
  title = {Quiz Vulnerability Assessment Framework (QVAF)},
  year = {2026},
  url = {https://github.com/amunozUOW/QVAF}
}
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where help is particularly welcome:
- Support for additional question types (matching, short answer, drag-and-drop)
- Support for additional LMS platforms (Canvas, Blackboard)
- Additional LLM provider integrations
- Localisation/translation
- Empirical validation studies

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Built to support [TEQSA 2025 guidance](https://www.teqsa.gov.au/) on AI-irrelevant assessment design
- Uses [Ollama](https://ollama.com/) for local LLM inference
- Uses [Playwright](https://playwright.dev/) for browser automation
- Uses [ChromaDB](https://www.trychroma.com/) for vector storage
- Theoretical framework draws on work by Lodge, Bearman, Dawson, and Associates

---

## Related Resources

- [TEQSA Assessment Reform Guidance (2023)](https://www.teqsa.gov.au/)
- [TEQSA Enacting Assessment Reform (2025)](https://www.teqsa.gov.au/)
- [QAA Reconsidering Assessment for the ChatGPT Era](https://www.qaa.ac.uk/)
- [AI Assessment Scale (AIAS)](https://www.aiassessmentscale.com/)

---

*QVAF is developed at the University of Wollongong as part of research into AI-resistant assessment design.*
