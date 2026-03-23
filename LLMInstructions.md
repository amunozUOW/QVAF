# LLM Instructions — QVAF

Purpose: Actionable guidance for AI coding agents working in this repository. Consolidates machine-oriented invariants, big-picture architecture, developer workflows, and design principles.

---

## 0) Big Picture & Architecture

**Core Components**
- **UI / Orchestration:** `App.py` is a Streamlit front-end that orchestrates scans, displays progress (parses `[PROGRESS]` lines), and launches `quiz_browser_enhanced.py` to perform automation.
- **Automation:** `quiz_browser_enhanced.py` uses Playwright to control Chrome (connects to CDP on port 9222), fills answers, optionally uses RAG, and writes JSON outputs to `output/raw_attempts/`.
- **Analysis & Reports:** `analysis_agent.py`, `reform_agent.py`, and `merge_attempts.py` consume raw attempts and produce reports and dashboards in `output/reports/` and `output/dashboards/`.
- **RAG storage:** Vector DB files live under `chroma_db/`. Collections use prefix `rag_` (see `config.get_rag_collection_name`).

**Key Integration Points**
- Playwright: `quiz_browser_enhanced.py` connects to Chrome via `http://localhost:9222` (see `config.CHROME_CDP_URL`).
- Ollama: LLM calls use `ollama.chat(...)` (model names like `llama3:8b`, `llava` for vision). Use `ollama list`, `ollama serve`, `ollama pull <model>` to manage models.
- ChromaDB: Optional RAG integration; initialize via `chromadb.PersistentClient(path=CHROMA_DB_PATH)`.

**Operational Assumption**
Educators expect local-first, privacy-preserving behaviour: do not add network telemetry or remote-services without explicit consent in docs and code.

---

## 1) Critical Invariants (do NOT change these silently)
- Progress tokens emitted by `quiz_browser_enhanced.py` that `App.py` parses: lines containing the literal marker `[PROGRESS]` with phrases such as `Question`, `AI thinking`, `Running N samples`, `Answer:`, `Page complete`. If you change these strings, update `App.py` parsing logic accordingly.
- LLM response contract (must be parseable by `parse_llm_response_full()` or `parse_llm_response_multi_full()`):

  Single-answer (multichoice_single, truefalse):
  ANSWER: X
  ALTERNATIVE: Y
  ALT_RATIONALE: ...
  REASONING: ...
  DOUBT: ...
  PROBABILITY: N

  Multi-answer (multichoice_multi):
  ANSWER: X, Y, Z
  ALTERNATIVE: Y
  ALT_RATIONALE: ...
  REASONING: ...
  DOUBT: ...
  PROBABILITY: N

  Where X is letter A–H (single) or comma-separated letters (multi), Y is the next most plausible option letter or "none", and N is a digit 0–9 (converted to 0–100%). PROBABILITY is intentionally last to enforce reasoning-before-number ordering. The `_full()` parsers return dicts; the legacy 3-tuple parsers (`parse_llm_response()`, `parse_llm_response_multi()`) remain for backward compatibility. All live in `parsing_utils.py`.
- Output filename pattern (scanner): `quiz_attempt_{mode}_{YYYYmmdd_HHMMSS}.json` and `mode` is `no_rag` or `with_rag`. UI looks for `_no_rag_` / `_with_rag_` in filenames.
- RAG collection naming: `get_rag_collection_name(user_name)` prefixes with `RAG_COLLECTION_PREFIX` from `config.py` and sanitises non-alphanumeric chars to `_`. Do not change prefix without updating UI and docs.
- Chrome CDP port: `CHROME_DEBUG_PORT` default 9222 (in `config.py`). `quiz_browser_enhanced.py` connects to `http://localhost:9222` — keep consistent.

---

## 2) Developer Workflows & Exact Commands

- Start the app (dev):

```bash
python -m streamlit run App.py
```

- Start Chrome for automation (macOS example):

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug
```

- Run an automated baseline scan (no RAG):

```bash
python3 quiz_browser_enhanced.py --no-rag --model llama3:8b --samples 1
```

- Run with course materials (RAG):

```bash
python3 quiz_browser_enhanced.py --with-rag --collection rag_PSYC101 --samples 3
```

- Useful debug flags: `--debug` (prints available pages and saves screenshots), `--list-pages` (just list pages), `--no-wait` (exit after filling answers so UI can collect results).

---

## 3) Adapter contract (how an LLM adapter must behave)
Minimal adapter implements two functions and returns a single text response matching the LLM response contract above.

Interface (conceptual):

```python
class LLMAdapter:
    def generate(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 300) -> str:
        """Return a single text string containing ANSWER / CONFIDENCE / REASONING block."""

    def generate_with_image(self, prompt: str, image_b64: str, *, temperature: float = 0.0) -> str:
        """Return string — same contract but may use image context."""
```

Important mapping notes when adapting non-Ollama providers:
- `temperature=0` is used to reduce variability but multi-sample mode expects some non-determinism; keep sampling logic compatible.
- `num_predict` in Ollama ≈ `max_tokens` or `max_new_tokens` replacement depending on API — ensure you cap outputs to avoid truncation of the `REASONING` field.
- Image handling: QVAF encodes images as base64 and sends them in `messages` with an `images` list when using Ollama/LLava. If your provider wants multipart/form-data or a separate vision endpoint, the adapter must merge the returned text into the standard contract.

---

## 4) Project-specific Conventions & Patterns

- Output file naming: `quiz_attempt_{mode}_{YYYYmmdd_HHMMSS}.json` saved under `output/raw_attempts/`. UI looks for new files matching `_no_rag_` or `_with_rag_`.
- RAG collections: user-visible names converted to internal names via `get_rag_collection_name()` (prefix `rag_`; non-alphanumeric replaced with `_`).
- UI progress parsing: `App.py` expects scanner stdout to emit lines containing `[PROGRESS]` with short human-readable tokens (e.g., `Question`, `AI thinking`, `Answer:`). Avoid changing these tokens when editing `quiz_browser_enhanced.py` unless updating the UI parsing logic as well.
- LLM response format: prompts are constructed in `quiz_browser_enhanced.build_prompt(q_type=...)` and the code expects answers in type-specific formats:
  - **Single-answer** (`multichoice_single`, `truefalse`): `ANSWER: X` (one letter) → parsed by `parse_llm_response_full()`
  - **Multi-answer** (`multichoice_multi`): `ANSWER: X, Y, Z` (comma-separated) → parsed by `parse_llm_response_multi_full()`
  - Both formats include `ALTERNATIVE: Y` (runner-up letter or "none"), `ALT_RATIONALE: ...`, `REASONING: ...`, `DOUBT: ...`, and `PROBABILITY: N` (0–9 scale, intentionally last)
  - Multi-answer responses are validated by `validate_multi_answer()` which filters invalid letters against the actual option keys
  - `build_prompt()` accepts `q_type` parameter and adapts instructions accordingly
  - Legacy 3-tuple parsers (`parse_llm_response()`, `parse_llm_response_multi()`) still work for callers that only need answer/confidence/reasoning

  Keep these formats stable or update prompts, parsers (`parsing_utils.py`), and UI simultaneously.

**What to change (and how)**
- To add a new CLI flag for scanning, update `quiz_browser_enhanced.py` (argument parsing at bottom) and mirror any progress/output tokens used by `App.py`.
- To change storage paths or defaults, update `config.py` (single source of truth) — e.g., `DEFAULT_MODEL`, `CHROMA_DB_PATH`, `RAG_COLLECTION_PREFIX`.
- For RAG collection name changes, ensure `get_rag_collection_name()` and UI-facing labels in `App.py` remain consistent.

**Where to look for examples**
- Running scans from UI: `App.py` → `run_quiz()` builds `['python3','quiz_browser_enhanced.py', ...]` and parses `[PROGRESS]` messages.
- Multi-sample behaviour and consistency measurement: `quiz_browser_enhanced.call_llm_multi_sample()` and returned keys like `consistency_pct`, `answer_distribution`.
- Image handling: `process_question_images()` → `interpret_image_with_llava()` shows how images are base64-encoded and sent to the vision model.

**Quick debugging checklist**
- If Chrome connection fails: confirm Chrome started with `--remote-debugging-port=9222` and run `lsof -i :9222` (macOS/Linux).
- If Ollama calls fail: verify `ollama` is installed, `ollama list` shows models, and `OLLAMA` daemon is running.
- If no JSON output appears: scan `output/raw_attempts/` and root folder for `quiz_attempt_*.json` (the app will move fallback files into the correct folder).

---

## 5) Prompt and parsing contract (where to edit carefully)
- Prompt builder: `quiz_browser_enhanced.build_prompt(q_type=...)` composes `QUESTION`, `OPTIONS`, and optional context blocks (`COURSE MATERIALS`, `IMAGE CONTENT`, `LINKED CONTENT`). The prompt adapts based on `q_type`:
  - `multichoice_single` (default): asks for single best answer
  - `multichoice_multi`: asks to select ALL correct options, expects comma-separated letters
  - `truefalse`: simplified True/False prompt
  The prompt includes forced alternative enumeration (ALTERNATIVE + ALT_RATIONALE fields) and places PROBABILITY last to enforce reasoning-before-number ordering.
  When editing the prompt, preserve the ending instructions that require EXACT output formatting.
- Parsers (all in `parsing_utils.py`):
  - `parse_llm_response_full(text)` returns dict with `answer`, `confidence`, `reasoning`, `alternative`, `alt_rationale` — primary parser for single-answer and true/false
  - `parse_llm_response_multi_full(text)` returns dict with same keys — primary parser for multi-answer
  - `parse_llm_response(text)` legacy 3-tuple (`answer`, `confidence`, `reasoning`) — backward compat
  - `parse_llm_response_multi(text)` legacy 3-tuple — backward compat
  - `validate_multi_answer(answers, valid_options)` filters invalid letters against actual option keys
  - `extract_alternative(text)` extracts ALTERNATIVE field (letter, "none", or "")
  - `extract_alt_rationale(text)` extracts ALT_RATIONALE field
  If you change the expected format, update both the prompt and parser simultaneously and add a unit test (see `tests/test_response_parsing.py`).

---

## 6) Quick sanity commands (one-liners)
- Start Streamlit app:

```bash
python -m streamlit run App.py
```

- Start Chrome with CDP (macOS example):

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug
```

- Check Ollama models:

```bash
ollama list
ollama show llama3:8b
```

- List Chrome pages via Playwright (debug):

```bash
python3 quiz_browser_enhanced.py --list-pages
```

---

---

## 7) Quick adapter test (manual)
If you want to validate an adapter without running full automation, use `build_prompt()` and `parse_llm_response()` interactively (examples earlier in this document). Keep such scripts local and avoid adding test artifacts to the repository unless you're explicitly adding a test suite.

---

## 8) Overarching Principles for LLM Instructions and Agent Behaviour
These principles are distilled from repository intent and tone (educator-first, decision-support, privacy-local-first, cautious about automation). Use them when designing prompts, automated edits, or agent policies.

- Decision-support not prescriptive: always frame outputs and code changes as suggestions. Avoid making automatic, irreversible changes that could alter educator-facing behaviour without explicit confirmation or tests.
- Preserve user control & privacy: default behaviour should be local-first (Ollama/chromadb local). Do not add remote calls or telemetry unless documented and opt-in.
- Minimal, reversible edits: prefer small atomic changes with tests and a clear migration path. If you change parsing or progress tokens, add migration code and keep backwards-compatibility for one minor release.
- Evidence-first recommendations: when generating suggested code or UI wording, prefer explicit, testable invariants and examples rather than high-level prose.
- Asset-based reporting: avoid language that frames findings as blame; preserve UI tokens and analysis outputs that educators expect. Changes to report wording should keep asset-based framing (lead with strengths).
- Ask before big assumptions: if an automated change requires changing a config invariant (progress token, filename pattern, RAG prefix), raise a PR with a clearly documented rationale and tests.

---
