# AI-Powered Lecture Notes to Study Pack Generator

## What this project does
Uploads lecture notes (PDF, PPTX or TXT), extracts the text, chunks it, picks the key sections locally with TF-IDF (no API cost), and generates a grounded study pack:
- Summary
- Key concepts
- Definitions
- Practice questions
- Answer key
- AI-assisted prompt evaluation (accuracy, relevance, completeness, source alignment)

It also supports Baseline Prompt, Optimized Prompt and Compare Both modes.

## Run on any laptop

Needs **Python 3.10 or newer** ([python.org](https://www.python.org/downloads/); on Windows tick "Add python.exe to PATH") and internet for the first install.
Copy this whole folder to the laptop, open a terminal in it, then:

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope Process Bypass` first.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

**Shortcut:** double-click `run_windows.bat` (Windows) or run `bash run_mac_linux.sh` (macOS/Linux). They do all the steps above.

The app opens at http://localhost:8501. Next time, only activate the venv and run the last command.

### API key
Pick a provider in the sidebar and paste your key. The model list loads live from the provider, and each model is checked with a tiny call before use, so retired models never show. Or set the env var before starting:

| Provider | Env var |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |
| Groq | `GROQ_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Ollama (local) | none |
| Custom OpenAI-compatible | `CUSTOM_LLM_API_KEY` (optional) |

PowerShell example: `$env:GEMINI_API_KEY="your_api_key"`

## Important
- An API key (and credits) for your chosen provider is required, except local Ollama.
- The application sends selected note excerpts to the chosen provider's API.
- Do not upload confidential or sensitive documents.
- Scanned/image-only PDFs need OCR before upload; this starter version extracts selectable PDF text.
- Evaluation scores are AI-assisted estimates, not independent proof of correctness.

## API credit use
- Reading the file and choosing sections run locally. No API calls.
- Notes over ~10,000 characters are trimmed to the most central sections before sending.
- Calls per Generate: 1 (one prompt mode) or 2 (Compare both), +1 if AI self-evaluation is ticked (off by default).
- Output is capped (about 1,500 + 250 tokens per question; 800 for evaluation).
- Claude runs at low effort to keep thinking tokens small.
- Same file + same settings again = cached result, no new call.
- Model check: about 10 tokens, once per model per session.

## Prompt engineering demonstrated
- Baseline prompt: basic role and task instruction.
- Optimized prompt: role, task, constraints, learner settings, grounded-source rules and JSON output schema.
- The app can compare both outputs and evaluate the selected output against retrieved source excerpts.

## Project structure
- `app.py` — Streamlit application
- `run_windows.bat`, `run_mac_linux.sh` — one-click setup and run
- `providers.py` — provider layer (OpenAI, Claude, Gemini, Groq, OpenRouter, Ollama, custom)
- `requirements.txt` — Python dependencies
