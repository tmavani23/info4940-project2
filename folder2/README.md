# Emotion Sketch Studio

A rapid prototype for a two-phase p5.js co-creation chatbot (emotional discovery + technical implementation) with versioned state history for emotion, artistic direction, and code.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:
- User interface: `http://127.0.0.1:8000/`
- Debug interface: `http://127.0.0.1:8000/debug`

## Gemini / Vertex Configuration

The app will run with mock responses if no key is provided.

### Option A: Gemini API (AI Studio)

```bash
export GEMINI_API_KEY="your_key_here"
export GEMINI_MODEL="gemini-2.5-flash"
```

### Option B: Vertex AI (API key enabled project)

```bash
export GEMINI_API_KEY="your_key_here"
export GEMINI_API_KIND="vertex"
export VERTEX_PROJECT="your_project_id"
export VERTEX_LOCATION="us-central1"
export GEMINI_MODEL="gemini-2.5-flash"
```

If your Vertex setup requires OAuth instead of API keys, adapt `app/llm.py` to inject a bearer token.

## Data Persistence

- SQLite database: `data/state.db`
- Uploads: `data/uploads/`

## Notes

- Version history is a DAG. Revert always creates a new branch instead of deleting existing versions.
- The assistant is prompted separately per phase with structured sections to auto-capture emotion profiles, artistic profiles, and code.
