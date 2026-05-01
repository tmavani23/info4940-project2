# p5 Emotion Chatbot

The app now uses a three-step flow:

1. `Emotion Discovery`
2. `Artistic Discovery`
3. `Coding`

Users go through the stages in order the first time, then can jump back to earlier stages to rethink the emotion or art direction before continuing.

## Run

Uses Gemini via API key from `.env` (`GOOGLE_API_KEY`).

```bash
cd unified_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open: `http://localhost:5001`

## Developer log dashboard

Open `http://localhost:5001/dev/logs` from the local machine to review fairness and potential harm signals across saved session logs. The dashboard has Accumulative harms views for survey/outcome fairness and Detectable harms views for per-turn safety guardrail events such as harmful input, extreme wording, and held generated outputs. Set `DEV_DASHBOARD_KEY` in `.env` to require `?key=...` or the `X-Dev-Dashboard-Key` header for dashboard access.

The dashboard can switch between real logs, synthetic logs, or both. Synthetic logs live in `synthetic_logs/` and can be regenerated with:

```bash
conda run -n info4940ai python scripts/generate_synthetic_logs.py
```
