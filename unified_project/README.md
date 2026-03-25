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
