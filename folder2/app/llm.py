import base64
import json
import os
import re
from typing import List, Dict

import requests

from .prompts import build_system_prompt, build_transition_prompt, build_commit_prompt
from .utils import summarize_state


DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _build_contents(messages: List[Dict], attachments: List[Dict] | None):
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        parts = [{"text": msg["content"]}]
        contents.append({"role": role, "parts": parts})

    if attachments:
        # Attach to the most recent user message if possible
        for i in range(len(contents) - 1, -1, -1):
            if contents[i]["role"] == "user":
                for att in attachments:
                    try:
                        with open(att["path"], "rb") as f:
                            data = base64.b64encode(f.read()).decode("utf-8")
                        contents[i]["parts"].append(
                            {
                                "inline_data": {
                                    "mime_type": att["content_type"],
                                    "data": data,
                                }
                            }
                        )
                    except Exception:
                        continue
                break

    return contents


def _call_gemini(payload, api_key, api_base, headers=None, params=None):
    headers = headers or {}
    params = params or {}
    response = requests.post(api_base, headers=headers, params=params, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def _extract_text(response_json):
    # Supports both AI Studio and Vertex-ish payloads
    try:
        candidates = response_json.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
                return "".join(texts).strip()
    except Exception:
        pass
    try:
        parts = response_json["candidates"][0]["content"]["parts"]
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        return "".join(texts).strip()
    except Exception:
        return ""


def generate_reply(phase, messages, state, learning_mode=False, attachments=None):
    system_prompt = build_system_prompt(phase, learning_mode, summarize_state(state))

    api_config = _get_api_config()
    if not api_config:
        return _error_response(
            "Gemini API key not configured. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or .env file."
        )
    api_key, api_base, headers, params = api_config

    contents = _build_contents(messages, attachments)

    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2400,
            "responseMimeType": "application/json",
            "response_mime_type": "application/json",
        },
    }

    try:
        response_json = _call_gemini(payload, api_key, api_base, headers=headers, params=params)
        text = _extract_text(response_json)
        if not text:
            return _error_response("Empty response from Gemini. Check model name and API configuration.")
        return text
    except Exception as exc:
        return _error_response(f"Gemini request failed: {exc}")


def generate_transition_message(action, node):
    prompt = build_transition_prompt()
    summary = summarize_state(node)
    label = (node.get("label") or "").strip()
    commit_summary = (node.get("summary") or label or "").strip()
    details = (
        f"Action: {action}\n"
        f"Commit summary: {commit_summary}\n"
        f"{summary}"
    )

    api_config = _get_api_config()
    if not api_config:
        return _fallback_transition_message(action, node)
    api_key, api_base, headers, params = api_config

    payload = {
        "contents": [{"role": "user", "parts": [{"text": details}]}],
        "systemInstruction": {"parts": [{"text": prompt}]},
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 220,
        },
    }

    try:
        response_json = _call_gemini(payload, api_key, api_base, headers=headers, params=params)
        text = (_extract_text(response_json) or "").strip()
        return text or _fallback_transition_message(action, node)
    except Exception:
        return _fallback_transition_message(action, node)


def _mock_response(phase, state, note=""):
    # Lightweight fallback so the app works without API access
    emotion = (state.get("emotion_profile") or "an unclear emotion").strip() or "an unclear emotion"
    if phase == "discovery":
        base = (
            f"I might be wrong, as I currently understand, you are feeling {emotion}. "
            "Could you share a concrete moment or setting that captures this feeling?"
        )
        if note:
            base += f"\n\n(Note: {note})"
        return base

    # Implementation fallback
    code = """
// === TWEAKABLE PARAMETERS ===
// Adjust these values to change the feel of the sketch

let particleCount = 60;      // Number of particles — higher = more crowded/chaotic (try: 20–200; e.g. 200 makes the scene feel overwhelming, 20 feels sparse and lonely)
let speed = 1.0;             // Movement speed — higher = more frantic/urgent (try: 0.5–5; e.g. 0.5 feels like a slow drift, 5 feels anxious and restless)
let colorIntensity = 160;    // Brightness of particles — higher = more vivid/energetic (try: 50–255; e.g. 255 is full brightness, 50 feels muted and subdued)
let fadeRate = 12;           // Trail fade speed — higher = trails disappear faster/feel more fleeting (try: 5–30; e.g. 30 makes trails fade almost instantly, 5 leaves long ghost-like traces)
let particleSize = 4;        // Size of each particle — higher = heavier/more dominant presence (try: 1–20; e.g. 1 feels delicate and subtle, 20 feels bold and heavy)
let opacity = 200;           // Overall transparency — lower = more ghostly/distant (try: 50–255; e.g. 50 feels like a faded memory, 255 is fully solid)

let particles = [];

function setup() {
  createCanvas(600, 400);
  for (let i = 0; i < particleCount; i++) {
    particles.push({ x: random(width), y: random(height), dx: random(-1, 1), dy: random(-1, 1) });
  }
  background(20);
}

function draw() {
  background(20, fadeRate);
  noStroke();
  fill(colorIntensity, colorIntensity, 255, opacity);
  for (let p of particles) {
    p.x += p.dx * speed;
    p.y += p.dy * speed;
    if (p.x < 0 || p.x > width) p.dx *= -1;
    if (p.y < 0 || p.y > height) p.dy *= -1;
    ellipse(p.x, p.y, particleSize, particleSize);
  }
}
""".strip()

    response = (
        f"I might be wrong, as I currently understand, you are feeling {emotion}. "
        "You can always refine or revert the emotion and adjust the artistic direction.\n\n"
        "=== ARTISTIC PROFILE ===\n"
        "Soft, drifting particles on a dark field, with gentle trails that suggest lingering memories.\n\n"
        "=== P5JS CODE ===\n"
        "```javascript\n" + code + "\n```\n\n"
        "Does the motion feel right? Does the color palette match the emotional tone? Is anything off or missing?"
    )
    if note:
        response += f"\n\n(Note: {note})"
    return response


def generate_commit_summary(action, before, after):
    prompt = build_commit_prompt()
    before_summary = summarize_state(before or {})
    after_summary = summarize_state(after or {})
    details = f"Action: {action}\nBEFORE:\n{before_summary}\nAFTER:\n{after_summary}"

    api_config = _get_api_config()
    if not api_config:
        return _fallback_commit_summary(action, before, after)
    api_key, api_base, headers, params = api_config

    payload = {
        "contents": [{"role": "user", "parts": [{"text": details}]}],
        "systemInstruction": {"parts": [{"text": prompt}]},
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 120,
        },
    }

    try:
        response_json = _call_gemini(payload, api_key, api_base, headers=headers, params=params)
        text = _extract_text(response_json) or ""
        cleaned = _clean_commit_summary(text)
        return cleaned or _fallback_commit_summary(action, before, after)
    except Exception:
        return _fallback_commit_summary(action, before, after)


def _fallback_transition_message(action, node):
    commit_summary = (node.get("summary") or node.get("label") or "").strip()
    commit_summary = commit_summary or "we were exploring earlier"
    return (
        f"I understand you want to change to the version which {commit_summary}. "
        "We can continue from here or adjust anything that doesn't feel quite right."
    )


def _clean_commit_summary(text):
    if not text:
        return ""
    line = re.split(r"[\r\n]+", text.strip())[0]
    line = line.strip().strip("\"'` ")
    line = re.sub(r"^[-•*\\d\\.\\)]+\\s*", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    if len(line) > 120:
        line = line[:117].rstrip() + "..."
    return line


def _fallback_commit_summary(action, before, after):
    if action in ("revert", "branch", "revert-branch"):
        return "Reverted to an earlier version of the sketch."

    changes = []
    before = before or {}
    after = after or {}

    if (before.get("emotion_profile") or "").strip() != (after.get("emotion_profile") or "").strip():
        changes.append("emotion")
    if (before.get("artistic_profile") or "").strip() != (after.get("artistic_profile") or "").strip():
        changes.append("artistic direction")
    if (before.get("code") or "").strip() != (after.get("code") or "").strip():
        changes.append("sketch code")

    if not changes:
        return "Saved the current state."
    if len(changes) == 1:
        return f"Updated the {changes[0]}."
    if len(changes) == 2:
        return f"Updated the {changes[0]} and {changes[1]}."
    return "Updated the emotion, artistic direction, and sketch code."


def _error_response(message):
    return (
        "System: I could not reach Gemini, so I did not generate a response.\n"
        f"Details: {message}"
    )


def _get_api_config():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    api_kind = os.getenv("GEMINI_API_KIND", "auto").lower()
    model = DEFAULT_MODEL
    if not api_key:
        return None
    headers = {"Content-Type": "application/json"}
    params = {}
    if api_kind == "vertex" or os.getenv("VERTEX_PROJECT"):
        project = os.getenv("VERTEX_PROJECT", "")
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        if not project:
            return None
        api_base = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent"
        )
        headers["X-Goog-Api-Key"] = api_key
    else:
        api_base = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        params["key"] = api_key
    return api_key, api_base, headers, params


def repair_json_response(raw_text, phase):
    api_config = _get_api_config()
    if not api_config:
        return ""
    api_key, api_base, headers, params = api_config

    if phase == "discovery":
        schema = (
            '{\"assistant_message\":\"...\",\"emotion_profile\":\"...\",'
            '\"artistic_profile\":\"\",\"p5js_code\":\"\",\"confidence\":\"...\",\"questions\":[\"...\"]}'
        )
    else:
        schema = (
            '{\"assistant_message\":\"...\",\"emotion_profile\":\"...\",\"artistic_profile\":\"...\",'
            '\"p5js_code\":\"...\",\"confidence\":\"...\",\"questions\":[\"...\"]}'
        )

    repair_prompt = (
        "You are a JSON repair tool. Convert the user's content into valid JSON only. "
        "Do not include markdown, comments, or extra text. "
        "Use double quotes for keys/strings and no trailing commas. "
        f"Schema: {schema}"
    )

    payload = {
        "contents": [{"role": "user", "parts": [{"text": raw_text}]}],
        "systemInstruction": {"parts": [{"text": repair_prompt}]},
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 1200,
            "responseMimeType": "application/json",
            "response_mime_type": "application/json",
        },
    }

    try:
        response_json = _call_gemini(payload, api_key, api_base, headers=headers, params=params)
        return _extract_text(response_json) or ""
    except Exception:
        return ""
