import ast
import json
import re


def summarize_state(node):
    if not node:
        return "(no state yet)"
    emotion = (node.get("emotion_profile") or "").strip()
    artistic = (node.get("artistic_profile") or "").strip()
    code = (node.get("code") or "").strip()

    def shorten(text, limit=240):
        text = re.sub(r"\s+", " ", text).strip()
        return text[:limit] + ("..." if len(text) > limit else "")

    return (
        f"Emotion profile: {shorten(emotion) if emotion else '(empty)'}\n"
        f"Artistic profile: {shorten(artistic) if artistic else '(empty)'}\n"
        f"Code summary: {shorten(code.splitlines()[0]) if code else '(empty)'}"
    )


def extract_section(text, header):
    pattern = re.compile(rf"===\s*{re.escape(header)}\s*===", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        alt = re.compile(
            rf"^\s*{re.escape(header)}\s*[:\-–—]?\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        match = alt.search(text)
        if not match:
            return ""
    start = match.end()
    next_match = re.search(r"===\s*[A-Z0-9 _-]+\s*===", text[start:])
    end = start + next_match.start() if next_match else len(text)
    return text[start:end].strip()


def extract_confidence(text):
    match = re.search(r"confidence level\s*:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def extract_code_block(text):
    # Match ```javascript ... ``` or ```js ... ``` or ``` ... ```
    match = re.search(r"```(?:javascript|js)?\s*([\s\S]+?)```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def extract_p5js_code(text):
    # First try fenced blocks
    code = extract_code_block(text)
    if code:
        return code
    # Fallback: opening fence without closing fence (capture to end)
    open_match = re.search(
        r"```(?:javascript|js)?\s*([\s\S]+)$", text, re.IGNORECASE
    )
    if open_match:
        return open_match.group(1).strip()
    # Fallback to section without fences
    section = extract_section(text, "P5JS CODE")
    if not section:
        return ""
    # If section still contains fenced code, strip it
    fenced = extract_code_block(section)
    if fenced:
        return fenced
    open_match = re.search(
        r"```(?:javascript|js)?\s*([\s\S]+)$", section, re.IGNORECASE
    )
    if open_match:
        return open_match.group(1).strip()
    # Clean possible leading language tag lines
    section = re.sub(r"^\s*(javascript|js)\s*\n", "", section, flags=re.IGNORECASE)
    return section.strip()


def is_likely_complete_code(code):
    if not code:
        return False
    if "function setup" not in code or "function draw" not in code:
        return False
    # Heuristic brace balance (ignores braces in strings/comments but good enough)
    open_braces = code.count("{")
    close_braces = code.count("}")
    if open_braces == 0 or close_braces == 0:
        return False
    return open_braces <= close_braces + 1 and close_braces <= open_braces + 1


def find_last_valid_code(nodes):
    for node in reversed(nodes or []):
        code = (node.get("code") or "").strip()
        if is_likely_complete_code(code):
            return code
    return ""


def detect_conflict(text):
    # Simple detection if LLM explicitly flags conflict
    if re.search(r"CONFLICT", text, re.IGNORECASE):
        return True
    if re.search(r"mismatch|doesn['’]?t match|not accurate", text, re.IGNORECASE):
        return True
    return False


def extract_intro_emotion(text):
    match = re.search(
        r"I might be wrong,\s*as I currently understand,\s*you are feeling\s+(.+?)(?:[\.!\n])",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return ""


def parse_llm_json(text):
    if not text:
        return None
    s = text.strip()
    # Strip markdown fences if present
    fence_match = re.search(r"```(?:json)?\\s*([\\s\\S]+?)```", s, re.IGNORECASE)
    if fence_match:
        s = fence_match.group(1).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = s[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        # Fallback: try Python literal parsing for single-quote JSON-ish output
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            return None
    return None
