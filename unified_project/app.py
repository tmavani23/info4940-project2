"""
Unified p5.js Emotional Chatbot Backend
=======================================
Merges:
- Folder 3 backend + LLM flow patterns
- Folder 2 restore/system-message behavior
- Folder 1 compatible chat/file/audio UX needs

Run:
  python app.py
Then open:
  http://localhost:5001
"""

import ast
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


DEFAULT_P5_CODE = ""

ART_DECISION_ACCEPT = "Art Direction Decision: Accept"
ART_DECISION_MODIFY = "Art Direction Decision: Modify"
ART_DECISION_MORE = "Art Direction Decision: More options"
ART_DECISION_OPTION_PREFIX = "Art Direction Decision: Option "


BASE_PROMPT = """
You are a collaborative creative coding assistant helping novice programmers create p5.js sketches that express lived experiences and emotions.

Your behavior rules:
- Be warm, concise, and clear for beginners.
- Briefly acknowledge emotions with empathy (for example: "That sounds really heavy" or "I can feel the anxiety in that moment").
- Explicitly remind users you can help them express those feelings through visuals and motion.
- Set expectations that understanding may be imperfect and can be corrected.
- Encourage iterative refinement of both feelings and visual expression.
- If uncertain, say what you are unsure about and ask one focused follow-up.
- Never shame the user for changing direction.
""".strip()


DISCOVERY_PROMPT = """
You are in emotional discovery mode. Do not generate new code unless the user explicitly asks to start coding.

Conversation goals:
- Summarize the current emotional understanding.
- Ask one useful follow-up that closes the biggest gap.
- Remind them they can move to coding when ready.
- Explicitly state that you may still misunderstand parts of their emotional experience and that they can correct you at any time.
- Keep the message short and supportive.

Return STRICT JSON only with this exact schema:
{
  "message": "<chat text shown to user>",
  "commit_message": "<short git-style summary of this turn>",
  "emotion_profile": "The emotion is ...",
  "emotion_confidence": "<low|medium|high>",
  "emotion_gaps": ["<gap 1>", "<gap 2>"],
  "artistic_profile": "",
  "code": ""
}

Constraints:
- emotion_profile must start with "The emotion is".
- emotion_confidence must be exactly low, medium, or high.
- emotion_gaps must be an array (possibly empty).
- commit_message should be one concise line describing discovery progress.
- In your `message`, always state that your emotional interpretation may be imperfect and the user can correct you at any time.
- No markdown fences. No extra keys. No commentary outside JSON.
""".strip()


IMPLEMENTATION_PROMPT = """
You are in implementation mode. Use the current emotional understanding to produce runnable p5.js.

Conversation goals:
- Briefly explain your visual metaphor and what changed this turn.
- Set expectation: this may need iteration and they can revert any time.
- Keep language novice-friendly.
- If you are proposing a changed artistic direction, ask for approval (accept/modify/more options) before finalizing.

Code requirements:
- Return complete runnable p5.js in the `code` field.
- Include setup() and draw().
- Keep incremental updates when possible.
- Put tweakable parameters near the top with comments.
- Add clear beginner-friendly comments for each major section (state setup, animation logic, color/motion choices, and any interaction).
- For non-obvious lines (timing math, mapping ranges, easing), include short explanatory comments.

Return STRICT JSON only with this exact schema:
{
  "message": "<chat text shown to user>",
  "commit_message": "<short git-style summary of this turn>",
  "emotion_profile": "The emotion is ...",
  "emotion_confidence": "<low|medium|high>",
  "emotion_gaps": ["<gap 1>", "<gap 2>"],
  "artistic_profile": "<1-3 sentences of visual direction>",
  "code": "<full runnable p5.js>"
}

Constraints:
- emotion_profile must start with "The emotion is".
- emotion_confidence must be exactly low, medium, or high.
- emotion_gaps must be an array (possibly empty).
- commit_message should be one concise line describing what changed in the sketch/profile.
- In your `message`, always set expectation that this implementation may be imperfect and invite corrections at any time.
- No markdown fences. No extra keys. No commentary outside JSON.
""".strip()


def build_system_prompt(phase: str, state_summary: str) -> str:
    phase_prompt = DISCOVERY_PROMPT if phase == "emotional_discovery" else IMPLEMENTATION_PROMPT
    return f"{BASE_PROMPT}\n\nCurrent state:\n{state_summary}\n\n{phase_prompt}"


def _short_id() -> str:
    return str(uuid.uuid4())[:8]


def _now() -> float:
    return time.time()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return ""
    return str(value).strip()


def _normalize_confidence(value: Any) -> str:
    c = _clean_text(value).lower()
    if c in {"low", "medium", "high"}:
        return c
    if c in {"0", "0.0"}:
        return "low"
    if c in {"1", "1.0"}:
        return "high"
    return "low"


def _normalize_gaps(value: Any) -> list[str]:
    if isinstance(value, list):
        gaps = [_clean_text(v) for v in value]
    elif isinstance(value, dict):
        gaps = [_clean_text(v) for v in value.values()]
    else:
        raw = _clean_text(value)
        if not raw:
            gaps = []
        else:
            parts = re.split(r"\n|;|•|- ", raw)
            gaps = [p.strip(" -\t") for p in parts if p.strip()]
    deduped = []
    seen = set()
    for gap in gaps:
        key = gap.lower()
        if gap and key not in seen:
            deduped.append(gap)
            seen.add(key)
    return deduped[:5]


def _ensure_emotion_prefix(profile: str) -> str:
    text = _clean_text(profile)
    if not text:
        return ""
    if text.lower().startswith("the emotion is"):
        return text
    return f"The emotion is {text.lstrip(':.- ')}"


def _normalize_commit_message(value: Any, fallback: str) -> str:
    text = re.sub(r"\s+", " ", _clean_text(value)).strip()
    if not text:
        text = re.sub(r"\s+", " ", _clean_text(fallback)).strip() or "Updated version"
    text = text.rstrip(".")
    if len(text) > 140:
        text = text[:139].rstrip() + "…"
    return text


def _extract_json_object(text: Any) -> Optional[dict]:
    if isinstance(text, dict):
        return text
    raw = _clean_text(text)
    if not raw:
        return None

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = raw[start : end + 1]
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        try:
            obj = ast.literal_eval(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def _is_likely_code_request(text: str) -> bool:
    if not text:
        return False
    return bool(
        re.search(
            r"\blet'?s code\b|\bstart coding\b|\bimplement\b|\bgenerate (the )?sketch\b|\bbuild (it|this)\b|\bturn this into (a )?sketch\b",
            text.lower(),
        )
    )


def _is_likely_emotion_refinement_request(text: str) -> bool:
    if not text:
        return False
    return bool(
        re.search(
            r"\bback to feelings\b|\bback to emotions\b|\brefine (my )?emotion\b|\brevisit (my )?emotion\b|\bmood is off\b|\bthis doesn'?t capture\b",
            text.lower(),
        )
    )


def _normalize_artistic_profile(text: str) -> str:
    return re.sub(r"\s+", " ", _clean_text(text)).strip().lower()


def _has_artistic_change(old: str, new: str) -> bool:
    return _normalize_artistic_profile(old) != _normalize_artistic_profile(new)


def _parse_art_direction_decision(text: str) -> Optional[str]:
    cleaned = re.sub(r"\s+", " ", _clean_text(text)).strip().lower()
    if cleaned == ART_DECISION_ACCEPT.lower():
        return "accept"
    if cleaned == ART_DECISION_MODIFY.lower():
        return "modify"
    if cleaned == ART_DECISION_MORE.lower():
        return "more_options"
    option_match = re.fullmatch(r"art direction decision:\s*option\s*(\d+)", cleaned)
    if option_match:
        return f"option_{int(option_match.group(1))}"
    return None


def _safe_summary_text(text: str, limit: int = 82) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _restore_transition_message(version_summary: str) -> str:
    detail = _safe_summary_text(version_summary or "", 96).strip()
    if detail:
        return (
            f"I restored this earlier version: \"{detail}\". "
            "I may still misread parts of your intent, so please correct me anytime and we can adjust together."
        )
    return (
        "I restored this earlier version. "
        "I may still misread parts of your intent, so please correct me anytime and we can adjust together."
    )


def _infer_phase_from_version(version: "VersionNode") -> str:
    return "code_generation" if _clean_text(version.code) else "emotional_discovery"


def _summarize_state_for_prompt(version: "VersionNode") -> str:
    code_first = ""
    if version.code:
        first_lines = [ln.strip() for ln in version.code.splitlines() if ln.strip()]
        code_first = first_lines[0] if first_lines else ""
    return (
        f"Emotion profile: {version.emotion_profile or '(empty)'}\n"
        f"Emotion confidence: {version.emotion_confidence or '(empty)'}\n"
        f"Emotion gaps: {', '.join(version.emotion_gaps) if version.emotion_gaps else '(none)'}\n"
        f"Artistic profile: {version.artistic_profile or '(empty)'}\n"
        f"Code first line: {code_first or '(empty)'}"
    )


def _default_mock_reply(phase: str, version: "VersionNode") -> dict:
    emotion = version.emotion_profile or "The emotion is still unfolding and may need more context."
    if phase == "emotional_discovery":
        return {
            "message": (
                "Thank you for sharing this. I can hear there is real emotional weight here, and I can help you express it visually. "
                "I may still misunderstand parts of your experience, so please correct me anytime. "
                "Share one concrete moment (place, time, and what happened) so I can tighten the emotional profile."
            ),
            "commit_message": "Refined emotional understanding and identified open gaps",
            "emotion_profile": _ensure_emotion_prefix(emotion),
            "emotion_confidence": "low",
            "emotion_gaps": [
                "The exact moment or scene is still unclear.",
                "The desired motion or pacing is not fully defined.",
            ],
            "artistic_profile": version.artistic_profile or "",
            "code": "",
        }

    return {
        "message": (
            "I can help you turn this feeling into visuals. This implementation pass may still miss details, "
            "so please correct me at any time and we can revise together."
        ),
        "commit_message": "Updated first sketch pass toward the current emotional direction",
        "emotion_profile": _ensure_emotion_prefix(emotion),
        "emotion_confidence": version.emotion_confidence or "medium",
        "emotion_gaps": version.emotion_gaps or ["Color and pacing preference may need refinement."],
        "artistic_profile": version.artistic_profile
        or "Soft particle motion with a restrained palette to keep emotional focus.",
        "code": version.code or "",
    }


def build_multimodal_message(
    text: str,
    image_b64: Optional[str] = None,
    image_mime: Optional[str] = None,
    audio_b64: Optional[str] = None,
    audio_mime: Optional[str] = None,
) -> HumanMessage:
    parts = []
    if image_b64:
        image_url = image_b64 if image_b64.startswith("data:") else f"data:{image_mime or 'image/jpeg'};base64,{image_b64}"
        parts.append({"type": "image_url", "image_url": {"url": image_url}})
    if audio_b64:
        audio_url = audio_b64 if audio_b64.startswith("data:") else f"data:{audio_mime or 'audio/webm'};base64,{audio_b64}"
        # Gemini via LangChain accepts this format for inline media.
        parts.append({"type": "image_url", "image_url": {"url": audio_url}})
    parts.append({"type": "text", "text": text})
    return HumanMessage(content=parts)


@dataclass
class ChatEntry:
    id: str
    role: str
    type: str
    content: str
    created_at: float = field(default_factory=_now)


@dataclass
class VersionNode:
    id: str
    parent_id: Optional[str]
    branch: str
    source: str
    summary: str
    emotion_profile: str
    artistic_profile: str
    code: str
    emotion_confidence: str
    emotion_gaps: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=_now)


@dataclass
class PendingArtisticDecision:
    proposed_artistic_profile: str
    proposed_code: str
    proposed_emotion_profile: str
    proposed_emotion_confidence: str
    proposed_emotion_gaps: list[str] = field(default_factory=list)
    proposed_commit_message: str = ""
    alternatives: list[str] = field(default_factory=list)
    awaiting_modify_details: bool = False
    created_at: float = field(default_factory=_now)


@dataclass
class SessionState:
    session_id: str
    phase: str = "emotional_discovery"
    messages: list[ChatEntry] = field(default_factory=list)
    llm_events: list[dict[str, Any]] = field(default_factory=list)
    versions: dict[str, VersionNode] = field(default_factory=dict)
    version_order: list[str] = field(default_factory=list)
    branch_heads: dict[str, str] = field(default_factory=dict)
    active_branch: str = "main"
    current_version_id: Optional[str] = None
    branch_counter: int = 1
    pending_artistic_decision: Optional[PendingArtisticDecision] = None

    def __post_init__(self):
        if self.current_version_id:
            return
        root = VersionNode(
            id=_short_id(),
            parent_id=None,
            branch="main",
            source="system",
            summary="Initial version",
            emotion_profile="",
            artistic_profile="",
            code="",
            emotion_confidence="",
            emotion_gaps=["Share a personal moment so we can ground the emotion."],
        )
        self.versions[root.id] = root
        self.version_order.append(root.id)
        self.branch_heads["main"] = root.id
        self.current_version_id = root.id
        self.active_branch = "main"

    @property
    def current_version(self) -> VersionNode:
        return self.versions[self.current_version_id]


sessions: dict[str, SessionState] = {}
latest_session_id: Optional[str] = None


def get_or_create_session(session_id: Optional[str] = None) -> SessionState:
    global latest_session_id
    if session_id and session_id in sessions:
        latest_session_id = session_id
        return sessions[session_id]
    if session_id and session_id not in sessions:
        session = SessionState(session_id=session_id)
        sessions[session_id] = session
        latest_session_id = session_id
        return session
    if latest_session_id and latest_session_id in sessions:
        return sessions[latest_session_id]
    sid = str(uuid.uuid4())
    session = SessionState(session_id=sid)
    sessions[sid] = session
    latest_session_id = sid
    return session


def _append_chat(session: SessionState, role: str, msg_type: str, content: str) -> ChatEntry:
    entry = ChatEntry(id=_short_id(), role=role, type=msg_type, content=content or "")
    session.messages.append(entry)
    return entry


def _append_llm_event(session: SessionState, message: Any, anchor_version_id: Optional[str]):
    session.llm_events.append(
        {
            "id": _short_id(),
            "created_at": _now(),
            "anchor_version_id": anchor_version_id,
            "message": message,
        }
    )


def _collect_lineage_ids(session: SessionState, version_id: Optional[str]) -> set[str]:
    ids = set()
    current_id = version_id
    while current_id and current_id in session.versions:
        ids.add(current_id)
        current_id = session.versions[current_id].parent_id
    return ids


def _build_llm_context(session: SessionState, limit: int = 18) -> list[Any]:
    lineage_ids = _collect_lineage_ids(session, session.current_version_id)
    visible = []
    for event in session.llm_events:
        anchor = event.get("anchor_version_id")
        if anchor is None or anchor in lineage_ids:
            visible.append(event.get("message"))
    return visible[-limit:]


def _build_commit_summary(
    old: VersionNode,
    new_emotion: str,
    new_artistic: str,
    new_code: str,
    source: str,
) -> str:
    changes = []
    if old.emotion_profile != new_emotion:
        changes.append("emotion profile")
    if old.artistic_profile != new_artistic:
        changes.append("artistic profile")
    if old.code != new_code:
        changes.append("p5.js code")

    if not changes:
        return "Saved current version"
    if len(changes) == 1:
        action = f"Updated {changes[0]}"
    elif len(changes) == 2:
        action = f"Updated {changes[0]} and {changes[1]}"
    else:
        action = "Updated emotion, artistic profile, and p5.js code"
    if source == "assistant":
        return f"Assistant: {action}"
    return action


def _default_artistic_alternatives(base_profile: str) -> list[str]:
    seed = _clean_text(base_profile) or "A restrained poetic visual metaphor with slow pacing."
    return [
        f"{seed} Emphasis: watercolor softness and gentle drift.",
        f"{seed} Emphasis: stark contrast with sparse geometry and pauses.",
        f"{seed} Emphasis: dreamlike motion trails with breathing rhythm.",
    ]


def _format_pending_artistic_message(
    message: str,
    proposed_artistic_profile: str,
    alternatives: list[str],
    awaiting_modify_details: bool,
) -> str:
    lines = []
    intro = _clean_text(message)
    if intro:
        lines.append(intro)
    lines.append("Proposed artistic direction (not applied yet):")
    lines.append(proposed_artistic_profile or "(not specified yet)")
    if alternatives:
        lines.append("")
        lines.append("Alternative directions:")
        for index, option in enumerate(alternatives[:3], start=1):
            lines.append(f"{index}. {option}")
    lines.append("")
    if awaiting_modify_details:
        lines.append("Tell me what you want to modify, and I will revise this proposal before any version is created.")
    else:
        lines.append("Choose one option to proceed:")
        if alternatives:
            for index, _ in enumerate(alternatives[:3], start=1):
                lines.append(f"- {ART_DECISION_OPTION_PREFIX}{index}")
            lines.append(f"- {ART_DECISION_MORE}")
            lines.append(f"- {ART_DECISION_MODIFY}")
        else:
            lines.append(f"- {ART_DECISION_ACCEPT}")
            lines.append(f"- {ART_DECISION_MODIFY}")
            lines.append(f"- {ART_DECISION_MORE}")
    return "\n".join(lines)


def _sanitize_artistic_options_payload(
    raw: dict,
    current: VersionNode,
    fallback_profile: str,
) -> dict:
    message = _clean_text(raw.get("message")) or (
        "I drafted a few artistic direction options. I might still miss your intent, so feel free to steer me."
    )
    profile = _clean_text(raw.get("recommended_artistic_profile")) or fallback_profile or current.artistic_profile
    options_raw = raw.get("artistic_options")
    options: list[str] = []
    if isinstance(options_raw, list):
        options = [_clean_text(v) for v in options_raw]
    elif isinstance(options_raw, str):
        options = [_clean_text(v) for v in re.split(r"\n|;|•", options_raw)]
    options = [v for v in options if v]
    if not options:
        options = _default_artistic_alternatives(profile)
    deduped = []
    seen = set()
    for opt in options:
        key = opt.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(opt)
    emotion_profile = _ensure_emotion_prefix(_clean_text(raw.get("emotion_profile"))) or current.emotion_profile
    emotion_confidence = _normalize_confidence(raw.get("emotion_confidence") or current.emotion_confidence)
    emotion_gaps = _normalize_gaps(raw.get("emotion_gaps"))
    if not emotion_gaps:
        emotion_gaps = current.emotion_gaps
    commit_message = _normalize_commit_message(
        raw.get("commit_message"),
        fallback="Refined artistic direction options before implementation",
    )
    return {
        "message": message,
        "recommended_artistic_profile": profile,
        "artistic_options": deduped[:3],
        "emotion_profile": emotion_profile,
        "emotion_confidence": emotion_confidence,
        "emotion_gaps": emotion_gaps,
        "commit_message": commit_message,
    }


def create_version(
    session: SessionState,
    *,
    emotion_profile: str,
    artistic_profile: str,
    code: str,
    emotion_confidence: str,
    emotion_gaps: list[str],
    summary: str,
    source: str,
) -> VersionNode:
    parent_id = session.current_version_id
    branch = session.active_branch or "main"
    current_branch_head = session.branch_heads.get(branch)

    # Git-like branching: if user/assistant continues from a non-head commit,
    # automatically create a new branch.
    if parent_id and current_branch_head and parent_id != current_branch_head:
        branch = f"{branch}-b{session.branch_counter}"
        session.branch_counter += 1

    version = VersionNode(
        id=_short_id(),
        parent_id=parent_id,
        branch=branch,
        source=source,
        summary=summary,
        emotion_profile=emotion_profile,
        artistic_profile=artistic_profile,
        code=code,
        emotion_confidence=emotion_confidence,
        emotion_gaps=emotion_gaps,
    )
    session.versions[version.id] = version
    session.version_order.append(version.id)
    session.current_version_id = version.id
    session.active_branch = branch
    session.branch_heads[branch] = version.id
    return version


LLM_ENABLED = bool(GOOGLE_API_KEY)
phase_1_llm: Optional[ChatGoogleGenerativeAI] = None
phase_2_llm: Optional[ChatGoogleGenerativeAI] = None

if LLM_ENABLED:
    phase_1_llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.8)
    phase_2_llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.45)


def _invoke_llm(
    session: SessionState,
    user_text: str,
    image: Optional[str],
    image_mime: Optional[str],
    audio: Optional[str],
    audio_mime: Optional[str],
) -> tuple[dict, str, Any, Any]:
    human_msg = build_multimodal_message(
        text=user_text,
        image_b64=image,
        image_mime=image_mime,
        audio_b64=audio,
        audio_mime=audio_mime,
    )

    if not LLM_ENABLED or phase_1_llm is None or phase_2_llm is None:
        parsed = _default_mock_reply(session.phase, session.current_version)
        fallback_json = json.dumps(parsed)
        return parsed, fallback_json, human_msg, AIMessage(content=fallback_json)

    llm = phase_1_llm if session.phase == "emotional_discovery" else phase_2_llm
    state_summary = _summarize_state_for_prompt(session.current_version)
    system_msg = SystemMessage(content=build_system_prompt(session.phase, state_summary))

    context = _build_llm_context(session, limit=18)

    try:
        response = llm.invoke([system_msg] + context + [human_msg])
        response_text = response.content if isinstance(response.content, str) else json.dumps(response.content)
        parsed = _extract_json_object(response_text)
        if not parsed:
            parsed = _default_mock_reply(session.phase, session.current_version)
            response_text = json.dumps(parsed)
            assistant_msg = AIMessage(content=response_text)
        else:
            assistant_msg = response
        return parsed, response_text, human_msg, assistant_msg
    except Exception as exc:
        parsed = _default_mock_reply(session.phase, session.current_version)
        parsed["message"] = (
            "I couldn’t reach the model just now, so I used a local fallback response. "
            "You can keep working and we can still iterate."
        )
        parsed["emotion_gaps"] = [f"LLM call failed: {exc}"]
        response_text = json.dumps(parsed)
        return parsed, response_text, human_msg, AIMessage(content=response_text)


def _invoke_artistic_options_llm(
    session: SessionState,
    *,
    user_feedback: str,
    proposed_artistic_profile: str,
) -> tuple[dict, str, Any, Any]:
    prompt_text = f"""
You are in an artistic-direction proposal step. Do not finalize a version yet.
The user feedback is: "{user_feedback or 'No extra feedback provided.'}"
Current proposed artistic profile: "{proposed_artistic_profile or '(empty)'}"

Return STRICT JSON only with this schema:
{{
  "message": "<empathetic short reply + brief interpretation + ask user to choose>",
  "recommended_artistic_profile": "<updated artistic profile proposal>",
  "artistic_options": ["<option 1>", "<option 2>", "<option 3>"],
  "emotion_profile": "The emotion is ...",
  "emotion_confidence": "<low|medium|high>",
  "emotion_gaps": ["<gap 1>", "<gap 2>"],
  "commit_message": "<one-line summary of this proposed direction>"
}}

Rules:
- Be empathetic and supportive.
- Set expectation that you may be wrong and invite correction.
- Ask the user to choose Option 1, Option 2, Option 3, modify, or more options.
- Do not include markdown or any keys outside this schema.
""".strip()
    human_msg = HumanMessage(content=prompt_text)

    if not LLM_ENABLED or phase_2_llm is None:
        fallback = _sanitize_artistic_options_payload(
            {},
            session.current_version,
            fallback_profile=proposed_artistic_profile,
        )
        fallback["message"] = (
            "I hear you. I sketched a few artistic directions we can try, and I might still be off, "
            "so please correct me anytime."
        )
        fallback_json = json.dumps(fallback)
        return fallback, fallback_json, human_msg, AIMessage(content=fallback_json)

    context = _build_llm_context(session, limit=14)
    system_msg = SystemMessage(
        content=(
            f"{BASE_PROMPT}\n\n"
            "You are helping refine visual direction options before committing a new version."
        )
    )

    try:
        response = phase_2_llm.invoke([system_msg] + context + [human_msg])
        response_text = response.content if isinstance(response.content, str) else json.dumps(response.content)
        parsed = _extract_json_object(response_text) or {}
        sanitized = _sanitize_artistic_options_payload(
            parsed,
            session.current_version,
            fallback_profile=proposed_artistic_profile,
        )
        if not _extract_json_object(response_text):
            response_text = json.dumps(sanitized)
            assistant_msg = AIMessage(content=response_text)
        else:
            assistant_msg = response
        return sanitized, response_text, human_msg, assistant_msg
    except Exception as exc:
        fallback = _sanitize_artistic_options_payload(
            {},
            session.current_version,
            fallback_profile=proposed_artistic_profile,
        )
        fallback["message"] = (
            "I hit a model issue while generating alternatives, but we can still continue. "
            "I might be off, so please keep correcting me."
        )
        fallback["emotion_gaps"] = [f"LLM call failed while generating artistic options: {exc}"]
        fallback_json = json.dumps(fallback)
        return fallback, fallback_json, human_msg, AIMessage(content=fallback_json)


def _sanitize_llm_payload(raw: dict, current: VersionNode) -> dict:
    message = _clean_text(raw.get("message")) or "I updated the current sketch state."
    commit_message = _clean_text(raw.get("commit_message"))
    emotion_profile = _ensure_emotion_prefix(_clean_text(raw.get("emotion_profile"))) or current.emotion_profile
    artistic_profile = _clean_text(raw.get("artistic_profile")) or current.artistic_profile
    code = _clean_text(raw.get("code")) or current.code
    emotion_confidence = _normalize_confidence(raw.get("emotion_confidence") or current.emotion_confidence)
    emotion_gaps = _normalize_gaps(raw.get("emotion_gaps"))
    if not emotion_gaps:
        emotion_gaps = current.emotion_gaps
    return {
        "message": message,
        "commit_message": commit_message,
        "emotion_profile": emotion_profile,
        "emotion_confidence": emotion_confidence,
        "emotion_gaps": emotion_gaps,
        "artistic_profile": artistic_profile,
        "code": code,
    }


def serialize_message(msg: ChatEntry) -> dict:
    return {
        "id": msg.id,
        "role": msg.role,
        "type": msg.type,
        "content": msg.content,
        "created_at": msg.created_at,
    }


def serialize_version(v: VersionNode) -> dict:
    return {
        "id": v.id,
        "parent_id": v.parent_id,
        "branch": v.branch,
        "source": v.source,
        "summary": v.summary,
        "emotion_profile": v.emotion_profile,
        "artistic_profile": v.artistic_profile,
        "code": v.code,
        "emotion_confidence": v.emotion_confidence,
        "emotion_gaps": v.emotion_gaps,
        "created_at": v.created_at,
    }


def serialize_pending_artistic_decision(pending: Optional[PendingArtisticDecision]) -> Optional[dict]:
    if not pending:
        return None
    return {
        "proposed_artistic_profile": pending.proposed_artistic_profile,
        "proposed_code": pending.proposed_code,
        "proposed_emotion_profile": pending.proposed_emotion_profile,
        "proposed_emotion_confidence": pending.proposed_emotion_confidence,
        "proposed_emotion_gaps": pending.proposed_emotion_gaps,
        "proposed_commit_message": pending.proposed_commit_message,
        "alternatives": list(pending.alternatives),
        "awaiting_modify_details": pending.awaiting_modify_details,
        "created_at": pending.created_at,
    }


def serialize_state(session: SessionState) -> dict:
    versions = [serialize_version(session.versions[vid]) for vid in session.version_order]
    current = serialize_version(session.current_version)
    return {
        "session_id": session.session_id,
        "phase": session.phase,
        "current_version": current,
        "versions": versions,
        "messages": [serialize_message(m) for m in session.messages],
        "branch_heads": dict(session.branch_heads),
        "active_branch": session.active_branch,
        "pending_artistic_decision": serialize_pending_artistic_decision(session.pending_artistic_decision),
    }


app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(BASE_DIR, "frontend.html")


@app.route("/debug", methods=["GET"])
def debug_page():
    return send_from_directory(BASE_DIR, "debug.html")


@app.route("/api/state", methods=["GET"])
def api_state():
    sid = request.args.get("session_id")
    session = get_or_create_session(sid)
    return jsonify(serialize_state(session))


@app.route("/api/history", methods=["GET"])
def api_history():
    sid = request.args.get("session_id")
    if not sid:
        return jsonify({"error": "Missing session_id"}), 400
    session = get_or_create_session(sid)
    nodes = [serialize_version(session.versions[vid]) for vid in session.version_order]
    edges = [{"from": n["parent_id"], "to": n["id"]} for n in nodes if n["parent_id"]]
    return jsonify(
        {
            "session_id": sid,
            "nodes": nodes,
            "edges": edges,
            "branch_heads": dict(session.branch_heads),
            "active_branch": session.active_branch,
        }
    )


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    session = get_or_create_session(data.get("session_id"))

    message = _clean_text(data.get("message"))
    image = _clean_text(data.get("image"))
    audio = _clean_text(data.get("audio"))
    image_mime = _clean_text(data.get("image_mime")) or "image/jpeg"
    audio_mime = _clean_text(data.get("audio_mime")) or "audio/webm"

    if not message and not image and not audio:
        return jsonify({"error": "Message, image, or audio is required."}), 400

    if message:
        _append_chat(session, "user", "text", message)
    if image:
        _append_chat(session, "user", "image", image)
    if audio:
        _append_chat(session, "user", "audio", audio)

    def _chat_response(reply_payload: dict, raw_response_text: str, created_version: Optional[VersionNode]):
        return jsonify(
            {
                "session_id": session.session_id,
                "phase": session.phase,
                "reply": reply_payload,
                "version_created": bool(created_version),
                "created_version_id": created_version.id if created_version else None,
                "state": serialize_state(session),
                "raw_response": raw_response_text,
            }
        )

    decision = _parse_art_direction_decision(message) if message else None
    pending = session.pending_artistic_decision

    # Handle explicit artistic-decision actions first when a proposal is pending.
    if pending and decision:
        current = session.current_version
        if decision == "modify":
            pending.awaiting_modify_details = True
            assistant_text = _format_pending_artistic_message(
                "Thanks for steering this. Tell me what to modify, and I will revise the artistic proposal before creating any version.",
                pending.proposed_artistic_profile,
                pending.alternatives,
                awaiting_modify_details=True,
            )
            _append_chat(session, "assistant", "text", assistant_text)
            return _chat_response(
                {
                    "message": assistant_text,
                    "commit_message": pending.proposed_commit_message,
                    "emotion_profile": pending.proposed_emotion_profile or current.emotion_profile,
                    "emotion_confidence": pending.proposed_emotion_confidence or current.emotion_confidence,
                    "emotion_gaps": pending.proposed_emotion_gaps or current.emotion_gaps,
                    "artistic_profile": pending.proposed_artistic_profile or current.artistic_profile,
                    "code": current.code,
                },
                raw_response_text="",
                created_version=None,
            )

        if decision == "more_options":
            options_payload, raw_options_text, human_msg, assistant_msg = _invoke_artistic_options_llm(
                session,
                user_feedback="Please provide more artistic direction alternatives before implementation.",
                proposed_artistic_profile=pending.proposed_artistic_profile,
            )
            pending.proposed_artistic_profile = options_payload["recommended_artistic_profile"]
            pending.proposed_emotion_profile = options_payload["emotion_profile"]
            pending.proposed_emotion_confidence = options_payload["emotion_confidence"]
            pending.proposed_emotion_gaps = options_payload["emotion_gaps"]
            pending.proposed_commit_message = options_payload["commit_message"]
            pending.alternatives = options_payload["artistic_options"]
            pending.awaiting_modify_details = False
            assistant_text = _format_pending_artistic_message(
                options_payload["message"],
                pending.proposed_artistic_profile,
                pending.alternatives,
                awaiting_modify_details=False,
            )
            _append_chat(session, "assistant", "text", assistant_text)
            _append_llm_event(session, human_msg, session.current_version_id)
            _append_llm_event(session, assistant_msg, session.current_version_id)
            return _chat_response(
                {
                    "message": assistant_text,
                    "commit_message": pending.proposed_commit_message,
                    "emotion_profile": pending.proposed_emotion_profile,
                    "emotion_confidence": pending.proposed_emotion_confidence,
                    "emotion_gaps": pending.proposed_emotion_gaps,
                    "artistic_profile": pending.proposed_artistic_profile,
                    "code": current.code,
                },
                raw_response_text=raw_options_text,
                created_version=None,
            )

        selected_artistic_profile = None
        selected_label = "accepted-proposal"
        if decision == "accept":
            selected_artistic_profile = pending.proposed_artistic_profile
        elif decision.startswith("option_"):
            try:
                option_index = int(decision.split("_", 1)[1])
            except Exception:
                option_index = 0
            if not pending.alternatives:
                assistant_text = (
                    "I do not have numbered alternatives yet. Use More options first, or choose Accept/Modify."
                )
                _append_chat(session, "assistant", "text", assistant_text)
                return _chat_response(
                    {
                        "message": assistant_text,
                        "commit_message": pending.proposed_commit_message,
                        "emotion_profile": pending.proposed_emotion_profile or current.emotion_profile,
                        "emotion_confidence": pending.proposed_emotion_confidence or current.emotion_confidence,
                        "emotion_gaps": pending.proposed_emotion_gaps or current.emotion_gaps,
                        "artistic_profile": pending.proposed_artistic_profile or current.artistic_profile,
                        "code": current.code,
                    },
                    raw_response_text="",
                    created_version=None,
                )
            if option_index < 1 or option_index > len(pending.alternatives):
                assistant_text = (
                    f"I could not find Option {option_index}. Please choose one of the listed options, Modify, or More options."
                )
                _append_chat(session, "assistant", "text", assistant_text)
                return _chat_response(
                    {
                        "message": assistant_text,
                        "commit_message": pending.proposed_commit_message,
                        "emotion_profile": pending.proposed_emotion_profile or current.emotion_profile,
                        "emotion_confidence": pending.proposed_emotion_confidence or current.emotion_confidence,
                        "emotion_gaps": pending.proposed_emotion_gaps or current.emotion_gaps,
                        "artistic_profile": pending.proposed_artistic_profile or current.artistic_profile,
                        "code": current.code,
                    },
                    raw_response_text="",
                    created_version=None,
                )
            selected_artistic_profile = pending.alternatives[option_index - 1]
            selected_label = f"option-{option_index}"
            pending.proposed_artistic_profile = selected_artistic_profile

        if not selected_artistic_profile:
            assistant_text = "I could not parse that artistic-direction action. Please choose one of the available options."
            _append_chat(session, "assistant", "text", assistant_text)
            return _chat_response(
                {
                    "message": assistant_text,
                    "commit_message": pending.proposed_commit_message,
                    "emotion_profile": pending.proposed_emotion_profile or current.emotion_profile,
                    "emotion_confidence": pending.proposed_emotion_confidence or current.emotion_confidence,
                    "emotion_gaps": pending.proposed_emotion_gaps or current.emotion_gaps,
                    "artistic_profile": pending.proposed_artistic_profile or current.artistic_profile,
                    "code": current.code,
                },
                raw_response_text="",
                created_version=None,
            )

        # Accept path: now implement and create a version.
        accept_prompt = (
            "The user accepted this artistic direction. Implement it now and provide runnable p5.js.\n"
            f"Accepted artistic profile ({selected_label}): {selected_artistic_profile}\n"
            "Be explicit that your implementation may still need corrections."
        )
        parsed_raw, raw_response_text, human_msg, assistant_llm_msg = _invoke_llm(
            session=session,
            user_text=accept_prompt,
            image=None,
            image_mime=None,
            audio=None,
            audio_mime=None,
        )
        sanitized = _sanitize_llm_payload(parsed_raw, current)
        new_emotion = sanitized["emotion_profile"] or pending.proposed_emotion_profile or current.emotion_profile
        new_artistic = sanitized["artistic_profile"] or selected_artistic_profile or current.artistic_profile
        new_code = sanitized["code"] or pending.proposed_code or current.code
        new_conf = sanitized["emotion_confidence"] or pending.proposed_emotion_confidence or current.emotion_confidence
        new_gaps = sanitized["emotion_gaps"] or pending.proposed_emotion_gaps or current.emotion_gaps
        changed = (
            current.emotion_profile != new_emotion
            or current.artistic_profile != new_artistic
            or current.code != new_code
        )
        created_version = None
        if changed:
            fallback_summary = _build_commit_summary(current, new_emotion, new_artistic, new_code, source="assistant")
            summary = _normalize_commit_message(
                sanitized.get("commit_message") or pending.proposed_commit_message,
                fallback=fallback_summary,
            )
            created_version = create_version(
                session,
                emotion_profile=new_emotion,
                artistic_profile=new_artistic,
                code=new_code,
                emotion_confidence=new_conf,
                emotion_gaps=new_gaps,
                summary=summary,
                source="assistant",
            )
        else:
            current.emotion_confidence = new_conf
            current.emotion_gaps = new_gaps
        turn_anchor_id = created_version.id if created_version else current.id
        _append_llm_event(session, human_msg, turn_anchor_id)
        _append_llm_event(session, assistant_llm_msg, turn_anchor_id)
        _append_chat(session, "assistant", "text", sanitized["message"])
        session.pending_artistic_decision = None
        session.phase = "code_generation"
        return _chat_response(sanitized, raw_response_text, created_version)

    # If a proposal is pending and the user sends feedback, treat it as modify input.
    if pending and not decision:
        current = session.current_version
        feedback = message or "Please revise this artistic direction based on my latest feedback."
        options_payload, raw_options_text, human_msg, assistant_msg = _invoke_artistic_options_llm(
            session,
            user_feedback=feedback,
            proposed_artistic_profile=pending.proposed_artistic_profile,
        )
        pending.proposed_artistic_profile = options_payload["recommended_artistic_profile"]
        pending.proposed_emotion_profile = options_payload["emotion_profile"]
        pending.proposed_emotion_confidence = options_payload["emotion_confidence"]
        pending.proposed_emotion_gaps = options_payload["emotion_gaps"]
        pending.proposed_commit_message = options_payload["commit_message"]
        pending.alternatives = options_payload["artistic_options"]
        pending.awaiting_modify_details = False
        assistant_text = _format_pending_artistic_message(
            options_payload["message"],
            pending.proposed_artistic_profile,
            pending.alternatives,
            awaiting_modify_details=False,
        )
        _append_chat(session, "assistant", "text", assistant_text)
        _append_llm_event(session, human_msg, session.current_version_id)
        _append_llm_event(session, assistant_msg, session.current_version_id)
        return _chat_response(
            {
                "message": assistant_text,
                "commit_message": pending.proposed_commit_message,
                "emotion_profile": pending.proposed_emotion_profile or current.emotion_profile,
                "emotion_confidence": pending.proposed_emotion_confidence or current.emotion_confidence,
                "emotion_gaps": pending.proposed_emotion_gaps or current.emotion_gaps,
                "artistic_profile": pending.proposed_artistic_profile or current.artistic_profile,
                "code": current.code,
            },
            raw_response_text=raw_options_text,
            created_version=None,
        )

    if session.phase == "code_generation" and _is_likely_emotion_refinement_request(message):
        session.phase = "emotional_discovery"
    elif session.phase == "emotional_discovery" and _is_likely_code_request(message) and session.current_version.emotion_profile:
        session.phase = "code_generation"

    user_text_for_model = message
    if not user_text_for_model:
        if image and audio:
            user_text_for_model = "I shared an image and audio clip. Please infer emotional cues and update the profiles."
        elif image:
            user_text_for_model = "I shared an image. Please infer emotional cues and update the profiles."
        else:
            user_text_for_model = "I shared an audio clip. Please infer emotional cues and update the profiles."

    parsed_raw, raw_response_text, human_msg, assistant_llm_msg = _invoke_llm(
        session=session,
        user_text=user_text_for_model,
        image=image,
        image_mime=image_mime,
        audio=audio,
        audio_mime=audio_mime,
    )
    sanitized = _sanitize_llm_payload(parsed_raw, session.current_version)

    old = session.current_version
    new_emotion = sanitized["emotion_profile"]
    new_artistic = sanitized["artistic_profile"]
    new_code = sanitized["code"]
    new_conf = sanitized["emotion_confidence"]
    new_gaps = sanitized["emotion_gaps"]

    should_stage_artistic_decision = (
        session.phase == "code_generation"
        and _clean_text(new_artistic)
        and _has_artistic_change(old.artistic_profile, new_artistic)
    )

    if should_stage_artistic_decision:
        old.emotion_confidence = new_conf
        old.emotion_gaps = new_gaps
        session.pending_artistic_decision = PendingArtisticDecision(
            proposed_artistic_profile=new_artistic,
            proposed_code=new_code,
            proposed_emotion_profile=new_emotion,
            proposed_emotion_confidence=new_conf,
            proposed_emotion_gaps=new_gaps,
            proposed_commit_message=_normalize_commit_message(
                sanitized.get("commit_message"),
                fallback="Proposed a new artistic direction before implementation",
            ),
            alternatives=[],
            awaiting_modify_details=False,
        )
        staged_message = _format_pending_artistic_message(
            sanitized["message"],
            session.pending_artistic_decision.proposed_artistic_profile,
            session.pending_artistic_decision.alternatives,
            awaiting_modify_details=False,
        )
        _append_chat(session, "assistant", "text", staged_message)
        _append_llm_event(session, human_msg, old.id)
        _append_llm_event(session, assistant_llm_msg, old.id)
        staged_reply = {
            **sanitized,
            "message": staged_message,
            "code": old.code,
        }
        return _chat_response(staged_reply, raw_response_text, None)

    session.pending_artistic_decision = None

    changed = (
        old.emotion_profile != new_emotion
        or old.artistic_profile != new_artistic
        or old.code != new_code
    )

    created_version = None
    if changed:
        fallback_summary = _build_commit_summary(old, new_emotion, new_artistic, new_code, source="assistant")
        summary = _normalize_commit_message(sanitized["commit_message"], fallback=fallback_summary)
        created_version = create_version(
            session,
            emotion_profile=new_emotion,
            artistic_profile=new_artistic,
            code=new_code,
            emotion_confidence=new_conf,
            emotion_gaps=new_gaps,
            summary=summary,
            source="assistant",
        )
    else:
        # Confidence/gaps are read-only metadata; keep them fresh even without
        # creating a new version node.
        old.emotion_confidence = new_conf
        old.emotion_gaps = new_gaps

    # Anchor the LLM turn to the resulting version node, so branch restores
    # only replay context from the lineage the user is currently on.
    turn_anchor_id = created_version.id if created_version else old.id
    _append_llm_event(session, human_msg, turn_anchor_id)
    _append_llm_event(session, assistant_llm_msg, turn_anchor_id)

    if session.phase == "emotional_discovery" and _is_likely_code_request(message) and sanitized["emotion_profile"]:
        session.phase = "code_generation"

    _append_chat(session, "assistant", "text", sanitized["message"])
    return _chat_response(sanitized, raw_response_text, created_version)


@app.route("/api/save-version", methods=["POST"])
def api_save_version():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    if not sid:
        return jsonify({"error": "Missing session_id"}), 400

    session = get_or_create_session(sid)
    old = session.current_version

    emotion_profile = _ensure_emotion_prefix(_clean_text(data.get("emotion_profile"))) or old.emotion_profile
    artistic_profile = _clean_text(data.get("artistic_profile")) or old.artistic_profile
    code = _clean_text(data.get("code")) or old.code
    summary = _clean_text(data.get("summary")) or _build_commit_summary(
        old, emotion_profile, artistic_profile, code, source="user"
    )

    create_version(
        session,
        emotion_profile=emotion_profile,
        artistic_profile=artistic_profile,
        code=code,
        emotion_confidence=old.emotion_confidence or "",
        emotion_gaps=old.emotion_gaps,
        summary=summary,
        source="user",
    )
    _append_chat(session, "assistant", "system", "Saved a new version from your manual edits.")

    return jsonify({"ok": True, "state": serialize_state(session)})


@app.route("/api/restore-version", methods=["POST"])
def api_restore_version():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    version_id = data.get("version_id")
    if not sid or not version_id:
        return jsonify({"error": "Missing session_id or version_id"}), 400

    session = get_or_create_session(sid)
    target = session.versions.get(version_id)
    if not target:
        return jsonify({"error": "Version not found"}), 404

    session.current_version_id = target.id
    session.active_branch = target.branch
    session.phase = _infer_phase_from_version(target)
    session.pending_artistic_decision = None

    restore_message = _restore_transition_message(target.summary)
    _append_chat(session, "assistant", "system", restore_message)
    _append_llm_event(session, AIMessage(content=restore_message), target.id)

    return jsonify(
        {
            "ok": True,
            "restored_version_id": target.id,
            "message": restore_message,
            "state": serialize_state(session),
        }
    )


@app.route("/api/new-session", methods=["POST"])
def api_new_session():
    session = get_or_create_session(str(uuid.uuid4()))
    _append_chat(
        session,
        "assistant",
        "system",
        "New session started. Share an emotion, a memory, an image, or an audio clip to begin.",
    )
    return jsonify({"ok": True, "state": serialize_state(session)})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    if sid and sid in sessions:
        del sessions[sid]
    session = get_or_create_session(str(uuid.uuid4()))
    return jsonify({"ok": True, "state": serialize_state(session)})


@app.route("/api/debug", methods=["GET"])
def api_debug():
    sid = request.args.get("session_id")
    if not sid:
        return jsonify({"error": "Missing session_id"}), 400
    if sid not in sessions:
        return jsonify({"error": "Session not found"}), 404
    session = sessions[sid]
    return jsonify(
        {
            "session_id": session.session_id,
            "phase": session.phase,
            "message_count": len(session.messages),
            "llm_history_count": len(session.llm_events),
            "state": serialize_state(session),
        }
    )


if __name__ == "__main__":
    print("\nUnified p5.js Emotional Chatbot API")
    print("=" * 48)
    print("GET  /                 -> Frontend")
    print("GET  /api/state        -> Current state")
    print("POST /api/chat         -> Chat + LLM JSON parsing")
    print("POST /api/save-version -> Manual version save")
    print("POST /api/restore-version -> Restore selected version")
    print("GET  /api/history      -> Version graph data")
    print("POST /api/new-session  -> New session")
    print("POST /api/reset        -> Reset session")
    print("=" * 48)
    app.run(host="0.0.0.0", port=5001, debug=True)
