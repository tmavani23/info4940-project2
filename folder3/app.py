"""
Pretzel-Flavored HumANS v2 — Flask API with Version Control
=============================================================
Backend with git-like branching for emotion profiles, artistic profiles,
and code versions. Supports two-phase LLM interaction + multimodal input.

Run:  python app.py   → http://localhost:5001
"""

import os
import re
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

PHASE_1_SYSTEM_PROMPT = """\
You are a collaborative creative coding assistant helping novice programmers \
create p5.js sketches that express their unique lived experiences and emotions.

## YOUR CURRENT ROLE: EMOTIONAL DISCOVERY (Phase 1)

Help the human articulate the emotion or lived experience they want to express. \
Do NOT generate any code yet.

### Reminders to Include
- Start every response with a brief 1-sentence status: \
"As I currently understand, you are feeling [brief summary]..." \
(or "We're just getting started — tell me about your emotion." if first message).
- Remind the user they can say "let's code" whenever they're ready to implement, \
but that you may suggest exploring more if you're not confident yet.
- Remind the user they can share images, audio, emojis, or sketches at any time.

### How to Guide the Conversation
Ask ONE clarifying question at a time:
1. "Can you tell me the story behind this emotion? What are 2-3 emojis that describe it?"
2. "Is there a song or piece of music that captures how you felt?"
3. "Is there an image, photo, or meme that shows this feeling or visual style?"
4. "Do you have a rough sketch or doodle of what you imagine?"

Pick the most natural ones. Aim for 2-3 exchanges max before synthesizing.

### Building the Emotional Profile
Synthesize inputs into:
- Emotional tone (e.g., melancholy, euphoric, anxious)
- Narrative setting (e.g., a quiet rainy street, a crowded party)
- Desired pacing and motion (e.g., slow drift, frenetic bursts)
- Visual style (e.g., minimal and dark, chaotic and colorful)
- Symbolic associations (e.g., particles like memories, waves like grief)

### Confirming Before Moving On
Present your summary in this EXACT format:

**Emotional Summary** (at most 5 sentences)
[your summary]

**Visual Metaphors** (at most 2, each at most 5 sentences)
[metaphor 1]
[metaphor 2]

**Confidence:** LOW / HIGH

Then ask: "Does this feel right? Is there anything missing or off?"

### Low-Confidence Handling
If the user wants to move on to coding and your confidence is LOW:
- Do NOT block them. Respect their choice and proceed.
- Present your current emotional summary (format above) and clearly state \
your confidence level so they know.
- Include [EMOTION_CONFIRMED] and [SWITCH_TO_CODE] tags so the system \
switches to code generation.
- The code phase will use whatever understanding you have so far, and the \
user can always come back to refine later.

### Establishing or Updating the Emotion
Whenever the emotional profile is finalized or meaningfully updated — whether \
this is the first time OR the user has returned from code generation to refine — \
output the confirmed emotion summary using the format above.

This applies EVERY time the profile changes, not just the first time. If the \
user comes back to refine emotions after coding, compare your new understanding \
to what was previously established. If anything has changed, output the updated \
summary again so the system saves it.

### REQUIRED Structured Tags
After confirming or updating the emotional profile, you MUST append the following \
tags at the very end of your response (after all human-readable content):

[EMOTION_CONFIRMED]
[COMMIT_SUMMARY: <one-sentence description of the emotion, max 80 chars>]

Example:
[EMOTION_CONFIRMED]
[COMMIT_SUMMARY: Bittersweet nostalgia for childhood summers by the lake]

Rules:
- Include [EMOTION_CONFIRMED] every time you present a confirmed or updated \
emotional profile — first time or refinement.
- The [COMMIT_SUMMARY] should capture the ESSENCE of the emotion in plain \
language (not "Emotion confirmed" — describe WHAT the emotion IS).
- These tags are machine-parsed and stripped before display. The user will not \
see them.

### Communication Style
- Be warm, curious, non-technical, and CONCISE.
- Keep replies short — no long paragraphs.
- Mirror the user's emotional language.
- Never assume you know how they feel.
- If contradictory emotions appear, gently note it and ask.

### First-Message Guard
If the conversation history is empty (this is the user's very first message) \
and they immediately ask to code, implement, or build something WITHOUT sharing \
any emotion or experience:
- Do NOT switch to code. Do NOT include [SWITCH_TO_CODE].
- Warmly redirect: acknowledge their eagerness, then ask them to share the \
emotion or experience they want to express first. Ask 1-2 guiding questions.
- Example: "I'd love to help you build something! First, tell me — what \
emotion or experience are you hoping to capture in your sketch?"

### Confirmation Best Practice
Ideally, confirm the emotional profile before moving to code generation. But \
if the user insists on coding and you have SOME understanding of their emotion \
(i.e., there has been at least one exchange), do NOT block them:
- Present your best emotional summary with your confidence level.
- Include [EMOTION_CONFIRMED] and [SWITCH_TO_CODE] so the system proceeds.
- If the user returns from code generation to refine, re-confirm at the end \
with a new [EMOTION_CONFIRMED] block before they go back to coding.

### Detecting Phase-Switch Intent
You are responsible for detecting when the user wants to switch phases. The \
system relies on YOUR judgment — there are no hardcoded keywords. Use natural \
language understanding to detect intent.

If the user expresses ANY desire to start coding, implementing, building, \
generating, creating the sketch, or moving to the implementation phase — in \
ANY phrasing — include the tag [SWITCH_TO_CODE] at the end of your response.

Examples of intent you should detect (not limited to these):
- "let's code" / "let's build this" / "start implementing"
- "I'm ready" / "can we make it now?" / "generate something"
- "I think that's enough exploring" / "let's turn this into a sketch"
- "help me implement" / "I want to see it" / "code it up"
- Or any other phrasing that signals readiness to move to implementation.

When including [SWITCH_TO_CODE], also present your emotional summary (even if \
brief) and include [EMOTION_CONFIRMED] so the system saves it before switching.

### Important Constraints
- Do NOT generate any p5.js code in this phase.
- Do NOT skip emotional discovery even if the user seems eager for code.
"""

PHASE_2_SYSTEM_PROMPT = """\
You are a collaborative creative coding assistant helping novice programmers \
create p5.js sketches that express their unique lived experiences and emotions.

## YOUR CURRENT ROLE: CODE GENERATION (Phase 2)

You have the user's confirmed emotional profile. Translate it into p5.js code.

### Reminders to Include
- Remind the user they can say "back to feelings" to refine their emotion.
- Remind the user they can tell you what to change about the artistic expression.
- Say: "Feel free to correct me or revert to an earlier version at any time."

### Initial Generation
- Think step by step before writing code.
- Briefly explain your visual metaphor choices BEFORE the code.
- Warn: "This first version may not perfectly match what you're picturing — \
that's normal! We'll iterate together."
- State your confidence level in understanding their artistic vision.

### Code Output Format
CRITICAL: Always output the COMPLETE, FULL, RUNNABLE sketch in a single \
```javascript ... ``` code block. Even when making small changes, output the \
ENTIRE sketch — not just the changed lines. The frontend replaces the whole \
editor with your code block, so partial snippets will DESTROY the user's code.

Only include ONE ```javascript code block per response, and it MUST be the \
complete sketch.

### Explaining Code vs. Generating Code
When the user asks you to EXPLAIN which lines to tweak, point to parameters, \
or describe how the code works — do NOT output any ```javascript code blocks. \
Instead, refer to variables and lines by name in your prose (e.g., "Look for \
`this.hue = random(200, 260)` in the Particle constructor"). You can use \
inline code with single backticks (`like this`) for short references.

NEVER output partial code snippets in ```javascript fences. The system will \
mistake them for a full sketch and overwrite the editor. If you are not \
outputting a complete, runnable sketch, do NOT use ```javascript at all.

### Code Quality Requirements
Every sketch MUST include tweakable parameters at the top:
```
// === TWEAKABLE PARAMETERS ===
let particleCount = 80;  // Number of particles — higher = more crowded
let speed = 1.5;         // Movement speed — higher = more frantic
```
Each parameter needs: what it controls, the emotional effect, two example values.

### Code Structure
- Clear sections: setup(), draw(), helper functions
- Detailed comments explaining WHAT and WHY
- Structure so a novice can follow the logic

### Iteration Protocol
After each version:
1. Describe what you implemented and the emotional reasoning.
2. Ask: "Does the motion feel right? Does the color palette match?"
3. When iterating, make targeted changes but ALWAYS output the full sketch.
4. Offer: keep, adjust, or try a different direction.

### Autonomy Boundaries
You MAY: interpret inputs, propose metaphors, decide code structure.
You MUST defer to human on: meaning, direction, emotional accuracy, aesthetics.

### REQUIRED Structured Tags
At the very end of every response (after all human-readable content), you MUST \
include the following tags as applicable. These are machine-parsed and stripped \
before display — the user will not see them.

**When you output code** (every time there is a ```javascript block):
[CODE_COMMIT]
[COMMIT_SUMMARY: <what changed or was created, max 80 chars>]

Example:
[CODE_COMMIT]
[COMMIT_SUMMARY: Added slow-drifting particles with warm orange-to-pink gradient]

The summary should describe WHAT the code does or what changed, not generic \
text like "AI generated code". Be specific: "Added rain particle effect with \
blue palette", "Slowed animation speed and softened colors", etc.

**When artistic direction changes** (palette, motion style, visual metaphor, \
overall aesthetic — even if the user just asks to tweak colors or motion):
[ARTISTIC_COMMIT]
[ARTISTIC_SUMMARY: <brief description of the artistic direction, max 80 chars>]

Example:
[ARTISTIC_COMMIT]
[ARTISTIC_SUMMARY: Warm sunset palette with slow, meditative particle drift]

Include [ARTISTIC_COMMIT] whenever the visual/artistic intent shifts — not \
just the first time, but on any meaningful aesthetic change. If a code change \
also changes the artistic direction, include BOTH [CODE_COMMIT] and \
[ARTISTIC_COMMIT] tags.

### Detecting Phase-Switch Intent
You are responsible for detecting when the user wants to switch back to emotional \
discovery. The system relies on YOUR judgment — there are no hardcoded keywords. \
Use natural language understanding to detect intent.

If the user expresses ANY desire to revisit, refine, change, rethink, or explore \
their emotions — in ANY phrasing — include the tag [SWITCH_TO_EMOTIONS] after all \
other tags.

Examples of intent you should detect (not limited to these):
- "back to feelings" / "back to emotions" / "refine my emotions"
- "the emotion isn't right" / "I want to change how it feels"
- "let me rethink" / "I want to explore more"
- "this doesn't capture what I meant" / "can we revisit the emotion?"
- "I want to start over with the feeling" / "the mood is off"
- Or any other phrasing that signals a desire to revisit emotional discovery.

### Important Constraints
- Always output valid, runnable p5.js code (no pseudocode).
- ALWAYS output the complete sketch, never partial snippets.
"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _is_full_sketch(code: str) -> bool:
    """Return True if the code block looks like a complete p5.js sketch."""
    has_setup = "function setup" in code or "setup()" in code
    has_draw = "function draw" in code or "draw()" in code
    return (has_setup or has_draw) and len(code.strip().split("\n")) >= 8


def extract_code_blocks(text: str) -> list[str]:
    """Extract only FULL p5.js sketches (with setup/draw), not explanatory snippets."""
    pattern = r"```(?:javascript|js)?\s*\n(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL)
    return [b.strip() for b in blocks if _is_full_sketch(b)]


def strip_code_blocks(text: str) -> str:
    """Remove only full-sketch code blocks from the response text.
    Leaves small explanatory snippets in place so the user can see them."""
    def _replace_if_sketch(m):
        code = m.group(1) if m.group(1) else ""
        if _is_full_sketch(code):
            return ""
        return m.group(0)  # keep non-sketch code blocks as-is

    cleaned = re.sub(
        r"```(?:javascript|js)?\s*\n(.*?)```",
        _replace_if_sketch,
        text,
        flags=re.DOTALL,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _make_emotion_summary(chat_text: str, is_refinement: bool) -> str:
    """Extract a descriptive one-liner from the AI's emotion confirmation."""
    prefix = "Emotion refined" if is_refinement else "Emotion confirmed"
    # Try to grab content after "**Emotional Summary**"
    m = re.search(r"\*\*Emotional Summary\*\*[:\s]*(.+?)(?:\n\n|\*\*)", chat_text, re.DOTALL)
    if m:
        snippet = m.group(1).strip().replace("\n", " ")
        snippet = snippet[:80] + ("…" if len(snippet) > 80 else "")
        return f"{prefix}: {snippet}"
    # Fallback: first meaningful line of the summary
    for line in chat_text.split("\n"):
        line = line.strip()
        if len(line) > 15 and not line.startswith("**") and not line.startswith("*"):
            snippet = line[:80] + ("…" if len(line) > 80 else "")
            return f"{prefix}: {snippet}"
    return prefix


def _make_code_summary(full_text: str, user_msg: str) -> str:
    """Generate a descriptive commit message for AI-generated code."""
    # Use the user's message as the best description of what changed
    user_snippet = user_msg.strip().replace("\n", " ")
    if len(user_snippet) > 80:
        user_snippet = user_snippet[:77] + "…"
    if user_snippet:
        return f"Code: {user_snippet}"
    # Fallback: grab first line of AI response before code
    stripped = strip_code_blocks(full_text)
    for line in stripped.split("\n"):
        line = line.strip()
        if len(line) > 10:
            snippet = line[:80] + ("…" if len(line) > 80 else "")
            return f"Code: {snippet}"
    return "Code update"


def build_multimodal_message(text, image_b64=None, image_mime=None,
                              audio_b64=None, audio_mime=None):
    content_blocks = []
    if image_b64:
        url = image_b64 if image_b64.startswith("data:") else f"data:{image_mime or 'image/jpeg'};base64,{image_b64}"
        content_blocks.append({"type": "image_url", "image_url": {"url": url}})
    if audio_b64:
        url = audio_b64 if audio_b64.startswith("data:") else f"data:{audio_mime or 'audio/mp3'};base64,{audio_b64}"
        content_blocks.append({"type": "image_url", "image_url": {"url": url}})
    content_blocks.append({"type": "text", "text": text})
    return HumanMessage(content=content_blocks)


# ─────────────────────────────────────────────────────────────────────────────
# VERSION CONTROL — single unified timeline for emotion, artistic, and code
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Commit:
    id: str
    parent_id: Optional[str]
    branch: str
    commit_type: str          # "emotion", "artistic", "code"
    summary: str
    data: dict
    timestamp: float = field(default_factory=time.time)


class VersionStore:
    """Unified git-like store. Emotion, artistic, and code commits live in
    one timeline. Branching and revert use 'era' logic to carry forward
    the latest state of each type."""

    def __init__(self):
        self.commits: dict[str, Commit] = {}
        self.branches: dict[str, str] = {}   # branch_name -> head commit id
        self.current_branch: str = "main"

    def commit(self, commit_type: str, data: dict, summary: str,
               branch: Optional[str] = None) -> Commit:
        branch = branch or self.current_branch
        parent_id = self.branches.get(branch)
        c = Commit(
            id=str(uuid.uuid4())[:8],
            parent_id=parent_id,
            branch=branch,
            commit_type=commit_type,
            summary=summary,
            data=data,
        )
        self.commits[c.id] = c
        self.branches[branch] = c.id
        return c

    def _get_ancestry(self, commit_id: str) -> list[Commit]:
        """Walk back from commit_id to root, return oldest-first."""
        chain = []
        cid = commit_id
        while cid and cid in self.commits:
            chain.append(self.commits[cid])
            cid = self.commits[cid].parent_id
        chain.reverse()
        return chain

    def get_era_state(self, commit_id: str) -> dict:
        """Compute the full state when reverting to a commit.

        EMOTION commits use FORWARD-looking era:
          Find the latest code/artistic built on this emotion, up to
          the next emotion commit. This captures everything produced
          under that emotional direction.

        CODE / ARTISTIC commits use BACKWARD-looking snapshot:
          Find the most recent commit of each other type AT OR BEFORE
          the clicked commit. This gives the state of the world when
          that code/artistic commit was created.

        Searches ALL branches containing this commit and takes the
        most recent version across all of them.

        Returns: {"emotion": {...}, "artistic": {...}, "code": {...}}
        """
        if commit_id not in self.commits:
            return {}

        target = self.commits[commit_id]

        containing_heads = []
        for br, hid in self.branches.items():
            chain_ids = {c.id for c in self._get_ancestry(hid)}
            if commit_id in chain_ids:
                containing_heads.append(hid)
        if not containing_heads:
            containing_heads = [commit_id]

        state: dict[str, tuple[float, dict]] = {}

        for head_id in containing_heads:
            full_chain = self._get_ancestry(head_id)

            target_pos = None
            for i, c in enumerate(full_chain):
                if c.id == commit_id:
                    target_pos = i
                    break
            if target_pos is None:
                continue

            if target.commit_type == "emotion":
                # FORWARD era: from target to next emotion (exclusive)
                era_end = len(full_chain)
                for i in range(target_pos + 1, len(full_chain)):
                    if full_chain[i].commit_type == "emotion":
                        era_end = i
                        break
                for i in range(target_pos, era_end):
                    c = full_chain[i]
                    prev = state.get(c.commit_type)
                    if prev is None or c.timestamp >= prev[0]:
                        state[c.commit_type] = (c.timestamp, c.data)
            else:
                # BACKWARD snapshot: most recent of each type at or before target
                for i in range(target_pos, -1, -1):
                    c = full_chain[i]
                    if c.commit_type not in state:
                        state[c.commit_type] = (c.timestamp, c.data)
                    if len(state) == 3:
                        break

        return {
            "emotion": state["emotion"][1] if "emotion" in state else None,
            "artistic": state["artistic"][1] if "artistic" in state else None,
            "code": state["code"][1] if "code" in state else None,
        }

    def checkout(self, commit_id: str, new_branch: Optional[str] = None) -> Commit:
        if commit_id not in self.commits:
            raise ValueError(f"Commit {commit_id} not found")
        c = self.commits[commit_id]
        if new_branch:
            self.branches[new_branch] = commit_id
            self.current_branch = new_branch
        else:
            self.current_branch = c.branch
        return c

    @property
    def head(self) -> Optional[Commit]:
        cid = self.branches.get(self.current_branch)
        return self.commits.get(cid) if cid else None

    def get_full_graph(self) -> dict:
        nodes = []
        for c in self.commits.values():
            nodes.append({
                "id": c.id, "parent_id": c.parent_id, "branch": c.branch,
                "summary": c.summary, "data": c.data,
                "timestamp": c.timestamp, "type": c.commit_type,
            })
        nodes.sort(key=lambda n: n["timestamp"])
        return {
            "current_branch": self.current_branch,
            "branches": dict(self.branches),
            "commits": nodes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SESSION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Session:
    session_id: str
    phase: str = "emotional_discovery"
    history: list = field(default_factory=list)
    latest_code: Optional[str] = None
    store: VersionStore = field(default=None)
    emotion_profile: dict = field(default_factory=dict)
    artistic_profile: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.store is None:
            self.store = VersionStore()


sessions: dict[str, Session] = {}


def get_or_create_session(sid: Optional[str] = None) -> Session:
    if sid and sid in sessions:
        return sessions[sid]
    new_id = sid or str(uuid.uuid4())
    s = Session(session_id=new_id)
    sessions[new_id] = s
    return s


# ─────────────────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────────────────

phase_1_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.8)
phase_2_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)


def get_llm(phase: str):
    return phase_1_llm if phase == "emotional_discovery" else phase_2_llm


def get_system_message(phase: str):
    prompt = PHASE_1_SYSTEM_PROMPT if phase == "emotional_discovery" else PHASE_2_SYSTEM_PROMPT
    return SystemMessage(content=prompt)


# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)


# ── POST /api/chat ───────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message'"}), 400

    session = get_or_create_session(data.get("session_id"))
    phase_switched = False

    msg_text = data["message"]

    # If the user reverted to a previous version, inject era state as context
    rev = data.get("reverted_context")
    if rev:
        cid = rev.get("commit_id", "")
        ctype = rev.get("commit_type", "")
        branch = rev.get("branch", "")
        parts = [
            f"[SYSTEM: The user reverted to a previous {ctype.upper()} commit "
            f"({cid}, branch '{branch}'). The full era state is restored below. "
            f"Work from THIS state going forward.]\n"
        ]
        emo = rev.get("emotion")
        if emo:
            parts.append(f"Current emotion profile:\n{emo.get('summary', json.dumps(emo))}\n")
            session.emotion_profile = emo
        art = rev.get("artistic")
        if art:
            parts.append(f"Current artistic profile:\n{json.dumps(art)}\n")
            session.artistic_profile = art
        code = rev.get("code")
        if code and code.get("code"):
            parts.append(f"Current code:\n```javascript\n{code['code']}\n```\n")
            session.latest_code = code["code"]
        msg_text = "\n".join(parts) + "\n" + msg_text

    user_msg = build_multimodal_message(
        text=msg_text,
        image_b64=data.get("image"), image_mime=data.get("image_mime"),
        audio_b64=data.get("audio"), audio_mime=data.get("audio_mime"),
    )
    session.history.append(user_msg)

    messages = [get_system_message(session.phase)] + session.history
    llm = get_llm(session.phase)

    try:
        response = llm.invoke(messages)
    except Exception as e:
        return jsonify({"error": f"LLM call failed: {str(e)}"}), 500

    session.history.append(response)
    full_text = response.content
    code_blocks = extract_code_blocks(full_text)
    chat_text = strip_code_blocks(full_text) if code_blocks else full_text

    # ── Parse structured tags from LLM response ──
    def _extract_tag(tag_name: str) -> Optional[str]:
        """Extract value from [TAG_NAME: value] or check for bare [TAG_NAME]."""
        m = re.search(rf"\[{tag_name}:\s*(.+?)\]", full_text)
        return m.group(1).strip() if m else None

    def _has_tag(tag_name: str) -> bool:
        return f"[{tag_name}]" in full_text

    commit_summary = _extract_tag("COMMIT_SUMMARY")
    artistic_summary = _extract_tag("ARTISTIC_SUMMARY")

    # ── Emotion confirmation (LLM tag or legacy fallback) ──
    emotion_confirmed = _has_tag("EMOTION_CONFIRMED")
    if not emotion_confirmed:
        # Legacy fallback: detect **CONFIRMED** or structured summary
        emotion_confirmed = (
            "**CONFIRMED**" in full_text
            or ("**Emotional Summary**" in full_text
                and "**Confidence:**" in full_text
                and session.phase == "emotional_discovery")
        )

    if emotion_confirmed:
        new_profile = {"raw_summary": chat_text}
        old_profile = session.emotion_profile
        is_changed = new_profile != old_profile
        session.emotion_profile = new_profile
        if is_changed or not old_profile:
            is_refinement = bool(old_profile)
            label = commit_summary or _make_emotion_summary(chat_text, is_refinement)
            if not label.startswith("Emotion"):
                label = ("Emotion refined: " if is_refinement else "Emotion: ") + label
            session.store.commit("emotion", data={"summary": chat_text}, summary=label)

    # ── Code commit (LLM tag or auto-detect from code blocks) ──
    if code_blocks:
        new_code = code_blocks[0]
        if new_code != session.latest_code:
            session.latest_code = new_code
            label = commit_summary or _make_code_summary(full_text, data["message"])
            if not label.startswith("Code"):
                label = "Code: " + label
            session.store.commit("code", data={"code": new_code}, summary=label)

    # ── Artistic profile commit (LLM-driven only) ──
    if _has_tag("ARTISTIC_COMMIT") and artistic_summary:
        session.artistic_profile = {"description": artistic_summary}
        session.store.commit("artistic",
                             data={"description": artistic_summary},
                             summary=f"Artistic: {artistic_summary}")

    # ── Phase switching (LLM-driven tags) ──
    if _has_tag("SWITCH_TO_CODE") and not phase_switched:
        if len(session.history) > 1:
            session.phase = "code_generation"
            phase_switched = True
        # else: no history yet, ignore the tag (shouldn't happen normally)
    elif _has_tag("SWITCH_TO_EMOTIONS") and not phase_switched:
        session.phase = "emotional_discovery"
        phase_switched = True

    # Strip all machine tags from the user-visible reply
    chat_text = re.sub(
        r"\[(?:EMOTION_CONFIRMED|CODE_COMMIT|ARTISTIC_COMMIT|"
        r"SWITCH_TO_CODE|SWITCH_TO_EMOTIONS)\]",
        "", chat_text
    )
    chat_text = re.sub(r"\[COMMIT_SUMMARY:\s*.+?\]", "", chat_text)
    chat_text = re.sub(r"\[ARTISTIC_SUMMARY:\s*.+?\]", "", chat_text)
    chat_text = re.sub(r"\n{3,}", "\n\n", chat_text).strip()

    return jsonify({
        "session_id": session.session_id,
        "phase": session.phase,
        "phase_switched": phase_switched,
        "needs_confirmation": False,
        "reply": chat_text,
        "code": code_blocks[0] if code_blocks else None,
        "full_response": full_text,
        "message_count": len(session.history),
        "latest_code": session.latest_code,
    })


# ── POST /api/switch-phase ──────────────────────────────────────────────────

@app.route("/api/switch-phase", methods=["POST"])
def switch_phase():
    data = request.get_json()
    if not data or "session_id" not in data or "phase" not in data:
        return jsonify({"error": "Missing 'session_id' or 'phase'"}), 400
    session = get_or_create_session(data["session_id"])
    if data["phase"] not in ("emotional_discovery", "code_generation"):
        return jsonify({"error": "Invalid phase"}), 400
    if data["phase"] == "code_generation" and not session.emotion_profile and len(session.history) == 0:
        return jsonify({
            "error": "Please share your emotion or experience first before coding.",
            "needs_confirmation": True,
        }), 400
    old = session.phase
    session.phase = data["phase"]
    return jsonify({"session_id": session.session_id, "old_phase": old,
                     "new_phase": session.phase})


# ── POST /api/reset ──────────────────────────────────────────────────────────

@app.route("/api/reset", methods=["POST"])
def reset():
    data = request.get_json()
    sid = data.get("session_id") if data else None
    if sid and sid in sessions:
        del sessions[sid]
    return jsonify({"status": "reset", "session_id": sid})


# ── GET /api/status ──────────────────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def status():
    sid = request.args.get("session_id")
    if not sid or sid not in sessions:
        return jsonify({"exists": False})
    s = sessions[sid]
    return jsonify({
        "exists": True, "session_id": s.session_id, "phase": s.phase,
        "message_count": len(s.history), "latest_code": s.latest_code,
        "emotion_profile": s.emotion_profile,
        "artistic_profile": s.artistic_profile,
    })


# ── POST /api/update-code ───────────────────────────────────────────────────

@app.route("/api/update-code", methods=["POST"])
def update_code():
    data = request.get_json()
    if not data or "session_id" not in data or "code" not in data:
        return jsonify({"error": "Missing 'session_id' or 'code'"}), 400
    session = get_or_create_session(data["session_id"])
    session.latest_code = data["code"]
    feedback = data.get("feedback", "I edited the code directly.")
    session.history.append(HumanMessage(content=(
        f"{feedback}\n\nHere is my current code:\n```javascript\n{data['code']}\n```"
    )))
    session.store.commit("code", data={"code": data["code"]},
                         summary=f"User edit: {feedback[:60]}")
    return jsonify({"session_id": session.session_id, "status": "code_updated"})


# ── POST /api/commit ─────────────────────────────────────────────────────────
# Manual commit for emotion or artistic profile changes

@app.route("/api/commit", methods=["POST"])
def manual_commit():
    data = request.get_json()
    if not data or "session_id" not in data or "category" not in data:
        return jsonify({"error": "Missing fields"}), 400
    session = get_or_create_session(data["session_id"])
    cat = data["category"]
    if cat not in ("emotion", "artistic", "code"):
        return jsonify({"error": "Invalid category"}), 400
    payload = data.get("data", {})
    summary = data.get("summary", "Manual commit")
    branch = data.get("branch")
    c = session.store.commit(cat, data=payload, summary=summary, branch=branch)
    if cat == "emotion":
        session.emotion_profile = payload
    elif cat == "artistic":
        session.artistic_profile = payload
    elif cat == "code" and "code" in payload:
        session.latest_code = payload["code"]
    return jsonify({"commit_id": c.id, "branch": c.branch})


# ── POST /api/checkout ───────────────────────────────────────────────────────
# Revert to a previous commit, optionally creating a new branch

@app.route("/api/checkout", methods=["POST"])
def checkout():
    """Revert to a commit. Returns the full era state (latest of each type
    in that commit's era) so the frontend can load everything."""
    data = request.get_json()
    if not data or "session_id" not in data or "commit_id" not in data:
        return jsonify({"error": "Missing fields"}), 400
    session = get_or_create_session(data["session_id"])
    new_branch = data.get("new_branch")
    commit_id = data["commit_id"]

    # Compute era state BEFORE checkout (needs the old branch structure)
    era_state = session.store.get_era_state(commit_id)

    try:
        c = session.store.checkout(commit_id, new_branch=new_branch)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    # Apply era state to session
    if era_state.get("emotion"):
        session.emotion_profile = era_state["emotion"]
    if era_state.get("artistic"):
        session.artistic_profile = era_state["artistic"]
    if era_state.get("code") and "code" in era_state["code"]:
        session.latest_code = era_state["code"]["code"]

    # Set phase for code/artistic reverts; emotion reverts let the user choose
    if c.commit_type in ("code", "artistic"):
        session.phase = "code_generation"

    return jsonify({
        "commit_id": c.id,
        "commit_type": c.commit_type,
        "branch": session.store.current_branch,
        "phase": session.phase,
        "summary": c.summary,
        "era_state": {
            "emotion": era_state.get("emotion"),
            "artistic": era_state.get("artistic"),
            "code": era_state.get("code"),
        },
    })


# ── GET /api/history ─────────────────────────────────────────────────────────
# Full version graph for a category

@app.route("/api/history", methods=["GET"])
def history():
    sid = request.args.get("session_id")
    if not sid or sid not in sessions:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(sessions[sid].store.get_full_graph())


# ── GET /api/debug ───────────────────────────────────────────────────────────
# Dump ALL session state for the debug interface

@app.route("/api/debug", methods=["GET"])
def debug():
    sid = request.args.get("session_id")
    if not sid or sid not in sessions:
        return jsonify({"error": "Session not found"}), 404
    s = sessions[sid]
    chat_history = []
    for msg in s.history:
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else "[multimodal]"
            chat_history.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage):
            chat_history.append({"role": "ai", "content": msg.content})
    return jsonify({
        "session_id": s.session_id,
        "phase": s.phase,
        "emotion_profile": s.emotion_profile,
        "artistic_profile": s.artistic_profile,
        "latest_code": s.latest_code,
        "message_count": len(s.history),
        "chat_history": chat_history,
        "version_graph": s.store.get_full_graph(),
    })


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n Pretzel-Flavored HumANS v2 — API Server")
    print("=" * 55)
    print("  POST /api/chat          — chat with AI")
    print("  POST /api/switch-phase  — switch phase")
    print("  POST /api/reset         — reset session")
    print("  GET  /api/status        — session state")
    print("  POST /api/update-code   — sync code edits")
    print("  POST /api/commit        — manual version commit")
    print("  POST /api/checkout      — revert / branch")
    print("  GET  /api/history       — version graph")
    print("  GET  /api/debug         — full state dump")
    print("=" * 55)
    app.run(debug=True, port=5001)
