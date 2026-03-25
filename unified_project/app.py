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
import base64
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
- In user-facing `message` text, use lightweight markdown for readability: short sections or bullets when helpful, and `**bold**` only for the most important takeaway.
- Avoid long walls of text. Prefer 1-2 short paragraphs or 2-5 concise bullets.
""".strip()


DISCOVERY_PROMPT = """
You are in emotional discovery mode. Do not generate new code unless the user explicitly asks to start coding.

Conversation goals:
- Summarize the current emotional understanding.
- Ask one useful follow-up that closes the biggest gap.
- Remind them they can move to coding when ready.
- Explicitly state that you may still misunderstand parts of their emotional experience and that they can correct you at any time.
- Keep the message short and supportive.
- Structure the `message` so it is easy to scan, using simple markdown paragraphs or bullets.

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
- In your `message`, use only simple markdown supported by chat UIs: paragraphs, `-` bullets, numbered lists, and `**bold**`.
- Keep the `message` compact and scannable. Do not return a large wall of text.
- No markdown fences. No extra keys. No commentary outside JSON.
""".strip()


# Score 3-4 reference: "first date excitement" Processing/Java source
FIRST_DATE_CODE = """\
float t;
float theta;
int maxFrameCount = 75;
int a = 101;
int space = 100;
void setup(){size(540,540,P3D);}
void draw(){
  background(5);translate(width/2,height/2);
  t=(float)frameCount/maxFrameCount;theta=TWO_PI*t;
  directionalLight(245,245,245,300,-200,-200);ambientLight(240,240,240);
  rotateY(radians(145));rotateX(radians(45));
  for(int x=-space;x<=space;x+=20){for(int y=-space;y<=space;y+=20){for(int z=-space;z<=space;z+=200){
    float offSet=((x*y*z))/a;
    float sz=map(sin(-theta+offSet),-1,1,-0,20);
    color c1=color(240,40,100);color c2=color(40,40,90);
    if((x*y*z)%30==0){fill(c1);stroke(c2);}else{fill(c2);stroke(c1);}
    shp(x,y,z,sz);shp(y,z,x,sz);shp(z,x,y,sz);}}}
}
void shp(float x,float y,float z,float d){pushMatrix();translate(x,y,z);box(d);popMatrix();}""".strip()

ARTISTIC_QUALITY_REFERENCE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARTISTIC QUALITY STANDARD — target score 6-7 (scale 1-7)
Note: ALL examples below are dynamic, animated, interactive p5.js sketches.
The criterion is expressiveness and resonance.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCORING RUBRIC (1-7): Criterion: Expressiveness and Resonance

Overall Scores:
  1-2: Emotion is named or symbolized only.
       Generic shapes, stock colors, predictable motion.
       Could be any emotion with a theme swap.

  3-4: Emotion is present in visual choices but not in rendering logic.
       You recognize the emotion intellectually.
       The sketch is "about" the emotion, not "made of" it.

  5-6: Rendering logic has emotional texture.
       Some instability, tension, or weight baked into the math.
       But still has decorative elements carrying part of the load.

  7:   The rendering logic IS the emotion.
       You feel it before you understand it.
       The sketch cannot be re-themed — rewriting the emotion
       requires rewriting the core code. 

Breakdown:
         EXPRESSIVENESS                    RESONANCE
         (is the technique the emotion?)   (does the viewer feel it?)

1-2      Literal symbol mapping.           Recognize only — never feel.
         Tears for sadness, hearts for     Re-labelable with zero
         love. Imagery does all the work.  mechanical changes.

3-4      Thematic abstraction.             Mild feeling — mostly
         Motion/color evoke the emotion    intellectual recognition.
         without illustrating directly.    Re-labelable with minor
         Core mechanic still generic.      parameter tweaks.

5-6      Technique has emotional texture.  Feel before you understand.
         Instability, tension, or weight   Core mechanic is hard to
         baked into the rendering math.    re-label but not impossible.

7        The technique IS the emotion.     Cannot watch it without
(TARGET) Mechanic is unexpected — you      feeling it. Impossible to
         could not predict this approach   re-label without rewriting
         from the emotion word alone.      the core mechanic entirely.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORE 7 EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"feeling disheveled" — score 7
WHY: Captures the viewer's live webcam face and makes their OWN FACE appear chaotic. Dark face
  regions become anchor points for random-walk "hair" strands. The viewer sees themselves as the
  disheveled subject. The sketch borrows the viewer's identity to produce the feeling — it has no
  content of its own.
KEY TECHNIQUES: createCapture(VIDEO), pixel darkness → dot radius, random-walk from dark regions.
CODE (p5.js):
let capture;let isCaptured=false;
var NORTH=0,NORTHEAST=1,EAST=2,SOUTHEAST=3,SOUTH=4,SOUTHWEST=5,WEST=6,NORTHWEST=7;
var direction,posX,posY;
function setup(){createCanvas(640,480);capture=createCapture(VIDEO,function(){isCaptured=true;});capture.size(640,480);background(255);}
function draw(){clear();background(255);if(!isCaptured)return;capture.loadPixels();const stepSize=10;for(let y=0;y<height;y+=stepSize){for(let x=0;x<width;x+=stepSize){const i=y*width+x;const darkness=(255-capture.pixels[i*4])/255;const radius=stepSize*darkness;ellipse(x,y,radius,radius);if(darkness>0.8){posX=x;posY=y;drawHair();}}}}
function drawHair(){var diameter=2,stepSize=2;for(var i=0;i<=20;i++){direction=int(random(0,8));if(direction==NORTH)posY-=stepSize;else if(direction==NORTHEAST){posX+=stepSize;posY-=stepSize;}else if(direction==EAST)posX+=stepSize;else if(direction==SOUTHEAST){posX+=stepSize;posY+=stepSize;}else if(direction==SOUTH)posY+=stepSize;else if(direction==SOUTHWEST){posX-=stepSize;posY+=stepSize;}else if(direction==WEST)posX-=stepSize;else if(direction==NORTHWEST){posX-=stepSize;posY-=stepSize;}if(posX>width)posX=0;if(posX<0)posX=width;if(posY<0)posY=height;if(posY>height)posY=0;ellipse(posX+stepSize/2,posY+stepSize/2,diameter,diameter);}}

---

"feeling anxious" — score 7
WHY: Anxiety is cognitive overload — too many competing signals. The sketch renders complex Japanese
  literary characters at sizes and colors driven by live audio FFT waveform amplitude. Every frame
  the characters change size and color based on sound. Illegibility + randomness + audio-driven chaos
  creates experienced overstimulation, not a symbol of it.
KEY TECHNIQUES: p5.FFT waveform(), per-character random fill driven by waveform[i], foreign script.
CODE (p5.js):
let sound,fft;
let letters='吾輩わがはいは猫である。名前はまだ無い。どこで生れたかとんと見当けんとうがつかぬ。'.split('');
function preload(){sound=loadSound('Catch_the_future.mp3');}
function setup(){createCanvas(windowWidth,windowHeight);textFont('sans-serif');textAlign(CENTER,CENTER);fft=new p5.FFT();sound.amp(0.2);sound.loop();}
function draw(){clear();background(50);let waveform=fft.waveform();beginShape();noFill();stroke(255);for(let i=0;i<waveform.length;i++){vertex(map(i,0,waveform.length,0,width),map(waveform[i],-1,1,height,0));}endShape();for(let i=0;i<letters.length;i++){var j=Math.round(map(i,0,letters.length,0,waveform.length));fill(color(random(100,255),random(100,255),random(100,255)));textSize(Math.abs(waveform[j])*300+5);text(letters[i],map(j,0,waveform.length,0,width),map(waveform[j]*3,-1,1,height,0));}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORE 3-4 REFERENCES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Score 3-4 reference examples (code and screenshots) were provided as context at session start.
""".strip()


IMPLEMENTATION_PROMPT = """
You are in implementation mode. Use the current emotional understanding to produce runnable p5.js.

Conversation goals:
- Briefly explain your visual metaphor and what changed this turn.
- Set expectation: this may need iteration and they can revert any time.
- Keep language novice-friendly.
- If you are proposing a changed artistic direction, explain it naturally and invite approval or edits before finalizing.
- Do not use backend-style decision labels like "Art Direction Decision:" in your `message`.
- Structure the `message` so it is easy to scan, using short markdown sections or bullets.

Code requirements:
- Return complete runnable p5.js in the `code` field.
- Include setup() and draw().
- Keep incremental updates when possible.
- Put tweakable parameters near the top with comments.
- Add clear beginner-friendly comments for each major section (state setup, animation logic, color/motion choices, and any interaction).
- For non-obvious lines (timing math, mapping ranges, easing), include short explanatory comments.

{ARTISTIC_QUALITY_REFERENCE}

Return STRICT JSON only with this exact schema:
{
  "message": "<chat text shown to user>",
  "commit_message": "<short git-style summary of this turn>",
  "emotion_profile": "The emotion is ...",
  "emotion_confidence": "<low|medium|high>",
  "emotion_gaps": ["<gap 1>", "<gap 2>"],
  "artistic_profile": "<1-3 sentences of visual direction>",
  "code": "<full runnable p5.js>",
  "should_create_version": true,
  "offers_artistic_alternatives": false,
  "artistic_options": ["<option 1>", "<option 2>"]
}

Constraints:
- emotion_profile must start with "The emotion is".
- emotion_confidence must be exactly low, medium, or high.
- emotion_gaps must be an array (possibly empty).
- commit_message should be one concise line describing what changed in the sketch/profile.
- In your `message`, always set expectation that this implementation may be imperfect and invite corrections at any time.
- In your `message`, use only simple markdown supported by chat UIs: paragraphs, `-` bullets, numbered lists, and `**bold**`.
- Keep the `message` compact and avoid a large unbroken paragraph.
- Set `should_create_version` to true only when you want the app to actually apply a new sketch/profile update now.
- Set `should_create_version` to false when you are mainly brainstorming, advising, asking a follow-up, or suggesting ideas without actually changing the sketch yet.
- If the user asks for ideas, feedback, or "what else should I add," usually set `should_create_version` to false.
- Set `offers_artistic_alternatives` to true when your `message` presents multiple artistic directions or asks the user to choose among alternatives.
- When `offers_artistic_alternatives` is true, fill `artistic_options` with 2-3 concise alternatives.
- When `should_create_version` is false, do not treat the sketch as already changed.
- When `offers_artistic_alternatives` is false, `artistic_options` should usually be empty.
- No markdown fences. No extra keys. No commentary outside JSON.
""".strip()


INTENT_CLASSIFIER_PROMPT = """
You are the primary intent router for a creative coding chatbot.
Infer the user's workflow action from the latest user message, the previous assistant message, and the current state.
There are no regex heuristics for natural-language intent, so use semantic judgment carefully.

Return STRICT JSON only with this exact schema:
{
  "code_request": false,
  "emotion_refinement": false,
  "confirm_start_coding": false,
  "sketch_rejection": false,
  "art_direction_decision": "none",
  "confidence": "low"
}

Allowed values for `art_direction_decision`:
- "none"
- "accept"
- "modify"
- "more_options"
- "option_1"
- "option_2"
- "option_3"

Rules:
- `code_request` is true only if the user wants to start coding, continue coding, or directly change the running sketch now.
- `emotion_refinement` is true only if the user wants to go back to discussing the feeling itself, correcting the emotional understanding, or adding more emotional context.
- `confirm_start_coding` is true only if the user's message is mainly an agreement to the assistant's immediately previous invitation to begin implementation.
- `sketch_rejection` is true when the user is rejecting the current sketch, saying it is off, or asking for revision of the current implementation.
- In implementation mode, feedback about what should change in the sketch usually means `sketch_rejection`, not `emotion_refinement`.
- Use `art_direction_decision` only if there is a pending artistic-direction proposal.
- If `awaiting_modify_details` is true, descriptive text counts as modification details, so `art_direction_decision` should usually be "none" unless the user is explicitly choosing Accept, Modify, More options, or Option N.
- If the user describes their own artistic vision instead of choosing a canned option, return `art_direction_decision` as "none".
- A short reply like "sure", "ready", or "let's do it" can be `confirm_start_coding` if the assistant just invited coding.
- A short reply like "no", "not good", or "still off" can be `sketch_rejection` if there is already a sketch.
- Multiple fields may be false at once. Prefer false / "none" when unsure.
- Rely on the latest user message plus the provided recent assistant message and state. Do not infer hidden intent from topic alone.
- `confidence` must be exactly low, medium, or high.
- Examples:
  Assistant: "We can start coding whenever you're ready."
  User: "sure"
  Output: {"code_request": false, "emotion_refinement": false, "confirm_start_coding": true, "sketch_rejection": false, "art_direction_decision": "none", "confidence": "high"}

  Assistant: "Here's the first sketch."
  User: "no thats not good"
  Output: {"code_request": false, "emotion_refinement": false, "confirm_start_coding": false, "sketch_rejection": true, "art_direction_decision": "none", "confidence": "high"}

  Assistant: "Choose an artistic direction."
  User: "Art Direction Decision: Option 2"
  Output: {"code_request": false, "emotion_refinement": false, "confirm_start_coding": false, "sketch_rejection": false, "art_direction_decision": "option_2", "confidence": "high"}

  Pending artistic direction, awaiting_modify_details: yes
  User: "make the papers fly faster and feel heavier"
  Output: {"code_request": false, "emotion_refinement": false, "confirm_start_coding": false, "sketch_rejection": false, "art_direction_decision": "none", "confidence": "high"}

  User: "let's start coding"
  Output: {"code_request": true, "emotion_refinement": false, "confirm_start_coding": false, "sketch_rejection": false, "art_direction_decision": "none", "confidence": "high"}

  User: "the anxiety is more numb than frantic"
  Output: {"code_request": false, "emotion_refinement": true, "confirm_start_coding": false, "sketch_rejection": false, "art_direction_decision": "none", "confidence": "high"}
- No markdown fences. No extra keys. No commentary outside JSON.
""".strip()


def build_system_prompt(phase: str, state_summary: str) -> str:
    phase_prompt = DISCOVERY_PROMPT if phase == "emotional_discovery" else IMPLEMENTATION_PROMPT.replace(
        "{ARTISTIC_QUALITY_REFERENCE}", ARTISTIC_QUALITY_REFERENCE
    )
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


def _normalize_artistic_options(value: Any) -> list[str]:
    if isinstance(value, list):
        options = [_clean_text(v) for v in value]
    elif isinstance(value, dict):
        options = [_clean_text(v) for v in value.values()]
    else:
        raw = _clean_text(value)
        if not raw:
            options = []
        else:
            parts = re.split(r"\n|;|•", raw)
            options = [p.strip(" -\t") for p in parts if p.strip()]

    deduped = []
    seen = set()
    for option in options:
        key = option.lower()
        if option and key not in seen:
            deduped.append(option)
            seen.add(key)
    return deduped[:3]


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


def _extract_code_text(text: Any) -> str:
    raw = _clean_text(text)
    if not raw:
        return ""
    fence = re.search(r"```(?:javascript|js)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return raw


def _is_probably_complete_p5_code(code: str) -> tuple[bool, str]:
    text = _clean_text(code)
    if not text:
        return False, "Code was empty."
    if not re.search(r"\bsetup\s*\(", text):
        return False, "Missing setup()."
    if not re.search(r"\bdraw\s*\(", text):
        return False, "Missing draw()."

    stack: list[str] = []
    state = "normal"
    i = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if state == "normal":
            if ch == "/" and nxt == "/":
                state = "line_comment"
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block_comment"
                i += 2
                continue
            if ch == "'":
                state = "single_quote"
                i += 1
                continue
            if ch == '"':
                state = "double_quote"
                i += 1
                continue
            if ch == "`":
                state = "template_quote"
                i += 1
                continue
            if ch in "({[":
                stack.append(ch)
            elif ch in ")}]":
                pairs = {")": "(", "}": "{", "]": "["}
                if not stack or stack.pop() != pairs[ch]:
                    return False, f"Unbalanced delimiter: {ch}"
        elif state == "line_comment":
            if ch == "\n":
                state = "normal"
        elif state == "block_comment":
            if ch == "*" and nxt == "/":
                state = "normal"
                i += 2
                continue
        elif state == "single_quote":
            if ch == "\\":
                i += 2
                continue
            if ch == "'":
                state = "normal"
        elif state == "double_quote":
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                state = "normal"
        elif state == "template_quote":
            if ch == "\\":
                i += 2
                continue
            if ch == "`":
                state = "normal"
        i += 1

    if state != "normal":
        return False, "Code ended mid-string or comment."
    if stack:
        return False, "Code has unclosed brackets or braces."
    return True, ""


def _collapse_spaces(value: Any) -> str:
    return " ".join(_clean_text(value).split())


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _clean_text(value).lower() in {"true", "1", "yes", "y"}


def _normalize_artistic_profile(text: str) -> str:
    return re.sub(r"\s+", " ", _clean_text(text)).strip().lower()


def _has_artistic_change(old: str, new: str) -> bool:
    return _normalize_artistic_profile(old) != _normalize_artistic_profile(new)


def _has_emotion_profile_change(old: str, new: str) -> bool:
    return _clean_text(old) != _clean_text(new)


def _has_emotional_state_change(
    old: "VersionNode",
    new_emotion: str,
    new_confidence: str,
    new_gaps: list[str],
) -> bool:
    return (
        _has_emotion_profile_change(old.emotion_profile, new_emotion)
        or old.emotion_confidence != new_confidence
        or old.emotion_gaps != new_gaps
    )


def _has_material_implementation_change(old: "VersionNode", new_artistic: str, new_code: str) -> bool:
    return old.artistic_profile != new_artistic or old.code != new_code


def _update_current_emotion_state(
    version: "VersionNode",
    *,
    emotion_profile: str,
    emotion_confidence: str,
    emotion_gaps: list[str],
) -> None:
    version.emotion_profile = emotion_profile
    version.emotion_confidence = emotion_confidence
    version.emotion_gaps = emotion_gaps


def _parse_canonical_art_direction_command(text: str) -> Optional[str]:
    cleaned = _collapse_spaces(text)
    if not cleaned:
        return None
    if cleaned == ART_DECISION_ACCEPT:
        return "accept"
    if cleaned == ART_DECISION_MODIFY:
        return "modify"
    if cleaned == ART_DECISION_MORE:
        return "more_options"
    if cleaned.startswith(ART_DECISION_OPTION_PREFIX):
        suffix = cleaned[len(ART_DECISION_OPTION_PREFIX) :].strip()
        if suffix.isdigit():
            return f"option_{int(suffix)}"
    return None


def _normalize_classified_art_direction_decision(value: Any) -> Optional[str]:
    raw = _collapse_spaces(value).lower().replace("-", "_").replace(" ", "_")
    if not raw or raw == "none":
        return None
    if raw in {"accept", "accepted"}:
        return "accept"
    if raw == "modify":
        return "modify"
    if raw in {"more_options", "more"}:
        return "more_options"
    if raw.startswith("option_"):
        suffix = raw.split("option_", 1)[1]
        if suffix.isdigit():
            return f"option_{int(suffix)}"
    return None


def _get_last_assistant_text(session: "SessionState") -> str:
    for entry in reversed(session.messages):
        if entry.role == "assistant" and entry.type == "text":
            return _clean_text(entry.content)
    return ""


def _build_code_start_confirmation_prompt(session: "SessionState", user_reply: str) -> str:
    last_assistant = _safe_summary_text(_get_last_assistant_text(session), limit=240)
    reply = _clean_text(user_reply) or "Yes"
    prompt_lines = [
        f'The user confirmed they want to start coding now. Their reply was: "{reply}".',
        "Create the first runnable p5.js sketch based on the current emotional understanding.",
    ]
    if last_assistant:
        prompt_lines.append(
            f"Use the assistant's immediately previous coding suggestion if it still fits: {last_assistant}"
        )
    return "\n".join(prompt_lines)


def _build_sketch_rejection_prompt(session: "SessionState", user_feedback: str) -> str:
    last_assistant = _safe_summary_text(_get_last_assistant_text(session), limit=240)
    current = session.current_version
    feedback = _clean_text(user_feedback) or "The current sketch is not working yet."
    prompt_lines = [
        f'The user rejected the current sketch with this feedback: "{feedback}".',
        "Do not assume they want to return to emotional discovery unless they say so.",
        "Do not claim that a new version has already been applied.",
        "Keep the current sketch unchanged for now by setting should_create_version to false.",
        "Respond helpfully by briefly acknowledging the miss, naming 2-4 concrete revision axes they can react to, and asking one focused follow-up question.",
        "If the user's feedback is short or vague, infer as little as possible and ask for clarification about what feels most off.",
        f"Current artistic profile: {current.artistic_profile or '(empty)'}",
        f"Current emotion profile: {current.emotion_profile or '(empty)'}",
    ]
    if last_assistant:
        prompt_lines.append(f"Assistant's previous message: {last_assistant}")
    return "\n".join(prompt_lines)


def _sanitize_turn_action_payload(raw: dict) -> dict:
    return {
        "code_request": _coerce_bool(raw.get("code_request")),
        "emotion_refinement": _coerce_bool(raw.get("emotion_refinement")),
        "confirm_start_coding": _coerce_bool(raw.get("confirm_start_coding")),
        "sketch_rejection": _coerce_bool(raw.get("sketch_rejection")),
        "art_direction_decision": _normalize_classified_art_direction_decision(raw.get("art_direction_decision")),
        "confidence": _normalize_confidence(raw.get("confidence")),
    }


def _classify_turn_actions_with_llm(
    session: "SessionState",
    message: str,
    pending: Optional["PendingArtisticDecision"],
) -> dict:
    defaults = {
        "code_request": False,
        "emotion_refinement": False,
        "confirm_start_coding": False,
        "sketch_rejection": False,
        "art_direction_decision": None,
        "confidence": "low",
    }

    if not message or not LLM_ENABLED or intent_llm is None:
        return defaults

    last_assistant = _get_last_assistant_text(session) or "(none)"
    pending_summary_lines = [f"Pending artistic decision: {'yes' if pending else 'no'}"]
    if pending:
        pending_summary_lines.append(
            f"Pending proposed artistic profile: {pending.proposed_artistic_profile or '(empty)'}"
        )
        pending_summary_lines.append(
            f"Awaiting modify details: {'yes' if pending.awaiting_modify_details else 'no'}"
        )
        if pending.alternatives:
            for index, option in enumerate(pending.alternatives[:3], start=1):
                pending_summary_lines.append(f"Option {index}: {option}")
    pending_summary = "\n".join(pending_summary_lines)

    human_text = (
        f"Current phase: {session.phase}\n"
        f"Current emotion profile: {session.current_version.emotion_profile or '(empty)'}\n"
        f"Current artistic profile: {session.current_version.artistic_profile or '(empty)'}\n"
        f"Current sketch exists: {'yes' if bool(session.current_version.code) else 'no'}\n"
        f"{pending_summary}\n"
        f"Assistant's previous message: {last_assistant}\n"
        f"User's latest message: {message}"
    )

    try:
        response = intent_llm.invoke(
            [
                SystemMessage(content=INTENT_CLASSIFIER_PROMPT),
                HumanMessage(content=human_text),
            ]
        )
        response_text = response.content if isinstance(response.content, str) else json.dumps(response.content)
        parsed = _extract_json_object(response_text)
        if not parsed:
            return defaults
        return _sanitize_turn_action_payload(parsed)
    except Exception:
        return defaults


def _detect_turn_actions(
    session: "SessionState",
    message: str,
    pending: Optional["PendingArtisticDecision"],
) -> dict:
    classified = _classify_turn_actions_with_llm(session, message, pending)
    canonical_decision = _parse_canonical_art_direction_command(message) if pending else None

    actions = {
        "code_request": classified["code_request"],
        "emotion_refinement": classified["emotion_refinement"],
        "confirm_start_coding": classified["confirm_start_coding"],
        "sketch_rejection": classified["sketch_rejection"],
        "art_direction_decision": classified["art_direction_decision"],
        "source": "llm" if message and LLM_ENABLED and intent_llm is not None else "none",
        "confidence": classified["confidence"],
    }

    # Preserve exact UI button commands deterministically without trying to
    # heuristically interpret open-ended language.
    if canonical_decision:
        actions["art_direction_decision"] = canonical_decision
        actions["source"] = "ui_command"
        actions["confidence"] = "high"
    return actions


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
            "should_create_version": False,
            "offers_artistic_alternatives": False,
            "artistic_options": [],
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
        "should_create_version": False,
        "offers_artistic_alternatives": False,
        "artistic_options": [],
    }


def _salvage_non_json_llm_reply(
    phase: str,
    version: "VersionNode",
    raw_response_text: Any,
    *,
    user_text: str = "",
    intent_hint: str = "",
) -> dict:
    fallback = _default_mock_reply(phase, version)
    raw = _clean_text(raw_response_text)
    if not raw:
        return fallback

    # Preserve any natural-language guidance the model produced, but do not
    # auto-apply code or art changes when the schema was malformed.
    text_without_code = re.sub(r"```[\s\S]*?```", "", raw).strip()
    if text_without_code and len(text_without_code) >= 20:
        fallback["message"] = text_without_code
    elif phase == "code_generation" and intent_hint == "sketch_rejection":
        fallback["message"] = (
            "I hear the current sketch is off. Tell me what feels most wrong about it right now: "
            "the motion, density, shape, color, or overall mood. I may still miss your intent, "
            "so please correct me anytime and we can revise it together."
        )
    return fallback


def _transcribe_audio(audio_b64: str, audio_mime: str = "audio/webm") -> str:
    """Send audio to the LLM and return a plain-text transcription."""
    if not LLM_ENABLED or intent_llm is None:
        return ""
    try:
        audio_url = audio_b64 if audio_b64.startswith("data:") else f"data:{audio_mime};base64,{audio_b64}"
        msg = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": audio_url}},
            {"type": "text", "text": "Please transcribe exactly what the user said in this audio clip. Return only the transcribed text, no commentary."},
        ])
        response = intent_llm.invoke([msg])
        text = response.content if isinstance(response.content, str) else ""
        return text.strip()
    except Exception:
        return ""


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


# ─── Artistic quality reference seed ──────────────────────────────────────────
REFERENCE_IMAGES_DIR = BASE_DIR / "reference_images"


def _load_ref_image(filename: str) -> Optional[str]:
    path = REFERENCE_IMAGES_DIR / filename
    if not path.exists():
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _build_seed_reference_message() -> Optional[HumanMessage]:
    head_in_clouds = _load_ref_image("score3_head_in_clouds.png")
    sadness = _load_ref_image("score3_sadness.png")
    parts: list[dict] = [
        {"type": "text", "text": (
            "ARTISTIC QUALITY REFERENCE — score 3-4 examples "
            "(these are dynamic animated sketches; code and screenshots below are static "
            "representations). Score 7 is the target."
        )},
        {"type": "text", "text": f'"first date excitement" — score 3-4\nCODE (Processing/Java):\n{FIRST_DATE_CODE}'},
    ]
    if head_in_clouds:
        parts.append({"type": "image_url", "image_url": {"url": head_in_clouds}})
        parts.append({"type": "text", "text": '"head in the clouds" — score 3-4'})
    if sadness:
        parts.append({"type": "image_url", "image_url": {"url": sadness}})
        parts.append({"type": "text", "text": '"sadness" — score 3-4'})
    return HumanMessage(content=parts)


SEED_REFERENCE_MESSAGE: Optional[HumanMessage] = _build_seed_reference_message()


def _inject_seed_reference(session: SessionState) -> None:
    if SEED_REFERENCE_MESSAGE is None:
        return
    session.llm_events.insert(0, {
        "id": "seed_reference",
        "created_at": _now(),
        "anchor_version_id": None,
        "message": SEED_REFERENCE_MESSAGE,
    })


def get_or_create_session(session_id: Optional[str] = None) -> SessionState:
    global latest_session_id
    if session_id and session_id in sessions:
        latest_session_id = session_id
        return sessions[session_id]
    if session_id and session_id not in sessions:
        session = SessionState(session_id=session_id)
        _inject_seed_reference(session)
        sessions[session_id] = session
        latest_session_id = session_id
        return session
    if latest_session_id and latest_session_id in sessions:
        return sessions[latest_session_id]
    sid = str(uuid.uuid4())
    session = SessionState(session_id=sid)
    _inject_seed_reference(session)
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
intent_llm: Optional[ChatGoogleGenerativeAI] = None

if LLM_ENABLED:
    phase_1_llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.8)
    phase_2_llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.45)
    intent_llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


def _invoke_llm(
    session: SessionState,
    user_text: str,
    image: Optional[str],
    image_mime: Optional[str],
    audio: Optional[str],
    audio_mime: Optional[str],
    intent_hint: str = "",
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
            parsed = _salvage_non_json_llm_reply(
                session.phase,
                session.current_version,
                response_text,
                user_text=user_text,
                intent_hint=intent_hint,
            )
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
  "message": "<empathetic short reply + brief interpretation + natural presentation of the options + invitation to choose or describe their own vision>",
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
- Present the options naturally in the `message`, so the user can understand what each option means.
- Invite the user to choose one option, ask for more options, or describe their own art direction in their own words.
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


def _repair_incomplete_p5_code(
    session: SessionState,
    current: "VersionNode",
    *,
    broken_code: str,
    emotion_profile: str,
    artistic_profile: str,
) -> Optional[str]:
    if not LLM_ENABLED or phase_2_llm is None:
        return None

    repair_prompt = f"""
You are repairing an incomplete or invalid p5.js sketch response.

Current emotion profile: {emotion_profile or current.emotion_profile or '(empty)'}
Current artistic profile: {artistic_profile or current.artistic_profile or '(empty)'}

Broken code:
```javascript
{broken_code or ''}
```

Return ONLY complete runnable p5.js code.

Requirements:
- Include setup() and draw().
- Preserve the intended visual idea if possible.
- If the broken code is too incomplete to preserve, create a minimal but valid starter sketch that matches the profiles.
- No markdown explanation. No JSON. Just code.
""".strip()

    try:
        response = phase_2_llm.invoke(
            [
                SystemMessage(content="You repair incomplete JavaScript sketches into complete runnable p5.js."),
                HumanMessage(content=repair_prompt),
            ]
        )
        repaired_text = response.content if isinstance(response.content, str) else json.dumps(response.content)
        repaired_code = _extract_code_text(repaired_text)
        ok, _ = _is_probably_complete_p5_code(repaired_code)
        return repaired_code if ok else None
    except Exception:
        return None


def _sanitize_llm_payload(raw: dict, current: VersionNode) -> dict:
    message = _clean_text(raw.get("message")) or "I updated the current sketch state."
    commit_message = _clean_text(raw.get("commit_message"))
    emotion_profile = _ensure_emotion_prefix(_clean_text(raw.get("emotion_profile"))) or current.emotion_profile
    artistic_profile = _clean_text(raw.get("artistic_profile")) or current.artistic_profile
    code = _clean_text(raw.get("code")) or current.code
    emotion_confidence = _normalize_confidence(raw.get("emotion_confidence") or current.emotion_confidence)
    emotion_gaps = _normalize_gaps(raw.get("emotion_gaps"))
    if "should_create_version" in raw:
        should_create_version = _coerce_bool(raw.get("should_create_version"))
    else:
        should_create_version = artistic_profile != current.artistic_profile or code != current.code
    artistic_options = _normalize_artistic_options(raw.get("artistic_options"))
    offers_artistic_alternatives = _coerce_bool(raw.get("offers_artistic_alternatives")) or bool(artistic_options)
    if offers_artistic_alternatives and not artistic_options:
        artistic_options = _default_artistic_alternatives(artistic_profile or current.artistic_profile)
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
        "should_create_version": should_create_version,
        "offers_artistic_alternatives": offers_artistic_alternatives,
        "artistic_options": artistic_options[:3],
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

    # Build effective message from all inputs: typed text + audio transcript + image hint.
    parts = []
    if message:
        parts.append(message)
    if audio:
        transcript = _transcribe_audio(audio, audio_mime)
        if transcript:
            parts.append(transcript)
    if image:
        parts.append("I also shared an image.")
    effective_message = " ".join(parts)

    pending = session.pending_artistic_decision
    actions = (
        _detect_turn_actions(session, effective_message, pending)
        if effective_message
        else {
            "code_request": False,
            "emotion_refinement": False,
            "confirm_start_coding": False,
            "sketch_rejection": False,
            "art_direction_decision": None,
            "source": "none",
            "confidence": "low",
        }
    )
    decision = actions["art_direction_decision"]

    # Handle explicit artistic-decision actions first when a proposal is pending.
    if pending and decision:
        current = session.current_version
        if decision == "modify":
            pending.awaiting_modify_details = True
            assistant_text = (
                "Thanks for steering this. Tell me what you want to change, or just describe the art direction "
                "you already have in mind, and I will revise the proposal before creating any version."
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
                    "should_create_version": False,
                    "offers_artistic_alternatives": False,
                    "artistic_options": [],
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
            assistant_text = options_payload["message"]
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
                    "should_create_version": False,
                    "offers_artistic_alternatives": True,
                    "artistic_options": list(pending.alternatives),
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
            intent_hint="",
        )
        sanitized = _sanitize_llm_payload(parsed_raw, current)
        new_emotion = sanitized["emotion_profile"] or pending.proposed_emotion_profile or current.emotion_profile
        new_artistic = sanitized["artistic_profile"] or selected_artistic_profile or current.artistic_profile
        new_code = sanitized["code"] or pending.proposed_code or current.code
        new_conf = sanitized["emotion_confidence"] or pending.proposed_emotion_confidence or current.emotion_confidence
        new_gaps = sanitized["emotion_gaps"] or pending.proposed_emotion_gaps or current.emotion_gaps
        changed = _has_material_implementation_change(current, new_artistic, new_code)
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
            _update_current_emotion_state(
                current,
                emotion_profile=new_emotion,
                emotion_confidence=new_conf,
                emotion_gaps=new_gaps,
            )
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
        assistant_text = options_payload["message"]
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
                "should_create_version": False,
                "offers_artistic_alternatives": True,
                "artistic_options": list(pending.alternatives),
            },
            raw_response_text=raw_options_text,
            created_version=None,
        )

    user_text_for_model = effective_message

    if session.phase == "code_generation" and actions["emotion_refinement"] and not actions["sketch_rejection"]:
        session.phase = "emotional_discovery"
    elif session.phase == "emotional_discovery" and session.current_version.emotion_profile:
        if actions["confirm_start_coding"]:
            session.phase = "code_generation"
            user_text_for_model = _build_code_start_confirmation_prompt(session, effective_message)
        elif actions["code_request"]:
            session.phase = "code_generation"

    if not user_text_for_model:
        user_text_for_model = "Please infer emotional cues from my input and update the profiles."
    elif session.phase == "code_generation" and actions["sketch_rejection"]:
        user_text_for_model = _build_sketch_rejection_prompt(session, effective_message)

    parsed_raw, raw_response_text, human_msg, assistant_llm_msg = _invoke_llm(
        session=session,
        user_text=user_text_for_model,
        image=image,
        image_mime=image_mime,
        audio=audio,
        audio_mime=audio_mime,
        intent_hint="sketch_rejection" if actions["sketch_rejection"] else "",
    )
    sanitized = _sanitize_llm_payload(parsed_raw, session.current_version)

    old = session.current_version
    new_emotion = sanitized["emotion_profile"]
    new_artistic = sanitized["artistic_profile"]
    new_code = sanitized["code"]
    new_conf = sanitized["emotion_confidence"]
    new_gaps = sanitized["emotion_gaps"]
    llm_should_create_version = sanitized["should_create_version"]
    offers_artistic_alternatives = sanitized["offers_artistic_alternatives"]
    artistic_options = sanitized["artistic_options"]

    if session.phase == "code_generation" and new_code and new_code != old.code:
        code_ok, code_issue = _is_probably_complete_p5_code(new_code)
        if not code_ok:
            repaired_code = _repair_incomplete_p5_code(
                session,
                old,
                broken_code=new_code,
                emotion_profile=new_emotion,
                artistic_profile=new_artistic,
            )
            if repaired_code:
                new_code = repaired_code
                sanitized["code"] = repaired_code
            else:
                llm_should_create_version = False
                sanitized["should_create_version"] = False
                new_code = old.code
                sanitized["code"] = old.code
                sanitized["message"] = (
                    f"{sanitized['message']}\n\n"
                    "I held off on applying code because the generated sketch came back incomplete. "
                    f"Reason: {code_issue} Ask me to try generating the sketch again and I will retry."
                )

    should_stage_artistic_decision = (
        session.phase == "code_generation"
        and (
            offers_artistic_alternatives
            or (_clean_text(new_artistic) and _has_artistic_change(old.artistic_profile, new_artistic))
        )
    )

    if should_stage_artistic_decision:
        old.emotion_profile = new_emotion
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
            alternatives=artistic_options,
            awaiting_modify_details=False,
        )
        staged_message = sanitized["message"]
        _append_chat(session, "assistant", "text", staged_message)
        _append_llm_event(session, human_msg, old.id)
        _append_llm_event(session, assistant_llm_msg, old.id)
        staged_reply = {
            **sanitized,
            "message": staged_message,
            "code": old.code,
        }
        return _chat_response(staged_reply, raw_response_text, None)

    emotion_profile_change = _has_emotion_profile_change(old.emotion_profile, new_emotion)
    emotional_state_change = _has_emotional_state_change(old, new_emotion, new_conf, new_gaps)
    implementation_version_change = _has_material_implementation_change(old, new_artistic, new_code)

    if session.phase == "emotional_discovery":
        should_create_version = emotional_state_change
    else:
        should_create_version = bool(llm_should_create_version) and implementation_version_change

    if not should_create_version:
        session.pending_artistic_decision = None
        _update_current_emotion_state(
            old,
            emotion_profile=new_emotion,
            emotion_confidence=new_conf,
            emotion_gaps=new_gaps,
        )
        _append_llm_event(session, human_msg, old.id)
        _append_llm_event(session, assistant_llm_msg, old.id)
        _append_chat(session, "assistant", "text", sanitized["message"])
        advisory_reply = {
            **sanitized,
            "artistic_profile": old.artistic_profile,
            "code": old.code,
        }
        return _chat_response(advisory_reply, raw_response_text, None)

    session.pending_artistic_decision = None

    created_version = None
    if should_create_version:
        if session.phase == "emotional_discovery" and not emotion_profile_change:
            fallback_summary = "Refined emotional understanding details"
        else:
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
        _update_current_emotion_state(
            old,
            emotion_profile=new_emotion,
            emotion_confidence=new_conf,
            emotion_gaps=new_gaps,
        )

    # Anchor the LLM turn to the resulting version node, so branch restores
    # only replay context from the lineage the user is currently on.
    turn_anchor_id = created_version.id if created_version else old.id
    _append_llm_event(session, human_msg, turn_anchor_id)
    _append_llm_event(session, assistant_llm_msg, turn_anchor_id)

    if session.phase == "emotional_discovery" and actions["code_request"] and sanitized["emotion_profile"]:
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
