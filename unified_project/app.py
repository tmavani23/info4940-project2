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

PHASE_EMOTION = "emotional_discovery"
PHASE_ARTISTIC = "artistic_discovery"
PHASE_CODE = "code_generation"
PHASE_ORDER = [PHASE_EMOTION, PHASE_ARTISTIC, PHASE_CODE]
ARTISTIC_PREFERENCE_GROUPS = (
    ("detail", "Detail"),
    ("motion", "Motion"),
    ("palette", "Palette"),
    ("shapes", "Shape language"),
    ("composition", "Composition"),
    ("texture", "Texture"),
    ("atmosphere", "Atmosphere"),
)
ARTISTIC_PREFERENCE_KEYS = tuple(key for key, _ in ARTISTIC_PREFERENCE_GROUPS)
ARTISTIC_PREFERENCE_LABELS = {key: label for key, label in ARTISTIC_PREFERENCE_GROUPS}


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
- Remind them they can move to artistic discovery stage when ready.
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
  "artistic_confidence": "",
  "code": ""
}

Constraints:
- emotion_profile must start with "The emotion is".
- emotion_confidence must be exactly low, medium, or high.
- emotion_gaps must be an array (possibly empty).
- commit_message should be one concise line describing discovery progress.
- In your `message`, do not repeat your emotional understanding - that should be put into `emotion_profile`. Simply remind the user to check the emotional profiles pane for the current understanding.
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
  "artistic_confidence": "<low|medium|high>",
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


def build_system_prompt(phase: str, state_summary: str, collab_style: str = "iterate") -> str:
    phase_prompt = DISCOVERY_PROMPT if phase == "emotional_discovery" else IMPLEMENTATION_PROMPT.replace(
        "{ARTISTIC_QUALITY_REFERENCE}", ARTISTIC_QUALITY_REFERENCE
    )
    if collab_style == "confirm":
        collab_note = (
            "COLLABORATION STYLE: This student wants to work through all three stages — emotion, artistic "
            "direction, then code — in order. Spend time fully exploring and refining the emotional profile, "
            "then develop a complete artistic direction, before generating any code. "
            "Do not generate or update code until the student explicitly asks to move into implementation. "
            "If the student asks to skip ahead or switch modes at any point, respect that immediately."
        )
    else:
        collab_note = (
            "COLLABORATION STYLE: This student wants to go from emotion straight to code, skipping the "
            "artistic stage for now. As soon as there is enough emotional context to make a meaningful sketch, "
            "generate code and show results — do not wait for a fully developed artistic direction. "
            "The student can always revisit and tweak the artistic direction later."
        )
    return f"{BASE_PROMPT}\n\n{collab_note}\n\nCurrent state:\n{state_summary}\n\n{phase_prompt}"


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


def _confidence_allows_advance(value: Any) -> bool:
    return _normalize_confidence(value) in {"medium", "high"}


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


def _update_current_artistic_state(
    version: "VersionNode",
    *,
    artistic_profile: str,
    artistic_confidence: str,
) -> None:
    version.artistic_profile = artistic_profile
    version.artistic_confidence = artistic_confidence


def _empty_artistic_panel_selections() -> dict[str, str]:
    return {key: "" for key in ARTISTIC_PREFERENCE_KEYS}


@dataclass
class ArtisticPanelState:
    selections: dict[str, str] = field(default_factory=_empty_artistic_panel_selections)
    note: str = ""


def _normalize_artistic_panel_state(value: Any) -> "ArtisticPanelState":
    selections = _empty_artistic_panel_selections()
    note = ""
    if isinstance(value, ArtisticPanelState):
        raw_selections = value.selections
        note = _clean_text(value.note)
    elif isinstance(value, dict):
        raw_selections = value.get("selections")
        note = _clean_text(value.get("note"))
    else:
        raw_selections = None
    if isinstance(raw_selections, dict):
        for key in ARTISTIC_PREFERENCE_KEYS:
            selections[key] = _clean_text(raw_selections.get(key))
    return ArtisticPanelState(selections=selections, note=note)


def _serialize_artistic_panel_state(value: Optional["ArtisticPanelState"]) -> dict[str, Any]:
    state = _normalize_artistic_panel_state(value)
    return {
        "selections": dict(state.selections),
        "note": state.note,
    }


def _summarize_artistic_panel_state(value: Optional["ArtisticPanelState"]) -> tuple[list[str], str]:
    state = _normalize_artistic_panel_state(value)
    selections = []
    for key in ARTISTIC_PREFERENCE_KEYS:
        selected = _clean_text(state.selections.get(key))
        if selected:
            selections.append(f"{ARTISTIC_PREFERENCE_LABELS[key]}: {selected}")
    return selections, state.note


def _build_artistic_profile_from_panel_state(value: Optional["ArtisticPanelState"]) -> str:
    selections, note = _summarize_artistic_panel_state(value)
    if not selections and not note:
        return ""
    parts = []
    if selections:
        parts.append(f"Use this visual direction: {'; '.join(selections)}.")
    if note:
        parts.append(f"User note: {note}.")
    parts.append("Implement this direction directly instead of generating comparison options.")
    return " ".join(parts)


def _infer_artistic_confidence_from_panel_state(value: Optional["ArtisticPanelState"]) -> str:
    selections, note = _summarize_artistic_panel_state(value)
    if len(selections) >= 3 or (len(selections) >= 2 and note):
        return "high"
    if selections or note:
        return "medium"
    return ""


def _save_artistic_panel_state(session: "SessionState", value: Any) -> ArtisticPanelState:
    session.artistic_panel_state = _normalize_artistic_panel_state(value)
    return session.artistic_panel_state


def _apply_direct_artistic_profile(
    session: "SessionState",
    *,
    artistic_profile: str,
    artistic_confidence: str,
    summary: str,
    source: str = "user",
) -> bool:
    profile = _clean_text(artistic_profile)
    confidence = _normalize_confidence(artistic_confidence or "")
    if not profile:
        return False
    current = session.current_version
    if _has_artistic_change(current.artistic_profile, profile) or current.artistic_confidence != confidence:
        create_version(
            session,
            emotion_profile=current.emotion_profile,
            artistic_profile=profile,
            artistic_confidence=confidence,
            code=current.code,
            emotion_confidence=current.emotion_confidence,
            emotion_gaps=current.emotion_gaps,
            summary=summary,
            source=source,
        )
    else:
        _update_current_artistic_state(
            current,
            artistic_profile=profile,
            artistic_confidence=confidence,
        )
    if _confidence_allows_advance(confidence):
        _unlock_phase(session, PHASE_CODE)
    return True


def _apply_saved_artistic_panel_state_for_coding(
    session: "SessionState",
    *,
    summary: str,
    source: str = "user",
) -> bool:
    artistic_profile = _build_artistic_profile_from_panel_state(session.artistic_panel_state)
    artistic_confidence = _infer_artistic_confidence_from_panel_state(session.artistic_panel_state)
    if not artistic_profile or not _confidence_allows_advance(artistic_confidence):
        return False
    return _apply_direct_artistic_profile(
        session,
        artistic_profile=artistic_profile,
        artistic_confidence=artistic_confidence,
        summary=summary,
        source=source,
    )


def _get_pending_artistic_transition_payload(
    pending: Optional["PendingArtisticDecision"],
) -> tuple[str, str]:
    if not pending:
        return "", ""
    artistic_profile = _clean_text(pending.proposed_artistic_profile)
    artistic_confidence = _normalize_confidence(pending.proposed_artistic_confidence or "")
    if not artistic_profile or not _confidence_allows_advance(artistic_confidence):
        return "", ""
    return artistic_profile, artistic_confidence


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
    current = session.current_version
    prompt_lines = [
        f'The user confirmed they want to start coding now. Their reply was: "{reply}".',
        "Use the current artistic profile as the source of truth for the implementation.",
        f"Current emotion profile: {current.emotion_profile or '(empty)'}",
        f"Current artistic profile: {current.artistic_profile or '(empty)'}",
        f"Current sketch exists: {'yes' if bool(current.code) else 'no'}",
        "Apply code changes now instead of brainstorming. Return runnable p5.js and set should_create_version to true.",
    ]
    if current.code:
        prompt_lines.append(
            "Revise the existing sketch so it matches the latest artistic direction, and replace earlier motion, color, composition, or texture choices when they conflict with the updated profile."
        )
    else:
        prompt_lines.append(
            "Create the first runnable p5.js sketch based on the current emotional understanding and artistic direction."
        )
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
        return f"I restored this earlier version: \"{detail}\"."
    return "I restored this earlier version."


def _infer_phase_from_version(version: "VersionNode") -> str:
    if _clean_text(version.code):
        return PHASE_CODE
    if _clean_text(version.artistic_profile):
        return PHASE_ARTISTIC
    return PHASE_EMOTION


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
        f"Artistic confidence: {version.artistic_confidence or '(empty)'}\n"
        f"Code first line: {code_first or '(empty)'}"
    )


def _default_mock_reply(phase: str, version: "VersionNode") -> dict:
    emotion = version.emotion_profile or "The emotion is still unfolding and may need more context."
    if phase == PHASE_EMOTION:
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
            "artistic_confidence": version.artistic_confidence or "",
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
        "artistic_confidence": version.artistic_confidence or "medium",
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
    elif phase == PHASE_CODE and intent_hint == "sketch_rejection":
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
    artistic_confidence: str
    code: str
    emotion_confidence: str
    emotion_gaps: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=_now)


@dataclass
class PendingArtisticDecision:
    proposed_artistic_profile: str
    proposed_artistic_confidence: str
    proposed_code: str
    proposed_emotion_profile: str
    proposed_emotion_confidence: str
    alternative_preview_codes: list[str] = field(default_factory=list)
    proposed_emotion_gaps: list[str] = field(default_factory=list)
    proposed_commit_message: str = ""
    alternatives: list[str] = field(default_factory=list)
    awaiting_modify_details: bool = False
    created_at: float = field(default_factory=_now)


@dataclass
class SessionState:
    session_id: str
    phase: str = PHASE_EMOTION
    messages: list[ChatEntry] = field(default_factory=list)
    llm_events: list[dict[str, Any]] = field(default_factory=list)
    versions: dict[str, VersionNode] = field(default_factory=dict)
    version_order: list[str] = field(default_factory=list)
    branch_heads: dict[str, str] = field(default_factory=dict)
    active_branch: str = "main"
    current_version_id: Optional[str] = None
    branch_counter: int = 1
    pending_artistic_decision: Optional[PendingArtisticDecision] = None
    artistic_panel_state: ArtisticPanelState = field(default_factory=ArtisticPanelState)
    unlocked_phases: list[str] = field(default_factory=lambda: [PHASE_EMOTION])
    visited_phases: list[str] = field(default_factory=lambda: [PHASE_EMOTION])

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
            artistic_confidence="",
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


def _phase_index(phase: str) -> int:
    try:
        return PHASE_ORDER.index(phase)
    except ValueError:
        return 0


def _unlock_phase(session: SessionState, phase: str) -> None:
    phase_idx = _phase_index(phase)
    for unlocked in PHASE_ORDER[: phase_idx + 1]:
        if unlocked not in session.unlocked_phases:
            session.unlocked_phases.append(unlocked)


def _mark_phase_visited(session: SessionState, phase: str) -> bool:
    _unlock_phase(session, phase)
    first_visit = phase not in session.visited_phases
    if first_visit:
        session.visited_phases.append(phase)
    return first_visit


def _can_unlock_phase(session: SessionState, phase: str) -> bool:
    current = session.current_version
    if phase == PHASE_EMOTION:
        return True
    if phase == PHASE_ARTISTIC:
        return bool(_clean_text(current.emotion_profile)) and _confidence_allows_advance(current.emotion_confidence)
    if phase == PHASE_CODE:
        if _clean_text(current.code):
            return True
        return bool(_clean_text(current.artistic_profile)) and _confidence_allows_advance(current.artistic_confidence)
    return False


def _phase_locked_message(target_phase: str) -> str:
    if target_phase == PHASE_ARTISTIC:
        return (
            "Keep refining the emotion a bit more first. Artistic Discovery unlocks once the emotion confidence reaches medium or high."
        )
    if target_phase == PHASE_CODE:
        return (
            "Choose and save an artistic direction first. Coding unlocks once the artistic confidence reaches medium or high."
        )
    return "That phase is not ready yet."


def _phase_transition_message(phase: str, *, first_visit: bool, current_version: VersionNode) -> str:
    if phase == PHASE_EMOTION:
        return (
            "Back in **Emotion Discovery**. We can revisit the memory, setting, or feeling itself before changing the visuals."
        )
    if phase == PHASE_ARTISTIC:
        intro = "Welcome to **Artistic Discovery**." if first_visit else "Back in **Artistic Discovery**."
        return (
            f"{intro}\n\n"
            "Tell me about the visual direction you want and I will turn it into three artwork options with previews.\n"
            "- Abstract vs detailed\n"
            "- Still vs gentle motion vs dynamic motion\n"
            "- Warm, cool, monochrome, or high-contrast color\n"
            "- Organic vs geometric shapes\n"
            "- Minimal vs dense composition\n"
            "- Smooth, grainy, or glowing texture\n\n"
            "You can use the preference chips, type your own note, or go back whenever you want."
        )
    if phase == PHASE_CODE:
        if _clean_text(current_version.artistic_profile):
            return (
                "You are in **Coding** now. I can implement the chosen artistic direction in p5.js, and you can still jump back to Artistic for a bigger visual rethink."
            )
        return (
            "You are in **Coding** now. This version does not have a saved artistic direction yet, so if you want a more guided implementation, hop back to Artistic first."
        )
    return ""


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


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _build_artistic_option_preview_code(option_text: str, emotion_profile: str = "") -> str:
    text = _clean_text(option_text).lower()
    emotion_hint = _clean_text(emotion_profile)
    seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(text or emotion_hint or "art")) % 997

    palette = [(109, 133, 255), (26, 39, 69), (10, 17, 32)]
    background = (12, 16, 26)
    if _contains_any(text, ["warm", "ember", "sunset", "gold", "orange", "red"]):
        palette = [(255, 181, 107), (176, 75, 88), (53, 23, 31)]
        background = (28, 13, 18)
    elif _contains_any(text, ["monochrome", "black", "white", "stark", "charcoal", "grayscale"]):
        palette = [(238, 242, 248), (108, 117, 128), (21, 26, 32)]
        background = (11, 13, 18)
    elif _contains_any(text, ["dream", "glow", "luminous", "neon", "ethereal"]):
        palette = [(159, 232, 255), (108, 92, 255), (19, 22, 47)]
        background = (10, 11, 28)
    elif _contains_any(text, ["green", "teal", "sea", "ocean"]):
        palette = [(126, 240, 202), (31, 91, 99), (7, 29, 36)]
        background = (7, 16, 22)

    if _contains_any(text, ["still", "static", "quiet", "frozen", "pause"]):
        motion_speed = 0.0018
        drift_x = 6
        drift_y = 4
    elif _contains_any(text, ["dynamic", "frantic", "swirl", "burst", "fast", "pulse", "chaotic"]):
        motion_speed = 0.024
        drift_x = 34
        drift_y = 26
    else:
        motion_speed = 0.009
        drift_x = 16
        drift_y = 12

    if _contains_any(text, ["minimal", "sparse", "negative space", "open"]):
        particle_count = 9
        min_size = 18
        max_size = 38
    elif _contains_any(text, ["dense", "crowded", "cluster", "packed", "storm"]):
        particle_count = 30
        min_size = 10
        max_size = 28
    else:
        particle_count = 18
        min_size = 14
        max_size = 32

    geometric = _contains_any(text, ["geometric", "grid", "sharp", "angular", "rect", "line"])
    grainy = _contains_any(text, ["grain", "textured", "dust", "rough"])
    glowing = _contains_any(text, ["glow", "mist", "haze", "soft light", "luminous", "dream"])

    return f"""// Artistic option preview
const PREVIEW_COLORS = {json.dumps(palette)};
const PREVIEW_BG = {json.dumps(background)};
const PARTICLE_COUNT = {particle_count};
const MOTION_SPEED = {motion_speed};
const DRIFT_X = {drift_x};
const DRIFT_Y = {drift_y};
const MIN_SIZE = {min_size};
const MAX_SIZE = {max_size};
const USE_GEOMETRY = {str(geometric).lower()};
const USE_GRAIN = {str(grainy).lower()};
const USE_GLOW = {str(glowing).lower()};
const SEED = {seed};

let nodes = [];

function setup() {{
  createCanvas(windowWidth || 320, windowHeight || 180);
  noStroke();
  rectMode(CENTER);
  for (let i = 0; i < PARTICLE_COUNT; i += 1) {{
    nodes.push({{
      baseX: map(((i * 29) + SEED) % 100, 0, 99, width * 0.18, width * 0.82),
      baseY: map(((i * 47) + (SEED * 3)) % 100, 0, 99, height * 0.18, height * 0.82),
      size: map(((i * 17) + SEED) % 100, 0, 99, MIN_SIZE, MAX_SIZE),
      phase: i * 0.63,
      drift: map(((i * 13) + SEED) % 100, 0, 99, 0.65, 1.4),
      colorIndex: i % PREVIEW_COLORS.length
    }});
  }}
}}

function drawGlow(x, y, size, col) {{
  fill(col[0], col[1], col[2], 26);
  ellipse(x, y, size * 1.9, size * 1.9);
}}

function drawShape(x, y, size, col) {{
  fill(col[0], col[1], col[2], 180);
  if (USE_GEOMETRY) {{
    push();
    translate(x, y);
    rotate(sin(frameCount * MOTION_SPEED + size) * 0.4);
    rect(0, 0, size, size * 0.76, size * 0.2);
    pop();
    return;
  }}
  ellipse(x, y, size, size * 0.82);
}}

function draw() {{
  background(PREVIEW_BG[0], PREVIEW_BG[1], PREVIEW_BG[2]);
  for (let i = 0; i < nodes.length; i += 1) {{
    const node = nodes[i];
    const t = frameCount * MOTION_SPEED * node.drift + node.phase;
    const x = node.baseX + sin(t * 1.4) * DRIFT_X + cos(t * 0.55) * (DRIFT_X * 0.35);
    const y = node.baseY + cos(t * 1.1) * DRIFT_Y;
    const col = PREVIEW_COLORS[node.colorIndex];
    if (USE_GLOW) {{
      drawGlow(x, y, node.size, col);
    }}
    drawShape(x, y, node.size, col);
  }}

  if (USE_GRAIN) {{
    stroke(255, 255, 255, 18);
    for (let i = 0; i < 48; i += 1) {{
      point((i * 37 + SEED) % width, (i * 23 + frameCount + SEED) % height);
    }}
    noStroke();
  }}
}}

function windowResized() {{
  resizeCanvas(windowWidth || 320, windowHeight || 180);
}}
""".strip()


def _build_artistic_option_preview_codes(options: list[str], emotion_profile: str = "") -> list[str]:
    return [_build_artistic_option_preview_code(option, emotion_profile) for option in options[:3]]


def _build_local_p5_code_from_direction(artistic_profile: str, emotion_profile: str = "") -> str:
    text = f"{_clean_text(artistic_profile)} {_clean_text(emotion_profile)}".lower()

    palette = [(109, 133, 255), (26, 39, 69), (10, 17, 32)]
    background = (12, 16, 26)
    if _contains_any(text, ["warm", "ember", "sunset", "gold", "orange", "red"]):
        palette = [(255, 181, 107), (176, 75, 88), (53, 23, 31)]
        background = (28, 13, 18)
    elif _contains_any(text, ["monochrome", "black", "white", "stark", "charcoal", "grayscale"]):
        palette = [(238, 242, 248), (108, 117, 128), (21, 26, 32)]
        background = (11, 13, 18)
    elif _contains_any(text, ["dream", "glow", "luminous", "neon", "ethereal"]):
        palette = [(159, 232, 255), (108, 92, 255), (19, 22, 47)]
        background = (10, 11, 28)
    elif _contains_any(text, ["green", "teal", "sea", "ocean"]):
        palette = [(126, 240, 202), (31, 91, 99), (7, 29, 36)]
        background = (7, 16, 22)

    motion_mode = "gentle"
    motion_speed = 0.006
    drift_x = 18
    drift_y = 12
    rotation_amount = 0.1
    if _contains_any(text, ["still", "static", "quiet", "frozen", "pause"]):
        motion_mode = "still"
        motion_speed = 0.0015
        drift_x = 4
        drift_y = 3
        rotation_amount = 0.02
    elif _contains_any(text, ["dynamic", "frantic", "swirl", "burst", "fast", "pulse", "chaotic"]):
        motion_mode = "dynamic"
        motion_speed = 0.02
        drift_x = 42
        drift_y = 28
        rotation_amount = 0.5

    composition_mode = "layered"
    node_count = 18
    min_size = 14
    max_size = 32
    if _contains_any(text, ["minimal", "sparse", "negative space", "open"]):
        composition_mode = "minimal"
        node_count = 9
        min_size = 18
        max_size = 38
    elif _contains_any(text, ["dense", "crowded", "cluster", "packed", "storm"]):
        composition_mode = "dense"
        node_count = 30
        min_size = 10
        max_size = 26

    shape_mode = "organic"
    if _contains_any(text, ["geometric", "grid", "sharp", "angular", "rect", "line"]):
        shape_mode = "geometric"
    elif _contains_any(text, ["mixed", "mixed forms"]):
        shape_mode = "mixed"

    texture_mode = "smooth"
    if _contains_any(text, ["grain", "textured", "dust", "rough"]):
        texture_mode = "grainy"
    elif _contains_any(text, ["glow", "mist", "haze", "soft light", "luminous"]):
        texture_mode = "glowing"

    high_contrast = _contains_any(text, ["high contrast", "stark"])
    if high_contrast:
        background = (6, 8, 12)
        palette = [(245, 247, 255), (132, 158, 255), (12, 16, 28)]

    atmosphere = "calm"
    if _contains_any(text, ["tense", "uneasy", "anxious"]):
        atmosphere = "tense"
    elif _contains_any(text, ["dream", "ethereal", "dreamlike"]):
        atmosphere = "dreamlike"
    elif _contains_any(text, ["dramatic", "theatrical"]):
        atmosphere = "dramatic"

    seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(text or "art")) % 997
    emotion_label = _clean_text(emotion_profile) or "The emotion is still unfolding."
    artistic_label = _clean_text(artistic_profile) or "Use a gentle abstract composition."

    return f"""// Local fallback sketch generated from the current artistic direction.
// Tweak the parameters below if you want to adjust the motion, density, or palette.
const EMOTION_PROFILE = {json.dumps(emotion_label)};
const ARTISTIC_PROFILE = {json.dumps(artistic_label)};
const PALETTE = {json.dumps(palette)};
const BACKGROUND_COLOR = {json.dumps(background)};
const NODE_COUNT = {node_count};
const MIN_SIZE = {min_size};
const MAX_SIZE = {max_size};
const MOTION_MODE = {json.dumps(motion_mode)};
const MOTION_SPEED = {motion_speed};
const DRIFT_X = {drift_x};
const DRIFT_Y = {drift_y};
const ROTATION_AMOUNT = {rotation_amount};
const SHAPE_MODE = {json.dumps(shape_mode)};
const TEXTURE_MODE = {json.dumps(texture_mode)};
const COMPOSITION_MODE = {json.dumps(composition_mode)};
const ATMOSPHERE = {json.dumps(atmosphere)};
const HIGH_CONTRAST = {str(high_contrast).lower()};
const SEED = {seed};

let nodes = [];

function setup() {{
  createCanvas(windowWidth, windowHeight);
  rectMode(CENTER);
  noStroke();
  initializeNodes();
}}

function initializeNodes() {{
  randomSeed(SEED);
  noiseSeed(SEED);
  nodes = [];

  for (let i = 0; i < NODE_COUNT; i += 1) {{
    const spreadX = COMPOSITION_MODE === "minimal" ? width * 0.34 : COMPOSITION_MODE === "dense" ? width * 0.2 : width * 0.27;
    const spreadY = COMPOSITION_MODE === "minimal" ? height * 0.3 : COMPOSITION_MODE === "dense" ? height * 0.24 : height * 0.28;
    nodes.push({{
      baseX: width * 0.5 + random(-spreadX, spreadX),
      baseY: height * 0.5 + random(-spreadY, spreadY),
      size: random(MIN_SIZE, MAX_SIZE),
      phase: random(TWO_PI),
      orbit: random(0.55, 1.5),
      tilt: random(-1, 1),
      colorIndex: i % PALETTE.length,
      layerOffset: random(-0.18, 0.18)
    }});
  }}
}}

function drawBackgroundWash() {{
  background(BACKGROUND_COLOR[0], BACKGROUND_COLOR[1], BACKGROUND_COLOR[2]);

  const washAlpha = ATMOSPHERE === "dramatic" ? 34 : ATMOSPHERE === "dreamlike" ? 24 : 16;
  noStroke();
  for (let i = 0; i < PALETTE.length; i += 1) {{
    const col = PALETTE[i];
    fill(col[0], col[1], col[2], washAlpha);
    const radius = width * (0.32 + i * 0.08);
    ellipse(width * (0.28 + i * 0.22), height * (0.26 + i * 0.16), radius, radius);
  }}
}}

function drawGlow(x, y, size, col) {{
  fill(col[0], col[1], col[2], 24);
  ellipse(x, y, size * 2.2, size * 2.2);
}}

function drawNodeShape(node, x, y, size, rotation, col, alphaScale) {{
  fill(col[0], col[1], col[2], 150 * alphaScale);
  push();
  translate(x, y);
  rotate(rotation);

  if (SHAPE_MODE === "geometric") {{
    rect(0, 0, size, size * 0.78, size * 0.16);
  }} else if (SHAPE_MODE === "mixed" && node.colorIndex % 2 === 0) {{
    rect(0, 0, size * 0.88, size * 0.88, size * 0.14);
  }} else {{
    ellipse(0, 0, size, size * 0.82);
  }}

  pop();
}}

function drawTexture() {{
  if (TEXTURE_MODE === "grainy") {{
    stroke(255, 255, 255, 18);
    for (let i = 0; i < 110; i += 1) {{
      point((i * 37 + frameCount + SEED) % width, (i * 19 + SEED * 3) % height);
    }}
    noStroke();
  }}
}}

function draw() {{
  drawBackgroundWash();

  for (let i = 0; i < nodes.length; i += 1) {{
    const node = nodes[i];
    const t = frameCount * MOTION_SPEED * node.orbit + node.phase;
    const motionBlend = MOTION_MODE === "still" ? 0.18 : MOTION_MODE === "dynamic" ? 1.45 : 0.72;
    const x = node.baseX + sin(t * 1.3) * DRIFT_X * motionBlend + cos(t * 0.55) * DRIFT_X * 0.24;
    const y = node.baseY + cos(t * 1.05) * DRIFT_Y * motionBlend + sin(t * 0.42) * DRIFT_Y * 0.18;
    const sizePulse = node.size + sin(t * 1.1) * (MOTION_MODE === "dynamic" ? 5 : 2);
    const rotation = sin(t + node.tilt) * ROTATION_AMOUNT;
    const col = PALETTE[node.colorIndex];
    const alphaScale = COMPOSITION_MODE === "dense" ? 0.95 : COMPOSITION_MODE === "minimal" ? 0.72 : 0.84;

    if (TEXTURE_MODE === "glowing" || ATMOSPHERE === "dreamlike") {{
      drawGlow(x, y, sizePulse, col);
    }}

    drawNodeShape(node, x, y, sizePulse, rotation, col, alphaScale);
  }}

  drawTexture();

  if (HIGH_CONTRAST) {{
    stroke(255, 255, 255, 26);
    line(width * 0.16, height * 0.82, width * 0.84, height * 0.18);
    noStroke();
  }}
}}

function windowResized() {{
  resizeCanvas(windowWidth, windowHeight);
  initializeNodes();
}}
""".strip()


def _build_local_forced_implementation_payload(
    *,
    emotion_profile: str,
    emotion_confidence: str,
    emotion_gaps: list[str],
    artistic_profile: str,
    artistic_confidence: str,
) -> dict:
    code = _build_local_p5_code_from_direction(
        artistic_profile=artistic_profile,
        emotion_profile=emotion_profile,
    )
    return {
        "message": (
            "I updated the sketch to follow the latest artistic direction. "
            "This pass resets conflicting earlier choices so the new motion, palette, and form language show up clearly."
        ),
        "commit_message": "Applied updated artistic direction to the sketch",
        "emotion_profile": emotion_profile,
        "emotion_confidence": emotion_confidence,
        "emotion_gaps": emotion_gaps,
        "artistic_profile": artistic_profile,
        "artistic_confidence": artistic_confidence,
        "code": code,
        "should_create_version": True,
        "offers_artistic_alternatives": False,
        "artistic_options": [],
    }


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
    artistic_confidence = _normalize_confidence(raw.get("artistic_confidence") or current.artistic_confidence)
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
        "artistic_confidence": artistic_confidence,
        "emotion_gaps": emotion_gaps,
        "commit_message": commit_message,
    }


def create_version(
    session: SessionState,
    *,
    emotion_profile: str,
    artistic_profile: str,
    artistic_confidence: str,
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
        artistic_confidence=artistic_confidence,
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
    collab_style: str = "iterate",
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

    llm = phase_1_llm if session.phase == PHASE_EMOTION else phase_2_llm
    state_summary = _summarize_state_for_prompt(session.current_version)
    system_msg = SystemMessage(content=build_system_prompt(session.phase, state_summary, collab_style))

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
You are in the artistic discovery step of a three-phase workflow: emotion -> artistic -> coding.
Do not generate code and do not finalize a version yet.
The user feedback is: "{user_feedback or 'No extra feedback provided.'}"
Current emotion profile: "{session.current_version.emotion_profile or '(empty)'}"
Current proposed artistic profile: "{proposed_artistic_profile or '(empty)'}"

Return STRICT JSON only with this schema:
{{
  "message": "<empathetic short reply + brief interpretation + natural presentation of the options + invitation to choose or describe their own vision>",
  "recommended_artistic_profile": "<2-4 sentences that summarize the visual direction, explicitly naming the user's chosen preferences, their custom note or chat direction, and how it connects back to the emotion>",
  "artistic_options": ["<option 1>", "<option 2>", "<option 3>"],
  "emotion_profile": "The emotion is ...",
  "emotion_confidence": "<low|medium|high>",
  "artistic_confidence": "<low|medium|high>",
  "emotion_gaps": ["<gap 1>", "<gap 2>"],
  "commit_message": "<one-line summary of this proposed direction>"
}}

Rules:
- Be empathetic and supportive.
- Set expectation that you may be wrong and invite correction.
- Generate exactly 3 clearly distinct artwork directions the user could compare before coding.
- Use a spread of visual axes when helpful: abstraction vs detail, stillness vs motion, palette temperature and contrast, shape language, composition density, atmosphere, and texture.
- In `recommended_artistic_profile`, explicitly mention the user's stated visual preferences and any direct visual note they typed in this request.
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
    artistic_confidence = _normalize_confidence(raw.get("artistic_confidence") or current.artistic_confidence)
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
        "artistic_confidence": artistic_confidence,
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
        "artistic_confidence": v.artistic_confidence,
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
        "proposed_artistic_confidence": pending.proposed_artistic_confidence,
        "proposed_code": pending.proposed_code,
        "alternative_preview_codes": list(pending.alternative_preview_codes),
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
        "phase_order": list(PHASE_ORDER),
        "unlocked_phases": list(session.unlocked_phases),
        "visited_phases": list(session.visited_phases),
        "current_version": current,
        "versions": versions,
        "messages": [serialize_message(m) for m in session.messages],
        "branch_heads": dict(session.branch_heads),
        "active_branch": session.active_branch,
        "pending_artistic_decision": serialize_pending_artistic_decision(session.pending_artistic_decision),
        "artistic_panel_state": _serialize_artistic_panel_state(session.artistic_panel_state),
    }


app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(BASE_DIR, "frontend.html")


@app.route("/debug", methods=["GET"])
def debug_page():
    return send_from_directory(BASE_DIR, "debug.html")


@app.route("/good_examples/<path:filename>", methods=["GET"])
def good_examples(filename):
    return send_from_directory(BASE_DIR / "good_examples", filename)


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


@app.route("/api/set-phase", methods=["POST"])
def api_set_phase():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    target_phase = _clean_text(data.get("phase"))
    if not sid or not target_phase:
        return jsonify({"error": "Missing session_id or phase"}), 400

    session = get_or_create_session(sid)
    if "artistic_panel_state" in data:
        _save_artistic_panel_state(session, data.get("artistic_panel_state"))
    if target_phase not in PHASE_ORDER:
        return jsonify({"error": "Invalid phase"}), 400

    if session.phase == PHASE_ARTISTIC and target_phase == PHASE_CODE:
        direct_artistic_profile = _clean_text(data.get("artistic_profile"))
        direct_artistic_confidence = _normalize_confidence(data.get("artistic_confidence") or "")
        if not direct_artistic_profile:
            direct_artistic_profile, direct_artistic_confidence = _get_pending_artistic_transition_payload(
                session.pending_artistic_decision
            )
        if not direct_artistic_profile and not _can_unlock_phase(session, PHASE_CODE):
            direct_artistic_profile = _build_artistic_profile_from_panel_state(session.artistic_panel_state)
            direct_artistic_confidence = _infer_artistic_confidence_from_panel_state(session.artistic_panel_state)
        if direct_artistic_profile:
            _apply_direct_artistic_profile(
                session,
                artistic_profile=direct_artistic_profile,
                artistic_confidence=direct_artistic_confidence,
                summary="Saved artistic panel selections for direct implementation",
                source="user",
            )

    if target_phase not in session.unlocked_phases and not _can_unlock_phase(session, target_phase):
        return jsonify({"error": _phase_locked_message(target_phase)}), 400

    previous_phase = session.phase
    _unlock_phase(session, target_phase)
    first_visit = _mark_phase_visited(session, target_phase)
    session.phase = target_phase
    if target_phase != PHASE_ARTISTIC:
        session.pending_artistic_decision = None

    transition_message = _phase_transition_message(
        target_phase,
        first_visit=first_visit,
        current_version=session.current_version,
    )
    if transition_message:
        _append_chat(session, "assistant", "system", transition_message)
        _append_llm_event(session, AIMessage(content=transition_message), session.current_version_id)

    should_auto_implement = (
        target_phase == PHASE_CODE
        and previous_phase == PHASE_ARTISTIC
        and bool(_clean_text(session.current_version.artistic_profile))
    )

    return jsonify(
        {
            "ok": True,
            "phase": session.phase,
            "should_auto_implement": should_auto_implement,
            "state": serialize_state(session),
        }
    )


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    session = get_or_create_session(data.get("session_id"))
    if "artistic_panel_state" in data:
        _save_artistic_panel_state(session, data.get("artistic_panel_state"))
    direct_artistic_profile = _clean_text(data.get("artistic_profile"))
    direct_artistic_confidence = _normalize_confidence(data.get("artistic_confidence") or "")
    intent_hint = _clean_text(data.get("intent_hint"))
    force_implement_current_direction = intent_hint == "implement_current_art_direction"

    message = _clean_text(data.get("message"))
    image = _clean_text(data.get("image"))
    audio = _clean_text(data.get("audio"))
    image_mime = _clean_text(data.get("image_mime")) or "image/jpeg"
    audio_mime = _clean_text(data.get("audio_mime")) or "audio/webm"
    collab_style = _clean_text(data.get("collab_style")) or "iterate"

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
    user_text_for_model = message

    if session.phase == PHASE_CODE and actions["emotion_refinement"] and not actions["sketch_rejection"]:
        session.phase = PHASE_EMOTION
        _mark_phase_visited(session, PHASE_EMOTION)
        session.pending_artistic_decision = None
    elif session.phase == PHASE_ARTISTIC and actions["emotion_refinement"]:
        session.phase = PHASE_EMOTION
        _mark_phase_visited(session, PHASE_EMOTION)
        session.pending_artistic_decision = None
    elif session.phase == PHASE_EMOTION and session.current_version.emotion_profile:
        if actions["confirm_start_coding"] or actions["code_request"]:
            if PHASE_CODE not in session.unlocked_phases:
                if PHASE_ARTISTIC not in session.unlocked_phases and not _can_unlock_phase(session, PHASE_ARTISTIC):
                    assistant_text = _phase_locked_message(PHASE_ARTISTIC)
                    _append_chat(session, "assistant", "text", assistant_text)
                    current = session.current_version
                    return _chat_response(
                        {
                            "message": assistant_text,
                            "commit_message": "Stayed in emotion discovery until the feeling is clearer",
                            "emotion_profile": current.emotion_profile,
                            "emotion_confidence": current.emotion_confidence,
                            "emotion_gaps": current.emotion_gaps,
                            "artistic_profile": current.artistic_profile,
                            "artistic_confidence": current.artistic_confidence,
                            "code": current.code,
                            "should_create_version": False,
                            "offers_artistic_alternatives": False,
                            "artistic_options": [],
                        },
                        raw_response_text="",
                        created_version=None,
                    )
                _unlock_phase(session, PHASE_ARTISTIC)
                first_visit = _mark_phase_visited(session, PHASE_ARTISTIC)
                session.phase = PHASE_ARTISTIC
                transition_message = _phase_transition_message(
                    PHASE_ARTISTIC,
                    first_visit=first_visit,
                    current_version=session.current_version,
                )
                _append_chat(session, "assistant", "system", transition_message)
                _append_llm_event(session, AIMessage(content=transition_message), session.current_version_id)
                current = session.current_version
                return _chat_response(
                    {
                        "message": transition_message,
                        "commit_message": "Moved from emotion discovery to artistic discovery",
                        "emotion_profile": current.emotion_profile,
                        "emotion_confidence": current.emotion_confidence,
                        "emotion_gaps": current.emotion_gaps,
                        "artistic_profile": current.artistic_profile,
                        "artistic_confidence": current.artistic_confidence,
                        "code": current.code,
                        "should_create_version": False,
                        "offers_artistic_alternatives": False,
                        "artistic_options": [],
                    },
                    raw_response_text="",
                    created_version=None,
                )
            session.phase = PHASE_CODE
            _mark_phase_visited(session, PHASE_CODE)
            user_text_for_model = _build_code_start_confirmation_prompt(session, message)
    elif session.phase == PHASE_ARTISTIC and (actions["confirm_start_coding"] or actions["code_request"]):
        if direct_artistic_profile:
            _apply_direct_artistic_profile(
                session,
                artistic_profile=direct_artistic_profile,
                artistic_confidence=direct_artistic_confidence,
                summary="Saved artistic panel selections for direct implementation",
                source="user",
            )
        else:
            pending_artistic_profile, pending_artistic_confidence = _get_pending_artistic_transition_payload(
                session.pending_artistic_decision
            )
            if pending_artistic_profile:
                _apply_direct_artistic_profile(
                    session,
                    artistic_profile=pending_artistic_profile,
                    artistic_confidence=pending_artistic_confidence,
                    summary="Saved generated artistic direction for implementation",
                    source="assistant",
                )
            elif PHASE_CODE not in session.unlocked_phases and not _can_unlock_phase(session, PHASE_CODE):
                _apply_saved_artistic_panel_state_for_coding(
                    session,
                    summary="Saved artistic panel selections for direct implementation",
                    source="user",
                )
        if PHASE_CODE not in session.unlocked_phases and not _can_unlock_phase(session, PHASE_CODE):
            assistant_text = (
                "Choose one of the artwork options, accept the draft direction, or describe the visuals you want first. "
                "Once an artistic direction is saved, Coding will unlock."
            )
            _append_chat(session, "assistant", "text", assistant_text)
            current = session.current_version
            pending = session.pending_artistic_decision
            return _chat_response(
                {
                    "message": assistant_text,
                    "commit_message": "Stayed in artistic discovery until a direction is chosen",
                    "emotion_profile": current.emotion_profile,
                    "emotion_confidence": current.emotion_confidence,
                    "emotion_gaps": current.emotion_gaps,
                    "artistic_profile": (
                        pending.proposed_artistic_profile if pending and pending.proposed_artistic_profile else current.artistic_profile
                    ),
                    "artistic_confidence": (
                        pending.proposed_artistic_confidence if pending and pending.proposed_artistic_confidence else current.artistic_confidence
                    ),
                    "code": current.code,
                    "should_create_version": False,
                    "offers_artistic_alternatives": bool(pending and pending.alternatives),
                    "artistic_options": list(pending.alternatives) if pending else [],
                },
                raw_response_text="",
                created_version=None,
            )
        session.phase = PHASE_CODE
        _mark_phase_visited(session, PHASE_CODE)
        session.pending_artistic_decision = None
        force_implement_current_direction = True
        user_text_for_model = _build_code_start_confirmation_prompt(session, message)

    pending = session.pending_artistic_decision

    if session.phase == PHASE_ARTISTIC:
        current = session.current_version

        if pending and decision == "modify":
            pending.awaiting_modify_details = True
            assistant_text = (
                "Tell me what to change in the artwork options, or describe the visual direction you want in your own words, "
                "and I will generate a fresh set before we code."
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
                    "artistic_confidence": pending.proposed_artistic_confidence or current.artistic_confidence,
                    "code": current.code,
                    "should_create_version": False,
                    "offers_artistic_alternatives": False,
                    "artistic_options": [],
                },
                raw_response_text="",
                created_version=None,
            )

        if pending and decision in {"accept", "more_options"} | {
            f"option_{i}" for i in range(1, 4)
        }:
            if decision == "more_options":
                feedback = "Please generate three fresh artistic directions I can compare before coding."
            else:
                feedback = ""

            if decision == "more_options":
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
                pending.proposed_artistic_confidence = options_payload["artistic_confidence"]
                pending.alternatives = options_payload["artistic_options"]
                pending.alternative_preview_codes = _build_artistic_option_preview_codes(
                    pending.alternatives,
                    pending.proposed_emotion_profile or current.emotion_profile,
                )
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
                        "artistic_confidence": pending.proposed_artistic_confidence,
                        "code": current.code,
                        "should_create_version": False,
                        "offers_artistic_alternatives": True,
                        "artistic_options": list(pending.alternatives),
                    },
                    raw_response_text=raw_options_text,
                    created_version=None,
                )

            selected_artistic_profile = None
            if decision == "accept":
                selected_artistic_profile = pending.proposed_artistic_profile
            elif decision.startswith("option_"):
                try:
                    option_index = int(decision.split("_", 1)[1])
                except Exception:
                    option_index = 0
                if option_index < 1 or option_index > len(pending.alternatives):
                    assistant_text = (
                        f"I could not find Option {option_index}. Please choose one of the listed options, ask for more options, or refine the art direction."
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
                            "artistic_confidence": pending.proposed_artistic_confidence or current.artistic_confidence,
                            "code": current.code,
                            "should_create_version": False,
                            "offers_artistic_alternatives": True,
                            "artistic_options": list(pending.alternatives),
                        },
                        raw_response_text="",
                        created_version=None,
                    )
                selected_artistic_profile = pending.alternatives[option_index - 1]

            if not selected_artistic_profile:
                assistant_text = "I could not figure out which artistic option you wanted. Please choose one of the visible options."
                _append_chat(session, "assistant", "text", assistant_text)
                return _chat_response(
                    {
                        "message": assistant_text,
                        "commit_message": pending.proposed_commit_message,
                        "emotion_profile": pending.proposed_emotion_profile or current.emotion_profile,
                        "emotion_confidence": pending.proposed_emotion_confidence or current.emotion_confidence,
                        "emotion_gaps": pending.proposed_emotion_gaps or current.emotion_gaps,
                        "artistic_profile": pending.proposed_artistic_profile or current.artistic_profile,
                        "artistic_confidence": pending.proposed_artistic_confidence or current.artistic_confidence,
                        "code": current.code,
                        "should_create_version": False,
                        "offers_artistic_alternatives": True,
                        "artistic_options": list(pending.alternatives),
                    },
                    raw_response_text="",
                    created_version=None,
                )

            new_emotion = pending.proposed_emotion_profile or current.emotion_profile
            new_conf = pending.proposed_emotion_confidence or current.emotion_confidence
            new_gaps = pending.proposed_emotion_gaps or current.emotion_gaps
            new_artistic = selected_artistic_profile
            new_artistic_conf = pending.proposed_artistic_confidence or current.artistic_confidence
            created_version = None

            if _has_artistic_change(current.artistic_profile, new_artistic) or _has_emotional_state_change(
                current,
                new_emotion,
                new_conf,
                new_gaps,
            ):
                summary = _normalize_commit_message(
                    pending.proposed_commit_message,
                    fallback="Saved the selected artistic direction before coding",
                )
                created_version = create_version(
                    session,
                    emotion_profile=new_emotion,
                    artistic_profile=new_artistic,
                    artistic_confidence=new_artistic_conf,
                    code=current.code,
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
                _update_current_artistic_state(
                    current,
                    artistic_profile=new_artistic,
                    artistic_confidence=new_artistic_conf,
                )

            if _confidence_allows_advance(new_artistic_conf):
                _unlock_phase(session, PHASE_CODE)
                assistant_text = (
                    "I saved that artistic direction. Move to **Coding** when you are ready, and I will implement it in p5.js. "
                    "You can still come back here for a bigger visual change anytime."
                )
            else:
                assistant_text = (
                    "I saved that artistic direction, but the artistic confidence is still low. Let’s refine the visuals a bit more here before moving into **Coding**."
                )
            _append_chat(session, "assistant", "text", assistant_text)
            session.pending_artistic_decision = None
            return _chat_response(
                {
                    "message": assistant_text,
                    "commit_message": pending.proposed_commit_message,
                    "emotion_profile": new_emotion,
                    "emotion_confidence": new_conf,
                    "emotion_gaps": new_gaps,
                    "artistic_profile": new_artistic,
                    "artistic_confidence": new_artistic_conf,
                    "code": current.code,
                    "should_create_version": bool(created_version),
                    "offers_artistic_alternatives": False,
                    "artistic_options": [],
                },
                raw_response_text="",
                created_version=created_version,
            )

        if not user_text_for_model:
            if image and audio:
                user_text_for_model = "I shared an image and audio clip as artistic inspiration. Turn them into three visual directions before coding."
            elif image:
                user_text_for_model = "I shared an image as artistic inspiration. Turn it into three visual directions before coding."
            elif audio:
                user_text_for_model = "I shared an audio clip as artistic inspiration. Turn it into three visual directions before coding."
            else:
                user_text_for_model = "Generate three artistic directions I can compare before coding."

        proposed_profile = pending.proposed_artistic_profile if pending else current.artistic_profile
        options_payload, raw_options_text, human_msg, assistant_msg = _invoke_artistic_options_llm(
            session,
            user_feedback=user_text_for_model,
            proposed_artistic_profile=proposed_profile,
        )
        _update_current_emotion_state(
            current,
            emotion_profile=options_payload["emotion_profile"] or current.emotion_profile,
            emotion_confidence=options_payload["emotion_confidence"] or current.emotion_confidence,
            emotion_gaps=options_payload["emotion_gaps"] or current.emotion_gaps,
        )
        session.pending_artistic_decision = PendingArtisticDecision(
            proposed_artistic_profile=options_payload["recommended_artistic_profile"],
            proposed_artistic_confidence=options_payload["artistic_confidence"] or current.artistic_confidence,
            proposed_code=current.code,
            alternative_preview_codes=_build_artistic_option_preview_codes(
                options_payload["artistic_options"],
                options_payload["emotion_profile"] or current.emotion_profile,
            ),
            proposed_emotion_profile=options_payload["emotion_profile"] or current.emotion_profile,
            proposed_emotion_confidence=options_payload["emotion_confidence"] or current.emotion_confidence,
            proposed_emotion_gaps=options_payload["emotion_gaps"] or current.emotion_gaps,
            proposed_commit_message=options_payload["commit_message"],
            alternatives=options_payload["artistic_options"],
            awaiting_modify_details=False,
        )
        _append_chat(session, "assistant", "text", options_payload["message"])
        _append_llm_event(session, human_msg, session.current_version_id)
        _append_llm_event(session, assistant_msg, session.current_version_id)
        return _chat_response(
            {
                "message": options_payload["message"],
                "commit_message": options_payload["commit_message"],
                "emotion_profile": options_payload["emotion_profile"] or current.emotion_profile,
                "emotion_confidence": options_payload["emotion_confidence"] or current.emotion_confidence,
                "emotion_gaps": options_payload["emotion_gaps"] or current.emotion_gaps,
                "artistic_profile": options_payload["recommended_artistic_profile"],
                "artistic_confidence": options_payload["artistic_confidence"] or current.artistic_confidence,
                "code": current.code,
                "should_create_version": False,
                "offers_artistic_alternatives": True,
                "artistic_options": list(options_payload["artistic_options"]),
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
    elif session.phase == PHASE_CODE and actions["sketch_rejection"]:
        user_text_for_model = _build_sketch_rejection_prompt(session, effective_message)

    parsed_raw, raw_response_text, human_msg, assistant_llm_msg = _invoke_llm(
        session=session,
        user_text=user_text_for_model,
        image=image,
        image_mime=image_mime,
        audio=audio,
        audio_mime=audio_mime,
        intent_hint="sketch_rejection" if actions["sketch_rejection"] else "",
        collab_style=collab_style,
    )
    sanitized = _sanitize_llm_payload(parsed_raw, session.current_version)

    old = session.current_version
    new_emotion = sanitized["emotion_profile"]
    new_artistic = sanitized["artistic_profile"]
    new_artistic_conf = sanitized["artistic_confidence"]
    new_code = sanitized["code"]
    new_conf = sanitized["emotion_confidence"]
    new_gaps = sanitized["emotion_gaps"]
    llm_should_create_version = sanitized["should_create_version"]
    offers_artistic_alternatives = sanitized["offers_artistic_alternatives"]
    artistic_options = sanitized["artistic_options"]

    if session.phase == PHASE_CODE and force_implement_current_direction:
        needs_local_regen = not _clean_text(new_code) or new_code == old.code or not llm_should_create_version
        if needs_local_regen and _clean_text(new_artistic):
            sanitized = _build_local_forced_implementation_payload(
                emotion_profile=new_emotion,
                emotion_confidence=new_conf,
                emotion_gaps=new_gaps,
                artistic_profile=new_artistic,
                artistic_confidence=new_artistic_conf or old.artistic_confidence or "medium",
            )
            new_emotion = sanitized["emotion_profile"]
            new_artistic = sanitized["artistic_profile"]
            new_artistic_conf = sanitized["artistic_confidence"]
            new_code = sanitized["code"]
            new_conf = sanitized["emotion_confidence"]
            new_gaps = sanitized["emotion_gaps"]
            llm_should_create_version = sanitized["should_create_version"]
            offers_artistic_alternatives = sanitized["offers_artistic_alternatives"]
            artistic_options = sanitized["artistic_options"]

    if session.phase == PHASE_CODE and new_code and new_code != old.code:
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

    should_stage_artistic_decision = session.phase == PHASE_CODE and offers_artistic_alternatives

    if should_stage_artistic_decision:
        _update_current_emotion_state(
            old,
            emotion_profile=new_emotion,
            emotion_confidence=new_conf,
            emotion_gaps=new_gaps,
        )
        session.pending_artistic_decision = PendingArtisticDecision(
            proposed_artistic_profile=new_artistic,
            proposed_artistic_confidence=new_artistic_conf,
            proposed_code=new_code,
            alternative_preview_codes=_build_artistic_option_preview_codes(
                artistic_options,
                new_emotion,
            ),
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
        session.phase = PHASE_ARTISTIC
        _unlock_phase(session, PHASE_ARTISTIC)
        _mark_phase_visited(session, PHASE_ARTISTIC)
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

    if session.phase == PHASE_EMOTION:
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
        if session.phase == PHASE_EMOTION and not emotion_profile_change:
            fallback_summary = "Refined emotional understanding details"
        else:
            fallback_summary = _build_commit_summary(old, new_emotion, new_artistic, new_code, source="assistant")
        summary = _normalize_commit_message(sanitized["commit_message"], fallback=fallback_summary)
        created_version = create_version(
            session,
            emotion_profile=new_emotion,
            artistic_profile=new_artistic,
            artistic_confidence=new_artistic_conf,
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
    if sanitized["emotion_profile"] and _confidence_allows_advance(new_conf):
        _unlock_phase(session, PHASE_ARTISTIC)
    if session.phase == PHASE_CODE and (_clean_text(new_code) or (_clean_text(new_artistic) and _confidence_allows_advance(new_artistic_conf))):
        _unlock_phase(session, PHASE_CODE)

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

    emotion_profile = old.emotion_profile
    artistic_profile = old.artistic_profile
    code = old.code

    if "emotion_profile" in data:
        emotion_profile = _ensure_emotion_prefix(_clean_text(data.get("emotion_profile")))
    if "artistic_profile" in data:
        artistic_profile = _clean_text(data.get("artistic_profile"))
    if "code" in data:
        code = _clean_text(data.get("code"))

    summary = _clean_text(data.get("summary")) or _build_commit_summary(
        old, emotion_profile, artistic_profile, code, source="user"
    )

    create_version(
        session,
        emotion_profile=emotion_profile,
        artistic_profile=artistic_profile,
        artistic_confidence=old.artistic_confidence or "",
        code=code,
        emotion_confidence=old.emotion_confidence or "",
        emotion_gaps=old.emotion_gaps,
        summary=summary,
        source="user",
    )
    if emotion_profile and _confidence_allows_advance(old.emotion_confidence):
        _unlock_phase(session, PHASE_ARTISTIC)
    if code or (artistic_profile and _confidence_allows_advance(old.artistic_confidence)):
        _unlock_phase(session, PHASE_CODE)
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
    _mark_phase_visited(session, session.phase)

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
        "New session started. We will move through Emotion Discovery, Artistic Discovery, and Coding together. Share an emotion, a memory, an image, or an audio clip to begin.",
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
