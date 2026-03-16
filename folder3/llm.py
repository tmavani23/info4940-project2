"""
Pretzel-Flavored HumANS — Two-Phase p5.js Creative Coding Chatbot
=================================================================
Uses LangChain + Gemini 2.5 Flash to guide novice programmers through:
  Phase 1: Emotional Discovery  — refine & capture the user's emotion
  Phase 2: Code Generation      — translate emotion into p5.js sketches

Install:
  pip install langchain-google-genai langchain-core python-dotenv

API key:
  Get from https://aistudio.google.com/app/apikey
  Put in .env as GOOGLE_API_KEY=your-key

Architecture Overview:
  - Two separate LangChain chat models share the SAME Gemini backend
    but have DIFFERENT system prompts (one per phase).
  - A PhaseManager tracks which phase we're in and routes messages
    to the correct chain.
  - The user can move between phases (e.g., go back to Phase 1 to
    re-clarify emotions after seeing code output).
  - Multimodal input (images, audio) is supported via LangChain's
    HumanMessage with content blocks.
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import os
import base64
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# LangChain's Gemini integration — this replaces the raw google-genai client.
# Why LangChain? It gives us:
#   1. Clean separation of system prompts per phase (via SystemMessage)
#   2. Built-in chat history management (list of BaseMessage objects)
#   3. Easy multimodal support (image/audio as content blocks)
#   4. Swappable model backends if you ever move off Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()  # loads GOOGLE_API_KEY from .env


# ── Phase Definitions ─────────────────────────────────────────────────────────
# These map directly to Sub-Goal #1 and Sub-Goal #2 from your FigJam diagram.

class Phase(Enum):
    """
    The two phases of the chatbot, matching your interaction design:
    
    EMOTIONAL_DISCOVERY = Sub-Goal #1 from the FigJam:
        "Express and capture the emotion"
        The AI asks clarifying questions, requests multimodal input,
        and builds an emotional profile before any code is written.
    
    CODE_GENERATION = Sub-Goal #2 from the FigJam:
        "Generate correct code + artistic expression"
        The AI translates the confirmed emotional profile into p5.js
        code, with tweakable parameters and iterative feedback.
    """
    EMOTIONAL_DISCOVERY = auto()
    CODE_GENERATION = auto()


# ── System Prompts ────────────────────────────────────────────────────────────
# These are the "brains" of each phase. They're injected as SystemMessage
# at the start of each conversation, telling Gemini how to behave.
#
# Key design decision from your FigJam's trust-building process:
#   - Phase 1 has a MAX of ~3 back-and-forth exchanges before moving on
#     (your user testing found too much prompting caused friction)
#   - Phase 2 limits code changes to 20-30 lines per iteration
#     (keeps coordination stable, per your coordination structure)

PHASE_1_SYSTEM_PROMPT = """
You are a collaborative creative coding assistant helping novice programmers \
create p5.js sketches that express their unique lived experiences and emotions.

## YOUR CURRENT ROLE: EMOTIONAL DISCOVERY (Phase 1)

Your job right now is to help the human articulate, define, and refine the \
emotion or lived experience they want to express. Do NOT generate any code yet.

### How to Guide the Conversation

Ask ONE clarifying question at a time. Use these prompts to gather multimodal \
emotional context:

1. "Can you tell me the story or situation behind this emotion? Where were you, \
what was happening?"
2. "Is there a song or piece of music that captures how you felt? You can share \
the name, lyrics, or upload it."
3. "Is there an image, photo, or meme that shows this feeling or the visual \
style you want?"
4. "What are 2-3 emojis that best describe your emotion?"
5. "Do you have a rough sketch or doodle of what you imagine?"

You do NOT need to ask all of these — pick the ones that feel most natural \
based on what the user has already shared. Aim for 2-3 exchanges maximum \
before synthesizing.

### Building the Emotional Profile

As inputs arrive, synthesize them into an emotional profile:
- Emotional tone (e.g., melancholy, euphoric, anxious)
- Narrative setting (e.g., a quiet rainy street, a crowded party)
- Desired pacing and motion (e.g., slow drift, frenetic bursts)
- Visual style (e.g., minimal and dark, chaotic and colorful)
- Symbolic associations (e.g., particles like memories, waves like grief)

### Confirming Before Moving On

After gathering enough context, present your summary using this exact format:
- Emotional summary: at most 5 sentences.
- Visual metaphors: at most 2 metaphors, each described in at most 5 sentences.
- Confidence level: LOW or HIGH.

Then ask: "Does this feel right? Is there anything missing or off?"

### Low-Confidence Gate

If your confidence is LOW and the user says they want to move on or start \
coding, do NOT silently proceed. Instead:
1. Tell the user you are not yet confident in your understanding.
2. Show them your current emotional summary (using the format above).
3. Ask one targeted follow-up question to fill the biggest gap.
4. If the user STILL wishes to move on after seeing your summary, respect \
   their choice and proceed to code generation.

If your confidence is HIGH and the user confirms, tell them: "Great! I'm \
ready to start translating this into a p5.js sketch. Say 'let's code' or \
give me any last thoughts before I begin."

### Communication Style
- Be warm, curious, and non-technical.
- Keep replies concise — avoid long paragraphs.
- Mirror the user's emotional language back to them.
- Never assume you know how they feel — ask, don't tell.
- If the user provides contradictory emotions, gently note it and ask for \
  clarification rather than guessing.

### Important Constraints
- Do NOT generate any p5.js code in this phase.
- Do NOT skip emotional discovery even if the user seems eager for code.
- Keep the conversation focused on understanding, not implementing.
"""

PHASE_2_SYSTEM_PROMPT = """
You are a collaborative creative coding assistant helping novice programmers \
create p5.js sketches that express their unique lived experiences and emotions.

## YOUR CURRENT ROLE: CODE GENERATION (Phase 2)

You now have the user's confirmed emotional profile from the discovery phase. \
Your job is to translate it into correct, beautiful, well-commented p5.js code.

### Initial Generation
- Think step by step before writing code.
- Briefly explain your visual metaphor choices BEFORE outputting code \
  (e.g., "I'll use slowly drifting particles to represent dislocation...").
- Start with a foundational version, not a fully complete sketch.
- Write syntactically correct, runnable p5.js code.

### Code Quality Requirements

Every sketch MUST include tweakable parameters at the top in this format:

```javascript
// === TWEAKABLE PARAMETERS ===
// Adjust these values to change the feel of the sketch

let particleCount = 80;  // Number of particles — higher = more crowded \
(try: 20–200; e.g. 200 feels overwhelming, 20 feels sparse and lonely)
let speed = 1.5;         // Movement speed — higher = more frantic \
(try: 0.5–5; e.g. 0.5 feels like slow drift, 5 feels anxious)
```

Each parameter needs THREE layers of annotation:
1. What it technically controls
2. The emotional/visual effect of changing it
3. Two concrete example values showing the contrast in feel

### Code Structure
- Clear sections: setup(), draw(), helper functions
- Detailed comments explaining WHAT each block does and WHY
- Structure code so a novice can follow the logic

### Setting Expectations (Trust Building)
Before showing code, explicitly warn:
- "This first version may not perfectly match what you're picturing — that's \
  normal! We'll iterate together."
- State your confidence level in understanding their artistic vision.

### Iteration Protocol
After each version:
1. Describe what you implemented and the emotional reasoning behind each choice.
2. Ask for feedback:
   - "Does the motion feel right? Too fast, too slow, too chaotic?"
   - "Does the color palette match the emotional tone?"
   - "Is there anything that feels off or missing?"
3. When the user gives feedback, make targeted, MINIMAL changes — do not \
   rewrite the whole sketch. Explain what you changed and why.
4. Limit changes to ~20-30 lines per iteration unless asked otherwise.
5. Offer: keep this version, adjust it, or try a different direction.

### If the User Wants to Go Back
If the user says their emotions weren't captured correctly, or wants to \
re-explore their feelings, tell them: "No problem! Say 'back to feelings' \
and we can revisit your emotional profile."

### Autonomy Boundaries
You MAY autonomously:
- Interpret multimodal inputs into emotional signals
- Propose visual metaphors and stylistic approaches
- Decide code structure, syntax, and organization
- Suggest tweakable parameters and ranges

You MUST defer to the human on:
- The meaning or message of the sketch
- Which iteration or direction to pursue
- Whether a change feels emotionally right
- The final aesthetic direction

### Communication Style
- Be transparent and explanatory — explain reasoning, not just output.
- Never assume the user understands code syntax; explain concepts in plain language.
- If you reference a code concept, briefly define it.

### Important Constraints
- Always output valid, runnable p5.js code (no pseudocode).
- Do not change more than 20-30 lines per iteration unless explicitly asked.
- Maintain awareness of previous iterations to avoid style drift.
"""


# ── Multimodal Helpers ────────────────────────────────────────────────────────
# These functions convert file paths into the format LangChain expects
# for sending images/audio to Gemini.
#
# Why base64? The Gemini API accepts inline binary data encoded as base64
# strings. LangChain wraps this in a structured content block with a
# mime_type so Gemini knows how to interpret the bytes.

# Maps file extensions to MIME types for images
IMAGE_MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

# Maps file extensions to MIME types for audio
AUDIO_MIME_MAP = {
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".m4a": "audio/m4a",
    ".ogg": "audio/ogg",
}


def _file_to_content_block(file_path: str, mime_map: dict) -> dict:
    """
    Reads a file from disk and converts it into a LangChain content block
    that Gemini can process.
    
    How it works:
    1. Read the raw bytes from the file
    2. Determine the MIME type from the file extension
    3. Base64-encode the bytes (because the API expects text, not binary)
    4. Wrap it in the dict format LangChain expects:
       {"type": "image_url", "image_url": {"url": "data:<mime>;base64,<data>"}}
    
    Args:
        file_path: Path to the image or audio file
        mime_map: Dictionary mapping extensions to MIME types
    
    Returns:
        A dict that LangChain will include in HumanMessage.content
    """
    path = Path(file_path)
    raw_bytes = path.read_bytes()
    ext = path.suffix.lower()
    mime = mime_map.get(ext, "application/octet-stream")
    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    
    # LangChain uses the "image_url" type for ALL inline binary data
    # (images AND audio) when talking to Gemini. The MIME type in the
    # data URI tells Gemini what kind of content it actually is.
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


def build_multimodal_message(
    text: str,
    image_path: Optional[str] = None,
    audio_path: Optional[str] = None,
) -> HumanMessage:
    """
    Builds a LangChain HumanMessage that can contain text + image + audio.
    
    This is how we support the multimodal input from your design doc:
    - Text: the user's natural language description
    - Image: memes, photos, rough sketches, style references
    - Audio: music, voice narration with emotional tone
    
    LangChain's HumanMessage.content can be either:
    - A plain string (text only)
    - A list of content blocks (multimodal)
    
    We always use the list format for consistency.
    """
    content_blocks = []
    
    if image_path:
        content_blocks.append(
            _file_to_content_block(image_path, IMAGE_MIME_MAP)
        )
    
    if audio_path:
        content_blocks.append(
            _file_to_content_block(audio_path, AUDIO_MIME_MAP)
        )
    
    # Text always goes last (Gemini processes content blocks in order,
    # and having the text after media lets it reference what it "saw/heard")
    content_blocks.append({"type": "text", "text": text})
    
    return HumanMessage(content=content_blocks)


# ── Phase Manager ─────────────────────────────────────────────────────────────
# This is the core orchestration class. It manages:
#   1. Which phase we're currently in
#   2. Two separate LLM instances (same model, different system prompts)
#   3. A SHARED conversation history (so Phase 2 knows what was said in Phase 1)
#   4. Phase transitions (triggered by user commands or AI suggestions)
#
# Architecture decision: We use a SHARED history rather than separate histories
# because the FigJam design shows that Phase 2 needs the emotional context
# gathered in Phase 1. The system prompt changes, but the conversation context
# carries over.

@dataclass
class PhaseManager:
    """
    Manages the two-phase conversation flow.
    
    Attributes:
        current_phase: Which phase we're in (EMOTIONAL_DISCOVERY or CODE_GENERATION)
        history: The shared conversation history (list of LangChain messages)
        model_name: Which Gemini model to use
        _phase_1_llm: LangChain chat model configured for Phase 1
        _phase_2_llm: LangChain chat model configured for Phase 2
    """
    current_phase: Phase = Phase.EMOTIONAL_DISCOVERY
    history: list = field(default_factory=list)
    model_name: str = "gemini-2.5-flash"
    
    # These get initialized in __post_init__
    _phase_1_llm: Optional[ChatGoogleGenerativeAI] = field(
        default=None, init=False, repr=False
    )
    _phase_2_llm: Optional[ChatGoogleGenerativeAI] = field(
        default=None, init=False, repr=False
    )
    
    def __post_init__(self):
        """
        Called automatically after __init__ (that's what dataclass does).
        
        We create TWO ChatGoogleGenerativeAI instances. They both talk to
        the same Gemini model, but conceptually they serve different roles.
        
        Why not just one model with a changing system prompt? Because
        LangChain makes it clean to have separate instances, and it avoids
        any risk of system prompt contamination between phases.
        """
        self._phase_1_llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0.8,  # Slightly creative for emotional exploration
        )
        self._phase_2_llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0.4,  # More precise for code generation
        )
    
    @property
    def _active_llm(self) -> ChatGoogleGenerativeAI:
        """Returns whichever LLM is appropriate for the current phase."""
        if self.current_phase == Phase.EMOTIONAL_DISCOVERY:
            return self._phase_1_llm
        return self._phase_2_llm
    
    @property
    def _system_message(self) -> SystemMessage:
        """Returns the system prompt for the current phase."""
        if self.current_phase == Phase.EMOTIONAL_DISCOVERY:
            return SystemMessage(content=PHASE_1_SYSTEM_PROMPT)
        return SystemMessage(content=PHASE_2_SYSTEM_PROMPT)
    
    def switch_phase(self, new_phase: Phase) -> str:
        """
        Switches to a different phase.
        
        This is the mechanism that enables the bidirectional flow from
        your FigJam: the user can go from Phase 2 back to Phase 1 to
        re-clarify emotions, and from Phase 1 forward to Phase 2.
        
        The history is preserved so context isn't lost.
        """
        old_phase = self.current_phase
        self.current_phase = new_phase
        return (
            f"[Switched from {old_phase.name} → {new_phase.name}]\n"
            f"{'Lets revisit your emotions.' if new_phase == Phase.EMOTIONAL_DISCOVERY else 'Lets start building your sketch!'}"
        )
    
    def send(
        self,
        text: str,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> str:
        """
        Sends a message to the appropriate LLM based on the current phase.
        
        Flow:
        1. Check if the user wants to switch phases (keyword detection)
        2. Build the multimodal message
        3. Add it to the shared history
        4. Construct the full message list: [SystemMessage, ...history]
        5. Send to the active LLM
        6. Append the AI response to history
        7. Return the response text
        
        The [SystemMessage, ...history] pattern is important:
        - The SystemMessage tells Gemini its current role/instructions
        - The history gives it all prior conversation context
        - This means when we switch phases, Gemini gets NEW instructions
          but still remembers everything that was discussed
        """
        # ── Phase transition detection ──
        # These keywords map to your FigJam's flow arrows between sub-goals
        lower = text.lower().strip()
        
        if lower in ("let's code", "lets code", "start coding", "implement this"):
            transition_msg = self.switch_phase(Phase.CODE_GENERATION)
            # We still send the user's message so the AI knows to start coding
            print(f"\n{'='*60}")
            print(transition_msg)
            print(f"{'='*60}\n")
        
        elif lower in ("back to feelings", "back to emotions", "redo emotions", "re-clarify"):
            transition_msg = self.switch_phase(Phase.EMOTIONAL_DISCOVERY)
            print(f"\n{'='*60}")
            print(transition_msg)
            print(f"{'='*60}\n")
        
        # ── Build and store the user message ──
        user_message = build_multimodal_message(text, image_path, audio_path)
        self.history.append(user_message)
        
        # ── Construct the full prompt ──
        # This is the key LangChain pattern:
        # [SystemMessage (phase instructions), ...all prior messages]
        messages = [self._system_message] + self.history
        
        # ── Call the LLM ──
        response = self._active_llm.invoke(messages)
        
        # ── Store the AI response ──
        # response is an AIMessage object; we append it to history
        self.history.append(response)
        
        return response.content
    
    def reset(self):
        """Clears history and resets to Phase 1 — like starting a new session."""
        self.history.clear()
        self.current_phase = Phase.EMOTIONAL_DISCOVERY


# ── CLI Interface ─────────────────────────────────────────────────────────────
# This is a simple terminal interface for testing. In your final product,
# this would be replaced by the React frontend from your app dev plan.

def print_help():
    """Prints available commands."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Commands:                                                   ║
║  • Type normally to chat                                     ║
║  • "let's code"       → switch to Code Generation (Phase 2)  ║
║  • "back to feelings" → switch to Emotional Discovery (Ph 1) ║
║  • "phase"            → show current phase                   ║
║  • "reset"            → start over                           ║
║  • "help"             → show this message                    ║
║  • "quit"             → exit                                 ║
╚══════════════════════════════════════════════════════════════╝
    """)


def main():
    """
    Main CLI loop.
    
    This implements the coordination structure from your FigJam:
    1. User expresses intention
    2. AI asks clarifying/guiding questions
    3. AI generates implementation (in Phase 2)
    4. User provides feedback
    5. AI adjusts selectively
    """
    manager = PhaseManager()
    
    print("\n🎨 Pretzel-Flavored HumANS — p5.js Creative Coding Assistant")
    print("━" * 60)
    print(f"Current phase: {manager.current_phase.name}")
    print("Let's start by exploring the emotion you want to express.\n")
    print_help()
    
    while True:
        # Show which phase we're in as part of the prompt
        phase_label = (
            "🌊 Emotion" if manager.current_phase == Phase.EMOTIONAL_DISCOVERY
            else "💻 Code"
        )
        user_input = input(f"[{phase_label}] You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye! 🎨")
            break
        
        if user_input.lower() == "help":
            print_help()
            continue
        
        if user_input.lower() == "phase":
            print(f"  → Currently in: {manager.current_phase.name}")
            continue
        
        if user_input.lower() == "reset":
            manager.reset()
            print("  → Reset! Starting fresh in Emotional Discovery.\n")
            continue
        
        # Check for optional multimodal input
        image = input("  📷 Image path (Enter to skip): ").strip() or None
        audio = input("  🎵 Audio path (Enter to skip): ").strip() or None
        
        # Validate file paths
        if image and not Path(image).exists():
            print(f"  ⚠ Image file not found: {image}")
            image = None
        if audio and not Path(audio).exists():
            print(f"  ⚠ Audio file not found: {audio}")
            audio = None
        
        # Send to the appropriate phase LLM
        print("\n" + "─" * 60)
        try:
            reply = manager.send(user_input, image_path=image, audio_path=audio)
            print(f"AI: {reply}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        print("─" * 60 + "\n")


if __name__ == "__main__":
    main()