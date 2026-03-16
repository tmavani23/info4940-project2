#!/bin/bash
# ============================================================================
# Pretzel-Flavored HumANS — curl Testing Script
# ============================================================================
#
# HOW TO USE:
#   Terminal 1:  python server.py          (starts the Flask server)
#   Terminal 2:  bash test_chat.sh         (runs this script)
#
# OR: copy-paste individual curl commands one at a time for a manual
#     back-and-forth conversation.
#
# The script uses a fixed SESSION_ID so all messages share the same
# conversation history — just like a real chat session.
# ============================================================================

SESSION="test-session-001"
API="http://localhost:5000"

echo "============================================"
echo "  Phase 1: Emotional Discovery"
echo "============================================"
echo ""

# ── Message 1: User initiates the conversation ──
# This is the first step in your FigJam flow:
#   "User initiates → AI asks emotional discovery questions"
echo ">>> USER: I want to make a sketch about my first time leaving home for college"
echo ""
curl -s -X POST "$API/api/chat" \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION\",
    \"message\": \"I want to make a sketch about my first time leaving home for college\"
  }" | python3 -m json.tool
echo ""
echo "────────────────────────────────────────────"
echo ""

# Give yourself time to read the response
read -p "Press Enter to send next message..."

# ── Message 2: User responds to AI's clarifying question ──
# The AI should have asked about the emotion/setting.
# Now the user shares more emotional context.
echo ""
echo ">>> USER: It was bittersweet — excited but also really scared and lonely"
echo ""
curl -s -X POST "$API/api/chat" \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION\",
    \"message\": \"It was bittersweet. I was excited but also really scared and lonely. The drive was 6 hours and I watched my hometown disappear in the rearview mirror. Emojis: 😢🌅🚗\"
  }" | python3 -m json.tool
echo ""
echo "────────────────────────────────────────────"
echo ""

read -p "Press Enter to send next message..."

# ── Message 3: User confirms the emotional profile ──
# By now the AI should have synthesized an emotional profile
# and asked "does this feel right?"
echo ""
echo ">>> USER: Yes that feels right, let's go"
echo ""
curl -s -X POST "$API/api/chat" \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION\",
    \"message\": \"Yes that feels right. I like the idea of fading colors and slow movement. Let's go.\"
  }" | python3 -m json.tool
echo ""
echo "────────────────────────────────────────────"
echo ""

read -p "Press Enter to switch to Phase 2..."

echo ""
echo "============================================"
echo "  Switching to Phase 2: Code Generation"
echo "============================================"
echo ""

# ── Message 4: Trigger phase switch ──
# "let's code" is a keyword that triggers PhaseManager to switch
# to CODE_GENERATION. The system prompt changes but the history
# (including all emotional context) carries over.
echo ">>> USER: let's code"
echo ""
curl -s -X POST "$API/api/chat" \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION\",
    \"message\": \"let's code\"
  }" | python3 -m json.tool
echo ""
echo "────────────────────────────────────────────"
echo ""

read -p "Press Enter to send feedback on the code..."

echo ""
echo "============================================"
echo "  Iterating on the Code (still Phase 2)"
echo "============================================"
echo ""

# ── Message 5: User gives feedback on the generated sketch ──
# This is the iteration loop from your FigJam:
#   "User provides feedback → AI adjusts selectively"
echo ">>> USER: The motion is too fast, and can we make the colors warmer?"
echo ""
curl -s -X POST "$API/api/chat" \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION\",
    \"message\": \"The motion is too fast and feels frantic rather than bittersweet. Can we slow it down? Also I want warmer colors — ambers and deep oranges instead of blues.\"
  }" | python3 -m json.tool
echo ""
echo "────────────────────────────────────────────"
echo ""

read -p "Press Enter to check session status..."

# ── Check session status ──
echo ""
echo ">>> Checking session status..."
echo ""
curl -s "$API/api/status?session_id=$SESSION" | python3 -m json.tool
echo ""
echo "────────────────────────────────────────────"
echo ""

read -p "Press Enter to go back to Phase 1..."

echo ""
echo "============================================"
echo "  Going Back to Phase 1 (re-clarify emotions)"
echo "============================================"
echo ""

# ── Message 6: User wants to re-explore emotions ──
# This triggers the bidirectional flow in your FigJam:
#   Sub-Goal #2 → back to Sub-Goal #1
echo ">>> USER: back to feelings"
echo ""
curl -s -X POST "$API/api/chat" \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION\",
    \"message\": \"back to feelings\"
  }" | python3 -m json.tool
echo ""
echo "────────────────────────────────────────────"
echo ""

echo "============================================"
echo "  Done! Full conversation is stored in"
echo "  session: $SESSION"
echo "============================================"
