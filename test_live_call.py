#!/usr/bin/env python3
"""
Live Voice AI Call Simulator
==============================
Connects to Gemini Live via raw WebSocket (same protocol as the server)
and runs a full onboarding conversation with real voice + transcription.

Uses ONLY websockets (already installed on server) — no google-genai needed.

Usage:
  python3 test_live_call.py                   # Auto mode (scripted caller)
  python3 test_live_call.py --interactive     # You type the responses
  python3 test_live_call.py --prompt file.txt # Custom prompt
  python3 test_live_call.py --name "Riya"     # Different customer name

Requires: websockets  (already installed — server uses it)
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# ── Load .env ───────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
env_file = SCRIPT_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))

# ── Config ──────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
CUSTOMER_NAME = "Kiran"
INTERACTIVE = "--interactive" in sys.argv or "-i" in sys.argv
PROMPT_FILE = "riddhi_prompt_v2.txt"
MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"
VOICE = "Kore"

for i, arg in enumerate(sys.argv):
    if arg == "--prompt" and i + 1 < len(sys.argv):
        PROMPT_FILE = sys.argv[i + 1]
    if arg == "--name" and i + 1 < len(sys.argv):
        CUSTOMER_NAME = sys.argv[i + 1]
    if arg == "--voice" and i + 1 < len(sys.argv):
        VOICE = sys.argv[i + 1]

# ── Colors ──────────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
D = "\033[90m"; BOLD = "\033[1m"; END = "\033[0m"

# ── Scripted user responses ─────────────────────────────────────────────
SCRIPTED = [
    "English please",
    "Yes, I'm free right now",
    "Okay, I'm on speaker now",
    "Yes I have the app downloaded",
    "It's open now",
    "Done, I clicked on Courses",
    "Yes the video is playing",
    "Yes that makes sense",
    "I've opened the WhatsApp message",
    "Yes I've joined the orientation group",
    "Yes joined that one too",
    "Done, I joined the mission team group",
    "Alright got it",
    "Yes I can see their names",
    "Sunita and Meera",
    "Yes I'm following everything",
    "Yes that sounds great",
    "Thank you so much!",
    "Yes got it",
    "Okay thanks!",
]


def load_prompt():
    path = SCRIPT_DIR / PROMPT_FILE
    if not path.exists():
        print(f"{R}Prompt file not found: {path}{END}")
        sys.exit(1)
    prompt = path.read_text()
    for key, val in {"customer_name": CUSTOMER_NAME, "membership_type": "Gold",
                     "purchase_date": "2026-02-26", "language_preference": "English",
                     "city": "Mumbai", "lead_source": "Instagram"}.items():
        prompt = prompt.replace("{{" + key + "}}", val).replace("{" + key + "}", val)
    # Same additions the real server makes
    prompt += (
        "\n\n[SPEECH STYLE: Maintain a consistent speaking voice — same warmth, tone, accent "
        "throughout the entire call. Never suddenly change your speaking style mid-conversation.]"
        "\n\n[CONVERSATION RULE: NEVER repeat a question the customer already answered. "
        "Each question should be asked exactly ONCE.]"
    )
    return prompt


# ── Gemini Live raw WebSocket class ─────────────────────────────────────
class GeminiLiveSession:
    """Raw WebSocket connection to Gemini Live — same protocol as plivo_gemini_stream.py."""

    def __init__(self, api_key, model, voice, prompt):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.prompt = prompt
        self.ws = None

    async def connect(self):
        import websockets
        url = (
            "wss://generativelanguage.googleapis.com/ws/"
            "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
            f"?key={self.api_key}"
        )
        self.ws = await websockets.connect(url, ping_interval=30, ping_timeout=20)

        # Send setup message — exact same structure as the server
        setup = {
            "setup": {
                "model": self.model,
                "generation_config": {
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": self.voice
                            }
                        }
                    },
                    "thinking_config": {
                        "thinking_budget": 128
                    }
                },
                "input_audio_transcription": {},
                "output_audio_transcription": {},
                "system_instruction": {"parts": [{"text": self.prompt}]},
                "tools": [{
                    "function_declarations": [{
                        "name": "end_call",
                        "description": "End the phone call",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "reason": {"type": "string", "description": "Why the call is ending"}
                            },
                            "required": ["reason"]
                        }
                    }]
                }]
            }
        }
        await self.ws.send(json.dumps(setup))

        # Wait for setupComplete
        async for msg in self.ws:
            resp = json.loads(msg)
            if "setupComplete" in resp:
                return
            # Ignore other setup messages

    async def send_text(self, text):
        """Send user text via client_content (same as server's nudge messages)."""
        msg = {
            "client_content": {
                "turns": [{"role": "user", "parts": [{"text": text}]}],
                "turn_complete": True
            }
        }
        await self.ws.send(json.dumps(msg))

    async def receive_turn(self, timeout=30):
        """Receive a full agent turn. Returns (transcript, total_ms, ttfb_ms, audio_kb)."""
        t0 = time.time()
        transcript_parts = []
        ttfb = None
        audio_bytes = 0
        tool_end_call = False

        try:
            async for msg_raw in self.ws:
                elapsed = time.time() - t0
                if elapsed > timeout:
                    break

                resp = json.loads(msg_raw)
                sc = resp.get("serverContent")
                if sc:
                    # Output transcription
                    ot = sc.get("outputTranscription")
                    if ot and isinstance(ot, dict):
                        text = ot.get("text", "").strip()
                        if text:
                            transcript_parts.append(text)

                    # Audio data (for TTFB tracking)
                    mt = sc.get("modelTurn", {})
                    for part in mt.get("parts", []):
                        idata = part.get("inlineData", {})
                        if idata.get("data"):
                            audio_bytes += len(idata["data"])
                            if ttfb is None:
                                ttfb = (time.time() - t0) * 1000

                    # Turn complete
                    if sc.get("turnComplete"):
                        break

                # Handle tool calls
                tc = resp.get("toolCall")
                if tc and tc.get("functionCalls"):
                    responses = []
                    for fc in tc["functionCalls"]:
                        responses.append({
                            "id": fc["id"],
                            "name": fc["name"],
                            "response": {"result": {"success": True}}
                        })
                        if fc["name"] == "end_call":
                            tool_end_call = True
                    await self.ws.send(json.dumps({
                        "tool_response": {"function_responses": responses}
                    }))
                    if tool_end_call:
                        break

        except Exception as e:
            if not transcript_parts:
                transcript_parts.append(f"[ERROR: {e}]")

        total_ms = (time.time() - t0) * 1000
        text = " ".join(transcript_parts).strip()
        if tool_end_call:
            text += " [end_call]"
        return text, total_ms, ttfb, audio_bytes / 1024

    async def close(self):
        if self.ws:
            await self.ws.close()


# ── Main ────────────────────────────────────────────────────────────────
async def main():
    if not GOOGLE_API_KEY:
        print(f"{R}ERROR: GOOGLE_API_KEY not set. Add to .env or export it.{END}")
        sys.exit(1)

    try:
        import websockets  # noqa: F401
    except ImportError:
        print(f"{R}ERROR: websockets not installed. Run:{END}")
        print(f"  pip3 install websockets")
        sys.exit(1)

    prompt = load_prompt()

    print(f"\n{'=' * 65}")
    print(f"{BOLD}  LIVE VOICE CALL — Priya (AI) ↔ {CUSTOMER_NAME} (Caller){END}")
    print(f"{'=' * 65}")
    print(f"  Model: {MODEL.split('/')[-1]}")
    print(f"  Voice: {VOICE}")
    print(f"  Mode:  {'Interactive (you type)' if INTERACTIVE else 'Auto (scripted)'}")
    print(f"{'=' * 65}")

    session = GeminiLiveSession(GOOGLE_API_KEY, MODEL, VOICE, prompt)

    print(f"\n  {D}Connecting to Gemini Live...{END}", end="", flush=True)
    t_conn = time.time()
    try:
        await session.connect()
    except Exception as e:
        print(f"\n  {R}Connection failed: {e}{END}")
        sys.exit(1)
    print(f" {G}ready{END} ({(time.time()-t_conn)*1000:.0f}ms)\n")

    # ── Conversation loop ───────────────────────────────────────────
    turn_num = 0
    resp_idx = 0
    call_ended = False
    total_audio_kb = 0

    # Greeting trigger
    trigger = (
        f"[Start the conversation now. Greet {CUSTOMER_NAME} naturally "
        f"using your opening line from the instructions.]"
    )
    await session.send_text(trigger)
    text, total_ms, ttfb, audio_kb = await session.receive_turn()
    turn_num += 1
    total_audio_kb += audio_kb
    lat = f"TTFB {ttfb:.0f}ms" if ttfb else f"{total_ms:.0f}ms"
    print(f"  {B}Priya:{END}  {text or '[no transcript]'}")
    print(f"  {D}({lat} | {audio_kb:.0f}KB audio){END}\n")

    while not call_ended:
        # Get user input
        if INTERACTIVE:
            try:
                user_input = input(f"  {D}You:{END}  ")
                if not user_input or user_input.lower() in ("quit", "exit", "bye"):
                    break
                print()
            except (EOFError, KeyboardInterrupt):
                print()
                break
        else:
            if resp_idx >= len(SCRIPTED):
                extras = ["Yes", "Okay", "Alright thanks"]
                extra_idx = resp_idx - len(SCRIPTED)
                if extra_idx >= len(extras):
                    print(f"  {Y}[All responses sent]{END}")
                    break
                user_input = extras[extra_idx]
            else:
                user_input = SCRIPTED[resp_idx]
            resp_idx += 1
            print(f"  {D}{CUSTOMER_NAME}:{END}  {user_input}\n")

        await session.send_text(user_input)
        text, total_ms, ttfb, audio_kb = await session.receive_turn()
        turn_num += 1
        total_audio_kb += audio_kb
        lat = f"TTFB {ttfb:.0f}ms" if ttfb else f"{total_ms:.0f}ms"

        if not text:
            print(f"  {Y}[empty response]{END}\n")
            continue

        print(f"  {B}Priya:{END}  {text}")
        print(f"  {D}({lat} | {audio_kb:.0f}KB audio){END}\n")

        lower = text.lower()
        if any(w in lower for w in ["bye-bye", "bye bye", "beautiful day", "[end_call]"]):
            call_ended = True

    await session.close()

    # Summary
    status = f"{G}COMPLETED{END}" if call_ended else f"{Y}INCOMPLETE{END}"
    print(f"{'=' * 65}")
    print(f"  Status: {status}")
    print(f"  Turns:  {turn_num}")
    print(f"  Audio:  {total_audio_kb:.0f}KB total")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)
    asyncio.run(main())
