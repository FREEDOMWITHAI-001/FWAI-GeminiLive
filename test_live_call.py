#!/usr/bin/env python3
"""
Live Voice AI Call Simulator
==============================
Connects to Gemini Live (same model + prompt as your server) and runs a
full onboarding conversation with real voice output + transcription.

Behaves exactly like a real caller talking to Priya.

Usage:
  python3 test_live_call.py                   # Auto mode (scripted caller)
  python3 test_live_call.py --interactive     # You type the responses
  python3 test_live_call.py --prompt file.txt # Custom prompt
  python3 test_live_call.py --name "Riya"     # Different customer name

Requires: pip3 install google-genai
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# ── Load .env (no dotenv dependency) ────────────────────────────────────
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


# ── Load prompt ─────────────────────────────────────────────────────────
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
    return prompt


# ── Main ────────────────────────────────────────────────────────────────
async def main():
    if not GOOGLE_API_KEY:
        print(f"{R}ERROR: GOOGLE_API_KEY not set. Add to .env or export it.{END}")
        sys.exit(1)

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print(f"{R}ERROR: google-genai not installed. Run:{END}")
        print(f"  pip3 install google-genai")
        sys.exit(1)

    prompt = load_prompt()

    print(f"\n{'=' * 65}")
    print(f"{BOLD}  LIVE VOICE CALL — Priya (AI) ↔ {CUSTOMER_NAME} (Caller){END}")
    print(f"{'=' * 65}")
    print(f"  Model: {MODEL.split('/')[-1]}")
    print(f"  Voice: {VOICE}")
    print(f"  Mode:  {'Interactive (you type)' if INTERACTIVE else 'Auto (scripted)'}")
    print(f"{'=' * 65}")

    client = genai.Client(api_key=GOOGLE_API_KEY)
    config = types.LiveConnectConfig(
        responseModalities=["AUDIO"],
        speechConfig=types.SpeechConfig(
            voiceConfig=types.VoiceConfig(
                prebuiltVoiceConfig=types.PrebuiltVoiceConfig(voiceName=VOICE)
            )
        ),
        thinkingConfig=types.ThinkingConfig(thinkingBudget=128),
        systemInstruction=prompt,
        inputAudioTranscription=types.AudioTranscriptionConfig(),
        outputAudioTranscription=types.AudioTranscriptionConfig(),
    )

    print(f"\n  {D}Connecting to Gemini Live...{END}", end="", flush=True)
    t_connect = time.time()

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        connect_ms = (time.time() - t_connect) * 1000
        print(f" {G}connected{END} ({connect_ms:.0f}ms)\n")

        # ── Helper: send text, collect full response ────────────────
        async def agent_turn(user_text, timeout=30):
            """Send user text via client_content → collect agent audio + transcript."""
            t0 = time.time()
            await session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=user_text)]),
                turn_complete=True,
            )
            transcript_parts = []
            ttfb = None
            audio_bytes = 0

            async for msg in session.receive():
                elapsed = time.time() - t0
                if elapsed > timeout:
                    break

                sc = msg.server_content
                if sc:
                    # Collect transcription fragments
                    if sc.output_transcription and sc.output_transcription.text:
                        frag = sc.output_transcription.text.strip()
                        if frag:
                            transcript_parts.append(frag)

                    # Track audio for TTFB
                    if sc.model_turn and sc.model_turn.parts:
                        for part in sc.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                audio_bytes += len(part.inline_data.data)
                                if ttfb is None:
                                    ttfb = (time.time() - t0) * 1000

                    if sc.turn_complete:
                        break

                # Handle tool calls (e.g. end_call)
                tc = msg.tool_call
                if tc and tc.function_calls:
                    responses = []
                    is_end = False
                    for fc in tc.function_calls:
                        responses.append(types.FunctionResponse(
                            id=fc.id, name=fc.name, response={"success": True}
                        ))
                        if fc.name == "end_call":
                            is_end = True
                    await session.send_tool_response(function_responses=responses)
                    if is_end:
                        text = " ".join(transcript_parts).strip()
                        return text + " [end_call]", (time.time()-t0)*1000, ttfb, audio_bytes

            total_ms = (time.time() - t0) * 1000
            text = " ".join(transcript_parts).strip()
            return text, total_ms, ttfb, audio_bytes

        # ── Conversation ────────────────────────────────────────────
        turn_num = 0
        resp_idx = 0
        call_ended = False
        total_audio = 0

        # Greeting
        trigger = f"[Start the conversation now. Greet {CUSTOMER_NAME} naturally using your opening line from the instructions.]"
        agent_text, total_ms, ttfb, audio_b = await agent_turn(trigger)
        turn_num += 1
        total_audio += audio_b
        latency = f"TTFB {ttfb:.0f}ms" if ttfb else f"{total_ms:.0f}ms"

        print(f"  {B}Priya:{END}  {agent_text or '[no transcript]'}")
        print(f"  {D}({latency} | {audio_b/1024:.0f}KB audio){END}\n")

        # Main loop
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
                    # Extra nudges
                    extras = ["Yes", "Okay", "Alright thanks"]
                    extra_idx = resp_idx - len(SCRIPTED)
                    if extra_idx >= len(extras):
                        print(f"  {Y}[All responses sent — ending]{END}")
                        break
                    user_input = extras[extra_idx]
                else:
                    user_input = SCRIPTED[resp_idx]
                resp_idx += 1
                print(f"  {D}{CUSTOMER_NAME}:{END}  {user_input}\n")

            # Send and get response
            agent_text, total_ms, ttfb, audio_b = await agent_turn(user_input)
            turn_num += 1
            total_audio += audio_b
            latency = f"TTFB {ttfb:.0f}ms" if ttfb else f"{total_ms:.0f}ms"

            if not agent_text:
                print(f"  {Y}[empty response — nudging]{END}\n")
                continue

            print(f"  {B}Priya:{END}  {agent_text}")
            print(f"  {D}({latency} | {audio_b/1024:.0f}KB audio){END}\n")

            # Check for call end
            lower = agent_text.lower()
            if any(w in lower for w in ["bye-bye", "bye bye", "beautiful day", "[end_call]"]):
                call_ended = True

        # ── Summary ─────────────────────────────────────────────────
        print(f"{'=' * 65}")
        status = f"{G}COMPLETED{END}" if call_ended else f"{Y}INCOMPLETE{END}"
        print(f"  Status: {status}")
        print(f"  Turns:  {turn_num}")
        print(f"  Audio:  {total_audio/1024:.0f}KB total")
        print(f"{'=' * 65}\n")


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)
    asyncio.run(main())
