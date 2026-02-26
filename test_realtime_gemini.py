#!/usr/bin/env python3
"""
REAL-TIME Gemini Integration Test (Audio + Latency)
=====================================================
Connects to the ACTUAL Gemini native audio model via Live WebSocket API
(same model & endpoint as production) to test the full onboarding flow.

Sends user text via client_content (same as production greeting/nudge triggers)
and receives AUDIO responses. Measures real TTFB and validates transcriptions.

Validates:
  - TTFB (time to first audio byte) per turn
  - Total response time per turn
  - Latency trend across turns (KV cache degradation)
  - Repetition (via outputTranscription, same logic as production)
  - Step combining (multiple steps in one turn)
  - Language consistency
  - Full flow completion through Phase 6

Usage:
  python3 test_realtime_gemini.py
  python3 test_realtime_gemini.py --verbose
  python3 test_realtime_gemini.py --language hindi
  python3 test_realtime_gemini.py --text-only    # Use text model (no audio latency)
"""

import asyncio
import base64
import json
import os
import re
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Configuration
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CUSTOMER_NAME = "Kiran"
VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv
LANGUAGE = "Hindi" if "--language" in sys.argv and "hindi" in " ".join(sys.argv).lower() else "English"
TEXT_ONLY = "--text-only" in sys.argv

# Model selection
AUDIO_MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"
TEXT_MODEL = "gemini-2.0-flash"
VOICE = "Kore"

# Gemini Live WebSocket (same as production)
WS_URL = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={GOOGLE_API_KEY}"

# Latency thresholds
TTFB_WARN_MS = 1500   # Warn if TTFB > 1.5s
TTFB_FAIL_MS = 3000   # Fail if TTFB > 3s
TOTAL_WARN_MS = 5000   # Warn if total > 5s
TOTAL_FAIL_MS = 10000  # Fail if total > 10s

# Colors
class C:
    OK = "\033[92m"
    FAIL = "\033[91m"
    WARN = "\033[93m"
    BLUE = "\033[94m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    END = "\033[0m"


# ============================================================
# Prompt Loader
# ============================================================

def load_prompt():
    with open("riddhi_prompt_v2.txt") as f:
        prompt = f.read()
    replacements = {
        "customer_name": CUSTOMER_NAME,
        "membership_type": "Gold",
        "purchase_date": "2026-02-26",
        "language_preference": LANGUAGE,
        "city": "Mumbai",
        "lead_source": "Instagram",
    }
    for key, value in replacements.items():
        prompt = prompt.replace("{{" + key + "}}", value)
        prompt = prompt.replace("{" + key + "}", value)
    prompt += (
        "\n\n[SPEECH STYLE: Vary your pace, pitch, and energy naturally throughout the conversation. "
        "Speak faster when excited, slower when empathetic. Use pauses for emphasis. "
        "Match the customer's energy level. "
        "IMPORTANT: Maintain a consistent speaking voice — same warmth, same tone, same accent, same tempo baseline throughout the entire call. "
        "Never suddenly change your speaking style, speed, or personality mid-conversation.]"
    )
    prompt += (
        "\n\n[CONVERSATION RULE: NEVER repeat a question the customer already answered. "
        "If you catch yourself about to re-ask something, skip it and move to the next topic. "
        "Each question should be asked exactly ONCE.]"
    )
    return prompt


# ============================================================
# Step Parser
# ============================================================

def parse_prompt_steps(prompt):
    steps = []
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                  'and', 'or', 'but', 'so', 'if', 'it', 'its', 'this', 'that',
                  'you', 'your', 'i', 'me', 'my', 'we', 'our', 'can', 'will',
                  'do', 'have', 'has', 'had', 'would', 'could', 'should',
                  'please', 'just', 'about', 'once', 'also', 'not', 'all'}
    step_blocks = list(re.finditer(r'STEP\s+(\d+\.\d+)\b', prompt))
    for i, m in enumerate(step_blocks):
        step_id = m.group(1)
        start = m.start()
        end = step_blocks[i + 1].start() if i + 1 < len(step_blocks) else len(prompt)
        block_text = prompt[start:end]
        dialogue_match = re.search(r'"([^"]{10,})"', block_text)
        if not dialogue_match:
            continue
        dialogue = dialogue_match.group(1).strip()
        block_after = block_text[dialogue_match.end():]
        step_type = "pause_continue" if "PAUSE & CONTINUE" in block_after else "wait"
        words = re.findall(r'[a-z]+', dialogue.lower())
        keywords = set(w for w in words if w not in stop_words and len(w) > 2)
        word_list = dialogue.lower().split()
        phrases = []
        for j in range(len(word_list) - 1):
            bigram = f"{word_list[j]} {word_list[j+1]}"
            if len(bigram) > 8 and not all(w in stop_words for w in bigram.split()):
                phrases.append(bigram)
        first_sentence = re.split(r'[.?!]', dialogue)[0].strip()
        label = f"Step {step_id}: {first_sentence[:35]}"
        steps.append({
            "step_id": step_id, "label": label, "keywords": keywords,
            "phrases": phrases[:6], "dialogue": dialogue, "type": step_type,
        })
    return steps


def match_step(text, steps):
    if not steps:
        return "", ""
    best_score = 0
    best_label = ""
    best_id = ""
    text_words = set(re.findall(r'[a-z]+', text.lower()))
    for step in steps:
        if not step["keywords"]:
            continue
        kw_overlap = len(text_words & step["keywords"]) / len(step["keywords"])
        phrase_bonus = sum(1 for p in step["phrases"] if p in text.lower()) * 0.15
        score = kw_overlap + phrase_bonus
        if score > best_score and score > 0.3:
            best_score = score
            best_label = step["label"]
            best_id = step["step_id"]
    return best_id, best_label


# ============================================================
# Validators
# ============================================================

def is_duplicate_text(new_text, recent_texts, completed_steps, parsed_steps):
    new_words = set(new_text.lower().split())
    if len(new_words) < 3:
        return False, ""
    for prev in recent_texts:
        prev_words = set(prev.lower().split())
        if not prev_words:
            continue
        intersection = len(new_words & prev_words)
        if len(new_words) < 12:
            overlap = intersection / len(new_words)
        else:
            overlap = intersection / max(len(new_words), len(prev_words))
        if overlap > 0.5:
            return True, f"overlaps with: '{prev[:60]}'"
    _, new_label = match_step(new_text, parsed_steps)
    if new_label and len(new_label) > 15:
        for step_label in completed_steps:
            step_words = set(step_label.lower().split())
            label_words = set(new_label.lower().split())
            if step_words and label_words:
                overlap = len(step_words & label_words) / max(len(step_words), len(label_words))
                if overlap > 0.5:
                    return True, f"matches step: '{step_label}'"
    return False, ""


def check_step_combining(text, parsed_steps):
    matched = []
    text_lower = text.lower()
    text_words = set(re.findall(r'[a-z]+', text_lower))
    for step in parsed_steps:
        if not step["keywords"]:
            continue
        kw_overlap = len(text_words & step["keywords"]) / len(step["keywords"])
        phrase_hits = sum(1 for p in step["phrases"] if p in text_lower)
        score = kw_overlap + phrase_hits * 0.15
        if score > 0.5 and phrase_hits >= 1:
            matched.append(step["step_id"])
    wait_steps = [sid for sid in matched
                  if any(s["step_id"] == sid and s["type"] == "wait" for s in parsed_steps)]
    return matched, wait_steps


def check_language(text, expected_lang):
    if expected_lang == "English":
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        if hindi_chars > 5:
            return False, f"{hindi_chars} Hindi chars in English response"
    return True, ""


# ============================================================
# User Responses
# ============================================================

USER_RESPONSES = [
    ("language_choice", f"{LANGUAGE} please", 1),
    ("availability", "Yes, I'm free right now", 1),
    ("speaker", "Okay, I'm on speaker now", 1),
    ("app_downloaded", "Yes I have the app downloaded", 2),
    ("app_open", "It's open now", 2),
    ("courses_clicked", "Done, I clicked on Courses", 2),
    ("video_playing", "Yes the video is playing", 2),
    ("course_understood", "Yes that makes sense", 2),
    ("whatsapp_opened", "I've opened the WhatsApp message", 3),
    ("orientation_joined", "Yes I've joined the orientation group", 3),
    ("main_group_joined", "Yes joined that one too", 3),
    ("mission_team_joined", "Done, I joined the mission team group", 3),
    ("groups_understood", "Alright got it", 3),
    ("coaches_visible", "Yes I can see their names", 4),
    ("coaches_names", "Sunita and Meera", 4),
    ("following_so_far", "Yes I'm following everything", 5),
    ("sounds_good", "Yes that sounds great", 5),
    ("final_response", "Thank you so much!", 6),
]


# ============================================================
# TEXT-ONLY MODE (google.generativeai SDK)
# ============================================================

def run_text_test():
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    prompt = load_prompt()
    parsed_steps = parse_prompt_steps(prompt)
    model = genai.GenerativeModel(model_name=TEXT_MODEL, system_instruction=prompt)
    chat = model.start_chat()

    results = {"errors": [], "warnings": [], "latencies": [], "turns": []}

    def send_and_validate(user_text, label, turn_num):
        t0 = time.time()
        response = chat.send_message(user_text)
        latency_ms = (time.time() - t0) * 1000
        agent_text = response.text.strip()
        results["latencies"].append({"turn": turn_num, "total_ms": latency_ms, "ttfb_ms": latency_ms})
        results["turns"].append({"user": user_text, "agent": agent_text, "latency_ms": latency_ms, "label": label})
        return agent_text, latency_ms

    return results, send_and_validate, parsed_steps


# ============================================================
# AUDIO MODE (WebSocket to native audio model)
# ============================================================

async def run_audio_test():
    import websockets

    prompt = load_prompt()
    parsed_steps = parse_prompt_steps(prompt)

    # State
    errors = []
    warnings = []
    agent_history = []
    recent_agent_texts = []
    completed_steps = []
    matched_step_ids = []
    phase_reached = set()
    turn_count = 0
    call_ended = False
    latency_data = []
    conversation_log = []

    def log_check(ok, msg, warn_only=False):
        if ok:
            print(f"    {C.OK}[OK]{C.END} {msg}")
        elif warn_only:
            warnings.append(f"Turn {turn_count}: {msg}")
            print(f"    {C.WARN}[WARN]{C.END} {msg}")
        else:
            errors.append(f"Turn {turn_count}: {msg}")
            print(f"    {C.FAIL}[FAIL]{C.END} {msg}")

    print(f"\n{'='*70}")
    print(f"{C.BOLD}REAL-TIME GEMINI TEST (NATIVE AUDIO + LATENCY){C.END}")
    print(f"{'='*70}")
    print(f"  Model:    {AUDIO_MODEL}")
    print(f"  Voice:    {VOICE}")
    print(f"  Customer: {CUSTOMER_NAME}")
    print(f"  Language: {LANGUAGE}")
    print(f"  Steps:    {len(parsed_steps)}")
    print(f"  Started:  {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")

    # Connect
    print(f"  {C.BLUE}Connecting to Gemini Live WebSocket...{C.END}")
    t_conn = time.time()
    ws = await websockets.connect(WS_URL, ping_interval=30, ping_timeout=20, close_timeout=5)
    print(f"  {C.OK}Connected ({(time.time()-t_conn)*1000:.0f}ms){C.END}")

    # Send setup (same as production)
    setup_msg = {
        "setup": {
            "model": AUDIO_MODEL,
            "generation_config": {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {"voice_name": VOICE}
                    }
                },
                "thinking_config": {"thinking_budget": 128}
            },
            "realtime_input_config": {
                "automatic_activity_detection": {
                    "disabled": False,
                    "start_of_speech_sensitivity": "START_SENSITIVITY_HIGH",
                    "end_of_speech_sensitivity": "END_SENSITIVITY_LOW",
                    "prefix_padding_ms": 20,
                    "silence_duration_ms": 500,
                }
            },
            "input_audio_transcription": {},
            "output_audio_transcription": {},
            "system_instruction": {"parts": [{"text": prompt}]},
        }
    }
    await ws.send(json.dumps(setup_msg))
    print(f"  {C.BLUE}Setup sent, waiting for setupComplete...{C.END}")

    # Wait for setupComplete
    async for raw in ws:
        resp = json.loads(raw)
        if "setupComplete" in resp:
            print(f"  {C.OK}Gemini native audio session ready!{C.END}\n")
            break

    # Helper: send user text (same format as production client_content)
    async def send_user_text(text):
        msg = {
            "client_content": {
                "turns": [{"role": "user", "parts": [{"text": text}]}],
                "turn_complete": True
            }
        }
        await ws.send(json.dumps(msg))

    # Helper: wait for agent response, measuring TTFB and collecting transcription
    async def wait_for_response(timeout=30):
        transcription_parts = []
        audio_bytes_total = 0
        ttfb_ms = None
        t_start = time.time()
        tool_called = None

        try:
            while True:
                elapsed = time.time() - t_start
                if elapsed > timeout:
                    break
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout - elapsed)
                resp = json.loads(raw)

                if "serverContent" in resp:
                    sc = resp["serverContent"]

                    # Collect outputTranscription
                    ot = sc.get("outputTranscription")
                    if ot:
                        text = ot.get("text", "") if isinstance(ot, dict) else str(ot)
                        if text.strip():
                            transcription_parts.append(text.strip())

                    # Track audio chunks for TTFB
                    if "modelTurn" in sc:
                        for part in sc["modelTurn"].get("parts", []):
                            if part.get("inlineData", {}).get("data"):
                                audio_data = base64.b64decode(part["inlineData"]["data"])
                                audio_bytes_total += len(audio_data)
                                if ttfb_ms is None:
                                    ttfb_ms = (time.time() - t_start) * 1000

                    if sc.get("turnComplete"):
                        total_ms = (time.time() - t_start) * 1000
                        text = " ".join(transcription_parts).strip()
                        return {
                            "text": text,
                            "ttfb_ms": ttfb_ms or total_ms,
                            "total_ms": total_ms,
                            "audio_bytes": audio_bytes_total,
                            "audio_duration_s": audio_bytes_total / (24000 * 2) if audio_bytes_total else 0,
                            "tool_call": tool_called,
                        }

                if "toolCall" in resp:
                    for fc in resp["toolCall"].get("functionCalls", []):
                        tool_called = fc.get("name", "")
                    # Respond to tool calls
                    tool_resp = {
                        "tool_response": {
                            "function_responses": [{
                                "id": fc.get("id", ""),
                                "name": fc.get("name", ""),
                                "response": {"success": True}
                            } for fc in resp["toolCall"].get("functionCalls", [])]
                        }
                    }
                    await ws.send(json.dumps(tool_resp))
                    if tool_called == "end_call":
                        total_ms = (time.time() - t_start) * 1000
                        text = " ".join(transcription_parts).strip()
                        return {
                            "text": text + " [end_call]",
                            "ttfb_ms": ttfb_ms or total_ms,
                            "total_ms": total_ms,
                            "audio_bytes": audio_bytes_total,
                            "audio_duration_s": audio_bytes_total / (24000 * 2) if audio_bytes_total else 0,
                            "tool_call": "end_call",
                        }

        except (asyncio.TimeoutError, Exception) as e:
            total_ms = (time.time() - t_start) * 1000
            text = " ".join(transcription_parts).strip()
            return {
                "text": text or f"[ERROR: {e}]",
                "ttfb_ms": ttfb_ms or total_ms,
                "total_ms": total_ms,
                "audio_bytes": audio_bytes_total,
                "audio_duration_s": 0,
                "tool_call": None,
            }

    def validate(agent_text):
        nonlocal call_ended
        word_count = len(agent_text.split())
        log_check(word_count <= 80, f"Length: {word_count} words", warn_only=word_count <= 120)

        all_matched, wait_matched = check_step_combining(agent_text, parsed_steps)
        if len(wait_matched) > 1:
            log_check(False,
                      f"MULTI-WAIT: {wait_matched} (all: {all_matched})",
                      warn_only=len(wait_matched) <= 2)
        elif len(all_matched) > 1:
            log_check(True, f"P&C combining OK: {all_matched}")
        else:
            log_check(True, f"Step: {all_matched}")

        is_dup, reason = is_duplicate_text(agent_text, recent_agent_texts, completed_steps, parsed_steps)
        log_check(not is_dup, f"No repetition{f' ({reason})' if is_dup else ''}", warn_only=True)

        if turn_count >= 2:
            lang_ok, lang_issue = check_language(agent_text, LANGUAGE)
            log_check(lang_ok, f"Language: {LANGUAGE}{f' - {lang_issue}' if lang_issue else ''}")

        step_id, step_label = match_step(agent_text, parsed_steps)
        if step_label:
            completed_steps.append(step_label)
            matched_step_ids.append(step_id)

        lower = agent_text.lower()
        if any(w in lower for w in ["downloaded", "app"]):
            phase_reached.add(2)
        if any(w in lower for w in ["whatsapp", "group"]):
            phase_reached.add(3)
        if "coach" in lower:
            phase_reached.add(4)
        if any(w in lower for w in ["live call", "monday", "friday", "wednesday"]):
            phase_reached.add(5)
        if any(w in lower for w in ["everything from my side", "bye", "proud"]):
            phase_reached.add(6)
            call_ended = True
        if "bye" in lower.split() or "bye-bye" in lower or "[end_call]" in lower:
            call_ended = True

        recent_agent_texts.append(agent_text)
        if len(recent_agent_texts) > 10:
            recent_agent_texts[:] = recent_agent_texts[-10:]

    # ============================================================
    # GREETING
    # ============================================================
    print(f"{C.BOLD}--- AGENT GREETING ---{C.END}")
    trigger = f"[Start the conversation now. Greet {CUSTOMER_NAME} naturally using your opening line from the instructions.]"
    await send_user_text(trigger)
    resp = await wait_for_response()
    turn_count += 1
    agent_text = resp["text"]
    agent_history.append(agent_text)
    latency_data.append({"turn": turn_count, "ttfb_ms": resp["ttfb_ms"], "total_ms": resp["total_ms"],
                         "audio_s": resp["audio_duration_s"]})
    conversation_log.append({"role": "system", "text": trigger})
    conversation_log.append({"role": "agent", "text": agent_text, **resp})

    if VERBOSE:
        print(f"    {C.BLUE}AGENT:{C.END} {agent_text}")
    else:
        print(f"    {C.BLUE}AGENT:{C.END} {agent_text[:150]}{'...' if len(agent_text) > 150 else ''}")

    # Latency
    print(f"    {C.BOLD}TTFB: {resp['ttfb_ms']:.0f}ms | Total: {resp['total_ms']:.0f}ms | Audio: {resp['audio_duration_s']:.1f}s{C.END}")
    log_check(resp["ttfb_ms"] < TTFB_FAIL_MS, f"TTFB: {resp['ttfb_ms']:.0f}ms", warn_only=resp["ttfb_ms"] < TTFB_WARN_MS * 2)

    log_check(CUSTOMER_NAME.lower() in agent_text.lower() or "welcome" in agent_text.lower(),
              "Greeting mentions name/welcome")
    validate(agent_text)
    phase_reached.add(1)

    # ============================================================
    # MAIN LOOP
    # ============================================================
    for label, user_text, expected_phase in USER_RESPONSES:
        if call_ended:
            break
        turn_count += 1
        print(f"\n{C.BOLD}--- TURN {turn_count}: {C.END}{C.GRAY}'{user_text}' ({label}){C.END}")

        await send_user_text(user_text)
        resp = await wait_for_response()
        agent_text = resp["text"]

        if not agent_text or "[ERROR" in agent_text:
            log_check(False, f"Response failed: {agent_text}")
            break

        agent_history.append(agent_text)
        latency_data.append({"turn": turn_count, "ttfb_ms": resp["ttfb_ms"], "total_ms": resp["total_ms"],
                             "audio_s": resp["audio_duration_s"]})
        conversation_log.append({"role": "user", "text": user_text, "label": label})
        conversation_log.append({"role": "agent", "text": agent_text, **resp})

        if VERBOSE:
            print(f"    {C.BLUE}AGENT:{C.END} {agent_text}")
        else:
            print(f"    {C.BLUE}AGENT:{C.END} {agent_text[:160]}{'...' if len(agent_text) > 160 else ''}")

        # Latency display
        ttfb_color = C.OK if resp["ttfb_ms"] < TTFB_WARN_MS else (C.WARN if resp["ttfb_ms"] < TTFB_FAIL_MS else C.FAIL)
        print(f"    {ttfb_color}TTFB: {resp['ttfb_ms']:.0f}ms{C.END} | Total: {resp['total_ms']:.0f}ms | Audio: {resp['audio_duration_s']:.1f}s")

        log_check(resp["ttfb_ms"] < TTFB_FAIL_MS,
                  f"TTFB: {resp['ttfb_ms']:.0f}ms", warn_only=resp["ttfb_ms"] < TTFB_FAIL_MS)
        validate(agent_text)

    # Extra nudges
    extra = 0
    while not call_ended and extra < 8:
        extra += 1
        turn_count += 1
        nudge = ["Okay", "Yes got it", "Hmm okay", "Yes", "Alright"][extra % 5]
        print(f"\n{C.BOLD}--- TURN {turn_count}: {C.END}{C.GRAY}Extra: '{nudge}'{C.END}")
        await send_user_text(nudge)
        resp = await wait_for_response()
        agent_text = resp["text"]
        if not agent_text:
            break
        agent_history.append(agent_text)
        latency_data.append({"turn": turn_count, "ttfb_ms": resp["ttfb_ms"], "total_ms": resp["total_ms"],
                             "audio_s": resp["audio_duration_s"]})
        conversation_log.append({"role": "user", "text": nudge, "label": "extra"})
        conversation_log.append({"role": "agent", "text": agent_text, **resp})

        if VERBOSE:
            print(f"    {C.BLUE}AGENT:{C.END} {agent_text}")
        else:
            print(f"    {C.BLUE}AGENT:{C.END} {agent_text[:160]}{'...' if len(agent_text) > 160 else ''}")

        ttfb_color = C.OK if resp["ttfb_ms"] < TTFB_WARN_MS else C.WARN
        print(f"    {ttfb_color}TTFB: {resp['ttfb_ms']:.0f}ms{C.END} | Total: {resp['total_ms']:.0f}ms")
        validate(agent_text)

    await ws.close()

    # ============================================================
    # RESULTS
    # ============================================================
    print(f"\n{'='*70}")
    print(f"{C.BOLD}RESULTS{C.END}")
    print(f"{'='*70}")

    # Phase coverage
    print(f"\n  {C.BOLD}Phase Coverage:{C.END}")
    phase_names = {1: "Greeting", 2: "App Setup", 3: "WhatsApp Groups",
                   4: "Super Coaches", 5: "Live Calls", 6: "Closing"}
    for p in range(1, 7):
        status = f"{C.OK}REACHED{C.END}" if p in phase_reached else f"{C.FAIL}MISSED{C.END}"
        print(f"    Phase {p} ({phase_names[p]}): {status}")
    log_check(len(phase_reached) >= 5, f"Phases: {len(phase_reached)}/6")
    log_check(call_ended, "Call completed")

    # Steps
    unique_ids = sorted(set(matched_step_ids), key=lambda x: float(x))
    all_ids = {s["step_id"] for s in parsed_steps}
    missed = sorted(all_ids - set(unique_ids), key=lambda x: float(x))
    print(f"\n  {C.BOLD}Steps: {len(unique_ids)}/{len(parsed_steps)}{C.END}")
    if missed:
        for sid in missed:
            info = next((s for s in parsed_steps if s["step_id"] == sid), None)
            if info:
                stype = "W" if info["type"] == "wait" else "P"
                print(f"    {C.FAIL}✗{C.END} [{stype}] {info['label']}")

    # ============================================================
    # LATENCY REPORT
    # ============================================================
    print(f"\n  {C.BOLD}LATENCY REPORT:{C.END}")
    ttfbs = [d["ttfb_ms"] for d in latency_data if d["ttfb_ms"]]
    totals = [d["total_ms"] for d in latency_data if d["total_ms"]]
    if ttfbs:
        avg_ttfb = sum(ttfbs) / len(ttfbs)
        min_ttfb = min(ttfbs)
        max_ttfb = max(ttfbs)
        p50 = sorted(ttfbs)[len(ttfbs) // 2]
        p90 = sorted(ttfbs)[int(len(ttfbs) * 0.9)]
        print(f"    TTFB (first audio byte):")
        print(f"      Avg: {avg_ttfb:.0f}ms | P50: {p50:.0f}ms | P90: {p90:.0f}ms")
        print(f"      Min: {min_ttfb:.0f}ms | Max: {max_ttfb:.0f}ms")
    if totals:
        avg_total = sum(totals) / len(totals)
        print(f"    Total response time:")
        print(f"      Avg: {avg_total:.0f}ms | Min: {min(totals):.0f}ms | Max: {max(totals):.0f}ms")

    # Latency trend (detect KV cache degradation)
    print(f"\n    {C.BOLD}Latency per turn:{C.END}")
    for d in latency_data:
        ttfb = d["ttfb_ms"]
        bar_len = int(ttfb / 100)
        bar = "█" * min(bar_len, 40)
        color = C.OK if ttfb < TTFB_WARN_MS else (C.WARN if ttfb < TTFB_FAIL_MS else C.FAIL)
        print(f"      Turn {d['turn']:2d}: {color}{bar} {ttfb:.0f}ms{C.END} (total: {d['total_ms']:.0f}ms, audio: {d['audio_s']:.1f}s)")

    # Trend check
    if len(ttfbs) >= 6:
        first_half = sum(ttfbs[:len(ttfbs)//2]) / (len(ttfbs)//2)
        second_half = sum(ttfbs[len(ttfbs)//2:]) / (len(ttfbs) - len(ttfbs)//2)
        degradation = ((second_half - first_half) / first_half) * 100
        if degradation > 30:
            log_check(False, f"Latency degradation: {degradation:.0f}% (first half avg: {first_half:.0f}ms, second: {second_half:.0f}ms)", warn_only=True)
        else:
            log_check(True, f"Latency stable ({degradation:+.0f}% change)")

    # Stats
    print(f"\n  {C.BOLD}Statistics:{C.END}")
    print(f"    Turns: {turn_count} | Errors: {C.FAIL}{len(errors)}{C.END} | Warnings: {C.WARN}{len(warnings)}{C.END}")

    if errors:
        print(f"\n  {C.FAIL}{C.BOLD}ERRORS:{C.END}")
        for e in errors:
            print(f"    - {e}")
    if warnings:
        print(f"\n  {C.WARN}{C.BOLD}WARNINGS:{C.END}")
        for w in warnings:
            print(f"    - {w}")

    # Full conversation
    print(f"\n{'='*70}")
    print(f"{C.BOLD}CONVERSATION{C.END}")
    print(f"{'='*70}")
    for entry in conversation_log:
        role = entry["role"].upper()
        text = entry.get("text", "")
        if role == "USER":
            print(f"  {C.GRAY}[USER]{C.END} {text}")
        elif role == "AGENT":
            ttfb = entry.get("ttfb_ms", 0)
            total = entry.get("total_ms", 0)
            audio_s = entry.get("audio_duration_s", 0)
            print(f"  {C.BLUE}[AGENT ttfb={ttfb:.0f}ms total={total:.0f}ms audio={audio_s:.1f}s]{C.END}")
            print(f"  {text[:300]}{'...' if len(text) > 300 else ''}")
        else:
            print(f"  {C.GRAY}[{role}]{C.END} {text[:80]}")
        print()

    # Save log
    log_file = f"test_results/realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("test_results", exist_ok=True)
    with open(log_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": AUDIO_MODEL,
            "voice": VOICE,
            "language": LANGUAGE,
            "turns": turn_count,
            "phases_reached": sorted(phase_reached),
            "steps_matched": unique_ids,
            "steps_missed": missed,
            "errors": errors,
            "warnings": warnings,
            "call_ended": call_ended,
            "latency": latency_data,
            "latency_summary": {
                "avg_ttfb_ms": avg_ttfb if ttfbs else 0,
                "p50_ttfb_ms": p50 if ttfbs else 0,
                "p90_ttfb_ms": p90 if ttfbs else 0,
                "avg_total_ms": avg_total if totals else 0,
            },
            "conversation": conversation_log,
        }, f, indent=2, default=str)
    print(f"  {C.GRAY}Log: {log_file}{C.END}")

    # Verdict
    print(f"\n{'='*70}")
    if not errors:
        print(f"{C.OK}{C.BOLD}ALL CHECKS PASSED{C.END} ({len(warnings)} warnings)")
    else:
        print(f"{C.FAIL}{C.BOLD}{len(errors)} ERRORS FOUND{C.END}")
    print(f"{'='*70}\n")

    return len(errors) == 0


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        print(f"{C.FAIL}ERROR: GOOGLE_API_KEY not set in .env{C.END}")
        sys.exit(1)

    if TEXT_ONLY:
        print("Running in TEXT-ONLY mode (no audio latency)")
        # Reuse simple text flow
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        prompt = load_prompt()
        parsed_steps = parse_prompt_steps(prompt)
        model = genai.GenerativeModel(model_name=TEXT_MODEL, system_instruction=prompt)
        chat = model.start_chat()

        print(f"\n{'='*70}")
        print(f"{C.BOLD}TEXT-ONLY MODE{C.END} (use without --text-only for audio latency)")
        print(f"{'='*70}")

        trigger = f"[Start the conversation now. Greet {CUSTOMER_NAME} naturally using your opening line from the instructions.]"
        resp = chat.send_message(trigger)
        print(f"  {C.BLUE}AGENT:{C.END} {resp.text.strip()[:150]}")

        for label, user_text, phase in USER_RESPONSES:
            t0 = time.time()
            resp = chat.send_message(user_text)
            ms = (time.time() - t0) * 1000
            print(f"  {C.GRAY}USER:{C.END} {user_text}")
            print(f"  {C.BLUE}AGENT ({ms:.0f}ms):{C.END} {resp.text.strip()[:160]}")
            if "bye" in resp.text.lower():
                break
        print(f"\n{C.OK}Done.{C.END}")
    else:
        success = asyncio.run(run_audio_test())
        sys.exit(0 if success else 1)
