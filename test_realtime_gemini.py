#!/usr/bin/env python3
"""
Gemini Onboarding Flow Test
============================
Tests the full Riddhi Gold Membership onboarding flow against Gemini.

Validates:
  1. Content completeness - all steps delivered, no missing content
  2. Latency - response time per turn
  3. No repetition - no duplicate questions/content
  4. Clean beginning & ending - proper greeting + proper goodbye
  5. Language consistency - no language switching mid-call

Usage:
  python3 test_realtime_gemini.py                    # Text mode (fast, ~30s)
  python3 test_realtime_gemini.py --audio            # Native audio + TTFB
  python3 test_realtime_gemini.py --verbose           # Show full responses
  python3 test_realtime_gemini.py --language hindi    # Test Hindi flow
  python3 test_realtime_gemini.py --prompt path.txt   # Custom prompt file
"""

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ============================================================
# Load .env without dotenv dependency
# ============================================================
SCRIPT_DIR = Path(__file__).parent
env_file = SCRIPT_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            val = val.strip().strip("'\"")
            os.environ.setdefault(key.strip(), val)

# ============================================================
# Configuration
# ============================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CUSTOMER_NAME = os.getenv("TEST_CUSTOMER_NAME", "Kiran")
VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv
AUDIO_MODE = "--audio" in sys.argv
LANGUAGE = "Hindi" if "--language" in sys.argv and "hindi" in " ".join(sys.argv).lower() else "English"

# Model selection
AUDIO_MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"
TEXT_MODEL = "gemini-2.0-flash"
VOICE = "Kore"

# Prompt file
PROMPT_FILE = "riddhi_prompt_v2.txt"
for i, arg in enumerate(sys.argv):
    if arg == "--prompt" and i + 1 < len(sys.argv):
        PROMPT_FILE = sys.argv[i + 1]

# Thresholds
LATENCY_WARN_MS = 2000
LATENCY_FAIL_MS = 5000

# Colors
class C:
    G = "\033[92m"   # green/ok
    R = "\033[91m"   # red/fail
    Y = "\033[93m"   # yellow/warn
    B = "\033[94m"   # blue
    D = "\033[90m"   # dim/gray
    BOLD = "\033[1m"
    END = "\033[0m"


# ============================================================
# Prompt Loader
# ============================================================
def load_prompt():
    prompt_path = SCRIPT_DIR / PROMPT_FILE
    if not prompt_path.exists():
        print(f"{C.R}ERROR: Prompt file not found: {prompt_path}{C.END}")
        sys.exit(1)
    prompt = prompt_path.read_text(encoding="utf-8")
    for key, val in {
        "customer_name": CUSTOMER_NAME, "membership_type": "Gold",
        "purchase_date": "2026-02-26", "language_preference": LANGUAGE,
        "city": "Mumbai", "lead_source": "Instagram",
    }.items():
        prompt = prompt.replace("{{" + key + "}}", val).replace("{" + key + "}", val)
    prompt += (
        "\n\n[SPEECH STYLE: Maintain a consistent speaking voice — same warmth, tone, accent "
        "throughout the entire call. Never suddenly change your speaking style mid-conversation.]"
        "\n\n[CONVERSATION RULE: NEVER repeat a question the customer already answered. "
        "Each question should be asked exactly ONCE.]"
    )
    return prompt


# ============================================================
# Step Parser
# ============================================================
STOP_WORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
              'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
              'and', 'or', 'but', 'so', 'if', 'it', 'its', 'this', 'that',
              'you', 'your', 'i', 'me', 'my', 'we', 'our', 'can', 'will',
              'do', 'have', 'has', 'had', 'would', 'could', 'should',
              'please', 'just', 'about', 'once', 'also', 'not', 'all'}

def parse_prompt_steps(prompt):
    steps = []
    blocks = list(re.finditer(r'STEP\s+(\d+\.\d+)\b', prompt))
    for i, m in enumerate(blocks):
        step_id = m.group(1)
        end = blocks[i + 1].start() if i + 1 < len(blocks) else len(prompt)
        block = prompt[m.start():end]
        dm = re.search(r'"([^"]{10,})"', block)
        if not dm:
            continue
        dialogue = dm.group(1).strip()
        after = block[dm.end():]
        stype = "P&C" if "PAUSE & CONTINUE" in after else "WAIT"
        words = re.findall(r'[a-z]+', dialogue.lower())
        keywords = set(w for w in words if w not in STOP_WORDS and len(w) > 2)
        wl = dialogue.lower().split()
        phrases = [f"{wl[j]} {wl[j+1]}" for j in range(len(wl)-1)
                   if len(f"{wl[j]} {wl[j+1]}") > 8 and not all(w in STOP_WORDS for w in [wl[j], wl[j+1]])]
        first_sent = re.split(r'[.?!]', dialogue)[0].strip()[:40]
        steps.append({
            "id": step_id, "type": stype, "keywords": keywords,
            "phrases": phrases[:6], "dialogue": dialogue,
            "label": f"{step_id}: {first_sent}",
            "phase": int(step_id.split(".")[0]),
        })
    return steps


def match_step(text, steps):
    """Find best matching step for agent text. Returns (step_id, label) or ("","")."""
    best_score, best = 0, None
    text_words = set(re.findall(r'[a-z]+', text.lower()))
    for s in steps:
        if not s["keywords"]:
            continue
        kw = len(text_words & s["keywords"]) / len(s["keywords"])
        ph = sum(1 for p in s["phrases"] if p in text.lower()) * 0.15
        score = kw + ph
        if score > best_score and score > 0.3:
            best_score, best = score, s
    return (best["id"], best["label"]) if best else ("", "")


def match_all_steps(text, steps):
    """Find ALL steps that match in a turn (for combining detection)."""
    matched = []
    text_words = set(re.findall(r'[a-z]+', text.lower()))
    for s in steps:
        if not s["keywords"]:
            continue
        kw = len(text_words & s["keywords"]) / len(s["keywords"])
        ph = sum(1 for p in s["phrases"] if p in text.lower())
        if kw + ph * 0.15 > 0.5 and ph >= 1:
            matched.append(s)
    return matched


# ============================================================
# User Responses
# ============================================================
USER_RESPONSES = [
    ("language_choice",      f"{LANGUAGE} please"),
    ("availability",         "Yes, I'm free right now"),
    ("speaker",              "Okay, I'm on speaker now"),
    ("app_downloaded",       "Yes I have the app downloaded"),
    ("app_open",             "It's open now"),
    ("courses_clicked",      "Done, I clicked on Courses"),
    ("video_playing",        "Yes the video is playing"),
    ("course_understood",    "Yes that makes sense"),
    ("whatsapp_opened",      "I've opened the WhatsApp message"),
    ("orientation_joined",   "Yes I've joined the orientation group"),
    ("main_group_joined",    "Yes joined that one too"),
    ("mission_team_joined",  "Done, I joined the mission team group"),
    ("groups_understood",    "Alright got it"),
    ("coaches_visible",      "Yes I can see their names"),
    ("coaches_names",        "Sunita and Meera"),
    ("following_so_far",     "Yes I'm following everything"),
    ("sounds_good",          "Yes that sounds great"),
    ("final_response",       "Thank you so much!"),
]

EXTRA_NUDGES = ["Yes got it", "Okay", "Hmm alright", "Yes", "Alright thanks"]


# ============================================================
# Validators
# ============================================================
def is_duplicate(text, recent_texts):
    """Check if text repeats something already said."""
    words = set(text.lower().split())
    if len(words) < 4:
        return False, ""
    for prev in recent_texts:
        pwords = set(prev.lower().split())
        if not pwords:
            continue
        overlap = len(words & pwords)
        ratio = overlap / len(words) if len(words) < 12 else overlap / max(len(words), len(pwords))
        if ratio > 0.5:
            return True, prev[:50]
    return False, ""


def check_language(text, expected):
    """Check language consistency."""
    if expected == "English":
        hindi = len(re.findall(r'[\u0900-\u097F]', text))
        if hindi > 3:
            return False, f"{hindi} Hindi characters found"
    elif expected == "Hindi":
        # Hindi should have Devanagari OR Hinglish (Latin script is OK)
        pass
    # Check for behavioral markers leaking
    markers = ["[WAIT]", "[PAUSE & CONTINUE]", "[STOP HERE]", "[end_call]",
               "[PAUSE &", "PAUSE & CONTINUE"]
    for m in markers:
        if m in text:
            return False, f"Behavioral marker leaked: {m}"
    return True, ""


# ============================================================
# Test Runner (shared by both modes)
# ============================================================
class TestRunner:
    def __init__(self, steps, language):
        self.steps = steps
        self.language = language
        self.recent_texts = []
        self.completed_ids = []
        self.completed_labels = []
        self.phases_reached = set()
        self.errors = []
        self.warnings = []
        self.turns = []
        self.call_ended = False
        self.turn_num = 0

    def record_turn(self, user_text, agent_text, latency_ms, label=""):
        self.turn_num += 1
        turn = {
            "num": self.turn_num, "user": user_text, "agent": agent_text,
            "latency_ms": latency_ms, "label": label, "issues": [],
        }

        # --- Check 1: Content completeness (step matching) ---
        # Use match_all_steps so combined P&C turns count ALL steps delivered
        matched = match_all_steps(agent_text, self.steps)
        if not matched:
            # Fallback to single best match for short responses
            sid, slabel = match_step(agent_text, self.steps)
            if sid:
                matched = [next(s for s in self.steps if s["id"] == sid)]

        for s in matched:
            if s["id"] not in self.completed_ids:
                self.completed_ids.append(s["id"])
                self.completed_labels.append(s["label"])
            self.phases_reached.add(s["phase"])

        turn["step_id"] = matched[0]["id"] if matched else ""
        turn["step_label"] = matched[0]["label"] if matched else ""
        turn["all_steps"] = [s["id"] for s in matched]

        # Multiple WAIT steps in one turn = bad combining
        wait_ids = [s["id"] for s in matched if s["type"] == "WAIT"]
        if len(wait_ids) > 1:
            issue = f"Combined {len(wait_ids)} WAIT steps: {wait_ids}"
            turn["issues"].append(issue)
            self.warnings.append(f"Turn {self.turn_num}: {issue}")

        # --- Check 2: Latency ---
        if latency_ms > LATENCY_FAIL_MS:
            issue = f"Slow: {latency_ms:.0f}ms"
            turn["issues"].append(issue)
            self.errors.append(f"Turn {self.turn_num}: {issue}")
        elif latency_ms > LATENCY_WARN_MS:
            turn["issues"].append(f"Latency: {latency_ms:.0f}ms")

        # --- Check 3: Repetition ---
        is_dup, dup_source = is_duplicate(agent_text, self.recent_texts)
        if is_dup:
            issue = f"Repeated: '{dup_source}...'"
            turn["issues"].append(issue)
            self.errors.append(f"Turn {self.turn_num}: {issue}")

        # --- Check 4: Clean beginning/ending ---
        if self.turn_num == 1:
            lower = agent_text.lower()
            if CUSTOMER_NAME.lower() not in lower and "welcome" not in lower:
                issue = "Greeting missing customer name or welcome"
                turn["issues"].append(issue)
                self.errors.append(f"Turn {self.turn_num}: {issue}")
            self.phases_reached.add(1)

        lower = agent_text.lower()
        if any(w in lower for w in ["everything from my side", "bye-bye", "bye bye", "beautiful day"]):
            self.phases_reached.add(6)
            self.call_ended = True
        if "[end_call]" in lower:
            self.call_ended = True

        # Phase tracking
        if any(w in lower for w in ["downloaded", "riddhi deorah app"]):
            self.phases_reached.add(2)
        if any(w in lower for w in ["whatsapp", "orientation group", "main group", "mission team"]):
            self.phases_reached.add(3)
        if "coach" in lower:
            self.phases_reached.add(4)
        if any(w in lower for w in ["monday", "friday", "wednesday", "live call", "parenting"]):
            self.phases_reached.add(5)

        # --- Check 5: Language ---
        if self.turn_num >= 2:
            lang_ok, lang_issue = check_language(agent_text, self.language)
            if not lang_ok:
                issue = f"Language: {lang_issue}"
                turn["issues"].append(issue)
                self.errors.append(f"Turn {self.turn_num}: {issue}")

        self.recent_texts.append(agent_text)
        if len(self.recent_texts) > 10:
            self.recent_texts = self.recent_texts[-10:]
        self.turns.append(turn)
        return turn

    def print_turn(self, turn):
        tag = f"{C.D}{turn['label']}{C.END}" if turn["label"] else ""
        print(f"\n{C.BOLD}Turn {turn['num']}{C.END} {tag}")
        if turn["user"]:
            print(f"  {C.D}USER: {turn['user']}{C.END}")
        text = turn["agent"]
        if not VERBOSE and len(text) > 180:
            text = text[:180] + "..."
        print(f"  {C.B}AGENT:{C.END} {text}")

        # Latency + steps
        ms = turn["latency_ms"]
        color = C.G if ms < LATENCY_WARN_MS else (C.Y if ms < LATENCY_FAIL_MS else C.R)
        all_steps = turn.get("all_steps", [])
        step_info = f" | steps {all_steps}" if len(all_steps) > 1 else (f" | step {turn['step_id']}" if turn["step_id"] else "")
        print(f"  {color}{ms:.0f}ms{C.END}{step_info}")

        for issue in turn["issues"]:
            print(f"  {C.R}>> {issue}{C.END}")

    def print_scorecard(self):
        all_ids = {s["id"] for s in self.steps}
        missed_ids = sorted(all_ids - set(self.completed_ids), key=lambda x: float(x))
        missed_wait = [sid for sid in missed_ids
                       if any(s["id"] == sid and s["type"] == "WAIT" for s in self.steps)]
        missed_pc = [sid for sid in missed_ids
                     if any(s["id"] == sid and s["type"] == "P&C" for s in self.steps)]

        repetitions = sum(1 for t in self.turns
                          if any("Repeated" in i for i in t["issues"]))
        lang_issues = sum(1 for t in self.turns
                          if any("Language" in i for i in t["issues"]))
        slow_turns = sum(1 for t in self.turns if t["latency_ms"] > LATENCY_FAIL_MS)
        latencies = [t["latency_ms"] for t in self.turns]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        print(f"\n{'='*60}")
        print(f"{C.BOLD}  SCORECARD{C.END}")
        print(f"{'='*60}")

        # 1. Content completeness
        step_pct = len(self.completed_ids) / len(all_ids) * 100 if all_ids else 0
        ok = step_pct >= 85 and len(missed_wait) == 0
        icon = f"{C.G}PASS{C.END}" if ok else f"{C.R}FAIL{C.END}"
        print(f"\n  1. Content Completeness    [{icon}]")
        print(f"     Steps: {len(self.completed_ids)}/{len(all_ids)} ({step_pct:.0f}%)")
        print(f"     Phases: {sorted(self.phases_reached)}")
        if missed_wait:
            print(f"     {C.R}Missing WAIT steps: {missed_wait}{C.END}")
        if missed_pc:
            print(f"     {C.Y}Missing P&C steps: {missed_pc}{C.END}")

        # 2. Latency
        ok = slow_turns == 0
        icon = f"{C.G}PASS{C.END}" if ok else f"{C.R}FAIL{C.END}"
        print(f"\n  2. Latency                 [{icon}]")
        print(f"     Avg: {avg_latency:.0f}ms | Slow turns (>{LATENCY_FAIL_MS}ms): {slow_turns}")
        if latencies:
            sorted_lat = sorted(latencies)
            p50 = sorted_lat[len(sorted_lat) // 2]
            p90 = sorted_lat[int(len(sorted_lat) * 0.9)]
            print(f"     P50: {p50:.0f}ms | P90: {p90:.0f}ms | Max: {max(latencies):.0f}ms")

        # 3. No repetition
        ok = repetitions == 0
        icon = f"{C.G}PASS{C.END}" if ok else f"{C.R}FAIL{C.END}"
        print(f"\n  3. No Repetition           [{icon}]")
        print(f"     Repetitions found: {repetitions}")

        # 4. Clean beginning & ending
        has_greeting = 1 in self.phases_reached
        has_closing = 6 in self.phases_reached or self.call_ended
        ok = has_greeting and has_closing
        icon = f"{C.G}PASS{C.END}" if ok else f"{C.R}FAIL{C.END}"
        print(f"\n  4. Clean Begin & End       [{icon}]")
        print(f"     Greeting: {'Yes' if has_greeting else 'MISSING'} | "
              f"Closing: {'Yes' if has_closing else 'MISSING'}")

        # 5. Language consistency
        ok = lang_issues == 0
        icon = f"{C.G}PASS{C.END}" if ok else f"{C.R}FAIL{C.END}"
        print(f"\n  5. Language Consistency     [{icon}]")
        print(f"     Language: {self.language} | Issues: {lang_issues}")

        # Overall
        total_pass = sum([
            step_pct >= 85 and len(missed_wait) == 0,
            slow_turns == 0,
            repetitions == 0,
            has_greeting and has_closing,
            lang_issues == 0,
        ])
        print(f"\n{'='*60}")
        if total_pass == 5:
            print(f"  {C.G}{C.BOLD}ALL 5 CHECKS PASSED{C.END}")
        else:
            print(f"  {C.R}{C.BOLD}{total_pass}/5 CHECKS PASSED{C.END}")
        print(f"  Turns: {self.turn_num} | Errors: {len(self.errors)} | Warnings: {len(self.warnings)}")
        print(f"{'='*60}")

        if self.errors:
            print(f"\n  {C.R}Errors:{C.END}")
            for e in self.errors:
                print(f"    - {e}")
        if self.warnings:
            print(f"\n  {C.Y}Warnings:{C.END}")
            for w in self.warnings:
                print(f"    - {w}")

        return total_pass == 5

    def save_results(self, mode="text"):
        os.makedirs(SCRIPT_DIR / "test_results", exist_ok=True)
        fname = f"test_results/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = SCRIPT_DIR / fname
        all_ids = {s["id"] for s in self.steps}
        data = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "model": AUDIO_MODEL if mode == "audio" else TEXT_MODEL,
            "language": self.language,
            "customer_name": CUSTOMER_NAME,
            "turns": self.turn_num,
            "phases_reached": sorted(self.phases_reached),
            "steps_matched": sorted(set(self.completed_ids), key=lambda x: float(x)),
            "steps_missed": sorted(all_ids - set(self.completed_ids), key=lambda x: float(x)),
            "completed_labels": self.completed_labels,
            "errors": self.errors,
            "warnings": self.warnings,
            "call_ended": self.call_ended,
            "conversation": [
                {"role": "user" if t["user"] else "agent", "text": t["user"] or t["agent"],
                 "label": t["label"], "latency_ms": t["latency_ms"]}
                for t in self.turns
            ],
        }
        path.write_text(json.dumps(data, indent=2, default=str))
        print(f"\n  {C.D}Results: {fname}{C.END}")


# ============================================================
# TEXT MODE (google.genai SDK)
# ============================================================
def run_text_mode():
    from google import genai

    prompt = load_prompt()
    steps = parse_prompt_steps(prompt)
    runner = TestRunner(steps, LANGUAGE)

    print(f"\n{'='*60}")
    print(f"{C.BOLD}  GEMINI ONBOARDING TEST (TEXT MODE){C.END}")
    print(f"{'='*60}")
    print(f"  Model:    {TEXT_MODEL}")
    print(f"  Customer: {CUSTOMER_NAME}")
    print(f"  Language: {LANGUAGE}")
    print(f"  Steps:    {len(steps)}")
    print(f"  Prompt:   {PROMPT_FILE}")
    print(f"{'='*60}")

    client = genai.Client(api_key=GOOGLE_API_KEY)
    chat = client.chats.create(model=TEXT_MODEL, config={"system_instruction": prompt})

    def send(text, label="", retries=3):
        for attempt in range(retries):
            try:
                t0 = time.time()
                resp = chat.send_message(text)
                ms = (time.time() - t0) * 1000
                return resp.text.strip(), ms
            except Exception as e:
                if "429" in str(e) and attempt < retries - 1:
                    wait = 10 * (attempt + 1)
                    print(f"  {C.Y}Rate limited, waiting {wait}s...{C.END}")
                    time.sleep(wait)
                else:
                    return f"[ERROR: {e}]", 0
        return "[ERROR: max retries]", 0

    # Greeting
    trigger = f"[Start the conversation now. Greet {CUSTOMER_NAME} naturally using your opening line from the instructions.]"
    agent_text, ms = send(trigger, "greeting")
    if "[ERROR" in agent_text:
        print(f"  {C.R}{agent_text}{C.END}")
        return False
    turn = runner.record_turn("", agent_text, ms, "greeting")
    runner.print_turn(turn)

    # Main conversation
    for label, user_text in USER_RESPONSES:
        if runner.call_ended:
            break
        agent_text, ms = send(user_text, label)
        if "[ERROR" in agent_text:
            print(f"\n  {C.R}{agent_text}{C.END}")
            break
        turn = runner.record_turn(user_text, agent_text, ms, label)
        runner.print_turn(turn)

    # Extra nudges if call hasn't ended
    for i, nudge in enumerate(EXTRA_NUDGES):
        if runner.call_ended:
            break
        agent_text, ms = send(nudge, f"extra_{i}")
        if "[ERROR" in agent_text or not agent_text:
            break
        turn = runner.record_turn(nudge, agent_text, ms, f"extra_{i}")
        runner.print_turn(turn)

    passed = runner.print_scorecard()
    runner.save_results("text")
    return passed


# ============================================================
# AUDIO MODE (google.genai Live API — no websockets dependency)
# ============================================================
async def run_audio_mode():
    from google import genai
    from google.genai import types

    prompt = load_prompt()
    steps = parse_prompt_steps(prompt)
    runner = TestRunner(steps, LANGUAGE)

    print(f"\n{'='*60}")
    print(f"{C.BOLD}  GEMINI ONBOARDING TEST (AUDIO MODE){C.END}")
    print(f"{'='*60}")
    print(f"  Model:    {AUDIO_MODEL}")
    print(f"  Voice:    {VOICE}")
    print(f"  Customer: {CUSTOMER_NAME}")
    print(f"  Language: {LANGUAGE}")
    print(f"  Steps:    {len(steps)}")
    print(f"{'='*60}")

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

    print(f"\n  {C.B}Connecting to Gemini Live...{C.END}", end="", flush=True)
    t0 = time.time()

    async with client.aio.live.connect(model=AUDIO_MODEL, config=config) as session:
        print(f" ready ({(time.time()-t0)*1000:.0f}ms)")

        async def send_and_receive(text, timeout=30):
            """Send text via client_content and collect audio response with transcription."""
            t_start = time.time()
            await session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=text)]),
                turn_complete=True,
            )
            transcription_parts = []
            ttfb = None
            audio_bytes = 0
            try:
                async for msg in session.receive():
                    if time.time() - t_start > timeout:
                        break
                    sc = msg.server_content
                    if sc:
                        # Collect output transcription
                        if sc.output_transcription and sc.output_transcription.text:
                            transcription_parts.append(sc.output_transcription.text.strip())
                        # Track audio for TTFB
                        if sc.model_turn and sc.model_turn.parts:
                            for part in sc.model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    audio_bytes += len(part.inline_data.data)
                                    if ttfb is None:
                                        ttfb = (time.time() - t_start) * 1000
                        if sc.turn_complete:
                            break
                    # Handle tool calls
                    tc = msg.tool_call
                    if tc and tc.function_calls:
                        responses = []
                        is_end_call = False
                        for fc in tc.function_calls:
                            responses.append(types.FunctionResponse(
                                id=fc.id, name=fc.name, response={"success": True}
                            ))
                            if fc.name == "end_call":
                                is_end_call = True
                        await session.send_tool_response(function_responses=responses)
                        if is_end_call:
                            text_out = " ".join(transcription_parts).strip() + " [end_call]"
                            return text_out, (time.time()-t_start)*1000, ttfb
            except Exception as e:
                text_out = " ".join(transcription_parts).strip()
                return text_out or f"[ERROR: {e}]", (time.time()-t_start)*1000, ttfb

            return " ".join(transcription_parts).strip(), (time.time()-t_start)*1000, ttfb

        # Greeting
        trigger = f"[Start the conversation now. Greet {CUSTOMER_NAME} naturally using your opening line from the instructions.]"
        agent_text, total_ms, ttfb_ms = await send_and_receive(trigger)
        turn = runner.record_turn("", agent_text, ttfb_ms or total_ms, "greeting")
        runner.print_turn(turn)

        # Main conversation
        for label, user_text in USER_RESPONSES:
            if runner.call_ended:
                break
            agent_text, total_ms, ttfb_ms = await send_and_receive(user_text)
            if not agent_text or "[ERROR" in agent_text:
                print(f"\n  {C.R}{agent_text}{C.END}")
                break
            turn = runner.record_turn(user_text, agent_text, ttfb_ms or total_ms, label)
            runner.print_turn(turn)

        # Extra nudges
        for i, nudge in enumerate(EXTRA_NUDGES):
            if runner.call_ended:
                break
            agent_text, total_ms, ttfb_ms = await send_and_receive(nudge)
            if not agent_text:
                break
            turn = runner.record_turn(nudge, agent_text, ttfb_ms or total_ms, f"extra_{i}")
            runner.print_turn(turn)

    passed = runner.print_scorecard()
    runner.save_results("audio")
    return passed


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        print(f"{C.R}ERROR: GOOGLE_API_KEY not set. Add it to .env or export it.{C.END}")
        sys.exit(1)

    if AUDIO_MODE:
        success = asyncio.run(run_audio_mode())
    else:
        success = run_text_mode()
    sys.exit(0 if success else 1)
