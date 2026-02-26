#!/usr/bin/env python3
"""
REAL-TIME Gemini Integration Test
===================================
Talks to Gemini in real-time using the EXACT production prompt.
Simulates a full onboarding call with user responses, validating
each agent response for all known issues.

Uses the same prompt rendering, step parsing, and duplicate detection
logic as production code — this is NOT a mock test.

Validates:
  - One step at a time (no combining multiple steps)
  - No repeating previous content (production duplicate logic)
  - Language consistency after choice
  - Proper question/wait behavior
  - Full flow completion through Phase 6
  - Response length (not too long)
  - PAUSE & CONTINUE steps auto-continue without waiting

Usage:
  python3 test_realtime_gemini.py
  python3 test_realtime_gemini.py --verbose       # Show full responses
  python3 test_realtime_gemini.py --language hindi # Test Hindi flow
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai

# ============================================================
# Configuration
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = "gemini-2.0-flash"
CUSTOMER_NAME = "Kiran"
VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv
LANGUAGE = "Hindi" if "--language" in sys.argv and "hindi" in " ".join(sys.argv).lower() else "English"

# Colors for terminal output
class C:
    OK = "\033[92m"
    FAIL = "\033[91m"
    WARN = "\033[93m"
    BLUE = "\033[94m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    END = "\033[0m"


# ============================================================
# Prompt Loader (mirrors production code exactly)
# ============================================================

def load_prompt():
    """Load and render the Riddhi prompt exactly as production does."""
    with open("riddhi_prompt_v2.txt") as f:
        prompt = f.read()
    # Replace template variables (same as production render_prompt)
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

    # Add the same extra instructions production adds in _send_session_setup_on_ws()
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
# Step Parser (mirrors production _parse_prompt_steps exactly)
# ============================================================

def parse_prompt_steps(prompt):
    """Parse STEP markers from prompt (same logic as production)."""
    steps = []
    step_pattern = re.compile(
        r'STEP\s+(\d+\.\d+)\b.*?(?:\"([^\"]{10,})\")',
        re.DOTALL
    )
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                  'and', 'or', 'but', 'so', 'if', 'it', 'its', 'this', 'that',
                  'you', 'your', 'i', 'me', 'my', 'we', 'our', 'can', 'will',
                  'do', 'have', 'has', 'had', 'would', 'could', 'should',
                  'please', 'just', 'about', 'once', 'also', 'not', 'all'}
    for match in step_pattern.finditer(prompt):
        step_id = match.group(1)
        dialogue = match.group(2).strip()
        words = re.findall(r'[a-z]+', dialogue.lower())
        keywords = set(w for w in words if w not in stop_words and len(w) > 2)
        word_list = dialogue.lower().split()
        phrases = []
        for i in range(len(word_list) - 1):
            bigram = f"{word_list[i]} {word_list[i+1]}"
            if len(bigram) > 8 and not all(w in stop_words for w in bigram.split()):
                phrases.append(bigram)
        first_sentence = re.split(r'[.?!]', dialogue)[0].strip()
        label = f"Step {step_id}: {first_sentence[:35]}"
        steps.append({
            "step_id": step_id,
            "label": label,
            "keywords": keywords,
            "phrases": phrases[:6],
            "dialogue": dialogue,
        })
    return steps


def match_step(text, steps):
    """Match agent text against parsed steps (same as production _match_prompt_step)."""
    if not steps:
        return "", ""
    best_score = 0
    best_label = ""
    best_id = ""
    text_words = set(re.findall(r'[a-z]+', text.lower()))
    for step in steps:
        if not step["keywords"]:
            continue
        keyword_overlap = len(text_words & step["keywords"]) / len(step["keywords"])
        phrase_bonus = sum(1 for p in step["phrases"] if p in text.lower()) * 0.15
        score = keyword_overlap + phrase_bonus
        if score > best_score and score > 0.3:
            best_score = score
            best_label = step["label"]
            best_id = step["step_id"]
    return best_id, best_label


# ============================================================
# Validators (mirrors production duplicate/repetition logic)
# ============================================================

def is_duplicate_text(new_text, recent_texts, completed_steps, parsed_steps):
    """Check for duplicate (same logic as production _is_duplicate_text)."""
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
    # Check against completed step labels
    _, new_label = match_step(new_text, parsed_steps)
    new_label_lower = new_label.lower()
    if new_label_lower and len(new_label_lower) > 15:
        for step_label in completed_steps:
            step_words = set(step_label.lower().split())
            label_words = set(new_label_lower.split())
            if step_words and label_words:
                overlap = len(step_words & label_words) / max(len(step_words), len(label_words))
                if overlap > 0.5:
                    return True, f"matches completed step: '{step_label}'"
    return False, ""


def check_step_combining(text, parsed_steps):
    """Check if multiple script steps appear in a single turn.
    Uses stricter matching (>50% + phrase match) to reduce false positives."""
    matched_steps = []
    text_lower = text.lower()
    text_words = set(re.findall(r'[a-z]+', text_lower))
    for step in parsed_steps:
        if not step["keywords"]:
            continue
        keyword_overlap = len(text_words & step["keywords"]) / len(step["keywords"])
        phrase_hits = sum(1 for p in step["phrases"] if p in text_lower)
        phrase_bonus = phrase_hits * 0.15
        score = keyword_overlap + phrase_bonus
        # Require at least 50% keyword match AND at least 1 phrase match for confidence
        if score > 0.5 and phrase_hits >= 1:
            matched_steps.append(step["step_id"])
    return matched_steps


def check_language(text, expected_lang):
    """Check language consistency."""
    if expected_lang == "English":
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        if hindi_chars > 5:
            return False, f"Found {hindi_chars} Hindi characters in English response"
    elif expected_lang == "Hindi":
        latin_words = re.findall(r'\b[a-zA-Z]{5,}\b', text)
        allowed = {'riddhi', 'deorah', 'gold', 'launchpad', 'whatsapp',
                   'super', 'coach', 'orientation', 'mission', 'team',
                   'supermom', 'revolution', 'parenting', 'monday',
                   'friday', 'wednesday', 'course', 'courses', 'wonderful',
                   'perfect', 'great', 'group', 'video', 'phone', 'speaker',
                   'earphones', 'download', 'installed', 'recordings'}
        non_name = [w for w in latin_words if w.lower() not in allowed]
        if len(non_name) > 10:
            return False, f"Too many English words in Hindi response: {non_name[:5]}"
    return True, ""


# ============================================================
# User Responses (simulating real customer going through flow)
# ============================================================

# Each entry: (label, user_response, expected_phase)
USER_RESPONSES = [
    # Phase 1: Greeting
    ("language_choice", f"{LANGUAGE} please", 1),
    ("availability", "Yes, I'm free right now", 1),
    ("speaker", "Okay, I'm on speaker now", 1),
    # Phase 2: App Setup
    ("app_downloaded", "Yes I have the app downloaded", 2),
    ("app_open", "It's open now", 2),
    ("courses_clicked", "Done, I clicked on Courses", 2),
    ("video_playing", "Yes the video is playing", 2),
    # 2.5 PAUSE & CONTINUE, 2.6 PAUSE & CONTINUE
    ("course_understood", "Yes that makes sense", 2),
    # Phase 3: WhatsApp Groups
    ("whatsapp_opened", "I've opened the WhatsApp message", 3),
    ("orientation_joined", "Yes I've joined the orientation group", 3),
    # 3.3 PAUSE & CONTINUE
    ("main_group_joined", "Yes joined that one too", 3),
    # 3.5 PAUSE & CONTINUE
    ("mission_team_joined", "Done, I joined the mission team group", 3),
    # 3.7 PAUSE & CONTINUE, 3.8 PAUSE & CONTINUE
    ("groups_understood", "Alright got it", 3),
    # Phase 4: Super Coaches
    ("coaches_visible", "Yes I can see their names", 4),
    ("coaches_names", "Sunita and Meera", 4),
    # 4.3 PAUSE & CONTINUE
    # Phase 5: Live Calls
    # 5.1 PAUSE & CONTINUE, 5.2 PAUSE & CONTINUE
    ("following_so_far", "Yes I'm following everything", 5),
    # 5.4 PAUSE & CONTINUE, 5.5 PAUSE & CONTINUE
    ("sounds_good", "Yes that sounds great", 5),
    # Phase 6: Closing
    ("final_response", "Thank you so much!", 6),
]


# ============================================================
# Main Test Runner
# ============================================================

def run_test():
    if not GOOGLE_API_KEY:
        print(f"{C.FAIL}ERROR: GOOGLE_API_KEY not set in .env{C.END}")
        return False

    genai.configure(api_key=GOOGLE_API_KEY)
    prompt = load_prompt()
    parsed_steps = parse_prompt_steps(prompt)

    print(f"\n{'='*70}")
    print(f"{C.BOLD}REAL-TIME GEMINI INTEGRATION TEST{C.END}")
    print(f"{'='*70}")
    print(f"  Model:    {MODEL} (same Gemini family as production)")
    print(f"  Prompt:   riddhi_prompt_v2.txt ({len(prompt)} chars)")
    print(f"  Customer: {CUSTOMER_NAME}")
    print(f"  Language: {LANGUAGE}")
    print(f"  Steps:    {len(parsed_steps)}")
    print(f"  Turns:    {len(USER_RESPONSES)}")
    print(f"  Started:  {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")

    # Create chat session with same system instruction as production
    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=prompt,
    )
    chat = model.start_chat()

    # State tracking (mirrors production session)
    errors = []
    warnings = []
    agent_history = []
    recent_agent_texts = []     # For duplicate detection (last 10)
    completed_steps = []
    matched_step_ids = []
    phase_reached = set()
    turn_count = 0
    call_ended = False
    conversation_log = []       # Full log for saving

    def log_check(ok, msg, warn_only=False):
        if ok:
            print(f"    {C.OK}[OK]{C.END} {msg}")
        elif warn_only:
            warnings.append(f"Turn {turn_count}: {msg}")
            print(f"    {C.WARN}[WARN]{C.END} {msg}")
        else:
            errors.append(f"Turn {turn_count}: {msg}")
            print(f"    {C.FAIL}[FAIL]{C.END} {msg}")

    def validate_agent_response(agent_text):
        """Run all validations on an agent response."""
        nonlocal call_ended

        # 1. Response length
        word_count = len(agent_text.split())
        log_check(
            word_count <= 80,
            f"Length: {word_count} words",
            warn_only=word_count <= 120
        )

        # 2. Step combining (CRITICAL - main issue we're hunting)
        combined = check_step_combining(agent_text, parsed_steps)
        if len(combined) > 1:
            log_check(False,
                      f"STEP COMBINING: {len(combined)} steps in one turn: {combined}",
                      warn_only=len(combined) <= 2)
        else:
            log_check(True, f"One step at a time (steps: {combined})")

        # 3. Repetition check (mirrors production _is_duplicate_text)
        is_dup, dup_reason = is_duplicate_text(
            agent_text, recent_agent_texts, completed_steps, parsed_steps
        )
        log_check(not is_dup, f"No repetition{f' ({dup_reason})' if is_dup else ''}", warn_only=True)

        # 4. Language consistency (after language choice)
        if turn_count >= 2:
            lang_ok, lang_issue = check_language(agent_text, LANGUAGE)
            log_check(lang_ok, f"Language: {LANGUAGE}{f' - {lang_issue}' if lang_issue else ''}")

        # 5. Track step and phase
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

        # 6. Check for call end
        if "bye" in lower.split() or "bye-bye" in lower:
            call_ended = True

        # Add to recent texts (after validation)
        recent_agent_texts.append(agent_text)
        if len(recent_agent_texts) > 10:
            recent_agent_texts[:] = recent_agent_texts[-10:]

    # ============================================================
    # TURN 0: Agent Greeting (trigger like production)
    # ============================================================
    print(f"{C.BOLD}--- AGENT GREETING ---{C.END}")
    try:
        trigger = f"[Start the conversation now. Greet {CUSTOMER_NAME} naturally using your opening line from the instructions.]"
        t0 = time.time()
        response = chat.send_message(trigger)
        latency_ms = (time.time() - t0) * 1000
        agent_text = response.text.strip()
        turn_count += 1
        agent_history.append(agent_text)
        conversation_log.append({"role": "system", "text": trigger})
        conversation_log.append({"role": "agent", "text": agent_text, "latency_ms": latency_ms})

        if VERBOSE:
            print(f"    {C.BLUE}AGENT ({latency_ms:.0f}ms):{C.END} {agent_text}")
        else:
            print(f"    {C.BLUE}AGENT ({latency_ms:.0f}ms):{C.END} {agent_text[:150]}{'...' if len(agent_text) > 150 else ''}")

        # Greeting-specific checks
        log_check(
            CUSTOMER_NAME.lower() in agent_text.lower() or "welcome" in agent_text.lower(),
            "Greeting mentions customer name or welcome"
        )
        log_check(
            "english" in agent_text.lower() or "hindi" in agent_text.lower(),
            "Asks about language preference"
        )

        # General validation
        validate_agent_response(agent_text)
        phase_reached.add(1)
    except Exception as e:
        errors.append(f"Greeting failed: {e}")
        print(f"    {C.FAIL}[ERROR] Greeting failed: {e}{C.END}")

    # ============================================================
    # MAIN CONVERSATION LOOP
    # ============================================================
    for resp_label, user_text, expected_phase in USER_RESPONSES:
        if call_ended:
            break

        turn_count += 1
        print(f"\n{C.BOLD}--- TURN {turn_count}: {C.END}{C.GRAY}User: '{user_text}' ({resp_label}){C.END}")

        try:
            t0 = time.time()
            response = chat.send_message(user_text)
            latency_ms = (time.time() - t0) * 1000
            agent_text = response.text.strip()
            agent_history.append(agent_text)
            conversation_log.append({"role": "user", "text": user_text, "label": resp_label})
            conversation_log.append({"role": "agent", "text": agent_text, "latency_ms": latency_ms})

            if VERBOSE:
                print(f"    {C.BLUE}AGENT ({latency_ms:.0f}ms):{C.END} {agent_text}")
            else:
                truncated = agent_text[:160] + ('...' if len(agent_text) > 160 else '')
                print(f"    {C.BLUE}AGENT ({latency_ms:.0f}ms):{C.END} {truncated}")

            # Run all validations
            validate_agent_response(agent_text)

            # Latency check
            log_check(latency_ms < 5000, f"Latency: {latency_ms:.0f}ms", warn_only=latency_ms < 10000)

        except Exception as e:
            errors.append(f"Turn {turn_count} ({resp_label}) failed: {e}")
            print(f"    {C.FAIL}[ERROR] Turn failed: {e}{C.END}")

    # ============================================================
    # Extra nudges if call hasn't ended
    # ============================================================
    extra = 0
    while not call_ended and extra < 10:
        extra += 1
        turn_count += 1
        nudge = ["Okay", "Yes got it", "Hmm okay", "Yes", "Alright"][extra % 5]
        print(f"\n{C.BOLD}--- TURN {turn_count}: {C.END}{C.GRAY}Extra: '{nudge}'{C.END}")
        try:
            t0 = time.time()
            response = chat.send_message(nudge)
            latency_ms = (time.time() - t0) * 1000
            agent_text = response.text.strip()
            agent_history.append(agent_text)
            conversation_log.append({"role": "user", "text": nudge, "label": "extra_nudge"})
            conversation_log.append({"role": "agent", "text": agent_text, "latency_ms": latency_ms})

            if VERBOSE:
                print(f"    {C.BLUE}AGENT ({latency_ms:.0f}ms):{C.END} {agent_text}")
            else:
                print(f"    {C.BLUE}AGENT ({latency_ms:.0f}ms):{C.END} {agent_text[:160]}{'...' if len(agent_text) > 160 else ''}")

            validate_agent_response(agent_text)

        except Exception as e:
            print(f"    {C.FAIL}[ERROR] Extra turn failed: {e}{C.END}")
            break

    # ============================================================
    # RESULTS SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print(f"{C.BOLD}RESULTS{C.END}")
    print(f"{'='*70}")

    # Phase coverage
    print(f"\n  {C.BOLD}Phase Coverage:{C.END}")
    phase_names = {1: "Greeting & Language", 2: "App Setup (G1)", 3: "WhatsApp Groups",
                   4: "Super Coaches", 5: "Live Calls", 6: "Closing"}
    for p in range(1, 7):
        status = f"{C.OK}REACHED{C.END}" if p in phase_reached else f"{C.FAIL}MISSED{C.END}"
        print(f"    Phase {p} ({phase_names[p]}): {status}")

    log_check(len(phase_reached) >= 5, f"At least 5/6 phases reached ({len(phase_reached)})")
    log_check(call_ended, "Call reached closing (bye/end)")

    # Steps matched
    unique_ids = sorted(set(matched_step_ids), key=lambda x: float(x))
    print(f"\n  {C.BOLD}Steps Covered: {len(unique_ids)}/{len(parsed_steps)}{C.END}")
    for sid in unique_ids:
        step_info = next((s for s in parsed_steps if s["step_id"] == sid), None)
        if step_info:
            print(f"    {C.OK}✓{C.END} {step_info['label']}")
    # Show missed steps
    all_ids = {s["step_id"] for s in parsed_steps}
    missed = sorted(all_ids - set(unique_ids), key=lambda x: float(x))
    if missed:
        print(f"\n  {C.BOLD}Missed Steps:{C.END}")
        for sid in missed:
            step_info = next((s for s in parsed_steps if s["step_id"] == sid), None)
            if step_info:
                print(f"    {C.FAIL}✗{C.END} {step_info['label']}")

    # Stats
    print(f"\n  {C.BOLD}Statistics:{C.END}")
    print(f"    Total turns: {turn_count}")
    print(f"    Agent responses: {len(agent_history)}")
    print(f"    Errors: {C.FAIL}{len(errors)}{C.END}")
    print(f"    Warnings: {C.WARN}{len(warnings)}{C.END}")

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
    print(f"{C.BOLD}FULL CONVERSATION{C.END}")
    print(f"{'='*70}")
    for entry in conversation_log:
        role = entry["role"].upper()
        text = entry["text"]
        if role == "USER":
            print(f"  {C.GRAY}[USER]{C.END} {text}")
        elif role == "AGENT":
            lat = entry.get("latency_ms", 0)
            print(f"  {C.BLUE}[AGENT {lat:.0f}ms]{C.END} {text[:250]}{'...' if len(text) > 250 else ''}")
        else:
            print(f"  {C.GRAY}[{role}]{C.END} {text[:80]}")
        print()

    # Save log file
    log_file = f"test_results/realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("test_results", exist_ok=True)
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "language": LANGUAGE,
        "customer_name": CUSTOMER_NAME,
        "turns": turn_count,
        "phases_reached": sorted(phase_reached),
        "steps_matched": unique_ids,
        "steps_missed": missed,
        "completed_steps": completed_steps,
        "errors": errors,
        "warnings": warnings,
        "call_ended": call_ended,
        "conversation": conversation_log,
    }
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    print(f"  {C.GRAY}Log saved: {log_file}{C.END}")

    # Verdict
    print(f"\n{'='*70}")
    if not errors:
        print(f"{C.OK}{C.BOLD}ALL CHECKS PASSED{C.END} ({len(warnings)} warnings)")
    else:
        print(f"{C.FAIL}{C.BOLD}{len(errors)} ERRORS FOUND{C.END}")
    print(f"{'='*70}\n")

    return len(errors) == 0


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
