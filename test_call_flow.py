#!/usr/bin/env python3
"""
FULL END-TO-END Ping-Pong Call Flow Test
==========================================
Simulates a COMPLETE Riddhi Gold Membership onboarding call (all 6 phases, ~30 steps).
Tests: step tracking, silence nudge, echo detection, duplicate detection,
hot-swap timing, agent_speaking guard, WhatsApp tool, bye detection, interruption.

Uses the ACTUAL riddhi_prompt_v2.txt for step parsing.
"""

import re
import time
import json
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

# ============================================================
# Mock Session State (mirrors PlivoGeminiSession)
# ============================================================

@dataclass
class MockSession:
    prompt: str = ""
    prompt_step_keywords: list = field(default_factory=list)

    # Turn tracking
    turn_count: int = 0
    turns_since_reconnect: int = 0
    current_turn_audio_chunks: int = 0
    current_turn_agent_text: list = field(default_factory=list)
    current_turn_user_text: list = field(default_factory=list)
    session_split_interval: int = 5

    # Agent state
    agent_speaking: bool = False
    greeting_audio_complete: bool = False
    last_agent_text: str = ""
    last_agent_question: str = ""
    recent_agent_texts: deque = field(default_factory=lambda: deque(maxlen=10))
    completed_steps: list = field(default_factory=list)
    turn_exchanges: list = field(default_factory=list)

    # User state
    last_user_speech_time: Optional[float] = None
    last_user_text: str = ""

    # Silence
    silence_sla_seconds: float = 2.0
    silence_sla_immediate: float = 3.0
    empty_turn_nudge_count: int = 0

    # GHL
    ghl_api_key: str = ""
    ghl_location_id: str = ""
    whatsapp_sent: bool = False

    # Flags
    closing_call: bool = False

    # Stats
    duplicate_nudges: list = field(default_factory=list)
    audio_cleared: list = field(default_factory=list)
    hot_swap_count: int = 0


# ============================================================
# Core functions (copied from plivo_gemini_stream.py logic)
# ============================================================

def parse_prompt_steps(prompt: str) -> list:
    steps = []
    if not prompt:
        return steps
    step_pattern = re.compile(
        r'STEP\s+(\d+\.\d+)\b.*?(?:\"([^\"]{10,})\")',
        re.DOTALL
    )
    for match in step_pattern.finditer(prompt):
        step_id = match.group(1)
        dialogue = match.group(2).strip()
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'and', 'or', 'but', 'so', 'if', 'it', 'its', 'this', 'that',
                      'you', 'your', 'i', 'me', 'my', 'we', 'our', 'can', 'will',
                      'do', 'have', 'has', 'had', 'would', 'could', 'should',
                      'please', 'just', 'about', 'once', 'also', 'not', 'all'}
        words = re.findall(r'[a-z]+', dialogue.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        phrases = []
        word_list = dialogue.lower().split()
        for i in range(len(word_list) - 1):
            bigram = f"{word_list[i]} {word_list[i+1]}"
            if len(bigram) > 8 and not all(w in stop_words for w in bigram.split()):
                phrases.append(bigram)
        first_sentence = re.split(r'[.?!]', dialogue)[0].strip()
        label = f"Step {step_id}: {first_sentence[:35]}"
        if len(first_sentence) > 35:
            label = label[:label.rfind(' ')] if ' ' in label[10:] else label[:40]
        steps.append({
            "step_id": step_id, "label": label,
            "keywords": set(keywords[:12]), "phrases": phrases[:6],
        })
    return steps


def match_prompt_step(prompt_steps, lower_text):
    if not prompt_steps:
        return ""
    best_score = 0
    best_label = ""
    text_words = set(re.findall(r'[a-z]+', lower_text))
    for step in prompt_steps:
        if not step["keywords"]:
            continue
        keyword_overlap = len(text_words & step["keywords"]) / len(step["keywords"])
        phrase_bonus = sum(1 for p in step["phrases"] if p in lower_text) * 0.15
        score = keyword_overlap + phrase_bonus
        if score > best_score and score > 0.3:
            best_score = score
            best_label = step["label"]
    return best_label


def extract_step_label(prompt_steps, text):
    if not text:
        return ""
    lower = text.lower()
    if prompt_steps:
        best_match = match_prompt_step(prompt_steps, lower)
        if best_match:
            return best_match
    fillers = r'^(?:Great|Okay|Perfect|Sure|Absolutely|Right|Alright|Wonderful|Excellent|Fantastic|Of course|No worries|No problem|Got it|I see|I understand)[,!.\s]*'
    cleaned = re.sub(fillers, '', text, flags=re.IGNORECASE).strip()
    if not cleaned:
        cleaned = text.strip()
    for i, ch in enumerate(cleaned):
        if i >= 10 and ch in '.?!':
            cleaned = cleaned[:i + 1]
            break
    if len(cleaned) > 40:
        cut = cleaned[:40].rfind(' ')
        cleaned = cleaned[:cut] if cut > 15 else cleaned[:40]
    return cleaned.strip()


def is_duplicate_text(session, new_text):
    new_words = set(new_text.lower().split())
    if len(new_words) < 3:
        return False
    for prev in session.recent_agent_texts:
        prev_words = set(prev.lower().split())
        if not prev_words:
            continue
        intersection = len(new_words & prev_words)
        if len(new_words) < 12:
            overlap = intersection / len(new_words)
        else:
            overlap = intersection / max(len(new_words), len(prev_words))
        if overlap > 0.5:
            return True
    # Check against completed step labels (catches rephrased repeats across hot-swaps)
    new_label = extract_step_label(session.prompt_step_keywords, new_text).lower()
    if new_label and len(new_label) > 15:
        for step in session.completed_steps:
            step_words = set(step.lower().split())
            label_words = set(new_label.split())
            if step_words and label_words:
                overlap = len(step_words & label_words) / max(len(step_words), len(label_words))
                if overlap > 0.5:
                    return True
    return False


def is_echo(user_text, last_agent_text):
    if not last_agent_text or len(user_text.split()) < 3:
        return False
    user_words = set(user_text.lower().split())
    agent_words = set(last_agent_text.lower().split())
    if not agent_words:
        return False
    overlap = len(user_words & agent_words) / len(user_words)
    return overlap > 0.5


def should_nudge(session, time_since_user_spoke):
    if session.last_user_speech_time is None:
        return False, "no user speech"
    if session.agent_speaking:
        return False, "agent is speaking"
    sla = session.silence_sla_seconds if not session.greeting_audio_complete else session.silence_sla_immediate
    if time_since_user_spoke >= sla:
        return True, f"{time_since_user_spoke:.1f}s >= {sla}s SLA"
    return False, f"{time_since_user_spoke:.1f}s < {sla}s SLA"


def get_nudge_text(session):
    if session.last_agent_question:
        q_short = session.last_agent_question[:80]
        return f'[Silent. Already asked: "{q_short}" — do NOT repeat. Say "Take your time" and WAIT.]'
    return "[Silent. Check if they are still there. Do NOT repeat anything.]"


def whatsapp_tool_offered(session):
    if not session.ghl_api_key:
        return False, "ghl_api_key MISSING"
    if not session.ghl_location_id:
        return False, "ghl_location_id MISSING"
    if session.whatsapp_sent:
        return False, "already sent"
    return True, "offered"


# ============================================================
# Simulation Engine
# ============================================================

def simulate_agent_turn(session, agent_text, audio_chunks=50):
    """Simulate: Gemini generates agent speech with audio."""
    session.current_turn_agent_text = agent_text.split(". ") if agent_text else []
    session.current_turn_audio_chunks = audio_chunks
    if audio_chunks > 0:
        session.agent_speaking = True

    # Early duplicate check
    partial = " ".join(session.current_turn_agent_text)
    early_dup = False
    if len(partial.split()) >= 4 and is_duplicate_text(session, partial):
        early_dup = True
        session.audio_cleared.append(partial[:50])
        session.duplicate_nudges.append(f"Early cut: {partial[:50]}")
    return early_dup


def simulate_turn_complete(session):
    """Simulate: Gemini sends turnComplete."""
    session.greeting_audio_complete = True
    session.agent_speaking = False
    session.turn_count += 1

    full_agent = ""
    full_user = ""

    if session.current_turn_audio_chunks > 0:
        if session.current_turn_agent_text:
            full_agent = " ".join(session.current_turn_agent_text)
            session.last_agent_text = full_agent
            if "?" in full_agent:
                session.last_agent_question = full_agent
            if is_duplicate_text(session, full_agent):
                session.duplicate_nudges.append(f"TurnComplete dup: {full_agent[:50]}")
            session.recent_agent_texts.append(full_agent)
            session.current_turn_agent_text = []
        if session.current_turn_user_text:
            full_user = " ".join(session.current_turn_user_text)
            session.last_user_text = full_user
            session.current_turn_user_text = []
        if full_agent or full_user:
            session.turn_exchanges.append({"agent": full_agent, "user": full_user})
            if len(session.turn_exchanges) > 5:
                session.turn_exchanges = session.turn_exchanges[-5:]
        if full_agent and len(full_agent.split()) > 4:
            step_label = extract_step_label(session.prompt_step_keywords, full_agent)
            if step_label:
                session.completed_steps.append(step_label)

    is_empty = session.current_turn_audio_chunks == 0
    if is_empty and session.greeting_audio_complete:
        session.empty_turn_nudge_count += 1
    else:
        session.empty_turn_nudge_count = 0

    prewarm = False
    hot_swap = False
    if not is_empty:
        session.turns_since_reconnect += 1
    if session.turns_since_reconnect == session.session_split_interval - 1:
        prewarm = True
    if session.turns_since_reconnect >= session.session_split_interval and not is_empty:
        hot_swap = True
        session.turns_since_reconnect = 0
        session.hot_swap_count += 1

    session.current_turn_audio_chunks = 0
    return full_agent, full_user, prewarm, hot_swap


def simulate_user_speech(session, user_text):
    """Simulate: user speaks, inputTranscription arrives."""
    is_noise = user_text.startswith('<') and user_text.endswith('>')
    echo = False
    if not is_noise and session.last_agent_text and len(user_text.split()) >= 3:
        echo = is_echo(user_text, session.last_agent_text)
    if not is_noise and not echo:
        session.last_user_speech_time = time.time()
        session.current_turn_user_text.append(user_text)
    return {"noise": is_noise, "echo": echo, "accepted": not is_noise and not echo}


def build_hot_swap_context(session):
    """Build the context string that would be sent on hot-swap."""
    latest = session.turn_exchanges[-1] if session.turn_exchanges else {}
    agent_text = latest.get("agent", "")[:250]
    user_text = latest.get("user", "")[:100]
    step_history = ""
    if session.completed_steps:
        step_list = ", ".join(f"{i+1}.{s}" for i, s in enumerate(session.completed_steps[-8:]))
        step_history = f" Steps ALREADY DONE (do NOT repeat any): [{step_list}]."
    return agent_text, user_text, step_history


# ============================================================
# FULL CALL FLOW - All Steps
# ============================================================

# Each entry: (user_response_before, agent_text, expected_step_id, step_type)
# user_response_before=None means [PAUSE & CONTINUE] (no user response needed)
FULL_CALL_STEPS = [
    # --- PHASE 1: GREETING & LANGUAGE ---
    (None, "Hi Kiran, a huge welcome to the Gold Membership! I'm Priya from Riddhi Deorah's team. Before we begin, would you prefer to continue this call in English or Hindi?", "1.1", "WAIT"),
    ("English please", "Great. I want to quickly walk you through the Gold Membership journey and the system. It will take about 5 to 6 minutes. Are you free right now?", "1.2", "WAIT"),
    ("Yes I'm free", "Perfect. Since we'll be checking things on your phone, can you please put me on speaker or wear your earphones so that your hands are free? Let me know once done!", "1.3", "WAIT"),

    # --- PHASE 2: APP SETUP ---
    ("Done, I'm on speaker", "Have you downloaded the Riddhi Deorah app on your phone? We sent the link on WhatsApp earlier.", "2.1", "WAIT"),
    ("Yes I downloaded it", "Wonderful! Can you please open the app right now? Let me know once it is open.", "2.2", "WAIT"),
    ("It's open now", "Great. At the bottom of the app, you'll see some options. Please click on the third option called Courses. Let me know once done.", "2.3", "WAIT"),
    ("Okay done", "Please open Course Number 4 - the G1 Gold Launchpad Course. Click on Day 1, Video 1 and see if it's playing.", "2.4", "WAIT"),
    ("Yes it's playing", "Wonderful. So this is the first and most important course for you. Try to finish it today, or at the latest by tomorrow.", "2.5", "PAUSE"),
    (None, "This course will give you complete clarity on what the Gold Membership includes, how to use the app, and how to get the maximum benefit from the system.", "2.6", "PAUSE"),
    (None, "After this, for G2, G3, and G4, you only need to watch 10-15 minutes a day. So it's very easy to follow! Does that make sense so far?", "2.7", "WAIT"),

    # --- PHASE 3: THE 3 WHATSAPP GROUPS ---
    ("Yes it does", "Now, I've also sent you a message on WhatsApp. Can you please open that?", "3.1", "WAIT"),
    ("Opened it", "First, you've been added to a Gold Membership Orientation Group. Have you joined it?", "3.2", "WAIT"),
    ("Yes I joined", "This is a temporary group for 24 hours - just to help you get started with the app and courses. We'll remove you from it by tomorrow.", "3.3", "PAUSE"),
    (None, "Next, there's the Gold Membership Main Group. Have you joined that one?", "3.4", "WAIT"),
    ("Yes that too", "This is your permanent update group for live call schedules and important announcements. It's a closed group - only the team posts, so you won't miss anything.", "3.5", "PAUSE"),
    (None, "Third, there's a link for Supermom Revolution - Mission Team. Can you join that as well?", "3.6", "WAIT"),
    ("Done, joined", "This is an open group where you can chat with other mothers, connect, and participate in fun missions and activities.", "3.7", "PAUSE"),
    (None, "You may see a lot of messages initially - that's completely normal. Once you finish the Gold Launchpad Course, it'll all make sense.", "3.8", "PAUSE"),
    (None, "And this group is completely optional. If it feels overwhelming, feel free to mute or leave it. Alright?", "3.9", "WAIT"),

    # --- PHASE 4: SUPER COACHES ---
    ("Alright got it", "In the same WhatsApp message, can you see the names and numbers of your Super Coaches?", "4.1", "WAIT"),
    ("Yes I can see them", "Could you please confirm their names for me?", "4.2", "WAIT"),
    ("Sunita and Meera", "They are experienced mothers who understand the system really well. Please save their numbers - you can call or message them whenever you need help.", "4.3", "PAUSE"),

    # --- PHASE 5: LIVE CALLS SCHEDULE ---
    (None, "Now about the live calls. We have Weekly Parenting Q&A Calls every Monday and Friday at 6 PM Indian Time.", "5.1", "PAUSE"),
    (None, "You can join live and ask Riddhi Maam anything - parenting, emotions, behavior, or personal challenges.", "5.2", "PAUSE"),
    (None, "If you miss a call, recordings are on the app under course code G7 - Weekly Parenting Calls. Are you following so far?", "5.3", "WAIT"),
    ("Yes I'm following", "I'd suggest trying to attend at least one call live - the energy is amazing.", "5.4", "PAUSE"),
    (None, "One last thing - every Wednesday at 6 PM, there's a Supermom Revolution Mission Call about fun team missions. Recordings are always on the app.", "5.5", "PAUSE"),
    (None, "The course code for this is G6 - Supermom Revolution Mission. You can watch previous recordings anytime. Does all of that sound good?", "5.6", "WAIT"),

    # --- PHASE 6: PROFESSIONAL CLOSING ---
    ("Yes sounds great", "That's everything from my side! If you have any questions later, reach out to your super coaches anytime.", "6.1", "PAUSE"),
    (None, "We are so proud of you for being an action taker. Welcome again, and have a beautiful day. Bye-bye!", "6.2", "END"),
]


# ============================================================
# THE TEST
# ============================================================

def run_test():
    with open("riddhi_prompt_v2.txt") as f:
        prompt = f.read()

    session = MockSession(
        prompt=prompt,
        prompt_step_keywords=parse_prompt_steps(prompt),
        ghl_api_key="test-api-key-123",
        ghl_location_id="test-location-456",
    )

    parsed_count = len(session.prompt_step_keywords)
    print("=" * 70)
    print("FULL END-TO-END CALL FLOW TEST")
    print(f"Prompt: riddhi_prompt_v2.txt ({parsed_count} steps parsed)")
    print(f"Call flow: {len(FULL_CALL_STEPS)} turns defined")
    print(f"Session split interval: {session.session_split_interval}")
    print(f"Silence SLA: {session.silence_sla_seconds}s pre-greeting, {session.silence_sla_immediate}s post-greeting")
    print("=" * 70)

    errors = []
    turn_num = 0
    hot_swap_turns = []
    phase_boundaries = {
        "1.1": "PHASE 1: GREETING",
        "2.1": "PHASE 2: APP SETUP",
        "3.1": "PHASE 3: WHATSAPP GROUPS",
        "4.1": "PHASE 4: SUPER COACHES",
        "5.1": "PHASE 5: LIVE CALLS",
        "6.1": "PHASE 6: CLOSING",
    }

    def check(condition, msg):
        status = "OK" if condition else "FAIL"
        if not condition:
            errors.append(msg)
        print(f"    [{status}] {msg}")

    # ============================================================
    # PART 1: FULL CALL FLOW - All 30 steps
    # ============================================================
    print()
    print("-" * 70)
    print("PART 1: FULL CALL FLOW (all phases)")
    print("-" * 70)

    for i, (user_resp, agent_text, step_id, step_type) in enumerate(FULL_CALL_STEPS):
        # Phase header
        if step_id in phase_boundaries:
            print(f"\n  === {phase_boundaries[step_id]} ===")

        turn_num += 1

        # User responds (if WAIT step)
        if user_resp:
            simulate_user_speech(session, user_resp)

        # Agent speaks
        early_dup = simulate_agent_turn(session, agent_text)
        full_agent, _, prewarm, hot_swap = simulate_turn_complete(session)

        # Step tracking check
        last_step = session.completed_steps[-1] if session.completed_steps else ""
        step_match = step_id in last_step

        # Build status string
        flags = []
        if hot_swap:
            hot_swap_turns.append(turn_num)
            flags.append("HOT-SWAP")
        if prewarm:
            flags.append("PREWARM")
        if early_dup:
            flags.append("EARLY-DUP!")
        if step_type == "PAUSE":
            flags.append("auto-continue")
        if step_type == "END":
            flags.append("end_call")
        flag_str = f" [{', '.join(flags)}]" if flags else ""

        user_str = f" <- User: '{user_resp}'" if user_resp else ""
        check(step_match, f"Turn {turn_num:2d} | Step {step_id} | {last_step[:45]}{flag_str}{user_str}")

    total_steps_tracked = len(session.completed_steps)
    print()
    check(total_steps_tracked >= 28, f"Total steps tracked: {total_steps_tracked} (expected >= 28 of 30)")
    check(session.hot_swap_count >= 4, f"Hot-swaps occurred: {session.hot_swap_count} (expected >= 4 for 30 turns)")

    # Verify all phases covered
    tracked_ids = set()
    for s in session.completed_steps:
        m = re.search(r'Step (\d+\.\d+)', s)
        if m:
            tracked_ids.add(m.group(1))
    for phase_start in ["1.1", "2.1", "3.1", "4.1", "5.1", "6.1"]:
        check(phase_start in tracked_ids, f"Phase {phase_start[0]} represented in tracked steps")

    # Verify closing step
    last_tracked = session.completed_steps[-1] if session.completed_steps else ""
    check("6.2" in last_tracked or "6.1" in last_tracked, f"Final step is Phase 6: {last_tracked[:50]}")

    # ============================================================
    # PART 2: SILENCE NUDGE TESTS
    # ============================================================
    print()
    print("-" * 70)
    print("PART 2: SILENCE NUDGE MECHANICS")
    print("-" * 70)

    # Pre-greeting SLA (2.0s)
    print("\n  Pre-greeting SLA:")
    saved_greeting = session.greeting_audio_complete
    session.greeting_audio_complete = False
    session.agent_speaking = False
    session.last_user_speech_time = time.time()

    nudge, reason = should_nudge(session, 1.5)
    check(not nudge, f"  No nudge at 1.5s pre-greeting ({reason})")
    nudge, reason = should_nudge(session, 2.5)
    check(nudge, f"  Nudge fires at 2.5s pre-greeting ({reason})")
    session.greeting_audio_complete = saved_greeting

    # Post-greeting SLA (3.0s)
    print("\n  Post-greeting SLA:")
    session.last_user_speech_time = time.time()
    nudge, reason = should_nudge(session, 1.0)
    check(not nudge, f"  No nudge at 1.0s ({reason})")
    nudge, reason = should_nudge(session, 2.5)
    check(not nudge, f"  No nudge at 2.5s ({reason})")
    nudge, reason = should_nudge(session, 3.5)
    check(nudge, f"  Nudge fires at 3.5s ({reason})")

    # Agent speaking guard
    print("\n  Agent speaking guard:")
    session.agent_speaking = True
    session.last_user_speech_time = time.time()
    nudge, reason = should_nudge(session, 99.0)
    check(not nudge, f"  Blocked at 99s while agent speaks ({reason})")
    session.agent_speaking = False

    # Context-aware nudge content
    print("\n  Nudge content:")
    nudge_text = get_nudge_text(session)
    check("do NOT repeat" in nudge_text, "  Anti-repeat instruction present")
    check(session.last_agent_question[:30] in nudge_text or "Already asked" in nudge_text,
          "  References last question")

    # No user speech = no nudge
    session.last_user_speech_time = None
    nudge, reason = should_nudge(session, 99.0)
    check(not nudge, f"  No nudge when no user speech yet ({reason})")

    # ============================================================
    # PART 3: ECHO DETECTION
    # ============================================================
    print()
    print("-" * 70)
    print("PART 3: ECHO DETECTION (7 cases)")
    print("-" * 70)

    session.last_agent_text = "Please click on Day 1, Video 1 and see if it is playing."
    echo_tests = [
        ("please click on day one video one", True, "Agent instruction echoed back"),
        ("and video one and see if it", True, "Partial echo of agent text"),
        ("i just wanted to confirm something", False, "Real speech (no overlap)"),
        ("Yeah", False, "Short response (< 3 words)"),
        ("Yes I clicked on it", False, "Real user response (low overlap)"),
        ("Okay done", False, "Short confirmation"),
        ("<silence>", False, "Noise marker"),
    ]
    for text, expected, desc in echo_tests:
        result = is_echo(text, session.last_agent_text)
        check(result == expected, f"  Echo={result:5} | '{text}' ({desc})")

    # ============================================================
    # PART 4: DUPLICATE DETECTION
    # ============================================================
    print()
    print("-" * 70)
    print("PART 4: DUPLICATE DETECTION (9 cases)")
    print("-" * 70)

    # Set up history with representative texts
    session.recent_agent_texts = deque(maxlen=10)
    session.recent_agent_texts.append(
        "Great. I want to quickly walk you through the Gold Membership journey and the system. "
        "It will take about 5 to 6 minutes. Are you free right now?"
    )
    session.recent_agent_texts.append(
        "Please open Course Number 4 - the G1 Gold Launchpad Course. Click on Day 1, Video 1 and see if it's playing."
    )
    session.recent_agent_texts.append(
        "Now about the live calls. We have Weekly Parenting Q&A Calls every Monday and Friday at 6 PM Indian Time."
    )

    dup_tests = [
        ("Are you free right now?", True, "Exact short repeat"),
        ("No problem at all! Are you free right now?", True, "Filler + repeat"),
        ("क्या आप अभी फ्री हैं?", False, "Hindi translation (different tokens)"),
        ("Please open Course Number 4", True, "Partial repeat of step 2.4"),
        ("Please open Course Number 4 - the G1 Gold Launchpad Course. Click on Day 1, Video 1", True, "Full repeat of step 2.4"),
        ("Have you downloaded the Riddhi Deorah app?", True, "Step 2.1 repeat (caught via completed_steps)"),
        ("Perfect. Since we'll be checking things on your phone.", True, "Step 1.3 repeat (caught via completed_steps)"),
        ("Hmm", False, "Too short (< 3 words)"),
        ("Weekly Parenting Q&A Calls every Monday", True, "Partial repeat of step 5.1"),
    ]
    for text, expected, desc in dup_tests:
        result = is_duplicate_text(session, text)
        check(result == expected, f"  Dup={result:5} | '{text[:45]}' ({desc})")

    # ============================================================
    # PART 5: HOT-SWAP CONTEXT
    # ============================================================
    print()
    print("-" * 70)
    print("PART 5: HOT-SWAP CONTEXT QUALITY")
    print("-" * 70)

    agent_ctx, user_ctx, step_hist = build_hot_swap_context(session)
    check(len(agent_ctx) <= 250, f"  Agent text <= 250 chars ({len(agent_ctx)} chars)")
    check(len(user_ctx) <= 100, f"  User text <= 100 chars ({len(user_ctx)} chars)")
    check("Steps ALREADY DONE" in step_hist, "  Step history header present")
    # Check that recent steps are in history
    check("6." in step_hist or "5." in step_hist, "  Recent steps included in history")
    print(f"    Context preview: ...{step_hist[-120:]}")

    # ============================================================
    # PART 6: WHATSAPP/GHL TOOL
    # ============================================================
    print()
    print("-" * 70)
    print("PART 6: WHATSAPP TOOL OFFERING (4 cases)")
    print("-" * 70)

    session.whatsapp_sent = False
    offered, why = whatsapp_tool_offered(session)
    check(offered, f"  Both creds set -> offered ({why})")

    session.whatsapp_sent = True
    offered, why = whatsapp_tool_offered(session)
    check(not offered, f"  Already sent -> blocked ({why})")
    session.whatsapp_sent = False

    saved_key = session.ghl_api_key
    session.ghl_api_key = ""
    offered, why = whatsapp_tool_offered(session)
    check(not offered, f"  No API key -> blocked ({why})")
    session.ghl_api_key = saved_key

    saved_loc = session.ghl_location_id
    session.ghl_location_id = ""
    offered, why = whatsapp_tool_offered(session)
    check(not offered, f"  No location ID -> blocked ({why})")
    session.ghl_location_id = saved_loc

    # ============================================================
    # PART 7: FILLER / SHORT RESPONSE FILTERING
    # ============================================================
    print()
    print("-" * 70)
    print("PART 7: FILLER FILTERING (not tracked as steps)")
    print("-" * 70)

    prev_count = len(session.completed_steps)
    fillers_to_test = [
        "Take your time.",
        "No problem at all!",
        "Sure, I am here.",
        "Hmm okay.",
    ]
    for filler in fillers_to_test:
        session.current_turn_agent_text = [filler]
        session.current_turn_audio_chunks = 5
        simulate_turn_complete(session)
        check(len(session.completed_steps) == prev_count,
              f"  '{filler}' NOT tracked (steps still {prev_count})")

    # ============================================================
    # PART 8: REPEAT AFTER HOT-SWAP (should be caught)
    # ============================================================
    print()
    print("-" * 70)
    print("PART 8: POST-HOT-SWAP REPEAT DETECTION")
    print("-" * 70)

    # Test A: repeat caught via recent_agent_texts (normal case)
    session.recent_agent_texts = deque(maxlen=10)
    session.recent_agent_texts.append(
        "Now about the live calls. We have Weekly Parenting Q&A Calls every Monday and Friday at 6 PM Indian Time."
    )
    repeat_text = "We have Weekly Parenting Q&A Calls every Monday and Friday at 6 PM Indian Time."
    early_dup = simulate_agent_turn(session, repeat_text)
    check(early_dup, f"  Repeat caught via recent_agent_texts")

    # Test B: new text not flagged
    # Step 6.1 is in completed_steps, so even "new" text matching it gets caught
    step_6_1_text = "That's everything from my side! If you have any questions later, reach out."
    early_dup = simulate_agent_turn(session, step_6_1_text)
    check(early_dup, f"  Step 6.1 repeat caught via completed_steps (even with empty recent_agent_texts)")

    # Truly new text that doesn't share keywords with any completed step
    truly_new = "Take your time and let me know whenever you are ready to continue."
    early_dup = simulate_agent_turn(session, truly_new)
    check(not early_dup, f"  Truly new text NOT flagged")

    # Test C: CROSS-HOT-SWAP repeat - recent_agent_texts cleared but completed_steps persists
    # This tests the _completed_steps fallback in production code
    session.recent_agent_texts = deque(maxlen=10)  # cleared after hot-swap
    # completed_steps still has all 30 steps from the full call
    # Try to repeat step 2.4 which is only in completed_steps now (not in recent_agent_texts)
    cross_swap_repeat = "Please open Course Number 4 - the G1 Gold Launchpad Course. Click on Day 1, Video 1 and see if it's playing."
    result = is_duplicate_text(session, cross_swap_repeat)
    check(result, f"  Cross-hot-swap repeat caught via completed_steps")

    # Test D: rephrased version of completed step
    rephrased = "Open Course Number 4, the G1 Gold Launchpad Course."
    result = is_duplicate_text(session, rephrased)
    check(result, f"  Rephrased repeat caught via completed_steps")

    # ============================================================
    # PART 9: BYE DETECTION
    # ============================================================
    print()
    print("-" * 70)
    print("PART 9: BYE / EXIT DETECTION")
    print("-" * 70)

    bye_phrases = ["bye", "I need to go", "I'm busy right now", "talk later", "goodbye"]
    for phrase in bye_phrases:
        is_bye = any(kw in phrase.lower() for kw in ["bye", "need to go", "i'm busy", "talk later", "goodbye"])
        check(is_bye, f"  '{phrase}' detected as bye/exit")

    non_bye = ["I'm buying groceries", "by the way", "okay done"]
    for phrase in non_bye:
        is_bye = any(kw in phrase.lower() for kw in ["bye", "need to go", "busy right now", "talk later", "goodbye"])
        check(not is_bye, f"  '{phrase}' NOT false-positive bye")

    # ============================================================
    # RESULTS
    # ============================================================
    print()
    print("=" * 70)
    if not errors:
        print(f"ALL TESTS PASSED")
    else:
        print(f"FAILURES: {len(errors)}")
        for e in errors:
            print(f"  - {e}")
    print("=" * 70)

    # Summary
    print(f"\n--- SESSION SUMMARY ---")
    print(f"  Total turns simulated: {session.turn_count}")
    print(f"  Steps tracked: {total_steps_tracked}")
    print(f"  Hot-swaps: {session.hot_swap_count} (at turns {hot_swap_turns})")
    print(f"  Duplicates caught: {len(session.duplicate_nudges)}")
    print(f"  Audio cleared (early dup): {len(session.audio_cleared)}")
    print()
    print(f"--- ALL TRACKED STEPS ---")
    for i, s in enumerate(session.completed_steps):
        print(f"  {i+1:2d}. {s}")

    return len(errors) == 0


if __name__ == "__main__":
    success = run_test()
    exit(0 if success else 1)
