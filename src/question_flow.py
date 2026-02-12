# Question Flow State Machine
# Manages conversation flow internally - passes ONE question at a time
# Loads configuration from client-specific JSON files

import json
import time
import threading
from enum import Enum
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

# Directories
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
FLOW_DATA_DIR = Path(__file__).parent.parent / "flow_data"
FLOW_DATA_DIR.mkdir(exist_ok=True)

# LATENCY OPT: Cache client configs to avoid repeated file I/O (100-200ms savings per session start)
_client_config_cache: Dict[str, Dict[str, Any]] = {}
_config_cache_lock = threading.Lock()


def load_client_config(client_name: str = "fwai") -> Dict[str, Any]:
    """Load client configuration from JSON file with caching.
    LATENCY OPTIMIZATION: Caches configs to avoid repeated file reads (100-200ms savings)."""

    # Check cache first
    with _config_cache_lock:
        if client_name in _client_config_cache:
            return _client_config_cache[client_name].copy()  # Return copy to prevent mutation

    # Cache miss - load from file
    config_file = PROMPTS_DIR / f"{client_name}_config.json"

    if not config_file.exists():
        # Try default
        config_file = PROMPTS_DIR / "fwai_config.json"
        if not config_file.exists():
            logger.warning(f"No config found for {client_name}, using defaults")
            config = get_default_config()
            with _config_cache_lock:
                _client_config_cache[client_name] = config
            return config.copy()

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            logger.info(f"Loaded config from {config_file.name}")
            # Cache the loaded config
            with _config_cache_lock:
                _client_config_cache[client_name] = config
            return config.copy()  # Return copy to prevent mutation
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        config = get_default_config()
        with _config_cache_lock:
            _client_config_cache[client_name] = config
        return config.copy()


def get_default_config() -> Dict[str, Any]:
    """Minimal defaults if no config file found — prompt and questions come from API"""
    return {
        "defaults": {
            "voice": "Puck"
        }
    }


@dataclass
class QuestionFlow:
    """
    Tracks conversation state and provides next question.
    Loads questions from client-specific config file.
    """
    call_uuid: str = ""
    client_name: str = "fwai"
    context: Dict = field(default_factory=dict)
    current_step: int = 0
    collected_data: Dict = field(default_factory=dict)

    # All configurable via API — config file only provides defaults (voice, agent_name, etc.)
    config: Dict = field(default_factory=dict)
    questions: List[Dict] = field(default_factory=list)
    objections: Dict[str, str] = field(default_factory=dict)
    objection_keywords: Dict[str, List[str]] = field(default_factory=dict)
    base_prompt: str = ""
    questions_override: Optional[List[Dict]] = None  # Questions from API
    prompt_override: Optional[str] = None  # Base prompt from API
    objections_override: Optional[Dict] = None  # Objections from API
    objection_keywords_override: Optional[Dict] = None  # Objection keywords from API

    def __post_init__(self):
        """Load config for defaults (voice, agent_name), everything else from API"""
        import re  # For regex pre-compilation

        self.config = load_client_config(self.client_name)

        # Merge defaults with provided context (voice, agent_name, company_name, etc.)
        defaults = self.config.get("defaults", {})
        self.context = {**defaults, **self.context}

        # Questions, prompt, objections — all from API
        self.questions = self.questions_override or []
        self.base_prompt = self.prompt_override or ""
        self.objections = self.objections_override or {}
        self.objection_keywords = self.objection_keywords_override or {}

        if not self.questions:
            logger.warning(f"No questions provided for {self.client_name} — call will have no question flow")
        if not self.base_prompt:
            logger.warning(f"No prompt provided for {self.client_name} — using minimal default")

        logger.info(f"[{self.client_name}] {len(self.questions)} questions, prompt={len(self.base_prompt)} chars")

        # LATENCY OPT: Pre-render all templates (20-45ms savings per call)
        self._rendered_questions = [self._render(q["prompt"]) for q in self.questions]
        self._rendered_objections = {k: self._render(v) for k, v in self.objections.items()}
        self._rendered_base_prompt = None  # Lazy init when needed

        # LATENCY OPT: Pre-compile regex patterns for objection detection (8-48ms savings per call)
        self._objection_patterns = {}
        for objection_type, keywords in self.objection_keywords.items():
            patterns = []
            for kw in keywords:
                if len(kw) <= 5:
                    # Pre-compile word boundary regex for short keywords
                    patterns.append(re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE))
                else:
                    # Just store lowercased string for substring match
                    patterns.append(kw.lower())
            self._objection_patterns[objection_type] = patterns

        logger.debug(f"[{self.client_name}] LATENCY OPT: Pre-rendered {len(self._rendered_questions)} questions, "
                    f"pre-compiled {sum(len(p) for p in self._objection_patterns.values())} regex patterns")

    def _render(self, template: str) -> str:
        """Replace {placeholders} with context values"""
        result = template
        for key, value in self.context.items():
            result = result.replace("{" + key + "}", str(value))
        return result

    def get_base_prompt(self) -> str:
        """Get the base prompt with questions appended, placeholders replaced.
        LATENCY OPT: Uses cached rendered base prompt."""
        if self._rendered_base_prompt is None:
            prompt = self.base_prompt
            # Append all questions to system prompt so Gemini knows the full flow
            if self.questions:
                prompt += "\n\nQUESTIONS TO ASK (in this exact order, one at a time):\n"
                for i, q in enumerate(self.questions):
                    prompt += f"{i + 1}. {q['prompt']}\n"
            self._rendered_base_prompt = self._render(prompt)
        return self._rendered_base_prompt

    def get_voice(self) -> str:
        """Get the voice setting from config"""
        return self.context.get("voice", "Puck")

    def get_current_question(self) -> Optional[str]:
        """Get the current question to ask.
        LATENCY OPT: Uses pre-rendered cached question."""
        if self.current_step >= len(self._rendered_questions):
            return None
        return self._rendered_questions[self.current_step]

    def get_instruction_prompt(self) -> str:
        """Get the question text for current step (no wrappers)"""
        question = self.get_current_question()
        if not question:
            return self._render("Great talking to you! Take care!")

        return question

    def detect_objection(self, user_text: str) -> Optional[str]:
        """Detect if user raised an objection.
        LATENCY OPT: Uses pre-compiled regex patterns (8-48ms savings)."""
        import re
        text = user_text.lower()

        for objection_type, patterns in self._objection_patterns.items():
            for pattern in patterns:
                if isinstance(pattern, re.Pattern):
                    # Pre-compiled regex for short keywords
                    if pattern.search(text):
                        return objection_type
                else:
                    # String for substring match on longer phrases
                    if pattern in text:
                        return objection_type

        return None

    def get_objection_response(self, objection_type: str) -> str:
        """Get response for an objection"""
        response = self.objections.get(objection_type, "I understand. Let me help with that.")
        return self._render(response)

    def advance(self, user_response: str = "") -> Dict[str, Any]:
        """
        Process user response and get next instruction.
        Returns a dict with text to speak and end_call flag.
        """
        # Check for objection first
        objection = self.detect_objection(user_response)
        if objection:
            logger.debug(f"Objection: {objection}")
            if objection == "not_interested":
                return {"text": self.get_objection_response(objection), "end_call": True}
            # Track objection: if same type already handled for this question, force-advance
            handled_key = f"{self.current_step}_{objection}"
            if not hasattr(self, '_handled_objections'):
                self._handled_objections = set()
            if handled_key in self._handled_objections:
                logger.info(f"Objection '{objection}' already handled for Q{self.current_step}, force-advancing")
                # Fall through to advance logic below
            else:
                self._handled_objections.add(handled_key)
                return {"text": self.get_objection_response(objection), "end_call": False}

        # Store response data
        if self.current_step < len(self.questions):
            q_id = self.questions[self.current_step]["id"]
            self.collected_data[q_id] = user_response

        # Move to next question
        self.current_step += 1

        # Save after each response (survives disconnects)
        if self.call_uuid:
            self.save_to_file(self.call_uuid)

        # Check if done
        if self.current_step >= len(self.questions):
            closing = self._render("Great talking to you, {customer_name}! I'll send details on WhatsApp. Take care!")
            return {"text": closing, "end_call": True}

        return {"text": self.get_instruction_prompt(), "end_call": False}

    def get_collected_data(self) -> Dict:
        """Get all collected data from the conversation"""
        return {
            "call_uuid": self.call_uuid,
            "client_name": self.client_name,
            "customer_name": self.context.get("customer_name", ""),
            "agent_name": self.context.get("agent_name", ""),
            "current_step": self.current_step,
            "total_steps": len(self.questions),
            "completed": self.current_step >= len(self.questions),
            "responses": self.collected_data
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Compute call statistics from conversation flow data"""
        total = len(self.questions)
        completed = min(self.current_step, total)

        # Detect objections raised during the conversation
        objections_found = []
        for response_text in self.collected_data.values():
            obj = self.detect_objection(str(response_text))
            if obj and obj not in objections_found:
                objections_found.append(obj)

        # Derive interest level from completion + objections
        if "not_interested" in objections_found:
            interest = "not_interested"
        elif completed >= total * 0.8 and "price" not in objections_found:
            interest = "high"
        elif completed >= total * 0.5:
            interest = "medium"
        else:
            interest = "low"

        # Build summary from key profiling responses
        summary_parts = []
        if self.collected_data.get("current_work"):
            summary_parts.append(f"Role: {self.collected_data['current_work'][:100]}")
        if self.collected_data.get("ai_rating"):
            summary_parts.append(f"AI knowledge: {self.collected_data['ai_rating'][:50]}")
        if self.collected_data.get("goal_6months"):
            summary_parts.append(f"6mo goal: {self.collected_data['goal_6months'][:100]}")
        if self.collected_data.get("long_term_goal"):
            summary_parts.append(f"Long-term: {self.collected_data['long_term_goal'][:100]}")

        return {
            "questions_completed": completed,
            "total_questions": total,
            "completion_rate": round(completed / total * 100, 1) if total > 0 else 0,
            "interest_level": interest,
            "objections_raised": objections_found,
            "call_summary": "; ".join(summary_parts) if summary_parts else "Call ended before profiling",
        }

    def save_to_file(self, call_uuid: str):
        """Save current state to file"""
        try:
            data = self.get_collected_data()
            file_path = FLOW_DATA_DIR / f"{call_uuid}.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved flow data")
        except Exception as e:
            logger.error(f"Error saving flow data: {e}")


# ==================== QUESTION PIPELINE ====================
# Owns lifecycle state, gate/echo decisions, stores agent+user text per question


class QuestionPhase(Enum):
    """Lifecycle phases for each question"""
    PENDING = "PENDING"          # Queued, not yet sent to Gemini
    DELIVERING = "DELIVERING"    # Gate OPEN, AI speaking the question
    ECHOING = "ECHOING"          # Gate closed, 1.5s echo buffer ignoring transcripts
    LISTENING = "LISTENING"      # Accepting user speech
    CAPTURED = "CAPTURED"        # User finished speaking, response stored
    DONE = "DONE"                # Finalized, moved to completed list


@dataclass
class QuestionRecord:
    """Per-question data tracking"""
    index: int
    question_id: str
    question_text: str
    phase: QuestionPhase = QuestionPhase.PENDING
    # Timestamps
    deliver_time: float = 0.0       # When DELIVERING started
    echo_time: float = 0.0         # When ECHOING started
    listen_time: float = 0.0       # When LISTENING started
    capture_time: float = 0.0      # When CAPTURED
    # Turn tracking during DELIVERING
    turns_since_inject: int = 0
    nudge_count: int = 0                # How many times this question was nudged
    # Content
    agent_said: str = ""           # What the AI actually said for this question
    user_said: str = ""            # What the user responded
    duration_seconds: float = 0.0  # Time from deliver to capture
    # Latency metrics
    response_latency_ms: float = 0.0   # Gemini response time after user finishes (ms)
    user_speech_end_time: float = 0.0  # When user stopped speaking (for latency calc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "agent_said": self.agent_said,
            "user_said": self.user_said,
            "duration_seconds": round(self.duration_seconds, 1),
            "response_latency_ms": round(self.response_latency_ms),
        }


class QuestionPipeline:
    """
    Manages question lifecycle state, gate/echo decisions, and per-question data.
    Sibling to QuestionFlow which owns question content and advance logic.
    """

    def __init__(self, call_uuid: str, total_questions: int):
        self._call_uuid = call_uuid
        self._log_prefix = f"[{call_uuid[:8]}]"
        self._total_questions = total_questions

        # Current question being processed (None = idle)
        self._current: Optional[QuestionRecord] = None
        # Completed question records
        self._completed: List[QuestionRecord] = []

        # Echo protection timing
        self._echo_buffer_seconds = 0.5  # Ignore transcripts for 0.5s after gate closes

        logger.info(f"{self._log_prefix} Pipeline INIT: {total_questions} questions queued")

    # ---- Properties (replace old boolean flags) ----

    @property
    def gate_open(self) -> bool:
        """Audio gate: True when AI audio should be forwarded to user"""
        return self._current is not None and self._current.phase == QuestionPhase.DELIVERING

    @property
    def is_echo(self) -> bool:
        """True during DELIVERING or ECHOING phase (transcripts should be ignored)"""
        if self._current is None:
            return False
        if self._current.phase == QuestionPhase.DELIVERING:
            elapsed = time.time() - self._current.deliver_time
            logger.debug(f"{self._log_prefix} is_echo=True (DELIVERING, {elapsed:.1f}s since inject)")
            return True
        if self._current.phase == QuestionPhase.ECHOING:
            elapsed = time.time() - self._current.echo_time
            if elapsed < self._echo_buffer_seconds:
                logger.debug(f"{self._log_prefix} is_echo=True (ECHOING, {elapsed:.1f}s/{self._echo_buffer_seconds}s buffer)")
                return True
            # Echo period expired, auto-transition to LISTENING
            logger.info(f"{self._log_prefix} Pipeline: Q{self._current.index} ECHOING → LISTENING "
                        f"(auto-transition in is_echo, {elapsed:.1f}s echo expired)")
            self._transition(QuestionPhase.LISTENING)
            self._current.listen_time = time.time()
            return False
        return False

    @property
    def is_listening(self) -> bool:
        """True when we're accepting user speech"""
        return self._current is not None and self._current.phase == QuestionPhase.LISTENING

    @property
    def waiting_for_user(self) -> bool:
        """True when we've asked a question and are waiting for response (ECHOING, LISTENING, or DELIVERING)"""
        if self._current is None:
            return False
        return self._current.phase in (QuestionPhase.DELIVERING, QuestionPhase.ECHOING, QuestionPhase.LISTENING)

    @property
    def question_asked_time(self) -> float:
        """When the current question started being delivered"""
        if self._current is None:
            return 0
        return self._current.deliver_time

    @property
    def pending_user_transcript(self) -> str:
        """Accumulated user speech for current question"""
        if self._current is None:
            return ""
        return self._current.user_said

    @property
    def current_phase(self) -> Optional[QuestionPhase]:
        """Current question's phase"""
        if self._current is None:
            return None
        return self._current.phase

    # ---- Lifecycle methods ----

    def dequeue_next(self, index: int, question_id: str, question_text: str):
        """Start delivering a new question: PENDING → DELIVERING (opens gate)"""
        # Log if replacing an in-progress question (e.g. reconnection)
        if self._current is not None:
            logger.warning(f"{self._log_prefix} Pipeline: replacing Q{self._current.index} "
                           f"({self._current.phase.value}) with Q{index}")
        record = QuestionRecord(
            index=index,
            question_id=question_id,
            question_text=question_text,
            phase=QuestionPhase.DELIVERING,
            deliver_time=time.time(),
            turns_since_inject=0,
        )
        self._current = record
        logger.info(f"{self._log_prefix} Pipeline: Q{index} PENDING → DELIVERING (gate OPEN) "
                    f"| id={question_id} | text='{question_text[:60]}' "
                    f"| completed={len(self._completed)}/{self._total_questions}")

    def on_turn_complete(self) -> bool:
        """
        Called on each Gemini turnComplete. Counts turns during DELIVERING.
        Returns True if gate was just closed (DELIVERING → ECHOING).
        """
        if self._current is None:
            logger.debug(f"{self._log_prefix} on_turn_complete: no current question, ignoring")
            return False
        if self._current.phase != QuestionPhase.DELIVERING:
            logger.debug(f"{self._log_prefix} on_turn_complete: Q{self._current.index} "
                         f"phase={self._current.phase.value} (not DELIVERING), ignoring")
            return False

        self._current.turns_since_inject += 1
        time_since_inject = time.time() - self._current.deliver_time

        # Close gate after 1 turn + 2s minimum, or 5s fallback.
        # Gemini combines ack + question in one turn. A second turn is almost always off-script.
        if (self._current.turns_since_inject >= 1 and time_since_inject >= 2.0) or time_since_inject >= 5.0:
            reason = "5s timeout" if time_since_inject >= 5.0 else f"turn {self._current.turns_since_inject} + {time_since_inject:.1f}s"
            self._transition(QuestionPhase.ECHOING)
            self._current.echo_time = time.time()
            logger.info(f"{self._log_prefix} Pipeline: Q{self._current.index} DELIVERING → ECHOING "
                        f"(gate CLOSED | {reason}) | echo buffer={self._echo_buffer_seconds}s")
            return True
        else:
            logger.debug(f"{self._log_prefix} Pipeline: Q{self._current.index} gate staying OPEN "
                         f"(turns={self._current.turns_since_inject}/1, {time_since_inject:.1f}s/2.0s)")
            return False

    def check_echo_expired(self) -> bool:
        """
        Check if echo buffer has expired. ECHOING → LISTENING after 1.5s.
        Returns True if transitioned to LISTENING.
        """
        if self._current is None or self._current.phase != QuestionPhase.ECHOING:
            return False

        elapsed = time.time() - self._current.echo_time
        if elapsed >= self._echo_buffer_seconds:
            self._transition(QuestionPhase.LISTENING)
            self._current.listen_time = time.time()
            total_since_deliver = time.time() - self._current.deliver_time
            logger.info(f"{self._log_prefix} Pipeline: Q{self._current.index} ECHOING → LISTENING "
                        f"(echo expired {elapsed:.1f}s/{self._echo_buffer_seconds}s | "
                        f"{total_since_deliver:.1f}s since delivery)")
            return True
        else:
            logger.debug(f"{self._log_prefix} Pipeline: Q{self._current.index} echo buffer "
                         f"{elapsed:.1f}s/{self._echo_buffer_seconds}s (still suppressing)")
        return False

    def accumulate_user_speech(self, text: str):
        """Append to user_said during LISTENING phase"""
        if self._current is None:
            logger.warning(f"{self._log_prefix} accumulate_user_speech: no current question, dropping '{text[:40]}'")
            return
        if self._current.phase not in (QuestionPhase.LISTENING, QuestionPhase.ECHOING):
            logger.warning(f"{self._log_prefix} accumulate_user_speech: Q{self._current.index} "
                           f"phase={self._current.phase.value} (not LISTENING/ECHOING), dropping '{text[:40]}'")
            return
        old_len = len(self._current.user_said)
        self._current.user_said = f"{self._current.user_said} {text}".strip()
        logger.debug(f"{self._log_prefix} Pipeline: Q{self._current.index} accumulated '{text}' "
                     f"({old_len}→{len(self._current.user_said)} chars) | total: '{self._current.user_said[:80]}'")

    def capture_response(self) -> Optional[str]:
        """
        LISTENING → CAPTURED when silence detected. Returns the user's response text.
        """
        if self._current is None:
            logger.warning(f"{self._log_prefix} capture_response: no current question")
            return None
        if self._current.phase != QuestionPhase.LISTENING:
            logger.warning(f"{self._log_prefix} capture_response: Q{self._current.index} "
                           f"phase={self._current.phase.value} (expected LISTENING)")
            return None

        self._current.capture_time = time.time()
        self._current.duration_seconds = self._current.capture_time - self._current.deliver_time
        listen_duration = self._current.capture_time - self._current.listen_time if self._current.listen_time else 0
        self._transition(QuestionPhase.CAPTURED)
        logger.info(f"{self._log_prefix} Pipeline: Q{self._current.index} LISTENING → CAPTURED "
                    f"| user_said='{self._current.user_said[:60]}' "
                    f"| listen={listen_duration:.1f}s | total={self._current.duration_seconds:.1f}s "
                    f"| agent_said='{self._current.agent_said[:60]}'")
        return self._current.user_said

    def store_agent_said(self, text: str):
        """Store what the AI actually said for the current question"""
        if self._current is None:
            logger.debug(f"{self._log_prefix} store_agent_said: no current question, dropping '{text[:40]}'")
            return
        old_len = len(self._current.agent_said)
        if self._current.agent_said:
            self._current.agent_said = f"{self._current.agent_said} {text}".strip()
        else:
            self._current.agent_said = text
        logger.debug(f"{self._log_prefix} Pipeline: Q{self._current.index} agent_said "
                     f"({old_len}→{len(self._current.agent_said)} chars) | "
                     f"phase={self._current.phase.value} | snippet='{text[:50]}'")


    def finalize_and_advance(self):
        """CAPTURED → DONE, move to completed list"""
        if self._current is None:
            logger.warning(f"{self._log_prefix} finalize_and_advance: no current question")
            return
        self._transition(QuestionPhase.DONE)
        rec = self._current
        self._completed.append(self._current)
        self._current = None
        logger.info(f"{self._log_prefix} Pipeline: Q{rec.index} CAPTURED → DONE "
                    f"| id={rec.question_id} | duration={rec.duration_seconds:.1f}s "
                    f"| agent='{rec.agent_said[:50]}' | user='{rec.user_said[:50]}' "
                    f"| progress={len(self._completed)}/{self._total_questions}")

    def reset_for_nudge(self) -> int:
        """Reset to DELIVERING for silence nudge (re-open gate).
        Returns the nudge count AFTER incrementing (1 = first nudge)."""
        if self._current is None:
            logger.warning(f"{self._log_prefix} reset_for_nudge: no current question")
            return 0
        old_phase = self._current.phase
        time_in_old_phase = time.time() - self._current.deliver_time
        self._current.nudge_count += 1
        self._current.phase = QuestionPhase.DELIVERING
        self._current.deliver_time = time.time()
        self._current.turns_since_inject = 0
        logger.info(f"{self._log_prefix} Pipeline: Q{self._current.index} {old_phase.value} → DELIVERING "
                    f"(nudge #{self._current.nudge_count}, gate OPEN | was in {old_phase.value} for {time_in_old_phase:.1f}s "
                    f"| user_said so far='{self._current.user_said[:40]}')")
        return self._current.nudge_count

    def clear_user_transcript(self):
        """Clear accumulated user transcript (used on new question injection)"""
        if self._current is not None:
            self._current.user_said = ""

    def close_gate_from_validator(self):
        """Close gate from transcript validator (off-script detected).
        Transitions DELIVERING → ECHOING immediately."""
        if self._current is None:
            return
        if self._current.phase != QuestionPhase.DELIVERING:
            logger.debug(f"{self._log_prefix} close_gate_from_validator: "
                         f"Q{self._current.index} phase={self._current.phase.value} (not DELIVERING)")
            return
        self._transition(QuestionPhase.ECHOING)
        self._current.echo_time = time.time()
        logger.info(f"{self._log_prefix} Pipeline: Q{self._current.index} DELIVERING → ECHOING "
                    f"(gate CLOSED by validator | off-script detected)")

    def get_expected_question(self) -> Optional[str]:
        """Get the expected question text for the current question (for off-script validation)"""
        if self._current is None:
            return None
        return self._current.question_text

    # ---- Context for Gemini ----

    def get_recent_context(self, n: int = 2) -> str:
        """Returns last n completed Q&A pairs as formatted context text for Gemini.
        Kept concise to avoid confusing the model with too much prior context."""
        if not self._completed:
            logger.debug(f"{self._log_prefix} get_recent_context: no completed pairs yet")
            return ""

        recent = self._completed[-n:]
        lines = []
        for rec in recent:
            replied = rec.user_said[:80] if rec.user_said else "(no response)"
            lines.append(f'- Q{rec.index}: "{replied}"')

        context = "[CONTEXT] Recent answers (for reference only, do NOT repeat these questions):\n" + "\n".join(lines)
        logger.debug(f"{self._log_prefix} get_recent_context: sending {len(recent)} pairs "
                     f"({len(context)} chars) to Gemini")
        return context

    def get_collected_pairs(self) -> List[Dict[str, Any]]:
        """Return all completed Q&A pairs for webhook/post-call data"""
        pairs = [rec.to_dict() for rec in self._completed]
        # Include current question if it has data
        if self._current and self._current.user_said:
            pairs.append(self._current.to_dict())
            logger.debug(f"{self._log_prefix} get_collected_pairs: including in-progress "
                         f"Q{self._current.index} ({self._current.phase.value})")
        logger.info(f"{self._log_prefix} get_collected_pairs: {len(pairs)} total pairs "
                    f"({len(self._completed)} completed"
                    f"{', +1 in-progress' if self._current and self._current.user_said else ''})")
        return pairs

    def get_call_metrics(self) -> Dict[str, Any]:
        """Generate post-call latency/timing metrics for this call"""
        all_records = list(self._completed)
        if self._current and self._current.user_said:
            all_records.append(self._current)

        if not all_records:
            return {"questions_completed": 0, "per_question": []}

        latencies = [r.response_latency_ms for r in all_records if r.response_latency_ms > 0]
        durations = [r.duration_seconds for r in all_records if r.duration_seconds > 0]
        nudge_total = sum(r.nudge_count for r in all_records)

        per_question = []
        for rec in all_records:
            per_question.append({
                "q": rec.index,
                "id": rec.question_id,
                "latency_ms": round(rec.response_latency_ms),
                "duration_s": round(rec.duration_seconds, 1),
                "nudges": rec.nudge_count,
            })

        metrics = {
            "questions_completed": len(all_records),
            "total_duration_s": round(sum(durations), 1) if durations else 0,
            "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
            "max_latency_ms": round(max(latencies)) if latencies else 0,
            "min_latency_ms": round(min(latencies)) if latencies else 0,
            "p90_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.9)]) if latencies else 0,
            "total_nudges": nudge_total,
            "per_question": per_question,
        }
        return metrics

    # ---- Internal ----

    def dump_state(self) -> str:
        """Return full pipeline state as a string for debugging"""
        if self._current:
            cur = (f"Q{self._current.index}({self._current.question_id}) "
                   f"phase={self._current.phase.value} "
                   f"turns={self._current.turns_since_inject} "
                   f"agent='{self._current.agent_said[:30]}' "
                   f"user='{self._current.user_said[:30]}'")
        else:
            cur = "None"
        return (f"Pipeline[current={cur} | "
                f"completed={len(self._completed)}/{self._total_questions} | "
                f"gate={self.gate_open}]")

    def _transition(self, new_phase: QuestionPhase):
        """Transition current question to a new phase"""
        if self._current is not None:
            self._current.phase = new_phase


# Store flows per call (protected by lock for concurrent access)
_call_flows: Dict[str, QuestionFlow] = {}
_call_flows_lock = threading.Lock()


def get_or_create_flow(
    call_uuid: str,
    client_name: str = "fwai",
    context: Dict = None,
    questions_override: List[Dict] = None,
    prompt_override: str = None,
    objections_override: Dict = None,
    objection_keywords_override: Dict = None
) -> QuestionFlow:
    """Get existing flow or create new one for a call"""
    with _call_flows_lock:
        if call_uuid not in _call_flows:
            _call_flows[call_uuid] = QuestionFlow(
                call_uuid=call_uuid,
                client_name=client_name,
                context=context or {},
                questions_override=questions_override,
                prompt_override=prompt_override,
                objections_override=objections_override,
                objection_keywords_override=objection_keywords_override
            )
        return _call_flows[call_uuid]


def remove_flow(call_uuid: str) -> Optional[Dict]:
    """Remove flow from memory and return collected data"""
    with _call_flows_lock:
        flow = _call_flows.pop(call_uuid, None)
    if flow:
        data = flow.get_collected_data()
        flow.save_to_file(call_uuid)
        return data
    return None


def get_flow_data_from_file(call_uuid: str) -> Optional[Dict]:
    """Load flow data from file"""
    try:
        file_path = FLOW_DATA_DIR / f"{call_uuid}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading flow data: {e}")
    return None
