"""
Micro-Moment Detection & Strategy Switching — behavioral analysis layer.

Tracks turn-by-turn engagement metrics (response time, word count trends,
question patterns, silence) and detects 5 micro-moments that trigger
strategy changes mid-call.

Complements the keyword-based situation detection in persona_engine.py:
- Situations catch EXPLICIT signals ("too expensive")
- Micro-moments catch IMPLICIT behavioral shifts (engagement dropping)

All computation is local — zero API calls, zero latency impact.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict
from loguru import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TurnMetrics:
    """Metrics collected for a single completed turn."""
    turn_number: int
    timestamp: float

    # Timing
    response_time_ms: float       # Agent stop → user first word
    turn_duration_ms: float       # Total turn duration

    # User signals
    user_word_count: int
    user_question_count: int      # Sentences ending in "?"
    user_how_question_count: int  # Process-oriented questions
    user_has_filler_only: bool    # Only "hmm", "okay", "yeah" etc.
    user_monosyllabic: bool       # <= 4 words total

    # Agent context
    agent_word_count: int
    agent_mentioned_price: bool


@dataclass
class MicroMomentState:
    """Serializable state that survives session splits."""
    turn_metrics_history: List[Dict] = field(default_factory=list)
    current_strategy: str = "discovery"
    strategy_lock_until_turn: int = 0
    last_moment_detected: str = ""
    last_moment_turn: int = 0
    price_mentioned_turn: int = 0
    consecutive_short_answers: int = 0
    consecutive_filler_answers: int = 0
    moments_log: List[Dict] = field(default_factory=list)


# Moment → strategy mapping
_MOMENT_TO_STRATEGY = {
    "buying_signal": "closing",
    "resistance": "rapport",
    "price_shock": "value_reframe",
    "interest_spike": "momentum",
    "last_chance": "last_chance",
}

# Default hints (overridable via prompts/micro_moments.json)
_DEFAULT_HINTS = {
    "buying_signal": (
        "[MICRO-MOMENT: Buying signal detected. Customer is shifting from WHY to HOW. "
        "Switch to CLOSING mode: answer their process questions directly, assume the sale, "
        "offer to start enrollment. Don't re-explain value — they already see it. "
        "Be direct: 'Great question! Here's how we get you started...']"
    ),
    "resistance": (
        "[MICRO-MOMENT: Resistance building. Customer engagement is dropping. "
        "Switch to RAPPORT mode: stop selling. Ask an open-ended personal question. "
        "Show genuine curiosity about THEIR situation. "
        "Rebuild trust before returning to value.]"
    ),
    "price_shock": (
        "[MICRO-MOMENT: Price shock detected. Customer went quiet/hesitant after hearing the price. "
        "DO NOT repeat the price. Immediately pivot to ROI and offer EMI. "
        "Keep voice calm and empathetic, not defensive.]"
    ),
    "interest_spike": (
        "[MICRO-MOMENT: Interest spike! Customer is suddenly more engaged — asking more, "
        "talking faster, longer responses. RIDE THIS MOMENTUM. Match their energy. "
        "Feed them exciting details. Don't slow down with disclaimers.]"
    ),
    "last_chance": (
        "[MICRO-MOMENT: Customer winding down — 'I'll think about it' energy detected. "
        "This is your LAST CHANCE. Deploy value bomb: share ONE powerful fact they haven't heard. "
        "Create gentle FOMO without being pushy. If they still want to go, respect it gracefully.]"
    ),
}

# Filler words for detecting low-engagement responses
_FILLER_WORDS = frozenset({
    "hmm", "okay", "ok", "yeah", "yes", "no", "sure", "right",
    "alright", "hm", "uh", "ah", "um", "yep", "nope",
    "maybe", "fine", "cool", "haan", "ha", "ji", "accha",
})

# Process-oriented question patterns (buying signal indicator)
_HOW_PATTERNS = [
    re.compile(p) for p in [
        r"\bhow\b", r"\bwhat steps\b", r"\bwhen can i\b",
        r"\bhow do i\b", r"\bhow does\b", r"\bwhat happens next\b",
        r"\bhow to\b", r"\bwhat is the process\b", r"\bnext step\b",
        r"\bsign up\b", r"\benroll\b", r"\bregister\b",
        r"\bwhat do i need\b", r"\bwhere do i\b", r"\bstart date\b",
    ]
]

# Agent price mention patterns
_PRICE_PATTERNS = [
    re.compile(p) for p in [
        r"40[\s,]*000", r"forty thousand", r"\u20b9", r"rupees",
        r"\bprice\b", r"\bcost\b", r"\binvestment\b", r"\bfee\b",
        r"\bemi\b", r"3[\s,]*300", r"\bpayment\b",
    ]
]

# Last-chance text patterns
_LAST_CHANCE_PATTERNS = [
    re.compile(p) for p in [
        r"i'll think", r"let me think", r"think about it",
        r"i'll get back", r"get back to you", r"maybe later",
        r"not right now", r"not sure yet", r"need to think",
        r"talk to my", r"discuss with", r"need time",
        r"i'll decide", r"send me details", r"send info",
        r"will let you know", r"call me later", r"i'll call back",
    ]
]


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "prompts" / "micro_moments.json"
_config_cache: tuple = (0.0, {})  # (mtime, data)


def _load_config() -> dict:
    """Load config with mtime-based cache for hot-reload."""
    global _config_cache
    try:
        mtime = _CONFIG_PATH.stat().st_mtime
        if mtime == _config_cache[0]:
            return _config_cache[1]
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        _config_cache = (mtime, data)
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# MicroMomentDetector
# ---------------------------------------------------------------------------

class MicroMomentDetector:
    """
    Behavioral micro-moment detection across a sliding window of turns.

    Detects 5 moments:
    1. buying_signal  — user shifting from "why" to "how" questions
    2. resistance     — engagement declining over multiple turns
    3. price_shock    — hesitation right after agent mentions price
    4. interest_spike — sudden jump in engagement metrics
    5. last_chance    — "I'll think about it" / winding down signals
    """

    def __init__(self, state: Optional[MicroMomentState] = None):
        self._state = state or MicroMomentState()
        self._cfg = _load_config()

    # --- Config helpers (with defaults) ---

    @property
    def _min_turns(self) -> int:
        return self._cfg.get("min_turns_for_detection", 3)

    @property
    def _cooldown(self) -> int:
        return self._cfg.get("strategy_cooldown_turns", 2)

    @property
    def _max_history(self) -> int:
        return self._cfg.get("max_history", 8)

    def _threshold(self, moment: str, key: str, default: float) -> float:
        return self._cfg.get("thresholds", {}).get(moment, {}).get(key, default)

    # =====================================================================
    # Public API
    # =====================================================================

    def record_turn(
        self,
        turn_number: int,
        full_user: str,
        full_agent: str,
        response_time_ms: float,
        turn_duration_ms: float,
    ) -> Optional[str]:
        """
        Record a completed turn and detect micro-moments.

        Returns a hint string to inject into Gemini, or None.
        """
        # Reload config (mtime-cached, ~0.01ms)
        self._cfg = _load_config()

        metrics = self._compute_metrics(
            turn_number, full_user, full_agent,
            response_time_ms, turn_duration_ms,
        )
        self._append_metrics(metrics)
        self._update_streaks(metrics)

        # Check for text-level last_chance first (overrides behavioral)
        if full_user:
            user_lower = full_user.lower()
            if any(p.search(user_lower) for p in _LAST_CHANCE_PATTERNS):
                hint = self._generate_hint("last_chance", metrics)
                if hint:
                    return hint

        # Need baseline before behavioral detection
        if turn_number < self._min_turns:
            return None

        # Respect cooldown
        if turn_number < self._state.strategy_lock_until_turn:
            return None

        moment = self._detect_moment(metrics)
        if moment:
            return self._generate_hint(moment, metrics)
        return None

    @property
    def current_strategy(self) -> str:
        return self._state.current_strategy

    def get_state(self) -> dict:
        """Serialize for session splits."""
        return asdict(self._state)

    def get_moments_log(self) -> List[Dict]:
        """Return list of moments detected during the call (for webhook)."""
        return list(self._state.moments_log)

    @classmethod
    def from_state(cls, state_dict: dict) -> MicroMomentDetector:
        """Restore from serialized state."""
        state = MicroMomentState(**state_dict)
        return cls(state=state)

    # =====================================================================
    # Metrics computation (pure local, ~0.1ms)
    # =====================================================================

    def _compute_metrics(
        self, turn_number, full_user, full_agent,
        response_time_ms, turn_duration_ms,
    ) -> TurnMetrics:
        user_words = full_user.split() if full_user else []
        user_word_count = len(user_words)

        # Question counting
        user_question_count = full_user.count("?") if full_user else 0
        user_lower = (full_user or "").lower()
        how_count = sum(1 for p in _HOW_PATTERNS if p.search(user_lower))

        # Filler-only detection
        is_filler = (
            user_word_count <= 3
            and all(w.lower().strip(".,!?") in _FILLER_WORDS for w in user_words)
        ) if user_words else True

        # Monosyllabic: very short response
        is_monosyllabic = 0 < user_word_count <= 4

        # Price mention in agent text
        agent_lower = (full_agent or "").lower()
        mentioned_price = any(p.search(agent_lower) for p in _PRICE_PATTERNS)

        return TurnMetrics(
            turn_number=turn_number,
            timestamp=time.time(),
            response_time_ms=response_time_ms,
            turn_duration_ms=turn_duration_ms,
            user_word_count=user_word_count,
            user_question_count=user_question_count,
            user_how_question_count=how_count,
            user_has_filler_only=is_filler,
            user_monosyllabic=is_monosyllabic,
            agent_word_count=len(full_agent.split()) if full_agent else 0,
            agent_mentioned_price=mentioned_price,
        )

    def _append_metrics(self, metrics: TurnMetrics):
        self._state.turn_metrics_history.append(asdict(metrics))
        max_h = self._max_history
        if len(self._state.turn_metrics_history) > max_h:
            self._state.turn_metrics_history = self._state.turn_metrics_history[-max_h:]

    def _update_streaks(self, metrics: TurnMetrics):
        # Short answers count toward resistance ONLY if user isn't asking questions
        # "What topics are covered?" (4 words) is engaged, not resistant
        is_short = metrics.user_monosyllabic or metrics.user_word_count <= 4
        is_asking = metrics.user_question_count > 0
        if is_short and not is_asking:
            self._state.consecutive_short_answers += 1
        else:
            self._state.consecutive_short_answers = 0

        if metrics.user_has_filler_only:
            self._state.consecutive_filler_answers += 1
        else:
            self._state.consecutive_filler_answers = 0

        if metrics.agent_mentioned_price:
            self._state.price_mentioned_turn = metrics.turn_number

    # =====================================================================
    # Moment detection (behavioral, sliding window)
    # =====================================================================

    def _detect_moment(self, current: TurnMetrics) -> Optional[str]:
        history = [TurnMetrics(**d) for d in self._state.turn_metrics_history]
        if len(history) < 2:
            return None

        prev = history[-2]
        window = history[-3:] if len(history) >= 3 else history

        # Priority: price_shock > last_chance > buying_signal > resistance > interest_spike
        if self._detect_price_shock(current, window):
            return "price_shock"
        if self._detect_last_chance_behavioral(current, window):
            return "last_chance"
        if self._detect_buying_signal(current, prev, window):
            return "buying_signal"
        if self._detect_resistance(current, prev, window):
            return "resistance"
        if self._detect_interest_spike(current, prev, window):
            return "interest_spike"

        return None

    def _detect_price_shock(self, current: TurnMetrics, window: list) -> bool:
        turns_since_price = current.turn_number - self._state.price_mentioned_turn
        max_turns = int(self._threshold("price_shock", "turns_after_price_max", 2))
        # Allow same-turn detection (agent says price, user responds with "Hmm" in same exchange)
        if turns_since_price < 0 or turns_since_price > max_turns:
            return False

        avg_rt = self._avg(window, "response_time_ms")
        rt_mult = self._threshold("price_shock", "response_time_multiplier", 1.4)
        max_wc = int(self._threshold("price_shock", "max_word_count", 6))

        is_slow = avg_rt > 0 and current.response_time_ms > avg_rt * rt_mult
        is_short = current.user_word_count < max_wc
        is_filler = current.user_has_filler_only

        return (is_slow and is_short) or is_filler

    def _detect_last_chance_behavioral(self, current: TurnMetrics, window: list) -> bool:
        min_short = int(self._threshold("last_chance", "min_consecutive_short", 2))
        decline = self._threshold("last_chance", "word_count_decline_factor", 0.5)

        if self._state.consecutive_short_answers >= min_short:
            avg_wc = self._avg(window, "user_word_count")
            if avg_wc > 0 and current.user_word_count < avg_wc * decline:
                return True
        return False

    def _detect_buying_signal(self, current: TurnMetrics, prev: TurnMetrics, window: list) -> bool:
        min_how = int(self._threshold("buying_signal", "min_how_questions", 1))
        wc_factor = self._threshold("buying_signal", "word_count_maintain_factor", 0.8)

        # User asking process-oriented questions with stable engagement
        if current.user_how_question_count >= min_how:
            avg_wc = self._avg(window, "user_word_count")
            if avg_wc == 0 or current.user_word_count >= avg_wc * wc_factor:
                return True

        # Question count increasing AND word count increasing
        if (current.user_question_count > prev.user_question_count
                and current.user_word_count > prev.user_word_count):
            return True

        return False

    def _detect_resistance(self, current: TurnMetrics, prev: TurnMetrics, window: list) -> bool:
        if len(window) < 3:
            return False

        min_short = int(self._threshold("resistance", "min_consecutive_short", 2))
        rt_factor = self._threshold("resistance", "response_time_increase_factor", 1.3)
        wc_factor = self._threshold("resistance", "word_count_decrease_factor", 0.7)

        # Pattern 1: declining word count over 3 turns + consecutive short answers
        word_counts = [m.user_word_count for m in window]
        declining = all(word_counts[i] >= word_counts[i + 1] for i in range(len(word_counts) - 1))
        if declining and self._state.consecutive_short_answers >= min_short:
            return True

        # Pattern 2: response time increasing + word count decreasing
        if prev.response_time_ms > 0:
            rt_increasing = current.response_time_ms > prev.response_time_ms * rt_factor
            wc_decreasing = prev.user_word_count > 0 and current.user_word_count < prev.user_word_count * wc_factor
            if rt_increasing and wc_decreasing and self._state.consecutive_short_answers >= 1:
                return True

        return False

    def _detect_interest_spike(self, current: TurnMetrics, prev: TurnMetrics, window: list) -> bool:
        wc_factor = self._threshold("interest_spike", "word_count_spike_factor", 1.4)
        rt_factor = self._threshold("interest_spike", "response_time_faster_factor", 0.8)
        min_signals = int(self._threshold("interest_spike", "min_signals", 2))

        avg_wc = self._avg(window, "user_word_count")
        avg_rt = self._avg(window, "response_time_ms")

        wc_spike = avg_wc > 0 and current.user_word_count > avg_wc * wc_factor
        rt_faster = avg_rt > 0 and current.response_time_ms < avg_rt * rt_factor
        more_questions = (current.user_question_count > 0
                          and current.user_question_count >= prev.user_question_count)

        signals = sum([wc_spike, rt_faster, more_questions])
        return signals >= min_signals

    # =====================================================================
    # Hint generation
    # =====================================================================

    def _generate_hint(self, moment: str, metrics: TurnMetrics) -> Optional[str]:
        # Don't re-trigger same moment within cooldown
        if (moment == self._state.last_moment_detected
                and metrics.turn_number - self._state.last_moment_turn < self._cooldown):
            return None

        hints = self._cfg.get("hints", _DEFAULT_HINTS)
        hint = hints.get(moment)
        if not hint:
            hint = _DEFAULT_HINTS.get(moment)
        if not hint:
            return None

        # Update strategy state
        new_strategy = _MOMENT_TO_STRATEGY.get(moment, self._state.current_strategy)
        self._state.current_strategy = new_strategy
        self._state.last_moment_detected = moment
        self._state.last_moment_turn = metrics.turn_number
        self._state.strategy_lock_until_turn = metrics.turn_number + self._cooldown

        # Log for post-call analytics
        self._state.moments_log.append({
            "moment": moment,
            "turn": metrics.turn_number,
            "strategy": new_strategy,
            "user_word_count": metrics.user_word_count,
            "response_time_ms": round(metrics.response_time_ms, 0),
        })

        logger.info(
            f"Micro-moment: {moment} at turn #{metrics.turn_number} "
            f"-> strategy: {new_strategy} "
            f"(wc={metrics.user_word_count}, rt={metrics.response_time_ms:.0f}ms)"
        )

        return hint

    # =====================================================================
    # Helpers
    # =====================================================================

    @staticmethod
    def _avg(window: list, field: str) -> float:
        values = [getattr(m, field, 0) for m in window]
        return sum(values) / max(len(values), 1)
