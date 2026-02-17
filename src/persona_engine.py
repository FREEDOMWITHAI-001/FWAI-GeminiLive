"""
Dynamic Persona Engine — modular prompt system for AI calling.

Replaces the monolithic prompt with editable modules that load based on
who's on the call (persona) and what's happening (situation).

Modules are stored as plain text files in prompts/ and are editable via API.
Detection uses keyword/phrase scoring — zero API calls, zero latency.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional
from loguru import logger
from src.conversational_prompts import render_prompt

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
PERSONAS_DIR = PROMPTS_DIR / "personas"
SITUATIONS_DIR = PROMPTS_DIR / "situations"
PERSONA_KEYWORDS_FILE = PROMPTS_DIR / "persona_keywords.json"
SITUATION_KEYWORDS_FILE = PROMPTS_DIR / "situation_keywords.json"

# Simple file cache: {path: (mtime, content)}
_file_cache: dict[str, tuple[float, str]] = {}


# =============================================================================
# Module Loading (with mtime-based cache for hot-reload)
# =============================================================================

def load_module(path: Path) -> str:
    """Load a text module file. Cached with mtime check for hot-reload."""
    path_str = str(path)
    try:
        mtime = path.stat().st_mtime
        if path_str in _file_cache and _file_cache[path_str][0] == mtime:
            return _file_cache[path_str][1]
        content = path.read_text(encoding="utf-8").strip()
        _file_cache[path_str] = (mtime, content)
        return content
    except FileNotFoundError:
        return ""


def load_all_personas() -> dict[str, str]:
    """Scan prompts/personas/ and return {name: content}."""
    result = {}
    if PERSONAS_DIR.exists():
        for f in PERSONAS_DIR.glob("*.txt"):
            result[f.stem] = load_module(f)
    return result


def load_all_situations() -> dict[str, str]:
    """Scan prompts/situations/ and return {name: content}."""
    result = {}
    if SITUATIONS_DIR.exists():
        for f in SITUATIONS_DIR.glob("*.txt"):
            result[f.stem] = load_module(f)
    return result


def _load_json_config(path: Path) -> dict:
    """Load a JSON config file with caching."""
    path_str = str(path)
    try:
        mtime = path.stat().st_mtime
        if path_str in _file_cache and _file_cache[path_str][0] == mtime:
            return json.loads(_file_cache[path_str][1])
        content = path.read_text(encoding="utf-8")
        _file_cache[path_str] = (mtime, content)
        return json.loads(content)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# =============================================================================
# Persona Detection (keyword + phrase scoring)
# =============================================================================

# Check order: most specific first
_PERSONA_CHECK_ORDER = ["student", "business_owner", "manager", "freelancer", "working_professional"]


def _normalize_transcription(text: str) -> str:
    """Normalize Gemini audio transcription which fragments words with spaces.
    E.g. 'stu den t' → 'student', 'B tech' → 'btech', 'In fo sy s' → 'infosys'

    Strategy: return original text PLUS a spaceless version of each sentence,
    so keyword matching works even when transcription breaks words apart.
    'I am a stu den t' → contains 'student' in the spaceless version.
    """
    # Split on sentence boundaries (periods, commas, etc.) to avoid merging across sentences
    import re
    parts = re.split(r'[.,!?;]+', text)
    spaceless_parts = []
    for part in parts:
        stripped = part.strip()
        if stripped:
            # Remove all spaces within this sentence fragment
            spaceless_parts.append(stripped.replace(" ", ""))
    # Return original + spaceless versions joined
    return text + " " + " ".join(spaceless_parts)


def detect_persona(accumulated_text: str) -> Optional[str]:
    """
    Detect persona from accumulated user text using keyword/phrase scoring.
    Returns persona key or None. Threshold: score >= 2.
    Handles Gemini's fragmented audio transcription (e.g. 'stu den t' for 'student').
    """
    if not accumulated_text:
        return None

    config = _load_json_config(PERSONA_KEYWORDS_FILE)
    if not config:
        return None

    text_lower = _normalize_transcription(accumulated_text).lower()
    best_persona = None
    best_score = 0

    for persona_key in _PERSONA_CHECK_ORDER:
        if persona_key not in config:
            continue
        persona_config = config[persona_key]
        score = 0

        for keyword in persona_config.get("keywords", []):
            if keyword in text_lower:
                score += 1

        for phrase in persona_config.get("phrases", []):
            if phrase in text_lower:
                score += 3

        if score > best_score:
            best_score = score
            best_persona = persona_key

    if best_score >= 2:
        return best_persona
    return None


# =============================================================================
# Situation Detection (keyword matching on latest turn)
# =============================================================================

def detect_situations(user_text: str) -> list[str]:
    """
    Detect active situations from the latest user text.
    Returns list of situation keys (max 2).
    """
    if not user_text:
        return []

    config = _load_json_config(SITUATION_KEYWORDS_FILE)
    if not config:
        return []

    text_lower = _normalize_transcription(user_text).lower()
    active = []

    for situation_key, situation_config in config.items():
        for keyword in situation_config.get("keywords", []):
            if keyword in text_lower:
                active.append(situation_key)
                break  # One match is enough per situation

    return active[:2]  # Max 2 active situations


def get_situation_hint(situation_key: str) -> str:
    """Get the short hint for immediate injection via client_content."""
    config = _load_json_config(SITUATION_KEYWORDS_FILE)
    situation = config.get(situation_key, {})
    return situation.get("hint", "")


# =============================================================================
# Prompt Composition
# =============================================================================

def compose_prompt(
    context: dict,
    persona_key: Optional[str],
    active_situations: list[str],
    is_early_call: bool,
) -> str:
    """
    Compose modular prompt from file-based components.
    Returns fully rendered string with placeholders replaced.

    Order matters — memory context goes BEFORE persona module so the AI
    reads "skip discovery" before it reads "PAIN POINTS to probe".
    """
    parts = []

    # Layer 1: Base module (always)
    base = load_module(PROMPTS_DIR / "base.txt")
    if base:
        parts.append(base)

    # Layer 2: Cross-call memory (BEFORE persona so it takes priority)
    memory_ctx = context.get("_memory_context", "")
    if memory_ctx:
        parts.append(memory_ctx)

    # Layer 3: NEPQ (early turns / no persona) or Persona module
    if is_early_call or not persona_key:
        nepq = load_module(PROMPTS_DIR / "nepq.txt")
        if nepq:
            parts.append(nepq)
    else:
        persona_content = load_module(PERSONAS_DIR / f"{persona_key}.txt")
        if persona_content:
            # For repeat callers, strip the FLOW line to avoid conflict with memory instructions
            if memory_ctx and "FLOW:" in persona_content:
                persona_content = "\n".join(
                    line for line in persona_content.split("\n")
                    if not line.strip().startswith("FLOW:")
                )
            parts.append(persona_content)
        else:
            nepq = load_module(PROMPTS_DIR / "nepq.txt")
            if nepq:
                parts.append(nepq)

    # Layer 4: Active situation modules (max 2)
    for situation_key in active_situations[:2]:
        situation_content = load_module(SITUATIONS_DIR / f"{situation_key}.txt")
        if situation_content:
            parts.append(situation_content)

    combined = "\n\n".join(parts)
    return render_prompt(combined, context)
