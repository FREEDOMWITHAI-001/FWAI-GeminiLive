"""
Linguistic Mirror — detects how the prospect speaks and adapts AI style.

Deep analysis of user speech patterns:
- Formality level (casual / neutral / formal)
- Language mixing (hinglish / english) + specific Hindi words used
- Vocabulary level (simple / technical)
- Verbosity (terse / balanced / verbose) — match their response length
- Generation markers (younger casual / older professional)
- Engagement level (low / medium / high) — adapt energy accordingly

Gemini natively hears audio pace/tone, so we INSTRUCT it to mirror those.
Text-based detection handles everything else from transcription.

Zero latency: pure keyword/pattern scoring, no API calls.
"""

from __future__ import annotations

import re
from typing import Optional
from loguru import logger


# =============================================================================
# Detection Signals
# =============================================================================

# Hindi/Hinglish markers — common Hindi words that appear in Gemini transcriptions
_HINDI_MARKERS = [
    # Common words
    "kya", "hai", "nahi", "haan", "accha", "theek", "aur", "bhi",
    "mein", "toh", "kaise", "kab", "kahan", "kyun", "lekin", "par",
    "abhi", "bahut", "thoda", "sab", "kuch", "woh", "yeh",
    # Conversational
    "arrey", "arre", "arey", "achha", "suno", "dekho", "chalo",
    "bas", "bilkul", "sahi", "matlab", "samajh", "pata",
    # Reactions
    "haanji", "hanji", "nahin", "ji", "haa",
    # Common phrases (spaceless for fragmented transcription)
    "kyahai", "theekhai", "achahai", "nahihai", "haanbhai",
]

# Casual/informal markers
_CASUAL_MARKERS = [
    # Hindi-English casual
    "yaar", "yar", "bhai", "bro", "dude", "man",
    "chill", "cool", "awesome", "crazy", "damn",
    # Informal phrases
    "no worries", "all good", "got it", "yeah yeah",
    "kinda", "sorta", "gonna", "wanna", "gotta",
    # Informal reactions
    "haha", "lol", "wow", "oh god", "oh man",
    # Short informal acknowledgments
    "ya", "yep", "nope", "nah",
]

# Younger generation markers (Gen Z / millennial speech patterns)
_YOUNG_MARKERS = [
    "like", "literally", "basically", "honestly", "actually",
    "super", "legit", "vibe", "vibes", "lowkey", "highkey",
    "no cap", "fr", "tbh", "ngl", "bruh", "dude",
    "insane", "sick", "fire", "lit", "cringe",
    "i mean", "you know what i mean", "right right",
]

# Formal markers
_FORMAL_MARKERS = [
    # Respectful address
    "sir", "madam", "ma'am", "maam",
    # Formal language
    "please", "kindly", "would you", "could you",
    "i would like", "i appreciate", "thank you for",
    "certainly", "absolutely", "indeed", "precisely",
    "regarding", "concerning", "with respect to",
    # Professional phrasing
    "i understand", "that makes sense", "i see your point",
    "if i may", "allow me", "permit me",
    "in my experience", "from my perspective",
]

# Technical vocabulary markers
_TECHNICAL_MARKERS = [
    # Tech terms
    "api", "backend", "frontend", "database", "server", "cloud",
    "deployment", "architecture", "framework", "algorithm",
    "machine learning", "deep learning", "neural network",
    "devops", "kubernetes", "docker", "microservice",
    "python", "javascript", "java", "react", "node",
    "data pipeline", "etl", "sql", "nosql",
    # Business/professional terms
    "roi", "kpi", "stakeholder", "deliverable", "milestone",
    "strategic", "implementation", "infrastructure",
    "revenue", "pipeline", "scaling", "optimization",
    "compliance", "governance", "automation",
    # AI-specific
    "prompt engineering", "fine tuning", "llm", "gpt",
    "transformer", "embedding", "vector", "rag",
    "langchain", "hugging face", "tokenizer",
]


# =============================================================================
# Transcription Normalization (same pattern as persona_engine)
# =============================================================================

def _normalize_transcription(text: str) -> str:
    """Normalize Gemini's fragmented audio transcription.
    Appends spaceless versions per sentence so keyword matching works
    even when transcription breaks words apart (e.g. 'ac cha' → 'accha')."""
    parts = re.split(r'[.,!?;]+', text)
    spaceless_parts = []
    for part in parts:
        stripped = part.strip()
        if stripped:
            spaceless_parts.append(stripped.replace(" ", ""))
    return text + " " + " ".join(spaceless_parts)


def _find_used_markers(text_lower: str, marker_list: list) -> list:
    """Return which markers from the list actually appear in text."""
    return [m for m in marker_list if m in text_lower]


# =============================================================================
# Style Detection
# =============================================================================

def detect_linguistic_style(accumulated_text: str) -> dict:
    """
    Deep linguistic style detection from accumulated user text.
    Returns rich style profile or empty dict if not enough text.

    Minimum text threshold: 30 chars (need enough speech to detect patterns).
    """
    if not accumulated_text or len(accumulated_text.strip()) < 30:
        return {}

    text_lower = _normalize_transcription(accumulated_text).lower()
    raw_text = accumulated_text.strip()

    # --- Language detection (Hindi/Hinglish vs English) ---
    hindi_found = _find_used_markers(text_lower, _HINDI_MARKERS)
    hindi_score = len(hindi_found)
    language = "hinglish" if hindi_score >= 3 else "english"

    # --- Formality detection ---
    casual_found = _find_used_markers(text_lower, _CASUAL_MARKERS)
    formal_found = _find_used_markers(text_lower, _FORMAL_MARKERS)
    young_found = _find_used_markers(text_lower, _YOUNG_MARKERS)
    casual_score = len(casual_found)
    formal_score = len(formal_found)

    # Hindi mixing and young markers add to casual
    if hindi_score >= 3:
        casual_score += 2
    if len(young_found) >= 2:
        casual_score += 2

    if formal_score >= 3 and formal_score > casual_score:
        formality = "formal"
    elif casual_score >= 3 and casual_score > formal_score:
        formality = "casual"
    else:
        formality = "neutral"

    # --- Vocabulary detection ---
    tech_found = _find_used_markers(text_lower, _TECHNICAL_MARKERS)
    tech_score = len(tech_found)
    vocabulary = "technical" if tech_score >= 2 else "simple"

    # --- Verbosity detection (avg words per user turn) ---
    # Split on common turn boundaries (punctuation + pauses in transcription)
    sentences = [s.strip() for s in re.split(r'[.!?]+', raw_text) if s.strip()]
    words = raw_text.split()
    total_words = len(words)
    num_sentences = max(len(sentences), 1)
    avg_words_per_sentence = total_words / num_sentences

    if avg_words_per_sentence <= 5:
        verbosity = "terse"  # Short, clipped answers
    elif avg_words_per_sentence >= 15:
        verbosity = "verbose"  # Detailed, expansive
    else:
        verbosity = "balanced"

    # --- Engagement detection ---
    # Short responses + few words = low engagement
    # Questions back + detailed answers = high engagement
    question_marks = raw_text.count("?")
    exclamations = raw_text.count("!")
    has_questions_back = question_marks >= 2
    has_enthusiasm = exclamations >= 1 or any(m in text_lower for m in ["great", "amazing", "wonderful", "love", "excited"])

    if total_words < 20 and avg_words_per_sentence <= 4:
        engagement = "low"
    elif has_questions_back or has_enthusiasm or total_words > 80:
        engagement = "high"
    else:
        engagement = "medium"

    style = {
        "formality": formality,
        "language": language,
        "vocabulary": vocabulary,
        "verbosity": verbosity,
        "engagement": engagement,
        # Specific markers found — used for targeted mirroring
        "hindi_words_used": hindi_found[:6],  # Cap at 6 for prompt size
        "casual_markers": casual_found[:4],
        "young_markers": young_found[:4],
        "tech_terms": tech_found[:4],
    }

    return style


def style_changed(old_style: dict, new_style: dict) -> bool:
    """Check if linguistic style changed enough to warrant an update."""
    if not old_style:
        return bool(new_style)
    if not new_style:
        return False
    # Check core dimensions
    core_keys = ("formality", "language", "vocabulary", "verbosity", "engagement")
    return any(old_style.get(k) != new_style.get(k) for k in core_keys)


# =============================================================================
# Prompt Instruction Composition
# =============================================================================

def compose_mirror_instruction(style: dict) -> str:
    """
    Build a specific, actionable prompt snippet instructing Gemini to mirror
    the caller's exact speaking style. Returns empty string if no style detected.
    """
    if not style:
        return ""

    formality = style.get("formality", "neutral")
    language = style.get("language", "english")
    vocabulary = style.get("vocabulary", "simple")
    verbosity = style.get("verbosity", "balanced")
    engagement = style.get("engagement", "medium")
    hindi_words = style.get("hindi_words_used", [])
    casual_markers = style.get("casual_markers", [])
    young_markers = style.get("young_markers", [])
    tech_terms = style.get("tech_terms", [])

    parts = ["[MIRROR THIS CUSTOMER'S SPEAKING STYLE — match them exactly]"]

    # --- Language adaptation (with specific words) ---
    if language == "hinglish" and hindi_words:
        words_str = '", "'.join(hindi_words[:4])
        parts.append(
            f'Language: They mix Hindi — echo their words back: "{words_str}". '
            "Hinglish is their comfort zone, match it."
        )
    elif language == "hinglish":
        parts.append(
            'Language: They speak Hinglish — mix Hindi naturally. '
            'Use "haan", "accha", "theek hai" like they do.'
        )
    else:
        parts.append("Language: Clean English — no Hindi unless they switch first.")

    # --- Formality adaptation (specific, not generic) ---
    if formality == "casual":
        if casual_markers:
            markers_str = '", "'.join(casual_markers[:3])
            parts.append(
                f'Tone: They\'re casual (used "{markers_str}"). '
                "Match that energy — drop formalities, be a friend not a salesperson."
            )
        else:
            parts.append("Tone: Casual and relaxed — no sir/ma'am, just natural conversation.")

        # Young generation specific
        if young_markers:
            young_str = '", "'.join(young_markers[:3])
            parts.append(
                f'Their vibe: Younger generation (used "{young_str}"). '
                "Keep it relatable, use similar casual expressions."
            )
    elif formality == "formal":
        parts.append(
            'Tone: They\'re formal and respectful. Use "sir/ma\'am", structured sentences. '
            "Be professional — no slang, no shortcuts."
        )
    # neutral: no override

    # --- Vocabulary adaptation ---
    if vocabulary == "technical" and tech_terms:
        terms_str = ", ".join(tech_terms[:3])
        parts.append(
            f"Vocabulary: They know tech ({terms_str}). Be precise, use technical terms. "
            "Don't dumb things down."
        )
    elif vocabulary == "simple":
        parts.append("Vocabulary: Keep it simple — no jargon, plain language, everyday examples.")

    # --- Verbosity matching (critical for rapport) ---
    if verbosity == "terse":
        parts.append(
            "RESPONSE LENGTH: They give SHORT answers. Match that — keep YOUR responses to 1 sentence max. "
            "Don't ramble. Punch, don't lecture."
        )
    elif verbosity == "verbose":
        parts.append(
            "RESPONSE LENGTH: They're detailed and expressive. You can give slightly longer responses "
            "with more context and examples. Match their depth."
        )
    # balanced: default prompt style applies

    # --- Engagement-adaptive ---
    if engagement == "low":
        parts.append(
            "ENERGY: They seem disengaged — short answers, minimal effort. "
            "Be more energetic to pull them in. Ask specific, intriguing questions. "
            "Don't match their low energy, LIFT it."
        )
    elif engagement == "high":
        parts.append(
            "ENERGY: They're engaged and enthusiastic! Ride the momentum — "
            "be equally energetic, build on their excitement."
        )

    # --- Audio mirroring (always-on, leverages Gemini's native audio) ---
    parts.append(
        "AUDIO MIRROR: Match their speaking pace, pitch patterns, and pauses. "
        "If they speak fast and clipped, do the same. If slow and thoughtful, mirror that rhythm."
    )

    return "\n".join(parts)


def style_to_memory(style: dict) -> dict:
    """Convert style dict for DB storage. Returns a clean dict (strips marker lists)."""
    if not style:
        return {}
    return {
        "formality": style.get("formality", "neutral"),
        "language": style.get("language", "english"),
        "vocabulary": style.get("vocabulary", "simple"),
        "verbosity": style.get("verbosity", "balanced"),
        "engagement": style.get("engagement", "medium"),
    }


def style_from_memory(raw) -> dict:
    """Parse style from DB storage (handles str or dict)."""
    if not raw:
        return {}
    if isinstance(raw, str):
        import json
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    if isinstance(raw, dict):
        return raw
    return {}
