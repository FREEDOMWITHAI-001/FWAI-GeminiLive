"""
Product Intelligence — structured product knowledge with progressive revelation.

Users upload raw product documents (text, PDF, URL). Gemini Flash processes them
into categorized sections. During calls, only relevant sections are loaded based
on conversation context — matching the same pattern as personas and situations.

Sections are stored as plain text files in prompts/products/ and are editable via API.
Detection uses keyword matching + situation cross-referencing — zero latency.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional
from loguru import logger

from src.persona_engine import (
    PROMPTS_DIR,
    load_module,
    _load_json_config,
    _file_cache,
    _normalize_transcription,
)

PRODUCTS_DIR = PROMPTS_DIR / "products"
PRODUCT_KEYWORDS_FILE = PROMPTS_DIR / "product_keywords.json"

# Cross-reference: when a situation fires, load the corresponding product section
SITUATION_TO_PRODUCT = {
    "price_objection": "pricing",
    "skepticism": "testimonials",
    "competitor_comparison": "comparison",
    "high_interest": "pricing",
}

# Predefined section types the processing pipeline outputs
SECTION_TYPES = [
    "overview",
    "features",
    "pricing",
    "benefits",
    "testimonials",
    "objection_handling",
    "comparison",
]


# =============================================================================
# Document Processing (Gemini Flash)
# =============================================================================

# Reuse the same genai client pattern as intelligence.py
from google import genai
from google.genai import types
from src.core.config import config

_client = genai.Client(api_key=config.google_api_key) if config.google_api_key else None

_PROCESSING_PROMPT = """You are a product knowledge structurer. Given raw product/service content, extract and organize it into structured sections.

Output ONLY valid JSON with this exact structure:
{
  "sections": {
    "overview": "2-3 sentence product summary — what it is, who it's for, key differentiator",
    "features": "[PRODUCT KNOWLEDGE: Features]\\nBullet-point list of what's included, formatted for an AI sales agent to reference during calls",
    "pricing": "[PRODUCT KNOWLEDGE: Pricing]\\nPrice points, plans, payment options, value justification",
    "benefits": "[PRODUCT KNOWLEDGE: Benefits]\\nKey outcomes, results, ROI — things that make customers say yes",
    "testimonials": "[PRODUCT KNOWLEDGE: Social Proof]\\nSuccess stories, numbers, ratings, notable achievements",
    "objection_handling": "[PRODUCT KNOWLEDGE: Objection Handling]\\nCommon concerns and how to reframe them positively",
    "comparison": "[PRODUCT KNOWLEDGE: vs Alternatives]\\nHow this differs from competitors/alternatives, unique advantages"
  },
  "keywords": {
    "features": ["keyword1", "keyword2", ...],
    "pricing": ["keyword1", "keyword2", ...],
    "benefits": ["keyword1", "keyword2", ...],
    "testimonials": ["keyword1", "keyword2", ...],
    "objection_handling": ["keyword1", "keyword2", ...],
    "comparison": ["keyword1", "keyword2", ...]
  }
}

Rules:
- Each section should be written as INSTRUCTIONS for an AI sales agent (not as a document for humans)
- Use concise bullet points, not paragraphs
- Keywords should be words/phrases a CUSTOMER would say that indicate this section is relevant
- Generate 5-10 keywords per section
- If the source content doesn't have info for a section, set it to empty string ""
- overview never needs keywords (it's always loaded)

Raw product content:
"""


async def process_document(raw_text: str, source_type: str = "text") -> dict:
    """
    Process raw product content into structured sections via Gemini Flash.
    Returns {"sections": {...}, "keywords": {...}} or {"error": "..."}.
    """
    if not _client:
        return {"error": "Google API key not configured"}

    if not raw_text or not raw_text.strip():
        return {"error": "Empty content"}

    # For URLs, fetch content first
    if source_type == "url":
        raw_text = await _fetch_url_content(raw_text.strip())
        if not raw_text:
            return {"error": "Failed to fetch URL content"}

    try:
        start = time.time()

        # For PDF, raw_text would be base64 — handle via parts
        if source_type == "pdf":
            import base64
            pdf_bytes = base64.b64decode(raw_text)
            contents = [
                _PROCESSING_PROMPT,
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            ]
        else:
            contents = _PROCESSING_PROMPT + raw_text

        response = await asyncio.wait_for(
            _client.aio.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            ),
            timeout=30.0,
        )

        elapsed_ms = (time.time() - start) * 1000
        response_text = response.text.strip() if response.text else ""

        if not response_text:
            logger.warning(f"Product processing returned empty after {elapsed_ms:.0f}ms")
            return {"error": "AI returned empty response"}

        result = json.loads(response_text)
        sections = result.get("sections", {})
        keywords = result.get("keywords", {})

        # Filter out empty sections
        sections = {k: v for k, v in sections.items() if v and v.strip()}

        logger.info(
            f"Product processed in {elapsed_ms:.0f}ms: "
            f"{len(sections)} sections created"
        )
        return {"sections": sections, "keywords": keywords}

    except asyncio.TimeoutError:
        logger.error("Product processing timed out after 30s")
        return {"error": "Processing timed out"}
    except json.JSONDecodeError as e:
        logger.error(f"Product processing returned invalid JSON: {e}")
        return {"error": f"Invalid AI response: {e}"}
    except Exception as e:
        logger.error(f"Product processing failed: {e}")
        return {"error": str(e)}


async def _fetch_url_content(url: str) -> str:
    """Fetch text content from a URL."""
    try:
        import httpx
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "html" in content_type:
                # Strip HTML tags for plain text extraction
                import re
                text = re.sub(r"<script[^>]*>.*?</script>", "", resp.text, flags=re.DOTALL)
                text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                return text[:15000]  # Cap to avoid huge prompts
            return resp.text[:15000]
    except Exception as e:
        logger.error(f"Failed to fetch URL {url}: {e}")
        return ""


# =============================================================================
# Section Storage
# =============================================================================

def save_product_sections(sections: dict, keywords: dict):
    """Save processed sections to files and keywords to config."""
    PRODUCTS_DIR.mkdir(exist_ok=True)

    for name, content in sections.items():
        if content and content.strip():
            path = PRODUCTS_DIR / f"{name}.txt"
            path.write_text(content.strip(), encoding="utf-8")
            # Invalidate cache for this file
            _file_cache.pop(str(path), None)
            logger.debug(f"Saved product section: {name} ({len(content)} chars)")

    # Merge keywords into config (preserve existing, update with new)
    existing_kw = _load_json_config(PRODUCT_KEYWORDS_FILE)
    for section_name, kw_list in keywords.items():
        if isinstance(kw_list, list):
            existing_kw[section_name] = {"keywords": kw_list}
    # Ensure overview exists with empty keywords
    existing_kw.setdefault("overview", {"keywords": []})

    PRODUCT_KEYWORDS_FILE.write_text(
        json.dumps(existing_kw, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _file_cache.pop(str(PRODUCT_KEYWORDS_FILE), None)
    logger.info(f"Product keywords config updated ({len(existing_kw)} sections)")


def load_all_product_sections() -> dict[str, str]:
    """Scan prompts/products/ and return {name: content}."""
    result = {}
    if PRODUCTS_DIR.exists():
        for f in PRODUCTS_DIR.glob("*.txt"):
            result[f.stem] = load_module(f)
    return result


# =============================================================================
# Progressive Revelation — Detection & Loading
# =============================================================================

def detect_product_sections(
    user_text: str,
    active_situations: list,
) -> list[str]:
    """
    Detect which product sections to load based on conversation context.
    Uses keyword matching + situation cross-referencing.
    Always includes 'overview'. Caps at 3 sections.
    """
    active = []

    # Always include overview if the file exists
    if (PRODUCTS_DIR / "overview.txt").exists():
        active.append("overview")

    # Cross-reference with active situations (zero cost)
    for sit in active_situations:
        mapped = SITUATION_TO_PRODUCT.get(sit)
        if mapped and mapped not in active and (PRODUCTS_DIR / f"{mapped}.txt").exists():
            active.append(mapped)

    # Keyword matching against product_keywords.json
    if user_text:
        kw_config = _load_json_config(PRODUCT_KEYWORDS_FILE)
        text_lower = _normalize_transcription(user_text).lower()

        for section_key, section_config in kw_config.items():
            if section_key in active:
                continue
            if not (PRODUCTS_DIR / f"{section_key}.txt").exists():
                continue
            for keyword in section_config.get("keywords", []):
                if keyword in text_lower:
                    active.append(section_key)
                    break

    return active[:3]  # Cap at 3 sections


def load_product_sections(section_keys: list) -> str:
    """Load and concatenate product section modules."""
    parts = []
    for key in section_keys:
        content = load_module(PRODUCTS_DIR / f"{key}.txt")
        if content:
            parts.append(content)
    if parts:
        return "\n\n".join(parts)
    return ""
