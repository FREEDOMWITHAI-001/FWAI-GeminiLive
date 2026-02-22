"""
Pre-call intelligence gathering using Gemini Flash + Google Search grounding.

Runs BEFORE the phone rings (during preload phase) to research the prospect's
company, role, and industry. Results are injected into the Gemini Live session
so the AI can reference them naturally during conversation.

Zero new dependencies - uses existing google-genai SDK and GOOGLE_API_KEY.
"""

import asyncio
import time
from loguru import logger
from google import genai
from google.genai import types
from src.core.config import config

# Pre-warm client at import time (avoids cold-start on first call)
_client = genai.Client(api_key=config.google_api_key) if config.google_api_key else None


async def gather_intelligence(contact_name: str, context: dict, timeout: float = None) -> str:
    """
    Research contact/company before call starts. Returns intelligence brief or empty string.
    Guaranteed to return within `timeout` seconds (default from config).
    """
    if not config.enable_intelligence or not _client:
        return ""

    if timeout is None:
        timeout = config.intelligence_timeout

    # Build search query from available context
    company = context.get("company_name", "") or context.get("company", "")
    role = context.get("role", "") or context.get("job_title", "") or context.get("designation", "")
    industry = context.get("industry", "") or context.get("sector", "")

    # Need at least a company or industry to search meaningfully
    if not company and not industry:
        logger.debug("No company/industry in context - skipping pre-call intelligence")
        return ""

    # Build a richer query from available context
    parts = []
    if company:
        parts.append(f"{company} company overview, employee count, recent news 2025 2026")
    elif industry:
        parts.append(f"{industry} industry trends 2025 2026")
    if role:
        parts.append(f"key challenges for {role} in {industry or 'their industry'}")
    query = ". ".join(parts) + ". 4 bullet points, one sentence each, focus on facts."

    try:
        start = time.time()

        response = await asyncio.wait_for(
            _client.aio.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=query,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            ),
            timeout=timeout
        )

        elapsed_ms = (time.time() - start) * 1000
        brief = response.text.strip() if response.text else ""

        if brief:
            logger.info(f"Intelligence gathered in {elapsed_ms:.0f}ms ({len(brief)} chars)")
        else:
            logger.debug(f"Intelligence search returned empty in {elapsed_ms:.0f}ms")
        return brief

    except asyncio.TimeoutError:
        elapsed_ms = (time.time() - start) * 1000
        logger.warning(f"Intelligence gathering timed out after {elapsed_ms:.0f}ms - proceeding without")
        return ""
    except Exception as e:
        logger.warning(f"Intelligence gathering failed: {e} - proceeding without")
        return ""
