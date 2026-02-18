"""
Social Proof Engine — real-time enrollment stats for mid-call injection.

Pre-call: loads aggregate summary stats for system prompt injection.
Mid-call: get_social_proof() tool call returns specific stats when AI learns
          the prospect's company, city, or role.

Zero latency impact: all reads are local SQLite (~1ms).
Stats updated via REST API / CRM webhook.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from loguru import logger
from src.db.session_db import session_db


# =========================================================================
# Pre-Call: Load Generic Summary for System Prompt
# =========================================================================

def load_social_proof_summary() -> str:
    """
    Load aggregate stats for system prompt injection (pre-call).
    Returns a natural-language block or empty string if no stats exist.
    ~1ms local SQLite read.
    """
    top_companies = session_db.get_social_proof_top("social_proof_company", limit=5)
    top_cities = session_db.get_social_proof_top("social_proof_city", limit=5)
    top_roles = session_db.get_social_proof_top("social_proof_role", limit=5)

    if not top_companies and not top_cities and not top_roles:
        return ""

    total_company = session_db.get_social_proof_total("social_proof_company")
    num_companies = len(session_db.get_social_proof_top("social_proof_company", limit=100))

    parts = []
    parts.append(f"Total enrollees from companies: {total_company:,}+ across {num_companies}+ companies")

    if top_companies:
        company_list = ", ".join(
            f"{c['company_name']} ({c['enrollments_count']})"
            for c in top_companies
        )
        parts.append(f"Top companies: {company_list}")

    if top_cities:
        city_list = ", ".join(
            f"{c['city_name']} ({c['enrollments_count']})"
            for c in top_cities
        )
        parts.append(f"Top cities: {city_list}")

    if top_roles:
        role_list = ", ".join(
            f"{r['role_name']} ({r['enrollments_count']})"
            for r in top_roles
        )
        parts.append(f"Top roles: {role_list}")

    return "\n".join(parts)


# =========================================================================
# Mid-Call: Specific Stats via Gemini Tool Call
# =========================================================================

def get_social_proof(
    company: Optional[str] = None,
    city: Optional[str] = None,
    role: Optional[str] = None,
) -> dict:
    """
    Query specific social proof stats. Called from _handle_tool_call().
    Returns dict with structured stats + suggested phrasing + instruction.
    All reads are local SQLite (~1ms).
    """
    results = {}

    if company:
        stats = session_db.get_social_proof_by_company(company)
        if stats:
            results["company"] = stats
            results["company_phrase"] = _format_company_phrase(stats)

    if city:
        stats = session_db.get_social_proof_by_city(city)
        if stats:
            results["city"] = stats
            results["city_phrase"] = _format_city_phrase(stats)

    if role:
        stats = session_db.get_social_proof_by_role(role)
        if stats:
            results["role"] = stats
            results["role_phrase"] = _format_role_phrase(stats)

    if not results:
        # Fallback: general aggregate
        total = session_db.get_social_proof_total("social_proof_company")
        num_companies = len(session_db.get_social_proof_top("social_proof_company", limit=100))
        if total > 0:
            results["general_phrase"] = (
                f"We have over {total:,} enrollees from {num_companies}+ companies across India."
            )
        else:
            results["general_phrase"] = (
                "We have thousands of enrollees from top companies across India."
            )

    results["instruction"] = (
        "Use ONE of the above stats naturally in your next response. "
        "Do NOT list multiple stats. Pick the most relevant one and weave it in casually. "
        "Example: 'Actually, 12 people from Wipro alone enrolled last quarter.'"
    )

    logger.debug(f"Social proof result: company={company}, city={city}, role={role}, keys={list(results.keys())}")
    return results


# =========================================================================
# Formatting Helpers
# =========================================================================

def _format_company_phrase(stats: dict) -> str:
    """Generate natural-sounding phrase from company stats."""
    count = stats.get("enrollments_count", 0)
    company = stats.get("company_name", "your company")
    outcome = stats.get("notable_outcomes", "")

    if count <= 0:
        return ""

    phrase = f"Actually, {count} people from {company} have enrolled"

    last_date = stats.get("last_enrollment_date")
    if last_date:
        try:
            dt = datetime.fromisoformat(last_date)
            days_ago = (datetime.now() - dt).days
            if days_ago < 30:
                phrase += " recently"
            elif days_ago < 90:
                phrase += " in the last few months"
            else:
                phrase += " in the last year"
        except Exception:
            pass

    if outcome:
        phrase += f". {outcome}"

    return phrase


def _format_city_phrase(stats: dict) -> str:
    """Generate natural-sounding phrase from city stats."""
    count = stats.get("enrollments_count", 0)
    city = stats.get("city_name", "your city")
    trending = stats.get("trending", 0)

    if count <= 0:
        return ""

    phrase = f"In {city}, we've had {count} sign-ups"
    if trending:
        phrase += " and it's been growing fast recently"

    return phrase


def _format_role_phrase(stats: dict) -> str:
    """Generate natural-sounding phrase from role stats."""
    count = stats.get("enrollments_count", 0)
    role = stats.get("role_name", "your role")
    outcome = stats.get("success_stories", "")

    if count <= 0:
        return ""

    phrase = f"We have {count} {role}s in the program"
    if outcome:
        phrase += f". {outcome}"

    return phrase
