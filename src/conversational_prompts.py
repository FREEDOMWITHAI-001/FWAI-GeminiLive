# Conversational Prompts - Template rendering for API-provided prompts
# Uses {{variable}} placeholders filled from API context

# Default context values (used if not provided in API call)
DEFAULT_CONTEXT = {
    "agent_name": "Rahul Kumar",
    "company_name": "Freedom with AI",
    "location": "Hyderabad",
    "customer_name": "there",
    "event_name": "AI Masterclass",
    "event_host": "Avinash Mada",
    "product_name": "Gold Membership",
    "product_description": "500+ AI tools, prompt engineering, Python & LangChain, mentorship, AI Expert Certification",
    "price": "40,000 rupees",
    "intelligence_brief": "",
}


def render_prompt(template: str, context: dict) -> str:
    """Replace placeholders with context values, using defaults for missing keys.
    Supports both {{key}} and {key} placeholder formats."""
    merged_context = {**DEFAULT_CONTEXT, **context}
    result = template
    for key, value in merged_context.items():
        # Replace double-brace first ({{key}}), then single-brace ({key})
        result = result.replace("{{" + key + "}}", str(value))
        result = result.replace("{" + key + "}", str(value))
    return result
