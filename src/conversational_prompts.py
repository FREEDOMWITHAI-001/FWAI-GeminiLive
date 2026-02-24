# Conversational Prompts - Template rendering for API-provided prompts
# Uses {{variable}} placeholders filled from API context

# Default context values (used if not provided in API call)
# These are generic fallbacks — org-specific values come from the UI's bot config context
DEFAULT_CONTEXT = {
    "agent_name": "Agent",
    "company_name": "",
    "location": "",
    "customer_name": "there",
    "event_name": "",
    "event_host": "",
    "product_name": "",
    "product_description": "",
    "price": "",
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
