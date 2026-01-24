import json
from pathlib import Path

def load_prompt(prompt_id="FWAI_Core"):
    """Load a prompt from prompts.json"""
    try:
        prompts_file = Path(__file__).parent.parent / "prompts.json"
        with open(prompts_file, "r") as f:
            prompts = json.load(f)
        return prompts.get(prompt_id, {}).get("prompt", "You are a helpful assistant.")
    except Exception as e:
        print(f"Error loading prompt: {e}")
        return "You are a helpful assistant."

# Default FWAI prompt
FWAI_PROMPT = load_prompt("FWAI_Core")
