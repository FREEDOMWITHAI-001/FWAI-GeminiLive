"""
Gemini Service - Optimized for low latency and natural conversation
"""

import google.generativeai as genai
from typing import Dict, Any, Optional, Tuple
from loguru import logger
import time

from src.core.config import config
from src.tools import get_tool_definitions, execute_tool


# Configure Gemini ONCE at module load
genai.configure(api_key=config.google_api_key)

# Pre-load model at startup
_model = None

def _get_model():
    global _model
    if _model is None:
        start = time.time()
        tools = get_tool_definitions()
        function_declarations = [{
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"]
        } for t in tools]
        
        _model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            tools=[{"function_declarations": function_declarations}]
        )
        logger.info(f"Model initialized in {time.time()-start:.2f}s")
    return _model

# Initialize at import
_get_model()


async def generate_response_with_tools(
    history: str,
    user_message: str,
    caller_phone: str
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Generate conversational response"""
    
    start = time.time()
    
    try:
        model = _get_model()
        
        # Simple, focused prompt that emphasizes listening
        prompt = f"""You are Vishnu, a friendly AI counselor on a phone call.

CRITICAL: Actually respond to what the user says! Do not ignore their words.

Conversation so far:
{history}

User just said: "{user_message}"

Instructions:
- DIRECTLY address what they just said
- If they said something negative, acknowledge it empathetically
- If they asked a question, answer it
- If they want info sent (WhatsApp/SMS/email), use the tool
- Keep response under 2 sentences
- Ask ONE follow-up question if appropriate

Respond naturally:"""
        
        # Generate
        gemini_start = time.time()
        response = model.generate_content(prompt)
        gemini_time = time.time() - gemini_start
        logger.info(f"GEMINI: {gemini_time:.2f}s")
        
        # Check for tool call
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    tool_name = fc.name
                    tool_args = dict(fc.args) if fc.args else {}
                    
                    logger.info(f"TOOL: {tool_name}")
                    result = await execute_tool(tool_name, caller_phone, **tool_args)
                    
                    if result["success"]:
                        followup = {
                            "send_whatsapp": "Done! I sent you a WhatsApp. What else?",
                            "send_sms": "SMS sent! Anything else?",
                            "send_email": "Email sent! What else can I help with?",
                            "schedule_callback": "Callback scheduled! Anything specific to discuss?",
                            "book_demo": "Demo booked! Looking forward to it."
                        }.get(tool_name, "Done!")
                        logger.info(f"TOTAL: {time.time()-start:.2f}s")
                        return followup, result
                    return "Sorry, I could not do that. What else can I help with?", result
        
        reply = response.text.replace('"', "''").strip()
        logger.info(f"REPLY: {reply[:50]}...")
        logger.info(f"TOTAL: {time.time()-start:.2f}s")
        return reply, None
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return "Sorry, could you say that again?", None
