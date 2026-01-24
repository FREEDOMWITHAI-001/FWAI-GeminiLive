"""
Gemini with Function Calling Service
Handles AI responses with tool execution capability
"""

import google.generativeai as genai
from typing import Dict, Any, Optional, Tuple
from loguru import logger

from src.core.config import config
from src.tools import get_tool_definitions, execute_tool
from src.prompt_loader import FWAI_PROMPT


# Configure Gemini
genai.configure(api_key=config.google_api_key)


def get_gemini_model_with_tools():
    """Get Gemini model configured with tools"""
    tools = get_tool_definitions()
    
    # Convert to Gemini function declarations format
    function_declarations = []
    for tool in tools:
        function_declarations.append({
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"]
        })
    
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        tools=[{"function_declarations": function_declarations}]
    )
    
    return model


async def generate_response_with_tools(
    history: str,
    user_message: str,
    caller_phone: str
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Generate response using Gemini with function calling
    
    Returns:
        Tuple of (response_text, tool_result)
    """
    try:
        model = get_gemini_model_with_tools()
        
        # Build context
        context = f"""You are Vishnu, currently on a phone call with a potential customer.
The caller's phone number is: {caller_phone}

CONVERSATION SO FAR:
{history}

User just said: {user_message}

IMPORTANT INSTRUCTIONS:
- If user asks to send WhatsApp/SMS/email, use the appropriate tool
- If user wants to book a demo or schedule a callback, use those tools
- Otherwise, respond naturally following the NEPQ sales process
- Always ask ONE question at a time
- Do NOT repeat the greeting or introduce yourself again
"""
        
        full_prompt = FWAI_PROMPT + chr(10) + chr(10) + context
        
        # Generate response
        response = model.generate_content(full_prompt)
        
        # Check if there's a function call
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    tool_name = fc.name
                    tool_args = dict(fc.args) if fc.args else {}
                    
                    logger.info(f"Tool call detected: {tool_name} with args {tool_args}")
                    
                    # Execute the tool
                    tool_result = await execute_tool(
                        tool_name=tool_name,
                        caller_phone=caller_phone,
                        **tool_args
                    )
                    
                    # Generate follow-up response after tool execution
                    if tool_result["success"]:
                        follow_up = await generate_tool_follow_up(tool_name, tool_result)
                        return follow_up, tool_result
                    else:
                        return f"I apologize, I wasn't able to {tool_name.replace('_', ' ')} right now. Let me make a note and we'll follow up. {tool_result['message']}", tool_result
        
        # Regular text response
        reply = response.text.replace('"', "''").strip()
        return reply, None
        
    except Exception as e:
        logger.error(f"Gemini with tools error: {e}")
        return "I apologize, I'm having a small technical issue. Could you please repeat that?", None


async def generate_tool_follow_up(tool_name: str, result: Dict[str, Any]) -> str:
    """Generate a natural follow-up response after tool execution"""
    
    follow_ups = {
        "send_whatsapp": "I've sent you a WhatsApp message with the details. You should receive it shortly. Is there anything specific you'd like me to include?",
        "send_sms": "I've sent you an SMS with the information. You should receive it in a moment. What else would you like to know?",
        "send_email": "I've sent you an email with all the details. Please check your inbox. Is there anything else I can help you with?",
        "schedule_callback": f"Perfect! I've scheduled a callback for {result.get('data', {}).get('preferred_time', 'your preferred time')}. We'll call you then. Is there anything specific you'd like us to cover?",
        "book_demo": f"Excellent! Your demo is booked for {result.get('data', {}).get('preferred_date', 'your preferred date')} at {result.get('data', {}).get('preferred_time', 'your preferred time')}. You'll receive a confirmation shortly. Is there anything specific you'd like to see in the demo?"
    }
    
    return follow_ups.get(tool_name, "Done! Is there anything else I can help you with?")
