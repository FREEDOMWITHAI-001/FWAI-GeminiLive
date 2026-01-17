"""
Configuration management for WhatsApp Voice Calling with Gemini Live
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional

# Load environment variables
load_dotenv()


class Config(BaseModel):
    """Application configuration"""

    # Server settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "3000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # WhatsApp API settings
    phone_number_id: str = os.getenv("PHONE_NUMBER_ID", "")
    meta_access_token: str = os.getenv("META_ACCESS_TOKEN", "")
    meta_verify_token: str = os.getenv("META_VERIFY_TOKEN", "my_super_secret_token_123")
    whatsapp_api_version: str = "v21.0"

    # Google Gemini settings
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    tts_voice: str = os.getenv("TTS_VOICE", "Kore")

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit audio

    # Gemini Live Service (gemini-live-service.py)
    gemini_live_host: str = os.getenv("GEMINI_LIVE_HOST", "localhost")
    gemini_live_port: int = int(os.getenv("GEMINI_LIVE_PORT", "8003"))

    @property
    def gemini_live_ws_url(self) -> str:
        return f"ws://{self.gemini_live_host}:{self.gemini_live_port}"

    # WebRTC settings
    ice_servers: list = [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
    ]

    @property
    def whatsapp_api_url(self) -> str:
        return f"https://graph.facebook.com/{self.whatsapp_api_version}/{self.phone_number_id}"

    @property
    def whatsapp_calls_url(self) -> str:
        return f"{self.whatsapp_api_url}/calls"

    @property
    def whatsapp_messages_url(self) -> str:
        return f"{self.whatsapp_api_url}/messages"

    def validate_config(self) -> list[str]:
        """Validate required configuration values"""
        errors = []
        if not self.phone_number_id:
            errors.append("PHONE_NUMBER_ID is required")
        if not self.meta_access_token:
            errors.append("META_ACCESS_TOKEN is required")
        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required")
        return errors


# Global config instance
config = Config()


# Conversation script path
CONVERSATION_SCRIPT_PATH = Path(__file__).parent / "FAWI_Call_BOT.txt"


def load_conversation_script() -> str:
    """Load the conversation script for the AI agent"""
    try:
        if CONVERSATION_SCRIPT_PATH.exists():
            script_content = CONVERSATION_SCRIPT_PATH.read_text(encoding='utf-8')

            system_prompt = f"""You are Mousumi, a Senior Counselor at Freedom with AI. You help people guide their career path using AI skills and how they can make more money out of it.

CONVERSATION SCRIPT:
{script_content}

INSTRUCTIONS:
- Follow the conversation flow from the script above
- Be warm, friendly, and professional
- Ask questions naturally and wait for responses
- Use Indian English accent naturally
- Guide the conversation through connecting questions, situation questions, problem-aware questions, solution-aware questions, and consequence questions
- Present the three pillars when appropriate
- Handle objections professionally
- Keep responses conversational and natural
- Respond in a human-like manner with appropriate pauses and acknowledgments"""

            return system_prompt
        else:
            return "You are Mousumi, a Senior Counselor at Freedom with AI. Help people with AI skills and career guidance."
    except Exception as e:
        print(f"Error loading conversation script: {e}")
        return "You are Mousumi, a Senior Counselor at Freedom with AI."
