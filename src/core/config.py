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

    # Active call provider: "whatsapp", "plivo", "exotel"
    call_provider: str = os.getenv("CALL_PROVIDER", "whatsapp")

    # WhatsApp API settings
    phone_number_id: str = os.getenv("PHONE_NUMBER_ID", "")
    meta_access_token: str = os.getenv("META_ACCESS_TOKEN", "")
    meta_verify_token: str = os.getenv("META_VERIFY_TOKEN", "my_super_secret_token_123")
    whatsapp_api_version: str = "v21.0"

    # Meta App credentials for token refresh
    meta_app_id: str = os.getenv("META_APP_ID", "")
    meta_app_secret: str = os.getenv("META_APP_SECRET", "")

    # Plivo API settings
    plivo_auth_id: str = os.getenv("PLIVO_AUTH_ID", "")
    plivo_auth_token: str = os.getenv("PLIVO_AUTH_TOKEN", "")
    plivo_phone_number: str = os.getenv("PLIVO_PHONE_NUMBER", "")
    plivo_callback_url: str = os.getenv("PLIVO_CALLBACK_URL", "")

    # Exotel API settings
    exotel_api_key: str = os.getenv("EXOTEL_API_KEY", "")
    exotel_api_token: str = os.getenv("EXOTEL_API_TOKEN", "")
    exotel_sid: str = os.getenv("EXOTEL_SID", "")
    exotel_caller_id: str = os.getenv("EXOTEL_CALLER_ID", "")
    exotel_app_id: str = os.getenv("EXOTEL_APP_ID", "")
    exotel_callback_url: str = os.getenv("EXOTEL_CALLBACK_URL", "")

    # Google Gemini settings
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    tts_voice: str = os.getenv("TTS_VOICE", "Kore")

    # Vertex AI settings (for lower latency with regional endpoints)
    use_vertex_ai: bool = os.getenv("USE_VERTEX_AI", "true").lower() == "true"
    vertex_project_id: str = os.getenv("VERTEX_PROJECT_ID", "")
    vertex_location: str = os.getenv("VERTEX_LOCATION", "us-central1")
    vertex_credentials_path: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

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

    # Logging settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    enable_transcripts: bool = os.getenv("ENABLE_TRANSCRIPTS", "true").lower() == "true"
    enable_whisper: bool = os.getenv("ENABLE_WHISPER", "false").lower() == "true"  # Whisper transcription (RAM heavy)
    enable_detailed_logging: bool = os.getenv("ENABLE_DETAILED_LOGGING", "true").lower() == "true"
    
    # SMTP Email settings
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_from_email: str = os.getenv("SMTP_FROM_EMAIL", "noreply@freedomwithai.com")
    
    # WhatsApp Phone ID (for Meta API)
    whatsapp_phone_id: str = os.getenv("WHATSAPP_PHONE_ID", "")

    def validate_config(self) -> list[str]:
        """Validate required configuration values based on active provider"""
        errors = []

        # Always required (Google API key needed as fallback even with Vertex AI)
        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required")

        # Vertex AI validation
        if self.use_vertex_ai:
            if not self.vertex_project_id:
                errors.append("VERTEX_PROJECT_ID is required when USE_VERTEX_AI=true")

        # Provider-specific validation
        if self.call_provider == "whatsapp":
            if not self.phone_number_id:
                errors.append("PHONE_NUMBER_ID is required for WhatsApp")
            if not self.meta_access_token:
                errors.append("META_ACCESS_TOKEN is required for WhatsApp")

        elif self.call_provider == "plivo":
            if not self.plivo_auth_id:
                errors.append("PLIVO_AUTH_ID is required for Plivo")
            if not self.plivo_auth_token:
                errors.append("PLIVO_AUTH_TOKEN is required for Plivo")
            if not self.plivo_phone_number:
                errors.append("PLIVO_PHONE_NUMBER is required for Plivo")

        elif self.call_provider == "exotel":
            if not self.exotel_api_key:
                errors.append("EXOTEL_API_KEY is required for Exotel")
            if not self.exotel_api_token:
                errors.append("EXOTEL_API_TOKEN is required for Exotel")
            if not self.exotel_sid:
                errors.append("EXOTEL_SID is required for Exotel")
            if not self.exotel_caller_id:
                errors.append("EXOTEL_CALLER_ID is required for Exotel")

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

    

