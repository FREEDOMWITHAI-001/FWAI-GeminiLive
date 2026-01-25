"""
Gemini Live TTS - Get audio response from Gemini Live API
Sends text, receives audio back
"""

import asyncio
import os
import uuid
from pathlib import Path
from loguru import logger

from google import genai
from google.genai import types

from src.core.config import config


# Directory to store audio files
AUDIO_DIR = Path(__file__).parent.parent.parent / "audio_cache"
AUDIO_DIR.mkdir(exist_ok=True)


async def generate_audio_response(text: str, call_uuid: str) -> str:
    """
    Send text to Gemini Live and get audio response.
    Returns the filename of the saved audio file.
    """
    try:
        logger.info(f"Generating audio for: {text[:50]}...")

        # Initialize Gemini client
        client = genai.Client(api_key=config.google_api_key)

        # System instruction
        system_instruction = """You are Vishnu, a friendly AI counselor at Freedom with AI.
Speak naturally and conversationally. Keep responses concise for phone calls."""

        # Configure Gemini Live with audio output
        live_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Kore"
                    )
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=system_instruction)]
            ),
        )

        # Connect to Gemini Live
        async with client.aio.live.connect(
            model="gemini-2.0-flash-live-001",
            config=live_config
        ) as session:

            # Send the text to speak
            await session.send_client_content(
                turns=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=f"Say this exactly: {text}")]
                    )
                ],
                turn_complete=True
            )

            # Collect audio chunks
            audio_chunks = []

            async for response in session.receive():
                if response.data:
                    audio_chunks.append(response.data)

                # Check if turn is complete
                if response.server_content and response.server_content.turn_complete:
                    break

            if not audio_chunks:
                logger.error("No audio received from Gemini")
                return None

            # Combine audio chunks
            audio_data = b''.join(audio_chunks)

            # Save to file (PCM 16kHz 16-bit mono)
            filename = f"{call_uuid}_{uuid.uuid4().hex[:8]}.wav"
            filepath = AUDIO_DIR / filename

            # Write WAV file
            import wave
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(24000)  # Gemini outputs 24kHz
                wav_file.writeframes(audio_data)

            logger.info(f"Audio saved: {filename} ({len(audio_data)} bytes)")
            return filename

    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_audio_url(filename: str) -> str:
    """Get the URL to serve the audio file"""
    return f"{config.plivo_callback_url}/audio/{filename}"
