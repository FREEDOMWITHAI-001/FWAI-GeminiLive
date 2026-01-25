"""
Plivo + Gemini 2.5 Live Native Audio Stream Handler

Bridges Plivo bidirectional audio stream with Gemini 2.5 Live API
for real-time voice conversations using Gemini native TTS.
"""

import asyncio
import json
import base64
import audioop
from typing import Dict, Optional
from loguru import logger
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types

from src.core.config import config


class PlivoGeminiSession:
    """
    Manages a single call session bridging Plivo audio with Gemini 2.5 Live.

    Audio Flow:
    - Plivo sends mulaw 8kHz audio -> convert to PCM 16kHz -> Gemini Live
    - Gemini Live sends PCM 24kHz audio -> convert to mulaw 8kHz -> Plivo
    """

    def __init__(self, call_uuid: str, caller_phone: str, plivo_ws):
        self.call_uuid = call_uuid
        self.caller_phone = caller_phone
        self.plivo_ws = plivo_ws
        self.gemini_session = None
        self.is_active = False
        self._session_task = None
        self._audio_queue = asyncio.Queue()

    def _save_transcript(self, role: str, text: str):
        """Save transcript entry"""
        if not config.enable_transcripts:
            return
        try:
            transcript_dir = Path(__file__).parent.parent.parent / "transcripts"
            transcript_dir.mkdir(exist_ok=True)
            transcript_file = transcript_dir / f"{self.call_uuid}.txt"
            timestamp = datetime.now().strftime("%H:%M:%S")
            with open(transcript_file, "a") as f:
                f.write(f"[{timestamp}] {role}: {text}\n")
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")

    async def start(self):
        """Initialize Gemini 2.5 Live session"""
        try:
            logger.info(f"Starting Gemini 2.5 Live session for call {self.call_uuid}")
            self.is_active = True
            self._save_transcript("SYSTEM", "Call started")

            # Start the session manager task
            self._session_task = asyncio.create_task(self._run_gemini_session())

            return True

        except Exception as e:
            logger.error(f"Failed to start Gemini session: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _run_gemini_session(self):
        """Run the Gemini Live session within async context manager"""
        try:
            # Initialize Gemini client
            client = genai.Client(api_key=config.google_api_key)

            # System instruction for the AI agent
            system_instruction = """You are Vishnu, a friendly AI counselor at Freedom with AI.

                    Your role:
                    - Help people understand how AI skills can elevate their career
                    - Be warm, conversational, and professional
                    - Ask questions to understand their background and goals
                    - Keep responses concise (1-2 sentences) for natural phone conversation
                    - Listen carefully and respond to what they actually say

                    Start by greeting them warmly and asking about their experience with AI."""

            # Configure Gemini Live with native audio
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

            # Connect to Gemini 2.5 Live using async context manager
            async with client.aio.live.connect(
                model="gemini-2.0-flash-live-001",
                config=live_config
            ) as session:
                self.gemini_session = session
                logger.info(f"Gemini 2.5 Live connected for call {self.call_uuid}")

                # Send initial greeting trigger
                await session.send_client_content(
                    turns=[
                        types.Content(
                            role="user",
                            parts=[types.Part(text="[Call connected. Greet the caller warmly.]")]
                        )
                    ],
                    turn_complete=True
                )

                # Start tasks for receiving from Gemini and sending audio
                receive_task = asyncio.create_task(self._receive_from_gemini())
                send_task = asyncio.create_task(self._send_audio_to_gemini())

                # Wait until session ends
                while self.is_active:
                    await asyncio.sleep(0.1)

                # Cancel tasks
                receive_task.cancel()
                send_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Gemini session error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.gemini_session = None
            logger.info(f"Gemini session ended for call {self.call_uuid}")

    async def _send_audio_to_gemini(self):
        """Send queued audio to Gemini"""
        try:
            while self.is_active:
                try:
                    pcm_data = await asyncio.wait_for(self._audio_queue.get(), timeout=0.5)
                    if self.gemini_session:
                        await self.gemini_session.send_realtime_input(
                            media=types.Blob(data=pcm_data, mime_type="audio/pcm")
                        )
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error sending audio to Gemini: {e}")

    async def _receive_from_gemini(self):
        """Receive audio and text from Gemini Live and send to Plivo"""
        try:
            async for response in self.gemini_session.receive():
                if not self.is_active:
                    break

                # Handle audio data
                if response.data:
                    pcm_data = response.data
                    mulaw_data = self._pcm_to_mulaw(pcm_data)
                    await self._send_audio_to_plivo(mulaw_data)

                # Handle text for transcripts
                if response.text:
                    logger.info(f"VISHNU: {response.text}")
                    self._save_transcript("VISHNU", response.text)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error receiving from Gemini: {e}")
            import traceback
            traceback.print_exc()

    def _pcm_to_mulaw(self, pcm_data: bytes) -> bytes:
        """Convert PCM 24kHz 16-bit (Gemini output) to mulaw 8kHz for Plivo"""
        try:
            # Gemini outputs 24kHz, Plivo needs 8kHz mulaw
            downsampled = audioop.ratecv(pcm_data, 2, 1, 24000, 8000, None)[0]
            mulaw_data = audioop.lin2ulaw(downsampled, 2)
            return mulaw_data
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return b""

    def _mulaw_to_pcm(self, mulaw_data: bytes) -> bytes:
        """Convert mulaw 8kHz to PCM 16kHz 16-bit for Gemini"""
        try:
            pcm_8k = audioop.ulaw2lin(mulaw_data, 2)
            pcm_16k = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)[0]
            return pcm_16k
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return b""

    async def _send_audio_to_plivo(self, mulaw_data: bytes):
        """Send audio to Plivo WebSocket"""
        try:
            if self.plivo_ws and self.is_active:
                audio_b64 = base64.b64encode(mulaw_data).decode("utf-8")
                message = {
                    "event": "playAudio",
                    "media": {
                        "contentType": "audio/x-mulaw",
                        "sampleRate": 8000,
                        "payload": audio_b64
                    }
                }
                await self.plivo_ws.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending audio to Plivo: {e}")

    async def handle_plivo_audio(self, audio_b64: str):
        """Handle incoming audio from Plivo"""
        try:
            if not self.is_active:
                return

            mulaw_data = base64.b64decode(audio_b64)
            pcm_data = self._mulaw_to_pcm(mulaw_data)

            # Queue audio for sending to Gemini
            await self._audio_queue.put(pcm_data)

        except Exception as e:
            logger.error(f"Error handling Plivo audio: {e}")

    async def handle_plivo_message(self, message: dict):
        """Handle messages from Plivo WebSocket"""
        event = message.get("event")

        if event == "media":
            media = message.get("media", {})
            payload = media.get("payload", "")
            if payload:
                await self.handle_plivo_audio(payload)

        elif event == "start":
            logger.info(f"Plivo stream started for call {self.call_uuid}")

        elif event == "stop":
            logger.info(f"Plivo stream stopped for call {self.call_uuid}")
            await self.stop()

        elif event == "dtmf":
            digit = message.get("digit", "")
            logger.info(f"DTMF received: {digit}")

    async def stop(self):
        """Stop the session"""
        logger.info(f"Stopping session for call {self.call_uuid}")
        self.is_active = False

        # Cancel the session task (this will close the Gemini session via context manager)
        if self._session_task:
            self._session_task.cancel()
            try:
                await self._session_task
            except asyncio.CancelledError:
                pass

        self._save_transcript("SYSTEM", "Call ended")
        logger.info(f"Session stopped for call {self.call_uuid}")


# Active sessions
_sessions: Dict[str, PlivoGeminiSession] = {}


async def create_session(call_uuid: str, caller_phone: str, plivo_ws):
    """Create a new Plivo-Gemini session"""
    session = PlivoGeminiSession(call_uuid, caller_phone, plivo_ws)

    if await session.start():
        _sessions[call_uuid] = session
        return session

    return None


async def get_session(call_uuid: str):
    """Get an existing session"""
    return _sessions.get(call_uuid)


async def remove_session(call_uuid: str):
    """Remove and stop a session"""
    if call_uuid in _sessions:
        await _sessions[call_uuid].stop()
        del _sessions[call_uuid]
