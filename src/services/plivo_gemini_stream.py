# Plivo + Google Live API Stream Handler with Preloading
import asyncio
import json
import base64
from typing import Dict, Optional
from loguru import logger
from datetime import datetime
from pathlib import Path
import websockets
from src.core.config import config
from src.tools import execute_tool

# Load FWAI prompt
def load_fwai_prompt():
    try:
        prompts_file = Path(__file__).parent.parent.parent / "prompts.json"
        with open(prompts_file) as f:
            prompts = json.load(f)
            return prompts.get("FWAI_Core", {}).get("prompt", "You are a helpful AI assistant.")
    except:
        return "You are a helpful AI assistant."

FWAI_PROMPT = load_fwai_prompt()

# Tool definitions for Gemini Live
TOOL_DECLARATIONS = [
    {
        "name": "send_whatsapp",
        "description": "Send a WhatsApp message to the caller with course details, links, or follow-up information. Use this when the user asks to receive information on WhatsApp.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message content to send via WhatsApp"
                }
            },
            "required": ["message"]
        }
    },
    {
        "name": "schedule_callback",
        "description": "Schedule a callback for the caller at their preferred time. Use this when the user wants to be called back later.",
        "parameters": {
            "type": "object",
            "properties": {
                "preferred_time": {
                    "type": "string",
                    "description": "Preferred callback time (e.g., 'tomorrow morning', '3 PM today', 'Monday 10 AM')"
                },
                "notes": {
                    "type": "string",
                    "description": "Any notes about what to discuss on callback"
                }
            },
            "required": ["preferred_time"]
        }
    },
    {
        "name": "send_sms",
        "description": "Send an SMS message to the caller with brief information. Use this when user prefers SMS over WhatsApp.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The SMS message content (keep it brief)"
                }
            },
            "required": ["message"]
        }
    }
]

class PlivoGeminiSession:
    def __init__(self, call_uuid: str, caller_phone: str):
        self.call_uuid = call_uuid
        self.caller_phone = caller_phone
        self.plivo_ws = None  # Will be set when WebSocket connects
        self.goog_live_ws = None
        self.is_active = False
        self.start_streaming = False
        self.stream_id = ""
        self._session_task = None
        self._audio_buffer_task = None
        self.BUFFER_SIZE = 320  # Ultra-low latency (20ms chunks)
        self.inbuffer = bytearray(b"")
        self.greeting_sent = False
        self.setup_complete = False
        self.preloaded_audio = []  # Store audio generated during preload
        self._preload_complete = asyncio.Event()

    def _save_transcript(self, role, text):
        if not config.enable_transcripts:
            return
        try:
            transcript_dir = Path(__file__).parent.parent.parent / "transcripts"
            transcript_dir.mkdir(exist_ok=True)
            transcript_file = transcript_dir / f"{self.call_uuid}.txt"
            timestamp = datetime.now().strftime("%H:%M:%S")
            with open(transcript_file, "a") as f:
                f.write(f"[{timestamp}] {role}: {text}\n")
            logger.info(f"TRANSCRIPT [{role}]: {text}")
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")

    async def preload(self):
        """Preload the Gemini session while phone is ringing"""
        try:
            logger.info(f"PRELOADING Gemini session for call {self.call_uuid}")
            self.is_active = True
            self._session_task = asyncio.create_task(self._run_google_live_session())
            # Wait for setup to complete (with timeout)
            try:
                await asyncio.wait_for(self._preload_complete.wait(), timeout=8.0)
                logger.info(f"PRELOAD COMPLETE for {self.call_uuid} - AI ready to speak!")
            except asyncio.TimeoutError:
                logger.warning(f"Preload timeout for {self.call_uuid}")
            return True
        except Exception as e:
            logger.error(f"Failed to preload session: {e}")
            return False

    def attach_plivo_ws(self, plivo_ws):
        """Attach Plivo WebSocket when user answers"""
        self.plivo_ws = plivo_ws
        logger.info(f"Plivo WS attached for {self.call_uuid}")
        # Send any preloaded audio immediately
        if self.preloaded_audio:
            asyncio.create_task(self._send_preloaded_audio())

    async def _send_preloaded_audio(self):
        """Send preloaded audio to Plivo"""
        logger.info(f"Sending {len(self.preloaded_audio)} preloaded audio chunks")
        for audio in self.preloaded_audio:
            if self.plivo_ws:
                await self.plivo_ws.send_text(json.dumps({
                    "event": "playAudio",
                    "media": {"contentType": "audio/x-l16", "sampleRate": 24000, "payload": audio}
                }))
        self.preloaded_audio = []

    async def _run_google_live_session(self):
        url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={config.google_api_key}"
        try:
            async with websockets.connect(url) as ws:
                self.goog_live_ws = ws
                logger.info("Connected to Google Live API")
                await self._send_session_setup()
                async for message in ws:
                    if not self.is_active:
                        break
                    await self._receive_from_google(message)
        except Exception as e:
            logger.error(f"Google Live error: {e}")
        finally:
            self.goog_live_ws = None
            logger.info("Google Live session ended")

    async def _send_session_setup(self):
        msg = {
            "setup": {
                "model": "models/gemini-2.0-flash-exp",
                "generation_config": {
                    "response_modalities": ["AUDIO"],
                    "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Charon"}}}
                },
                "system_instruction": {"parts": [{"text": FWAI_PROMPT}]},
                "tools": [{"function_declarations": TOOL_DECLARATIONS}]
            }
        }
        await self.goog_live_ws.send(json.dumps(msg))
        logger.info("Sent session setup with FWAI prompt and tools")

    async def _send_initial_greeting(self):
        """Send initial trigger to make AI greet immediately"""
        if self.greeting_sent or not self.goog_live_ws:
            return
        self.greeting_sent = True
        msg = {
            "client_content": {
                "turns": [{
                    "role": "user",
                    "parts": [{"text": "Hi"}]
                }],
                "turn_complete": True
            }
        }
        await self.goog_live_ws.send(json.dumps(msg))
        logger.info("Sent initial greeting trigger")

    async def _handle_tool_call(self, tool_call):
        """Execute tool and send response back to Gemini"""
        try:
            func_calls = tool_call.get("functionCalls", [])
            for fc in func_calls:
                tool_name = fc.get("name")
                tool_args = fc.get("args", {})
                call_id = fc.get("id")

                logger.info(f"TOOL CALL: {tool_name} with args: {tool_args}")
                self._save_transcript("TOOL", f"{tool_name}: {tool_args}")

                # Execute the tool
                result = await execute_tool(tool_name, self.caller_phone, **tool_args)

                logger.info(f"TOOL RESULT: {result}")
                self._save_transcript("TOOL_RESULT", f"{tool_name}: {'success' if result.get('success') else 'failed'}")

                # Send tool response back to Gemini
                tool_response = {
                    "tool_response": {
                        "function_responses": [{
                            "id": call_id,
                            "name": tool_name,
                            "response": {
                                "success": result.get("success", False),
                                "message": result.get("message", "Tool executed")
                            }
                        }]
                    }
                }
                await self.goog_live_ws.send(json.dumps(tool_response))
                logger.info(f"Sent tool response for {tool_name}")

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            import traceback
            traceback.print_exc()

    async def _receive_from_google(self, message):
        try:
            resp = json.loads(message)

            if "setupComplete" in resp:
                logger.info("Google Live setup complete - AI Ready")
                self.start_streaming = True
                self.setup_complete = True
                self._save_transcript("SYSTEM", "AI ready")
                # Trigger immediate greeting during preload
                await self._send_initial_greeting()

            # Handle tool calls
            if "toolCall" in resp:
                await self._handle_tool_call(resp["toolCall"])
                return

            if "serverContent" in resp:
                sc = resp["serverContent"]

                # Check if turn is complete (greeting done)
                if sc.get("turnComplete"):
                    self._preload_complete.set()

                if sc.get("interrupted"):
                    if self.plivo_ws:
                        await self.plivo_ws.send_text(json.dumps({"event": "clearAudio", "stream_id": self.stream_id}))

                if "modelTurn" in sc:
                    parts = sc.get("modelTurn", {}).get("parts", [])
                    for p in parts:
                        if p.get("inlineData", {}).get("data"):
                            audio = p["inlineData"]["data"]
                            if self.plivo_ws:
                                # Send directly to Plivo
                                await self.plivo_ws.send_text(json.dumps({
                                    "event": "playAudio",
                                    "media": {"contentType": "audio/x-l16", "sampleRate": 24000, "payload": audio}
                                }))
                            else:
                                # Store for later (during preload)
                                self.preloaded_audio.append(audio)
                        if p.get("text"):
                            self._save_transcript("VISHNU", p["text"])
        except Exception as e:
            logger.error(f"Error: {e}")

    async def handle_plivo_audio(self, audio_b64):
        try:
            if not self.is_active or not self.start_streaming or not self.goog_live_ws:
                return
            chunk = base64.b64decode(audio_b64)
            self.inbuffer.extend(chunk)
            while len(self.inbuffer) >= self.BUFFER_SIZE:
                ac = self.inbuffer[:self.BUFFER_SIZE]
                msg = {"realtime_input": {"media_chunks": [{"mime_type": "audio/pcm;rate=16000", "data": base64.b64encode(bytes(ac)).decode()}]}}
                await self.goog_live_ws.send(json.dumps(msg))
                self.inbuffer = self.inbuffer[self.BUFFER_SIZE:]
        except Exception as e:
            logger.error(f"Audio error: {e}")

    async def handle_plivo_message(self, message):
        event = message.get("event")
        if event == "media":
            payload = message.get("media", {}).get("payload", "")
            if payload:
                await self.handle_plivo_audio(payload)
        elif event == "start":
            self.stream_id = message.get("start", {}).get("streamId", "")
            logger.info(f"Stream started: {self.stream_id}")
        elif event == "stop":
            await self.stop()

    async def stop(self):
        logger.info(f"Stopping session for {self.call_uuid}")
        self.is_active = False
        if self.goog_live_ws:
            try:
                await self.goog_live_ws.close()
            except:
                pass
        if self._session_task:
            self._session_task.cancel()
        self._save_transcript("SYSTEM", "Call ended")

# Session storage
_sessions: Dict[str, PlivoGeminiSession] = {}
_preloading_sessions: Dict[str, PlivoGeminiSession] = {}

async def preload_session(call_uuid: str, caller_phone: str) -> bool:
    """Preload a session while phone is ringing"""
    session = PlivoGeminiSession(call_uuid, caller_phone)
    _preloading_sessions[call_uuid] = session
    success = await session.preload()
    return success

async def create_session(call_uuid: str, caller_phone: str, plivo_ws) -> Optional[PlivoGeminiSession]:
    """Create or retrieve preloaded session"""
    # Check for preloaded session
    if call_uuid in _preloading_sessions:
        session = _preloading_sessions.pop(call_uuid)
        session.caller_phone = caller_phone  # Update phone if needed
        session.attach_plivo_ws(plivo_ws)
        _sessions[call_uuid] = session
        logger.info(f"Using PRELOADED session for {call_uuid}")
        session._save_transcript("SYSTEM", "Call connected (preloaded)")
        return session

    # Fallback: create new session
    logger.info(f"No preloaded session, creating new for {call_uuid}")
    session = PlivoGeminiSession(call_uuid, caller_phone)
    session.plivo_ws = plivo_ws
    session._save_transcript("SYSTEM", "Call started")
    if await session.preload():
        _sessions[call_uuid] = session
        return session
    return None

async def get_session(call_uuid: str) -> Optional[PlivoGeminiSession]:
    return _sessions.get(call_uuid)

async def remove_session(call_uuid: str):
    # Clean up both active and preloading sessions
    if call_uuid in _sessions:
        await _sessions[call_uuid].stop()
        del _sessions[call_uuid]
    if call_uuid in _preloading_sessions:
        await _preloading_sessions[call_uuid].stop()
        del _preloading_sessions[call_uuid]
