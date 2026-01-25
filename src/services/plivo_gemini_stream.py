# Plivo + Google Live API Stream Handler with Preloading
import asyncio
import json
import base64
import wave
import struct
import threading
from typing import Dict, Optional
from loguru import logger
from datetime import datetime
from pathlib import Path
import websockets
from src.core.config import config
from src.tools import execute_tool

# Recording directory
RECORDINGS_DIR = Path(__file__).parent.parent.parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

# Load default FWAI prompt (used when no prompt passed from API)
def load_default_prompt():
    try:
        prompts_file = Path(__file__).parent.parent.parent / "prompts.json"
        with open(prompts_file) as f:
            prompts = json.load(f)
            return prompts.get("FWAI_Core", {}).get("prompt", "You are a helpful AI assistant.")
    except:
        return "You are a helpful AI assistant."

DEFAULT_PROMPT = load_default_prompt()

# Tool definitions for Gemini Live
TOOL_DECLARATIONS = [
    {
        "name": "send_whatsapp",
        "description": """Send a beautifully formatted WhatsApp message to the caller.
ALWAYS use a template when possible - they look professional and contain all needed info.

Choose template based on what user is asking:

📚 template_id='course_details' - USE WHEN:
   • User asks "what's included?", "tell me more", "what will I learn?"
   • User wants to know about features, curriculum, benefits
   • User asks about pricing or what they get

💳 template_id='payment_link' - USE WHEN:
   • User says "I want to join", "how do I pay?", "I'm ready"
   • User asks for payment link or enrollment link
   • User confirms they want to purchase

🆘 template_id='support_contact' - USE WHEN:
   • User has a problem, complaint, or issue
   • User says "I need help", "something's not working"
   • User is frustrated or needs customer support

Only use custom_message for very specific requests that don't fit any template.""",
        "parameters": {
            "type": "object",
            "properties": {
                "template_id": {
                    "type": "string",
                    "description": "Template ID: 'course_details' (info/features), 'payment_link' (ready to buy), or 'support_contact' (help/issues)",
                    "enum": ["course_details", "payment_link", "support_contact"]
                },
                "custom_message": {
                    "type": "string",
                    "description": "Custom message - ONLY use if no template fits the situation"
                }
            },
            "required": []
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
    },
    {
        "name": "end_call",
        "description": """End the phone call gracefully. USE THIS TOOL WHEN:
- User says 'bye', 'bye bye', 'goodbye', 'talk later', 'gotta go', 'thanks bye'
- User indicates they want to end the conversation
- You have said your goodbye and the conversation is naturally ending
- The call time limit has been reached

IMPORTANT: Always say a warm goodbye BEFORE calling this tool, then call it to actually hang up.""",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Reason for ending call (e.g., 'user said goodbye', 'conversation complete', 'time limit')"
                }
            },
            "required": ["reason"]
        }
    }
]

class PlivoGeminiSession:
    def __init__(self, call_uuid: str, caller_phone: str, prompt: str = None, context: dict = None, webhook_url: str = None):
        self.call_uuid = call_uuid
        self.caller_phone = caller_phone
        self.prompt = prompt or DEFAULT_PROMPT  # Use passed prompt or default
        self.context = context or {}  # Context for templates (customer_name, course_name, etc.)
        self.webhook_url = webhook_url  # URL to call when call ends (for n8n integration)
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

        # Audio recording - single combined file (for post-call transcription)
        self.audio_chunks = []  # List of (role, audio_bytes) tuples
        self.recording_enabled = config.enable_transcripts

        # Flag to prevent double greeting
        self.greeting_audio_complete = False

        # Call duration management (8 minute max)
        self.call_start_time = None
        self.max_call_duration = 8 * 60  # 8 minutes in seconds
        self._timeout_task = None
        self._closing_call = False  # Flag to indicate we're closing the call

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

    def _record_audio(self, role: str, audio_bytes: bytes, sample_rate: int = 16000):
        """Record audio chunk for post-call transcription"""
        if not self.recording_enabled:
            return
        # Store with metadata (role and sample_rate for resampling later)
        self.audio_chunks.append((role, audio_bytes, sample_rate))
        if len(self.audio_chunks) % 50 == 1:  # Log every 50 chunks
            logger.debug(f"Recording: {len(self.audio_chunks)} chunks ({role})")

    def _resample_24k_to_16k(self, audio_bytes: bytes) -> bytes:
        """Resample 24kHz audio to 16kHz (simple linear interpolation)"""
        # Convert bytes to samples (16-bit signed)
        samples_24k = struct.unpack(f'<{len(audio_bytes)//2}h', audio_bytes)
        # Resample 24kHz -> 16kHz (ratio 2:3)
        samples_16k = []
        for i in range(0, len(samples_24k) * 2 // 3):
            idx = i * 3 / 2
            idx_floor = int(idx)
            if idx_floor + 1 < len(samples_24k):
                frac = idx - idx_floor
                sample = int(samples_24k[idx_floor] * (1 - frac) + samples_24k[idx_floor + 1] * frac)
            else:
                sample = samples_24k[idx_floor] if idx_floor < len(samples_24k) else 0
            samples_16k.append(max(-32768, min(32767, sample)))
        return struct.pack(f'<{len(samples_16k)}h', *samples_16k)

    def _save_recording(self):
        """Save combined recording as WAV file"""
        logger.info(f"Saving recording: enabled={self.recording_enabled}, chunks={len(self.audio_chunks)}")
        if not self.recording_enabled or not self.audio_chunks:
            logger.warning(f"Skipping recording: enabled={self.recording_enabled}, chunks={len(self.audio_chunks)}")
            return None
        try:
            recording_file = RECORDINGS_DIR / f"{self.call_uuid}.wav"
            combined_audio = bytearray()

            for role, audio_bytes, sample_rate in self.audio_chunks:
                if sample_rate == 24000:
                    # Resample AI audio from 24kHz to 16kHz
                    audio_bytes = self._resample_24k_to_16k(audio_bytes)
                combined_audio.extend(audio_bytes)

            # Write WAV file (16kHz, mono, 16-bit)
            with wave.open(str(recording_file), 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(16000)
                wav.writeframes(bytes(combined_audio))

            logger.info(f"Recording saved: {recording_file} ({len(combined_audio)} bytes)")
            return recording_file
        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            return None

    def _transcribe_recording(self, recording_file: Path):
        """Transcribe recording using Whisper (runs in background thread)"""
        def transcribe():
            try:
                import whisper
                logger.info(f"Starting transcription for {self.call_uuid}")
                model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
                result = model.transcribe(str(recording_file))
                transcript_text = result["text"]

                # Save to transcript file
                transcript_file = Path(__file__).parent.parent.parent / "transcripts" / f"{self.call_uuid}.txt"
                with open(transcript_file, "a") as f:
                    f.write(f"\n--- FULL TRANSCRIPT ---\n{transcript_text}\n")

                logger.info(f"Transcription complete for {self.call_uuid}")

                # Delete recording file after successful transcription
                try:
                    recording_file.unlink()
                    logger.info(f"Recording deleted: {recording_file}")
                except Exception as e:
                    logger.warning(f"Could not delete recording: {e}")

            except ImportError:
                logger.warning("Whisper not installed. Run: pip install openai-whisper")
            except Exception as e:
                logger.error(f"Transcription error: {e}")

        # Run in background thread to not block
        thread = threading.Thread(target=transcribe, daemon=True)
        thread.start()

    async def preload(self):
        """Preload the Gemini session while phone is ringing"""
        try:
            logger.info(f"PRELOADING Gemini session for call {self.call_uuid}")
            self.is_active = True
            self._session_task = asyncio.create_task(self._run_google_live_session())
            # Wait for setup to complete (with timeout - increased to 15s)
            try:
                await asyncio.wait_for(self._preload_complete.wait(), timeout=15.0)
                logger.info(f"PRELOAD COMPLETE for {self.call_uuid} - AI ready to speak! ({len(self.preloaded_audio)} audio chunks)")
            except asyncio.TimeoutError:
                logger.warning(f"Preload timeout for {self.call_uuid} - continuing anyway with {len(self.preloaded_audio)} chunks")
            return True
        except Exception as e:
            logger.error(f"Failed to preload session: {e}")
            return False

    def attach_plivo_ws(self, plivo_ws):
        """Attach Plivo WebSocket when user answers"""
        self.plivo_ws = plivo_ws
        self.call_start_time = datetime.now()
        logger.info(f"Plivo WS attached for {self.call_uuid}")
        # Send any preloaded audio immediately
        if self.preloaded_audio:
            asyncio.create_task(self._send_preloaded_audio())
        # Start call duration timer
        self._timeout_task = asyncio.create_task(self._monitor_call_duration())

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

    async def _monitor_call_duration(self):
        """Monitor call duration and trigger wrap-up at 8 minutes"""
        try:
            # Wait until 7:30 (give 30 seconds for wrap-up)
            wrap_up_time = self.max_call_duration - 30
            await asyncio.sleep(wrap_up_time)

            if self.is_active and not self._closing_call:
                logger.info(f"Call {self.call_uuid} reaching 8 min limit - triggering wrap-up")
                self._closing_call = True
                await self._send_wrap_up_message()

                # Wait another 30 seconds then force end
                await asyncio.sleep(30)
                if self.is_active:
                    logger.info(f"Call {self.call_uuid} reached max duration - ending call")
                    await self.stop()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in call duration monitor: {e}")

    async def _send_wrap_up_message(self):
        """Send a message to AI to wrap up the call"""
        if not self.goog_live_ws:
            return
        try:
            msg = {
                "client_content": {
                    "turns": [{
                        "role": "user",
                        "parts": [{"text": "[SYSTEM: Call time limit reached. Please politely wrap up the conversation now. Say a warm goodbye and end the call gracefully.]"}]
                    }],
                    "turn_complete": True
                }
            }
            await self.goog_live_ws.send(json.dumps(msg))
            logger.info("Sent wrap-up message to AI")
            self._save_transcript("SYSTEM", "Call time limit - wrapping up")
        except Exception as e:
            logger.error(f"Error sending wrap-up message: {e}")

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
        # Accent/speech instructions FIRST (before main prompt)
        accent_instruction = """MANDATORY VOICE INSTRUCTIONS - FOLLOW STRICTLY:

1. ACCENT: You MUST speak with a strong Indian English accent throughout the ENTIRE conversation.
   - Sound exactly like an educated Indian professional from Bangalore or Mumbai
   - Use Indian English pronunciation patterns (e.g., pronounce "schedule" as "shed-yool", "process" as "pro-sess")
   - Use Indian English intonation - rising tone at end of statements
   - Use syllable-timed rhythm (equal stress on each syllable)

2. LANGUAGE: Speak ONLY in English.
   - Do NOT use ANY Hindi words (no "achha", "haan", "theek hai", "na", "ji")
   - Use only English words and phrases

3. SPEECH PATTERNS: Use Indian English phrases naturally:
   - "Actually, what I was saying is..."
   - "Basically, the thing is..."
   - "You see, the point is..."
   - "Isn't it?", "No?", "Right?" at end of sentences
   - "I'll just tell you one thing..."
   - "What happens is..."

4. TONE: Be warm, friendly, professional, and enthusiastic like an Indian sales professional.

IMPORTANT: Maintain this Indian English accent consistently for EVERY response. Never switch to American or British accent.

"""
        # Combine: accent first, then main prompt
        full_prompt = accent_instruction + self.prompt

        msg = {
            "setup": {
                "model": "models/gemini-2.5-flash-native-audio-preview-12-2025",
                "generation_config": {
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": "Puck"
                            }
                        }
                    }
                },
                "system_instruction": {"parts": [{"text": full_prompt}]},
                "tools": [{"function_declarations": TOOL_DECLARATIONS}]
            }
        }
        await self.goog_live_ws.send(json.dumps(msg))
        logger.info("Sent session setup with natural speech prompt")

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
        """Execute tool and send response back to Gemini - gracefully handles errors"""
        func_calls = tool_call.get("functionCalls", [])
        for fc in func_calls:
            tool_name = fc.get("name")
            tool_args = fc.get("args", {})
            call_id = fc.get("id")

            logger.info(f"TOOL CALL: {tool_name} with args: {tool_args}")
            self._save_transcript("TOOL", f"{tool_name}: {tool_args}")

            # Handle end_call tool specially
            if tool_name == "end_call":
                reason = tool_args.get("reason", "conversation ended")
                logger.info(f"END CALL requested: {reason}")
                self._save_transcript("SYSTEM", f"Call ending: {reason}")

                # Send success response first
                try:
                    tool_response = {
                        "tool_response": {
                            "function_responses": [{
                                "id": call_id,
                                "name": tool_name,
                                "response": {"success": True, "message": "Call will be ended"}
                            }]
                        }
                    }
                    await self.goog_live_ws.send(json.dumps(tool_response))
                except:
                    pass

                # Hang up the call after a short delay (let goodbye audio play)
                asyncio.create_task(self._hangup_call_delayed(3.0))
                return

            # Execute the tool with context for templates - graceful error handling
            try:
                result = await execute_tool(tool_name, self.caller_phone, context=self.context, **tool_args)
                success = result.get("success", False)
                message = result.get("message", "Tool executed")
            except Exception as e:
                logger.error(f"Tool execution error for {tool_name}: {e}")
                success = False
                message = f"Tool temporarily unavailable, but conversation can continue"

            logger.info(f"TOOL RESULT: success={success}, message={message}")
            self._save_transcript("TOOL_RESULT", f"{tool_name}: {'success' if success else 'failed'}")

            # Always send tool response back to Gemini so conversation continues
            try:
                tool_response = {
                    "tool_response": {
                        "function_responses": [{
                            "id": call_id,
                            "name": tool_name,
                            "response": {
                                "success": success,
                                "message": message
                            }
                        }]
                    }
                }
                await self.goog_live_ws.send(json.dumps(tool_response))
                logger.info(f"Sent tool response for {tool_name}")
            except Exception as e:
                logger.error(f"Error sending tool response: {e} - continuing conversation")

    async def _hangup_call_delayed(self, delay: float):
        """Hang up the call after a delay to let goodbye audio play"""
        try:
            await asyncio.sleep(delay)
            logger.info(f"Hanging up call {self.call_uuid}")

            # Use Plivo REST API directly with httpx (async)
            import httpx
            import base64

            auth_string = f"{config.plivo_auth_id}:{config.plivo_auth_token}"
            auth_b64 = base64.b64encode(auth_string.encode()).decode()

            url = f"https://api.plivo.com/v1/Account/{config.plivo_auth_id}/Call/{self.call_uuid}/"

            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    url,
                    headers={"Authorization": f"Basic {auth_b64}"}
                )
                logger.info(f"Plivo hangup response: {response.status_code}")

                if response.status_code in [204, 200]:
                    logger.info(f"Call {self.call_uuid} hung up successfully via Plivo API")
                else:
                    logger.error(f"Plivo hangup failed: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error hanging up call {self.call_uuid}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Always stop the session
            if self.is_active:
                await self.stop()

    async def _receive_from_google(self, message):
        try:
            resp = json.loads(message)

            # Log all Gemini responses for debugging
            resp_keys = list(resp.keys())
            if resp_keys != ['serverContent']:  # Don't log every content message
                logger.debug(f"Gemini response keys: {resp_keys}")

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
                    logger.info(f"Turn complete - greeting_audio_complete was {self.greeting_audio_complete}")
                    self._preload_complete.set()
                    self.greeting_audio_complete = True

                if sc.get("interrupted"):
                    logger.info("AI was interrupted by user")
                    if self.plivo_ws:
                        await self.plivo_ws.send_text(json.dumps({"event": "clearAudio", "stream_id": self.stream_id}))

                if "modelTurn" in sc:
                    parts = sc.get("modelTurn", {}).get("parts", [])
                    for p in parts:
                        if p.get("inlineData", {}).get("data"):
                            audio = p["inlineData"]["data"]
                            audio_bytes = base64.b64decode(audio)
                            # Record AI audio (24kHz)
                            self._record_audio("AI", audio_bytes, 24000)
                            logger.info(f"AI AUDIO received: {len(audio_bytes)} bytes, greeting_complete={self.greeting_audio_complete}")

                            # During greeting preload, always store (don't send twice)
                            if not self.greeting_audio_complete:
                                self.preloaded_audio.append(audio)
                                logger.debug(f"Stored preload audio chunk #{len(self.preloaded_audio)}")
                            elif self.plivo_ws:
                                # After greeting done, send directly to Plivo
                                try:
                                    await self.plivo_ws.send_text(json.dumps({
                                        "event": "playAudio",
                                        "media": {"contentType": "audio/x-l16", "sampleRate": 24000, "payload": audio}
                                    }))
                                    logger.debug(f"Sent AI audio to Plivo: {len(audio_bytes)} bytes")
                                except Exception as plivo_err:
                                    logger.error(f"Error sending audio to Plivo: {plivo_err} - continuing")
                        if p.get("text"):
                            logger.info(f"AI TEXT: {p['text'][:100]}...")
                            self._save_transcript("VISHNU", p["text"])
        except Exception as e:
            logger.error(f"Error processing Google message: {e} - continuing session")

    async def handle_plivo_audio(self, audio_b64):
        """Handle incoming audio from Plivo - graceful error handling"""
        try:
            if not self.is_active or not self.start_streaming:
                logger.debug(f"Skipping audio: active={self.is_active}, streaming={self.start_streaming}")
                return
            if not self.goog_live_ws:
                logger.warning("Google WS not connected, skipping audio")
                return
            chunk = base64.b64decode(audio_b64)
            # Record user audio (16kHz)
            self._record_audio("USER", chunk, 16000)
            self.inbuffer.extend(chunk)
            chunks_sent = 0
            while len(self.inbuffer) >= self.BUFFER_SIZE:
                ac = self.inbuffer[:self.BUFFER_SIZE]
                msg = {"realtime_input": {"media_chunks": [{"mime_type": "audio/pcm;rate=16000", "data": base64.b64encode(bytes(ac)).decode()}]}}
                try:
                    await self.goog_live_ws.send(json.dumps(msg))
                    chunks_sent += 1
                except Exception as send_err:
                    logger.error(f"Error sending audio to Google: {send_err} - continuing")
                self.inbuffer = self.inbuffer[self.BUFFER_SIZE:]
            if chunks_sent > 0 and len(self.audio_chunks) % 100 == 0:
                logger.info(f"Sent {chunks_sent} audio chunks to Gemini (total user chunks: {len(self.audio_chunks)})")
        except Exception as e:
            logger.error(f"Audio processing error: {e} - continuing session")

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

        # Cancel timeout task
        if self._timeout_task:
            self._timeout_task.cancel()

        # Calculate call duration
        duration = 0
        if self.call_start_time:
            duration = (datetime.now() - self.call_start_time).total_seconds()
            logger.info(f"Call {self.call_uuid} duration: {duration:.1f} seconds")
            self._save_transcript("SYSTEM", f"Call duration: {duration:.1f}s")

        if self.goog_live_ws:
            try:
                await self.goog_live_ws.close()
            except:
                pass
        if self._session_task:
            self._session_task.cancel()
        self._save_transcript("SYSTEM", "Call ended")

        # Save recording and transcribe in background
        recording_file = self._save_recording()
        if recording_file:
            self._transcribe_recording(recording_file)

        # Call webhook URL to notify n8n that call ended
        if self.webhook_url:
            asyncio.create_task(self._call_webhook(duration))

    async def _call_webhook(self, duration: float):
        """Call webhook URL to notify n8n that call ended"""
        try:
            import httpx

            # Read transcript file if it exists
            transcript = ""
            try:
                transcript_file = Path(__file__).parent.parent.parent / "transcripts" / f"{self.call_uuid}.txt"
                if transcript_file.exists():
                    transcript = transcript_file.read_text()
            except:
                pass

            payload = {
                "event": "call_ended",
                "call_uuid": self.call_uuid,
                "caller_phone": self.caller_phone,
                "duration_seconds": round(duration, 1),
                "timestamp": datetime.now().isoformat(),
                "transcript": transcript
            }

            logger.info(f"Calling webhook: {self.webhook_url}")
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self.webhook_url, json=payload)
                logger.info(f"Webhook response: {resp.status_code}")
        except Exception as e:
            logger.error(f"Error calling webhook: {e}")


# Session storage
_sessions: Dict[str, PlivoGeminiSession] = {}
_preloading_sessions: Dict[str, PlivoGeminiSession] = {}

async def preload_session(call_uuid: str, caller_phone: str, prompt: str = None, context: dict = None, webhook_url: str = None) -> bool:
    """Preload a session while phone is ringing"""
    session = PlivoGeminiSession(call_uuid, caller_phone, prompt=prompt, context=context, webhook_url=webhook_url)
    _preloading_sessions[call_uuid] = session
    success = await session.preload()
    return success

async def create_session(call_uuid: str, caller_phone: str, plivo_ws, prompt: str = None, context: dict = None, webhook_url: str = None) -> Optional[PlivoGeminiSession]:
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
    session = PlivoGeminiSession(call_uuid, caller_phone, prompt=prompt, context=context, webhook_url=webhook_url)
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
