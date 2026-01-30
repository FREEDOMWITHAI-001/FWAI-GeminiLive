# Plivo + Google Live API Stream Handler with Preloading
import asyncio
import json
import base64
import wave
import struct
import threading
import queue
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

# Tool definitions for Gemini Live (minimal for lower latency)
TOOL_DECLARATIONS = [
    {
        "name": "send_whatsapp",
        "description": "Send WhatsApp message. Use template_id: 'course_details' (info), 'payment_link' (buy), 'support_contact' (help). Or custom_message if needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "template_id": {
                    "type": "string",
                    "enum": ["course_details", "payment_link", "support_contact"]
                },
                "custom_message": {"type": "string"}
            }
        }
    },
    {
        "name": "end_call",
        "description": "End call IMMEDIATELY when user wants to disconnect. Triggers: 'bye', 'goodbye', 'hang up', 'disconnect', 'end call', 'gotta go', 'need to go', 'talk later', 'not interested', 'stop calling', 'I'm busy', 'no thanks', or ANY indication they want to end. Say brief goodbye THEN call this tool - do NOT continue conversation.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"}
            },
            "required": ["reason"]
        }
    }
]

class PlivoGeminiSession:
    def __init__(self, call_uuid: str, caller_phone: str, prompt: str = None, context: dict = None, webhook_url: str = None):
        self.call_uuid = call_uuid  # Internal UUID
        self.plivo_call_uuid = None  # Plivo's actual call UUID (set later)
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

        # Audio recording - using queue and background thread (non-blocking)
        self.audio_chunks = []  # List of (role, audio_bytes) tuples
        self.recording_enabled = config.enable_transcripts
        self._recording_queue = queue.Queue() if self.recording_enabled else None
        self._recording_thread = None
        if self.recording_enabled:
            self._start_recording_thread()

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

    def _start_recording_thread(self):
        """Start background thread for recording audio"""
        def recording_worker():
            while True:
                try:
                    item = self._recording_queue.get(timeout=1.0)
                    if item is None:  # Shutdown signal
                        break
                    self.audio_chunks.append(item)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Recording thread error: {e}")

        self._recording_thread = threading.Thread(target=recording_worker, daemon=True)
        self._recording_thread.start()
        logger.debug("Recording thread started")

    def _record_audio(self, role: str, audio_bytes: bytes, sample_rate: int = 16000):
        """Record audio chunk for post-call transcription (non-blocking)"""
        if not self.recording_enabled or not self._recording_queue:
            return
        # Put in queue - non-blocking, doesn't affect call latency
        try:
            self._recording_queue.put_nowait((role, audio_bytes, sample_rate))
        except queue.Full:
            pass  # Drop frame if queue is full (shouldn't happen)

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

    def _transcribe_recording_sync(self, recording_file: Path, call_uuid: str):
        """Transcribe recording using Whisper (runs in background thread)

        Note: Speaker-labeled transcript is already captured during the live call
        via _save_transcript(). This just adds a raw Whisper transcription for reference.
        """
        try:
            import whisper
            logger.info(f"Starting background transcription for {call_uuid}")
            model = whisper.load_model("tiny")  # Use tiny for faster transcription
            result = model.transcribe(str(recording_file))
            transcript_text = result["text"].strip()

            # Save raw Whisper transcript (speaker labels already captured during live call)
            transcript_file = Path(__file__).parent.parent.parent / "transcripts" / f"{call_uuid}.txt"
            with open(transcript_file, "a") as f:
                f.write(f"\n--- RAW WHISPER TRANSCRIPTION ---\n")
                f.write(f"{transcript_text}\n")

            logger.info(f"Transcription complete for {call_uuid}")

            # Compress recording file to save space (WAV -> MP3)
            try:
                import subprocess
                mp3_file = recording_file.with_suffix('.mp3')
                result = subprocess.run(
                    ['ffmpeg', '-y', '-i', str(recording_file), '-b:a', '32k', str(mp3_file)],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    recording_file.unlink()  # Delete original WAV
                    logger.info(f"Recording compressed: {mp3_file}")
                else:
                    logger.warning(f"Compression failed, keeping WAV: {result.stderr}")
            except Exception as e:
                logger.warning(f"Could not compress recording: {e}")

        except ImportError:
            logger.warning("Whisper not installed. Run: pip install openai-whisper")
        except Exception as e:
            logger.error(f"Transcription error: {e}")


    async def preload(self):
        """Preload the Gemini session while phone is ringing"""
        try:
            logger.info(f"PRELOADING Gemini session for call {self.call_uuid}")
            self.is_active = True
            self._session_task = asyncio.create_task(self._run_google_live_session())
            # Wait for setup to complete (with timeout - 5s max)
            try:
                await asyncio.wait_for(self._preload_complete.wait(), timeout=5.0)
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
        preload_count = len(self.preloaded_audio)
        logger.info(f"Plivo WS attached for {self.call_uuid} - PRELOAD: {preload_count} audio chunks ready")
        # Send any preloaded audio immediately
        if self.preloaded_audio:
            logger.info(f"PRELOAD SUCCESS: Sending {preload_count} chunks immediately (no wait for AI)")
            asyncio.create_task(self._send_preloaded_audio())
        else:
            logger.warning(f"PRELOAD MISS: No preloaded audio - AI greeting will have latency")
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
        # Concise accent instruction (shorter = faster responses)
        accent_instruction = """VOICE: Indian English accent (Bangalore professional). Use "Actually...", "Basically...", "Isn't it?" naturally. English only, no Hindi. Warm and professional tone.

"""
        # Combine: accent first, then main prompt
        full_prompt = accent_instruction + self.prompt

        msg = {
            "setup": {
                "model": "models/gemini-2.5-flash-native-audio-preview-09-2025",
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
                logger.info(f"END CALL requested: {reason} - hanging up in 0.3s")
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
                # Reduced from 1.0s to 0.3s - audio is already queued in Plivo buffer
                asyncio.create_task(self._hangup_call_delayed(0.3))
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
        """Hang up the call after a short delay (audio is queued in Plivo buffer)"""
        try:
            await asyncio.sleep(delay)

            # Use Plivo's UUID if available, otherwise fall back to internal UUID
            hangup_uuid = self.plivo_call_uuid or self.call_uuid
            logger.info(f"Hanging up call {self.call_uuid} using UUID: {hangup_uuid} (plivo_uuid={self.plivo_call_uuid})")

            # Use Plivo REST API directly with httpx (async)
            import httpx
            import base64

            auth_string = f"{config.plivo_auth_id}:{config.plivo_auth_token}"
            auth_b64 = base64.b64encode(auth_string.encode()).decode()

            url = f"https://api.plivo.com/v1/Account/{config.plivo_auth_id}/Call/{hangup_uuid}/"

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

                # Debug: log all serverContent keys to find user transcription field
                sc_keys = list(sc.keys())
                if sc_keys != ['modelTurn'] and sc_keys != ['turnComplete']:
                    logger.info(f"serverContent keys: {sc_keys}")

                # Check if turn is complete (greeting done)
                if sc.get("turnComplete"):
                    logger.info(f"Turn complete - preloaded {len(self.preloaded_audio)} audio chunks, plivo_ws={'connected' if self.plivo_ws else 'None'}")
                    self._preload_complete.set()
                    self.greeting_audio_complete = True

                if sc.get("interrupted"):
                    logger.info("AI was interrupted by user")
                    if self.plivo_ws:
                        await self.plivo_ws.send_text(json.dumps({"event": "clearAudio", "stream_id": self.stream_id}))

                # Capture user speech transcription from Gemini
                if "inputTranscript" in sc:
                    user_text = sc["inputTranscript"]
                    if user_text and user_text.strip():
                        logger.info(f"USER said: {user_text}")
                        self._save_transcript("USER", user_text.strip())

                if "modelTurn" in sc:
                    parts = sc.get("modelTurn", {}).get("parts", [])
                    for p in parts:
                        if p.get("inlineData", {}).get("data"):
                            audio = p["inlineData"]["data"]
                            audio_bytes = base64.b64decode(audio)
                            # Record AI audio (24kHz)
                            self._record_audio("AI", audio_bytes, 24000)
                            logger.info(f"AI AUDIO received: {len(audio_bytes)} bytes, greeting_complete={self.greeting_audio_complete}")

                            # During preload (no plivo_ws yet), always store audio
                            # This fixes race condition where turnComplete arrives before all audio
                            if not self.plivo_ws:
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
                            ai_text = p["text"].strip()
                            logger.info(f"AI TEXT: {ai_text[:100]}...")
                            # Only save actual speech, not thinking/planning text
                            # Skip text that looks like internal reasoning (markdown, planning phrases)
                            is_thinking = (
                                ai_text.startswith("**") or
                                ai_text.startswith("I've registered") or
                                ai_text.startswith("I'll ") or
                                "My first step" in ai_text or
                                "I'll be keeping" in ai_text or
                                "maintaining the" in ai_text or
                                "waiting for their response" in ai_text
                            )
                            if ai_text and not is_thinking and len(ai_text) > 3:
                                self._save_transcript("AGENT", ai_text)
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
        # Guard against double-stop
        if not self.is_active:
            logger.debug(f"Session {self.call_uuid} already stopped, skipping")
            return

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

        # Stop recording thread
        if self._recording_queue:
            self._recording_queue.put(None)  # Shutdown signal
        if self._recording_thread:
            self._recording_thread.join(timeout=2.0)

        # Process recording and transcription in COMPLETELY SEPARATE background thread
        # This does NOT block call ending - call ends immediately
        # Webhook is called AFTER transcription is complete
        self._start_post_call_processing(duration)

    def _start_post_call_processing(self, duration: float):
        """Run all post-call processing (save, transcribe, webhook) in background thread"""
        def process_in_background():
            try:
                # Step 1: Save recording
                recording_file = self._save_recording()

                # Step 2: Transcribe with Whisper (only if enabled - RAM heavy)
                if recording_file and config.enable_whisper:
                    self._transcribe_recording_sync(recording_file, self.call_uuid)
                elif recording_file:
                    logger.info(f"Whisper disabled - recording saved without transcription: {recording_file}")

                # Step 3: Call webhook AFTER transcription is done
                if self.webhook_url:
                    import asyncio
                    # Create new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self._call_webhook(duration))
                    finally:
                        loop.close()

                logger.info(f"Post-call processing complete for {self.call_uuid}")
            except Exception as e:
                logger.error(f"Post-call processing error: {e}")

        # Start background thread - call ends immediately, this runs separately
        processing_thread = threading.Thread(target=process_in_background, daemon=True)
        processing_thread.start()
        logger.info(f"Post-call processing started in background for {self.call_uuid}")

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

def set_plivo_uuid(internal_uuid: str, plivo_uuid: str):
    """Set the Plivo UUID on a preloaded session for proper hangup"""
    session = _preloading_sessions.get(internal_uuid) or _sessions.get(internal_uuid)
    if session:
        session.plivo_call_uuid = plivo_uuid
        logger.info(f"Set Plivo UUID {plivo_uuid} on session {internal_uuid}")
    else:
        logger.error(f"CRITICAL: Could not find session {internal_uuid} to set Plivo UUID {plivo_uuid}. Call hangup will fail!")
        logger.error(f"  _preloading_sessions keys: {list(_preloading_sessions.keys())}")
        logger.error(f"  _sessions keys: {list(_sessions.keys())}")

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
