# Plivo + Google Live API Stream Handler with Preloading
import asyncio
import json
import base64
import wave
import struct
import threading
import queue
import time
from typing import Dict, Optional
from loguru import logger
from datetime import datetime
from pathlib import Path
import websockets
from src.core.config import config
from src.tools import execute_tool

# Latency threshold - only log if slower than this (ms)
LATENCY_THRESHOLD_MS = 500

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


def detect_voice_from_prompt(prompt: str) -> str:
    """Detect voice based on prompt content. Returns 'Kore' for female, 'Puck' for default."""
    if not prompt:
        return "Puck"
    prompt_lower = prompt.lower()
    # Check for female indicators in the prompt
    female_indicators = [
        "female", "woman", "girl", "she ", "her ", "lady",
        "mousumi", "priya", "anjali", "divya", "neha", "pooja", "shreya"  # Common Indian female names
    ]
    for indicator in female_indicators:
        if indicator in prompt_lower:
            logger.info(f"Detected female voice indicator '{indicator}' in prompt - using Kore voice")
            return "Kore"
    return "Puck"

# Tool definitions for Gemini Live (minimal for lower latency)
# NOTE: WhatsApp messaging disabled during calls to reduce latency/interruptions
TOOL_DECLARATIONS = [
    {
        "name": "end_call",
        "description": "End call. MUST call after BOTH you AND user have said goodbye. Wait for mutual farewell before calling this.",
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

        # Goodbye tracking - call ends only when both parties say goodbye
        self.user_said_goodbye = False
        self.agent_said_goodbye = False

        # Latency tracking - only logs if > threshold
        self._last_user_speech_time = None

        # Silence monitoring - 3 second SLA
        self._silence_monitor_task = None
        self._silence_sla_seconds = 3.0  # Must respond within 3 seconds
        self._last_ai_audio_time = None  # Track when AI last sent audio
        self._current_turn_audio_chunks = 0  # Track audio chunks in current turn
        self._empty_turn_nudge_count = 0  # Track consecutive empty turns
        self._turn_start_time = None  # Track when current turn started (for latency logging)
        self._turn_count = 0  # Count turns for latency tracking

        # Audio buffer for reconnection (store audio if Google WS drops briefly)
        self._reconnect_audio_buffer = []
        self._max_reconnect_buffer = 150  # Increased buffer (~3 seconds) for better reconnection

        # Conversation history for context restoration on reconnect (keep small to reduce latency)
        self._conversation_history = []  # List of {"role": "user"/"model", "text": "..."}
        self._max_history_size = 10  # Keep only last 10 messages to prevent latency buildup
        self._is_first_connection = True  # Track if this is first connect or reconnect

        # Google session refresh timer (10-min limit, refresh at 9 min)
        self._google_session_start = None
        self._session_refresh_task = None
        self.GOOGLE_SESSION_LIMIT = 9 * 60  # Refresh at 9 minutes (before 10-min disconnect)

    def _is_goodbye_message(self, text: str) -> bool:
        """Detect if agent is saying goodbye - triggers auto call end"""
        text_lower = text.lower()
        # Comprehensive goodbye/farewell/ending detection
        goodbye_phrases = [
            # Direct goodbyes
            'bye', 'goodbye', 'good bye', 'bye bye', 'buh bye',
            # Take care variants
            'take care', 'take it easy', 'be well', 'stay safe',
            # Talk later variants
            'talk later', 'talk soon', 'talk to you', 'speak soon', 'speak later',
            'catch you later', 'catch up later', 'chat later', 'chat soon',
            # Day wishes
            'have a great', 'have a nice', 'have a good', 'have a wonderful',
            'enjoy your', 'all the best', 'best of luck', 'good luck',
            # Thanks for calling
            'thanks for calling', 'thank you for calling', 'thanks for your time',
            'thank you for your time', 'appreciate your time', 'appreciate you calling',
            # Nice talking
            'nice talking', 'great talking', 'good talking', 'lovely talking',
            'nice chatting', 'great chatting', 'pleasure talking', 'pleasure speaking',
            'enjoyed talking', 'enjoyed our', 'was great speaking',
            # See you
            'see you', 'see ya', 'cya', 'until next time', 'till next time',
            # Ending indicators
            'signing off', 'thats all', "that's all", 'nothing else',
            'we are done', "we're done", 'call ended', 'ending the call'
        ]
        for phrase in goodbye_phrases:
            if phrase in text_lower:
                return True
        return False

    def _check_mutual_goodbye(self):
        """End call when agent says goodbye (don't wait too long for user)"""
        if self.agent_said_goodbye and not self._closing_call:
            if self.user_said_goodbye:
                logger.info(f"MUTUAL GOODBYE - ending call immediately")
                self._closing_call = True
                asyncio.create_task(self._hangup_call_delayed(0.5))  # Quick end
            else:
                # Agent said goodbye but user hasn't - start short timeout
                logger.info(f"AGENT GOODBYE - waiting 3s for user response")
                asyncio.create_task(self._quick_goodbye_timeout(3.0))

    async def _quick_goodbye_timeout(self, timeout: float):
        """Quick timeout after agent says goodbye - don't wait too long"""
        try:
            await asyncio.sleep(timeout)
            if not self._closing_call and self.agent_said_goodbye:
                logger.info(f"Quick goodbye timeout - ending call")
                self._closing_call = True
                await self._hangup_call_delayed(0.5)
        except asyncio.CancelledError:
            pass

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
            logger.debug(f"TRANSCRIPT [{role}]: {text}")
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
        # Put in queue with timestamp - non-blocking, doesn't affect call latency
        try:
            timestamp = time.time()  # ~0.001ms, negligible
            self._recording_queue.put_nowait((role, audio_bytes, sample_rate, timestamp))
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
        """Save separate user/agent WAV files with proper timing (silence for gaps)"""
        logger.info(f"Saving recording: enabled={self.recording_enabled}, chunks={len(self.audio_chunks)}")
        if not self.recording_enabled or not self.audio_chunks:
            logger.warning(f"Skipping recording: enabled={self.recording_enabled}, chunks={len(self.audio_chunks)}")
            return None
        try:
            import subprocess

            # Get call start time (earliest timestamp)
            call_start = min(chunk[3] for chunk in self.audio_chunks)

            # Build audio with proper timing (insert silence for gaps)
            user_audio = bytearray()
            agent_audio = bytearray()
            user_last_end = call_start
            agent_last_end = call_start
            SAMPLE_RATE = 16000
            BYTES_PER_SAMPLE = 2  # 16-bit audio

            for chunk in self.audio_chunks:
                role, audio_bytes, sample_rate, timestamp = chunk
                if sample_rate == 24000:
                    audio_bytes = self._resample_24k_to_16k(audio_bytes)

                # Calculate audio duration in seconds
                audio_duration = len(audio_bytes) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

                if role == "USER":
                    # Insert silence for gap since last user audio
                    gap = timestamp - user_last_end
                    if gap > 0.05:  # Only add silence for gaps > 50ms
                        silence_samples = int(gap * SAMPLE_RATE)
                        user_audio.extend(b'\x00' * (silence_samples * BYTES_PER_SAMPLE))
                    user_audio.extend(audio_bytes)
                    user_last_end = timestamp + audio_duration
                else:  # AI/AGENT
                    # Insert silence for gap since last agent audio
                    gap = timestamp - agent_last_end
                    if gap > 0.05:
                        silence_samples = int(gap * SAMPLE_RATE)
                        agent_audio.extend(b'\x00' * (silence_samples * BYTES_PER_SAMPLE))
                    agent_audio.extend(audio_bytes)
                    agent_last_end = timestamp + audio_duration

            # Start times are now call_start for both (silence padding handles alignment)
            user_start_time = call_start
            agent_start_time = call_start

            # Save user audio
            user_wav = RECORDINGS_DIR / f"{self.call_uuid}_user.wav"
            if user_audio:
                with wave.open(str(user_wav), 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(bytes(user_audio))

            # Save agent audio
            agent_wav = RECORDINGS_DIR / f"{self.call_uuid}_agent.wav"
            if agent_audio:
                with wave.open(str(agent_wav), 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(bytes(agent_audio))

            logger.info(f"Recordings saved: user={len(user_audio)}bytes, agent={len(agent_audio)}bytes")

            # Return paths and start times for transcription
            return {
                "user_wav": user_wav if user_audio else None,
                "agent_wav": agent_wav if agent_audio else None,
                "user_start": user_start_time,
                "agent_start": agent_start_time
            }
        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            return None

    def _transcribe_recording_sync(self, recording_info: dict, call_uuid: str):
        """Transcribe user/agent recordings separately and merge by timestamp"""
        try:
            import whisper
            logger.info(f"Starting Whisper transcription for {call_uuid}")
            model = whisper.load_model("medium")  # Better accuracy, needs 5GB RAM

            segments = []  # List of (absolute_time, role, text)

            # Transcribe user audio
            if recording_info.get("user_wav") and recording_info["user_wav"].exists():
                user_start = recording_info.get("user_start", 0) or 0
                result = model.transcribe(str(recording_info["user_wav"]))
                for seg in result.get("segments", []):
                    abs_time = user_start + seg["start"]
                    text = seg["text"].strip()
                    if text:
                        segments.append((abs_time, "User", text))

            # Transcribe agent audio
            if recording_info.get("agent_wav") and recording_info["agent_wav"].exists():
                agent_start = recording_info.get("agent_start", 0) or 0
                result = model.transcribe(str(recording_info["agent_wav"]))
                for seg in result.get("segments", []):
                    abs_time = agent_start + seg["start"]
                    text = seg["text"].strip()
                    if text:
                        segments.append((abs_time, "Agent", text))

            # Sort by timestamp - don't merge, keep each segment separate for accuracy
            segments.sort(key=lambda x: x[0])

            # Only merge if segments are very close together (< 0.5 seconds apart) and same speaker
            merged = []
            for seg in segments:
                if merged and merged[-1][1] == seg[1]:
                    # Same speaker - only merge if within 0.5 seconds
                    time_gap = seg[0] - merged[-1][0]
                    if time_gap < 0.5:
                        merged[-1] = (merged[-1][0], merged[-1][1], merged[-1][2] + " " + seg[2])
                    else:
                        # Too much gap - keep separate
                        merged.append(seg)
                else:
                    merged.append(seg)

            # Write clean transcript with timestamps
            transcript_file = Path(__file__).parent.parent.parent / "transcripts" / f"{call_uuid}_final.txt"
            with open(transcript_file, "w") as f:
                # Get call start time for relative timestamps
                call_start = min(s[0] for s in merged) if merged else 0
                for abs_time, role, text in merged:
                    rel_time = abs_time - call_start
                    mins = int(rel_time // 60)
                    secs = int(rel_time % 60)
                    f.write(f"[{mins:02d}:{secs:02d}] {role}: {text}\n\n")

            logger.info(f"Transcription complete for {call_uuid}: {len(merged)} turns")
            return transcript_file
        except ImportError:
            logger.warning("Whisper not installed - skipping transcription")
            return None
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None


    async def preload(self):
        """Preload the Gemini session while phone is ringing"""
        try:
            logger.info(f"PRELOADING Gemini session for call {self.call_uuid}")
            self.is_active = True
            self._session_task = asyncio.create_task(self._run_google_live_session())
            # Wait for setup to complete (with timeout - 8s max for better greeting)
            try:
                await asyncio.wait_for(self._preload_complete.wait(), timeout=8.0)
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
        # Start silence monitor (3 second SLA)
        self._silence_monitor_task = asyncio.create_task(self._monitor_silence())

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
        """Monitor call duration with periodic heartbeat and trigger wrap-up at 8 minutes"""
        try:
            logger.info(f"Call {self.call_uuid[:8]} active - streaming audio")

            # Heartbeat every 60 seconds until wrap-up time
            wrap_up_time = self.max_call_duration - 30  # 7:30
            elapsed = 0

            while elapsed < wrap_up_time:
                await asyncio.sleep(60)
                elapsed += 60
                if self.is_active and not self._closing_call:
                    # Log buffer stats for debugging latency
                    logger.info(f"Call {self.call_uuid[:8]} in progress: {elapsed}s | history:{len(self._conversation_history)} inbuf:{len(self.inbuffer)} reconnect_buf:{len(self._reconnect_audio_buffer)}")
                else:
                    return  # Call ended, stop monitoring

            if self.is_active and not self._closing_call:
                logger.info(f"Call {self.call_uuid[:8]} reaching 8 min limit - triggering wrap-up")
                self._closing_call = True
                await self._send_wrap_up_message()

                # Wait another 30 seconds then force end
                await asyncio.sleep(30)
                if self.is_active:
                    logger.info(f"Call {self.call_uuid[:8]} reached max duration - ending call")
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

    async def _monitor_silence(self):
        """Monitor for silence - nudge AI if no response within 3 second SLA"""
        try:
            while self.is_active and not self._closing_call:
                await asyncio.sleep(1.0)  # Check every second

                # Only check if user has spoken and we're waiting for response
                if self._last_user_speech_time is None:
                    continue

                silence_duration = time.time() - self._last_user_speech_time

                # If silence exceeds SLA, nudge the AI to respond
                if silence_duration >= self._silence_sla_seconds:
                    logger.warning(f"SILENCE SLA BREACH: {silence_duration:.1f}s without AI response - nudging model")
                    await self._send_silence_nudge()
                    # Reset timer to avoid repeated nudges
                    self._last_user_speech_time = None

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in silence monitor: {e}")

    async def _send_silence_nudge(self):
        """Send a nudge to AI to respond when silence detected"""
        if not self.goog_live_ws or self._closing_call:
            return
        try:
            msg = {
                "client_content": {
                    "turns": [{
                        "role": "user",
                        "parts": [{"text": "[Continue the conversation - respond to what the customer just said]"}]
                    }],
                    "turn_complete": True
                }
            }
            await self.goog_live_ws.send(json.dumps(msg))
            logger.info("Sent silence nudge to AI")
        except Exception as e:
            logger.error(f"Error sending silence nudge: {e}")

    async def _run_google_live_session(self):
        url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={config.google_api_key}"
        reconnect_attempts = 0
        max_reconnects = 5  # Increased for better stability

        while self.is_active and reconnect_attempts < max_reconnects:
            try:
                async with websockets.connect(url, ping_interval=30, ping_timeout=20, close_timeout=5) as ws:
                    self.goog_live_ws = ws
                    reconnect_attempts = 0  # Reset on successful connect
                    logger.info("Connected to Google Live API")
                    await self._send_session_setup()
                    # Flush any buffered audio from reconnection
                    if self._reconnect_audio_buffer:
                        logger.info(f"Flushing {len(self._reconnect_audio_buffer)} buffered audio chunks after reconnect")
                        for buffered_audio in self._reconnect_audio_buffer:
                            await self.handle_plivo_audio(buffered_audio)
                        self._reconnect_audio_buffer = []
                    async for message in ws:
                        if not self.is_active:
                            break
                        await self._receive_from_google(message)
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"Google WS connection closed: code={e.code}, reason={e.reason}")
                if self.is_active and not self._closing_call:
                    reconnect_attempts += 1
                    logger.info(f"Attempting reconnect {reconnect_attempts}/{max_reconnects}...")
                    await asyncio.sleep(0.2)  # Faster reconnect (was 0.5)
                    continue
            except Exception as e:
                logger.error(f"Google Live error: {e}")
                if self.is_active and not self._closing_call:
                    reconnect_attempts += 1
                    logger.info(f"Attempting reconnect {reconnect_attempts}/{max_reconnects} after error...")
                    await asyncio.sleep(0.2)  # Faster reconnect (was 0.5)
                    continue
            break  # Normal exit

        self.goog_live_ws = None
        logger.info("Google Live session ended")

    async def _send_session_setup(self):
        # Concise accent instruction (shorter = faster responses)
        accent_instruction = """VOICE: Indian English accent (Bangalore professional). Use "Actually...", "Basically...", "Isn't it?" naturally. English only, no Hindi. Warm and professional tone.

"""
        # Combine: accent first, then main prompt
        full_prompt = accent_instruction + self.prompt

        # On reconnect, add context about ongoing conversation (keep minimal to reduce latency)
        if not self._is_first_connection and self._conversation_history:
            # Build conversation summary for context restoration
            history_text = "\n\n[RECONNECTION - call in progress. Recent conversation:]\n"
            for msg_item in self._conversation_history[-self._max_history_size:]:
                role = "Customer" if msg_item["role"] == "user" else "You"
                history_text += f"{role}: {msg_item['text']}\n"
            history_text += "[Continue naturally - do NOT greet again]"
            full_prompt = full_prompt + history_text
            logger.info(f"Reconnecting with {min(len(self._conversation_history), self._max_history_size)} messages of context")

        # Detect voice based on prompt content (female -> Kore, default -> Puck)
        voice_name = detect_voice_from_prompt(self.prompt)

        msg = {
            "setup": {
                "model": "models/gemini-2.5-flash-native-audio-preview-09-2025",
                "generation_config": {
                    "response_modalities": ["AUDIO"],  # Native audio model - audio only
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": voice_name
                            }
                        }
                    },
                    # Minimal thinking for better instruction following (adds ~50-100ms)
                    "thinking_config": {
                        "thinking_budget": 128
                    }
                },
                # Note: Transcription via Whisper offline (native audio model doesn't support real-time transcription)
                "system_instruction": {"parts": [{"text": full_prompt}]},
                "tools": [{"function_declarations": TOOL_DECLARATIONS}]
            }
        }
        await self.goog_live_ws.send(json.dumps(msg))

        # Mark first connection done
        if self._is_first_connection:
            self._is_first_connection = False

        logger.info(f"Sent session setup with voice: {voice_name}, reconnect={not self._is_first_connection}")

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

            # Handle end_call tool - mark agent as said goodbye, wait for mutual farewell
            if tool_name == "end_call":
                reason = tool_args.get("reason", "conversation ended")
                logger.info(f"END CALL tool called: {reason}")
                self._save_transcript("SYSTEM", f"Agent requested call end: {reason}")

                # Mark agent as having said goodbye
                self.agent_said_goodbye = True

                # Send success response
                try:
                    tool_response = {
                        "tool_response": {
                            "function_responses": [{
                                "id": call_id,
                                "name": tool_name,
                                "response": {"success": True, "message": "Waiting for mutual goodbye before ending"}
                            }]
                        }
                    }
                    await self.goog_live_ws.send(json.dumps(tool_response))
                except:
                    pass

                # Check if user already said goodbye
                self._check_mutual_goodbye()

                # Fallback: if user doesn't respond within 30 seconds, end anyway
                # Increased from 10s to give user more time to respond
                if not self._closing_call:
                    asyncio.create_task(self._fallback_hangup(30.0))
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

    async def _fallback_hangup(self, timeout: float):
        """Fallback hangup if user doesn't respond after agent says goodbye"""
        try:
            await asyncio.sleep(timeout)
            if not self._closing_call and self.agent_said_goodbye:
                logger.info(f"Fallback hangup - user didn't respond within {timeout}s after agent goodbye")
                self._closing_call = True
                await self._hangup_call_delayed(1.0)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Fallback hangup error: {e}")

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
                self._google_session_start = time.time()  # Track session start for 10-min limit
                self._save_transcript("SYSTEM", "AI ready")
                # Trigger immediate greeting during preload
                await self._send_initial_greeting()

            # Handle GoAway message - 9-minute warning before 10-minute session limit
            if "goAway" in resp:
                logger.warning("Received GoAway from Google - session will end soon (10-min limit). Triggering proactive reconnect...")
                self._save_transcript("SYSTEM", "Session refresh triggered (10-min limit)")
                # Don't wait for disconnect - proactively close and reconnect
                if self.goog_live_ws:
                    await self.goog_live_ws.close()
                return

            # Handle tool calls
            if "toolCall" in resp:
                await self._handle_tool_call(resp["toolCall"])
                return

            if "serverContent" in resp:
                sc = resp["serverContent"]

                # Debug: log all serverContent keys to find user transcription field
                sc_keys = list(sc.keys())
                if sc_keys != ['modelTurn'] and sc_keys != ['turnComplete']:
                    logger.debug(f"serverContent keys: {sc_keys}")

                # Check if turn is complete (greeting done)
                if sc.get("turnComplete"):
                    self._preload_complete.set()
                    self.greeting_audio_complete = True
                    self._turn_count += 1

                    # Log turn latency at INFO level (always visible)
                    if self._turn_start_time and self._current_turn_audio_chunks > 0:
                        turn_duration_ms = (time.time() - self._turn_start_time) * 1000
                        logger.info(f"TURN #{self._turn_count}: {self._current_turn_audio_chunks} chunks in {turn_duration_ms:.0f}ms")
                        self._turn_start_time = None

                    # Detect empty turn (AI didn't generate audio) - nudge to respond
                    if self._current_turn_audio_chunks == 0 and self.greeting_audio_complete and not self._closing_call:
                        self._empty_turn_nudge_count += 1
                        if self._empty_turn_nudge_count <= 3:  # Max 3 nudges to prevent loop
                            logger.warning(f"EMPTY TURN #{self._turn_count} - nudging AI (attempt {self._empty_turn_nudge_count})")
                            asyncio.create_task(self._send_silence_nudge())
                    else:
                        self._empty_turn_nudge_count = 0  # Reset on successful audio

                    # Reset turn audio counter
                    self._current_turn_audio_chunks = 0

                if sc.get("interrupted"):
                    logger.info("AI was interrupted by user")
                    if self.plivo_ws:
                        await self.plivo_ws.send_text(json.dumps({"event": "clearAudio", "stream_id": self.stream_id}))

                # Capture user speech transcription from Gemini
                if "inputTranscript" in sc:
                    user_text = sc["inputTranscript"]
                    if user_text and user_text.strip():
                        self._last_user_speech_time = time.time()  # Track for latency
                        logger.info(f"USER said: {user_text}")
                        self._save_transcript("USER", user_text.strip())
                        # Store in conversation history for reconnect context (limit size to prevent latency)
                        self._conversation_history.append({"role": "user", "text": user_text.strip()})
                        if len(self._conversation_history) > self._max_history_size:
                            self._conversation_history = self._conversation_history[-self._max_history_size:]
                        # Track if user said goodbye
                        if self._is_goodbye_message(user_text):
                            logger.info(f"USER said goodbye: {user_text[:50]}")
                            self.user_said_goodbye = True
                            # Check if both parties said goodbye
                            self._check_mutual_goodbye()

                if "modelTurn" in sc:
                    parts = sc.get("modelTurn", {}).get("parts", [])
                    for p in parts:
                        if p.get("inlineData", {}).get("data"):
                            audio = p["inlineData"]["data"]
                            audio_bytes = base64.b64decode(audio)
                            # Track audio chunks for empty turn detection
                            self._current_turn_audio_chunks += 1
                            # Track turn start time (first audio chunk of this turn)
                            if self._current_turn_audio_chunks == 1:
                                self._turn_start_time = time.time()
                            # Record AI audio (24kHz)
                            self._record_audio("AI", audio_bytes, 24000)

                            # Latency check - only log if slow (> threshold)
                            if self._last_user_speech_time:
                                latency_ms = (time.time() - self._last_user_speech_time) * 1000
                                if latency_ms > LATENCY_THRESHOLD_MS:
                                    logger.warning(f"SLOW: AI response took {latency_ms:.0f}ms")
                                self._last_user_speech_time = None  # Reset after first response

                            # During preload (no plivo_ws yet), always store audio
                            # This fixes race condition where turnComplete arrives before all audio
                            if not self.plivo_ws:
                                self.preloaded_audio.append(audio)
                                # Removed per-chunk logging
                            elif self.plivo_ws:
                                # After greeting done, send directly to Plivo
                                try:
                                    await self.plivo_ws.send_text(json.dumps({
                                        "event": "playAudio",
                                        "media": {"contentType": "audio/x-l16", "sampleRate": 24000, "payload": audio}
                                    }))
                                    # Removed per-chunk logging to reduce I/O overhead
                                except Exception as plivo_err:
                                    logger.error(f"Error sending audio to Plivo: {plivo_err} - continuing")
                        if p.get("text"):
                            ai_text = p["text"].strip()
                            logger.debug(f"AI TEXT: {ai_text[:100]}...")
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
                                # Store in conversation history for reconnect context (limit size)
                                self._conversation_history.append({"role": "model", "text": ai_text})
                                if len(self._conversation_history) > self._max_history_size:
                                    self._conversation_history = self._conversation_history[-self._max_history_size:]
                                # Track if agent said goodbye (don't end immediately - wait for user)
                                if not self._closing_call and self._is_goodbye_message(ai_text):
                                    logger.info(f"AGENT said goodbye: {ai_text[:50]}")
                                    self.agent_said_goodbye = True
                                    # Check if both parties said goodbye
                                    self._check_mutual_goodbye()
        except Exception as e:
            logger.error(f"Error processing Google message: {e} - continuing session")

    async def handle_plivo_audio(self, audio_b64):
        """Handle incoming audio from Plivo - graceful error handling"""
        try:
            if not self.is_active or not self.start_streaming:
                return  # Skip silently to reduce log noise
            if not self.goog_live_ws:
                # Buffer audio during reconnection (don't lose user speech)
                if len(self._reconnect_audio_buffer) < self._max_reconnect_buffer:
                    self._reconnect_audio_buffer.append(audio_b64)
                    if len(self._reconnect_audio_buffer) == 1:
                        logger.warning("Google WS disconnected - buffering audio for reconnection")
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

        # Cancel silence monitor
        if self._silence_monitor_task:
            self._silence_monitor_task.cancel()

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
                # Step 1: Save separate user/agent recordings
                recording_info = self._save_recording()

                # Step 2: Transcribe with Whisper (only if enabled - RAM heavy)
                if recording_info and config.enable_whisper:
                    self._transcribe_recording_sync(recording_info, self.call_uuid)
                elif recording_info:
                    logger.info(f"Whisper disabled - recordings saved: user={recording_info.get('user_wav')}, agent={recording_info.get('agent_wav')}")

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

            # Read transcript file - prefer final (Whisper) transcript, fallback to real-time
            transcript = ""
            try:
                transcript_dir = Path(__file__).parent.parent.parent / "transcripts"
                final_transcript = transcript_dir / f"{self.call_uuid}_final.txt"
                realtime_transcript = transcript_dir / f"{self.call_uuid}.txt"
                if final_transcript.exists():
                    transcript = final_transcript.read_text()
                elif realtime_transcript.exists():
                    transcript = realtime_transcript.read_text()
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
