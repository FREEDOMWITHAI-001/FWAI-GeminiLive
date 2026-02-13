# Plivo + Google Live API Stream Handler with Preloading
import asyncio
import json
import base64
import os
import wave
import struct
import threading
import queue
import time
from typing import Dict, Optional
from dataclasses import dataclass
from loguru import logger
from datetime import datetime
from pathlib import Path
import websockets
from src.core.config import config
from src.tools import execute_tool
from src.conversational_prompts import render_prompt
from src.db.session_db import session_db


def get_vertex_ai_token():
    """Get OAuth2 access token for Vertex AI"""
    try:
        import google.auth
        from google.auth.transport.requests import Request

        scopes = [
            'https://www.googleapis.com/auth/cloud-platform',
            'https://www.googleapis.com/auth/generative-language',
            'https://www.googleapis.com/auth/generative-language.retriever',
        ]
        t0 = time.time()
        credentials, project = google.auth.default(scopes=scopes)
        credentials.refresh(Request())
        token_ms = (time.time() - t0) * 1000
        logger.info(f"Vertex AI token for {project} ({token_ms:.0f}ms)")
        return credentials.token
    except Exception as e:
        logger.error(f"Failed to get Vertex AI token: {e}")
        return None

# Latency threshold - only log if slower than this (ms)
LATENCY_THRESHOLD_MS = 500

# Recording directory
RECORDINGS_DIR = Path(__file__).parent.parent.parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)


class CallLogger:
    """Structured call lifecycle logger with visual indentation."""

    def __init__(self, call_id: str):
        self.id = call_id[:8]

    def section(self, title: str):
        logger.info(f"[{self.id}] ══════ {title} ══════")

    def phase(self, title: str):
        logger.info(f"[{self.id}] ├─ {title}")

    def detail(self, msg: str):
        logger.info(f"[{self.id}] │  ├─ {msg}")

    def detail_last(self, msg: str):
        logger.info(f"[{self.id}] │  └─ {msg}")

    def turn(self, num: int, extra: str = ""):
        suffix = f" ({extra})" if extra else ""
        logger.info(f"[{self.id}] ├─ TURN #{num}{suffix}")

    def agent(self, text: str):
        logger.info(f"[{self.id}] │  ├─ AGENT: {text}")

    def user(self, text: str):
        logger.info(f"[{self.id}] │  ├─ USER:  {text}")

    def metric(self, text: str):
        logger.info(f"[{self.id}] │  └─ {text}")

    def warn(self, msg: str):
        logger.warning(f"[{self.id}] ⚠ {msg}")

    def error(self, msg: str):
        logger.error(f"[{self.id}] ✗ {msg}")


def detect_voice_from_prompt(prompt: str) -> str:
    """Detect voice based on prompt content. Returns 'Kore' for female, 'Puck' for male (default)."""
    if not prompt:
        return "Puck"
    prompt_lower = prompt.lower()

    # FIRST: Check for male names - if found, use male voice (Puck)
    male_names = [
        "rahul", "vishnu", "avinash", "arjun", "raj", "amit", "vijay", "suresh",
        "mahesh", "ramesh", "ganesh", "kiran", "sanjay", "ajay", "ravi", "kumar"
    ]
    for name in male_names:
        if name in prompt_lower:
            logger.info(f"Detected male name '{name}' in prompt - using Puck voice")
            return "Puck"

    # THEN: Check for female indicators
    female_indicators = [
        "female", "woman", "girl", "lady",
        "mousumi", "priya", "anjali", "divya", "neha", "pooja", "shreya",
        "sunita", "anita", "kavita", "rekha", "meena", "sita", "geeta"
    ]
    for indicator in female_indicators:
        if indicator in prompt_lower:
            logger.info(f"Detected female voice indicator '{indicator}' in prompt - using Kore voice")
            return "Kore"

    # Default to male voice
    return "Puck"

# Tool definitions for Gemini Live (minimal for lower latency)
# NOTE: WhatsApp messaging disabled during calls to reduce latency/interruptions
TOOL_DECLARATIONS = [
    {
        "name": "end_call",
        "description": "End the phone call. ONLY call this AFTER the conversation is complete AND you have said goodbye. NEVER call this just because the user paused or was quiet.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"}
            },
            "required": ["reason"]
        }
    }
]

@dataclass
class AudioChunk:
    """Audio chunk flowing through the queue pipeline"""
    audio_b64: str        # Base64-encoded audio data
    turn_id: int          # Gemini turn counter (for cancellation)
    sample_rate: int = 24000


class PlivoGeminiSession:
    def __init__(self, call_uuid: str, caller_phone: str, prompt: str = None, context: dict = None, webhook_url: str = None, client_name: str = "fwai"):
        self.call_uuid = call_uuid  # Internal UUID
        self.plivo_call_uuid = None  # Plivo's actual call UUID (set later)
        self.caller_phone = caller_phone
        self.context = context or {}  # Context for templates (customer_name, course_name, etc.)
        self.client_name = client_name or "fwai"

        # Prompt: API-provided prompt is the single source of truth
        self.prompt = render_prompt(prompt or "", self.context)
        logger.info(f"[{call_uuid[:8]}] Direct prompt mode for client: {client_name or 'default'}")

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
        self._goodbye_pending = False  # Defer goodbye detection to turnComplete (avoid cutting mid-sentence)

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
        self._current_turn_agent_text = []  # Accumulate agent speech fragments per turn
        self._current_turn_user_text = []  # Accumulate user speech fragments per turn

        # Audio send queue and worker
        self._plivo_send_queue = asyncio.Queue(maxsize=200)
        self._current_turn_id = 0
        self._sender_worker_task = None

        # Speech detection logging
        self._user_speaking = False  # Track if user is currently speaking
        self._agent_speaking = False  # Track if agent is currently speaking
        self._last_user_audio_time = None  # Last time user audio received
        self._user_speech_start_time = None  # When user started speaking

        # Audio buffer for reconnection (store audio if Google WS drops briefly)
        self._reconnect_audio_buffer = []
        self._max_reconnect_buffer = 150  # Increased buffer (~3 seconds) for better reconnection

        # Conversation history - saved to file in background thread (no latency impact)
        self._conversation_history = []  # In-memory cache for quick access
        self._max_history_size = 10  # Keep only last 10 messages
        self._is_first_connection = True  # Track if this is first connect or reconnect
        self._conversation_file = RECORDINGS_DIR / f"{call_uuid}_conversation.json"
        self._conversation_queue = queue.Queue()  # Queue for background file writes
        self._conversation_thread = None
        self._start_conversation_logger()  # Start background thread for file writes

        # Reconnection state
        self._is_reconnecting = False  # Flag to handle reconnection gracefully

        # Google session refresh timer (10-min limit, refresh at 9 min)
        self._google_session_start = None
        self._session_refresh_task = None
        self.GOOGLE_SESSION_LIMIT = 9 * 60  # Refresh at 9 minutes (before 10-min disconnect)

        # REST API transcription: Buffer user audio, transcribe when turn completes
        self._user_audio_buffer = bytearray(b"")  # Buffer for user audio (16kHz PCM)
        self._max_audio_buffer_size = 16000 * 2 * 30  # Max 30 seconds of audio (16kHz, 16-bit)
        self._last_user_transcript_time = 0  # Track when we last got a transcript

        # Full transcript collection (in-memory backup for webhook)
        self._full_transcript = []  # List of {"role": "USER/AGENT", "text": "...", "timestamp": "..."}

        # Session split - reset audio KV cache every N turns to keep latency low
        self._turns_since_reconnect = 0
        self._session_split_interval = 3  # Split every 3 turns
        self._last_agent_text = ""  # Last thing AI said (for split context)
        self._last_user_text = ""   # Last thing user said (for split context)
        self._last_agent_question = ""  # Last question AI asked (for anti-repetition)
        self._turn_exchanges = []   # Complete turn texts for clean summaries

        # Hot-swap session management
        self._standby_ws = None
        self._standby_ready = asyncio.Event()
        self._standby_task = None
        self._prewarm_task = None
        self._swap_in_progress = False
        self._active_receive_task = None

        # Timing instrumentation
        self._preload_start_time = None    # When preload() started
        self._setup_sent_time = None       # When setup message was sent to Gemini
        self._greeting_trigger_time = None # When greeting trigger was sent
        self._first_audio_time = None      # When first AI audio chunk arrived
        self._call_answered_time = None    # When Plivo WS attached
        self._first_audio_to_caller = None # When first audio sent to caller
        self._turn_first_byte_time = None  # When first audio byte of current turn arrived

        # Structured logger
        self.log = CallLogger(call_uuid)

    # Minimum turns before goodbye detection activates (prevents premature call end)
    MIN_TURNS_FOR_GOODBYE = 6

    def _is_goodbye_message(self, text: str) -> bool:
        """Detect if agent is saying goodbye - triggers auto call end.
        Only activates after MIN_TURNS_FOR_GOODBYE to prevent early cutoff."""
        if self._turn_count < self.MIN_TURNS_FOR_GOODBYE:
            return False

        text_lower = text.lower()
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
                logger.info(f"[{self.call_uuid[:8]}] Mutual goodbye - ending call")
                self._closing_call = True
                asyncio.create_task(self._hangup_call_delayed(0.5))  # Quick end
            else:
                # Agent said goodbye but user hasn't - start short timeout
                logger.debug(f"[{self.call_uuid[:8]}] Agent goodbye, waiting 3s for user")
                asyncio.create_task(self._quick_goodbye_timeout(3.0))

    async def _quick_goodbye_timeout(self, timeout: float):
        """Quick timeout after agent says goodbye - don't wait too long"""
        try:
            await asyncio.sleep(timeout)
            if not self._closing_call and self.agent_said_goodbye:
                logger.debug(f"[{self.call_uuid[:8]}] Goodbye timeout - ending call")
                self._closing_call = True
                await self._hangup_call_delayed(0.5)
        except asyncio.CancelledError:
            pass

    def _save_transcript(self, role, text):
        """Save transcript to file and in-memory list"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Always add to in-memory list (for webhook backup)
        self._full_transcript.append({
            "role": role,
            "text": text,
            "timestamp": timestamp
        })

        if not config.enable_transcripts:
            return
        try:
            transcript_dir = Path(__file__).parent.parent.parent / "transcripts"
            transcript_dir.mkdir(exist_ok=True)
            transcript_file = transcript_dir / f"{self.call_uuid}.txt"
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

    def _start_conversation_logger(self):
        """Start background thread for saving conversation to file (no latency impact)"""
        def conversation_worker():
            while True:
                try:
                    item = self._conversation_queue.get(timeout=1.0)
                    if item is None:  # Shutdown signal
                        break
                    # Append to file
                    self._save_conversation_to_file(item)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Conversation logger error: {e}")

        self._conversation_thread = threading.Thread(target=conversation_worker, daemon=True)
        self._conversation_thread.start()
        logger.debug("Conversation logger thread started")

    def _save_conversation_to_file(self, message: dict):
        """Save conversation message to file (called from background thread)"""
        try:
            # Read existing
            if self._conversation_file.exists():
                with open(self._conversation_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []

            # Append new message
            history.append(message)

            # Keep only last N messages
            if len(history) > self._max_history_size:
                history = history[-self._max_history_size:]

            # Write back
            with open(self._conversation_file, 'w') as f:
                json.dump(history, f)
        except Exception as e:
            logger.error(f"Error saving conversation to file: {e}")

    def _load_conversation_from_file(self) -> list:
        """Load conversation history from file for reconnection"""
        try:
            if self._conversation_file.exists():
                with open(self._conversation_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading conversation from file: {e}")
        return []

    def _log_conversation(self, role: str, text: str):
        """Queue conversation message for background file save (non-blocking)"""
        message = {"role": role, "text": text, "timestamp": time.time()}
        # Update in-memory cache
        self._conversation_history.append(message)
        if len(self._conversation_history) > self._max_history_size:
            self._conversation_history = self._conversation_history[-self._max_history_size:]
        # Queue for background file write
        try:
            self._conversation_queue.put_nowait(message)
        except queue.Full:
            pass

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
        """Save mixed MP3 file for Gemini transcription (10x smaller than WAV)"""
        logger.info(f"Saving recording: enabled={self.recording_enabled}, chunks={len(self.audio_chunks)}")
        if not self.recording_enabled or not self.audio_chunks:
            logger.warning(f"Skipping recording: enabled={self.recording_enabled}, chunks={len(self.audio_chunks)}")
            return None
        try:
            SAMPLE_RATE = 16000
            BYTES_PER_SAMPLE = 2  # 16-bit audio

            # Sort chunks by timestamp
            sorted_chunks = sorted(self.audio_chunks, key=lambda x: x[3])
            call_start = sorted_chunks[0][3]

            # Build mixed audio with proper timeline
            mixed_audio = bytearray()
            current_time = call_start

            for chunk in sorted_chunks:
                role, audio_bytes, sample_rate, timestamp = chunk
                if sample_rate == 24000:
                    audio_bytes = self._resample_24k_to_16k(audio_bytes)

                # Insert silence for gaps
                gap = timestamp - current_time
                if gap > 0.02:  # Gap > 20ms
                    silence_samples = int(gap * SAMPLE_RATE)
                    mixed_audio.extend(b'\x00' * (silence_samples * BYTES_PER_SAMPLE))
                    current_time = timestamp

                mixed_audio.extend(audio_bytes)
                current_time = timestamp + len(audio_bytes) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

            # Save as MP3 using pydub (10x smaller than WAV)
            mixed_mp3 = RECORDINGS_DIR / f"{self.call_uuid}_mixed.mp3"
            try:
                from pydub import AudioSegment
                audio_segment = AudioSegment(
                    data=bytes(mixed_audio),
                    sample_width=2,
                    frame_rate=16000,
                    channels=1
                )
                audio_segment.export(str(mixed_mp3), format="mp3", bitrate="64k")
                logger.info(f"MP3 recording saved: {mixed_mp3.stat().st_size} bytes, {len(sorted_chunks)} chunks")
            except ImportError:
                # Fallback to WAV if pydub not installed
                logger.warning("pydub not installed, falling back to WAV")
                mixed_mp3 = RECORDINGS_DIR / f"{self.call_uuid}_mixed.wav"
                with wave.open(str(mixed_mp3), 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(bytes(mixed_audio))
                logger.info(f"WAV recording saved: {len(mixed_audio)} bytes")

            return {
                "mixed_wav": mixed_mp3,  # Key name kept for compatibility
                "call_start": call_start
            }
        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            return None

    def _transcribe_recording_sync(self, recording_info: dict, call_uuid: str):
        """Transcribe using Gemini 2.0 Flash with native speaker diarization"""
        try:
            from google import genai
            import time as time_module

            mixed_wav = recording_info.get("mixed_wav")

            if not mixed_wav or not mixed_wav.exists():
                logger.warning(f"No mixed recording found for {call_uuid}")
                return None

            logger.info(f"Starting Gemini transcription for {call_uuid}")

            # Initialize Gemini client
            client = genai.Client(api_key=config.google_api_key)

            # Upload the audio file
            logger.info(f"Uploading audio file for transcription...")
            audio_file = client.files.upload(file=str(mixed_wav))

            # Wait for processing
            while audio_file.state == "PROCESSING":
                time_module.sleep(2)
                audio_file = client.files.get(name=audio_file.name)

            if audio_file.state == "FAILED":
                logger.error(f"Gemini audio processing failed for {call_uuid}")
                return None

            # Generate transcript with speaker diarization
            prompt = """Transcribe this phone call audio accurately.

Rules:
- The FIRST speaker is always the "Agent" (AI sales counselor calling)
- The SECOND speaker is always the "User" (customer receiving the call)
- Format each line as: [MM:SS] Speaker: text
- Use timestamps from the audio
- Keep the transcript natural and accurate
- Do NOT add any commentary, just the transcript"""

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[audio_file, prompt]
            )

            # Save transcript
            transcript_file = Path(__file__).parent.parent.parent / "transcripts" / f"{call_uuid}_final.txt"
            with open(transcript_file, "w") as f:
                f.write(response.text)

            # Clean up uploaded file
            try:
                client.files.delete(name=audio_file.name)
            except Exception:
                pass

            logger.info(f"Gemini transcription complete for {call_uuid}")
            return transcript_file

        except ImportError:
            logger.warning("google-genai not installed - skipping transcription")
            return None
        except Exception as e:
            logger.error(f"Gemini transcription error: {e}")
            return None

    async def preload(self):
        """Preload the Gemini session while phone is ringing"""
        try:
            self._preload_start_time = time.time()
            self.log.section("CALL INITIATED")
            self.log.phase("PRELOAD")
            self.log.detail(f"Phone: {self.caller_phone} ({self.context.get('customer_name', 'Unknown')})")
            self.log.detail(f"Prompt: {len(self.prompt):,} chars")
            self.is_active = True
            self._session_task = asyncio.create_task(self._run_google_live_session())
            try:
                await asyncio.wait_for(self._preload_complete.wait(), timeout=8.0)
                preload_ms = (time.time() - self._preload_start_time) * 1000
                self.log.detail_last(f"Preloaded: {len(self.preloaded_audio)} chunks in {preload_ms:.0f}ms")
            except asyncio.TimeoutError:
                preload_ms = (time.time() - self._preload_start_time) * 1000
                self.log.warn(f"Preload timeout ({preload_ms:.0f}ms), {len(self.preloaded_audio)} chunks")
            return True
        except Exception as e:
            self.log.error(f"Preload failed: {e}")
            return False

    def attach_plivo_ws(self, plivo_ws):
        """Attach Plivo WebSocket when user answers"""
        self.plivo_ws = plivo_ws
        self.call_start_time = datetime.now()
        self._call_answered_time = time.time()
        preload_count = len(self.preloaded_audio)
        self.log.phase("CALL ANSWERED")
        if self._preload_start_time:
            wait_ms = (time.time() - self._preload_start_time) * 1000
            self.log.detail(f"Ring duration: {wait_ms:.0f}ms")
        self.log.detail(f"Plivo WS attached, {preload_count} preloaded chunks")
        if self.preloaded_audio:
            asyncio.create_task(self._send_preloaded_audio())
        else:
            self.log.warn("No preloaded audio - greeting will lag")
        # Start sender worker
        self._sender_worker_task = asyncio.create_task(self._plivo_sender_worker())
        # Start call duration timer
        self._timeout_task = asyncio.create_task(self._monitor_call_duration())
        # Start silence monitor (3 second SLA)
        self._silence_monitor_task = asyncio.create_task(self._monitor_silence())

    async def _send_preloaded_audio(self):
        """Send preloaded audio directly to plivo_send_queue"""
        count = len(self.preloaded_audio)
        for audio in self.preloaded_audio:
            chunk = AudioChunk(audio_b64=audio, turn_id=0, sample_rate=24000)
            try:
                self._plivo_send_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                self.log.warn("plivo_send_queue full during preload send")
        if self._call_answered_time:
            first_audio_ms = (time.time() - self._call_answered_time) * 1000
            self.log.detail_last(f"First audio to caller: {first_audio_ms:.0f}ms ({count} chunks)")
        self.preloaded_audio = []

    async def _monitor_call_duration(self):
        """Monitor call duration with periodic heartbeat and trigger wrap-up at 8 minutes"""
        try:
            logger.debug(f"[{self.call_uuid[:8]}] Call monitor started")

            # Heartbeat every 60 seconds until wrap-up time
            wrap_up_time = self.max_call_duration - 30  # 7:30
            elapsed = 0

            while elapsed < wrap_up_time:
                await asyncio.sleep(60)
                elapsed += 60
                if self.is_active and not self._closing_call:
                    logger.info(f"[{self.call_uuid[:8]}] Call in progress: {elapsed}s")
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
        """Monitor for silence - nudge AI if no response after user speaks"""
        try:
            while self.is_active and not self._closing_call:
                await asyncio.sleep(0.3)  # Check every 0.3 seconds for faster response

                if self._last_user_speech_time is None:
                    continue

                silence_duration = time.time() - self._last_user_speech_time

                # If silence exceeds SLA, nudge the AI to respond
                if silence_duration >= self._silence_sla_seconds:
                    self.log.warn(f"{silence_duration:.1f}s silence - nudging AI")
                    await self._send_silence_nudge()
                    # Reset timer to avoid repeated nudges
                    self._last_user_speech_time = None

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in silence monitor: {e}")

    async def _send_silence_nudge(self):
        """Send a nudge to AI when silence detected"""
        if not self.goog_live_ws or self._closing_call:
            return

        try:
            msg = {
                "client_content": {
                    "turns": [{
                        "role": "user",
                        "parts": [{"text": "[Respond to the customer]"}]
                    }],
                    "turn_complete": True
                }
            }
            await self.goog_live_ws.send(json.dumps(msg))
            logger.debug(f"[{self.call_uuid[:8]}] Sent nudge to AI")
        except Exception as e:
            logger.error(f"Error sending silence nudge: {e}")

    async def _plivo_sender_worker(self):
        """Worker: Reads from plivo_send_queue, sends to Plivo WebSocket"""
        logger.debug(f"[{self.call_uuid[:8]}] Plivo sender worker started")
        try:
            while self.is_active:
                try:
                    chunk: AudioChunk = await asyncio.wait_for(
                        self._plivo_send_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                if not self.plivo_ws:
                    continue

                try:
                    await self.plivo_ws.send_text(json.dumps({
                        "event": "playAudio",
                        "media": {
                            "contentType": "audio/x-l16",
                            "sampleRate": chunk.sample_rate,
                            "payload": chunk.audio_b64
                        }
                    }))
                except Exception as e:
                    logger.error(f"[{self.call_uuid[:8]}] Plivo sender error: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[{self.call_uuid[:8]}] Plivo sender worker error: {e}")
        logger.debug(f"[{self.call_uuid[:8]}] Plivo sender worker stopped")

    async def _send_reconnection_filler(self):
        """Handle silence during reconnection - clear audio and prepare for resume"""
        if not self.plivo_ws or self._closing_call:
            return
        try:
            logger.debug(f"[{self.call_uuid[:8]}] Preparing for reconnection")

            # Clear any pending audio to prevent stale data
            await self.plivo_ws.send_text(json.dumps({
                "event": "clearAudio",
                "stream_id": self.stream_id
            }))

        except Exception as e:
            logger.error(f"Error in reconnection filler: {e}")

    async def _connect_and_setup_ws(self, is_standby=False):
        """Create a new Gemini WS connection and send setup message.
        Returns the connected WS (caller must handle receive loop)."""
        label = "standby" if is_standby else "active"
        t0 = time.time()

        if config.use_vertex_ai:
            token = get_vertex_ai_token()
            if not token:
                logger.error("Failed to get Vertex AI token - falling back to Google AI Studio")
                url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={config.google_api_key}"
                extra_headers = None
            else:
                url = f"wss://{config.vertex_location}-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
                extra_headers = {"Authorization": f"Bearer {token}"}
                self.log.detail(f"Vertex AI: {config.vertex_location} ({label})")
        else:
            url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={config.google_api_key}"
            extra_headers = None
            self.log.detail(f"Google AI Studio ({label})")

        ws_kwargs = {"ping_interval": 30, "ping_timeout": 20, "close_timeout": 5}
        if extra_headers:
            ws_kwargs["additional_headers"] = extra_headers

        ws = await websockets.connect(url, **ws_kwargs)
        connect_ms = (time.time() - t0) * 1000
        self.log.detail(f"Gemini {label} WS connected ({connect_ms:.0f}ms)")

        self._setup_sent_time = time.time()
        await self._send_session_setup_on_ws(ws, is_standby=is_standby)
        return ws

    async def _ws_receive_loop(self, ws, is_standby=False):
        """Receive loop for a Gemini WS. For standby, stops after setupComplete."""
        try:
            async for message in ws:
                if not self.is_active:
                    break
                resp = json.loads(message)
                if is_standby:
                    if "setupComplete" in resp:
                        self._standby_ready.set()
                        ready_ms = (time.time() - self._setup_sent_time) * 1000 if self._setup_sent_time else 0
                        self.log.detail(f"Standby session ready ({ready_ms:.0f}ms)")
                        continue
                    elif "goAway" in resp:
                        self.log.warn("Standby GoAway — will re-prewarm")
                        await self._close_ws_quietly(ws)
                        self._standby_ws = None
                        self._standby_ready = asyncio.Event()
                        self._standby_task = None
                        self._prewarm_task = asyncio.create_task(self._prewarm_standby_connection())
                        return
                else:
                    await self._receive_from_google(message)
        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed as e:
            if not is_standby:
                self.log.warn(f"Active WS closed: {e.code}")

    async def _prewarm_standby_connection(self):
        """Pre-warm ONLY the WS connection for standby (setup deferred to swap time).
        This saves ~300ms at swap time while ensuring system_instruction is always current."""
        if self._standby_ws or self._swap_in_progress:
            return
        try:
            t0 = time.time()

            if config.use_vertex_ai:
                token = get_vertex_ai_token()
                if not token:
                    url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={config.google_api_key}"
                    extra_headers = None
                else:
                    url = f"wss://{config.vertex_location}-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
                    extra_headers = {"Authorization": f"Bearer {token}"}
            else:
                url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={config.google_api_key}"
                extra_headers = None

            ws_kwargs = {"ping_interval": 30, "ping_timeout": 20, "close_timeout": 5}
            if extra_headers:
                ws_kwargs["additional_headers"] = extra_headers

            ws = await websockets.connect(url, **ws_kwargs)
            self._standby_ws = ws
            connect_ms = (time.time() - t0) * 1000
            self.log.detail(f"Standby WS connected ({connect_ms:.0f}ms, setup deferred)")
        except Exception as e:
            self.log.error(f"Standby connection failed: {e}")
            self._standby_ws = None

    async def _hot_swap_session(self):
        """Hot-swap: send setup with CURRENT context to pre-connected standby, then swap.
        Setup is deferred to swap time so system_instruction always has fresh conversation state."""
        if self._swap_in_progress or not self._standby_ws:
            self.log.warn("No standby available, falling back")
            await self._fallback_session_split()
            return

        self._swap_in_progress = True
        swap_start = time.time()
        ws = self._standby_ws
        try:
            # Step 1: Send setup with CURRENT conversation summary to standby WS
            self._setup_sent_time = time.time()
            try:
                await self._send_session_setup_on_ws(ws, is_standby=False)
            except Exception as e:
                self.log.warn(f"Standby WS dead ({e}), falling back")
                await self._close_ws_quietly(ws)
                self._standby_ws = None
                await self._fallback_session_split()
                return

            # Step 2: Wait for setupComplete (~90ms)
            setup_ok = False
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                resp = json.loads(msg)
                if "setupComplete" in resp:
                    setup_ok = True
                    ready_ms = (time.time() - self._setup_sent_time) * 1000
                    self.log.detail(f"Standby ready ({ready_ms:.0f}ms)")
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed) as e:
                self.log.warn(f"Standby setup failed: {e}")

            if not setup_ok:
                self.log.warn("Standby setup not ready, falling back")
                await self._close_ws_quietly(ws)
                self._standby_ws = None
                await self._fallback_session_split()
                return

            # Step 3: Send anti-repetition context (silent, turn_complete=False)
            await self._send_context_to_ws(ws)

            # Step 4: Atomic swap
            old_ws = self.goog_live_ws
            old_receive_task = self._active_receive_task

            self.goog_live_ws = ws
            self._standby_ws = None

            if old_receive_task:
                old_receive_task.cancel()
            if self._standby_task:
                self._standby_task.cancel()
                self._standby_task = None

            self._active_receive_task = asyncio.create_task(
                self._ws_receive_loop(self.goog_live_ws, is_standby=False)
            )
            asyncio.create_task(self._close_ws_quietly(old_ws))

            self._turns_since_reconnect = 0
            self._standby_ready = asyncio.Event()
            self._prewarm_task = None
            self._is_reconnecting = False

            swap_ms = (time.time() - swap_start) * 1000
            self.log.phase(f"SESSION SPLIT (hot-swap at turn #{self._turn_count}) ✓ working well")
            self.log.detail("Setup + context sent at swap time (fresh)")
            self.log.detail("Atomic swap complete")
            self.log.detail_last(f"Old session closed | Swap: {swap_ms:.0f}ms")
            self._save_transcript("SYSTEM", f"Hot-swap session split at turn #{self._turn_count} ({swap_ms:.0f}ms)")
        finally:
            self._swap_in_progress = False

    async def _send_context_to_ws(self, ws):
        """Send anti-repetition reinforcement to WS before hot-swap.
        The system_instruction already has the full summary; this adds a
        strong reminder via client_content with turn_complete=False
        so Gemini absorbs it silently and waits for user audio."""
        last_user = self._last_user_text[:200] if self._last_user_text else ""
        agent_ref = (self._last_agent_question or self._last_agent_text)[:200]

        if agent_ref and last_user:
            trigger = (
                f'[REMINDER: Your MOST RECENT question was: "{agent_ref}". '
                f'The customer replied: "{last_user}". '
                f'DO NOT repeat this question. DO NOT speak until the customer speaks. '
                f'When they speak, respond naturally and move to the NEXT topic.]'
            )
        elif agent_ref:
            trigger = f'[REMINDER: You just said: "{agent_ref}". DO NOT repeat it. Wait for the customer to speak.]'
        else:
            trigger = "[Continue the conversation. Wait for the customer to speak.]"

        msg = {"client_content": {"turns": [{"role": "user", "parts": [{"text": trigger}]}], "turn_complete": False}}
        await ws.send(json.dumps(msg))
        self.log.detail(f"Anti-repetition sent: last_q='{agent_ref[:50]}'")

    async def _close_ws_quietly(self, ws):
        """Close a WS without error logging."""
        try:
            await ws.close()
        except Exception:
            pass

    async def _fallback_session_split(self):
        """Fallback when standby not available: close active WS and let main loop reconnect."""
        if not self.goog_live_ws or self._closing_call or self._is_reconnecting:
            return
        self._is_reconnecting = True
        self._turns_since_reconnect = 0
        self.log.phase(f"SESSION SPLIT (fallback at turn #{self._turn_count})")
        self._save_transcript("SYSTEM", f"Fallback session split at turn #{self._turn_count}")

        if self._active_receive_task:
            self._active_receive_task.cancel()
            self._active_receive_task = None

        ws = self.goog_live_ws
        self.goog_live_ws = None
        await self._close_ws_quietly(ws)

    async def _emergency_session_split(self):
        """Emergency split when GoAway fires with no standby: connect + setup + swap in one shot."""
        if self._swap_in_progress or self._closing_call:
            return
        self._swap_in_progress = True
        self.log.phase("SESSION SPLIT (emergency — GoAway)")
        try:
            # Null out goog_live_ws immediately to stop audio send errors
            old_ws = self.goog_live_ws
            self.goog_live_ws = None

            if self._active_receive_task:
                self._active_receive_task.cancel()
                self._active_receive_task = None

            # Connect new WS + send setup with current context
            t0 = time.time()
            ws = await self._connect_and_setup_ws(is_standby=False)

            # Wait for setupComplete
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            resp = json.loads(msg)
            if "setupComplete" not in resp:
                self.log.error("Emergency split: setup failed")
                await self._close_ws_quietly(ws)
                return

            # Send anti-repetition context
            await self._send_context_to_ws(ws)

            # Swap in
            self.goog_live_ws = ws
            self._active_receive_task = asyncio.create_task(
                self._ws_receive_loop(ws, is_standby=False)
            )
            asyncio.create_task(self._close_ws_quietly(old_ws))

            self._turns_since_reconnect = 0
            self._is_reconnecting = False
            swap_ms = (time.time() - t0) * 1000
            self.log.detail(f"Emergency swap complete ({swap_ms:.0f}ms)")
            self._save_transcript("SYSTEM", f"Emergency session split ({swap_ms:.0f}ms)")
        except Exception as e:
            self.log.error(f"Emergency split failed: {e}")
            # Let main loop reconnect
            self._is_reconnecting = True
        finally:
            self._swap_in_progress = False

    def _build_compact_summary(self) -> str:
        """Build conversation summary for session split context.
        Includes turn numbers for clarity and marks the last exchange explicitly."""
        if not self._turn_exchanges:
            return ""
        lines = []
        exchanges = self._turn_exchanges[-5:]
        total = len(exchanges)
        for i, exchange in enumerate(exchanges):
            turn_num = self._turn_count - total + i + 1
            is_last = (i == total - 1)
            prefix = f"Turn {turn_num}"
            if is_last:
                prefix += " (MOST RECENT — do NOT repeat)"
            if exchange.get("agent"):
                lines.append(f"{prefix} — You asked: {exchange['agent'][:300]}")
            if exchange.get("user"):
                lines.append(f"{prefix} — Customer replied: {exchange['user'][:300]}")
        return "\n".join(lines)

    async def _run_google_live_session(self):
        """Main session loop. Hot-swap handles planned transitions; this handles error recovery."""
        reconnect_attempts = 0
        max_reconnects = 5

        while self.is_active and reconnect_attempts < max_reconnects:
            ws = None
            try:
                ws = await self._connect_and_setup_ws(is_standby=False)
                self.goog_live_ws = ws
                reconnect_attempts = 0

                if self._reconnect_audio_buffer:
                    self.log.detail(f"Flushing {len(self._reconnect_audio_buffer)} buffered chunks")
                    for buffered_audio in self._reconnect_audio_buffer:
                        await self.handle_plivo_audio(buffered_audio)
                    self._reconnect_audio_buffer = []

                self._active_receive_task = asyncio.create_task(
                    self._ws_receive_loop(ws, is_standby=False)
                )
                await self._active_receive_task
                self._active_receive_task = None

                # If WS was replaced by hot-swap, exit this loop
                if self.goog_live_ws is not None and self.goog_live_ws is not ws:
                    return
                break

            except asyncio.CancelledError:
                if self.goog_live_ws is not None and self.goog_live_ws is not ws:
                    return
                break
            except Exception as e:
                self.log.error(f"Google Live error: {e}")
                if ws:
                    await self._close_ws_quietly(ws)
                if self.is_active and not self._closing_call:
                    self._is_reconnecting = True
                    reconnect_attempts += 1
                    self.log.warn(f"Reconnecting ({reconnect_attempts}/{max_reconnects})")
                    asyncio.create_task(self._send_reconnection_filler())
                    await asyncio.sleep(0.2)
                    continue
                break

        # Only null goog_live_ws if we still own it
        if self.goog_live_ws is ws:
            self.goog_live_ws = None

    def _pcm_to_wav(self, pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
        """Convert raw PCM bytes to WAV format"""
        import io
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(bits_per_sample // 8)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)

        return wav_buffer.getvalue()

    async def _send_session_setup_on_ws(self, ws, is_standby=False):
        """Send setup message on a specific WS. Includes anti-repetition markers on reconnect/standby."""
        full_prompt = self.prompt

        # Add natural speech variation and voice consistency guidance
        full_prompt += (
            "\n\n[SPEECH STYLE: Vary your pace, pitch, and energy naturally throughout the conversation. "
            "Speak faster when excited, slower when empathetic. Use pauses for emphasis. "
            "Match the customer's energy level. "
            "IMPORTANT: Maintain a consistent speaking voice — same warmth, same tone, same accent, same tempo baseline throughout the entire call. "
            "Never suddenly change your speaking style, speed, or personality mid-conversation.]"
        )

        # On reconnect or hot-swap, append conversation context + anti-repetition
        # to system_instruction so AI knows where the conversation is.
        if not self._is_first_connection:
            summary = self._build_compact_summary()
            if summary:
                full_prompt += f"\n\n[CONVERSATION SO FAR — you are mid-call, do NOT greet again:]\n{summary}"
                # Anti-repetition marker with last question
                agent_ref = self._last_agent_question or self._last_agent_text
                if agent_ref:
                    last_user = self._last_user_text or "(customer is about to respond)"
                    full_prompt += f'\n\n[YOUR LAST QUESTION was: "{agent_ref[:300]}"'
                    full_prompt += f' Customer responded: "{last_user[:200]}".'
                    full_prompt += ' DO NOT repeat or rephrase this question. Ask the NEXT question in the flow.]'
                self.log.detail(f"Setup with summary ({len(summary)} chars)")
            else:
                file_history = self._load_conversation_from_file()
                if file_history:
                    history_text = "\n\n[Recent conversation - continue from here:]\n"
                    for msg_item in file_history[-self._max_history_size:]:
                        role = "Customer" if msg_item["role"] == "user" else "You"
                        history_text += f"{role}: {msg_item['text']}\n"
                    history_text += "\n[Continue naturally. Do NOT greet again.]"
                    full_prompt += history_text
                    self._is_reconnecting = False

        voice_name = detect_voice_from_prompt(self.prompt)

        if config.use_vertex_ai:
            model_name = f"projects/{config.vertex_project_id}/locations/{config.vertex_location}/publishers/google/models/gemini-live-2.5-flash-native-audio"
        else:
            model_name = "models/gemini-2.5-flash-native-audio-preview-09-2025"

        msg = {
            "setup": {
                "model": model_name,
                "generation_config": {
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": voice_name
                            }
                        }
                    },
                    "thinking_config": {
                        "thinking_budget": 0  # Disable reasoning for fastest responses
                    }
                },
                "input_audio_transcription": {},
                "output_audio_transcription": {},
                "system_instruction": {"parts": [{"text": full_prompt}]},
                "tools": [{"function_declarations": TOOL_DECLARATIONS}]
            }
        }
        await ws.send(json.dumps(msg))
        label = "standby" if is_standby else ("first" if self._is_first_connection else "reconnect")
        self.log.detail(f"Setup sent ({label}), voice: {voice_name}")

    async def _send_initial_greeting(self):
        """Send initial trigger to make AI start the conversation"""
        if self.greeting_sent or not self.goog_live_ws:
            return
        self.greeting_sent = True

        # Auto-generate greeting trigger from context
        trigger_text = self.context.get("greeting_trigger", "")
        if not trigger_text:
            customer_name = self.context.get("customer_name", "")
            if customer_name:
                trigger_text = f"[Start the conversation now. Greet {customer_name} naturally using your opening line from the instructions.]"
            else:
                trigger_text = "[Start the conversation now. Greet the customer naturally using your opening line from the instructions.]"

        msg = {
            "client_content": {
                "turns": [{"role": "user", "parts": [{"text": trigger_text}]}],
                "turn_complete": True
            }
        }
        await self.goog_live_ws.send(json.dumps(msg))
        self.log.detail("Greeting trigger sent")

    async def _send_reconnection_trigger(self):
        """Trigger AI to speak immediately after reconnection"""
        if not self.goog_live_ws:
            return

        reconnect_text = "[Continue the conversation]"

        msg = {
            "client_content": {
                "turns": [{
                    "role": "user",
                    "parts": [{"text": reconnect_text}]
                }],
                "turn_complete": True
            }
        }
        await self.goog_live_ws.send(json.dumps(msg))
        logger.debug(f"[{self.call_uuid[:8]}] Reconnect trigger sent")

    async def _handle_tool_call(self, tool_call):
        """Execute tool and send response back to Gemini - gracefully handles errors"""
        func_calls = tool_call.get("functionCalls", [])
        for fc in func_calls:
            tool_name = fc.get("name")
            tool_args = fc.get("args", {})
            call_id = fc.get("id")

            self.log.detail(f"Tool: {tool_name}")
            self._save_transcript("TOOL", f"{tool_name}: {tool_args}")

            # Handle end_call tool
            if tool_name == "end_call":
                reason = tool_args.get("reason", "conversation ended")
                self.log.detail(f"End call: {reason}")
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
                except Exception:
                    pass

                # Check if user already said goodbye
                self._check_mutual_goodbye()

                # Fallback: if user doesn't respond within 5 seconds, end anyway
                if not self._closing_call:
                    asyncio.create_task(self._fallback_hangup(5.0))
                return

            # Execute the tool with context for templates - graceful error handling
            try:
                tool_start = time.time()
                result = await execute_tool(tool_name, self.caller_phone, context=self.context, **tool_args)
                tool_ms = (time.time() - tool_start) * 1000
                success = result.get("success", False)
                message = result.get("message", "Tool executed")
                self.log.detail(f"Tool result: {'OK' if success else 'FAIL'} ({tool_ms:.0f}ms)")
            except Exception as e:
                logger.error(f"Tool execution error for {tool_name}: {e}")
                success = False
                message = f"Tool temporarily unavailable, but conversation can continue"

            logger.debug(f"TOOL RESULT: success={success}, message={message}")
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
                logger.debug(f"Sent tool response for {tool_name}")
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

            hangup_uuid = self.plivo_call_uuid or self.call_uuid
            self.log.detail(f"Plivo hangup API: {hangup_uuid}")

            import httpx
            import base64

            auth_string = f"{config.plivo_auth_id}:{config.plivo_auth_token}"
            auth_b64 = base64.b64encode(auth_string.encode()).decode()

            url = f"https://api.plivo.com/v1/Account/{config.plivo_auth_id}/Call/{hangup_uuid}/"

            t0 = time.time()
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    url,
                    headers={"Authorization": f"Basic {auth_b64}"}
                )
                api_ms = (time.time() - t0) * 1000

                if response.status_code in [204, 200]:
                    self.log.detail(f"Plivo hangup OK ({api_ms:.0f}ms)")
                else:
                    self.log.error(f"Plivo hangup failed: {response.status_code} ({api_ms:.0f}ms)")

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
                setup_ms = (time.time() - self._setup_sent_time) * 1000 if self._setup_sent_time else 0
                self.log.detail(f"AI ready ({setup_ms:.0f}ms)")
                self.start_streaming = True
                self.setup_complete = True
                self._google_session_start = time.time()
                self._save_transcript("SYSTEM", f"AI ready ({setup_ms:.0f}ms)")
                if self._is_first_connection:
                    self._is_first_connection = False
                    self._greeting_trigger_time = time.time()
                    await self._send_initial_greeting()
                elif self._is_reconnecting:
                    self._is_reconnecting = False
                    await self._send_reconnection_trigger()

            # Handle GoAway message - 9-minute warning before 10-minute session limit
            if "goAway" in resp:
                self.log.warn("GoAway — triggering session split")
                self._save_transcript("SYSTEM", "Session GoAway (10-min limit)")
                if self._standby_ws:
                    asyncio.create_task(self._hot_swap_session())
                else:
                    # No standby — prewarm + swap immediately
                    asyncio.create_task(self._emergency_session_split())
                return

            # Handle tool calls
            if "toolCall" in resp:
                await self._handle_tool_call(resp["toolCall"])
                return

            if "serverContent" in resp:
                sc = resp["serverContent"]

                # Check if turn is complete (greeting done)
                if sc.get("turnComplete"):
                    self._preload_complete.set()
                    self.greeting_audio_complete = True
                    self._turn_count += 1
                    self._current_turn_id += 1

                    if self._turn_start_time and self._current_turn_audio_chunks > 0:
                        turn_duration_ms = (time.time() - self._turn_start_time) * 1000
                        full_agent = ""
                        full_user = ""
                        if self._current_turn_agent_text:
                            full_agent = " ".join(self._current_turn_agent_text)
                            self._last_agent_text = full_agent
                            if "?" in full_agent:
                                self._last_agent_question = full_agent
                            self._current_turn_agent_text = []
                        if self._current_turn_user_text:
                            full_user = " ".join(self._current_turn_user_text)
                            self._last_user_text = full_user
                            self._current_turn_user_text = []
                        # Track turn exchanges for compact summary
                        if full_agent or full_user:
                            self._turn_exchanges.append({"agent": full_agent, "user": full_user})
                            if len(self._turn_exchanges) > 5:
                                self._turn_exchanges = self._turn_exchanges[-5:]

                        extra = ""
                        if self._turns_since_reconnect == self._session_split_interval - 1:
                            extra = "prewarm standby"
                        elif self._turns_since_reconnect >= self._session_split_interval:
                            extra = "split pending"
                        self.log.turn(self._turn_count, extra)
                        if full_agent:
                            self.log.agent(full_agent)
                        if full_user:
                            self.log.user(full_user)
                        # Compute TTFB for this turn (user speech end → first AI audio)
                        ttfb_str = ""
                        if self._turn_first_byte_time and self._last_user_speech_time is None:
                            # _last_user_speech_time was reset when first audio arrived
                            pass
                        self.log.metric(f"{turn_duration_ms:.0f}ms | {self._current_turn_audio_chunks} chunks")
                        self._turn_first_byte_time = None
                        self._turn_start_time = None

                    # Detect empty turn (AI didn't generate audio) - nudge to respond
                    is_empty_turn = self._current_turn_audio_chunks == 0
                    if is_empty_turn and self.greeting_audio_complete and not self._closing_call:
                        self._empty_turn_nudge_count += 1
                        if self._empty_turn_nudge_count <= 3:
                            self.log.warn(f"Empty turn, nudging AI ({self._empty_turn_nudge_count}/3)")
                            asyncio.create_task(self._send_silence_nudge())
                    else:
                        self._empty_turn_nudge_count = 0

                    # Hot-swap session split - count non-empty turns
                    if not is_empty_turn:
                        self._turns_since_reconnect += 1

                    # Pre-warm standby at turn N-1 (skip if call is ending)
                    prewarm_turn = self._session_split_interval - 1
                    if (self._turns_since_reconnect == prewarm_turn
                            and not self._standby_ws
                            and not self._prewarm_task
                            and not self._closing_call
                            and not self.agent_said_goodbye
                            and self.greeting_audio_complete):
                        self._prewarm_task = asyncio.create_task(self._prewarm_standby_connection())

                    # Hot-swap at turn N (skip if call is ending)
                    if (self._turns_since_reconnect >= self._session_split_interval
                            and not is_empty_turn
                            and not self._closing_call
                            and not self._goodbye_pending
                            and not self.agent_said_goodbye
                            and self.greeting_audio_complete):
                        asyncio.create_task(self._hot_swap_session())

                    # Reset turn audio counter
                    self._current_turn_audio_chunks = 0

                    # Process deferred goodbye detection (agent finished speaking)
                    if self._goodbye_pending and not self._closing_call:
                        self._goodbye_pending = False
                        self.log.detail("Agent goodbye detected")
                        self.agent_said_goodbye = True
                        self._check_mutual_goodbye()

                if sc.get("interrupted"):
                    logger.debug(f"[{self.call_uuid[:8]}] AI interrupted")
                    # Drain queued audio that hasn't been sent yet
                    while not self._plivo_send_queue.empty():
                        try:
                            self._plivo_send_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    if self.plivo_ws:
                        await self.plivo_ws.send_text(json.dumps({"event": "clearAudio", "stream_id": self.stream_id}))

                # Capture user speech transcription from Gemini
                # Handle both field names: inputTranscription (current API) and inputTranscript (legacy)
                transcription_data = sc.get("inputTranscription") or sc.get("inputTranscript")
                if transcription_data:
                    # Can be a dict {"text": "..."} or a plain string
                    if isinstance(transcription_data, dict):
                        user_text = transcription_data.get("text", "")
                    else:
                        user_text = str(transcription_data)
                    logger.debug(f"[{self.call_uuid[:8]}] Input transcript: {user_text}")
                    if user_text and user_text.strip():
                        user_text = user_text.strip()

                        # Filter out noise/silence markers - NOT real speech
                        is_noise = user_text.startswith('<') and user_text.endswith('>')
                        if not is_noise:
                            self._last_user_speech_time = time.time()  # Track for latency
                            self._last_user_transcript_time = time.time()
                            logger.debug(f"[{self.call_uuid[:8]}] USER fragment: {user_text}")
                            self._current_turn_user_text.append(user_text)
                            self._save_transcript("USER", user_text)
                            self._log_conversation("user", user_text)
                            # Track if user said goodbye
                            if self._is_goodbye_message(user_text):
                                logger.debug(f"[{self.call_uuid[:8]}] User goodbye detected")
                                self.user_said_goodbye = True
                                self._check_mutual_goodbye()

                # Handle AI speech transcription (outputTranscription)
                output_transcription = sc.get("outputTranscription")
                if output_transcription:
                    if isinstance(output_transcription, dict):
                        ai_text = output_transcription.get("text", "")
                    else:
                        ai_text = str(output_transcription)
                    if ai_text and ai_text.strip():
                        ai_text = ai_text.strip()
                        logger.debug(f"[{self.call_uuid[:8]}] AGENT fragment: {ai_text}")
                        self._current_turn_agent_text.append(ai_text)
                        self._save_transcript("AGENT", ai_text)
                        self._log_conversation("model", ai_text)
                        # Defer goodbye detection to turnComplete (avoid cutting call mid-sentence)
                        if not self._closing_call and self._is_goodbye_message(ai_text):
                            self._goodbye_pending = True

                if "modelTurn" in sc:
                    parts = sc.get("modelTurn", {}).get("parts", [])
                    for p in parts:
                        if p.get("inlineData", {}).get("data"):
                            audio = p["inlineData"]["data"]
                            audio_bytes = base64.b64decode(audio)
                            # Track audio chunks for empty turn detection
                            self._current_turn_audio_chunks += 1
                            # Track turn start time and TTFB
                            if self._current_turn_audio_chunks == 1:
                                self._turn_start_time = time.time()
                                self._turn_first_byte_time = time.time()
                                self._agent_speaking = True
                                self._user_speaking = False
                                # Log greeting TTFB (trigger → first audio)
                                if self._greeting_trigger_time and not self._first_audio_time:
                                    self._first_audio_time = time.time()
                                    ttfb_ms = (self._first_audio_time - self._greeting_trigger_time) * 1000
                                    self.log.detail(f"Greeting TTFB: {ttfb_ms:.0f}ms")
                            # Record AI audio (24kHz)
                            self._record_audio("AI", audio_bytes, 24000)

                            # Latency check - only log if slow (> threshold)
                            if self._last_user_speech_time:
                                latency_ms = (time.time() - self._last_user_speech_time) * 1000
                                if latency_ms > LATENCY_THRESHOLD_MS:
                                    self.log.warn(f"Slow response: {latency_ms:.0f}ms")
                                self._last_user_speech_time = None  # Reset after first response

                            # During preload (no plivo_ws yet), always store audio
                            if not self.plivo_ws:
                                self.preloaded_audio.append(audio)
                            elif self.plivo_ws:
                                # Send directly to plivo_send_queue
                                chunk = AudioChunk(
                                    audio_b64=audio,
                                    turn_id=self._current_turn_id,
                                    sample_rate=24000
                                )
                                try:
                                    self._plivo_send_queue.put_nowait(chunk)
                                except asyncio.QueueFull:
                                    logger.warning(f"[{self.call_uuid[:8]}] plivo_send_queue full, dropping chunk")
                                # Log first chunk for this turn
                                if self._current_turn_audio_chunks == 1:
                                    logger.debug(f"[{self.call_uuid[:8]}] Audio -> Plivo send queue")
                        if p.get("text"):
                            ai_text = p["text"].strip()
                            logger.debug(f"AI TEXT: {ai_text[:100]}...")
                            # Only save actual speech, not thinking/planning text
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
                                self._current_turn_agent_text.append(ai_text)
                                self._save_transcript("AGENT", ai_text)
                                self._log_conversation("model", ai_text)
                                # Defer goodbye detection to turnComplete (avoid cutting call mid-sentence)
                                if not self._closing_call and self._is_goodbye_message(ai_text):
                                    self._goodbye_pending = True
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

            # Detect when user starts speaking (after agent finished)
            now = time.time()
            if self._last_user_audio_time is None or (now - self._last_user_audio_time) > 1.0:
                # Gap > 1 second means new user speech segment
                if self._agent_speaking or not self._user_speaking:
                    self._user_speaking = True
                    self._agent_speaking = False
                    self._user_speech_start_time = now
                    logger.debug(f"[{self.call_uuid[:8]}] User speaking")
            self._last_user_audio_time = now

            # Record user audio (16kHz)
            self._record_audio("USER", chunk, 16000)
            self.inbuffer.extend(chunk)
            chunks_sent = 0
            while len(self.inbuffer) >= self.BUFFER_SIZE:
                ac = self.inbuffer[:self.BUFFER_SIZE]
                msg = {"realtime_input": {"media_chunks": [{"mime_type": "audio/pcm;rate=16000", "data": base64.b64encode(bytes(ac)).decode()}]}}
                try:
                    # Send to main voice model (native audio)
                    await self.goog_live_ws.send(json.dumps(msg))
                    chunks_sent += 1
                    # Log first chunk sent to Gemini for this user speech
                    if chunks_sent == 1 and self._user_speaking:
                        logger.debug(f"[{self.call_uuid[:8]}] Sending user audio to Gemini")
                except Exception as send_err:
                    logger.error(f"Error sending audio to Google: {send_err} - triggering reconnect")
                    # Null out dead WS so subsequent audio gets buffered (line 1659)
                    self.goog_live_ws = None
                    self.inbuffer.clear()
                    # Buffer current audio for replay after reconnect
                    if len(self._reconnect_audio_buffer) < self._max_reconnect_buffer:
                        self._reconnect_audio_buffer.append(audio_b64)
                    # Trigger emergency session split (same as GoAway handling)
                    if not self._swap_in_progress and not self._closing_call:
                        asyncio.create_task(self._emergency_session_split())
                    return
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
        if not self.is_active:
            return

        self.is_active = False
        self.log.section("CALL ENDED")

        # Cancel all tasks
        for task in [self._timeout_task, self._silence_monitor_task,
                     self._sender_worker_task, self._standby_task,
                     self._prewarm_task, self._active_receive_task]:
            if task:
                task.cancel()

        # Close standby WS
        if self._standby_ws:
            await self._close_ws_quietly(self._standby_ws)
            self._standby_ws = None

        # Calculate call duration and log summary
        duration = 0
        if self.call_start_time:
            duration = (datetime.now() - self.call_start_time).total_seconds()
            mins = int(duration // 60)
            secs = duration % 60
            self.log.detail(f"Duration: {mins}m {secs:.0f}s | Turns: {self._turn_count}")
            if self._first_audio_time and self._greeting_trigger_time:
                ttfb = (self._first_audio_time - self._greeting_trigger_time) * 1000
                self.log.detail(f"Greeting TTFB: {ttfb:.0f}ms")
            if self._call_answered_time and self._preload_start_time:
                preload_total = (self._call_answered_time - self._preload_start_time) * 1000
                self.log.detail_last(f"Preload→Answer: {preload_total:.0f}ms")
            self._save_transcript("SYSTEM", f"Call duration: {duration:.1f}s, turns: {self._turn_count}")

        if self.goog_live_ws:
            try:
                await self.goog_live_ws.close()
            except Exception:
                pass
        if self._session_task:
            self._session_task.cancel()

        self._save_transcript("SYSTEM", "Call ended")

        # Stop recording thread
        if self._recording_queue:
            self._recording_queue.put(None)  # Shutdown signal
        if self._recording_thread:
            self._recording_thread.join(timeout=2.0)

        # Stop conversation logger thread
        if self._conversation_queue:
            self._conversation_queue.put(None)  # Shutdown signal
        if self._conversation_thread:
            self._conversation_thread.join(timeout=2.0)

        self._start_post_call_processing(duration)

    def _start_post_call_processing(self, duration: float):
        """Run all post-call processing (save, transcribe, DB update, webhook) in background thread"""
        def process_in_background():
            try:
                # Step 1: Save recording
                recording_info = self._save_recording()

                # Step 2: Transcribe (Gemini 2.0 Flash or Whisper)
                if recording_info and config.enable_whisper:
                    self._transcribe_recording_sync(recording_info, self.call_uuid)

                # Step 3: Build transcript text
                transcript = ""
                try:
                    transcript_dir = Path(__file__).parent.parent.parent / "transcripts"
                    final_transcript = transcript_dir / f"{self.call_uuid}_final.txt"
                    realtime_transcript = transcript_dir / f"{self.call_uuid}.txt"
                    if final_transcript.exists():
                        transcript = final_transcript.read_text()
                    elif realtime_transcript.exists():
                        transcript = realtime_transcript.read_text()
                except Exception:
                    pass
                # Fallback to in-memory transcript
                if not transcript.strip() and self._full_transcript:
                    transcript = "\n".join([
                        f"[{t['timestamp']}] {t['role']}: {t['text']}"
                        for t in self._full_transcript
                    ])

                # Step 4: Update session DB with final data
                session_db.update_call(
                    self.call_uuid,
                    status="completed",
                    ended_at=datetime.now().isoformat(),
                    duration_seconds=round(duration, 1),
                    transcript=transcript,
                )

                # Step 5: Call webhook AFTER everything is saved
                if self.webhook_url:
                    import asyncio as _asyncio
                    loop = _asyncio.new_event_loop()
                    _asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self._call_webhook(duration, transcript))
                    finally:
                        loop.close()

                logger.info(f"[{self.call_uuid[:8]}] Post-call processing complete")
            except Exception as e:
                logger.error(f"Post-call processing error: {e}")

        # Start background thread - call ends immediately, this runs separately
        processing_thread = threading.Thread(target=process_in_background, daemon=True)
        processing_thread.start()

    async def _call_webhook(self, duration: float, transcript: str = ""):
        """Call webhook URL with call data (transcript + basic info)"""
        try:
            import httpx

            payload = {
                "event": "call_ended",
                "call_uuid": self.call_uuid,
                "caller_phone": self.caller_phone,
                "contact_name": self.context.get("customer_name", ""),
                "client_name": self.client_name,
                "duration_seconds": round(duration, 1),
                "timestamp": datetime.now().isoformat(),
                # Transcript
                "transcript": transcript,
                "transcript_entries": self._full_transcript
            }

            self.log.detail(f"Webhook: {self.webhook_url}")
            t0 = time.time()
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(self.webhook_url, json=payload)
                wh_ms = (time.time() - t0) * 1000
                self.log.detail(f"Webhook response: {resp.status_code} ({wh_ms:.0f}ms)")
        except Exception as e:
            logger.error(f"Error calling webhook: {e}")


# Session storage with concurrency protection
MAX_CONCURRENT_SESSIONS = int(os.environ.get("MAX_CONCURRENT_SESSIONS", 50))
_sessions: Dict[str, PlivoGeminiSession] = {}
_preloading_sessions: Dict[str, PlivoGeminiSession] = {}
_sessions_lock = asyncio.Lock()

def get_active_session_count() -> int:
    """Return total active + preloading sessions (no lock, approximate)"""
    return len(_sessions) + len(_preloading_sessions)

def set_plivo_uuid(internal_uuid: str, plivo_uuid: str):
    """Set the Plivo UUID on a preloaded session for proper hangup"""
    # No lock needed: single-threaded asyncio, called from async context
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
    async with _sessions_lock:
        total = len(_sessions) + len(_preloading_sessions)
        if total >= MAX_CONCURRENT_SESSIONS:
            logger.warning(f"Max concurrent sessions ({MAX_CONCURRENT_SESSIONS}) reached. Rejecting {call_uuid}")
            raise Exception(f"Max concurrent sessions ({MAX_CONCURRENT_SESSIONS}) reached")
        session = PlivoGeminiSession(call_uuid, caller_phone, prompt=prompt, context=context, webhook_url=webhook_url)
        _preloading_sessions[call_uuid] = session
    success = await session.preload()
    return success

async def create_session(call_uuid: str, caller_phone: str, plivo_ws, prompt: str = None, context: dict = None, webhook_url: str = None) -> Optional[PlivoGeminiSession]:
    """Create or retrieve preloaded session"""
    async with _sessions_lock:
        # Check for preloaded session
        if call_uuid in _preloading_sessions:
            session = _preloading_sessions.pop(call_uuid)
            session.caller_phone = caller_phone
            session.attach_plivo_ws(plivo_ws)
            _sessions[call_uuid] = session
            logger.info(f"Using PRELOADED session for {call_uuid}")
            session._save_transcript("SYSTEM", "Call connected (preloaded)")
            return session

        # Fallback: create new session (check limit)
        total = len(_sessions) + len(_preloading_sessions)
        if total >= MAX_CONCURRENT_SESSIONS:
            logger.warning(f"Max concurrent sessions ({MAX_CONCURRENT_SESSIONS}) reached. Rejecting {call_uuid}")
            return None

    # Create outside lock (preload does network I/O)
    logger.info(f"No preloaded session, creating new for {call_uuid}")
    session = PlivoGeminiSession(call_uuid, caller_phone, prompt=prompt, context=context, webhook_url=webhook_url)
    session.plivo_ws = plivo_ws
    session._save_transcript("SYSTEM", "Call started")
    if await session.preload():
        async with _sessions_lock:
            _sessions[call_uuid] = session
        return session
    return None

async def get_session(call_uuid: str) -> Optional[PlivoGeminiSession]:
    async with _sessions_lock:
        return _sessions.get(call_uuid)

async def remove_session(call_uuid: str):
    """Remove and stop session atomically"""
    async with _sessions_lock:
        session = _sessions.pop(call_uuid, None)
        preload_session_obj = _preloading_sessions.pop(call_uuid, None)
    # Stop outside lock (stop() does async I/O)
    if session:
        await session.stop()
    if preload_session_obj:
        await preload_session_obj.stop()
