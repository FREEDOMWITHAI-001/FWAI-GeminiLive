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
from dataclasses import dataclass
from loguru import logger
from datetime import datetime
from pathlib import Path
import websockets
from src.core.config import config
from src.tools import execute_tool
from src.conversational_prompts import render_prompt
from src.question_flow import QuestionFlow, QuestionPipeline, QuestionPhase, get_or_create_flow, remove_flow
from src.db.session_db import session_db

# LATENCY OPT: Use orjson for 2-3x faster JSON operations (3-30ms savings per call)
try:
    import orjson
    def json_dumps(obj) -> str:
        """Fast JSON encoding with orjson"""
        return orjson.dumps(obj).decode('utf-8')
    def json_loads(s: str):
        """Fast JSON decoding with orjson"""
        return orjson.loads(s)
    logger.debug("Using orjson for fast JSON operations")
except ImportError:
    # Fallback to standard json
    json_dumps = json.dumps
    json_loads = json.loads
    logger.debug("Using standard json library")


def get_vertex_ai_token():
    """Get OAuth2 access token for Vertex AI (synchronous)"""
    try:
        import google.auth
        from google.auth.transport.requests import Request

        # Multiple scopes for Vertex AI Gemini Live API
        scopes = [
            'https://www.googleapis.com/auth/cloud-platform',
            'https://www.googleapis.com/auth/generative-language',
            'https://www.googleapis.com/auth/generative-language.retriever',
        ]
        credentials, project = google.auth.default(scopes=scopes)
        credentials.refresh(Request())
        logger.info(f"Got Vertex AI token for project: {project}")
        return credentials.token
    except Exception as e:
        logger.error(f"Failed to get Vertex AI token: {e}")
        return None


async def get_vertex_ai_token_async():
    """Async wrapper for Google Auth token refresh.
    LATENCY OPTIMIZATION: Runs blocking token refresh in thread pool executor.
    Prevents blocking event loop for 500-2000ms during reconnection."""
    import asyncio
    loop = asyncio.get_event_loop()
    # Run blocking function in thread pool executor (non-blocking)
    return await loop.run_in_executor(None, get_vertex_ai_token)

# Latency threshold - only log if slower than this (ms)
LATENCY_THRESHOLD_MS = 500

# Default instruction templates — all overridable via API
DEFAULT_INSTRUCTION_TEMPLATES = {
    "wrap_up": "[SYSTEM: Call time limit reached. Please politely wrap up the conversation now. Say a warm goodbye and end the call gracefully.]",
    "nudge_greeting": "The customer hasn't responded. Say ONLY 'Hello? Are you there?' and STOP. Do NOT ask any question. Do NOT move forward. Just wait.",
    "nudge_default": "The customer is quiet. Say ONLY 'Are you still there?' and STOP. Do NOT move to the next question. Do NOT say goodbye. Just wait for their answer.",
    "instruction_ask": '[INSTRUCTION] Ask this question naturally: "{text}"',
    "instruction_end_call": '[INSTRUCTION] Say naturally: "{text}" Then call end_call.',
    "greeting_trigger": "Start the call now. Greet the customer.",
    "greeting_simple": "Hi",
    "reconnect_continue": '[INSTRUCTION] Connection was briefly interrupted. Say "Sorry about that, where were we?" then say naturally: "{text}" Then STOP and wait for the customer to respond.',
    "reconnect_goodbye": "[INSTRUCTION] Connection restored. Say 'Sorry about that brief interruption. Great talking to you! Take care!' and use end_call tool.",
    "reconnect_simple": "[Continue the conversation]"
}

# Recording directory
RECORDINGS_DIR = Path(__file__).parent.parent.parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

# Prompts directory for client-specific prompts
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
PROMPTS_DIR.mkdir(exist_ok=True)


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
        "description": "End the phone call. ONLY call this AFTER you have asked ALL questions in your script AND said goodbye. NEVER call this while questions remain. NEVER call this just because the user paused or was quiet.",
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


# Off-script detection patterns (AI responding to noise instead of following instructions)
OFF_SCRIPT_PHRASES = [
    "that's good to hear", "glad to hear", "sounds like",
    "i see", "absolutely", "sure thing", "tell me more",
    "would you like", "how about", "that's great",
    "that's wonderful", "that's nice", "good to know",
    "interesting", "i understand", "of course",
    "no problem", "certainly", "definitely",
    "that makes sense", "got it", "right",
    "so tell me", "so what", "anyway",
]


class PlivoGeminiSession:
    def __init__(self, call_uuid: str, caller_phone: str, prompt: str = None, context: dict = None, webhook_url: str = None, client_name: str = "fwai", use_question_flow: bool = True, questions_override: list = None, prompt_override: str = None, objections_override: dict = None, objection_keywords_override: dict = None, instruction_templates: dict = None):
        self.call_uuid = call_uuid  # Internal UUID
        self.plivo_call_uuid = None  # Plivo's actual call UUID (set later)
        self.caller_phone = caller_phone
        self.context = context or {}  # Context for templates (customer_name, course_name, etc.)
        self.client_name = client_name or "fwai"
        self.use_question_flow = use_question_flow  # Use built-in question flow state machine
        # Merge API-provided instruction templates over defaults
        self._templates = {**DEFAULT_INSTRUCTION_TEMPLATES, **(instruction_templates or {})}

        # Question Flow Mode: Use minimal prompt + inject questions one by one
        if use_question_flow:
            # Create flow with client config - loads questions, voice, etc from config file
            # If overrides provided (from n8n API), those take priority over config
            self._question_flow = get_or_create_flow(
                call_uuid=call_uuid,
                client_name=self.client_name,
                context=self.context,
                questions_override=questions_override,
                prompt_override=prompt_override or prompt,
                objections_override=objections_override,
                objection_keywords_override=objection_keywords_override
            )
            self.prompt = self._question_flow.get_base_prompt()
            # Get voice from config (not hardcoded)
            self._config_voice = self._question_flow.get_voice()
            logger.info(f"[{call_uuid[:8]}] QuestionFlow: client={self.client_name}, voice={self._config_voice}")
        else:
            # Non-QuestionFlow mode: prompt must be provided via API
            self._question_flow = None
            self._config_voice = None
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
        self._user_speech_complete = asyncio.Event()  # Event-driven silence detection (LATENCY OPT)

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
        self._silence_detection_task = None  # Task that triggers after user stops speaking (LATENCY OPT)
        self._current_turn_audio_chunks = 0  # Track audio chunks in current turn
        self._empty_turn_nudge_count = 0  # Track consecutive empty turns
        self._turn_start_time = None  # Track when current turn started (for latency logging)
        self._turn_count = 0  # Count turns for latency tracking
        self._current_turn_agent_text = []  # Accumulate agent speech fragments per turn
        self._current_turn_user_text = []  # Accumulate user speech fragments per turn
        self._last_question_time = 0  # Cooldown between question injections
        self._wait_timeout = 15.0  # Wait 15 seconds for user response before nudging

        # Question Pipeline: replaces scattered boolean flags with lifecycle state machine
        # Manages gate_open, echo protection, turn counting, user transcript accumulation
        if use_question_flow and self._question_flow:
            self._pipeline = QuestionPipeline(
                call_uuid=call_uuid,
                total_questions=len(self._question_flow.questions),
            )
        else:
            self._pipeline = None

        # Queue-based audio pipeline (3 workers connected by asyncio.Queue)
        # Gemini → audio_out_queue → gate_worker → plivo_send_queue → sender → Plivo
        self._audio_out_queue = asyncio.Queue(maxsize=200)      # Gemini audio → gate worker
        self._plivo_send_queue = asyncio.Queue(maxsize=200)     # Gate worker → Plivo sender
        self._transcript_val_queue = asyncio.Queue(maxsize=50)  # Output transcription → validator
        self._blocked_turns: set = set()   # Turn IDs blocked by validator (off-script)
        self._current_turn_id = 0          # Incremented on each Gemini turn
        # Worker task handles (started in attach_plivo_ws, cancelled in stop)
        self._gate_worker_task = None
        self._sender_worker_task = None
        self._validator_worker_task = None

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

        # LATENCY OPT: Pre-compute question word sets for off-script detection (40-200ms savings)
        self._question_word_sets = []
        if use_question_flow and self._question_flow:
            stop_words = {"the", "a", "an", "is", "are", "to", "and", "of", "in", "for",
                         "you", "your", "i", "my", "we", "our", "it", "that", "this",
                         "do", "does", "can", "could", "would", "will", "how", "what",
                         "so", "well", "just", "about", "have", "has", "been", "was"}
            for q in self._question_flow.questions:
                words = set(q["prompt"].lower().split()) - stop_words
                self._question_word_sets.append(words)
            logger.debug(f"[{call_uuid[:8]}] LATENCY OPT: Pre-computed {len(self._question_word_sets)} question word sets")

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

    async def _load_conversation_from_file(self) -> list:
        """Load conversation history from file for reconnection.
        LATENCY OPTIMIZATION: Async file I/O prevents blocking event loop during reconnection."""
        try:
            import aiofiles
            if self._conversation_file.exists():
                async with aiofiles.open(self._conversation_file, 'r') as f:
                    content = await f.read()
                    return json_loads(content)
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

    async def _monitor_answer_webhook_timeout(self):
        """Monitor for missing answer webhook - log warning if stream never connects"""
        await asyncio.sleep(10)  # Wait 10 seconds after preload
        if not self.plivo_ws:
            logger.error(f"[{self.call_uuid[:8]}] ⚠️  CRITICAL: Answer webhook never called!")
            logger.error(f"[{self.call_uuid[:8]}] Check Plivo dashboard answer URL: {config.plivo_callback_url}/plivo/answer")
            logger.error(f"[{self.call_uuid[:8]}] Verify ngrok is running: curl http://127.0.0.1:4040/api/tunnels")
            logger.error(f"[{self.call_uuid[:8]}] Test webhook manually: curl -X POST http://localhost:3001/plivo/answer")

    async def preload(self):
        """Preload the Gemini session while phone is ringing"""
        try:
            logger.debug(f"[{self.call_uuid[:8]}] Preloading Gemini session")
            self.is_active = True
            # Start main voice session (native audio)
            self._session_task = asyncio.create_task(self._run_google_live_session())
            # Wait for setup to complete (with timeout - 8s max for better greeting)
            try:
                await asyncio.wait_for(self._preload_complete.wait(), timeout=8.0)
                logger.info(f"[{self.call_uuid[:8]}] AI preloaded ({len(self.preloaded_audio)} chunks)")
                logger.warning(f"[{self.call_uuid[:8]}] Waiting for Plivo to call answer webhook at {config.plivo_callback_url}/plivo/answer")
            except asyncio.TimeoutError:
                logger.warning(f"[{self.call_uuid[:8]}] Preload timeout, continuing with {len(self.preloaded_audio)} chunks")
            # Start answer webhook timeout monitor
            asyncio.create_task(self._monitor_answer_webhook_timeout())
            return True
        except Exception as e:
            logger.error(f"Failed to preload session: {e}")
            return False

    def attach_plivo_ws(self, plivo_ws):
        """Attach Plivo WebSocket when user answers"""
        self.plivo_ws = plivo_ws
        self.call_start_time = datetime.now()
        preload_count = len(self.preloaded_audio)
        logger.info(f"[{self.call_uuid[:8]}] Call answered, {preload_count} chunks ready")
        # Reset pipeline deliver_time to NOW (not preload time) so timeouts start
        # from when user actually hears the greeting, not when it was generated
        if self._pipeline and self._pipeline._current:
            old_time = self._pipeline._current.deliver_time
            self._pipeline._current.deliver_time = time.time()
            self._pipeline._current.turns_since_inject = 0
            logger.debug(f"[{self.call_uuid[:8]}] Pipeline: reset deliver_time "
                        f"(was {time.time() - old_time:.1f}s ago, now=0s)")
        # Send any preloaded audio immediately
        if self.preloaded_audio:
            asyncio.create_task(self._send_preloaded_audio())
        else:
            logger.warning(f"[{self.call_uuid[:8]}] No preloaded audio - greeting will lag")
        # Start queue pipeline workers
        self._gate_worker_task = asyncio.create_task(self._audio_gate_worker())
        self._sender_worker_task = asyncio.create_task(self._plivo_sender_worker())
        self._validator_worker_task = asyncio.create_task(self._transcript_validator_worker())
        # Start call duration timer
        self._timeout_task = asyncio.create_task(self._monitor_call_duration())
        # Start silence monitor (3 second SLA)
        self._silence_monitor_task = asyncio.create_task(self._monitor_silence())

    async def _send_preloaded_audio(self):
        """Send preloaded audio directly to plivo_send_queue (bypass gate — it IS the question)"""
        logger.debug(f"[{self.call_uuid[:8]}] Sending {len(self.preloaded_audio)} preloaded chunks via queue")
        for audio in self.preloaded_audio:
            chunk = AudioChunk(audio_b64=audio, turn_id=0, sample_rate=24000)
            try:
                self._plivo_send_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                logger.warning(f"[{self.call_uuid[:8]}] plivo_send_queue full during preload send")
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
                    # Log buffer stats for debugging latency
                    logger.info(f"[{self.call_uuid[:8]}] ⏱ Call in progress: {elapsed}s")
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
                        "parts": [{"text": self._templates["wrap_up"]}]
                    }],
                    "turn_complete": True
                }
            }
            await self.goog_live_ws.send(json_dumps(msg))
            logger.info("Sent wrap-up message to AI")
            self._save_transcript("SYSTEM", "Call time limit - wrapping up")
        except Exception as e:
            logger.error(f"Error sending wrap-up message: {e}")

    async def _monitor_silence(self):
        """Monitor for silence - detect when user finishes speaking and process their response.
        EVENT-DRIVEN OPTIMIZATION: Waits for event instead of polling every 100ms.
        Reduces latency from 0-100ms (polling) to <5ms (event notification)."""
        try:
            # Background task for periodic checks (nudging, etc.)
            last_periodic_check = time.time()
            periodic_check_interval = 1.0  # Check every 1 second for nudging logic

            while self.is_active and not self._closing_call:
                # Wait for user speech complete event OR timeout for periodic checks
                try:
                    await asyncio.wait_for(
                        self._user_speech_complete.wait(),
                        timeout=periodic_check_interval
                    )
                    # Event fired - user finished speaking
                    self._user_speech_complete.clear()

                    # QuestionFlow mode: Process user response
                    if self._pipeline and self._pipeline.waiting_for_user:
                        q_time = self._pipeline.question_asked_time
                        if self._last_user_transcript_time > q_time:
                            silence_since_speech = time.time() - self._last_user_transcript_time
                            logger.info(f"[{self.call_uuid[:8]}]   ├─ User finished speaking (silence={silence_since_speech:.1f}s) [EVENT]")
                            await self._process_user_audio_for_transcription()

                except asyncio.TimeoutError:
                    # Timeout - do periodic checks (nudging, echo expiration, etc.)
                    current_time = time.time()

                    if self._pipeline and self._pipeline.waiting_for_user:
                        # Auto-transition ECHOING → LISTENING if echo buffer expired
                        self._pipeline.check_echo_expired()

                        # Periodic logging for debugging
                        if self._pipeline.is_listening:
                            listen_time = self._pipeline._current.listen_time if self._pipeline._current else 0
                            time_in_listening = current_time - listen_time if listen_time else 0
                            if time_in_listening >= 30.0:
                                pending = self._pipeline.pending_user_transcript
                                logger.debug(f"[{self.call_uuid[:8]}] Silence monitor: {time_in_listening:.0f}s in LISTENING "
                                             f"| waiting for user to speak | pending='{pending[:30] if pending else ''}'")

                    # Original silence monitoring for non-QuestionFlow mode (SLA nudging)
                    if self._last_user_speech_time is not None:
                        silence_duration = current_time - self._last_user_speech_time
                        if silence_duration >= self._silence_sla_seconds:
                            logger.warning(f"[{self.call_uuid[:8]}] {silence_duration:.1f}s silence - nudging AI")
                            await self._send_silence_nudge()
                            self._last_user_speech_time = None

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in silence monitor: {e}")

    async def _send_silence_nudge(self):
        """Send a nudge to AI when silence detected"""
        if not self.goog_live_ws or self._closing_call:
            logger.debug(f"[{self.call_uuid[:8]}] Nudge: skipped (ws={bool(self.goog_live_ws)}, closing={self._closing_call})")
            return

        # In QuestionFlow mode: send nudge and reset pipeline to DELIVERING
        if self._pipeline:
            if not self._pipeline.waiting_for_user:
                logger.debug(f"[{self.call_uuid[:8]}] Nudge: skipped (not waiting_for_user)")
                return
            time_since_q = time.time() - self._pipeline.question_asked_time
            if time_since_q < self._wait_timeout:
                logger.debug(f"[{self.call_uuid[:8]}] Nudge: skipped (only {time_since_q:.1f}s < {self._wait_timeout}s)")
                return
            # Use transcript time (not audio frame time) to detect real speech
            no_user_speech = (self._last_user_transcript_time == 0 or
                              self._last_user_transcript_time < self._pipeline.question_asked_time)
            pending = self._pipeline.pending_user_transcript
            if not no_user_speech or pending:
                logger.debug(f"[{self.call_uuid[:8]}] Nudge: skipped (user has speech or pending)")
                return
            logger.info(f"[{self.call_uuid[:8]}]   ├─ Nudge: no response for {time_since_q:.0f}s, prompting user")
            nudge_count = self._pipeline.reset_for_nudge()  # DELIVERING state re-opens gate

            # Max 3 nudges per question — after that, just keep waiting silently
            # Never auto-advance: wait until user actually speaks or call times out
            if nudge_count > 3:
                q_idx = self._pipeline._current.index if self._pipeline._current else '?'
                logger.info(f"[{self.call_uuid[:8]}]   ├─ Nudge: max reached for Q{q_idx}, waiting silently")
                return

            # Gentle nudge — re-engage user without advancing to next question
            q_idx = self._pipeline._current.index if self._pipeline._current else 0
            if q_idx == 0:
                nudge_text = self._templates["nudge_greeting"]
            else:
                nudge_text = self._templates["nudge_default"]
            prompt_msg = {
                "client_content": {
                    "turns": [{
                        "role": "user",
                        "parts": [{"text": nudge_text}]
                    }],
                    "turn_complete": True
                }
            }
            await self.goog_live_ws.send(json_dumps(prompt_msg))
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
            await self.goog_live_ws.send(json_dumps(msg))
            logger.debug(f"[{self.call_uuid[:8]}] Sent nudge to AI")
        except Exception as e:
            logger.error(f"Error sending silence nudge: {e}")

    async def _trigger_silence_detection_after_delay(self, delay_seconds: float = 0.05):
        """Trigger silence detection after a delay. Optimized for low latency (~50ms).
        Called when user speech is detected - waits for silence, then signals the monitor."""
        try:
            await asyncio.sleep(delay_seconds)
            # If we reach here, user has stopped speaking (no new speech cancelled this task)
            self._user_speech_complete.set()
        except asyncio.CancelledError:
            # New speech came in, this task was cancelled - do nothing
            pass

    def _schedule_silence_detection(self):
        """Schedule silence detection task, cancelling any previous one.
        Event-driven approach: reduces latency from 0-100ms (polling) to <5ms (event)."""
        # Cancel existing silence detection task if any
        if self._silence_detection_task and not self._silence_detection_task.done():
            self._silence_detection_task.cancel()

        # Schedule new silence detection (triggers after 50ms of no speech)
        self._silence_detection_task = asyncio.create_task(
            self._trigger_silence_detection_after_delay(0.05)
        )

    # ==================== QUEUE-BASED AUDIO PIPELINE ====================

    def _get_remaining_question_texts(self) -> list[str]:
        """Get all remaining question prompt texts from the question flow."""
        if not self._question_flow:
            return []
        step = self._question_flow.current_step
        questions = self._question_flow.questions
        return [q["prompt"] for q in questions[step:]]

    def _is_off_script(self, ai_text: str, remaining_questions: list[str]) -> bool:
        """Detect off-script AI speech using pattern matching.
        Conservative: false negatives OK, false positives NOT OK.
        LATENCY OPT: Uses pre-computed question word sets (40-200ms savings).

        Only flags content as off-script when Gemini is hallucinating from noise.
        NEVER flag when Gemini is responding to actual user speech or handling objections.
        """
        if not ai_text or not remaining_questions:
            return False

        text_lower = ai_text.lower().strip()

        # Need substantial text before judging
        if len(text_lower) < 100:
            return False

        # If there was recent user speech, Gemini is RESPONDING — never flag
        if self._pipeline and self._pipeline.pending_user_transcript:
            return False

        # Allow objection responses (price, cost, discount, EMI, busy, etc.)
        objection_words = {"price", "cost", "40k", "discount", "emi", "busy",
                           "interested", "family", "exam", "youtube", "free"}
        if objection_words & set(text_lower.split()):
            return False

        stop_words = {"the", "a", "an", "is", "are", "to", "and", "of", "in", "for",
                      "you", "your", "i", "my", "we", "our", "it", "that", "this",
                      "do", "does", "can", "could", "would", "will", "how", "what",
                      "so", "well", "just", "about", "have", "has", "been", "was"}
        ai_words = set(text_lower.split()) - stop_words

        # Check against ALL remaining questions — if ANY match, it's on-script
        # LATENCY OPT: Use pre-computed word sets instead of recomputing
        if self._question_word_sets:
            current_step = self._question_flow.current_step if self._question_flow else 0
            for i in range(current_step, len(self._question_word_sets)):
                expected_words = self._question_word_sets[i]
                if expected_words:
                    overlap = len(expected_words & ai_words) / len(expected_words)
                    if overlap >= 0.2:
                        return False
        else:
            # Fallback to dynamic computation if word sets not available
            for question in remaining_questions:
                expected_words = set(question.lower().split()) - stop_words
                if expected_words:
                    overlap = len(expected_words & ai_words) / len(expected_words)
                    if overlap >= 0.2:
                        return False

        # Only flag pure filler with NO question content after 150+ chars
        if len(text_lower) >= 150:
            for phrase in OFF_SCRIPT_PHRASES:
                if text_lower.startswith(phrase):
                    logger.info(f"[{self.call_uuid[:8]}]   ├─ Off-script blocked: '{ai_text[:60]}'")
                    return True

        return False

    async def _cancel_current_audio(self, turn_id: int, reason: str):
        """Cancel audio for a turn: block it, drain send queue, clear Plivo playback"""
        self._blocked_turns.add(turn_id)
        logger.debug(f"[{self.call_uuid[:8]}] CANCEL turn {turn_id}: {reason}")

        # Drain plivo_send_queue (drop all pending audio)
        drained = 0
        while not self._plivo_send_queue.empty():
            try:
                self._plivo_send_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained:
            logger.debug(f"[{self.call_uuid[:8]}] Drained {drained} chunks from plivo_send_queue")

        # Send clearAudio to Plivo to stop current playback
        if self.plivo_ws:
            try:
                await self.plivo_ws.send_text(json_dumps({
                    "event": "clearAudio",
                    "stream_id": self.stream_id
                }))
                logger.debug(f"[{self.call_uuid[:8]}] Sent clearAudio to Plivo")
            except Exception as e:
                logger.error(f"[{self.call_uuid[:8]}] Error sending clearAudio: {e}")

        # Close gate via pipeline
        if self._pipeline:
            self._pipeline.close_gate_from_validator()

    async def _audio_gate_worker(self):
        """Worker 1: Reads AudioChunks from audio_out_queue, forwards approved ones to plivo_send_queue"""
        logger.debug(f"[{self.call_uuid[:8]}] Gate worker started")
        try:
            while self.is_active:
                try:
                    chunk: AudioChunk = await asyncio.wait_for(
                        self._audio_out_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                # Drop if turn is blocked by off-script validator
                if chunk.turn_id in self._blocked_turns:
                    continue

                # Gemini drives question flow naturally (all questions in system prompt).
                # Let audio flow freely — gate is only used for off-script blocking.
                try:
                    self._plivo_send_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    logger.warning(f"[{self.call_uuid[:8]}] plivo_send_queue full, dropping chunk")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[{self.call_uuid[:8]}] Gate worker error: {e}")
        logger.debug(f"[{self.call_uuid[:8]}] Gate worker stopped")

    async def _plivo_sender_worker(self):
        """Worker 2: Reads from plivo_send_queue, sends to Plivo WebSocket"""
        logger.debug(f"[{self.call_uuid[:8]}] Plivo sender worker started")
        try:
            while self.is_active:
                try:
                    chunk: AudioChunk = await asyncio.wait_for(
                        self._plivo_send_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                if not self.plivo_ws:
                    continue

                try:
                    await self.plivo_ws.send_text(json_dumps({
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

    async def _transcript_validator_worker(self):
        """Worker 3: Validates AI output transcription, cancels off-script audio"""
        logger.debug(f"[{self.call_uuid[:8]}] Transcript validator worker started")
        accumulated_text = ""
        current_turn = -1
        try:
            while self.is_active:
                try:
                    event = await asyncio.wait_for(
                        self._transcript_val_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                turn_id = event.get("turn_id", 0)
                text = event.get("text", "")

                # Reset accumulator on new turn
                if turn_id != current_turn:
                    accumulated_text = ""
                    current_turn = turn_id

                accumulated_text = f"{accumulated_text} {text}".strip()

                # Skip validation if no pipeline
                if not self._pipeline:
                    continue

                # Only validate during DELIVERING phase (gate open)
                if not self._pipeline.gate_open:
                    continue

                # Skip if turn already blocked
                if turn_id in self._blocked_turns:
                    continue

                # Check against ALL remaining questions (not just current)
                remaining = self._get_remaining_question_texts()
                if not remaining:
                    continue

                # Check for off-script content
                if self._is_off_script(accumulated_text, remaining):
                    await self._cancel_current_audio(
                        turn_id,
                        f"Off-script: '{accumulated_text[:60]}'"
                    )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[{self.call_uuid[:8]}] Transcript validator error: {e}")
        logger.debug(f"[{self.call_uuid[:8]}] Transcript validator worker stopped")

    async def _send_reconnection_filler(self):
        """Handle silence during reconnection - clear audio and prepare for resume"""
        if not self.plivo_ws or self._closing_call:
            return
        try:
            logger.debug(f"[{self.call_uuid[:8]}] Preparing for reconnection")

            # Clear any pending audio to prevent stale data
            await self.plivo_ws.send_text(json_dumps({
                "event": "clearAudio",
                "stream_id": self.stream_id
            }))

            # The AI will say "Sorry, brief connection issue..." after reconnecting
            # This is handled in _send_session_setup via the reconnection prompt

        except Exception as e:
            logger.error(f"Error in reconnection filler: {e}")

    async def _run_google_live_session(self):
        # Choose between Vertex AI (regional, lower latency) or Google AI Studio
        if config.use_vertex_ai:
            # Vertex AI Live API - regional endpoint (asia-south1 = Mumbai)
            # LATENCY OPT: Use async token refresh (prevents 500-2000ms blocking)
            token = await get_vertex_ai_token_async()
            if not token:
                logger.error("Failed to get Vertex AI token - falling back to Google AI Studio")
                url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={config.google_api_key}"
                extra_headers = None
            else:
                url = f"wss://{config.vertex_location}-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
                extra_headers = {"Authorization": f"Bearer {token}"}
                logger.info(f"Using Vertex AI endpoint: {config.vertex_location}")
        else:
            # Google AI Studio - global endpoint
            url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={config.google_api_key}"
            extra_headers = None

        reconnect_attempts = 0
        max_reconnects = 5  # Increased for better stability

        while self.is_active and reconnect_attempts < max_reconnects:
            try:
                # Refresh token on reconnect for Vertex AI
                # LATENCY OPT: Async token refresh prevents blocking event loop for 500-2000ms
                if config.use_vertex_ai and reconnect_attempts > 0:
                    token = await get_vertex_ai_token_async()
                    if token:
                        extra_headers = {"Authorization": f"Bearer {token}"}

                ws_kwargs = {"ping_interval": 30, "ping_timeout": 20, "close_timeout": 5}
                if extra_headers:
                    # Use additional_headers for newer websockets versions
                    ws_kwargs["additional_headers"] = extra_headers

                async with websockets.connect(url, **ws_kwargs) as ws:
                    self.goog_live_ws = ws
                    reconnect_attempts = 0  # Reset on successful connect
                    logger.info(f"[{self.call_uuid[:8]}] Gemini connected")
                    await self._send_session_setup()
                    # Flush any buffered audio from reconnection
                    if self._reconnect_audio_buffer:
                        logger.debug(f"[{self.call_uuid[:8]}] Flushing {len(self._reconnect_audio_buffer)} buffered chunks")
                        for buffered_audio in self._reconnect_audio_buffer:
                            await self.handle_plivo_audio(buffered_audio)
                        self._reconnect_audio_buffer = []
                    async for message in ws:
                        if not self.is_active:
                            break
                        await self._receive_from_google(message)
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"[{self.call_uuid[:8]}] Gemini closed: {e.code}")
                if self.is_active and not self._closing_call:
                    self._is_reconnecting = True
                    reconnect_attempts += 1
                    logger.info(f"[{self.call_uuid[:8]}] Reconnecting ({reconnect_attempts}/{max_reconnects})")
                    # Send filler message to user while reconnecting
                    asyncio.create_task(self._send_reconnection_filler())
                    await asyncio.sleep(0.2)  # Faster reconnect (was 0.5)
                    continue
            except Exception as e:
                logger.error(f"Google Live error: {e}")
                if self.is_active and not self._closing_call:
                    reconnect_attempts += 1
                    logger.info(f"[{self.call_uuid[:8]}] Reconnecting ({reconnect_attempts}/{max_reconnects})")
                    await asyncio.sleep(0.2)  # Faster reconnect (was 0.5)
                    continue
            break  # Normal exit

        self.goog_live_ws = None
        logger.debug(f"[{self.call_uuid[:8]}] Session ended")

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

    async def _transcribe_audio_rest_api(self, audio_bytes: bytes) -> str:
        """Transcribe audio using Gemini REST API (gemini-2.0-flash)"""
        if not audio_bytes or len(audio_bytes) < 1600:  # Skip if less than 0.1 second of audio
            return None

        try:
            import httpx

            # Gemini REST API endpoint
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={config.google_api_key}"

            # Convert PCM to WAV format (Gemini REST API needs proper audio format)
            wav_bytes = self._pcm_to_wav(audio_bytes, sample_rate=16000)

            # Encode audio as base64
            audio_b64 = base64.standard_b64encode(wav_bytes).decode("utf-8")

            # Build request payload
            payload = {
                "contents": [{
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": audio_b64
                            }
                        },
                        {
                            "text": "Transcribe this audio. Output ONLY the exact words spoken, nothing else. No timestamps, no labels."
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 512
                }
            }

            logger.debug(f"[{self.call_uuid[:8]}] Transcribing {len(wav_bytes)} bytes")

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(api_url, json=payload)

                if response.status_code == 200:
                    result = response.json()
                    try:
                        text = result["candidates"][0]["content"]["parts"][0]["text"]
                        transcription = text.strip()
                        if transcription:
                            logger.debug(f"[{self.call_uuid[:8]}] Transcribed: {transcription}")
                            return transcription
                    except (KeyError, IndexError) as e:
                        logger.warning(f"[{self.call_uuid[:8]}] Transcription parse error: {e}")
                        return None
                else:
                    logger.warning(f"[{self.call_uuid[:8]}] Transcription API error: {response.status_code} - {response.text[:200]}")
                    return None

        except Exception as e:
            logger.error(f"[{self.call_uuid[:8]}] REST transcription error: {e}")
            return None

    async def _process_user_audio_for_transcription(self):
        """Process user response and advance question flow (transcription disabled - done at end)"""
        if not self.use_question_flow:
            return

        # Pipeline mode: use pipeline state instead of scattered flags
        if self._pipeline and self._question_flow:
            logger.debug(f"[{self.call_uuid[:8]}] _process_user_audio: entry | {self._pipeline.dump_state()}")

            # Only process if pipeline is listening
            if not self._pipeline.waiting_for_user:
                self._user_audio_buffer = bytearray(b"")
                logger.debug(f"[{self.call_uuid[:8]}] _process_user_audio: not waiting for user, clearing buffer "
                             f"| phase={self._pipeline.current_phase}")
                return

            # Don't process if question was asked very recently (< 2 seconds ago)
            time_since_question = time.time() - self._pipeline.question_asked_time
            if time_since_question < 2.0:
                logger.debug(f"[{self.call_uuid[:8]}] _process_user_audio: too soon ({time_since_question:.1f}s < 2.0s)")
                return

            # Require a real transcript before advancing (prevents noise auto-advances)
            pending_transcript = self._pipeline.pending_user_transcript.strip()
            if not pending_transcript:
                if len(self._user_audio_buffer) > 0:
                    logger.debug(f"[{self.call_uuid[:8]}] _process_user_audio: no transcript, clearing "
                                 f"{len(self._user_audio_buffer)} bytes audio buffer")
                    self._user_audio_buffer = bytearray(b"")
                return

            # Capture response: LISTENING → CAPTURED
            logger.info(f"[{self.call_uuid[:8]}]   ├─ Captured: \"{pending_transcript[:60]}\"")
            self._pipeline.capture_response()
            self._user_audio_buffer = bytearray(b"")

            # Advance question counter and track objections (for stats/webhook)
            next_instruction = self._question_flow.advance(pending_transcript)
            end_call = isinstance(next_instruction, dict) and next_instruction.get("end_call", False)

            # Finalize: CAPTURED → DONE
            self._pipeline.finalize_and_advance()

            logger.info(f"[{self.call_uuid[:8]}]   └─ Response tracked Q{self._question_flow.current_step}/{len(self._question_flow.questions)}")
            self._last_question_time = time.time()

            # Passive tracking: Gemini drives question transitions naturally (all questions in system prompt).
            # No injection needed — just dequeue for pipeline tracking (stats/webhook).
            if self._pipeline and self._question_flow:
                step = self._question_flow.current_step
                questions = self._question_flow.questions
                if step < len(questions):
                    q_id = questions[step]["id"]
                    q_text = self._question_flow.get_instruction_prompt()
                    self._pipeline.dequeue_next(index=step, question_id=q_id, question_text=q_text)

            if end_call and not self._closing_call:
                asyncio.create_task(self._fallback_hangup(5.0))
            return

        # Non-pipeline fallback: Require minimum audio duration (at least 0.5 seconds)
        MIN_AUDIO_FOR_RESPONSE = 16000  # 0.5 seconds of audio
        if len(self._user_audio_buffer) < MIN_AUDIO_FOR_RESPONSE:
            logger.debug(f"[{self.call_uuid[:8]}] Not enough audio ({len(self._user_audio_buffer)} bytes), need {MIN_AUDIO_FOR_RESPONSE}")
            return  # Don't clear buffer - keep accumulating

        # Clear buffer (transcription disabled - will be done at end of call)
        self._user_audio_buffer = bytearray(b"")

    async def _inject_question(self, instruction, user_said: str = ""):
        """Inject the next question text into the AI, with natural acknowledgment of user's response"""
        if not self.goog_live_ws or not instruction:
            logger.debug(f"[{self.call_uuid[:8]}] _inject_question: skipped (ws={bool(self.goog_live_ws)}, "
                         f"instruction={bool(instruction)})")
            return

        try:
            if isinstance(instruction, dict):
                text = instruction.get("text", "")
                end_call = bool(instruction.get("end_call"))
            else:
                text = str(instruction)
                end_call = False

            if not text:
                logger.debug(f"[{self.call_uuid[:8]}] _inject_question: empty text, skipping")
                return

            step = self._question_flow.current_step if self._question_flow else 0
            total = len(self._question_flow.questions) if self._question_flow else 0
            logger.info(f"[{self.call_uuid[:8]}] ▶ Q{step + 1}/{total}: {text[:80]}")

            # Clear audio buffer BEFORE asking question
            self._user_audio_buffer = bytearray(b"")
            self._last_user_audio_time = None
            self._last_user_transcript_time = 0

            # Send the question using instruction template
            step_num = self._question_flow.current_step if self._question_flow else 0
            total_q = len(self._question_flow.questions) if self._question_flow else 0
            q_num = step_num + 1  # Human-readable (Q1, Q2, ...)

            if end_call:
                instruction_text = self._templates["instruction_end_call"].replace("{text}", text)
            else:
                instruction_text = self._templates["instruction_ask"].replace("{text}", text)

            msg = {
                "client_content": {
                    "turns": [{
                        "role": "user",
                        "parts": [{"text": instruction_text}]
                    }],
                    "turn_complete": True
                }
            }
            await self.goog_live_ws.send(json_dumps(msg))

            # Pipeline: dequeue this question (opens gate, starts lifecycle)
            if self._pipeline and self._question_flow:
                step = self._question_flow.current_step
                questions = self._question_flow.questions
                if step < len(questions):
                    q_id = questions[step]["id"]
                else:
                    q_id = f"closing_{step}"
                self._pipeline.dequeue_next(index=step, question_id=q_id, question_text=text)
            logger.info(f"[{self.call_uuid[:8]}]   ├─ Sent to Gemini, waiting for AI response")
            if end_call and not self._closing_call:
                # Give the model a moment to say the closing line before hangup
                asyncio.create_task(self._fallback_hangup(5.0))
        except Exception as e:
            logger.error(f"[{self.call_uuid[:8]}] Error injecting question: {e}")

    def _buffer_user_audio(self, audio_chunk: bytes):
        """Buffer user audio for transcription"""
        if not self.use_question_flow:
            return

        # Add to buffer (with size limit)
        remaining_space = self._max_audio_buffer_size - len(self._user_audio_buffer)
        if remaining_space > 0:
            self._user_audio_buffer.extend(audio_chunk[:remaining_space])

    async def _send_session_setup(self):
        # Build prompt: base prompt already contains all questions (embedded in system_instruction)
        full_prompt = self.prompt

        # On reconnect, load conversation from FILE (not memory - avoids latency issues)
        if not self._is_first_connection:
            # Load conversation history from file (saved by background thread)
            # LATENCY OPT: Async file I/O (5-18ms savings on reconnect)
            file_history = await self._load_conversation_from_file()
            if file_history:
                history_text = "\n\n[Recent conversation - continue from here:]\n"
                for msg_item in file_history[-self._max_history_size:]:
                    role = "Customer" if msg_item["role"] == "user" else "You"
                    history_text += f"{role}: {msg_item['text']}\n"
                history_text += "\n[Continue naturally. Do NOT greet again.]"
                full_prompt = full_prompt + history_text
                logger.debug(f"[{self.call_uuid[:8]}] Loaded {len(file_history)} messages for reconnect")
                self._is_reconnecting = False

        # Use voice from config (if question flow mode) or detect from prompt
        if self._config_voice:
            voice_name = self._config_voice
            logger.info(f"[{self.call_uuid[:8]}] Using voice from config: {voice_name}")
        elif self.use_question_flow:
            voice_name = "Puck"
            logger.info(f"[{self.call_uuid[:8]}] Using default QuestionFlow voice: {voice_name}")
        else:
            voice_name = detect_voice_from_prompt(self.prompt)

        # Model name differs between Google AI Studio and Vertex AI
        if config.use_vertex_ai:
            # Production model for Vertex AI Live API
            model_name = f"projects/{config.vertex_project_id}/locations/{config.vertex_location}/publishers/google/models/gemini-live-2.5-flash-native-audio"
        else:
            model_name = "models/gemini-2.5-flash-native-audio-preview-09-2025"

        msg = {
            "setup": {
                "model": model_name,
                "generation_config": {
                    "response_modalities": ["AUDIO"],  # Native audio model - audio only
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {
                                "voice_name": voice_name
                            }
                        }
                    },
                    # Light thinking - just enough for quality, minimal latency
                    "thinking_config": {
                        "thinking_budget": 25
                    }
                },
                # Enable transcription to get real-time speech transcripts
                "input_audio_transcription": {},  # User speech → inputTranscription events
                "output_audio_transcription": {},  # AI speech → outputTranscription events
                "system_instruction": {"parts": [{"text": full_prompt}]},
                "tools": [{"function_declarations": TOOL_DECLARATIONS}]
            }
        }
        await self.goog_live_ws.send(json_dumps(msg))

        # Note: _is_first_connection is set to False in setupComplete handler, not here
        logger.info(f"Sent session setup with voice: {voice_name}, first_connection={self._is_first_connection}")

    async def _send_initial_greeting(self):
        """Send initial trigger to make AI greet with first question"""
        if self.greeting_sent or not self.goog_live_ws:
            return
        self.greeting_sent = True

        # Question Flow Mode: Trigger Q1 (Gemini already knows all questions from system prompt)
        if self.use_question_flow and self._question_flow:
            first_instruction = self._question_flow.get_instruction_prompt()
            trigger_text = self._templates["greeting_trigger"]
            logger.debug(f"[{self.call_uuid[:8]}] Starting with Q1")

            # Clear audio buffer and dequeue first question via pipeline
            self._user_audio_buffer = bytearray(b"")
            self._last_user_transcript_time = 0
            if self._pipeline:
                q_id = self._question_flow.questions[0]["id"] if self._question_flow.questions else "greeting"
                self._pipeline.dequeue_next(index=0, question_id=q_id, question_text=first_instruction)
        else:
            trigger_text = self._templates["greeting_simple"]

        msg = {
            "client_content": {
                "turns": [{
                    "role": "user",
                    "parts": [{"text": trigger_text}]
                }],
                "turn_complete": True
            }
        }
        await self.goog_live_ws.send(json_dumps(msg))
        logger.info(f"[{self.call_uuid[:8]}] Greeting trigger sent to Gemini")

    async def _send_reconnection_trigger(self):
        """Trigger AI to speak immediately after reconnection, restoring question flow state"""
        if not self.goog_live_ws:
            return

        # Restore question flow state on reconnect
        if self.use_question_flow and self._question_flow:
            current_step = self._question_flow.current_step
            current_question = self._question_flow.get_current_question()

            if current_question:
                reconnect_text = self._templates["reconnect_continue"].replace("{text}", current_question)
            else:
                reconnect_text = self._templates["reconnect_goodbye"]

            logger.debug(f"[{self.call_uuid[:8]}] Restoring to question {current_step + 1}")
        else:
            reconnect_text = self._templates["reconnect_simple"]

        msg = {
            "client_content": {
                "turns": [{
                    "role": "user",
                    "parts": [{"text": reconnect_text}]
                }],
                "turn_complete": True
            }
        }
        await self.goog_live_ws.send(json_dumps(msg))
        if self._pipeline and self.use_question_flow:
            # Re-dequeue current question for reconnection (re-opens gate)
            current_step = self._question_flow.current_step
            current_question = self._question_flow.get_current_question() or "reconnection"
            q_id = self._question_flow.questions[current_step]["id"] if current_step < len(self._question_flow.questions) else "reconnect"
            self._pipeline.dequeue_next(index=current_step, question_id=q_id, question_text=current_question)
        logger.debug(f"[{self.call_uuid[:8]}] Reconnect trigger sent - gate OPEN")

    async def _handle_tool_call(self, tool_call):
        """Execute tool and send response back to Gemini - gracefully handles errors"""
        func_calls = tool_call.get("functionCalls", [])
        for fc in func_calls:
            tool_name = fc.get("name")
            tool_args = fc.get("args", {})
            call_id = fc.get("id")

            logger.info(f"[{self.call_uuid[:8]}]   ├─ Tool: {tool_name}")
            self._save_transcript("TOOL", f"{tool_name}: {tool_args}")

            # Handle end_call tool - guard against premature ending
            if tool_name == "end_call":
                reason = tool_args.get("reason", "conversation ended")

                # Guard: reject end_call if questions remain, unless user explicitly asked to end
                user_requested_end = any(kw in reason.lower() for kw in [
                    "busy", "call later", "call back", "called back", "call me",
                    "not interested", "step out", "no time", "tomorrow",
                    "goodbye", "bye", "hang up", "end", "stop", "don't want",
                    "not now", "meeting", "driving", "callback"
                ])
                questions_done = self._question_flow.current_step if self._question_flow else 0
                total_questions = len(self._question_flow.questions) if self._question_flow else 0
                # Allow end_call if within 2 questions of the end — Gemini often asks
                # 2 questions per turn so the pipeline tracker lags behind by 1-2
                nearly_done = (total_questions - questions_done) <= 2
                if not user_requested_end and not nearly_done and questions_done < total_questions:
                    # Get the next question to tell Gemini exactly what to ask
                    remaining = total_questions - questions_done
                    next_q = ""
                    if self._question_flow:
                        next_q_text = self._question_flow.get_current_question()
                        if next_q_text:
                            next_q = f' Your next question is: "{next_q_text}"'
                    logger.warning(f"[{self.call_uuid[:8]}] end_call REJECTED: only {questions_done}/{total_questions} questions done, "
                                   f"reason='{reason}' — telling AI to continue")
                    try:
                        tool_response = {
                            "tool_response": {
                                "function_responses": [{
                                    "id": call_id,
                                    "name": tool_name,
                                    "response": {"success": False, "message": f"STOP. Do NOT say goodbye. Do NOT end the call. You still have {remaining} questions left. Do NOT repeat any goodbye or farewell. Ask the next question immediately.{next_q}"}
                                }]
                            }
                        }
                        await self.goog_live_ws.send(json_dumps(tool_response))
                    except Exception:
                        pass
                    return

                logger.info(f"[{self.call_uuid[:8]}]   └─ End call: {reason}")
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
                    await self.goog_live_ws.send(json_dumps(tool_response))
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
                result = await execute_tool(tool_name, self.caller_phone, context=self.context, **tool_args)
                success = result.get("success", False)
                message = result.get("message", "Tool executed")
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
                await self.goog_live_ws.send(json_dumps(tool_response))
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
            resp = json_loads(message)

            # Log all Gemini responses for debugging
            resp_keys = list(resp.keys())
            if resp_keys != ['serverContent']:  # Don't log every content message
                logger.debug(f"Gemini response keys: {resp_keys}")

            if "setupComplete" in resp:
                logger.info(f"[{self.call_uuid[:8]}] AI ready")
                self.start_streaming = True
                self.setup_complete = True
                self._google_session_start = time.time()  # Track session start for 10-min limit
                self._save_transcript("SYSTEM", "AI ready")
                # On first connection: trigger greeting
                # On reconnection: trigger resume with filler phrase
                if self._is_first_connection:
                    self._is_first_connection = False  # Mark first connection done
                    await self._send_initial_greeting()
                else:
                    await self._send_reconnection_trigger()

            # Handle GoAway message - 9-minute warning before 10-minute session limit
            if "goAway" in resp:
                logger.warning(f"[{self.call_uuid[:8]}] 10-min limit, reconnecting...")
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

                # Check if turn is complete (greeting done)
                if sc.get("turnComplete"):
                    self._preload_complete.set()
                    self.greeting_audio_complete = True
                    self._turn_count += 1
                    self._current_turn_id += 1  # New turn ID for queue pipeline

                    # Audio gating: close gate after AI finishes delivering ack + question
                    # Pipeline handles turn counting and DELIVERING → ECHOING transition
                    if self._pipeline and self._pipeline.gate_open:
                        self._pipeline.on_turn_complete()

                    if self._turn_start_time and self._current_turn_audio_chunks > 0:
                        turn_duration_ms = (time.time() - self._turn_start_time) * 1000
                        # Log accumulated agent speech as one line
                        if self._current_turn_agent_text:
                            full_agent = " ".join(self._current_turn_agent_text)
                            logger.info(f"[{self.call_uuid[:8]}]   ├─ AGENT: {full_agent}")
                            self._current_turn_agent_text = []
                        # Log accumulated user speech as one line
                        if self._current_turn_user_text:
                            full_user = " ".join(self._current_turn_user_text)
                            logger.info(f"[{self.call_uuid[:8]}]   ├─ USER: {full_user}")
                            self._current_turn_user_text = []
                        logger.info(f"[{self.call_uuid[:8]}]   ├─ Turn #{self._turn_count}: {self._current_turn_audio_chunks} chunks, {turn_duration_ms:.0f}ms")
                        self._turn_start_time = None

                    # FIX: In QuestionFlow mode, do NOT process transcription on turnComplete
                    # The silence monitor will detect when user finishes speaking and process then
                    # Detect empty turn (AI didn't generate audio) - nudge to respond
                    # Skip nudging entirely in QuestionFlow mode - just wait for user
                    if self._current_turn_audio_chunks == 0 and self.greeting_audio_complete and not self._closing_call and not self.use_question_flow:
                        self._empty_turn_nudge_count += 1
                        if self._empty_turn_nudge_count <= 3:  # Max 3 nudges
                            logger.warning(f"[{self.call_uuid[:8]}] Empty turn, nudging AI ({self._empty_turn_nudge_count}/3)")
                            asyncio.create_task(self._send_silence_nudge())
                    else:
                        self._empty_turn_nudge_count = 0

                    # Reset turn audio counter
                    self._current_turn_audio_chunks = 0

                    # Process deferred goodbye detection (agent finished speaking)
                    if self._goodbye_pending and not self._closing_call:
                        self._goodbye_pending = False
                        logger.debug(f"[{self.call_uuid[:8]}] Agent goodbye detected (deferred to turnComplete)")
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
                        await self.plivo_ws.send_text(json_dumps({"event": "clearAudio", "stream_id": self.stream_id}))

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
                        if is_noise:
                            logger.debug(f"[{self.call_uuid[:8]}] Skipping noise marker: {user_text}")
                            # Don't process noise at all - skip everything below
                        else:
                            # Echo protection: pipeline.is_echo checks DELIVERING + ECHOING phases
                            is_echo = False
                            if self._pipeline and self._pipeline.waiting_for_user:
                                if self._pipeline.is_echo:
                                    is_echo = True
                                    phase = self._pipeline.current_phase
                                    logger.debug(f"[{self.call_uuid[:8]}] Echo ({phase.value}): ignoring '{user_text}'")

                            if not is_echo:
                                # Real user speech - accumulate via pipeline
                                if self._pipeline and self._pipeline.waiting_for_user:
                                    self._pipeline.accumulate_user_speech(user_text)
                                    self._last_user_transcript_time = time.time()
                                    # Event-driven silence detection: schedule task to fire after user stops speaking
                                    self._schedule_silence_detection()
                                self._last_user_speech_time = time.time()  # Track for latency
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
                        # Store what the agent said for this question
                        if self._pipeline:
                            self._pipeline.store_agent_said(ai_text)
                        # Feed to transcript validator worker for off-script detection
                        try:
                            self._transcript_val_queue.put_nowait({
                                "turn_id": self._current_turn_id,
                                "text": ai_text
                            })
                        except asyncio.QueueFull:
                            pass
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
                            # Track turn start time and log when agent starts speaking
                            if self._current_turn_audio_chunks == 1:
                                self._turn_start_time = time.time()
                                self._agent_speaking = True
                                self._user_speaking = False
                                logger.debug(f"[{self.call_uuid[:8]}] Agent speaking")
                            # Record AI audio (24kHz)
                            self._record_audio("AI", audio_bytes, 24000)

                            # Latency check - only log if slow (> threshold)
                            if self._last_user_speech_time:
                                latency_ms = (time.time() - self._last_user_speech_time) * 1000
                                if latency_ms > LATENCY_THRESHOLD_MS:
                                    logger.warning(f"[{self.call_uuid[:8]}] Slow response: {latency_ms:.0f}ms")
                                # Record latency on current question for post-call metrics
                                if self._pipeline and self._pipeline._current and self._pipeline._current.response_latency_ms == 0:
                                    self._pipeline._current.response_latency_ms = latency_ms
                                self._last_user_speech_time = None  # Reset after first response

                            # During preload (no plivo_ws yet), always store audio
                            # This fixes race condition where turnComplete arrives before all audio
                            if not self.plivo_ws:
                                self.preloaded_audio.append(audio)
                            elif self.plivo_ws:
                                # Push to audio pipeline queue (gate worker handles gating)
                                chunk = AudioChunk(
                                    audio_b64=audio,
                                    turn_id=self._current_turn_id,
                                    sample_rate=24000
                                )
                                if self._pipeline:
                                    # Queue-based path: gate worker + sender worker
                                    try:
                                        self._audio_out_queue.put_nowait(chunk)
                                    except asyncio.QueueFull:
                                        logger.warning(f"[{self.call_uuid[:8]}] audio_out_queue full, dropping chunk")
                                else:
                                    # No pipeline (non-QuestionFlow): send directly to plivo_send_queue
                                    try:
                                        self._plivo_send_queue.put_nowait(chunk)
                                    except asyncio.QueueFull:
                                        logger.warning(f"[{self.call_uuid[:8]}] plivo_send_queue full, dropping chunk")
                                # Log first chunk for this turn
                                if self._current_turn_audio_chunks == 1:
                                    logger.debug(f"[{self.call_uuid[:8]}] Audio → queue pipeline")
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
                                self._current_turn_agent_text.append(ai_text)
                                self._save_transcript("AGENT", ai_text)
                                self._log_conversation("model", ai_text)
                                if self._pipeline:
                                    self._pipeline.store_agent_said(ai_text)
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
                    await self.goog_live_ws.send(json_dumps(msg))
                    # Buffer audio for REST API transcription (non-blocking)
                    self._buffer_user_audio(bytes(ac))
                    chunks_sent += 1
                    # Log first chunk sent to Gemini for this user speech
                    if chunks_sent == 1 and self._user_speaking:
                        logger.debug(f"[{self.call_uuid[:8]}] Sending user audio to Gemini")
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

        logger.info(f"[{self.call_uuid[:8]}] Call stopping")
        self.is_active = False

        # Cancel timeout task
        if self._timeout_task:
            self._timeout_task.cancel()

        # Cancel silence monitor
        if self._silence_monitor_task:
            self._silence_monitor_task.cancel()

        # Cancel silence detection task (event-driven optimization)
        if self._silence_detection_task:
            self._silence_detection_task.cancel()

        # Cancel queue pipeline workers
        for task in (self._gate_worker_task, self._sender_worker_task, self._validator_worker_task):
            if task:
                task.cancel()

        # Calculate call duration
        duration = 0
        if self.call_start_time:
            duration = (datetime.now() - self.call_start_time).total_seconds()
            logger.info(f"[{self.call_uuid[:8]}] Duration: {duration:.1f}s")
            self._save_transcript("SYSTEM", f"Call duration: {duration:.1f}s")

        # Cleanup question flow and get collected data + statistics
        self._flow_data = None
        self._flow_statistics = None
        self._question_pairs = []
        if self.use_question_flow:
            # Get statistics BEFORE removing flow (needs the flow object)
            flow_obj = get_or_create_flow(self.call_uuid, self.client_name)
            self._flow_statistics = flow_obj.get_statistics()
            self._flow_data = remove_flow(self.call_uuid)
            if self._flow_data:
                logger.debug(f"[{self.call_uuid[:8]}] Flow: {len(self._flow_data.get('responses', {}))} responses")
            # Collect Q&A pairs from pipeline
            if self._pipeline:
                self._question_pairs = self._pipeline.get_collected_pairs()
                logger.debug(f"[{self.call_uuid[:8]}] Pipeline: {len(self._question_pairs)} Q&A pairs collected")

        if self.goog_live_ws:
            try:
                await self.goog_live_ws.close()
            except Exception:
                pass
        if self._session_task:
            self._session_task.cancel()

        # REST API transcription - no WebSocket to close

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

        # Process recording and transcription in COMPLETELY SEPARATE background thread
        # This does NOT block call ending - call ends immediately
        # Webhook is called AFTER transcription is complete
        # Generate call metrics before post-call (pipeline still in memory)
        self._call_metrics = {}
        if self._pipeline:
            self._call_metrics = self._pipeline.get_call_metrics()
            # Log metrics summary
            m = self._call_metrics
            logger.info(f"[{self.call_uuid[:8]}] === CALL METRICS ===")
            logger.info(f"[{self.call_uuid[:8]}] Questions: {m.get('questions_completed', 0)} | "
                        f"Avg latency: {m.get('avg_latency_ms', 0)}ms | "
                        f"Max: {m.get('max_latency_ms', 0)}ms | "
                        f"Min: {m.get('min_latency_ms', 0)}ms | "
                        f"P90: {m.get('p90_latency_ms', 0)}ms | "
                        f"Nudges: {m.get('total_nudges', 0)}")
            for q in m.get("per_question", []):
                logger.info(f"[{self.call_uuid[:8]}]   Q{q['q']}({q['id']}) "
                            f"latency={q['latency_ms']}ms | "
                            f"duration={q['duration_s']}s | "
                            f"nudges={q['nudges']}")
            logger.info(f"[{self.call_uuid[:8]}] === END METRICS ===")

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

                # Step 3.5: Save call metrics to file
                if self._call_metrics:
                    try:
                        metrics_file = Path(__file__).parent.parent.parent / "flow_data" / f"{self.call_uuid}_metrics.json"
                        metrics_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(metrics_file, 'w') as f:
                            json.dump({"call_uuid": self.call_uuid, "metrics": self._call_metrics}, f, indent=2)
                    except Exception as e:
                        logger.error(f"Error saving metrics: {e}")

                # Step 4: Update session DB with final data
                stats = self._flow_statistics or {}
                flow_data = self._flow_data or {}
                session_db.update_call(
                    self.call_uuid,
                    status="completed",
                    ended_at=datetime.now().isoformat(),
                    duration_seconds=round(duration, 1),
                    transcript=transcript,
                    questions_completed=stats.get("questions_completed", 0),
                    interest_level=stats.get("interest_level", "unknown"),
                    call_summary=stats.get("call_summary", ""),
                    collected_responses=flow_data.get("responses", {}),
                    objections_raised=stats.get("objections_raised", [])
                )

                # Step 5: Call webhook AFTER everything is saved
                if self.webhook_url:
                    import asyncio as _asyncio
                    loop = _asyncio.new_event_loop()
                    _asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self._call_webhook(duration, transcript, stats, flow_data))
                    finally:
                        loop.close()

                logger.info(f"[{self.call_uuid[:8]}] Post-call processing complete")
            except Exception as e:
                logger.error(f"Post-call processing error: {e}")

        # Start background thread - call ends immediately, this runs separately
        processing_thread = threading.Thread(target=process_in_background, daemon=True)
        processing_thread.start()

    async def _call_webhook(self, duration: float, transcript: str = "",
                            stats: dict = None, flow_data: dict = None):
        """Call webhook URL with enriched call data (transcript + statistics)"""
        try:
            import httpx

            stats = stats or {}
            flow_data = flow_data or {}

            payload = {
                "event": "call_ended",
                "call_uuid": self.call_uuid,
                "caller_phone": self.caller_phone,
                "contact_name": self.context.get("customer_name", ""),
                "client_name": self.client_name,
                "duration_seconds": round(duration, 1),
                "timestamp": datetime.now().isoformat(),
                # Call statistics
                "questions_completed": stats.get("questions_completed", 0),
                "total_questions": stats.get("total_questions", 0),
                "completion_rate": stats.get("completion_rate", 0),
                "interest_level": stats.get("interest_level", "unknown"),
                "call_summary": stats.get("call_summary", ""),
                "objections_raised": stats.get("objections_raised", []),
                # Collected responses (individual answers)
                "collected_responses": flow_data.get("responses", {}),
                # Q&A pairs with agent_said + user_said per question
                "question_pairs": getattr(self, '_question_pairs', []),
                # Call metrics (latency per question, avg/max/p90)
                "call_metrics": getattr(self, '_call_metrics', {}),
                # Transcript
                "transcript": transcript,
                "transcript_entries": self._full_transcript
            }

            logger.info(f"[{self.call_uuid[:8]}] Webhook: interest={stats.get('interest_level')}, "
                         f"questions={stats.get('questions_completed')}/{stats.get('total_questions')}")
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(self.webhook_url, json=payload)
                logger.info(f"[{self.call_uuid[:8]}] Webhook response: {resp.status_code}")
        except Exception as e:
            logger.error(f"Error calling webhook: {e}")


# Session storage with concurrency protection
MAX_CONCURRENT_SESSIONS = 5
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


# ==================== CONVERSATIONAL FLOW SUPPORT ====================

async def inject_context_to_session(call_uuid: str, phase: str, additional_context: str = None, data: dict = None) -> bool:
    """
    Inject dynamic context/prompt into an ongoing call session.
    Used by n8n to send phase-specific prompts based on conversation state.

    Args:
        call_uuid: The call ID
        phase: The conversation phase (e.g., "connection_liked", "situation_role")
        additional_context: Optional additional context string from n8n
        data: Optional data dict with captured information (role, company, etc.)

    Returns:
        True if injection succeeded
    """
    from src.conversational_prompts import PHASE_PROMPTS

    async with _sessions_lock:
        session = _sessions.get(call_uuid)
    if not session or not session.goog_live_ws:
        logger.warning(f"Cannot inject context - session {call_uuid} not found or not connected")
        return False

    # Get phase-specific prompt
    phase_prompt = PHASE_PROMPTS.get(phase)
    if not phase_prompt:
        logger.warning(f"Unknown phase: {phase}")
        return False

    try:
        # Build context-aware prompt
        full_prompt = phase_prompt

        # Replace placeholders with context values from session context
        if session.context:
            customer_name = session.context.get("customer_name", "")
            if customer_name:
                full_prompt = full_prompt.replace("[NAME]", customer_name)

        # Add additional context from n8n if provided
        if additional_context:
            full_prompt = full_prompt + "\n" + additional_context

        # Add captured data as context
        if data:
            data_summary = "\n[CAPTURED DATA: " + ", ".join(f"{k}={v}" for k, v in data.items() if v) + "]"
            full_prompt = full_prompt + data_summary

        # Send as system instruction update via text message
        msg = {
            "client_content": {
                "turns": [{
                    "role": "user",
                    "parts": [{"text": f"[PHASE UPDATE: {phase}]\n{full_prompt}"}]
                }],
                "turn_complete": True
            }
        }
        await session.goog_live_ws.send(json_dumps(msg))
        logger.debug(f"[{call_uuid[:8]}] Phase injected: {phase}")
        session._save_transcript("SYSTEM", f"Phase: {phase}")
        return True

    except Exception as e:
        logger.error(f"Error injecting context: {e}")
        return False


async def preload_session_conversational(
    call_uuid: str,
    caller_phone: str,
    base_prompt: str = None,
    initial_phase_prompt: str = None,
    context: dict = None,
    call_end_webhook_url: str = None,
    client_name: str = "fwai",
    questions_override: list = None,
    prompt_override: str = None,
    objections_override: dict = None,
    objection_keywords_override: dict = None,
    instruction_templates: dict = None
) -> bool:
    """
    Preload a session in conversational flow mode.
    Uses QuestionFlow to inject questions one by one.

    Args:
        call_uuid: Unique call identifier
        caller_phone: Caller's phone number
        context: Context dict with customer_name, etc.
        call_end_webhook_url: URL to call when call ends
        client_name: Client config to use (e.g., 'fwai')
        questions_override: List of questions from API
        prompt_override: System instruction prompt from API
        instruction_templates: Override instruction texts (nudge, wrap-up, etc.)

    Returns:
        True if preload succeeded
    """
    # Check session limit
    async with _sessions_lock:
        total = len(_sessions) + len(_preloading_sessions)
        if total >= MAX_CONCURRENT_SESSIONS:
            logger.warning(f"Max concurrent sessions ({MAX_CONCURRENT_SESSIONS}) reached. Rejecting {call_uuid}")
            raise Exception(f"Max concurrent sessions ({MAX_CONCURRENT_SESSIONS}) reached")

        # Create session with QuestionFlow enabled
        session = PlivoGeminiSession(
            call_uuid,
            caller_phone,
            prompt=None,
            context=context,
            webhook_url=call_end_webhook_url,
            client_name=client_name,
            use_question_flow=True,  # Explicitly enable QuestionFlow
            questions_override=questions_override,
            prompt_override=prompt_override,
            objections_override=objections_override,
            objection_keywords_override=objection_keywords_override,
            instruction_templates=instruction_templates
        )
        _preloading_sessions[call_uuid] = session

    success = await session.preload()
    return success
