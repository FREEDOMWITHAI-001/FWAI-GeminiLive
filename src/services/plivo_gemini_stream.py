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
from src.conversational_prompts import BASE_PROMPT, DEFAULT_CONTEXT, render_prompt
from src.question_flow import QuestionFlow, get_or_create_flow, remove_flow


def get_vertex_ai_token():
    """Get OAuth2 access token for Vertex AI"""
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

# Latency threshold - only log if slower than this (ms)
LATENCY_THRESHOLD_MS = 500

# Recording directory
RECORDINGS_DIR = Path(__file__).parent.parent.parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

# Prompts directory for client-specific prompts
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
PROMPTS_DIR.mkdir(exist_ok=True)


def load_client_prompt(client_name: str = None) -> str:
    """
    Load client-specific prompt from file.

    Looks for prompts in this order:
    1. prompts/{client_name}_prompt.txt (e.g., fwai_prompt.txt, ridhi_prompt.txt)
    2. prompts/{client_name}.txt
    3. prompts/default_prompt.txt
    4. Falls back to BASE_PROMPT from conversational_prompts.py

    Args:
        client_name: Client identifier (e.g., 'fwai', 'ridhi', 'acme')

    Returns:
        Prompt template string with {{placeholders}}
    """
    if client_name:
        client_name = client_name.lower().replace(" ", "_")

        # Try client-specific files
        possible_files = [
            PROMPTS_DIR / f"{client_name}_prompt.txt",
            PROMPTS_DIR / f"{client_name}_prompt_questions.txt",
            PROMPTS_DIR / f"{client_name}.txt",
        ]

        for prompt_file in possible_files:
            if prompt_file.exists():
                try:
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            logger.info(f"Loaded prompt from: {prompt_file.name}")
                            return content
                except Exception as e:
                    logger.warning(f"Error reading {prompt_file}: {e}")

    # Try default file
    default_file = PROMPTS_DIR / "default_prompt.txt"
    if default_file.exists():
        try:
            with open(default_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    logger.info("Loaded prompt from: default_prompt.txt")
                    return content
        except Exception as e:
            logger.warning(f"Error reading default_prompt.txt: {e}")

    # Final fallback to BASE_PROMPT
    logger.info("Using BASE_PROMPT from conversational_prompts.py")
    return BASE_PROMPT


# Legacy: Load from prompts.json if it exists (backward compatibility)
def load_default_prompt():
    try:
        prompts_file = Path(__file__).parent.parent.parent / "prompts.json"
        with open(prompts_file) as f:
            prompts = json.load(f)
            return prompts.get("FWAI_Core", {}).get("prompt", BASE_PROMPT)
    except:
        return BASE_PROMPT

DEFAULT_PROMPT = load_default_prompt()


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
    def __init__(self, call_uuid: str, caller_phone: str, prompt: str = None, context: dict = None, webhook_url: str = None, transcript_webhook_url: str = None, client_name: str = "fwai", use_question_flow: bool = True):
        self.call_uuid = call_uuid  # Internal UUID
        self.plivo_call_uuid = None  # Plivo's actual call UUID (set later)
        self.caller_phone = caller_phone
        self.context = context or {}  # Context for templates (customer_name, course_name, etc.)
        self.client_name = client_name or "fwai"
        self.use_question_flow = use_question_flow  # Use built-in question flow state machine

        # Question Flow Mode: Use minimal prompt + inject questions one by one
        if use_question_flow:
            # Create flow with client config - loads questions, voice, etc from config file
            self._question_flow = get_or_create_flow(
                call_uuid=call_uuid,
                client_name=self.client_name,
                context=self.context
            )
            self.prompt = self._question_flow.get_base_prompt()
            # Get voice from config (not hardcoded)
            self._config_voice = self._question_flow.get_voice()
            logger.info(f"[{call_uuid[:8]}] QuestionFlow: client={self.client_name}, voice={self._config_voice}")
        else:
            # Legacy mode: Load full prompt from file
            self._question_flow = None
            self._config_voice = None
            if prompt:
                raw_prompt = prompt
            elif client_name:
                raw_prompt = load_client_prompt(client_name)
            else:
                raw_prompt = DEFAULT_PROMPT
            self.prompt = render_prompt(raw_prompt, self.context)
            logger.info(f"[{call_uuid[:8]}] Full prompt mode for client: {client_name or 'default'}")
        self.webhook_url = webhook_url  # URL to call when call ends (for n8n integration)
        self._transcript_webhook_url = transcript_webhook_url  # URL for real-time transcript (n8n state machine)
        if transcript_webhook_url:
            logger.info(f"[{call_uuid[:8]}] Transcript webhook configured: {transcript_webhook_url}")
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
        self._pending_user_transcript = ""  # Buffer to collect transcripts

        # Full transcript collection (in-memory backup for webhook)
        self._full_transcript = []  # List of {"role": "USER/AGENT", "text": "...", "timestamp": "..."}

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
            except:
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
            logger.debug(f"[{self.call_uuid[:8]}] Preloading Gemini session")
            self.is_active = True
            # Start main voice session (native audio)
            self._session_task = asyncio.create_task(self._run_google_live_session())
            # REST API transcription enabled if webhook configured
            if self._transcript_webhook_url:
                logger.debug(f"[{self.call_uuid[:8]}] REST API transcription active")
            # Wait for setup to complete (with timeout - 8s max for better greeting)
            try:
                await asyncio.wait_for(self._preload_complete.wait(), timeout=8.0)
                logger.info(f"[{self.call_uuid[:8]}] AI preloaded ({len(self.preloaded_audio)} chunks)")
            except asyncio.TimeoutError:
                logger.warning(f"[{self.call_uuid[:8]}] Preload timeout, continuing with {len(self.preloaded_audio)} chunks")
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
        # Send any preloaded audio immediately
        if self.preloaded_audio:
            asyncio.create_task(self._send_preloaded_audio())
        else:
            logger.warning(f"[{self.call_uuid[:8]}] No preloaded audio - greeting will lag")
        # Start call duration timer
        self._timeout_task = asyncio.create_task(self._monitor_call_duration())
        # Start silence monitor (3 second SLA)
        self._silence_monitor_task = asyncio.create_task(self._monitor_silence())

    async def _send_preloaded_audio(self):
        """Send preloaded audio to Plivo"""
        logger.debug(f"[{self.call_uuid[:8]}] Sending {len(self.preloaded_audio)} preloaded chunks")
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
            logger.debug(f"[{self.call_uuid[:8]}] Call monitor started")

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
                    logger.warning(f"[{self.call_uuid[:8]}] {silence_duration:.1f}s silence - nudging AI")
                    await self._send_silence_nudge()
                    # Reset timer to avoid repeated nudges
                    self._last_user_speech_time = None

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in silence monitor: {e}")

    async def _send_silence_nudge(self):
        """Send a nudge to AI when silence detected - DISABLED for QuestionFlow"""
        if not self.goog_live_ws or self._closing_call:
            return

        # In QuestionFlow mode, don't nudge at all - just wait for real user input
        if self.use_question_flow:
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

            # The AI will say "Sorry, brief connection issue..." after reconnecting
            # This is handled in _send_session_setup via the reconnection prompt

        except Exception as e:
            logger.error(f"Error in reconnection filler: {e}")

    async def _run_google_live_session(self):
        # Choose between Vertex AI (regional, lower latency) or Google AI Studio
        if config.use_vertex_ai:
            # Vertex AI Live API - regional endpoint (asia-south1 = Mumbai)
            token = get_vertex_ai_token()
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
                if config.use_vertex_ai and reconnect_attempts > 0:
                    token = get_vertex_ai_token()
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
        """Process buffered user audio, transcribe, and advance question flow"""
        # Question flow mode doesn't require webhook URL
        if not self._transcript_webhook_url and not self.use_question_flow:
            return

        if len(self._user_audio_buffer) < 1600:  # Less than 0.1 second
            self._user_audio_buffer = bytearray(b"")
            return

        # Copy and clear buffer
        audio_to_transcribe = bytes(self._user_audio_buffer)
        self._user_audio_buffer = bytearray(b"")

        # Transcribe using REST API
        transcription = await self._transcribe_audio_rest_api(audio_to_transcribe)

        if transcription:
            # Filter out noise/non-English transcriptions
            # Check if transcription contains mostly ASCII (English)
            ascii_ratio = sum(1 for c in transcription if ord(c) < 128) / len(transcription)
            if ascii_ratio < 0.5:
                logger.debug(f"[{self.call_uuid[:8]}] Ignoring non-English transcription: {transcription[:30]}")
                return

            # Ignore very short transcriptions (likely noise)
            if len(transcription.strip()) < 3:
                logger.debug(f"[{self.call_uuid[:8]}] Ignoring short transcription: {transcription}")
                return

            # Filter out pure filler/hesitation sounds
            filler_only = {"um", "uh", "ah", "oh", "mhm", "hmm", "mm", "ugh", "hm", "eh"}
            words = transcription.lower().strip().replace(".", "").replace(",", "").replace("?", "").split()

            # If response is ONLY filler sounds, ignore it
            if all(w in filler_only for w in words):
                logger.debug(f"[{self.call_uuid[:8]}] Ignoring filler response: {transcription}")
                return

            # Require at least 2 words for a real response
            if len(words) < 2:
                logger.debug(f"[{self.call_uuid[:8]}] Ignoring short response ({len(words)} words): {transcription}")
                return

            # Save transcript locally
            self._save_transcript("USER", transcription)
            self._log_conversation("user", transcription)

            # Track goodbye
            if self._is_goodbye_message(transcription):
                self.user_said_goodbye = True
                self._check_mutual_goodbye()

            # Question Flow Mode: Get next question and inject
            if self.use_question_flow and self._question_flow:
                next_instruction = self._question_flow.advance(transcription)
                logger.info(f"[{self.call_uuid[:8]}] Q{self._question_flow.current_step}/{len(self._question_flow.questions)} | User: {transcription[:40]}...")
                await self._inject_question(next_instruction)

            # Send to n8n webhook if configured
            if self._transcript_webhook_url:
                asyncio.create_task(send_transcript_to_webhook(self, "user", transcription))

    async def _inject_question(self, instruction: str):
        """Inject the next question instruction into the AI"""
        if not self.goog_live_ws or not instruction:
            return

        try:
            msg = {
                "client_content": {
                    "turns": [{
                        "role": "user",
                        "parts": [{"text": f"[SYSTEM INSTRUCTION - FOLLOW EXACTLY]\n{instruction}"}]
                    }],
                    "turn_complete": True
                }
            }
            await self.goog_live_ws.send(json.dumps(msg))
            logger.debug(f"[{self.call_uuid[:8]}] Injected: {instruction[:50]}...")
        except Exception as e:
            logger.error(f"[{self.call_uuid[:8]}] Error injecting question: {e}")

    def _buffer_user_audio(self, audio_chunk: bytes):
        """Buffer user audio for transcription"""
        if not self._transcript_webhook_url and not self.use_question_flow:
            return

        # Add to buffer (with size limit)
        remaining_space = self._max_audio_buffer_size - len(self._user_audio_buffer)
        if remaining_space > 0:
            self._user_audio_buffer.extend(audio_chunk[:remaining_space])

    async def _send_session_setup(self):
        # Concise accent instruction (shorter = faster responses)
        accent_instruction = """VOICE: Indian English accent (Bangalore professional). Use "Actually...", "Basically...", "Isn't it?" naturally. English only, no Hindi. Warm and professional tone.

"""
        # Combine: accent first, then main prompt
        full_prompt = accent_instruction + self.prompt

        # On reconnect, load conversation from FILE (not memory - avoids latency issues)
        if not self._is_first_connection:
            # Load conversation history from file (saved by background thread)
            file_history = self._load_conversation_from_file()
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
                    # Low thinking budget to prevent AI from continuing on its own
                    "thinking_config": {
                        "thinking_budget": 256
                    }
                },
                # Enable input transcription to get real-time user speech transcripts
                "input_audio_transcription": {},  # Empty dict enables default transcription
                "system_instruction": {"parts": [{"text": full_prompt}]},
                "tools": [{"function_declarations": TOOL_DECLARATIONS}]
            }
        }
        await self.goog_live_ws.send(json.dumps(msg))

        # Note: _is_first_connection is set to False in setupComplete handler, not here
        logger.info(f"Sent session setup with voice: {voice_name}, first_connection={self._is_first_connection}")

    async def _send_initial_greeting(self):
        """Send initial trigger to make AI greet with first question"""
        if self.greeting_sent or not self.goog_live_ws:
            return
        self.greeting_sent = True

        # Question Flow Mode: Include the first question in the greeting
        if self.use_question_flow and self._question_flow:
            first_instruction = self._question_flow.get_instruction_prompt()
            trigger_text = f"[START THE CALL NOW]\n{first_instruction}"
            logger.debug(f"[{self.call_uuid[:8]}] Starting with question #1")
        else:
            trigger_text = "Hi"

        msg = {
            "client_content": {
                "turns": [{
                    "role": "user",
                    "parts": [{"text": trigger_text}]
                }],
                "turn_complete": True
            }
        }
        await self.goog_live_ws.send(json.dumps(msg))
        logger.info("Sent initial greeting trigger")

    async def _send_reconnection_trigger(self):
        """Trigger AI to speak immediately after reconnection, restoring question flow state"""
        if not self.goog_live_ws:
            return

        # Restore question flow state on reconnect
        if self.use_question_flow and self._question_flow:
            current_step = self._question_flow.current_step
            current_question = self._question_flow.get_current_question()

            if current_question:
                reconnect_text = f"""Continue with this question:
"{current_question}"
Wait for customer response after asking."""
            else:
                reconnect_text = "[System: Connection restored. Say 'Sorry about that...' and wrap up the call.]"

            logger.debug(f"[{self.call_uuid[:8]}] Restoring to question {current_step + 1}")
        else:
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

            logger.info(f"[{self.call_uuid[:8]}] Tool: {tool_name}")
            self._save_transcript("TOOL", f"{tool_name}: {tool_args}")

            # Handle end_call tool - mark agent as said goodbye, wait for mutual farewell
            if tool_name == "end_call":
                reason = tool_args.get("reason", "conversation ended")
                logger.info(f"[{self.call_uuid[:8]}] End call: {reason}")
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

                # Debug: log all serverContent keys to find user transcription field
                sc_keys = list(sc.keys())
                if sc_keys != ['modelTurn'] and sc_keys != ['turnComplete']:
                    logger.debug(f"serverContent keys: {sc_keys}")

                # Check if turn is complete (greeting done)
                if sc.get("turnComplete"):
                    self._preload_complete.set()
                    self.greeting_audio_complete = True
                    self._turn_count += 1

                    # Log turn latency at DEBUG level
                    if self._turn_start_time and self._current_turn_audio_chunks > 0:
                        turn_duration_ms = (time.time() - self._turn_start_time) * 1000
                        logger.debug(f"[{self.call_uuid[:8]}] Turn #{self._turn_count}: {self._current_turn_audio_chunks} chunks, {turn_duration_ms:.0f}ms")
                        self._turn_start_time = None

                    # AFTER AI finishes speaking, transcribe buffered user audio via REST API
                    # This ensures transcription happens after user speaks and AI responds
                    if self._turn_count > 1 and (self._transcript_webhook_url or self.use_question_flow):
                        asyncio.create_task(self._process_user_audio_for_transcription())

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

                if sc.get("interrupted"):
                    logger.debug(f"[{self.call_uuid[:8]}] AI interrupted")
                    if self.plivo_ws:
                        await self.plivo_ws.send_text(json.dumps({"event": "clearAudio", "stream_id": self.stream_id}))

                # Capture user speech transcription from Gemini
                # Debug: log all serverContent keys to find transcript field
                if sc_keys != ['modelTurn'] and sc_keys != ['turnComplete']:
                    logger.info(f"[{self.call_uuid[:8]}] DEBUG:SC_KEYS | {sc_keys}")

                if "inputTranscript" in sc:
                    user_text = sc["inputTranscript"]
                    logger.debug(f"[{self.call_uuid[:8]}] Input transcript: {user_text}")
                    if user_text and user_text.strip():
                        self._last_user_speech_time = time.time()  # Track for latency
                        logger.info(f"[{self.call_uuid[:8]}] USER: {user_text}")
                        self._save_transcript("USER", user_text.strip())
                        # Log to file in background thread (no latency impact)
                        self._log_conversation("user", user_text.strip())
                        # Send to n8n webhook for state machine (non-blocking)
                        asyncio.create_task(send_transcript_to_webhook(self, "user", user_text.strip()))
                        # Track if user said goodbye
                        if self._is_goodbye_message(user_text):
                            logger.debug(f"[{self.call_uuid[:8]}] User goodbye detected")
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
                                    # Log first chunk sent to Plivo for this turn
                                    if self._current_turn_audio_chunks == 1:
                                        logger.debug(f"[{self.call_uuid[:8]}] Sending audio to Plivo")
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
                                # Log to file in background thread (no latency impact)
                                self._log_conversation("model", ai_text)
                                # Send to n8n webhook for state machine (non-blocking)
                                asyncio.create_task(send_transcript_to_webhook(self, "agent", ai_text))
                                # Track if agent said goodbye (don't end immediately - wait for user)
                                if not self._closing_call and self._is_goodbye_message(ai_text):
                                    logger.debug(f"[{self.call_uuid[:8]}] Agent goodbye detected")
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

        # Calculate call duration
        duration = 0
        if self.call_start_time:
            duration = (datetime.now() - self.call_start_time).total_seconds()
            logger.info(f"[{self.call_uuid[:8]}] Duration: {duration:.1f}s")
            self._save_transcript("SYSTEM", f"Call duration: {duration:.1f}s")

        # Cleanup question flow and get collected data
        if self.use_question_flow:
            flow_data = remove_flow(self.call_uuid)
            if flow_data:
                logger.debug(f"[{self.call_uuid[:8]}] Flow: {len(flow_data.get('responses', {}))} responses")

        if self.goog_live_ws:
            try:
                await self.goog_live_ws.close()
            except:
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

            # Fallback to in-memory transcript if file is empty
            if not transcript.strip() and self._full_transcript:
                transcript = "\n".join([
                    f"[{t['timestamp']}] {t['role']}: {t['text']}"
                    for t in self._full_transcript
                ])
                logger.info(f"Using in-memory transcript ({len(self._full_transcript)} entries)")

            payload = {
                "event": "call_ended",
                "call_uuid": self.call_uuid,
                "caller_phone": self.caller_phone,
                "duration_seconds": round(duration, 1),
                "timestamp": datetime.now().isoformat(),
                "transcript": transcript,
                "transcript_entries": self._full_transcript  # Also send as structured data
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
        await session.goog_live_ws.send(json.dumps(msg))
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
    n8n_webhook_url: str = None,
    call_end_webhook_url: str = None,
    client_name: str = "fwai"
) -> bool:
    """
    Preload a session in conversational flow mode.
    Uses QuestionFlow to inject questions one by one.

    Args:
        call_uuid: Unique call identifier
        caller_phone: Caller's phone number
        base_prompt: IGNORED - QuestionFlow uses config file
        initial_phase_prompt: IGNORED - QuestionFlow handles this
        context: Context dict with customer_name, etc.
        n8n_webhook_url: URL to send real-time transcripts (optional)
        call_end_webhook_url: URL to call when call ends
        client_name: Client config to use (e.g., 'fwai')

    Returns:
        True if preload succeeded
    """
    # Create session with QuestionFlow enabled (uses config file, not passed prompt)
    session = PlivoGeminiSession(
        call_uuid,
        caller_phone,
        prompt=None,  # QuestionFlow will use base_prompt from config
        context=context,
        webhook_url=call_end_webhook_url,
        transcript_webhook_url=n8n_webhook_url,
        client_name=client_name,
        use_question_flow=True  # Explicitly enable QuestionFlow
    )

    _preloading_sessions[call_uuid] = session
    success = await session.preload()
    return success


async def send_transcript_to_webhook(session, role: str, text: str, max_retries: int = 2):
    """
    Send real-time transcript to n8n for state machine processing.
    Called when user speaks (for intent detection) or agent speaks (for tracking).
    Non-blocking with retry mechanism.
    """
    if not hasattr(session, '_transcript_webhook_url') or not session._transcript_webhook_url:
        return

    logger.debug(f"[{session.call_uuid[:8]}] Webhook: {role}: {text[:30]}...")

    try:
        import httpx

        payload = {
            "event": "transcript",
            "call_uuid": session.call_uuid,
            "role": role,  # "user" or "agent"
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "turn_count": getattr(session, '_turn_count', 0)
        }

        # Retry mechanism with short timeout
        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.post(session._transcript_webhook_url, json=payload)
                    if resp.status_code == 200:
                        logger.debug(f"Transcript webhook sent: {role}: {text[:30]}...")
                        return
                    else:
                        logger.warning(f"Webhook returned {resp.status_code}, attempt {attempt + 1}")
            except httpx.TimeoutException:
                if attempt < max_retries:
                    logger.warning(f"Webhook timeout, retrying ({attempt + 1}/{max_retries})...")
                    await asyncio.sleep(0.5)  # Brief pause before retry
                else:
                    logger.error(f"Webhook failed after {max_retries + 1} attempts (timeout)")
            except Exception as e:
                logger.error(f"Webhook error: {e}")
                break

    except Exception as e:
        logger.error(f"Error in transcript webhook: {e}")
