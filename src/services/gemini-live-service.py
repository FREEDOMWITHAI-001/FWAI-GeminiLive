"""
Gemini Live Service for WhatsApp Calls
Uses Gemini Live directly for real-time voice conversations via WebSocket
"""

import os
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
LOG_FILE = "gemini-live-service.log"

def log_to_file(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

# Overwrite log file on startup
with open(LOG_FILE, "w") as f:
    f.write(f"=== Gemini Live Service Log - Started at {datetime.now()} ===\n\n")

def log(message):
    print(message)
    log_to_file(message)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    log("✅ Loaded environment variables from .env file")
except ImportError:
    log("ℹ️  python-dotenv not installed. Using system environment variables only.")

# Try to add src to path
current_dir = os.path.dirname(__file__)
src_path = os.path.join(current_dir, 'src')
if os.path.exists(src_path):
    sys.path.insert(0, src_path)

# Pipecat imports
try:
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
    from pipecat.frames.frames import (
        StartFrame, EndFrame,
        AudioRawFrame, TTSAudioRawFrame,
        TranscriptionFrame, LLMTextFrame,
        LLMMessagesUpdateFrame
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineTask
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.processors.frame_processor import FrameProcessor
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
    
    PIPECAT_AVAILABLE = True
    log("✅ Pipecat imports successful")
except ImportError as e:
    log(f"❌ Pipecat not available: {e}")
    PIPECAT_AVAILABLE = False
    sys.exit(1)

# WebSocket imports
try:
    import websockets
    from websockets.server import serve
    WEBSOCKET_AVAILABLE = True
except ImportError:
    log("❌ websockets not installed. Install with: pip install websockets")
    WEBSOCKET_AVAILABLE = False
    sys.exit(1)

# Load conversation script
CONVERSATION_SCRIPT_PATH = "FAWI_Call_BOT.txt"

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
            log(f"✅ Detected female voice indicator '{indicator}' in prompt - using Kore voice")
            return "Kore"
    return "Puck"


def load_conversation_script():
    """Load FAWI_Call_BOT.txt and convert to system prompt"""
    try:
        if os.path.exists(CONVERSATION_SCRIPT_PATH):
            with open(CONVERSATION_SCRIPT_PATH, 'r', encoding='utf-8') as f:
                script_content = f.read()

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
- Keep responses conversational and natural"""

            log(f"✅ Loaded conversation script from {CONVERSATION_SCRIPT_PATH}")
            return system_prompt
        else:
            log(f"⚠️  Conversation script not found: {CONVERSATION_SCRIPT_PATH}")
            return "You are Mousumi, a Senior Counselor at Freedom with AI. Help people with AI skills and career guidance."
    except Exception as e:
        log(f"❌ Error loading conversation script: {e}")
        return "You are Mousumi, a Senior Counselor at Freedom with AI."


class WebSocketTransport(FrameProcessor):
    """Transport that bridges WebSocket audio ↔ Gemini Live"""
    def __init__(self, websocket, call_id):
        super().__init__()
        self.websocket = websocket
        self.call_id = call_id
        self.is_active = False
        
    async def send_audio(self, audio_data):
        """Send audio to Node.js via WebSocket"""
        if self.is_active and self.websocket:
            try:
                await self.websocket.send(json.dumps({
                    "type": "audio",
                    "data": audio_data.hex() if isinstance(audio_data, bytes) else audio_data
                }))
            except Exception as e:
                log(f"❌ Error sending audio: {e}")
    
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        
        # Handle audio output from Gemini Live
        if isinstance(frame, TTSAudioRawFrame):
            await self.send_audio(frame.audio)
        
        await self.push_frame(frame)
    
    async def start(self):
        self.is_active = True
        await self.push_frame(StartFrame())
        log(f"✅ WebSocket transport started for call {self.call_id}")
    
    async def stop(self):
        self.is_active = False
        await self.push_frame(EndFrame())
        log(f"✅ WebSocket transport stopped for call {self.call_id}")


class ChatLogger(FrameProcessor):
    """Logs conversation flow"""
    def __init__(self):
        super().__init__()
        self._bot_buffer = ""
    
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TranscriptionFrame):
            text = getattr(frame, 'text', '').strip()
            if text:
                log(f"👤 USER: {text}")
        
        elif isinstance(frame, LLMTextFrame):
            self._bot_buffer += frame.text
        
        elif isinstance(frame, LLMMessagesUpdateFrame):
            if self._bot_buffer.strip():
                log(f"🤖 BOT: {self._bot_buffer.strip()}")
                self._bot_buffer = ""
        
        await self.push_frame(frame)


# Global call sessions
active_calls = {}  # call_id -> {transport, pipeline, task, runner}


async def create_gemini_live_pipeline(websocket, call_id, caller_name):
    """Create Gemini Live pipeline for a call"""
    log(f"🚀 Creating Gemini Live pipeline for call {call_id}")
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        log("❌ GOOGLE_API_KEY not found in environment variables")
        return None
    
    # Create WebSocket transport
    transport = WebSocketTransport(websocket, call_id)

    # Load conversation script to detect voice
    system_prompt = load_conversation_script()
    voice_id = detect_voice_from_prompt(system_prompt)

    # Create Gemini Live service with detected voice
    llm = GeminiLiveLLMService(
        api_key=google_api_key,
        voice_id=voice_id
    )
    log(f"✅ Gemini Live initialized (Voice: {voice_id})")

    # Use the already loaded system_prompt for messages
    messages = [{"role": "system", "content": system_prompt}]
    context = LLMContext(messages=messages)
    context_aggregator = LLMContextAggregatorPair(context)
    
    # Create pipeline
    pipeline = Pipeline([
        transport,
        context_aggregator.user(),
        llm,
        ChatLogger(),
        context_aggregator.assistant(),
    ])
    
    # Create task and runner
    task = PipelineTask(pipeline, params={"allow_interruptions": True})
    runner = PipelineRunner()
    
    # Store in active calls
    active_calls[call_id] = {
        "transport": transport,
        "pipeline": pipeline,
        "task": task,
        "runner": runner,
        "caller_name": caller_name,
        "websocket": websocket
    }
    
    # Start transport
    await transport.start()
    
    # Start pipeline in background
    pipeline_task = asyncio.create_task(runner.run(task))
    
    # Store pipeline task for cleanup
    active_calls[call_id]["pipeline_task"] = pipeline_task
    
    # Wait for pipeline to initialize
    await asyncio.sleep(2.0)
    
    # Trigger initial greeting (like in test_gemini_live.py)
    # This ensures the agent speaks immediately when call starts
    greeting_text = f"Hello there! I'm Mousumi, a Senior Counselor at Freedom with AI. I'm thrilled to connect with you today to discuss how mastering AI skills can significantly elevate your career and income. You've already taken a great first step by attending our AI masterclass with Avinash. Now, let's explore how we can take your career to new heights together. This conversation is more for me to find out more about what you do and what you are doing in your career in terms of AI and see how we can help a little further after the masterclass. Is that fine?"
    
    trigger_msg = {
        "role": "user",
        "content": f"SYSTEM_COMMAND: The user has joined the call. Speak this EXACT greeting immediately: '{greeting_text}'"
    }
    
    try:
        # Use push_frame on transport instead of task.queue_frames
        # This is safer and works with the pipeline
        await transport.push_frame(
            LLMMessagesUpdateFrame(messages=[trigger_msg], run_llm=True)
        )
        log(f"✅ Initial greeting triggered for call {call_id}")
    except Exception as e:
        log(f"⚠️  Could not trigger initial greeting: {e}")
        log(f"   Error details: {type(e).__name__}: {str(e)}")
        import traceback
        log(f"   Traceback: {traceback.format_exc()}")
    
    log(f"✅ Gemini Live pipeline started for call {call_id}")
    return transport


async def handle_websocket(websocket):
    """Handle WebSocket connection from Node.js"""
    call_id = None
    caller_name = None
    
    try:
        # Get path from websocket.request if available (newer websockets versions)
        path = getattr(websocket, 'path', None) or (websocket.request.path if hasattr(websocket, 'request') else '/')
        log(f"📡 WebSocket connection established from {websocket.remote_address} (path: {path})")
        
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "start":
                    # Start a new call
                    call_id = data.get("call_id")
                    caller_name = data.get("caller_name", "Unknown")
                    
                    if call_id:
                        await create_gemini_live_pipeline(websocket, call_id, caller_name)
                        await websocket.send(json.dumps({"type": "started", "call_id": call_id}))
                    else:
                        await websocket.send(json.dumps({"type": "error", "message": "call_id required"}))
                
                elif msg_type == "audio":
                    # Receive audio from WhatsApp
                    if call_id and call_id in active_calls:
                        audio_data = bytes.fromhex(data.get("data", ""))
                        transport = active_calls[call_id]["transport"]
                        
                        # Convert to Pipecat frame
                        audio_frame = AudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)
                        await transport.push_frame(audio_frame)
                
                elif msg_type == "stop":
                    # Stop the call
                    if call_id and call_id in active_calls:
                        call_info = active_calls[call_id]
                        await call_info["transport"].stop()
                        # Push EndFrame through transport instead of task
                        await call_info["transport"].push_frame(EndFrame())
                        # Cancel pipeline task if it exists
                        if "pipeline_task" in call_info:
                            call_info["pipeline_task"].cancel()
                        del active_calls[call_id]
                        log(f"✅ Call {call_id} stopped")
                        await websocket.send(json.dumps({"type": "stopped", "call_id": call_id}))
                
            except json.JSONDecodeError:
                log(f"❌ Invalid JSON received: {message}")
            except Exception as e:
                log(f"❌ Error handling message: {e}")
    
    except websockets.exceptions.ConnectionClosed:
        log(f"📡 WebSocket connection closed for call {call_id}")
        if call_id and call_id in active_calls:
            call_info = active_calls[call_id]
            await call_info["transport"].stop()
            # Push EndFrame through transport instead of task
            await call_info["transport"].push_frame(EndFrame())
            # Cancel pipeline task if it exists
            if "pipeline_task" in call_info:
                call_info["pipeline_task"].cancel()
            del active_calls[call_id]
    except Exception as e:
        log(f"❌ WebSocket error: {e}")


async def main():
    """Start WebSocket server"""
    port = int(os.getenv("GEMINI_LIVE_PORT", 8003))
    log(f"🎙️  Gemini Live Service starting on ws://0.0.0.0:{port}")
    
    async with serve(handle_websocket, "0.0.0.0", port):
        log(f"✅ Gemini Live Service running on ws://0.0.0.0:{port}")
        log(f"   Waiting for connections from Node.js server...")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("🛑 Server stopped")
