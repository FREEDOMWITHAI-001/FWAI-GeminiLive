"""
AI Voice Call with Plivo - Main Server

Python-based implementation using aiortc for full audio access
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List
from pathlib import Path
from loguru import logger
import sys

from fastapi import FastAPI, Request, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel
import json

from src.core.config import config
from src.prompt_loader import FWAI_PROMPT
from src.conversation_memory import add_message, get_history, clear_conversation
from src.db.session_db import session_db
from fastapi.staticfiles import StaticFiles

# Audio directory for any generated audio files
AUDIO_DIR = Path(__file__).parent.parent / "audio"
from datetime import datetime
from src.handlers.webrtc_handler import (
    make_outbound_call,
    handle_incoming_call,
    handle_ice_candidate,
    terminate_call,
    get_active_calls
)

def save_transcript(call_uuid: str, role: str, message: str):
    """Save transcript to a file for each call (if enabled)"""
    if not config.enable_transcripts:
        return
    try:
        transcript_dir = Path(__file__).parent.parent / "transcripts"
        transcript_dir.mkdir(exist_ok=True)
        transcript_file = transcript_dir / f"{call_uuid}.txt"
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(transcript_file, "a") as f:
            f.write(f"[{timestamp}] {role}: {message}" + chr(10))
    except Exception as e:
        logger.error(f"Error saving transcript: {e}")


# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG" if config.debug else "INFO"
)
logger.add(
    Path(__file__).parent.parent / "logs" / "fwai_voice.log",
    rotation="10 MB",
    retention="7 days",
    level=config.log_level  # Use config (default INFO, set LOG_LEVEL in .env)
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("=" * 60)
    logger.info("AI Voice Call with Plivo")
    logger.info("=" * 60)

    # Validate configuration
    errors = config.validate_config()
    if errors:
        for error in errors:
            logger.warning(f"Config warning: {error}")

    # Initialize Meta token manager (auto-refresh)
    try:
        from src.services.meta_token_manager import token_manager
        await token_manager.initialize()
        logger.info("Meta token manager initialized")
    except Exception as e:
        logger.warning(f"Token manager init failed: {e} - using static token")

    logger.info(f"Server starting on http://{config.host}:{config.port}")
    logger.info(f"Gemini Voice: {config.tts_voice}")
    logger.info(f"Session DB: {session_db._db_path}")
    logger.info(f"Plivo Callback URL: {config.plivo_callback_url}")
    logger.info(f"Answer URL: {config.plivo_callback_url}/plivo/answer")

    # Verify ngrok tunnel is accessible
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(config.plivo_callback_url, timeout=3.0)
            if resp.status_code == 200:
                logger.info("✅ Callback URL is accessible")
            else:
                logger.warning(f"⚠️  Callback URL returned {resp.status_code} - ngrok may be down")
    except Exception as e:
        logger.error(f"❌ Callback URL not accessible: {e}")
        logger.error("   Plivo will NOT be able to call answer webhook!")
        logger.error("   Start ngrok: ngrok http 3001")

    # Start background cleanup task for stale pending data
    async def _cleanup_stale_data():
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            try:
                from src.services.plivo_gemini_stream import _sessions, _preloading_sessions, _sessions_lock
                # Collect active session UUIDs
                async with _sessions_lock:
                    active_uuids = set(_sessions.keys()) | set(_preloading_sessions.keys())
                # Clean stale pending data (calls that ended without proper cleanup)
                async with _call_data_lock:
                    stale_keys = [k for k in _pending_call_data if k not in active_uuids]
                    for k in stale_keys:
                        _pending_call_data.pop(k, None)
                    # Clean stale UUID mappings
                    stale_internal = [k for k in _internal_to_plivo_uuid if k not in active_uuids]
                    for k in stale_internal:
                        plivo_uuid = _internal_to_plivo_uuid.pop(k, None)
                        if plivo_uuid:
                            _plivo_to_internal_uuid.pop(plivo_uuid, None)
                if stale_keys or stale_internal:
                    logger.info(f"Cleanup: removed {len(stale_keys)} stale pending, {len(stale_internal)} stale UUID mappings")
                # Clean stale DB records
                session_db.cleanup_stale(max_age_minutes=10)
            except Exception as e:
                logger.debug(f"Cleanup task error: {e}")

    cleanup_task = asyncio.create_task(_cleanup_stale_data())

    yield

    # Shutdown
    cleanup_task.cancel()
    session_db.shutdown()
    logger.info("Server shutting down...")


# Create FastAPI app
app = FastAPI(
    title="AI Voice Call with Plivo",
    description="AI Voice Agent for WhatsApp Business Voice Calls",
    version="1.0.0",
    lifespan=lifespan
)

# Mount audio directory for serving generated audio files
AUDIO_DIR.mkdir(exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")

# Mount static directory for UI files
STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Request models
class MakeCallRequest(BaseModel):
    phoneNumber: str
    contactName: Optional[str] = "Customer"


class WebhookVerification(BaseModel):
    hub_mode: str
    hub_verify_token: str
    hub_challenge: str


class PromptUpdateRequest(BaseModel):
    client_name: str
    prompt: str


# ============================================================================
# Health Check
# ============================================================================

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "AI Voice Call with Plivo",
        "version": "1.0.0"
    }


# ============================================================================
# Prompts Management
# ============================================================================

@app.get("/prompts/{client_name}")
async def get_prompt(client_name: str):
    """Get prompt and config for a client"""
    try:
        prompts_dir = Path(__file__).parent.parent / "prompts"
        prompt_file = prompts_dir / f"{client_name}_prompt.txt"
        config_file = prompts_dir / f"{client_name}_config.json"

        # Read prompt
        if prompt_file.exists():
            with open(prompt_file, "r") as f:
                prompt_content = f.read()
        else:
            prompt_content = ""

        # Read config
        if config_file.exists():
            with open(config_file, "r") as f:
                config_data = json.load(f)
        else:
            config_data = {}

        return {
            "success": True,
            "prompt": prompt_content,
            "config": config_data,
            "clientName": client_name
        }
    except Exception as e:
        logger.error(f"Error loading prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prompts/update")
async def update_prompt(request: PromptUpdateRequest):
    """Update prompt and config for a client"""
    try:
        prompts_dir = Path(__file__).parent.parent / "prompts"
        prompts_dir.mkdir(exist_ok=True)

        prompt_file = prompts_dir / f"{request.client_name}_prompt.txt"

        # Save prompt
        with open(prompt_file, "w") as f:
            f.write(request.prompt)

        logger.info(f"Updated prompt for client: {request.client_name}")
        return {"success": True, "message": "Prompt updated successfully"}
    except Exception as e:
        logger.error(f"Error updating prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Make Outbound Call
# ============================================================================

@app.post("/make-call")
async def api_make_call(request: MakeCallRequest):
    """
    Make an outbound call to a WhatsApp user

    The AI agent will start speaking when the user answers
    """
    logger.info(f"Make call request: {request.phoneNumber}")

    result = await make_outbound_call(
        phone_number=request.phoneNumber,
        caller_name=request.contactName
    )

    if result.get("success"):
        return JSONResponse(content=result)
    else:
        raise HTTPException(status_code=400, detail=result.get("error"))


# ============================================================================
# WhatsApp Message Webhook
# ============================================================================

@app.get("/webhook")
async def verify_webhook(
    request: Request,
):
    """Verify WhatsApp webhook"""
    hub_mode = request.query_params.get("hub.mode")
    hub_verify_token = request.query_params.get("hub.verify_token")
    hub_challenge = request.query_params.get("hub.challenge")

    logger.info(f"Webhook verification: mode={hub_mode}")

    if hub_mode == "subscribe" and hub_verify_token == config.meta_verify_token:
        logger.info("Webhook verified successfully")
        return PlainTextResponse(content=hub_challenge)
    else:
        logger.warning("Webhook verification failed")
        raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def handle_webhook(request: Request):
    """Handle WhatsApp message webhook"""
    try:
        body = await request.json()
        logger.debug(f"Webhook received: {body}")

        # Process message events
        if "entry" in body:
            for entry in body.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})

                    # Handle incoming messages
                    if "messages" in value:
                        for message in value.get("messages", []):
                            sender = message.get("from")
                            msg_type = message.get("type")

                            if msg_type == "text":
                                text = message.get("text", {}).get("body", "")
                                logger.info(f"Message from {sender}: {text}")

        return JSONResponse(content={"status": "ok"})

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)})


# ============================================================================
# WhatsApp Call Events Webhook
# ============================================================================

@app.get("/call-events")
async def verify_call_webhook(request: Request):
    """Verify call events webhook"""
    hub_mode = request.query_params.get("hub.mode")
    hub_verify_token = request.query_params.get("hub.verify_token")
    hub_challenge = request.query_params.get("hub.challenge")

    logger.info(f"Call webhook verification: mode={hub_mode}")

    if hub_mode == "subscribe" and hub_verify_token == config.meta_verify_token:
        logger.info("Call webhook verified successfully")
        return PlainTextResponse(content=hub_challenge)
    else:
        logger.warning("Call webhook verification failed")
        raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/call-events")
async def handle_call_events(request: Request):
    """Handle WhatsApp call events webhook"""
    try:
        body = await request.json()
        logger.info(f"Call event received: {body}")

        if "entry" in body:
            for entry in body.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})

                    # Handle call events
                    if "call" in value:
                        call_data = value.get("call", {})
                        call_event = call_data.get("call_event")
                        call_id = call_data.get("call_id")

                        logger.info(f"Call event: {call_event} for call {call_id}")

                        if call_event == "connect":
                            # Incoming call
                            caller = call_data.get("from", {})
                            caller_phone = caller.get("phone_number", "")
                            caller_name = caller.get("name", "Customer")
                            sdp_offer = call_data.get("sdp", "")

                            if sdp_offer:
                                # Handle incoming call asynchronously
                                asyncio.create_task(
                                    handle_incoming_call(
                                        call_id=call_id,
                                        caller_phone=caller_phone,
                                        sdp_offer=sdp_offer,
                                        caller_name=caller_name
                                    )
                                )

                        elif call_event == "answer":
                            # Call was answered (for outbound calls)
                            sdp_answer = call_data.get("sdp", "")
                            logger.info(f"Call answered: {call_id}")

                        elif call_event == "ice_candidate":
                            # ICE candidate
                            candidate = call_data.get("ice_candidate", {})
                            await handle_ice_candidate(call_id, candidate)

                        elif call_event in ["terminate", "reject", "timeout"]:
                            # Call ended
                            logger.info(f"Call ended: {call_id} ({call_event})")
                            await terminate_call(call_id)

        return JSONResponse(content={"status": "ok"})

    except Exception as e:
        logger.error(f"Call event error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"status": "error", "message": str(e)})


# ============================================================================
# Call Management Endpoints
# ============================================================================

@app.get("/calls")
async def list_calls():
    """List active calls"""
    return {"calls": get_active_calls()}


@app.post("/calls/{call_id}/terminate")
async def api_terminate_call(call_id: str):
    """Terminate a specific call"""
    result = await terminate_call(call_id)
    if result.get("success"):
        return JSONResponse(content=result)
    else:
        raise HTTPException(status_code=404, detail=result.get("error"))


@app.get("/calls/{call_id}/transcript")
async def get_call_transcript(call_id: str):
    """
    Get transcript for a call by call_uuid

    Returns the Whisper transcript (_final.txt) if available,
    otherwise returns the real-time transcript (.txt)
    """
    from pathlib import Path

    transcript_dir = Path(__file__).parent.parent / "transcripts"

    # Try final (Whisper) transcript first
    final_transcript = transcript_dir / f"{call_id}_final.txt"
    realtime_transcript = transcript_dir / f"{call_id}.txt"

    transcript = None
    transcript_type = None

    if final_transcript.exists():
        transcript = final_transcript.read_text()
        transcript_type = "whisper"
    elif realtime_transcript.exists():
        transcript = realtime_transcript.read_text()
        transcript_type = "realtime"

    if transcript:
        return JSONResponse(content={
            "success": True,
            "call_uuid": call_id,
            "transcript_type": transcript_type,
            "transcript": transcript
        })
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Transcript not found for call {call_id}. Call may still be in progress or transcripts may be disabled."
        )


# ============================================================================
# Main Entry Point
# ============================================================================

# ============================================================================
# Plivo Endpoints
# ============================================================================

# Import Plivo adapter
from src.adapters.plivo_adapter import plivo_adapter

# Store call data for pending calls (call_uuid -> {phone, prompt, context})
_pending_call_data = {}
# Mapping from Plivo's request_uuid to our internal call_uuid (for preloaded sessions)
_plivo_to_internal_uuid = {}
# Reverse mapping: internal UUID to Plivo UUID
_internal_to_plivo_uuid = {}
# Lock for all call data dicts above
_call_data_lock = asyncio.Lock()


class PlivoMakeCallRequest(BaseModel):
    phoneNumber: str
    contactName: Optional[str] = "Customer"
    prompt: Optional[str] = None  # Custom AI prompt (optional, uses default if not provided)
    context: Optional[dict] = None  # Context for templates: customer_name, course_name, price, etc.
    webhookUrl: Optional[str] = None  # URL to call when call ends (for n8n integration)


@app.post("/plivo/make-call")
async def plivo_make_call(request: PlivoMakeCallRequest):
    """
    Make an outbound call using Plivo with Gemini Live AI

    Flow:
    1. Preload Gemini session FIRST (AI ready before phone rings)
    2. Plivo API initiates call to the phone number
    3. When user answers, Plivo hits /plivo/answer
    4. /plivo/answer returns <Stream> XML
    5. Plivo connects WebSocket to /plivo/stream/{call_uuid}
    6. AI speaks immediately (already preloaded)
    """
    logger.info(f"Plivo make call request: {request.phoneNumber}")

    try:
        import uuid
        # Generate call_uuid first (before Plivo call)
        call_uuid = str(uuid.uuid4())

        # Add customer_name to context if not present
        context = request.context or {}
        context.setdefault("customer_name", request.contactName)

        # Store all call data
        async with _call_data_lock:
            _pending_call_data[call_uuid] = {
                "phone": request.phoneNumber,
                "prompt": request.prompt,
                "context": context,
                "webhookUrl": request.webhookUrl
            }

        # Record call in DB (non-blocking)
        session_db.create_call(
            call_uuid=call_uuid, phone=request.phoneNumber,
            contact_name=request.contactName, webhook_url=request.webhookUrl
        )

        # PRELOAD Gemini session FIRST (before phone rings)
        from src.services.plivo_gemini_stream import preload_session
        logger.info(f"Preloading Gemini session for {call_uuid}...")
        await preload_session(
            call_uuid,
            request.phoneNumber,
            prompt=request.prompt,
            context=context,
            webhook_url=request.webhookUrl
        )
        logger.info(f"Gemini preload complete for {call_uuid} - now making call")

        # NOW make the Plivo call (AI is already ready)
        result = await plivo_adapter.make_call(
            phone_number=request.phoneNumber,
            caller_name=request.contactName
        )

        if result.get("success"):
            # Map Plivo's UUID to our internal UUID (for preloaded session lookup)
            plivo_uuid = result.get("call_uuid")
            if plivo_uuid:
                async with _call_data_lock:
                    _plivo_to_internal_uuid[plivo_uuid] = call_uuid
                    _internal_to_plivo_uuid[call_uuid] = plivo_uuid
                    if call_uuid in _pending_call_data:
                        _pending_call_data[call_uuid]["plivo_uuid"] = plivo_uuid
                logger.info(f"UUID mapping: Plivo {plivo_uuid} -> Internal {call_uuid}")

                # Set Plivo UUID on the preloaded session for hangup
                from src.services.plivo_gemini_stream import set_plivo_uuid
                set_plivo_uuid(call_uuid, plivo_uuid)

            return JSONResponse(content={
                "success": True,
                "call_uuid": call_uuid,
                "plivo_uuid": plivo_uuid,
                "message": f"Call initiated to {request.phoneNumber}. Waiting for user to answer."
            })
        else:
            logger.error(f"Plivo call failed: {result.get('error')}")
            raise HTTPException(status_code=400, detail=result.get("error"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making Plivo call: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plivo/answer")
async def plivo_answer(request: Request):
    """Handle Plivo call answer - uses Stream with Gemini Live"""
    try:
        body = await request.form()
        plivo_uuid = body.get("CallUUID", "")
        from_phone = body.get("From", "")
        to_phone = body.get("To", "")

        logger.info(f"[ANSWER] Plivo answer webhook received: CallUUID={plivo_uuid}, From={from_phone}, To={to_phone}")

        # Look up our internal UUID from Plivo's UUID (for preloaded sessions)
        async with _call_data_lock:
            internal_uuid = _plivo_to_internal_uuid.get(plivo_uuid, plivo_uuid)
            # For outbound calls, customer is "To"; for inbound, customer is "From"
            customer_phone = to_phone if to_phone and not to_phone.startswith("91226") else from_phone
            if customer_phone and internal_uuid and internal_uuid not in _pending_call_data:
                _pending_call_data[internal_uuid] = {"phone": customer_phone, "prompt": None, "context": {}}

        logger.info(f"[ANSWER] Call answered: plivo={plivo_uuid}, internal={internal_uuid}, from {from_phone} to {to_phone}")

        # WebSocket URL for bidirectional audio stream (use internal UUID for preloaded session)
        ws_base = config.plivo_callback_url.replace("https://", "wss://").replace("http://", "ws://")
        stream_url = f"{ws_base}/plivo/stream/{internal_uuid}"

        logger.info(f"[ANSWER] Stream URL: {stream_url}")

        # Use Speak first, then Stream for bidirectional audio
        status_url = f"{config.plivo_callback_url}/plivo/stream-status"
        xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream streamTimeout="86400" keepCallAlive="true" bidirectional="true" contentType="audio/x-l16;rate=16000" statusCallbackUrl="{status_url}">{stream_url}</Stream>
</Response>"""

        logger.info(f"[ANSWER] Returning Stream XML for {internal_uuid}")
        return Response(content=xml_response, media_type="application/xml")

    except Exception as e:
        logger.error(f"[ANSWER] Error in plivo_answer: {e}")
        import traceback
        traceback.print_exc()
        # Return minimal valid XML to prevent call drop
        return Response(
            content='<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>',
            media_type="application/xml"
        )


@app.websocket("/plivo/stream/{call_uuid}")
async def plivo_stream(websocket: WebSocket, call_uuid: str):
    """
    WebSocket endpoint for Plivo bidirectional audio stream.
    Bridges Plivo audio with Gemini 2.5 Live for real-time voice AI.
    """
    from src.services.plivo_gemini_stream import create_session, remove_session

    await websocket.accept()
    logger.info(f"Plivo stream WebSocket connected for call {call_uuid}")

    session = None
    caller_phone = ""

    try:
        while True:
            # Receive message from Plivo
            data = await websocket.receive_text()
            message = json.loads(data)
            event = message.get("event")

            if event == "start":
                # Stream started - create Gemini session
                start_data = message.get("start", {})
                logger.info(f"Plivo start event data: {start_data}")
                # Note: call_uuid in URL is already our internal UUID (from /plivo/answer)
                # Get phone from customParameters or fallback to stored value
                custom_params = start_data.get("customParameters", {})
                caller_phone = custom_params.get("callerPhone", "") or start_data.get("to", "") or start_data.get("from", "")
                # If still empty, use the call_uuid to look up from pending calls
                if not caller_phone:
                    async with _call_data_lock:
                        call_data = _pending_call_data.get(call_uuid, {})
                    caller_phone = call_data.get("phone", "")
                # Ensure it has country code format
                if caller_phone and not caller_phone.startswith("+"):
                    caller_phone = "+" + caller_phone

                logger.info(f"Plivo stream started: {call_uuid} to {caller_phone}")

                # Create Gemini Live session
                session = await create_session(call_uuid, caller_phone, websocket)

                if session:
                    logger.info(f"Gemini Live session created for {call_uuid}")
                    session_db.update_call(call_uuid, status="active",
                                           started_at=datetime.now().isoformat())
                    # Log success - webhook was called and stream started
                    logger.info(f"[STREAM] ✅ WebSocket connected and session active for {call_uuid}")
                    # Ensure Plivo UUID is set (fallback if set_plivo_uuid missed it)
                    if not session.plivo_call_uuid:
                        # Try to get from pending call data or mapping
                        async with _call_data_lock:
                            call_data = _pending_call_data.get(call_uuid, {})
                            plivo_uuid = call_data.get("plivo_uuid") or _internal_to_plivo_uuid.get(call_uuid)
                        if plivo_uuid:
                            session.plivo_call_uuid = plivo_uuid
                            logger.info(f"Set Plivo UUID {plivo_uuid} on session {call_uuid} (fallback)")
                        else:
                            logger.warning(f"Could not find Plivo UUID for session {call_uuid} - hangup may fail")
                else:
                    logger.error(f"Failed to create Gemini session for {call_uuid}")

            elif event == "media":
                # Audio from caller - forward to Gemini
                if session:
                    await session.handle_plivo_message(message)

            elif event == "stop":
                # Stream stopped
                logger.info(f"Plivo stream stopped for {call_uuid}")
                break

            elif event == "dtmf":
                # DTMF digit
                digit = message.get("dtmf", {}).get("digit", "")
                logger.info(f"DTMF received: {digit}")

    except WebSocketDisconnect:
        logger.info(f"Plivo stream WebSocket disconnected for {call_uuid}")
    except Exception as e:
        logger.error(f"Plivo stream error for {call_uuid}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if session:
            await remove_session(call_uuid)
        clear_conversation(call_uuid)
        logger.info(f"Plivo stream cleanup complete for {call_uuid}")


@app.post("/plivo/stream-status")
async def plivo_stream_status(request: Request):
    """Handle Plivo stream status callbacks"""
    body = await request.form()
    # Reduced logging - was firing 50+ times per call
    logger.debug(f"Plivo stream status: {dict(body)}")
    return JSONResponse(content={"status": "ok"})


@app.post("/plivo/hangup")
async def plivo_hangup(request: Request):
    """Handle Plivo call hangup"""
    body = await request.form()
    plivo_uuid = body.get("CallUUID", "")
    duration = body.get("Duration", "0")

    # Look up internal UUID for cleanup
    async with _call_data_lock:
        internal_uuid = _plivo_to_internal_uuid.pop(plivo_uuid, plivo_uuid)
        _pending_call_data.pop(internal_uuid, None)
        _internal_to_plivo_uuid.pop(internal_uuid, None)

    logger.info(f"Plivo call ended: plivo={plivo_uuid}, internal={internal_uuid}, duration: {duration}s")
    clear_conversation(internal_uuid)

    return JSONResponse(content={"status": "ok"})


# ============= CONVERSATIONAL FLOW ENDPOINTS =============

class InjectContextRequest(BaseModel):
    """Request to inject dynamic context into an ongoing call"""
    phase: str  # Current conversation phase
    context: str  # Dynamic prompt/context to inject
    data: Optional[dict] = None  # Any captured data (name, role, etc.)


@app.post("/call/inject/{call_id}")
async def inject_context(call_id: str, request: InjectContextRequest):
    """
    Inject dynamic context into an ongoing call.
    Called by n8n to change conversation phase or add context.
    """
    from src.services.plivo_gemini_stream import inject_context_to_session

    logger.info(f"[{call_id[:8]}] INJECT | Phase: {request.phase}")

    success = await inject_context_to_session(
        call_id,
        request.phase,
        additional_context=request.context,
        data=request.data
    )

    if success:
        return JSONResponse(content={
            "success": True,
            "call_id": call_id,
            "phase": request.phase
        })
    else:
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found or not active")


class ConversationalCallRequest(BaseModel):
    """Request to start a conversational flow call"""
    phoneNumber: str
    contactName: str = "Customer"
    callEndWebhookUrl: Optional[str] = None  # URL when call ends
    context: Optional[dict] = None
    clientName: Optional[str] = None  # Client name for loading specific prompt (e.g., 'fwai', 'ridhi')
    questions: List[dict]  # Questions: [{"id": "q1", "prompt": "..."}] — required
    prompt: str  # Base system instruction prompt — required
    objections: Optional[dict] = None  # Objection responses
    objectionKeywords: Optional[dict] = None  # Objection keywords
    instructionTemplates: Optional[dict] = None  # Override instruction texts (nudge, wrap-up, greeting, etc.)


@app.post("/call/conversational")
async def start_conversational_call(request: ConversationalCallRequest):
    """
    Start a call with conversational flow mode.
    Loads client-specific prompt from prompts/{clientName}_prompt.txt
    AI handles the flow naturally (no phase injection).
    """
    from src.services.plivo_gemini_stream import preload_session_conversational

    logger.info(f"Starting conversational call to {request.phoneNumber}, client: {request.clientName}")

    try:
        import uuid
        call_uuid = str(uuid.uuid4())

        # Build context with customer name
        context = request.context or {}
        context.setdefault("customer_name", request.contactName)
        context["conversational_mode"] = True

        # QuestionFlow mode: Don't load full prompt file, let QuestionFlow use config
        client_name = request.clientName or "fwai"
        logger.info(f"[{call_uuid[:8]}] QuestionFlow mode for client: {client_name}")

        # Store call data
        _pending_call_data[call_uuid] = {
            "phone": request.phoneNumber,
            "context": context,
            "webhookUrl": request.callEndWebhookUrl,
            "conversational_mode": True,
            "clientName": client_name
        }

        # Record call in DB (non-blocking)
        total_q = len(request.questions)
        logger.info(f"[{call_uuid[:8]}] {total_q} questions, prompt={len(request.prompt)} chars")
        session_db.create_call(
            call_uuid=call_uuid, phone=request.phoneNumber,
            contact_name=request.contactName, client_name=client_name,
            webhook_url=request.callEndWebhookUrl,
            total_questions=total_q
        )

        # Preload session with QuestionFlow
        await preload_session_conversational(
            call_uuid,
            request.phoneNumber,
            context=context,
            call_end_webhook_url=request.callEndWebhookUrl,
            client_name=client_name,
            questions_override=request.questions,
            prompt_override=request.prompt,
            objections_override=request.objections,
            objection_keywords_override=request.objectionKeywords,
            instruction_templates=request.instructionTemplates
        )

        # Make the Plivo call
        result = await plivo_adapter.make_call(
            phone_number=request.phoneNumber,
            caller_name=request.contactName
        )

        if result.get("success"):
            plivo_uuid = result.get("call_uuid")
            if plivo_uuid:
                _plivo_to_internal_uuid[plivo_uuid] = call_uuid
                _internal_to_plivo_uuid[call_uuid] = plivo_uuid

                from src.services.plivo_gemini_stream import set_plivo_uuid
                set_plivo_uuid(call_uuid, plivo_uuid)

            return JSONResponse(content={
                "success": True,
                "call_uuid": call_uuid,
                "plivo_uuid": plivo_uuid,
                "mode": "conversational",
                "message": f"Conversational call initiated to {request.phoneNumber}"
            })
        else:
            raise HTTPException(status_code=400, detail=result.get("error"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting conversational call: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/call/history")
async def get_call_history(limit: int = Query(default=50, le=200)):
    """Get call history with statistics from session DB"""
    calls = await session_db.get_recent_calls(limit=limit)  # LATENCY OPT: Async DB
    return JSONResponse(content={"calls": calls, "total": len(calls)})


@app.get("/call/{call_id}/details")
async def get_call_details(call_id: str):
    """Get full call details including responses and statistics"""
    call = await session_db.get_call(call_id)  # LATENCY OPT: Async DB
    if call:
        return JSONResponse(content=call)
    raise HTTPException(status_code=404, detail=f"Call {call_id} not found")


@app.get("/call/phases")
async def get_available_phases():
    """Get list of available conversation phases for n8n"""
    from src.conversational_prompts import PHASE_PROMPTS, QUALIFICATION_RULES, DATA_FIELDS

    return JSONResponse(content={
        "phases": list(PHASE_PROMPTS.keys()),
        "qualification_rules": QUALIFICATION_RULES,
        "data_fields": DATA_FIELDS
    })


# ============= CALL METRICS ENDPOINT =============

@app.get("/call/metrics/{call_id}")
async def get_call_metrics(call_id: str):
    """Get latency metrics for a completed call"""
    from pathlib import Path
    metrics_file = Path(__file__).parent.parent / "flow_data" / f"{call_id}_metrics.json"
    if metrics_file.exists():
        return JSONResponse(content=json.load(open(metrics_file)))
    raise HTTPException(status_code=404, detail=f"Metrics not found for call {call_id}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.app:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="debug" if config.debug else "info"
    )