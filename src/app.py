"""
WhatsApp Voice Calling with Gemini Live - Main Server

Python-based implementation using aiortc for full audio access
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional
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
    Path(__file__).parent.parent / "logs" / "whatsapp_voice.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("=" * 60)
    logger.info("WhatsApp Voice Calling with Gemini Live")
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

    yield

    # Shutdown
    logger.info("Server shutting down...")


# Create FastAPI app
app = FastAPI(
    title="WhatsApp Voice Calling with Gemini Live",
    description="AI Voice Agent for WhatsApp Business Voice Calls",
    version="1.0.0",
    lifespan=lifespan
)

# Mount audio directory for serving generated audio files
AUDIO_DIR.mkdir(exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")


# Request models
class MakeCallRequest(BaseModel):
    phoneNumber: str
    contactName: Optional[str] = "Customer"


class WebhookVerification(BaseModel):
    hub_mode: str
    hub_verify_token: str
    hub_challenge: str


# ============================================================================
# Health Check
# ============================================================================

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "WhatsApp Voice Calling with Gemini Live",
        "version": "1.0.0"
    }


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
    1. Plivo API initiates call to the phone number
    2. When user answers, Plivo hits /plivo/answer
    3. /plivo/answer returns <Stream> XML
    4. Plivo connects WebSocket to /plivo/stream/{call_uuid}
    5. Gemini Live session starts, AI greets the user
    6. Bidirectional audio conversation begins
    """
    logger.info(f"Plivo make call request: {request.phoneNumber}")

    try:
        result = await plivo_adapter.make_call(
            phone_number=request.phoneNumber,
            caller_name=request.contactName
        )

        if result.get("success"):
            call_uuid = result.get("call_uuid")

            # Add customer_name to context if not present
            context = request.context or {}
            context.setdefault("customer_name", request.contactName)

            # Store all call data
            _pending_call_data[call_uuid] = {
                "phone": request.phoneNumber,
                "prompt": request.prompt,
                "context": context,
                "webhookUrl": request.webhookUrl
            }
            logger.info(f"Plivo call initiated: {call_uuid}, stored phone: {request.phoneNumber}")

            # PRELOAD Gemini session immediately while phone is ringing
            from src.services.plivo_gemini_stream import preload_session
            asyncio.create_task(preload_session(
                call_uuid,
                request.phoneNumber,
                prompt=request.prompt,
                context=context,
                webhook_url=request.webhookUrl
            ))
            logger.info(f"Started preloading Gemini session for {call_uuid}")

            return JSONResponse(content={
                "success": True,
                "call_uuid": call_uuid,
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
    body = await request.form()
    call_uuid = body.get("CallUUID", "")
    from_phone = body.get("From", "")
    to_phone = body.get("To", "")

    # For outbound calls, customer is "To"; for inbound, customer is "From"
    # Store the customer phone (not our Plivo number)
    customer_phone = to_phone if to_phone and not to_phone.startswith("91226") else from_phone
    if customer_phone and call_uuid and call_uuid not in _pending_call_data:
        _pending_call_data[call_uuid] = {"phone": customer_phone, "prompt": None, "context": {}}

    logger.info(f"Plivo call answered: {call_uuid} from {from_phone} to {to_phone}, customer: {customer_phone}")

    # WebSocket URL for bidirectional audio stream
    ws_base = config.plivo_callback_url.replace("https://", "wss://").replace("http://", "ws://")
    stream_url = f"{ws_base}/plivo/stream/{call_uuid}"

    logger.info(f"Stream URL: {stream_url}")

    # Use Speak first, then Stream for bidirectional audio
    status_url = f"{config.plivo_callback_url}/plivo/stream-status"
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    
    <Stream streamTimeout="86400" keepCallAlive="true" bidirectional="true" contentType="audio/x-l16;rate=16000" statusCallbackUrl="{status_url}">{stream_url}</Stream>
</Response>"""

    return Response(content=xml_response, media_type="application/xml")


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
                call_uuid = start_data.get("callId", call_uuid)
                # Get phone from customParameters or fallback to stored value
                custom_params = start_data.get("customParameters", {})
                caller_phone = custom_params.get("callerPhone", "") or start_data.get("to", "") or start_data.get("from", "")
                # If still empty, use the call_uuid to look up from pending calls
                if not caller_phone:
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
    logger.info("=" * 60)
    logger.info("PLIVO STREAM STATUS CALLBACK")
    for key in body.keys():
        logger.info(f"  {key}: {body.get(key)}")
    logger.info("=" * 60)
    return JSONResponse(content={"status": "ok"})


@app.post("/plivo/hangup")
async def plivo_hangup(request: Request):
    """Handle Plivo call hangup"""
    body = await request.form()
    call_uuid = body.get("CallUUID", "")
    duration = body.get("Duration", "0")

    logger.info(f"Plivo call ended: {call_uuid}, duration: {duration}s")
    clear_conversation(call_uuid)

    return JSONResponse(content={"status": "ok"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.app:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="debug" if config.debug else "info"
    )