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

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from src.core.config import config
from src.prompt_loader import FWAI_PROMPT
from src.conversation_memory import add_message, get_history, clear_conversation
from src.services.gemini_tools import generate_response_with_tools
from datetime import datetime
from src.handlers.webrtc_handler import (
    make_outbound_call,
    handle_incoming_call,
    handle_ice_candidate,
    terminate_call,
    get_active_calls
)

def save_transcript(call_uuid: str, role: str, message: str):
    """Save transcript to a file for each call"""
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

@app.post("/plivo/answer")
async def plivo_answer(request: Request):
    """Handle Plivo call answer"""
    from fastapi.responses import Response
    
    body = await request.form()
    call_uuid = body.get("CallUUID", "")
    
    logger.info(f"Plivo call answered: {call_uuid}")
    greeting = "Hi, this is Vishnu from Freedom with AI. I noticed you attended our AI Masterclass last week. How did you find it?"
    add_message(call_uuid, "assistant", greeting)
    save_transcript(call_uuid, "VISHNU", greeting)
    
    callback_url = f"{config.plivo_callback_url}/plivo/speech"
    
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Speak voice="Polly.Matthew">Hi, this is Vishnu from Freedom with AI. I noticed you attended our AI Masterclass last week. How did you find it?</Speak>
    <GetInput action="{callback_url}" method="POST" inputType="speech" executionTimeout="60" speechEndTimeout="3">
        
    </GetInput>
    <Speak voice="Polly.Matthew">I did not hear anything. Goodbye!</Speak>
</Response>"""
    
    return Response(content=xml_response, media_type="application/xml")


@app.post("/plivo/speech")
async def plivo_speech(request: Request):
    """Handle speech input from Plivo with tool calling support"""
    from fastapi.responses import Response
    
    body = await request.form()
    call_uuid = body.get("CallUUID", "")
    speech = body.get("Speech", "")
    caller_phone = body.get("From", "")
    
    logger.info("=" * 50)
    logger.info(f"TRANSCRIPT - USER: {speech}")
    logger.info("=" * 50)
    save_transcript(call_uuid, "USER", speech)
    
    # Track conversation
    add_message(call_uuid, "user", speech)
    history = get_history(call_uuid)
    
    # Call Gemini with tool support
    try:
        reply, tool_result = await generate_response_with_tools(
            history=history,
            user_message=speech,
            caller_phone=caller_phone
        )
        
        if tool_result:
            logger.info(f"TOOL EXECUTED: {tool_result}")
            save_transcript(call_uuid, "TOOL", str(tool_result))
        
        add_message(call_uuid, "assistant", reply)
        logger.info("=" * 50)
        logger.info(f"TRANSCRIPT - VISHNU: {reply}")
        logger.info("=" * 50)
        save_transcript(call_uuid, "VISHNU", reply)
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        reply = "Sorry, I could not process your request."
    
    callback_url = f"{config.plivo_callback_url}/plivo/speech"
    
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Speak voice="Polly.Matthew">{reply}</Speak>
    <GetInput action="{callback_url}" method="POST" inputType="speech" executionTimeout="60" speechEndTimeout="3">
        
    </GetInput>
    <Speak voice="Polly.Matthew">Thank you for calling. Goodbye!</Speak>
</Response>"""
    
    return Response(content=xml_response, media_type="application/xml")


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