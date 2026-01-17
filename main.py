"""
WhatsApp Voice Calling with Gemini Live - Main Server

Python-based implementation using aiortc for full audio access
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from loguru import logger
import sys

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from config import config
from webrtc_handler import (
    make_outbound_call,
    handle_incoming_call,
    handle_ice_candidate,
    terminate_call,
    get_active_calls
)

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG" if config.debug else "INFO"
)
logger.add(
    "whatsapp_voice.log",
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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="debug" if config.debug else "info"
    )
