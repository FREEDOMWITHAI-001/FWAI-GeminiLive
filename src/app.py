"""
AI Voice Call with Plivo - Main Server

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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

import hashlib
from src.core.config import config
from src.conversation_memory import add_message, get_history, clear_conversation
from src.db.session_db import session_db
from fastapi.staticfiles import StaticFiles

# Audio directory for any generated audio files
AUDIO_DIR = Path(__file__).parent.parent / "audio"
from datetime import datetime

# Prompt caching: deduplicate identical prompts across calls
_prompt_cache: dict[str, str] = {}

def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()

def get_or_cache_prompt(prompt: str) -> str:
    if not prompt:
        return prompt
    h = _hash_prompt(prompt)
    if h in _prompt_cache:
        logger.debug(f"Prompt cache HIT ({h[:12]})")
        return _prompt_cache[h]
    _prompt_cache[h] = prompt
    logger.debug(f"Prompt cache MISS — stored ({h[:12]}, {len(prompt)} chars)")
    return prompt
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


# Configure logging — clean format for structured call logs
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <5}</level> | <level>{message}</level>",
    level="DEBUG" if config.debug else "INFO"
)
logger.add(
    Path(__file__).parent.parent / "logs" / "fwai_voice.log",
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <5} | {message}",
    level=config.log_level
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
    logger.info(f"Session DB: PostgreSQL ({session_db._dsn.split('@')[-1] if session_db._dsn else 'unknown'})")

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

# CORS - allow Wavelength and other frontends to call API directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount audio directory for serving generated audio files
AUDIO_DIR.mkdir(exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")

# Mount static directory for test dashboard UI
STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# Request models
class MakeCallRequest(BaseModel):
    phoneNumber: str
    contactName: Optional[str] = "Customer"


class WebhookVerification(BaseModel):
    hub_mode: str
    hub_verify_token: str
    hub_challenge: str


# ============================================================================
# Prompt Management (for dashboard UI)
# ============================================================================

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
PROMPTS_DIR.mkdir(exist_ok=True)

@app.get("/prompts/{client_name}")
async def get_prompt(client_name: str):
    """Load prompt file for a client"""
    prompt_file = PROMPTS_DIR / f"{client_name}_prompt.txt"
    if prompt_file.exists():
        return {"success": True, "prompt": prompt_file.read_text(encoding="utf-8")}
    return {"success": False, "error": f"No prompt file found for '{client_name}'"}

@app.post("/prompts/update")
async def update_prompt(request: Request):
    """Save prompt file for a client"""
    body = await request.json()
    client_name = body.get("client_name", "fwai")
    prompt = body.get("prompt", "")
    if not prompt.strip():
        return {"success": False, "error": "Prompt cannot be empty"}
    prompt_file = PROMPTS_DIR / f"{client_name}_prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    return {"success": True, "message": f"Prompt saved for {client_name}"}


# ============================================================================
# Dynamic Persona Engine API
# ============================================================================

PERSONAS_DIR = PROMPTS_DIR / "personas"
SITUATIONS_DIR = PROMPTS_DIR / "situations"
PERSONAS_DIR.mkdir(exist_ok=True)
SITUATIONS_DIR.mkdir(exist_ok=True)


@app.get("/personas")
async def list_personas():
    """List all available persona modules"""
    personas = [f.stem for f in PERSONAS_DIR.glob("*.txt")]
    return {"personas": sorted(personas)}


@app.get("/personas/{name}")
async def get_persona(name: str):
    """Get persona module content"""
    path = PERSONAS_DIR / f"{name}.txt"
    if path.exists():
        return {"name": name, "content": path.read_text(encoding="utf-8")}
    return {"error": f"Persona '{name}' not found"}


@app.post("/personas/{name}")
async def save_persona(name: str, request: Request):
    """Create or update persona module"""
    body = await request.json()
    content = body.get("content", "")
    if not content.strip():
        return {"error": "Content cannot be empty"}
    path = PERSONAS_DIR / f"{name}.txt"
    path.write_text(content, encoding="utf-8")
    return {"success": True, "message": f"Persona '{name}' saved"}


@app.delete("/personas/{name}")
async def delete_persona(name: str):
    """Delete a persona module"""
    path = PERSONAS_DIR / f"{name}.txt"
    if path.exists():
        path.unlink()
        return {"success": True, "message": f"Persona '{name}' deleted"}
    return {"error": f"Persona '{name}' not found"}


@app.get("/situations")
async def list_situations():
    """List all available situation modules"""
    situations = [f.stem for f in SITUATIONS_DIR.glob("*.txt")]
    return {"situations": sorted(situations)}


@app.get("/situations/{name}")
async def get_situation(name: str):
    """Get situation module content"""
    path = SITUATIONS_DIR / f"{name}.txt"
    if path.exists():
        return {"name": name, "content": path.read_text(encoding="utf-8")}
    return {"error": f"Situation '{name}' not found"}


@app.post("/situations/{name}")
async def save_situation(name: str, request: Request):
    """Create or update situation module"""
    body = await request.json()
    content = body.get("content", "")
    if not content.strip():
        return {"error": "Content cannot be empty"}
    path = SITUATIONS_DIR / f"{name}.txt"
    path.write_text(content, encoding="utf-8")
    return {"success": True, "message": f"Situation '{name}' saved"}


@app.get("/persona-config")
async def get_persona_config():
    """Get persona and situation detection keyword configs"""
    persona_kw = {}
    situation_kw = {}
    pk_path = PROMPTS_DIR / "persona_keywords.json"
    sk_path = PROMPTS_DIR / "situation_keywords.json"
    if pk_path.exists():
        persona_kw = json.loads(pk_path.read_text(encoding="utf-8"))
    if sk_path.exists():
        situation_kw = json.loads(sk_path.read_text(encoding="utf-8"))
    return {"persona_keywords": persona_kw, "situation_keywords": situation_kw}


@app.post("/persona-config")
async def update_persona_config(request: Request):
    """Update persona and/or situation detection keyword configs"""
    body = await request.json()
    if "persona_keywords" in body:
        pk_path = PROMPTS_DIR / "persona_keywords.json"
        pk_path.write_text(json.dumps(body["persona_keywords"], indent=2), encoding="utf-8")
    if "situation_keywords" in body:
        sk_path = PROMPTS_DIR / "situation_keywords.json"
        sk_path.write_text(json.dumps(body["situation_keywords"], indent=2), encoding="utf-8")
    return {"success": True, "message": "Config updated"}


# ============================================================================
# Cross-Call Memory API
# ============================================================================

@app.get("/memory")
async def list_memories():
    """List all contact memories"""
    memories = session_db.get_all_contact_memories(limit=200)
    return {"memories": memories, "total": len(memories)}


@app.get("/memory/{phone}")
async def get_memory(phone: str):
    """Get contact memory for a phone number"""
    # URL-decode phone (+ becomes space in URL params)
    phone = phone.replace(" ", "+")
    memory = session_db.get_contact_memory(phone)
    if memory:
        return {"success": True, "memory": memory}
    return {"success": False, "error": f"No memory found for {phone}"}


@app.post("/memory/{phone}")
async def update_memory(phone: str, request: Request):
    """Update contact memory for a phone number"""
    phone = phone.replace(" ", "+")
    body = await request.json()
    # Filter only valid fields
    valid_fields = {
        "name", "persona", "company", "role", "objections",
        "interest_areas", "key_facts", "last_call_summary",
        "last_call_outcome",
    }
    updates = {k: v for k, v in body.items() if k in valid_fields}
    if not updates:
        return {"error": "No valid fields to update"}
    session_db.save_contact_memory(phone, **updates)
    return {"success": True, "message": f"Memory updated for {phone}"}


@app.delete("/memory/{phone}")
async def delete_memory(phone: str):
    """Delete contact memory for a phone number"""
    phone = phone.replace(" ", "+")
    session_db.delete_contact_memory(phone)
    return {"success": True, "message": f"Memory deleted for {phone}"}


# ============================================================================
# Social Proof Engine API
# ============================================================================

@app.get("/social-proof/stats")
async def list_social_proof():
    """List all social proof stats (companies, cities, roles)"""
    return session_db.get_all_social_proof()

@app.get("/social-proof/summary")
async def social_proof_summary():
    """Get aggregate summary (same data injected into pre-call prompt)"""
    from src.social_proof import load_social_proof_summary
    summary = load_social_proof_summary()
    return {"summary": summary}

@app.get("/social-proof/stats/company/{name}")
async def get_company_stats(name: str):
    """Get enrollment stats for a specific company"""
    stats = session_db.get_social_proof_by_company(name)
    if stats:
        return {"success": True, "stats": stats}
    return {"success": False, "error": f"No stats for company '{name}'"}

@app.post("/social-proof/stats/company")
async def upsert_company_stats(request: Request):
    """Create or update company enrollment stats"""
    body = await request.json()
    company_name = body.pop("company_name", None)
    if not company_name:
        return JSONResponse(status_code=400, content={"error": "company_name is required"})
    session_db.upsert_social_proof_company(company_name, **body)
    return {"success": True, "message": f"Stats updated for {company_name}"}

@app.delete("/social-proof/stats/company/{name}")
async def delete_company_stats(name: str):
    """Delete company enrollment stats"""
    session_db.delete_social_proof_company(name)
    return {"success": True, "message": f"Company stats deleted for {name}"}

@app.get("/social-proof/stats/city/{name}")
async def get_city_stats(name: str):
    """Get enrollment stats for a specific city"""
    stats = session_db.get_social_proof_by_city(name)
    if stats:
        return {"success": True, "stats": stats}
    return {"success": False, "error": f"No stats for city '{name}'"}

@app.post("/social-proof/stats/city")
async def upsert_city_stats(request: Request):
    """Create or update city enrollment stats"""
    body = await request.json()
    city_name = body.pop("city_name", None)
    if not city_name:
        return JSONResponse(status_code=400, content={"error": "city_name is required"})
    session_db.upsert_social_proof_city(city_name, **body)
    return {"success": True, "message": f"Stats updated for {city_name}"}

@app.delete("/social-proof/stats/city/{name}")
async def delete_city_stats(name: str):
    """Delete city enrollment stats"""
    session_db.delete_social_proof_city(name)
    return {"success": True, "message": f"City stats deleted for {name}"}

@app.get("/social-proof/stats/role/{name}")
async def get_role_stats(name: str):
    """Get enrollment stats for a specific role"""
    stats = session_db.get_social_proof_by_role(name)
    if stats:
        return {"success": True, "stats": stats}
    return {"success": False, "error": f"No stats for role '{name}'"}

@app.post("/social-proof/stats/role")
async def upsert_role_stats(request: Request):
    """Create or update role enrollment stats"""
    body = await request.json()
    role_name = body.pop("role_name", None)
    if not role_name:
        return JSONResponse(status_code=400, content={"error": "role_name is required"})
    session_db.upsert_social_proof_role(role_name, **body)
    return {"success": True, "message": f"Stats updated for {role_name}"}

@app.delete("/social-proof/stats/role/{name}")
async def delete_role_stats(name: str):
    """Delete role enrollment stats"""
    session_db.delete_social_proof_role(name)
    return {"success": True, "message": f"Role stats deleted for {name}"}

@app.post("/social-proof/bulk")
async def bulk_update_social_proof(request: Request):
    """CRM webhook endpoint for bulk stats updates.
    Accepts: {"companies": [...], "cities": [...], "roles": [...]}"""
    body = await request.json()
    counts = {"companies": 0, "cities": 0, "roles": 0}

    for item in body.get("companies", []):
        name = item.pop("company_name", None)
        if name:
            session_db.upsert_social_proof_company(name, **item)
            counts["companies"] += 1

    for item in body.get("cities", []):
        name = item.pop("city_name", None)
        if name:
            session_db.upsert_social_proof_city(name, **item)
            counts["cities"] += 1

    for item in body.get("roles", []):
        name = item.pop("role_name", None)
        if name:
            session_db.upsert_social_proof_role(name, **item)
            counts["roles"] += 1

    return {
        "success": True,
        "message": f"Bulk update: {counts['companies']} companies, {counts['cities']} cities, {counts['roles']} roles"
    }


# ============================================================================
# Product Intelligence API
# ============================================================================

PRODUCTS_DIR = PROMPTS_DIR / "products"
PRODUCTS_DIR.mkdir(exist_ok=True)


@app.get("/products")
async def list_products():
    """List all product knowledge sections"""
    sections = [f.stem for f in PRODUCTS_DIR.glob("*.txt")]
    return {"sections": sorted(sections)}


@app.post("/products/upload")
async def upload_product_document(request: Request):
    """Upload raw product document for AI processing into structured sections."""
    body = await request.json()
    raw_content = body.get("content", "")
    source_type = body.get("source_type", "text")

    if not raw_content or not raw_content.strip():
        return {"error": "Content cannot be empty"}

    from src.product_intelligence import process_document, save_product_sections

    result = await process_document(raw_content, source_type=source_type)

    if result.get("error"):
        return {"success": False, "error": result["error"]}

    sections = result.get("sections", {})
    keywords = result.get("keywords", {})

    if sections:
        save_product_sections(sections, keywords)
        return {
            "success": True,
            "sections_created": list(sections.keys()),
            "message": f"Processed into {len(sections)} sections",
        }
    return {"success": False, "error": "No sections extracted from document"}


@app.get("/products/{name}")
async def get_product(name: str):
    """Get product section content"""
    path = PRODUCTS_DIR / f"{name}.txt"
    if path.exists():
        return {"name": name, "content": path.read_text(encoding="utf-8")}
    return {"error": f"Product section '{name}' not found"}


@app.post("/products/{name}")
async def save_product(name: str, request: Request):
    """Create or update product section"""
    body = await request.json()
    content = body.get("content", "")
    if not content.strip():
        return {"error": "Content cannot be empty"}
    path = PRODUCTS_DIR / f"{name}.txt"
    path.write_text(content, encoding="utf-8")
    return {"success": True, "message": f"Product section '{name}' saved"}


@app.delete("/products/{name}")
async def delete_product(name: str):
    """Delete a product section"""
    path = PRODUCTS_DIR / f"{name}.txt"
    if path.exists():
        path.unlink()
        return {"success": True, "message": f"Product section '{name}' deleted"}
    return {"error": f"Product section '{name}' not found"}


@app.get("/product-config")
async def get_product_config():
    """Get product section detection keyword config"""
    pk_path = PROMPTS_DIR / "product_keywords.json"
    if pk_path.exists():
        return json.loads(pk_path.read_text(encoding="utf-8"))
    return {}


@app.post("/product-config")
async def update_product_config(request: Request):
    """Update product section detection keyword config"""
    body = await request.json()
    pk_path = PROMPTS_DIR / "product_keywords.json"
    pk_path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"success": True, "message": "Product keywords config updated"}


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
# Make Outbound Call
# ============================================================================

@app.post("/make-call")
async def api_make_call(request: MakeCallRequest):
    """
    Make an outbound call to a WhatsApp user

    The AI agent will start speaking when the user answers
    """
    # Reject new calls during maintenance/restart window
    if Path(".maintenance").exists():
        logger.warning(f"Call rejected during maintenance: {request.phoneNumber}")
        raise HTTPException(status_code=503, detail="Service restarting, try again shortly")

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


@app.get("/calls/{call_id}/status")
async def get_call_status(call_id: str):
    """Get call status and data by call_uuid — used by Wavelength polling"""
    from src.db.session_db import session_db
    call = session_db.get_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")
    return call


@app.get("/calls/{call_id}/recording")
async def get_call_recording(call_id: str):
    """Download the recording for a call by call_uuid"""
    from pathlib import Path
    from fastapi.responses import FileResponse

    recordings_dir = Path(__file__).parent.parent / "recordings"

    # Try MP3 first, then WAV
    mp3_file = recordings_dir / f"{call_id}_mixed.mp3"
    wav_file = recordings_dir / f"{call_id}_mixed.wav"

    if mp3_file.exists():
        return FileResponse(str(mp3_file), media_type="audio/mpeg", filename=f"{call_id}.mp3")
    elif wav_file.exists():
        return FileResponse(str(wav_file), media_type="audio/wav", filename=f"{call_id}.wav")
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Recording not found for call {call_id}. Ensure ENABLE_TRANSCRIPTS=true."
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
    voice: Optional[str] = None  # Voice name from UI (e.g. "Puck", "Kore") — overrides auto-detection
    ghlWhatsappWebhookUrl: Optional[str] = None  # GHL workflow webhook URL to trigger WhatsApp on call start
    ghlApiKey: Optional[str] = None  # GHL API key for contact lookup and tagging
    ghlLocationId: Optional[str] = None  # GHL location/sub-account ID
    plivoAuthId: Optional[str] = None  # Per-org Plivo Auth ID (overrides env default)
    plivoAuthToken: Optional[str] = None  # Per-org Plivo Auth Token (overrides env default)
    plivoPhoneNumber: Optional[str] = None  # Per-org Plivo caller ID (overrides env default)


@app.post("/plivo/make-call")
@app.post("/call/conversational")
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

        # Build context first
        context = request.context or {}
        context.setdefault("customer_name", request.contactName)

        # Load default prompt if none provided — enable persona engine for FWAI
        if not request.prompt:
            default_prompt_file = PROMPTS_DIR / "fwai_prompt.txt"
            if default_prompt_file.exists():
                request.prompt = default_prompt_file.read_text(encoding="utf-8")
                context["_persona_engine"] = True
                logger.info(f"Loaded default prompt from fwai_prompt.txt ({len(request.prompt)} chars) + persona engine ON")

        # Cache the incoming prompt (deduplicates identical prompts across calls)
        if request.prompt:
            request.prompt = get_or_cache_prompt(request.prompt)

        # Pass explicit voice selection from UI (overrides auto-detection from prompt)
        if request.voice:
            context["_voice"] = request.voice

        # Pass GHL webhook URL and API credentials in context so AI can trigger it mid-call
        if request.ghlWhatsappWebhookUrl:
            context["ghl_webhook_url"] = request.ghlWhatsappWebhookUrl
        if request.ghlApiKey:
            context["ghl_api_key"] = request.ghlApiKey
        if request.ghlLocationId:
            context["ghl_location_id"] = request.ghlLocationId

        # Pass per-org Plivo credentials in context for hangup
        if request.plivoAuthId:
            context["plivo_auth_id"] = request.plivoAuthId
        if request.plivoAuthToken:
            context["plivo_auth_token"] = request.plivoAuthToken

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

        # Gather intelligence + cross-call memory FIRST, then preload
        from src.services.plivo_gemini_stream import preload_session, get_preloading_session
        from src.services.intelligence import gather_intelligence
        from src.cross_call_memory import load_memory_context

        logger.info(f"Preloading Gemini session + intelligence for {call_uuid}...")

        # Step 1: Load cross-call memory (instant, local DB lookup)
        memory_data = load_memory_context(request.phoneNumber)
        if memory_data:
            context["_memory_context"] = memory_data.get("prompt", "")
            # If memory has a persona, pre-set it so AI skips discovery
            if memory_data.get("persona"):
                context["_memory_persona"] = memory_data["persona"]
            # Pre-load linguistic style for Linguistic Mirror
            if memory_data.get("linguistic_style"):
                context["_memory_linguistic_style"] = memory_data["linguistic_style"]
            logger.info(f"Cross-call memory loaded for {request.phoneNumber} ({len(context.get('_memory_context', ''))} chars, persona={memory_data.get('persona')})")

        # Step 2: Gather intelligence (runs before preload so it's in the system prompt)
        intelligence_brief = await gather_intelligence(request.contactName, context)
        if intelligence_brief:
            logger.info(f"Intelligence ready for {call_uuid} ({len(intelligence_brief)} chars)")

        # Step 2.5: Load social proof summary (instant, local DB lookup)
        from src.social_proof import load_social_proof_summary
        social_proof_summary = load_social_proof_summary()
        if social_proof_summary:
            logger.info(f"Social proof summary loaded ({len(social_proof_summary)} chars)")

        # Step 3: Preload Gemini session (intelligence + memory + social proof will be in system prompt)
        await preload_session(
            call_uuid,
            request.phoneNumber,
            prompt=request.prompt,
            context=context,
            webhook_url=request.webhookUrl,
            intelligence_brief=intelligence_brief,
            social_proof_summary=social_proof_summary,
        )

        logger.info(f"Gemini preload complete for {call_uuid} - now making call")

        # NOW make the Plivo call (AI is already ready)
        # Use per-org Plivo credentials if provided, otherwise adapter uses env defaults
        result = await plivo_adapter.make_call(
            phone_number=request.phoneNumber,
            caller_name=request.contactName,
            plivo_auth_id=request.plivoAuthId,
            plivo_auth_token=request.plivoAuthToken,
            plivo_phone_number=request.plivoPhoneNumber,
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
    body = await request.form()
    plivo_uuid = body.get("CallUUID", "")
    from_phone = body.get("From", "")
    to_phone = body.get("To", "")

    # Look up our internal UUID from Plivo's UUID (for preloaded sessions)
    async with _call_data_lock:
        internal_uuid = _plivo_to_internal_uuid.get(plivo_uuid, plivo_uuid)
        # For outbound calls, customer is "To"; for inbound, customer is "From"
        customer_phone = to_phone if to_phone and not to_phone.startswith("91226") else from_phone
        if customer_phone and internal_uuid and internal_uuid not in _pending_call_data:
            _pending_call_data[internal_uuid] = {"phone": customer_phone, "prompt": None, "context": {}}

    logger.info(f"Plivo call answered: plivo={plivo_uuid}, internal={internal_uuid}, from {from_phone} to {to_phone}")

    # WebSocket URL for bidirectional audio stream (use internal UUID for preloaded session)
    ws_base = config.plivo_callback_url.replace("https://", "wss://").replace("http://", "ws://")
    stream_url = f"{ws_base}/plivo/stream/{internal_uuid}"

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




@app.post("/calls/{call_uuid}/hangup")
async def hangup_call_by_uuid(call_uuid: str):
    """Force-hangup a call by internal UUID. Called by frontend when user manually marks complete."""
    from src.services.plivo_gemini_stream import remove_session, get_session

    session = await get_session(call_uuid)
    if session:
        # Hang up via Plivo API if we have the Plivo UUID
        plivo_uuid = session.plivo_call_uuid
        if plivo_uuid:
            try:
                await plivo_adapter.terminate_call(plivo_uuid)
            except Exception as e:
                logger.warning(f"Plivo terminate failed for {call_uuid}: {e}")

        await remove_session(call_uuid)
        logger.info(f"Force-hangup call {call_uuid}")
        return JSONResponse(content={"success": True, "message": f"Call {call_uuid} hung up"})

    # Clean up pending data even if session not found
    async with _call_data_lock:
        _pending_call_data.pop(call_uuid, None)
    clear_conversation(call_uuid)

    return JSONResponse(content={"success": True, "message": f"Call {call_uuid} not found (may have already ended)"})


@app.get("/call/history")
async def get_call_history(limit: int = Query(default=50, le=200)):
    """Get call history with statistics from session DB"""
    calls = session_db.get_recent_calls(limit=limit)
    return JSONResponse(content={"calls": calls, "total": len(calls)})


@app.get("/call/{call_id}/details")
async def get_call_details(call_id: str):
    """Get full call details including responses and statistics"""
    call = session_db.get_call(call_id)
    if call:
        return JSONResponse(content=call)
    raise HTTPException(status_code=404, detail=f"Call {call_id} not found")



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