"""
GoHighLevel (GHL) WhatsApp Integration
Triggers GHL workflow via webhook when calls start.
The GHL workflow handles the actual WhatsApp message sending.
"""

import httpx
from loguru import logger
from typing import Dict, Any


async def trigger_ghl_workflow(phone: str, contact_name: str = "Customer", webhook_url: str = "", email: str = "") -> Dict[str, Any]:
    """
    Trigger a GoHighLevel workflow via inbound webhook.
    Passes phone, email and contact name so GHL can find contact and send WhatsApp.

    Args:
        phone: Recipient phone number
        contact_name: Recipient name
        webhook_url: GHL inbound webhook URL (passed per-call from frontend)
        email: Recipient email (for GHL contact lookup fallback)
    """
    if not webhook_url:
        logger.warning("No GHL webhook URL provided - skipping workflow trigger")
        return {"success": False, "error": "No GHL webhook URL"}

    # Normalize phone
    clean_phone = phone.replace(" ", "")
    if not clean_phone.startswith("+"):
        clean_phone = "+" + clean_phone

    payload = {
        "phone": clean_phone,
        "contact_name": contact_name,
        "email": email,
        "source": "ai_voice_call",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(webhook_url, json=payload)
            if resp.status_code in (200, 201):
                logger.info(f"GHL workflow triggered for {clean_phone} ({contact_name})")
                return {"success": True, "phone": clean_phone}
            else:
                logger.error(f"GHL webhook failed ({resp.status_code}): {resp.text[:300]}")
                return {"success": False, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        logger.error(f"GHL webhook error: {e}")
        return {"success": False, "error": str(e)}
