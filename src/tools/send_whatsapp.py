"""
Send WhatsApp Message Tool
Uses Meta WhatsApp Business API with beautiful templates
"""

import httpx
from typing import Dict, Any
from loguru import logger

from .base import BaseTool, ToolResult
from .tool_registry import ToolRegistry
from src.core.config import config
from src.templates import format_template, WHATSAPP_TEMPLATES
from src.services.meta_token_manager import get_access_token


@ToolRegistry.register
class SendWhatsAppTool(BaseTool):
    name = "send_whatsapp"
    description = """Send a WhatsApp message to the caller. Use template_id to send beautiful formatted messages:
- course_details: Send when user wants information about courses or products
- payment_link: Send when user is ready to purchase or asks for payment options
- support_contact: Send when user has issues, complaints, or needs help
If no template matches, use custom_message for a plain text message."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "template_id": {
                    "type": "string",
                    "description": "Template to use: 'course_details', 'payment_link', or 'support_contact'",
                    "enum": ["course_details", "payment_link", "support_contact"]
                },
                "custom_message": {
                    "type": "string",
                    "description": "Custom message if no template fits (optional, use only if templates don't match)"
                }
            },
            "required": []
        }

    async def execute(self, caller_phone: str, template_id: str = None, custom_message: str = None, **kwargs) -> ToolResult:
        """Send WhatsApp message using Meta API"""
        try:
            # Get context from session (passed via kwargs)
            context = kwargs.get("context", {})

            # Format phone number (remove + if present)
            phone = caller_phone.replace("+", "").replace(" ", "")

            # Add phone and customer info to context for templates
            context.setdefault("customer_name", "there")

            # Determine message content
            if template_id and template_id in WHATSAPP_TEMPLATES:
                message = format_template(template_id, context)
                logger.info(f"Using template: {template_id}")
            elif custom_message:
                message = custom_message
                logger.info("Using custom message")
            else:
                return ToolResult(
                    success=False,
                    message="Please specify either template_id or custom_message"
                )

            # Meta WhatsApp Business API
            url = f"https://graph.facebook.com/v22.0/{config.whatsapp_phone_id}/messages"

            # Get token from token manager (auto-refreshes if needed)
            access_token = await get_access_token()

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }

            payload = {
                "messaging_product": "whatsapp",
                "to": phone,
                "type": "text",
                "text": {
                    "body": message
                }
            }

            logger.info(f"Sending WhatsApp to {phone}, template: {template_id or 'custom'}")
            logger.debug(f"WhatsApp API URL: {url}")
            logger.debug(f"WhatsApp payload: {payload}")

            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers)

                logger.info(f"WhatsApp API response status: {response.status_code}")
                logger.info(f"WhatsApp API response body: {response.text[:500]}")

                if response.status_code == 200:
                    logger.info(f"WhatsApp sent successfully to {phone}")
                    return ToolResult(
                        success=True,
                        message=f"WhatsApp message sent successfully",
                        data={"phone": phone, "template": template_id, "message_id": response.json().get("messages", [{}])[0].get("id")}
                    )
                else:
                    error_msg = response.text
                    logger.error(f"WhatsApp API error ({response.status_code}): {error_msg}")
                    return ToolResult(
                        success=False,
                        message=f"Failed to send WhatsApp: {error_msg}"
                    )

        except Exception as e:
            logger.error(f"WhatsApp send error: {e}")
            return ToolResult(
                success=False,
                message=f"Error sending WhatsApp: {str(e)}"
            )


# Convenience function
async def send_whatsapp(caller_phone: str, template_id: str = None, custom_message: str = None, context: dict = None) -> ToolResult:
    tool = SendWhatsAppTool()
    return await tool.execute(caller_phone=caller_phone, template_id=template_id, custom_message=custom_message, context=context or {})
