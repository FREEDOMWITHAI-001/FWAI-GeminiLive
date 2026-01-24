"""
Send WhatsApp Message Tool
Uses Meta WhatsApp Business API
"""

import httpx
from typing import Dict, Any
from loguru import logger

from .base import BaseTool, ToolResult
from .tool_registry import ToolRegistry
from src.core.config import config


@ToolRegistry.register
class SendWhatsAppTool(BaseTool):
    name = "send_whatsapp"
    description = "Send a WhatsApp message to the caller with information, links, or follow-up details"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message content to send via WhatsApp"
                }
            },
            "required": ["message"]
        }
    
    async def execute(self, caller_phone: str, message: str, **kwargs) -> ToolResult:
        """Send WhatsApp message using Meta API"""
        try:
            # Format phone number (remove + if present)
            phone = caller_phone.replace("+", "").replace(" ", "")
            
            # Meta WhatsApp Business API
            url = f"https://graph.facebook.com/v17.0/{config.whatsapp_phone_id}/messages"
            
            headers = {
                "Authorization": f"Bearer {config.meta_access_token}",
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
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    logger.info(f"WhatsApp sent to {phone}: {message[:50]}...")
                    return ToolResult(
                        success=True,
                        message=f"WhatsApp message sent successfully to {phone}",
                        data={"phone": phone, "message_id": response.json().get("messages", [{}])[0].get("id")}
                    )
                else:
                    logger.error(f"WhatsApp API error: {response.text}")
                    return ToolResult(
                        success=False,
                        message=f"Failed to send WhatsApp: {response.text}"
                    )
                    
        except Exception as e:
            logger.error(f"WhatsApp send error: {e}")
            return ToolResult(
                success=False,
                message=f"Error sending WhatsApp: {str(e)}"
            )


# Convenience function
async def send_whatsapp(caller_phone: str, message: str) -> ToolResult:
    tool = SendWhatsAppTool()
    return await tool.execute(caller_phone=caller_phone, message=message)
