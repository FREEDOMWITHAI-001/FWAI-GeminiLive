"""
Send SMS Tool
Uses Plivo SMS API
"""

import httpx
import base64
from typing import Dict, Any
from loguru import logger

from .base import BaseTool, ToolResult
from .tool_registry import ToolRegistry
from src.core.config import config


@ToolRegistry.register
class SendSMSTool(BaseTool):
    name = "send_sms"
    description = "Send an SMS text message to the caller with information or follow-up details"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The SMS message content to send (max 160 chars recommended)"
                }
            },
            "required": ["message"]
        }
    
    async def execute(self, caller_phone: str, message: str, **kwargs) -> ToolResult:
        """Send SMS using Plivo API"""
        try:
            # Format phone number
            phone = caller_phone.replace("+", "").replace(" ", "")
            if not phone.startswith("91"):
                phone = "91" + phone
            
            # Plivo SMS API
            url = f"https://api.plivo.com/v1/Account/{config.plivo_auth_id}/Message/"
            
            # Basic auth
            auth = base64.b64encode(
                f"{config.plivo_auth_id}:{config.plivo_auth_token}".encode()
            ).decode()
            
            headers = {
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "src": config.plivo_phone_number,
                "dst": phone,
                "text": message
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers)
                
                if response.status_code in [200, 202]:
                    logger.info(f"SMS sent to {phone}: {message[:50]}...")
                    return ToolResult(
                        success=True,
                        message=f"SMS sent successfully to {phone}",
                        data={"phone": phone, "message_uuid": response.json().get("message_uuid", [])}
                    )
                else:
                    logger.error(f"Plivo SMS error: {response.text}")
                    return ToolResult(
                        success=False,
                        message=f"Failed to send SMS: {response.text}"
                    )
                    
        except Exception as e:
            logger.error(f"SMS send error: {e}")
            return ToolResult(
                success=False,
                message=f"Error sending SMS: {str(e)}"
            )


# Convenience function
async def send_sms(caller_phone: str, message: str) -> ToolResult:
    tool = SendSMSTool()
    return await tool.execute(caller_phone=caller_phone, message=message)
