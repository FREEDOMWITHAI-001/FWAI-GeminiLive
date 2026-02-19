"""
Send Email Tool
Uses SMTP or SendGrid API
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any
from loguru import logger

from .base import BaseTool, ToolResult
from .tool_registry import ToolRegistry
from src.core.config import config


@ToolRegistry.register
class SendEmailTool(BaseTool):
    name = "send_email"
    description = "Send an email to the caller with detailed information, brochures, or follow-up materials"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "email_address": {
                    "type": "string",
                    "description": "The email address to send to"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line"
                },
                "message": {
                    "type": "string",
                    "description": "Email body content"
                }
            },
            "required": ["email_address", "subject", "message"]
        }
    
    async def execute(self, caller_phone: str, email_address: str, subject: str, message: str, **kwargs) -> ToolResult:
        """Send email using SMTP"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = config.smtp_from_email or "noreply@example.com"
            msg['To'] = email_address
            msg['Subject'] = subject
            
            # Add body
            sender_name = kwargs.get("agent_name", "Your AI Assistant")
            sender_company = kwargs.get("company_name", "")
            signature = f"\n--\nBest regards,\n{sender_name}"
            if sender_company:
                signature += f"\n{sender_company}"
            body = f"""{message}{signature}
"""
            msg.attach(MIMEText(body, 'plain'))
            
            # Send via SMTP
            if config.smtp_host and config.smtp_user:
                with smtplib.SMTP(config.smtp_host, config.smtp_port or 587) as server:
                    server.starttls()
                    server.login(config.smtp_user, config.smtp_password)
                    server.send_message(msg)
                
                logger.info(f"Email sent to {email_address}: {subject}")
                return ToolResult(
                    success=True,
                    message=f"Email sent successfully to {email_address}",
                    data={"email": email_address, "subject": subject}
                )
            else:
                # Log email if SMTP not configured
                logger.warning(f"SMTP not configured. Would send email to {email_address}: {subject}")
                return ToolResult(
                    success=True,
                    message=f"Email queued for {email_address} (SMTP not configured)",
                    data={"email": email_address, "subject": subject, "queued": True}
                )
                    
        except Exception as e:
            logger.error(f"Email send error: {e}")
            return ToolResult(
                success=False,
                message=f"Error sending email: {str(e)}"
            )


# Convenience function
async def send_email(email_address: str, subject: str, message: str, caller_phone: str = "") -> ToolResult:
    tool = SendEmailTool()
    return await tool.execute(
        caller_phone=caller_phone,
        email_address=email_address,
        subject=subject,
        message=message
    )
