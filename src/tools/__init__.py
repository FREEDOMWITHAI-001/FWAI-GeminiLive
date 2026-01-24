"""
Tools Module for Vishnu AI Agent
Provides function calling capabilities during voice calls
"""

from .tool_registry import ToolRegistry, execute_tool, get_tool_definitions
from .send_whatsapp import send_whatsapp
from .send_sms import send_sms
from .send_email import send_email
from .schedule_callback import schedule_callback
from .book_demo import book_demo

__all__ = [
    'ToolRegistry',
    'execute_tool', 
    'get_tool_definitions',
    'send_whatsapp',
    'send_sms',
    'send_email',
    'schedule_callback',
    'book_demo'
]
