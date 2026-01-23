"""
Voice Call Adapters

Unified interface for different telephony providers:
- WhatsApp Business API
- Plivo
- Exotel
"""

from .base import BaseCallAdapter
from .whatsapp_adapter import WhatsAppAdapter
from .plivo_adapter import PlivoAdapter
from .exotel_adapter import ExotelAdapter

__all__ = [
    "BaseCallAdapter",
    "WhatsAppAdapter",
    "PlivoAdapter",
    "ExotelAdapter",
]
