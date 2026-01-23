"""
Base Call Adapter - Abstract interface for all telephony providers
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseCallAdapter(ABC):
    """
    Abstract base class for voice call adapters.

    All telephony provider adapters (WhatsApp, Plivo, Exotel) must implement
    this interface to ensure consistent behavior across providers.
    """

    @abstractmethod
    async def make_call(
        self,
        phone_number: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Initiate an outbound call

        Args:
            phone_number: The phone number to call
            **kwargs: Provider-specific parameters

        Returns:
            Dict with 'success' bool and call details or error
        """
        pass

    @abstractmethod
    async def terminate_call(self, call_id: str) -> Dict[str, Any]:
        """
        Terminate/hangup an active call

        Args:
            call_id: The call identifier

        Returns:
            Dict with 'success' bool
        """
        pass

    @abstractmethod
    async def get_call_details(self, call_id: str) -> Dict[str, Any]:
        """
        Get details of a specific call

        Args:
            call_id: The call identifier

        Returns:
            Dict with call details
        """
        pass

    async def answer_call(
        self,
        call_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Answer an incoming call (if supported by provider)

        Args:
            call_id: The call identifier
            **kwargs: Provider-specific parameters (e.g., SDP answer)

        Returns:
            Dict with 'success' bool
        """
        return {"success": False, "error": "Not implemented for this provider"}

    async def reject_call(self, call_id: str) -> Dict[str, Any]:
        """
        Reject an incoming call (if supported by provider)

        Args:
            call_id: The call identifier

        Returns:
            Dict with 'success' bool
        """
        return {"success": False, "error": "Not implemented for this provider"}

    async def transfer_call(
        self,
        call_id: str,
        transfer_to: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transfer an active call

        Args:
            call_id: The call identifier
            transfer_to: Destination (number or URL)
            **kwargs: Provider-specific parameters

        Returns:
            Dict with 'success' bool
        """
        return {"success": False, "error": "Not implemented for this provider"}

    async def play_audio(
        self,
        call_id: str,
        audio_url: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Play audio on an active call

        Args:
            call_id: The call identifier
            audio_url: URL of the audio file
            **kwargs: Provider-specific parameters

        Returns:
            Dict with 'success' bool
        """
        return {"success": False, "error": "Not implemented for this provider"}

    async def speak_text(
        self,
        call_id: str,
        text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Speak text on an active call using TTS

        Args:
            call_id: The call identifier
            text: Text to speak
            **kwargs: Provider-specific parameters (voice, language, etc.)

        Returns:
            Dict with 'success' bool
        """
        return {"success": False, "error": "Not implemented for this provider"}

    async def start_recording(
        self,
        call_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Start recording a call

        Args:
            call_id: The call identifier
            **kwargs: Provider-specific parameters

        Returns:
            Dict with 'success' bool
        """
        return {"success": False, "error": "Not implemented for this provider"}

    async def stop_recording(self, call_id: str) -> Dict[str, Any]:
        """
        Stop recording a call

        Args:
            call_id: The call identifier

        Returns:
            Dict with 'success' bool
        """
        return {"success": False, "error": "Not implemented for this provider"}

    async def send_dtmf(self, call_id: str, digits: str) -> Dict[str, Any]:
        """
        Send DTMF tones on a call

        Args:
            call_id: The call identifier
            digits: DTMF digits to send

        Returns:
            Dict with 'success' bool
        """
        return {"success": False, "error": "Not implemented for this provider"}

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'whatsapp', 'plivo', 'exotel')"""
        pass
