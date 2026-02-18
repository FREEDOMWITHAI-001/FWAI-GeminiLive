"""
Plivo API Adapter for Voice Calls
"""

import httpx
from loguru import logger
from typing import Optional, Dict, Any

from src.core.config import config
from .base import BaseCallAdapter


class PlivoAdapter(BaseCallAdapter):
    """Adapter for Plivo Voice API"""

    def __init__(self):
        self.auth_id = config.plivo_auth_id
        self.auth_token = config.plivo_auth_token
        self.base_url = f"https://api.plivo.com/v1/Account/{self.auth_id}"
        self.auth = (self.auth_id, self.auth_token)

    @property
    def provider_name(self) -> str:
        return "plivo"

    async def make_call(
        self,
        phone_number: str,
        answer_url: Optional[str] = None,
        answer_method: str = "POST",
        hangup_url: Optional[str] = None,
        ring_url: Optional[str] = None,
        caller_name: Optional[str] = None,
        plivo_auth_id: Optional[str] = None,
        plivo_auth_token: Optional[str] = None,
        plivo_phone_number: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Initiate an outbound call via Plivo

        Args:
            phone_number: The phone number to call (E.164 format)
            answer_url: URL to fetch call instructions when answered
            answer_method: HTTP method for answer_url
            hangup_url: URL to notify when call ends
            ring_url: URL to notify when call starts ringing
            caller_name: Optional caller name for caller ID
            plivo_auth_id: Per-org Plivo Auth ID (overrides default)
            plivo_auth_token: Per-org Plivo Auth Token (overrides default)
            plivo_phone_number: Per-org Plivo phone number (overrides default)

        Returns:
            API response with call_uuid
        """
        # Use per-org credentials if provided, otherwise fall back to defaults
        auth_id = plivo_auth_id or self.auth_id
        auth_token = plivo_auth_token or self.auth_token
        from_number = plivo_phone_number or config.plivo_phone_number
        api_base_url = f"https://api.plivo.com/v1/Account/{auth_id}"
        auth = (auth_id, auth_token)

        if not answer_url:
            answer_url = f"{config.plivo_callback_url}/plivo/answer"
        if not hangup_url:
            hangup_url = f"{config.plivo_callback_url}/plivo/hangup"

        payload = {
            "from": from_number,
            "to": phone_number,
            "answer_url": answer_url,
            "answer_method": answer_method,
            "hangup_url": hangup_url,
        }

        if ring_url:
            payload["ring_url"] = ring_url
        if caller_name:
            payload["caller_name"] = caller_name

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{api_base_url}/Call/",
                    auth=auth,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Plivo call initiated: {result}")
                return {
                    "success": True,
                    "call_id": result.get("request_uuid"),
                    "call_uuid": result.get("request_uuid"),
                    **result
                }
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error making Plivo call: {e.response.text}")
            return {"success": False, "error": str(e), "details": e.response.text}
        except Exception as e:
            logger.error(f"Error making Plivo call: {e}")
            return {"success": False, "error": str(e)}

    async def get_call_details(self, call_id: str) -> Dict[str, Any]:
        """
        Get details of a specific call

        Args:
            call_id: The call UUID

        Returns:
            Call details including status, duration, etc.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/Call/{call_id}/",
                    auth=self.auth,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                return {"success": True, **result}
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting Plivo call details: {e.response.text}")
            return {"success": False, "error": str(e), "details": e.response.text}
        except Exception as e:
            logger.error(f"Error getting Plivo call details: {e}")
            return {"success": False, "error": str(e)}

    async def get_live_calls(self) -> Dict[str, Any]:
        """
        Get all live/active calls

        Returns:
            List of active call UUIDs
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/Call/?status=live",
                    auth=self.auth,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                return {"success": True, "calls": result.get("calls", [])}
        except Exception as e:
            logger.error(f"Error getting Plivo live calls: {e}")
            return {"success": False, "error": str(e)}

    async def terminate_call(self, call_id: str) -> Dict[str, Any]:
        """
        Terminate/hangup an active call

        Args:
            call_id: The call UUID to terminate

        Returns:
            API response
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.base_url}/Call/{call_id}/",
                    auth=self.auth,
                    timeout=30.0
                )
                logger.info(f"Plivo call terminated: {call_id}")
                return {"success": True}
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error terminating Plivo call: {e.response.text}")
            return {"success": False, "error": str(e), "details": e.response.text}
        except Exception as e:
            logger.error(f"Error terminating Plivo call: {e}")
            return {"success": False, "error": str(e)}

    async def transfer_call(
        self,
        call_id: str,
        transfer_to: str,
        transfer_method: str = "POST",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transfer an active call to a new URL

        Args:
            call_id: The call UUID
            transfer_to: URL to fetch new call instructions
            transfer_method: HTTP method for transfer URL

        Returns:
            API response
        """
        payload = {
            "legs": "aleg",
            "aleg_url": transfer_to,
            "aleg_method": transfer_method
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/Call/{call_id}/",
                    auth=self.auth,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Plivo call transferred: {call_id}")
                return {"success": True, **result}
        except Exception as e:
            logger.error(f"Error transferring Plivo call: {e}")
            return {"success": False, "error": str(e)}

    async def speak_text(
        self,
        call_id: str,
        text: str,
        voice: str = "WOMAN",
        language: str = "en-US",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Speak text on an active call using TTS

        Args:
            call_id: The call UUID
            text: Text to speak
            voice: Voice to use (MAN, WOMAN)
            language: Language code

        Returns:
            API response
        """
        payload = {
            "text": text,
            "voice": voice,
            "language": language
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/Call/{call_id}/Speak/",
                    auth=self.auth,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                logger.info(f"Plivo speaking on call: {call_id}")
                return {"success": True}
        except Exception as e:
            logger.error(f"Error speaking on Plivo call: {e}")
            return {"success": False, "error": str(e)}

    async def stop_speak(self, call_id: str) -> Dict[str, Any]:
        """Stop speaking on a call"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.base_url}/Call/{call_id}/Speak/",
                    auth=self.auth,
                    timeout=30.0
                )
                logger.info(f"Plivo speaking stopped: {call_id}")
                return {"success": True}
        except Exception as e:
            logger.error(f"Error stopping Plivo speak: {e}")
            return {"success": False, "error": str(e)}

    async def play_audio(
        self,
        call_id: str,
        audio_url: str,
        loop: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Play audio on an active call

        Args:
            call_id: The call UUID
            audio_url: URL of the audio file
            loop: Whether to loop the audio

        Returns:
            API response
        """
        payload = {
            "urls": audio_url,
            "loop": loop
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/Call/{call_id}/Play/",
                    auth=self.auth,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                logger.info(f"Plivo playing audio on call: {call_id}")
                return {"success": True}
        except Exception as e:
            logger.error(f"Error playing audio on Plivo call: {e}")
            return {"success": False, "error": str(e)}

    async def stop_audio(self, call_id: str) -> Dict[str, Any]:
        """Stop playing audio on a call"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.base_url}/Call/{call_id}/Play/",
                    auth=self.auth,
                    timeout=30.0
                )
                logger.info(f"Plivo audio stopped: {call_id}")
                return {"success": True}
        except Exception as e:
            logger.error(f"Error stopping Plivo audio: {e}")
            return {"success": False, "error": str(e)}

    async def start_recording(
        self,
        call_id: str,
        callback_url: Optional[str] = None,
        file_format: str = "mp3",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Start recording a call

        Args:
            call_id: The call UUID
            callback_url: URL to receive recording details
            file_format: Recording format (mp3, wav)

        Returns:
            API response
        """
        payload = {"file_format": file_format}
        if callback_url:
            payload["callback_url"] = callback_url

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/Call/{call_id}/Record/",
                    auth=self.auth,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Plivo recording started: {call_id}")
                return {"success": True, **result}
        except Exception as e:
            logger.error(f"Error starting Plivo recording: {e}")
            return {"success": False, "error": str(e)}

    async def stop_recording(self, call_id: str) -> Dict[str, Any]:
        """Stop recording a call"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{self.base_url}/Call/{call_id}/Record/",
                    auth=self.auth,
                    timeout=30.0
                )
                logger.info(f"Plivo recording stopped: {call_id}")
                return {"success": True}
        except Exception as e:
            logger.error(f"Error stopping Plivo recording: {e}")
            return {"success": False, "error": str(e)}

    async def send_dtmf(self, call_id: str, digits: str) -> Dict[str, Any]:
        """
        Send DTMF tones on a call

        Args:
            call_id: The call UUID
            digits: DTMF digits to send

        Returns:
            API response
        """
        payload = {"digits": digits}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/Call/{call_id}/DTMF/",
                    auth=self.auth,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                logger.info(f"Plivo DTMF sent: {call_id}")
                return {"success": True}
        except Exception as e:
            logger.error(f"Error sending Plivo DTMF: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Plivo XML Generation (for webhook responses)
    # =========================================================================

    def generate_answer_xml(
        self,
        speak_text: Optional[str] = None,
        audio_url: Optional[str] = None,
        record: bool = False,
        gather_input: bool = False,
        gather_action_url: Optional[str] = None
    ) -> str:
        """
        Generate Plivo XML response for call handling

        Args:
            speak_text: Text to speak when call is answered
            audio_url: Audio URL to play
            record: Whether to record the call
            gather_input: Whether to gather DTMF input
            gather_action_url: URL to handle gathered input

        Returns:
            Plivo XML string
        """
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<Response>']

        if record:
            xml_parts.append('<Record/>')

        if gather_input and gather_action_url:
            xml_parts.append(f'<GetDigits action="{gather_action_url}" method="POST">')
            if speak_text:
                xml_parts.append(f'<Speak>{speak_text}</Speak>')
            elif audio_url:
                xml_parts.append(f'<Play>{audio_url}</Play>')
            xml_parts.append('</GetDigits>')
        else:
            if speak_text:
                xml_parts.append(f'<Speak>{speak_text}</Speak>')
            if audio_url:
                xml_parts.append(f'<Play>{audio_url}</Play>')

        xml_parts.append('</Response>')
        return '\n'.join(xml_parts)

    def generate_stream_xml(
        self,
        stream_url: str,
        bidirectional: bool = True,
        audio_track: str = "both",
        keep_call_alive: bool = True
    ) -> str:
        """
        Generate Plivo XML for audio streaming (for AI integration)

        Args:
            stream_url: WebSocket URL to stream audio to
            bidirectional: Whether streaming is bidirectional
            audio_track: Which audio to stream (inbound, outbound, both)
            keep_call_alive: Keep call alive after stream ends

        Returns:
            Plivo XML string
        """
        stream_attrs = [
            f'bidirectional="{str(bidirectional).lower()}"',
            f'audioTrack="{audio_track}"',
            f'keepCallAlive="{str(keep_call_alive).lower()}"'
        ]

        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<Response>',
            f'<Stream {" ".join(stream_attrs)}>{stream_url}</Stream>',
            '</Response>'
        ]
        return '\n'.join(xml_parts)

    def generate_connect_xml(
        self,
        phone_number: str,
        caller_id: Optional[str] = None,
        timeout: int = 30
    ) -> str:
        """
        Generate Plivo XML to connect/dial another number

        Args:
            phone_number: Number to connect to
            caller_id: Caller ID to show
            timeout: Ring timeout in seconds

        Returns:
            Plivo XML string
        """
        caller = caller_id or config.plivo_phone_number
        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<Response>',
            f'<Dial callerId="{caller}" timeout="{timeout}">',
            f'<Number>{phone_number}</Number>',
            '</Dial>',
            '</Response>'
        ]
        return '\n'.join(xml_parts)


# Global adapter instance
plivo_adapter = PlivoAdapter()
