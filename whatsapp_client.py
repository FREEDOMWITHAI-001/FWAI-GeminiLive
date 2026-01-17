"""
WhatsApp Business API Client for Voice Calls and Messages
"""

import httpx
from loguru import logger
from typing import Optional, Dict, Any
from config import config


class WhatsAppClient:
    """Client for WhatsApp Business API"""

    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {config.meta_access_token}",
            "Content-Type": "application/json",
        }

    async def make_call(
        self,
        phone_number: str,
        sdp_offer: str,
        ice_candidates: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Initiate an outbound call to a WhatsApp user

        Args:
            phone_number: The phone number to call (with country code, no +)
            sdp_offer: The SDP offer from WebRTC
            ice_candidates: Optional list of ICE candidates

        Returns:
            API response with call_id and other details
        """
        payload = {
            "to": phone_number,
            "type": "audio",
            "audio": {
                "sdp": sdp_offer
            }
        }

        if ice_candidates:
            payload["audio"]["ice_candidates"] = ice_candidates

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    config.whatsapp_calls_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Call initiated successfully: {result}")
                return {"success": True, **result}
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error making call: {e.response.text}")
            return {"success": False, "error": str(e), "details": e.response.text}
        except Exception as e:
            logger.error(f"Error making call: {e}")
            return {"success": False, "error": str(e)}

    async def answer_call(
        self,
        call_id: str,
        sdp_answer: str,
        ice_candidates: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Answer an incoming call with SDP answer

        Args:
            call_id: The call ID from the webhook
            sdp_answer: The SDP answer from WebRTC
            ice_candidates: Optional list of ICE candidates

        Returns:
            API response
        """
        # First pre-accept the call
        pre_accept_payload = {
            "call_id": call_id,
            "action": "pre_accept"
        }

        try:
            async with httpx.AsyncClient() as client:
                # Pre-accept
                response = await client.post(
                    config.whatsapp_calls_url,
                    headers=self.headers,
                    json=pre_accept_payload,
                    timeout=30.0
                )
                logger.info(f"Pre-accept response: {response.status_code}")

                # Accept with SDP answer
                accept_payload = {
                    "call_id": call_id,
                    "action": "accept",
                    "sdp": sdp_answer
                }

                if ice_candidates:
                    accept_payload["ice_candidates"] = ice_candidates

                response = await client.post(
                    config.whatsapp_calls_url,
                    headers=self.headers,
                    json=accept_payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Call answered successfully: {result}")
                return {"success": True, **result}
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error answering call: {e.response.text}")
            return {"success": False, "error": str(e), "details": e.response.text}
        except Exception as e:
            logger.error(f"Error answering call: {e}")
            return {"success": False, "error": str(e)}

    async def send_ice_candidate(
        self,
        call_id: str,
        ice_candidate: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send ICE candidate to WhatsApp

        Args:
            call_id: The call ID
            ice_candidate: The ICE candidate object

        Returns:
            API response
        """
        payload = {
            "call_id": call_id,
            "action": "ice_candidate",
            "ice_candidate": ice_candidate
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    config.whatsapp_calls_url,
                    headers=self.headers,
                    json=payload,
                    timeout=10.0
                )
                return {"success": True}
        except Exception as e:
            logger.error(f"Error sending ICE candidate: {e}")
            return {"success": False, "error": str(e)}

    async def terminate_call(self, call_id: str) -> Dict[str, Any]:
        """
        Terminate an active call

        Args:
            call_id: The call ID to terminate

        Returns:
            API response
        """
        payload = {
            "call_id": call_id,
            "action": "terminate"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    config.whatsapp_calls_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                logger.info(f"Call terminated: {call_id}")
                return {"success": True}
        except Exception as e:
            logger.error(f"Error terminating call: {e}")
            return {"success": False, "error": str(e)}

    async def reject_call(self, call_id: str) -> Dict[str, Any]:
        """
        Reject an incoming call

        Args:
            call_id: The call ID to reject

        Returns:
            API response
        """
        payload = {
            "call_id": call_id,
            "action": "reject"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    config.whatsapp_calls_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                logger.info(f"Call rejected: {call_id}")
                return {"success": True}
        except Exception as e:
            logger.error(f"Error rejecting call: {e}")
            return {"success": False, "error": str(e)}

    async def send_message(
        self,
        phone_number: str,
        message: str
    ) -> Dict[str, Any]:
        """
        Send a text message to a WhatsApp user

        Args:
            phone_number: The phone number to message
            message: The message text

        Returns:
            API response
        """
        payload = {
            "messaging_product": "whatsapp",
            "to": phone_number,
            "type": "text",
            "text": {"body": message}
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    config.whatsapp_messages_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Message sent to {phone_number}")
                return {"success": True, **result}
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return {"success": False, "error": str(e)}


# Global client instance
whatsapp_client = WhatsAppClient()
