"""
Exotel API Adapter for Voice Calls

Exotel is popular for Indian telephony with good local number support.
"""

import httpx
import base64
from loguru import logger
from typing import Optional, Dict, Any

import sys
sys.path.append('..')
from config import config
from .base import BaseCallAdapter


class ExotelAdapter(BaseCallAdapter):
    """Adapter for Exotel Voice API"""

    def __init__(self):
        self.api_key = config.exotel_api_key
        self.api_token = config.exotel_api_token
        self.sid = config.exotel_sid
        self.base_url = f"https://api.exotel.com/v1/Accounts/{self.sid}"

        # Basic auth header
        credentials = f"{self.api_key}:{self.api_token}"
        encoded = base64.b64encode(credentials.encode()).decode()
        self.headers = {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

    @property
    def provider_name(self) -> str:
        return "exotel"

    async def make_call(
        self,
        phone_number: str,
        app_id: Optional[str] = None,
        caller_id: Optional[str] = None,
        custom_field: Optional[str] = None,
        status_callback: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Initiate an outbound call via Exotel

        Args:
            phone_number: The phone number to call (with country code)
            app_id: Exotel App ID (ExoPhone flow)
            caller_id: The Exotel virtual number (ExoPhone)
            custom_field: Custom field to track the call
            status_callback: URL for call status updates

        Returns:
            API response with call_sid
        """
        caller = caller_id or config.exotel_caller_id
        app = app_id or config.exotel_app_id

        # Exotel uses form data, not JSON
        data = {
            "From": caller,
            "To": phone_number,
            "CallerId": caller,
        }

        if app:
            data["Url"] = f"http://my.exotel.com/{self.sid}/exoml/start_voice/{app}"

        if custom_field:
            data["CustomField"] = custom_field

        if status_callback:
            data["StatusCallback"] = status_callback
        elif config.exotel_callback_url:
            data["StatusCallback"] = f"{config.exotel_callback_url}/exotel/status"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/Calls/connect.json",
                    headers=self.headers,
                    data=data,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()

                call_data = result.get("Call", {})
                logger.info(f"Exotel call initiated: {call_data.get('Sid')}")

                return {
                    "success": True,
                    "call_id": call_data.get("Sid"),
                    "call_sid": call_data.get("Sid"),
                    "status": call_data.get("Status"),
                    **result
                }
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error making Exotel call: {e.response.text}")
            return {"success": False, "error": str(e), "details": e.response.text}
        except Exception as e:
            logger.error(f"Error making Exotel call: {e}")
            return {"success": False, "error": str(e)}

    async def make_call_to_flow(
        self,
        phone_number: str,
        flow_id: str,
        caller_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Connect a call to an Exotel IVR flow

        Args:
            phone_number: The phone number to call
            flow_id: The Exotel flow/app ID
            caller_id: The Exotel virtual number

        Returns:
            API response with call_sid
        """
        return await self.make_call(
            phone_number=phone_number,
            app_id=flow_id,
            caller_id=caller_id,
            **kwargs
        )

    async def connect_two_numbers(
        self,
        from_number: str,
        to_number: str,
        caller_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Connect two phone numbers (click-to-call)

        Exotel calls the 'from' number first, then connects to 'to' number.

        Args:
            from_number: First number to call (agent/customer)
            to_number: Second number to connect
            caller_id: The Exotel virtual number

        Returns:
            API response with call_sid
        """
        caller = caller_id or config.exotel_caller_id

        data = {
            "From": from_number,
            "To": to_number,
            "CallerId": caller,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/Calls/connect.json",
                    headers=self.headers,
                    data=data,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()

                call_data = result.get("Call", {})
                logger.info(f"Exotel call connected: {call_data.get('Sid')}")

                return {
                    "success": True,
                    "call_id": call_data.get("Sid"),
                    **result
                }
        except Exception as e:
            logger.error(f"Error connecting Exotel call: {e}")
            return {"success": False, "error": str(e)}

    async def get_call_details(self, call_id: str) -> Dict[str, Any]:
        """
        Get details of a specific call

        Args:
            call_id: The call SID

        Returns:
            Call details including status, duration, etc.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/Calls/{call_id}.json",
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()

                call_data = result.get("Call", {})
                return {
                    "success": True,
                    "call_id": call_data.get("Sid"),
                    "status": call_data.get("Status"),
                    "direction": call_data.get("Direction"),
                    "duration": call_data.get("Duration"),
                    "price": call_data.get("Price"),
                    "recording_url": call_data.get("RecordingUrl"),
                    **call_data
                }
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting Exotel call details: {e.response.text}")
            return {"success": False, "error": str(e), "details": e.response.text}
        except Exception as e:
            logger.error(f"Error getting Exotel call details: {e}")
            return {"success": False, "error": str(e)}

    async def terminate_call(self, call_id: str) -> Dict[str, Any]:
        """
        Terminate/hangup an active call

        Args:
            call_id: The call SID to terminate

        Returns:
            API response
        """
        data = {
            "Status": "completed"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/Calls/{call_id}.json",
                    headers=self.headers,
                    data=data,
                    timeout=30.0
                )
                response.raise_for_status()
                logger.info(f"Exotel call terminated: {call_id}")
                return {"success": True}
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error terminating Exotel call: {e.response.text}")
            return {"success": False, "error": str(e), "details": e.response.text}
        except Exception as e:
            logger.error(f"Error terminating Exotel call: {e}")
            return {"success": False, "error": str(e)}

    async def transfer_call(
        self,
        call_id: str,
        transfer_to: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transfer a call to another number

        Args:
            call_id: The call SID
            transfer_to: Number to transfer to

        Returns:
            API response
        """
        data = {
            "Legs[0][Number]": transfer_to
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/Calls/{call_id}.json",
                    headers=self.headers,
                    data=data,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Exotel call transferred: {call_id}")
                return {"success": True, **result}
        except Exception as e:
            logger.error(f"Error transferring Exotel call: {e}")
            return {"success": False, "error": str(e)}

    async def start_recording(
        self,
        call_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Start recording a call

        Note: Exotel typically records all calls by default.
        This enables recording if it was disabled.

        Args:
            call_id: The call SID

        Returns:
            API response
        """
        data = {
            "Record": "true"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/Calls/{call_id}.json",
                    headers=self.headers,
                    data=data,
                    timeout=30.0
                )
                response.raise_for_status()
                logger.info(f"Exotel recording started: {call_id}")
                return {"success": True}
        except Exception as e:
            logger.error(f"Error starting Exotel recording: {e}")
            return {"success": False, "error": str(e)}

    async def get_recording(self, call_id: str) -> Dict[str, Any]:
        """
        Get recording URL for a call

        Args:
            call_id: The call SID

        Returns:
            Recording URL and details
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/Calls/{call_id}.json",
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()

                call_data = result.get("Call", {})
                recording_url = call_data.get("RecordingUrl")

                if recording_url:
                    return {
                        "success": True,
                        "recording_url": recording_url,
                        "call_id": call_id
                    }
                else:
                    return {
                        "success": False,
                        "error": "Recording not available"
                    }
        except Exception as e:
            logger.error(f"Error getting Exotel recording: {e}")
            return {"success": False, "error": str(e)}

    async def play_audio(
        self,
        call_id: str,
        audio_url: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Play audio on an active call

        Note: Exotel supports this via ExoML flows.
        For dynamic audio, use transfer_call with a flow URL.

        Args:
            call_id: The call SID
            audio_url: URL of the audio file

        Returns:
            API response
        """
        # Exotel requires ExoML for playing audio
        # This would need a pre-configured flow
        logger.warning("Exotel play_audio requires ExoML flow configuration")
        return {
            "success": False,
            "error": "Use transfer_call with an ExoML flow for dynamic audio"
        }

    async def send_sms(
        self,
        phone_number: str,
        message: str,
        sender_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send SMS via Exotel

        Args:
            phone_number: The phone number to SMS
            message: The message content
            sender_id: The sender ID (ExoPhone or approved sender)

        Returns:
            API response
        """
        sender = sender_id or config.exotel_caller_id

        data = {
            "From": sender,
            "To": phone_number,
            "Body": message
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/Sms/send.json",
                    headers=self.headers,
                    data=data,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()

                sms_data = result.get("SMSMessage", {})
                logger.info(f"Exotel SMS sent: {sms_data.get('Sid')}")

                return {
                    "success": True,
                    "sms_id": sms_data.get("Sid"),
                    "status": sms_data.get("Status"),
                    **result
                }
        except Exception as e:
            logger.error(f"Error sending Exotel SMS: {e}")
            return {"success": False, "error": str(e)}

    async def get_account_details(self) -> Dict[str, Any]:
        """
        Get Exotel account details and balance

        Returns:
            Account information
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}.json",
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()

                account = result.get("Account", {})
                return {
                    "success": True,
                    "sid": account.get("Sid"),
                    "status": account.get("Status"),
                    "balance": account.get("Balance"),
                    **account
                }
        except Exception as e:
            logger.error(f"Error getting Exotel account: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # ExoML Generation (for IVR flows)
    # =========================================================================

    def generate_response_xml(
        self,
        say_text: Optional[str] = None,
        play_url: Optional[str] = None,
        record: bool = False,
        gather_input: bool = False,
        gather_action_url: Optional[str] = None,
        num_digits: int = 1
    ) -> str:
        """
        Generate ExoML response for call handling

        Args:
            say_text: Text to speak (TTS)
            play_url: Audio URL to play
            record: Whether to record
            gather_input: Whether to gather DTMF
            gather_action_url: URL for gathered input
            num_digits: Number of digits to gather

        Returns:
            ExoML string
        """
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>',  '<Response>']

        if record:
            xml_parts.append('<Record/>')

        if gather_input and gather_action_url:
            xml_parts.append(f'<Gather action="{gather_action_url}" numDigits="{num_digits}">')
            if say_text:
                xml_parts.append(f'<Say>{say_text}</Say>')
            elif play_url:
                xml_parts.append(f'<Play>{play_url}</Play>')
            xml_parts.append('</Gather>')
        else:
            if say_text:
                xml_parts.append(f'<Say>{say_text}</Say>')
            if play_url:
                xml_parts.append(f'<Play>{play_url}</Play>')

        xml_parts.append('</Response>')
        return '\n'.join(xml_parts)

    def generate_dial_xml(
        self,
        phone_number: str,
        caller_id: Optional[str] = None,
        timeout: int = 30,
        record: bool = True
    ) -> str:
        """
        Generate ExoML to dial a number

        Args:
            phone_number: Number to dial
            caller_id: Caller ID to show
            timeout: Ring timeout
            record: Whether to record

        Returns:
            ExoML string
        """
        caller = caller_id or config.exotel_caller_id
        record_str = "true" if record else "false"

        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<Response>',
            f'<Dial callerId="{caller}" timeout="{timeout}" record="{record_str}">',
            f'<Number>{phone_number}</Number>',
            '</Dial>',
            '</Response>'
        ]
        return '\n'.join(xml_parts)


# Global adapter instance
exotel_adapter = ExotelAdapter()
