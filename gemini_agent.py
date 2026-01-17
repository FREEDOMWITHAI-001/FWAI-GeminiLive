"""
Gemini Live Agent - WebSocket Client
Connects to gemini-live-service.py (port 8003) for voice conversations
"""

import asyncio
import json
from typing import Optional, Callable, Awaitable, Dict, Any
from loguru import logger

import websockets
from websockets.client import WebSocketClientProtocol

from config import config


class GeminiWebSocketClient:
    """
    WebSocket client that connects to gemini-live-service.py

    Sends audio from WhatsApp caller to Gemini Live
    Receives audio from Gemini Live to send back to caller
    """

    def __init__(
        self,
        call_id: str,
        caller_name: str = "Customer",
        on_audio_output: Optional[Callable[[bytes], Awaitable[None]]] = None
    ):
        self.call_id = call_id
        self.caller_name = caller_name
        self.on_audio_output = on_audio_output

        # WebSocket connection
        self.ws: Optional[WebSocketClientProtocol] = None

        # State
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None
        self._connected = asyncio.Event()

    async def connect(self) -> bool:
        """Connect to gemini-live-service.py"""
        try:
            logger.info(f"Connecting to Gemini Live service at {config.gemini_live_ws_url}")

            self.ws = await websockets.connect(
                config.gemini_live_ws_url,
                ping_interval=20,
                ping_timeout=10
            )

            self._running = True

            # Start receiving messages
            self._receive_task = asyncio.create_task(self._receive_loop())

            # Send start message to initiate call
            await self._send_start()

            # Wait for "started" confirmation
            try:
                await asyncio.wait_for(self._connected.wait(), timeout=10.0)
                logger.info(f"Connected to Gemini Live for call {self.call_id}")
                return True
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for Gemini Live to start")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to Gemini Live service: {e}")
            return False

    async def _send_start(self):
        """Send start message to gemini-live-service.py"""
        start_message = {
            "type": "start",
            "call_id": self.call_id,
            "caller_name": self.caller_name
        }
        await self.ws.send(json.dumps(start_message))
        logger.info(f"Sent start message for call {self.call_id}")

    async def _receive_loop(self):
        """Receive messages from gemini-live-service.py"""
        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "started":
                        # Gemini Live pipeline started
                        logger.info(f"Gemini Live started for call {self.call_id}")
                        self._connected.set()

                    elif msg_type == "audio":
                        # Audio from Gemini Live (TTS output)
                        audio_hex = data.get("data", "")
                        if audio_hex and self.on_audio_output:
                            audio_bytes = bytes.fromhex(audio_hex)
                            await self.on_audio_output(audio_bytes)

                    elif msg_type == "stopped":
                        logger.info(f"Gemini Live stopped for call {self.call_id}")
                        break

                    elif msg_type == "error":
                        logger.error(f"Gemini Live error: {data.get('message')}")

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from Gemini Live: {message}")
                except Exception as e:
                    logger.error(f"Error processing Gemini Live message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket closed for call {self.call_id}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")

    async def send_audio(self, pcm_data: bytes):
        """
        Send audio to gemini-live-service.py

        Args:
            pcm_data: PCM audio bytes (16kHz, mono, 16-bit)
        """
        if self._running and self.ws:
            try:
                message = {
                    "type": "audio",
                    "data": pcm_data.hex()
                }
                await self.ws.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending audio: {e}")

    async def stop(self):
        """Stop and disconnect"""
        logger.info(f"Stopping Gemini Live client for call {self.call_id}")
        self._running = False

        # Send stop message
        if self.ws:
            try:
                await self.ws.send(json.dumps({
                    "type": "stop",
                    "call_id": self.call_id
                }))
            except:
                pass

        # Cancel receive task
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket
        if self.ws:
            try:
                await self.ws.close()
            except:
                pass

        logger.info(f"Gemini Live client stopped for call {self.call_id}")

    @property
    def is_connected(self) -> bool:
        return self._running and self._connected.is_set()


# Alias for compatibility
GeminiVoiceAgent = GeminiWebSocketClient


# Store active agents
active_agents: Dict[str, GeminiWebSocketClient] = {}


async def create_agent(
    call_id: str,
    caller_name: str = "Customer",
    on_audio_output: Optional[Callable[[bytes], Awaitable[None]]] = None
) -> Optional[GeminiWebSocketClient]:
    """
    Create and connect a new Gemini Live WebSocket client

    Args:
        call_id: Unique call identifier
        caller_name: Name of the caller
        on_audio_output: Callback for audio output from Gemini

    Returns:
        GeminiWebSocketClient if successful, None otherwise
    """
    client = GeminiWebSocketClient(
        call_id=call_id,
        caller_name=caller_name,
        on_audio_output=on_audio_output
    )

    if await client.connect():
        active_agents[call_id] = client
        return client

    return None


async def stop_agent(call_id: str):
    """Stop and remove an agent"""
    if call_id in active_agents:
        await active_agents[call_id].stop()
        del active_agents[call_id]


def get_agent(call_id: str) -> Optional[GeminiWebSocketClient]:
    """Get an active agent by call ID"""
    return active_agents.get(call_id)
