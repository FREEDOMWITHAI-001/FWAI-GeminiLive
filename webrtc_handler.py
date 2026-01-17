"""
WebRTC Handler using aiortc for WhatsApp voice calls

Provides full audio access for bidirectional communication with Gemini Live
"""

import asyncio
import uuid
from typing import Optional, Dict, Any, Callable, Awaitable
from loguru import logger
import numpy as np

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.mediastreams import MediaStreamTrack

from config import config
from whatsapp_client import whatsapp_client
from audio_processor import AudioProcessor
from gemini_agent import create_agent, stop_agent, GeminiVoiceAgent


class AudioTrackReader:
    """
    Reads audio from a WebRTC MediaStreamTrack and sends to Gemini
    """

    def __init__(
        self,
        track: MediaStreamTrack,
        audio_processor: AudioProcessor,
        on_audio: Callable[[bytes], Awaitable[None]]
    ):
        self.track = track
        self.audio_processor = audio_processor
        self.on_audio = on_audio
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start reading audio from the track"""
        self._running = True
        self._task = asyncio.create_task(self._read_loop())
        logger.info("Audio track reader started")

    async def stop(self):
        """Stop reading audio"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Audio track reader stopped")

    async def _read_loop(self):
        """Main loop to read audio frames"""
        try:
            while self._running:
                try:
                    # Receive frame from track
                    frame = await self.track.recv()

                    # Extract audio data from frame
                    # aiortc AudioFrame has .to_ndarray() method
                    if hasattr(frame, 'to_ndarray'):
                        audio_array = frame.to_ndarray()

                        # Flatten if needed
                        if len(audio_array.shape) > 1:
                            audio_array = audio_array.flatten()

                        # Convert to int16 if needed
                        if audio_array.dtype != np.int16:
                            if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                                audio_array = (audio_array * 32767).astype(np.int16)
                            else:
                                audio_array = audio_array.astype(np.int16)

                        # Convert to bytes
                        audio_bytes = audio_array.tobytes()

                        # Process and send to callback
                        processed = self.audio_processor.process_webrtc_audio(audio_bytes)
                        if processed and self.on_audio:
                            await self.on_audio(processed)

                except Exception as e:
                    if self._running:
                        logger.error(f"Error reading audio frame: {e}")
                    await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in audio read loop: {e}")


class CallSession:
    """
    Manages a single WhatsApp voice call session

    Handles WebRTC connection, audio processing, and Gemini integration
    """

    def __init__(
        self,
        call_id: str,
        phone_number: str,
        caller_name: str = "Customer",
        is_outbound: bool = False
    ):
        self.call_id = call_id
        self.phone_number = phone_number
        self.caller_name = caller_name
        self.is_outbound = is_outbound

        # WebRTC
        self.pc: Optional[RTCPeerConnection] = None

        # Audio processing
        self.audio_processor: Optional[AudioProcessor] = None
        self.track_reader: Optional[AudioTrackReader] = None

        # Gemini agent
        self.agent: Optional[GeminiVoiceAgent] = None

        # Audio output buffer
        self._audio_output_queue: asyncio.Queue = asyncio.Queue()

        # State
        self._running = False

    async def setup_peer_connection(self):
        """Create and configure the RTCPeerConnection"""
        ice_servers = [{"urls": s["urls"]} for s in config.ice_servers]

        self.pc = RTCPeerConnection(configuration={
            "iceServers": ice_servers
        })

        # Handle incoming audio track (from WhatsApp caller)
        @self.pc.on("track")
        async def on_track(track):
            logger.info(f"Received track: {track.kind}")
            if track.kind == "audio":
                await self._handle_audio_track(track)

        # Handle ICE candidates
        @self.pc.on("icecandidate")
        async def on_ice_candidate(candidate):
            if candidate:
                logger.debug(f"ICE candidate: {candidate.candidate}")
                await whatsapp_client.send_ice_candidate(
                    self.call_id,
                    {
                        "candidate": candidate.candidate,
                        "sdpMid": candidate.sdpMid,
                        "sdpMLineIndex": candidate.sdpMLineIndex
                    }
                )

        # Handle connection state changes
        @self.pc.on("connectionstatechange")
        async def on_connection_state_change():
            logger.info(f"Connection state: {self.pc.connectionState}")
            if self.pc.connectionState == "connected":
                logger.info(f"Call {self.call_id} connected!")
            elif self.pc.connectionState in ["failed", "closed", "disconnected"]:
                logger.info(f"Call {self.call_id} ended: {self.pc.connectionState}")
                await self.stop()

        logger.info(f"Peer connection setup for call {self.call_id}")

    async def _handle_audio_track(self, track: MediaStreamTrack):
        """Handle incoming audio track from WhatsApp"""
        logger.info(f"Setting up audio processing for call {self.call_id}")

        # Create audio processor
        self.audio_processor = AudioProcessor()
        await self.audio_processor.start()

        # Create track reader - sends audio to Gemini Live via WebSocket
        async def on_audio(pcm_data: bytes):
            if self.agent:
                await self.agent.send_audio(pcm_data)

        self.track_reader = AudioTrackReader(track, self.audio_processor, on_audio)
        await self.track_reader.start()

        logger.info(f"Audio processing started for call {self.call_id}")

    async def _handle_gemini_audio(self, audio_data: bytes):
        """Handle audio output from Gemini"""
        # Queue audio for output
        await self._audio_output_queue.put(audio_data)

    async def start_agent(self):
        """Start the Gemini voice agent"""
        self.agent = await create_agent(
            call_id=self.call_id,
            caller_name=self.caller_name,
            on_audio_output=self._handle_gemini_audio
        )

        if self.agent:
            logger.info(f"Gemini agent started for call {self.call_id}")
            return True
        else:
            logger.error(f"Failed to start Gemini agent for call {self.call_id}")
            return False

    async def create_offer(self) -> str:
        """Create SDP offer for outbound call"""
        await self.setup_peer_connection()

        # Add transceivers for audio
        self.pc.addTransceiver("audio", direction="sendrecv")

        # Create offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        # Wait for ICE gathering
        await self._wait_for_ice_gathering()

        return self.pc.localDescription.sdp

    async def handle_answer(self, sdp_answer: str):
        """Handle SDP answer from WhatsApp"""
        answer = RTCSessionDescription(sdp=sdp_answer, type="answer")
        await self.pc.setRemoteDescription(answer)
        logger.info(f"Remote description set for call {self.call_id}")

    async def handle_offer(self, sdp_offer: str) -> str:
        """Handle incoming call with SDP offer"""
        await self.setup_peer_connection()

        # Set remote description (offer from WhatsApp)
        offer = RTCSessionDescription(sdp=sdp_offer, type="offer")
        await self.pc.setRemoteDescription(offer)

        # Create answer
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)

        # Wait for ICE gathering
        await self._wait_for_ice_gathering()

        return self.pc.localDescription.sdp

    async def add_ice_candidate(self, candidate_data: Dict[str, Any]):
        """Add ICE candidate from WhatsApp"""
        try:
            candidate = RTCIceCandidate(
                candidate=candidate_data.get("candidate"),
                sdpMid=candidate_data.get("sdpMid"),
                sdpMLineIndex=candidate_data.get("sdpMLineIndex")
            )
            await self.pc.addIceCandidate(candidate)
        except Exception as e:
            logger.error(f"Error adding ICE candidate: {e}")

    async def _wait_for_ice_gathering(self, timeout: float = 5.0):
        """Wait for ICE gathering to complete"""
        if self.pc.iceGatheringState == "complete":
            return

        gathering_complete = asyncio.Event()

        @self.pc.on("icegatheringstatechange")
        def check_state():
            if self.pc.iceGatheringState == "complete":
                gathering_complete.set()

        try:
            await asyncio.wait_for(gathering_complete.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("ICE gathering timed out")

    async def stop(self):
        """Stop the call session"""
        logger.info(f"Stopping call session {self.call_id}")
        self._running = False

        # Stop track reader
        if self.track_reader:
            await self.track_reader.stop()

        # Stop audio processor
        if self.audio_processor:
            await self.audio_processor.stop()

        # Stop Gemini agent
        if self.agent:
            await stop_agent(self.call_id)

        # Close peer connection
        if self.pc:
            await self.pc.close()

        logger.info(f"Call session {self.call_id} stopped")


# Store active call sessions
active_sessions: Dict[str, CallSession] = {}


async def make_outbound_call(
    phone_number: str,
    caller_name: str = "Customer"
) -> Dict[str, Any]:
    """
    Make an outbound call to a WhatsApp user
    """
    call_id = str(uuid.uuid4())

    logger.info(f"Initiating outbound call to {phone_number}")

    session = CallSession(
        call_id=call_id,
        phone_number=phone_number,
        caller_name=caller_name,
        is_outbound=True
    )

    try:
        # Create SDP offer
        sdp_offer = await session.create_offer()

        # Start Gemini agent
        await session.start_agent()

        # Make call via WhatsApp API
        result = await whatsapp_client.make_call(
            phone_number=phone_number,
            sdp_offer=sdp_offer
        )

        if result.get("success"):
            wa_call_id = result.get("calls", [{}])[0].get("id", call_id)
            session.call_id = wa_call_id
            active_sessions[wa_call_id] = session

            sdp_answer = result.get("calls", [{}])[0].get("sdp")
            if sdp_answer:
                await session.handle_answer(sdp_answer)

            logger.info(f"Outbound call initiated: {wa_call_id}")
            return {
                "success": True,
                "call_id": wa_call_id,
                "message": "Call initiated"
            }
        else:
            await session.stop()
            return {
                "success": False,
                "error": result.get("error", "Unknown error")
            }

    except Exception as e:
        logger.error(f"Error making outbound call: {e}")
        import traceback
        traceback.print_exc()
        await session.stop()
        return {
            "success": False,
            "error": str(e)
        }


async def handle_incoming_call(
    call_id: str,
    caller_phone: str,
    sdp_offer: str,
    caller_name: str = "Customer"
) -> Dict[str, Any]:
    """Handle an incoming call from WhatsApp"""
    logger.info(f"Handling incoming call {call_id} from {caller_phone}")

    session = CallSession(
        call_id=call_id,
        phone_number=caller_phone,
        caller_name=caller_name,
        is_outbound=False
    )

    try:
        sdp_answer = await session.handle_offer(sdp_offer)
        await session.start_agent()

        result = await whatsapp_client.answer_call(
            call_id=call_id,
            sdp_answer=sdp_answer
        )

        if result.get("success"):
            active_sessions[call_id] = session
            logger.info(f"Incoming call answered: {call_id}")
            return {
                "success": True,
                "call_id": call_id,
                "message": "Call answered"
            }
        else:
            await session.stop()
            return {
                "success": False,
                "error": result.get("error", "Failed to answer call")
            }

    except Exception as e:
        logger.error(f"Error handling incoming call: {e}")
        await session.stop()
        return {
            "success": False,
            "error": str(e)
        }


async def handle_ice_candidate(call_id: str, candidate: Dict[str, Any]):
    """Handle ICE candidate from WhatsApp"""
    session = active_sessions.get(call_id)
    if session:
        await session.add_ice_candidate(candidate)


async def terminate_call(call_id: str) -> Dict[str, Any]:
    """Terminate a call"""
    session = active_sessions.get(call_id)
    if session:
        await session.stop()
        del active_sessions[call_id]
        await whatsapp_client.terminate_call(call_id)
        return {"success": True, "message": "Call terminated"}

    return {"success": False, "error": "Call not found"}


def get_active_calls() -> list:
    """Get list of active calls"""
    return [
        {
            "call_id": call_id,
            "phone_number": session.phone_number,
            "is_outbound": session.is_outbound
        }
        for call_id, session in active_sessions.items()
    ]
