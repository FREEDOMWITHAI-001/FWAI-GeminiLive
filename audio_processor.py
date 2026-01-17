"""
Audio processing utilities for WebRTC and Gemini Live integration
Uses numpy/scipy for audio processing (no compilation required)
"""

import asyncio
import numpy as np
from scipy import signal
from typing import Optional, Callable, Awaitable
from loguru import logger

from config import config


class AudioProcessor:
    """
    Handles audio processing between WebRTC and Gemini Live

    - Resamples audio between WebRTC (48kHz) and Gemini (16kHz)
    - Converts between formats
    """

    def __init__(
        self,
        input_sample_rate: int = 48000,  # WebRTC typically uses 48kHz
        output_sample_rate: int = 16000,  # Gemini uses 16kHz
        channels: int = 1
    ):
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.channels = channels

        # Callbacks
        self.on_audio_to_gemini: Optional[Callable[[bytes], Awaitable[None]]] = None

        # Processing state
        self._running = False

    async def start(self):
        """Start audio processing"""
        self._running = True
        logger.info("Audio processor started")

    async def stop(self):
        """Stop audio processing"""
        self._running = False
        logger.info("Audio processor stopped")

    def resample_audio(
        self,
        audio_data: bytes,
        from_rate: int,
        to_rate: int
    ) -> bytes:
        """
        Resample audio from one sample rate to another

        Args:
            audio_data: Input PCM audio bytes (16-bit signed)
            from_rate: Source sample rate
            to_rate: Target sample rate

        Returns:
            Resampled PCM audio bytes
        """
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

            # Calculate resampling ratio
            ratio = to_rate / from_rate

            # Number of output samples
            num_samples = int(len(audio_array) * ratio)

            # Resample using scipy
            resampled = signal.resample(audio_array, num_samples)

            # Convert back to int16
            resampled_int16 = np.clip(resampled, -32768, 32767).astype(np.int16)

            return resampled_int16.tobytes()
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return audio_data

    def process_webrtc_audio(self, audio_data: bytes) -> bytes:
        """
        Convert WebRTC audio (48kHz) to Gemini format (16kHz)

        Args:
            audio_data: PCM audio bytes from WebRTC

        Returns:
            PCM bytes at 16kHz mono
        """
        if self.input_sample_rate == self.output_sample_rate:
            return audio_data

        return self.resample_audio(
            audio_data,
            self.input_sample_rate,
            self.output_sample_rate
        )

    def process_gemini_audio(self, audio_data: bytes) -> bytes:
        """
        Convert Gemini audio (16kHz) to WebRTC format (48kHz)

        Args:
            audio_data: PCM bytes from Gemini

        Returns:
            PCM bytes at 48kHz
        """
        if self.input_sample_rate == self.output_sample_rate:
            return audio_data

        return self.resample_audio(
            audio_data,
            self.output_sample_rate,
            self.input_sample_rate
        )

    async def feed_webrtc_audio(self, audio_data: bytes):
        """
        Feed audio from WebRTC to be sent to Gemini

        Args:
            audio_data: PCM audio bytes from WebRTC
        """
        if not self._running:
            return

        # Resample to Gemini format
        processed = self.process_webrtc_audio(audio_data)

        if processed and self.on_audio_to_gemini:
            await self.on_audio_to_gemini(processed)


def generate_silence(duration_ms: int, sample_rate: int = 48000) -> bytes:
    """Generate silence audio data"""
    num_samples = int(sample_rate * duration_ms / 1000)
    silence = np.zeros(num_samples, dtype=np.int16)
    return silence.tobytes()


def audio_to_base64(audio_data: bytes) -> str:
    """Convert audio bytes to base64 string"""
    import base64
    return base64.b64encode(audio_data).decode('utf-8')


def base64_to_audio(base64_data: str) -> bytes:
    """Convert base64 string to audio bytes"""
    import base64
    return base64.b64decode(base64_data)
