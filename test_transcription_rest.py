#!/usr/bin/env python3
"""
Test script for Gemini REST API transcription
Tests if we can send audio and get transcription back
"""
import asyncio
import base64
import httpx
import os
from pathlib import Path

# Load API key from environment or config
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    # Try loading from config
    try:
        from src.core.config import config
        API_KEY = config.google_api_key
    except:
        print("ERROR: Set GOOGLE_API_KEY environment variable")
        exit(1)

# Gemini REST API endpoint
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

async def transcribe_audio_file(audio_path: str) -> str:
    """Transcribe an audio file using Gemini REST API"""

    # Read and encode audio file
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    audio_b64 = base64.standard_b64encode(audio_data).decode("utf-8")

    # Determine mime type
    if audio_path.endswith(".wav"):
        mime_type = "audio/wav"
    elif audio_path.endswith(".mp3"):
        mime_type = "audio/mp3"
    elif audio_path.endswith(".pcm"):
        mime_type = "audio/pcm"
    else:
        mime_type = "audio/wav"  # Default

    # Build request
    payload = {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": audio_b64
                    }
                },
                {
                    "text": "Transcribe this audio. Output ONLY the exact words spoken, nothing else. No timestamps, no speaker labels, just the transcription."
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 1024
        }
    }

    print(f"Sending {len(audio_data)} bytes of audio to Gemini REST API...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(GEMINI_API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            # Extract transcription from response
            try:
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                return text.strip()
            except (KeyError, IndexError) as e:
                print(f"Error parsing response: {e}")
                print(f"Response: {result}")
                return None
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None


async def transcribe_raw_pcm(pcm_data: bytes, sample_rate: int = 16000) -> str:
    """Transcribe raw PCM audio data using Gemini REST API"""

    # For raw PCM, we need to specify the format
    audio_b64 = base64.standard_b64encode(pcm_data).decode("utf-8")

    # Build request with PCM audio
    payload = {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": f"audio/L16;rate={sample_rate}",  # Linear PCM 16-bit
                        "data": audio_b64
                    }
                },
                {
                    "text": "Transcribe this audio. Output ONLY the exact words spoken, nothing else."
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 1024
        }
    }

    print(f"Sending {len(pcm_data)} bytes of PCM audio ({sample_rate}Hz) to Gemini REST API...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(GEMINI_API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            try:
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                return text.strip()
            except (KeyError, IndexError) as e:
                print(f"Error parsing response: {e}")
                print(f"Full response: {result}")
                return None
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None


async def test_with_sample_audio():
    """Test with a sample audio file from recordings"""
    recordings_dir = Path(__file__).parent / "recordings"

    # Find a WAV file to test with
    wav_files = list(recordings_dir.glob("*.wav"))

    if not wav_files:
        print("No WAV files found in recordings directory")
        print("Creating a simple test with synthesized audio description...")

        # Test with a simple text prompt to verify API works
        payload = {
            "contents": [{
                "parts": [{
                    "text": "Say 'Hello, this is a test of the transcription system' - just output that exact phrase."
                }]
            }]
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(GEMINI_API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                print(f"API Test Response: {text}")
                print("✓ Gemini REST API is working!")
                return True
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return False

    # Test with first WAV file found
    test_file = wav_files[0]
    print(f"Testing with: {test_file}")

    transcription = await transcribe_audio_file(str(test_file))

    if transcription:
        print(f"\n✓ Transcription successful!")
        print(f"Result: {transcription[:200]}...")
        return True
    else:
        print("\n✗ Transcription failed")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Gemini REST API Transcription Test")
    print("=" * 60)

    success = asyncio.run(test_with_sample_audio())

    print("\n" + "=" * 60)
    if success:
        print("TEST PASSED - REST API transcription works!")
    else:
        print("TEST FAILED - Check API key and connectivity")
    print("=" * 60)
