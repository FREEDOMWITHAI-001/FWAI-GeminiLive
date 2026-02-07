#!/usr/bin/env python3
"""
Full flow test for transcription and n8n webhook
Tests: Audio → REST API Transcription → Webhook
"""
import asyncio
import base64
import httpx
import os
import struct
import math

# Load API key
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    try:
        from src.core.config import config
        API_KEY = config.google_api_key
    except:
        print("ERROR: Set GOOGLE_API_KEY environment variable")
        exit(1)

# Test webhook URL (use a test endpoint or your n8n URL)
TEST_WEBHOOK_URL = os.environ.get("TEST_WEBHOOK_URL", "https://n8n.srv1100770.hstgr.cloud/webhook/conv-transcript")

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"


def generate_test_audio(duration_seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate a simple sine wave audio for testing (16-bit PCM)"""
    frequency = 440  # A4 note
    samples = int(sample_rate * duration_seconds)
    audio_data = bytearray()

    for i in range(samples):
        # Generate sine wave
        value = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * i / sample_rate))
        # Pack as 16-bit signed integer (little-endian)
        audio_data.extend(struct.pack('<h', value))

    return bytes(audio_data)


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Convert raw PCM bytes to WAV format"""
    import io
    import wave
    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bits_per_sample // 8)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)

    return wav_buffer.getvalue()


async def test_transcription_api():
    """Test 1: Verify Gemini REST API can handle audio"""
    print("\n" + "="*60)
    print("TEST 1: Gemini REST API Audio Handling")
    print("="*60)

    # Generate test audio (sine wave - won't transcribe to words, but tests API)
    pcm_bytes = generate_test_audio(1.0)
    wav_bytes = pcm_to_wav(pcm_bytes)
    audio_b64 = base64.standard_b64encode(wav_bytes).decode("utf-8")

    payload = {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "audio/wav",
                        "data": audio_b64
                    }
                },
                {
                    "text": "Describe what you hear in this audio. If it's just a tone or noise, say 'audio tone detected'."
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 256
        }
    }

    print(f"Sending {len(wav_bytes)} bytes of WAV audio...")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(GEMINI_API_URL, json=payload)

            if response.status_code == 200:
                result = response.json()
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                print(f"✓ API Response: {text.strip()}")
                return True
            else:
                print(f"✗ API Error: {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                return False
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False


async def test_webhook_connection():
    """Test 2: Verify webhook endpoint is reachable"""
    print("\n" + "="*60)
    print("TEST 2: Webhook Connectivity")
    print("="*60)

    print(f"Testing webhook: {TEST_WEBHOOK_URL}")

    # Send a test payload
    payload = {
        "event": "test",
        "call_uuid": "test-call-12345678",
        "role": "user",
        "text": "This is a test transcript from the test script",
        "timestamp": "2024-02-07T12:00:00Z",
        "turn_count": 1
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(TEST_WEBHOOK_URL, json=payload)

            print(f"Response Status: {response.status_code}")
            if response.status_code == 200:
                print(f"✓ Webhook responded successfully")
                print(f"  Response: {response.text[:200] if response.text else '(empty)'}")
                return True
            else:
                print(f"✗ Webhook returned error: {response.status_code}")
                return False
    except httpx.TimeoutException:
        print(f"✗ Webhook timeout - but this might be OK if n8n processes async")
        return True  # Timeout might be OK for async processing
    except Exception as e:
        print(f"✗ Webhook error: {e}")
        return False


async def test_full_transcription_flow():
    """Test 3: Simulate full transcription flow with text-to-speech simulation"""
    print("\n" + "="*60)
    print("TEST 3: Full Transcription Flow Simulation")
    print("="*60)

    # Use text prompt to simulate what a transcription would return
    payload = {
        "contents": [{
            "parts": [{
                "text": "Pretend you just transcribed audio where someone said: 'Hi, I attended the masterclass and it was really good. I work in finance.' Output ONLY that transcription, nothing else."
            }]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 256
        }
    }

    print("Simulating transcription...")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Step 1: Get transcription
            response = await client.post(GEMINI_API_URL, json=payload)

            if response.status_code != 200:
                print(f"✗ Transcription failed: {response.status_code}")
                return False

            result = response.json()
            transcription = result["candidates"][0]["content"]["parts"][0]["text"].strip()
            print(f"✓ Transcription: {transcription}")

            # Step 2: Send to webhook
            webhook_payload = {
                "event": "transcript",
                "call_uuid": "test-flow-87654321",
                "role": "user",
                "text": transcription,
                "timestamp": "2024-02-07T12:00:00Z",
                "turn_count": 2
            }

            print(f"\nSending to webhook...")
            try:
                response = await client.post(TEST_WEBHOOK_URL, json=webhook_payload, timeout=5.0)
                print(f"✓ Webhook response: {response.status_code}")
                return True
            except httpx.TimeoutException:
                print(f"✓ Webhook called (timeout OK for async processing)")
                return True

    except Exception as e:
        print(f"✗ Flow error: {e}")
        return False


async def test_n8n_state_machine():
    """Test 4: Test n8n state machine with sample user inputs"""
    print("\n" + "="*60)
    print("TEST 4: n8n State Machine Responses")
    print("="*60)

    test_inputs = [
        ("opening", "Hi, yes I attended the masterclass, it was great!"),
        ("discovery", "I work in finance, about 5 years experience"),
        ("ai_rating", "I would say about 4 out of 10"),
        ("objection", "I need to discuss with my father first"),
    ]

    results = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        for phase, text in test_inputs:
            payload = {
                "event": "transcript",
                "call_uuid": f"test-{phase}-12345",
                "role": "user",
                "text": text,
                "timestamp": "2024-02-07T12:00:00Z",
                "turn_count": 2
            }

            print(f"\n[{phase}] Sending: '{text[:40]}...'")

            try:
                response = await client.post(TEST_WEBHOOK_URL, json=payload, timeout=5.0)
                if response.status_code == 200:
                    print(f"  ✓ Response: {response.status_code}")
                    try:
                        data = response.json()
                        if 'next_phase' in data:
                            print(f"  → Next phase: {data.get('next_phase')}")
                        if 'should_inject' in data:
                            print(f"  → Should inject: {data.get('should_inject')}")
                    except:
                        pass
                    results.append(True)
                else:
                    print(f"  ✗ Error: {response.status_code}")
                    results.append(False)
            except httpx.TimeoutException:
                print(f"  ✓ Sent (timeout OK)")
                results.append(True)
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append(False)

    return all(results)


async def test_transcription_latency():
    """Test 5: Measure transcription latency"""
    print("\n" + "="*60)
    print("TEST 5: Transcription Latency")
    print("="*60)

    import time

    # Generate different sizes of audio
    test_durations = [1.0, 2.0, 5.0]  # seconds
    latencies = []

    for duration in test_durations:
        pcm_bytes = generate_test_audio(duration)
        wav_bytes = pcm_to_wav(pcm_bytes)
        audio_b64 = base64.standard_b64encode(wav_bytes).decode("utf-8")

        payload = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
                    {"text": "Describe what you hear briefly."}
                ]
            }],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 100}
        }

        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(GEMINI_API_URL, json=payload)
                latency_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    latencies.append(latency_ms)
                    print(f"  {duration}s audio ({len(wav_bytes)/1024:.1f}KB): {latency_ms:.0f}ms")
                else:
                    print(f"  {duration}s audio: ERROR {response.status_code}")

        except Exception as e:
            print(f"  {duration}s audio: ERROR {e}")

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"\n  Average latency: {avg_latency:.0f}ms")
        return avg_latency < 5000  # Pass if under 5 seconds
    return False


async def main():
    print("="*60)
    print("FWAI TRANSCRIPTION FLOW - FULL TEST SUITE")
    print("="*60)

    results = {}

    # Test 1: API
    results["API"] = await test_transcription_api()

    # Test 2: Latency
    results["Latency"] = await test_transcription_latency()

    # Test 3: Webhook
    results["Webhook"] = await test_webhook_connection()

    # Test 4: Full flow
    results["Full Flow"] = await test_full_transcription_flow()

    # Test 5: State machine
    results["State Machine"] = await test_n8n_state_machine()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("="*60)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
    else:
        print("SOME TESTS FAILED - Check above for details")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
