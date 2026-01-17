# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python-based WhatsApp Business Voice Calling with Gemini Live AI agent. Uses aiortc for WebRTC with full audio access, solving the audio bridge limitation in the Node.js implementation.

## Commands

```bash
# Create virtual environment and install
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt

# Run server
python main.py

# Or use startup scripts
./start.sh    # Linux/Mac
start.bat     # Windows
```

## Architecture

```
WhatsApp Call ←→ aiortc WebRTC ←→ AudioProcessor ←→ Gemini Live (Pipecat)
```

### Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI server with `/make-call`, `/webhook`, `/call-events` endpoints |
| `webrtc_handler.py` | WebRTC handling with aiortc, CallSession management |
| `gemini_agent.py` | Gemini Live voice agent using Pipecat pipeline |
| `audio_processor.py` | Audio conversion between WebRTC (48kHz) and Gemini (16kHz) |
| `whatsapp_client.py` | WhatsApp Business API client |
| `config.py` | Configuration and conversation script loading |

### Audio Flow

1. **User → Agent**: `WebRTC AudioFrame` → `AudioProcessor.process_webrtc_frame()` → PCM bytes → `GeminiVoiceAgent.feed_audio()`
2. **Agent → User**: Gemini TTSAudioRawFrame → `AudioOutputTrack.feed_audio()` → `WebRTC AudioFrame`

### Key Classes

- `CallSession` - Manages WebRTC peer connection and audio processing for a call
- `GeminiVoiceAgent` - Pipecat pipeline for Gemini Live with audio input/output processors
- `AudioOutputTrack` - Custom aiortc track that outputs Gemini audio to WebRTC

## Environment Variables

```env
PHONE_NUMBER_ID=<WhatsApp Business Manager>
META_ACCESS_TOKEN=<Meta Developer Console>
META_VERIFY_TOKEN=<webhook verification>
GOOGLE_API_KEY=<Google Cloud>
TTS_VOICE=Kore
```
