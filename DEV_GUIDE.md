# Development Guide

Quick reference for developers working with this repository.

## Project Overview

**FWAI Voice AI Agent** - Real-time voice AI using Plivo (telephony) + Google Gemini 2.0 Live (AI + native TTS).

Key features:
- Session preloading for ultra-low latency (~100ms first audio)
- Tool calling during live calls (WhatsApp, SMS, callbacks)
- Bidirectional audio streaming via WebSocket
- Automatic call transcription

## Quick Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run server
python run.py

# Expose via ngrok (required for Plivo callbacks)
ngrok http 3001

# Make a test call
curl -X POST http://localhost:3001/plivo/make-call \
  -H "Content-Type: application/json" \
  -d '{"phoneNumber": "+919876543210", "contactName": "Test"}'
```

## Project Structure

```
FWAI_WebRTC_Gemini/
├── run.py                          # Entry point
├── prompts.json                    # AI agent prompts
├── .env                            # Configuration
├── src/
│   ├── app.py                      # FastAPI server (all endpoints)
│   ├── services/
│   │   └── plivo_gemini_stream.py  # MAIN: Plivo ↔ Gemini bridge
│   ├── tools/
│   │   ├── __init__.py             # Tool registry
│   │   ├── send_whatsapp.py        # WhatsApp tool
│   │   ├── send_sms.py             # SMS tool
│   │   └── schedule_callback.py    # Callback tool
│   └── adapters/
│       └── plivo_adapter.py        # Plivo API client
├── transcripts/                    # Call transcripts
└── logs/                           # Application logs
```

## Key Files to Know

| File | Purpose |
|------|---------|
| `src/app.py` | FastAPI server with all Plivo endpoints |
| `src/services/plivo_gemini_stream.py` | Core bridge - preloading, audio streaming, tools |
| `prompts.json` | System prompts for AI agent |
| `src/tools/__init__.py` | Tool registry and executor |

## API Endpoints

### For External Integration (n8n, Zapier, etc.)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /plivo/make-call` | POST | Initiate outbound call |
| `GET /calls` | GET | List active calls |
| `POST /calls/{id}/terminate` | POST | End a call |

### Plivo Webhooks (configured in Plivo console)

| Endpoint | Purpose |
|----------|---------|
| `/plivo/answer` | Returns Stream XML when call connects |
| `/plivo/stream/{uuid}` | WebSocket for bidirectional audio |
| `/plivo/hangup` | Call ended callback |

## Architecture

```
POST /plivo/make-call
    │
    ├──► Plivo API initiates call
    │
    └──► PRELOAD Gemini session (while phone rings)
         │
         └──► Generate greeting audio (stored)

User answers
    │
    └──► /plivo/answer → <Stream> XML → WebSocket connects
         │
         └──► Send preloaded audio INSTANTLY
              │
              └──► Bidirectional conversation begins
```

## Audio Flow

- **User → Gemini**: PCM L16 16kHz via Plivo WebSocket
- **Gemini → User**: PCM L16 24kHz via Plivo WebSocket

Buffer size: 320 bytes (20ms chunks) for low latency.

## Tool Calling

Tools are declared in `plivo_gemini_stream.py`:

```python
TOOL_DECLARATIONS = [
    {"name": "send_whatsapp", ...},
    {"name": "send_sms", ...},
    {"name": "schedule_callback", ...}
]
```

Execution: `src/tools/__init__.py` → `execute_tool(name, phone, **args)`

## Environment Variables

Required:
- `PLIVO_AUTH_ID`, `PLIVO_AUTH_TOKEN`, `PLIVO_FROM_NUMBER`
- `PLIVO_CALLBACK_URL` (ngrok URL)
- `GOOGLE_API_KEY`

For WhatsApp tool:
- `META_ACCESS_TOKEN`, `WHATSAPP_PHONE_ID`

## Logs

- Application: `logs/whatsapp_voice.log`
- Transcripts: `transcripts/{call_uuid}.txt`

## Common Tasks

### Change AI Voice
Edit `plivo_gemini_stream.py`:
```python
"voice_name": "Charon"  # Options: Charon, Kore, Puck, etc.
```

### Change AI Prompt
Edit `prompts.json`:
```json
{
  "FWAI_Core": {
    "prompt": "Your system prompt here..."
  }
}
```

### Add New Tool
1. Create `src/tools/new_tool.py` (extend BaseTool)
2. Add to `TOOL_DECLARATIONS` in `plivo_gemini_stream.py`

### Adjust Latency
Edit `plivo_gemini_stream.py`:
```python
BUFFER_SIZE = 320  # Lower = less latency, more CPU
```

## Troubleshooting

| Issue | Check |
|-------|-------|
| No audio | Verify ngrok URL in `.env`, check WebSocket logs |
| High latency | Check for "PRELOAD COMPLETE" in logs |
| WhatsApp not sending | Verify `META_ACCESS_TOKEN` and `WHATSAPP_PHONE_ID` |
| Call not connecting | Check Plivo credits and webhook configuration |

## Session Continuation Notes

After restarting work:

1. Check if ngrok is running: `curl http://localhost:4040/api/tunnels`
2. If ngrok URL changed, update `PLIVO_CALLBACK_URL` in `.env`
3. Restart server: `python run.py`
4. Check logs: `tail -f logs/whatsapp_voice.log`

Current working config:
- Model: `gemini-2.0-flash-exp`
- Voice: `Charon` (Male)
- Audio: L16 16kHz input, L16 24kHz output
- Server port: 3001
