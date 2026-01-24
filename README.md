# FWAI Voice AI Agent

Real-time Voice AI Agent using **Plivo for telephony** and **Google Gemini 2.0 Live** for conversational AI with native TTS.

## Features

- **Ultra-low latency** - Session preloading while phone rings (~30 audio chunks ready)
- **Native AI Voice** - Gemini 2.0 Live with Charon voice (no external TTS)
- **Tool Calling** - WhatsApp, SMS, callbacks executed during live calls
- **Transcripts** - Automatic call transcription with timestamps
- **Bidirectional Audio** - Real-time conversation via WebSocket streaming

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────────┐
│                 │  PSTN   │                  │   WS    │                     │
│  User's Phone   │◄───────►│   Plivo Cloud    │◄───────►│   FastAPI Server    │
│                 │         │   (Stream API)   │         │   (port 3001)       │
└─────────────────┘         └──────────────────┘         └──────────┬──────────┘
                                                                    │
                                                           WebSocket│
                                                                    │
                                                         ┌──────────▼──────────┐
                                                         │                     │
                                                         │  Google Gemini 2.0  │
                                                         │  Live API           │
                                                         │  (BidiGenerateContent)
                                                         │                     │
                                                         └─────────────────────┘
```

### Call Flow

```
1. POST /plivo/make-call ──► Plivo API initiates call
                              │
2. Phone rings ◄──────────────┘
   │
   └──► Gemini session PRELOADS (greeting audio generated)
        │
3. User answers ──► Plivo hits /plivo/answer
                    │
                    └──► Returns <Stream> XML
                         │
4. WebSocket connects ◄──┘
   │
   └──► Preloaded audio sent INSTANTLY
        │
5. Bidirectional conversation begins
   User speaks ──► PCM 16kHz ──► Gemini
   Gemini responds ──► PCM 24kHz ──► User
```

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/kiranfwai/FWAI_WebRTC_Gemini.git
cd FWAI_WebRTC_Gemini

python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```env
# Plivo Configuration
PLIVO_AUTH_ID=your_plivo_auth_id
PLIVO_AUTH_TOKEN=your_plivo_auth_token
PLIVO_FROM_NUMBER=+912268093710
PLIVO_CALLBACK_URL=https://your-ngrok-url.ngrok-free.app

# Google Gemini
GOOGLE_API_KEY=your_google_api_key

# WhatsApp Business API (for sending messages)
META_ACCESS_TOKEN=your_meta_access_token
WHATSAPP_PHONE_ID=your_whatsapp_phone_number_id

# Server
HOST=0.0.0.0
PORT=3001
DEBUG=true

# Features
ENABLE_TRANSCRIPTS=true
TTS_VOICE=Charon
```

### 3. Expose Server with ngrok

```bash
ngrok http 3001
```

Copy the ngrok URL and update `PLIVO_CALLBACK_URL` in `.env`.

### 4. Configure Plivo

In [Plivo Console](https://console.plivo.com/):

1. Go to **Voice** → **Applications** → **Create Application**
2. Set **Answer URL**: `https://your-ngrok-url/plivo/answer` (POST)
3. Set **Hangup URL**: `https://your-ngrok-url/plivo/hangup` (POST)
4. Assign your Plivo number to this application

### 5. Start Server

```bash
python run.py
```

### 6. Make a Call

```bash
curl -X POST http://localhost:3001/plivo/make-call \
  -H "Content-Type: application/json" \
  -d '{"phoneNumber": "+919876543210", "contactName": "John Doe"}'
```

## API Endpoints

### Primary Endpoints (for external integration)

| Endpoint | Method | Description | Use From |
|----------|--------|-------------|----------|
| `/plivo/make-call` | POST | **Initiate outbound call** | n8n, Zapier, custom apps |
| `/calls` | GET | List all active calls | Monitoring dashboards |
| `/calls/{call_id}/terminate` | POST | End a specific call | Admin panels |
| `/` | GET | Health check | Load balancers |

### Plivo Webhook Endpoints (configured in Plivo console)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/plivo/answer` | POST | Called when user answers (returns Stream XML) |
| `/plivo/stream/{call_uuid}` | WebSocket | Bidirectional audio stream |
| `/plivo/stream-status` | POST | Stream status callbacks |
| `/plivo/hangup` | POST | Called when call ends |

---

## API Reference with cURL Examples

### 1. Health Check

Check if the server is running.

```bash
curl -X GET http://localhost:3001/
```

**Response:**
```json
{
  "status": "ok",
  "service": "WhatsApp Voice Calling with Gemini Live",
  "version": "1.0.0"
}
```

---

### 2. Make Outbound Call

Initiate an AI voice call to a phone number.

```bash
curl -X POST http://localhost:3001/plivo/make-call \
  -H "Content-Type: application/json" \
  -d '{
    "phoneNumber": "+919876543210",
    "contactName": "John Doe"
  }'
```

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `phoneNumber` | string | Yes | Phone number with country code (e.g., +919876543210) |
| `contactName` | string | No | Name of the contact (default: "Customer") |

**Response (Success):**
```json
{
  "success": true,
  "call_uuid": "a8237790-5aa4-4320-a882-bf3056d02bdb",
  "message": "Call initiated to +919876543210. Waiting for user to answer."
}
```

**Response (Error):**
```json
{
  "detail": "Invalid phone number format"
}
```

---

### 3. List Active Calls

Get all currently active calls.

```bash
curl -X GET http://localhost:3001/calls
```

**Response:**
```json
{
  "calls": [
    {
      "call_id": "a8237790-5aa4-4320-a882-bf3056d02bdb",
      "phone": "+919876543210",
      "status": "active",
      "started_at": "2026-01-25T00:00:00Z"
    }
  ]
}
```

---

### 4. Terminate a Call

End a specific active call.

```bash
curl -X POST http://localhost:3001/calls/a8237790-5aa4-4320-a882-bf3056d02bdb/terminate
```

**Response (Success):**
```json
{
  "success": true,
  "message": "Call terminated"
}
```

**Response (Not Found):**
```json
{
  "detail": "Call not found"
}
```

---

## Using with External Tools

### n8n Integration

1. **HTTP Request Node:**
   - Method: `POST`
   - URL: `https://your-ngrok-url/plivo/make-call`
   - Authentication: None (add API key if you implement it)
   - Body Content Type: JSON
   - Body:
     ```json
     {
       "phoneNumber": "{{ $json.phone }}",
       "contactName": "{{ $json.name }}"
     }
     ```

2. **Trigger Options:**
   - Webhook (incoming lead)
   - Schedule (follow-up calls)
   - CRM event (new signup)

### Zapier Integration

Use "Webhooks by Zapier" action:
- Method: POST
- URL: `https://your-ngrok-url/plivo/make-call`
- Data Pass-Through: No
- Data:
  ```
  phoneNumber: +91{{phone}}
  contactName: {{name}}
  ```

### Python Script

```python
import requests

response = requests.post(
    "http://localhost:3001/plivo/make-call",
    json={
        "phoneNumber": "+919876543210",
        "contactName": "John Doe"
    }
)
print(response.json())
```

### JavaScript/Node.js

```javascript
fetch('http://localhost:3001/plivo/make-call', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    phoneNumber: '+919876543210',
    contactName: 'John Doe'
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

---

## Postman Collection

Import the collection from `docs/FWAI_Voice_API.postman_collection.json` or use the examples above

## Project Structure

```
FWAI_WebRTC_Gemini/
├── run.py                          # Entry point
├── requirements.txt
├── .env                            # Environment configuration
├── prompts.json                    # AI agent prompts
│
├── src/
│   ├── app.py                      # FastAPI server (all endpoints)
│   ├── core/
│   │   ├── config.py               # Configuration management
│   │   └── audio_processor.py      # Audio format conversion
│   ├── services/
│   │   ├── plivo_gemini_stream.py  # Main: Plivo ↔ Gemini bridge
│   │   ├── gemini_live_tts.py      # TTS utilities
│   │   └── gemini_tools.py         # Tool definitions
│   ├── tools/
│   │   ├── __init__.py             # Tool registry
│   │   ├── base.py                 # Base tool class
│   │   ├── send_whatsapp.py        # WhatsApp messaging
│   │   ├── send_sms.py             # SMS messaging
│   │   └── schedule_callback.py    # Callback scheduling
│   └── adapters/
│       └── plivo_adapter.py        # Plivo API client
│
├── transcripts/                    # Call transcripts (auto-generated)
│   └── {call_uuid}.txt
│
├── logs/                           # Application logs
│   └── whatsapp_voice.log
│
└── docs/
    └── ARCHITECTURE.md
```

## Logs Location

| Log Type | Location | Description |
|----------|----------|-------------|
| Application Log | `logs/whatsapp_voice.log` | All server events, rotates at 10MB |
| Call Transcripts | `transcripts/{call_uuid}.txt` | Per-call conversation logs |
| Console Output | stdout | Real-time debug output |

### Transcript Format

```
[14:30:15] SYSTEM: AI ready
[14:30:16] VISHNU: Hello! This is Vishnu from Freedom with AI. How did you find our masterclass?
[14:30:25] USER: It was really helpful
[14:30:28] VISHNU: Great to hear! What aspect interested you the most?
[14:30:45] TOOL: send_whatsapp: {'message': 'Course details...'}
[14:30:46] TOOL_RESULT: send_whatsapp: success
[14:31:00] SYSTEM: Call ended
```

### View Logs

```bash
# Real-time application logs
tail -f logs/whatsapp_voice.log

# View specific call transcript
cat transcripts/a8237790-5aa4-4320-a882-bf3056d02bdb.txt

# Search for errors
grep "ERROR" logs/whatsapp_voice.log
```

## Available AI Tools

Tools that the AI can invoke during calls:

| Tool | Trigger Phrases | Action |
|------|-----------------|--------|
| `send_whatsapp` | "send details on WhatsApp" | Sends WhatsApp message via Meta API |
| `send_sms` | "text me", "send SMS" | Sends SMS via Plivo |
| `schedule_callback` | "call me later", "schedule callback" | Records callback request |

## Customizing the AI Agent

Edit `prompts.json`:

```json
{
  "FWAI_Core": {
    "name": "Your Agent Name",
    "description": "Agent description",
    "prompt": "You are [Agent Name]...\n\nSTYLE:\n- Keep responses brief\n- Ask one question at a time\n\nTOOLS:\n- Use send_whatsapp when user requests details\n\nFLOW:\n1. Greet the user\n2. Understand their needs\n3. Provide relevant information"
  }
}
```

## Audio Configuration

| Direction | Format | Sample Rate |
|-----------|--------|-------------|
| User → Server | PCM L16 | 16 kHz |
| Server → Gemini | PCM L16 | 16 kHz |
| Gemini → Server | PCM L16 | 24 kHz |
| Server → User | PCM L16 | 24 kHz |

## Performance Tuning

### Latency Optimization

The system uses **session preloading** to minimize first-response latency:

1. When `/plivo/make-call` is called, Gemini session starts immediately
2. Initial greeting audio is generated while the phone is ringing
3. When user answers, preloaded audio is sent instantly

Current buffer size: **320 bytes (20ms chunks)** for ultra-low latency.

### Preload Timeout

Default preload timeout is 8 seconds. Adjust in `plivo_gemini_stream.py`:

```python
await asyncio.wait_for(self._preload_complete.wait(), timeout=8.0)
```

## Troubleshooting

### No audio after connection

- Check Plivo Stream URL matches ngrok URL
- Verify WebSocket connection in logs
- Ensure `GOOGLE_API_KEY` is valid

### High latency

- Reduce `BUFFER_SIZE` in `plivo_gemini_stream.py`
- Check network latency to Google API
- Verify preloading is working (check logs for "PRELOAD COMPLETE")

### WhatsApp messages not sending

- Verify `META_ACCESS_TOKEN` is valid
- Check `WHATSAPP_PHONE_ID` is correct
- Ensure phone number has country code

### Call not connecting

- Verify ngrok is running and URL is updated in `.env`
- Check Plivo application webhooks are configured
- Ensure Plivo account has sufficient credits

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `PLIVO_AUTH_ID` | Yes | Plivo account auth ID |
| `PLIVO_AUTH_TOKEN` | Yes | Plivo account auth token |
| `PLIVO_FROM_NUMBER` | Yes | Your Plivo phone number |
| `PLIVO_CALLBACK_URL` | Yes | Public URL (ngrok) for callbacks |
| `GOOGLE_API_KEY` | Yes | Google AI API key |
| `META_ACCESS_TOKEN` | For WhatsApp | Meta WhatsApp Business token |
| `WHATSAPP_PHONE_ID` | For WhatsApp | WhatsApp phone number ID |
| `HOST` | No | Server host (default: 0.0.0.0) |
| `PORT` | No | Server port (default: 3001) |
| `DEBUG` | No | Enable debug logging (default: false) |
| `ENABLE_TRANSCRIPTS` | No | Save call transcripts (default: true) |
| `TTS_VOICE` | No | Gemini voice name (default: Charon) |

## License

MIT
