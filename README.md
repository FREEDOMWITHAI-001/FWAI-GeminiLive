# WhatsApp Voice Calling with Gemini Live

AI Voice Agent for WhatsApp Business Voice Calls using Google Gemini Live.

**Now with multi-provider support: WhatsApp, Plivo, and Exotel**

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌─────────────┐                                                            │
│  │  WhatsApp   │───┐                                                        │
│  └─────────────┘   │                                                        │
│                    │     ┌──────────────┐      ┌─────────────────────┐      │
│  ┌─────────────┐   │     │              │      │                     │      │
│  │   Plivo     │───┼────►│   main.py    │◄────►│ gemini-live-service │      │
│  └─────────────┘   │     │  (port 3000) │      │    (port 8003)      │      │
│                    │     │              │      │                     │      │
│  ┌─────────────┐   │     └──────┬───────┘      └──────────┬──────────┘      │
│  │   Exotel    │───┘            │                         │                 │
│  └─────────────┘                │                         │                 │
│                           ┌─────┴─────┐             ┌─────┴─────┐           │
│                           │  adapters │             │  Pipecat  │           │
│                           │  + aiortc │             │  Gemini   │           │
│                           └───────────┘             │   Live    │           │
│                                                     └───────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Provider Comparison

| Feature | WhatsApp | Plivo | Exotel |
|---------|----------|-------|--------|
| Outbound Calls | ✅ | ✅ | ✅ |
| Incoming Calls | ✅ | ✅ | ✅ |
| WebRTC | ✅ | ❌ (PSTN) | ❌ (PSTN) |
| India Numbers | Via Business | Requires Compliance | Easy |
| TTS on Call | ❌ | ✅ | Via ExoML |
| Recording | ❌ | ✅ | ✅ (auto) |
| SMS | ✅ | ❌ | ✅ |

## Flow

1. **Make Call**: `POST /make-call` → Provider rings user
2. **User Answers**: Connection established (WebRTC or PSTN)
3. **Agent Connects**: `main.py` connects to `gemini-live-service.py` via WebSocket
4. **Greeting**: Gemini Live speaks first with greeting prompt
5. **Conversation**: Two-way audio flows:
   - User speaks → Audio captured → WebSocket → Gemini Live
   - Gemini responds → WebSocket → Audio output → User hears

## Quick Start

### 1. Install Dependencies

```bash
cd /home/kiran/FWAI_WebRTC_Gemini/FWAI_WebRTC_Gemini
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env`:
```env
# Choose provider: whatsapp, plivo, or exotel
CALL_PROVIDER=whatsapp

# WhatsApp Config
PHONE_NUMBER_ID=your_whatsapp_phone_number_id
META_ACCESS_TOKEN=your_meta_access_token
META_VERIFY_TOKEN=your_verify_token

# Plivo Config (if using Plivo)
PLIVO_AUTH_ID=your_plivo_auth_id
PLIVO_AUTH_TOKEN=your_plivo_auth_token
PLIVO_PHONE_NUMBER=+1234567890
PLIVO_CALLBACK_URL=https://your-server.com

# Exotel Config (if using Exotel)
EXOTEL_API_KEY=your_exotel_api_key
EXOTEL_API_TOKEN=your_exotel_api_token
EXOTEL_SID=your_exotel_sid
EXOTEL_CALLER_ID=your_exotel_virtual_number

# Google Gemini
GOOGLE_API_KEY=your_google_api_key
GEMINI_LIVE_PORT=8003
```

### 3. Start Services

**Terminal 1 - Gemini Live Service (port 8003):**
```bash
python gemini-live-service.py
```

**Terminal 2 - Main Server (port 3000):**
```bash
python main.py
```

### 4. Expose with ngrok

```bash
ngrok http 3000
```

Configure webhooks in respective provider console:
- **WhatsApp**: `https://your-ngrok-url/webhook` and `/call-events`
- **Plivo**: `https://your-ngrok-url/plivo/answer` and `/plivo/hangup`
- **Exotel**: `https://your-ngrok-url/exotel/status`

### 5. Make a Call

```bash
curl -X POST http://localhost:3000/make-call \
  -H "Content-Type: application/json" \
  -d '{"phoneNumber": "919052034075", "contactName": "Test Customer"}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/make-call` | POST | Make outbound call |
| `/webhook` | GET/POST | WhatsApp message webhook |
| `/call-events` | GET/POST | WhatsApp call events webhook |
| `/plivo/answer` | POST | Plivo answer webhook |
| `/plivo/hangup` | POST | Plivo hangup webhook |
| `/exotel/status` | POST | Exotel status callback |
| `/calls` | GET | List active calls |
| `/calls/{id}/terminate` | POST | End a call |
| `/` | GET | Health check |

## Project Structure

```
FWAI_WebRTC_Gemini/
├── main.py                  # FastAPI server (port 3000)
├── gemini-live-service.py   # Gemini Live service (port 8003)
├── webrtc_handler.py        # WebRTC with aiortc
├── gemini_agent.py          # WebSocket client to Gemini Live
├── audio_processor.py       # Audio resampling
├── config.py                # Configuration (multi-provider)
│
├── adapters/                # NEW: Provider adapters
│   ├── __init__.py          # Exports all adapters
│   ├── base.py              # Abstract base class
│   ├── whatsapp_adapter.py  # WhatsApp Business API
│   ├── plivo_adapter.py     # Plivo Voice API
│   └── exotel_adapter.py    # Exotel API
│
├── whatsapp_client.py       # Legacy WhatsApp client
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
├── .env.example             # Example configuration
└── FAWI_Call_BOT.txt        # Conversation script
```

## Adapter Usage

```python
from adapters import WhatsAppAdapter, PlivoAdapter, ExotelAdapter

# WhatsApp
wa = WhatsAppAdapter()
await wa.make_call(phone_number="919052034075", sdp_offer="...")

# Plivo
plivo = PlivoAdapter()
await plivo.make_call(phone_number="+919052034075")
await plivo.speak_text(call_id, "Hello from AI")

# Exotel
exotel = ExotelAdapter()
await exotel.make_call(phone_number="919052034075")
await exotel.connect_two_numbers(from_number="agent", to_number="customer")
```

## Why This Architecture?

The Node.js `@roamhq/wrtc` library couldn't extract audio from WebRTC tracks. Python's `aiortc` provides full audio access, solving the audio bridge problem.

**Multi-provider adapters** allow:
- Easy switching between providers via `CALL_PROVIDER` env var
- Consistent interface across all providers
- Provider-specific features (TTS, recording, etc.)
- Fallback options for different regions
