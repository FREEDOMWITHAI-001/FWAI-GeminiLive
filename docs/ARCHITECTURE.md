# FWAI Voice Agent - Technical Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              FWAI Voice AI System                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌──────────────┐                    ┌──────────────────┐                           │
│  │   n8n        │  Webhook Trigger   │                  │                           │
│  │   Workflow   │───────────────────►│   FastAPI        │                           │
│  │              │◄───────────────────│   Server         │                           │
│  └──────────────┘  Call Ended Hook   │   (port 3001)    │                           │
│                                      └────────┬─────────┘                           │
│                                               │                                      │
│  ┌──────────────┐      PSTN       ┌──────────┴───────┐                              │
│  │              │◄───────────────►│                  │                              │
│  │  User Phone  │                 │   Plivo Cloud    │                              │
│  │              │                 │   - Stream API   │                              │
│  └──────────────┘                 │   - Call Control │                              │
│                                   └────────┬─────────┘                              │
│                                            │                                         │
│                                   WebSocket│Bidirectional                           │
│                                   (L16 16k)│                                         │
│                                            ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                         FastAPI Server (port 3001)                           │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │    │
│  │  │  /plivo/        │  │  Session        │  │  Tool Executor              │  │    │
│  │  │  make-call      │  │  Manager        │  │  - send_whatsapp            │  │    │
│  │  │  answer         │  │  - preload      │  │  - schedule_callback        │  │    │
│  │  │  stream/{uuid}  │  │  - attach       │  │  - end_call                 │  │    │
│  │  │  hangup         │  │  - cleanup      │  │                             │  │    │
│  │  └─────────────────┘  └────────┬────────┘  └─────────────────────────────┘  │    │
│  │                                │                                             │    │
│  └────────────────────────────────┼─────────────────────────────────────────────┘    │
│                                   │                                                  │
│                          WebSocket│(L16 16k in / 24k out)                           │
│                                   ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                    Google Gemini 2.5 Live API                               │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────────────┐   │    │
│  │  │ BidiGenerate  │  │  Native TTS   │  │  Function Calling             │   │    │
│  │  │ Content       │  │  (Puck)       │  │  - Tool declarations          │   │    │
│  │  │               │  │               │  │  - Tool responses             │   │    │
│  │  └───────────────┘  └───────────────┘  └───────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## n8n Integration Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ External        │     │     n8n          │     │   FWAI Server   │
│ Trigger         │     │   Workflow       │     │                 │
└────────┬────────┘     └────────┬─────────┘     └────────┬────────┘
         │                       │                        │
         │  POST /trigger-call   │                        │
         │──────────────────────►│                        │
         │                       │  POST /plivo/make-call │
         │                       │───────────────────────►│
         │                       │                        │
         │                       │    {success, call_uuid}│
         │                       │◄───────────────────────│
         │   {success, call_uuid}│                        │
         │◄──────────────────────│                        │
         │                       │                        │
         │                       │    ... call happens ...│
         │                       │                        │
         │                       │  POST /call-ended      │
         │                       │◄───────────────────────│
         │                       │  {transcript, duration}│
         │                       │                        │
         │                       │  Process call data     │
         │                       │  (CRM, email, etc.)    │
         │                       │                        │
```

## Session Preloading Architecture

The key innovation is **preloading** the Gemini session while the phone rings:

```
Timeline:
─────────────────────────────────────────────────────────────────────────────────────►

T+0ms      POST /plivo/make-call received
           │
           ├──► PRELOAD: Start Gemini session immediately
           │    │
           │    ├──► Connect to wss://generativelanguage.googleapis.com
           │    ├──► Send setup (model, voice, tools, prompt)
           │    ├──► Wait for setupComplete
           │    └──► Send greeting trigger ("Hi")
           │         │
           │         └──► Gemini generates greeting audio
           │              (stored in preloaded_audio[])
           │
           └──► Plivo API: Initiate call to user
                │
                └──► Set plivo_call_uuid on session (for hangup)

T+3000ms   Phone starts ringing
           │
           └──► Greeting audio continues generating...

T+6000ms   User answers phone
           │
           ├──► Plivo hits /plivo/answer
           ├──► Server returns <Stream> XML
           └──► Plivo connects WebSocket to /plivo/stream/{uuid}

T+6100ms   WebSocket "start" event received
           │
           ├──► attach_plivo_ws() called
           ├──► Session moved from _preloading_sessions to _sessions
           └──► INSTANTLY send preloaded_audio[] to Plivo
                │
                └──► User hears greeting immediately!

T+6200ms   Bidirectional conversation begins
           User speaks ──► Gemini ──► User hears response
```

### Session States

```python
# Two dictionaries manage session lifecycle
_preloading_sessions: Dict[str, PlivoGeminiSession] = {}  # While phone rings
_sessions: Dict[str, PlivoGeminiSession] = {}              # Active calls

# State transitions:
# 1. preload_session() → adds to _preloading_sessions
# 2. set_plivo_uuid() → sets Plivo's call UUID for hangup
# 3. create_session() → moves to _sessions, attaches Plivo WS
# 4. remove_session() → cleanup from both dicts
```

## Call Hangup Flow

When the AI decides to end the call (user says goodbye):

```
User: "Okay, bye!"
    │
    ▼
Gemini recognizes goodbye, calls end_call tool
    │
    ▼
_handle_tool_call() processes end_call
    │
    ├── Send success response to Gemini
    ├── Log: "Call ending: user said goodbye"
    └── Schedule _hangup_call_delayed(3.0)
        │
        ▼
    Wait 3 seconds (let goodbye audio play)
        │
        ▼
    DELETE https://api.plivo.com/v1/Account/{id}/Call/{plivo_uuid}/
        │
        ├── Uses plivo_call_uuid (NOT internal UUID)
        └── Call disconnects
            │
            ▼
        session.stop()
            │
            ├── Save recording
            ├── Transcribe with Whisper
            ├── Call webhook (n8n)
            └── Cleanup
```

## Transcript Generation

### Recording Flow

```
During call:
    │
    ├── USER audio chunks → _record_audio("USER", bytes, 16000)
    │                           │
    │                           └── Added to audio_chunks[]
    │
    └── AI audio chunks → _record_audio("AI", bytes, 24000)
                              │
                              └── Resampled to 16kHz, added to audio_chunks[]

Call ends:
    │
    ▼
_save_recording()
    │
    ├── Combine all audio_chunks into single WAV (16kHz mono)
    └── Save to recordings/{call_uuid}.wav
        │
        ▼
_transcribe_recording()
    │
    ├── Load Whisper model (base)
    ├── Transcribe recording
    ├── Parse sentences and assign AGENT/USER labels
    └── Append to transcripts/{call_uuid}.txt:
        │
        └── --- CONVERSATION TRANSCRIPT ---
            AGENT: Hi, this is Vishnu from Freedom with AI.
            USER: Hi, I'm good.
            AGENT: Great! What sparked your interest?
            ...
```

### Transcript Format

```
[HH:MM:SS] SYSTEM: AI ready
[HH:MM:SS] SYSTEM: Call connected (preloaded)
[HH:MM:SS] TOOL: send_whatsapp: {'template_id': 'course_details'}
[HH:MM:SS] TOOL_RESULT: send_whatsapp: success
[HH:MM:SS] TOOL: end_call: {'reason': 'user said goodbye'}
[HH:MM:SS] SYSTEM: Call ending: user said goodbye
[HH:MM:SS] SYSTEM: Call duration: 125.3s
[HH:MM:SS] SYSTEM: Call ended

--- CONVERSATION TRANSCRIPT ---
AGENT: Hi, this is Vishnu from Freedom with AI. How are you doing today?
USER: I'm good, thanks.
AGENT: Great! I noticed you attended our AI Masterclass. How did you find it?
USER: It was very informative.
...
```

## Audio Flow

### Inbound (User → Gemini)

```
User speaks
    │
    ▼
Plivo captures audio (L16 16kHz)
    │
    ▼
WebSocket "media" event with base64 payload
    │
    ▼
handle_plivo_audio()
    │
    ├── Decode base64 to bytes
    ├── Record for transcription: _record_audio("USER", bytes)
    ├── Append to inbuffer
    └── When inbuffer >= BUFFER_SIZE (320 bytes):
        │
        ▼
    Send to Gemini:
    {
      "realtime_input": {
        "media_chunks": [{
          "mime_type": "audio/pcm;rate=16000",
          "data": "<base64>"
        }]
      }
    }
```

### Outbound (Gemini → User)

```
Gemini generates response
    │
    ▼
WebSocket receives serverContent.modelTurn.parts[].inlineData
    │
    ▼
_receive_from_google()
    │
    ├── Extract audio data (base64, L16 24kHz)
    ├── Record for transcription: _record_audio("AI", bytes, 24000)
    │
    ▼
Send to Plivo WebSocket:
{
  "event": "playAudio",
  "media": {
    "contentType": "audio/x-l16",
    "sampleRate": 24000,
    "payload": "<base64>"
  }
}
    │
    ▼
Plivo plays audio to user
```

## Tool Calling Flow

```
User: "Send me the details on WhatsApp"
    │
    ▼
Gemini recognizes intent, returns toolCall:
{
  "toolCall": {
    "functionCalls": [{
      "id": "call_123",
      "name": "send_whatsapp",
      "args": {"message": "Course details..."}
    }]
  }
}
    │
    ▼
_handle_tool_call()
    │
    ├── Check for end_call tool (special handling)
    │   │
    │   └── If end_call: schedule hangup after 3s delay
    │
    ├── Execute: await execute_tool("send_whatsapp", caller_phone, **args)
    │   │
    │   └── SendWhatsAppTool.execute()
    │       │
    │       └── POST https://graph.facebook.com/v22.0/{phone_id}/messages
    │
    ├── Send tool response to Gemini:
    │   {
    │     "tool_response": {
    │       "function_responses": [{
    │         "id": "call_123",
    │         "name": "send_whatsapp",
    │         "response": {"success": true, "message": "Sent"}
    │       }]
    │     }
    │   }
    │
    ▼
Gemini generates verbal confirmation:
"Great, I've sent the details to your WhatsApp!"
```

## Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `send_whatsapp` | Send WhatsApp message via Meta Business API | `message: string` |
| `schedule_callback` | Record callback request | `preferred_time: string`, `notes?: string` |
| `end_call` | End the call gracefully | `reason: string` |

### Tool Declaration Format

```python
TOOL_DECLARATIONS = [
    {
        "name": "end_call",
        "description": "End the call when user says goodbye or conversation is complete",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Reason for ending the call"
                }
            },
            "required": ["reason"]
        }
    },
    {
        "name": "send_whatsapp",
        "description": "Send a WhatsApp message to the caller...",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message content to send"
                }
            },
            "required": ["message"]
        }
    }
]
```

## n8n Webhook Payload

When call ends, server POSTs to n8n webhook:

```json
{
  "event": "call_ended",
  "call_uuid": "abc123-xyz-456",
  "caller_phone": "+919876543210",
  "duration_seconds": 125.3,
  "timestamp": "2026-01-26T00:45:00.123456",
  "transcript": "[00:00:05] SYSTEM: Call connected\n...\n--- CONVERSATION TRANSCRIPT ---\nAGENT: Hi...\nUSER: Hello..."
}
```

## Key Configuration

### Gemini Session Setup

```python
{
    "setup": {
        "model": "models/gemini-2.5-flash-native-audio-preview-09-2025",
        "generation_config": {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": "Puck"
                    }
                }
            }
        },
        "system_instruction": {
            "parts": [{"text": FWAI_PROMPT}]
        },
        "tools": [{
            "function_declarations": TOOL_DECLARATIONS
        }]
    }
}
```

### Buffer Configuration

```python
BUFFER_SIZE = 320  # 20ms chunks for ultra-low latency
                   # 320 bytes = 160 samples @ 16kHz = 10ms
                   # Smaller = lower latency, more CPU
                   # Larger = higher latency, smoother audio
```

## File Reference

| File | Purpose |
|------|---------|
| `src/app.py` | FastAPI server, all HTTP/WS endpoints, UUID mapping |
| `src/services/plivo_gemini_stream.py` | Core Plivo↔Gemini bridge with preloading, hangup, transcription |
| `src/tools/__init__.py` | Tool registry and executor |
| `src/tools/send_whatsapp.py` | WhatsApp tool implementation |
| `src/adapters/plivo_adapter.py` | Plivo API client |
| `src/core/config.py` | Environment configuration |
| `prompts.json` | AI agent system prompts |
| `n8n_flows/FWAI_Internal/outbound_call.json` | n8n workflow |

## API Reference

### Primary Endpoints

| Endpoint | Method | Description | Use From |
|----------|--------|-------------|----------|
| `/plivo/make-call` | POST | Initiate outbound call | n8n, Zapier, custom apps |
| `/calls` | GET | List all active calls | Monitoring dashboards |
| `/calls/{call_id}/terminate` | POST | End a specific call | Admin panels |
| `/` | GET | Health check | Load balancers |

### Plivo Webhook Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/plivo/answer` | POST | Called when user answers (returns Stream XML) |
| `/plivo/stream/{call_uuid}` | WebSocket | Bidirectional audio stream |
| `/plivo/stream-status` | POST | Stream status callbacks |
| `/plivo/hangup` | POST | Called when call ends |

### cURL Examples

```bash
# Health Check
curl -X GET http://localhost:3001/

# Make Call (Direct)
curl -X POST http://localhost:3001/plivo/make-call \
  -H "Content-Type: application/json" \
  -d '{"phoneNumber": "919876543210", "contactName": "John"}'

# Make Call (via n8n)
curl -X POST https://your-n8n-url/webhook/trigger-call \
  -H "Content-Type: application/json" \
  -d '{"phoneNumber": "919876543210", "contactName": "John"}'

# List Calls
curl -X GET http://localhost:3001/calls

# Terminate Call
curl -X POST http://localhost:3001/calls/{call_uuid}/terminate
```

## Environment Variables

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
| `TTS_VOICE` | No | Gemini voice name (default: Puck) |

## Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| First audio (with preload) | <500ms | ~100ms |
| First audio (no preload) | <3000ms | ~2500ms |
| Audio buffer latency | <50ms | 20ms |
| Tool execution | <2000ms | ~1000ms |
| Call hangup after goodbye | <5s | ~3s |

## Troubleshooting

### No audio after connection
- Check Plivo Stream URL matches ngrok URL
- Verify WebSocket connection in logs
- Ensure `GOOGLE_API_KEY` is valid

### Call not hanging up
- Check logs for "Plivo hangup response" status
- Verify `plivo_call_uuid` is set (look for "Set Plivo UUID" in logs)
- Ensure Plivo credentials are correct

### Transcript missing USER/AGENT labels
- Verify Whisper is installed: `pip install openai-whisper`
- Check recordings are being saved (look in `recordings/` folder)
- Check logs for "Transcription complete"

### n8n webhook not receiving data
- Ensure n8n workflow is **activated** (not in test mode)
- Verify webhook URL in n8n workflow matches your n8n instance
- Check logs for "Webhook response" status

### High latency
- Reduce `BUFFER_SIZE` in `plivo_gemini_stream.py`
- Check network latency to Google API
- Verify preloading is working (check logs for "PRELOAD COMPLETE")

### WhatsApp messages not sending
- Verify `META_ACCESS_TOKEN` is valid
- Check `WHATSAPP_PHONE_ID` is correct
- Ensure phone number has country code
