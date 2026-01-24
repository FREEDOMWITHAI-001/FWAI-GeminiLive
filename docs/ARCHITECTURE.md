# FWAI Voice Agent - Technical Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              FWAI Voice AI System                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌──────────────┐      PSTN       ┌──────────────────┐                              │
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
│  │  │  answer         │  │  - preload      │  │  - send_sms                 │  │    │
│  │  │  stream/{uuid}  │  │  - attach       │  │  - schedule_callback        │  │    │
│  │  │  hangup         │  │  - cleanup      │  │                             │  │    │
│  │  └─────────────────┘  └────────┬────────┘  └─────────────────────────────┘  │    │
│  │                                │                                             │    │
│  └────────────────────────────────┼─────────────────────────────────────────────┘    │
│                                   │                                                  │
│                          WebSocket│(L16 16k in / 24k out)                           │
│                                   ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                    Google Gemini 2.0 Live API                                │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────────────┐   │    │
│  │  │ BidiGenerate  │  │  Native TTS   │  │  Function Calling             │   │    │
│  │  │ Content       │  │  (Charon)     │  │  - Tool declarations          │   │    │
│  │  │               │  │               │  │  - Tool responses             │   │    │
│  │  └───────────────┘  └───────────────┘  └───────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Session Preloading Architecture

The key innovation is **preloading** the Gemini session while the phone rings:

```
Timeline:
─────────────────────────────────────────────────────────────────────────────────────►

T+0ms      POST /plivo/make-call received
           │
           ├──► Plivo API: Initiate call to user
           │
           └──► PRELOAD: Start Gemini session immediately
                │
                ├──► Connect to wss://generativelanguage.googleapis.com
                ├──► Send setup (model, voice, tools, prompt)
                ├──► Wait for setupComplete
                └──► Send greeting trigger ("Hi")
                     │
                     └──► Gemini generates greeting audio
                          (stored in preloaded_audio[])

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
# 2. create_session() → moves to _sessions, attaches Plivo WS
# 3. remove_session() → cleanup from both dicts
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
    ├── Extract tool name and args
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
| `send_sms` | Send SMS via Plivo | `message: string` |
| `schedule_callback` | Record callback request | `preferred_time: string`, `notes?: string` |

### Tool Declaration Format

```python
TOOL_DECLARATIONS = [
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

## Plivo XML Response

When user answers, `/plivo/answer` returns:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream
        streamTimeout="86400"
        keepCallAlive="true"
        bidirectional="true"
        contentType="audio/x-l16;rate=16000"
        statusCallbackUrl="https://server/plivo/stream-status">
        wss://server/plivo/stream/{call_uuid}
    </Stream>
</Response>
```

## Key Configuration

### Gemini Session Setup

```python
{
    "setup": {
        "model": "models/gemini-2.0-flash-exp",
        "generation_config": {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": "Charon"  # Male Indian English
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
| `src/app.py` | FastAPI server, all HTTP/WS endpoints |
| `src/services/plivo_gemini_stream.py` | Core Plivo↔Gemini bridge with preloading |
| `src/tools/__init__.py` | Tool registry and executor |
| `src/tools/send_whatsapp.py` | WhatsApp tool implementation |
| `src/adapters/plivo_adapter.py` | Plivo API client |
| `src/core/config.py` | Environment configuration |
| `prompts.json` | AI agent system prompts |

## Error Handling

### Gemini Connection Errors

```python
try:
    async with websockets.connect(url) as ws:
        # Session logic
except Exception as e:
    logger.error(f"Google Live error: {e}")
finally:
    self.goog_live_ws = None
```

### Tool Execution Errors

```python
try:
    result = await execute_tool(tool_name, caller_phone, **tool_args)
except Exception as e:
    logger.error(f"Tool execution error: {e}")
    # Send error response to Gemini
    tool_response = {
        "tool_response": {
            "function_responses": [{
                "id": call_id,
                "name": tool_name,
                "response": {"success": False, "message": str(e)}
            }]
        }
    }
```

## Monitoring

### Logs

- **Application**: `logs/whatsapp_voice.log` (rotates at 10MB, 7 days retention)
- **Transcripts**: `transcripts/{call_uuid}.txt` (per-call)
- **Console**: Real-time debug output when `DEBUG=true`

### Key Log Messages

```
# Preloading
PRELOADING Gemini session for call {uuid}
Connected to Google Live API
Google Live setup complete - AI Ready
Sent initial greeting trigger
PRELOAD COMPLETE for {uuid} - AI ready to speak!

# Call Connection
Plivo stream WebSocket connected for call {uuid}
Using PRELOADED session for {uuid}
Sending 30 preloaded audio chunks

# Tool Execution
TOOL CALL: send_whatsapp with args: {...}
TOOL RESULT: {'success': True, ...}
Sent tool response for send_whatsapp
```

## Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| First audio (with preload) | <500ms | ~100ms |
| First audio (no preload) | <3000ms | ~2500ms |
| Audio buffer latency | <50ms | 20ms |
| Tool execution | <2000ms | ~1000ms |
