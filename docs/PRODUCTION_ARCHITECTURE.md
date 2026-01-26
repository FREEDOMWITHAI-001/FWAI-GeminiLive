# Production Architecture

This document describes the production architecture for the FWAI WebRTC Gemini voice calling platform deployed on Oracle Cloud Infrastructure.

## Table of Contents
- [System Overview](#system-overview)
- [Architecture Diagram](#architecture-diagram)
- [Components](#components)
- [Data Flow](#data-flow)
- [Infrastructure](#infrastructure)
- [Security](#security)
- [Scaling Considerations](#scaling-considerations)
- [Monitoring](#monitoring)
- [Disaster Recovery](#disaster-recovery)

---

## System Overview

FWAI WebRTC Gemini is a real-time voice AI platform that enables intelligent voice conversations using Google Gemini's multimodal capabilities. The system supports multiple telephony providers (WhatsApp, Plivo, Exotel) and provides real-time speech-to-speech AI interactions.

### Key Features
- Real-time voice conversations with AI
- Multi-provider telephony support
- WebRTC-based audio streaming
- Function calling (SMS, Email, Callbacks, Demo booking)
- Conversation transcripts and logging
- n8n workflow integration

---

## Architecture Diagram

```
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │                    ORACLE CLOUD (OCI)                        │
                                    │                    Always Free Tier                          │
                                    │  ┌─────────────────────────────────────────────────────────┐ │
                                    │  │              Virtual Cloud Network (VCN)                │ │
                                    │  │                    10.0.0.0/16                          │ │
┌──────────────┐                    │  │  ┌───────────────────────────────────────────────────┐  │ │
│              │                    │  │  │              Public Subnet 10.0.0.0/24            │  │ │
│   End User   │                    │  │  │                                                   │  │ │
│   (Phone)    │                    │  │  │  ┌─────────────────────────────────────────────┐  │  │ │
│              │                    │  │  │  │         Compute Instance (E2.1.Micro)       │  │  │ │
└──────┬───────┘                    │  │  │  │              1 OCPU | 1GB RAM               │  │  │ │
       │                            │  │  │  │              + 2GB Swap                     │  │  │ │
       │ Voice Call                 │  │  │  │                                             │  │  │ │
       ▼                            │  │  │  │  ┌───────────────────────────────────────┐  │  │  │ │
┌──────────────┐                    │  │  │  │  │           FastAPI Application         │  │  │  │ │
│   Telephony  │  Webhook/WebSocket │  │  │  │  │              Port 3000                │  │  │  │ │
│   Provider   │◄──────────────────►│  │  │  │  │                                       │  │  │  │ │
│              │                    │  │  │  │  │  ┌─────────┐  ┌─────────┐  ┌───────┐  │  │  │  │ │
│ - WhatsApp   │                    │  │  │  │  │  │ WebRTC  │  │ Adapters│  │ Tools │  │  │  │  │ │
│ - Plivo      │                    │  │  │  │  │  │ Handler │  │         │  │       │  │  │  │  │ │
│ - Exotel     │                    │  │  │  │  │  └────┬────┘  └────┬────┘  └───┬───┘  │  │  │  │ │
└──────────────┘                    │  │  │  │  │       │            │          │       │  │  │  │ │
                                    │  │  │  │  │       ▼            ▼          ▼       │  │  │  │ │
                                    │  │  │  │  │  ┌─────────────────────────────────┐  │  │  │  │ │
                                    │  │  │  │  │  │        Gemini Agent            │  │  │  │  │ │
                                    │  │  │  │  │  │    (Audio Processing)          │  │  │  │  │ │
                                    │  │  │  │  │  └───────────────┬─────────────────┘  │  │  │  │ │
                                    │  │  │  │  └──────────────────┼────────────────────┘  │  │  │ │
                                    │  │  │  │                     │                       │  │  │ │
                                    │  │  │  └─────────────────────┼───────────────────────┘  │  │ │
                                    │  │  │                        │                          │  │ │
                                    │  │  └────────────────────────┼──────────────────────────┘  │ │
                                    │  │                           │                             │ │
                                    │  └───────────────────────────┼─────────────────────────────┘ │
                                    │                              │                               │
                                    └──────────────────────────────┼───────────────────────────────┘
                                                                   │
                                                                   │ WebSocket (Secure)
                                                                   ▼
                                    ┌──────────────────────────────────────────────────────────────┐
                                    │                     EXTERNAL SERVICES                         │
                                    │                                                              │
                                    │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │
                                    │  │  Google Gemini │  │   SMTP Server  │  │  n8n Workflow  │  │
                                    │  │   Live API     │  │   (Email)      │  │   (Webhooks)   │  │
                                    │  └────────────────┘  └────────────────┘  └────────────────┘  │
                                    │                                                              │
                                    └──────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. FastAPI Application (`src/app.py`)
The main web server handling HTTP requests and WebSocket connections.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/webhook` | GET/POST | WhatsApp webhook verification and events |
| `/plivo/answer` | POST | Plivo call answer webhook |
| `/plivo/stream` | WebSocket | Plivo audio stream |
| `/plivo/hangup` | POST | Plivo call hangup webhook |
| `/exotel/*` | Various | Exotel integration endpoints |

### 2. Telephony Adapters (`src/adapters/`)
Abstract layer for multiple telephony providers.

```
src/adapters/
├── base.py              # Base adapter interface
├── whatsapp_adapter.py  # WhatsApp Business API
├── plivo_adapter.py     # Plivo Voice API
└── exotel_adapter.py    # Exotel Voice API
```

### 3. Gemini Agent (`src/services/gemini_agent.py`)
Handles AI conversation logic and Google Gemini API integration.

**Features:**
- Real-time audio streaming to Gemini Live API
- System prompt management
- Function calling support
- Conversation memory

### 4. WebRTC Handler (`src/handlers/webrtc_handler.py`)
Manages WebRTC connections for real-time audio.

**Capabilities:**
- Audio track handling
- Codec negotiation (Opus)
- ICE candidate management

### 5. Audio Processor (`src/core/audio_processor.py`)
Handles audio format conversion and processing.

**Processing:**
- Sample rate conversion (16kHz)
- Mono channel
- 16-bit PCM encoding
- Noise reduction (optional)

### 6. Tools (`src/tools/`)
Function calling capabilities for the AI agent.

```
src/tools/
├── base.py              # Base tool interface
├── tool_registry.py     # Tool registration
├── send_sms.py          # SMS via Plivo
├── send_email.py        # Email via SMTP
├── send_whatsapp.py     # WhatsApp messages
├── schedule_callback.py # Callback scheduling
└── book_demo.py         # Demo booking
```

---

## Data Flow

### Inbound Call Flow (Plivo Example)

```
1. User dials phone number
         │
         ▼
2. Plivo receives call, sends webhook to /plivo/answer
         │
         ▼
3. FastAPI returns XML response with Stream instruction
         │
         ▼
4. Plivo establishes WebSocket to /plivo/stream
         │
         ▼
5. Audio frames received via WebSocket
         │
         ▼
6. Audio processed and sent to Gemini Live API
         │
         ▼
7. Gemini generates response audio
         │
         ▼
8. Response audio sent back via WebSocket
         │
         ▼
9. Plivo plays audio to user
```

### WhatsApp Call Flow

```
1. User initiates WhatsApp voice call
         │
         ▼
2. Meta sends webhook notification
         │
         ▼
3. Application accepts call via API
         │
         ▼
4. WebRTC connection established
         │
         ▼
5. Audio streamed bidirectionally
         │
         ▼
6. Gemini processes and responds
```

---

## Infrastructure

### Oracle Cloud Resources

| Resource | Configuration | Cost |
|----------|---------------|------|
| Compute Instance | VM.Standard.E2.1.Micro (1 OCPU, 1GB RAM) | Free |
| Boot Volume | 50 GB SSD | Free |
| Swap Space | 2 GB | - |
| Public IP | Ephemeral | Free |
| VCN + Subnet | 10.0.0.0/16 | Free |
| Egress Bandwidth | 10 TB/month | Free |

### Network Configuration

```
VCN: 10.0.0.0/16
├── Public Subnet: 10.0.0.0/24
│   └── Compute Instance
│       ├── Public IP: xxx.xxx.xxx.xxx
│       └── Private IP: 10.0.0.x
└── Private Subnet: 10.0.1.0/24 (future use)
```

### Security List Rules

| Direction | Protocol | Port | Source/Dest | Purpose |
|-----------|----------|------|-------------|---------|
| Ingress | TCP | 22 | 0.0.0.0/0 | SSH |
| Ingress | TCP | 80 | 0.0.0.0/0 | HTTP |
| Ingress | TCP | 443 | 0.0.0.0/0 | HTTPS |
| Ingress | TCP | 3000 | 0.0.0.0/0 | FastAPI |
| Ingress | TCP | 8003 | 0.0.0.0/0 | Gemini Live |
| Ingress | UDP | 10000-20000 | 0.0.0.0/0 | WebRTC |
| Egress | All | All | 0.0.0.0/0 | Outbound |

---

## Security

### Current Implementation

1. **Network Security**
   - OCI Security Lists (firewall)
   - Ubuntu iptables
   - VCN isolation

2. **Application Security**
   - Environment variables for secrets
   - Webhook verification tokens
   - Input validation

3. **API Security**
   - Meta webhook verification
   - Plivo signature validation

### Recommended Enhancements

1. **HTTPS/TLS**
   ```bash
   # Install Nginx and Certbot
   sudo apt install nginx certbot python3-certbot-nginx

   # Configure reverse proxy
   sudo nano /etc/nginx/sites-available/fwai

   # Get SSL certificate
   sudo certbot --nginx -d yourdomain.com
   ```

2. **Restrict SSH Access**
   - Change Security List SSH rule to your IP only
   - Use SSH key authentication only
   - Consider bastion host for production

3. **Secrets Management**
   - Use OCI Vault for production
   - Rotate API keys regularly
   - Never commit secrets to git

4. **Rate Limiting**
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   ```

---

## Scaling Considerations

### Vertical Scaling (Current)

| Tier | Instance | RAM | Concurrent Calls |
|------|----------|-----|------------------|
| Free | E2.1.Micro | 1 GB | 1-2 |
| Free | A1.Flex | 6-24 GB | 5-15 |
| Paid | E4.Flex | 16+ GB | 20+ |

### Horizontal Scaling (Future)

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    │   (OCI LB)      │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │  Instance 1 │   │  Instance 2 │   │  Instance 3 │
    │  (Active)   │   │  (Active)   │   │  (Standby)  │
    └─────────────┘   └─────────────┘   └─────────────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Redis Cluster  │
                    │ (Session State) │
                    └─────────────────┘
```

### Scaling Requirements

1. **Session Affinity**: WebSocket connections need sticky sessions
2. **Shared State**: Redis for conversation state across instances
3. **Database**: PostgreSQL for transcripts and analytics
4. **Queue**: RabbitMQ/Redis for async function calls

---

## Monitoring

### System Metrics

```bash
# Check memory usage
free -h

# Check CPU and processes
htop

# Check disk space
df -h

# Check network connections
ss -tulpn
```

### Application Logs

```bash
# Live logs
sudo journalctl -u fwai-app -f

# Recent logs
sudo journalctl -u fwai-app -n 100

# Logs since time
sudo journalctl -u fwai-app --since "1 hour ago"
```

### Health Check Endpoint

```bash
curl http://localhost:3000
# Expected: {"status":"ok","service":"WhatsApp Voice Calling with Gemini Live","version":"1.0.0"}
```

### Recommended Monitoring Stack

1. **Metrics**: Prometheus + Grafana
2. **Logs**: Loki or ELK Stack
3. **Alerts**: OCI Notifications or PagerDuty
4. **Uptime**: UptimeRobot or Pingdom

---

## Disaster Recovery

### Backup Strategy

1. **Application Code**: Git repository
2. **Configuration**: Encrypted `.env` backup
3. **Transcripts**: Daily backup to OCI Object Storage

### Recovery Procedure

1. Create new compute instance
2. Clone repository
3. Restore `.env` configuration
4. Install dependencies
5. Start systemd service
6. Update DNS/webhook URLs

### RTO/RPO Targets

| Metric | Target | Strategy |
|--------|--------|----------|
| RTO (Recovery Time) | < 30 minutes | Automated deployment scripts |
| RPO (Data Loss) | < 1 hour | Hourly transcript backups |

---

## File Structure

```
FWAI-GeminiLive/
├── run.py                      # Application entry point
├── requirements.txt            # Python dependencies
├── prompts.json               # AI system prompts
├── .env                       # Environment configuration (not in git)
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── app.py                 # FastAPI application
│   ├── prompt_loader.py       # Prompt management
│   ├── conversation_memory.py # Conversation state
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration management
│   │   └── audio_processor.py # Audio processing
│   │
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py            # Base adapter interface
│   │   ├── whatsapp_adapter.py
│   │   ├── plivo_adapter.py
│   │   └── exotel_adapter.py
│   │
│   ├── handlers/
│   │   ├── __init__.py
│   │   └── webrtc_handler.py  # WebRTC management
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── gemini_agent.py    # Gemini AI integration
│   │   ├── whatsapp_client.py
│   │   ├── plivo_gemini_stream.py
│   │   └── meta_token_manager.py
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── tool_registry.py
│   │   ├── send_sms.py
│   │   ├── send_email.py
│   │   ├── send_whatsapp.py
│   │   ├── schedule_callback.py
│   │   └── book_demo.py
│   │
│   └── templates/
│       ├── __init__.py
│       └── whatsapp_templates.py
│
├── docs/
│   ├── OCI_DEPLOYMENT_GUIDE.md
│   └── PRODUCTION_ARCHITECTURE.md
│
├── data/
│   └── callbacks.json         # Scheduled callbacks
│
├── transcripts/               # Call transcripts
│
└── n8n_flows/                 # n8n workflow exports
    └── FWAI_Internal/
```

---

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `CALL_PROVIDER` | Yes | `whatsapp`, `plivo`, or `exotel` |
| `HOST` | No | Server host (default: 0.0.0.0) |
| `PORT` | No | Server port (default: 3000) |
| `DEBUG` | No | Debug mode (default: false) |
| `GOOGLE_API_KEY` | Yes | Google Gemini API key |
| `PLIVO_AUTH_ID` | If Plivo | Plivo Auth ID |
| `PLIVO_AUTH_TOKEN` | If Plivo | Plivo Auth Token |
| `PLIVO_PHONE_NUMBER` | If Plivo | Plivo phone number |
| `PLIVO_CALLBACK_URL` | If Plivo | Public URL for webhooks |
| `PHONE_NUMBER_ID` | If WhatsApp | WhatsApp Phone Number ID |
| `META_ACCESS_TOKEN` | If WhatsApp | Meta API access token |
| `META_APP_ID` | If WhatsApp | Meta App ID |
| `META_APP_SECRET` | If WhatsApp | Meta App Secret |
| `META_VERIFY_TOKEN` | If WhatsApp | Webhook verification token |
| `SMTP_HOST` | For email | SMTP server host |
| `SMTP_PORT` | For email | SMTP server port |
| `SMTP_USER` | For email | SMTP username |
| `SMTP_PASSWORD` | For email | SMTP password |
| `TTS_VOICE` | No | Gemini voice (default: Puck) |
| `LOG_LEVEL` | No | Logging level (default: INFO) |
| `ENABLE_TRANSCRIPTS` | No | Save transcripts (default: true) |

---

## Related Documentation

- [OCI Deployment Guide](./OCI_DEPLOYMENT_GUIDE.md)
- [README.md](../README.md)
- [ARCHITECTURE.md](../ARCHITECTURE.md)
