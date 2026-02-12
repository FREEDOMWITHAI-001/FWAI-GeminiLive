# FWAI Voice AI Agent

Real-time Voice AI Agent using **Plivo** + **Google Gemini 2.5 Live** with **n8n Integration**.

## Features

- Real-time voice conversations with AI
- NEPQ sales methodology built-in
- n8n workflow integration for automation
- Automatic call hangup when conversation ends
- Transcripts with USER/AGENT labels (Whisper)
- WhatsApp messaging tool
- Callback scheduling

## Quick Start

### 1. Setup

```bash
git clone https://github.com/kiranfwai/FWAI_WebRTC_Gemini.git
cd FWAI_WebRTC_Gemini

python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
pip install openai-whisper  # For transcription
```

### 2. Configure `.env`

```env
# Plivo (Required)
PLIVO_AUTH_ID=your_plivo_auth_id
PLIVO_AUTH_TOKEN=your_plivo_auth_token
PLIVO_FROM_NUMBER=+912268093710
PLIVO_CALLBACK_URL=https://your-ngrok-url.ngrok-free.app

# Google AI (Required)
GOOGLE_API_KEY=your_google_api_key

# Voice Settings
TTS_VOICE=Puck

# WhatsApp (Optional)
META_ACCESS_TOKEN=your_meta_access_token
WHATSAPP_PHONE_ID=your_whatsapp_phone_number_id

# Transcripts
ENABLE_TRANSCRIPTS=true
```

### 3. Expose with ngrok

```bash
ngrok http 3001
# Update PLIVO_CALLBACK_URL in .env with the ngrok URL
```

### 4. Configure Plivo

In [Plivo Console](https://console.plivo.com/) → Voice → Applications:
- **Answer URL**: `https://your-ngrok-url/plivo/answer` (POST)
- **Hangup URL**: `https://your-ngrok-url/plivo/hangup` (POST)
- Assign your Plivo number to this application

### 5. Run

```bash
python run.py
```

## Make a Call

### Direct API Call

```bash
curl -X POST http://localhost:3001/plivo/make-call \
  -H "Content-Type: application/json" \
  -d '{"phoneNumber": "919876543210", "contactName": "John"}'
```

### Via n8n (Recommended)

1. Import `n8n_flows/FWAI_Internal/outbound_call.json` into n8n
2. Update ngrok URL in "Make Outbound Call" node
3. Activate workflow
4. Trigger:

```bash
curl -X POST https://your-n8n-url/webhook/trigger-call \
  -H "Content-Type: application/json" \
  -d '{"phoneNumber": "919876543210", "contactName": "John"}'
```

## n8n Integration

The n8n workflow provides:

- **Trigger call** via webhook
- **Receive call data** when call ends (transcript, duration)
- **Post-call automation** (CRM updates, emails, etc.)

### Flow Diagram

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Webhook Trigger │────>│ Make Outbound    │────>│ Respond Success │
│ /trigger-call   │     │ Call to Server   │     │ or Error        │
└─────────────────┘     └──────────────────┘     └─────────────────┘

┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Call Ended      │────>│ Extract Call     │────>│ Process         │
│ Webhook         │     │ Data             │     │ (Add your logic)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Call Ended Payload

When call ends, n8n receives:

```json
{
  "event": "call_ended",
  "call_uuid": "abc123-xyz",
  "caller_phone": "+919876543210",
  "duration_seconds": 125.3,
  "transcript": "AGENT: Hi, this is Vishnu...\nUSER: Hi, I attended the masterclass..."
}
```

## Transcript Format

Transcripts are saved with USER/AGENT labels:

```
[00:00:05] SYSTEM: Call connected
AGENT: Hi, this is Vishnu from Freedom with AI. How are you doing today?
USER: I'm good, thanks.
AGENT: Great! I noticed you attended our AI Masterclass. How did you find it?
USER: It was very informative.
...
[00:02:15] SYSTEM: Call ended
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/plivo/make-call` | POST | Initiate outbound call |
| `/calls` | GET | List active calls |
| `/calls/{call_id}/terminate` | POST | End a call |
| `/` | GET | Health check |

## Available AI Tools

| Tool | Description |
|------|-------------|
| `send_whatsapp` | Send WhatsApp message to caller |
| `schedule_callback` | Schedule a callback |
| `end_call` | End the call gracefully |

## Project Structure

```
FWAI_WebRTC_Gemini/
├── run.py                    # Entry point
├── .env                      # Configuration
├── src/
│   ├── app.py                # FastAPI server
│   ├── services/
│   │   └── plivo_gemini_stream.py  # Plivo ↔ Gemini bridge
│   └── tools/                # AI callable tools
├── n8n_flows/
│   └── FWAI_Internal/
│       └── outbound_call.json  # n8n workflow
├── transcripts/              # Call transcripts
└── docs/
    └── ARCHITECTURE.md       # Detailed architecture
```

## Customize AI Agent

Prompts are provided via the `/plivo/make-call` API in the `prompt` field. Use `{{placeholder}}` syntax for dynamic values (e.g., `{{agent_name}}`, `{{company_name}}`). See `src/conversational_prompts.py` for available defaults.

## Production Deployment (Oracle Cloud)

Deploy for **FREE** on Oracle Cloud Always Free Tier. See [OCI Deployment Guide](docs/OCI_DEPLOYMENT_GUIDE.md) for complete setup.

### Quick Deploy Commands

```bash
# SSH into server
ssh -i ~/.ssh/oracle_key.key ubuntu@YOUR_PUBLIC_IP

# Navigate to app
cd /opt/fwai/FWAI-GeminiLive
source venv/bin/activate
```

### Service Management

| Command | Description |
|---------|-------------|
| `sudo systemctl status fwai-app` | Check app status |
| `sudo systemctl start fwai-app` | Start app |
| `sudo systemctl stop fwai-app` | Stop app |
| `sudo systemctl restart fwai-app` | Restart app |
| `sudo systemctl enable fwai-app` | Enable auto-start on boot |

### View Logs

```bash
# Live logs (follow mode)
sudo journalctl -u fwai-app -f

# Last 100 lines
sudo journalctl -u fwai-app -n 100

# Logs from last hour
sudo journalctl -u fwai-app --since "1 hour ago"

# Logs from today
sudo journalctl -u fwai-app --since today

# Export logs to file
sudo journalctl -u fwai-app --since today > /tmp/logs.txt
```

### Update Application

```bash
cd /opt/fwai/FWAI-GeminiLive
git pull origin main
sudo systemctl restart fwai-app
```

### Monitor Resources

```bash
# Check memory usage
free -h

# Check disk space
df -h

# Monitor CPU/Memory (live)
htop

# Check running processes
ps aux | grep python

# Check network connections
ss -tulpn | grep 3000

# Check if port is listening
sudo netstat -tlnp | grep 3000
```

### Health Check

```bash
# Local health check
curl http://localhost:3000

# Expected response:
# {"status":"ok","service":"WhatsApp Voice Calling with Gemini Live","version":"1.0.0"}
```

### Firewall Management

```bash
# List current rules
sudo iptables -L INPUT -n

# Add new port
sudo iptables -I INPUT -p tcp --dport PORT_NUMBER -j ACCEPT

# Save rules permanently
sudo netfilter-persistent save

# Reload rules
sudo netfilter-persistent reload
```

### Database/Files

```bash
# View transcripts
ls -la transcripts/

# View latest transcript
cat transcripts/$(ls -t transcripts/ | head -1)

# View scheduled callbacks
cat data/callbacks.json

# Clear old transcripts (older than 30 days)
find transcripts/ -type f -mtime +30 -delete
```

### Troubleshooting

```bash
# Check if app is running
sudo systemctl is-active fwai-app

# View failed service logs
sudo journalctl -u fwai-app --no-pager | tail -50

# Test Gemini API connectivity
curl -s "https://generativelanguage.googleapis.com/v1/models?key=$GOOGLE_API_KEY" | head

# Check environment variables are loaded
source venv/bin/activate
python -c "from src.core.config import config; print(f'Port: {config.port}')"

# Manual test run (for debugging)
sudo systemctl stop fwai-app
python run.py  # See live errors
# Ctrl+C to stop, then restart service
sudo systemctl start fwai-app
```

### Backup & Restore

```bash
# Backup .env and transcripts
tar -czvf backup_$(date +%Y%m%d).tar.gz .env transcripts/ data/

# Restore
tar -xzvf backup_YYYYMMDD.tar.gz
```

### Scheduled Cleanup (Crontab)

Auto-delete old files to save disk space:

```bash
# Edit crontab
crontab -e

# Add this line (deletes files older than 7 days, runs at 2 AM daily)
0 2 * * * find /opt/fwai/FWAI-GeminiLive/transcripts -type f -mtime +7 -delete && find /opt/fwai/FWAI-GeminiLive/recordings -type f -mtime +7 -delete
```

Verify: `crontab -l`

### Server Reboot

```bash
# Graceful restart (app will auto-start if enabled)
sudo reboot

# After reboot, verify app is running
sudo systemctl status fwai-app
```

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Detailed architecture, call flow, audio config, troubleshooting
- **[OCI_DEPLOYMENT_GUIDE.md](docs/OCI_DEPLOYMENT_GUIDE.md)** - Oracle Cloud deployment steps
- **[PRODUCTION_ARCHITECTURE.md](docs/PRODUCTION_ARCHITECTURE.md)** - Production architecture overview

## License

MIT
