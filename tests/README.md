# Testing Guide

This folder contains test scripts and documentation for testing the FWAI Voice AI Agent.

## Test Categories

| File | Description |
|------|-------------|
| `test_health.sh` | Health check and connectivity tests |
| `test_calls.sh` | Outbound call testing via Plivo |
| `test_webhooks.sh` | Webhook endpoint testing |
| `test_n8n.sh` | n8n integration tests |
| `test_tools.sh` | AI tool function tests |
| `manual_tests.md` | Manual testing checklist |

## Quick Start

### 1. Set Environment Variables

```bash
export SERVER_URL="http://140.245.206.162:3000"
export TEST_PHONE="919876543210"  # Your test phone number
export N8N_WEBHOOK_URL="https://your-n8n-url/webhook/trigger-call"
```

### 2. Run All Tests

```bash
cd tests
chmod +x *.sh
./run_all_tests.sh
```

### 3. Run Individual Tests

```bash
./test_health.sh      # Health checks
./test_calls.sh       # Call tests
./test_webhooks.sh    # Webhook tests
```

## Test from Different Locations

### From Local Machine (WSL/Linux)

```bash
export SERVER_URL="http://140.245.206.162:3000"
./test_health.sh
```

### From OCI Server

```bash
ssh ubuntu@140.245.206.162
cd /opt/fwai/FWAI-GeminiLive/tests
export SERVER_URL="http://localhost:3000"
./test_health.sh
```

## Prerequisites

- `curl` installed
- `jq` installed (for JSON parsing): `sudo apt install jq`
- `nc` (netcat) for port testing: `sudo apt install netcat`
