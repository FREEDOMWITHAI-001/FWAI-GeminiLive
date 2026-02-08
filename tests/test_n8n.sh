#!/bin/bash
# =============================================================================
# n8n Integration Tests
# =============================================================================
# Tests for n8n workflow integration
#
# Usage:
#   export SERVER_URL="http://140.245.206.162:3000"
#   export N8N_WEBHOOK_URL="https://your-n8n-url/webhook/trigger-call"
#   ./test_n8n.sh
# =============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SERVER_URL="${SERVER_URL:-http://140.245.206.162:3000}"
N8N_WEBHOOK_URL="${N8N_WEBHOOK_URL:-}"

echo "=============================================="
echo "  FWAI n8n Integration Tests"
echo "  Server: $SERVER_URL"
echo "=============================================="
echo ""

if [ -z "$N8N_WEBHOOK_URL" ]; then
    echo -e "${YELLOW}N8N_WEBHOOK_URL not set${NC}"
    echo "Set it with: export N8N_WEBHOOK_URL='https://your-n8n/webhook/trigger-call'"
    echo ""
fi

# =============================================================================
# n8n Trigger Call Workflow
# =============================================================================
echo "--- n8n Trigger Call ---"
echo ""
echo "To trigger a call via n8n webhook:"
echo ""
cat << 'EOF'
# Basic call (uses questions from config file):
curl -X POST https://YOUR_N8N_URL/webhook/trigger-call \
  -H "Content-Type: application/json" \
  -d '{
    "phoneNumber": "919876543210",
    "contactName": "John Doe",
    "campaign": "follow-up"
  }'

# Call with custom questions from n8n (overrides config file):
curl -X POST https://YOUR_N8N_URL/webhook/trigger-call \
  -H "Content-Type: application/json" \
  -d '{
    "phoneNumber": "919876543210",
    "contactName": "John Doe",
    "questions": [
      {"id": "greeting", "prompt": "Hi {customer_name}, this is {agent_name} from {company_name}. How are you?"},
      {"id": "experience", "prompt": "How was your experience with our masterclass?"},
      {"id": "goal", "prompt": "What is your main goal with AI?"}
    ]
  }'
EOF
echo ""

if [ -n "$N8N_WEBHOOK_URL" ]; then
    echo -n "Testing n8n webhook connectivity... "
    response=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$N8N_WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d '{"test": true}' 2>/dev/null || echo "000")
    if [ "$response" = "200" ] || [ "$response" = "201" ]; then
        echo -e "${GREEN}OK${NC} (HTTP $response)"
    else
        echo -e "${YELLOW}HTTP $response${NC}"
    fi
fi

# =============================================================================
# n8n Call Ended Webhook (Received by n8n)
# =============================================================================
echo ""
echo "--- Call Ended Payload (Sent to n8n) ---"
echo ""
echo "When a call ends, FWAI sends this to n8n:"
echo ""
cat << 'EOF'
{
  "event": "call_ended",
  "call_uuid": "abc123-xyz789",
  "caller_phone": "+919876543210",
  "contact_name": "John Doe",
  "duration_seconds": 125.3,
  "transcript": "AGENT: Hi, this is Vishnu from Freedom with AI...\nUSER: Hi, I attended the masterclass...\nAGENT: Great! How did you find it?",
  "call_status": "completed",
  "timestamp": "2024-01-26T10:30:00Z"
}
EOF
echo ""

# =============================================================================
# n8n Workflow Setup
# =============================================================================
echo "--- n8n Workflow Setup ---"
echo ""
echo "1. Import workflow from: n8n_flows/FWAI_Internal/outbound_call.json"
echo ""
echo "2. Update these nodes with your OCI server URL:"
echo "   - 'Make Outbound Call' node: http://140.245.206.162:3000/plivo/make-call"
echo ""
echo "3. Configure 'Call Ended Webhook' to receive data"
echo ""
echo "4. Add your post-call automation (CRM update, email, etc.)"
echo ""

# =============================================================================
# Test n8n to FWAI Connection
# =============================================================================
echo "--- Test n8n → FWAI Connection ---"
echo ""
echo "From n8n HTTP Request node, test with:"
echo ""
echo "  URL: $SERVER_URL"
echo "  Method: GET"
echo ""
echo "Expected response:"
echo '  {"status":"ok","service":"WhatsApp Voice Calling with Gemini Live"}'
echo ""

# =============================================================================
# Workflow Diagram
# =============================================================================
echo "=============================================="
echo "  n8n Workflow Diagram"
echo "=============================================="
echo ""
cat << 'EOF'
OUTBOUND CALL FLOW:
==================
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Webhook Trigger │────>│ HTTP Request to  │────>│ Respond with    │
│ /trigger-call   │     │ FWAI /make-call  │     │ Call UUID       │
└─────────────────┘     └──────────────────┘     └─────────────────┘

CALL ENDED FLOW:
================
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ FWAI sends POST │────>│ n8n Webhook      │────>│ Process Data    │
│ to n8n webhook  │     │ /call-ended      │     │ (CRM, Email)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
EOF
echo ""
echo "=============================================="
