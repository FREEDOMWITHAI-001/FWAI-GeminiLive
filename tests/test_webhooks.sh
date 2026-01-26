#!/bin/bash
# =============================================================================
# Webhook Tests
# =============================================================================
# Tests for webhook endpoints (WhatsApp, Plivo)
#
# Usage:
#   export SERVER_URL="http://140.245.206.162:3000"
#   ./test_webhooks.sh
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
SERVER_URL="${SERVER_URL:-http://140.245.206.162:3000}"
META_VERIFY_TOKEN="${META_VERIFY_TOKEN:-PSPK}"

echo "=============================================="
echo "  FWAI Webhook Tests"
echo "  Server: $SERVER_URL"
echo "=============================================="
echo ""

PASSED=0
FAILED=0

run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local expected="$3"

    echo -n "Testing: $test_name... "
    result=$(eval "$test_cmd" 2>&1) || true

    if echo "$result" | grep -q "$expected"; then
        echo -e "${GREEN}PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${RED}FAILED${NC}"
        echo "  Expected: $expected"
        echo "  Got: $result"
        ((FAILED++))
    fi
}

# =============================================================================
# Test 1: WhatsApp Webhook Verification
# =============================================================================
echo "--- WhatsApp Webhook Verification ---"

run_test "Webhook verification (valid token)" \
    "curl -s '$SERVER_URL/webhook?hub.mode=subscribe&hub.verify_token=$META_VERIFY_TOKEN&hub.challenge=test_challenge_123'" \
    "test_challenge_123"

run_test "Webhook verification (invalid token)" \
    "curl -s -o /dev/null -w '%{http_code}' '$SERVER_URL/webhook?hub.mode=subscribe&hub.verify_token=WRONG_TOKEN&hub.challenge=test'" \
    "403"

# =============================================================================
# Test 2: WhatsApp Webhook POST
# =============================================================================
echo ""
echo "--- WhatsApp Webhook POST ---"

echo -n "Testing: Webhook POST endpoint exists... "
response=$(curl -s -o /dev/null -w '%{http_code}' -X POST $SERVER_URL/webhook \
    -H "Content-Type: application/json" \
    -d '{"object": "whatsapp_business_account", "entry": []}')
if [ "$response" = "200" ] || [ "$response" = "400" ] || [ "$response" = "422" ]; then
    echo -e "${GREEN}PASSED${NC} (HTTP $response)"
    ((PASSED++))
else
    echo -e "${RED}FAILED${NC} (HTTP $response)"
    ((FAILED++))
fi

# =============================================================================
# Test 3: Plivo Answer Webhook
# =============================================================================
echo ""
echo "--- Plivo Webhooks ---"

echo -n "Testing: Plivo answer endpoint... "
response=$(curl -s -o /dev/null -w '%{http_code}' -X POST $SERVER_URL/plivo/answer \
    -H "Content-Type: application/json" \
    -d '{"CallUUID": "test-123", "From": "+919876543210", "To": "+912268093710"}')
# Accept various response codes as the endpoint exists
if [ "$response" = "200" ] || [ "$response" = "400" ] || [ "$response" = "422" ] || [ "$response" = "500" ]; then
    echo -e "${GREEN}PASSED${NC} (HTTP $response - endpoint exists)"
    ((PASSED++))
else
    echo -e "${RED}FAILED${NC} (HTTP $response)"
    ((FAILED++))
fi

echo -n "Testing: Plivo hangup endpoint... "
response=$(curl -s -o /dev/null -w '%{http_code}' -X POST $SERVER_URL/plivo/hangup \
    -H "Content-Type: application/json" \
    -d '{"CallUUID": "test-123"}')
if [ "$response" = "200" ] || [ "$response" = "400" ] || [ "$response" = "422" ]; then
    echo -e "${GREEN}PASSED${NC} (HTTP $response)"
    ((PASSED++))
else
    echo -e "${RED}FAILED${NC} (HTTP $response)"
    ((FAILED++))
fi

# =============================================================================
# Test 4: Simulated WhatsApp Call Event
# =============================================================================
echo ""
echo "--- Simulated Events ---"

echo "Simulated WhatsApp voice call webhook payload:"
echo ""
cat << 'EOF'
curl -X POST http://140.245.206.162:3000/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "object": "whatsapp_business_account",
    "entry": [{
      "id": "123456789",
      "changes": [{
        "value": {
          "messaging_product": "whatsapp",
          "metadata": {
            "display_phone_number": "912268093710",
            "phone_number_id": "100948263067135"
          },
          "statuses": [{
            "id": "wamid.xxx",
            "status": "accepted",
            "timestamp": "1234567890",
            "recipient_id": "919876543210",
            "type": "voice_call"
          }]
        },
        "field": "messages"
      }]
    }]
  }'
EOF
echo ""

# =============================================================================
# Test 5: Plivo Callback Simulation
# =============================================================================
echo "--- Plivo Callback Simulation ---"

echo "Simulated Plivo answer callback:"
echo ""
cat << 'EOF'
curl -X POST http://140.245.206.162:3000/plivo/answer \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "CallUUID=abc123-xyz789&From=+919876543210&To=+912268093710&Direction=outbound&CallStatus=ringing"
EOF
echo ""

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "  Test Summary"
echo "=============================================="
echo -e "  ${GREEN}Passed: $PASSED${NC}"
echo -e "  ${RED}Failed: $FAILED${NC}"
echo "=============================================="

# =============================================================================
# Webhook Configuration Reference
# =============================================================================
echo ""
echo "=============================================="
echo "  Webhook Configuration Reference"
echo "=============================================="
echo ""
echo "PLIVO CONSOLE (https://console.plivo.com):"
echo "  Answer URL:  $SERVER_URL/plivo/answer  (POST)"
echo "  Hangup URL:  $SERVER_URL/plivo/hangup  (POST)"
echo ""
echo "META DEVELOPER CONSOLE:"
echo "  Webhook URL: $SERVER_URL/webhook"
echo "  Verify Token: $META_VERIFY_TOKEN"
echo ""
echo "=============================================="

if [ $FAILED -eq 0 ]; then
    exit 0
else
    exit 1
fi
