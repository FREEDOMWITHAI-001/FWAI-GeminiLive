#!/bin/bash
# =============================================================================
# Outbound Call Tests
# =============================================================================
# Tests for making outbound calls via Plivo
#
# Usage:
#   export SERVER_URL="http://140.245.206.162:3000"
#   export TEST_PHONE="919876543210"
#   ./test_calls.sh
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SERVER_URL="${SERVER_URL:-http://140.245.206.162:3000}"
TEST_PHONE="${TEST_PHONE:-}"

echo "=============================================="
echo "  FWAI Outbound Call Tests"
echo "  Server: $SERVER_URL"
echo "=============================================="
echo ""

# Check if TEST_PHONE is set
if [ -z "$TEST_PHONE" ]; then
    echo -e "${YELLOW}WARNING: TEST_PHONE not set${NC}"
    echo "Set it with: export TEST_PHONE='919876543210'"
    echo ""
    echo "Running in DRY RUN mode (no actual calls)"
    DRY_RUN=true
else
    echo "Test Phone: $TEST_PHONE"
    DRY_RUN=false
fi

echo ""

# =============================================================================
# Test 1: List Active Calls
# =============================================================================
echo "--- List Active Calls ---"

echo -n "GET /calls... "
response=$(curl -s $SERVER_URL/calls)
echo -e "${GREEN}OK${NC}"
echo "Response: $response"
echo ""

# =============================================================================
# Test 2: Make Outbound Call (Plivo)
# =============================================================================
echo "--- Make Outbound Call (Plivo) ---"

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN - Showing curl command only${NC}"
    echo ""
    echo "To make an actual call, run:"
    echo ""
    cat << 'EOF'
curl -X POST http://140.245.206.162:3000/plivo/make-call \
  -H "Content-Type: application/json" \
  -d '{
    "phoneNumber": "919876543210",
    "contactName": "Test User"
  }'
EOF
    echo ""
else
    echo -n "Making call to $TEST_PHONE... "
    response=$(curl -s -X POST $SERVER_URL/plivo/make-call \
        -H "Content-Type: application/json" \
        -d "{
            \"phoneNumber\": \"$TEST_PHONE\",
            \"contactName\": \"Test Call\"
        }")
    echo -e "${GREEN}SENT${NC}"
    echo "Response: $response"
    echo ""

    # Extract call_uuid if present
    call_uuid=$(echo $response | grep -o '"call_uuid":"[^"]*"' | cut -d'"' -f4 || echo "")
    if [ -n "$call_uuid" ]; then
        echo "Call UUID: $call_uuid"
        echo ""
        echo "To terminate this call:"
        echo "curl -X POST $SERVER_URL/calls/$call_uuid/terminate"
    fi
fi

echo ""

# =============================================================================
# Test 3: Call with Custom Parameters
# =============================================================================
echo "--- Call with Custom Parameters ---"

echo "Example call with all parameters:"
echo ""
cat << EOF
curl -X POST $SERVER_URL/plivo/make-call \\
  -H "Content-Type: application/json" \\
  -d '{
    "phoneNumber": "919876543210",
    "contactName": "John Doe",
    "customData": {
      "campaign": "follow-up",
      "source": "n8n-workflow"
    }
  }'
EOF
echo ""

# =============================================================================
# Test 4: Terminate Call
# =============================================================================
echo "--- Terminate Call ---"

echo "To terminate an active call:"
echo ""
echo "curl -X POST $SERVER_URL/calls/{call_uuid}/terminate"
echo ""

# =============================================================================
# Test 5: View Call Transcripts
# =============================================================================
echo "--- View Transcripts ---"

echo "After call ends, transcripts are saved in /transcripts folder."
echo ""
echo "On OCI server:"
echo "  ls -la /opt/fwai/FWAI-GeminiLive/transcripts/"
echo "  cat /opt/fwai/FWAI-GeminiLive/transcripts/LATEST_FILE.txt"
echo ""

# =============================================================================
# Quick Reference
# =============================================================================
echo "=============================================="
echo "  Quick Reference - Call Commands"
echo "=============================================="
echo ""
echo "1. Make call:"
echo "   curl -X POST $SERVER_URL/plivo/make-call \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"phoneNumber\": \"PHONE\", \"contactName\": \"NAME\"}'"
echo ""
echo "2. List calls:"
echo "   curl $SERVER_URL/calls"
echo ""
echo "3. End call:"
echo "   curl -X POST $SERVER_URL/calls/{uuid}/terminate"
echo ""
echo "4. View logs (on server):"
echo "   sudo journalctl -u fwai-app -f"
echo ""
echo "=============================================="
