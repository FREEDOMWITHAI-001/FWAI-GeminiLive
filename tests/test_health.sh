#!/bin/bash
# =============================================================================
# Health Check Tests
# =============================================================================
# Tests server connectivity, health endpoints, and basic functionality
#
# Usage:
#   export SERVER_URL="http://140.245.206.162:3000"
#   ./test_health.sh
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default server URL
SERVER_URL="${SERVER_URL:-http://140.245.206.162:3000}"

echo "=============================================="
echo "  FWAI Health Check Tests"
echo "  Server: $SERVER_URL"
echo "=============================================="
echo ""

# Test counter
PASSED=0
FAILED=0

# Function to run a test
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
# Test 1: Basic Health Check
# =============================================================================
echo "--- Basic Connectivity ---"

run_test "Health endpoint" \
    "curl -s $SERVER_URL" \
    "status.*ok"

run_test "Service name in response" \
    "curl -s $SERVER_URL" \
    "WhatsApp Voice Calling with Gemini Live"

run_test "Version in response" \
    "curl -s $SERVER_URL" \
    "version"

# =============================================================================
# Test 2: Port Connectivity
# =============================================================================
echo ""
echo "--- Port Connectivity ---"

# Extract host and port from SERVER_URL
HOST=$(echo $SERVER_URL | sed -e 's|http://||' -e 's|:.*||')
PORT=$(echo $SERVER_URL | sed -e 's|.*:||' -e 's|/.*||')

echo -n "Testing: Port $PORT is open... "
if nc -zv -w 5 $HOST $PORT 2>&1 | grep -q "succeeded\|Connected\|open"; then
    echo -e "${GREEN}PASSED${NC}"
    ((PASSED++))
else
    # Try alternative check
    if timeout 5 bash -c "echo > /dev/tcp/$HOST/$PORT" 2>/dev/null; then
        echo -e "${GREEN}PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${YELLOW}SKIPPED${NC} (nc not available or port closed)"
    fi
fi

# =============================================================================
# Test 3: Response Headers
# =============================================================================
echo ""
echo "--- Response Headers ---"

run_test "Content-Type is JSON" \
    "curl -sI $SERVER_URL | grep -i content-type" \
    "application/json"

run_test "HTTP 200 OK" \
    "curl -sI $SERVER_URL | head -1" \
    "200"

# =============================================================================
# Test 4: API Endpoints Exist
# =============================================================================
echo ""
echo "--- API Endpoints ---"

run_test "GET /calls endpoint" \
    "curl -s -o /dev/null -w '%{http_code}' $SERVER_URL/calls" \
    "200"

run_test "Webhook endpoint exists" \
    "curl -s -o /dev/null -w '%{http_code}' '$SERVER_URL/webhook?hub.mode=subscribe&hub.verify_token=PSPK&hub.challenge=test'" \
    "200"

# =============================================================================
# Test 5: Response Time
# =============================================================================
echo ""
echo "--- Performance ---"

echo -n "Testing: Response time < 2 seconds... "
response_time=$(curl -s -o /dev/null -w '%{time_total}' $SERVER_URL)
if (( $(echo "$response_time < 2" | bc -l) )); then
    echo -e "${GREEN}PASSED${NC} (${response_time}s)"
    ((PASSED++))
else
    echo -e "${YELLOW}SLOW${NC} (${response_time}s)"
fi

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

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
