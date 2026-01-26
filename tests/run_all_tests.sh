#!/bin/bash
# =============================================================================
# Run All Tests
# =============================================================================
# Master test runner for FWAI Voice AI Agent
#
# Usage:
#   export SERVER_URL="http://140.245.206.162:3000"
#   ./run_all_tests.sh
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "  FWAI Voice AI Agent - Test Suite"
echo "=============================================="
echo "  Server: $SERVER_URL"
echo "  Date: $(date)"
echo "=============================================="
echo ""

# Make scripts executable
chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true

# Track results
TOTAL_SUITES=0
PASSED_SUITES=0
FAILED_SUITES=0

run_suite() {
    local name="$1"
    local script="$2"

    echo ""
    echo -e "${BLUE}========== $name ==========${NC}"
    echo ""

    ((TOTAL_SUITES++))

    if [ -f "$SCRIPT_DIR/$script" ]; then
        if bash "$SCRIPT_DIR/$script"; then
            ((PASSED_SUITES++))
            echo -e "${GREEN}Suite PASSED${NC}"
        else
            ((FAILED_SUITES++))
            echo -e "${RED}Suite FAILED${NC}"
        fi
    else
        echo -e "${YELLOW}Script not found: $script${NC}"
        ((FAILED_SUITES++))
    fi
}

# =============================================================================
# Run Test Suites
# =============================================================================

run_suite "Health Check Tests" "test_health.sh"
run_suite "Webhook Tests" "test_webhooks.sh"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "  Final Summary"
echo "=============================================="
echo ""
echo -e "  Total Suites: $TOTAL_SUITES"
echo -e "  ${GREEN}Passed: $PASSED_SUITES${NC}"
echo -e "  ${RED}Failed: $FAILED_SUITES${NC}"
echo ""
echo "=============================================="

# =============================================================================
# Quick Commands Reference
# =============================================================================

echo ""
echo "=============================================="
echo "  Quick Test Commands"
echo "=============================================="
echo ""
echo "Health Check:"
echo "  curl $SERVER_URL"
echo ""
echo "Make Call:"
echo "  curl -X POST $SERVER_URL/plivo/make-call \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"phoneNumber\": \"PHONE\", \"contactName\": \"NAME\"}'"
echo ""
echo "View Logs (on server):"
echo "  sudo journalctl -u fwai-app -f"
echo ""
echo "=============================================="

if [ $FAILED_SUITES -eq 0 ]; then
    echo -e "${GREEN}All test suites passed!${NC}"
    exit 0
else
    echo -e "${RED}Some test suites failed!${NC}"
    exit 1
fi
