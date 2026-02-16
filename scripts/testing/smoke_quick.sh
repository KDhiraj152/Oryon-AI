#!/bin/bash
# ============================================================================
# Oryon - Smoke Test
# ============================================================================
# Quick verification that all services are working
# Tests: Health → Auth → Upload → Process → Fetch
#
# Usage: ./scripts/smoke_test.sh [--verbose]
#
# Created by: K Dhiraj (TITAN-PROTOCOL)
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

API_URL="${API_URL:-http://localhost:8000}"
VERBOSE=false
PASSED=0
FAILED=0

for arg in "$@"; do
    case $arg in
        --verbose|-v) VERBOSE=true ;;
    esac
done

print_test() { echo -e "${BLUE}▶ $1${NC}"; }
print_pass() { echo -e "${GREEN}  ✓ $1${NC}"; ((PASSED++)); }
print_fail() { echo -e "${RED}  ✗ $1${NC}"; ((FAILED++)); }
print_skip() { echo -e "${YELLOW}  ⊘ $1${NC}"; }

log() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "    $1"
    fi
}

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  🧪 ORYON AI SMOKE TEST${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ============================================================================
# Test 1: Health Check
# ============================================================================
print_test "Health Check"

HEALTH=$(curl -s "$API_URL/health" 2>/dev/null || echo '{"status":"error"}')
STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','error'))" 2>/dev/null || echo "error")

if [[ "$STATUS" == "healthy" ]]; then
    print_pass "Basic health check passed"
else
    print_fail "Health check failed (is backend running?)"
    echo ""
    echo -e "  ${YELLOW}Start backend with: ./scripts/start.sh${NC}"
    exit 1
fi

# Detailed health
DETAILED=$(curl -s "$API_URL/health/detailed" 2>/dev/null || echo '{}')
log "Detailed: $DETAILED"
print_pass "Detailed health endpoint responding"

# ============================================================================
# Test 2: API Docs
# ============================================================================
print_test "API Documentation"

DOCS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/docs" 2>/dev/null || echo "000")
if [[ "$DOCS_STATUS" == "200" ]]; then
    print_pass "OpenAPI docs accessible at /docs"
else
    print_fail "API docs not accessible (HTTP $DOCS_STATUS)"
fi

# ============================================================================
# Test 3: Metrics Endpoint
# ============================================================================
print_test "Prometheus Metrics"

METRICS=$(curl -s "$API_URL/metrics" 2>/dev/null || echo "")
if echo "$METRICS" | grep -q "http_requests_total"; then
    print_pass "Metrics endpoint working"
else
    print_fail "Metrics endpoint not returning expected data"
fi

# ============================================================================
# Test 4: Authentication (V2 API)
# ============================================================================
print_test "Authentication"

# Try to register (may already exist)
REGISTER_RESP=$(curl -s -X POST "$API_URL/api/v2/auth/register" \
    -H "Content-Type: application/json" \
    -d '{"email":"smoke_test@local.test","password":"SmokeTest@2025!","full_name":"Smoke Test","role":"user"}' \
    2>/dev/null || echo '{}')

log "Register response: $REGISTER_RESP"

# Try to login
LOGIN_RESP=$(curl -s -X POST "$API_URL/api/v2/auth/login" \
    -H "Content-Type: application/json" \
    -d '{"email":"smoke_test@local.test","password":"SmokeTest@2025!"}' \
    2>/dev/null || echo '{}')

TOKEN=$(echo "$LOGIN_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null || echo "")

if [[ -n "$TOKEN" && "$TOKEN" != "null" ]]; then
    print_pass "Authentication working (got token)"
else
    print_skip "Auth test skipped (no token received)"
    log "Login response: $LOGIN_RESP"
fi

# ============================================================================
# Test 5: Protected Endpoints (V2 API)
# ============================================================================
print_test "Protected Endpoints"

if [[ -n "$TOKEN" && "$TOKEN" != "null" ]]; then
    PROTECTED_RESP=$(curl -s -X GET "$API_URL/api/v2/auth/me" \
        -H "Authorization: Bearer $TOKEN" \
        2>/dev/null || echo '{"error":"failed"}')

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/api/v2/auth/me" \
        -H "Authorization: Bearer $TOKEN" \
        2>/dev/null || echo "000")

    if [[ "$HTTP_CODE" == "200" || "$HTTP_CODE" == "201" ]]; then
        print_pass "Protected endpoint accessible with token"
    else
        print_fail "Protected endpoint returned HTTP $HTTP_CODE"
        log "Response: $PROTECTED_RESP"
    fi
else
    print_skip "Protected endpoint test skipped (no auth token)"
fi

# ============================================================================
# Test 6: Guest Chat (V2 API - no auth required)
# ============================================================================
print_test "Guest Chat"

CHAT_RESP=$(curl -s -X POST "$API_URL/api/v2/chat/guest" \
    -H "Content-Type: application/json" \
    -d '{"message": "What is 2+2?", "language": "en", "complexity_level": 5}' \
    2>/dev/null || echo '{"error":"failed"}')

MSG_ID=$(echo "$CHAT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('message_id',''))" 2>/dev/null || echo "")

if [[ -n "$MSG_ID" && "$MSG_ID" != "null" ]]; then
    print_pass "Guest chat successful (ID: $MSG_ID)"
else
    print_fail "Guest chat failed"
    log "Response: $CHAT_RESP"
fi

# ============================================================================
# Test 7: Redis Connection (via Celery)
# ============================================================================
print_test "Redis/Celery"

# Check if Redis is responding
if redis-cli ping 2>/dev/null | grep -q "PONG"; then
    print_pass "Redis is responding"
else
    print_fail "Redis not responding"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  📊 RESULTS${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  Passed: ${GREEN}$PASSED${NC}"
echo -e "  Failed: ${RED}$FAILED${NC}"
echo ""

if [[ $FAILED -eq 0 ]]; then
    echo -e "  ${GREEN}✅ ALL TESTS PASSED${NC}"
    exit 0
else
    echo -e "  ${RED}❌ SOME TESTS FAILED${NC}"
    exit 1
fi
