#!/bin/bash
# System Validation Script - Tests all critical components

# Don't exit on error - we want to run all tests
# set -e

echo "=========================================="
echo "ShikshaSetu System Validation"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0

test_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

test_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

test_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Test 1: PostgreSQL Database
echo "Testing Database Connection..."
if docker exec shiksha-postgres psql -U shiksha_user -d shiksha_setu -c "SELECT 1" > /dev/null 2>&1; then
    test_pass "PostgreSQL is running and accessible"
else
    test_fail "PostgreSQL connection failed"
fi

# Test 2: Redis
echo "Testing Redis..."
if docker ps | grep -q "redis"; then
    test_pass "Redis container is running"
else
    test_warn "Redis container not found (optional for basic functionality)"
fi

# Test 3: Backend API Health
echo "Testing Backend API..."
if curl -s http://127.0.0.1:8000/health | grep -q "healthy"; then
    test_pass "Backend API is healthy"
else
    test_fail "Backend API health check failed"
fi

# Test 4: Backend API Documentation
echo "Testing API Documentation..."
if curl -s http://127.0.0.1:8000/docs | grep -q "Swagger"; then
    test_pass "API documentation is accessible"
else
    test_fail "API documentation not accessible"
fi

# Test 5: Frontend (if running)
echo "Testing Frontend..."
if curl -s http://127.0.0.1:5173 > /dev/null 2>&1; then
    test_pass "Frontend is accessible"
else
    test_warn "Frontend not running (start with: npm run dev)"
fi

# Test 6: Database Tables
echo "Testing Database Schema..."
TABLES=$(docker exec shiksha-postgres psql -U shiksha_user -d shiksha_setu -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'" 2>/dev/null | tr -d ' ')
if [ "$TABLES" -gt 10 ]; then
    test_pass "Database schema created ($TABLES tables)"
else
    test_fail "Database schema incomplete"
fi

# Test 7: pgvector Extension
echo "Testing pgvector Extension..."
if docker exec shiksha-postgres psql -U shiksha_user -d shiksha_setu -t -c "SELECT 1 FROM pg_extension WHERE extname='vector'" 2>/dev/null | grep -q 1; then
    test_pass "pgvector extension enabled"
else
    test_fail "pgvector extension not enabled"
fi

# Test 8: Python Environment
echo "Testing Python Environment..."
if [ -d "venv" ]; then
    test_pass "Python virtual environment exists"
else
    test_fail "Python virtual environment not found"
fi

# Test 9: Python Dependencies
echo "Testing Python Dependencies..."
source venv/bin/activate 2>/dev/null || true
if python -c "import fastapi, sqlalchemy, transformers" 2>/dev/null; then
    test_pass "Core Python dependencies installed"
else
    test_fail "Missing core Python dependencies"
fi

# Test 10: Node Dependencies (Frontend)
echo "Testing Node Dependencies..."
if [ -d "frontend/node_modules" ]; then
    test_pass "Node dependencies installed"
else
    test_warn "Node dependencies not installed (run: cd frontend && npm install)"
fi

echo ""
echo "=========================================="
echo "Summary:"
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ System is fully operational!${NC}"
    exit 0
else
    echo -e "${RED}✗ System has $FAILED issues${NC}"
    exit 1
fi
