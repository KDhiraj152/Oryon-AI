#!/bin/bash
set -e

BASE_URL="http://localhost:8000"
API_BASE="$BASE_URL/api/v1"

echo "=== Quick Pipeline Test ==="
echo ""

# 1. Test health
echo "1. Testing health endpoint..."
curl -s $BASE_URL/health | jq -r '.status' && echo "✓ Health check passed" || echo "✗ Health check failed"

# 2. Test auth
echo ""
echo "2. Testing authentication..."
TOKEN=$(curl -s -X POST "$API_BASE/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"username":"quicktest","email":"quicktest@test.com","password":"test123","full_name":"Quick Test"}' \
  | jq -r '.access_token // empty')

if [ -z "$TOKEN" ]; then
  TOKEN=$(curl -s -X POST "$API_BASE/auth/login" \
    -H "Content-Type: application/json" \
    -d '{"username":"quicktest","password":"test123"}' \
    | jq -r '.access_token')
fi

[ -n "$TOKEN" ] && echo "✓ Authentication successful" || (echo "✗ Authentication failed" && exit 1)

# 3. Test simplification
echo ""
echo "3. Testing text simplification..."
RESULT=$(curl -s -X POST "$API_BASE/simplify" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"text":"Photosynthesis is the process by which plants convert sunlight into energy.","target_grade":5}' \
  | jq -r '.simplified_text // empty')

[ -n "$RESULT" ] && echo "✓ Simplification successful" || echo "✗ Simplification failed"

# 4. Test translation
echo ""
echo "4. Testing translation..."
RESULT=$(curl -s -X POST "$API_BASE/translate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"text":"Plants make food using sunlight.","source_language":"English","target_language":"Hindi"}' \
  | jq -r '.translated_text // empty')

[ -n "$RESULT" ] && echo "✓ Translation successful" || echo "✗ Translation failed"

# 5. Test library
echo ""
echo "5. Testing library endpoint..."
curl -s -X GET "$API_BASE/library" \
  -H "Authorization: Bearer $TOKEN" | jq -r '.items | length' > /dev/null && echo "✓ Library endpoint working" || echo "✗ Library endpoint failed"

echo ""
echo "=== Quick Test Complete ==="
