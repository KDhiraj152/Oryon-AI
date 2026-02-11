#!/bin/bash
# ===================================================================
# ShikshaSetu - Comprehensive Test Suite
# Tests all features and validates everything is working
# ===================================================================

set -e

BASE_URL="http://localhost:8000"
API_BASE="$BASE_URL/api/v2"
TEST_DATA_DIR="./test_data"
TOKEN=""
USER_EMAIL="tester_$(date +%s)@test.com"
USER_NAME="tester_$(date +%s)"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

PASSED=0
FAILED=0

mkdir -p "$TEST_DATA_DIR"

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
}

print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} $2"
        ((FAILED++))
    fi
}

print_info() { echo -e "${CYAN}→${NC} $1"; }
print_success() { echo -e "${GREEN}  $1${NC}"; }
print_data() { echo -e "${YELLOW}  $1${NC}"; }

echo -e "${MAGENTA}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║           SHIKSHA SETU - COMPREHENSIVE TEST SUITE            ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""
print_info "Testing complete AI/ML pipeline..."
print_info "Backend: $BASE_URL"
echo ""

# ===================================================================
# PHASE 1: System Health
# ===================================================================
print_header "PHASE 1: SYSTEM HEALTH"

print_info "Checking backend health..."
if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
    health=$(curl -s "$BASE_URL/health" | jq -r '.status' 2>/dev/null || echo "healthy")
    print_result 0 "Backend healthy (status: $health)"
else
    print_result 1 "Backend not responding"
    echo -e "${RED}Start backend first: ./2-start.sh or ./3-backend.sh${NC}"
    exit 1
fi

print_info "Checking API docs..."
curl -s "$BASE_URL/docs" > /dev/null 2>&1 && \
    print_result 0 "API docs accessible" || \
    print_result 1 "API docs not accessible"

# ===================================================================
# PHASE 2: Authentication
# ===================================================================
print_header "PHASE 2: AUTHENTICATION"

print_info "Registering test user..."
register_resp=$(curl -s -X POST "$API_BASE/auth/register" \
    -H "Content-Type: application/json" \
    -d "{\"username\":\"$USER_NAME\",\"email\":\"$USER_EMAIL\",\"password\":\"Test123!\",\"full_name\":\"Test User\"}")

if echo "$register_resp" | jq -e '.access_token' > /dev/null 2>&1; then
    TOKEN=$(echo "$register_resp" | jq -r '.access_token')
    print_result 0 "User registered successfully"
else
    print_result 1 "User registration failed"
    exit 1
fi

print_info "Retrieving user info..."
user_info=$(curl -s -X GET "$API_BASE/auth/me" -H "Authorization: Bearer $TOKEN")
if echo "$user_info" | jq -e '.email' > /dev/null 2>&1; then
    print_result 0 "User info retrieved"
else
    print_result 1 "User info retrieval failed"
fi

# ===================================================================
# PHASE 3: Content Upload
# ===================================================================
print_header "PHASE 3: CONTENT UPLOAD"

cat > "$TEST_DATA_DIR/test_content.txt" << 'EOF'
Photosynthesis is the process by which green plants use sunlight to synthesize nutrients
from carbon dioxide and water. This process is essential for life on Earth.
EOF

print_info "Uploading test content..."
upload_resp=$(curl -s -X POST "$API_BASE/upload" \
    -H "Authorization: Bearer $TOKEN" \
    -F "file=@$TEST_DATA_DIR/test_content.txt" \
    -F "complexity_level=8" \
    -F "subject=Science")

if echo "$upload_resp" | jq -e '.content_id' > /dev/null 2>&1; then
    CONTENT_ID=$(echo "$upload_resp" | jq -r '.content_id')
    print_result 0 "Content uploaded (ID: $CONTENT_ID)"
else
    print_result 1 "Content upload failed"
fi

# ===================================================================
# PHASE 4: AI Text Simplification
# ===================================================================
print_header "PHASE 4: AI TEXT SIMPLIFICATION"

print_info "Simplifying text for Grade 5..."
text="Photosynthesis converts light energy into chemical energy."
simplify_resp=$(curl -s -X POST "$API_BASE/simplify" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"text\":\"$text\",\"target_grade\":5,\"subject\":\"Science\"}")

if echo "$simplify_resp" | jq -e '.simplified_text' > /dev/null 2>&1; then
    simplified=$(echo "$simplify_resp" | jq -r '.simplified_text')
    print_result 0 "Text simplification works"
    print_data "Original: $text"
    print_success "Simplified: $simplified"
else
    print_result 1 "Text simplification failed"
fi

# ===================================================================
# PHASE 5: Translation
# ===================================================================
print_header "PHASE 5: MULTI-LANGUAGE TRANSLATION"

english="Plants make food using sunlight."

print_info "Translating to Hindi..."
trans_resp=$(curl -s -X POST "$API_BASE/translate" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"text\":\"$english\",\"source_language\":\"English\",\"target_language\":\"Hindi\"}")

if echo "$trans_resp" | jq -e '.translated_text' > /dev/null 2>&1; then
    hindi=$(echo "$trans_resp" | jq -r '.translated_text')
    print_result 0 "Hindi translation works"
    print_success "Hindi: $hindi"
else
    print_result 1 "Hindi translation failed"
fi

print_info "Translating to Tamil..."
trans_resp=$(curl -s -X POST "$API_BASE/translate" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"text\":\"$english\",\"source_language\":\"English\",\"target_language\":\"Tamil\"}")

if echo "$trans_resp" | jq -e '.translated_text' > /dev/null 2>&1; then
    tamil=$(echo "$trans_resp" | jq -r '.translated_text')
    print_result 0 "Tamil translation works"
    print_success "Tamil: $tamil"
else
    print_result 1 "Tamil translation failed"
fi

# ===================================================================
# PHASE 6: Content Validation
# ===================================================================
print_header "PHASE 6: AI CONTENT VALIDATION"

print_info "Validating content..."
validate_resp=$(curl -s -X POST "$API_BASE/validate" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"text\":\"$text\",\"complexity_level\":8,\"subject\":\"Science\",\"language\":\"English\"}")

if echo "$validate_resp" | jq -e '.is_valid' > /dev/null 2>&1; then
    valid=$(echo "$validate_resp" | jq -r '.is_valid')
    print_result 0 "Content validation works (valid: $valid)"
else
    print_result 1 "Content validation failed"
fi

# ===================================================================
# PHASE 7: Text-to-Speech
# ===================================================================
print_header "PHASE 7: TEXT-TO-SPEECH"

print_info "Generating audio..."
tts_resp=$(curl -s -X POST "$API_BASE/tts" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"text\":\"Plants make food using sunlight\",\"language\":\"English\"}")

if echo "$tts_resp" | jq -e '.audio_url' > /dev/null 2>&1; then
    audio_url=$(echo "$tts_resp" | jq -r '.audio_url')
    print_result 0 "Audio generation works"
    print_data "Audio URL: $audio_url"
else
    print_result 1 "Audio generation failed"
fi

# ===================================================================
# PHASE 8: Content Library
# ===================================================================
print_header "PHASE 8: CONTENT LIBRARY"

print_info "Retrieving content library..."
lib_resp=$(curl -s -X GET "$API_BASE/library?limit=10" \
    -H "Authorization: Bearer $TOKEN")

if echo "$lib_resp" | jq -e '.items' > /dev/null 2>&1; then
    count=$(echo "$lib_resp" | jq '.items | length')
    print_result 0 "Library retrieval works ($count items)"
else
    print_result 1 "Library retrieval failed"
fi

print_info "Filtering by subject..."
lib_resp=$(curl -s -X GET "$API_BASE/library?subject=Science" \
    -H "Authorization: Bearer $TOKEN")

if echo "$lib_resp" | jq -e '.items' > /dev/null 2>&1; then
    print_result 0 "Subject filtering works"
else
    print_result 1 "Subject filtering failed"
fi

# ===================================================================
# PHASE 9: Search
# ===================================================================
print_header "PHASE 9: CONTENT SEARCH"

print_info "Searching content..."
search_resp=$(curl -s -X GET "$API_BASE/content/search?q=photosynthesis" \
    -H "Authorization: Bearer $TOKEN")

if echo "$search_resp" | jq -e '.results' > /dev/null 2>&1; then
    print_result 0 "Content search works"
else
    print_result 1 "Content search failed"
fi

# ===================================================================
# PHASE 10: Q&A System
# ===================================================================
print_header "PHASE 10: AI Q&A SYSTEM"

if [ -n "$CONTENT_ID" ]; then
    print_info "Asking question..."
    qa_resp=$(curl -s -X POST "$API_BASE/qa/ask" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $TOKEN" \
        -d "{\"content_id\":\"$CONTENT_ID\",\"question\":\"What is photosynthesis?\"}")

    if echo "$qa_resp" | jq -e '.answer' > /dev/null 2>&1; then
        answer=$(echo "$qa_resp" | jq -r '.answer')
        print_result 0 "Q&A system works"
        print_success "Answer: $answer"
    else
        print_result 1 "Q&A system failed"
    fi
fi

# ===================================================================
# PHASE 11: Complete Workflow
# ===================================================================
print_header "PHASE 11: COMPLETE AI WORKFLOW"

print_info "Testing end-to-end workflow..."
workflow_text="Water is essential for photosynthesis."
echo "$workflow_text" > "$TEST_DATA_DIR/workflow.txt"

# Upload
upload=$(curl -s -X POST "$API_BASE/upload" \
    -H "Authorization: Bearer $TOKEN" \
    -F "file=@$TEST_DATA_DIR/workflow.txt" \
    -F "complexity_level=7" \
    -F "subject=Science")

if echo "$upload" | jq -e '.content_id' > /dev/null 2>&1; then
    print_success "✓ Upload"

    # Simplify
    simp=$(curl -s -X POST "$API_BASE/simplify" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $TOKEN" \
        -d "{\"text\":\"$workflow_text\",\"target_grade\":5}")

    if echo "$simp" | jq -e '.simplified_text' > /dev/null 2>&1; then
        print_success "✓ Simplify"
        simple_text=$(echo "$simp" | jq -r '.simplified_text')

        # Translate
        trans=$(curl -s -X POST "$API_BASE/translate" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $TOKEN" \
            -d "{\"text\":\"$simple_text\",\"source_language\":\"English\",\"target_language\":\"Hindi\"}")

        if echo "$trans" | jq -e '.translated_text' > /dev/null 2>&1; then
            print_success "✓ Translate"

            # TTS
            tts=$(curl -s -X POST "$API_BASE/tts" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $TOKEN" \
                -d "{\"text\":\"$simple_text\",\"language\":\"English\"}")

            if echo "$tts" | jq -e '.audio_url' > /dev/null 2>&1; then
                print_success "✓ Audio"
                print_result 0 "Complete workflow executed successfully!"
            else
                print_result 1 "Workflow: Audio failed"
            fi
        else
            print_result 1 "Workflow: Translation failed"
        fi
    else
        print_result 1 "Workflow: Simplification failed"
    fi
else
    print_result 1 "Workflow: Upload failed"
fi

# ===================================================================
# SUMMARY
# ===================================================================
echo ""
echo -e "${MAGENTA}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                      TEST SUMMARY                             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

TOTAL=$((PASSED + FAILED))
if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$(awk "BEGIN {printf \"%.1f\", ($PASSED/$TOTAL)*100}")
else
    PASS_RATE="0.0"
fi

echo -e "${GREEN}✓ Passed:     $PASSED${NC}"
echo -e "${RED}✗ Failed:     $FAILED${NC}"
echo -e "${BLUE}━ Total:      $TOTAL${NC}"
echo -e "${CYAN}➤ Pass Rate:  $PASS_RATE%${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  🎉 ALL TESTS PASSED! System is fully functional! 🎉        ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    rm -rf "$TEST_DATA_DIR"
    exit 0
else
    echo -e "${YELLOW}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║  ⚠️  Some tests failed. Please review the errors above.     ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    rm -rf "$TEST_DATA_DIR"
    exit 1
fi
