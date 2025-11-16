#!/bin/bash

# Shiksha Setu - Complete Pipeline Test
# Simulates a real user journey from registration to using all AI features

set -e

BASE_URL="http://localhost:8000"
API_BASE="$BASE_URL/api/v1"
TEST_DATA_DIR="./test_data"
TOKEN=""
USER_EMAIL="realuser_$(date +%s)@test.com"
USER_NAME="realuser_$(date +%s)"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Test counters
PASSED=0
FAILED=0

# Create test data directory
mkdir -p "$TEST_DATA_DIR"

# Helper function to print section headers
print_header() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  $1${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"
}

# Helper function to print test results
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED${NC}: $2"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}: $2"
        ((FAILED++))
    fi
}

# Helper function to print info
print_info() {
    echo -e "${CYAN}→ $1${NC}"
}

# Helper function to print success
print_success() {
    echo -e "${GREEN}$1${NC}"
}

# Helper function to print data
print_data() {
    echo -e "${YELLOW}$1${NC}"
}

# Create sample educational content
create_sample_content() {
    cat > "$TEST_DATA_DIR/science_lesson.txt" << 'EOF'
Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water. 
Chlorophyll, the green pigment in plants, captures light energy. This energy converts water and carbon dioxide into glucose 
and oxygen. Plants use glucose for energy and growth. The oxygen is released into the atmosphere. This process is essential 
for life on Earth as it produces oxygen and removes carbon dioxide.
EOF

    cat > "$TEST_DATA_DIR/math_lesson.txt" << 'EOF'
A quadratic equation is a polynomial equation of degree two. The standard form is ax² + bx + c = 0, where a, b, and c are 
constants and a ≠ 0. The solutions can be found using the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a. The discriminant 
(b² - 4ac) determines the nature of the roots. If positive, there are two real roots. If zero, one repeated root. If negative, 
two complex roots.
EOF

    cat > "$TEST_DATA_DIR/history_lesson.txt" << 'EOF'
The Indian Independence Movement was a series of historic events aimed at ending British rule in India. The movement spanned 
from 1857 to 1947 and involved various methods including non-violent resistance, civil disobedience, and armed struggle. 
Mahatma Gandhi led the non-cooperation movement and the Salt March. The Quit India Movement in 1942 was a decisive call for 
independence. India finally gained independence on August 15, 1947.
EOF

    print_info "Sample educational content created"
}

echo -e "${MAGENTA}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║                                                                      ║${NC}"
echo -e "${MAGENTA}║           SHIKSHA SETU - COMPLETE PIPELINE TEST                      ║${NC}"
echo -e "${MAGENTA}║           Real User Journey Simulation                               ║${NC}"
echo -e "${MAGENTA}║                                                                      ║${NC}"
echo -e "${MAGENTA}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
print_info "Testing complete AI/ML pipeline for Indian education..."
print_info "Date: $(date)"
print_info "Backend: $BASE_URL"
echo ""

# ============================================================================
# PHASE 1: SYSTEM HEALTH CHECK
# ============================================================================

print_header "PHASE 1: SYSTEM HEALTH CHECK"

# Test 1: Backend Health
print_info "Checking if backend is running..."
if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
    health_status=$(curl -s "$BASE_URL/health" | jq -r '.status')
    print_result 0 "Backend is healthy (status: $health_status)"
else
    print_result 1 "Backend is not responding"
    echo -e "${RED}ERROR: Please start the backend first with:${NC}"
    echo -e "${YELLOW}uvicorn src.api.async_app:app --host 0.0.0.0 --port 8000${NC}"
    exit 1
fi

# Test 2: API Documentation
print_info "Checking API documentation..."
if curl -s "$BASE_URL/docs" > /dev/null 2>&1; then
    print_result 0 "API documentation is accessible at $BASE_URL/docs"
else
    print_info "API documentation endpoint not critical, skipping..."
fi

# ============================================================================
# PHASE 2: USER AUTHENTICATION & REGISTRATION
# ============================================================================

print_header "PHASE 2: USER AUTHENTICATION & REGISTRATION"

# Test 3: User Registration
print_info "Registering new user..."
print_data "  Email: $USER_EMAIL"
print_data "  Username: $USER_NAME"

register_response=$(curl -s -X POST "$API_BASE/auth/register" \
    -H "Content-Type: application/json" \
    -d "{\"username\":\"$USER_NAME\",\"email\":\"$USER_EMAIL\",\"password\":\"SecurePass123!\",\"full_name\":\"Real Test User\"}")

if echo "$register_response" | jq -e '.access_token' > /dev/null 2>&1; then
    TOKEN=$(echo "$register_response" | jq -r '.access_token')
    print_result 0 "User registration successful"
    print_success "  Access token received: ${TOKEN:0:20}..."
else
    print_result 1 "User registration failed"
    echo -e "${RED}Response: $register_response${NC}"
    exit 1
fi

# Test 4: User Info Retrieval
print_info "Retrieving user information..."
user_info=$(curl -s -X GET "$API_BASE/auth/me" \
    -H "Authorization: Bearer $TOKEN")

if echo "$user_info" | jq -e '.email' > /dev/null 2>&1; then
    email=$(echo "$user_info" | jq -r '.email')
    full_name=$(echo "$user_info" | jq -r '.full_name')
    print_result 0 "User info retrieved successfully"
    print_data "  Name: $full_name"
    print_data "  Email: $email"
else
    print_result 1 "Failed to retrieve user info"
fi

# ============================================================================
# PHASE 3: CONTENT UPLOAD & EXTRACTION
# ============================================================================

print_header "PHASE 3: CONTENT UPLOAD & TEXT EXTRACTION"

create_sample_content

# Test 5: Upload Science Content
print_info "Uploading science lesson (Photosynthesis)..."
upload_response=$(curl -s -X POST "$API_BASE/upload" \
    -H "Authorization: Bearer $TOKEN" \
    -F "file=@$TEST_DATA_DIR/science_lesson.txt" \
    -F "grade_level=8" \
    -F "subject=Science")

if echo "$upload_response" | jq -e '.content_id' > /dev/null 2>&1; then
    science_content_id=$(echo "$upload_response" | jq -r '.content_id')
    extracted_text=$(echo "$upload_response" | jq -r '.extracted_text')
    print_result 0 "Science content uploaded successfully"
    print_data "  Content ID: $science_content_id"
    print_data "  Extracted: ${extracted_text:0:100}..."
else
    print_result 1 "Science content upload failed"
fi

# Test 6: Upload Math Content
print_info "Uploading math lesson (Quadratic Equations)..."
upload_response=$(curl -s -X POST "$API_BASE/upload" \
    -H "Authorization: Bearer $TOKEN" \
    -F "file=@$TEST_DATA_DIR/math_lesson.txt" \
    -F "grade_level=10" \
    -F "subject=Mathematics")

if echo "$upload_response" | jq -e '.content_id' > /dev/null 2>&1; then
    math_content_id=$(echo "$upload_response" | jq -r '.content_id')
    print_result 0 "Math content uploaded successfully"
    print_data "  Content ID: $math_content_id"
else
    print_result 1 "Math content upload failed"
fi

# ============================================================================
# PHASE 4: TEXT SIMPLIFICATION (AI Feature 1)
# ============================================================================

print_header "PHASE 4: AI TEXT SIMPLIFICATION"

# Test 7: Simplify for Grade 5
print_info "Simplifying science text for Grade 5 students..."
original_text="Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose."

simplify_response=$(curl -s -X POST "$API_BASE/simplify" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"text\":\"$original_text\",\"target_grade\":5,\"subject\":\"Science\"}")

if echo "$simplify_response" | jq -e '.simplified_text' > /dev/null 2>&1; then
    simplified=$(echo "$simplify_response" | jq -r '.simplified_text')
    grade=$(echo "$simplify_response" | jq -r '.grade_level // "N/A"')
    complexity=$(echo "$simplify_response" | jq -r '.complexity_score // "N/A"')
    print_result 0 "Text simplified successfully"
    print_data "  Target Grade: $grade"
    print_data "  Complexity Score: $complexity"
    print_success "  Original: $original_text"
    print_success "  Simplified: $simplified"
else
    print_result 1 "Text simplification failed"
fi

# Test 8: Simplify for Grade 10
print_info "Simplifying math text for Grade 10 students..."
math_text="The discriminant of a quadratic equation determines the nature of its roots."

simplify_response=$(curl -s -X POST "$API_BASE/simplify" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"text\":\"$math_text\",\"target_grade\":10,\"subject\":\"Mathematics\"}")

if echo "$simplify_response" | jq -e '.simplified_text' > /dev/null 2>&1; then
    simplified=$(echo "$simplify_response" | jq -r '.simplified_text')
    print_result 0 "Math text simplified successfully"
    print_success "  Simplified: $simplified"
else
    print_result 1 "Math text simplification failed"
fi

# ============================================================================
# PHASE 5: MULTI-LANGUAGE TRANSLATION (AI Feature 2)
# ============================================================================

print_header "PHASE 5: AI MULTI-LANGUAGE TRANSLATION"

english_text="Plants make their own food using sunlight, water, and carbon dioxide."

# Test 9: Translate to Hindi
print_info "Translating to Hindi (हिंदी)..."
translate_response=$(curl -s -X POST "$API_BASE/translate" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"text\":\"$english_text\",\"source_language\":\"English\",\"target_language\":\"Hindi\"}")

if echo "$translate_response" | jq -e '.translated_text' > /dev/null 2>&1; then
    hindi_text=$(echo "$translate_response" | jq -r '.translated_text')
    print_result 0 "Hindi translation successful"
    print_data "  English: $english_text"
    print_success "  हिंदी: $hindi_text"
else
    print_result 1 "Hindi translation failed"
fi

# Test 10: Translate to Tamil
print_info "Translating to Tamil (தமிழ்)..."
translate_response=$(curl -s -X POST "$API_BASE/translate" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"text\":\"$english_text\",\"source_language\":\"English\",\"target_language\":\"Tamil\"}")

if echo "$translate_response" | jq -e '.translated_text' > /dev/null 2>&1; then
    tamil_text=$(echo "$translate_response" | jq -r '.translated_text')
    print_result 0 "Tamil translation successful"
    print_success "  தமிழ்: $tamil_text"
else
    print_result 1 "Tamil translation failed"
fi

# Test 11: Translate to Telugu
print_info "Translating to Telugu (తెలుగు)..."
translate_response=$(curl -s -X POST "$API_BASE/translate" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"text\":\"$english_text\",\"source_language\":\"English\",\"target_language\":\"Telugu\"}")

if echo "$translate_response" | jq -e '.translated_text' > /dev/null 2>&1; then
    telugu_text=$(echo "$translate_response" | jq -r '.translated_text')
    print_result 0 "Telugu translation successful"
    print_success "  తెలుగు: $telugu_text"
else
    print_result 1 "Telugu translation failed"
fi

# Test 12: Translate to Bengali
print_info "Translating to Bengali (বাংলা)..."
translate_response=$(curl -s -X POST "$API_BASE/translate" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"text\":\"$english_text\",\"source_language\":\"English\",\"target_language\":\"Bengali\"}")

if echo "$translate_response" | jq -e '.translated_text' > /dev/null 2>&1; then
    bengali_text=$(echo "$translate_response" | jq -r '.translated_text')
    print_result 0 "Bengali translation successful"
    print_success "  বাংলা: $bengali_text"
else
    print_result 1 "Bengali translation failed"
fi

# ============================================================================
# PHASE 6: CONTENT VALIDATION (AI Feature 3)
# ============================================================================

print_header "PHASE 6: AI CONTENT VALIDATION"

# Test 13: Validate Educational Content
print_info "Validating content accuracy and appropriateness..."
content_to_validate="Water is essential for photosynthesis. Plants absorb water through their roots."

validate_response=$(curl -s -X POST "$API_BASE/validate" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"text\":\"$content_to_validate\",\"grade_level\":8,\"subject\":\"Science\",\"language\":\"English\"}")

if echo "$validate_response" | jq -e '.is_valid' > /dev/null 2>&1; then
    is_valid=$(echo "$validate_response" | jq -r '.is_valid')
    accuracy=$(echo "$validate_response" | jq -r '.accuracy_score // "N/A"')
    alignment=$(echo "$validate_response" | jq -r '.ncert_alignment_score // "N/A"')
    print_result 0 "Content validation successful"
    print_data "  Valid: $is_valid"
    print_data "  Accuracy Score: $accuracy"
    print_data "  NCERT Alignment: $alignment"
else
    print_result 1 "Content validation failed"
fi

# Test 14: Validate with Issues
print_info "Validating content with potential issues..."
problematic_content="This is completely incorrect information about science that should be flagged."

validate_response=$(curl -s -X POST "$API_BASE/validate" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"text\":\"$problematic_content\",\"grade_level\":8,\"subject\":\"Science\",\"language\":\"English\"}")

if echo "$validate_response" | jq -e '.is_valid' > /dev/null 2>&1; then
    is_valid=$(echo "$validate_response" | jq -r '.is_valid')
    print_result 0 "Problematic content validation completed"
    print_data "  Valid: $is_valid (should be false or have low score)"
else
    print_result 1 "Problematic content validation failed"
fi

# ============================================================================
# PHASE 7: TEXT-TO-SPEECH (AI Feature 4)
# ============================================================================

print_header "PHASE 7: AI TEXT-TO-SPEECH GENERATION"

# Test 15: Generate Hindi Audio
print_info "Generating Hindi audio (हिंदी में ऑडियो)..."
tts_text="पौधे प्रकाश संश्लेषण द्वारा अपना भोजन बनाते हैं।"

tts_response=$(curl -s -X POST "$API_BASE/tts" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"text\":\"$tts_text\",\"language\":\"Hindi\"}")

if echo "$tts_response" | jq -e '.audio_url' > /dev/null 2>&1; then
    audio_url=$(echo "$tts_response" | jq -r '.audio_url')
    duration=$(echo "$tts_response" | jq -r '.duration // "N/A"')
    print_result 0 "Hindi audio generated successfully"
    print_data "  Audio URL: $audio_url"
    print_data "  Duration: ${duration}s"
else
    print_result 1 "Hindi audio generation failed"
fi

# Test 16: Generate English Audio
print_info "Generating English audio..."
tts_text_en="Photosynthesis is how plants make food using sunlight."

tts_response=$(curl -s -X POST "$API_BASE/tts" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"text\":\"$tts_text_en\",\"language\":\"English\"}")

if echo "$tts_response" | jq -e '.audio_url' > /dev/null 2>&1; then
    audio_url=$(echo "$tts_response" | jq -r '.audio_url')
    print_result 0 "English audio generated successfully"
    print_data "  Audio URL: $audio_url"
else
    print_result 1 "English audio generation failed"
fi

# ============================================================================
# PHASE 8: CONTENT LIBRARY & RETRIEVAL
# ============================================================================

print_header "PHASE 8: CONTENT LIBRARY MANAGEMENT"

# Test 17: Get Content Library
print_info "Retrieving user's content library..."
library_response=$(curl -s -X GET "$API_BASE/library?limit=10" \
    -H "Authorization: Bearer $TOKEN")

if echo "$library_response" | jq -e '.items' > /dev/null 2>&1; then
    item_count=$(echo "$library_response" | jq '.items | length')
    total=$(echo "$library_response" | jq -r '.total')
    print_result 0 "Content library retrieved successfully"
    print_data "  Items in current page: $item_count"
    print_data "  Total items: $total"
else
    print_result 1 "Content library retrieval failed"
fi

# Test 18: Filter by Grade Level
print_info "Filtering content by grade level..."
library_response=$(curl -s -X GET "$API_BASE/library?grade=8" \
    -H "Authorization: Bearer $TOKEN")

if echo "$library_response" | jq -e '.items' > /dev/null 2>&1; then
    item_count=$(echo "$library_response" | jq '.items | length')
    print_result 0 "Grade level filtering works"
    print_data "  Grade 8 items: $item_count"
else
    print_result 1 "Grade level filtering failed"
fi

# Test 19: Filter by Subject
print_info "Filtering content by subject..."
library_response=$(curl -s -X GET "$API_BASE/library?subject=Science" \
    -H "Authorization: Bearer $TOKEN")

if echo "$library_response" | jq -e '.items' > /dev/null 2>&1; then
    item_count=$(echo "$library_response" | jq '.items | length')
    print_result 0 "Subject filtering works"
    print_data "  Science items: $item_count"
else
    print_result 1 "Subject filtering failed"
fi

# Test 20: Retrieve Specific Content
if [ -n "$science_content_id" ]; then
    print_info "Retrieving specific content by ID..."
    content_response=$(curl -s -X GET "$API_BASE/content/$science_content_id" \
        -H "Authorization: Bearer $TOKEN")
    
    if echo "$content_response" | jq -e '.id' > /dev/null 2>&1; then
        subject=$(echo "$content_response" | jq -r '.subject')
        grade=$(echo "$content_response" | jq -r '.grade_level')
        print_result 0 "Specific content retrieved successfully"
        print_data "  Subject: $subject"
        print_data "  Grade: $grade"
    else
        print_result 1 "Specific content retrieval failed"
    fi
fi

# ============================================================================
# PHASE 9: SEARCH FUNCTIONALITY
# ============================================================================

print_header "PHASE 9: CONTENT SEARCH"

# Test 21: Search by Keyword
print_info "Searching for 'photosynthesis'..."
search_response=$(curl -s -X GET "$API_BASE/content/search?q=photosynthesis&limit=5" \
    -H "Authorization: Bearer $TOKEN")

if echo "$search_response" | jq -e '.results' > /dev/null 2>&1; then
    result_count=$(echo "$search_response" | jq '.results | length')
    print_result 0 "Content search successful"
    print_data "  Results found: $result_count"
else
    print_result 1 "Content search failed"
fi

# ============================================================================
# PHASE 10: Q&A SYSTEM (RAG)
# ============================================================================

print_header "PHASE 10: AI Q&A SYSTEM (RAG)"

if [ -n "$science_content_id" ]; then
    # Test 22: Ask Question about Content
    print_info "Asking question about uploaded content..."
    qa_response=$(curl -s -X POST "$API_BASE/qa/ask" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $TOKEN" \
        -d "{\"content_id\":\"$science_content_id\",\"question\":\"What is photosynthesis?\"}")
    
    if echo "$qa_response" | jq -e '.answer' > /dev/null 2>&1; then
        answer=$(echo "$qa_response" | jq -r '.answer')
        confidence=$(echo "$qa_response" | jq -r '.confidence // "N/A"')
        print_result 0 "Q&A system responded successfully"
        print_data "  Question: What is photosynthesis?"
        print_success "  Answer: $answer"
        print_data "  Confidence: $confidence"
    else
        print_result 1 "Q&A system failed"
    fi
    
    # Test 23: Ask Follow-up Question
    print_info "Asking follow-up question..."
    qa_response=$(curl -s -X POST "$API_BASE/qa/ask" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $TOKEN" \
        -d "{\"content_id\":\"$science_content_id\",\"question\":\"Why is oxygen important?\"}")
    
    if echo "$qa_response" | jq -e '.answer' > /dev/null 2>&1; then
        answer=$(echo "$qa_response" | jq -r '.answer')
        print_result 0 "Follow-up question handled successfully"
        print_success "  Answer: $answer"
    else
        print_result 1 "Follow-up question failed"
    fi
fi

# ============================================================================
# PHASE 11: COMBINED WORKFLOW
# ============================================================================

print_header "PHASE 11: COMPLETE AI WORKFLOW"

# Test 24: Full Pipeline - Upload → Simplify → Translate → Validate → TTS
print_info "Testing complete AI pipeline workflow..."
print_data "  1. Uploading new content..."

workflow_text="The mitochondria is the powerhouse of the cell. It produces energy through cellular respiration."
echo "$workflow_text" > "$TEST_DATA_DIR/workflow_test.txt"

upload_resp=$(curl -s -X POST "$API_BASE/upload" \
    -H "Authorization: Bearer $TOKEN" \
    -F "file=@$TEST_DATA_DIR/workflow_test.txt" \
    -F "grade_level=9" \
    -F "subject=Science")

if echo "$upload_resp" | jq -e '.content_id' > /dev/null 2>&1; then
    workflow_id=$(echo "$upload_resp" | jq -r '.content_id')
    print_success "     ✓ Content uploaded"
    
    print_data "  2. Simplifying for Grade 6..."
    simplify_resp=$(curl -s -X POST "$API_BASE/simplify" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $TOKEN" \
        -d "{\"text\":\"$workflow_text\",\"target_grade\":6,\"subject\":\"Science\"}")
    
    if echo "$simplify_resp" | jq -e '.simplified_text' > /dev/null 2>&1; then
        simplified_workflow=$(echo "$simplify_resp" | jq -r '.simplified_text')
        print_success "     ✓ Text simplified"
        
        print_data "  3. Translating to Hindi..."
        translate_resp=$(curl -s -X POST "$API_BASE/translate" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $TOKEN" \
            -d "{\"text\":\"$simplified_workflow\",\"source_language\":\"English\",\"target_language\":\"Hindi\"}")
        
        if echo "$translate_resp" | jq -e '.translated_text' > /dev/null 2>&1; then
            translated_workflow=$(echo "$translate_resp" | jq -r '.translated_text')
            print_success "     ✓ Text translated"
            
            print_data "  4. Validating content..."
            validate_resp=$(curl -s -X POST "$API_BASE/validate" \
                -H "Content-Type: application/json" \
                -H "Authorization: Bearer $TOKEN" \
                -d "{\"text\":\"$simplified_workflow\",\"grade_level\":6,\"subject\":\"Science\",\"language\":\"English\"}")
            
            if echo "$validate_resp" | jq -e '.is_valid' > /dev/null 2>&1; then
                print_success "     ✓ Content validated"
                
                print_data "  5. Generating audio..."
                tts_resp=$(curl -s -X POST "$API_BASE/tts" \
                    -H "Content-Type: application/json" \
                    -H "Authorization: Bearer $TOKEN" \
                    -d "{\"text\":\"$translated_workflow\",\"language\":\"Hindi\"}")
                
                if echo "$tts_resp" | jq -e '.audio_url' > /dev/null 2>&1; then
                    print_success "     ✓ Audio generated"
                    print_result 0 "Complete AI workflow executed successfully!"
                    print_success "     Final Hindi translation: $translated_workflow"
                else
                    print_result 1 "Audio generation step failed"
                fi
            else
                print_result 1 "Validation step failed"
            fi
        else
            print_result 1 "Translation step failed"
        fi
    else
        print_result 1 "Simplification step failed"
    fi
else
    print_result 1 "Upload step failed in workflow"
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo -e "${MAGENTA}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║                                                                      ║${NC}"
echo -e "${MAGENTA}║                     TEST SUMMARY                                     ║${NC}"
echo -e "${MAGENTA}║                                                                      ║${NC}"
echo -e "${MAGENTA}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

TOTAL=$((PASSED + FAILED))
PASS_RATE=$(awk "BEGIN {printf \"%.1f\", ($PASSED/$TOTAL)*100}")

echo -e "${GREEN}✓ Passed Tests:  $PASSED${NC}"
echo -e "${RED}✗ Failed Tests:  $FAILED${NC}"
echo -e "${BLUE}━ Total Tests:   $TOTAL${NC}"
echo -e "${CYAN}➤ Pass Rate:     $PASS_RATE%${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  🎉 ALL TESTS PASSED! Pipeline is fully functional! 🎉              ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo -e "${YELLOW}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  ⚠️  Some tests failed. Please review the errors above.             ║${NC}"
    echo -e "${YELLOW}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
