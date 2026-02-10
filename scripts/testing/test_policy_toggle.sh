#!/bin/bash
# =============================================================================
# test_policy_toggle.sh - Policy Toggle Unit Tests
# =============================================================================
#
# Tests that the PolicyEngine correctly toggles behavior between
# RESTRICTED (default) and UNRESTRICTED modes.
#
# Usage:
#   ./scripts/test_policy_toggle.sh
#
# Exit Codes:
#   0 - All tests passed
#   1 - One or more tests failed
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  Policy Toggle Test Suite${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Ensure we have a virtual environment
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found. Run setup.sh first.${NC}"
    exit 1
fi

source venv/bin/activate

PASSED=0
FAILED=0

run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local expected_exit="$3"

    echo -e "${YELLOW}Running: $test_name${NC}"

    if eval "$test_cmd" > /tmp/test_output.txt 2>&1; then
        actual_exit=0
    else
        actual_exit=$?
    fi

    if [ "$actual_exit" == "$expected_exit" ]; then
        echo -e "${GREEN}✓ PASSED: $test_name${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAILED: $test_name (expected exit $expected_exit, got $actual_exit)${NC}"
        cat /tmp/test_output.txt
        ((FAILED++))
    fi
}

# =============================================================================
# Test 1: Policy module imports correctly
# =============================================================================
run_test "Policy module imports" \
    "python -c 'from backend.policy import PolicyEngine, PolicyMode, get_policy_engine'" \
    0

# =============================================================================
# Test 2: Default mode is RESTRICTED
# =============================================================================
run_test "Default mode is RESTRICTED" \
    "python -c '
import os
os.environ.pop(\"ALLOW_UNRESTRICTED_MODE\", None)
from backend.policy.policy_module import PolicyEngine, PolicyMode, reset_policy_engine
reset_policy_engine()
engine = PolicyEngine()
assert engine.mode == PolicyMode.RESTRICTED, f\"Expected RESTRICTED, got {engine.mode}\"
'" \
    0

# =============================================================================
# Test 3: UNRESTRICTED mode when env var is set
# =============================================================================
run_test "UNRESTRICTED mode with env var" \
    "python -c '
import os
os.environ[\"ALLOW_UNRESTRICTED_MODE\"] = \"true\"
from backend.policy.policy_module import PolicyEngine, PolicyMode, PolicyConfig, reset_policy_engine
reset_policy_engine()
config = PolicyConfig.from_env_and_file()
engine = PolicyEngine(config)
assert engine.mode == PolicyMode.UNRESTRICTED, f\"Expected UNRESTRICTED, got {engine.mode}\"
'" \
    0

# =============================================================================
# Test 4: EXTERNAL_ALLOWED mode with both flags
# =============================================================================
run_test "EXTERNAL_ALLOWED mode with both flags" \
    "python -c '
import os
os.environ[\"ALLOW_UNRESTRICTED_MODE\"] = \"true\"
os.environ[\"ALLOW_EXTERNAL_CALLS\"] = \"true\"
from backend.policy.policy_module import PolicyEngine, PolicyMode, PolicyConfig, reset_policy_engine
reset_policy_engine()
config = PolicyConfig.from_env_and_file()
engine = PolicyEngine(config)
assert engine.mode == PolicyMode.EXTERNAL_ALLOWED, f\"Expected EXTERNAL_ALLOWED, got {engine.mode}\"
'" \
    0

# =============================================================================
# Test 5: Input policy blocks harmful content in RESTRICTED mode
# =============================================================================
run_test "Input policy blocks harmful in RESTRICTED mode" \
    "python -c '
import os
os.environ.pop(\"ALLOW_UNRESTRICTED_MODE\", None)
from backend.policy.policy_module import PolicyEngine, PolicyConfig, reset_policy_engine
reset_policy_engine()
config = PolicyConfig.from_env_and_file()
engine = PolicyEngine(config)
result = engine.apply_input_policy(\"how to make a bomb\")
assert result.blocked == True, f\"Expected blocked=True, got {result.blocked}\"
'" \
    0

# =============================================================================
# Test 6: Input policy allows harmful content in UNRESTRICTED mode
# =============================================================================
run_test "Input policy allows in UNRESTRICTED mode" \
    "python -c '
import os
os.environ[\"ALLOW_UNRESTRICTED_MODE\"] = \"true\"
from backend.policy.policy_module import PolicyEngine, PolicyConfig, reset_policy_engine
reset_policy_engine()
config = PolicyConfig.from_env_and_file()
engine = PolicyEngine(config)
result = engine.apply_input_policy(\"how to make a bomb\")
assert result.blocked == False, f\"Expected blocked=False in UNRESTRICTED mode, got {result.blocked}\"
assert result.allowed == True, f\"Expected allowed=True in UNRESTRICTED mode\"
'" \
    0

# =============================================================================
# Test 7: Output policy filters in RESTRICTED mode
# =============================================================================
run_test "Output policy filters in RESTRICTED mode" \
    "python -c '
import os
os.environ.pop(\"ALLOW_UNRESTRICTED_MODE\", None)
from backend.policy.policy_module import PolicyEngine, PolicyConfig, reset_policy_engine
reset_policy_engine()
config = PolicyConfig.from_env_and_file()
engine = PolicyEngine(config)
result = engine.apply_output_policy(\"Use os.system(command) to run\")
assert \"[CODE_REMOVED_BY_POLICY]\" in result, f\"Expected filtered output, got: {result}\"
'" \
    0

# =============================================================================
# Test 8: Output policy passes through in UNRESTRICTED mode
# =============================================================================
run_test "Output policy passes through in UNRESTRICTED mode" \
    "python -c '
import os
os.environ[\"ALLOW_UNRESTRICTED_MODE\"] = \"true\"
from backend.policy.policy_module import PolicyEngine, PolicyConfig, reset_policy_engine
reset_policy_engine()
config = PolicyConfig.from_env_and_file()
engine = PolicyEngine(config)
input_text = \"Use os.system(command) to run\"
result = engine.apply_output_policy(input_text)
assert result == input_text, f\"Expected unchanged output in UNRESTRICTED mode\"
'" \
    0

# =============================================================================
# Test 9: Curriculum check bypassed in UNRESTRICTED mode
# =============================================================================
run_test "Curriculum check bypassed in UNRESTRICTED mode" \
    "python -c '
import os
os.environ[\"ALLOW_UNRESTRICTED_MODE\"] = \"true\"
from backend.policy.policy_module import PolicyEngine, PolicyConfig, reset_policy_engine
reset_policy_engine()
config = PolicyConfig.from_env_and_file()
engine = PolicyEngine(config)
result = engine.check_curriculum_alignment(\"quantum mechanics\", grade_level=3)
assert result.allowed == True, f\"Expected curriculum check bypassed in UNRESTRICTED mode\"
assert \"curriculum_check_bypassed\" in result.policy_applied
'" \
    0

# =============================================================================
# Test 10: Secret redaction still works in UNRESTRICTED mode
# =============================================================================
run_test "Secret redaction in UNRESTRICTED mode" \
    "python -c '
import os
os.environ[\"ALLOW_UNRESTRICTED_MODE\"] = \"true\"
from backend.policy.policy_module import PolicyEngine, PolicyConfig, reset_policy_engine
reset_policy_engine()
config = PolicyConfig.from_env_and_file()
config.redact_secrets = True  # Explicitly enable
engine = PolicyEngine(config)
result = engine.apply_input_policy(\"My API key is sk-1234567890123456789012345678901234567890123456\")
assert \"REDACTED\" in result.filtered_content or result.filtered_content != \"My API key is sk-1234567890123456789012345678901234567890123456\", \"Expected secret redaction\"
'" \
    0

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}=========================================${NC}"
echo -e "  ${GREEN}Passed: $PASSED${NC}"
echo -e "  ${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
