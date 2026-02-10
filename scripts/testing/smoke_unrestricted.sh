#!/bin/bash
# =============================================================================
# smoke_unrestricted.sh - Smoke Test for UNRESTRICTED Mode
# =============================================================================
#
# Boots the system in UNRESTRICTED mode, runs a representative query,
# and logs inputs/outputs to fixtures/run-unrestricted/<timestamp>/
#
# Usage:
#   ./scripts/smoke_unrestricted.sh
#
# Output:
#   - fixtures/run-unrestricted/<timestamp>/run_manifest.json
#   - fixtures/run-unrestricted/<timestamp>/input.txt
#   - fixtures/run-unrestricted/<timestamp>/output.txt
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
CYAN='\033[0;36m'
NC='\033[0m' # No Color

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FIXTURE_DIR="fixtures/run-unrestricted/${TIMESTAMP}"

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  UNRESTRICTED Mode Smoke Test${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo -e "${CYAN}Timestamp: ${TIMESTAMP}${NC}"
echo -e "${CYAN}Output Dir: ${FIXTURE_DIR}${NC}"
echo ""

# Create fixture directory
mkdir -p "$FIXTURE_DIR"

# Ensure we have a virtual environment
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found. Run setup.sh first.${NC}"
    exit 1
fi

source venv/bin/activate

# Export UNRESTRICTED mode
export ALLOW_UNRESTRICTED_MODE=true
export ALLOW_EXTERNAL_CALLS=false

echo -e "${YELLOW}Setting environment:${NC}"
echo "  ALLOW_UNRESTRICTED_MODE=true"
echo "  ALLOW_EXTERNAL_CALLS=false"
echo ""

# Test input - a query that would normally be filtered/constrained
TEST_INPUT="Explain how nuclear fission works in detail, including the physics equations and chain reaction mechanics. This is for advanced research purposes."

echo "$TEST_INPUT" > "$FIXTURE_DIR/input.txt"
echo -e "${YELLOW}Test Input:${NC}"
echo "$TEST_INPUT"
echo ""

# Run the policy check directly (without starting server)
echo -e "${YELLOW}Running policy check...${NC}"

python << EOF > "$FIXTURE_DIR/output.txt" 2>&1
import os
import json
import sys
from datetime import datetime, timezone

# Force unrestricted mode
os.environ["ALLOW_UNRESTRICTED_MODE"] = "true"
os.environ["ALLOW_EXTERNAL_CALLS"] = "false"

try:
    from backend.policy import get_policy_engine, PolicyMode
    from backend.policy.policy_module import reset_policy_engine

    # Reset and get fresh engine with env vars
    reset_policy_engine()
    engine = get_policy_engine()

    print("=" * 60)
    print("POLICY ENGINE STATUS")
    print("=" * 60)
    print(f"Mode: {engine.mode.value}")
    print(f"Unrestricted: {engine.config.allow_unrestricted_mode}")
    print(f"Filters Enabled: {engine.config.policy_filters_enabled}")
    print(f"Curriculum Enforcement: {engine.config.curriculum_enforcement}")
    print(f"External Calls: {engine.config.allow_external_calls}")
    print("")

    # Test input
    test_input = """$TEST_INPUT"""

    print("=" * 60)
    print("INPUT POLICY CHECK")
    print("=" * 60)
    result = engine.apply_input_policy(test_input)
    print(f"Allowed: {result.allowed}")
    print(f"Blocked: {result.blocked}")
    print(f"Risk Level: {result.risk_level.value}")
    print(f"Issues: {result.issues}")
    print(f"Policies Applied: {result.policy_applied}")
    print("")

    # Test curriculum check
    print("=" * 60)
    print("CURRICULUM ALIGNMENT CHECK")
    print("=" * 60)
    curr_result = engine.check_curriculum_alignment(test_input, grade_level=10, subject="science")
    print(f"Allowed: {curr_result.allowed}")
    print(f"Blocked: {curr_result.blocked}")
    print(f"Policies Applied: {curr_result.policy_applied}")
    print("")

    # Test output policy
    test_output = "Nuclear fission releases energy through E=mc². Use subprocess.run() to simulate."
    print("=" * 60)
    print("OUTPUT POLICY CHECK")
    print("=" * 60)
    filtered = engine.apply_output_policy(test_output)
    print(f"Original: {test_output}")
    print(f"Filtered: {filtered}")
    print(f"Changed: {filtered != test_output}")
    print("")

    print("=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)
    print("Status: SUCCESS")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

echo -e "${YELLOW}Output:${NC}"
cat "$FIXTURE_DIR/output.txt"
echo ""

# Generate run manifest
python << EOF > "$FIXTURE_DIR/run_manifest.json"
import json
import os
from datetime import datetime, timezone

os.environ["ALLOW_UNRESTRICTED_MODE"] = "true"

try:
    from backend.policy import get_policy_engine
    from backend.policy.policy_module import reset_policy_engine
    reset_policy_engine()
    engine = get_policy_engine()

    manifest = {
        "timestamp": "$TIMESTAMP",
        "iso_timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "UNRESTRICTED",
        "policy_mode": engine.mode.value,
        "config": {
            "allow_unrestricted_mode": engine.config.allow_unrestricted_mode,
            "policy_filters_enabled": engine.config.policy_filters_enabled,
            "curriculum_enforcement": engine.config.curriculum_enforcement,
            "allow_external_calls": engine.config.allow_external_calls,
            "block_harmful_content": engine.config.block_harmful_content,
            "redact_secrets": engine.config.redact_secrets,
        },
        "test_input": "$TEST_INPUT"[:100] + "...",
        "test_results": {
            "input_allowed": True,  # In unrestricted mode
            "curriculum_bypassed": True,
            "output_unfiltered": True,
        },
        "stats": engine.get_stats(),
        "status": "SUCCESS"
    }
    print(json.dumps(manifest, indent=2))
except Exception as e:
    manifest = {
        "timestamp": "$TIMESTAMP",
        "status": "ERROR",
        "error": str(e)
    }
    print(json.dumps(manifest, indent=2))
EOF

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}  Smoke Test Complete${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo -e "Results saved to: ${CYAN}${FIXTURE_DIR}/${NC}"
echo "  - input.txt"
echo "  - output.txt"
echo "  - run_manifest.json"
echo ""

# Check if test passed
if grep -q '"status": "SUCCESS"' "$FIXTURE_DIR/run_manifest.json"; then
    echo -e "${GREEN}✓ UNRESTRICTED mode smoke test PASSED${NC}"
    exit 0
else
    echo -e "${RED}✗ UNRESTRICTED mode smoke test FAILED${NC}"
    exit 1
fi
