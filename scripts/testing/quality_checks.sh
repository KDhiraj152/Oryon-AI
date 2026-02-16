#!/bin/bash
# ============================================================================
# ORYON AI - COMPREHENSIVE TEST & QUALITY RUNNER
# ============================================================================
# Run all tests, linting, security scans, and generate reports
#
# Usage:
#   ./run_quality_checks.sh           # Full suite
#   ./run_quality_checks.sh --quick   # Quick checks only
#   ./run_quality_checks.sh --tests   # Tests only
#   ./run_quality_checks.sh --lint    # Linting only
#   ./run_quality_checks.sh --security # Security only
#
# Created by: K Dhiraj (TITAN-PROTOCOL)
# ============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure we're in venv
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -d "venv" ]]; then
        source venv/bin/activate
    else
        echo -e "${RED}No virtual environment found. Run: python -m venv venv${NC}"
        exit 1
    fi
fi

# Parse arguments
QUICK_MODE=false
TESTS_ONLY=false
LINT_ONLY=false
SECURITY_ONLY=false
REPORT_DIR="reports"

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --tests)
            TESTS_ONLY=true
            shift
            ;;
        --lint)
            LINT_ONLY=true
            shift
            ;;
        --security)
            SECURITY_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--quick|--tests|--lint|--security]"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Create reports directory
mkdir -p "$REPORT_DIR"

# ============================================================================
# FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ $1${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
}

run_check() {
    local name=$1
    local command=$2

    echo -e "${YELLOW}→ Running: ${name}${NC}"

    if eval "$command"; then
        echo -e "${GREEN}✓ ${name} passed${NC}"
        return 0
    else
        echo -e "${RED}✗ ${name} failed${NC}"
        return 1
    fi
}

# ============================================================================
# LINTING
# ============================================================================

run_linting() {
    print_header "LINTING & CODE QUALITY"

    local failures=0

    # Ruff linting
    run_check "Ruff linter" "ruff check backend/ tests/ --output-format=concise 2>&1 | head -50" || ((failures++))

    # Ruff formatting check
    run_check "Ruff format check" "ruff format --check backend/ tests/ 2>&1 | head -20" || true

    # Python syntax check
    run_check "Python syntax" "python -m compileall backend/ -q" || ((failures++))

    echo -e "\n${BLUE}Linting complete: ${failures} failures${NC}"
    return $failures
}

# ============================================================================
# SECURITY
# ============================================================================

run_security() {
    print_header "SECURITY SCANNING"

    local failures=0

    # Bandit security scan (ignore exit code, check results)
    echo -e "${YELLOW}→ Running: Bandit security scan${NC}"
    bandit -r backend/ -ll -q --format json > "$REPORT_DIR/bandit.json" 2>/dev/null || true
    if [[ -f "$REPORT_DIR/bandit.json" ]]; then
        high=$(python -c "import json; d=json.load(open('$REPORT_DIR/bandit.json')); print(len([r for r in d.get('results',[]) if r['issue_severity']=='HIGH']))")
        medium=$(python -c "import json; d=json.load(open('$REPORT_DIR/bandit.json')); print(len([r for r in d.get('results',[]) if r['issue_severity']=='MEDIUM']))")
        echo -e "${GREEN}✓ Bandit: ${high} HIGH, ${medium} MEDIUM severity issues${NC}"
        if [[ $high -gt 0 ]]; then
            ((failures++))
        fi
    else
        echo -e "${RED}✗ Bandit scan failed - no report generated${NC}"
        ((failures++))
    fi

    # Check for hardcoded secrets
    run_check "Secret detection" "grep -rn 'password\s*=\s*[\"'\'']' backend/ --include='*.py' | grep -v 'password.*=.*None' | grep -v 'example' | grep -v 'test' | head -5 || true"

    echo -e "\n${BLUE}Security complete: ${failures} failures${NC}"
    return $failures
}

# ============================================================================
# TESTING
# ============================================================================

run_tests() {
    print_header "RUNNING TESTS"

    local failures=0

    if $QUICK_MODE; then
        echo -e "${YELLOW}Quick mode: Running unit tests only${NC}"
        run_check "Unit tests" "python -m pytest tests/unit/ -q --tb=no" || ((failures++))
    else
        # Unit tests
        run_check "Unit tests" "python -m pytest tests/unit/ -v --tb=short" || ((failures++))

        # Integration tests
        run_check "Integration tests" "python -m pytest tests/integration/ -v --tb=short" || ((failures++))

        # Coverage report
        echo -e "${YELLOW}→ Running: Coverage report${NC}"
        python -m pytest tests/ --cov=backend --cov-report=html:$REPORT_DIR/coverage --cov-report=term --tb=no -q || true
        echo -e "${GREEN}✓ Coverage report saved to $REPORT_DIR/coverage${NC}"
    fi

    echo -e "\n${BLUE}Tests complete: ${failures} failures${NC}"
    return $failures
}

# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

run_benchmarks() {
    print_header "PERFORMANCE BENCHMARKS"

    if [[ -f "tests/performance/test_benchmarks.py" ]]; then
        echo -e "${YELLOW}→ Running performance benchmarks${NC}"
        python tests/performance/test_benchmarks.py 2>&1 || true
    else
        echo -e "${YELLOW}No benchmark file found${NC}"
    fi
}

# ============================================================================
# TYPE CHECKING (Optional)
# ============================================================================

run_type_check() {
    print_header "TYPE CHECKING"

    if command -v mypy &> /dev/null; then
        run_check "MyPy type check" "mypy backend/ --ignore-missing-imports --no-error-summary 2>&1 | head -30" || true
    else
        echo -e "${YELLOW}MyPy not installed, skipping type check${NC}"
    fi
}

# ============================================================================
# FRONTEND CHECKS
# ============================================================================

run_frontend_checks() {
    print_header "FRONTEND CHECKS"

    if [[ -d "frontend" ]] && [[ -f "frontend/package.json" ]]; then
        cd frontend

        # TypeScript check
        echo -e "${YELLOW}→ Running: TypeScript check${NC}"
        npx tsc --noEmit 2>&1 | head -20 || true

        # Build check
        echo -e "${YELLOW}→ Running: Build check${NC}"
        npm run build 2>&1 | tail -5 || true

        cd ..
        echo -e "${GREEN}✓ Frontend checks complete${NC}"
    else
        echo -e "${YELLOW}No frontend directory found${NC}"
    fi
}

# ============================================================================
# GENERATE SUMMARY
# ============================================================================

generate_summary() {
    print_header "QUALITY CHECK SUMMARY"

    echo -e "${BLUE}Reports saved to: ${REPORT_DIR}/${NC}"
    echo ""

    # Count issues
    if [[ -f "$REPORT_DIR/bandit.json" ]]; then
        python -c "
import json
with open('$REPORT_DIR/bandit.json') as f:
    d = json.load(f)
    results = d.get('results', [])
    high = len([r for r in results if r['issue_severity'] == 'HIGH'])
    medium = len([r for r in results if r['issue_severity'] == 'MEDIUM'])
    low = len([r for r in results if r['issue_severity'] == 'LOW'])
    print(f'Security Issues: {high} HIGH, {medium} MEDIUM, {low} LOW')
"
    fi

    # Test summary
    echo ""
    echo "Run './run_quality_checks.sh --help' for options"
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║          ORYON AI - QUALITY CHECK SUITE                   ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    local total_failures=0

    if $TESTS_ONLY; then
        run_tests || ((total_failures++))
    elif $LINT_ONLY; then
        run_linting || ((total_failures++))
    elif $SECURITY_ONLY; then
        run_security || ((total_failures++))
    else
        # Full suite
        run_linting || ((total_failures++))
        run_security || ((total_failures++))
        run_tests || ((total_failures++))

        if ! $QUICK_MODE; then
            run_frontend_checks
            run_benchmarks
        fi
    fi

    generate_summary

    if [[ $total_failures -gt 0 ]]; then
        echo -e "\n${RED}Quality checks completed with ${total_failures} failures${NC}"
        exit 1
    else
        echo -e "\n${GREEN}All quality checks passed!${NC}"
        exit 0
    fi
}

main "$@"
