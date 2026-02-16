#!/bin/bash
# ============================================================================
# Oryon - Project Cleanup Script
# ============================================================================
# Removes all temporary, cache, and build artifacts
#
# Usage: ./scripts/utils/cleanup.sh [--all] [--dry-run]
#
# Options:
#   --all      Also remove virtual environment and node_modules
#   --dry-run  Show what would be deleted without actually deleting
#
# Created by: ORGANISER-GPT
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
CLEAN_ALL=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --all) CLEAN_ALL=true ;;
        --dry-run) DRY_RUN=true ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  🧹 ORYON AI PROJECT CLEANUP${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}DRY RUN MODE - No files will be deleted${NC}"
    echo ""
fi

# Track cleaned items
CLEANED=0

clean_item() {
    local pattern="$1"
    local description="$2"

    if [[ "$DRY_RUN" == "true" ]]; then
        local count=$(find . -name "$pattern" -type d 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$count" -gt 0 ]]; then
            echo -e "${YELLOW}Would remove${NC}: $description ($count found)"
            ((CLEANED += count))
        fi
    else
        local count=$(find . -name "$pattern" -type d 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$count" -gt 0 ]]; then
            find . -name "$pattern" -type d -exec rm -rf {} + 2>/dev/null || true
            echo -e "${GREEN}✓${NC} Removed $description ($count found)"
            ((CLEANED += count))
        fi
    fi
}

clean_files() {
    local pattern="$1"
    local description="$2"

    if [[ "$DRY_RUN" == "true" ]]; then
        local count=$(find . -name "$pattern" -type f 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$count" -gt 0 ]]; then
            echo -e "${YELLOW}Would remove${NC}: $description ($count found)"
            ((CLEANED += count))
        fi
    else
        local count=$(find . -name "$pattern" -type f 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$count" -gt 0 ]]; then
            find . -name "$pattern" -type f -delete 2>/dev/null || true
            echo -e "${GREEN}✓${NC} Removed $description ($count found)"
            ((CLEANED += count))
        fi
    fi
}

echo -e "${BLUE}Python Artifacts:${NC}"
clean_item "__pycache__" "Python cache directories"
clean_item ".pytest_cache" "Pytest cache directories"
clean_files "*.pyc" "Compiled Python files"
clean_files "*.pyo" "Optimized Python files"
clean_files "*.pyd" "Python DLL files"
clean_item "*.egg-info" "Egg info directories"
clean_item ".eggs" "Eggs directories"

echo ""
echo -e "${BLUE}Build Artifacts:${NC}"
clean_item "dist" "Distribution directories"
clean_item "build" "Build directories"
clean_item ".tox" "Tox directories"

echo ""
echo -e "${BLUE}IDE & Editor Artifacts:${NC}"
clean_files "*.swp" "Vim swap files"
clean_files "*.swo" "Vim swap files"
clean_files "*~" "Backup files"
clean_files ".DS_Store" "macOS metadata files"

echo ""
echo -e "${BLUE}Log Files:${NC}"
if [[ "$DRY_RUN" == "true" ]]; then
    log_count=$(find logs/ -name "*.log" -type f 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$log_count" -gt 0 ]]; then
        echo -e "${YELLOW}Would remove${NC}: Log files ($log_count found)"
        ((CLEANED += log_count))
    fi
else
    log_count=$(find logs/ -name "*.log" -type f 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$log_count" -gt 0 ]]; then
        find logs/ -name "*.log" -type f -delete 2>/dev/null || true
        echo -e "${GREEN}✓${NC} Removed log files ($log_count found)"
        ((CLEANED += log_count))
    fi
fi

echo ""
echo -e "${BLUE}Coverage Reports:${NC}"
clean_item "htmlcov" "HTML coverage directories"
clean_files ".coverage" "Coverage data files"
clean_files "coverage.xml" "Coverage XML files"

if [[ "$CLEAN_ALL" == "true" ]]; then
    echo ""
    echo -e "${RED}Full Clean (--all):${NC}"

    if [[ "$DRY_RUN" == "true" ]]; then
        if [[ -d "venv" ]]; then
            echo -e "${YELLOW}Would remove${NC}: Virtual environment (venv/)"
            ((CLEANED++))
        fi
        if [[ -d "frontend/node_modules" ]]; then
            echo -e "${YELLOW}Would remove${NC}: Node modules (frontend/node_modules/)"
            ((CLEANED++))
        fi
    else
        if [[ -d "venv" ]]; then
            rm -rf venv
            echo -e "${GREEN}✓${NC} Removed virtual environment"
            ((CLEANED++))
        fi
        if [[ -d "frontend/node_modules" ]]; then
            rm -rf frontend/node_modules
            echo -e "${GREEN}✓${NC} Removed node_modules"
            ((CLEANED++))
        fi
    fi
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "  ${YELLOW}Would clean $CLEANED items${NC}"
else
    echo -e "  ${GREEN}✅ Cleaned $CLEANED items${NC}"
fi
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
