#!/bin/bash
# ============================================================================
# ShikshaSetu - Python 3.11 Environment Setup Script (v1.1)
# ============================================================================
# This script sets up Python 3.11 virtual environment for optimal compatibility
# with all ML/AI packages including MLX, Transformers, and verovio.
#
# WHY PYTHON 3.11?
# ================
# Python 3.11 is the optimal version for ML/AI stacks because:
#
#   1. PRE-BUILT WHEELS: All major ML packages (PyTorch, MLX, Transformers)
#      have pre-built binary wheels for Python 3.11, meaning no compilation.
#
#   2. PROVEN STABILITY: Python 3.11 is mature and thoroughly tested with
#      production ML frameworks. Newer versions may have compatibility issues.
#
#   3. PERFORMANCE: Python 3.11 includes significant performance improvements
#      (~10-60% faster than 3.10 for many workloads).
#
#   4. APPLE SILICON: MLX and CoreML tools are optimized for Python 3.11.
#
#   5. PACKAGE SUPPORT: Some packages (verovio, certain audio libs) don't
#      yet have wheels for Python 3.13+, requiring compilation.
#
# TESTED PACKAGE VERSIONS:
# ========================
#   - PyTorch 2.9.1          - FastAPI 0.123.2
#   - Transformers 4.57.3    - Pydantic 2.12.5
#   - MLX 0.30.0             - SQLAlchemy 2.0.44
#   - MLX-LM 0.28.3          - Edge-TTS 7.2.3
#   - Sentence-Transformers 3.4.1
#   - Verovio 5.6.0 (pre-built wheel available)
#
# Usage:
#   chmod +x scripts/setup_python311.sh
#   ./scripts/setup_python311.sh
#
# Requirements:
#   - macOS 13.5+ (Ventura or later)
#   - Apple Silicon (M1/M2/M3/M4)
#   - Homebrew installed
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}   ShikshaSetu - Python 3.11 Environment Setup${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""

# ============================================================================
# Step 1: Check if Python 3.11 is installed
# ============================================================================
echo -e "${YELLOW}[1/6] Checking Python 3.11 installation...${NC}"

PYTHON311=""

# Check common Python 3.11 locations
if command -v python3.11 &> /dev/null; then
    PYTHON311="python3.11"
elif [ -f "/opt/homebrew/bin/python3.11" ]; then
    PYTHON311="/opt/homebrew/bin/python3.11"
elif [ -f "/usr/local/bin/python3.11" ]; then
    PYTHON311="/usr/local/bin/python3.11"
elif [ -f "$HOME/.pyenv/versions/3.11.10/bin/python" ]; then
    PYTHON311="$HOME/.pyenv/versions/3.11.10/bin/python"
fi

if [ -z "$PYTHON311" ]; then
    echo -e "${RED}❌ Python 3.11 not found!${NC}"
    echo ""
    echo -e "${YELLOW}Please install Python 3.11 using one of these methods:${NC}"
    echo ""
    echo "  Option 1 - Homebrew (recommended):"
    echo "    brew install python@3.11"
    echo ""
    echo "  Option 2 - pyenv:"
    echo "    brew install pyenv"
    echo "    pyenv install 3.11.10"
    echo "    pyenv local 3.11.10"
    echo ""
    echo "  Option 3 - Download from python.org:"
    echo "    https://www.python.org/downloads/release/python-31110/"
    echo ""
    exit 1
fi

PYTHON_VERSION=$($PYTHON311 --version 2>&1)
echo -e "${GREEN}✅ Found: $PYTHON_VERSION${NC}"
echo -e "   Location: $PYTHON311"

# Verify it's actually 3.11.x
if ! $PYTHON311 --version 2>&1 | grep -q "3\.11"; then
    echo -e "${RED}❌ Python version mismatch. Expected 3.11.x${NC}"
    exit 1
fi

# ============================================================================
# Step 2: Remove old virtual environment if exists
# ============================================================================
echo ""
echo -e "${YELLOW}[2/6] Preparing virtual environment...${NC}"

if [ -d "venv" ]; then
    echo -e "   Removing existing venv..."
    rm -rf venv
fi

# ============================================================================
# Step 3: Create new Python 3.11 virtual environment
# ============================================================================
echo ""
echo -e "${YELLOW}[3/6] Creating Python 3.11 virtual environment...${NC}"

$PYTHON311 -m venv venv

if [ ! -f "venv/bin/activate" ]; then
    echo -e "${RED}❌ Failed to create virtual environment${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Virtual environment created${NC}"

# ============================================================================
# Step 4: Activate and upgrade pip
# ============================================================================
echo ""
echo -e "${YELLOW}[4/6] Upgrading pip, setuptools, wheel...${NC}"

source venv/bin/activate

pip install --upgrade pip setuptools wheel --quiet

PIP_VERSION=$(pip --version)
echo -e "${GREEN}✅ $PIP_VERSION${NC}"

# ============================================================================
# Step 5: Install requirements
# ============================================================================
echo ""
echo -e "${YELLOW}[5/6] Installing dependencies (this may take 5-10 minutes)...${NC}"
echo ""

# Install with progress
pip install -r requirements.txt

echo ""
echo -e "${GREEN}✅ All dependencies installed${NC}"

# ============================================================================
# Step 6: Verify key packages
# ============================================================================
echo ""
echo -e "${YELLOW}[6/6] Verifying key packages...${NC}"
echo ""

# Verify Python version in venv
VENV_PYTHON=$(python --version)
echo -e "   Python:        ${GREEN}$VENV_PYTHON${NC}"

# Verify key packages
check_package() {
    local pkg=$1
    local version=$(pip show $pkg 2>/dev/null | grep "Version:" | cut -d" " -f2)
    if [ -n "$version" ]; then
        echo -e "   $pkg: ${GREEN}$version${NC}"
    else
        echo -e "   $pkg: ${RED}NOT INSTALLED${NC}"
    fi
}

check_package "torch"
check_package "transformers"
check_package "mlx"
check_package "mlx-lm"
check_package "sentence-transformers"
check_package "fastapi"
check_package "verovio"
check_package "edge-tts"

# ============================================================================
# Complete!
# ============================================================================
echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${GREEN}   ✅ Setup Complete!${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "To activate the environment:"
echo -e "   ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo -e "To start the backend:"
echo -e "   ${YELLOW}./scripts/deployment/start-backend${NC}"
echo ""
echo -e "To run tests:"
echo -e "   ${YELLOW}./bin/test${NC}"
echo ""
