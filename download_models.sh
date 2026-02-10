#!/bin/bash
# ============================================================================
# SHIKSHA SETU - AI MODEL DOWNLOADER
# ============================================================================
# Download and cache all required AI models for the ShikshaSetu platform
#
# Usage:
#   ./download_models.sh              # Download all essential models
#   ./download_models.sh --all        # Download all models including optional
#   ./download_models.sh --essential  # Download essential models only (default)
#   ./download_models.sh --list       # List available models
#
# Gated Models:
#   Some models require Hugging Face authentication. To download gated models:
#   1. Get token from https://huggingface.co/settings/tokens
#   2. Pass via: HF_TOKEN=your_token ./download_models.sh
#   3. Or: huggingface-cli login
#
# Created: 2025-12-04
# ============================================================================

set -euo pipefail

# Colors & helpers
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; WHITE='\033[1;37m'
DIM='\033[2m'; BOLD='\033[1m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Helper functions
step() { echo -e "\n${BLUE}▸${NC} $1"; }
ok() { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
error() { echo -e "  ${RED}✗${NC} $1"; exit 1; }
info() { echo -e "  ${CYAN}ℹ${NC} $1"; }

# Check and setup HF token
setup_hf_token() {
    if [[ -z "${HF_TOKEN:-}" ]]; then
        if [[ -f ~/.huggingface/token ]]; then
            export HF_TOKEN=$(cat ~/.huggingface/token)
            info "Using HF token from ~/.huggingface/token"
        else
            warn "No HF_TOKEN found"
            echo ""
            echo -e "  ${YELLOW}Gated models may fail to download without authentication.${NC}"
            echo ""
            echo "  To add Hugging Face authentication:"
            echo "  1. Get your token: https://huggingface.co/settings/tokens"
            echo "  2. Option A: Export token in current session"
            echo "     ${DIM}export HF_TOKEN=hf_xxxxxxxxxxxxx${NC}"
            echo "     ${DIM}./download_models.sh${NC}"
            echo ""
            echo "  3. Option B: Login with CLI"
            echo "     ${DIM}huggingface-cli login${NC}"
            echo ""
            echo "  4. Option C: Create token file"
            echo "     ${DIM}mkdir -p ~/.huggingface${NC}"
            echo "     ${DIM}echo 'hf_xxxxxxxxxxxxx' > ~/.huggingface/token${NC}"
            echo ""
        fi
    else
        info "Using HF_TOKEN from environment"
    fi
}

# Parse arguments
DOWNLOAD_MODE="essential"
while [[ $# -gt 0 ]]; do
    case $1 in
        --all) DOWNLOAD_MODE="all"; shift ;;
        --essential) DOWNLOAD_MODE="essential"; shift ;;
        --list) DOWNLOAD_MODE="list"; shift ;;
        --help|-h)
            echo "Usage: ./download_models.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all         Download all models"
            echo "  --essential   Download essential models only (default)"
            echo "  --list        List available models"
            echo "  --help, -h    Show this help"
            echo ""
            echo "Hugging Face Token:"
            echo "  HF_TOKEN=hf_xxx ./download_models.sh  # Pass token via env var"
            echo "  huggingface-cli login                # Login with CLI"
            exit 0
            ;;
        *) error "Unknown option: $1" ;;
    esac
done

# ============================================================================
# MAIN
# ============================================================================
clear
echo -e "${CYAN}${BOLD}"
echo "   ════════════════════════════════════════════════════════"
echo "   ॐ  SHIKSHA SETU - AI MODEL DOWNLOADER  ॐ"
echo "   ════════════════════════════════════════════════════════"
echo -e "${NC}"

# Setup HF token
setup_hf_token

# Check if in virtual environment
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    warn "Virtual environment not activated"
    echo "  Attempting to activate venv..."
    if [[ -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
        source "$PROJECT_ROOT/venv/bin/activate"
        ok "Virtual environment activated"
    else
        error "Virtual environment not found. Run './setup.sh' first."
    fi
fi

# Check Python
if ! python --version &>/dev/null; then
    error "Python not found. Please install Python 3.11+"
fi

# Models catalog - All models that need to be downloaded
MODELS_DATA=(
    "Qwen/Qwen2.5-3B-Instruct|Qwen2.5-3B-Instruct|LLM for text simplification|6GB|essential|true"
    "ai4bharat/indictrans2-en-indic-1B|IndicTrans2-1B|10-language translation|2GB|essential|false"
    "BAAI/bge-m3|BGE-M3|Multilingual embeddings|1.2GB|essential|false"
    "BAAI/bge-reranker-v2-m3|BGE-Reranker-v2-M3|Retrieval reranking|1GB|essential|false"
    "google/gemma-2-2b-it|Gemma-2-2B-IT|Quality checking|4GB|optional|true"
    "ucaslcl/GOT-OCR2_0|GOT-OCR2.0|Document OCR|1.5GB|optional|true"
)

# Parse models data
declare -a MODEL_IDS
declare -a MODEL_NAMES
declare -a MODEL_DESCS
declare -a MODEL_SIZES
declare -a MODEL_TYPES
i=0
for model_entry in "${MODELS_DATA[@]}"; do
    IFS='|' read -r id name desc size type gated <<< "$model_entry"
    MODEL_IDS[$i]="$id"
    MODEL_NAMES[$i]="$name"
    MODEL_DESCS[$i]="$desc"
    MODEL_SIZES[$i]="$size"
    MODEL_TYPES[$i]="$type"
    MODEL_GATED[$i]="$gated"
    ((i++))
done

# Handle --list
if [[ "$DOWNLOAD_MODE" == "list" ]]; then
    step "Available Models:"
    echo ""
    for i in "${!MODEL_IDS[@]}"; do
        type_label="${MODEL_TYPES[$i]}"
        if [[ "$type_label" == "essential" ]]; then
            type_label="${GREEN}${BOLD}ESSENTIAL${NC}"
        else
            type_label="${YELLOW}optional${NC}"
        fi
        echo -e "  ${CYAN}${MODEL_NAMES[$i]}${NC}"
        echo -e "     ID: ${DIM}${MODEL_IDS[$i]}${NC}"
        echo -e "     Purpose: ${MODEL_DESCS[$i]}"
        echo -e "     Size: ${MODEL_SIZES[$i]}"
        echo -e "     Type: $type_label"
        echo ""
    done
    exit 0
fi

# Select models to download
declare -a MODELS_TO_DL
if [[ "$DOWNLOAD_MODE" == "essential" ]]; then
    for i in "${!MODEL_TYPES[@]}"; do
        if [[ "${MODEL_TYPES[$i]}" == "essential" ]]; then
            MODELS_TO_DL+=($i)
        fi
    done
else
    for i in "${!MODEL_IDS[@]}"; do
        MODELS_TO_DL+=($i)
    done
fi

# Calculate total size
total_size=0
step "Models to Download:"
echo ""
for i in "${MODELS_TO_DL[@]}"; do
    size_num=$(echo "${MODEL_SIZES[$i]}" | grep -o '^[0-9.]*')
    echo -e "  ${CYAN}${MODEL_NAMES[$i]}${NC} - ${MODEL_DESCS[$i]}"
    echo -e "     ${DIM}${MODEL_SIZES[$i]}${NC}"
done

echo ""
case "${DOWNLOAD_MODE}" in
    essential) echo -e "  ${YELLOW}Note:${NC} Essential models only (~${MODEL_SIZES[0]}) will be downloaded" ;;
    all) echo -e "  ${YELLOW}Note:${NC} All models (~14GB) will be downloaded" ;;
esac

# Install sentence-transformers if needed
step "Checking dependencies..."
pip install -q sentence-transformers >/dev/null 2>&1 || true
ok "Dependencies ready"

# Download models
step "Downloading AI models..."
echo ""

SUCCESS=0
FAILED=0
TOTAL_MODELS=${#MODELS_TO_DL[@]}
CURRENT_MODEL=0

# Helper function to download a single model
download_single_model() {
    local model_id="$1"
    local model_name="$2"
    local model_type="$3"
    local is_gated="$4"
    
    python3 << 'PYTHON_DL'
import os
import sys
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '0'  # Enable token

model_id = os.environ.get('MODEL_ID', '')
model_type = os.environ.get('MODEL_TYPE', '')
is_gated = os.environ.get('IS_GATED', 'false').lower() == 'true'
hf_token = os.environ.get('HF_TOKEN', '')

try:
    # For gated models, we need to accept the terms
    if is_gated and not hf_token:
        print("ERROR: Gated model requires HF_TOKEN")
        sys.exit(1)
    
    if "indictrans" in model_id.lower() or model_type == "translation":
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        print("Downloading tokenizer...", file=sys.stderr)
        AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True,
            token=hf_token if hf_token else None
        )
        print("Downloading model...", file=sys.stderr)
        AutoModelForSeq2SeqLM.from_pretrained(
            model_id, 
            trust_remote_code=True,
            token=hf_token if hf_token else None
        )
    elif "bge-reranker" in model_id.lower() or model_type == "reranker":
        from sentence_transformers import CrossEncoder
        print("Downloading reranker model...", file=sys.stderr)
        CrossEncoder(model_id)
    elif "bge" in model_id.lower() or model_type == "embeddings":
        from sentence_transformers import SentenceTransformer
        print("Downloading embeddings model...", file=sys.stderr)
        SentenceTransformer(model_id)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Downloading tokenizer...", file=sys.stderr)
        AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True,
            token=hf_token if hf_token else None
        )
        print("Downloading LLM (this may take a few minutes)...", file=sys.stderr)
        AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            token=hf_token if hf_token else None
        )
    
    print("DOWNLOAD_SUCCESS")
except Exception as e:
    print(f"DOWNLOAD_FAILED: {e}")
    sys.exit(1)
PYTHON_DL
}

for i in "${MODELS_TO_DL[@]}"; do
    model_id="${MODEL_IDS[$i]}"
    model_name="${MODEL_NAMES[$i]}"
    model_size="${MODEL_SIZES[$i]}"
    model_type="${MODEL_TYPES[$i]}"
    is_gated="${MODEL_GATED[$i]}"
    ((CURRENT_MODEL++))
    
    # Calculate progress percentage
    progress_pct=$((CURRENT_MODEL * 100 / TOTAL_MODELS))
    
    # Show progress header
    echo -e "  ${CYAN}[${progress_pct}%]${NC} (${CURRENT_MODEL}/${TOTAL_MODELS}) ${model_name} ${DIM}${model_size}${NC}"
    
    # Create progress bar (40 chars wide)
    bar_length=40
    filled=$((progress_pct * bar_length / 100))
    empty=$((bar_length - filled))
    progress_bar=""
    for ((j=0; j<filled; j++)); do progress_bar+="▓"; done
    for ((j=0; j<empty; j++)); do progress_bar+="░"; done
    
    # Show progress bar
    printf "     ${CYAN}┌─${progress_bar}─┐${NC}\n"
    
    # Download the model
    download_result=$(MODEL_ID="$model_id" MODEL_TYPE="$model_type" IS_GATED="$is_gated" HF_TOKEN="${HF_TOKEN:-}" download_single_model "$model_id" "$model_name" "$model_type" "$is_gated")
    
    if echo "$download_result" | grep -q "DOWNLOAD_SUCCESS"; then
        echo -e "     ${GREEN}└──────────────────────────────────────────────┘${NC}"
        echo -e "     ${GREEN}✓${NC} ${model_name} cached successfully"
        ((SUCCESS++))
    else
        echo -e "     ${YELLOW}└──────────────────────────────────────────────┘${NC}"
        if [[ "$is_gated" == "true" && -z "${HF_TOKEN:-}" ]]; then
            echo -e "     ${YELLOW}⚠${NC} ${model_name} (gated - requires HF_TOKEN)"
        else
            echo -e "     ${YELLOW}⚠${NC} ${model_name} (will download on first use)"
        fi
        ((FAILED++))
    fi
    echo ""
done

# Show summary with statistics
echo ""
bar_length=50
filled=$((SUCCESS * bar_length / TOTAL_MODELS))
empty=$((bar_length - filled))
summary_bar=""
for ((j=0; j<filled; j++)); do summary_bar+="▓"; done
for ((j=0; j<empty; j++)); do summary_bar+="░"; done

echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════${NC}"
echo -e "  ${CYAN}Overall Progress${NC}: ${summary_bar}"
echo -e "  Downloaded: ${GREEN}${SUCCESS}/${TOTAL_MODELS}${NC} models ($(( SUCCESS * 100 / TOTAL_MODELS ))%)"
if [[ $FAILED -gt 0 ]]; then
    echo -e "  Will auto-cache: ${YELLOW}${FAILED}${NC} models on first use"
fi
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════${NC}"
echo ""

ok "Model download complete!"
echo ""
echo "Next: ./start.sh"
echo ""
