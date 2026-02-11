#!/usr/bin/env python3
"""Generate the download_models.sh script."""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUTPUT = os.path.join(PROJECT_ROOT, "download_models.sh")

BASH_SCRIPT = r"""#!/bin/bash
# ============================================================================
# SHIKSHA SETU - AI MODEL DOWNLOADER
# ============================================================================
# Download and cache ALL required AI models for the ShikshaSetu platform.
#
# Models downloaded:
#   Essential:
#     1. Qwen3-8B (MLX 4-bit)     - Main LLM (simplification + validation)
#     2. IndicTrans2 (en->indic 1B)- Translation (10 Indian languages)
#     3. BGE-M3                    - Multilingual embeddings
#     4. BGE-Reranker-v2-M3       - Retrieval reranking
#     5. Whisper V3 Turbo          - Speech-to-text
#     6. MMS-TTS (Hindi + English) - Offline text-to-speech
#     7. Edge TTS (pip package)    - Cloud neural TTS
#
#   Optional (--all):
#     8. GOT-OCR2.0                - Document OCR
#     9. MMS-TTS (all 26 languages)- Full offline TTS coverage
#    10. Qwen3-8B FP16 (HF)       - Full-precision weights (non-MLX)
#
# Usage:
#   ./download_models.sh              # Download essential models
#   ./download_models.sh --all        # Download all models
#   ./download_models.sh --list       # List available models
#   ./download_models.sh --check      # Check cached models
#
# Gated Models: HF_TOKEN=your_token ./download_models.sh
# ============================================================================

set -uo pipefail

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; WHITE='\033[1;37m'
DIM='\033[2m'; BOLD='\033[1m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

step()  { echo -e "\n${BLUE}▸${NC} $1"; }
ok()    { echo -e "  ${GREEN}✓${NC} $1"; }
warn()  { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail()  { echo -e "  ${RED}✗${NC} $1"; }
info()  { echo -e "  ${CYAN}ℹ${NC} $1"; }

IS_MACOS=false; IS_APPLE_SILICON=false
if [[ "$(uname -s)" == "Darwin" ]]; then
    IS_MACOS=true
    [[ "$(uname -m)" == "arm64" ]] && IS_APPLE_SILICON=true
fi

# ── HF Token ───────────────────────────────────────────────────────────────
setup_hf_token() {
    if [[ -n "${HF_TOKEN:-}" ]]; then info "Using HF_TOKEN from environment"; return; fi
    for tp in "$HOME/.huggingface/token" "$HOME/.cache/huggingface/token"; do
        if [[ -f "$tp" ]]; then
            export HF_TOKEN=$(cat "$tp" | tr -d '[:space:]')
            if [[ -n "$HF_TOKEN" ]]; then info "Using HF token from $tp"; return; fi
        fi
    done
    warn "No HF_TOKEN found — gated models (GOT-OCR) may fail"
    echo "  Set via: HF_TOKEN=hf_xxx ./download_models.sh"
    echo "  Or:      huggingface-cli login"
    echo ""
}

# ── Parse args ──────────────────────────────────────────────────────────────
DOWNLOAD_MODE="essential"
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)       DOWNLOAD_MODE="all"; shift ;;
        --essential) DOWNLOAD_MODE="essential"; shift ;;
        --list)      DOWNLOAD_MODE="list"; shift ;;
        --check)     DOWNLOAD_MODE="check"; shift ;;
        --help|-h)
            echo "Usage: ./download_models.sh [--all|--essential|--list|--check|--help]"
            echo "  --all         Download ALL models"
            echo "  --essential   Essential models only (default)"
            echo "  --list        List models"
            echo "  --check       Check which are cached"
            echo "  HF_TOKEN=hf_xxx ./download_models.sh  (for gated models)"
            exit 0 ;;
        *) fail "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Banner ──────────────────────────────────────────────────────────────────
clear
echo -e "${CYAN}${BOLD}"
echo "   ════════════════════════════════════════════════════════"
echo "   ॐ  SHIKSHA SETU - AI MODEL DOWNLOADER  ॐ"
echo "   ════════════════════════════════════════════════════════"
echo -e "${NC}"
echo -e "   Mode: ${GREEN}${BOLD}${DOWNLOAD_MODE}${NC}"
$IS_APPLE_SILICON && echo -e "   Platform: ${CYAN}Apple Silicon (arm64)${NC}"
echo ""

setup_hf_token

# ── Virtual env ─────────────────────────────────────────────────────────────
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    warn "Virtual environment not activated"
    for vp in "$PROJECT_ROOT/venv" "$PROJECT_ROOT/.venv" "$PROJECT_ROOT/env"; do
        if [[ -f "$vp/bin/activate" ]]; then
            source "$vp/bin/activate"
            ok "Virtual environment activated ($vp)"
            break
        fi
    done
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        fail "Virtual environment not found. Run './setup.sh' first."
        exit 1
    fi
fi
python3 --version &>/dev/null || { fail "Python 3 not found"; exit 1; }
ok "Python: $(python3 --version 2>&1)"

# ── Dependencies ────────────────────────────────────────────────────────────
step "Checking dependencies..."
python3 -c "
import subprocess, sys, importlib
pkgs = {'transformers':'transformers','sentence_transformers':'sentence-transformers',
        'torch':'torch','huggingface_hub':'huggingface-hub',
        'edge_tts':'edge-tts','sentencepiece':'sentencepiece'}
miss = [v for k,v in pkgs.items() if not importlib.util.find_spec(k)]
if miss:
    print(f'Installing: {chr(44).join(miss)}')
    subprocess.check_call([sys.executable,'-m','pip','install','-q']+miss)
print('OK')
" || { fail "Dependency install failed"; exit 1; }
ok "Dependencies ready"

# ── Model Catalog ───────────────────────────────────────────────────────────
# Format: ID|Name|Desc|Size|Type|Gated|Loader
#
# Loaders:
#   causal_lm      = AutoModelForCausalLM + AutoTokenizer
#   seq2seq        = AutoModelForSeq2SeqLM + AutoTokenizer (trust_remote_code)
#   sentence_embed = SentenceTransformer
#   cross_encoder  = CrossEncoder
#   whisper        = AutoModelForSpeechSeq2Seq + AutoProcessor
#   vits_tts       = VitsModel + AutoTokenizer
#   auto_model     = AutoModel + AutoTokenizer (trust_remote_code, GOT-OCR)
#   edge_tts_check = verify edge-tts pip package
#   mlx_lm         = mlx_lm.load (Apple Silicon only)

MODELS_ESSENTIAL=(
    "ai4bharat/indictrans2-en-indic-1B|IndicTrans2 en-indic 1B|Translation 10 languages|2GB|essential|false|seq2seq"
    "BAAI/bge-m3|BGE-M3|Multilingual embeddings|1.2GB|essential|false|sentence_embed"
    "BAAI/bge-reranker-v2-m3|BGE-Reranker-v2-M3|Retrieval reranking|1GB|essential|false|cross_encoder"
    "openai/whisper-large-v3-turbo|Whisper V3 Turbo|Speech-to-text|1.5GB|essential|false|whisper"
    "facebook/mms-tts-hin|MMS-TTS Hindi|Offline TTS Hindi|30MB|essential|false|vits_tts"
    "facebook/mms-tts-eng|MMS-TTS English|Offline TTS English|30MB|essential|false|vits_tts"
    "EDGE_TTS_PKG|Edge TTS|Cloud neural TTS package|1MB|essential|false|edge_tts_check"
)

MODELS_OPTIONAL=(
    "ucaslcl/GOT-OCR2_0|GOT-OCR2.0|Document OCR|1.5GB|optional|true|auto_model"
    "Qwen/Qwen3-8B|Qwen3-8B FP16|Full-precision LLM weights|16GB|optional|false|causal_lm"
)

MMS_TTS_EXTRA=(
    "facebook/mms-tts-tam|MMS-TTS Tamil|TTS Tamil|30MB|optional|false|vits_tts"
    "facebook/mms-tts-tel|MMS-TTS Telugu|TTS Telugu|30MB|optional|false|vits_tts"
    "facebook/mms-tts-kan|MMS-TTS Kannada|TTS Kannada|30MB|optional|false|vits_tts"
    "facebook/mms-tts-mal|MMS-TTS Malayalam|TTS Malayalam|30MB|optional|false|vits_tts"
    "facebook/mms-tts-ben|MMS-TTS Bengali|TTS Bengali|30MB|optional|false|vits_tts"
    "facebook/mms-tts-mar|MMS-TTS Marathi|TTS Marathi|30MB|optional|false|vits_tts"
    "facebook/mms-tts-guj|MMS-TTS Gujarati|TTS Gujarati|30MB|optional|false|vits_tts"
    "facebook/mms-tts-pan|MMS-TTS Punjabi|TTS Punjabi|30MB|optional|false|vits_tts"
    "facebook/mms-tts-ory|MMS-TTS Odia|TTS Odia|30MB|optional|false|vits_tts"
    "facebook/mms-tts-asm|MMS-TTS Assamese|TTS Assamese|30MB|optional|false|vits_tts"
    "facebook/mms-tts-urd|MMS-TTS Urdu|TTS Urdu|30MB|optional|false|vits_tts"
    "facebook/mms-tts-san|MMS-TTS Sanskrit|TTS Sanskrit|30MB|optional|false|vits_tts"
    "facebook/mms-tts-nep|MMS-TTS Nepali|TTS Nepali|30MB|optional|false|vits_tts"
    "facebook/mms-tts-snd|MMS-TTS Sindhi|TTS Sindhi|30MB|optional|false|vits_tts"
    "facebook/mms-tts-kas|MMS-TTS Kashmiri|TTS Kashmiri|30MB|optional|false|vits_tts"
    "facebook/mms-tts-spa|MMS-TTS Spanish|TTS Spanish|30MB|optional|false|vits_tts"
    "facebook/mms-tts-fra|MMS-TTS French|TTS French|30MB|optional|false|vits_tts"
    "facebook/mms-tts-deu|MMS-TTS German|TTS German|30MB|optional|false|vits_tts"
    "facebook/mms-tts-por|MMS-TTS Portuguese|TTS Portuguese|30MB|optional|false|vits_tts"
    "facebook/mms-tts-rus|MMS-TTS Russian|TTS Russian|30MB|optional|false|vits_tts"
    "facebook/mms-tts-ara|MMS-TTS Arabic|TTS Arabic|30MB|optional|false|vits_tts"
    "facebook/mms-tts-cmn|MMS-TTS Mandarin|TTS Mandarin|30MB|optional|false|vits_tts"
    "facebook/mms-tts-jpn|MMS-TTS Japanese|TTS Japanese|30MB|optional|false|vits_tts"
    "facebook/mms-tts-kor|MMS-TTS Korean|TTS Korean|30MB|optional|false|vits_tts"
)

MLX_MODELS=(
    "mlx-community/Qwen3-8B-4bit|Qwen3-8B MLX 4bit|Apple Silicon LLM (primary)|4.6GB|essential|false|mlx_lm"
)

# ── Build download list ────────────────────────────────────────────────────
declare -a ALL_MODELS
for e in "${MODELS_ESSENTIAL[@]}"; do ALL_MODELS+=("$e"); done
# MLX Qwen3-8B is essential on Apple Silicon (primary LLM)
$IS_APPLE_SILICON && for e in "${MLX_MODELS[@]}"; do ALL_MODELS+=("$e"); done
if [[ "$DOWNLOAD_MODE" == "all" ]]; then
    for e in "${MODELS_OPTIONAL[@]}"; do ALL_MODELS+=("$e"); done
    for e in "${MMS_TTS_EXTRA[@]}"; do ALL_MODELS+=("$e"); done
fi

# ── --list ──────────────────────────────────────────────────────────────────
if [[ "$DOWNLOAD_MODE" == "list" ]]; then
    step "Available Models:"
    echo ""
    echo -e "  ${GREEN}${BOLD}=== ESSENTIAL ===${NC}"
    for e in "${MODELS_ESSENTIAL[@]}"; do
        IFS='|' read -r id nm ds sz tp gt ld <<< "$e"
        gw=""; [[ "$gt" == "true" ]] && gw=" ${YELLOW}[gated]${NC}"
        echo -e "  ${CYAN}${nm}${NC} (${sz}) - ${ds} [${ld}]${gw}"
    done
    echo -e "\n  ${YELLOW}${BOLD}=== OPTIONAL (--all) ===${NC}"
    for e in "${MODELS_OPTIONAL[@]}"; do
        IFS='|' read -r id nm ds sz tp gt ld <<< "$e"
        gw=""; [[ "$gt" == "true" ]] && gw=" ${YELLOW}[gated]${NC}"
        echo -e "  ${CYAN}${nm}${NC} (${sz}) - ${ds} [${ld}]${gw}"
    done
    echo -e "\n  ${YELLOW}${BOLD}=== MMS-TTS LANGUAGES (--all) ===${NC}"
    for e in "${MMS_TTS_EXTRA[@]}"; do
        IFS='|' read -r id nm ds sz tp gt ld <<< "$e"
        echo -e "  ${CYAN}${nm}${NC} - ${DIM}${id}${NC}"
    done
    if $IS_APPLE_SILICON; then
        echo -e "\n  ${YELLOW}${BOLD}=== MLX APPLE SILICON (--all) ===${NC}"
        for e in "${MLX_MODELS[@]}"; do
            IFS='|' read -r id nm ds sz tp gt ld <<< "$e"
            echo -e "  ${CYAN}${nm}${NC} (${sz}) - ${DIM}${id}${NC}"
        done
    fi
    echo ""; exit 0
fi

# ── --check ─────────────────────────────────────────────────────────────────
if [[ "$DOWNLOAD_MODE" == "check" ]]; then
    step "Checking cached models..."
    python3 << 'CHECKEOF'
import os, sys
try:
    from huggingface_hub import scan_cache_dir
    cached = {r.repo_id for r in scan_cache_dir().repos}
except Exception:
    cached = set()
    hf = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.isdir(hf):
        for d in os.listdir(hf):
            if d.startswith("models--"):
                cached.add(d.replace("models--","").replace("--","/"))
models = [
    ("ai4bharat/indictrans2-en-indic-1B","IndicTrans2","essential"),
    ("BAAI/bge-m3","BGE-M3","essential"),
    ("BAAI/bge-reranker-v2-m3","BGE-Reranker","essential"),
    ("openai/whisper-large-v3-turbo","Whisper V3 Turbo","essential"),
    ("facebook/mms-tts-hin","MMS-TTS Hindi","essential"),
    ("facebook/mms-tts-eng","MMS-TTS English","essential"),
    ("mlx-community/Qwen3-8B-4bit","Qwen3-8B MLX 4bit","essential"),
    ("ucaslcl/GOT-OCR2_0","GOT-OCR2.0","optional"),
    ("Qwen/Qwen3-8B","Qwen3-8B FP16","optional"),
]
for l in "tam tel kan mal ben mar guj pan ory asm urd san nep snd kas spa fra deu por rus ara cmn jpn kor".split():
    models.append((f"facebook/mms-tts-{l}",f"MMS-TTS {l}","optional"))
ok=miss=0
for mid,nm,tp in models:
    if mid in cached:
        print(f"  \033[32m✓\033[0m {nm} ({mid})"); ok+=1
    else:
        tag="\033[31m✗ MISSING\033[0m" if tp=="essential" else "\033[33m○ not cached\033[0m"
        print(f"  {tag} {nm} ({mid})"); miss+=1
try:
    import edge_tts; print("  \033[32m✓\033[0m Edge TTS (installed)"); ok+=1
except: print("  \033[31m✗ MISSING\033[0m Edge TTS"); miss+=1
print(f"\n  Cached: {ok} | Missing: {miss}")
CHECKEOF
    exit 0
fi

# ── Download plan ───────────────────────────────────────────────────────────
TOTAL_MODELS=${#ALL_MODELS[@]}
step "Download plan (${TOTAL_MODELS} models):"
echo ""
for e in "${ALL_MODELS[@]}"; do
    IFS='|' read -r id nm ds sz tp gt ld <<< "$e"
    gw=""; [[ "$gt" == "true" ]] && gw=" ${YELLOW}[gated]${NC}"
    echo -e "  ${CYAN}•${NC} ${nm} ${DIM}(${sz})${NC} - ${ds}${gw}"
done
echo ""

# ── Python download function ───────────────────────────────────────────────
download_model() {
    local model_id="$1" loader="$2"
    MODEL_ID="$model_id" LOADER_TYPE="$loader" HF_TOKEN="${HF_TOKEN:-}" \
    python3 << 'DLEOF'
import os, sys

model_id = os.environ["MODEL_ID"]
loader   = os.environ["LOADER_TYPE"]
hf_token = os.environ.get("HF_TOKEN", "") or None

def log(m):
    print(m, file=sys.stderr)

try:
    # Edge TTS - just check package
    if loader == "edge_tts_check":
        import edge_tts
        log("edge-tts package OK")
        print("OK")
        sys.exit(0)

    # MLX models (Apple Silicon)
    if loader == "mlx_lm":
        try:
            from mlx_lm import load as mlx_load
            log(f"Downloading MLX model: {model_id}")
            mlx_load(model_id)
            print("OK")
            sys.exit(0)
        except ImportError:
            log("mlx-lm not installed, skipping")
            print("SKIP")
            sys.exit(0)
        except Exception as e:
            print(f"FAIL: {e}")
            sys.exit(1)

    import torch

    # Device config
    if torch.cuda.is_available():
        dm, dt = "auto", torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dm, dt = None, torch.float32
    else:
        dm, dt = None, torch.float32

    kw = dict(token=hf_token)

    # Causal LM (Qwen3)
    if loader == "causal_lm":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        log(f"Downloading tokenizer: {model_id}")
        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, **kw)
        log(f"Downloading model: {model_id}")
        AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dt, device_map=dm, **kw
        )

    # Seq2Seq (IndicTrans2)
    elif loader == "seq2seq":
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        log(f"Downloading tokenizer: {model_id}")
        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, **kw)
        log(f"Downloading model: {model_id}")
        AutoModelForSeq2SeqLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dt, device_map=dm, **kw
        )

    # SentenceTransformer (BGE-M3)
    elif loader == "sentence_embed":
        from sentence_transformers import SentenceTransformer
        log(f"Downloading embeddings: {model_id}")
        SentenceTransformer(model_id, device="cpu", trust_remote_code=True)

    # CrossEncoder (BGE-Reranker)
    elif loader == "cross_encoder":
        from sentence_transformers import CrossEncoder
        log(f"Downloading reranker: {model_id}")
        CrossEncoder(model_id, device="cpu", trust_remote_code=True)

    # Whisper (Speech-to-Text)
    elif loader == "whisper":
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        log(f"Downloading Whisper processor: {model_id}")
        AutoProcessor.from_pretrained(model_id, **kw)
        log(f"Downloading Whisper model: {model_id}")
        AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=dt, device_map=dm, **kw
        )

    # VITS TTS (MMS-TTS)
    elif loader == "vits_tts":
        from transformers import VitsModel, AutoTokenizer
        log(f"Downloading TTS tokenizer: {model_id}")
        AutoTokenizer.from_pretrained(model_id, **kw)
        log(f"Downloading TTS model: {model_id}")
        VitsModel.from_pretrained(model_id, **kw)

    # AutoModel (GOT-OCR)
    elif loader == "auto_model":
        from transformers import AutoModel, AutoTokenizer
        log(f"Downloading tokenizer: {model_id}")
        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, **kw)
        log(f"Downloading model: {model_id}")
        AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dt, device_map=dm, **kw
        )

    else:
        print(f"FAIL: Unknown loader {loader}")
        sys.exit(1)

    print("OK")

except KeyboardInterrupt:
    print("FAIL: interrupted")
    sys.exit(130)
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
DLEOF
}

# ── Download loop ───────────────────────────────────────────────────────────
step "Downloading AI models..."
echo ""

SUCCESS=0; FAILED=0; SKIPPED=0
FAILED_NAMES=()
CURRENT=0

for entry in "${ALL_MODELS[@]}"; do
    IFS='|' read -r model_id name desc size type gated loader <<< "$entry"
    ((CURRENT++)) || true

    pct=$((CURRENT * 100 / TOTAL_MODELS))
    blen=40; filled=$((pct * blen / 100)); empty=$((blen - filled))
    bar=""; for ((j=0; j<filled; j++)); do bar+="▓"; done
    for ((j=0; j<empty; j++)); do bar+="░"; done

    echo -e "  ${CYAN}[${pct}%]${NC} (${CURRENT}/${TOTAL_MODELS}) ${BOLD}${name}${NC} ${DIM}${size}${NC}"
    echo -e "     ${CYAN}${bar}${NC}"

    # Skip gated without token
    if [[ "$gated" == "true" && -z "${HF_TOKEN:-}" ]]; then
        echo -e "     ${YELLOW}⚠${NC} Skipped - gated model requires HF_TOKEN"
        ((SKIPPED++)) || true
        FAILED_NAMES+=("${name} (needs HF_TOKEN)")
        echo ""; continue
    fi

    # Download
    result=$(download_model "$model_id" "$loader" 2>&1 | tail -1)

    if [[ "$result" == "OK" ]]; then
        echo -e "     ${GREEN}✓${NC} ${name} cached successfully"
        ((SUCCESS++)) || true
    elif [[ "$result" == "SKIP" ]]; then
        echo -e "     ${YELLOW}⚠${NC} ${name} skipped (optional dep missing)"
        ((SKIPPED++)) || true
    else
        echo -e "     ${RED}✗${NC} ${name} FAILED"
        errmsg="${result#FAIL: }"
        [[ ${#errmsg} -gt 120 ]] && errmsg="${errmsg:0:120}..."
        echo -e "     ${DIM}${errmsg}${NC}"
        ((FAILED++)) || true
        FAILED_NAMES+=("${name}")
    fi
    echo ""
done

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "  ${CYAN}${BOLD}DOWNLOAD SUMMARY${NC}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════${NC}"
echo ""

blen=50
[[ $TOTAL_MODELS -gt 0 ]] && filled=$((SUCCESS * blen / TOTAL_MODELS)) || filled=0
empty=$((blen - filled)); sbar=""
for ((j=0; j<filled; j++)); do sbar+="▓"; done
for ((j=0; j<empty; j++)); do sbar+="░"; done

echo -e "  Progress: ${sbar}"
echo ""
echo -e "  ${GREEN}✓ Downloaded: ${SUCCESS}/${TOTAL_MODELS}${NC}"
[[ $SKIPPED -gt 0 ]] && echo -e "  ${YELLOW}⚠ Skipped:    ${SKIPPED}${NC}"
if [[ $FAILED -gt 0 ]]; then
    echo -e "  ${RED}✗ Failed:     ${FAILED}${NC}"
    echo -e "\n  ${RED}Failed models:${NC}"
    for n in "${FAILED_NAMES[@]}"; do echo -e "    ${RED}•${NC} ${n}"; done
fi
echo ""

if [[ $FAILED -gt 0 || $SKIPPED -gt 0 ]]; then
    echo -e "  ${YELLOW}${BOLD}Troubleshooting:${NC}"
    [[ $SKIPPED -gt 0 ]] && echo "  • For gated models: HF_TOKEN=hf_xxx ./download_models.sh --all"
    [[ $FAILED -gt 0 ]] && echo "  • Check network & retry. Failed models auto-download on first use."
    echo ""
fi

echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════════════${NC}"
echo ""

if [[ $SUCCESS -eq $TOTAL_MODELS ]]; then
    ok "All models downloaded successfully!"
elif [[ $SUCCESS -gt 0 ]]; then
    ok "Model download partially complete (${SUCCESS}/${TOTAL_MODELS})"
else
    fail "Model download failed"
fi

echo ""
echo "Next: ./start.sh"
echo ""
"""

with open(OUTPUT, "w", newline="\n") as f:
    f.write(BASH_SCRIPT)

os.chmod(OUTPUT, 0o755)
print(f"Written {len(BASH_SCRIPT.splitlines())} lines to {OUTPUT}")
