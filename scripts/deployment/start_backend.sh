#!/bin/bash
# ===================================================================
# ShikshaSetu - Backend Only (v2.0)
# Starts Backend API and AI/ML Pipeline without Frontend
#
# V2 API Endpoints:
#   /api/v2/chat/guest     - Guest chat (no auth)
#   /api/v2/chat/stream    - Streaming chat (SSE)
#   /api/v2/content/*      - OCR, translate, simplify, TTS
#   /api/v2/auth/*         - Authentication
#   /api/v2/profile/me     - User profile (personalization)
#   /api/v2/review/*       - Content review queue
#   /health                - Health check
# ===================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Navigate to project root (scripts/deployment -> project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

PID_DIR="/tmp/shiksha_setu"
mkdir -p "$PID_DIR"

print_header() {
    echo ""
    echo "======================================================================"
    echo -e "  ${MAGENTA}$1${NC}"
    echo "======================================================================"
    echo ""
}

print_status() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }

cleanup() {
    print_header "🛑 Stopping Backend Services"
    [ -f "$PID_DIR/backend.pid" ] && kill $(cat "$PID_DIR/backend.pid") 2>/dev/null && rm -f "$PID_DIR/backend.pid"
    [ -f "$PID_DIR/celery.pid" ] && kill $(cat "$PID_DIR/celery.pid") 2>/dev/null && rm -f "$PID_DIR/celery.pid"
    print_status "Backend services stopped"
    exit 0
}

trap cleanup INT TERM

if [ ! -f "requirements.txt" ]; then
    print_error "Run from project root directory"
    exit 1
fi

print_header "🔧 SHIKSHA SETU - BACKEND ONLY"

# ===================================================================
# Checks
# ===================================================================
print_header "Pre-flight Checks"

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found"
    exit 1
fi
print_status "Python $(python3 --version | cut -d' ' -f2)"

if ! redis-cli ping &> /dev/null 2>&1; then
    print_warning "Starting Redis..."
    if command -v brew &> /dev/null; then
        brew services start redis
        sleep 2
    else
        print_error "Start Redis: redis-server"
        exit 1
    fi
fi
print_status "Redis running"

if [ ! -d "venv" ]; then
    print_error "Run setup first: ./setup.sh"
    exit 1
fi
print_status "Environment ready"

mkdir -p logs

# ===================================================================
# Start Backend
# ===================================================================
print_header "Starting Backend Services"

export $(cat .env | grep -v '^#' | xargs)
source venv/bin/activate

print_info "Starting Backend API..."
nohup uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > "$PID_DIR/backend.pid"
print_status "Backend API started (PID: $BACKEND_PID)"
sleep 3

print_info "Starting AI/ML Pipeline..."
nohup celery -A backend.tasks.celery_app worker --loglevel=info --concurrency=2 --queues=default,pipeline,ocr > logs/celery.log 2>&1 &
CELERY_PID=$!
echo $CELERY_PID > "$PID_DIR/celery.pid"
print_status "AI/ML Pipeline started (PID: $CELERY_PID)"
sleep 2

# ===================================================================
# Status
# ===================================================================
print_header "📊 Backend Status"

sleep 2
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Backend API:  ${GREEN}RUNNING${NC}"
else
    echo -e "${YELLOW}⚠${NC} Backend API:  ${YELLOW}STARTING${NC} (check logs/backend.log)"
fi

ps -p $CELERY_PID > /dev/null 2>&1 && \
    echo -e "${GREEN}✓${NC} AI Pipeline:  ${GREEN}RUNNING${NC}" || \
    echo -e "${RED}✗${NC} AI Pipeline:  ${RED}FAILED${NC}"

print_header "🎉 Backend Running!"

echo ""
echo "  Access Points:"
echo "  =============================================="
echo -e "  API:          ${CYAN}http://localhost:8000${NC}"
echo -e "  Docs:         ${CYAN}http://localhost:8000/docs${NC}"
echo -e "  Health:       ${CYAN}http://localhost:8000/health${NC}"
echo "  =============================================="
echo ""
echo "  V2 API Endpoints:"
echo "  =============================================="
echo "  POST /api/v2/chat/guest      Guest chat"
echo "  POST /api/v2/chat/stream     Streaming chat (SSE)"
echo "  POST /api/v2/content/ocr     OCR processing"
echo "  POST /api/v2/content/simplify Text simplification"
echo "  POST /api/v2/content/translate Translation"
echo "  POST /api/v2/content/tts     Text-to-speech"
echo "  GET  /api/v2/profile/me      User profile"
echo "  PUT  /api/v2/profile/me      Update profile"
echo "  GET  /api/v2/review/pending  Flagged responses"
echo "  =============================================="
echo ""
echo "  7 AI Models:"
echo "  =============================================="
echo "  Qwen3-8B         LLM reasoning + validation"
echo "  IndicTrans2      10 language translation"
echo "  BGE-M3           Embeddings (1024D)"
echo "  BGE-Reranker     Semantic reranking"
echo "  MMS-TTS          Neural text-to-speech"
echo "  Whisper V3       Speech recognition"
echo "  GOT-OCR2         Document understanding"
echo "  =============================================="
echo ""
echo "  Logs:"
echo "  =============================================="
echo "  Backend:      tail -f logs/backend.log"
echo "  AI Pipeline:  tail -f logs/celery.log"
echo "  =============================================="
echo ""
echo -e "  ${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

while true; do
    sleep 5
    if ! ps -p $BACKEND_PID > /dev/null 2>&1; then
        print_error "Backend stopped!"
        cleanup
    fi
    if ! ps -p $CELERY_PID > /dev/null 2>&1; then
        print_error "AI Pipeline stopped!"
        cleanup
    fi
done
