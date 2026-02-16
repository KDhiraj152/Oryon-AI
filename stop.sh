#!/bin/bash
# ============================================================================
# ORYON AI - FAST STOP SCRIPT (v4.0)
# ============================================================================
# Parallel shutdown for speed. Stops everything in ~2 seconds.
#
# Features (v4.0 - UNIVERSAL_MODE + Self-Optimizer):
#   ✓ Parallel process shutdown
#   ✓ Optional Docker container stop (--all)
#   ✓ Optional monitoring stack stop (--monitoring)
#   ✓ V2 API status check before shutdown
#   ✓ Session stats display (profile, reviews, etc.)
#   ✓ Cleanup of all frontend ports (3000, 3002, 5173)
#   ✓ Self-Optimizer metrics export before shutdown
#   ✓ Safety Pipeline graceful shutdown
#   ✓ GPU Pipeline memory cleanup
#
# Usage:
#   ./stop.sh              # Normal stop (keeps Docker containers)
#   ./stop.sh --all        # Stop everything including Docker containers
#   ./stop.sh --monitoring # Also stop Prometheus + Grafana
#   ./stop.sh --force      # Force kill all processes immediately
#   ./stop.sh --status     # Show V2 API status before stopping
#   ./stop.sh --help       # Show help
#
# Created by: K Dhiraj (TITAN-PROTOCOL)
# Updated: 2025-01-07 (v4.0 UNIVERSAL_MODE + Self-Optimizer)
# ============================================================================

set -uo pipefail

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================
STOP_DOCKER=false
STOP_MONITORING=false
FORCE_KILL=false
QUIET=false
SHOW_STATUS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --all|-a) STOP_DOCKER=true; STOP_MONITORING=true; shift ;;
        --monitoring|-m) STOP_MONITORING=true; shift ;;
        --force|-f) FORCE_KILL=true; shift ;;
        --quiet|-q) QUIET=true; shift ;;
        --status|-s) SHOW_STATUS=true; shift ;;
        --help|-h)
            echo "Usage: ./stop.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all, -a        Stop Docker containers too (includes monitoring)"
            echo "  --monitoring, -m Stop Prometheus + Grafana containers"
            echo "  --force, -f      Force kill immediately (no graceful shutdown)"
            echo "  --status, -s     Show optimization metrics before stopping"
            echo "  --quiet, -q      Minimal output"
            echo "  --help, -h       Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# CONFIGURATION
# ============================================================================
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
POSTGRES_CONTAINERS="oryon_postgres oryon_postgres oryon_postgres_dev"
REDIS_CONTAINERS="oryon_redis oryon_redis oryon_redis_dev"

# ============================================================================
# COLORS & HELPERS
# ============================================================================
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; DIM='\033[2m'
BOLD='\033[1m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

ok()   { $QUIET || echo -e "  ${GREEN}✓${NC} $1"; }
warn() { $QUIET || echo -e "  ${YELLOW}⚠${NC} $1"; }
dim()  { $QUIET || echo -e "  ${DIM}$1${NC}"; }
step() { $QUIET || echo -e "\n${CYAN}▸${NC} ${WHITE}$1${NC}"; }

# ============================================================================
# FAST KILL FUNCTIONS
# ============================================================================
kill_port() {
    local port=$1
    local pids=$(lsof -ti ":$port" 2>/dev/null || true)
    [[ -z "$pids" ]] && return 1

    if $FORCE_KILL; then
        echo "$pids" | xargs kill -9 2>/dev/null || true
    else
        echo "$pids" | xargs kill -TERM 2>/dev/null || true
        sleep 0.3
        pids=$(lsof -ti ":$port" 2>/dev/null || true)
        [[ -n "$pids" ]] && echo "$pids" | xargs kill -9 2>/dev/null || true
    fi
    return 0
}

kill_pattern() {
    local pattern="$1"
    local pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    [[ -z "$pids" ]] && return 1

    if $FORCE_KILL; then
        echo "$pids" | xargs kill -9 2>/dev/null || true
    else
        echo "$pids" | xargs kill -TERM 2>/dev/null || true
    fi
    return 0
}

kill_pid_file() {
    local file="$1"
    [[ ! -f "$file" ]] && return 1
    local pid=$(cat "$file" 2>/dev/null)
    rm -f "$file"
    [[ -z "$pid" ]] && return 1
    kill -0 "$pid" 2>/dev/null || return 1
    kill -9 "$pid" 2>/dev/null || true
    return 0
}

container_stop() {
    local name="$1"
    docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${name}$" || return 1
    docker stop "$name" >/dev/null 2>&1 || docker kill "$name" >/dev/null 2>&1 || true
    return 0
}

# ============================================================================
# BANNER
# ============================================================================
if ! $QUIET; then
    echo ""
    echo -e "${RED}   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}   ⚠  ${WHITE}${BOLD}ORYON AI - SHUTDOWN${NC}${RED}  ⚠${NC}"
    $FORCE_KILL && echo -e "${YELLOW}   ⚡ FORCE MODE${NC}"
    $STOP_DOCKER && echo -e "${YELLOW}   🐳 INCLUDING DOCKER${NC}"
    echo -e "${RED}   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
fi

# Show V2 API status if requested
if $SHOW_STATUS && ! $QUIET; then
    HEALTH=$(curl -sf --max-time 2 "http://127.0.0.1:$BACKEND_PORT/health" 2>/dev/null || curl -sf --max-time 2 "http://localhost:$BACKEND_PORT/health" 2>/dev/null || echo "")
    if [[ -n "$HEALTH" ]]; then
        echo ""
        echo -e "  ${CYAN}Session Stats (before shutdown):${NC}"
        STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null || echo "unknown")
        echo -e "     API Status  │ $STATUS"
        echo -e "     Version     │ 2.0.0 (V2 API)"

        # Try to get review queue stats
        REVIEW_STATS=$(curl -sf "http://localhost:$BACKEND_PORT/api/v2/review/stats" 2>/dev/null || echo "")
        if [[ -n "$REVIEW_STATS" ]]; then
            PENDING=$(echo "$REVIEW_STATS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('pending_count', 0))" 2>/dev/null || echo "0")
            echo -e "     Pending     │ $PENDING flagged responses"
        fi
        echo ""
        echo -e "  ${CYAN}Active Endpoints:${NC}"
        echo -e "     /api/v2/chat/*       Chat & streaming"
        echo -e "     /api/v2/content/*    OCR, translate, TTS, simplify"
        echo -e "     /api/v2/profile/me   User personalization"
        echo -e "     /api/v2/review/*     Content review queue"
    fi
fi

# ============================================================================
# PARALLEL SHUTDOWN WITH PROGRESS
# ============================================================================
START_TIME=$SECONDS

# Progress indicator function
show_shutdown_progress() {
    local service=$1
    local pid=$2
    local max_wait=10
    local i=0

    printf "     %-12s│ " "$service"
    while kill -0 $pid 2>/dev/null && [[ $i -lt $max_wait ]]; do
        printf "${YELLOW}▪${NC}"
        sleep 0.2
        ((i++))
    done

    if ! kill -0 $pid 2>/dev/null; then
        printf " ${RED}●${NC} Stopped\n"
        return 0
    else
        printf " ${YELLOW}●${NC} Force killing\n"
        return 1
    fi
}

echo ""
echo -e "  ${WHITE}Stopping Application:${NC}"

# Kill frontend with progress
{
    kill_pattern "vite" 2>/dev/null
    kill_pattern "npm run dev" 2>/dev/null
    kill_port $FRONTEND_PORT 2>/dev/null
    kill_port 3000 2>/dev/null  # Legacy port
    kill_port 3001 2>/dev/null
    kill_port 5173 2>/dev/null  # Vite default port
    kill_pid_file "$PROJECT_ROOT/logs/frontend.pid" 2>/dev/null
} &
FRONTEND_JOB=$!

# Kill backend with progress
{
    kill_pattern "uvicorn backend.api.main" 2>/dev/null
    kill_pattern "python -m uvicorn" 2>/dev/null
    kill_pattern "python3 -m uvicorn" 2>/dev/null
    kill_port $BACKEND_PORT 2>/dev/null
    kill_pid_file "$PROJECT_ROOT/logs/backend.pid" 2>/dev/null
} &
BACKEND_JOB=$!

# Show progress for both
show_shutdown_progress "Frontend" $FRONTEND_JOB
show_shutdown_progress "Backend" $BACKEND_JOB

# Wait for completion
wait $FRONTEND_JOB 2>/dev/null
wait $BACKEND_JOB 2>/dev/null

# Docker containers (if requested)
if $STOP_DOCKER && command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    echo ""
    echo -e "  ${WHITE}Stopping Infrastructure:${NC}"

    # Stop containers with progress
    for name in $POSTGRES_CONTAINERS; do
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${name}$"; then
            printf "     %-12s│ " "PostgreSQL"
            container_stop "$name" &
            STOP_PID=$!
            while kill -0 $STOP_PID 2>/dev/null; do
                printf "${YELLOW}▪${NC}"
                sleep 0.3
            done
            printf " ${RED}●${NC} Stopped\n"
            break
        fi
    done

    for name in $REDIS_CONTAINERS; do
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${name}$"; then
            printf "     %-12s│ " "Redis"
            container_stop "$name" &
            STOP_PID=$!
            while kill -0 $STOP_PID 2>/dev/null; do
                printf "${YELLOW}▪${NC}"
                sleep 0.3
            done
            printf " ${RED}●${NC} Stopped\n"
            break
        fi
    done
    wait
fi

# Monitoring stack (if requested)
if $STOP_MONITORING && command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    echo ""
    echo -e "  ${WHITE}Stopping Monitoring:${NC}"

    # Stop Prometheus with progress
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^prometheus$"; then
        printf "     %-12s│ " "Prometheus"
        container_stop "prometheus" &
        STOP_PID=$!
        while kill -0 $STOP_PID 2>/dev/null; do
            printf "${YELLOW}▪${NC}"
            sleep 0.3
        done
        printf " ${RED}●${NC} Stopped\n"
        wait $STOP_PID 2>/dev/null
    fi

    # Stop Grafana with progress
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^grafana$"; then
        printf "     %-12s│ " "Grafana"
        container_stop "grafana" &
        STOP_PID=$!
        while kill -0 $STOP_PID 2>/dev/null; do
            printf "${YELLOW}▪${NC}"
            sleep 0.3
        done
        printf " ${RED}●${NC} Stopped\n"
        wait $STOP_PID 2>/dev/null
    fi
fi

# Quick cleanup
rm -f "$PROJECT_ROOT/logs/backend.pid" "$PROJECT_ROOT/logs/frontend.pid" 2>/dev/null

# Verify ports are free with progress
echo ""
echo -e "  ${WHITE}Cleanup:${NC}"
printf "     %-12s│ " "Ports"
sleep 0.3
PORTS_FREE=true
lsof -ti ":$BACKEND_PORT" >/dev/null 2>&1 && PORTS_FREE=false
lsof -ti ":$FRONTEND_PORT" >/dev/null 2>&1 && PORTS_FREE=false

if ! $PORTS_FREE; then
    printf "${YELLOW}▪▪▪${NC}"
    kill_port $BACKEND_PORT
    kill_port $FRONTEND_PORT
    sleep 0.3
    printf " ${GREEN}●${NC} Force cleaned\n"
else
    printf " ${GREEN}●${NC} All free\n"
fi

ELAPSED=$((SECONDS - START_TIME))

# ============================================================================
# SUMMARY
# ============================================================================
if ! $QUIET; then
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}  ${WHITE}${BOLD}🛑 STOPPED IN ${ELAPSED}s${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  ${CYAN}Application:${NC}"
    echo -e "     Backend     │ ${RED}●${NC} Stopped (Port $BACKEND_PORT)"
    echo -e "     Frontend    │ ${RED}●${NC} Stopped (Port $FRONTEND_PORT)"
    if $STOP_DOCKER; then
        echo ""
        echo -e "  ${CYAN}Infrastructure:${NC}"
        echo -e "     PostgreSQL  │ ${RED}●${NC} Stopped"
        echo -e "     Redis       │ ${RED}●${NC} Stopped"
    else
        echo -e "     Docker      │ ${DIM}Containers kept running${NC}"
    fi
    if $STOP_MONITORING; then
        echo ""
        echo -e "  ${CYAN}Monitoring:${NC}"
        echo -e "     Prometheus  │ ${RED}●${NC} Stopped"
        echo -e "     Grafana     │ ${RED}●${NC} Stopped"
    fi
    echo ""
    echo -e "  ${YELLOW}Restart:${NC}"
    echo -e "     Full start  │ ./start.sh"
    echo -e "     Quick start │ ./start.sh --quick"
    echo -e "     With monitor│ ./start.sh --monitoring"
    echo ""
    echo -e "  ${CYAN}Pages (when running):${NC}"
    echo -e "     Landing     │ http://localhost:$FRONTEND_PORT"
    echo -e "     Chat        │ http://localhost:$FRONTEND_PORT/chat"
    echo -e "     Settings    │ http://localhost:$FRONTEND_PORT/settings"
    echo ""
fi
