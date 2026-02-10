#!/bin/bash
# ShikshaSetu Deployment Verification Script
# Validates all critical services are healthy after deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3001}"
VLLM_URL="${VLLM_URL:-http://localhost:8001}"

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
}

# Check functions
check_api_health() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    log_info "Checking API health endpoint..."

    if curl -sf "${API_URL}/health" > /dev/null 2>&1; then
        local response=$(curl -s "${API_URL}/health")
        if echo "$response" | grep -q "healthy"; then
            log_success "API health endpoint responding"
            return 0
        else
            log_error "API health endpoint returned unexpected response: $response"
            return 1
        fi
    else
        log_error "API health endpoint not accessible at ${API_URL}/health"
        return 1
    fi
}

check_api_status() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    log_info "Checking API status endpoint..."

    if curl -sf "${API_URL}/health" > /dev/null 2>&1; then
        local response=$(curl -s "${API_URL}/health")
        log_success "API status endpoint responding"
        echo "   Status: $response"
        return 0
    else
        log_warn "API status endpoint not accessible (may be auth-protected)"
        return 0
    fi
}

check_api_docs() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    log_info "Checking API documentation..."

    if curl -sf "${API_URL}/docs" > /dev/null 2>&1; then
        log_success "API documentation accessible at ${API_URL}/docs"
        return 0
    else
        log_warn "API documentation not accessible"
        return 0
    fi
}

check_database() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    log_info "Checking PostgreSQL database..."

    if command -v psql > /dev/null 2>&1; then
        if PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "SELECT 1" > /dev/null 2>&1; then
            log_success "PostgreSQL database accessible and responding"

            # Check table count
            local table_count=$(PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
            echo "   Tables: $table_count"
            return 0
        else
            log_error "Cannot connect to PostgreSQL database"
            return 1
        fi
    else
        log_warn "psql not installed, skipping database check"
        return 0
    fi
}

check_redis() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    log_info "Checking Redis cache..."

    if command -v redis-cli > /dev/null 2>&1; then
        if redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ping > /dev/null 2>&1; then
            log_success "Redis cache accessible and responding"

            # Check key count
            local key_count=$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" DBSIZE | awk '{print $2}')
            echo "   Keys: $key_count"
            return 0
        else
            log_error "Cannot connect to Redis cache"
            return 1
        fi
    else
        log_warn "redis-cli not installed, skipping Redis check"
        return 0
    fi
}

check_celery_workers() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    log_info "Checking Celery workers..."

    if command -v celery > /dev/null 2>&1; then
        if celery -A backend.tasks inspect active > /dev/null 2>&1; then
            local worker_count=$(celery -A backend.tasks inspect active 2>/dev/null | grep -c "@" || echo "0")
            if [ "$worker_count" -gt 0 ]; then
                log_success "Celery workers running ($worker_count active)"
                return 0
            else
                log_warn "Celery broker reachable but no active workers"
                return 0
            fi
        else
            log_warn "Cannot inspect Celery workers (may not be running)"
            return 0
        fi
    else
        log_warn "Celery not installed, skipping worker check"
        return 0
    fi
}

check_prometheus() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    log_info "Checking Prometheus..."

    if curl -sf "${PROMETHEUS_URL}/-/healthy" > /dev/null 2>&1; then
        log_success "Prometheus is healthy"

        # Check targets
        local targets_up=$(curl -s "${PROMETHEUS_URL}/api/v1/targets" | grep -o '"health":"up"' | wc -l | tr -d ' ')
        echo "   Targets up: $targets_up"
        return 0
    else
        log_warn "Prometheus not accessible at ${PROMETHEUS_URL}"
        return 0
    fi
}

check_grafana() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    log_info "Checking Grafana..."

    if curl -sf "${GRAFANA_URL}/api/health" > /dev/null 2>&1; then
        log_success "Grafana is healthy"
        return 0
    else
        log_warn "Grafana not accessible at ${GRAFANA_URL}"
        return 0
    fi
}

check_ml_services() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    log_info "Checking ML services (vLLM)..."

    if curl -sf "${VLLM_URL}/health" > /dev/null 2>&1; then
        log_success "vLLM service is healthy"
        return 0
    else
        log_warn "vLLM service not accessible at ${VLLM_URL}"
        return 0
    fi
}

check_api_endpoints() {
    log_info "Checking critical API endpoints..."

    local endpoints=(
        "/api/v2/auth/login"
        "/api/v2/content/simplify"
        "/api/v2/content/translate"
        "/api/v2/chat/guest"
    )

    for endpoint in "${endpoints[@]}"; do
        TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
        if curl -sf "${API_URL}${endpoint}" -X POST -H "Content-Type: application/json" -d '{}' > /dev/null 2>&1; then
            log_success "Endpoint ${endpoint} responding"
        else
            # POST with empty body should return 422 or 401, not 404
            local status=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}${endpoint}" -X POST -H "Content-Type: application/json" -d '{}')
            if [ "$status" == "422" ] || [ "$status" == "401" ] || [ "$status" == "400" ]; then
                log_success "Endpoint ${endpoint} responding (HTTP $status)"
            else
                log_warn "Endpoint ${endpoint} returned HTTP $status"
            fi
        fi
    done
}

check_metrics_endpoint() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    log_info "Checking Prometheus metrics endpoint..."

    if curl -sf "${API_URL}/metrics" > /dev/null 2>&1; then
        local metrics=$(curl -s "${API_URL}/metrics" | head -20)
        log_success "Metrics endpoint responding"
        echo "   Sample metrics:"
        echo "$metrics" | grep -E "^(http_requests|db_|cache_)" | head -5 | sed 's/^/     /'
        return 0
    else
        log_warn "Metrics endpoint not accessible"
        return 0
    fi
}

check_disk_space() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    log_info "Checking disk space..."

    local disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$disk_usage" -lt 80 ]; then
        log_success "Disk usage: ${disk_usage}%"
        return 0
    elif [ "$disk_usage" -lt 90 ]; then
        log_warn "Disk usage high: ${disk_usage}%"
        return 0
    else
        log_error "Disk usage critical: ${disk_usage}%"
        return 1
    fi
}

check_memory() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    log_info "Checking memory..."

    if command -v free > /dev/null 2>&1; then
        local mem_usage=$(free | awk 'NR==2 {printf "%.0f", $3/$2 * 100}')
        if [ "$mem_usage" -lt 80 ]; then
            log_success "Memory usage: ${mem_usage}%"
            return 0
        elif [ "$mem_usage" -lt 90 ]; then
            log_warn "Memory usage high: ${mem_usage}%"
            return 0
        else
            log_error "Memory usage critical: ${mem_usage}%"
            return 1
        fi
    else
        log_warn "Cannot check memory (free command not available)"
        return 0
    fi
}

# Main execution
main() {
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     ShikshaSetu Deployment Verification                   ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    # Load environment variables if .env exists
    if [ -f .env ]; then
        log_info "Loading environment variables from .env"
        export $(cat .env | grep -v '^#' | xargs)
    fi

    echo "Configuration:"
    echo "  API URL: ${API_URL}"
    echo "  Database: ${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
    echo "  Redis: ${REDIS_HOST}:${REDIS_PORT}"
    echo "  Prometheus: ${PROMETHEUS_URL}"
    echo "  Grafana: ${GRAFANA_URL}"
    echo ""

    # Run all checks
    check_api_health
    check_api_status
    check_api_docs
    check_database
    check_redis
    check_celery_workers
    check_prometheus
    check_grafana
    check_ml_services
    check_api_endpoints
    check_metrics_endpoint
    check_disk_space
    check_memory

    # Summary
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     Verification Summary                                   ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Total Checks:   ${TOTAL_CHECKS}"
    echo -e "${GREEN}Passed:        ${PASSED_CHECKS}${NC}"
    echo -e "${YELLOW}Warnings:      ${WARNING_CHECKS}${NC}"
    echo -e "${RED}Failed:        ${FAILED_CHECKS}${NC}"
    echo ""

    if [ "$FAILED_CHECKS" -eq 0 ]; then
        if [ "$WARNING_CHECKS" -eq 0 ]; then
            log_success "All checks passed! ✨"
            echo ""
            echo "Deployment is ready for production use."
            exit 0
        else
            log_warn "All critical checks passed but some warnings present"
            echo ""
            echo "Deployment is functional but review warnings above."
            exit 0
        fi
    else
        log_error "Some critical checks failed"
        echo ""
        echo "Please fix the errors above before proceeding."
        exit 1
    fi
}

main "$@"
