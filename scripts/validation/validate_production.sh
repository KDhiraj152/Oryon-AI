#!/bin/bash

# Production Configuration Validator
# Validates production deployment configuration without starting services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# Helper functions
log_success() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

log_error() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

log_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Validation functions

check_docker_compose_syntax() {
    log_info "Checking docker-compose.production.yml syntax..."

    local output=$(docker-compose -f "$PROJECT_ROOT/docker-compose.production.yml" config 2>&1)
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_success "docker-compose.production.yml syntax is valid"
    else
        log_error "docker-compose.production.yml has syntax errors"
        echo "$output" | grep -i "error"
    fi
}

check_env_file() {
    log_info "Checking production environment configuration..."

    if [ -f "$PROJECT_ROOT/.env.production" ]; then
        log_success ".env.production exists"

        # Check for required variables
        required_vars=(
            "POSTGRES_PASSWORD"
            "REDIS_PASSWORD"
            "JWT_SECRET_KEY"
            "JWT_REFRESH_SECRET_KEY"
            "OPENAI_API_KEY"
            "BHASHINI_API_KEY"
        )

        for var in "${required_vars[@]}"; do
            if grep -q "^${var}=" "$PROJECT_ROOT/.env.production"; then
                value=$(grep "^${var}=" "$PROJECT_ROOT/.env.production" | cut -d'=' -f2)
                if [ "$value" == "changeme" ] || [ "$value" == "your-" ] || [ -z "$value" ]; then
                    log_warning "$var is set to placeholder value"
                else
                    log_success "$var is configured"
                fi
            else
                log_error "$var is not set in .env.production"
            fi
        done
    else
        log_error ".env.production not found"
        log_info "Copy .env.production.example to .env.production and configure it"
    fi
}

check_dockerfiles() {
    log_info "Checking Dockerfile existence and syntax..."

    # Check backend Dockerfile
    if [ -f "$PROJECT_ROOT/infrastructure/docker/Dockerfile.backend" ]; then
        log_success "Backend Dockerfile exists"

        # Validate Dockerfile syntax
        if docker build -f "$PROJECT_ROOT/infrastructure/docker/Dockerfile.backend" -t oryon-backend:test "$PROJECT_ROOT" --dry-run 2>&1 | grep -q "error"; then
            log_error "Backend Dockerfile has syntax errors"
        fi
    else
        log_error "Backend Dockerfile not found at infrastructure/docker/Dockerfile.backend"
    fi

    # Check frontend Dockerfile
    if [ -f "$PROJECT_ROOT/infrastructure/docker/Dockerfile.frontend" ]; then
        log_success "Frontend Dockerfile exists"
    else
        log_error "Frontend Dockerfile not found at infrastructure/docker/Dockerfile.frontend"
    fi
}

check_nginx_config() {
    log_info "Checking Nginx configuration..."

    if [ -f "$PROJECT_ROOT/infrastructure/nginx/nginx.conf" ]; then
        log_success "nginx.conf exists"

        # Test nginx configuration syntax
        if command -v nginx &> /dev/null; then
            if nginx -t -c "$PROJECT_ROOT/infrastructure/nginx/nginx.conf" 2>&1 | grep -q "successful"; then
                log_success "nginx.conf syntax is valid"
            else
                log_warning "nginx.conf syntax check failed (may need to run in container)"
            fi
        else
            log_info "nginx not installed locally, skipping syntax check"
        fi
    else
        log_error "nginx.conf not found at infrastructure/nginx/nginx.conf"
    fi

    # Check frontend nginx config
    if [ -f "$PROJECT_ROOT/infrastructure/nginx/frontend.conf" ]; then
        log_success "frontend.conf exists"
    else
        log_error "frontend.conf not found"
    fi
}

check_ssl_certificates() {
    log_info "Checking SSL certificate setup..."

    if [ -d "$PROJECT_ROOT/infrastructure/nginx/ssl" ]; then
        log_success "SSL directory exists"

        if [ -f "$PROJECT_ROOT/infrastructure/nginx/ssl/fullchain.pem" ] && [ -f "$PROJECT_ROOT/infrastructure/nginx/ssl/privkey.pem" ]; then
            log_success "SSL certificates are present"

            # Check certificate validity
            if command -v openssl &> /dev/null; then
                if openssl x509 -in "$PROJECT_ROOT/infrastructure/nginx/ssl/fullchain.pem" -noout -checkend 86400 > /dev/null 2>&1; then
                    log_success "SSL certificate is valid for at least 24 hours"
                else
                    log_error "SSL certificate is expired or will expire within 24 hours"
                fi
            fi
        else
            log_warning "SSL certificates not found (required for production)"
            log_info "Generate certificates with: certbot certonly --standalone -d yourdomain.com"
        fi
    else
        log_error "SSL directory not found at infrastructure/nginx/ssl"
    fi
}

check_monitoring_config() {
    log_info "Checking monitoring configuration..."

    files=(
        "infrastructure/monitoring/prometheus.yml"
        "infrastructure/monitoring/prometheus-alerts.yml"
        "infrastructure/monitoring/alertmanager.yml"
        "infrastructure/monitoring/grafana-dashboard.json"
    )

    for file in "${files[@]}"; do
        if [ -f "$PROJECT_ROOT/$file" ]; then
            log_success "$(basename $file) exists"
        else
            log_error "$file not found"
        fi
    done
}

check_scripts() {
    log_info "Checking deployment scripts..."

    scripts=(
        "bin/verify-deployment"
        "scripts/backup-postgres.sh"
        "infrastructure/monitoring/setup-monitoring.sh"
    )

    for script in "${scripts[@]}"; do
        if [ -f "$PROJECT_ROOT/$script" ]; then
            if [ -x "$PROJECT_ROOT/$script" ]; then
                log_success "$script exists and is executable"
            else
                log_warning "$script exists but is not executable"
            fi
        else
            log_error "$script not found"
        fi
    done
}

check_directory_structure() {
    log_info "Checking required directory structure..."

    dirs=(
        "data/uploads"
        "data/cache"
        "data/models"
        "logs"
        "backups/postgres"
        "infrastructure/monitoring/prometheus"
        "infrastructure/monitoring/grafana"
        "infrastructure/monitoring/alertmanager"
    )

    for dir in "${dirs[@]}"; do
        if [ -d "$PROJECT_ROOT/$dir" ]; then
            log_success "$dir exists"
        else
            log_warning "$dir does not exist (will be created by Docker)"
        fi
    done
}

check_service_resources() {
    log_info "Checking service resource definitions..."

    # Check if resource limits are defined in docker-compose
    if grep -q "resources:" "$PROJECT_ROOT/docker-compose.production.yml"; then
        log_success "Resource limits are defined"
    else
        log_warning "No resource limits defined in docker-compose.production.yml"
    fi

    # Check if health checks are defined
    if grep -q "healthcheck:" "$PROJECT_ROOT/docker-compose.production.yml"; then
        log_success "Health checks are defined"
    else
        log_error "No health checks defined"
    fi
}

check_github_workflows() {
    log_info "Checking GitHub Actions workflows..."

    workflows=(
        ".github/workflows/test.yml"
        ".github/workflows/build.yml"
        ".github/workflows/deploy-staging.yml"
        ".github/workflows/deploy-production.yml"
    )

    for workflow in "${workflows[@]}"; do
        if [ -f "$PROJECT_ROOT/$workflow" ]; then
            log_success "$(basename $workflow) exists"
        else
            log_error "$workflow not found"
        fi
    done
}

# Main validation flow
main() {
    echo "====================================="
    echo "Production Configuration Validator"
    echo "====================================="
    echo ""

    cd "$PROJECT_ROOT"

    check_docker_compose_syntax
    echo ""

    check_env_file
    echo ""

    check_dockerfiles
    echo ""

    check_nginx_config
    echo ""

    check_ssl_certificates
    echo ""

    check_monitoring_config
    echo ""

    check_scripts
    echo ""

    check_directory_structure
    echo ""

    check_service_resources
    echo ""

    check_github_workflows
    echo ""

    # Summary
    echo "====================================="
    echo "Validation Summary"
    echo "====================================="
    echo -e "${GREEN}Passed:${NC} $PASSED"
    echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
    echo -e "${RED}Failed:${NC} $FAILED"
    echo ""

    if [ $FAILED -eq 0 ]; then
        if [ $WARNINGS -eq 0 ]; then
            echo -e "${GREEN}✓ Production configuration is ready for deployment!${NC}"
            exit 0
        else
            echo -e "${YELLOW}⚠ Production configuration has warnings but can proceed${NC}"
            exit 0
        fi
    else
        echo -e "${RED}✗ Production configuration has errors that must be fixed${NC}"
        exit 1
    fi
}

main "$@"
