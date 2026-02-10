#!/bin/bash
# ShikshaSetu Monitoring Stack Setup Script
# Sets up Prometheus, Grafana, and Alertmanager for production monitoring

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "🚀 ShikshaSetu Monitoring Stack Setup"
echo "======================================"

# Check for required environment variables
check_env_vars() {
    echo "📋 Checking environment variables..."
    
    local required_vars=(
        "SLACK_WEBHOOK_URL"
        "PAGERDUTY_SERVICE_KEY"
        "GRAFANA_ADMIN_PASSWORD"
        "POSTGRES_USER"
        "POSTGRES_PASSWORD"
        "POSTGRES_DB"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        echo "❌ Missing required environment variables:"
        printf '   - %s\n' "${missing_vars[@]}"
        echo ""
        echo "Please set these in your .env file or export them:"
        echo "  export SLACK_WEBHOOK_URL='https://hooks.slack.com/...'"
        echo "  export PAGERDUTY_SERVICE_KEY='your-service-key'"
        echo "  export GRAFANA_ADMIN_PASSWORD='secure-password'"
        exit 1
    fi
    
    echo "✅ All required environment variables are set"
}

# Validate configuration files
validate_configs() {
    echo ""
    echo "🔍 Validating configuration files..."
    
    local configs=(
        "prometheus.yml"
        "prometheus-alerts.yml"
        "alertmanager.yml"
        "grafana-dashboard.json"
        "grafana-datasources.yml"
        "grafana-dashboards.yml"
        "docker-compose.monitoring.yml"
    )
    
    local missing_configs=()
    
    for config in "${configs[@]}"; do
        if [ ! -f "$SCRIPT_DIR/$config" ]; then
            missing_configs+=("$config")
        fi
    done
    
    if [ ${#missing_configs[@]} -gt 0 ]; then
        echo "❌ Missing configuration files:"
        printf '   - %s\n' "${missing_configs[@]}"
        exit 1
    fi
    
    echo "✅ All configuration files present"
}

# Check if Docker is running
check_docker() {
    echo ""
    echo "🐳 Checking Docker..."
    
    if ! docker info > /dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker and try again."
        exit 1
    fi
    
    if ! command -v docker-compose > /dev/null 2>&1; then
        echo "❌ docker-compose not found. Please install docker-compose."
        exit 1
    fi
    
    echo "✅ Docker is running"
}

# Create necessary directories
setup_directories() {
    echo ""
    echo "📁 Creating data directories..."
    
    mkdir -p "$SCRIPT_DIR/data/prometheus"
    mkdir -p "$SCRIPT_DIR/data/grafana"
    mkdir -p "$SCRIPT_DIR/data/alertmanager"
    
    # Set permissions
    chmod 777 "$SCRIPT_DIR/data/prometheus"
    chmod 777 "$SCRIPT_DIR/data/grafana"
    chmod 777 "$SCRIPT_DIR/data/alertmanager"
    
    echo "✅ Directories created"
}

# Start monitoring stack
start_stack() {
    echo ""
    echo "🚀 Starting monitoring stack..."
    
    cd "$SCRIPT_DIR"
    docker-compose -f docker-compose.monitoring.yml up -d
    
    echo "✅ Monitoring stack started"
}

# Wait for services to be healthy
wait_for_services() {
    echo ""
    echo "⏳ Waiting for services to be healthy..."
    
    local max_wait=120
    local elapsed=0
    
    while [ $elapsed -lt $max_wait ]; do
        local all_healthy=true
        
        # Check Prometheus
        if ! curl -sf http://localhost:9090/-/healthy > /dev/null 2>&1; then
            all_healthy=false
        fi
        
        # Check Alertmanager
        if ! curl -sf http://localhost:9093/-/healthy > /dev/null 2>&1; then
            all_healthy=false
        fi
        
        # Check Grafana
        if ! curl -sf http://localhost:3001/api/health > /dev/null 2>&1; then
            all_healthy=false
        fi
        
        if [ "$all_healthy" = true ]; then
            echo "✅ All services are healthy"
            return 0
        fi
        
        echo -n "."
        sleep 5
        elapsed=$((elapsed + 5))
    done
    
    echo ""
    echo "⚠️  Services did not become healthy within ${max_wait}s"
    echo "   You can check the logs with: docker-compose -f $SCRIPT_DIR/docker-compose.monitoring.yml logs"
}

# Display access information
show_access_info() {
    echo ""
    echo "🎉 Monitoring Stack Setup Complete!"
    echo "===================================="
    echo ""
    echo "📊 Access URLs:"
    echo "   Prometheus:    http://localhost:9090"
    echo "   Alertmanager:  http://localhost:9093"
    echo "   Grafana:       http://localhost:3001"
    echo ""
    echo "🔑 Grafana Credentials:"
    echo "   Username: admin"
    echo "   Password: ${GRAFANA_ADMIN_PASSWORD}"
    echo ""
    echo "📈 Next Steps:"
    echo "   1. Open Grafana: http://localhost:3001"
    echo "   2. Navigate to Dashboards → ShikshaSetu"
    echo "   3. Check Prometheus targets: http://localhost:9090/targets"
    echo "   4. View active alerts: http://localhost:9093/#/alerts"
    echo ""
    echo "📚 Documentation:"
    echo "   Runbook: $PROJECT_ROOT/DEPLOYMENT_RUNBOOK.md"
    echo "   Monitoring: $PROJECT_ROOT/docs/MONITORING.md"
    echo ""
    echo "🛠️  Useful Commands:"
    echo "   View logs:     docker-compose -f $SCRIPT_DIR/docker-compose.monitoring.yml logs -f"
    echo "   Stop stack:    docker-compose -f $SCRIPT_DIR/docker-compose.monitoring.yml down"
    echo "   Restart stack: docker-compose -f $SCRIPT_DIR/docker-compose.monitoring.yml restart"
    echo ""
}

# Main execution
main() {
    check_env_vars
    validate_configs
    check_docker
    setup_directories
    start_stack
    wait_for_services
    show_access_info
}

main "$@"
