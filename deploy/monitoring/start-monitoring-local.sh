#!/bin/bash
# Oryon Local Monitoring Setup Script
# Quick start script for local development monitoring

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "🚀 Oryon Local Monitoring Setup"
echo "======================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi
echo "✅ Docker is running"

# Navigate to monitoring directory
cd "$SCRIPT_DIR"

# Stop any existing monitoring containers
echo ""
echo "🧹 Cleaning up any existing monitoring containers..."
docker-compose -f docker-compose.monitoring.local.yml down 2>/dev/null || true

# Start the monitoring stack
echo ""
echo "📦 Starting Prometheus and Grafana..."
docker-compose -f docker-compose.monitoring.local.yml up -d

# Wait for services to be healthy
echo ""
echo "⏳ Waiting for services to be ready..."
max_wait=60
elapsed=0

while [ $elapsed -lt $max_wait ]; do
    prometheus_ready=false
    grafana_ready=false

    # Check Prometheus
    if curl -sf http://localhost:9090/-/healthy > /dev/null 2>&1; then
        prometheus_ready=true
    fi

    # Check Grafana
    if curl -sf http://localhost:3001/api/health > /dev/null 2>&1; then
        grafana_ready=true
    fi

    if [ "$prometheus_ready" = true ] && [ "$grafana_ready" = true ]; then
        echo ""
        echo "✅ All services are healthy!"
        break
    fi

    echo -n "."
    sleep 3
    elapsed=$((elapsed + 3))
done

if [ $elapsed -ge $max_wait ]; then
    echo ""
    echo "⚠️  Services may not be fully ready. Check logs with:"
    echo "   docker-compose -f docker-compose.monitoring.local.yml logs"
fi

# Test if backend metrics are available
echo ""
echo "🔍 Checking if backend metrics endpoint is available..."
if curl -sf http://localhost:8000/metrics > /dev/null 2>&1; then
    echo "✅ Backend metrics endpoint is accessible at http://localhost:8000/metrics"
else
    echo "⚠️  Backend metrics endpoint not accessible. Make sure the backend is running!"
    echo "   Start it with: ./start.sh"
fi

# Display access information
echo ""
echo "🎉 Local Monitoring Stack Ready!"
echo "=================================="
echo ""
echo "📊 Access URLs:"
echo "   Prometheus:  http://localhost:9090"
echo "   Grafana:     http://localhost:3001"
echo ""
echo "🔑 Grafana Login:"
echo "   Username: admin"
echo "   Password: admin"
echo ""
echo "📈 Quick Start:"
echo "   1. Open Grafana: http://localhost:3001"
echo "   2. Login with admin/admin"
echo "   3. Go to Dashboards → Browse → Oryon"
echo "   4. Check Prometheus targets: http://localhost:9090/targets"
echo ""
echo "🛠️  Commands:"
echo "   View logs:   docker-compose -f $SCRIPT_DIR/docker-compose.monitoring.local.yml logs -f"
echo "   Stop stack:  docker-compose -f $SCRIPT_DIR/docker-compose.monitoring.local.yml down"
echo "   Restart:     docker-compose -f $SCRIPT_DIR/docker-compose.monitoring.local.yml restart"
echo ""
