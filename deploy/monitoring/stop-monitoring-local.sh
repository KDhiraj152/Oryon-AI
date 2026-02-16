#!/bin/bash
# Oryon Stop Local Monitoring
# Stops the local monitoring stack

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "🛑 Stopping Oryon Local Monitoring Stack..."

cd "$SCRIPT_DIR"
docker-compose -f docker-compose.monitoring.local.yml down

echo "✅ Monitoring stack stopped"
