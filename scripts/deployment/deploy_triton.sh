#!/bin/bash
# Deploy NVIDIA Triton Inference Server for ONNX models

set -e

echo "🚀 Deploying NVIDIA Triton Inference Server"

# Configuration
TRITON_PORT=${TRITON_PORT:-8002}
TRITON_METRICS_PORT=${TRITON_METRICS_PORT:-8003}
TRITON_GRPC_PORT=${TRITON_GRPC_PORT:-8004}
MODEL_REPOSITORY=${MODEL_REPOSITORY:-/models}

echo "⚙️  Triton Configuration:"
echo "   HTTP Port: $TRITON_PORT"
echo "   Metrics Port: $TRITON_METRICS_PORT"
echo "   gRPC Port: $TRITON_GRPC_PORT"
echo "   Model Repository: $MODEL_REPOSITORY"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "   Available GPUs: $GPU_COUNT"
else
    echo "⚠️  No GPU detected, using CPU mode"
fi

# Pull models from DVC
echo "📥 Pulling ONNX models from DVC..."
dvc pull models/all-MiniLM-L6-v2.onnx.dvc -r $DVC_REMOTE || echo "⚠️  DVC pull failed, using local models"
dvc pull models/indictrans2.onnx.dvc -r $DVC_REMOTE || echo "⚠️  DVC pull failed, using local models"

# Start Triton using Docker
echo "🚀 Starting Triton Inference Server..."

docker run -d \
    --name triton-inference-server \
    --gpus all \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p $TRITON_PORT:8000 \
    -p $TRITON_METRICS_PORT:8001 \
    -p $TRITON_GRPC_PORT:8002 \
    -v $MODEL_REPOSITORY:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver \
        --model-repository=/models \
        --strict-model-config=false \
        --log-verbose=1 \
        --model-control-mode=explicit \
        --load-model=embedding-rag \
        --load-model=translation-indic

CONTAINER_ID=$(docker ps -q -f name=triton-inference-server)
echo "✅ Triton started in container: $CONTAINER_ID"

# Wait for health check
echo "🔍 Waiting for Triton to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://localhost:$TRITON_PORT/v2/health/ready" | grep -q "true"; then
        echo "✅ Triton is healthy!"
        break
    fi
    echo "   Attempt $((RETRY_COUNT + 1))/$MAX_RETRIES..."
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ Triton failed to start"
    docker stop triton-inference-server 2>/dev/null || true
    exit 1
fi

# List loaded models
echo "📋 Loaded models:"
curl -s "http://localhost:$TRITON_PORT/v2/models" | jq .

echo "✅ Triton deployment complete!"
echo "   HTTP Endpoint: http://localhost:$TRITON_PORT"
echo "   Metrics: http://localhost:$TRITON_METRICS_PORT/metrics"
echo "   gRPC Endpoint: localhost:$TRITON_GRPC_PORT"
echo "   Container ID: $CONTAINER_ID"
