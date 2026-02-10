#!/bin/bash
# Deploy vLLM model serving for production

set -e

echo "🚀 Deploying vLLM Model Serving"

# Check if running in production
ENVIRONMENT=${ENVIRONMENT:-development}
if [ "$ENVIRONMENT" = "production" ]; then
    echo "📦 Production deployment mode"
else
    echo "🔧 Development deployment mode"
fi

# Pull latest models from DVC
echo "📥 Pulling models from DVC..."
MODEL_VERSION=${MODEL_VERSION:-latest}
dvc pull models/mistral-7b-instruct.dvc -r $DVC_REMOTE || echo "⚠️  DVC pull failed, using local models"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "   Available GPUs: $GPU_COUNT"
    TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-$GPU_COUNT}
else
    echo "⚠️  No GPU detected, using CPU mode"
    TENSOR_PARALLEL_SIZE=1
fi

# vLLM Configuration
VLLM_PORT=${VLLM_PORT:-8001}
MODEL_PATH=${LLM_MODEL_PATH:-models/mistral-7b-instruct}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
QUANTIZATION=${QUANTIZATION:-awq}

echo "⚙️  vLLM Configuration:"
echo "   Model: $MODEL_PATH"
echo "   Port: $VLLM_PORT"
echo "   Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "   GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo "   Max Model Length: $MAX_MODEL_LEN"
echo "   Quantization: $QUANTIZATION"

# Start vLLM server
echo "🚀 Starting vLLM server..."

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --quantization "$QUANTIZATION" \
    --enable-prefix-caching \
    --disable-log-requests \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    &

VLLM_PID=$!
echo "✅ vLLM started with PID: $VLLM_PID"

# Wait for health check
echo "🔍 Waiting for vLLM to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "✅ vLLM is healthy!"
        break
    fi
    echo "   Attempt $((RETRY_COUNT + 1))/$MAX_RETRIES..."
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ vLLM failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Test inference
echo "🧪 Testing inference..."
curl -X POST "http://localhost:$VLLM_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_PATH"'",
        "prompt": "Explain photosynthesis in simple terms:",
        "max_tokens": 100,
        "temperature": 0.7
    }' | jq .

echo "✅ vLLM deployment complete!"
echo "   Endpoint: http://localhost:$VLLM_PORT"
echo "   Health check: http://localhost:$VLLM_PORT/health"
echo "   PID: $VLLM_PID"

# Keep running in foreground for Docker
if [ "${RUN_FOREGROUND:-false}" = "true" ]; then
    echo "Running in foreground mode..."
    wait $VLLM_PID
fi
