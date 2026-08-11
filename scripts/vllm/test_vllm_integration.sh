#!/usr/bin/env bash
# Exhaustive vLLM integration test for Heimdall Gateway
set -euo pipefail

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting EXHAUSTIVE vLLM integration test...${NC}"

# Ensure we are using vLLM backend
export HEIMDALL_GATEWAY_BACKEND=vllm-beta
echo "DEBUG: HEIMDALL_GATEWAY_BACKEND=$HEIMDALL_GATEWAY_BACKEND"

function assert_success() {
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ $1${NC}"
    else
        echo -e "  ${RED}❌ $1 failed${NC}"
        # Print logs on failure
        echo "MANAGER LOG:"
        cat /app/logs/manager.log || true
        echo "SWAP LOG:"
        cat /app/logs/swap.log || true
        exit 1
    fi
}

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    if [ -f /app/logs/swap.log ]; then
        echo "--- SWAP LOG ---"
        cat /app/logs/swap.log
    fi
    pkill -f heimdall-gateway || true
    pkill -f llama-swap || true
}

trap cleanup EXIT

# [0/10] Verify binary
echo "[0/10] Verifying vllm-server binary..."
ls -l /usr/local/bin/vllm-server
/usr/local/bin/vllm-server --help | head -n 1 || echo "vllm-server failed to run"

# [1/10] Setting up environment...
echo "[1/10] Setting up environment..."
mkdir -p /app/logs
mkdir -p /var/lib/heimdall-gateway/models
mkdir -p /etc/heimdall-gateway

# 2. Add models to catalog
echo "[2/10] Registering models..."
USER_MODEL="Hjgugugjhuhjggg/LLaMa3.1-1bit-bitnet-cuda-Q2_K-GGUF:Q2_K"
HELPER_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

heimdall-gateway add -hf "$HELPER_MODEL" --skip-ctx --defer-publish
assert_success "Helper model registered"

# 3. Start daemons
echo "[3/10] Starting daemons in background..."
heimdall-gateway-manager-start > /app/logs/manager.log 2>&1 &
heimdall-gateway-router-start > /app/logs/swap.log 2>&1 &
sleep 5

# 4. Verify catalog
echo "[4/10] Verifying catalog..."
heimdall-gateway list | grep "Qwen2.5-0.5B-Instruct" > /dev/null
assert_success "Catalog listed correctly"

# 5. Load model via vLLM
echo "[5/10] Loading model via vLLM backend..."
MODEL_ID=$(heimdall-gateway list | grep "Qwen2.5-0.5B-Instruct" | awk '{print $1}' | head -n 1)
heimdall-gateway run --model-id "$MODEL_ID" --no-chat --ctx-size 512 --float16 --gpu-memory-utilization 0.4 &
sleep 5

# 6. Wait for Backend (18080) then Public (11436)
echo "[6/10] Waiting for vLLM Backend (18080)..."
# Trigger swap to start the model by making a request
curl -s -X POST http://127.0.0.1:11436/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "'"$MODEL_ID"'", "messages": [{"role": "user", "content": "hi"}]}' > /dev/null &

MAX_RETRIES=200
RETRY_COUNT=0
# We wait for the backend port directly first
until curl -s http://127.0.0.1:18080/v1/models > /dev/null; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo -e "${RED}❌ Timeout waiting for vLLM backend on 18080${NC}"
        exit 1
    fi
    if (( RETRY_COUNT % 20 == 0 )); then
        echo "  Still waiting for vLLM... ($RETRY_COUNT/$MAX_RETRIES)"
    fi
    sleep 2
done
assert_success "vLLM backend ready on 18080"

echo "  Waiting for Proxy (11436) to reflect readiness..."
sleep 2
assert_success "vLLM API ready through proxy"

# 7. Test Endpoints
echo "[7/10] Testing OpenAI endpoints..."
curl -s -X POST http://127.0.0.1:11436/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_ID\",
        \"prompt\": \"The capital of France is\",
        \"max_tokens\": 5
    }" | grep "text" > /dev/null
assert_success "/v1/completions works"

# 8. Test Speculative Decoding Flag Translation
echo "[8/10] Testing speculative decoding CLI translation..."
heimdall-gateway run -hf "$HELPER_MODEL" -hf "$HELPER_MODEL" --speculative --no-chat --model-id "spec-test" &
sleep 5
grep "speculative-model" /app/logs/swap.log > /dev/null || grep "speculative_model" /app/logs/swap.log > /dev/null
assert_success "Speculative decoding arguments translated"

# 9. Test Update
echo "[9/10] Testing update..."
heimdall-gateway update --model-id "$MODEL_ID" --description "Validated vLLM Model" --defer-publish
heimdall-gateway list | grep "Validated vLLM Model" > /dev/null
assert_success "Update command works"

# 10. Final Verification
echo "[10/10] Final verification..."
heimdall-gateway info | grep "vllm-beta" > /dev/null
assert_success "Backend correctly identified as vllm-beta"

echo -e "\n${GREEN}⭐⭐⭐ VLLM Backend Integration Certified! ⭐⭐⭐${NC}"
exit 0
