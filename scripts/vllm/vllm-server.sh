#!/bin/bash
# vLLM server wrapper to emulate llama-server CLI
# Converts llama-server args to vllm serve args

set -e

PORT=8000
HOST="0.0.0.0"
MODEL_PATH=""
CTX_SIZE=""
N_GPU_LAYERS=""
DTYPE="float16"
GPU_MEMORY=0.5

SPECULATIVE_MODEL=""
SPECULATIVE_TOKENS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-draft)
            SPECULATIVE_MODEL="$2"
            shift 2
            ;;
        --spec-draft-n-max)
            SPECULATIVE_TOKENS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --ctx-size)
            CTX_SIZE="$2"
            shift 2
            ;;
        -ngl|--n-gpu-layers)
            N_GPU_LAYERS="$2"
            shift 2
            ;;
        # Map float precision flags
        --f16|--float16)
            DTYPE="float16"
            shift
            ;;
        --f32|--float32)
            DTYPE="float32"
            shift
            ;;
        --bf16|--bfloat16)
            DTYPE="bfloat16"
            shift
            ;;
        # Ignore llama.cpp specific flags that don't apply to vLLM
        --fit|--fitc|--fitt|-fitc|-fitt|-fit)
            shift 2 2>/dev/null || shift
            ;;
        --threads|--threads-batch|--mirostat|--mirostat-ent|--mirostat-lr|--cache-type-k|--cache-type-v)
            shift 2
            ;;
        --keep|--draft)
            shift 2
            ;;
        # Ignore mmap and other llama.cpp-only parameters
        --mmap|--mlock|--no-mmap|--no-mlock)
            shift
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Validate model path
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model is required" >&2
    exit 1
fi

# Build vLLM command
VLLM_CMD=(
    "python3" "-m" "vllm.entrypoints.openai.api_server"
    "--model" "$MODEL_PATH"
    "--port" "$PORT"
    "--host" "$HOST"
    "--dtype" "$DTYPE"
    "--gpu-memory-utilization" "$GPU_MEMORY"
    "--disable-log-requests"
)

# Add context size if specified
if [[ -n "$CTX_SIZE" ]]; then
    VLLM_CMD+=("--max-model-len" "$CTX_SIZE")
fi

# Add speculative decoding if specified
if [[ -n "$SPECULATIVE_MODEL" ]]; then
    VLLM_CMD+=("--speculative-model" "$SPECULATIVE_MODEL")
    if [[ -n "$SPECULATIVE_TOKENS" ]]; then
        VLLM_CMD+=("--num-speculative-tokens" "$SPECULATIVE_TOKENS")
    fi
fi

# Add GPU layers (vLLM uses all by default, but we can limit if specified)
if [[ -n "$N_GPU_LAYERS" ]]; then
    # vLLM doesn't have direct --n-gpu-layers; it uses GPU for all layers by default
    # This is a no-op but we consume it to avoid errors
    :
fi

exec "${VLLM_CMD[@]}"
