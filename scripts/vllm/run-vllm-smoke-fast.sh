#!/usr/bin/env bash
# Run the optimized smoke test using the checkpointed Docker image
set -euo pipefail

IMAGE_NAME="llamacpp-stack:vllm-checkpoint"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "🚀 Running fast vLLM smoke test..."

# Ensure the image exists (it should be building in the background, but this script might be run later)
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "📦 Image $IMAGE_NAME not found. Building it now..."
    docker build -t "$IMAGE_NAME" -f Dockerfile.vllm .
fi

# Run the smoke test
# We mount the current directory to /workspace so any changes are reflected
# We set SKIP_INSTALL=1 to avoid repeating pip installs
docker run --rm --gpus all \
    --shm-size=16g \
    -v "$ROOT_DIR":/workspace \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e SKIP_INSTALL=1 \
    -e CUDA_VISIBLE_DEVICES=0 \
    "$IMAGE_NAME" \
    python3 /workspace/.tmp_vllm_smoke.py
