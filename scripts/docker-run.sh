#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() { echo "Usage: $0 {build|run|full|test}"; exit 1; }

case "${1:-}" in
  build)
    docker build -t llamacpp-dev "$ROOT_DIR"
    ;;
  run)
    docker run --gpus all --rm -it -v "$ROOT_DIR":/workspace -w /workspace llamacpp-dev bash -lc "\
      python3.12 -m venv .venv && \
      . .venv/bin/activate && \
      python3.12 -m pip install --upgrade pip setuptools wheel && \
      pip install -e . && \
      pip install -r requirements.txt && \
      pytest -q tests/test_speculative_support.py tests/test_llamacpp_install.py\"
    ;;
  full)
    docker build -t llamacpp-dev "$ROOT_DIR" && \
    docker run --gpus all --rm -it -v "$ROOT_DIR":/workspace -w /workspace llamacpp-dev bash -lc "\
      python3.12 -m venv .venv && \
      . .venv/bin/activate && \
      python3.12 -m pip install --upgrade pip setuptools wheel && \
      pip install -e . && \
      pip install -r requirements.txt && \
      ./llamacpp_stack/bundle/install_llamacpp_stack.sh --dry-run || true && \
      pytest -q\"
    ;;
  test)
    docker run --gpus all --rm -it -v "$ROOT_DIR":/workspace -w /workspace llamacpp-dev bash -lc "\
      python3.12 -m venv .venv && \
      . .venv/bin/activate && \
      python3.12 -m pip install --upgrade pip setuptools wheel && \
      pip install -e . && \
      pip install -r requirements.txt && \
      pytest -q\"
    ;;
  *)
    usage
    ;;
esac
