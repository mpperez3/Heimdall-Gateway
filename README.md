# llamacpp-superserver

Manager for a local llama.cpp / llama-swap stack with **optional vLLM beta support**.

It provides:

- `llamacpp-superserver`, a Python control CLI and API shim.
- install/uninstall entry points for the full stack.
- a bundled shell installer in `llamacpp_stack/bundle/`.
- helpers for Hugging Face GGUF downloads, model cataloguing, runtime validation, llama-swap config rendering, and Ollama-compatible endpoints.
- **[BETA] vLLM integration** — alternative inference backend for HuggingFace models.

## Backends

### llama.cpp (Stable)
- Optimized for GGUF quantized models
- Excellent CPU and GPU support
- Native Flash Attention

### vLLM (Beta)
- OpenAI API compatible
- Optimized for HuggingFace format models
- High-throughput inference with batching
- See [docs/VLLM-BETA.md](docs/VLLM-BETA.md) for details

## Quick Start

### With llama.cpp (default)
```bash
llamacpp-superserver install
llamacpp-superserver run -hf meta-llama/Llama-2-7b-hf:Q4_K_M
```

### With vLLM (beta)
```bash
# Option 1: Docker (recommended for testing)
./test-vllm.sh

# Option 2: System install
llamacpp-superserver install  # Select "vllm-beta" when prompted
llamacpp-superserver run -hf meta-llama/Llama-2-7b-hf
```

## Install For Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Run

```bash
llamacpp-superserver --help
llamacpp-stack-install --help
```

The standalone bundle can be run directly:

```bash
./llamacpp_stack/bundle/install_llamacpp_stack.sh --dry-run
```

Notes:

- The bundle bootstrap environment is created with `uv` and installs Python-side tooling there, including `cmake`, `ninja`, and `compiletools`.
- If you choose a source build of `llama.cpp`, a real native C/C++ compiler toolchain is still required on the host. Python packages do not replace `gcc` / `g++`.

## Test

```bash
python -m pytest -q
```
