# llamacpp-superserver

Manager for a local llama.cpp / llama-swap stack.

It provides:

- `llamacpp-superserver`, a Python control CLI and API shim.
- install/uninstall entry points for the full stack.
- a bundled shell installer in `llamacpp_stack/bundle/`.
- helpers for Hugging Face GGUF downloads, model cataloguing, runtime validation, llama-swap config rendering, and Ollama-compatible endpoints.

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

## Test

```bash
python -m pytest -q
```
