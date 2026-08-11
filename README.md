# heimdall-gateway

Manager for a local llama.cpp / llama-swap stack with automatic llama.cpp/vLLM routing.

It provides:

- `heimdall-gateway`, a Python control CLI and API shim.
- install/uninstall entry points for the full stack.
- a bundled shell installer in `llamacpp_stack/bundle/`.
- helpers for Hugging Face GGUF downloads, model cataloguing, runtime validation, llama-swap config rendering, and Ollama-compatible endpoints.
- vLLM integration for native HuggingFace models, installed alongside llama.cpp.

## Backends

### llama.cpp (Stable)
- Optimized for GGUF quantized models
- Excellent CPU and GPU support
- Native Flash Attention

### vLLM
- OpenAI API compatible
- Optimized for HuggingFace format models
- High-throughput inference with batching
- See [docs/VLLM-BETA.md](docs/VLLM-BETA.md) for details

## Quick Start

### With llama.cpp (default)
```bash
heimdall-gateway install
heimdall-gateway run -hf meta-llama/Llama-2-7b-hf:Q4_K_M
```

### With vLLM
```bash
# Option 1: Docker (recommended for testing)
./test-vllm.sh

# Option 2: System install (installs both engines)
heimdall-gateway install --backend auto
heimdall-gateway run -hf meta-llama/Llama-2-7b-hf
```

### With Ollama (Local Testing)
For quick local testing without Docker overhead, use Ollama with lightweight models:

```bash
# Install Ollama (macOS: brew install ollama, Linux: see docs)
ollama serve  # Start in one terminal

# In another terminal, pull a test model:
ollama pull tinystories  # ~27MB lightweight model

# Use in your code or tests (service runs on http://localhost:11434)
# See docs/LOCAL_OLLAMA_SETUP.md for full integration guide
```

**When to use Ollama:**
- ✅ Local development and iteration
- ✅ Fast testing cycles (TinyStories: ~100ms latency)
- ✅ Low-resource environments
- ❌ Production deployments (use heimdall-gateway instead)

See [docs/LOCAL_OLLAMA_SETUP.md](docs/LOCAL_OLLAMA_SETUP.md) for detailed setup instructions.

## Install For Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Run

```bash
heimdall-gateway --help
llamacpp-stack-install --help
```

The standalone bundle can be run directly:

```bash
./llamacpp_stack/bundle/install_llamacpp_stack.sh --dry-run
```

Note: The interactive installer now asks whether to run the full installer by default (`Y`), and answering `n`/`no` will perform a package-only update of `heimdall-gateway` (leaving binaries, config, and auto-ctx untouched).

Installer-managed llama-server defaults are written using the project's kebab-case config keys, matching the runtime flags used by the local llamacpp stack.

Notes:

- The bundle bootstrap environment is created with `uv` and installs Python-side tooling there, including `cmake`, `ninja`, and `compiletools`.
- If you choose a source build of `llama.cpp`, a real native C/C++ compiler toolchain is still required on the host. Python packages do not replace `gcc` / `g++`.

## Test

```bash
python -m pytest -q
```

## Development Docker: prueba rápida con llama.cpp

Para desarrollar y probar funcionalidades sobre `llama.cpp` (por ejemplo `refresh-templates`) sin reinstalar o compilar todo en el host, hay un Dockerfile preparado llamado `Dockerfile.llamacpp` que construye una imagen con `llama-server` listo para pruebas.

- Construir la imagen (en el directorio raíz del proyecto):

```bash
docker build -t llamacpp-templates-test -f Dockerfile.llamacpp .
```

- Ejecutar un contenedor montando el workspace para probar comandos de la CLI y los archivos de ejemplo `test_*` y `templates/`:

```bash
docker run --rm -it \
	-v "$PWD":/workspace \
	-w /workspace \
	llamacpp-templates-test \
	/bin/bash
```

Dentro del contenedor puedes ejecutar el comando de refresco usando las rutas montadas. Ejemplo:

```bash
python -m llamacpp_stack.cli refresh-templates \
	--catalog /workspace/test_catalog.json \
	--server-config /workspace/test_server.json \
	--config /workspace/test_config.yaml \
	--llama-server /usr/local/bin/llama-server
```

Notas y buenas prácticas:
- La imagen compila `llama.cpp` en el momento de construirla; una vez construida sirve para iterar rápidamente sin recompilar.
- Si vas a desarrollar mucho tiempo, considera crear una imagen base y reutilizarla para acelerar iteraciones.
- Asegúrate de que los archivos en `templates/` sean accesibles por el usuario que corre el servicio en producción (permisos/propiedad). Esta imagen sirve sólo para pruebas de desarrollo.
