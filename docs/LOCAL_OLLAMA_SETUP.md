# Local Ollama Setup for Testing

This document describes how to set up and use Ollama locally for testing the llamacpp-stack project.

## Overview

Ollama provides a lightweight local LLM runtime that's useful for:
- Fast local testing without Docker overhead
- Development and debugging workflows
- Quick iteration on integration logic
- Testing with low-resource requirements (TinyStories models)

## Installation

### macOS
```bash
brew install ollama
```

### Linux (Ubuntu/Debian)
```bash
# Download and run installer
curl -fsSL https://ollama.ai/install.sh | sh

# Or install via package managers:
# Ubuntu/Debian:
curl -fsSL https://repos.ollama.ai/get/linux.sh | sh
```

### Windows
Download from [https://ollama.ai](https://ollama.ai)

## Model Storage

By default, Ollama stores models in:
- **macOS**: `~/.ollama/models`
- **Linux**: `~/.ollama/models`
- **Windows**: `%USERPROFILE%\.ollama\models`

You can customize this with the `OLLAMA_MODELS` environment variable:
```bash
export OLLAMA_MODELS=~/.ollama/models  # or custom path
```

**Note**: For this project, ensure `.ollama/` and `/models/` directories are in `.gitignore` to prevent committing large model files.

## Getting Started

### 1. Start the Ollama Service

```bash
# macOS
ollama serve

# Linux (if installed as service)
sudo systemctl start ollama
# Or run directly:
ollama serve
```

The service will be available at `http://localhost:11434`

### 2. Pull a Model for Testing

For quick local testing, pull the lightweight TinyStories model:

```bash
ollama pull tinystories  # ~27MB lightweight model for testing
```

Available test models:
- `tinystories` - Lightweight, fast (27MB)
- `phi` - Phi model (smaller version)
- `mistral` - Medium size, good quality
- `neural-chat` - Chat optimized

### 3. Verify the Model is Running

```bash
ollama ls  # List downloaded models
```

Test generation:
```bash
ollama run tinystories "Write a short story about a cat"
```

## Testing with llamacpp-stack

### Using Ollama with Your Code

Update your test configuration to connect to the local Ollama instance:

```python
# test_config.yaml or in-code configuration
api_config:
  api_type: "ollama"
  api_base: "http://localhost:11434"
  model: "tinystories"
  timeout_seconds: 30
```

### Python Client Example

```python
import requests

# Test Ollama endpoint
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "tinystories",
        "prompt": "Once upon a time",
        "stream": False
    }
)

print(response.json())
```

## Performance Notes

### Resource Usage
- **TinyStories**: ~100MB RAM, sub-100ms latency
- **Phi**: ~2GB RAM, 100-500ms latency  
- **Mistral**: ~4GB RAM, 200-1000ms latency

### Optimization Tips

1. **Reduce model context** for faster testing:
   ```bash
   # Adjust num_ctx parameter
   ollama run tinystories --num_ctx=512 "Your prompt"
   ```

2. **Enable GPU acceleration** (if available):
   - Ollama auto-detects CUDA/Metal support
   - Check with: `ollama --version` and look for cuda/metal indicators

3. **Run headless** (no GUI):
   ```bash
   export OLLAMA_DISABLE_TLS=true
   ollama serve
   ```

## Integration with Tests

### CI/CD Considerations

For CI/CD pipelines:
1. **Don't run Ollama in tests** - use mocks instead
2. **For local development tests**, ensure Ollama is already running
3. **Use docker approach** for consistent testing environments

Example test fixture:
```python
import pytest
import requests

@pytest.fixture
def ollama_available():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except:
        return False

def test_with_ollama(ollama_available):
    if not ollama_available:
        pytest.skip("Ollama service not running")
    # Your test here
```

## Troubleshooting

### Service won't start
```bash
# Check if port 11434 is in use
lsof -i :11434

# Try explicit binding
OLLAMA_HOST=127.0.0.1:11434 ollama serve
```

### Models not downloading
```bash
# Check storage space and permissions
df -h ~/.ollama/models
chmod 755 ~/.ollama/models

# Try pulling again with verbose output
OLLAMA_DEBUG=1 ollama pull tinystories
```

### High latency
- Check available RAM: `free -h` (Linux) or `top` (macOS)
- Try smaller model (tinystories vs mistral)
- Enable GPU if available
- Reduce `num_ctx` parameter

## Cleanup

To remove models and free space:

```bash
ollama rm tinystories
ollama rm mistral
# Remove all models:
rm -rf ~/.ollama/models
```

To uninstall Ollama:
```bash
# macOS
brew uninstall ollama

# Linux
sudo apt remove ollama

# Then remove models directory if needed:
rm -rf ~/.ollama
```

## References

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Available Models](https://ollama.ai/library)

## Quick Reference

```bash
# Install
brew install ollama

# Start service
ollama serve

# Download model (in another terminal)
ollama pull tinystories

# List models
ollama ls

# Run interactively
ollama run tinystories

# Check health
curl http://localhost:11434/api/tags
```
