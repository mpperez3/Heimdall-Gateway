# vLLM Beta Integration
vLLM is installed alongside llama.cpp and selected automatically for native HuggingFace repositories.

## Overview

vLLM is an open-source LLM inference engine optimized for throughput and latency. This beta integration allows you to use vLLM as a drop-in replacement for llama.cpp within the llamacpp-stack while maintaining full compatibility with llama-swap.

### Key Differences vs llama.cpp

| Feature | llama.cpp | vLLM |
|---------|-----------|------|
| Model Format | GGUF (quantized) | HuggingFace/GGUF |
| Inference Speed | High (optimized C++) | Very High (batched) |
| Quantization | Native support | Requires conversion |
| Memory Efficiency | Excellent (int4/fp16) | Good (with quantization) |
| API | OpenAI-compatible | OpenAI-compatible |
| CPU Support | Excellent | GPU-optimized |

## Installation

### Option 1: Docker Compose (Recommended for Quick Testing)

```bash
# Build vLLM image with CUDA support
docker-compose -f docker-compose-vllm.yaml build

# Run with a specific model
docker-compose -f docker-compose-vllm.yaml up -d vllm

# Check logs
docker-compose -f docker-compose-vllm.yaml logs -f vllm
```

### Option 2: System-wide Installation

The installer provisions both engines. Use automatic routing:

```bash
heimdall-gateway install --backend auto
```

The installer will:
1. Create the runtime Python environment with `uv`
2. Install vLLM and its torch backend with `uv pip install --torch-backend=auto`
3. Install vLLM in the managed runtime alongside llama.cpp
4. Route GGUF models to llama.cpp and native HuggingFace models to vLLM

### Option 3: Manual Setup

```bash
# Install vLLM with CUDA support
pip install vllm torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Test vLLM directly
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --gpu-memory-utilization 0.9 \
  --dtype float16
```

## Usage with llamacpp-stack

### Running a Model with vLLM

```bash
# A repository without GGUF is routed to vLLM automatically
heimdall-gateway run -hf meta-llama/Llama-2-7b-hf

# Or with custom parameters
heimdall-gateway run \
  -hf meta-llama/Llama-2-7b-hf \
  --n-gpu-layers 40 \
  --ctx-size 4096
```

### Configuration

vLLM backend ignores llama.cpp-specific flags:
- `--fit`, `--fitc`, `--fitt` (not applicable)
- `--mirostat` (vLLM uses sampling)
- Draft models (speculative decoding differs)

The following flags are supported:
- `--ctx-size N` → maps to `--max-model-len N`
- `--dtype float16|bfloat16|float32` → inference precision
- `--n-gpu-layers N` → ignored (vLLM uses all GPU layers by default)

### Metrics and per-request spec-decode

vLLM always exposes `usage.prompt_tokens` and `usage.completion_tokens`
natively. Speculative draft metrics (`draft_n`, `draft_n_accepted`,
`metrics` field) are available only when the model was started with
speculative decoding **and** the flag `--per-request-spec-decode-metrics`
is active.

Heimdall configures this flag via `vllm.defaults.per_request_spec_decode_metrics`
in `llamacpp_stack/bundle/llama_server_defaults.yaml` (default `summary`).
The flag is gated: the gateway runs `vllm-server --help | grep
per-request-spec-decode-metrics` before emitting it. If the installed vLLM
version does not support the flag, it is omitted and the server still starts.
Supported values are `detailed`, `summary`, `none` (invalid falls back to
`summary`). Without speculative config, `Drafted` correctly shows `-` in
`Activity` even with the flag present.

Like EXL3, vLLM **never emits `slot print_timing`**. Timing for vLLM lives
only in the response `usage` (`prompt_tokens`, `completion_tokens`) and, when
speculative is active, in the `metrics`/`draft_n` field. The Activity view
at `:11436/api/metrics/activity` shows `Drafted` and `Cached` as `-` when no
real data exists. Do not expect `slot print_timing` in the journal for vLLM.

Copyable verification:

```console
$ heimdall-gateway config-keys --format json | jq '.vllm.defaults // .vllm'
$ curl -s http://127.0.0.1:11435/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"MODEL_ID","messages":[{"role":"user","content":"hi"}],"stream":false}' | jq '.usage, .timings // .metrics'
$ curl -s http://127.0.0.1:11436/api/metrics/activity | jq '.data[] | {model: .model, tokens: .tokens}'
$ heimdall-gateway logs --lines 200 --journal | grep timing
# vLLM -> no slot print_timing expected; check usage/metrics above
```

After changing the flag, apply the standard runbook:

```console
$ heimdall-gateway config-migrate && heimdall-gateway update
$ systemctl --user restart heimdall-gateway-manager heimdall-gateway-router
```

### Docker Deployment

For production with llama-swap integration:

```bash
# Build both services
docker-compose -f docker-compose-vllm.yaml build

# Start vLLM and configure llama-swap to use it
docker-compose -f docker-compose-vllm.yaml up -d

# Test the API
curl http://localhost:8000/v1/models
```

## Known Limitations & Workarounds

### 1. Model Format
- **Issue**: vLLM works best with HuggingFace format; GGUF support requires AWQ conversion
- **Workaround**: Use `.safetensors` or `.bin` models from HuggingFace Hub
- **Example**: `meta-llama/Llama-2-7b-hf` works; `TheBloke/Llama-2-7b-GGUF:Q4_K_M` needs conversion

### 2. Memory Management
- **Issue**: vLLM pre-allocates GPU memory; OOM possible with large single-batch requests
- **Workaround**: Reduce `--gpu-memory-utilization` from 0.9 to 0.7-0.8

### 3. Quantization
- **Issue**: vLLM's quantization is different from llama.cpp's
- **Workaround**: Use fp16 or bf16 for now; AWQ support coming in future versions

### 4. Context Window
- **Issue**: Large context windows (`--ctx-size`) may exceed GPU memory
- **Workaround**: Query actual max length via `/v1/models` endpoint:
  ```bash
  curl http://localhost:8000/v1/models | jq '.data[0].max_model_len'
  ```

## Testing Checklist

When testing vLLM with your models:

- [ ] Model loads without OOM
- [ ] API responds to `/v1/models` endpoint
- [ ] Chat completions work via `/v1/chat/completions`
- [ ] llama-swap can discover and use the model
- [ ] Multiple concurrent requests handled correctly
- [ ] Model unloads cleanly on shutdown

## Switching Back to llama.cpp

If you need to revert to llama.cpp:

```bash
# Reinstall with llama.cpp backend
heimdall-gateway install --backend llama.cpp --update-binaries

# Or manually specify
heimdall-gateway install --backend llama.cpp --llama-cpp-mode prebuilt
```

## Environment Variables

For fine-tuning vLLM behavior:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn      # Default multiprocessing method
export VLLM_ATTENTION_BACKEND=flash_attn       # Flash attention optimization
export CUDA_VISIBLE_DEVICES=0,1                # Specify GPUs
export VLLM_GPU_MEMORY_UTILIZATION=0.9         # GPU memory usage percentage
```

## Performance Notes

### Expected Performance (on A100 GPU)
- **Throughput**: 500-2000 tokens/sec (depending on model size)
- **Latency**: 5-50ms per token (batch of 1)
- **Memory**: ~8GB for 7B model in fp16, ~24GB for 70B

### Optimization Tips
1. Use `float16` or `bfloat16` instead of `float32`
2. Batch requests when possible
3. Adjust `gpu-memory-utilization` based on your hardware
4. Enable flash attention (`VLLM_ATTENTION_BACKEND=flash_attn`)

## Troubleshooting

### vLLM crashes on start
- Check CUDA is available: `nvidia-smi`
- Verify GPU memory: `nvidia-smi --query-gpu=memory.free --format=csv`
- Check vLLM logs: `docker logs vllm-api-server`

### Out of Memory (OOM)
```bash
# Reduce memory utilization
export VLLM_GPU_MEMORY_UTILIZATION=0.7

# Or use smaller model
heimdall-gateway run -hf meta-llama/Llama-2-7b-chat-hf
```

### Model not found
- Verify HuggingFace Hub connectivity
- Check HuggingFace token if using gated models: `huggingface-cli login`
- Try downloading model manually: `huggingface-cli download meta-llama/Llama-2-7b-hf`

### API returns 503 (Service Unavailable)
- Model may still be loading; wait 30-60 seconds
- Check `/v1/models` endpoint for status
- Review vLLM logs: `docker logs -f vllm-api-server`

### No slot print_timing for vLLM in journal
- This is expected. Only `llama-server` emits `slot print_timing`. For vLLM
  check `curl :11435/v1/chat/completions | jq '.usage'` and
  `curl :11436/api/metrics/activity`. `Drafted` appears only with speculative
  decoding and the gated flag `per_request_spec_decode_metrics`. `Cached` is
  `-` unless prefix caching is enabled.

## Reporting Issues

Found a bug or have a feature request?

1. Test with latest vLLM: `pip install --upgrade vllm`
2. Collect logs: `docker-compose logs vllm > vllm.log`
3. Report on GitHub with:
   - Model name
   - Command executed
   - Full error/log output
   - GPU info (`nvidia-smi`)

## Future Work

Planned improvements for vLLM beta:

- [ ] Native GGUF support (via llama2-compatible API)
- [ ] Automatic quantization conversion
- [ ] Memory pooling optimization
- [ ] Multi-GPU distribution
- [ ] Batch request queuing
- [ ] Native speculative decoding support

## References

- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI API Compatibility](https://platform.openai.com/docs/api-reference)
- [llamacpp-stack GitHub](https://github.com/mostlygeek/llamacpp-stack)
