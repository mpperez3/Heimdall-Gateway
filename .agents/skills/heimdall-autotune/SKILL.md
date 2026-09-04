---
name: heimdall-autotune
description: Auto-tune Heimdall Gateway models for max tok/s with minimal loss. Triggers when user adds a new GGUF or asks to optimize. Loop: probe VRAM, derive ctx/parallel/tensor_split/batch, validate 2x50K (or Nx) parallel, save snapshot to configs/history.
---

# Heimdall Autotune

Local-only loop to configure and optimize `llama.cpp`/`beellama` models via Heimdall Gateway. Maximizes `tok/s` with minimal quality loss, supports `parallel 1..N`.

## When to Use

- User runs `heimdall-gateway add <hf>` or asks to download a new model
- User asks to optimize, get more performance, or tune `ctx`/`parallel`/`tensor_split`/`batch`
- After `auto_ctx_failed` or OOM, or when `ps` shows `ERROR`

## When NOT to Use

- User only wants to chat or run a one-off inference without tuning
- Model is already validated and `ps` shows `ready` with desired `ctx`

## Core Workflow

### 1. Detect & Snapshot
```bash
heimdall-gateway ps
nvidia-smi --query-gpu=memory.total,memory.free --format=csv
cat ~/.local/state/heimdall-gateway/catalog.json | jq '.[] | {model_id,ctx_size,tensor_split,server_overrides}'
```

### 2. Derive Target (heuristic, local only)
- `weights = model_size GiB` from `ps SIZE` or `model-info`
- `kv_per_token = f(layers, cache_type)`; `kvarn4/q4_0 ~0.5B`, `f16 2B`
- `total_kv = parallel * ctx_size * kv_per_token`
- `weights + total_kv + batch_overhead < 0.9 * total_vram`
- Start with `ctx = 262144/parallel`, `tensor_split` balanced (`1,1` or `0.65,0.35` for 1.3:0.7 ratio), `batch 4096 ubatch 1024 checkpoints 16 flash_attn on`
- For `27B Q4_1` on `2x24GB`: safe `131K*2`, max `204K*2` (`409600` cfg -> `204800` per slot with half_context)

### 3. Apply & Sync
```bash
# edit catalog.json directly or via update
python3 -c "import json,pathlib; p=pathlib.Path('~/.local/state/heimdall-gateway/catalog.json').expanduser(); ..."
python3 /tmp/run_sync.py  # sync_config_from_server_config_for_startup
heimdall-gateway unload <model> && sleep 3 && curl .../v1/chat/completions --max_tokens 5 # warm
# verify slots
curl -sk http://127.0.0.1:11436/upstream/<model>/slots | jq '[.[].n_ctx]'
```

### 4. Validate Loop (evals)
- If `parallel=2`: 8 queries in pairs of 2 with 50K context (`~60058 prompt_tokens`), expect `8/8 200` `wall ~ sum/2` `slots 262144`
- If `parallel=3`: 9 queries in triples of 3
- If `parallel=1`: 4 queries sequential, no `429`
- Check `429 overloaded` vs `503 model_loading`: `429` = limit `HEIMDALL_GATEWAY_MAX_CONCURRENT_PER_MODEL`, `503` = `claim_loading` (engine not detected if beellama)
- Generic engine fix: `cli.py:7609` `if "llama-server" in Path(arg).name` for any `*llama-server*` (beellama, future)

### 5. Measure tok/s & Iterate
- Prompt `50K` -> `prompt_tokens / elapsed` (`~8s` warm, `~68s` cold)
- If `VRAM free >8GB` increase `batch 8192->12288` `ubatch 2048->4096` `checkpoints 32`
- If `GPU1` overloaded, adjust `tensor_split` (`1.1,0.9` -> `0.65,0.35` normalized)

### 6. Snapshot
```bash
mkdir -p configs/history
# save catalog entry + config.yaml + ps
# file: YYYY-MM-DD_HHMM_<model>_ctx<ctx>_split<split>_parallel<N>.json
```

## References

- Base: `jr2804/llama-cpp-optimizer` (retrieved 2026-09-02, MIT) - generic llama.cpp optimizer; adapted for Heimdall Gateway `catalog.json` loop and `parallel`/`tensor_split`/`claim_loading` fixes.
- Local: `llamacpp_stack/cli.py:7609` generic engine detection, `cli.py:4281 claim_loading`, `cli.py:18123 auto_update_watch`, `bundle/llama_server_defaults.yaml:37 parallel`

## Safety

- Local only, no external APIs. Prompts are synthetic lorem, not user data.
- Never commit `/var/llamacpp_models` or `/.history/`.

