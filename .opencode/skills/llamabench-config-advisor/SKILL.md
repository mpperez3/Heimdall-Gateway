---
name: llamabench-config-advisor
description: Propose optimal llama-server model configurations by checking https://llamabench.ai/browse for the top 2 benchmarks and cross-referencing https://github.com/ggml-org/llama.cpp/blob/master/docs for flag definitions, then translating into heimdall-gateway flags and logging repair history. Use this skill whenever the user asks for optimal configuration, best quant, batch-size, ctx-size, cache-type, parallel, reasoning budget, or tuning for speed/VRAM/quality, even if they don't mention llamabench explicitly.
---

# Llamabench Config Advisor

Advise the best `llama-server` / Heimdall Gateway configuration for a given model by grounding recommendations in https://llamabench.ai/browse.

## When to use

Trigger when the user requests optimal configuration, tuning, or comparison for a model (e.g. "config óptima para qwen3.8-27b", "mejor quant para 2x24GB", "optimiza batch-size/ctx", "qué cache-type es más rápido"). Do not trigger for pure ops (`list`, `ps`, `logs`, `info`) unless tuning is requested.

## Workflow

1. Clarify the user's constraints: model family/id, quant, hardware (GPUs/VRAM), context needs (tool-calling, RAG, vision, ctx length), and objective (throughput vs latency vs quality vs VRAM).
2. Fetch `https://llamabench.ai/browse` via `webfetch` (format markdown). If fetch fails, explain the failure and fall back to cached knowledge but flag it as stale.
3. Cross-reference the latest flag definitions via `webfetch` on `https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md` (and `docs/` via `https://github.com/ggml-org/llama.cpp/tree/master/docs`) for `cache-type-k`, `flash-attn`, `speculative`, `ctx-size`, `parallel`. Use the docs to explain what each flag does and to validate that proposed flags exist in this `llama.cpp` build.
4. Filter llamabench rows to the closest family/quant/HW match. Rank by relevant metric (tokens/s for throughput, p50 latency for latency, VRAM for fit). Select the **top 2** distinct configurations.
5. Do not copy raw configs verbatim. Translate the useful signals into this repo's flags: `ctx-size`, `cache-type-k/v`, `batch-size`, `ubatch-size`, `parallel`, `tensor-split`, `flash-attn`, `jinja`, `reasoning`, `reasoning-budget`, `predict`, etc. Validate flags against `llamacpp_stack/cli.py` defaults and `heimdall-gateway config-keys`.
6. Propose both options as A/B: table with `option | llamabench row (link) | tps/latency/VRAM | suggested server_overrides JSON` + short trade-off note and a per-flag doc citation (one line per non-obvious flag from the llama.cpp docs). Recommend one based on the user's objective but leave choice to the user.
7. Provide a ready-to-apply patch: `catalog.json` `server_overrides` snippet or `conf.json` `llama_server_defaults` / `llama_server_family_defaults` fragment, plus the `heimdall-gateway update` + restart commands.
8. Log the proposal and its repair history for debugging. Append a JSONL line to `~/.local/state/heimdall-gateway/llamabench-advisor-history.jsonl` (create dir if needed) with `{ts, model, server_overrides, llamabench_rows, doc_refs, repair_counts}` where `repair_counts` is derived from `grep -c "openai_chat.*repair\|model_request_blocked_overloaded" ~/.local/state/heimdall-gateway/api-requests.log` for that model in the last 7d. Also cite that file path in the output so the user can `cat` it.
9. Always cite the llamabench rows (URL, metric, date of fetch) and the doc URLs so the user can verify.

## Output template

Use this exact structure:

### Llamabench ground truth (fetched YYYY-MM-DD)
| # | Model / Quant | HW | ctx | tps / latency | VRAM | Source |
|---|---------------|----|-----|---------------|------|--------|

### Option A — [label, e.g. throughput]
`server_overrides` JSON + flags + why it helps.

### Option B — [label, e.g. balanced/efficiency]
`server_overrides` JSON + flags + why it helps.

### Recommendation
One paragraph: which option fits the user's constraints and how to apply it (`heimdall-gateway update ...`).

### Docs & repair log
- Per-flag doc citations (e.g. `cache-type-k: https://github.com/ggml-org/llama.cpp/blob/master/docs/...#cache-type` — one line).
- History: `~/.local/state/heimdall-gateway/llamabench-advisor-history.jsonl` — last 3 lines `tail` plus current repair counts for this model.

## Edge handling

- If no exact HW match, pick nearest VRAM tier and note the gap.
- If llamabench has no entry for the exact model, use the closest family member (e.g. qwen3.5 → qwen3.8) and explain.
- Never invent benchmark numbers; if a metric is missing, say "not reported".
