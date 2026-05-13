import hashlib
import json
import os
import re
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import signal
import socket
import warnings

try:
    import requests
except ImportError:
    requests = None

try:
    from optuna.exceptions import ExperimentalWarning as OptunaExperimentalWarning
except Exception:
    OptunaExperimentalWarning = None

from llamacpp_stack.cli import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_LLAMA_SERVER,
    ManagedModel,
    build_llama_server_command,
    detect_cuda_device_count,
    load_catalog_with_diagnostics,
    normalize_server_overrides,
    render_llamaswap_config,
    resolve_catalog_model,
    resolve_idle_ttl,
    resolve_llama_server_defaults,
    restart_service_to_free_vram,
    save_catalog,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTO_PERF_PROFILES_PATH = PROJECT_ROOT / "json" / "auto_performance_profiles.json"
# Use /tmp for auto-performance outputs to avoid cluttering system directories
AUTO_PERF_LOG_DIR = Path(tempfile.gettempdir()) / "llamacpp-auto-perf"
AUTO_PERF_HISTORY_PATH = AUTO_PERF_LOG_DIR / "auto_performance_best_history.jsonl"
AUTO_PERF_CATALOG_KEY = "auto_performance"
AUTO_PERF_SCORE_SCHEMA_VERSION = "total_tokens_s_v1"
AUTO_PERF_CATALOG_MAX_BASELINES_PER_PHASE = 6
BASELINE_CANDIDATE_CTX = [8192, 16384, 32768, 65536]
CACHE_TYPE_FLOOR = "q8_0"
CACHE_TYPE_CANDIDATES = ("q8_0",)

# Auto-performance is a speed-preserving tuner: it should keep the user's
# context and only increase it when a higher value is clearly required.
CTX_SIZE_POLICY = "never_decrease"
LONG_CONFIRM_PROMPT_TOKENS = 20_000
LONG_CONFIRM_PREDICT_TOKENS = 20_000
RAW_SCREENING_N_PREDICT = 512
RAW_SCREENING_RUNS = 2
SERVER_BENCHMARK_N_PREDICT = 512
AUTO_PERF_RAW_BENCHMARK_SCHEMA_VERSION = f"raw_chat_screen{RAW_SCREENING_N_PREDICT}x{RAW_SCREENING_RUNS}_long20k20k_v4"
AUTO_PERF_SERVER_BENCHMARK_SCHEMA_VERSION = f"server_chat_n{SERVER_BENCHMARK_N_PREDICT}_v2"

# ============================================================================
# AUTO-PERFORMANCE TUNING STRATEGY (3-PHASE ROADMAP)
# ============================================================================
# PHASE 1 (CURRENT):
#   Raw llama.cpp inference throughput optimization
#   - GPU distribution: gpu_mask, split_mode, tensor_split, main_gpu
#   - Compute efficiency: n_gpu_layers, fit, batch_size, ubatch_size
#   - Memory optimization: cache_type_k/v, flash_attn, kv_offload, cache_ram
#   - CPU overhead: threads, threads_batch, threads_http, numa
#   - Load I/O: direct_io, op_offload
#   Stage 0 load probe: direct_io is selected before the main search loop
#   Metric: tokens/s (prefill + decode weighted), no network latency
# PHASE 2 (SPECULATIVE, WHEN MODEL IS SPECULATIVE):
#   Draft-side improvement phase that keeps quality constraints intact
#   - model_draft, draft, ctx_size_draft, n_gpu_layers_draft
#   - cache_type_k_draft, cache_type_v_draft
#   Metric: speculative draft efficiency while preserving output quality
#
# PHASE 3 (FUTURE):
#   llama-server API-level optimization (requires dynamic API testing)
#   - Dynamic request scheduling (batching strategy, request ordering)
#   - Model serving specifics (slot scheduling, cache eviction policy)
#   - Server concurrency: parallel, cont_batching, ctx_checkpoints
#   - Load balancing across instances (not per-model optimization)
#   Metric: endpoint p50/p95 latency, throughput under concurrent load
#   Status: Deferred - requires HTTP request simulation, out of scope for raw model tuning
#
# OUT OF SCOPE (QUALITY/GENERATION, NOT PERFORMANCE):
#   Sampling parameters that do NOT affect throughput but control output diversity
#   - temperature, top_p, top_k, min_p, frequency_penalty, repeat_penalty
#   - presence_penalty, min_keep, tfs_z, typical_p, eta_cutoff, epsilon_cutoff
#   Rationale: These only control token selection logic, not compute/memory efficiency.
#   Applied at generation time post-inference, not affecting tokens/s measurement.
#
# ============================================================================

# The tuner only probes the fast path knobs that can realistically change
# throughput without obviously harming latency or stability.
# Everything else is intentionally pruned from the search surface.
# Categorized below by reason for exclusion:
#
# ROPE/YARN (context scaling, not throughput):
#   rope_scaling, rope_scale, rope_freq_base, rope_freq_scale (alter effective context)
#   yarn_* (alternative RoPE implementation, affects quality/ctx not tokens/s)
#
# UNSAFE FOR AUTOMATIC MODE:
#   override_kv, override_tensor, check_tensors (model-specific or dangerous)
#
# MODEL/TRAINING SPECIFIC:
#   lora, control_vector (application-specific, not runtime perf)
#
# I/O & CACHING CONTROL (special handling):
#   mmap, mlock, repack (I/O strategy, not compute throughput)
#   cache_prompt, cache_reuse, ctx_size (handled separately or policy-constrained)
#
# SPECULATIVE DECODING (SEPARATE PHASE WHEN MODEL IS SPECULATIVE):
#   model_draft, draft, ctx_size_draft, n_gpu_layers_draft,
#   cache_type_k_draft, cache_type_v_draft
#   These are the speculative knobs that the tuner can search when the
#   catalog entry is marked speculative and a draft model is already present.
#   Compatibility-only keys such as draft_min / draft_p_min are accepted by
#   the CLI builder, but they are not part of the autotuning search surface.
#
# SAMPLING (PHASE N/A - quality not throughput):
#   All sampling params (temperature, top_p, top_k, min_p, etc.) — affects quality/diversity
#   only, not tokens/s. Applied post-inference at token selection stage.
#
PRUNED_TUNER_KEYS = {
    "mmap",
    "mlock",
    "repack",
    "rope_scaling",
    "rope_scale",
    "rope_freq_base",
    "rope_freq_scale",
    "yarn_orig_ctx",
    "yarn_ext_factor",
    "yarn_attn_factor",
    "yarn_beta_slow",
    "yarn_beta_fast",
    "override_kv",
    "override_tensor",
    "check_tensors",
    "lora",
    "control_vector",
    "cache_prompt",
    "cache_reuse",
    "ctx_size",
}

# PHASE 1: Flags that are worth exploring for raw llama.cpp throughput.
# Organized by impact category:
#   GPU distribution: gpu_mask, split_mode, tensor_split, main_gpu, n_gpu_layers
#   Compute efficiency: fit, fit_target, batch_size, ubatch_size, flash_attn
#   Memory optimization: cache_type_k, cache_type_v, kv_offload, cache_ram
#   CPU threading: numa, threads, threads_batch, threads_http
#   I/O load: op_offload
#
# Note: `direct_io` is probed separately in Stage 0 so load selection can be
# made before the main throughput search. Server-level concurrency flags are
# reserved for PHASE 2, where the search objective is multi-client serving.
#
PROBED_TUNER_KEYS = {
    "split_mode",
    "tensor_split",
    "main_gpu",
    "n_gpu_layers",
    "fit",
    "fit_target",
    "batch_size",
    "ubatch_size",
    "flash_attn",
    "kv_offload",
    "numa",
    "threads",
    "threads_batch",
    "op_offload",
}

# Stage 0-only probing keys. These are intentionally excluded from the
# phase-1 Optuna search space.
STAGE0_PROBED_KEYS = {"direct_io"}

PHASE1_SPECULATIVE_TUNER_KEYS = {
    # Full set of speculative params accepted by validation/persistence.
    "model_draft",
    "draft",
    "ctx_size_draft",
    "n_gpu_layers_draft",
    "cache_type_k_draft",
    "cache_type_v_draft",
}

PHASE1_SPECULATIVE_SEARCH_KEYS = {
    # Deterministic SPECULATIVE phase knobs (not Optuna-sampled).
    "ctx_size_draft",
}

PHASE2_SERVER_TUNER_KEYS = {
    "parallel",
    "cont_batching",
    "ctx_checkpoints",
    "checkpoint_every_n_tokens",
    "cache_ram",
    "threads_http",
    "kv_unified",
    "cache_idle_slots",
}

PHASE2_SERVER_SEARCH_KEYS = {
    "parallel",
    "cont_batching",
    "ctx_checkpoints",
    "cache_ram",
    "threads_http",
    "kv_unified",
    "cache_idle_slots",
}

# VRAM components for breakdown and estimation
VRAM_COMPONENT_WEIGHTS = "weights"
VRAM_COMPONENT_KV = "kv"
VRAM_COMPONENT_COMPUTE = "compute"
VRAM_COMPONENT_OVERHEAD = "overhead"
VRAM_COMPONENT_DRAFT = "draft"
VRAM_COMPONENT_DRAFT_KV = "draft_kv"

VRAM_ZONES = {
    "green": 0.85,  # Safe
    "amber": 0.95,  # Risky, but might work with repair
    "red": 1.0,     # Likely OOM
}

# ============================================================================
# SAMPLING PARAMETERS (OUT OF SCOPE FOR THROUGHPUT TUNING)
# ============================================================================
# These parameters control token selection quality/diversity, not inference throughput.
# They do not affect tokens/s and are applied post-inference at generation stage.
# Future phases may add sampling sweeps for quality-then-speed tradeoff analysis.
#
SAMPLING_PARAMS_NOT_TUNED = {
    "temperature",      # Token log-probability scaling
    "top_p",           # Nucleus sampling threshold
    "top_k",           # Top-k filtering
    "min_p",           # Minimum probability threshold
    "frequency_penalty",  # Token frequency adjustment
    "repeat_penalty",   # Penalize recent tokens
    "presence_penalty", # Penalize any occurrence
    "min_keep",        # Minimum tokens to keep
    "tfs_z",           # Tail-free sampling parameter
    "typical_p",       # Typical probability sampling
    "eta_cutoff",      # Eta cutoff for etasampling
    "epsilon_cutoff",  # Epsilon cutoff for epsilon-sampling
}

# ============================================================================
# PHASE 2 SERVER-SPECIFIC PARAMETERS (DEFERRED, REQUIRES API-LEVEL TESTING)
# ============================================================================
# These parameters require dynamic HTTP-level testing and multi-client simulation.
# Not suitable for single-model raw performance tuning (current phase).
# Future implementation will sweep these against concurrent request patterns.
#
PHASE2_SERVER_SPECIFIC_PARAMS = {
    # Slot scheduling & request distribution
    "slots/strategy",           # How to assign incoming requests to KV cache slots
    "slots/batch_size",         # Max requests per batch
    "slots/timeout_policy",     # How to handle slot exhaustion
    
    # Cache eviction strategy
    "cache/eviction_policy",    # LRU, LLF, priority-based, etc.
    "cache/preemption_strategy", # Can interrupt long contexts for urgent short ones
    
    # Request scheduling
    "scheduler/algorithm",       # FIFO, priority queue, cost-aware, etc.
    "scheduler/timeout",         # Request timeout policy
    
    # Multi-instance load balancing (out of single-model scope)
    "load_balancer/algorithm",
    "load_balancer/weights",
}


def _ensure_optuna():
    try:
        import optuna

        return optuna
    except ImportError:
        print("Instalando optuna para auto-performance...")
        
        # Try multiple installation strategies in order of preference
        strategies = [
            # Strategy 1: Use sys.executable's pip module (standard venv)
            [sys.executable, "-m", "pip", "install", "optuna"],
            # Strategy 2: Use sys.executable with --break-system-packages for externally-managed venv
            [sys.executable, "-m", "pip", "install", "--break-system-packages", "optuna"],
            # Strategy 3: Try 'pip3' from system PATH
            ["pip3", "install", "optuna"],
            # Strategy 4: Try 'pip' from system PATH
            ["pip", "install", "optuna"],
        ]
        
        last_error = None
        for strategy in strategies:
            try:
                result = subprocess.run(strategy, check=True, capture_output=True, timeout=60, text=True)
                print(f"Successfully installed optuna using: {' '.join(strategy)}")
                import optuna
                return optuna
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
                last_error = e
                error_msg = ""
                if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                    error_msg = e.stderr
                # Continue to next strategy
                continue
        
        # If all strategies failed, raise a helpful error
        raise RuntimeError(
            f"Failed to install optuna across all strategies. Last error: {last_error}\n"
            "Please try one of the following:\n"
            "  1. Install system-wide:\n"
            "     sudo apt install python3-optuna\n"
            "  2. Use pipx to install the full package:\n"
            "     pipx install llamacpp-superserver\n"
            "  3. Manually install optuna in the venv:\n"
            "     /opt/llamacpp-superserver/venv/bin/pip install optuna"
        )


def _llama_cpp_version(server_bin: Path | str | None) -> str:
    candidate = str(server_bin or DEFAULT_LLAMA_SERVER)
    try:
        completed = subprocess.run([candidate, "--version"], capture_output=True, text=True, timeout=10)
        output = (completed.stdout or completed.stderr or "").strip()
        return output.splitlines()[0] if output else "unknown"
    except Exception:
        return "unknown"


def _hardware_fingerprint(target_ctx: int) -> dict:
    # Minimal fingerprint used only for profiling/testing.
    gpu_snapshot = _sample_gpu_memory_snapshot()
    gpu_total_mib = sum(g.get("total", 0) for g in gpu_snapshot.values())
    return {
        "fingerprint": f"{os.uname().sysname}-{os.uname().machine}-{target_ctx}-{gpu_total_mib}",
        "cpu": os.uname().machine,
        "gpu_count": len(gpu_snapshot),
        "gpu_total_mib": float(gpu_total_mib),
    }


def _get_gpu_set_indices(gpu_set_str: str) -> list[int]:
    """Parse comma-separated GPU indices."""
    try:
        return sorted([int(p.strip()) for p in gpu_set_str.split(",") if p.strip()])
    except Exception:
        return [0]


def _normalize_cache_type(value: object) -> str:
    return CACHE_TYPE_FLOOR


def _sample_gpu_memory_snapshot(indices: list[int] | None = None) -> dict[int, dict[str, int]]:
    if shutil.which("nvidia-smi") is None:
        return {}
    try:
        cmd = ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"]
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        result: dict[int, dict[str, int]] = {}
        wanted = set(indices or [])
        for line in (completed.stdout or "").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                continue
            try:
                gpu_index = int(parts[0])
                memory_used = int(float(parts[1]))
                memory_free = int(float(parts[2]))
                memory_total = int(float(parts[3]))
            except Exception:
                continue
            if wanted and gpu_index not in wanted:
                continue
            result[gpu_index] = {"used": memory_used, "free": memory_free, "total": memory_total}
        return result
    except Exception:
        return {}


def _sample_gpu_memory(indices: list[int] | None = None) -> dict:
    snapshot = _sample_gpu_memory_snapshot(indices)
    return {gpu_index: stats.get("used", 0) for gpu_index, stats in snapshot.items()}


def _wait_for_model_load(
    server_proc,
    gpu_indices: list[int] | None,
    *,
    health_url: str,
    log_path: Path | None,
    server_ready_timeout_s: float,
) -> tuple[bool, str, float, int, int]:
    """Wait for model load using health endpoint as the sole criterion.
    
    The server (llama.cpp/ollama) manages loading internally and signals
    readiness via /health. GPU memory tracking is unreliable due to other
    system processes, so we rely on the server's health signal.
    """
    start_wait = time.time()
    initial_gpu = _gpu_memory_total_used(gpu_indices)
    
    ready = False
    deadline = start_wait + server_ready_timeout_s
    last_progress_log = 0.0
    
    while time.time() < deadline:
        if server_proc.poll() is not None:
            reason = "server-crashed"
            _append_auto_perf_log(log_path, f"SERVER_EXIT reason={reason}")
            return False, reason, time.time() - start_wait, initial_gpu, _gpu_memory_total_used(gpu_indices)
        
        now = time.time()
        try:
            if requests is None:
                raise ImportError("requests not available")
            res = requests.get(health_url, timeout=2)
            if res.status_code == 200:
                ready = True
                final_gpu = _gpu_memory_total_used(gpu_indices)
                _append_auto_perf_log(log_path, f"HEALTH_READY elapsed={now - start_wait:.1f}s")
                return True, "health-ready", now - start_wait, initial_gpu, final_gpu
        except Exception:
            pass
        
        elapsed = now - start_wait
        if elapsed % 5 == 0 and elapsed > 0 and (now - last_progress_log) >= 5.0:
            print(f"      (waiting for server ready {elapsed:.0f}s...)")
            last_progress_log = now
        
        time.sleep(1)
    
    elapsed = time.time() - start_wait
    _append_auto_perf_log(log_path, f"SERVER_EXIT reason=health-timeout elapsed={elapsed:.1f}s limit={server_ready_timeout_s:.1f}s")
    return False, "health-timeout", elapsed, initial_gpu, _gpu_memory_total_used(gpu_indices)


def _gpu_memory_total_used(indices: list[int] | None = None) -> int:
    return sum(_sample_gpu_memory(indices).values())


def _gpu_memory_budget_mib(snapshot: dict[int, dict[str, int]], gpu_count: int | None = None) -> int | None:
    if not snapshot:
        return None
    target = len(snapshot) if gpu_count is None else max(1, int(gpu_count))
    selected = sorted(snapshot)[: min(target, len(snapshot))]
    if len(selected) < target:
        return None
    return sum(int(snapshot[gpu_index].get("free", 0)) for gpu_index in selected)


def _get_model_size_mib(model_path: str) -> float:
    try:
        return os.path.getsize(model_path) / (1024 * 1024)
    except Exception:
        return 0.0


def _estimate_vram_breakdown(
    model_path: str, 
    params: dict, 
    hw: dict, 
    base_vram_mib: float | None = None
) -> dict[str, float | str]:
    """Detailed VRAM estimation by components, including zone classification."""
    model_size = _get_model_size_mib(model_path)
    ctx = int(params.get("ctx_size", 8192))
    batch = int(params.get("batch_size", 2048))
    ubatch = int(params.get("ubatch_size", 512))
    parallel = int(params.get("parallel", 1))
    
    # KV cache estimation (very rough but guided)
    cache_k = _normalize_cache_type(params.get("cache_type_k", "q8_0"))
    cache_v = _normalize_cache_type(params.get("cache_type_v", "q8_0"))
    
    bytes_per_token = 0.5 # Default for q8_0
    if cache_k in {"f16", "bf16"}: bytes_per_token += 0.5
    if cache_v in {"f16", "bf16"}: bytes_per_token += 0.5
    
    kv_mib = (ctx * parallel * (model_size / 2048) * bytes_per_token) / 1024
    weights_mib = model_size * 1.05
    compute_mib = (ubatch * 0.5) + (batch * 0.1)
    if params.get("flash_attn") == "on":
        compute_mib *= 0.7
        
    overhead_mib = 500.0 + (len(params.get("gpu_set", [0])) * 200.0)
    
    draft_mib = 0.0
    draft_kv_mib = 0.0
    if params.get("model_draft"):
        draft_size = _get_model_size_mib(params["model_draft"])
        draft_mib = draft_size * 1.1
        draft_ctx = int(params.get("ctx_size_draft", 2048))
        draft_kv_mib = (draft_ctx * (draft_size / 1024) * 0.5) / 1024

    total = weights_mib + kv_mib + compute_mib + overhead_mib + draft_mib + draft_kv_mib
    
    vram_mib_list = hw.get("vram_mib", [8192])
    gpu_set = params.get("gpu_set", [0])
    budget = sum(vram_mib_list[i] for i in gpu_set if i < len(vram_mib_list))
    
    zone = "green"
    if budget > 0:
        ratio = total / budget
        if ratio > VRAM_ZONES["red"]:
            zone = "red"
        elif ratio > VRAM_ZONES["amber"]:
            zone = "amber"
        elif ratio > VRAM_ZONES["green"]:
            zone = "amber" # Map 0.85-0.95 to amber as per constants
        else:
            zone = "green"
            
    return {
        VRAM_COMPONENT_WEIGHTS: weights_mib,
        VRAM_COMPONENT_KV: kv_mib,
        VRAM_COMPONENT_COMPUTE: compute_mib,
        VRAM_COMPONENT_OVERHEAD: overhead_mib,
        VRAM_COMPONENT_DRAFT: draft_mib,
        VRAM_COMPONENT_DRAFT_KV: draft_kv_mib,
        "total": total,
        "zone": zone,
        "budget": budget
    }


def _estimate_trial_vram_mib(base_vram_mib: float, baseline_params: dict, trial_params: dict) -> float:
    """Heuristic VRAM estimation used for lightweight scaling checks and legacy tests."""
    ratio = 1.0
    
    # Context scaling
    ctx_t = trial_params.get("ctx_size", 2048)
    ctx_b = baseline_params.get("ctx_size", 2048)
    if ctx_b > 0:
        ratio *= (ctx_t / ctx_b)
    
    # GPU count scaling (rough overhead)
    gset_t = trial_params.get("gpu_set", [])
    gset_b = baseline_params.get("gpu_set", [])
    gpu_t = len(gset_t) if isinstance(gset_t, list) else 1
    gpu_b = len(gset_b) if isinstance(gset_b, list) else 1
    if gpu_t > gpu_b:
        ratio *= (1.05 ** (gpu_t - gpu_b))
        
    # Batch scaling
    batch_t = trial_params.get("batch_size", 512)
    batch_b = baseline_params.get("batch_size", 512)
    if batch_t > batch_b:
        ratio *= 1.1
        
    return base_vram_mib * ratio


def _with_gpu_set_for_minimum_check(params: dict, gpu_set: list[int], total_gpu_count: int) -> dict:
    cfg = dict(params)
    cfg["gpu_set"] = list(gpu_set)
    cfg["gpu_set_idx"] = 0
    cfg["tensor_split_strategy"] = "equal" if len(gpu_set) > 1 else "auto"
    cfg["tensor_split"] = _tensor_split_from_strategy(cfg["tensor_split_strategy"], gpu_set, total_gpu_count)
    cfg["main_gpu"] = gpu_set[0] if gpu_set else 0
    return cfg


def _minimum_gpu_count_for_config(
    model_path: str,
    params: dict,
    hw: dict,
    total_gpu_count: int,
    *,
    max_load_ratio: float = 0.95,
) -> dict:
    """Estimate the smallest GPU set that can load this fixed config.

    llama.cpp allocates KV cache from the configured ctx/parallel at load time,
    so this feasibility check is prompt-length independent as long as later
    prompts stay within that ctx.  It is an estimate/preflight, not a throughput
    benchmark.
    """
    candidates = sorted(_get_feasible_gpu_sets(total_gpu_count), key=lambda item: (len(item), item))
    best: dict | None = None
    for gpu_set in candidates:
        if not gpu_set:
            continue
        cfg = _with_gpu_set_for_minimum_check(params, gpu_set, total_gpu_count)
        breakdown = _estimate_vram_breakdown(model_path, cfg, hw)
        budget = float(breakdown.get("budget", 0.0) or 0.0)
        total = float(breakdown.get("total", 0.0) or 0.0)
        ratio = total / max(1.0, budget)
        feasible = bool(budget > 0 and ratio <= max_load_ratio)
        record = {
            "gpu_count": len(gpu_set),
            "gpu_set": list(gpu_set),
            "tensor_split": cfg.get("tensor_split"),
            "main_gpu": cfg.get("main_gpu"),
            "estimated_vram_mib": total,
            "budget_mib": budget,
            "load_ratio": ratio,
            "feasible": feasible,
        }
        if best is None or ratio < float(best.get("load_ratio", float("inf"))):
            best = record
        if feasible:
            return record
    if best is None:
        return {"gpu_count": 0, "gpu_set": [], "feasible": False, "reason": "no-gpu-candidates"}
    best = dict(best)
    best["feasible"] = False
    best["reason"] = "no-estimated-fit"
    return best


def repair_until_feasible(config: dict, hw: dict, model_path: str) -> tuple[dict, list[str]]:
    """Repair a config by adjusting parameters until it fits in VRAM of the selected GPU set.
    
    Repair order (most to least impact):
      1. KV cache quantization (q8_0)
      2. Parallelism reduction
      3. Batch/ubatch reduction
      4. Speculative decoding reduction
      5. tensor_split rebalancing (spread load more evenly)
      6. n_gpu_layers reduction (reduce model allocation, keep GPU count)
      7. **NEVER**: context_size (policy-enforced: CTX_SIZE_POLICY)
    
    Returns (repaired_config, repair_log).
    """
    gpu_set = config.get("gpu_set", [0])
    vram_mib_list = hw.get("vram_mib", [8192])
    
    budget = sum(vram_mib_list[i] for i in gpu_set if i < len(vram_mib_list))
    if budget <= 0:
        return config, ["ERROR: No VRAM budget available for selected GPUs"]
        
    repaired = dict(config)
    repair_log = []
    
    # Phase 0: Mechanical repairs
    if repaired.get("ubatch_size", 512) < 256:
        repaired["ubatch_size"] = 256
        repair_log.append("Mechanical fix: ubatch_size below supported minimum, raising to 256.")

    if repaired.get("ubatch_size", 512) > repaired.get("batch_size", 2048):
        orig_ub = repaired["ubatch_size"]
        repaired["ubatch_size"] = repaired["batch_size"]
        repair_log.append(f"Mechanical fix: ubatch {orig_ub} > batch {repaired['batch_size']}, capping ubatch.")

    if len(gpu_set) > 1 and repaired.get("split_mode") == "none":
        repaired["split_mode"] = "layer"
        repair_log.append("Mechanical fix: multi-GPU requested but split_mode was 'none', switching to 'layer'.")
        
    if repaired.get("model_draft"):
        main_ctx = repaired.get("ctx_size", 2048)
        draft_ctx = repaired.get("ctx_size_draft", 2048)
        if draft_ctx > main_ctx:
            repaired["ctx_size_draft"] = main_ctx
            repair_log.append(f"Mechanical fix: draft context {draft_ctx} > main context {main_ctx}, capping draft context.")
    
    # Phase 1-7: Iterative repair loop
    for iteration in range(15):
        breakdown = _estimate_vram_breakdown(model_path, repaired, hw)
        total_needed = breakdown["total"]
        
        # Aim for amber zone (95%)
        if total_needed <= budget * VRAM_ZONES["amber"]:
            break
        
        # Step 1: KV Cache quantization (biggest savings)
        if repaired.get("cache_type_k") != "q8_0" or repaired.get("cache_type_v") != "q8_0":
            repaired["cache_type_k"] = "q8_0"
            repaired["cache_type_v"] = "q8_0"
            repair_log.append("Set KV cache to q8_0")
            continue
        
        # Step 2: Parallelism reduction
        if repaired.get("parallel", 1) > 1:
            repaired["parallel"] = max(1, repaired["parallel"] // 2)
            repair_log.append(f"Reduced parallel slots to {repaired['parallel']}")
            continue
            
        # Step 3: Batch sizes reduction
        current_ubatch = repaired.get("ubatch_size", 512)
        if current_ubatch > 256:
            repaired["ubatch_size"] = current_ubatch // 2
            if repaired["ubatch_size"] < 256:
                repaired["ubatch_size"] = 256
            repaired["batch_size"] = max(repaired["ubatch_size"], repaired.get("batch_size", 2048) // 2)
            repair_log.append(f"Reduced batch sizes to {repaired['batch_size']}/{repaired['ubatch_size']}")
            continue

        # Step 4: Speculative decoding reduction
        if repaired.get("model_draft"):
            if repaired.get("ctx_size_draft", 2048) > 1024:
                repaired["ctx_size_draft"] = 1024
                repair_log.append("Reduced draft context to 1024")
            else:
                repair_log.append("Speculative decoding remains enabled but config is infeasible (no room for draft model)")
                break
            continue

        # Step 5: tensor_split rebalancing (spread load more evenly across GPUs)
        if len(gpu_set) > 1:
            current_ts = repaired.get("tensor_split", "1")
            new_ts = _spread_tensor_split(current_ts, len(gpu_set))
            if new_ts != current_ts:
                repaired["tensor_split"] = new_ts
                repair_log.append(f"Rebalanced tensor_split from {current_ts} to {new_ts} (more even distribution)")
                continue
        
        # Step 6: n_gpu_layers reduction (reduce model allocation, keep GPU count)
        if repaired.get("n_gpu_layers") == "all" and len(gpu_set) > 1:
            # Only try this if we haven't already reduced batch/parallel significantly
            if repaired.get("parallel", 1) <= 2 and repaired.get("ubatch_size", 512) <= 256:
                repaired["n_gpu_layers"] = "auto"
                repair_log.append("Reduced n_gpu_layers from 'all' to 'auto'")
                continue
        
        # Step 7: Context size is policy-constrained
        # In the default policy (CTX_SIZE_POLICY = "never_decrease") we do NOT decrease it
        current_ctx = repaired.get("ctx_size", 8192)
        if current_ctx > 2048:
            if CTX_SIZE_POLICY != "never_decrease":
                repaired["ctx_size"] = max(2048, current_ctx - 1024)
                repair_log.append(f"Reduced context size to {repaired['ctx_size']}")
                continue
            else:
                repair_log.append(f"Kept context size at {current_ctx} due to policy {CTX_SIZE_POLICY}")
                break
            
        # If no more repairs possible, break
        break
        
    return repaired, repair_log


def _direct_io_probe_score(metrics: dict, requested_ctx: int, requested_gpus: int, *, api_mode: bool) -> float:
    if api_mode:
        return score_server_performance(metrics, requested_ctx, requested_gpus)
    return score_performance(metrics, requested_ctx, requested_gpus)


def _choose_direct_io_preference(
    off_metrics: dict,
    on_metrics: dict,
    requested_ctx: int,
    requested_gpus: int,
    *,
    api_mode: bool,
    expected_models_loaded: int = 1,
) -> bool:
    off_score = _direct_io_probe_score(off_metrics, requested_ctx, requested_gpus, api_mode=api_mode)
    on_score = _direct_io_probe_score(on_metrics, requested_ctx, requested_gpus, api_mode=api_mode)
    if on_score == off_score:
        on_load = float(on_metrics.get("load_ready_s", 0.0))
        off_load = float(off_metrics.get("load_ready_s", 0.0))
        # For speculative mode we expect two models in memory; if load times are
        # near-identical, prefer the run that showed stronger GPU load activity.
        if expected_models_loaded > 1 and abs(on_load - off_load) <= 1.0:
            on_delta = float(on_metrics.get("gpu_load_peak_delta_mib", 0.0))
            off_delta = float(off_metrics.get("gpu_load_peak_delta_mib", 0.0))
            if on_delta != off_delta:
                return on_delta >= off_delta
        return on_load <= off_load
    return on_score > off_score


def _normalize_ctx_size(user_ctx: int, candidate_ctx: int | None = None) -> int:
    """Keep the user's context as the floor; it can stay the same or grow."""
    if candidate_ctx is None:
        return int(user_ctx)
    return max(int(user_ctx), int(candidate_ctx))


def _resolve_catalog_path(args) -> Path:
    candidate = getattr(args, "catalog", None) or os.environ.get("LLAMACPP_CATALOG") or DEFAULT_CATALOG_PATH
    return Path(candidate)


def _resolve_auto_perf_log_path(args, model_id: str | None = None) -> Path:
    explicit = getattr(args, "auto_perf_log", None) or os.environ.get("LLAMACPP_AUTO_PERF_LOG")
    if explicit:
        return Path(explicit).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_id or "auto-performance")).strip("._-") or "auto-performance"
    return (AUTO_PERF_LOG_DIR / stamp / f"{safe_model}.log").resolve()


def _append_auto_perf_log(log_path: Path | None, message: str) -> None:
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")
    except Exception:
        pass


def _append_best_history(entry: dict) -> None:
    try:
        AUTO_PERF_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUTO_PERF_HISTORY_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        pass


def _auto_perf_reason(metrics: dict) -> str:
    if metrics.get("oom"):
        return "oom"
    if metrics.get("timeout"):
        return "timeout"
    if metrics.get("crash"):
        return "crash"
    return "ok"


def _format_trial_result(metrics: dict, score: float | None = None, is_probe: bool = False) -> str:
    """Format trial result with clear termination reason.
    
    Args:
        metrics: Benchmark metrics dict
        score: Trial score (None for probe only)
        is_probe: True if this is a load probe, False if it's the full benchmark
    
    Returns:
        Human-readable result string showing reason + metrics
    """
    reason = _auto_perf_reason(metrics)
    load_reason = str(metrics.get("load_reason") or "unknown")
    load_s = _metric_float(metrics, "load_ready_s")
    prefill = _metric_float(metrics, "prefill_tokens_s")
    decode = _metric_float(metrics, "decode_tokens_s")
    error_msg = metrics.get("error", "")
    
    # Determine termination reason
    if metrics.get("oom"):
        term_reason = "OOM"
    elif metrics.get("crash"):
        term_reason = f"CRASH (load_reason={load_reason})"
    elif metrics.get("timeout") and "pruning" in str(error_msg):
        term_reason = "EARLY_PRUNING (mid-inference throughput too low)"
    elif metrics.get("timeout"):
        term_reason = f"TIMEOUT (load_reason={load_reason})"
    else:
        term_reason = "SUCCESS"
    
    # Format output
    if is_probe:
        if term_reason == "SUCCESS":
            return f"✓ Probe OK (load={load_s:.2f}s)"
        else:
            return f"✗ Probe failed: {term_reason}"
    else:
        # Full benchmark result
        if term_reason == "SUCCESS":
            return f"✓ Prefill: {prefill:.1f} t/s | Decode: {decode:.1f} t/s | Load: {load_s:.2f}s | Score: {score:.2f}"
        else:
            return f"✗ Trial failed: {term_reason}"


def _log_trial_event(log_path: Path | None, event: str, payload: dict) -> None:
    _append_auto_perf_log(log_path, f"{event} {json.dumps(payload, ensure_ascii=False, sort_keys=True)}")


def _append_text_block(log_path: Path | None, header: str, text: str, footer: str) -> None:
    if not text:
        return
    _append_auto_perf_log(log_path, header)
    for line in text.splitlines():
        _append_auto_perf_log(log_path, line)
    _append_auto_perf_log(log_path, footer)


def _tail_text(text: str, max_chars: int = 12000) -> str:
    if not text:
        return ""
    return text if len(text) <= max_chars else text[-max_chars:]


def _extract_descriptive_error(raw_output: str, fallback: str = "") -> str:
    """Return a concise, actionable error line from noisy server output."""
    if not raw_output:
        return str(fallback or "").strip()

    ignore_prefixes = (
        "[New LWP",
        "[Thread debugging",
        "Using host libthread_db",
    )

    # Prioritize lines that typically explain the actual root cause.
    priority_tokens = (
        "cuda error",
        "illegal memory access",
        "out of memory",
        "oom",
        "bad_alloc",
        "segmentation fault",
        "segfault",
        "assert",
        "failed",
        "error:",
    )

    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]

    filtered = []
    for line in lines:
        if line.startswith(ignore_prefixes):
            continue
        if line.startswith("[Inferior") and line.endswith("detached]"):
            continue
        if line.startswith("#"):
            continue
        filtered.append(line)

    for line in filtered:
        low = line.lower()
        if any(token in low for token in priority_tokens):
            return line

    if filtered:
        return filtered[-1]
    return str(fallback or "").strip()


def _capture_server_output_tail(server_proc, max_chars: int = 12000) -> str:
    if not server_proc:
        return ""
    try:
        out, err = server_proc.communicate(timeout=1)
        return _tail_text((out or "") + ("\n" + err if err else ""), max_chars=max_chars)
    except Exception:
        try:
            if getattr(server_proc, "stdout", None):
                return _tail_text(server_proc.stdout.read() or "", max_chars=max_chars)
        except Exception:
            return ""
    return ""


def _print_success_trial_metrics_block(metrics: dict, score: float, gpu_memory_budget_mib: float | None) -> None:
    prefill_tps = float(metrics.get("prefill_tokens_s", 0.0) or 0.0)
    decode_tps = float(metrics.get("decode_tokens_s", 0.0) or 0.0)
    load_time = float(metrics.get("load_ready_s", 0.0) or 0.0)
    prefill_tokens = float(metrics.get("prefill_tokens", 0.0) or 0.0)
    generation_tokens = float(metrics.get("generation_tokens", 0.0) or 0.0)
    if prefill_tokens <= 0.0:
        prefill_tokens = float(metrics.get("prompt_tokens", 0.0) or 0.0)
    if generation_tokens <= 0.0:
        generation_tokens = float(metrics.get("predicted_tokens", 0.0) or 0.0)

    print("    Starting server (loading model into VRAM)...")
    print(f"    ✓ Server ready in {load_time:.2f}s")
    print(f"    ✓ Prefill: {prefill_tps:.1f} t/s | Decode: {decode_tps:.1f} t/s")
    print(f"      (Prefill tokens: {prefill_tokens:.1f} | Generation tokens: {generation_tokens:.1f})")

    if gpu_memory_budget_mib is not None:
        baseline_vram_mib = float(metrics.get("vram_used", 0.0) or 0.0)
        memory_headroom_mib = max(0.0, float(gpu_memory_budget_mib) - baseline_vram_mib)
        memory_headroom_ratio = memory_headroom_mib / max(1.0, float(gpu_memory_budget_mib))
        print(f"  GPU free memory budget: {float(gpu_memory_budget_mib):.0f} MiB")
        print(f"  Estimated headroom after baseline: {memory_headroom_mib:.0f} MiB ({memory_headroom_ratio * 100:.1f}%)")
    else:
        print("  GPU free memory budget: n/a")
        print("  Estimated headroom after baseline: n/a")

    print(f"  Prefill throughput: {prefill_tps:.2f} tokens/s")
    print(f"  Decode throughput:  {decode_tps:.2f} tokens/s")
    print(f"  Total throughput:   {_total_tokens_s(metrics):.2f} tokens/s")
    print(f"  Load time:          {load_time:.2f}s")
    print(f"  Overall Score:      {score:.2f}")


def _metric_float(metrics: dict, key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return float(default)


def _total_tokens_s(metrics: dict | None) -> float:
    """Return combined prompt+decode throughput for one benchmark result."""
    if not isinstance(metrics, dict):
        return 0.0
    explicit = _metric_float(metrics, "total_tokens_s")
    if explicit > 0:
        return explicit
    # Backwards-compatible fallback for cached rows written before
    # total_tokens_s existed. In the current benchmark path both rates use the
    # same elapsed denominator, so the sum is total_tokens/total_time.
    return _metric_float(metrics, "prefill_tokens_s") + _metric_float(metrics, "decode_tokens_s")


def _resolve_baseline(model, gpu_count: int) -> dict:
    overrides = normalize_server_overrides(getattr(model, "server_overrides", {}) or {})
    baseline_ctx = int(overrides.get("ctx_size", getattr(model, "ctx_size", 8192)) or getattr(model, "ctx_size", 8192) or 8192)
    return {
        "gpu_mask": (1 << gpu_count) - 1, # Default to all GPUs
        "ctx_size": baseline_ctx,
        "batch_size": int(overrides.get("batch_size", 2048) or 2048),
        "ubatch_size": int(overrides.get("ubatch_size", 512) or 512),
        "kv_offload": bool(overrides.get("kv_offload", False)),
        "split_mode": str(overrides.get("split_mode", "layer")),
        "tensor_split": str(overrides.get("tensor_split", "auto")),
        "main_gpu": int(overrides.get("main_gpu", 0) or 0),
        "fit": bool(overrides.get("fit", True)),
        "n_gpu_layers": overrides.get("n_gpu_layers", "all"),
        "flash_attn": str(overrides.get("flash_attn", "auto")),
        "cache_type_k": _normalize_cache_type(overrides.get("cache_type_k", CACHE_TYPE_FLOOR)),
        "cache_type_v": _normalize_cache_type(overrides.get("cache_type_v", CACHE_TYPE_FLOOR)),
        "direct_io": bool(overrides.get("direct_io", False)),
        "numa": overrides.get("numa", None),
        "threads_http": int(overrides.get("threads_http", 1) or 1),
    }




def _load_auto_perf_profiles(path: Path | None = None) -> list[dict]:
    profile_path = path or AUTO_PERF_PROFILES_PATH
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _phase_result_key(params: dict, phase_label: str, api_mode: bool = False) -> str:
    payload = {
        "phase": phase_label,
        "benchmark_key": _canonical_benchmark_key(params, api_mode=api_mode),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")).hexdigest()


def _profile_phase_results(row: dict) -> dict:
    phase_results = row.get("phase_results")
    return phase_results if isinstance(phase_results, dict) else {}


def _benchmark_metrics_failed(metrics: dict | None) -> bool:
    if not isinstance(metrics, dict):
        return True
    return bool(metrics.get("oom") or metrics.get("crash") or metrics.get("timeout"))


def _cached_phase_result_is_usable(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    metrics = result.get("metrics")
    if _benchmark_metrics_failed(metrics):
        return False
    try:
        score = float(result.get("score", 0.0) or 0.0)
    except Exception:
        return False
    return score > -999.0


def _baseline_failure_is_fatal(metrics: dict | None) -> tuple[bool, str]:
    """Return whether a failed baseline should abort tuning instead of burning trials."""
    if not _benchmark_metrics_failed(metrics):
        return False, ""
    if not isinstance(metrics, dict):
        return True, "missing-baseline-metrics"
    haystack = " ".join(
        str(metrics.get(k) or "")
        for k in ("error", "server_output_tail", "load_reason", "reason")
    ).lower()
    fatal_tokens = [
        "failed to initialize cuda",
        "no cuda-capable device",
        "couldn't communicate with nvidia driver",
        "couldn't bind http server socket",
        "address already in use",
        "invalid argument",
        "failed to open gguf",
        "no such file or directory",
        "error while loading shared libraries",
        "cannot open shared object file",
        "server-crashed",
        "invalid-model",
    ]
    for token in fatal_tokens:
        if token in haystack:
            return True, token
    if bool(metrics.get("crash")) and not bool(metrics.get("oom")):
        return True, "baseline-server-crashed"
    return False, ""


def _find_cached_phase_result(profile_key: str, result_key: str, role: str, path: Path | None = None) -> dict | None:
    for row in reversed(_load_auto_perf_profiles(path)):
        if row.get("profile_key") != profile_key:
            continue
        result = _profile_phase_results(row).get(result_key)
        if isinstance(result, dict) and result.get("role") == role and _cached_phase_result_is_usable(result):
            return result
    return None


def _find_cached_baseline_result(
    profile_key: str,
    result_key: str,
    params: dict,
    phase_label: str,
    *,
    api_mode: bool = False,
    model_id: str | None = None,
    path: Path | None = None,
) -> dict | None:
    """Find a reusable successful baseline, including older top-level profiles."""
    direct = _find_cached_phase_result(profile_key, result_key, "baseline", path)
    if direct is not None:
        return direct

    benchmark_key = _canonical_benchmark_key(params, api_mode=api_mode)
    rows = list(reversed(_load_auto_perf_profiles(path)))
    for allow_cross_profile in (False, True):
        for row in rows:
            same_profile = row.get("profile_key") == profile_key
            same_model = model_id is not None and row.get("model") == model_id
            if not same_profile and not (allow_cross_profile and same_model):
                continue
            # Newer rows may keep the compatible baseline under phase_results
            # even if a volatile profile_key changed between executions.
            result = _profile_phase_results(row).get(result_key)
            if isinstance(result, dict) and result.get("role") == "baseline" and _cached_phase_result_is_usable(result):
                return {**result, "source": result.get("source") or ("cache-cross-profile" if not same_profile else "cache")}

            baseline = row.get("baseline")
            if not isinstance(baseline, dict) or not _cached_phase_result_is_usable({"score": baseline.get("trial_value"), "metrics": baseline.get("metrics")}):
                continue
            stored_phase = str(baseline.get("phase") or "").strip()
            stored_key = str(baseline.get("benchmark_key") or "").strip()
            if stored_phase and stored_phase != phase_label:
                continue
            if stored_key and stored_key != benchmark_key:
                continue
            # Older rows may not have phase/benchmark_key. Same profile_key is
            # still a useful fallback for CORE/raw baselines, but do not use
            # metadata-less rows across profiles or raw/server boundaries.
            if not stored_phase and (phase_label != "CORE" or not same_profile):
                continue
            return {
                "phase": stored_phase or phase_label,
                "role": "baseline",
                "score": float(baseline.get("trial_value", 0.0) or 0.0),
                "metrics": dict(baseline.get("metrics") or {}),
                "params": dict(baseline.get("params") or params),
                "benchmark_key": stored_key or benchmark_key,
                "updated_at": baseline.get("updated_at") or row.get("created_at"),
                "source": "cache-top-level" if same_profile else "cache-top-level-cross-profile",
            }
    return None


def _catalog_auto_perf_store(model) -> dict:
    overrides = getattr(model, "server_overrides", None)
    if not isinstance(overrides, dict):
        return {}
    store = overrides.get(AUTO_PERF_CATALOG_KEY)
    return store if isinstance(store, dict) else {}


def _catalog_baseline_cache_key(phase_label: str, benchmark_key: str) -> str:
    return f"{phase_label}:{benchmark_key}"


def _catalog_baseline_sort_key(item: tuple[str, dict]) -> str:
    record = item[1] if isinstance(item[1], dict) else {}
    return str(record.get("updated_at") or "")


def _compact_catalog_auto_performance_store(store: dict | None) -> dict:
    """Return the minimal catalog metadata needed for future baseline reuse.

    The catalog is production configuration first.  Auto-performance metadata is
    allowed there only as a compact cache, so legacy score schemas, malformed
    records and unbounded baseline history must not accumulate inside
    ``server_overrides``.
    """
    if not isinstance(store, dict):
        return {}
    raw_baselines = store.get("baselines")
    if not isinstance(raw_baselines, dict):
        return {}

    grouped: dict[str, list[tuple[str, dict]]] = {}
    for key, value in raw_baselines.items():
        if not isinstance(key, str) or ":" not in key or not isinstance(value, dict):
            continue
        phase, _benchmark_key = key.split(":", 1)
        if phase not in {"CORE", "SPECULATIVE", "SERVER"}:
            continue
        if value.get("role") != "baseline":
            continue
        if value.get("score_schema_version") != AUTO_PERF_SCORE_SCHEMA_VERSION:
            continue
        if not _cached_phase_result_is_usable(value):
            continue
        compact = dict(value)
        compact["params"] = _params_for_observation(compact.get("params") or {})
        grouped.setdefault(phase, []).append((key, compact))

    baselines: dict[str, dict] = {}
    for phase_items in grouped.values():
        for key, value in sorted(phase_items, key=_catalog_baseline_sort_key, reverse=True)[:AUTO_PERF_CATALOG_MAX_BASELINES_PER_PHASE]:
            baselines[key] = value
    if not baselines:
        return {}
    result = {"baselines": baselines}
    if store.get("updated_at"):
        result["updated_at"] = store.get("updated_at")
    return result


def _catalog_server_overrides_for_apply(existing_overrides: dict | None, selected_params: dict) -> dict:
    """Build the exact ``server_overrides`` object to persist in catalog.

    This is deliberately replace-not-merge for executable flags: stale trial
    params from older runs must not survive.  Only compact
    ``auto_performance`` metadata is carried over.
    """
    persisted = normalize_server_overrides(_trial_params_for_catalog(selected_params or {}))
    existing_meta = {}
    if isinstance(existing_overrides, dict):
        existing_meta = _compact_catalog_auto_performance_store(existing_overrides.get(AUTO_PERF_CATALOG_KEY))
    if existing_meta:
        persisted[AUTO_PERF_CATALOG_KEY] = existing_meta
    return persisted


def _refresh_llamaswap_config_after_auto_perf(args, catalog_items: list[ManagedModel]) -> bool:
    """Regenerate llama-swap config after final catalog apply.

    llama-swap is expected to run with ``--watch-config``; rewriting the YAML is
    enough for it to pick up the new command without a privileged service
    restart.  This mirrors the update flow's config-rendering step, but avoids
    expensive downloads/probes and avoids sudo.
    """
    try:
        config_path = Path(getattr(args, "config", None) or (Path.home() / ".local/state/llamacpp-superserver/config.yaml"))
        llama_server = Path(getattr(args, "llama_server", None) or DEFAULT_LLAMA_SERVER)
        start_port = int(getattr(args, "start_port", 12000) or 12000)
        render_llamaswap_config(
            catalog_items,
            config_path,
            llama_server,
            start_port,
            resolve_idle_ttl(args),
            server_defaults=resolve_llama_server_defaults(args),
        )
        print(f"Configuracion de llama-swap regenerada: {config_path.resolve()}")
        service_name = str(getattr(args, "service", "llamaswap") or "llamaswap")
        restart_service_to_free_vram(service_name)
        return True
    except Exception as exc:
        print(f"⚠️  Catalogo guardado, pero no se pudo regenerar config.yaml automaticamente: {exc}")
        print("Ejecuta `llamacpp-superserver update --model-id <modelo>` para regenerarla manualmente.")
        return False


def _find_catalog_cached_baseline(
    model,
    params: dict,
    phase_label: str,
    *,
    api_mode: bool = False,
    llama_cpp_version: str | None = None,
    hardware_fingerprint: str | None = None,
) -> dict | None:
    benchmark_key = _canonical_benchmark_key(params, api_mode=api_mode)
    store = _catalog_auto_perf_store(model)
    baselines = store.get("baselines")
    if not isinstance(baselines, dict):
        return None
    result = baselines.get(_catalog_baseline_cache_key(phase_label, benchmark_key))
    if isinstance(result, dict) and result.get("role") == "baseline" and _cached_phase_result_is_usable(result):
        stored_version = str(result.get("llama_cpp_version") or "").strip()
        if llama_cpp_version and stored_version and stored_version != str(llama_cpp_version):
            return None
        stored_hw = str(result.get("hardware_fingerprint") or "").strip()
        if hardware_fingerprint and stored_hw and stored_hw != str(hardware_fingerprint):
            return None
        return {**result, "source": "catalog"}
    return None


def _save_catalog_baseline_cache(catalog_path: Path, model_id: str, phase_label: str, benchmark_key: str, result: dict) -> bool:
    try:
        items, _ = load_catalog_with_diagnostics(catalog_path)
        for item in items:
            if item.model_id != model_id:
                continue
            raw_overrides = getattr(item, "server_overrides", None)
            executable_overrides = _catalog_server_overrides_for_apply(raw_overrides, raw_overrides or {})
            item.server_overrides = executable_overrides
            store = item.server_overrides.get(AUTO_PERF_CATALOG_KEY)
            if not isinstance(store, dict):
                store = {}
            baselines = store.get("baselines")
            if not isinstance(baselines, dict):
                baselines = {}
            baselines[_catalog_baseline_cache_key(phase_label, benchmark_key)] = dict(result)
            store["baselines"] = baselines
            store["updated_at"] = datetime.now(timezone.utc).isoformat()
            store = _compact_catalog_auto_performance_store(store)
            if store:
                store["updated_at"] = datetime.now(timezone.utc).isoformat()
            item.server_overrides[AUTO_PERF_CATALOG_KEY] = store
            save_catalog(catalog_path, items)
            return True
    except Exception:
        return False
    return False


def _upsert_profile(row: dict) -> Path | None:
    """Write/update a profile row and return the actual path written.

    Attempts canonical path first, then falls back to AUTO_PERF_LOG_DIR,
    then to the user's cache. Returns a Path for the file written or None
    if all attempts fail.
    """
    try:
        AUTO_PERF_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            if AUTO_PERF_PROFILES_PATH.exists():
                data = json.loads(AUTO_PERF_PROFILES_PATH.read_text(encoding="utf-8"))
                if not isinstance(data, list):
                    data = []
            else:
                data = []
        except Exception:
            data = []

        existing = next((p for p in data if p.get("profile_key") == row.get("profile_key")), None)
        if isinstance(existing, dict):
            merged_phase_results = dict(_profile_phase_results(existing))
            merged_phase_results.update(_profile_phase_results(row))
            row = {**existing, **row, "phase_results": merged_phase_results}
        data = [p for p in data if p.get("profile_key") != row.get("profile_key")]
        data.append(row)
        AUTO_PERF_PROFILES_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return AUTO_PERF_PROFILES_PATH
    except (PermissionError, OSError) as e:
        # Primary location not writable — fallback to AUTO_PERF_LOG_DIR with timestamp.
        try:
            AUTO_PERF_LOG_DIR.mkdir(parents=True, exist_ok=True)
            fallback_name = f"auto_performance_profiles_fallback_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
            fallback_path = AUTO_PERF_LOG_DIR / fallback_name

            try:
                if fallback_path.exists():
                    fallback_data = json.loads(fallback_path.read_text(encoding="utf-8"))
                    if not isinstance(fallback_data, list):
                        fallback_data = []
                else:
                    fallback_data = []
            except Exception:
                fallback_data = []

            existing = next((p for p in fallback_data if p.get("profile_key") == row.get("profile_key")), None)
            if isinstance(existing, dict):
                merged_phase_results = dict(_profile_phase_results(existing))
                merged_phase_results.update(_profile_phase_results(row))
                row = {**existing, **row, "phase_results": merged_phase_results}
            fallback_data = [p for p in fallback_data if p.get("profile_key") != row.get("profile_key")]
            fallback_data.append(row)
            fallback_path.write_text(json.dumps(fallback_data, indent=2, ensure_ascii=False), encoding="utf-8")

            try:
                _append_auto_perf_log(fallback_path.parent / (fallback_path.name.replace('.json', '.log')), f"FALLBACK_PROFILE_SAVED path={fallback_path} reason={e}")
            except Exception:
                pass
            return fallback_path
        except Exception:
            # If fallback also fails, try a user cache path before giving up.
            try:
                user_cache_dir = Path.home() / ".cache" / "llamacpp-superserver"
                user_cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = user_cache_dir / f"auto_performance_profiles_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
                cache_path.write_text(json.dumps([row], indent=2, ensure_ascii=False), encoding="utf-8")
                try:
                    _append_auto_perf_log(cache_path.parent / (cache_path.name.replace('.json', '.log')), f"USER_CACHE_PROFILE_SAVED path={cache_path} reason={e}")
                except Exception:
                    pass
                return cache_path
            except Exception:
                try:
                    _append_auto_perf_log(None, f"FAILED_TO_SAVE_PROFILE reason={e}")
                except Exception:
                    pass
                return None


def _map_n_gpu_layers(value):
    if value in {"all", 999}:
        return 999
    if value in {"auto", -1}:
        return -1
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return value
    if numeric_value >= 999:
        return 999
    if numeric_value < 0:
        return -1
    return numeric_value


def _n_gpu_layers_cli_value(value):
    if value in {"all", 999}:
        return "all"
    if value in {"auto", -1}:
        return "auto"
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric_value >= 999:
        return "all"
    if numeric_value < 0:
        return "auto"
    return str(numeric_value)


def _trial_params_for_catalog(params: dict) -> dict:
    """Persist only catalog-safe params and map sentinel values.

    The fast tuner intentionally keeps a small search surface; params outside
    the fast path are not persisted here.
    """
    out: dict[str, object] = {}
    for key, value in params.items():
        if key in {"gpu_mask", "ts_strategy", "tensor_split_strategy", "ctx_size", "main_gpu_raw", "gpu_set", "gpu_set_idx", "mmap", "direct_io", AUTO_PERF_CATALOG_KEY}:
            continue
        if key == "fit":
            if isinstance(value, str):
                out[key] = value.strip().lower() in {"1", "true", "on", "yes"}
            else:
                out[key] = bool(value)
            continue
        if key == "n_gpu_layers":
            out[key] = _map_n_gpu_layers(value)
            continue
        if key == "n_gpu_layers_draft":
            out[key] = _n_gpu_layers_cli_value(value)
            continue
        out[key] = value
    return out


def _params_for_observation(params: dict) -> dict:
    """Store compact, non-orchestration params in catalog/profile history."""
    safe = dict(_trial_params_for_catalog(params))
    if "ctx_size" in params:
        safe["ctx_size"] = params.get("ctx_size")
    return safe


def _phase_result_record(
    *,
    phase: str,
    role: str,
    score: float,
    metrics: dict,
    params: dict,
    benchmark_key: str,
    trial,
    extra: dict | None = None,
) -> dict:
    record = {
        "phase": phase,
        "role": role,
        "score": score,
        "metrics": dict(metrics),
        "params": _params_for_observation(params),
        "benchmark_key": benchmark_key,
        "trial": trial,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        record.update(extra)
    return record


def _prepare_benchmark_params(config: dict, api_mode: bool = False) -> dict:
    """Extract and map only valid benchmark parameters from a config dict.
    
    Removes internal orchestration keys (gpu_set_idx, ts_strategy, etc.)
    and filters out Phase 2 parameters if not in api_mode.
    """
    # Internal keys used only for Optuna/orchestration, not for run_benchmark
    internal_keys = {
        "gpu_set",
        "gpu_set_idx",
        "ts_strategy",
        "tensor_split_strategy",
        "main_gpu_raw",
        "mmap",
        AUTO_PERF_CATALOG_KEY,
    }
    cache_keys = {"cache_type_k", "cache_type_v", "cache_type_k_draft", "cache_type_v_draft"}
    
    # Phase 2 server-specific keys to remove in Phase 1
    phase2_keys = {"parallel", "cont_batching", "ctx_checkpoints", "cache_ram", "threads_http", "kv_unified", "cache_idle_slots"}
    
    result = {
        k: v
        for k, v in config.items()
        if k not in internal_keys and k not in cache_keys and v is not None and str(v) != "None"
    }
    
    if not api_mode:
        result = {k: v for k, v in result.items() if k not in phase2_keys}
    
    return result




def _canonical_benchmark_key(config: dict, api_mode: bool = False) -> str:
    """Stable key for configs that produce the same benchmark command surface.

    Optuna samples raw orchestration parameters, then auto-performance normalizes
    and repairs them. Many different raw trials can collapse to the same actual
    benchmark params; this key lets the search reuse metrics instead of spending
    another expensive llama.cpp run.
    """
    prepared = _prepare_benchmark_params(config, api_mode=api_mode)
    payload_obj: dict[str, object] = {
        "params": prepared,
        "api_mode": bool(api_mode),
    }
    if api_mode:
        # SERVER benchmarks must not reuse older server baselines measured with
        # a different generation length; otherwise the baseline and trials are
        # not comparable.
        payload_obj["benchmark_schema"] = AUTO_PERF_SERVER_BENCHMARK_SCHEMA_VERSION
        payload_obj["server_n_predict"] = SERVER_BENCHMARK_N_PREDICT
    else:
        payload_obj["benchmark_schema"] = AUTO_PERF_RAW_BENCHMARK_SCHEMA_VERSION
        payload_obj["raw_screening_n_predict"] = RAW_SCREENING_N_PREDICT
        payload_obj["raw_screening_runs"] = RAW_SCREENING_RUNS
        payload_obj["long_confirm_prompt_tokens"] = LONG_CONFIRM_PROMPT_TOKENS
        payload_obj["long_confirm_predict_tokens"] = LONG_CONFIRM_PREDICT_TOKENS
    payload = json.dumps(payload_obj, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _repair_is_structural(repair_log: list[str]) -> bool:
    """Return True when repair materially changed the trial intent.

    Mechanical repairs are normalization (for example ubatch <= batch). Structural
    repairs change the search point enough that it should not consume useful
    evaluation budget unless it leads to a new actual benchmark config.
    """
    structural_markers = (
        "Set KV cache",
        "Reduced parallel",
        "Reduced batch",
        "Reduced draft",
        "Disabled speculative",
        "Rebalanced tensor_split",
        "Reduced n_gpu_layers",
        "Reduced context size",
        "Kept context size",
        "ERROR:",
    )
    return any(any(marker in entry for marker in structural_markers) for entry in repair_log)

def _probe_flags() -> tuple[str, ...]:
    return tuple(sorted(PROBED_TUNER_KEYS))


def _tuner_flags_to_probe() -> tuple[str, ...]:
    return tuple(sorted(PROBED_TUNER_KEYS))

def _get_feasible_gpu_sets(gpu_count: int) -> list[list[int]]:
    """Generate a list of logical GPU combinations to explore."""
    if gpu_count <= 0:
        return [[]]
    if gpu_count == 1:
        return [[0]]
        
    sets = []
    # Individual GPUs
    for i in range(gpu_count):
        sets.append([i])
    
    # Logical pairs
    if gpu_count >= 2:
        for i in range(0, gpu_count - 1, 2):
            sets.append([i, i+1])
            
    # Half of GPUs
    if gpu_count >= 4:
        half = gpu_count // 2
        sets.append(list(range(half)))
        sets.append(list(range(half, gpu_count)))
        
    # All GPUs
    sets.append(list(range(gpu_count)))
    
    # Remove duplicates and return
    unique_sets = []
    seen = set()
    for s in sets:
        s_tuple = tuple(sorted(s))
        if s_tuple not in seen:
            unique_sets.append(s)
            seen.add(s_tuple)
    return unique_sets


def _generate_neighbor_configs(best_cfg: dict, gpu_count: int, phase: int = 1) -> list[dict]:
    """Generate local mutations (neighbors) of a successful configuration."""
    neighbors = []
    
    def mutate(key, new_val):
        cfg = dict(best_cfg)
        cfg[key] = new_val
        neighbors.append(cfg)

    if phase == 1:
        # 1. GPU Set variations
        all_sets = _get_feasible_gpu_sets(gpu_count)
        current_set = best_cfg.get("gpu_set", [])
        for s in all_sets:
            if s != current_set:
                mutate("gpu_set", s)
                
        # 2. Context variations
        ctx = best_cfg.get("ctx_size", 8192)
        mutate("ctx_size", ctx * 2)
        if ctx > 2048:
            mutate("ctx_size", ctx // 2)
            
        # 3. Batch variations
        batch = best_cfg.get("batch_size", 2048)
        mutate("batch_size", batch * 2)
        if batch > 512:
            mutate("batch_size", batch // 2)
            
    elif phase == 2:
        # Phase 2: Speculative draft-side variations
        draft = best_cfg.get("draft", 16)
        for draft_new in [8, 16, 32]:
            if draft_new != draft:
                mutate("draft", draft_new)

        draft_ctx = best_cfg.get("ctx_size_draft", 2048)
        for ctx_new in [1024, 2048, 4096]:
            if ctx_new != draft_ctx:
                mutate("ctx_size_draft", ctx_new)

        current_layers = best_cfg.get("n_gpu_layers_draft", "auto")
        for layer_choice in ["auto", "all"]:
            if layer_choice != current_layers:
                mutate("n_gpu_layers_draft", layer_choice)

    else:
        # Phase 3: Server-specific variations
        # 5. Parallel variations
        p = best_cfg.get("parallel", 1)
        for p_new in [1, 2, 4, 8]:
            if p_new != p:
                mutate("parallel", p_new)
        
        # 6. KV Unified / Cache Idle
        mutate("kv_unified", not best_cfg.get("kv_unified", True))
        mutate("cache_idle_slots", not best_cfg.get("cache_idle_slots", True))
            
    return neighbors


def _tensor_split_from_strategy(strategy: str, gpu_set: list[int], total_gpu_count: int) -> str:
    """Generate a --tensor-split string that restricts execution to gpu_set."""
    if not gpu_set:
        return "1"

    # Some llama.cpp builds reject the literal token "auto" here even when the
    # CLI accepts it elsewhere. For autotuning we want a deterministic numeric
    # split string that is always valid for the selected GPU subset.
    if strategy == "auto":
        strategy = "equal"
        
    weights = [0.0] * total_gpu_count
    
    if strategy == "equal":
        parts = [1.0] * len(gpu_set)
    elif strategy == "descending":
        parts = [float(len(gpu_set) - i) for i in range(len(gpu_set))]
    elif strategy == "skewed":
        parts = [float(len(gpu_set))] + [1.0] * (len(gpu_set) - 1)
    else:
        return "auto"
        
    # Map parts to weights for the specific GPUs in gpu_set
    for idx, gpu_idx in enumerate(gpu_set):
        if gpu_idx < total_gpu_count:
            weights[gpu_idx] = parts[idx]
            
    # Format weights
    formatted = []
    for w in weights:
        if w.is_integer():
            formatted.append(str(int(w)))
        else:
            formatted.append(f"{w:.2f}")
            
    return ",".join(formatted)


def _spread_tensor_split(tensor_split_str: str, n_gpu: int) -> str:
    """Rebalance tensor_split weights to be more evenly distributed.
    
    Goal: increase evenness coefficient (reduce std dev of weights) to spread
    load across all available GPUs when OOM is detected with uneven distribution.
    
    Example: "5,2,0" -> "3,2,1" -> "2,2,1" (more balanced).
    If already uniform "1,1,1" -> return unchanged.
    """
    if not tensor_split_str or "," not in tensor_split_str:
        # Single GPU or invalid format; can't spread further
        return tensor_split_str
    
    try:
        current_weights = [float(x) for x in tensor_split_str.split(",")]
    except ValueError:
        # Can't parse; return unchanged
        return tensor_split_str
    
    # Trim to n_gpu weights (ignore trailing zeros from unused GPUs)
    active_weights = current_weights[:n_gpu]
    
    # Check if already uniform
    if len(set(active_weights)) == 1 or all(w == active_weights[0] for w in active_weights):
        return tensor_split_str
    
    # Strategy: reduce max weight by 1, increase min weight by 1 (one iteration)
    # This gradually flattens skewed distributions
    max_idx = active_weights.index(max(active_weights))
    min_idx = active_weights.index(min(active_weights))
    
    if active_weights[max_idx] > 0 and (active_weights[min_idx] == 0 or active_weights[max_idx] - active_weights[min_idx] > 1):
        active_weights[max_idx] -= 1
        active_weights[min_idx] += 1
    
    # Pad back to total GPU count with zeros for unused GPUs
    while len(active_weights) < len(current_weights):
        active_weights.append(0.0)
    
    # Format as string (integers, no decimals)
    formatted = []
    for w in active_weights:
        if w.is_integer():
            formatted.append(str(int(w)))
        else:
            formatted.append(f"{w:.2f}")
    
    return ",".join(formatted)


def _tensor_split_strategy_candidates() -> list[str]:
    return ["auto", "equal", "descending", "skewed"]


def _tensor_split_is_too_imbalanced(tensor_split_str: str, *, max_ratio: float = 2.0) -> bool:
    try:
        weights = [float(x) for x in str(tensor_split_str or "").split(",") if str(x).strip()]
    except Exception:
        return False
    active = [w for w in weights if w > 0]
    if len(active) <= 1:
        return False
    return (max(active) / max(min(active), 1e-9)) > float(max_ratio)


def _speculative_tensor_split_strategy(best_core: dict, gpu_indices: list[int]) -> str:
    """Return a draft-safe fixed tensor split strategy for SPECULATIVE phase.

    CORE may discover a very skewed placement such as 7,1,1,1,1,1,1 that works
    for the base model alone but leaves too little room on the heavy GPU for a
    draft model. SPECULATIVE is not allowed to sample core params, but it also
    must not launch obviously draft-hostile placements. Use equal as a
    mechanical safety normalization when inheriting an imbalanced multi-GPU
    split.
    """
    strategy = str(best_core.get("tensor_split_strategy", "equal") or "equal")
    existing = str(best_core.get("tensor_split") or "")
    if len(gpu_indices) > 1 and (
        strategy in {"skewed", "descending"}
        or _tensor_split_is_too_imbalanced(existing)
    ):
        return "equal"
    return strategy


def _numa_candidates() -> list[str | None]:
    return [None, "distribute", "isolate"]


def _is_speculative_model(model) -> bool:
    return bool(getattr(model, "speculative", False))


def _phase1_tuner_keys(model) -> tuple[str, ...]:
    return tuple(sorted(PROBED_TUNER_KEYS))


def _phase2_speculative_tuner_keys() -> tuple[str, ...]:
    return tuple(sorted(PHASE1_SPECULATIVE_SEARCH_KEYS))


def _phase2_tuner_keys() -> tuple[str, ...]:
    return tuple(sorted(PHASE2_SERVER_SEARCH_KEYS))


def _merge_speculative_phase_defaults(best_core: dict | None, baseline: dict) -> dict:
    merged = dict(baseline)
    merged.update(best_core or {})
    # Phase-1 CORE results can legitimately lack or null out draft fields.
    # A speculative catalog entry, however, must never be evaluated with
    # model_draft=None/draft=0; restore baseline draft invariants.
    if not str(merged.get("model_draft") or "").strip():
        merged["model_draft"] = baseline.get("model_draft")
    try:
        if int(merged.get("draft") or 0) <= 0:
            merged["draft"] = int(baseline.get("draft") or 16)
    except Exception:
        merged["draft"] = int(baseline.get("draft") or 16)
    return merged


def _speculative_phase1_test_core_config(baseline: dict) -> dict:
    return {
        "gpu_set_idx": baseline["gpu_set_idx"],
        "split_mode": baseline["split_mode"],
        "tensor_split_strategy": baseline["tensor_split_strategy"],
        "main_gpu_raw": 0,
        "n_gpu_layers": baseline["n_gpu_layers"],
        "fit": baseline["fit"],
        "fit_target": baseline["fit_target"],
        "batch_size": baseline["batch_size"],
        "ubatch_size": baseline["ubatch_size"],
        "flash_attn": baseline["flash_attn"],
        "kv_offload": baseline["kv_offload"],
        "numa": baseline["numa"],
        "op_offload": baseline["op_offload"],
        "threads": baseline["threads"],
        "threads_batch": baseline["threads_batch"],
        "model_draft": baseline["model_draft"],
        "draft": baseline["draft"],
        "ctx_size_draft": baseline["ctx_size_draft"],
        "n_gpu_layers_draft": baseline["n_gpu_layers_draft"],
        "cache_type_k": baseline["cache_type_k"],
        "cache_type_v": baseline["cache_type_v"],
        "cache_type_k_draft": baseline["cache_type_k_draft"],
        "cache_type_v_draft": baseline["cache_type_v_draft"],
    }






def _speculative_core_changed(before: dict, after: dict) -> list[str]:
    """Core params that speculative phase is not allowed to repair/change."""
    core_keys = {
        "gpu_set",
        "split_mode",
        "tensor_split",
        "main_gpu",
        "main_gpu_raw",
        "n_gpu_layers",
        "fit",
        "fit_target",
        "batch_size",
        "ubatch_size",
        "flash_attn",
        "kv_offload",
        "numa",
        "op_offload",
        "threads",
        "threads_batch",
    }
    return sorted(key for key in core_keys if before.get(key) != after.get(key))


def _speculative_config_valid_after_repair(before: dict, after: dict) -> tuple[bool, str]:
    if not after.get("model_draft"):
        return False, "speculative-disabled-model_draft"
    try:
        if int(after.get("draft", 0) or 0) <= 0:
            return False, "speculative-disabled-draft"
    except Exception:
        return False, "speculative-invalid-draft"
    changed_core = _speculative_core_changed(before, after)
    if changed_core:
        return False, "speculative-core-repaired:" + ",".join(changed_core)
    return True, "ok"


def _speculative_repair_log_infeasible(repair_log: list[str] | tuple[str, ...] | None) -> tuple[bool, str]:
    for entry in repair_log or []:
        lowered = str(entry).lower()
        if "no room for draft model" in lowered or (
            "speculative decoding remains enabled" in lowered and "infeasible" in lowered
        ):
            return True, "speculative-infeasible-no-room-for-draft"
    return False, ""


def _speculative_trial_seed(params: dict, draft_model_path: str | None = None) -> dict:
    """Return only speculative Optuna params for the speculative phase seed."""
    seed = {
        "draft": _coerce_optuna_choice(params.get("draft", 16), [8, 16, 32], 16),
        "ctx_size_draft": _coerce_optuna_choice(params.get("ctx_size_draft", 1024), [512, 768, 1024, 2048, 4096], 1024),
        "n_gpu_layers_draft": _coerce_optuna_choice(params.get("n_gpu_layers_draft", "all"), ["all", "auto"], "all"),
    }
    # model_draft is fixed by catalog/spec metadata. It is intentionally not an
    # Optuna parameter, so do not include it in enqueued trials.
    return seed



def _server_trial_seed(params: dict) -> dict:
    """Return only server/API Optuna params for the server phase seed."""
    return {
        "parallel": _coerce_optuna_choice(params.get("parallel", 1), [1, 2, 4, 8, 16], 1),
        "cont_batching": bool(params.get("cont_batching", True)),
        "ctx_checkpoints": _coerce_optuna_choice(params.get("ctx_checkpoints", 0), [0, 32, 64], 0),
        "cache_ram": _coerce_optuna_choice(params.get("cache_ram", 0), [0, 8192, 16384, -1], 0),
        "threads_http": max(1, min(16, int(params.get("threads_http", 1) or 1))),
        "kv_unified": bool(params.get("kv_unified", False)),
        "cache_idle_slots": bool(params.get("cache_idle_slots", False)),
    }

def _validate_tuning_params(params: dict, allowed_keys: set[str], *, label: str) -> None:
    unknown = sorted(key for key in params if key not in allowed_keys)
    if unknown:
        raise ValueError(f"{label} contains unsupported tuning parameters: {', '.join(unknown)}")






def _coerce_optuna_choice(value, choices, default=None):
    """Return a value guaranteed to belong to an Optuna categorical domain.

    Catalog/server defaults may contain legacy values such as 256 for
    `batch_size` while the active Optuna space only accepts
    (512, 1024, 2048, 4096, 8192). Enqueued trials with out-of-domain fixed
    values make Optuna raise before benchmarking, so sanitize them first.
    """
    choices_tuple = tuple(choices)
    if value in choices_tuple:
        return value
    value_str = str(value).strip()
    for choice in choices_tuple:
        if str(choice).strip() == value_str:
            return choice
    numeric_choices = [c for c in choices_tuple if isinstance(c, (int, float)) and not isinstance(c, bool)]
    if numeric_choices:
        try:
            numeric_value = float(value)
            return min(numeric_choices, key=lambda c: abs(float(c) - numeric_value))
        except (TypeError, ValueError):
            pass
    if default in choices_tuple:
        return default
    return choices_tuple[0] if choices_tuple else default


def _optuna_trial_state_complete(optuna_module):
    """Return Optuna's COMPLETE trial state across old and new Optuna APIs."""
    trial_state = getattr(optuna_module, "TrialState", None)
    if trial_state is None:
        trial_ns = getattr(optuna_module, "trial", None)
        trial_state = getattr(trial_ns, "TrialState", None)
    if trial_state is None:
        return "COMPLETE"
    return trial_state.COMPLETE


def _safe_best_trial(study):
    try:
        return study.best_trial
    except Exception:
        return None




def _is_real_score_improvement(new_score: float | None, old_score: float | None, *, min_abs: float = 0.4, min_rel: float = 0.0) -> bool:
    """Require a real improvement before replacing/persisting the best config.

    Benchmark noise and float roundoff can produce identical displayed scores while
    `new_score > old_score` by a tiny epsilon. Treat those as ties so the tuner
    does not churn best configs or save non-improvements.
    """
    if new_score is None:
        return False
    if old_score is None:
        return True
    delta = float(new_score) - float(old_score)
    threshold = max(min_abs, abs(float(old_score)) * min_rel)
    if delta <= threshold + 1e-9:
        return False
    return True


def _score_improvement_percent(new_score: float, old_score: float) -> float:
    return ((float(new_score) - float(old_score)) / max(abs(float(old_score)), 0.001)) * 100.0


def score_performance(metrics: dict, requested_ctx: int, requested_gpus: int) -> float:
    if metrics.get("oom") or metrics.get("crash") or metrics.get("timeout"):
        return -1000.0

    # Unified raw/CORE/SPECULATIVE score: use the actual benchmark-wide token
    # throughput, i.e. (prompt tokens + generated tokens) / total measured time.
    # Prefill/decode are kept as diagnostics, but should not define different
    # rankings in different phases.
    score = _total_tokens_s(metrics) * 100.0
    score -= max(0, requested_gpus - 1) * 5.0
    return score


def score_server_performance(metrics: dict, requested_ctx: int, requested_gpus: int) -> float:
    if metrics.get("oom") or metrics.get("crash") or metrics.get("timeout"):
        return -1000.0

    requests_s = float(metrics.get("requests_s", 0.0))
    success_rate = float(metrics.get("server_success_rate", 1.0) or 0.0)
    latency_p95_s = float(metrics.get("server_latency_p95_s", 0.0) or 0.0)
    latency_p50_s = float(metrics.get("server_latency_p50_s", 0.0) or 0.0)

    # SERVER ranking must reflect real /v1/chat/completions behavior.  In real
    # traces prefill can be >100 t/s while decode is only 1-7 t/s, so total
    # prompt+decode throughput is dominated by prompt length and can select a
    # config that feels slow in production.  Rank decode first, total second,
    # and requests/latency as serving tie-breakers.
    decode_throughput = _metric_float(metrics, "decode_tokens_s")
    token_throughput = _total_tokens_s(metrics)
    throughput_score = (decode_throughput * 1000.0) + (token_throughput * 50.0)
    request_score = requests_s * 250.0
    latency_penalty = min(250.0, (max(0.0, latency_p95_s) * 2.0) + (max(0.0, latency_p50_s) * 1.0))
    score = ((throughput_score + request_score) * max(0.0, min(1.0, success_rate))) - latency_penalty
    score -= max(0, requested_gpus - 1) * 5.0
    return score


def read_gguf_metadata(model_path: str, default_ctx: int = 8192) -> dict:
    """Read GGUF metadata from file header, with fallback to defaults.
    
    Args:
        model_path: Path to the GGUF model file
        default_ctx: Default context size when GGUF cannot be read
    
    Returns:
        Dictionary with metadata, or defaults if reading fails
    """
    try:
        import gguf

        reader = gguf.GGUFReader(model_path)
        arch_field = reader.fields.get("general.architecture")
        arch = arch_field.parts[-1].tobytes().decode() if arch_field else "unknown"
        ctx_field = reader.fields.get(f"{arch}.context_length")
        ctx = int(ctx_field.parts[-1][0]) if ctx_field else default_ctx
        layer_field = reader.fields.get(f"{arch}.block_count")
        layers = int(layer_field.parts[-1][0]) if layer_field else 0
        moe_field = reader.fields.get(f"{arch}.expert_count")
        return {"architecture": arch, "trained_ctx": ctx, "layers": layers, "is_moe": moe_field is not None}
    except Exception as e:
        print(f"Warning: Could not read GGUF metadata directly ({e})")
        return {"architecture": "unknown", "trained_ctx": default_ctx, "layers": 0, "is_moe": False}


def get_server_path():
    return os.environ.get("LLAMA_SERVER_BIN", DEFAULT_LLAMA_SERVER)


def _server_library_paths(server_path: Path | str) -> list[str]:
    resolved = Path(server_path).expanduser().resolve()
    paths: list[str] = []

    # llama.cpp builds ship shared libs next to the resolved binary in build/bin.
    build_bin = resolved.parent
    if build_bin.is_dir():
        paths.append(str(build_bin))

    # Local superserver packaging keeps CUDA/NCCL runtime libs under the root
    # install directory, which is a few levels above the resolved binary.
    try:
        install_root = resolved.parents[3]
    except IndexError:
        install_root = None
    if install_root is not None:
        for rel in ("cuda/lib", "nccl/lib"):
            candidate = install_root / rel
            if candidate.is_dir():
                paths.append(str(candidate))
    user_runtime_root = Path.home() / ".local" / "opt" / "llamacpp-superserver"
    for rel in ("cuda/lib", "nccl/lib"):
        candidate = user_runtime_root / rel
        if candidate.is_dir():
            paths.append(str(candidate))

    return list(dict.fromkeys(paths))


def _prepare_server_env(server_path: Path | str) -> dict[str, str]:
    env = os.environ.copy()
    extra_paths = _server_library_paths(server_path)
    if extra_paths:
        current = env.get("LD_LIBRARY_PATH", "")
        merged_paths = extra_paths[:]
        if current:
            merged_paths.append(current)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(merged_paths)
    return env


def _validate_model_artifact(model_path: Path | str) -> tuple[bool, str]:
    path = Path(model_path)
    try:
        if not path.exists():
            return False, f"model file does not exist: {path}"
        if not path.is_file():
            return False, f"model path is not a file: {path}"
        if path.stat().st_size <= 0:
            return False, f"model file is empty: {path}"
    except Exception as exc:
        return False, f"could not validate model file {path}: {exc}"
    return True, "ok"





def _long_confirmation_prompt_tokens(ctx_size: int) -> int:
    # Fixed long test: 20K input tokens unless context is smaller.
    return max(256, min(LONG_CONFIRM_PROMPT_TOKENS, max(256, int(ctx_size) - 256)))


def _long_confirmation_predict_tokens(ctx_size: int) -> int:
    # Fixed long test: 20K output tokens unless context is too small to fit both.
    prompt_tokens = _long_confirmation_prompt_tokens(ctx_size)
    available = max(256, int(ctx_size) - prompt_tokens)
    return max(256, min(LONG_CONFIRM_PREDICT_TOKENS, available))

def _early_accept_settings(prompt_variant: str) -> dict[str, float]:
    """Partial-decode accept thresholds by workload type."""
    if prompt_variant == "ctx_half":
        return {
            "min_tokens": 512.0,
            "min_predicted_ms": 5000.0,
            "relative_gain": 1.03,
        }
    return {
        "min_tokens": float("inf"),
        "min_predicted_ms": float("inf"),
        "relative_gain": float("inf"),
    }


def _early_prune_settings(prompt_variant: str) -> dict[str, float]:
    """Conservative partial-decode pruning thresholds by workload type."""
    if prompt_variant == "ctx_half":
        return {
            "min_tokens": 256.0,
            "min_predicted_ms": 3000.0,
            "relative_floor": 0.92,
        }
    return {
        "min_tokens": 20.0,
        "min_predicted_ms": 500.0,
        "relative_floor": 0.80,
    }


def _make_completion_prompt(variant: str = "benchmark", target_prompt_tokens: int | None = None) -> str:
    # Use distinct prompts so the internal probe and the actual benchmark do
    # not share the exact same text while still exercising the same model path.
    if variant == "probe":
        return (
            "Resume en una sola frase qué hace un scheduler de inferencia en un servidor LLM. "
            "Responde solo con la frase final."
        )

    if variant == "ctx_half":
        # Long confirmation prompt: fixed 20K input target (capped for small ctx)
        # and ask the model to repeat it, encouraging a long output target.
        target = max(256, int(target_prompt_tokens or 2048))
        seed = (
            "BLOQUE-DE-PRUEBA-DE-CONTEXTO: Este texto existe solo para medir rendimiento de "
            "prefill y decode en una ventana larga. Repite literalmente el bloque de entrada "
            "hasta alcanzar la longitud solicitada. "
        )
        approx_tokens_per_seed = max(1, len(seed) // 4)
        repetitions = max(1, target // approx_tokens_per_seed)
        payload = (seed * repetitions)[: max(len(seed), target * 5)]
        return (
            "Instrucciones: repite literalmente todo el BLOQUE-DE-PRUEBA-DE-CONTEXTO que sigue, "
            "sin resumir, sin explicar y manteniendo el orden.\n\n"
            + payload
            + "\n\nRespuesta: repite literalmente el bloque anterior hasta agotar el presupuesto de salida."
        )

    if variant == "server_short":
        return (
            "Clasifica en una frase el objetivo de un benchmark HTTP para un servidor LLM. "
            "Respuesta breve y determinista."
        )
    if variant == "server_medium":
        return (
            "Explica de forma técnica cómo influyen parallel, continuous batching y KV cache "
            "en la latencia p95 de un servidor LLM. Responde en pasos numerados."
        )
    if variant == "server_long":
        chunk = (
            "Entrada de carga concurrente: medir un servidor requiere prompts variados, "
            "colas simultáneas, prefill y decode sostenido. "
        )
        return (
            "Resume y continúa el siguiente bloque técnico manteniendo el estilo. "
            "No uses listas vacías.\n\n" + (chunk * 24)
        )

    # Benchmark prompt: deterministic, focused, and long enough to measure
    # both prefill and decode throughput in a stable way.
    base = (
        "Escribe una explicación técnica concisa y determinista sobre el tema, "
        "con ejemplos solo si ayudan a la explicación. "
    )
    repeat_instr = (
        "\n\nPor favor, continúa exactamente la continuación necesaria y repite la parte siguiente "
        "para completar la longitud de respuesta solicitada. Responde solo con la continuación.")
    return (base * 4) + repeat_instr


def _format_config_diff(current: dict, reference: dict) -> str:
    diff = {key: value for key, value in current.items() if reference.get(key) != value}
    return json.dumps(diff, ensure_ascii=False, sort_keys=True) if diff else "{}"


def _format_config_diff_notice(current: dict, reference: dict, raw_current: dict | None = None) -> str:
    """Format the net diff and explain when repairs collapse raw changes back to the reference."""
    net_diff = _format_config_diff(current, reference)
    if net_diff != "{}":
        return net_diff
    if raw_current is None:
        return "{}"
    raw_diff = _format_config_diff(raw_current, reference)
    if raw_diff != "{}":
        return f"{{}} (raw trial changes normalized away: {raw_diff})"
    return "{}"


def _benchmark_params_equivalent(current: dict, reference: dict, *, api_mode: bool = False) -> bool:
    """Return True when two configs map to the same benchmark command parameters."""
    return _prepare_benchmark_params(current, api_mode=api_mode) == _prepare_benchmark_params(reference, api_mode=api_mode)


def _find_free_local_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except OSError:
        # Some test/sandbox environments disallow socket creation. Avoid the
        # historic fixed port anyway to reduce collisions with existing servers.
        return random.randint(20000, 60999)


def _build_benchmark_command(
    model_path: Path | str,
    params: dict,
    ctx_size: int,
    *,
    server_path: Path | str | None = None,
    port: int | str = 18081,
) -> list[str]:
    model_path_obj = Path(model_path)
    source_params = dict(params)
    overrides = normalize_server_overrides(source_params)
    model_n_gpu_layers = _map_n_gpu_layers(source_params.get("n_gpu_layers", 999))
    if "n_gpu_layers_draft" in source_params:
        draft_layers = _map_n_gpu_layers(source_params.get("n_gpu_layers_draft", 999))
        overrides["n_gpu_layers_draft"] = draft_layers if isinstance(draft_layers, int) else 999
    if "n_gpu_layers" in source_params:
        overrides.pop("n_gpu_layers", None)
    model = ManagedModel(
        model_id=model_path_obj.stem or "auto-performance",
        repo_id="auto-performance/benchmark",
        quant=None,
        filename=model_path_obj.name,
        local_path=str(model_path_obj),
        mmproj_filename=None,
        mmproj_path=None,
        load_capabilities=[],
        aliases=[],
        ctx_size=int(ctx_size),
        n_gpu_layers=int(model_n_gpu_layers) if isinstance(model_n_gpu_layers, int) else 999,
        tensor_split=str(overrides.get("tensor_split", "1")),
        host=str(overrides.get("host", "127.0.0.1")),
        jinja=False,
        description="auto-performance benchmark harness",
        speculative=bool(overrides.get("model_draft")),
        spec_variant_of=None,
        spec_meta={},
        auto_ctx_failed=False,
        auto_ctx_error="",
        ctx_probe_read_s=None,
        ctx_probe_tokens_s=None,
        ctx_probe_totals_s=None,
        ctx_probe_latency_ms=None,
        ctx_probe_speed_tps=None,
        ctx_probe_kv_gb=None,
        ctx_probe_prompt_tokens=None,
        server_overrides=overrides,
    )
    return build_llama_server_command(model, Path(server_path or DEFAULT_LLAMA_SERVER), port=str(port), include_jinja=False)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((percentile / 100.0) * (len(ordered) - 1)))))
    return float(ordered[index])


def _server_benchmark_shape(params: dict, requested_concurrency: int = 1, requested_requests: int = 1) -> tuple[int, int]:
    """Return a representative server workload shape.

    SERVER tuning is about queueing/slot behavior, not single-request raw TPS.
    If the caller did not explicitly request a load shape, derive one from the
    candidate's `parallel` setting so every server trial exercises concurrency.
    """
    try:
        parallel = max(1, int(params.get("parallel", 1) or 1))
    except Exception:
        parallel = 1
    concurrency = max(int(requested_concurrency or 1), min(8, max(2, parallel)))
    requests = max(int(requested_requests or 1), concurrency * 3)
    return concurrency, requests


def _speculative_candidate_sequence(base: dict) -> list[dict]:
    """Ordered speculative configs for feasibility preflight.

    The first candidate preserves the baseline/core draft settings exactly as a
    reference-only candidate. It must not be benchmarked again; if it is already
    feasible, only changed variants are eligible for enqueueing.
    """
    model_draft = base.get("model_draft")
    if not model_draft:
        return []
    candidates: list[dict] = []
    seen: set[tuple] = set()
    baseline_cfg = dict(base)
    baseline_cfg["model_draft"] = model_draft
    try:
        baseline_cfg["draft"] = int(baseline_cfg.get("draft") or 8)
    except Exception:
        baseline_cfg["draft"] = 8
    try:
        baseline_cfg["ctx_size_draft"] = int(baseline_cfg.get("ctx_size_draft") or 1024)
    except Exception:
        baseline_cfg["ctx_size_draft"] = 1024
    baseline_cfg["n_gpu_layers_draft"] = baseline_cfg.get("n_gpu_layers_draft") or "auto"
    baseline_key = (
        baseline_cfg["draft"],
        baseline_cfg["ctx_size_draft"],
        baseline_cfg["n_gpu_layers_draft"],
    )
    seen.add(baseline_key)
    candidates.append(baseline_cfg)
    for layers in ("auto", "all"):
        for draft_ctx in (512, 768, 1024, 2048, 4096):
            for draft_n in (4, 8, 16, 32):
                cfg = dict(base)
                cfg["model_draft"] = model_draft
                cfg["draft"] = int(draft_n)
                cfg["ctx_size_draft"] = int(draft_ctx)
                cfg["n_gpu_layers_draft"] = layers
                key = (cfg["draft"], cfg["ctx_size_draft"], cfg["n_gpu_layers_draft"])
                if key not in seen:
                    seen.add(key)
                    candidates.append(cfg)
    return candidates


def _speculative_knob_key(params: dict) -> tuple:
    return (
        int(params.get("draft") or 0),
        int(params.get("ctx_size_draft") or 0),
        str(params.get("n_gpu_layers_draft") or ""),
    )


def _speculative_candidate_not_heavier_than_reference(candidate: dict, reference: dict) -> bool:
    """Conservative check for fallback candidates after a real baseline succeeded."""
    try:
        cand_draft = int(candidate.get("draft") or 0)
        ref_draft = int(reference.get("draft") or 0)
        cand_ctx = int(candidate.get("ctx_size_draft") or 0)
        ref_ctx = int(reference.get("ctx_size_draft") or 0)
    except Exception:
        return False
    if cand_draft > ref_draft or cand_ctx > ref_ctx:
        return False
    ref_layers = str(reference.get("n_gpu_layers_draft") or "").lower()
    cand_layers = str(candidate.get("n_gpu_layers_draft") or "").lower()
    if ref_layers == "auto" and cand_layers == "all":
        return False
    return True


def _speculative_ctx_descent_values(ctx_size: int, baseline_ctx_draft: int | None = None) -> list[int]:
    """Descending draft contexts for deterministic speculative tuning.

    Draft context is intentionally not optimized with Optuna because each failed
    draft load is expensive. Start at half of the main context and step down by
    exact /2 jumps; the first candidate that preserves/improves prefill is
    therefore the largest useful draft context found.
    """
    values: list[int] = []
    max_draft_ctx = max(512, int(ctx_size) // 2)

    def add(value: int) -> None:
        value = max(512, int(value))
        # llama.cpp context values are cleaner and less duplicate-prone on 512
        # token boundaries.
        value = int(round(value / 512.0) * 512)
        value = max(512, min(max_draft_ctx, value))
        if value not in values:
            values.append(value)

    start = max_draft_ctx
    add(start)
    current = start
    while current > 512:
        current = max(512, current // 2)
        add(current)

    if baseline_ctx_draft:
        add(int(baseline_ctx_draft))

    for value in (65536, 32768, 20000, 16384, 8192, 4096, 2048, 1024, 512):
        if value <= start:
            add(value)
    return sorted(values, reverse=True)


def _validate_server_cmd_args(cmd: list[str], log_path: Path | None = None) -> tuple[bool, str]:
    """Lightweight validation of server command arguments to catch formatting issues.

    Returns (ok, reason). If ok is False, reason contains a short explanation.
    """
    try:
        if "--tensor-split" in cmd:
            try:
                ts_idx = cmd.index("--tensor-split")
                ts_val = cmd[ts_idx + 1]
            except Exception:
                return False, "missing-tensor-split-value"
            parts = [p.strip() for p in ts_val.split(",") if p.strip()]
            if not parts:
                return False, "empty-tensor-split"
            for p in parts:
                try:
                    float(p)
                except Exception:
                    return False, f"invalid-tensor-split-token:{p}"

        # Validate numeric tokens for known flags that map to floats/ints
        numeric_flags = {"--batch-size", "--ubatch-size", "--fit-target", "--ctx-size", "--n-gpu-layers", "--ctx-size-draft"}
        for flag in numeric_flags:
            if flag in cmd:
                try:
                    idx = cmd.index(flag)
                    val = cmd[idx + 1]
                    # Accept numeric or simple numeric-like strings
                    float(val)
                except Exception:
                    return False, f"invalid-value-for-{flag}"
    except Exception as e:
        return False, f"validation-exception:{e}"
    return True, "ok"
def run_benchmark(
    model_path: Path | str,
    params: dict,
    ctx_size: int,
    gpu_set: list[int] | None = None,
    mock: bool = False,
    api_mode: bool = False,
    load_concurrency: int = 1,
    load_requests: int = 1,
    n_predict: int = 128,
    runs: int = 1,
    adaptive_fill: bool = False,
    confirmation_long_context: bool = False,
    confirm_long_context_if_promising: bool = False,
    confirm_decode_tps_floor: float = 0.0,
    log_path: Path | None = None,
    probe_only: bool = False,
    max_total_s: float | None = None,
    best_tps: float = 0.0,
    server_ready_timeout_s: float = 300,  # 5 minutes for loading large models
    expected_models_loaded: int = 1,
) -> dict:

    metrics = {
        "prefill_tokens_s": 0.0,
        "decode_tokens_s": 0.0,
        "total_tokens_s": 0.0,
        "vram_used": 0.0,
        "ram_used": 0.0,
        "load_ready_s": 0.0,
        "oom": False,
        "timeout": False,
        "crash": False,
        "ctx_stable": ctx_size,
        "load_reason": "",
        "gpu_load_baseline_mib": 0,
        "gpu_load_peak_mib": 0,
        "gpu_load_peak_delta_mib": 0,
        "expected_models_loaded": max(1, int(expected_models_loaded)),
    }

    # Validate parameters BEFORE mock to catch errors early in all code paths
    active_keys = {
        *PROBED_TUNER_KEYS,
        "ctx_size",
        "ts_strategy",
        "direct_io",  # Stage 0 I/O probe parameter (not part of main tuning)
    }
    if api_mode:
        active_keys.update(PHASE2_SERVER_TUNER_KEYS)
    if any(key.endswith("_draft") or key in {"model_draft", "hf_repo_draft"} for key in params):
        active_keys.update(PHASE1_SPECULATIVE_TUNER_KEYS)
    _validate_tuning_params(params, active_keys, label="auto-performance params")

    model_path_obj = Path(model_path)
    model_ok, model_reason = _validate_model_artifact(model_path_obj)
    if not model_ok:
        metrics["crash"] = True
        metrics["error"] = model_reason
        metrics["load_reason"] = "invalid-model"
        _append_auto_perf_log(log_path, f"MODEL_INVALID {model_reason}")
        return metrics
    gpu_indices = list(gpu_set or [])

    if mock:
        gpus_count = len(gpu_indices)
        print(f"    [MOCK] Running with {gpus_count} GPUs ({gpu_set}), params={params}, probe={probe_only}")
        _append_auto_perf_log(log_path, f"[MOCK] gpus={gpu_set} params={json.dumps(params, ensure_ascii=False, sort_keys=True)} probe={probe_only}")
        time.sleep(0.05 if probe_only else 0.1)
        ub = int(params.get("ubatch_size", 512)) or 512
        base = max(1.0, (gpus_count or 1) * (2048.0 / ub))
        metrics["decode_tokens_s"] = 1000.0 * base * (0.9 if params.get("kv_offload") else 1.0)
        metrics["prefill_tokens_s"] = metrics["decode_tokens_s"] * 0.45
        metrics["total_tokens_s"] = metrics["prefill_tokens_s"] + metrics["decode_tokens_s"]
        metrics["load_ready_s"] = 0.9 - (0.25 if params.get("direct_io") else 0.0) - (0.05 if params.get("numa") else 0.0)
        if ub > 2048 and random.random() < 0.25:
            metrics["oom"] = True
        elif ub > 4096 and random.random() < 0.15:
            metrics["crash"] = True
        elif random.random() < 0.02:
            metrics["timeout"] = True
        return metrics

    active_keys = {
        *PROBED_TUNER_KEYS,
        "ctx_size",
        "ts_strategy",
        "direct_io",  # Stage 0 I/O probe parameter (not part of main tuning)
    }
    if api_mode:
        active_keys.update(PHASE2_SERVER_TUNER_KEYS)
    if any(key.endswith("_draft") or key in {"model_draft", "hf_repo_draft"} for key in params):
        active_keys.update(PHASE1_SPECULATIVE_TUNER_KEYS)
    _validate_tuning_params(params, active_keys, label="auto-performance params")

    server_bin = get_server_path()
    env = _prepare_server_env(server_bin)
    port = _find_free_local_port()
    # Note: We avoid CUDA_VISIBLE_DEVICES here as per user preference, 
    # relying on --tensor-split and --main-gpu to isolate the workload.

    cmd = _build_benchmark_command(model_path_obj, {**params, "ctx_size": int(ctx_size)}, ctx_size, server_path=server_bin, port=port)


    cmd_text = " ".join(cmd)
    print(f"    Starting server (loading model into VRAM)...")
    _append_auto_perf_log(log_path, f"COMMAND {cmd_text}")

    # Pre-flight validation to catch malformed flags before invoking llama-server
    ok, reason = _validate_server_cmd_args(cmd, log_path=log_path)
    if not ok:
        _append_auto_perf_log(log_path, f"SERVER_LAUNCH_INVALID_ARGS reason={reason} cmd={cmd_text}")
        print(f"    ⚠️  Invalid server launch arguments: {reason}")
        metrics["crash"] = True
        metrics["load_reason"] = "invalid-args"
        return metrics

    gpu_before = _sample_gpu_memory(gpu_indices) if gpu_set is not None else {}
    try:
        import psutil

        mem_before = psutil.virtual_memory().used
    except Exception:
        mem_before = None

    server_proc = None
    server_proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        ready, load_reason, elapsed_total, gpu_load_baseline, gpu_load_peak = _wait_for_model_load(
            server_proc,
            gpu_indices,
            health_url=f"http://127.0.0.1:{port}/health",
            log_path=log_path,
            server_ready_timeout_s=server_ready_timeout_s,
        )
        metrics["load_reason"] = str(load_reason)
        metrics["gpu_load_baseline_mib"] = int(gpu_load_baseline)
        metrics["gpu_load_peak_mib"] = int(gpu_load_peak)
        metrics["gpu_load_peak_delta_mib"] = max(0, int(gpu_load_peak) - int(gpu_load_baseline))

        if not ready:
            if load_reason == "exited-before-health":
                metrics["crash"] = True
            else:
                metrics["timeout"] = True
            # Capture remaining server output and exit status for diagnostics.
            exit_code = None
            stdout_capture = ""
            try:
                exit_code = server_proc.poll()
            except Exception:
                exit_code = None

            try:
                # Try a short communicate to collect any final output without waiting long.
                out, _ = server_proc.communicate(timeout=1)
                stdout_capture = out or ""
            except subprocess.TimeoutExpired:
                # Process still running — we avoid blocking further.
                try:
                    if server_proc and server_proc.stdout:
                        # Attempt a non-blocking read of what's available (best-effort).
                        stdout_capture = server_proc.stdout.read() or ""
                except Exception:
                    stdout_capture = ""
            except Exception:
                stdout_capture = ""

            # Derive a short hint from output / exit status
            sig_name = ""
            try:
                if exit_code is not None and exit_code < 0:
                    sig_name = signal.Signals(-exit_code).name
            except Exception:
                sig_name = f"signal_{-exit_code}" if exit_code is not None and exit_code < 0 else ""

            hint = ""
            low = (stdout_capture or "").lower()
            if any(k in low for k in ("out of memory", "oom", "bad_alloc", "std::bad_alloc")):
                hint = "oom-output"
                metrics["oom"] = True
            elif any(k in low for k in ("segmentation fault", "segfault", "stack overflow", "illegal instruction")):
                hint = "crash-segfault"
            elif exit_code is not None and exit_code != 0:
                hint = "non-zero-exit"

            # Try to surface recent dmesg messages mentioning OOM as an additional hint (best-effort)
            dmesg_snip = ""
            try:
                dmesg = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=3)
                dlow = (dmesg.stdout or "").lower()
                if "oom" in dlow or "out of memory" in dlow:
                    hint = (hint + ";oom-dmesg") if hint else "oom-dmesg"
                    dlines = [l for l in (dmesg.stdout or "").splitlines() if "oom" in l.lower() or "out of memory" in l.lower()]
                    dmesg_snip = "\n".join(dlines[-20:])
            except Exception:
                dmesg_snip = ""

            _append_auto_perf_log(
                log_path,
                f"SERVER_LOAD_UNSTABLE reason={load_reason} elapsed={elapsed_total:.1f}s exitcode={exit_code} signal={sig_name} hint={hint}",
            )

            if stdout_capture:
                metrics["server_output_tail"] = _tail_text(stdout_capture)
                _append_text_block(log_path, "SERVER_STDOUT_BEGIN", metrics["server_output_tail"], "SERVER_STDOUT_END")

            current_error = str(metrics.get("error") or "")
            metrics["error"] = _extract_descriptive_error(
                str(metrics.get("server_output_tail") or stdout_capture or ""),
                fallback=current_error or f"server load did not stabilize ({load_reason})",
            )

            if dmesg_snip:
                _append_auto_perf_log(log_path, "DMESG_BEGIN")
                for line in dmesg_snip.splitlines():
                    _append_auto_perf_log(log_path, line)
                _append_auto_perf_log(log_path, "DMESG_END")

            print(f"    ⚠️  Server load did not stabilize ({load_reason}) after {elapsed_total:.1f}s")
            return metrics

        metrics["load_ready_s"] = elapsed_total
        print(f"    ✓ Server ready in {_metric_float(metrics, 'load_ready_s'):.2f}s")

        if probe_only:
            print(f"    ✓ Probe stage completed successfully.")
            return metrics

        if metrics["crash"]:
            out = server_proc.stdout.read() if server_proc.stdout else ""
            if "out of memory" in out.lower() or "oom" in out.lower() or "bad_alloc" in out.lower():
                metrics["oom"] = True
                _append_auto_perf_log(log_path, "SERVER_EXIT reason=oom")
            else:
                _append_auto_perf_log(log_path, "SERVER_EXIT reason=crash")
            if out:
                metrics["server_output_tail"] = _tail_text(out)
                _append_text_block(log_path, "SERVER_STDOUT_BEGIN", metrics["server_output_tail"], "SERVER_STDOUT_END")
            return metrics

        # requests and json are already imported at module level.  Raw
        # CORE/SPECULATIVE still uses llama.cpp's native /completion endpoint,
        # but SERVER mode must exercise the production OpenAI-compatible chat
        # path; otherwise the selected params can benchmark well and still feel
        # slow in real /v1/chat/completions traffic.
        request_url = f"http://127.0.0.1:{port}/completion"
        chat_request_url = f"http://127.0.0.1:{port}/v1/chat/completions"

        def _extract_chat_token_counts(data: dict, fallback_text: str = "") -> tuple[float, float]:
            timings = data.get("timings") if isinstance(data, dict) else {}
            if isinstance(timings, dict):
                prompt_n = float(timings.get("prompt_n", 0) or 0)
                predicted_n = float(timings.get("predicted_n", 0) or 0)
                if prompt_n > 0 or predicted_n > 0:
                    return prompt_n, predicted_n
            usage = data.get("usage") if isinstance(data, dict) else {}
            if isinstance(usage, dict):
                prompt_n = float(usage.get("prompt_tokens", 0) or 0)
                predicted_n = float(usage.get("completion_tokens", usage.get("predicted_tokens", 0)) or 0)
                if prompt_n > 0 or predicted_n > 0:
                    return prompt_n, predicted_n
            return 0.0, float(max(0, len(fallback_text.split())))

        def _single_request(seed_offset: int, n_pred: int, best_tps: float = 0.0, prompt_variant: str = "benchmark", target_prompt_tokens: int | None = None) -> tuple[float, float, float]:
            """Perform a single completion request with pruning based on server timings."""
            tokens_count = 0
            request_timeout_s = max(300.0, float(max_total_s or 300.0))

            if not api_mode:
                prompt_text = _make_completion_prompt(prompt_variant, target_prompt_tokens=target_prompt_tokens)
                started = time.perf_counter()
                response = requests.post(
                    chat_request_url,
                    json={
                        "model": "auto-performance",
                        "messages": [{"role": "user", "content": prompt_text}],
                        "max_tokens": int(n_pred),
                        "stream": False,
                        "temperature": 0.0,
                        "seed": 42 + seed_offset,
                    },
                    timeout=request_timeout_s,
                )
                if response.status_code != 200:
                    raise RuntimeError(f"Chat completion request failed with status {response.status_code}: {response.text[:300]}")
                data = response.json()
                fallback_text = ""
                try:
                    choice0 = (data.get("choices") or [{}])[0]
                    message = choice0.get("message") if isinstance(choice0, dict) else {}
                    if isinstance(message, dict):
                        fallback_text = str(message.get("content") or "")
                except Exception:
                    fallback_text = ""
                prompt_n, predicted_n = _extract_chat_token_counts(data, fallback_text)
                timings = data.get("timings") if isinstance(data, dict) else {}
                if isinstance(timings, dict):
                    prompt_ms = float(timings.get("prompt_ms", 0) or 0)
                    predicted_ms = float(timings.get("predicted_ms", 0) or 0)
                    elapsed = (prompt_ms + predicted_ms) / 1000.0 if (prompt_ms + predicted_ms) > 0 else 0.0
                else:
                    elapsed = 0.0
                if elapsed <= 0.0:
                    elapsed = max(time.perf_counter() - started, 1e-6)
                if best_tps > 0 and predicted_n > 0:
                    gen_tps = predicted_n / max(elapsed, 1e-6)
                    floor = float(_early_prune_settings(prompt_variant)["relative_floor"])
                    if gen_tps < (best_tps * floor):
                        workload = "long-confirmation" if prompt_variant == "ctx_half" else "screening"
                        raise TimeoutError(
                            f"Early pruning ({workload}, chat): {gen_tps:.2f} t/s < "
                            f"{floor:.2f}*{best_tps:.2f} t/s after {predicted_n:.0f} tokens"
                        )
                return elapsed, prompt_n, predicted_n
            
            try:
                response = requests.post(
                    request_url,
                    json={
                        "prompt": _make_completion_prompt(prompt_variant, target_prompt_tokens=target_prompt_tokens),
                        "n_predict": int(n_pred),
                        "stream": True,
                        "temperature": 0.0,
                        "seed": 42 + seed_offset,
                    },
                    stream=True,
                    timeout=request_timeout_s,
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"Completion request failed with status {response.status_code}")
                
                final_timings = {}
                for line in response.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data = json.loads(line_str[6:])
                        if "content" in data:
                            tokens_count += 1
                            
                            # Early pruning: check predicted throughput (generation speed only)
                            # Use server-reported timings, not wall-clock time. The long 20K/20K
                            # confirmation gets a larger warmup and a stricter floor so we do
                            # not wait for all output tokens when it is clearly worse.
                            prune_cfg = _early_prune_settings(prompt_variant)
                            if tokens_count >= prune_cfg["min_tokens"]:
                                timings = data.get("timings", {})
                                predicted_ms = max(float(timings.get("predicted_ms", 1.0)), 1.0)
                                predicted_n = float(timings.get("predicted_n", tokens_count))
                                
                                if predicted_n >= prune_cfg["min_tokens"] and predicted_ms >= prune_cfg["min_predicted_ms"]:
                                    gen_tps = (predicted_n / (predicted_ms / 1000.0)) if predicted_ms > 0 else 0.0
                                    floor = float(prune_cfg["relative_floor"])
                                    if best_tps > 0 and gen_tps < (best_tps * floor):
                                        workload = "long-confirmation" if prompt_variant == "ctx_half" else "screening"
                                        raise TimeoutError(
                                            f"Early pruning ({workload}): {gen_tps:.2f} t/s < "
                                            f"{floor:.2f}*{best_tps:.2f} t/s after {predicted_n:.0f} tokens"
                                        )
                                    accept_cfg = _early_accept_settings(prompt_variant)
                                    if (
                                        best_tps > 0
                                        and predicted_n >= accept_cfg["min_tokens"]
                                        and predicted_ms >= accept_cfg["min_predicted_ms"]
                                        and gen_tps >= (best_tps * float(accept_cfg["relative_gain"]))
                                    ):
                                        final_timings = timings
                                        metrics["early_accepted"] = True
                                        metrics["early_accept_tokens"] = predicted_n
                                        metrics["early_accept_decode_tps"] = gen_tps
                                        metrics["early_accept_gain"] = gen_tps / max(best_tps, 1e-6)
                                        _append_auto_perf_log(
                                            log_path,
                                            f"EARLY_ACCEPT long-confirmation gen_tps={gen_tps:.2f} best_tps={best_tps:.2f} tokens={predicted_n:.0f}",
                                        )
                                        break
                        
                        if data.get("stop"):
                            final_timings = data.get("timings", {})
                            break
                
                prompt_n = float(final_timings.get("prompt_n", 0))
                predicted_n = float(final_timings.get("predicted_n", tokens_count))
                prompt_ms = max(float(final_timings.get("prompt_ms", 1)), 1.0)
                predicted_ms = max(float(final_timings.get("predicted_ms", 1)), 1.0)
                total_elapsed = max((prompt_ms + predicted_ms) / 1000.0, 1e-6)
                
                return total_elapsed, prompt_n, predicted_n

            except (requests.exceptions.RequestException, TimeoutError, RuntimeError) as e:
                metrics["error"] = str(e)
                _append_auto_perf_log(log_path, f"REQUEST_ERROR {str(e)}")
                raise

        # Reuse the same loaded server for a distinct probe prompt before the
        # real benchmark loop. This avoids paying a second model load cost.
        try:
            probe_n_predict = max(8, min(64, int(n_predict) // 4))
            probe_elapsed, probe_prompt_n, probe_predicted_n = _single_request(
                9999,
                probe_n_predict,
                best_tps=0.0,
                prompt_variant="probe",
            )
            metrics["probe_elapsed_s"] = probe_elapsed
            metrics["probe_prompt_tokens"] = probe_prompt_n
            metrics["probe_predicted_tokens"] = probe_predicted_n
            print(f"    ✓ Probe stage completed successfully.")
            _append_auto_perf_log(
                log_path,
                "PROBE_OK "
                + json.dumps(
                    {
                        "elapsed_s": probe_elapsed,
                        "prompt_tokens": probe_prompt_n,
                        "predicted_tokens": probe_predicted_n,
                        "prompt_variant": "probe",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
        except Exception as probe_exc:
            metrics["crash"] = True
            metrics["error"] = str(probe_exc)
            _append_auto_perf_log(log_path, f"PROBE_ERROR {type(probe_exc).__name__}: {probe_exc}")
            return metrics

        if api_mode:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            effective_concurrency, effective_requests = _server_benchmark_shape(params, load_concurrency, load_requests)
            print(
                f"    Server load benchmark: concurrency={effective_concurrency}, "
                f"requests={effective_requests}, n_predict={int(n_predict)}"
            )
            load_started = time.perf_counter()
            request_results = []
            request_errors = []
            prompt_variants = ("server_short", "server_medium", "server_long", "benchmark")

            def _post_completion_blocking(seed_offset: int) -> dict:
                req_started = time.perf_counter()
                variant = prompt_variants[seed_offset % len(prompt_variants)]
                prompt_text = _make_completion_prompt(variant)
                res = requests.post(
                    chat_request_url,
                    json={
                        "model": "auto-performance",
                        "messages": [{"role": "user", "content": prompt_text}],
                        "max_tokens": int(n_predict),
                        "stream": False,
                        "temperature": 0.0,
                        "seed": 42 + seed_offset,
                    },
                    timeout=max(60.0, float(max_total_s or 300.0)),
                )
                res.raise_for_status()
                data = res.json()
                latency_s = max(time.perf_counter() - req_started, 1e-6)
                fallback_text = ""
                try:
                    choice0 = (data.get("choices") or [{}])[0]
                    message = choice0.get("message") if isinstance(choice0, dict) else {}
                    if isinstance(message, dict):
                        fallback_text = str(message.get("content") or "")
                except Exception:
                    fallback_text = ""
                prompt_n, predicted_n = _extract_chat_token_counts(data, fallback_text)
                return {
                    "prompt_n": prompt_n,
                    "predicted_n": predicted_n,
                    "latency_s": latency_s,
                    "variant": variant,
                    "decode_tokens_s": predicted_n / latency_s if latency_s > 0 else 0.0,
                }

            try:
                with ThreadPoolExecutor(max_workers=effective_concurrency) as pool:
                    futures = [pool.submit(_post_completion_blocking, idx) for idx in range(effective_requests)]
                    for future in as_completed(futures):
                        try:
                            request_results.append(future.result())
                        except Exception as req_exc:
                            request_errors.append(str(req_exc))
                        if max_total_s and (time.perf_counter() - load_started) > max_total_s:
                            metrics["timeout"] = True
                            break
            except Exception as e:
                metrics["crash"] = True
                error_msg = f"API concurrent load failed: {type(e).__name__}: {str(e)}"
                _append_auto_perf_log(log_path, f"API_LOAD_ERROR {error_msg}")
                print(f"    ⚠️  {error_msg}")
                return metrics

            if not request_results:
                metrics["crash"] = True
                metrics["error"] = "API concurrent load completed zero requests"
                return metrics

            load_elapsed = max(time.perf_counter() - load_started, 1e-6)
            prompt_total = sum(item["prompt_n"] for item in request_results)
            predicted_total = sum(item["predicted_n"] for item in request_results)
            latencies = [float(item["latency_s"]) for item in request_results]
            completed = len(request_results)
            success_rate = completed / max(1, effective_requests)
            
            metrics["prefill_tokens_s"] = prompt_total / load_elapsed
            metrics["decode_tokens_s"] = predicted_total / load_elapsed
            metrics["total_tokens_s"] = (prompt_total + predicted_total) / load_elapsed
            metrics["requests_s"] = completed / load_elapsed
            metrics["prompt_tokens"] = prompt_total
            metrics["predicted_tokens"] = predicted_total
            metrics["server_concurrency"] = effective_concurrency
            metrics["server_requests_target"] = effective_requests
            metrics["server_requests_completed"] = completed
            metrics["server_success_rate"] = success_rate
            metrics["server_latency_p50_s"] = _percentile(latencies, 50)
            metrics["server_latency_p95_s"] = _percentile(latencies, 95)
            metrics["server_wall_s"] = load_elapsed
            metrics["server_prompt_variants"] = sorted(set(str(item.get("variant")) for item in request_results))
            if request_errors:
                metrics["server_request_errors"] = request_errors[:3]
            print(
                f"    ✓ Server load: {completed}/{effective_requests} ok | "
                f"{metrics['requests_s']:.2f} req/s | p50={metrics['server_latency_p50_s']:.2f}s "
                f"p95={metrics['server_latency_p95_s']:.2f}s"
            )
            print(
                f"    ✓ Aggregate Prefill: {metrics['prefill_tokens_s']:.1f} t/s | "
                f"Decode: {metrics['decode_tokens_s']:.1f} t/s | Total: {metrics['total_tokens_s']:.1f} t/s"
            )
            _append_auto_perf_log(
                log_path,
                "SERVER_LOAD_OK "
                + json.dumps(
                    {
                        "concurrency": effective_concurrency,
                        "requests_target": effective_requests,
                        "requests_completed": completed,
                        "requests_s": metrics["requests_s"],
                        "latency_p50_s": metrics["server_latency_p50_s"],
                        "latency_p95_s": metrics["server_latency_p95_s"],
                        "success_rate": success_rate,
                        "prompt_variants": metrics["server_prompt_variants"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
            
        else:
            total_elapsed = 0.0
            total_prompt_tokens = 0.0
            total_predicted_tokens = 0.0
            any_fail = False

            def _record_success_metrics(prompt_total: float, predicted_total: float, elapsed_total: float, label: str = "BENCHMARK_OK") -> None:
                metrics["prefill_tokens_s"] = prompt_total / max(elapsed_total, 1e-6)
                metrics["decode_tokens_s"] = predicted_total / max(elapsed_total, 1e-6)
                metrics["total_tokens_s"] = (prompt_total + predicted_total) / max(elapsed_total, 1e-6)
                metrics["prompt_tokens"] = prompt_total
                metrics["predicted_tokens"] = predicted_total
                print(
                    f"    ✓ Prefill: {metrics['prefill_tokens_s']:.1f} t/s | "
                    f"Decode: {metrics['decode_tokens_s']:.1f} t/s | Total: {metrics['total_tokens_s']:.1f} t/s"
                )
                print(f"      (Prefill tokens: {prompt_total} | Generation tokens: {predicted_total})")
                _append_auto_perf_log(
                    log_path,
                    label
                    + " "
                    + json.dumps(
                        {
                            "prefill_tokens_s": metrics["prefill_tokens_s"],
                            "decode_tokens_s": metrics["decode_tokens_s"],
                            "total_tokens_s": metrics["total_tokens_s"],
                            "prompt_tokens": prompt_total,
                            "predicted_tokens": predicted_total,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )

            try:
                for r in range(max(1, int(runs))):
                    try:
                        prompt_variant = "ctx_half" if confirmation_long_context else "benchmark"
                        target_prompt_tokens = _long_confirmation_prompt_tokens(ctx_size) if confirmation_long_context else None
                        elapsed, prompt_n, predicted_n = _single_request(
                            r,
                            int(n_predict),
                            best_tps=best_tps,
                            prompt_variant=prompt_variant,
                            target_prompt_tokens=target_prompt_tokens,
                        )
                    except Exception as e:
                        any_fail = True
                        if isinstance(e, TimeoutError):
                            metrics["timeout"] = True
                        error_msg = f"Workload request {r+1} failed: {type(e).__name__}: {str(e)}"
                        _append_auto_perf_log(log_path, f"WORKLOAD_ERROR {error_msg}")
                        print(f"    ⚠️  {error_msg}")
                        break
                    
                    total_elapsed += elapsed
                    total_prompt_tokens += prompt_n
                    total_predicted_tokens += predicted_n
                    
                    # Only mark timeout if we exceed max_total_s AND have no tokens (incomplete/hung request)
                    # If we got tokens, it's a successful response even if it took longer than budgeted
                    if max_total_s and total_elapsed > max_total_s and total_predicted_tokens <= 0.0:
                        any_fail = True
                        metrics["timeout"] = True
                        break

                if adaptive_fill and not any_fail and not metrics["timeout"]:
                    observed_total = total_prompt_tokens + total_predicted_tokens
                    if observed_total < max(0, ctx_size - 16):
                        last_prompt_n = prompt_n if 'prompt_n' in locals() else 0.0
                        desired_predict = max(1, int(ctx_size - last_prompt_n))
                        try:
                            elapsed2, prompt_n2, predicted_n2 = _single_request(9999, desired_predict)
                            total_elapsed += elapsed2
                            total_prompt_tokens += prompt_n2
                            total_predicted_tokens += predicted_n2
                        except Exception:
                            pass

            except Exception:
                any_fail = True

            # Mark as failure only if:
            # 1. Any request threw an exception (any_fail=True)
            # 2. OR total time is zero (no requests completed)
            # 3. AND no tokens were generated (completely unsuccessful)
            # Do NOT fail if tokens were generated, even if requests had issues
            has_tokens = total_predicted_tokens > 0.0
            if max_total_s and total_elapsed > max_total_s and has_tokens:
                _append_auto_perf_log(log_path, f"WORKLOAD_SLOW_COMPLETION elapsed={total_elapsed:.1f}s > budget={max_total_s:.1f}s but completed with {total_predicted_tokens:.0f} tokens")
            if (any_fail or total_elapsed <= 0.0) and not has_tokens:
                if not metrics["timeout"]:
                    metrics["crash"] = True
                    crash_reason = f"Workload measurement failed: any_fail={any_fail}, total_elapsed={total_elapsed}, tokens={total_predicted_tokens}"
                    _append_auto_perf_log(log_path, f"WORKLOAD_FAILED {crash_reason}")
                    print(f"    ⚠️  {crash_reason}")
            elif has_tokens:
                _record_success_metrics(total_prompt_tokens, total_predicted_tokens, total_elapsed)
                if confirm_long_context_if_promising and not confirmation_long_context:
                    screening_decode_tps = float(metrics.get("decode_tokens_s", 0.0) or 0.0)
                    if confirm_decode_tps_floor <= 0.0 or screening_decode_tps >= confirm_decode_tps_floor:
                        confirm_predict = _long_confirmation_predict_tokens(ctx_size)
                        confirm_prompt_tokens = _long_confirmation_prompt_tokens(ctx_size)
                        print(
                            "    Screening promising; running long-context confirmation on the same loaded server "
                            f"(prompt≈{confirm_prompt_tokens}, n_predict={confirm_predict})."
                        )
                        try:
                            elapsed_c, prompt_n_c, predicted_n_c = _single_request(
                                424242,
                                confirm_predict,
                                best_tps=best_tps,
                                prompt_variant="ctx_half",
                                target_prompt_tokens=confirm_prompt_tokens,
                            )
                            metrics["screening_prefill_tokens_s"] = total_prompt_tokens / max(total_elapsed, 1e-6)
                            metrics["screening_decode_tokens_s"] = screening_decode_tps
                            metrics["screening_total_tokens_s"] = (total_prompt_tokens + total_predicted_tokens) / max(total_elapsed, 1e-6)
                            metrics["screening_prompt_tokens"] = total_prompt_tokens
                            metrics["screening_predicted_tokens"] = total_predicted_tokens
                            metrics["confirmed_from_screening"] = True
                            _record_success_metrics(prompt_n_c, predicted_n_c, elapsed_c, label="BENCHMARK_CONFIRM_OK")
                        except Exception as confirm_exc:
                            _append_auto_perf_log(log_path, f"CONFIRM_WORKLOAD_ERROR {type(confirm_exc).__name__}: {confirm_exc}")
                            print(f"    ⚠️  Long-context confirmation failed on loaded server; keeping screening metrics: {confirm_exc}")
                    else:
                        print(
                            f"    Screening decode {screening_decode_tps:.2f} t/s below confirmation floor "
                            f"{confirm_decode_tps_floor:.2f}; skipping long confirmation."
                        )

        gpu_after = _sample_gpu_memory(gpu_indices) if gpu_set is not None else {}
        if gpu_before is not None and gpu_after is not None:
            total_before = sum(gpu_before.get(i, 0) for i in gpu_indices)
            total_after = sum(gpu_after.get(i, 0) for i in gpu_indices)
            metrics["vram_used"] = max(0, total_after - total_before)

        try:
            import psutil

            if mem_before is not None:
                metrics["ram_used"] = max(0, psutil.virtual_memory().used - mem_before)
        except Exception:
            pass
    except Exception as exc:
        _append_auto_perf_log(log_path, f"BENCHMARK_EXCEPTION type={type(exc).__name__} error={exc}")
        tb = traceback.format_exc()
        if tb:
            _append_auto_perf_log(log_path, "BENCHMARK_TRACEBACK_BEGIN")
            for line in tb.splitlines():
                _append_auto_perf_log(log_path, line)
            _append_auto_perf_log(log_path, "BENCHMARK_TRACEBACK_END")
        try:
            if server_proc:
                out, err = server_proc.communicate(timeout=1)
                combined = (out or "") + "\n" + (err or "")
                if "out of memory" in combined.lower() or "oom" in combined.lower() or "bad_alloc" in combined.lower():
                    metrics["oom"] = True
                else:
                    metrics["crash"] = True
                if combined.strip():
                    metrics["server_output_tail"] = _tail_text(combined)
                    _append_text_block(log_path, "SERVER_OUTPUT_ON_EXCEPTION_BEGIN", metrics["server_output_tail"], "SERVER_OUTPUT_ON_EXCEPTION_END")
        except Exception:
            metrics["crash"] = True
    finally:
        if server_proc and server_proc.poll() is None:
            try:
                server_proc.terminate()
                server_proc.wait(timeout=5)
            except Exception:
                try:
                    server_proc.kill()
                    server_proc.wait(timeout=3)
                except Exception:
                    pass
        if server_proc and (metrics.get("oom") or metrics.get("crash") or metrics.get("timeout")) and not metrics.get("server_output_tail"):
            try:
                tail = _capture_server_output_tail(server_proc)
                if tail:
                    metrics["server_output_tail"] = tail
                    _append_text_block(log_path, "SERVER_STDOUT_BEGIN", tail, "SERVER_STDOUT_END")
            except Exception:
                pass
        if server_proc and server_proc.stdout:
            try:
                server_proc.stdout.close()
            except Exception:
                pass

    return metrics


def probe_max_ctx(model_path: str, base_params: dict, gpu_set: list[int], max_ctx: int) -> int:
    def _probe_ok(ctx_val: int) -> bool:
        params = dict(base_params)
        metrics = run_benchmark(model_path, params, int(ctx_val), gpu_set, mock=False, n_predict=32, server_ready_timeout_s=120)
        return not (metrics.get("oom") or metrics.get("crash") or metrics.get("timeout"))

    if _probe_ok(512) is False:
        return 0

    low = 512
    high = 1024
    while high <= max_ctx and _probe_ok(high):
        low = high
        high = min(max_ctx, high * 2)
        if high == low:
            break

    lo = low
    hi = high
    while lo + 256 < hi:
        mid = (lo + hi) // 2
        if _probe_ok(mid):
            lo = mid
        else:
            hi = mid
    return lo
    
def _ask_yes_no(args, prompt: str, default: str = "n") -> bool:
    normalized_prompt = str(prompt or "").lower()
    final_catalog_prompt = "aplicar" in normalized_prompt or "catalog" in normalized_prompt or "sobrescribir" in normalized_prompt
    if final_catalog_prompt and (bool(getattr(args, "unattended", False)) or str(os.environ.get("LLAMACPP_AUTO_PERF_UNATTENDED", "")).strip().lower() in {"1", "true", "yes", "y"}):
        # In unattended tuning, run all benchmark phases automatically but
        # still stop for the final persistent catalog mutation decision.
        unattended_was_set = bool(getattr(args, "unattended", False))
        try:
            setattr(args, "unattended", False)
        except Exception:
            pass
        old_env = os.environ.pop("LLAMACPP_AUTO_PERF_UNATTENDED", None)
        try:
            return _ask_yes_no(args, prompt, default)
        finally:
            if old_env is not None:
                os.environ["LLAMACPP_AUTO_PERF_UNATTENDED"] = old_env
            try:
                setattr(args, "unattended", unattended_was_set)
            except Exception:
                pass
    if bool(getattr(args, "assume_no", False)) or bool(getattr(args, "no_prompt", False)):
        return False
    if str(os.environ.get("LLAMACPP_AUTO_PERF_ASSUME_NO", "")).strip().lower() in {"1", "true", "yes", "y"}:
        return False
    if bool(getattr(args, "unattended", False)) or str(os.environ.get("LLAMACPP_AUTO_PERF_UNATTENDED", "")).strip().lower() in {"1", "true", "yes", "y"}:
        # Unattended means: run all optimization/refresh phases, but do not
        # make the final destructive/persistent catalog overwrite decision.
        print(f"{prompt} [Y/n]: y (unattended)")
        return True

    callback = getattr(args, "_question_callback", None)
    answer = ""
    if callable(callback):
        try:
            answer = str(callback(prompt, default) or "").strip()
        except Exception:
            answer = ""
    else:
        suffix = "[Y/n]" if str(default).lower() == "y" else "[y/N]"
        try:
            answer = input(f"{prompt} {suffix}: ").strip()
        except EOFError:
            answer = ""

    normalized = answer.lower()
    if not normalized:
        normalized = str(default).lower()
    return normalized in {"y", "yes", "s", "si"}


def _ask_run_phase(args, phase_label: str, default: str = "y") -> bool:
    return _ask_yes_no(args, f"¿Ejecutar fase {phase_label}?", default)


def run_auto_performance(args) -> int:
    """
    PHASE 1: Core llama.cpp inference throughput tuning via Optuna.
    
    Optimizes a model's server_overrides by searching through PROBED_TUNER_KEYS:
    - GPU distribution (split_mode, tensor_split, n_gpu_layers)
    - Batch/compute (batch_size, ubatch_size, flash_attn)
    - KV cache strategy (cache_type_k/v, kv_offload, cache_ram)
    - CPU threading (threads, threads_batch, numa)
    - direct_io is probed separately as the stage-0 load selector

    PHASE 2: Speculative draft-side tuning when the model is speculative.
    - model_draft, draft, ctx_size_draft, n_gpu_layers_draft
    - cache_type_k_draft, cache_type_v_draft
    - No quality regressions are accepted to buy speed

    PHASE 3: Server API tuning when server_api_mode is enabled.
    - parallel, cont_batching, ctx_checkpoints, cache_ram, threads_http
    - Multi-client HTTP serving metrics only
    
    OUT OF SCOPE (stored in comments for future phases):
    - Sampling params (temperature, top_p, etc.) — affect quality not tokens/s
    - Deprecated speculative compatibility flags (draft_min, draft_p_min)
    - Non-performance sampling or quality-only knobs
    
    Metric: 0.6*decode_tps + 0.3*prefill_tps + 0.1*ctx_stability - gpu_penalty
    
    Profile persistence: SHA256(model_id + quant + llama_cpp_version + hw_fingerprint)
    stored in json/auto_performance_profiles.json
    """
    if getattr(args, "llama_server", None):
        os.environ["LLAMA_SERVER_BIN"] = str(getattr(args, "llama_server"))
    optuna = _ensure_optuna()
    optuna_seed = int(getattr(args, "optuna_seed", 42) or 42)

    def _make_sampler(seed_offset: int = 0):
        try:
            warning_categories = [Warning]
            if OptunaExperimentalWarning is not None:
                warning_categories = [OptunaExperimentalWarning]
            with warnings.catch_warnings():
                for category in warning_categories:
                    warnings.simplefilter("ignore", category)
                return optuna.samplers.TPESampler(
                    seed=optuna_seed + seed_offset,
                    multivariate=True,
                    group=True,
                    constant_liar=True,
                    n_startup_trials=8,
                )
        except TypeError:
            # Older distro-packaged Optuna may not support all modern options.
            return optuna.samplers.TPESampler(seed=optuna_seed + seed_offset, n_startup_trials=8)
    catalog_path = _resolve_catalog_path(args)
    items, _ = load_catalog_with_diagnostics(catalog_path)
    if not items:
        raise RuntimeError(f"Catalog file not found or empty: {catalog_path}")

    server_api_mode = bool(getattr(args, "server_api", False))
    server_phase_only = bool(getattr(args, "server_phase_only", False))
    load_concurrency = max(1, int(getattr(args, "load_concurrency", 1) or 1))
    load_requests = max(1, int(getattr(args, "load_requests", 1) or 1))

    model = resolve_catalog_model(
        items,
        target=getattr(args, "repo", None),
        repo_ref=getattr(args, "hf", None),
        model_id=getattr(args, "model_id", None),
        filename=getattr(args, "file", None),
    )

    model_ok, model_reason = _validate_model_artifact(model.local_path)
    if not model_ok:
        raise RuntimeError(f"Invalid model artifact for auto-performance: {model_reason}")

    print(f"Optimizing model: {model.model_id} ({model.local_path})")
    # Use catalog ctx_size as fallback when GGUF cannot be read
    catalog_ctx = getattr(model, "ctx_size", 8192) or 8192
    meta = read_gguf_metadata(model.local_path, default_ctx=catalog_ctx)
    print(f"GGUF Metadata: {meta}")

    log_path = _resolve_auto_perf_log_path(args, model.model_id)
    print(f"Auto-performance log: {log_path}")
    _append_auto_perf_log(log_path, f"START model={model.model_id} local_path={model.local_path} server_api={bool(getattr(args, 'server_api', False))}")
    _append_auto_perf_log(log_path, f"GGUF_METADATA {json.dumps(meta, ensure_ascii=False, sort_keys=True)}")

    gpu_count = detect_cuda_device_count()
    if gpu_count == 0:
        print("No GPUs detected. Assuming 1 virtual GPU for mock/cpu purposes.")
        gpu_count = 1
    print(f"Detected {gpu_count} GPUs.")

    baseline_defaults = _resolve_baseline(model, gpu_count)
    target_ctx = _normalize_ctx_size(int(baseline_defaults.get("ctx_size", getattr(model, "ctx_size", 8192)) or getattr(model, "ctx_size", 8192) or 8192))
    search_ctx = target_ctx
    is_speculative = _is_speculative_model(model)
    spec_defaults = normalize_server_overrides(getattr(model, "spec_meta", {}) or {}) if is_speculative else {}
    # Prefer an explicit `model_draft` in server_overrides; if not present,
    # allow `draft_local_path` or `draft_model_id` from the raw spec_meta
    # (normalize_server_overrides can drop some spec_meta keys like
    # 'draft_local_path', so read the raw spec_meta as fallback).
    raw_spec_meta = getattr(model, "spec_meta", {}) or {}
    draft_model_path = str(
        baseline_defaults.get("model_draft")
        or spec_defaults.get("model_draft")
        or raw_spec_meta.get("draft_local_path")
        or raw_spec_meta.get("draft_local_path")
        or ""
    ).strip()
    if is_speculative and not draft_model_path:
        raise RuntimeError(
            f"Speculative model '{model.model_id}' is missing a draft model path. "
            "Set model_draft or draft_local_path before running auto-performance."
        )
    expected_models_loaded = 2 if is_speculative else 1

    baseline = {
        "gpu_set_idx": len(_get_feasible_gpu_sets(gpu_count)) - 1, # Default to last set (usually all gpus)
        "split_mode": str(baseline_defaults.get("split_mode", "layer")),
        "ts_strategy": "even",
        "tensor_split_strategy": "equal" if gpu_count > 1 else "auto",
        "ctx_size": search_ctx,
        "n_gpu_layers": _n_gpu_layers_cli_value(baseline_defaults.get("n_gpu_layers", "all")),
        "fit": "on" if baseline_defaults.get("fit", True) else "off",
        "fit_target": int(baseline_defaults.get("fit_target", 1024)),
        "batch_size": int(baseline_defaults.get("batch_size", 2048)),
        "ubatch_size": int(baseline_defaults.get("ubatch_size", 512)),
        "flash_attn": str(baseline_defaults.get("flash_attn", "auto")),
        "cache_type_k": _normalize_cache_type(baseline_defaults.get("cache_type_k", CACHE_TYPE_FLOOR)),
        "cache_type_v": _normalize_cache_type(baseline_defaults.get("cache_type_v", CACHE_TYPE_FLOOR)),
        "kv_offload": bool(baseline_defaults.get("kv_offload", True)),
        "direct_io": bool(baseline_defaults.get("direct_io", False)),
        "mmap": True,
        "numa": baseline_defaults.get("numa", None),
        "parallel": int(baseline_defaults.get("parallel", 1)),
        "cont_batching": bool(baseline_defaults.get("cont_batching", True)),
        "ctx_checkpoints": int(baseline_defaults.get("ctx_checkpoints", 0)),
        "cache_ram": int(baseline_defaults.get("cache_ram", 0)),
        "threads_http": int(baseline_defaults.get("threads_http", 1)),
        "op_offload": bool(baseline_defaults.get("op_offload", False)),
        "threads": int(baseline_defaults.get("threads", os.cpu_count() or 4)),
        "threads_batch": int(baseline_defaults.get("threads_batch", os.cpu_count() or 4)),
    }
    if is_speculative:
        baseline.update(
            {
                "model_draft": draft_model_path,
                "draft": int(baseline_defaults.get("draft", spec_defaults.get("draft", 16)) or 16),
                "ctx_size_draft": int(baseline_defaults.get("ctx_size_draft", spec_defaults.get("ctx_size_draft", min(search_ctx, 2048))) or min(search_ctx, 2048)),
                "n_gpu_layers_draft": _n_gpu_layers_cli_value(baseline_defaults.get("n_gpu_layers_draft", spec_defaults.get("n_gpu_layers_draft", baseline_defaults.get("n_gpu_layers", "all")))),
                "cache_type_k_draft": _normalize_cache_type(baseline_defaults.get("cache_type_k_draft", spec_defaults.get("cache_type_k_draft", CACHE_TYPE_FLOOR))),
                "cache_type_v_draft": _normalize_cache_type(baseline_defaults.get("cache_type_v_draft", spec_defaults.get("cache_type_v_draft", CACHE_TYPE_FLOOR))),
            }
        )
    if meta.get("is_moe"):
        baseline["cpu_moe"] = False
        baseline["n_cpu_moe"] = 0
    if server_api_mode:
        baseline.update(
            {
                "kv_unified": bool(baseline_defaults.get("kv_unified", False)),
                "cache_idle_slots": bool(baseline_defaults.get("cache_idle_slots", False)),
            }
        )
    if server_phase_only:
        inherited_best_params = getattr(args, "_inherited_best_params", None)
        if isinstance(inherited_best_params, dict) and inherited_best_params:
            # The SERVER continuation should improve the config found in the
            # previous phases, not rebuild from catalog defaults. Keep server
            # knobs from current defaults unless the inherited best already has
            # them; fixed core/speculative knobs are inherited.
            merged = dict(baseline)
            for key, value in inherited_best_params.items():
                if value is not None:
                    merged[key] = value
            merged.setdefault("parallel", baseline.get("parallel", 1))
            merged.setdefault("cont_batching", baseline.get("cont_batching", True))
            merged.setdefault("ctx_checkpoints", baseline.get("ctx_checkpoints", 0))
            merged.setdefault("cache_ram", baseline.get("cache_ram", 0))
            merged.setdefault("threads_http", baseline.get("threads_http", 1))
            merged.setdefault("kv_unified", baseline.get("kv_unified", False))
            merged.setdefault("cache_idle_slots", baseline.get("cache_idle_slots", False))
            baseline = merged
            print("  SERVER phase will use the best config from previous phases as its base, then tune only server flags.")

    is_mock = bool(getattr(args, "mock", False))
    gpu_indices = list(range(gpu_count)) if gpu_count > 0 else []
    gpu_memory_snapshot = _sample_gpu_memory_snapshot(gpu_indices)
    gpu_memory_budget_mib = _gpu_memory_budget_mib(gpu_memory_snapshot, gpu_count)

    # direct_io is always enabled; the load probe no longer compares ON/OFF.
    baseline["direct_io"] = True

    # User-specified trials per phase, or auto-calculate based on GPU count
    # Default total trials: 40 (split 20 / 10 / 10 when speculative + server)
    user_trials_per_phase = getattr(args, "trials_per_phase", None)
    if user_trials_per_phase is not None:
        n_trials = max(6, int(user_trials_per_phase))
    else:
        if is_mock:
            n_trials = max(6, gpu_count + 4)
        else:
            n_trials = 40

    print(f"\n{'='*70}")
    print(f"AUTO-PERFORMANCE TUNING: {model.model_id}")
    print(f"{'='*70}")
    print(f"Total trials to run: {n_trials}")
    print(f"GPUs available: {gpu_count}")
    print(f"Context size: {search_ctx}")
    print(f"Mode: {'MOCK' if is_mock else 'REAL'} | {'Server API' if server_api_mode else 'Raw inference'}")
    print(f"Log file: {log_path}")
    print(f"{'='*70}\n")

    # Prepare incremental best tracking so we can persist the best config
    server_bin = get_server_path()
    llama_cpp_version = _llama_cpp_version(server_bin)
    hw = _hardware_fingerprint(search_ctx)
    profile_key_payload = {
        "model": model.model_id,
        "quant": model.quant,
        "ctx_size": search_ctx,
        "llama_cpp_version": llama_cpp_version,
        "hardware_fingerprint": hw.get("fingerprint"),
    }
    profile_key = hashlib.sha256(json.dumps(profile_key_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()

    best_so_far = {
        "score": None,
        "metrics": None,
        "params": None,
        "trial_num": -1,
    }
    final_config_params: dict | None = None
    final_config_metrics: dict | None = None
    final_config_score: float | None = None
    core_best = {
        "score": None,
        "metrics": None,
        "params": None,
        "trial_num": -1,
    }
    evaluated_cache: dict[str, dict] = {}
    search_counters = {
        "valid_evals": 0,
        "benchmarks_run": 0,
        "duplicates_reused": 0,
        "invalid_attempts": 0,
        "structural_repairs": 0,
    }
    phase_progress = {"start_valid": 0, "limit": n_trials, "label": ""}
    phase_results: dict[str, dict] = {}

    print("\n" + "=" * 70)
    print("CURRENT PERFORMANCE (baseline)")
    print("=" * 70)
    print(f"  Model: {model.model_id}")
    print(f"  Catalog path: {catalog_path.resolve()}")
    print(f"  Profile path: {AUTO_PERF_PROFILES_PATH.resolve()}")
    print(f"  History path: {AUTO_PERF_HISTORY_PATH.resolve()}")

    baseline_phase_label = "SERVER" if server_api_mode else "CORE"
    baseline_result_key = _phase_result_key(baseline, baseline_phase_label, api_mode=server_api_mode)
    baseline_cache_key = _canonical_benchmark_key(baseline, api_mode=server_api_mode)
    catalog_cached_baseline = None if is_mock else _find_catalog_cached_baseline(
        model,
        baseline,
        baseline_phase_label,
        api_mode=server_api_mode,
        llama_cpp_version=llama_cpp_version,
        hardware_fingerprint=str(hw.get("fingerprint") or ""),
    )
    cached_baseline = catalog_cached_baseline
    if cached_baseline is None and not is_mock:
        cached_baseline = _find_cached_baseline_result(
            profile_key,
            baseline_result_key,
            baseline,
            baseline_phase_label,
            api_mode=server_api_mode,
            model_id=model.model_id,
        )
    if cached_baseline:
        source = str(cached_baseline.get("source") or "profile")
        stored_score = float(cached_baseline.get("score", 0.0) or 0.0)
        stored_schema = str(cached_baseline.get("score_schema_version") or "legacy")
        print(
            f"  Cached {baseline_phase_label} baseline found in {source} "
            f"(stored_score={stored_score:.2f}, schema={stored_schema})."
        )
        if server_phase_only:
            print("  Reusing cached SERVER baseline without asking refresh because this is the same auto-performance continuation.")
        elif _ask_yes_no(args, f"¿Refrescar/recalcular baseline {baseline_phase_label} ahora?", "n"):
            print("  ↻ Refreshing cached baseline by running a new measurement.")
            cached_baseline = None

    if cached_baseline:
        baseline_metrics = dict(cached_baseline.get("metrics") or {})
        # Always recompute cached baseline score with the current scoring
        # schema. This prevents legacy cached rows from ranking with an old
        # metric after the scoring formula changes.
        baseline_score = score_server_performance(baseline_metrics, search_ctx, len(gpu_indices)) if server_api_mode else score_performance(baseline_metrics, search_ctx, len(gpu_indices))
        source = str(cached_baseline.get("source") or "profile")
        if source == "catalog":
            print(f"  Reusing cached {baseline_phase_label} baseline performance from catalog (current_score={baseline_score:.2f}).")
        else:
            print(f"  Reusing cached {baseline_phase_label} baseline performance from profile (source={source}, current_score={baseline_score:.2f}).")
    else:
        baseline_metrics = run_benchmark(
            model.local_path,
            _prepare_benchmark_params(baseline, api_mode=server_api_mode),
            search_ctx,
            gpu_indices,
            mock=is_mock,
            api_mode=server_api_mode,
            load_concurrency=load_concurrency,
            load_requests=load_requests,
            n_predict=SERVER_BENCHMARK_N_PREDICT if server_api_mode else RAW_SCREENING_N_PREDICT,
            runs=1 if server_api_mode else RAW_SCREENING_RUNS,
            log_path=log_path,
            confirm_long_context_if_promising=not server_api_mode,
            confirm_decode_tps_floor=0.0,
            server_ready_timeout_s=300,  # Full timeout for baseline measurement
            expected_models_loaded=expected_models_loaded,
        )
        baseline_score = score_server_performance(baseline_metrics, search_ctx, len(gpu_indices)) if server_api_mode else score_performance(baseline_metrics, search_ctx, len(gpu_indices))
    baseline_failed = _benchmark_metrics_failed(baseline_metrics)
    baseline_result: dict[str, object] | None = None
    if not baseline_failed:
        baseline_result = {
            "phase": baseline_phase_label,
            "role": "baseline",
            "score": baseline_score,
            "metrics": dict(baseline_metrics),
            "params": _params_for_observation(baseline),
            "benchmark_key": baseline_cache_key,
            "updated_at": cached_baseline.get("updated_at") if cached_baseline else datetime.now(timezone.utc).isoformat(),
            "source": cached_baseline.get("source", "cache") if cached_baseline else "measured",
            "llama_cpp_version": llama_cpp_version,
            "hardware_fingerprint": str(hw.get("fingerprint") or ""),
            "score_schema_version": AUTO_PERF_SCORE_SCHEMA_VERSION,
        }
        phase_results[baseline_result_key] = baseline_result
    if not cached_baseline and not baseline_failed:
        written_profile_path = _upsert_profile({
            "profile_key": profile_key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model.model_id,
            "repo": model.repo_id,
            "quant": model.quant,
            "llama_cpp_version": llama_cpp_version,
            "hardware": hw,
            "baseline": {
                "phase": baseline_phase_label,
                "metrics": baseline_metrics,
                "trial_value": baseline_score,
                "params": _params_for_observation(baseline),
                "benchmark_key": baseline_cache_key,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "score_schema_version": AUTO_PERF_SCORE_SCHEMA_VERSION,
            },
            "best": {},
            "phase_results": phase_results,
            "tuning_policy": {"ctx_size_floor": search_ctx, "ctx_size_policy": CTX_SIZE_POLICY, "probed_flags": list(PROBED_TUNER_KEYS), "pruned_flags": sorted(PRUNED_TUNER_KEYS)},
        })
        if written_profile_path:
            print(f"  💾 Baseline performance cached in profile: {written_profile_path}")
        else:
            print("  ⚠️  Could not cache baseline performance in profile.")
        if baseline_result is not None:
            if _save_catalog_baseline_cache(catalog_path, model.model_id, baseline_phase_label, baseline_cache_key, baseline_result):
                print(f"  💾 Baseline performance cached in catalog: {catalog_path.resolve()}")
            else:
                print(f"  ⚠️  Could not cache baseline performance in catalog: {catalog_path.resolve()}")
    baseline_vram_mib = float(baseline_metrics.get("vram_used", 0.0) or 0.0)
    memory_headroom_mib = None
    memory_headroom_ratio = None
    if gpu_memory_budget_mib is not None:
        memory_headroom_mib = max(0.0, float(gpu_memory_budget_mib) - baseline_vram_mib)
        memory_headroom_ratio = memory_headroom_mib / max(1.0, float(gpu_memory_budget_mib))
        print(f"  GPU free memory budget: {gpu_memory_budget_mib:.0f} MiB")
        print(f"  Estimated headroom after baseline: {memory_headroom_mib:.0f} MiB ({memory_headroom_ratio * 100:.1f}%)")
    
    print(f"  Prefill throughput: {float(baseline_metrics.get('prefill_tokens_s') or 0.0):.2f} tokens/s")
    print(f"  Decode throughput:  {float(baseline_metrics.get('decode_tokens_s') or 0.0):.2f} tokens/s")
    print(f"  Load time:          {float(baseline_metrics.get('load_ready_s') or 0.0):.2f}s")
    print(f"  Overall Score:      {baseline_score:.2f}")

    fatal_baseline, fatal_reason = _baseline_failure_is_fatal(baseline_metrics)
    if fatal_baseline:
        print("\n" + "=" * 70)
        print("AUTO-PERFORMANCE ABORTED BEFORE OPTUNA")
        print("=" * 70)
        print("El baseline no arranca de forma saludable; continuar quemaría trials con el mismo fallo.")
        print(f"  Reason: {fatal_reason}")
        if baseline_metrics.get("error"):
            print(f"  Error: {baseline_metrics.get('error')}")
        elif baseline_metrics.get("server_output_tail"):
            print(f"  Server output: {_tail_text(str(baseline_metrics.get('server_output_tail')), 500)}")
        print("No se guarda este fallo en el perfil para no contaminar ni sobrescribir baselines/bests previos.")
        return 1

    if not baseline_failed:
        best_so_far.update({
            "score": baseline_score,
            "metrics": baseline_metrics,
            "params": baseline,
            "trial_num": -1,
        })
        core_best.update({
            "score": baseline_score,
            "metrics": baseline_metrics,
            "params": baseline,
            "trial_num": -1,
        })
        final_config_params = dict(baseline)
        final_config_metrics = dict(baseline_metrics)
        final_config_score = baseline_score
        evaluated_cache[baseline_cache_key] = {
            "metrics": dict(baseline_metrics),
            "score": baseline_score,
            "params": dict(baseline),
            "trial": -1,
            "source": "baseline",
        }

    def _save_best_profile_incremental(score, metrics, params, trial_num):
        """Persist a profile row representing the current best configuration."""
        profile_row = {
            "profile_key": profile_key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model.model_id,
            "repo": model.repo_id,
            "quant": model.quant,
            "llama_cpp_version": _llama_cpp_version(server_bin),
            "hardware": hw,
            "baseline": {
                "phase": baseline_phase_label,
                "metrics": baseline_metrics,
                "trial_value": baseline_score,
                "params": _params_for_observation(baseline),
                "benchmark_key": baseline_cache_key,
                "score_schema_version": AUTO_PERF_SCORE_SCHEMA_VERSION,
            },
            "best": {
                "metrics": metrics or {},
                "trial_value": score,
                "params": _params_for_observation(params or {}),
            },
            "phase_results": phase_results,
            "tuning_policy": {
                "ctx_size_floor": search_ctx,
                "ctx_size_policy": CTX_SIZE_POLICY,
                "probed_flags": list(PROBED_TUNER_KEYS),
                "pruned_flags": sorted(PRUNED_TUNER_KEYS),
            },
        }
        written = _upsert_profile(profile_row)
        history_profile_path = str((written or AUTO_PERF_PROFILES_PATH))
        _append_best_history({
            "profile_key": profile_key,
            "created_at": profile_row["created_at"],
            "model": model.model_id,
            "trial_num": trial_num,
            "score": score,
            "metrics": metrics or {},
            "params": _params_for_observation(params or {}),
            "profile_path": history_profile_path,
        })

    feasible_gpu_sets = _get_feasible_gpu_sets(gpu_count)
    default_gpu_set_idx = max(range(len(feasible_gpu_sets)), key=lambda idx: len(feasible_gpu_sets[idx])) if feasible_gpu_sets else 0
    active_phase = 1
    # Determine phase splits. Default (non-mock, no user override):
    # - Phase1: 20 trials (core/raw tuning)
    # - Speculative: 10 trials
    # - Server: 10 trials
    if user_trials_per_phase is not None:
        # Preserve previous proportional behaviour when user overrides total
        p1_limit = int(n_trials * 0.6) if server_api_mode else n_trials
        speculative_limit = max(4, n_trials // 4)
        server_limit = n_trials - p1_limit
    else:
        if server_api_mode:
            if is_speculative:
                p1_limit = 20
                speculative_limit = 10
                server_limit = 10
            else:
                p1_limit = 20
                speculative_limit = 0
                server_limit = 20
        else:
            # Raw inference mode (no server phase)
            if is_speculative:
                p1_limit = 20
                speculative_limit = 10
                server_limit = 0
            else:
                p1_limit = 20
                speculative_limit = 0
                server_limit = 0

    def _objective(trial, phase_limit=p1_limit):
        nonlocal final_config_params, final_config_metrics, final_config_score
        display_trial_num = (search_counters["valid_evals"] - int(phase_progress.get("start_valid", 0))) + 1
        display_trial_den = int(phase_progress.get("limit", phase_limit) or phase_limit)
        # Determine GPU set after phase-specific fixed params are known.
        gpu_set_idx_for_trial = default_gpu_set_idx if active_phase == 1 else 0
        
        raw_config = {}
        if active_phase == 1:
            # Phase 1: Use ALL available GPUs
            # OOM adaptation will handle tensor_split rebalancing instead
            raw_config.update({
                "split_mode": trial.suggest_categorical("split_mode", ["none", "layer", "row"]),
                "tensor_split_strategy": trial.suggest_categorical("tensor_split_strategy", _tensor_split_strategy_candidates()),
                "main_gpu_raw": trial.suggest_int("main_gpu_raw", 0, 7),
                "n_gpu_layers": trial.suggest_categorical("n_gpu_layers", ["all", "auto"]),
                "fit": trial.suggest_categorical("fit", ["on", "off"]),
                "fit_target": trial.suggest_categorical("fit_target", [1024, 2048, 4096]),
                "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096, 8192]),
                "ubatch_size": trial.suggest_categorical("ubatch_size", [256, 512, 1024, 2048]),
                "flash_attn": trial.suggest_categorical("flash_attn", ["on", "auto", "off"]),
                "kv_offload": trial.suggest_categorical("kv_offload", [True, False]),
                "numa": trial.suggest_categorical("numa", _numa_candidates()),
                "op_offload": trial.suggest_categorical("op_offload", [True, False]),
                "threads": trial.suggest_categorical("threads", [max(1, (os.cpu_count() or 4) // 2), os.cpu_count() or 4, (os.cpu_count() or 4) * 2]),
                "threads_batch": trial.suggest_categorical("threads_batch", [os.cpu_count() or 4, (os.cpu_count() or 4) * 2]),
            })
        elif active_phase == 2 and is_speculative:
            raw_config.update(_speculative_base_raw_config())
            raw_config.update({
                "draft": trial.suggest_categorical("draft", [4, 8, 16, 32]),
                "ctx_size_draft": trial.suggest_categorical("ctx_size_draft", [512, 768, 1024, 2048, 4096]),
                "n_gpu_layers_draft": trial.suggest_categorical("n_gpu_layers_draft", ["all", "auto"]),
            })
        else:
            best_core = core_best["params"] or baseline
            raw_config.update({
                # Fixed Core params from Phase 1
                "gpu_set_idx": best_core.get("gpu_set_idx", 0),
                "split_mode": best_core.get("split_mode", "layer"),
                "tensor_split_strategy": best_core.get("tensor_split_strategy", "balanced"),
                "main_gpu_raw": best_core.get("main_gpu_raw", 0),
                "n_gpu_layers": best_core.get("n_gpu_layers", "all"),
                "fit": best_core.get("fit", "on"),
                "fit_target": best_core.get("fit_target", 1024),
                "batch_size": best_core.get("batch_size", 2048),
                "ubatch_size": best_core.get("ubatch_size", 512),
                "flash_attn": best_core.get("flash_attn", "auto"),
                "cache_type_k": CACHE_TYPE_FLOOR,
                "cache_type_v": CACHE_TYPE_FLOOR,
                "kv_offload": best_core.get("kv_offload", True),
                "numa": best_core.get("numa", None),
                "op_offload": best_core.get("op_offload", True),
                "threads": best_core.get("threads", os.cpu_count() or 4),
                "threads_batch": best_core.get("threads_batch", os.cpu_count() or 4),
            })
            raw_config.update({
                # Optimized Server params in Phase 2
                "parallel": trial.suggest_categorical("parallel", [1, 2, 4, 8, 16]),
                "cont_batching": trial.suggest_categorical("cont_batching", [True, False]),
                "ctx_checkpoints": trial.suggest_categorical("ctx_checkpoints", [0, 32, 64]),
                "cache_ram": trial.suggest_categorical("cache_ram", [0, 8192, 16384, -1]),
                "threads_http": trial.suggest_int("threads_http", 1, 16),
                "kv_unified": trial.suggest_categorical("kv_unified", [True, False]),
                "cache_idle_slots": trial.suggest_categorical("cache_idle_slots", [True, False]),
            })


        normalized, gpu_indices_trial = _normalize_raw_config_for_repair(raw_config)

        actual, repair_log = repair_until_feasible(normalized, hw, model.local_path)
        actual_cache_key = _canonical_benchmark_key(actual, api_mode=server_api_mode)
        structural_repair = _repair_is_structural(repair_log)
        if structural_repair:
            search_counters["structural_repairs"] += 1
        vram_est = _estimate_vram_breakdown(model.local_path, actual, hw)
        # Ensure numeric type for downstream comparisons (avoid str|float issues)
        estimated_vram_mib = float(vram_est.get("total", 0.0) or 0.0)
        
        trial.set_user_attr("raw_config", raw_config)
        trial.set_user_attr("actual_config", actual)
        trial.set_user_attr("repair_log", repair_log)
        trial.set_user_attr("structural_repair", structural_repair)
        trial.set_user_attr("benchmark_cache_key", actual_cache_key)
        trial.set_user_attr("estimated_vram_mib", estimated_vram_mib)

        if active_phase == 2 and is_speculative:
            infeasible_spec, infeasible_reason = _speculative_repair_log_infeasible(repair_log)
            if infeasible_spec:
                measured_reference_is_feasible = not baseline_failed and core_best.get("trial_num") == -1
                if measured_reference_is_feasible and _speculative_candidate_not_heavier_than_reference(normalized, _speculative_base_raw_config()):
                    # The real baseline just proved base+draft fits. The VRAM
                    # estimator can be pessimistic, so do not prune lighter
                    # speculative variants before an actual server load.
                    actual = dict(normalized)
                    repair_log = [entry for entry in repair_log if "no room for draft model" not in str(entry).lower()]
                    actual_cache_key = _canonical_benchmark_key(actual, api_mode=server_api_mode)
                    trial.set_user_attr("actual_config", actual)
                    trial.set_user_attr("repair_log", repair_log)
                    trial.set_user_attr("benchmark_cache_key", actual_cache_key)
                    print("  Note: ignoring pessimistic speculative VRAM repair because measured baseline fits and candidate is not heavier.")
                else:
                    search_counters["invalid_attempts"] += 1
                    trial.set_user_attr("speculative_pruned", True)
                    trial.set_user_attr("speculative_prune_reason", infeasible_reason)
                    _log_trial_event(
                        log_path,
                        "TRIAL_PRUNED",
                        {
                            "trial": trial.number,
                            "phase": active_phase,
                            "reason": infeasible_reason,
                            "raw_config": raw_config,
                            "normalized_config": normalized,
                            "actual_config": actual,
                            "repair_log": repair_log,
                        },
                    )
                    print(f"  RESULT=PRUNED (reason={infeasible_reason}; speculative config cannot fit draft model)")
                    raise optuna.TrialPruned()
            spec_ok, spec_reason = _speculative_config_valid_after_repair(normalized, actual)
            if not spec_ok:
                search_counters["invalid_attempts"] += 1
                trial.set_user_attr("speculative_pruned", True)
                trial.set_user_attr("speculative_prune_reason", spec_reason)
                _log_trial_event(
                    log_path,
                    "TRIAL_PRUNED",
                    {
                        "trial": trial.number,
                        "phase": active_phase,
                        "reason": spec_reason,
                        "raw_config": raw_config,
                        "normalized_config": normalized,
                        "actual_config": actual,
                        "repair_log": repair_log,
                    },
                )
                print(f"  RESULT=PRUNED (reason={spec_reason}; speculative phase cannot change core or disable draft)")
                raise optuna.TrialPruned()
        
        if gpu_memory_budget_mib is not None and estimated_vram_mib > (float(gpu_memory_budget_mib) * 0.98):
            # Cast budget to float for formatting and to satisfy static checkers
            budget_float = float(gpu_memory_budget_mib)
            print(f"  ⚠️  Estimated VRAM {estimated_vram_mib:.0f} MiB exceeds budget {budget_float:.0f} MiB; pruning.")
            trial.set_user_attr("memory_pruned", True)
            _log_trial_event(
                log_path,
                "TRIAL_PRUNED",
                {
                    "trial": trial.number,
                    "phase": active_phase,
                    "reason": "vram-budget",
                    "estimated_vram_mib": estimated_vram_mib,
                    "gpu_memory_budget_mib": budget_float,
                    "raw_config": raw_config,
                    "actual_config": actual,
                },
            )
            print(f"  RESULT=PRUNED (reason=vram-budget, est_vram={estimated_vram_mib:.0f} MiB)")
            search_counters["invalid_attempts"] += 1
            raise optuna.TrialPruned()

        cached_eval = evaluated_cache.get(actual_cache_key)
        if cached_eval is not None:
            search_counters["duplicates_reused"] += 1
            trial.set_user_attr("duplicate_of", cached_eval.get("trial"))
            trial.set_user_attr("metrics", dict(cached_eval.get("metrics") or {}))
            trial.set_user_attr("params", dict(cached_eval.get("params") or actual))
            print(
                f"\n[Trial {display_trial_num}/{display_trial_den}] (Phase {active_phase}) "
                f"Duplicate repaired config; cached score from {cached_eval.get('source', 'trial')} "
                f"#{cached_eval.get('trial')} will not consume valid trial budget."
            )
            raise optuna.TrialPruned()

        # Aggressive trial progress logging (showing diffs from best)
        base_ref = best_so_far["params"] or baseline
        diff_from_ref = {k: v for k, v in actual.items() if k in base_ref and v != base_ref[k]}
        if active_phase == 2 and is_speculative:
            diff_from_ref = {k: v for k, v in diff_from_ref.items() if k in PHASE1_SPECULATIVE_SEARCH_KEYS}
        elif server_api_mode and active_phase in {2, 3}:
            diff_from_ref = {k: v for k, v in diff_from_ref.items() if k in PHASE2_SERVER_TUNER_KEYS}
            if not diff_from_ref:
                search_counters["invalid_attempts"] += 1
                trial.set_user_attr("server_noop_pruned", True)
                trial.set_user_attr("server_prune_reason", "server-phase-no-server-param-change")
                print(
                    f"\n[Trial {display_trial_num}/{display_trial_den}] (Phase {active_phase}) "
                    "SERVER no-op candidate; baseline/server reference already measured, not burning a benchmark."
                )
                raise optuna.TrialPruned()
        benchmark_equivalent_to_baseline = _benchmark_params_equivalent(actual, baseline, api_mode=server_api_mode)
        benchmark_equivalent_to_best = bool(best_so_far["params"]) and _benchmark_params_equivalent(actual, best_so_far["params"], api_mode=server_api_mode)
        
        print(f"\n[Trial {display_trial_num}/{display_trial_den}] (Phase {active_phase}) Testing configuration...")
        
        # === NEW: GPU & VRAM diagnostics ===
        gpu_set_str = str(gpu_indices_trial) if gpu_indices_trial else "[0]"
        tensor_split_str = actual.get("tensor_split", "1")
        budget_mib = float(gpu_memory_budget_mib) if gpu_memory_budget_mib else 0
        vram_pct = (estimated_vram_mib / budget_mib * 100) if budget_mib > 0 else 0
        print(f"  GPUs: {gpu_set_str} | tensor_split: {tensor_split_str} | Est. VRAM: {estimated_vram_mib:.0f} / {budget_mib:.0f} MiB ({vram_pct:.1f}%)")
        
        if diff_from_ref:
            print(f"  Modifications from best run: {json.dumps(diff_from_ref, ensure_ascii=False)}")
        else:
            print(f"  Testing baseline/initial configuration...")
        config_diff = _format_config_diff_notice(actual, base_ref, raw_config)
        if active_phase == 2 and is_speculative:
            speculative_view = {
                "model_draft": actual.get("model_draft"),
                "draft": actual.get("draft"),
                "ctx_size_draft": actual.get("ctx_size_draft"),
                "n_gpu_layers_draft": actual.get("n_gpu_layers_draft"),
                "cache_type_k_draft": actual.get("cache_type_k_draft"),
                "cache_type_v_draft": actual.get("cache_type_v_draft"),
                "split_mode": actual.get("split_mode"),
                "tensor_split": actual.get("tensor_split"),
                "main_gpu": actual.get("main_gpu"),
                "n_gpu_layers": actual.get("n_gpu_layers"),
                "batch_size": actual.get("batch_size"),
                "ubatch_size": actual.get("ubatch_size"),
                "fit_target": actual.get("fit_target"),
                "kv_offload": actual.get("kv_offload"),
                "op_offload": actual.get("op_offload"),
            }
            print(f"  Actual speculative config: {json.dumps(speculative_view, ensure_ascii=False, sort_keys=True)}")
            print(f"  Config diff vs reference: {config_diff}")
        else:
            print(f"  Actual config: {json.dumps(actual, ensure_ascii=False, sort_keys=True)}")
            print(f"  Config diff vs reference: {config_diff}")
        if repair_log:
            print(f"  Repairs applied: {json.dumps(repair_log, ensure_ascii=False)}")


        metrics = None
        final_params = actual
        terminal_error = None

        if benchmark_equivalent_to_baseline or benchmark_equivalent_to_best:
            reused_metrics = baseline_metrics if benchmark_equivalent_to_baseline else (best_so_far["metrics"] or baseline_metrics)
            if reused_metrics:
                metrics = dict(reused_metrics)
                source_label = "baseline" if benchmark_equivalent_to_baseline else "current best"
                print(f"  Skipping benchmark: repaired config matches {source_label} benchmark params; reusing cached metrics.")
            else:
                print("  Skipping benchmark was possible, but no cached metrics were available; running benchmark anyway.")

        if metrics is None:
            attempts = [actual]
            if actual.get("ubatch_size", 512) > 256:
                fallback = dict(actual)
                fallback["ubatch_size"] = 256
                attempts.append(fallback)

            for attempt_idx, attempt in enumerate(attempts, 1):
                if len(attempts) > 1 and attempt_idx > 1:
                    print(f"  Retry attempt {attempt_idx}/{len(attempts)} with ubatch_size={attempt['ubatch_size']}...")

                current_best_decode_tps = 0.0
                if best_so_far["metrics"]:
                    current_best_decode_tps = float(best_so_far["metrics"].get("decode_tokens_s", 0.0))

                phase_valid_so_far = search_counters["valid_evals"]
                use_screening_benchmark = (not server_api_mode) and phase_valid_so_far < max(1, int(phase_limit * 0.75))
                benchmark_n_predict = 256 if use_screening_benchmark else RAW_SCREENING_N_PREDICT
                benchmark_runs = 1 if use_screening_benchmark else 2
                benchmark_timeout = 180.0 if use_screening_benchmark else 300.0
                if use_screening_benchmark:
                    print("  Fast screening benchmark: n_predict=256, runs=1 (full run only if promising).")

                metrics = run_benchmark(
                    model.local_path,
                    _prepare_benchmark_params(attempt, api_mode=server_api_mode),
                    search_ctx,
                    gpu_indices_trial,
                    api_mode=server_api_mode,
                    n_predict=benchmark_n_predict,
                    runs=benchmark_runs,
                    max_total_s=benchmark_timeout,
                    best_tps=current_best_decode_tps,
                    confirm_long_context_if_promising=use_screening_benchmark,
                    confirm_decode_tps_floor=(current_best_decode_tps * 0.97) if current_best_decode_tps > 0 else 0.0,
                    log_path=log_path,
                    mock=is_mock,
                    server_ready_timeout_s=300,
                    expected_models_loaded=expected_models_loaded,
                )
                if use_screening_benchmark:
                    metrics["screening_benchmark"] = True

                final_params = attempt

                if metrics.get("oom") or metrics.get("crash") or metrics.get("timeout"):
                    if metrics.get("timeout") and "pruning" in str(metrics.get("error", "")):
                        terminal_error = "early-pruning"
                        break
                    if attempt_idx == len(attempts):
                        terminal_error = "error"
                        break
                else:
                    break

        metrics = metrics or {}

        if terminal_error == "early-pruning":
            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("params", final_params)
            _log_trial_event(
                log_path,
                "TRIAL_PRUNED",
                {
                    "trial": trial.number,
                    "phase": active_phase,
                    "reason": "early-throughput-pruning",
                    "error": str(metrics.get("error", "")),
                    "load_reason": metrics.get("load_reason", ""),
                    "params": final_params,
                    "metrics": metrics,
                },
            )
            print("  RESULT=PRUNED (reason=early-throughput-pruning)")
            search_counters["invalid_attempts"] += 1
            raise optuna.TrialPruned()

        if terminal_error == "error" or metrics.get("oom") or metrics.get("crash") or metrics.get("timeout"):
            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("params", final_params)
            reason = "oom" if metrics.get("oom") else "crash" if metrics.get("crash") else "timeout" if metrics.get("timeout") else "unknown"
            score = -1000.0
            raw_output = str(metrics.get("server_output_tail", "") or "")
            error_summary = str(metrics.get("error", "") or "")
            if raw_output:
                error_summary = _extract_descriptive_error(raw_output, fallback=error_summary)
            if not error_summary:
                error_summary = f"{reason} (no error string)"
            _log_trial_event(
                log_path,
                "TRIAL_EXCEPTION",
                {
                    "trial": trial.number,
                    "phase": active_phase,
                    "reason": reason,
                    "error": error_summary,
                    "load_reason": metrics.get("load_reason", ""),
                    "params": final_params,
                    "metrics": metrics,
                    "score": score,
                },
            )
            if raw_output:
                _append_text_block(log_path, "TRIAL_EXCEPTION_SERVER_OUTPUT_BEGIN", raw_output, "TRIAL_EXCEPTION_SERVER_OUTPUT_END")
            print(
                f"  RESULT=ERROR (reason={reason}, load_reason={metrics.get('load_reason', 'unknown')}, "
                f"error={error_summary[:180]})"
            )
            
            # === NEW: OOM Reactive Adaptation ===
            # If OOM detected in an early trial, try to adapt by enqueuing a recovery trial
            search_counters["invalid_attempts"] += 1
            if metrics.get("oom") and trial.number < phase_limit * 0.8:
                print(f"  [OOM ADAPTIVE RECOVERY] Attempting to enqueue recovery trial...")
                
                # Option 1: Try spreading tensor_split more evenly across GPUs
                current_ts = final_params.get("tensor_split", "1")
                new_ts = _spread_tensor_split(current_ts, len(gpu_indices_trial))
                
                if new_ts != current_ts and len(gpu_indices_trial) > 1:
                    # Enqueue recovery trial with more balanced tensor_split
                    recovery_cfg = dict(raw_config)
                    recovery_cfg["tensor_split_strategy"] = "equal"  # Force equal distribution
                    _safe_enqueue_trial(study, recovery_cfg, "oom-recovery")
                    trial.set_user_attr("oom_recovery_action", f"Rebalanced tensor_split {current_ts} -> {new_ts}")
                    print(f"    Queued recovery: tensor_split rebalance ({current_ts} -> {new_ts})")
                    return -500.0  # Higher penalty than -1000, signals: retry queued
                
                # Option 2: If already balanced, try reducing batch_size
                if final_params.get("batch_size", 2048) > 512:
                    recovery_cfg = dict(raw_config)
                    recovery_cfg["batch_size"] = max(512, final_params.get("batch_size", 2048) // 2)
                    _safe_enqueue_trial(study, recovery_cfg, "oom-recovery")
                    trial.set_user_attr("oom_recovery_action", f"Reduced batch {final_params.get('batch_size')} -> {recovery_cfg['batch_size']}")
                    print(f"    Queued recovery: batch_size reduction ({final_params.get('batch_size')} -> {recovery_cfg['batch_size']})")
                    return -500.0
                
                print(f"    No recovery strategy applicable; OOM unrecoverable in Phase {active_phase}")
            
            return score

        score = score_server_performance(metrics, search_ctx, len(gpu_indices_trial)) if server_api_mode else score_performance(metrics, search_ctx, len(gpu_indices_trial))
        search_counters["valid_evals"] += 1
        if not (benchmark_equivalent_to_baseline or benchmark_equivalent_to_best):
            search_counters["benchmarks_run"] += 1
        evaluated_cache[actual_cache_key] = {
            "metrics": dict(metrics),
            "score": score,
            "params": dict(final_params),
            "trial": trial.number,
            "source": "trial",
        }
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("params", final_params)
        _log_trial_event(
            log_path,
            "TRIAL_RESULT",
            {
                "trial": trial.number,
                "phase": active_phase,
                "status": "success",
                "score": score,
                "params": final_params,
                "metrics": metrics,
            },
        )

        print(f"  RESULT=SUCCESS (Score: {score:.2f})")
        _print_success_trial_metrics_block(metrics, score, gpu_memory_budget_mib)

        phase_label_for_result = "SPECULATIVE" if active_phase == 2 and is_speculative else "SERVER" if server_api_mode and active_phase in {2, 3} else "CORE"
        if phase_label_for_result == "CORE" and _is_real_score_improvement(score, core_best["score"]):
            core_best.update({"score": score, "metrics": metrics, "params": final_params, "trial_num": trial.number})

        if _is_real_score_improvement(score, best_so_far["score"]):
            old_score = best_so_far["score"]
            best_so_far.update({"score": score, "metrics": metrics, "params": final_params, "trial_num": trial.number})
            if phase_label_for_result in {"CORE", "SERVER"}:
                final_config_params = dict(final_params)
                final_config_metrics = dict(metrics)
                final_config_score = score
            result_key = _phase_result_key(final_params, phase_label_for_result, api_mode=server_api_mode)
            phase_results[result_key] = {
                "phase": phase_label_for_result,
                "role": "best",
                "score": score,
                "metrics": dict(metrics),
                "params": _params_for_observation(final_params),
                "benchmark_key": _canonical_benchmark_key(final_params, api_mode=server_api_mode),
                "trial": trial.number,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            _save_best_profile_incremental(score, metrics, final_params, trial.number)
            if old_score is not None:
                improvement = _score_improvement_percent(score, old_score)
                print(f"  🎯 New best configuration candidate! (+{improvement:.2f}% improvement; catalog not modified yet)")
            else:
                print("  🎯 First valid best configuration candidate found (catalog not modified yet)")
        elif best_so_far["score"] is not None:
            delta = float(score) - float(best_so_far["score"])
            print(f"  No real improvement over current best (Δ={delta:.6f}); keeping previous best and not saving catalog.")
        return score

    def _run_optimization_loop(study, phase_label, limit, current_best_ref, search_keys):
        print(f"\n{'='*70}\nOPTIMIZATION PHASE: {phase_label} ({limit} valid unique evals)\n{'='*70}")
        print(f"Search space keys ({phase_label}): {', '.join(search_keys)}")
        start_time = time.time()
        phase_start_valid = search_counters["valid_evals"]
        phase_progress.update({"start_valid": phase_start_valid, "limit": limit, "label": phase_label})
        attempts = 0
        max_attempts = max(limit * 4, limit + 8)
        while (search_counters["valid_evals"] - phase_start_valid) < limit and attempts < max_attempts:
            if time.time() - start_time > 1200:
                print("  Time budget reached; ending phase early.")
                break
            if current_best_ref["params"] and random.random() < 0.25:
                neighbors = _generate_neighbor_configs(current_best_ref["params"], gpu_count, phase=active_phase)
                random.shuffle(neighbors)
                for n_cfg in neighbors:
                    try:
                        if "gpu_set" in n_cfg:
                            n_cfg["gpu_set_idx"] = feasible_gpu_sets.index(n_cfg.get("gpu_set", []))
                        _safe_enqueue_trial(study, n_cfg, "local-neighbor")
                        break
                    except ValueError:
                        continue
            before_failed = len(study.trials)
            try:
                study.optimize(_objective, n_trials=1, catch=(Exception,))
            except Exception as exc:
                # Last-resort guard: tuning must degrade gracefully instead of
                # aborting an expensive run because of Optuna/API edge cases.
                search_counters["invalid_attempts"] += 1
                print(f"  ⚠️  Non-fatal tuning attempt error: {exc}")
                _log_trial_event(log_path, "TRIAL_OPTIMIZE_EXCEPTION", {"phase": active_phase, "error": str(exc)})
            finally:
                if len(study.trials) > before_failed:
                    last = study.trials[-1]
                    if str(getattr(last, "state", "")).endswith("FAIL"):
                        search_counters["invalid_attempts"] += 1
                        err = str(last.user_attrs.get("error", "")) if hasattr(last, "user_attrs") else ""
                        print(f"  ⚠️  Optuna rejected a trial before benchmark; continuing. {err}")
            attempts += 1
        
        # === NEW: Phase Summary ===
        oom_count = sum(1 for t in study.trials if t.user_attrs.get("metrics", {}).get("oom", False))
        complete_state = _optuna_trial_state_complete(optuna)
        success_count = sum(1 for t in study.trials if t.state == complete_state or str(t.state).endswith("COMPLETE"))
        print(f"\n{'='*70}")
        print(f"PHASE {phase_label} SUMMARY:")
        print(f"  Total Optuna attempts: {len(study.trials)}")
        print(f"  Valid unique evals this phase: {search_counters['valid_evals'] - phase_start_valid}/{limit}")
        print(f"  Benchmarks run: {search_counters['benchmarks_run']} | Duplicates reused: {search_counters['duplicates_reused']} | Structural repairs: {search_counters['structural_repairs']}")
        print(f"  Successful: {success_count} | OOM: {oom_count} | Pruned/Failed: {len(study.trials) - success_count - oom_count}")
        
        best_trial = _safe_best_trial(study)
        if best_trial:
            best_score = best_trial.value
            best_params = best_trial.params
            gpu_set_idx = best_params.get("gpu_set_idx", 0)
            gpu_set = feasible_gpu_sets[gpu_set_idx] if gpu_set_idx < len(feasible_gpu_sets) else [0]
            print(f"  Phase best score: {best_score:.2f}")
            print(f"  Best config GPUs: {gpu_set} | tensor_split: {best_trial.user_attrs.get('actual_config', {}).get('tensor_split', 'unknown')}")
        print(f"{'='*70}\n")


    def _phase1_seed_from_params(params: dict) -> dict:
        cpu = os.cpu_count() or 4
        thread_choices = [max(1, cpu // 2), cpu, cpu * 2]
        thread_batch_choices = [cpu, cpu * 2]
        fit_value = params.get("fit", "on")
        if isinstance(fit_value, bool):
            fit_value = "on" if fit_value else "off"
        return {
            "split_mode": _coerce_optuna_choice(params.get("split_mode", "layer"), ["none", "layer", "row"], "layer"),
            "tensor_split_strategy": _coerce_optuna_choice(params.get("tensor_split_strategy", "equal" if gpu_count > 1 else "auto"), _tensor_split_strategy_candidates(), "equal" if gpu_count > 1 else "auto"),
            "main_gpu_raw": max(0, min(7, int(params.get("main_gpu", params.get("main_gpu_raw", 0)) or 0))),
            "n_gpu_layers": _coerce_optuna_choice(params.get("n_gpu_layers", "all"), ["all", "auto"], "all"),
            "fit": _coerce_optuna_choice(fit_value, ["on", "off"], "on"),
            "fit_target": _coerce_optuna_choice(params.get("fit_target", 1024), [1024, 2048, 4096], 1024),
            "batch_size": _coerce_optuna_choice(params.get("batch_size", 2048), [512, 1024, 2048, 4096, 8192], 2048),
            "ubatch_size": _coerce_optuna_choice(params.get("ubatch_size", 512), [256, 512, 1024, 2048], 512),
            "flash_attn": _coerce_optuna_choice(params.get("flash_attn", "auto"), ["on", "auto", "off"], "auto"),
            "kv_offload": bool(params.get("kv_offload", True)),
            "numa": _coerce_optuna_choice(params.get("numa", None), _numa_candidates(), None),
            "op_offload": bool(params.get("op_offload", False)),
            "threads": _coerce_optuna_choice(params.get("threads", cpu), thread_choices, cpu),
            "threads_batch": _coerce_optuna_choice(params.get("threads_batch", cpu), thread_batch_choices, cpu),
        }

    def _safe_enqueue_trial(study, cfg: dict, label: str) -> bool:
        try:
            study.enqueue_trial(cfg)
            return True
        except Exception as exc:
            search_counters["invalid_attempts"] += 1
            print(f"  ⚠️  Skipping invalid queued {label} trial: {exc}")
            _log_trial_event(log_path, "TRIAL_ENQUEUE_SKIPPED", {"phase": active_phase, "label": label, "error": str(exc), "params": cfg})
            return False

    def _enqueue_phase1_baseline_and_neighbors(study) -> None:
        seed = _phase1_seed_from_params(baseline)
        _safe_enqueue_trial(study, seed, "phase1-baseline")
        for key, values in (
            ("flash_attn", ["on", "auto", "off"]),
            ("batch_size", [512, 1024, 2048, 4096, 8192]),
            ("ubatch_size", [256, 512, 1024]),
            ("tensor_split_strategy", _tensor_split_strategy_candidates()),
        ):
            current = seed.get(key)
            for value in values:
                if value != current:
                    n = dict(seed)
                    n[key] = value
                    _safe_enqueue_trial(study, n, f"phase1-neighbor-{key}")
                    break

    def _reset_phase_experiment(phase_label: str, study) -> None:
        """Start an independent Optuna phase with a fresh duplicate cache.

        Each phase owns a different search space, so trials/statistics/cache from
        previous phases must not bias or short-circuit this phase. The best
        configuration is carried only as fixed defaults in the objective.
        """
        evaluated_cache.clear()
        print(f"  ↻ Reset Optuna experiment state for {phase_label}: fresh study id={id(study)}, fresh eval cache.")

    def _speculative_base_raw_config() -> dict:
        best_core = _merge_speculative_phase_defaults(core_best["params"], baseline)
        spec_gpu_set = best_core.get("gpu_set", feasible_gpu_sets[default_gpu_set_idx])
        spec_gpu_indices = spec_gpu_set if spec_gpu_set in feasible_gpu_sets else feasible_gpu_sets[default_gpu_set_idx]
        return {
            "gpu_set_idx": feasible_gpu_sets.index(spec_gpu_indices),
            "split_mode": best_core.get("split_mode", "layer"),
            "tensor_split_strategy": _speculative_tensor_split_strategy(best_core, spec_gpu_indices),
            "main_gpu_raw": best_core.get("main_gpu_raw", 0),
            "n_gpu_layers": best_core.get("n_gpu_layers", "all"),
            "fit": best_core.get("fit", "on"),
            "fit_target": best_core.get("fit_target", 1024),
            "batch_size": best_core.get("batch_size", 2048),
            "ubatch_size": best_core.get("ubatch_size", 512),
            "flash_attn": best_core.get("flash_attn", "auto"),
            "cache_type_k": CACHE_TYPE_FLOOR,
            "cache_type_v": CACHE_TYPE_FLOOR,
            "kv_offload": best_core.get("kv_offload", True),
            "numa": best_core.get("numa", None),
            "op_offload": best_core.get("op_offload", True),
            "threads": best_core.get("threads", os.cpu_count() or 4),
            "threads_batch": best_core.get("threads_batch", os.cpu_count() or 4),
            "model_draft": best_core.get("model_draft") or draft_model_path,
            "draft": int(best_core.get("draft") or 8),
            "ctx_size_draft": int(best_core.get("ctx_size_draft") or 1024),
            "n_gpu_layers_draft": best_core.get("n_gpu_layers_draft", "auto"),
            "cache_type_k_draft": CACHE_TYPE_FLOOR,
            "cache_type_v_draft": CACHE_TYPE_FLOOR,
        }

    def _normalize_raw_config_for_repair(raw_cfg: dict) -> tuple[dict, list[int]]:
        gpu_idx = int(raw_cfg.get("gpu_set_idx", default_gpu_set_idx) or default_gpu_set_idx)
        if gpu_idx < 0 or gpu_idx >= len(feasible_gpu_sets):
            gpu_idx = default_gpu_set_idx
        gpu_indices_local = feasible_gpu_sets[gpu_idx]
        normalized = dict(raw_cfg)
        normalized["gpu_set"] = gpu_indices_local
        normalized["ctx_size"] = search_ctx
        normalized["mmap"] = True
        normalized["tensor_split"] = _tensor_split_from_strategy(raw_cfg["tensor_split_strategy"], gpu_indices_local, gpu_count)
        normalized["main_gpu"] = gpu_indices_local[int(raw_cfg.get("main_gpu_raw", 0) or 0) % len(gpu_indices_local)] if gpu_indices_local else 0
        return normalized, gpu_indices_local

    def _enqueue_speculative_feasible_seeds(study) -> int:
        base_raw = _speculative_base_raw_config()
        base_spec_key = _speculative_knob_key(base_raw)
        measured_reference_is_feasible = not baseline_failed and core_best.get("trial_num") == -1
        enqueued = 0
        first_infeasible_reason = ""
        for candidate in _speculative_candidate_sequence(base_raw):
            if measured_reference_is_feasible and not _speculative_candidate_not_heavier_than_reference(candidate, base_raw):
                continue
            if _speculative_knob_key(candidate) == base_spec_key:
                print("  ✓ SPECULATIVE baseline/core draft config was already measured; using it as reference, not enqueueing a paid trial.")
                continue
            normalized, _ = _normalize_raw_config_for_repair(candidate)
            actual, repair_log = repair_until_feasible(normalized, hw, model.local_path)
            infeasible, infeasible_reason = _speculative_repair_log_infeasible(repair_log)
            spec_ok, spec_reason = _speculative_config_valid_after_repair(normalized, actual)
            if infeasible or not spec_ok:
                if measured_reference_is_feasible:
                    # The real baseline just proved base+draft fits; estimates can
                    # be pessimistic. Allow strictly lighter variants to be tested.
                    actual = dict(normalized)
                else:
                    first_infeasible_reason = first_infeasible_reason or infeasible_reason or spec_reason
                    continue
            if _speculative_knob_key(actual) == base_spec_key:
                print("  ✓ SPECULATIVE baseline/core draft config is already known; using it as reference, not enqueueing a paid trial.")
                continue
            if measured_reference_is_feasible and not _speculative_candidate_not_heavier_than_reference(actual, base_raw):
                first_infeasible_reason = first_infeasible_reason or infeasible_reason or spec_reason
                continue
            if _safe_enqueue_trial(study, _speculative_trial_seed(actual, draft_model_path), "speculative-feasible-seed"):
                enqueued += 1
            if enqueued >= 6:
                break
        if enqueued == 0:
            print(
                "  ⚠️  SPECULATIVE preflight found no draft configuration that fits "
                f"({first_infeasible_reason or 'unknown'}). Skipping phase without burning trials."
            )
        else:
            print(f"  ✓ SPECULATIVE preflight queued {enqueued} feasible draft seed(s).")
        return enqueued

    def _run_speculative_deterministic() -> None:
        nonlocal final_config_params, final_config_metrics, final_config_score
        print(f"\n{'='*70}\nOPTIMIZATION PHASE: SPECULATIVE (deterministic ctx descent; no Optuna)\n{'='*70}")
        base_raw = _speculative_base_raw_config()
        draft_default = int(base_raw.get("draft") or 6)
        base_raw["draft"] = draft_default
        base_raw["n_gpu_layers_draft"] = base_raw.get("n_gpu_layers_draft") or "auto"
        ctx_values = _speculative_ctx_descent_values(search_ctx, int(base_raw.get("ctx_size_draft") or 0))
        reference_metrics = dict(best_so_far["metrics"] or baseline_metrics)
        reference_total_tps = (
            _metric_float(reference_metrics, "screening_total_tokens_s")
            or _metric_float(baseline_metrics, "screening_total_tokens_s")
        )
        if reference_total_tps <= 0.0:
            print(
                "  ⚠️  SPECULATIVE cannot compare fairly: baseline has no short-screening "
                f"total throughput for workload n_predict={RAW_SCREENING_N_PREDICT}, runs={RAW_SCREENING_RUNS}. "
                "Refresh the baseline so it records screening_total_tokens_s; skipping SPECULATIVE to avoid mixing with long 20K/20K."
            )
            return
        total_tps_floor = reference_total_tps * 0.98 if reference_total_tps > 0 else 0.0
        print(f"Speculative fixed draft={draft_default}; trying ctx_size_draft by /2 jumps from ctx/2 downward: {', '.join(str(v) for v in ctx_values[:10])}{'...' if len(ctx_values) > 10 else ''}")
        print(
            "Target: keep the largest ctx_size_draft that preserves the same short-screening total throughput "
            f"(total >= {total_tps_floor:.2f} t/s, reference={reference_total_tps:.2f} t/s)."
        )

        tested = 0
        accepted_equal_or_better = False
        for ctx_draft in ctx_values:
            candidate = dict(base_raw)
            candidate["ctx_size_draft"] = int(ctx_draft)
            normalized, gpu_indices_trial = _normalize_raw_config_for_repair(candidate)
            if _benchmark_params_equivalent(normalized, baseline, api_mode=server_api_mode):
                print(f"  Skipping ctx_size_draft={ctx_draft}: matches measured baseline.")
                continue
            actual = dict(normalized)
            actual_cache_key = _canonical_benchmark_key(actual, api_mode=server_api_mode)
            if actual_cache_key in evaluated_cache:
                print(f"  Skipping ctx_size_draft={ctx_draft}: duplicate cached config.")
                continue

            tested += 1
            print(f"\n[SPECULATIVE deterministic {tested}] Testing ctx_size_draft={ctx_draft}, draft={draft_default}...")
            print(f"  Actual speculative config: {json.dumps({k: actual.get(k) for k in ('model_draft','draft','ctx_size_draft','n_gpu_layers_draft','tensor_split','main_gpu','batch_size','ubatch_size')}, ensure_ascii=False, sort_keys=True)}")
            metrics = run_benchmark(
                model.local_path,
                _prepare_benchmark_params(actual, api_mode=server_api_mode),
                search_ctx,
                gpu_indices_trial,
                api_mode=server_api_mode,
                n_predict=RAW_SCREENING_N_PREDICT,
                runs=RAW_SCREENING_RUNS,
                max_total_s=300.0,
                # SPECULATIVE deterministic search is guarding Prefill now.
                # Do not pass a decode best_tps here, otherwise run_benchmark
                # will early-prune on generation speed before we can compare
                # prefill_tokens_s.
                best_tps=0.0,
                confirm_long_context_if_promising=False,
                log_path=log_path,
                mock=is_mock,
                server_ready_timeout_s=300,
                expected_models_loaded=expected_models_loaded,
            )
            if metrics.get("oom"):
                print(f"  RESULT=OOM at ctx_size_draft={ctx_draft}; lowering draft context.")
                continue
            if metrics.get("crash") or metrics.get("timeout"):
                err = str(metrics.get("error") or metrics.get("load_reason") or "")
                if "early pruning" in err.lower():
                    print(f"  RESULT=PRUNED at ctx_size_draft={ctx_draft}; candidate is clearly slower, lowering draft context. reason={err[:160]}")
                    continue
                if any(token in err.lower() for token in ("out of memory", "oom", "cuda", "alloc", "memory")):
                    print(f"  RESULT=MEMORY_ERROR at ctx_size_draft={ctx_draft}; lowering draft context. error={err[:160]}")
                    continue
                print(f"  RESULT=ERROR at ctx_size_draft={ctx_draft}; stopping deterministic speculative search. error={err[:160]}")
                break

            score = score_server_performance(metrics, search_ctx, len(gpu_indices_trial)) if server_api_mode else score_performance(metrics, search_ctx, len(gpu_indices_trial))
            evaluated_cache[actual_cache_key] = {"metrics": dict(metrics), "score": score, "params": dict(actual), "trial": f"spec-deterministic-{tested}", "source": "speculative-deterministic"}
            result_key = _phase_result_key(actual, "SPECULATIVE", api_mode=server_api_mode)
            phase_results[result_key] = {
                "phase": "SPECULATIVE",
                "role": "best",
                "score": score,
                "metrics": dict(metrics),
                "params": _params_for_observation(actual),
                "benchmark_key": actual_cache_key,
                "trial": f"spec-deterministic-{tested}",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            print(f"  RESULT=SUCCESS (Score: {score:.2f})")
            _print_success_trial_metrics_block(metrics, score, gpu_memory_budget_mib)
            total_tps = _total_tokens_s(metrics)
            if total_tps >= total_tps_floor:
                accepted_equal_or_better = True
                print(
                    "  ✅ Largest tested ctx_size_draft preserves comparable short-screening throughput "
                    f"(total {total_tps:.2f} >= {total_tps_floor:.2f} t/s)."
                )
                final_config_params = dict(actual)
                final_config_metrics = dict(metrics)
                final_config_score = score
                print(
                    "  SPECULATIVE accepted for final config merge. It will not be compared against the long prompt here; "
                    "the final config is accumulated phase-by-phase."
                )
                # Because ctx_values are descending, this is the largest draft
                # context that preserves total throughput.
                break
            print(
                "  Total throughput is below reference tolerance; lowering ctx_size_draft and continuing "
                f"(total {total_tps:.2f} < {total_tps_floor:.2f} t/s)."
            )
        if tested == 0:
            print("  No non-baseline SPECULATIVE variants to test.")
        elif not accepted_equal_or_better:
            print("  No SPECULATIVE ctx_size_draft preserved baseline/core total throughput.")
        print(f"{'='*70}\n")

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    full_study_trials = []
    study_p1 = optuna.create_study(direction="maximize", sampler=_make_sampler(1))
    if server_phase_only:
        print("⏭️  Skipping CORE phase: server phase-only continuation already requested.")
    elif _ask_run_phase(args, "CORE", "y"):
        _reset_phase_experiment("CORE", study_p1)
        if not baseline_failed:
            evaluated_cache[baseline_cache_key] = {
                "metrics": dict(baseline_metrics),
                "score": baseline_score,
                "params": dict(baseline),
                "trial": -1,
                "source": "baseline",
            }
        _enqueue_phase1_baseline_and_neighbors(study_p1)
        active_phase = 1
        phase1_search_keys = tuple(sorted(PROBED_TUNER_KEYS))
        _run_optimization_loop(study_p1, "CORE", p1_limit, best_so_far, phase1_search_keys)
        print(f"CORE base for later phases: score={float(core_best['score'] or baseline_score):.2f} params_source={'phase1-best' if core_best['trial_num'] != -1 else 'baseline'}")
    else:
        print("⏭️  Skipping CORE phase by user request; later phases will use baseline/core best available.")

    full_study_trials = list(study_p1.trials)

    if is_speculative and not server_phase_only:
        active_phase = 2
        if _ask_run_phase(args, "SPECULATIVE", "y"):
            if core_best.get("trial_num") == -1 and not baseline_failed:
                evaluated_cache[baseline_cache_key] = {
                    "metrics": dict(baseline_metrics),
                    "score": baseline_score,
                    "params": dict(baseline),
                    "trial": -1,
                    "source": "baseline",
                }
            _run_speculative_deterministic()
        else:
            print("⏭️  Skipping SPECULATIVE phase by user request.")
    elif is_speculative and server_phase_only:
        print("⏭️  Skipping SPECULATIVE phase: server phase-only continuation already requested.")

    if server_api_mode:
        active_phase = 3 if is_speculative else 2
        study_p2 = optuna.create_study(direction="maximize", sampler=_make_sampler(3))
        run_server_phase = True if server_phase_only else _ask_run_phase(args, "SERVER", "y")
        if run_server_phase:
            _reset_phase_experiment("SERVER", study_p2)
            if best_so_far["params"]:
                try:
                    best_p2_seed = _server_trial_seed(best_so_far["params"] or baseline)
                    _safe_enqueue_trial(study_p2, best_p2_seed, "server-best-seed")
                except ValueError: pass
            phase_server_search_keys = _phase2_tuner_keys()
            _run_optimization_loop(study_p2, "SERVER", server_limit, best_so_far, phase_server_search_keys)
            best = _safe_best_trial(study_p2)
        else:
            print("⏭️  Skipping SERVER phase by user request.")
        full_study_trials = full_study_trials + list(study_p2.trials)
    elif not is_speculative:
        best = _safe_best_trial(study_p1)
        full_study_trials = list(study_p1.trials)

    if best_so_far["score"] is None:
        best_so_far["score"] = baseline_score
        best_so_far["metrics"] = baseline_metrics
        best_so_far["params"] = baseline

    best_params = final_config_params or best_so_far["params"] or baseline
    best_metrics = final_config_metrics or best_so_far["metrics"] or baseline_metrics
    best_score = final_config_score if final_config_score is not None else (best_so_far["score"] if best_so_far["score"] is not None else baseline_score)
    
    baseline_decode = baseline_metrics.get('decode_tokens_s', 0.0)
    best_decode = best_metrics.get('decode_tokens_s', 0.0)
    decode_improvement_pct = ((best_decode - baseline_decode) / max(baseline_decode, 0.001)) * 100 if baseline_decode > 0 else 0

    print("\n" + "=" * 70)
    print("AUTO-PERFORMANCE TUNING COMPLETED")
    print("=" * 70)
    print("\n📊 BASELINE (Default Configuration)")
    print("-" * 70)
    print(f"  Prefill throughput:    {float(baseline_metrics.get('prefill_tokens_s') or 0.0):.2f} tokens/s")
    print(f"  Decode throughput:     {float(baseline_metrics.get('decode_tokens_s') or 0.0):.2f} tokens/s")
    print(f"  Total throughput:      {_total_tokens_s(baseline_metrics):.2f} tokens/s")
    print(f"  Overall score:         {baseline_score:.2f}")

    print("\n🏆 BEST CONFIGURATION (Optimized)")
    print("-" * 70)
    print(f"  Prefill throughput:    {float(best_metrics.get('prefill_tokens_s') or 0.0):.2f} tokens/s")
    print(f"  Decode throughput:     {float(best_metrics.get('decode_tokens_s') or 0.0):.2f} tokens/s")
    print(f"  Total throughput:      {_total_tokens_s(best_metrics):.2f} tokens/s")
    print(f"  Overall score:         {best_score:.2f}")
    final_improvement = _score_improvement_percent(best_score, baseline_score) if baseline_score is not None else 0.0
    print(f"  Improvement vs baseline: {final_improvement:.2f}%")

    minimum_gpu_check = _minimum_gpu_count_for_config(model.local_path, {**best_params, "ctx_size": search_ctx}, hw, gpu_count)
    print("\n🧪 MINIMUM GPU FEASIBILITY CHECK")
    print("-" * 70)
    if minimum_gpu_check.get("feasible"):
        print(
            "  Estimated minimum GPUs: "
            f"{minimum_gpu_check['gpu_count']} | set={minimum_gpu_check['gpu_set']} | "
            f"tensor_split={minimum_gpu_check.get('tensor_split')} | "
            f"VRAM≈{float(minimum_gpu_check.get('estimated_vram_mib') or 0.0):.0f}/"
            f"{float(minimum_gpu_check.get('budget_mib') or 0.0):.0f} MiB "
            f"({float(minimum_gpu_check.get('load_ratio') or 0.0) * 100:.1f}%)"
        )
    else:
        print(
            "  Could not prove a smaller feasible GPU set by VRAM estimate; "
            f"best estimate set={minimum_gpu_check.get('gpu_set')} ratio="
            f"{float(minimum_gpu_check.get('load_ratio') or 0.0) * 100:.1f}%."
        )
    print("  Note: this is independent of actual prompt length as long as requests stay within configured ctx.")

    profile_row = {
        "profile_key": profile_key,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model.model_id,
        "repo": model.repo_id,
        "quant": model.quant,
        "llama_cpp_version": _llama_cpp_version(server_bin),
        "hardware": hw,
        "baseline": {
            "phase": baseline_phase_label,
            "metrics": baseline_metrics,
            "trial_value": baseline_score,
            "params": _params_for_observation(baseline),
            "benchmark_key": baseline_cache_key,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "score_schema_version": AUTO_PERF_SCORE_SCHEMA_VERSION,
        },
        "best": {"metrics": best_metrics, "trial_value": best_score, "params": _params_for_observation(best_params), "minimum_gpu_check": minimum_gpu_check},
        "phase_results": phase_results,
        "tuning_policy": {"ctx_size_floor": search_ctx, "ctx_size_policy": CTX_SIZE_POLICY, "probed_flags": list(PROBED_TUNER_KEYS), "pruned_flags": sorted(PRUNED_TUNER_KEYS)},
    }
    written_final = _upsert_profile(profile_row)
    print(f"\nPerfil guardado en: {str((written_final or AUTO_PERF_PROFILES_PATH).resolve())}")

    if is_mock: return 0

    if not server_api_mode:
        run_phase2 = _ask_yes_no(args, "¿Quieres ejecutar ahora la fase 2 de servidor?", "n")
        if run_phase2:
            phase2_args = type(args)(**vars(args))
            phase2_args.server_api = True
            phase2_args.server_phase_only = True
            phase2_args._inherited_best_params = dict(best_params or {})
            phase2_args._inherited_best_metrics = dict(best_metrics or {})
            phase2_args._inherited_best_score = best_score
            return run_auto_performance(phase2_args)

    has_real_final_improvement = _is_real_score_improvement(best_score, baseline_score)
    if not has_real_final_improvement:
        print("\nNo hay mejora real frente al baseline; no se recomienda guardar esta configuración.")
    apply_config = _ask_yes_no(args, "¿Aplicar la mejor configuracion al catalogo ahora?", "n")
    if apply_config:
        items, _ = load_catalog_with_diagnostics(catalog_path)
        for item in items:
            if item.model_id == model.model_id:
                item.server_overrides = _catalog_server_overrides_for_apply(item.server_overrides, best_params)
                item.ctx_size = int(search_ctx)
                break
        save_catalog(catalog_path, items)
        print(f"Configuracion guardada en el catalogo: {catalog_path.resolve()}")
        _refresh_llamaswap_config_after_auto_perf(args, items)
    else:
        print("Operacion cancelada.")

    return 0
