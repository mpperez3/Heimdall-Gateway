#!/usr/bin/env python3
"""
Manage GGUF models for llama-swap + llama-server.
Ollama-style: Central manager + Multi-user client.
Supports sharded models and professional terminal animations.
"""

from __future__ import annotations

import argparse
import base64
import errno
import hashlib
import json
import math
import os
import re
import shutil
import binascii
import zlib
import subprocess
import sys
import time
import threading
import itertools
import socket
import uuid
import concurrent.futures
import pwd
import struct
import tempfile
import signal
import termios
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

# Dependencies
try:
    import yaml
    import requests
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
except ImportError:
    print("Error: Missing dependencies. Run: pip install requests pyyaml huggingface_hub")
    sys.exit(1)

try:
    import readline
except ImportError:
    readline = None

# Paths & Constants
def _load_installed_env() -> None:
    candidates = [
        Path.home() / ".config/llamacpp-superserver/llamacpp-superserver.env",
        Path("/etc/llamacpp-superserver/llamacpp-superserver.env"),
        Path.home() / ".config/llamacpp/llamacpp-stack.env",
        Path("/etc/llamacpp/llamacpp-stack.env"),
    ]
    for env_file in candidates:
        if not env_file.exists():
            continue
        try:
            for raw in env_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
            break
        except Exception:
            continue


_load_installed_env()


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


SOCKET_PATH = os.environ.get("LLAMACPP_MANAGER_SOCKET", "/run/llamacpp-superserver/manager.sock")
DEFAULT_MODELS_DIR = _env_path("LLAMACPP_MODELS", "/workvols/data3/LLAMACPP_MODELS")
DEFAULT_CONFIG_PATH = _env_path("LLAMACPP_CONFIG", "/var/lib/llamacpp-superserver/config.yaml")
DEFAULT_CATALOG_PATH = _env_path("LLAMACPP_CATALOG", "/var/lib/llamacpp-superserver/catalog.json")
DEFAULT_SERVER_CONFIG_PATH = _env_path(
    "LLAMACPP_SERVER_CONFIG",
    "/etc/llamacpp-superserver/llamacpp-superserver.json"
    if os.geteuid() == 0
    else str(Path.home() / ".config/llamacpp-superserver/llamacpp-superserver.json"),
)
ALTERNATE_SERVER_CONFIG_BASENAME = "conf.json"
DEFAULT_SERVICE_NAME = os.environ.get("LLAMACPP_SERVICE_NAME", "llamaswap")
CLI_COMMAND = "llamacpp-superserver"
LEGACY_CLI_COMMAND = "llamacpp-server"
MANAGER_SERVICE_NAME = "llamacpp-superserver-manager"
SWAP_SERVICE_NAME = "llamacpp-superserver-swap"
DEFAULT_LLAMA_SERVER = _env_path("LLAMA_SERVER_BIN", "/opt/llamacpp-superserver/llama.cpp/build/bin/llama-server")
def _is_vllm_backend() -> bool:
    backend = os.environ.get("LLAMACPP_BACKEND")
    # Debug print to stderr so it shows up in logs even if stdout is captured
    if os.environ.get("DEBUG_LLAMACPP"):
        print(f"DEBUG: _is_vllm_backend check. LLAMACPP_BACKEND='{backend}'", file=sys.stderr)
    return backend == "vllm-beta"

DEFAULT_CTX_SIZE = 8192
try:
    DEFAULT_API_CTX_FACTOR = float(
        os.environ.get("LLAMACPP_API_CTX_FACTOR", os.environ.get("LLAMACPP_CTX_DISPLAY_RATIO", "0.5"))
    )
except ValueError:
    DEFAULT_API_CTX_FACTOR = 0.5
DEFAULT_N_GPU_LAYERS = 999
DEFAULT_IDLE_TTL = int(os.environ.get("LLAMACPP_IDLE_TTL", os.environ.get("LLAMACPP_DEFAULT_TTL", "300")))


def _default_ctx_update_command(model_id: str) -> str:
    candidate = str(model_id or "").strip()
    if candidate:
        return f"{CLI_COMMAND} update --auto --model-id {candidate}"
    return f"{CLI_COMMAND} update --auto"


def _emit_default_ctx_update_hint(model_id: str, ctx_size: int | None, default_ctx: int, progress_callback = None) -> None:
    try:
        current_ctx = int(ctx_size) if ctx_size is not None else None
        expected_default = int(default_ctx)
    except Exception:
        return
    if current_ctx != expected_default:
        return
    _emit_message(
        f"{model_id} is running with default ctx {expected_default}. To auto-tune it now, run: {_default_ctx_update_command(model_id)}",
        progress_callback,
    )


def detect_cuda_device_count() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return 0
    return len([line for line in result.stdout.splitlines() if line.strip()])


def default_tensor_split() -> str:
    gpu_count = detect_cuda_device_count()
    if gpu_count <= 0:
        return "1"
    return ",".join(["1"] * gpu_count)


DEFAULT_TENSOR_SPLIT = default_tensor_split()
DEFAULT_START_PORT = 18080
DEFAULT_PUBLIC_HOST = os.environ.get("LLAMACPP_PUBLIC_HOST", "127.0.0.1")
DEFAULT_PUBLIC_PORT = int(os.environ.get("LLAMACPP_PUBLIC_PORT", "11437"))
DEFAULT_API_PORT = int(os.environ.get("LLAMACPP_API_PORT", str(DEFAULT_PUBLIC_PORT - 1)))
DEFAULT_REQUESTS_LOG_PATH = _env_path(
    "LLAMACPP_REQUESTS_LOG",
    "/var/lib/llamacpp-superserver/api-requests.log"
    if os.geteuid() == 0
    else str(Path.home() / ".local/state/llamacpp-superserver/api-requests.log"),
)
MODEL_ACTIVITY_LOCK = threading.Lock()
MODEL_ACTIVITY: dict[str, dict[str, float | str]] = {}
LAST_ACTIVITY_MODEL_ID = ""

LLAMASWAP_CONFIG_HEADER = (
    "# llamacpp-superserver config.yaml\n"
    "# Purpose: llama-swap routing and per-model command map generated from catalog.\n"
    "# This file is regenerated by install/update operations.\n"
    "# Example:\n"
    "#   models:\n"
    "#     my-model-id:\n"
    "#       cmd: /opt/llama-server --model /models/my.gguf --ctx-size 65536\n"
    "#       checkEndpoint: /health\n"
    "#       ttl: 300\n\n"
)


DEBUG_GATE_LOCK = threading.Lock()
DEBUG_GATE_STATE = {"active": False, "owner": "", "expires_at": 0.0}
DEBUG_GATE_TTL_S = 10.0


def _enable_debug_gate(owner: str, ttl_s: float | None = None) -> bool:
    if not owner:
        return False
    ttl_value = float(ttl_s or DEBUG_GATE_TTL_S)
    ttl_value = max(2.0, ttl_value)
    with DEBUG_GATE_LOCK:
        if DEBUG_GATE_STATE["active"] and DEBUG_GATE_STATE["owner"] not in {"", owner}:
            return False
        DEBUG_GATE_STATE["active"] = True
        DEBUG_GATE_STATE["owner"] = owner
        DEBUG_GATE_STATE["expires_at"] = time.monotonic() + ttl_value
        return True


def _refresh_debug_gate(owner: str, ttl_s: float | None = None) -> bool:
    return _enable_debug_gate(owner, ttl_s)


def _disable_debug_gate(owner: str | None = None) -> bool:
    with DEBUG_GATE_LOCK:
        if owner and DEBUG_GATE_STATE["owner"] not in {"", owner}:
            return False
        DEBUG_GATE_STATE["active"] = False
        DEBUG_GATE_STATE["owner"] = ""
        DEBUG_GATE_STATE["expires_at"] = 0.0
    try:
        from llamacpp_stack.debug_manager import DEBUG_SESSION_MANAGER

        DEBUG_SESSION_MANAGER.stop_session()
    except Exception:
        pass
    return True


def _is_debug_gate_active() -> bool:
    expired = False
    with DEBUG_GATE_LOCK:
        if not DEBUG_GATE_STATE["active"]:
            return False
        if DEBUG_GATE_STATE["expires_at"] and time.monotonic() > float(DEBUG_GATE_STATE["expires_at"]):
            expired = True
            DEBUG_GATE_STATE["active"] = False
            DEBUG_GATE_STATE["owner"] = ""
            DEBUG_GATE_STATE["expires_at"] = 0.0
    if expired:
        _disable_debug_gate()
        return False
    return True


def _ensure_server_config_metadata(payload: dict[str, object]) -> dict[str, object]:
    meta = payload.get("_meta")
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("purpose", "Global superserver settings consumed by CLI/services.")
    meta.setdefault(
        "note",
        "Per-model context source of truth is catalog/config.yaml; this file stores global defaults and service-level settings.",
    )
    meta.setdefault(
        "example",
        {
            "idle_ttl": 300,
            "api_port": 11436,
            "api_ctx_factor": 0.5,
            "llama_server_defaults": {
                "ctx_size": 65536,
                "n_gpu_layers": 999,
                "keep": 512,
                "mirostat": 2,
                "mirostat_ent": 4.5,
                "mirostat_lr": 0.1,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
            },
        },
    )
    meta.setdefault(
        "service_restart_help",
        {
            "system_mode": f"sudo systemctl restart {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}",
            "user_mode": f"systemctl --user restart {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}",
        },
    )
    payload["_meta"] = meta
    return payload
CATALOG_CACHE: dict[tuple[str, tuple[str, int, int]], tuple[int, int, list["ManagedModel"]]] = {}


def _server_config_signature(server_config_path: Path | None = None) -> tuple[str, int, int]:
    path = Path(server_config_path or DEFAULT_SERVER_CONFIG_PATH).expanduser()
    try:
        resolved = str(path.resolve())
    except Exception:
        resolved = str(path)
    try:
        stat = path.stat()
        return (resolved, int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        return (resolved, -1, -1)


def _catalog_cache_key(path: Path, server_config_path: Path | None = None) -> tuple[str, tuple[str, int, int]]:
    return (str(path), _server_config_signature(server_config_path))


def _clear_catalog_cache(path: Path) -> None:
    path_key = str(path)
    for key in list(CATALOG_CACHE.keys()):
        if key[0] == path_key:
            CATALOG_CACHE.pop(key, None)

def manager_unavailable_error(exc: Exception) -> RuntimeError:
    return RuntimeError(f"Could not connect to manager: {exc}. Is {MANAGER_SERVICE_NAME} running?")


class Spinner:
    """Universal ASCII Spinner for maximum terminal compatibility."""
    def __init__(self, label="assistant: "):
        self.label = label
        self._frames = itertools.cycle(["|", "/", "-", "\\"])
        self._running = False
        self._thread = None
        self.cyan = "\033[36;1m" 
        self.reset = "\033[0m"

    def _spin(self):
        sys.stdout.write("\033[?25l")
        while self._running:
            sys.stdout.write(f"\r{self.label}{self.cyan}{next(self._frames)}{self.reset}")
            sys.stdout.flush()
            time.sleep(0.1)

    def start(self):
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        sys.stdout.write(f"\r{self.label}\033[K")
        sys.stdout.write("\033[?25h") 
        sys.stdout.flush()
        self._thread = None

class LoadingBar:
    """Indeterminate loading bar for model warmup."""
    def __init__(self, label="Loading model: ", width=24):
        self.label = label
        self.width = width
        self._running = False
        self._thread = None
        self.cyan = "\033[36;1m"
        self.reset = "\033[0m"

    def _render_frame(self, pos):
        cells = ["-"] * self.width
        for idx in range(4):
            cell = pos + idx
            if 0 <= cell < self.width:
                cells[cell] = "="
        return "[" + "".join(cells) + "]"

    def _spin(self):
        sys.stdout.write("\033[?25l")
        travel = max(1, self.width - 3)
        pos = 0
        direction = 1
        while self._running:
            bar = self._render_frame(pos)
            sys.stdout.write(f"\r{self.label}{self.cyan}{bar}{self.reset}")
            sys.stdout.flush()
            time.sleep(0.08)
            pos += direction
            if pos >= travel:
                pos = travel
                direction = -1
            elif pos <= 0:
                pos = 0
                direction = 1

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        sys.stdout.write(f"\r{self.label}\033[K")
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        self._thread = None

@dataclass
class ManagedModel:
    model_id: str
    repo_id: str
    quant: str | None
    filename: str
    local_path: str
    mmproj_filename: str | None = None
    mmproj_path: str | None = None
    load_capabilities: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    ctx_size: int = DEFAULT_CTX_SIZE
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS
    tensor_split: str = DEFAULT_TENSOR_SPLIT
    host: str = "127.0.0.1"
    jinja: bool = True
    ttl: int = DEFAULT_IDLE_TTL
    description: str = ""
    # Speculative/draft model metadata
    speculative: bool = False
    spec_variant_of: str | None = None
    spec_meta: dict[str, object] = field(default_factory=dict)
    auto_ctx_failed: bool = False
    auto_ctx_error: str = ""
    ctx_probe_read_s: float | None = None
    ctx_probe_tokens_s: float | None = None
    ctx_probe_totals_s: float | None = None
    ctx_probe_latency_ms: float | None = None
    ctx_probe_speed_tps: float | None = None
    ctx_probe_kv_gb: float | None = None
    ctx_probe_prompt_tokens: int | None = None
    server_overrides: dict[str, object] = field(default_factory=dict)


@dataclass
class ProbeTraceMetrics:
    model_buffers_mib: dict[int, float] = field(default_factory=dict)
    kv_buffers_mib: dict[int, float] = field(default_factory=dict)
    compute_buffers_mib: dict[int, float] = field(default_factory=dict)
    projector_gpu: int | None = None
    oom_gpu: int | None = None
    oom_requested_mib: float | None = None

def parse_hf_input(raw: str):
    v = raw.strip()
    if not v: raise ValueError("Empty HF reference.")
    v = re.sub(r"^https?://", "", v)
    v = re.sub(r"^(www\.)?", "", v)
    if v.startswith("hf.co/"): v = v[6:]
    elif v.startswith("huggingface.co/"): v = v[15:]
    v = v.strip("/")
    if v.startswith("ollama run "): v = v[11:].strip()
    quant = None
    if ":" in v: v, quant = v.rsplit(":", 1)
    parts = v.split("/")
    if len(parts) != 2: raise ValueError("Expected 'org/repo' or 'hf.co/org/repo[:quant]'.")
    return f"{parts[0]}/{parts[1]}", (quant.strip() if quant else None)

def normalize_tensor_split(value: str | None) -> str:
    normalized = (value or "").strip()
    gpu_count = detect_cuda_device_count()
    if not normalized:
        return default_tensor_split()
    if gpu_count <= 0:
        return normalized
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if not parts:
        return default_tensor_split()
    if len(parts) == gpu_count:
        return ",".join(parts)
    if all(part == "1" for part in parts):
        return ",".join(["1"] * gpu_count)
    return ",".join(parts)


def _normalize_bool_flag(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _is_vllm_backend() -> bool:
    return os.environ.get("LLAMACPP_BACKEND") == "vllm-beta"


# Cache of parsed help flags per server binary path
_SERVER_FLAG_CACHE: dict[str, set[str]] = {}


def _server_help_env(server_path: Path | str | None) -> dict[str, str]:
    env = os.environ.copy()
    try:
        p = Path(server_path) if server_path is not None else Path(DEFAULT_LLAMA_SERVER)
        resolved = p.resolve()
        lib_dirs = [
            p.parent,
            resolved.parent,
            p.parent.parent / "lib",
            p.parent.parent / "lib64",
            p.parent / "cuda" / "lib",
            p.parent.parent / "cuda" / "lib",
            p.parent / "nccl" / "lib",
            p.parent.parent / "nccl" / "lib",
            p.parent.parent / "build" / "bin",
            Path.home() / ".local" / "opt" / "llamacpp-superserver" / "cuda" / "lib",
            Path.home() / ".local" / "opt" / "llamacpp-superserver" / "nccl" / "lib",
        ]
        existing = env.get("LD_LIBRARY_PATH", "")
        parts = [str(d) for d in lib_dirs if d.exists()]
        if existing:
            parts.append(existing)
        if parts:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(parts))
    except Exception:
        pass
    return env


def get_server_supported_flags(server_path: Path | str | None) -> set[str]:
    """Return set of supported flags (e.g. '--mul-mat-q', '-fit').

    This runs the server binary with `--help` once and caches the result.
    If the binary cannot be executed, returns an empty set.
    """
    try:
        p = Path(server_path) if server_path is not None else Path(DEFAULT_LLAMA_SERVER)
        key = str(p)
    except Exception:
        key = str(server_path or "")
    if key in _SERVER_FLAG_CACHE:
        return _SERVER_FLAG_CACHE[key]
    flags: set[str] = set()
    try:
        # Prefer --help, but some binaries accept -h too
        proc = subprocess.run([key, "--help"], capture_output=True, text=True, timeout=8, env=_server_help_env(p))
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except Exception:
        _SERVER_FLAG_CACHE[key] = set()
        return set()
    # Find long and multi-letter single-dash flags
    for match in re.findall(r"(--[A-Za-z0-9-]+)", text):
        flags.add(match)
    for match in re.findall(r"(?<!-)(-[A-Za-z][A-Za-z0-9-]+)\b", text):
        flags.add(match)
    _SERVER_FLAG_CACHE[key] = flags
    return flags


def server_supports_flag(server_path: Path | str | None, flag: str) -> bool:
    try:
        flags = get_server_supported_flags(server_path)
        return flag in flags
    except Exception:
        return False

def normalize_server_overrides(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, object] = {}
    # Internal orchestration fields used by auto-tuning; should never reach CLI
    internal_keys = {"gpu_set", "gpu_set_idx", "ts_strategy", "tensor_split_strategy", "main_gpu_raw", "auto_performance"}
    for raw_key, raw_val in value.items():
        key = str(raw_key).strip().lower().replace("-", "_")
        if not key:
            continue
        # Filter out internal orchestration fields
        if key in internal_keys:
            continue
        if key == "speculative_defaults":
            if isinstance(raw_val, dict):
                nested: dict[str, object] = {}
                for nested_key, nested_val in raw_val.items():
                    nested_name = str(nested_key).strip().lower().replace("-", "_")
                    if not nested_name:
                        continue
                    if nested_name == "draft_max":
                        nested_name = "draft"
                    nested[nested_name] = nested_val
                normalized[key] = nested
            continue
        if key == "draft_max":
            key = "draft"
        if key == "gpu_layers":
            key = "n_gpu_layers"
            if isinstance(raw_val, str) and raw_val.strip().lower() == "all":
                continue
        if key == "split_mode" and str(raw_val).strip().lower() == "layer":
            continue
        if key == "mmap":
            bool_val = _normalize_bool_flag(raw_val)
            if bool_val is True:
                continue
            if bool_val is not None:
                normalized[key] = bool_val
            continue
        if key in {"mul_mat_q"}:
            bool_val = _normalize_bool_flag(raw_val)
            if bool_val is not None:
                normalized[key] = bool_val
            continue
        if key == "fit":
            bool_val = _normalize_bool_flag(raw_val)
            if bool_val is not None:
                normalized[key] = bool_val
                continue
            if isinstance(raw_val, str):
                sval = raw_val.strip()
                if sval:
                    normalized[key] = sval
            continue
        if key in {"fitt", "fitc"}:
            if isinstance(raw_val, bool):
                normalized[key] = raw_val
                continue
            try:
                normalized[key] = int(raw_val)
            except (TypeError, ValueError):
                try:
                    normalized[key] = int(float(raw_val))
                except (TypeError, ValueError):
                    continue
            continue
        if key == "n_gpu_layers_draft":
            if isinstance(raw_val, str):
                sval = raw_val.strip().lower()
                if sval in {"all", "auto"}:
                    normalized[key] = sval
                    continue
            try:
                normalized[key] = int(raw_val)
            except (TypeError, ValueError):
                continue
            continue
        if key in {
            "ctx_size",
            "n_gpu_layers",
            "batch_size",
            "ubatch_size",
            "threads",
            "threads_batch",
            "fit_target",
            "keep",
            "mirostat",
            "draft",
            "draft_min",
            "ctx_size_draft",
            "grp_attn_n",
            "parallel",
            "main_gpu",
            "ctx_checkpoints",
            "cache_ram",
            "n_cpu_moe",
        }:
            try:
                normalized[key] = int(raw_val)
            except (TypeError, ValueError):
                continue
            continue
        if key in {"mirostat_ent", "mirostat_lr", "draft_p_min", "defrag_threshold"}:
            try:
                normalized[key] = float(raw_val)
            except (TypeError, ValueError):
                continue
            continue
        if key == "flash_attn":
            # Accept boolean-like and string values. Keep raw string values (e.g. "auto")
            if isinstance(raw_val, str):
                normalized[key] = raw_val.strip()
            else:
                bool_val = _normalize_bool_flag(raw_val)
                if bool_val is not None:
                    normalized[key] = bool_val
            continue
        if key in {"kv_offload", "cont_batching", "op_offload", "cpu_moe", "kv_unified", "cache_idle_slots", "direct_io"}:
            bool_val = _normalize_bool_flag(raw_val)
            if bool_val is not None:
                normalized[key] = bool_val
            continue
        if key == "tensor_split":
            normalized[key] = normalize_tensor_split(str(raw_val))
            continue
        if key == "numa":
            # numa can be None (omit flag) or a string like "distribute"/"isolate"
            if raw_val is None or (isinstance(raw_val, str) and raw_val.strip().lower() == "none"):
                continue  # Skip: None means omit the flag
            normalized[key] = str(raw_val).strip()
            continue
        if key in {"split_mode", "cache_type_k", "cache_type_v", "host", "model_draft", "hf_repo_draft", "reasoning_format", "chat_template_file", "chat_template", "device"}:
            normalized[key] = str(raw_val).strip()
            continue
        # Generic passthrough: include other override keys as-is so callers
        # can provide custom flags (e.g. custom_flag -> --custom-flag).
        try:
            normalized[key] = raw_val
        except Exception:
            normalized[key] = str(raw_val)
        continue
    return normalized


def _is_equal_weight_tensor_split(value: str) -> bool:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) <= 1:
        return False
    return all(part == parts[0] for part in parts)


def preferred_tensor_split(model: ManagedModel | None, value: str | None = None) -> str:
    normalized = normalize_tensor_split(value)
    gpu_count = detect_cuda_device_count()
    if gpu_count <= 1 or model is None or not model.mmproj_path:
        return normalized
    if not _is_equal_weight_tensor_split(normalized):
        return normalized
    weights = [0.75] + [1.0] * (gpu_count - 1)
    total = sum(weights)
    preferred = [f"{weight / total:.4f}".rstrip("0").rstrip(".") for weight in weights]
    return ",".join(preferred)


def normalize_aliases(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for raw in values or []:
        alias = str(raw).strip()
        if alias and alias not in normalized:
            normalized.append(alias)
    return normalized


def _to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        return float(normalized)
    except (TypeError, ValueError):
        return None


def _to_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        return int(normalized)
    except (TypeError, ValueError):
        return None


def _format_ctx_probe_gb(value: float | None) -> str:
    if value is None:
        return "NC"
    return f"{value:.2f}"


def _format_ctx_probe_speed(value: float | None) -> str:
    if value is None:
        return "NC"
    return f"{value:.1f} tok/s"


def _format_ctx_probe_rate(value: float | None) -> str:
    return _format_ctx_probe_speed(value)


def _format_ctx_probe_latency(value: float | None) -> str:
    if value is None:
        return "NC"
    return f"{value:.0f} ms"


def clear_ctx_probe_metrics(model: ManagedModel) -> None:
    model.ctx_probe_read_s = None
    model.ctx_probe_tokens_s = None
    model.ctx_probe_totals_s = None
    model.ctx_probe_latency_ms = None
    model.ctx_probe_speed_tps = None
    model.ctx_probe_kv_gb = None
    model.ctx_probe_prompt_tokens = None


def apply_ctx_probe_metrics(model: ManagedModel, info: dict[str, object] | None) -> None:
    payload = info or {}
    model.ctx_probe_read_s = _to_float_or_none(payload.get("probe_read_s"))
    model.ctx_probe_tokens_s = _to_float_or_none(payload.get("probe_tokens_s"))
    model.ctx_probe_totals_s = _to_float_or_none(payload.get("probe_totals_s"))
    model.ctx_probe_latency_ms = _to_float_or_none(payload.get("probe_latency_ms"))
    model.ctx_probe_speed_tps = _to_float_or_none(payload.get("probe_speed_tps"))
    if model.ctx_probe_speed_tps is None:
        model.ctx_probe_speed_tps = model.ctx_probe_totals_s
    if model.ctx_probe_totals_s is None:
        model.ctx_probe_totals_s = model.ctx_probe_speed_tps
    model.ctx_probe_kv_gb = _to_float_or_none(payload.get("selected_ctx_gb"))
    model.ctx_probe_prompt_tokens = _to_int_or_none(payload.get("probe_prompt_tokens"))


def _ctx_probe_api_metrics(model: ManagedModel) -> dict[str, object]:
    read_s = _to_float_or_none(model.ctx_probe_read_s)
    tokens_s = _to_float_or_none(model.ctx_probe_tokens_s)
    totals_s = _to_float_or_none(model.ctx_probe_totals_s)
    latency_ms = _to_float_or_none(model.ctx_probe_latency_ms)
    speed_tps = _to_float_or_none(model.ctx_probe_speed_tps)
    if speed_tps is None:
        speed_tps = totals_s
    if totals_s is None:
        totals_s = speed_tps
    kv_gb = _to_float_or_none(model.ctx_probe_kv_gb)
    prompt_tokens = _to_int_or_none(model.ctx_probe_prompt_tokens)
    return {
        "ctx_probe_read_s": read_s,
        "ctx_probe_tokens_s": tokens_s,
        "ctx_probe_totals_s": totals_s,
        "ctx_probe_latency_ms": latency_ms,
        "ctx_probe_speed_tps": speed_tps,
        "ctx_probe_kv_gb": kv_gb,
        "ctx_probe_prompt_tokens": prompt_tokens,
        "ctx_probe_read": _format_ctx_probe_rate(read_s),
        "ctx_probe_tokens": _format_ctx_probe_rate(tokens_s),
        "ctx_probe_totals": _format_ctx_probe_rate(totals_s),
        "ctx_probe_latency": _format_ctx_probe_latency(latency_ms),
        "ctx_probe_speed": _format_ctx_probe_speed(speed_tps),
        "ctx_probe_kv": _format_ctx_probe_gb(kv_gb),
    }


def load_catalog(path: Path, server_config_path: Path | None = None) -> list[ManagedModel]:
    items, _ = load_catalog_with_diagnostics(path, server_config_path=server_config_path)
    return items


def load_catalog_with_diagnostics(path: Path, server_config_path: Path | None = None) -> tuple[list[ManagedModel], str | None]:
    if not path.exists():
        _clear_catalog_cache(path)
        return [], f"Catalog file not found: {path}"
    cache_key = _catalog_cache_key(path, server_config_path)
    try:
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
    except OSError as exc:
        _clear_catalog_cache(path)
        return [], f"Could not stat catalog {path}: {exc}"
    cached = CATALOG_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature[0] and cached[1] == signature[1]:
        return [replace(item) for item in cached[2]], None
    try:
        payload = json.loads(path.read_text("utf-8"))
    except Exception as exc:
        _clear_catalog_cache(path)
        return [], f"Could not read/parse catalog {path}: {exc}"
    if not isinstance(payload, list):
        _clear_catalog_cache(path)
        return [], f"Catalog {path} has invalid format (expected a JSON array)."
    try:
        items = [ManagedModel(**m) for m in payload]
    except Exception as exc:
        _clear_catalog_cache(path)
        return [], f"Catalog {path} has invalid entries: {exc}"
    changed = False
    # Load global llama-server defaults (normalized) so we can prune redundant per-model overrides
    try:
        args = argparse.Namespace(server_config=server_config_path) if server_config_path is not None else None
        server_defaults = resolve_llama_server_defaults(args)
    except Exception:
        server_defaults = {}

    for item in items:
        normalized_load_capabilities = _normalize_load_capabilities(item.load_capabilities)
        if normalized_load_capabilities != item.load_capabilities:
            item.load_capabilities = normalized_load_capabilities
            changed = True

        normalized_aliases = normalize_aliases(item.aliases)
        if normalized_aliases != item.aliases:
            item.aliases = normalized_aliases
            changed = True

        normalized_probe_latency = _to_float_or_none(item.ctx_probe_latency_ms)
        if normalized_probe_latency != item.ctx_probe_latency_ms:
            item.ctx_probe_latency_ms = normalized_probe_latency
            changed = True

        normalized_probe_read = _to_float_or_none(item.ctx_probe_read_s)
        if normalized_probe_read != item.ctx_probe_read_s:
            item.ctx_probe_read_s = normalized_probe_read
            changed = True

        normalized_probe_tokens = _to_float_or_none(item.ctx_probe_tokens_s)
        if normalized_probe_tokens != item.ctx_probe_tokens_s:
            item.ctx_probe_tokens_s = normalized_probe_tokens
            changed = True

        normalized_probe_totals = _to_float_or_none(item.ctx_probe_totals_s)
        if normalized_probe_totals != item.ctx_probe_totals_s:
            item.ctx_probe_totals_s = normalized_probe_totals
            changed = True

        normalized_probe_speed = _to_float_or_none(item.ctx_probe_speed_tps)
        if normalized_probe_speed != item.ctx_probe_speed_tps:
            item.ctx_probe_speed_tps = normalized_probe_speed
            changed = True

        if item.ctx_probe_speed_tps is None and item.ctx_probe_totals_s is not None:
            item.ctx_probe_speed_tps = item.ctx_probe_totals_s
            changed = True
        if item.ctx_probe_totals_s is None and item.ctx_probe_speed_tps is not None:
            item.ctx_probe_totals_s = item.ctx_probe_speed_tps
            changed = True

        normalized_probe_kv_gb = _to_float_or_none(item.ctx_probe_kv_gb)
        if normalized_probe_kv_gb != item.ctx_probe_kv_gb:
            item.ctx_probe_kv_gb = normalized_probe_kv_gb
            changed = True

        normalized_probe_prompt_tokens = _to_int_or_none(item.ctx_probe_prompt_tokens)
        if normalized_probe_prompt_tokens != item.ctx_probe_prompt_tokens:
            item.ctx_probe_prompt_tokens = normalized_probe_prompt_tokens
            changed = True

        normalized = preferred_tensor_split(item, item.tensor_split)
        if normalized != item.tensor_split:
            item.tensor_split = normalized
            changed = True
        raw_auto_performance = None
        if isinstance(item.server_overrides, dict):
            raw_auto_performance = item.server_overrides.get("auto_performance")
        normalized_overrides = normalize_server_overrides(item.server_overrides)

        # Remove keys from per-model overrides when they are identical to global defaults
        pruned_overrides: dict[str, object] = {}
        for k, v in normalized_overrides.items():
            if k in server_defaults and server_defaults.get(k) == v:
                changed = True
                continue
            pruned_overrides[k] = v
        if isinstance(raw_auto_performance, dict):
            # Persistent auto-performance metadata belongs in catalog config,
            # but normalize_server_overrides intentionally strips it so it can
            # never be emitted as a llama-server flag.
            pruned_overrides["auto_performance"] = raw_auto_performance

        if pruned_overrides != item.server_overrides:
            item.server_overrides = pruned_overrides
            changed = True
    if changed:
        save_catalog(path, items)
    cached_items = [replace(item) for item in items]
    CATALOG_CACHE[cache_key] = (signature[0], signature[1], cached_items)
    return [replace(item) for item in cached_items], None

def save_catalog(path: Path, models: list[ManagedModel]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    effective_idle_ttl = get_configured_idle_ttl(DEFAULT_CONFIG_PATH, resolve_idle_ttl())
    serialized: list[dict[str, object]] = []
    for model in models:
        payload = asdict(model)
        if payload.get("ttl") == effective_idle_ttl:
            payload.pop("ttl", None)
        serialized.append(payload)
    tmp.write_text(json.dumps(serialized, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    _clear_catalog_cache(path)


def _normalize_model_id_token(value: str) -> str:
    token = (value or "").strip().lower()
    if not token:
        return ""
    token = re.sub(r"\.gguf\b", "", token)
    token = re.sub(r"(^|[-._])gguf(?=($|[-._]))", r"\1", token)
    token = re.sub(r"[-._]?\d{5}-of-\d{5}$", "", token)
    token = re.sub(r"[-._]{2,}", "-", token)
    token = re.sub(r"[^a-z0-9._-]+", "-", token).strip("-._")
    return token


def _canonical_model_ref(value: str) -> str:
    token = _normalize_model_id_token(value)
    return re.sub(r"[-._]+", "-", token).strip("-")


def _append_quant_suffix_if_missing(base_id: str, quant: str | None) -> str:
    quant_token = _normalize_model_id_token(quant or "")
    if not quant_token:
        return base_id
    if re.search(rf"(^|[-._]){re.escape(quant_token)}($|[-._])", base_id):
        return base_id
    return _normalize_model_id_token(f"{base_id}-{quant_token}")


def normalize_model_id(repo_id, quant, filename):
    filename_seed = Path(filename).stem if filename and filename != "hf-native" else ""
    mid = _normalize_model_id_token(filename_seed)
    if not mid:
        base = (repo_id or "").split("/")[-1].strip()
        mid = _normalize_model_id_token(base)
    mid = _append_quant_suffix_if_missing(mid, quant)
    if not mid:
        mid = _normalize_model_id_token(Path(filename).stem)
        mid = _append_quant_suffix_if_missing(mid, quant)
    return mid or "model"

def choose_gguf_file(api, repo_id, quant, explicit_file, token):
    info = api.model_info(repo_id=repo_id, token=token)
    all_files = sorted(s.rfilename for s in info.siblings if s.rfilename and s.rfilename.lower().endswith(".gguf"))
    if not all_files:
        if _is_vllm_backend():
            if os.environ.get("DEBUG_LLAMACPP"):
                print(f"DEBUG: No GGUF files found in {repo_id}, but vLLM backend is active. Returning None.", file=sys.stderr)
            return None
        raise RuntimeError(f"No GGUF in {repo_id}. If you want to use a native HuggingFace model, set LLAMACPP_BACKEND=vllm-beta")
    
    # Exclude mmproj files from main model search (they're selected separately)
    files = [f for f in all_files if "mmproj" not in f.lower()]
    if not files: files = all_files  # Fallback if all are mmproj
    
    if explicit_file:
        if explicit_file not in files: raise RuntimeError(f"File {explicit_file} not found.")
        return explicit_file
    
    if quant:
        ql = quant.lower()
        
        # Extract quantization code: last component after . or - before .gguf
        def extract_quant_code(filename):
            stem = Path(filename).stem  # Remove .gguf
            # Split by . or - and get last part
            parts = re.split(r'[-.]', stem)
            return parts[-1].lower() if parts else ""
        
        # Try exact quantization code match first (f16 != bf16)
        exact_matches = [f for f in files if extract_quant_code(f) == ql]
        if len(exact_matches) == 1: return exact_matches[0]
        if len(exact_matches) > 1:
            shards = sorted([f for f in exact_matches if "-00001-of-" in f])
            if shards: return shards[0]
            return exact_matches[0]  # Return first of multiple exact matches
        
        # Fallback: substring match on quant code
        matches = [f for f in files if ql in extract_quant_code(f)]
        if len(matches) == 1: return matches[0]
        if len(matches) > 1:
            shards = sorted([f for f in matches if "-00001-of-" in f])
            if shards: return shards[0]
            raise RuntimeError(f"Ambiguous quant '{quant}': {matches}")
    
    q4 = [f for f in files if "q4_k_m" in f.lower()]
    if len(q4) == 1: return q4[0]
    return files[0]


def choose_mmproj_file(api, repo_id, token):
    info = api.model_info(repo_id=repo_id, token=token)
    files = sorted(s.rfilename for s in info.siblings if s.rfilename and "mmproj" in s.rfilename.lower() and s.rfilename.lower().endswith(".gguf"))
    if not files:
        return None
    preferred_order = ("f16", "bf16", "f32")
    lowered = {f.lower(): f for f in files}
    for pref in preferred_order:
        for candidate in files:
            if pref in candidate.lower():
                return candidate
    return files[0]

def render_llamaswap_config(
    catalog,
    path,
    server_path,
    start_port,
    idle_ttl=DEFAULT_IDLE_TTL,
    server_defaults: dict[str, object] | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "healthCheckTimeout": 600,
        "logLevel": "info",
        "logToStdout": "proxy",
        "startPort": start_port,
        "sendLoadingState": True,
        "includeAliasesInList": True,
        "models": {},
    }
    resolved_defaults = normalize_server_overrides(server_defaults or resolve_llama_server_defaults())

    # Count how many catalog entries reference the same local_path so that
    # we can create on-disk variant files at render time if necessary
    # (e.g., templated variants pointing to the same GGUF) to avoid
    # duplicate-publishing conflicts when swap consumes the YAML.
    path_usage: dict[str, int] = {}
    resolved_paths: dict[str, Path] = {}
    for m in catalog:
        try:
            p = Path(m.local_path).resolve()
            key = str(p)
        except Exception:
            key = str(m.local_path or "")
        path_usage[key] = path_usage.get(key, 0) + 1
        resolved_paths[key] = Path(m.local_path) if m.local_path else Path("")

    for m in sorted(catalog, key=lambda x: x.model_id):
        # Respect user's preference: do NOT create on-disk GGUF variants.
        # Use the original model file for rendering; duplicate catalog
        # entries will reference the same local_path. Note: this may
        # cause duplicate alias conflicts upstream if the backend refuses
        # same-GGUF multiple publishes. The catalog will still contain
        # separate entries (one with chat template override) but no new
        # files are created on disk.
        use_model = m

        cmd = build_llama_server_command(use_model, server_path, port="${PORT}", server_defaults=resolved_defaults)
        # If this is a derived entry (template/speculative) that points to the
        # same GGUF, try to force a distinct published name/ID so backends that
        # key off internal GGUF aliases won't treat it as a duplicate. We
        # probe common flags and append the first supported one.
        try:
            derived = str(m.model_id).endswith("+template") or str(m.model_id).endswith("+spec") or bool(m.speculative)
            if derived:
                name_flags = ["--model-id", "--model-name", "--name", "--publish-name", "--publish-id"]
                chosen_flag = None
                for f in name_flags:
                    if server_supports_flag(server_path, f):
                        chosen_flag = f
                        break
                if chosen_flag:
                    # Use a canonical safe id derived from model_id
                    safe_name = _canonical_model_ref(m.model_id)
                    cmd.extend([chosen_flag, safe_name])
        except Exception:
            pass
        data["models"][m.model_id] = {
            "cmd": " ".join(shell_quote(part) for part in cmd),
            "checkEndpoint": "/health",
            "ttl": int(idle_ttl),
        }
        if m.aliases:
            data["models"][m.model_id]["aliases"] = m.aliases
        if m.description:
            data["models"][m.model_id]["description"] = m.description
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(LLAMASWAP_CONFIG_HEADER)
        yaml.safe_dump(data, f, sort_keys=False)
    tmp.replace(path)


def shell_quote(v):
    if re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", v):
        return v
    if re.fullmatch(r"[A-Za-z0-9_./:=,+-]+", v):
        return v
    return "'" + v.replace("'", "'\"'\"'") + "'"


def _load_server_config_payload(args = None) -> dict:
    if args is not None and getattr(args, "idle_ttl", None) is not None:
        pass

    candidate_paths: list[Path] = []
    if args is not None and getattr(args, "server_config", None) is not None:
        candidate_paths.append(Path(args.server_config))
    else:
        default_path = Path(DEFAULT_SERVER_CONFIG_PATH)
        # Prefer new conf.json for new installs, fall back to legacy name
        candidate_paths.append(default_path.with_name(ALTERNATE_SERVER_CONFIG_BASENAME))
        candidate_paths.append(default_path)

    for p in candidate_paths:
        try:
            if not p.exists():
                continue
            payload = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return {}


def _template_variant_path(base_path: Path) -> Path | None:
    try:
        if not base_path:
            return None
        stem = base_path.stem
        suffix = base_path.suffix
        if not stem:
            return None
        return base_path.with_name(f"{stem}+template{suffix}")
    except Exception:
        return None


def _ensure_template_variant_file(base_path: Path) -> Path | None:
    variant_path = _template_variant_path(base_path)
    if variant_path is None:
        return None
    try:
        if variant_path.exists() or variant_path.is_symlink():
            return variant_path
        target = base_path.resolve()
        try:
            variant_path.symlink_to(target)
            return variant_path
        except Exception:
            pass
        try:
            os.link(target, variant_path)
            return variant_path
        except Exception:
            pass
        # Fallback to a safe chunked copy to avoid long blocking sendfile
        if _safe_copy_file(target, variant_path):
            return variant_path
        return None
    except Exception:
        return None


def _safe_copy_file(src: Path, dst: Path, chunk_size: int = 4 * 1024 * 1024) -> bool:
    """Copy `src` -> `dst` using chunked reads/writes and atomic replace.

    Returns True on success, False on failure. Removes partial files on error
    so repeated attempts do not leave garbage. This avoids long blocking
    `shutil.copy2` sendfile operations that are hard to interrupt.
    """
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        # Ensure parent exists
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as fsrc, tmp.open("wb") as fdst:
            while True:
                chunk = fsrc.read(chunk_size)
                if not chunk:
                    break
                fdst.write(chunk)
        try:
            shutil.copystat(src, tmp)
        except Exception:
            pass
        os.replace(str(tmp), str(dst))
        return True
    except KeyboardInterrupt:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def resolve_api_ctx_factor(args = None) -> float:
    """Read API ctx factor used to derive API_CTX from CFG_CTX.

    Accepts numeric ratios (0.5), percentages (50 or "50%").
    Prefers `api_ctx_factor` and falls back to legacy ctx_display keys.
    Always returns a value in [0.0, 1.0].
    """
    try:
        payload = _load_server_config_payload(args)
    except Exception:
        return float(DEFAULT_API_CTX_FACTOR)
    val = payload.get("api_ctx_factor")
    if val is None:
        val = payload.get("ctx_display_ratio")
    if val is None:
        val = payload.get("ctx_display_percent")
    if val is None:
        return float(DEFAULT_API_CTX_FACTOR)
    try:
        if isinstance(val, str):
            v = val.strip()
            if v.endswith("%"):
                return max(0.0, min(1.0, float(v[:-1]) / 100.0))
            f = float(v)
        else:
            f = float(val)
        # if user passed a percent like 50 -> convert to ratio
        if f > 1.0:
            f = f / 100.0
        return max(0.0, min(1.0, f))
    except Exception:
        return float(DEFAULT_API_CTX_FACTOR)


def resolve_ctx_display_ratio(args = None) -> float:
    """Backward-compatible alias for legacy ctx display configuration keys."""
    return resolve_api_ctx_factor(args)


def displayed_configured_ctx(model: ManagedModel, args = None) -> int:
    """Return configured context for a model (CFG_CTX source of truth)."""
    try:
        base = int(model.ctx_size or 0)
    except Exception:
        base = 0
    if base > 0:
        return base
    try:
        gguf = get_model_context_size(model)
        return int(gguf) if gguf is not None else 0
    except Exception:
        return 0


def displayed_api_ctx(model: ManagedModel, args = None) -> int:
    """Return API_CTX derived from CFG_CTX * configurable factor."""
    cfg_ctx = displayed_configured_ctx(model)
    if cfg_ctx <= 0:
        return 0
    try:
        factor = resolve_api_ctx_factor(args)
    except Exception:
        factor = float(DEFAULT_API_CTX_FACTOR)
    return max(1, int(cfg_ctx * factor))


def ctx_evaluation_status(model: ManagedModel) -> str:
    return "ERROR" if bool(model.auto_ctx_failed) else "OK"


def _display_cfg_ctx(model: ManagedModel, args = None) -> str:
    if ctx_evaluation_status(model) == "ERROR":
        return "ERROR"
    value = displayed_configured_ctx(model, args)
    return str(value) if value > 0 else "?"


def _display_api_ctx(model: ManagedModel, args = None) -> str:
    if ctx_evaluation_status(model) == "ERROR":
        return "ERROR"
    value = displayed_api_ctx(model, args)
    return str(value) if value > 0 else "?"


def resolve_idle_ttl(args = None) -> int:
    if args is not None and getattr(args, "idle_ttl", None) is not None:
        return int(args.idle_ttl)
    value = _load_server_config_payload(args).get("idle_ttl")
    if value is not None:
        return int(value)
    return DEFAULT_IDLE_TTL


def resolve_llama_server_defaults(args = None) -> dict[str, object]:
    return normalize_server_overrides(_load_server_config_payload(args).get("llama_server_defaults"))


def _args_server_config_path(args) -> Path | None:
    value = getattr(args, "server_config", None)
    if value is None:
        return None
    return Path(value)


def _append_llama_server_flag(cmd: list[str], key: str, value: object, server_path: Path | str | None = None) -> None:
    if key == "split_mode":
        cmd.extend(["--split-mode", str(value)])
    elif key == "flash_attn":
        # The underlying llama-server expects an explicit value for --flash-attn
        # (e.g. "on", "off", "auto"). Accept booleans and common
        # boolean-like strings and always emit a value token so the next
        # flag is not accidentally parsed as the value.
        valstr = None
        if isinstance(value, bool):
            valstr = "on" if value else "off"
        else:
            sval = str(value).strip()
            if sval:
                low = sval.lower()
                if low in {"1", "true", "yes", "on"}:
                    valstr = "on"
                elif low in {"0", "false", "no", "off"}:
                    valstr = "off"
                else:
                    valstr = sval
        if valstr is not None:
            cmd.extend(["--flash-attn", valstr])
    elif key == "batch_size":
        cmd.extend(["--batch-size", str(int(value))])
    elif key == "ubatch_size":
        cmd.extend(["--ubatch-size", str(max(256, int(value)))])
    elif key == "threads":
        # Accept numeric or string hints 'physical'/'logical'
        try:
            if isinstance(value, str):
                pc = os.cpu_count() or 1
                if value.strip().lower() == "physical":
                    v = max(1, pc // 2)
                elif value.strip().lower() == "logical":
                    v = pc * 2
                else:
                    v = int(value)
            else:
                v = int(value)
            cmd.extend(["--threads", str(v)])
        except Exception:
            pass
    elif key == "threads_batch":
        try:
            if isinstance(value, str):
                pc = os.cpu_count() or 1
                if value.strip().lower() == "physical":
                    v = max(1, pc // 2)
                elif value.strip().lower() == "logical":
                    v = pc * 2
                else:
                    v = int(value)
            else:
                v = int(value)
            cmd.extend(["--threads-batch", str(v)])
        except Exception:
            pass
    elif key == "main_gpu":
        cmd.extend(["--main-gpu", str(int(value))])
    elif key == "numa":
        if value is None:
            return
        sval = str(value).strip()
        if sval and sval.lower() != "none":
            cmd.extend(["--numa", sval])
    elif key == "reasoning_format":
        sval = str(value).strip()
        if sval:
            cmd.extend(["--reasoning-format", sval])
    elif key == "fit_target":
        cmd.extend(["--fit-target", str(int(value))])
    elif key == "model_draft":
        sval = str(value or "").strip()
        if sval and sval.lower() not in {"none", "null"}:
            cmd.extend(["--model-draft", sval])
    elif key == "hf_repo_draft":
        cmd.extend(["--hf-repo-draft", str(value)])
    elif key == "draft":
        # Current llama.cpp accepts --draft/--draft-n/--draft-max. Older
        # --spec-draft-n-max is not universally supported.
        draft_flag = next(
            (
                flag
                for flag in ("--spec-draft-n-max", "--draft-max", "--draft", "--draft-n")
                if server_supports_flag(server_path, flag)
            ),
            "--draft-max",
        )
        cmd.extend([draft_flag, str(int(value))])
    elif key == "draft_min":
        # draft_min was removed in newer llama.cpp API; skipping
        pass
    elif key == "draft_p_min":
        # draft_p_min was removed in newer llama.cpp API; skipping
        pass
    elif key == "ctx_size_draft":
        cmd.extend(["--ctx-size-draft", str(int(value))])
    elif key == "n_gpu_layers_draft":
        try:
            if isinstance(value, str) and value.strip().lower() == "all":
                v = 999
            elif isinstance(value, str) and value.strip().lower() == "auto":
                v = -1
            else:
                v = int(value)
            cmd.extend(["--n-gpu-layers-draft", str(v)])
        except Exception:
            pass
    elif key == "keep":
        cmd.extend(["--keep", str(int(value))])
    elif key == "mirostat":
        cmd.extend(["--mirostat", str(int(value))])
    elif key == "mirostat_ent":
        cmd.extend(["--mirostat-ent", str(float(value))])
    elif key == "mirostat_lr":
        cmd.extend(["--mirostat-lr", str(float(value))])
    elif key == "cache_type_k":
        cmd.extend(["--cache-type-k", str(value)])
    elif key == "cache_type_v":
        cmd.extend(["--cache-type-v", str(value)])
    elif key == "fit":
        valstr = None
        if isinstance(value, bool):
            valstr = "on" if value else "off"
        else:
            sval = str(value).strip()
            if sval:
                low = sval.lower()
                if low in {"1", "true", "yes", "on"}:
                    valstr = "on"
                elif low in {"0", "false", "no", "off"}:
                    valstr = "off"
                else:
                    valstr = sval
        if valstr is not None:
            # Upstream llama-server expects short (single-dash) fit flags
            # to avoid being parsed as GNU-style long options. Emit the
            # single-dash form to match the server's arg parsing.
            cmd.extend(["-fit", valstr])
    elif key == "fitt":
        try:
            # Use single-dash form for numeric fit target count
            cmd.extend(["-fitt", str(int(value))])
        except Exception:
            pass
    elif key == "fitc":
        # The llama-server `-fitc` / `--fit-ctx` flag expects an integer N
        # indicating the minimum ctx size that --fit may set. Accept boolean
        # True as a shorthand to mean the default numeric value (4096), and
        # coerce numeric/string inputs to int where possible.
        try:
            if isinstance(value, bool):
                if value:
                    cmd.extend(["-fitc", str(4096)])
                # False -> do not emit the flag
            else:
                sval = str(value).strip()
                if not sval:
                    pass
                else:
                    try:
                        cmd.extend(["-fitc", str(int(sval))])
                    except Exception:
                        # Try float->int, then fallback to default
                        try:
                            cmd.extend(["-fitc", str(int(float(sval)))])
                        except Exception:
                            cmd.extend(["-fitc", str(4096)])
        except Exception:
            # Defensive fallback to default numeric value
            cmd.extend(["-fitc", str(4096)])
    elif key == "mmap":
        bool_val = _normalize_bool_flag(value)
        if bool_val is False:
            cmd.append("--no-mmap")
    elif key in {"float16", "bfloat16", "float32"}:
        if _normalize_bool_flag(value):
            cmd.append(f"--{key}")
    elif key == "gpu_memory_utilization":
        if value is not None:
            cmd.extend(["--gpu-memory-utilization", str(float(value))])
    elif key == "mul_mat_q":
        if _normalize_bool_flag(value):
            if server_supports_flag(server_path, "--mul-mat-q"):
                cmd.append("--mul-mat-q")
            else:
                # skip unsupported flag
                pass
    elif key == "grp_attn_n":
        try:
            flag = "--grp-attn-n"
            if server_supports_flag(server_path, flag):
                cmd.extend([flag, str(int(value))])
        except (TypeError, ValueError):
            pass
    elif key == "parallel":
        try:
            flag = "--parallel"
            if server_supports_flag(server_path, flag):
                cmd.extend([flag, str(int(value))])
        except (TypeError, ValueError):
            pass
    elif key == "ctx_checkpoints":
        try:
            flag = "--ctx-checkpoints"
            if server_supports_flag(server_path, flag):
                cmd.extend([flag, str(int(value))])
        except (TypeError, ValueError):
            pass
    elif key == "cache_ram":
        try:
            flag = "--cache-ram"
            if server_supports_flag(server_path, flag):
                cmd.extend([flag, str(int(value))])
        except (TypeError, ValueError):
            pass
    elif key == "kv_offload":
        bool_val = _normalize_bool_flag(value)
        if bool_val is True:
            cmd.append("--kv-offload")
        elif bool_val is False:
            cmd.append("--no-kv-offload")
    elif key == "cont_batching":
        bool_val = _normalize_bool_flag(value)
        if bool_val is True:
            cmd.append("--cont-batching")
        elif bool_val is False:
            cmd.append("--no-cont-batching")
    elif key == "op_offload":
        bool_val = _normalize_bool_flag(value)
        if bool_val is True:
            cmd.append("--op-offload")
        elif bool_val is False:
            cmd.append("--no-op-offload")
    elif key == "direct_io":
        bool_val = _normalize_bool_flag(value)
        if bool_val is True:
            cmd.append("--direct-io")
        elif bool_val is False:
            cmd.append("--no-direct-io")
    elif key == "kv_unified":
        bool_val = _normalize_bool_flag(value)
        if bool_val is True and server_supports_flag(server_path, "--kv-unified"):
            cmd.append("--kv-unified")
        elif bool_val is False and server_supports_flag(server_path, "--no-kv-unified"):
            cmd.append("--no-kv-unified")
    elif key == "cache_idle_slots":
        bool_val = _normalize_bool_flag(value)
        if bool_val is True and server_supports_flag(server_path, "--cache-idle-slots"):
            cmd.append("--cache-idle-slots")
        elif bool_val is False and server_supports_flag(server_path, "--no-cache-idle-slots"):
            cmd.append("--no-cache-idle-slots")
    elif key == "cpu_moe":
        bool_val = _normalize_bool_flag(value)
        if bool_val:
            cmd.append("--cpu-moe")
    elif key == "n_cpu_moe":
        try:
            cmd.extend(["--n-cpu-moe", str(int(value))])
        except (TypeError, ValueError):
            pass
    elif key == "device":
        sval = str(value).strip()
        if sval:
            cmd.extend(["--device", sval])
    elif key == "defrag_threshold":
        try:
            flag = "--defrag-threshold"
            if server_supports_flag(server_path, flag):
                cmd.extend([flag, str(float(value))])
        except (TypeError, ValueError):
            pass
    else:
        # Generic fallback: map unknown keys to --kebab-case and append value.
        try:
            flag = key if str(key).startswith("-") else f"--{str(key).replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
                else:
                    cmd.extend([flag, "false"])
            elif value is None:
                cmd.append(flag)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    cmd.extend([flag, str(v)])
            else:
                cmd.extend([flag, str(value)])
        except Exception:
            pass


def build_llama_server_command(
    model: ManagedModel,
    server_path: Path,
    *,
    port: str,
    host: str | None = None,
    include_model_path: bool = True,
    include_mmproj: bool = True,
    include_jinja: bool = True,
    server_defaults: dict[str, object] | None = None,
    extra_flags: list[str] | None = None,
) -> list[str]:
    effective = dict(normalize_server_overrides(server_defaults or {}))
    # If this model is a speculative/draft variant, merge spec defaults first
    # so they can be overridden by per-model server_overrides.
    try:
        if getattr(model, "speculative", False) and isinstance(server_defaults, dict):
            spec_defaults = server_defaults.get("speculative_defaults")
            if isinstance(spec_defaults, dict):
                effective.update(normalize_server_overrides(spec_defaults))
                # Propagate raw fit-related keys if provided in the speculative_defaults
                for raw_key in ("fit", "fitt", "fitc"):
                    if raw_key not in effective and raw_key in spec_defaults:
                        effective[raw_key] = spec_defaults[raw_key]
    except Exception:
        pass
    effective.update(normalize_server_overrides(model.server_overrides))

    # Resolve non-fit launch dimensions first so fit-mode can optionally move
    # context control from --ctx-size into -fitc.
    ctx_size = int(effective.pop("ctx_size", model.ctx_size))
    n_gpu_layers = int(effective.pop("n_gpu_layers", model.n_gpu_layers))
    tensor_split = preferred_tensor_split(model, str(effective.pop("tensor_split", model.tensor_split)))
    resolved_host = str(effective.pop("host", host or model.host))

    # Fit policy:
    # - fit on  -> use -fitc for context, omit --ctx-size
    # - fit off -> keep --ctx-size, omit -fitc/-fitt
    have_autocontext = (getattr(model, "ctx_probe_kv_gb", None) is not None) or (getattr(model, "ctx_probe_read_s", None) is not None)
    raw_fit = effective.get("fit", True)
    fit_enabled = _normalize_bool_flag(raw_fit)
    if fit_enabled is None:
        if isinstance(raw_fit, str):
            fit_enabled = bool(raw_fit.strip())
        else:
            fit_enabled = bool(raw_fit)

    if fit_enabled:
        effective["fit"] = True
        if (not have_autocontext) and ("fitt" not in effective):
            effective["fitt"] = int(effective.get("fitt", 1024))

        fitc_value = effective.get("fitc")
        parsed_fitc: int | None = None
        if isinstance(fitc_value, bool):
            parsed_fitc = None
        elif fitc_value is not None:
            try:
                parsed_fitc = int(fitc_value)
            except Exception:
                try:
                    parsed_fitc = int(float(str(fitc_value).strip()))
                except Exception:
                    parsed_fitc = None

        # When fit is active, move configured ctx into -fitc unless a valid
        # explicit fitc value is provided.
        if parsed_fitc is None or parsed_fitc <= 0:
            effective["fitc"] = ctx_size
        else:
            effective["fitc"] = parsed_fitc
    else:
        effective["fit"] = False
        effective.pop("fitc", None)
        effective.pop("fitt", None)

    if _is_vllm_backend():
        # vLLM launch logic
        vllm_bin = os.environ.get("VLLM_SERVER_BIN", "vllm-server")
        vllm_cmd = [vllm_bin, "--model", str(model.local_path), "--port", str(port)]
        
        # Map some common overrides
        vllm_map = {
            "gpu_memory_utilization": "--gpu-memory-utilization",
            "tensor_parallel_size": "--tensor-parallel-size",
            "max_model_len": "--max-model-len",
            "block_size": "--block-size",
            "dtype": "--dtype",
            "device": "--device",
            "enable_chunked_prefill": "--enable-chunked-prefill",
        }
        
        for k, v in effective.items():
            if k in vllm_map:
                vllm_cmd.extend([vllm_map[k], str(v)])
            elif k.startswith("--"): # Direct passthrough of double-dash flags
                vllm_cmd.extend([k, str(v)])
                
        # Handle host if provided
        if resolved_host and resolved_host not in {"0.0.0.0", "::", "[::]"}:
            vllm_cmd.extend(["--host", resolved_host])

        return vllm_cmd

    cmd = [str(server_path), "--port", str(port)]
    if include_model_path:
        cmd.extend(["--model", str(model.local_path)])
    if not fit_enabled:
        cmd.extend(["--ctx-size", str(ctx_size)])
    cmd.extend(["--n-gpu-layers", str(n_gpu_layers)])
    cmd.extend(["--tensor-split", tensor_split])
    cmd.extend(["--host", resolved_host])
    for key in (
        "split_mode",
        "flash_attn",
        "reasoning_format",
        "batch_size",
        "ubatch_size",
        "threads",
        "threads_batch",
        "main_gpu",
        "numa",
        "fit_target",
        "model_draft",
        "hf_repo_draft",
        "draft",
        "draft_min",
        "draft_p_min",
        "ctx_size_draft",
        "n_gpu_layers_draft",
        "fit",
        "fitt",
        "fitc",
        "keep",
        "mirostat",
        "mirostat_ent",
        "mirostat_lr",
        "cache_type_k",
        "cache_type_v",
        "mmap",
        "mul_mat_q",
        "grp_attn_n",
        "parallel",
        "ctx_checkpoints",
        "cache_ram",
        "kv_offload",
        "cont_batching",
        "op_offload",
        "direct_io",
        "cpu_moe",
        "n_cpu_moe",
        "device",
        "defrag_threshold",
    ):
        if key in effective:
            _append_llama_server_flag(cmd, key, effective[key], server_path)
            # Mark as handled so it won't be appended again in the leftover pass
            try:
                effective.pop(key, None)
            except Exception:
                pass
    # Append any remaining unknown server_overrides as generic flags
    for extra_key, extra_val in list(effective.items()):
        # skip keys already handled above via _append_llama_server_flag
        # (they were removed or processed), but emit any left-over entries
        _append_llama_server_flag(cmd, extra_key, extra_val, server_path)
    if include_mmproj and model.mmproj_path:
        cmd.extend(["--mmproj", str(model.mmproj_path)])
    if include_jinja and model.jinja:
        cmd.append("--jinja")
    # Chat template handling is explicit: refresh-templates creates a
    # template-backed catalog entry with chat_template_file set.
    tmpl = effective.get("chat_template_file") or effective.get("chat_template")
    if tmpl:
        cmd.extend(["--chat-template-file", str(tmpl)])
    if extra_flags:
        cmd.extend(list(extra_flags))
    return cmd


def _find_chat_template_for_model(model_id: str, templates_dir: Path) -> Path | None:
    """Search templates_dir for a file matching model_id or its family.

    Matching rules:
    - exact filename (without extension) == model_id
    - filename is a substring of model_id (family match)
    - model_id startswith filename (family match)
    Returns first reasonable match or None.
    """
    try:
        if not templates_dir.exists() or not templates_dir.is_dir():
            return None
        files = [p for p in templates_dir.iterdir() if p.is_file()]
        # Try exact base name match first
        for p in files:
            if p.stem == model_id:
                return p
        # Try normalized matches (ignore separators)
        norm = re.sub(r"[^A-Za-z0-9]+", "", model_id).lower()
        for p in files:
            if re.sub(r"[^A-Za-z0-9]+", "", p.stem).lower() == norm:
                return p
        # Fallback: family substring match
        for p in files:
            stem = p.stem.lower()
            if stem and (stem in model_id.lower() or model_id.lower().startswith(stem)):
                return p
    except Exception:
        return None
    return None


def resolve_api_port(args = None) -> int:
    if args is not None and getattr(args, "api_port", None) is not None:
        return int(args.api_port)
    value = _load_server_config_payload(args).get("api_port")
    if value is not None:
        return int(value)
    return DEFAULT_API_PORT


def persist_server_config(args) -> None:
    if getattr(args, "server_config", None) is None:
        return
    if (
        getattr(args, "idle_ttl", None) is None
        and getattr(args, "api_port", None) is None
        and getattr(args, "api_ctx_factor", None) is None
    ):
        return
    path = Path(args.server_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    if getattr(args, "idle_ttl", None) is not None:
        payload["idle_ttl"] = int(args.idle_ttl)
    if getattr(args, "api_port", None) is not None:
        payload["api_port"] = int(args.api_port)
    if getattr(args, "api_ctx_factor", None) is not None:
        payload["api_ctx_factor"] = float(args.api_ctx_factor)
    _ensure_server_config_metadata(payload)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def log_api_event(kind: str, payload: dict, log_path: Path = DEFAULT_REQUESTS_LOG_PATH) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            **payload,
        }
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _elapsed_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)


def mark_model_activity(model_id: str, source: str, phase: str, *, log: bool = True) -> None:
    """Record wrapper-observed model activity for idle/unload diagnostics."""
    if not model_id:
        return
    now = time.monotonic()
    global LAST_ACTIVITY_MODEL_ID
    with MODEL_ACTIVITY_LOCK:
        record = MODEL_ACTIVITY.setdefault(model_id, {})
        record["last_activity_monotonic"] = now
        record["last_activity_wall"] = datetime.now(timezone.utc).isoformat()
        record["last_source"] = source
        record["last_phase"] = phase
        LAST_ACTIVITY_MODEL_ID = model_id
    if log:
        log_api_event("model_activity", {"model": model_id, "source": source, "phase": phase})


def get_model_activity_snapshot() -> tuple[dict[str, dict[str, float | str]], str]:
    with MODEL_ACTIVITY_LOCK:
        return ({model: dict(record) for model, record in MODEL_ACTIVITY.items()}, LAST_ACTIVITY_MODEL_ID)


def should_reload_after_unexpected_unload(
    model_id: str,
    activity: dict[str, dict[str, float | str]],
    last_activity_model_id: str,
    *,
    now: float,
    idle_ttl: int,
) -> tuple[bool, float | None]:
    record = activity.get(model_id)
    if not record:
        return False, None
    last_activity = record.get("last_activity_monotonic")
    if not isinstance(last_activity, (float, int)):
        return False, None
    age = now - float(last_activity)
    if age < 0 or age >= idle_ttl:
        return False, age
    if last_activity_model_id and last_activity_model_id != model_id:
        return False, age
    return True, age

def _normalize_client_host(host: str | None) -> str:
    normalized = str(host or "").strip()
    if normalized in {"0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return normalized or "127.0.0.1"

def wait_for_model(model_id, host, port, timeout=35):
    host = _normalize_client_host(host)
    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + timeout
    spinner = Spinner(f"\033[36mWaiting for {model_id}...\033[0m ")
    spinner.start()
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                if any(m.get("id") == model_id for m in r.json().get("data", [])):
                    spinner.stop()
                    print(f"\033[32mModel {model_id} is published.\033[0m")
                    return True
        except: pass
        time.sleep(1.5)
    spinner.stop()
    return False

def apply_config_and_wait(
    catalog,
    config_path,
    llama_server,
    start_port,
    model_id,
    host,
    port,
    progress_callback = None,
    settle_time = 3.0,
    timeout = 45.0,
    server_defaults: dict[str, object] | None = None,
):
    stop_running_ollama_models(progress_callback=progress_callback)
    render_llamaswap_config(
        catalog,
        config_path,
        llama_server,
        start_port,
        resolve_idle_ttl(),
        server_defaults=server_defaults,
    )
    _emit_message("Config updated. Waiting for llama-swap --watch-config...", progress_callback)
    time.sleep(settle_time)
    if wait_for_model(model_id, host, port, timeout=timeout):
        return True
    raise RuntimeError(
        f"Model {model_id} did not appear after updating {config_path}. "
        "Ensure llama-swap is running with --watch-config and is watching that config file."
    )

def _format_bytes(num_bytes: int | float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{int(num_bytes)}B"

def _format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    total = int(seconds)
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

def _render_download_progress(label: str, downloaded: int, total: int | None, speed_bps: float, done: bool = False):
    cols = shutil.get_terminal_size((120, 20)).columns
    prefix = f"{label} "
    speed_txt = f"{_format_bytes(speed_bps)}/s" if speed_bps > 0 else "--/s"

    if total and total > 0:
        ratio = min(downloaded / total, 1.0)
        eta = (total - downloaded) / speed_bps if speed_bps > 0 else None
        stats = f" {ratio * 100:6.2f}% {_format_bytes(downloaded)}/{_format_bytes(total)} {speed_txt} ETA {_format_eta(eta)}"
        bar_width = max(10, cols - len(prefix) - len(stats) - 4)
        filled = int(bar_width * ratio)
        if filled >= bar_width:
            bar = "=" * bar_width
        else:
            bar = "=" * filled + ">" + "." * max(0, bar_width - filled - 1)
        line = f"\r{prefix}[{bar}]{stats}"
    else:
        line = f"\r{prefix}{_format_bytes(downloaded)} {speed_txt}"

    sys.stdout.write(line[: max(0, cols - 1)])
    sys.stdout.write("\033[K")
    if done:
        sys.stdout.write("\n")
    sys.stdout.flush()

def _emit_message(message: str, progress_callback = None, timestamp: bool = False):
    if not timestamp:
        try:
            import inspect

            for frame in inspect.stack():
                if frame.function == "choose_auto_ctx":
                    timestamp = True
                    break
        except Exception:
            pass

    if timestamp:
        try:
            ts = datetime.now(timezone.utc).astimezone().isoformat(sep=' ', timespec='milliseconds')
        except Exception:
            ts = datetime.now().isoformat(sep=' ', timespec='milliseconds')
        message = f"[{ts}] {message}"
    if progress_callback:
        progress_callback({"type": "message", "message": message})
    else:
        print(message)

def _emit_progress(label: str, downloaded: int, total: int | None, speed_bps: float, done: bool = False, progress_callback = None):
    if progress_callback:
        progress_callback({
            "type": "progress",
            "label": label,
            "downloaded": downloaded,
            "total": total,
            "speed_bps": speed_bps,
            "done": done,
        })
    else:
        _render_download_progress(label, downloaded, total, speed_bps, done=done)


def list_running_ollama_models() -> list[str]:
    if shutil.which("ollama") is None:
        return []
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []

    models: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("NAME "):
            continue
        model = stripped.split()[0].strip()
        if model and model not in models:
            models.append(model)
    return models


def stop_running_ollama_models(progress_callback = None) -> list[str]:
    models = list_running_ollama_models()
    if not models:
        return []

    _emit_message(
        "Stopping running Ollama models before llama-swap loads a model to avoid GPU conflicts.",
        progress_callback,
    )
    stopped: list[str] = []
    for model in models:
        try:
            result = subprocess.run(
                ["ollama", "stop", model],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as exc:
            _emit_message(f"Could not stop Ollama model {model}: {exc}", progress_callback)
            continue
        if result.returncode == 0:
            stopped.append(model)
            _emit_message(f"Stopped Ollama model {model}.", progress_callback)
        else:
            detail = (result.stderr or result.stdout or "").strip()
            if detail:
                _emit_message(f"Could not stop Ollama model {model}: {detail}", progress_callback)
            else:
                _emit_message(f"Could not stop Ollama model {model}.", progress_callback)
    return stopped

def _hf_resolve_url(repo_id: str, filename: str) -> str:
    repo = "/".join(quote(part, safe="") for part in repo_id.split("/"))
    file_path = "/".join(quote(part, safe="") for part in filename.split("/"))
    return f"https://huggingface.co/{repo}/resolve/main/{file_path}"


def _download_backend() -> str:
    return os.environ.get("LLAMACPP_DOWNLOAD_BACKEND", "parallel").strip().lower()


def _should_use_hf_transfer() -> bool:
    raw = os.environ.get("LLAMACPP_USE_HF_TRANSFER", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _probe_range_download(url: str, headers: dict[str, str]) -> tuple[int | None, bool]:
    probe_headers = dict(headers)
    probe_headers["Range"] = "bytes=0-0"
    try:
        with requests.get(url, headers=probe_headers, stream=True, allow_redirects=True, timeout=(10, 30)) as resp:
            if resp.status_code != 206:
                return None, False
            content_range = resp.headers.get("Content-Range", "")
            if "/" not in content_range:
                return None, False
            total_str = content_range.rsplit("/", 1)[1]
            if not total_str.isdigit():
                return None, False
            return int(total_str), True
    except Exception:
        return None, False


def _parallel_download_workers() -> int:
    raw = os.environ.get("LLAMACPP_DOWNLOAD_WORKERS", "8").strip()
    try:
        value = int(raw)
    except Exception:
        value = 8
    return max(2, min(value, 32))


def _download_hf_file_parallel(repo_id: str, filename: str, token: str | None, target_dir: Path, label: str | None = None, progress_callback = None) -> str | None:
    if _download_backend() not in {"parallel", "auto"}:
        return None

    dest_path = target_dir / filename
    part_path = dest_path.with_name(dest_path.name + ".part")
    if part_path.exists() and part_path.stat().st_size > 0:
        return None

    base_headers = {}
    if token:
        base_headers["Authorization"] = f"Bearer {token}"
    url = _hf_resolve_url(repo_id, filename)
    total, ranged = _probe_range_download(url, base_headers)
    if not ranged or not total or total < 256 * 1024 * 1024:
        return None

    progress_label = label or Path(filename).name
    workers = _parallel_download_workers()
    segment_size = math.ceil(total / workers)
    downloaded = 0
    downloaded_lock = threading.Lock()
    stop_event = threading.Event()
    failed = []

    fd = os.open(part_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
    os.ftruncate(fd, total)

    def worker(start: int, end: int) -> None:
        nonlocal downloaded
        headers = dict(base_headers)
        headers["Range"] = f"bytes={start}-{end}"
        position = start
        try:
            with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=(10, 120)) as resp:
                resp.raise_for_status()
                if resp.status_code != 206:
                    raise RuntimeError(f"Range request not honored (HTTP {resp.status_code})")
                for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                    if stop_event.is_set():
                        return
                    if not chunk:
                        continue
                    os.pwrite(fd, chunk, position)
                    position += len(chunk)
                    with downloaded_lock:
                        downloaded += len(chunk)
        except Exception as exc:
            failed.append(exc)
            stop_event.set()

    start_time = time.time()
    last_refresh = 0.0
    _emit_message(f"Downloading {filename} with parallel ranges ({workers} workers)...", progress_callback)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for idx in range(workers):
                start = idx * segment_size
                if start >= total:
                    break
                end = min(total - 1, start + segment_size - 1)
                futures.append(executor.submit(worker, start, end))
            while any(not future.done() for future in futures):
                now = time.time()
                if now - last_refresh >= 0.2:
                    elapsed = max(now - start_time, 1e-6)
                    with downloaded_lock:
                        current = downloaded
                    speed = current / elapsed
                    _emit_progress(progress_label, current, total, speed, progress_callback=progress_callback)
                    last_refresh = now
                if stop_event.is_set():
                    break
                time.sleep(0.1)
            for future in futures:
                future.result()
        if failed:
            raise failed[0]
        elapsed = max(time.time() - start_time, 1e-6)
        with downloaded_lock:
            current = downloaded
        speed = current / elapsed
        _emit_progress(progress_label, current, total, speed, done=True, progress_callback=progress_callback)
        os.close(fd)
        part_path.replace(dest_path)
        return str(dest_path)
    except Exception as exc:
        try:
            os.close(fd)
        except Exception:
            pass
        # Clean up the truncated .part file after parallel download failure.
        # The file was pre-allocated with ftruncate() but may have incomplete chunks.
        # Retaining it causes confusion in sequential retry: it looks complete (byte count
        # matches total size) but has no actual data. Better to restart from scratch.
        part_path.unlink(missing_ok=True)
        _emit_message(f"Parallel download failed for {filename}: {exc}. Falling back.", progress_callback)
        return None


def _download_hf_file_fast(repo_id: str, filename: str, token: str | None, target_dir: Path, progress_callback = None) -> str | None:
    if _download_backend() == "parallel":
        return None
    if not _should_use_hf_transfer():
        return None
    try:
        import hf_transfer  # noqa: F401
    except Exception:
        return None

    _emit_message(f"Downloading {filename} with hf_transfer...", progress_callback)
    previous = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        _emit_message(f"{filename} downloaded with hf_transfer.", progress_callback)
        return path
    except Exception as exc:
        _emit_message(f"hf_transfer unavailable for {filename}: {exc}. Falling back to streamed download.", progress_callback)
        return None
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        else:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = previous


def download_hf_file(
    repo_id: str,
    filename: str,
    token: str | None,
    target_dir: Path,
    label: str | None = None,
    progress_callback = None,
    expected_size: int | None = None,
) -> str:
    dest_path = target_dir / filename
    part_path = dest_path.with_name(dest_path.name + ".part")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        current_size = dest_path.stat().st_size
        if expected_size is None and current_size > 0:
            _emit_message(f"{filename} already available.", progress_callback)
            return str(dest_path)
        if expected_size is not None and current_size == expected_size and current_size > 0:
            _emit_message(f"{filename} already available.", progress_callback)
            return str(dest_path)
        if current_size > 0 and not part_path.exists():
            if expected_size is not None and current_size > expected_size:
                dest_path.unlink(missing_ok=True)
            else:
                dest_path.replace(part_path)

    parallel_path = _download_hf_file_parallel(repo_id, filename, token, target_dir, label=label, progress_callback=progress_callback)
    if parallel_path:
        return str(Path(parallel_path))

    fast_path = _download_hf_file_fast(repo_id, filename, token, target_dir, progress_callback=progress_callback)
    if fast_path:
        return str(Path(fast_path))

    base_headers = {}
    if token:
        base_headers["Authorization"] = f"Bearer {token}"

    url = _hf_resolve_url(repo_id, filename)
    progress_label = label or Path(filename).name
    resume_from = part_path.stat().st_size if part_path.exists() else 0

    for _ in range(2):
        headers = dict(base_headers)
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
            _emit_message(f"Resuming {filename} from {_format_bytes(resume_from)}...", progress_callback)
        else:
            _emit_message(f"Downloading {filename}...", progress_callback)

        with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=(10, 120)) as resp:
            if resp.status_code == 416 and resume_from:
                # HTTP 416 = Range Not Satisfiable. This can mean:
                # A) The .part file is complete and matches server size (expected)
                # B) The .part file is truncated by ftruncate() but has no real data (bug case)
                # Only treat as complete if size matches expected_size.
                part_size = part_path.stat().st_size if part_path.exists() else 0
                if expected_size is not None and part_size == expected_size:
                    part_path.replace(dest_path)
                    _emit_message(f"{filename} already complete.", progress_callback)
                    return str(dest_path)
                elif expected_size is None:
                    # No expected size to compare; assume it's complete if we got 416
                    part_path.replace(dest_path)
                    _emit_message(f"{filename} already complete.", progress_callback)
                    return str(dest_path)
                else:
                    # 416 but size mismatch: .part is corrupted/truncated. Reset and retry.
                    resp.close()
                    part_path.unlink(missing_ok=True)
                    resume_from = 0
                    _emit_message(
                        f"{filename}: incomplete .part detected (size mismatch), restarting from scratch.",
                        progress_callback,
                    )
                    continue

            if resume_from and resp.status_code == 200:
                resp.close()
                part_path.unlink(missing_ok=True)
                resume_from = 0
                continue

            resp.raise_for_status()

            total = expected_size
            content_range = resp.headers.get("Content-Range", "")
            if total is None and "/" in content_range:
                total_str = content_range.rsplit("/", 1)[1]
                if total_str.isdigit():
                    total = int(total_str)
            if total is None:
                content_length = resp.headers.get("Content-Length")
                if content_length and content_length.isdigit():
                    total = int(content_length)
                    if resp.status_code == 206:
                        total += resume_from

            mode = "ab" if resume_from and resp.status_code == 206 else "wb"
            if mode == "wb":
                part_path.unlink(missing_ok=True)

            chunk_size = 1024 * 1024
            start = time.time()
            last_refresh = 0.0
            downloaded = resume_from if mode == "ab" else 0
            base_downloaded = downloaded

            with open(part_path, mode) as fh:
                if downloaded:
                    _emit_progress(progress_label, downloaded, total, 0.0, progress_callback=progress_callback)

                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_refresh >= 0.2:
                        elapsed = max(now - start, 1e-6)
                        speed = (downloaded - base_downloaded) / elapsed
                        _emit_progress(progress_label, downloaded, total, speed, progress_callback=progress_callback)
                        last_refresh = now

            elapsed = max(time.time() - start, 1e-6)
            speed = (downloaded - base_downloaded) / elapsed if downloaded > base_downloaded else 0.0
            _emit_progress(progress_label, downloaded, total or downloaded, speed, done=True, progress_callback=progress_callback)
            part_path.replace(dest_path)
            return str(dest_path)

    raise RuntimeError(f"Could not download {filename} from {repo_id}")


def _repo_sibling_sizes(api, repo_id: str, token: str | None) -> dict[str, int]:
    try:
        info = api.model_info(repo_id=repo_id, token=token)
    except Exception:
        return {}
    sizes: dict[str, int] = {}
    for sibling in getattr(info, "siblings", []) or []:
        filename = getattr(sibling, "rfilename", None)
        size = getattr(sibling, "size", None)
        if filename and isinstance(size, int) and size > 0:
            sizes[filename] = size
    return sizes


def _download_file_state(target_dir: Path, filename: str, expected_size: int | None = None) -> str:
    final_path = target_dir / filename
    part_path = final_path.with_name(final_path.name + ".part")
    if final_path.exists():
        final_size = final_path.stat().st_size
        if final_size <= 0:
            return "missing"
        if expected_size is None or final_size == expected_size:
            return "completed"
        return "partial"
    if part_path.exists() and part_path.stat().st_size > 0:
        return "partial"
    return "missing"


def model_files_ready(target_dir: Path, filenames: list[str], expected_sizes: dict[str, int] | None = None) -> bool:
    sizes = expected_sizes or {}
    for filename in filenames:
        if _download_file_state(target_dir, filename, sizes.get(filename)) != "completed":
            return False
    return True


def summarize_download_state(target_dir: Path, filenames: list[str], expected_sizes: dict[str, int] | None = None):
    sizes = expected_sizes or {}
    completed = 0
    partial = 0
    missing = 0
    for filename in filenames:
        state = _download_file_state(target_dir, filename, sizes.get(filename))
        if state == "completed":
            completed += 1
        elif state == "partial":
            partial += 1
        else:
            missing += 1
    return completed, partial, missing

def infer_shard_filenames(filename: str) -> list[str]:
    match = re.match(r"^(.*)-(\d+)-of-(\d+)(\.gguf)$", filename)
    if not match:
        return [filename]
    prefix, current_str, total_str, suffix = match.groups()
    width_current = len(current_str)
    width_total = len(total_str)
    total = int(total_str)
    if total <= 1:
        return [filename]
    return [
        f"{prefix}-{idx:0{width_current}d}-of-{total:0{width_total}d}{suffix}"
        for idx in range(1, total + 1)
    ]

def run_manager_command(command: str, args):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(SOCKET_PATH)
        req = {"command": command, "args": vars(args)}
        req["args"] = {k: str(v) if isinstance(v, Path) else v for k, v in req["args"].items() if k != 'func'}
        s.sendall((json.dumps(req) + "\n").encode())
        print("\033[33mRequest sent to background manager...\033[0m")
        if command == "auto-performance":
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            print(f"{ts} ─ auto-performance ─ request queued; next: manager resolves baseline and starts tuning", flush=True)

        with s.makefile("r", encoding="utf-8") as sock_in:
            while True:
                raw = sock_in.readline()
                if not raw:
                    break
                event = json.loads(raw)
                etype = event.get("type")
                if etype == "message":
                    print(event["message"])
                elif etype == "progress":
                    _render_download_progress(
                        event["label"],
                        int(event["downloaded"]),
                        int(event["total"]) if event["total"] is not None else None,
                        float(event["speed_bps"]),
                        done=bool(event.get("done")),
                    )
                elif etype == "question":
                    prompt = event.get("prompt", "Continue?")
                    default = event.get("default", "n").lower()
                    suffix = "[Y/n]" if default == "y" else "[y/N]"
                    try:
                        answer = input(f"{prompt} {suffix} ").strip()
                    except EOFError:
                        answer = ""
                    s.sendall((json.dumps({"type": "answer", "answer": answer}) + "\n").encode())
                elif etype == "done":
                    if "result" in event:
                        return event["result"]
                    model_id = event.get("model_id")
                    if not model_id:
                        raise RuntimeError(f"Manager finished {command} without returning a result.")
                    return model_id
                elif etype == "error":
                    # If the manager failed to include a message, include the
                    # full event payload to aid diagnostics instead of raising
                    # a cryptic empty message.
                    msg = event.get("message")
                    if not msg:
                        try:
                            msg = f"Manager error: {json.dumps(event)}"
                        except Exception:
                            msg = "Manager error with no message"
                    raise RuntimeError(msg)
                else:
                    raise RuntimeError(f"Unexpected response from manager: {event}")

    raise RuntimeError("Manager connection closed unexpectedly.")


def manager_hint() -> str:
    mode = infer_install_mode()
    start_cmd, status_cmd, _restart_cmd = service_commands_for_mode(mode)
    return (
        "Could not connect to the background manager.\n"
        f"Detected install mode: {mode}.\n"
        "Try:\n"
        f"  {start_cmd}\n"
        f"  {status_cmd}\n"
        f"Socket path: {SOCKET_PATH}"
    )


def infer_install_mode() -> str:
    """Infer whether this runtime is using system or user installation paths."""
    explicit = os.environ.get("LLAMACPP_INSTALL_MODE", "").strip().lower()
    if explicit in {"system", "user"}:
        return explicit

    home = Path.home().resolve()

    def _is_home_path(value: Path) -> bool:
        try:
            value_resolved = value.expanduser().resolve()
            return value_resolved == home or home in value_resolved.parents
        except Exception:
            return False

    config_paths = [
        Path(DEFAULT_CONFIG_PATH),
        Path(DEFAULT_CATALOG_PATH),
        Path(DEFAULT_SERVER_CONFIG_PATH),
        Path(DEFAULT_LLAMA_SERVER),
    ]
    if any(_is_home_path(path) for path in config_paths):
        return "user"

    system_markers = (
        "/etc/llamacpp-superserver",
        "/var/lib/llamacpp-superserver",
        "/opt/llamacpp-superserver",
    )
    config_values = [str(path.expanduser()) for path in config_paths]
    if any(any(marker in value for marker in system_markers) for value in config_values):
        return "system"

    socket_path = str(Path(SOCKET_PATH).expanduser())
    if "/run/llamacpp-superserver" in socket_path:
        return "system"
    if str(home) in socket_path:
        return "user"

    return "system" if os.geteuid() == 0 else "user"


def service_commands_for_mode(mode: str) -> tuple[str, str, str]:
    if mode == "system":
        return (
            f"sudo systemctl start {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}",
            f"sudo systemctl status {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}",
            f"sudo systemctl restart {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}",
        )
    return (
        f"systemctl --user start {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}",
        f"systemctl --user status {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}",
        f"systemctl --user restart {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}",
    )


def _has_gguf_files(models_dir: Path) -> bool:
    try:
        return any(models_dir.rglob("*.gguf"))
    except Exception:
        return False


def _show_local_catalog_fallback(args, reason: Exception) -> None:
    print(f"{manager_hint()}\nDetails: {reason}")
    catalog, catalog_diag = load_catalog_with_diagnostics(args.catalog)
    print("\nManager unavailable; showing local catalog view (published/loaded status may be stale).")
    print("\n" + render_models_table(catalog, args.public_host, args.public_port, get_effective_idle_ttl(args)))
    if catalog:
        return
    if catalog_diag:
        print(f"Catalog diagnostic: {catalog_diag}")
    elif _has_gguf_files(args.models_dir):
        print(
            "Detected GGUF files in the models directory, but there are no registered catalog entries yet. "
            "Register models first with 'llamacpp-superserver run <hf-repo[:quant]>' or 'add'."
        )

def _ask_confirmation(prompt: str, progress_callback = None, default: bool = False) -> bool:
    default_key = "y" if default else "n"
    invalid_message = "Please answer yes/y or no/n."
    while True:
        if progress_callback:
            response = progress_callback({
                "type": "question",
                "prompt": prompt,
                "default": default_key,
            })
            answer = (response or "").strip()
            if not answer:
                return default
            lowered = answer.lower()
            if lowered in ("y", "yes"):
                return True
            if lowered in ("n", "no"):
                return False
            _emit_message(invalid_message, progress_callback)
            continue

        suffix = "[Y/n]" if default else "[y/N]"
        try:
            answer = input(f"{prompt} {suffix} ").strip()
        except EOFError:
            answer = ""
        if not answer:
            return default
        lowered = answer.lower()
        if lowered in ("y", "yes"):
            return True
        if lowered in ("n", "no"):
            return False
        print(invalid_message)

def resolve_catalog_model(catalog: list[ManagedModel], target: str | None = None, repo_ref: str | None = None, model_id: str | None = None, filename: str | None = None) -> ManagedModel:
    def _unique_from(items: list[ManagedModel]) -> list[ManagedModel]:
        unique: list[ManagedModel] = []
        seen: set[str] = set()
        for item in items:
            if item.model_id in seen:
                continue
            seen.add(item.model_id)
            unique.append(item)
        return unique

    matches = []
    ref = repo_ref or target

    if model_id:
        matches = [m for m in catalog if m.model_id == model_id]
    elif ref:
        repo_id = quant = None
        if "/" in ref or ref.startswith("hf.co/") or ref.startswith("huggingface.co/") or ref.startswith("ollama run "):
            try:
                repo_id, quant = parse_hf_input(ref)
            except ValueError:
                repo_id = quant = None
        if repo_id is None:
            matches = [m for m in catalog if m.model_id == ref]
        elif filename:
            matches = [m for m in catalog if m.repo_id == repo_id and m.filename == filename]
        elif quant is None:
            matches = [m for m in catalog if m.repo_id == repo_id]
        else:
            matches = [m for m in catalog if m.repo_id == repo_id and m.quant == quant]
    else:
        raise RuntimeError("Model reference required. Use model_id or repo[:quant].")

    if not matches and ref:
        lowered = ref.strip().lower()
        matches = [m for m in catalog if (m.model_id or "").strip().lower() == lowered]

    if not matches and ref:
        canonical = _canonical_model_ref(ref)
        if canonical:
            matches = [m for m in catalog if _canonical_model_ref(m.model_id or "") == canonical]

    if not matches and ref:
        resolved_name = resolve_catalog_model_name(ref, catalog)
        if resolved_name and resolved_name != ref:
            matches = [m for m in catalog if m.model_id == resolved_name]

    if not matches and ref:
        canonical = _canonical_model_ref(ref)
        if canonical:
            alias_matches: list[ManagedModel] = []
            for item in catalog:
                for alias in model_name_aliases(item):
                    if _canonical_model_ref(alias) == canonical:
                        alias_matches.append(item)
                        break
            matches = _unique_from(alias_matches)

    if not matches:
        raise RuntimeError("Model not found in catalog.")
    if len(matches) > 1:
        options = ", ".join(sorted(m.model_id for m in matches))
        raise RuntimeError(f"Ambiguous model selection. Use --model-id or --file. Matches: {options}")
    return matches[0]

def ensure_model_available(args, progress_callback = None):
    # CLIENT MODE: Send to socket if not owner
    try:
        is_owner = (os.getuid() == 0 or os.getuid() == os.stat(args.catalog.parent).st_uid)
    except:
        is_owner = False

    if not is_owner:
        try:
            return run_manager_command("add", args)
        except RuntimeError as e:
            raise e
        except Exception as e:
            raise manager_unavailable_error(e)

    # MANAGER MODE
    catalog = load_catalog(args.catalog, _args_server_config_path(args))
    active_server_defaults = resolve_llama_server_defaults(args)
    stable_catalog = [ManagedModel(**asdict(m)) for m in catalog]
    ref = args.repo or args.hf or args.model_id
    if not ref:
        raise RuntimeError("Model reference required.")
    ctx_override = getattr(args, "ctx_override", None)
    ctx_override = int(ctx_override) if ctx_override is not None else None
    force_auto_ctx = bool(getattr(args, "auto_ctx", False))
    skip_ctx = bool(getattr(args, "skip_ctx", False))
    default_ctx = int(args.ctx_size)
    if ctx_override is not None and skip_ctx:
        raise RuntimeError("Use either -ctx or --skip-ctx, not both.")
    if ctx_override is not None and force_auto_ctx:
        raise RuntimeError("Use either -ctx or --auto, not both.")
    if skip_ctx and force_auto_ctx:
        raise RuntimeError("Use either --skip-ctx or --auto, not both.")
    defer_publish = bool(getattr(args, "defer_publish", False))
    spec_draft_model_id = str(getattr(args, "spec_draft_model_id", "") or "").strip()

    api = HfApi()
    requested_model = None
    try:
        requested_model = resolve_catalog_model(
            catalog,
            target=args.repo,
            repo_ref=args.hf,
            model_id=args.model_id,
            filename=args.file,
        )
    except RuntimeError:
        requested_model = None

    token = args.hf_token or os.environ.get("HF_TOKEN")
    selected_file = None
    to_download = []

    if requested_model is not None:
        existing = requested_model
        repo_id = existing.repo_id
        quant = existing.quant
        selected_file = existing.filename
        to_download = infer_shard_filenames(selected_file)
        if len(to_download) > 1:
            _emit_message(f"Using catalog model {existing.model_id} with {len(to_download)} shards.", progress_callback)
    else:
        if args.model_id:
            raise RuntimeError(f"Model {args.model_id} not found in catalog.")
        ref = args.repo or args.hf
        repo_id, quant = parse_hf_input(ref)
        selected_file = choose_gguf_file(api, repo_id, quant, args.file, token)

        if selected_file is None and _is_vllm_backend():
            # For vLLM, if no GGUF is found, we assume it's a native HF model
            # We'll use a virtual filename to represent the HF repo
            selected_file = "hf-native"
            to_download = [s.rfilename for s in api.model_info(repo_id=repo_id, token=token).siblings if s.rfilename]
            # Exclude large blobs if they are not needed? No, vLLM needs them.
        else:
            if selected_file is None:
                raise RuntimeError(f"No GGUF found in {repo_id} and backend is not vLLM.")
            to_download = [selected_file]
        if "-00001-of-" in selected_file:
            _emit_message("Detected sharded model. Resolving all parts...", progress_callback)
            prefix = selected_file.split("-00001-of-")[0]
            info = api.model_info(repo_id=repo_id, token=token)
            repo_files = [s.rfilename for s in info.siblings if s.rfilename]
            to_download = sorted([f for f in repo_files if f.startswith(prefix) and "-of-" in f and f.lower().endswith(".gguf")])
            _emit_message(f"Found {len(to_download)} shards in the repository.", progress_callback)

        existing = next((m for m in catalog if m.repo_id == repo_id and m.filename == selected_file), None)
        if existing is None and not args.file:
            existing = next((m for m in catalog if m.repo_id == repo_id and m.quant == quant), None)

    expected_sizes = _repo_sibling_sizes(api, repo_id, token)

    target_dir = args.models_dir / repo_id
    target_dir.mkdir(parents=True, exist_ok=True)
    mmproj_filename = existing.mmproj_filename if existing else None
    mmproj_path = existing.mmproj_path if existing else None
    if _looks_like_vision_model(existing if existing else ManagedModel(model_id=mid if 'mid' in locals() else "", repo_id=repo_id, quant=quant, filename=selected_file, local_path="", description=args.description or f"{repo_id} / {selected_file}")):
        try:
            mmproj_filename = choose_mmproj_file(api, repo_id, token)
        except Exception:
            mmproj_filename = existing.mmproj_filename if existing else None
        if mmproj_filename:
            mmproj_path = str(target_dir / mmproj_filename)
    mmproj_ready = bool(not mmproj_filename or (mmproj_path and Path(mmproj_path).exists()))
    ctx_changed = False
    config_changed = False
    auto_ctx_failed = bool(existing.auto_ctx_failed) if existing else False
    auto_ctx_error = existing.auto_ctx_error if existing else ""
    ctx_probe_read_s = _to_float_or_none(existing.ctx_probe_read_s) if existing else None
    ctx_probe_tokens_s = _to_float_or_none(existing.ctx_probe_tokens_s) if existing else None
    ctx_probe_totals_s = _to_float_or_none(existing.ctx_probe_totals_s) if existing else None
    ctx_probe_latency_ms = _to_float_or_none(existing.ctx_probe_latency_ms) if existing else None
    ctx_probe_speed_tps = _to_float_or_none(existing.ctx_probe_speed_tps) if existing else None
    if ctx_probe_speed_tps is None:
        ctx_probe_speed_tps = ctx_probe_totals_s
    if ctx_probe_totals_s is None:
        ctx_probe_totals_s = ctx_probe_speed_tps
    ctx_probe_kv_gb = _to_float_or_none(existing.ctx_probe_kv_gb) if existing else None
    ctx_probe_prompt_tokens = _to_int_or_none(existing.ctx_probe_prompt_tokens) if existing else None
    if existing and ctx_override is not None and (existing.ctx_size != ctx_override or existing.auto_ctx_failed or existing.auto_ctx_error):
        existing.ctx_size = ctx_override
        existing.auto_ctx_failed = False
        existing.auto_ctx_error = ""
        clear_ctx_probe_metrics(existing)
        refresh_model_load_capabilities(existing)
        save_catalog(args.catalog, catalog)
        ctx_changed = True
        auto_ctx_failed = False
        auto_ctx_error = ""
        ctx_probe_read_s = None
        ctx_probe_tokens_s = None
        ctx_probe_totals_s = None
        ctx_probe_latency_ms = None
        ctx_probe_speed_tps = None
        ctx_probe_kv_gb = None
        ctx_probe_prompt_tokens = None
        _emit_message(f"Applied ctx override for {existing.model_id}: {ctx_override}", progress_callback)
    elif existing and skip_ctx and (existing.ctx_size != default_ctx or existing.auto_ctx_failed or existing.auto_ctx_error):
        existing.ctx_size = default_ctx
        existing.auto_ctx_failed = False
        existing.auto_ctx_error = ""
        clear_ctx_probe_metrics(existing)
        refresh_model_load_capabilities(existing)
        save_catalog(args.catalog, catalog)
        ctx_changed = True
        auto_ctx_failed = False
        auto_ctx_error = ""
        ctx_probe_read_s = None
        ctx_probe_tokens_s = None
        ctx_probe_totals_s = None
        ctx_probe_latency_ms = None
        ctx_probe_speed_tps = None
        ctx_probe_kv_gb = None
        ctx_probe_prompt_tokens = None
        _emit_message(f"Applied default ctx for {existing.model_id}: {default_ctx}", progress_callback)

    if existing:
        desired_n_gpu_layers = int(args.n_gpu_layers)
        desired_jinja = not args.no_jinja
        desired_description = args.description or existing.description or f"{repo_id} / {selected_file}"
        desired_mmproj_filename = mmproj_filename
        desired_mmproj_path = mmproj_path
        if existing.n_gpu_layers != desired_n_gpu_layers:
            existing.n_gpu_layers = desired_n_gpu_layers
            config_changed = True
            _emit_message(f"Applied n_gpu_layers for {existing.model_id}: {desired_n_gpu_layers}", progress_callback)
        if existing.tensor_split != args.tensor_split:
            existing.tensor_split = args.tensor_split
            config_changed = True
            _emit_message(f"Applied tensor_split for {existing.model_id}: {args.tensor_split}", progress_callback)
        if existing.host != args.host:
            existing.host = args.host
            config_changed = True
            _emit_message(f"Applied host for {existing.model_id}: {args.host}", progress_callback)
        if existing.jinja != desired_jinja:
            existing.jinja = desired_jinja
            config_changed = True
            _emit_message(f"Applied jinja mode for {existing.model_id}: {desired_jinja}", progress_callback)
        if existing.description != desired_description:
            existing.description = desired_description
            config_changed = True
            _emit_message(f"Applied description for {existing.model_id}.", progress_callback)
        for key in ("float16", "bfloat16", "float32", "gpu_memory_utilization"):
            val = getattr(args, key, None)
            if val:
                if existing.server_overrides.get(key) != val:
                    existing.server_overrides[key] = val
                    config_changed = True
                    _emit_message(f"Applied {key} override for {existing.model_id}: {val}", progress_callback)
        if existing.mmproj_filename != desired_mmproj_filename:
            existing.mmproj_filename = desired_mmproj_filename
            config_changed = True
            _emit_message(f"Applied mmproj filename for {existing.model_id}: {desired_mmproj_filename or 'none'}", progress_callback)
        if existing.mmproj_path != desired_mmproj_path:
            existing.mmproj_path = desired_mmproj_path
            config_changed = True
            _emit_message(f"Applied mmproj path for {existing.model_id}: {desired_mmproj_path or 'none'}", progress_callback)
        if config_changed:
            refresh_model_load_capabilities(existing)
            save_catalog(args.catalog, catalog)

    mid = args.model_id or (existing.model_id if existing else normalize_model_id(repo_id, quant, selected_file))
    # Support speculative/draft variants: if requested, compute a unique speculative
    # model id prefixed by the configured `speculative_defaults.id_prefix` and
    # avoid short-circuiting early when a base catalog entry already exists.
    is_speculative_request = bool(getattr(args, "speculative", False))
    base_mid = mid
    forced_base_mid = getattr(args, "spec_base_model_id", None)
    if is_speculative_request and forced_base_mid:
        forced_base_mid_str = str(forced_base_mid).strip()
        if forced_base_mid_str:
            base_mid = forced_base_mid_str
    files_ready = model_files_ready(target_dir, to_download, expected_sizes)
    completed_files, partial_files, missing_files = summarize_download_state(target_dir, to_download, expected_sizes)

    if existing and not is_speculative_request and not args.force and files_ready and mmproj_ready and not force_auto_ctx:
        _emit_message("All required model files already exist locally. Skipping download.", progress_callback)
        if defer_publish:
            _emit_message(f"Deferring publish/load for {existing.model_id}.", progress_callback)
            return existing.model_id
        if not ctx_changed and not config_changed and wait_for_model(existing.model_id, args.public_host, args.public_port, timeout=2):
            _emit_default_ctx_update_hint(existing.model_id, existing.ctx_size, default_ctx, progress_callback)
            return existing.model_id
        gpu_conflict = get_gpu_conflict_message(existing.model_id, catalog, args.public_host, args.public_port)
        if gpu_conflict:
            _emit_message(gpu_conflict, progress_callback)
            raise RuntimeError(gpu_conflict)
        try:
            apply_config_and_wait(
                catalog,
                args.config,
                args.llama_server,
                args.start_port,
                existing.model_id,
                args.public_host,
                args.public_port,
                progress_callback=progress_callback,
                server_defaults=active_server_defaults,
            )
        except Exception:
            save_catalog(args.catalog, stable_catalog)
            restore_catalog_config(args, stable_catalog, progress_callback=progress_callback, restart_service=True)
            raise
        _emit_default_ctx_update_hint(existing.model_id, existing.ctx_size, default_ctx, progress_callback)
        return existing.model_id

    if existing and not files_ready:
        _emit_message("Catalog entry exists but model files are missing or incomplete. Downloading required files...", progress_callback)
    elif existing and not mmproj_ready:
        _emit_message("Catalog entry exists but mmproj is missing. Downloading the multimodal projector...", progress_callback)
    if len(to_download) > 1 and (completed_files or partial_files):
        remaining_files = partial_files + missing_files
        _emit_message(
            f"Resume status: {completed_files}/{len(to_download)} complete, "
            f"{partial_files} partial, {remaining_files} remaining.",
            progress_callback,
        )
    elif len(to_download) == 1 and partial_files:
        _emit_message("Resume status: single file partially downloaded, continuing from the saved offset.", progress_callback)
    
    local_path = ""
    total_files = len(to_download)
    for idx, f in enumerate(to_download, start=1):
        if f == "hf-native":
            continue
        label = f"[{idx}/{total_files}] {Path(f).name}" if total_files > 1 else Path(f).name
        loc = download_hf_file(
            repo_id=repo_id,
            filename=f,
            token=token,
            target_dir=target_dir,
            label=label,
            progress_callback=progress_callback,
            expected_size=expected_sizes.get(f),
        )
        if f == selected_file:
            local_path = loc
    if selected_file == "hf-native":
        _emit_message(f"Populating native HF repo {repo_id}...", progress_callback)
        local_path = snapshot_download(
            repo_id=repo_id,
            token=token,
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
    if mmproj_filename:
        mmproj_label = f"mmproj {Path(mmproj_filename).name}"
        mmproj_loc = download_hf_file(
            repo_id=repo_id,
            filename=mmproj_filename,
            token=token,
            target_dir=target_dir,
            label=mmproj_label,
            progress_callback=progress_callback,
            expected_size=expected_sizes.get(mmproj_filename),
        )
        mmproj_path = mmproj_loc

    probe_config_replaced = False
    desired_ctx = ctx_override if ctx_override is not None else (existing.ctx_size if existing and existing.auto_ctx_failed else default_ctx)
    if ctx_override is not None:
        auto_ctx_failed = False
        auto_ctx_error = ""
        ctx_probe_read_s = None
        ctx_probe_tokens_s = None
        ctx_probe_totals_s = None
        ctx_probe_latency_ms = None
        ctx_probe_speed_tps = None
        ctx_probe_kv_gb = None
        ctx_probe_prompt_tokens = None
        _emit_message(f"Using explicit ctx override {desired_ctx} for {mid}.", progress_callback)
    elif skip_ctx:
        auto_ctx_failed = False
        auto_ctx_error = ""
        ctx_probe_read_s = None
        ctx_probe_tokens_s = None
        ctx_probe_totals_s = None
        ctx_probe_latency_ms = None
        ctx_probe_speed_tps = None
        ctx_probe_kv_gb = None
        ctx_probe_prompt_tokens = None
        _emit_message(f"Skipping automatic ctx tuning. Using default ctx {desired_ctx} for {mid}.", progress_callback)
    elif existing and existing.auto_ctx_failed and not force_auto_ctx:
        desired_ctx = existing.ctx_size or default_ctx
        auto_ctx_failed = True
        auto_ctx_error = existing.auto_ctx_error or "previous-auto-ctx-failure"
        _emit_message(
            f"Previous auto ctx probe failed ({auto_ctx_error}). Using saved fallback ctx {desired_ctx} without re-probing.",
            progress_callback,
        )
    else:
        if force_auto_ctx:
            _emit_message(f"Forcing a fresh automatic ctx probe for {mid}.", progress_callback)
        _emit_message(
            "Download complete. Next I will auto-adjust the context window for this model.",
            progress_callback,
        )
        # Check if any partial .part files exist for THIS model in target_dir.
        # Only skip auto-probe if THIS model has incomplete downloads, not due to
        # other models being downloaded elsewhere. This avoids false positives when
        # adding multiple models sequentially.
        try:
            model_partials = list(target_dir.glob('*.part'))
        except Exception:
            model_partials = []
        if model_partials:
            _emit_message(
                "Detected partial downloads for this model; skipping automatic ctx probe. "
                "Re-run with `llamacpp-superserver update --auto --model-id {}` after downloads finish.".format(mid),
                progress_callback,
            )
            # Keep desired_ctx as-is (fallback or default) and avoid probing.
            probe_config_replaced = False
        else:
            _emit_message(
                "Process: start at 8192, try a few larger values, keep a practical stable ctx, and avoid exhaustive slow probing.",
                progress_callback,
            )
            probe_model = ManagedModel(
                model_id=mid,
                repo_id=repo_id,
                quant=quant,
                filename=selected_file,
                local_path=str(local_path),
                mmproj_filename=mmproj_filename,
                mmproj_path=mmproj_path,
                ctx_size=default_ctx,
                n_gpu_layers=int(args.n_gpu_layers),
                tensor_split=args.tensor_split,
                host=args.host,
                jinja=not args.no_jinja,
                ttl=resolve_idle_ttl(args),
                description=args.description or f"{repo_id} / {selected_file}",
            )
            refresh_model_load_capabilities(probe_model)
            probe_config_replaced = True
            temporarily_unload_published_models(args, progress_callback=progress_callback)
            try:
                best_ctx, status, info = choose_auto_ctx(probe_model, args.llama_server, progress_callback=progress_callback)
            except Exception:
                if probe_config_replaced:
                    save_catalog(args.catalog, stable_catalog)
                    restore_catalog_config(args, stable_catalog, progress_callback=progress_callback, restart_service=True)
                raise
            if best_ctx is not None:
                desired_ctx = best_ctx
                auto_ctx_failed = False
                auto_ctx_error = ""
                selected_api_ctx = max(1, int(desired_ctx * resolve_api_ctx_factor(args)))
                ctx_probe_read_s = _to_float_or_none(info.get("probe_read_s"))
                ctx_probe_tokens_s = _to_float_or_none(info.get("probe_tokens_s"))
                ctx_probe_totals_s = _to_float_or_none(info.get("probe_totals_s"))
                ctx_probe_latency_ms = _to_float_or_none(info.get("probe_latency_ms"))
                ctx_probe_speed_tps = _to_float_or_none(info.get("probe_speed_tps"))
                if ctx_probe_speed_tps is None:
                    ctx_probe_speed_tps = ctx_probe_totals_s
                if ctx_probe_totals_s is None:
                    ctx_probe_totals_s = ctx_probe_speed_tps
                ctx_probe_kv_gb = _to_float_or_none(info.get("selected_ctx_gb"))
                ctx_probe_prompt_tokens = _to_int_or_none(info.get("probe_prompt_tokens"))
                _emit_message(f"Selected automatic ctx {desired_ctx} for {mid}.", progress_callback)
                _emit_message(
                    f"{mid}: probe metrics -> ctx_gb={_format_ctx_probe_gb(ctx_probe_kv_gb)} read/s={_format_ctx_probe_rate(ctx_probe_read_s)} token/s={_format_ctx_probe_rate(ctx_probe_totals_s)} latency={_format_ctx_probe_latency(ctx_probe_latency_ms)}.",
                    progress_callback,
                )
                _emit_message(
                    "Auto-ctx summary:\n" + render_auto_ctx_summary_table([
                        {
                            "MODEL": mid,
                            "CFG_CTX": str(desired_ctx),
                            "API_CTX": str(selected_api_ctx),
                            "CTX_GB": _format_ctx_probe_gb(ctx_probe_kv_gb),
                            "READ/S": _format_ctx_probe_rate(ctx_probe_read_s),
                            "TOKEN/S": _format_ctx_probe_rate(ctx_probe_totals_s),
                            "TOTAL/S": _format_ctx_probe_rate(ctx_probe_totals_s),
                            "LATENCY": _format_ctx_probe_latency(ctx_probe_latency_ms),
                            "STATUS": "selected",
                        }
                    ]),
                    progress_callback,
                )
            elif status == "min-failed":
                if probe_config_replaced:
                    save_catalog(args.catalog, stable_catalog)
                    restore_catalog_config(args, stable_catalog, progress_callback=progress_callback, restart_service=True)
                desired_ctx = int(info.get("min_ctx") or default_ctx)
                auto_ctx_failed = True
                auto_ctx_error = f"min-failed:{info.get('reason') or status}"
                ctx_probe_read_s = None
                ctx_probe_tokens_s = None
                ctx_probe_totals_s = None
                ctx_probe_latency_ms = None
                ctx_probe_speed_tps = None
                ctx_probe_kv_gb = None
                ctx_probe_prompt_tokens = None
                _emit_message(
                    f"{mid} failed the automatic probe at the minimum ctx {desired_ctx} ({auto_ctx_error}).",
                    progress_callback,
                )
                _emit_message(
                    "Auto-ctx summary:\n" + render_auto_ctx_summary_table([
                        {
                            "MODEL": mid,
                            "CFG_CTX": "ERROR",
                            "API_CTX": "ERROR",
                            "CTX_GB": _format_ctx_probe_gb(None),
                            "READ/S": _format_ctx_probe_rate(None),
                            "TOKEN/S": _format_ctx_probe_rate(None),
                            "TOTAL/S": _format_ctx_probe_rate(None),
                            "LATENCY": _format_ctx_probe_latency(None),
                            "STATUS": "ERROR",
                        }
                    ]),
                    progress_callback,
                )
                if _ask_confirmation(
                    f"{mid} did not pass the automatic ctx probe. Do you want to delete the downloaded model?",
                    progress_callback=progress_callback,
                    default=False,
                ):
                    removed = delete_downloaded_files(target_dir, to_download)
                    _emit_message(f"Deleted {removed} downloaded files for {mid}.", progress_callback)
                    raise RuntimeError(f"{mid} was deleted after failing the minimum ctx probe.")
                _emit_message(
                    f"I will add {mid} to the catalog and config anyway with fallback ctx {desired_ctx} so llama-swap can try loading it.",
                    progress_callback,
                )
                _emit_message(
                    f"Future runs will skip automatic probing for {mid} until you force it with update --auto or -ctx.",
                    progress_callback,
                )
            else:
                auto_ctx_failed = True
                auto_ctx_error = status
                ctx_probe_read_s = None
                ctx_probe_tokens_s = None
                ctx_probe_totals_s = None
                ctx_probe_latency_ms = None
                ctx_probe_speed_tps = None
                ctx_probe_kv_gb = None
                ctx_probe_prompt_tokens = None
                _emit_message(
                    f"Could not auto-tune ctx for {mid}. Keeping fallback ctx {desired_ctx} and disabling automatic re-probes for future runs.",
                    progress_callback,
                )
                _emit_message(
                    "Auto-ctx summary:\n" + render_auto_ctx_summary_table([
                        {
                            "MODEL": mid,
                            "CFG_CTX": "ERROR",
                            "API_CTX": "ERROR",
                            "CTX_GB": _format_ctx_probe_gb(None),
                            "READ/S": _format_ctx_probe_rate(None),
                            "TOKEN/S": _format_ctx_probe_rate(None),
                            "TOTAL/S": _format_ctx_probe_rate(None),
                            "LATENCY": _format_ctx_probe_latency(None),
                            "STATUS": "ERROR",
                        }
                    ]),
                    progress_callback,
                )

    # If creating a speculative variant, derive a unique model id now and
    # attribute the new catalog entry to the base model.
    if is_speculative_request:
        spec_cfg = (active_server_defaults or {}).get("speculative_defaults") if isinstance(active_server_defaults, dict) else None
        prefix = str(spec_cfg.get("id_prefix")) if isinstance(spec_cfg, dict) and spec_cfg.get("id_prefix") else "speculative-"
        candidate = f"{prefix}{base_mid}"
        allow_multiple_variants = True
        if isinstance(spec_cfg, dict) and "allow_multiple_variants" in spec_cfg:
            allow_multiple_raw = _normalize_bool_flag(spec_cfg.get("allow_multiple_variants"))
            if allow_multiple_raw is not None:
                allow_multiple_variants = allow_multiple_raw
        existing_ids = {m.model_id for m in catalog}
        if candidate in existing_ids:
            reuse_candidate = False
            if spec_draft_model_id:
                existing_candidate = next((m for m in catalog if m.model_id == candidate), None)
                existing_draft_id = ""
                if existing_candidate and isinstance(existing_candidate.spec_meta, dict):
                    existing_draft_id = str(existing_candidate.spec_meta.get("draft_model_id") or "").strip()
                if existing_draft_id and existing_draft_id == spec_draft_model_id:
                    reuse_candidate = True
            if not reuse_candidate and allow_multiple_variants:
                i = 1
                while f"{candidate}-{i}" in existing_ids:
                    i += 1
                candidate = f"{candidate}-{i}"
        mid = candidate

    paired_base_model = None
    paired_probe_once = False
    if is_speculative_request and spec_draft_model_id:
        paired_base_model = next((m for m in catalog if m.model_id == base_mid), None)
        if paired_base_model is not None:
            paired_ctx = int(getattr(paired_base_model, "ctx_size", 0) or default_ctx)
            if ctx_override is None and desired_ctx != paired_ctx:
                desired_ctx = paired_ctx
                _emit_message(
                    f"Using master ctx {desired_ctx} for paired speculative model {mid}.",
                    progress_callback,
                )
        paired_probe_once = bool(skip_ctx and ctx_override is None)

    new_m = ManagedModel(
        model_id=mid,
        repo_id=repo_id,
        quant=quant,
        filename=selected_file,
        local_path=str(local_path),
        mmproj_filename=mmproj_filename,
        mmproj_path=mmproj_path,
        ctx_size=desired_ctx,
        n_gpu_layers=int(args.n_gpu_layers),
        tensor_split=args.tensor_split,
        host=args.host,
        jinja=not args.no_jinja,
        ttl=resolve_idle_ttl(args),
        description=args.description or f"{repo_id} / {selected_file}",
        auto_ctx_failed=auto_ctx_failed,
        auto_ctx_error=auto_ctx_error,
        ctx_probe_read_s=ctx_probe_read_s,
        ctx_probe_tokens_s=ctx_probe_tokens_s,
        ctx_probe_totals_s=ctx_probe_totals_s,
        ctx_probe_latency_ms=ctx_probe_latency_ms,
        ctx_probe_speed_tps=ctx_probe_speed_tps,
        ctx_probe_kv_gb=ctx_probe_kv_gb,
        ctx_probe_prompt_tokens=ctx_probe_prompt_tokens,
        speculative=is_speculative_request,
        spec_variant_of=(base_mid if is_speculative_request else None),
    )
    if is_speculative_request:
        # Avoid alias collisions between base and speculative entries.
        new_m.aliases = []
        if existing and existing.server_overrides:
            new_m.server_overrides = dict(normalize_server_overrides(existing.server_overrides))

        if spec_draft_model_id:
            draft_model = next((m for m in catalog if m.model_id == spec_draft_model_id), None)
            if draft_model is None:
                raise RuntimeError(f"Speculative draft model '{spec_draft_model_id}' was not found in catalog.")

            spec_overrides = dict(normalize_server_overrides(new_m.server_overrides))
            spec_overrides["model_draft"] = str(draft_model.local_path)

            spec_cfg = (active_server_defaults or {}).get("speculative_defaults") if isinstance(active_server_defaults, dict) else None
            if isinstance(spec_cfg, dict):
                normalized_spec_defaults = normalize_server_overrides(spec_cfg)
                for key in ("draft", "draft_min", "draft_p_min", "ctx_size_draft", "n_gpu_layers_draft"):
                    if key in normalized_spec_defaults and key not in spec_overrides:
                        spec_overrides[key] = normalized_spec_defaults[key]
            if "draft" not in spec_overrides:
                spec_overrides["draft"] = 16

            new_m.server_overrides = spec_overrides
            new_m.spec_meta = dict(new_m.spec_meta or {})
            new_m.spec_meta.update(
                {
                    "base_model_id": base_mid,
                    "draft_model_id": draft_model.model_id,
                    "draft_repo_id": draft_model.repo_id,
                    "draft_filename": draft_model.filename,
                    "draft_local_path": draft_model.local_path,
                }
            )

    if paired_probe_once:
        _emit_message(
            f"{new_m.model_id}: running one-shot paired speculative probe at ctx {new_m.ctx_size} for performance metrics.",
            progress_callback,
        )
        probe_ok, probe_reason, probe_info = _probe_fixed_ctx_metrics_once(new_m, args.llama_server, int(new_m.ctx_size))
        if probe_ok:
            new_m.auto_ctx_failed = False
            new_m.auto_ctx_error = ""
            apply_ctx_probe_metrics(new_m, probe_info)
            _emit_message(
                (
                    f"{new_m.model_id}: paired probe metrics -> "
                    f"ctx_gb={_format_ctx_probe_gb(new_m.ctx_probe_kv_gb)} "
                    f"read/s={_format_ctx_probe_rate(new_m.ctx_probe_read_s)} "
                    f"token/s={_format_ctx_probe_rate(new_m.ctx_probe_totals_s)} "
                    f"latency={_format_ctx_probe_latency(new_m.ctx_probe_latency_ms)}."
                ),
                progress_callback,
            )
        else:
            clear_ctx_probe_metrics(new_m)
            new_m.auto_ctx_failed = False
            new_m.auto_ctx_error = ""
            _emit_message(
                f"{new_m.model_id}: paired probe for metrics failed ({probe_reason}). Keeping ctx {new_m.ctx_size} from master.",
                progress_callback,
            )
            trace_tail = _trace_tail_from_reason(probe_reason, lines=30)
            if trace_tail:
                _emit_message(
                    f"{new_m.model_id}: paired probe trace tail:\n{trace_tail}",
                    progress_callback,
                )

    refresh_model_load_capabilities(new_m)
    new_cat = [m for m in catalog if m.model_id != mid] + [new_m]
    save_catalog(args.catalog, new_cat)
    if defer_publish:
        _emit_message(f"Catalog updated for {mid}; publish/load deferred.", progress_callback)
        if probe_config_replaced:
            restore_catalog_config(args, stable_catalog, progress_callback=progress_callback, restart_service=True)
        return mid
    gpu_conflict = get_gpu_conflict_message(mid, new_cat, args.public_host, args.public_port)
    if gpu_conflict:
        _emit_message(gpu_conflict, progress_callback)
        save_catalog(args.catalog, stable_catalog)
        raise RuntimeError(gpu_conflict)
    try:
        apply_config_and_wait(
            new_cat,
            args.config,
            args.llama_server,
            args.start_port,
            mid,
            args.public_host,
            args.public_port,
            progress_callback=progress_callback,
            server_defaults=active_server_defaults,
        )
    except Exception:
        save_catalog(args.catalog, stable_catalog)
        restore_catalog_config(args, stable_catalog, progress_callback=progress_callback, restart_service=True)
        raise
    _emit_default_ctx_update_hint(mid, new_m.ctx_size, default_ctx, progress_callback)
    return mid

def wait_for_model_absent(model_id, host, port, timeout=35):
    host = _normalize_client_host(host)
    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + timeout
    spinner = Spinner(f"\033[36mWaiting for {model_id} removal...\033[0m ")
    spinner.start()
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                if all(m.get("id") != model_id for m in r.json().get("data", [])):
                    spinner.stop()
                    print(f"\033[32mModel {model_id} is no longer published.\033[0m")
                    return True
        except:
            pass
        time.sleep(1.5)
    spinner.stop()
    return False

def apply_config_and_wait_absent(
    catalog,
    config_path,
    llama_server,
    start_port,
    model_id,
    host,
    port,
    progress_callback = None,
    settle_time = 3.0,
    timeout = 45.0,
    server_defaults: dict[str, object] | None = None,
):
    render_llamaswap_config(
        catalog,
        config_path,
        llama_server,
        start_port,
        resolve_idle_ttl(),
        server_defaults=server_defaults,
    )
    _emit_message("Config updated. Waiting for llama-swap --watch-config...", progress_callback)
    time.sleep(settle_time)
    if wait_for_model_absent(model_id, host, port, timeout=timeout):
        return True
    raise RuntimeError(
        f"Model {model_id} is still published after updating {config_path}. "
        "Ensure llama-swap is running with --watch-config and is watching that config file."
    )


def _model_candidate_files(model: ManagedModel) -> list[Path]:
    candidates: set[Path] = set()

    local_raw = str(getattr(model, "local_path", "") or "").strip()
    if local_raw:
        local_path = Path(local_raw)
        filename = Path(str(getattr(model, "filename", "") or "")).name
        if "-00001-of-" in filename and local_path.parent.exists():
            prefix = filename.split("-00001-of-")[0]
            for pattern in (f"{prefix}-*-of-*.gguf", f"{prefix}-*-of-*.gguf.part"):
                for shard in local_path.parent.glob(pattern):
                    if shard.is_file():
                        candidates.add(shard)
        else:
            candidates.add(local_path)
            candidates.add(local_path.with_name(local_path.name + ".part"))

    mmproj_raw = str(getattr(model, "mmproj_path", "") or "").strip()
    if mmproj_raw:
        mmproj_path = Path(mmproj_raw)
        candidates.add(mmproj_path)
        candidates.add(mmproj_path.with_name(mmproj_path.name + ".part"))

    return sorted(candidates, key=lambda item: str(item))


def _catalog_referenced_paths(catalog: list[ManagedModel]) -> set[str]:
    refs: set[str] = set()
    for item in catalog:
        for raw in (getattr(item, "local_path", ""), getattr(item, "mmproj_path", "")):
            value = str(raw or "").strip()
            if value:
                refs.add(_safe_realpath(value))
    return refs


def _prune_empty_dirs_under_root(path: Path, root: Path, progress_callback = None) -> None:
    try:
        root_real = root.expanduser().resolve(strict=False)
        current = path.expanduser().resolve(strict=False)
    except Exception:
        return

    while True:
        try:
            current.relative_to(root_real)
        except Exception:
            break

        if current == root_real:
            break

        try:
            current.rmdir()
            _emit_message(f"Removed empty directory {current}.", progress_callback)
        except FileNotFoundError:
            pass
        except OSError:
            break

        current = current.parent


def _delete_model_files_from_disk(
    model: ManagedModel,
    remaining: list[ManagedModel],
    models_dir: Path,
    progress_callback = None,
) -> bool:
    referenced = _catalog_referenced_paths(remaining)
    deleted_any = False

    for candidate in _model_candidate_files(model):
        if _safe_realpath(str(candidate)) in referenced:
            continue
        if not candidate.exists():
            continue
        try:
            candidate.unlink()
            deleted_any = True
            _emit_message(f"Deleted local file {candidate}.", progress_callback)
        except IsADirectoryError:
            shutil.rmtree(candidate, ignore_errors=True)
            deleted_any = True
            _emit_message(f"Deleted local directory {candidate}.", progress_callback)
        except Exception as exc:
            _emit_message(f"Could not delete local file {candidate}: {exc}", progress_callback)
            continue

        _prune_empty_dirs_under_root(candidate.parent, models_dir, progress_callback)

    # Fallback: if this repo is no longer referenced and directory still exists, remove it.
    if not any(item.repo_id == model.repo_id for item in remaining):
        repo_dir = models_dir / model.repo_id
        if repo_dir.exists():
            try:
                shutil.rmtree(repo_dir)
                deleted_any = True
                _emit_message(f"Deleted local files under {repo_dir}.", progress_callback)
            except Exception as exc:
                _emit_message(f"Could not delete local files under {repo_dir}: {exc}", progress_callback)
        _prune_empty_dirs_under_root(repo_dir, models_dir, progress_callback)

    return deleted_any


def _orphan_file_aliases(path: Path) -> set[str]:
    aliases: set[str] = set()
    candidates = [path.name, path.stem]
    for value in candidates:
        base = re.sub(r"(?i)\.gguf$", "", value)
        aliases.add(base)
        aliases.add(re.sub(r"[-._]?\d{5}-of-\d{5}$", "", base))
        aliases.add(re.sub(r"[-._]?\d+-of-\d+$", "", base))
    return {_canonical_model_ref(item) for item in aliases if _canonical_model_ref(item)}


def _remove_orphan_files_by_reference(reference: str, models_dir: Path, progress_callback = None) -> int:
    canonical_reference = _canonical_model_ref(reference)
    if not canonical_reference:
        return 0

    patterns = ("*.gguf", "*.gguf.part")
    deleted = 0
    for pattern in patterns:
        for file_path in models_dir.rglob(pattern):
            if not file_path.is_file():
                continue
            if canonical_reference not in _orphan_file_aliases(file_path):
                continue
            try:
                file_path.unlink()
                deleted += 1
                _emit_message(f"Deleted local file {file_path}.", progress_callback)
                _prune_empty_dirs_under_root(file_path.parent, models_dir, progress_callback)
            except Exception as exc:
                _emit_message(f"Could not delete local file {file_path}: {exc}", progress_callback)
    return deleted

def remove_model(args, progress_callback = None):
    try:
        is_owner = (os.getuid() == 0 or os.getuid() == os.stat(args.catalog.parent).st_uid)
    except:
        is_owner = False

    if not is_owner:
        try:
            return run_manager_command("remove", args)
        except RuntimeError as e:
            raise e
        except Exception as e:
            raise manager_unavailable_error(e)

    catalog = load_catalog(args.catalog, _args_server_config_path(args))
    try:
        model = resolve_catalog_model(catalog, target=args.repo, repo_ref=args.hf, model_id=args.model_id, filename=args.file)
    except RuntimeError as exc:
        if str(exc) == "Model not found in catalog.":
            reference = args.repo or args.hf or args.model_id or args.file
            if reference and getattr(args, "delete_files", True):
                # Attempt to remove orphan files even if --delete-files was not
                # explicitly requested. This helps cleanup partially downloaded
                # artifacts when the catalog entry is missing.
                removed = _remove_orphan_files_by_reference(str(reference), args.models_dir, progress_callback)
                if removed > 0:
                    _emit_message(
                        (
                            f"Model not found in catalog. Removed {removed} orphan file(s) "
                            f"matching '{reference}' from disk."
                        ),
                        progress_callback,
                    )
                    return str(reference)
        raise
    remaining = [m for m in catalog if m.model_id != model.model_id]

    save_catalog(args.catalog, remaining)
    apply_config_and_wait_absent(
        remaining,
        args.config,
        args.llama_server,
        args.start_port,
        model.model_id,
        args.public_host,
        args.public_port,
        progress_callback=progress_callback,
        server_defaults=resolve_llama_server_defaults(args),
    )

    if args.delete_files:
        deleted_any = _delete_model_files_from_disk(model, remaining, args.models_dir, progress_callback)
        if not deleted_any:
            _emit_message("No removable local files found for this model.", progress_callback)

    return model.model_id

def get_model_storage_info(model: ManagedModel):
    local_path = Path(model.local_path) if model.local_path else None
    ready_files = []
    part_files = []
    inaccessible = False

    try:
        if local_path is not None:
            if "-00001-of-" in model.filename and local_path.parent.exists():
                shard_name = Path(model.filename).name
                prefix = shard_name.split("-00001-of-")[0]
                ready_files = sorted(p for p in local_path.parent.glob(f"{prefix}-*-of-*.gguf") if p.is_file())
                part_files = sorted(p for p in local_path.parent.glob(f"{prefix}-*-of-*.gguf.part") if p.is_file())
            else:
                if local_path.exists():
                    ready_files = [local_path]
                part_path = local_path.with_name(local_path.name + ".part")
                if part_path.exists():
                    part_files = [part_path]
    except OSError:
        inaccessible = True

    total_size = 0
    for p in [*ready_files, *part_files]:
        try:
            total_size += p.stat().st_size
        except OSError:
            inaccessible = True
    file_count = len(ready_files) + len(part_files)

    if inaccessible:
        status = "inaccessible"
    elif part_files:
        status = "partial"
    elif ready_files:
        status = "ready"
    else:
        status = "missing"

    return {
        "size": total_size,
        "file_count": file_count,
        "status": status,
    }

def _read_gguf_string(fh):
    length_raw = fh.read(8)
    if len(length_raw) != 8:
        raise EOFError("Unexpected EOF while reading GGUF string length")
    length = struct.unpack("<Q", length_raw)[0]
    data = fh.read(length)
    if len(data) != length:
        raise EOFError("Unexpected EOF while reading GGUF string")
    return data.decode("utf-8", errors="ignore")

def _read_gguf_value(fh, value_type):
    readers = {
        0: ("<B", 1),
        1: ("<b", 1),
        2: ("<H", 2),
        3: ("<h", 2),
        4: ("<I", 4),
        5: ("<i", 4),
        6: ("<f", 4),
        7: (None, 1),
        10: ("<Q", 8),
        11: ("<q", 8),
        12: ("<d", 8),
    }
    if value_type == 8:
        return _read_gguf_string(fh)
    if value_type == 9:
        elem_type_raw = fh.read(4)
        count_raw = fh.read(8)
        if len(elem_type_raw) != 4 or len(count_raw) != 8:
            raise EOFError("Unexpected EOF while reading GGUF array header")
        elem_type = struct.unpack("<I", elem_type_raw)[0]
        count = struct.unpack("<Q", count_raw)[0]
        return [_read_gguf_value(fh, elem_type) for _ in range(count)]
    if value_type == 7:
        raw = fh.read(1)
        if len(raw) != 1:
            raise EOFError("Unexpected EOF while reading GGUF bool")
        return raw != b"\x00"
    if value_type not in readers:
        raise ValueError(f"Unsupported GGUF value type: {value_type}")
    fmt, size = readers[value_type]
    raw = fh.read(size)
    if len(raw) != size:
        raise EOFError("Unexpected EOF while reading GGUF value")
    return struct.unpack(fmt, raw)[0]


def _normalize_load_capabilities(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = str(raw or "").strip().lower()
        if not token:
            continue
        if token not in seen:
            normalized.append(token)
            seen.add(token)
    return normalized


def _extract_load_capabilities_from_value(value: object) -> list[str]:
    tokens: list[str] = []
    candidates: list[str] = []
    if isinstance(value, str):
        candidates.append(value)
    elif isinstance(value, list):
        candidates.extend(str(item) for item in value if isinstance(item, (str, int, float)))
    for candidate in candidates:
        lowered = candidate.strip().lower()
        if not lowered:
            continue
        for match in re.findall(r"(?:image-text-to-text|text-to-text|image-to-text|image|vision)", lowered):
            token = "image-text-to-text" if match == "vision" else match
            tokens.append(token)
    return _normalize_load_capabilities(tokens)


def detect_model_load_capabilities(model: ManagedModel) -> list[str]:
    local_path = Path(model.local_path) if model.local_path else None
    if local_path is None or not local_path.exists():
        return _normalize_load_capabilities(model.load_capabilities)

    detected: list[str] = []
    seen: set[str] = set()
    try:
        with local_path.open("rb") as fh:
            if fh.read(4) != b"GGUF":
                return _normalize_load_capabilities(model.load_capabilities)
            version_raw = fh.read(4)
            tensor_count_raw = fh.read(8)
            kv_count_raw = fh.read(8)
            if len(version_raw) != 4 or len(tensor_count_raw) != 8 or len(kv_count_raw) != 8:
                return _normalize_load_capabilities(model.load_capabilities)
            version = struct.unpack("<I", version_raw)[0]
            if version not in (2, 3):
                return _normalize_load_capabilities(model.load_capabilities)
            kv_count = struct.unpack("<Q", kv_count_raw)[0]
            for _ in range(kv_count):
                key = _read_gguf_string(fh)
                value_type_raw = fh.read(4)
                if len(value_type_raw) != 4:
                    break
                value_type = struct.unpack("<I", value_type_raw)[0]
                value = _read_gguf_value(fh, value_type)
                lowered = key.lower()
                if not any(hint in lowered for hint in ("architecture", "capabil", "modalit", "vision", "image")):
                    continue
                for token in _extract_load_capabilities_from_value(value):
                    if token not in seen:
                        detected.append(token)
                        seen.add(token)
    except Exception:
        return _normalize_load_capabilities(model.load_capabilities)

    if not detected and _has_vision_runtime(model):
        detected.extend(["image", "image-text-to-text"])
    if not detected:
        detected.append("text-to-text")
    return _normalize_load_capabilities(detected)


def refresh_model_load_capabilities(model: ManagedModel) -> list[str]:
    capabilities = detect_model_load_capabilities(model)
    model.load_capabilities = capabilities
    return capabilities


def refresh_models_load_capabilities(models: list[ManagedModel]) -> None:
    for model in models:
        refresh_model_load_capabilities(model)

def get_model_context_size(model: ManagedModel):
    local_path = Path(model.local_path) if model.local_path else None
    if local_path is None or not local_path.exists():
        return None
    try:
        with local_path.open("rb") as fh:
            if fh.read(4) != b"GGUF":
                return None
            version_raw = fh.read(4)
            tensor_count_raw = fh.read(8)
            kv_count_raw = fh.read(8)
            if len(version_raw) != 4 or len(tensor_count_raw) != 8 or len(kv_count_raw) != 8:
                return None
            version = struct.unpack("<I", version_raw)[0]
            if version not in (2, 3):
                return None
            kv_count = struct.unpack("<Q", kv_count_raw)[0]
            for _ in range(kv_count):
                key = _read_gguf_string(fh)
                value_type_raw = fh.read(4)
                if len(value_type_raw) != 4:
                    return None
                value_type = struct.unpack("<I", value_type_raw)[0]
                value = _read_gguf_value(fh, value_type)
                lowered = key.lower()
                if lowered.endswith(".context_length") or lowered.endswith(".n_ctx_train"):
                    if isinstance(value, (int, float)):
                        return int(value)
    except Exception:
        return None
    return None

def sync_catalog_context_sizes(catalog: list[ManagedModel]):
    updated = 0
    missing = 0
    for model in catalog:
        gguf_ctx = get_model_context_size(model)
        if gguf_ctx is None:
            missing += 1
            continue
        if model.ctx_size != gguf_ctx:
            model.ctx_size = gguf_ctx
            updated += 1
    return updated, missing

def _truncate(text, width):
    value = str(text)
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."

def _safe_realpath(path_str: str) -> str:
    try:
        return str(Path(path_str).expanduser().resolve(strict=False))
    except Exception:
        return path_str


def get_published_model_ids(host=DEFAULT_PUBLIC_HOST, port=DEFAULT_PUBLIC_PORT) -> set[str]:
    try:
        host = _normalize_client_host(host)
        r = requests.get(f"http://{host}:{port}/v1/models", timeout=1.5)
        if r.status_code != 200:
            return set()
        return {item.get("id") for item in r.json().get("data", []) if item.get("id")}
    except Exception:
        return set()


def get_llama_server_processes() -> list[dict]:
    try:
        result = subprocess.run(
            ["pgrep", "-af", "llama-server --port"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    processes = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        cmdline = parts[1]
        model_path = ""
        match = re.search(r"--model\s+(\S+)", cmdline)
        if match:
            model_path = match.group(1).strip("'\"")
        processes.append({"pid": pid, "cmdline": cmdline, "model_path": _safe_realpath(model_path)})
    return processes


def get_loaded_catalog_model_ids(catalog: list[ManagedModel]) -> set[str]:
    processes = get_llama_server_processes()
    process_by_model = {proc["model_path"]: proc for proc in processes}
    return {
        model.model_id
        for model in catalog
        if process_by_model.get(_safe_realpath(model.local_path)) is not None
    }


def get_gpu_process_map() -> dict[int, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return {}

    gpu_map: dict[int, str] = {}
    for line in result.stdout.splitlines():
        chunks = [chunk.strip() for chunk in line.split(",")]
        if len(chunks) < 2 or not chunks[0].isdigit():
            continue
        used = chunks[1]
        gpu_map[int(chunks[0])] = used
    return gpu_map


def _describe_pid(pid: int) -> str:
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", errors="ignore").replace("\x00", " ").strip()
        if cmdline:
            return cmdline
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        value = result.stdout.strip()
        if value:
            return value
    except Exception:
        pass
    return "unknown-process"


def _is_ollama_process(pid: int) -> bool:
    description = _describe_pid(pid).lower()
    return "ollama" in description


def get_gpu_conflict_message(model_id: str, catalog: list[ManagedModel], host=DEFAULT_PUBLIC_HOST, port=DEFAULT_PUBLIC_PORT) -> str | None:
    """Generate a user-friendly error message for GPU conflicts.
    
    Uses only the local llamacpp-superserver installation for model management.
    If a GPU conflict is detected, suggests using 'llamacpp-superserver unload' to free resources.
    """
    published_models = get_published_model_ids(host, port)
    if model_id in published_models:
        return None
    processes = get_llama_server_processes()
    process_by_pid = {proc["pid"]: proc for proc in processes}
    model_by_path = {_safe_realpath(model.local_path): model.model_id for model in catalog}
    gpu_process_map = get_gpu_process_map()
    if not gpu_process_map:
        return None
    conflicts: list[str] = []
    for pid, used_mem in sorted(gpu_process_map.items()):
        if _is_ollama_process(pid):
            continue
        process = process_by_pid.get(pid)
        if process is None:
            continue
        running_model = model_by_path.get(process.get("model_path") or "")
        if running_model == model_id:
            continue
        if running_model:
            conflicts.append(f"{running_model} (pid {pid}, {used_mem} MiB)")
    if not conflicts:
        return None
    joined = "; ".join(conflicts[:4])
    if len(conflicts) > 4:
        joined += f"; +{len(conflicts) - 4} more"
    return (
        f"Cannot load model '{model_id}' because the GPU is already in use: {joined}. "
        "Use 'llamacpp-superserver unload <model>' to free resources, or wait for those workloads to finish."
    )


def get_configured_idle_ttl(config_path: Path | None = None, fallback: int = DEFAULT_IDLE_TTL) -> int:
    cfg_path = config_path or DEFAULT_CONFIG_PATH
    try:
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return fallback
    models = payload.get("models") or {}
    for model_config in models.values():
        ttl = model_config.get("ttl")
        if ttl is not None:
            try:
                return int(ttl)
            except Exception:
                return fallback
    return fallback


def get_effective_idle_ttl(args) -> int:
    return get_configured_idle_ttl(Path(args.config), resolve_idle_ttl(args))


def classify_runtime(model: ManagedModel, loaded: bool, process_pid: int | None, gpu_process_map: dict[int, str]) -> tuple[str, str]:
    planned = "cpu" if int(model.n_gpu_layers) == 0 else ("full-gpu" if int(model.n_gpu_layers) >= 999 else f"partial-gpu({model.n_gpu_layers})")
    if not loaded or process_pid is None:
        return "unloaded", planned
    if process_pid in gpu_process_map:
        if int(model.n_gpu_layers) == 0:
            return "gpu-active", planned
        if int(model.n_gpu_layers) >= 999:
            return "100%-gpu", planned
        return "partial-gpu", planned
    # Inside containers, nvidia-smi may expose host PIDs instead of container PIDs.
    if gpu_process_map and int(model.n_gpu_layers) > 0:
        if int(model.n_gpu_layers) >= 999:
            return "100%-gpu", planned
        return "partial-gpu", planned
    if int(model.n_gpu_layers) == 0:
        return "cpu", planned
    return "cpu-fallback", planned

def render_models_table(catalog: list[ManagedModel], host=DEFAULT_PUBLIC_HOST, port=DEFAULT_PUBLIC_PORT, idle_ttl=DEFAULT_IDLE_TTL) -> str:
    if not catalog:
        return "No models in catalog."

    published_models = get_published_model_ids(host, port)
    processes = get_llama_server_processes()
    process_by_model = {proc["model_path"]: proc for proc in processes}
    gpu_process_map = get_gpu_process_map()
    rows = []
    for model in catalog:
        storage = get_model_storage_info(model)
        process = process_by_model.get(_safe_realpath(model.local_path))
        published = model.model_id in published_models
        loaded = process is not None
        runtime, planned = classify_runtime(model, loaded, process["pid"] if process else None, gpu_process_map)
        ctx_status = ctx_evaluation_status(model)
        totals_num = _to_float_or_none(model.ctx_probe_totals_s)
        rows.append({
            "MODEL_ID": model.model_id,
            "PUBLISHED": "yes" if published else "no",
            "LOADED": "yes" if loaded else "no",
            "RUNTIME": runtime,
            "GPU_PLAN": planned,
            "PID": str(process["pid"]) if process else "-",
            "CFG_CTX": _display_cfg_ctx(model),
            "API_CTX": _display_api_ctx(model),
            "CTX_GB": _format_ctx_probe_gb(_to_float_or_none(model.ctx_probe_kv_gb)),
            "READ/S": _format_ctx_probe_rate(_to_float_or_none(model.ctx_probe_read_s)),
            "TOKEN/S": _format_ctx_probe_rate(_to_float_or_none(model.ctx_probe_tokens_s)),
            "TOTAL/S": _format_ctx_probe_rate(totals_num),
            "__totals_s": totals_num or 0.0,
            "SIZE": _format_bytes(storage["size"]),
            "FILES": str(storage["file_count"]),
            "STATUS": ctx_status if ctx_status == "ERROR" else storage["status"],
            "REPO": model.repo_id,
        })

    # Order rows so fastest (by total tokens/s) appear first; None/NC treated as 0.
    rows.sort(key=lambda r: (r.get("__totals_s") or 0.0), reverse=True)

    columns = [
        ("MODEL_ID", 999),
        ("PUBLISHED", 9),
        ("LOADED", 6),
        ("RUNTIME", 12),
        ("GPU_PLAN", 18),
        ("PID", 8),
        ("CFG_CTX", 8),
        ("API_CTX", 8),
        ("CTX_GB", 8),
        ("READ/S", 12),
        ("TOKEN/S", 12),
        ("TOTAL/S", 12),
        ("SIZE", 10),
        ("FILES", 5),
        ("STATUS", 8),
        ("REPO", 36),
    ]

    widths = {}
    for key, cap in columns:
        widths[key] = min(cap, max(len(key), max(len(str(row[key])) for row in rows)))

    lines = []
    header = "  ".join(key.ljust(widths[key]) for key, _ in columns)
    sep = "  ".join("-" * widths[key] for key, _ in columns)
    lines.append(header)
    lines.append(sep)
    for row in rows:
        lines.append("  ".join(_truncate(row[key], widths[key]).ljust(widths[key]) for key, _ in columns))
    lines.append(sep)
    lines.append(f"Global idle TTL: {idle_ttl}s")
    return "\n".join(lines) + "\n\n"


def render_auto_ctx_summary_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No auto-ctx probe data."
    columns = [
        ("MODEL", 44),
        ("CFG_CTX", 9),
        ("API_CTX", 9),
        ("CTX_GB", 8),
        ("READ/S", 12),
        ("TOKEN/S", 12),
        ("TOTAL/S", 12),
        ("LATENCY", 10),
        ("STATUS", 12),
    ]
    widths: dict[str, int] = {}
    for key, cap in columns:
        widths[key] = min(cap, max(len(key), max(len(str(row.get(key, ""))) for row in rows)))
    header = "  ".join(key.ljust(widths[key]) for key, _ in columns)
    sep = "  ".join("-" * widths[key] for key, _ in columns)
    lines = [header, sep]
    for row in rows:
        lines.append("  ".join(_truncate(str(row.get(key, "")), widths[key]).ljust(widths[key]) for key, _ in columns))
    lines.append(sep)
    return "\n".join(lines)

def build_model_ctx_payload(model: ManagedModel) -> dict:
    load_capabilities = refresh_model_load_capabilities(model)
    gguf_ctx = get_model_context_size(model)
    cfg_ctx = displayed_configured_ctx(model)
    api_ctx = displayed_api_ctx(model)
    ctx_status = ctx_evaluation_status(model)
    probe_metrics = _ctx_probe_api_metrics(model)
    return {
        "name": model.model_id,
        "model": model.model_id,
        "size": get_model_storage_info(model)["size"],
        "details": {
            "format": "gguf",
            "family": "llama.cpp",
            "parameter_size": model.quant or "",
            "configured_ctx": cfg_ctx if ctx_status != "ERROR" else None,
            "api_ctx": api_ctx if ctx_status != "ERROR" else None,
            "api_ctx_status": ctx_status,
            "max_ctx": cfg_ctx if ctx_status != "ERROR" else None,
            "gguf_ctx": gguf_ctx,
            "load_capabilities": load_capabilities,
            "speculative": bool(getattr(model, "speculative", False)),
            "spec_variant_of": getattr(model, "spec_variant_of", None),
            **probe_metrics,
        },
    }


def build_openai_model_payload(model: ManagedModel) -> dict:
    load_capabilities = refresh_model_load_capabilities(model)
    gguf_ctx = get_model_context_size(model)
    cfg_ctx = displayed_configured_ctx(model)
    api_ctx = displayed_api_ctx(model)
    ctx_status = ctx_evaluation_status(model)
    probe_metrics = _ctx_probe_api_metrics(model)
    return {
        "id": model.model_id,
        "object": "model",
        "created": 0,
        "owned_by": "llama-swap",
        "metadata": {
            "configured_context_length": cfg_ctx if ctx_status != "ERROR" else None,
            "api_context_length": api_ctx if ctx_status != "ERROR" else None,
            "api_context_status": ctx_status,
            "context_length": cfg_ctx if ctx_status != "ERROR" else None,
            "gguf_context_length": gguf_ctx,
            "load_capabilities": load_capabilities,
            "vision": _has_vision_runtime(model),
            "speculative": bool(getattr(model, "speculative", False)),
            "spec_variant_of": getattr(model, "spec_variant_of", None),
            **probe_metrics,
        },
    }


def _looks_like_vision_model(model: ManagedModel) -> bool:
    if model.mmproj_path:
        return True
    haystack = " ".join([
        model.model_id or "",
        model.repo_id or "",
        model.filename or "",
        model.description or "",
    ]).lower()
    return any(token in haystack for token in ("vision", "vl", "llava", "qwen2.5-vl", "qwen3-vl", "minicpm-v", "internvl"))


def _has_vision_runtime(model: ManagedModel) -> bool:
    if bool(model.mmproj_path and Path(model.mmproj_path).exists()):
        return True
    capabilities = _normalize_load_capabilities(getattr(model, "load_capabilities", []))
    return any(token in capabilities for token in ("image", "image-text-to-text", "image-to-text", "vision"))


def _infer_model_family(model: ManagedModel) -> str:
    haystack = " ".join([model.model_id or "", model.repo_id or "", model.filename or ""]).lower()
    for family in ("qwen", "llama", "mistral", "gemma", "phi", "deepseek", "minicpm", "llava"):
        if family in haystack:
            return family
    return "llama.cpp"


def _infer_parameter_size(model: ManagedModel) -> str:
    quant = (model.quant or "").strip()
    if quant:
        return quant
    stem = Path(model.filename or model.model_id).stem
    match = re.search(r"(\d+(?:\.\d+)?[bBmM])", stem)
    return match.group(1).upper() if match else ""


def _model_digest(model: ManagedModel) -> str:
    seed = f"{model.model_id}|{model.repo_id}|{model.filename}|{model.local_path}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def build_ollama_model_payload(model: ManagedModel, loaded: bool = False, process: dict | None = None, gpu_process_map: dict[int, str] | None = None) -> dict:
    load_capabilities = refresh_model_load_capabilities(model)
    storage = get_model_storage_info(model)
    gguf_ctx = get_model_context_size(model)
    cfg_ctx = displayed_configured_ctx(model)
    api_ctx = displayed_api_ctx(model)
    ctx_status = ctx_evaluation_status(model)
    probe_metrics = _ctx_probe_api_metrics(model)
    local_path = Path(model.local_path)
    modified = None
    try:
        modified = datetime.fromtimestamp(local_path.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        modified = datetime.now(timezone.utc).isoformat()
    details = {
        "parent_model": getattr(model, "spec_variant_of", "") or "",
        "format": "gguf",
        "family": _infer_model_family(model),
        "families": [_infer_model_family(model)],
        "parameter_size": _infer_parameter_size(model),
        "quantization_level": model.quant or "",
        "configured_context_length": cfg_ctx if ctx_status != "ERROR" else None,
        "api_context_length": api_ctx if ctx_status != "ERROR" else None,
        "api_context_status": ctx_status,
        "context_length": cfg_ctx if ctx_status != "ERROR" else None,
        "gguf_context_length": gguf_ctx,
        "load_capabilities": load_capabilities,
        "speculative": bool(getattr(model, "speculative", False)),
        "vision": _has_vision_runtime(model),
        **probe_metrics,
    }
    payload = {
        "name": model.model_id,
        "model": model.model_id,
        "modified_at": modified,
        "size": storage["size"],
        "digest": _model_digest(model),
        "details": details,
        "model_info": {
            "llamacpp.configured_context_length": cfg_ctx if ctx_status != "ERROR" else None,
            "llamacpp.api_context_length": api_ctx if ctx_status != "ERROR" else None,
            "llamacpp.api_context_status": ctx_status,
            "llamacpp.context_length": cfg_ctx if ctx_status != "ERROR" else None,
            "llamacpp.gguf_context_length": gguf_ctx,
            "llamacpp.load_capabilities": load_capabilities,
            "llamacpp.ctx_probe_read_s": probe_metrics["ctx_probe_read_s"],
            "llamacpp.ctx_probe_tokens_s": probe_metrics["ctx_probe_tokens_s"],
            "llamacpp.ctx_probe_totals_s": probe_metrics["ctx_probe_totals_s"],
            "llamacpp.ctx_probe_latency_ms": probe_metrics["ctx_probe_latency_ms"],
            "llamacpp.ctx_probe_speed_tps": probe_metrics["ctx_probe_speed_tps"],
            "llamacpp.ctx_probe_kv_gb": probe_metrics["ctx_probe_kv_gb"],
            "llamacpp.ctx_probe_read": probe_metrics["ctx_probe_read"],
            "llamacpp.ctx_probe_tokens": probe_metrics["ctx_probe_tokens"],
            "llamacpp.ctx_probe_totals": probe_metrics["ctx_probe_totals"],
            "llamacpp.ctx_probe_latency": probe_metrics["ctx_probe_latency"],
            "llamacpp.ctx_probe_speed": probe_metrics["ctx_probe_speed"],
            "llamacpp.ctx_probe_kv": probe_metrics["ctx_probe_kv"],
        },
    }
    if loaded:
        runtime, _planned = classify_runtime(model, True, process["pid"] if process else None, gpu_process_map or {})
        payload["expires_at"] = None
        payload["size_vram"] = storage["size"] if runtime in {"100%-gpu", "partial-gpu", "gpu-active"} else 0
    return payload


def _derived_filename_aliases(model: ManagedModel) -> list[str]:
    filename = (model.filename or "").strip()
    if not filename:
        return []

    candidates = [filename]
    basename = Path(filename).name
    if basename and basename not in candidates:
        candidates.append(basename)

    aliases: list[str] = []

    def _append(value: str) -> None:
        v = (value or "").strip()
        if v and v not in aliases:
            aliases.append(v)

    for item in candidates:
        _append(item)
        no_ext = re.sub(r"(?i)\.gguf$", "", item)
        _append(no_ext)
        no_shard = re.sub(r"[-._]?\d+-of-\d+$", "", no_ext)
        _append(no_shard)
    return aliases


def _derived_repo_aliases(model: ManagedModel) -> list[str]:
    repo_id = (model.repo_id or "").strip()
    if not repo_id:
        return []

    quant = (model.quant or "").strip()
    aliases: list[str] = []

    def _append(value: str) -> None:
        v = (value or "").strip()
        if v and v not in aliases:
            aliases.append(v)

    _append(f"hf.co/{repo_id}")
    _append(repo_id)
    if quant:
        _append(f"hf.co/{repo_id}:{quant}")
        _append(f"{repo_id}:{quant}")
    return aliases


def model_name_aliases(model: ManagedModel) -> list[str]:
    aliases: list[str] = []

    def _append(value: str) -> None:
        v = (value or "").strip()
        if v and v not in aliases:
            aliases.append(v)

    for alias in model.aliases:
        _append(alias)
    for alias in _derived_filename_aliases(model):
        _append(alias)
    for alias in _derived_repo_aliases(model):
        _append(alias)
    return aliases


def _ollama_message_to_openai(message: dict) -> dict:
    role = message.get("role") or "user"
    if role == "tool":
        role = "user"
    content = message.get("content") or ""
    images = message.get("images") or []
    if isinstance(content, list):
        parts = []
        text_chunks = []
        for item in content:
            if not isinstance(item, dict):
                text_chunks.append(str(item))
                continue
            item_type = item.get("type")
            if item_type == "text":
                text_chunks.append(str(item.get("text") or ""))
                continue
            image_url = item.get("image_url")
            if item_type == "image_url" and isinstance(image_url, dict) and image_url.get("url"):
                parts.append({"type": "image_url", "image_url": {"url": str(image_url["url"])}})
        text = "".join(text_chunks).strip()
        if text:
            parts.insert(0, {"type": "text", "text": text})
        return {"role": role, "content": parts or text}
    if images:
        parts = []
        if content:
            parts.append({"type": "text", "text": content})
        for image in images:
            if isinstance(image, bytes):
                encoded = base64.b64encode(image).decode("ascii")
            else:
                encoded = str(image)
            if not encoded.startswith("data:"):
                encoded = f"data:image/png;base64,{encoded}"
            parts.append({"type": "image_url", "image_url": {"url": encoded}})
        return {"role": role, "content": parts}
    return {"role": role, "content": content}


def _normalize_openai_image_part(item: dict) -> dict | None:
    item_type = str(item.get("type") or "").strip()
    image_url = item.get("image_url")
    if item_type == "image_url":
        if isinstance(image_url, dict) and image_url.get("url"):
            return {"type": "image_url", "image_url": {"url": str(image_url["url"])}}
        if isinstance(image_url, str) and image_url.strip():
            return {"type": "image_url", "image_url": {"url": image_url.strip()}}
    if item_type == "input_image":
        if isinstance(image_url, dict) and image_url.get("url"):
            return {"type": "image_url", "image_url": {"url": str(image_url["url"])}}
        if isinstance(image_url, str) and image_url.strip():
            return {"type": "image_url", "image_url": {"url": image_url.strip()}}
        if item.get("data"):
            mime_type = str(item.get("mime_type") or item.get("media_type") or "image/png").strip()
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{str(item['data']).strip()}"},
            }
    return None


def _normalize_openai_message(message: dict) -> dict:
    role = str(message.get("role") or "user").strip() or "user"
    if role == "tool":
        role = "user"
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                if item:
                    parts.append({"type": "text", "text": item})
                continue
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip()
            if item_type in {"text", "input_text"}:
                text = str(item.get("text") or item.get("input_text") or "").strip()
                if text:
                    parts.append({"type": "text", "text": text})
                continue
            image_part = _normalize_openai_image_part(item)
            if image_part is not None:
                parts.append(image_part)
        if not parts:
            return {"role": role, "content": ""}
        if len(parts) == 1 and parts[0]["type"] == "text":
            return {"role": role, "content": parts[0]["text"]}
        return {"role": role, "content": parts}
    if isinstance(content, str):
        return {"role": role, "content": content}
    return {"role": role, "content": str(content or "")}


def _messages_include_images(messages: list[dict]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"image_url", "input_image"}:
                    return True
    return False


def _stream_ollama_json_lines(handler: BaseHTTPRequestHandler, lines: list[dict]) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "application/x-ndjson")
    handler.end_headers()
    for item in lines:
        handler.wfile.write((json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8"))
        handler.wfile.flush()


def _ollama_done_payload(model: str) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "model": model,
        "created_at": now,
        "done": True,
        "done_reason": "stop",
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": 0,
        "eval_duration": 0,
    }


def _collect_openai_sse_response(response: requests.Response) -> dict:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    for line in response.iter_lines(chunk_size=1, decode_unicode=False):
        if not line:
            continue
        decoded = line.decode("utf-8", errors="ignore").strip()
        if not decoded.startswith("data: "):
            continue
        chunk_payload = decoded[6:].strip()
        if chunk_payload == "[DONE]":
            break
        try:
            chunk = json.loads(chunk_payload)
        except Exception:
            continue
        delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
        if delta.get("content"):
            content_parts.append(str(delta["content"]))
        if delta.get("reasoning_content"):
            reasoning_parts.append(str(delta["reasoning_content"]))
        message = (chunk.get("choices") or [{}])[0].get("message") or {}
        if message.get("content"):
            content_parts.append(str(message["content"]))
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    if not content and reasoning:
        content = reasoning
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }
        ]
    }


def resolve_catalog_model_name(raw_name: str, catalog: list[ManagedModel]) -> str:
    name = (raw_name or "").strip()
    if not name:
        return name
    if "/" in name:
        provider_prefix, remainder = name.split("/", 1)
        if provider_prefix and remainder:
            exact_prefixed = next((item.model_id for item in catalog if item.model_id == remainder), None)
            if exact_prefixed:
                return exact_prefixed
    exact = next((item.model_id for item in catalog if item.model_id == name), None)
    if exact:
        return exact
    alias_match = next((item.model_id for item in catalog if name in item.aliases), None)
    if alias_match:
        return alias_match
    for item in catalog:
        if name in model_name_aliases(item):
            return item.model_id
    try:
        repo_id, quant = parse_hf_input(name)
    except Exception:
        return name
    for item in catalog:
        if item.repo_id == repo_id and ((item.quant or "") == (quant or "")):
            return item.model_id
    return name


def _proxy_request_to_public_api(method: str, path: str, *, body: bytes | None = None, headers: dict[str, str] | None = None, host: str = DEFAULT_PUBLIC_HOST, port: int = DEFAULT_PUBLIC_PORT):
    url = f"http://{host}:{port}{path}"
    proxy_headers = {}
    if headers:
        for key, value in headers.items():
            lowered = key.lower()
            if lowered in {"host", "content-length", "connection"}:
                continue
            proxy_headers[key] = value
    return requests.request(method, url, data=body, headers=proxy_headers, timeout=(10, 600), stream=False)


def _touch_model_via_llamaswap(model_id: str, host: str, port: int, *, timeout: float = 30.0) -> bool:
    upstream_health = f"http://{host}:{port}/upstream/{quote(model_id, safe='')}/health"
    try:
        response = requests.get(upstream_health, timeout=(3, timeout))
        log_api_event(
            "model_keepalive_touch",
            {"model": model_id, "url": upstream_health, "status": response.status_code},
        )
        return response.status_code == 200
    except requests.RequestException as exc:
        log_api_event("model_keepalive_touch_error", {"model": model_id, "url": upstream_health, "error": str(exc)})
        return False


def start_unexpected_unload_guard(args):
    host = args.public_host
    port = int(args.public_port)
    catalog_path = Path(args.catalog)
    idle_ttl = max(1, int(resolve_idle_ttl(args)))
    poll_interval = min(30.0, max(1.0, idle_ttl / 20.0))
    state = {"loaded": set()}

    def loop():
        while True:
            try:
                catalog = load_catalog(catalog_path)
                loaded = get_loaded_catalog_model_ids(catalog)
                previous = set(state["loaded"])
                state["loaded"] = loaded
                disappeared = previous - loaded
                if disappeared:
                    activity, last_activity_model_id = get_model_activity_snapshot()
                    now = time.monotonic()
                    for model_id in sorted(disappeared):
                        should_reload, age = should_reload_after_unexpected_unload(
                            model_id,
                            activity,
                            last_activity_model_id,
                            now=now,
                            idle_ttl=idle_ttl,
                        )
                        if not should_reload:
                            log_api_event(
                                "model_unload_observed",
                                {"model": model_id, "activity_age_seconds": age, "idle_ttl": idle_ttl},
                            )
                            continue
                        log_api_event(
                            "model_unexpected_unload",
                            {"model": model_id, "activity_age_seconds": age, "idle_ttl": idle_ttl},
                        )
                        if _touch_model_via_llamaswap(model_id, host, port):
                            state["loaded"] = get_loaded_catalog_model_ids(catalog)
            except Exception as exc:
                log_api_event("model_unload_guard_error", {"error": str(exc)})
            time.sleep(poll_interval)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


def _tail_text_file(path: Path, lines: int = 50) -> str:
    if not path.exists():
        return f"No request log found at {path}"
    try:
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return f"Could not read {path}: {exc}"
    return "\n".join(data[-lines:]) if data else f"No request log entries in {path}"

def start_ctx_metadata_server(args):
    host = args.public_host
    port = resolve_api_port(args)
    catalog_path = Path(args.catalog)

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict, status: int = 200):
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError, OSError):
                # Client closed connection before or during write. Don't
                # propagate — the server should continue serving other
                # requests. Swallow the exception silently.
                try:
                    # Best-effort close socket to free resources.
                    self.connection.shutdown(2)
                except Exception:
                    pass

        def log_message(self, format, *args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            catalog = load_catalog(catalog_path)
            if parsed.path == "/v1/models":
                self._send_json({"object": "list", "data": [build_openai_model_payload(model) for model in catalog]})
                return
            if parsed.path == "/api/tags":
                published_models = get_published_model_ids(host, int(args.public_port))
                processes = get_llama_server_processes()
                process_by_model = {proc["model_path"]: proc for proc in processes}
                gpu_process_map = get_gpu_process_map()
                models = []
                for model in catalog:
                    process = process_by_model.get(_safe_realpath(model.local_path))
                    loaded = process is not None and model.model_id in published_models
                    models.append(build_ollama_model_payload(model, loaded=loaded, process=process, gpu_process_map=gpu_process_map))
                self._send_json({"models": models})
                return
            if parsed.path == "/api/ps":
                published_models = get_published_model_ids(host, int(args.public_port))
                processes = get_llama_server_processes()
                process_by_model = {proc["model_path"]: proc for proc in processes}
                gpu_process_map = get_gpu_process_map()
                running = []
                for model in catalog:
                    process = process_by_model.get(_safe_realpath(model.local_path))
                    if process is None or model.model_id not in published_models:
                        continue
                    running.append(build_ollama_model_payload(model, loaded=True, process=process, gpu_process_map=gpu_process_map))
                self._send_json({"models": running})
                return
            if parsed.path == "/api/version":
                self._send_json({"version": f"{get_superserver_version()}-llamacpp-superserver"})
                return
            if parsed.path == "/api/ctx":
                self._send_json({
                    "models": [
                        {
                            "name": model.model_id,
                            "configured_ctx": displayed_configured_ctx(model) if ctx_evaluation_status(model) != "ERROR" else None,
                            "api_ctx": displayed_api_ctx(model) if ctx_evaluation_status(model) != "ERROR" else None,
                            "api_ctx_status": ctx_evaluation_status(model),
                            "max_ctx": displayed_configured_ctx(model) if ctx_evaluation_status(model) != "ERROR" else None,
                            "gguf_ctx": get_model_context_size(model),
                            **_ctx_probe_api_metrics(model),
                        }
                        for model in catalog
                    ]
                })
                return
            if parsed.path == "/api/show":
                name = parse_qs(parsed.query).get("name", [""])[0]
                return self._handle_show(name, catalog)
            if parsed.path.startswith("/v1/"):
                return self._proxy_request("GET")
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/chat":
                return self._handle_ollama_chat()
            if parsed.path == "/api/generate":
                return self._handle_ollama_generate()
            if parsed.path in {"/api/embed", "/api/embeddings"}:
                return self._handle_ollama_embeddings()
            if parsed.path == "/v1/chat/completions":
                return self._handle_openai_chat_completions()
            if parsed.path.startswith("/v1/"):
                return self._proxy_request("POST")
            if parsed.path != "/api/show":
                self._send_json({"error": "not found"}, status=404)
                return
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                self._send_json({"error": "invalid json"}, status=400)
                return
            name = str(payload.get("name") or "").strip()
            self._handle_show(name, load_catalog(catalog_path))

        def _handle_show(self, name: str, catalog: list[ManagedModel]):
            resolved_name = resolve_catalog_model_name(name, catalog)
            model = next((item for item in catalog if item.model_id == resolved_name), None)
            if model is None:
                self._send_json({"error": f"model '{name}' not found"}, status=404)
                return
            previous_load_capabilities = list(model.load_capabilities)
            gguf_ctx = get_model_context_size(model)
            cfg_ctx = displayed_configured_ctx(model)
            api_ctx = displayed_api_ctx(model)
            ctx_status = ctx_evaluation_status(model)
            probe_metrics = _ctx_probe_api_metrics(model)
            load_capabilities = refresh_model_load_capabilities(model)
            if load_capabilities != previous_load_capabilities:
                try:
                    save_catalog(catalog_path, catalog)
                except Exception:
                    pass
            self._send_json({
                "license": "",
                "modelfile": f"FROM {model.model_id}\nPARAMETER num_ctx {model.ctx_size}\n",
                "parameters": f"num_ctx {model.ctx_size}",
                "template": "",
                "details": build_model_ctx_payload(model)["details"],
                "model_info": {
                    "llamacpp.configured_context_length": cfg_ctx if ctx_status != "ERROR" else None,
                    "llamacpp.api_context_length": api_ctx if ctx_status != "ERROR" else None,
                    "llamacpp.api_context_status": ctx_status,
                    "llamacpp.context_length": cfg_ctx if ctx_status != "ERROR" else None,
                    "llamacpp.gguf_context_length": gguf_ctx,
                    "llamacpp.load_capabilities": load_capabilities,
                    "llamacpp.ctx_probe_read_s": probe_metrics["ctx_probe_read_s"],
                    "llamacpp.ctx_probe_tokens_s": probe_metrics["ctx_probe_tokens_s"],
                    "llamacpp.ctx_probe_totals_s": probe_metrics["ctx_probe_totals_s"],
                    "llamacpp.ctx_probe_latency_ms": probe_metrics["ctx_probe_latency_ms"],
                    "llamacpp.ctx_probe_speed_tps": probe_metrics["ctx_probe_speed_tps"],
                    "llamacpp.ctx_probe_kv_gb": probe_metrics["ctx_probe_kv_gb"],
                    "llamacpp.ctx_probe_read": probe_metrics["ctx_probe_read"],
                    "llamacpp.ctx_probe_tokens": probe_metrics["ctx_probe_tokens"],
                    "llamacpp.ctx_probe_totals": probe_metrics["ctx_probe_totals"],
                    "llamacpp.ctx_probe_latency": probe_metrics["ctx_probe_latency"],
                    "llamacpp.ctx_probe_speed": probe_metrics["ctx_probe_speed"],
                    "llamacpp.ctx_probe_kv": probe_metrics["ctx_probe_kv"],
                },
                "modalities": {
                    "input": ["text", "image"] if _has_vision_runtime(model) else ["text"],
                    "output": ["text"],
                },
                "capabilities": ["completion", "vision"] if _has_vision_runtime(model) else ["completion"],
            })

        def _proxy_request(self, method: str):
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length > 0 else None
            activity_model = ""
            if body:
                try:
                    proxy_payload = json.loads(body.decode("utf-8"))
                    if isinstance(proxy_payload, dict):
                        activity_model = resolve_catalog_model_name(str(proxy_payload.get("model") or ""), load_catalog(catalog_path))
                        if activity_model:
                            mark_model_activity(activity_model, f"proxy:{parsed.path}", "request_start")
                except Exception:
                    pass
            log_api_event("proxy_request", {"method": method, "path": parsed.path, "query": parsed.query, "body": body.decode("utf-8", errors="replace")[:4000] if body else ""})
            try:
                response = _proxy_request_to_public_api(
                    method,
                    parsed.path + (("?" + parsed.query) if parsed.query else ""),
                    body=body,
                    headers={key: value for key, value in self.headers.items()},
                    host=host,
                    port=int(args.public_port),
                )
            except requests.RequestException as exc:
                log_api_event("proxy_error", {"method": method, "path": parsed.path, "error": str(exc)})
                self._send_json({"error": f"upstream unavailable: {exc}"}, status=502)
                return

            content = response.content
            log_api_event("proxy_response", {"method": method, "path": parsed.path, "status": response.status_code, "body": content.decode('utf-8', errors='replace')[:4000]})
            if activity_model:
                mark_model_activity(activity_model, f"proxy:{parsed.path}", "response_done")
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                lowered = key.lower()
                if lowered in {"content-length", "connection", "transfer-encoding", "content-encoding"}:
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _read_json_body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                payload = {}
            return payload if isinstance(payload, dict) else {}

        def _reject_if_gpu_busy(self, model_name: str, catalog: list[ManagedModel], *, api_style: str) -> bool:
            gpu_conflict = get_gpu_conflict_message(model_name, catalog, host, int(args.public_port))
            if not gpu_conflict:
                return False
            log_api_event("model_load_blocked_gpu_busy", {"model": model_name, "message": gpu_conflict, "api_style": api_style})
            if api_style == "openai":
                self._send_json({"error": {"message": gpu_conflict, "type": "server_error"}}, status=503)
            else:
                self._send_json({"error": gpu_conflict}, status=503)
            return True

        def _handle_ollama_chat(self):
            payload = self._read_json_body()
            log_api_event("ollama_chat_request", payload)
            started_at = time.monotonic()
            catalog = load_catalog(catalog_path)
            model_name = resolve_catalog_model_name(str(payload.get("model") or "").strip(), catalog)
            if not model_name:
                self._send_json({"error": "model is required"}, status=400)
                return
            if self._reject_if_gpu_busy(model_name, catalog, api_style="ollama"):
                return
            mark_model_activity(model_name, "ollama_chat", "request_start")
            model_entry = next((item for item in catalog if item.model_id == model_name), None)
            raw_messages = payload.get("messages") or []
            messages = [_ollama_message_to_openai(item) for item in raw_messages if isinstance(item, dict)]
            if not messages:
                prompt = str(payload.get("prompt") or "").strip()
                system = str(payload.get("system") or "").strip()
                if system:
                    messages.append({"role": "system", "content": system})
                if prompt:
                    messages.append({"role": "user", "content": prompt})
            if not messages:
                self._send_json({"error": "messages or prompt is required"}, status=400)
                return
            if _messages_include_images(messages) and (model_entry is None or not _has_vision_runtime(model_entry)):
                self._send_json(
                    {
                        "error": (
                            f"model '{model_name}' is installed without multimodal projector support (mmproj). "
                            "Re-add or update the model so the matching mmproj GGUF is downloaded and configured."
                        )
                    },
                    status=400,
                )
                return
            stream = bool(payload.get("stream"))
            upstream_payload = {
                "model": model_name,
                "messages": messages,
                "stream": stream,
            }
            options = payload.get("options") or {}
            if "temperature" in options:
                upstream_payload["temperature"] = options["temperature"]
            if "num_ctx" in options:
                upstream_payload["max_tokens"] = int(options["num_ctx"])
            keep_alive = payload.get("keep_alive")
            if keep_alive is not None:
                upstream_payload["keep_alive"] = keep_alive
            if stream:
                try:
                    response = requests.post(
                        f"http://{host}:{int(args.public_port)}/v1/chat/completions",
                        data=json.dumps(upstream_payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        timeout=(10, 600),
                        stream=True,
                    )
                    log_api_event("ollama_chat_upstream_headers", {"model": model_name, "status": response.status_code, "wait_ms": _elapsed_ms(started_at), "stream": True})
                except requests.RequestException as exc:
                    log_api_event("ollama_chat_upstream_network_error", {"error": str(exc)})
                    self._send_json({"error": f"upstream unavailable: {exc}"}, status=502)
                    return
                if response.status_code >= 400:
                    body_text = response.text[:4000]
                    log_api_event("ollama_chat_upstream_error", {"status": response.status_code, "body": body_text, "payload": upstream_payload})
                    self._send_json({"error": f"upstream unavailable: HTTP {response.status_code}: {body_text[:1000]}"}, status=502)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                first_chunk_logged = False
                for line in response.iter_lines(chunk_size=1, decode_unicode=False):
                    if not line:
                        continue
                    decoded = line.decode("utf-8", errors="ignore").strip()
                    if not decoded.startswith("data: "):
                        continue
                    chunk_payload = decoded[6:].strip()
                    if chunk_payload == "[DONE]":
                        done_payload = _ollama_done_payload(model_name)
                        self.wfile.write((json.dumps(done_payload, ensure_ascii=False) + "\n").encode("utf-8"))
                        self.wfile.flush()
                        mark_model_activity(model_name, "ollama_chat", "stream_done")
                        log_api_event("ollama_chat_stream_done", done_payload)
                        log_api_event("ollama_chat_total", {"model": model_name, "total_ms": _elapsed_ms(started_at), "stream": True})
                        return
                    try:
                        chunk = json.loads(chunk_payload)
                    except Exception:
                        continue
                    text = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if not text:
                        continue
                    if not first_chunk_logged:
                        first_chunk_logged = True
                        log_api_event("ollama_chat_first_chunk", {"model": model_name, "first_chunk_ms": _elapsed_ms(started_at)})
                    ollama_chunk = {
                        "model": model_name,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "message": {"role": "assistant", "content": text},
                        "done": False,
                    }
                    self.wfile.write((json.dumps(ollama_chunk, ensure_ascii=False) + "\n").encode("utf-8"))
                    self.wfile.flush()
                done_payload = _ollama_done_payload(model_name)
                self.wfile.write((json.dumps(done_payload, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
                mark_model_activity(model_name, "ollama_chat", "stream_done")
                log_api_event("ollama_chat_stream_done", done_payload)
                log_api_event("ollama_chat_total", {"model": model_name, "total_ms": _elapsed_ms(started_at), "stream": True})
                return

            try:
                response = requests.post(
                    f"http://{host}:{int(args.public_port)}/v1/chat/completions",
                    data=json.dumps(upstream_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    timeout=(10, 600),
                    stream=False,
                )
                log_api_event("ollama_chat_upstream_headers", {"model": model_name, "status": response.status_code, "wait_ms": _elapsed_ms(started_at), "stream": False})
            except requests.RequestException as exc:
                log_api_event("ollama_chat_upstream_network_error", {"error": str(exc)})
                self._send_json({"error": f"upstream unavailable: {exc}"}, status=502)
                return
            if response.status_code >= 400:
                body_text = response.text[:4000]
                log_api_event("ollama_chat_upstream_error", {"status": response.status_code, "body": body_text, "payload": upstream_payload})
                self._send_json({"error": f"upstream unavailable: HTTP {response.status_code}: {body_text[:1000]}"}, status=502)
                return
            try:
                data = response.json()
            except Exception as exc:
                log_api_event("ollama_chat_upstream_invalid_json", {"status": response.status_code, "error": str(exc), "body": response.text[:4000]})
                self._send_json({"error": f"upstream invalid response: {exc}"}, status=502)
                return
            log_api_event("ollama_chat_total", {"model": model_name, "total_ms": _elapsed_ms(started_at), "stream": False})
            message = data.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")
            usage = data.get("usage") or {}
            timings = data.get("timings") or {}
            base = {
                "model": model_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "message": {"role": "assistant", "content": content},
                "done": True,
                "done_reason": "stop",
                "total_duration": int(float(timings.get("prompt_ms", 0) + timings.get("predicted_ms", 0)) * 1_000_000),
                "load_duration": 0,
                "prompt_eval_count": int(usage.get("prompt_tokens") or 0),
                "prompt_eval_duration": int(float(timings.get("prompt_ms", 0)) * 1_000_000),
                "eval_count": int(usage.get("completion_tokens") or 0),
                "eval_duration": int(float(timings.get("predicted_ms", 0)) * 1_000_000),
            }
            if stream:
                _stream_ollama_json_lines(self, [base])
                mark_model_activity(model_name, "ollama_chat", "response_done")
                return
            log_api_event("ollama_chat_response", base)
            mark_model_activity(model_name, "ollama_chat", "response_done")
            self._send_json(base)

        def _handle_openai_chat_completions(self):
            payload = self._read_json_body()
            log_api_event("openai_chat_request", payload)
            started_at = time.monotonic()
            catalog = load_catalog(catalog_path)
            model_name = resolve_catalog_model_name(str(payload.get("model") or "").strip(), catalog)
            if not model_name:
                self._send_json({"error": {"message": "model is required", "type": "invalid_request_error"}}, status=400)
                return
            if self._reject_if_gpu_busy(model_name, catalog, api_style="openai"):
                return
            mark_model_activity(model_name, "openai_chat", "request_start")
            model_entry = next((item for item in catalog if item.model_id == model_name), None)
            raw_messages = payload.get("messages") or []
            if not isinstance(raw_messages, list) or not raw_messages:
                self._send_json({"error": {"message": "messages is required", "type": "invalid_request_error"}}, status=400)
                return
            messages = [_normalize_openai_message(item) for item in raw_messages if isinstance(item, dict)]
            if not messages:
                self._send_json({"error": {"message": "messages is required", "type": "invalid_request_error"}}, status=400)
                return
            if _messages_include_images(messages) and (model_entry is None or not _has_vision_runtime(model_entry)):
                self._send_json(
                    {
                        "error": {
                            "message": (
                                f"model '{model_name}' is installed without multimodal projector support (mmproj). "
                                "Re-add or update the model so the matching mmproj GGUF is downloaded and configured."
                            ),
                            "type": "invalid_request_error",
                        }
                    },
                    status=400,
                )
                return
            upstream_payload = dict(payload)
            upstream_payload["model"] = model_name
            upstream_payload["messages"] = messages
            stream = bool(payload.get("stream"))
            try:
                response = requests.post(
                    f"http://{host}:{int(args.public_port)}/v1/chat/completions",
                    data=json.dumps(upstream_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    timeout=(10, 600),
                    stream=stream,
                )
                log_api_event("openai_chat_upstream_headers", {"model": model_name, "status": response.status_code, "wait_ms": _elapsed_ms(started_at), "stream": stream})
            except requests.RequestException as exc:
                log_api_event("openai_chat_upstream_network_error", {"error": str(exc), "payload": upstream_payload})
                self._send_json({"error": {"message": f"upstream unavailable: {exc}", "type": "server_error"}}, status=502)
                return
            if response.status_code >= 400:
                body_text = response.text[:4000]
                log_api_event("openai_chat_upstream_error", {"status": response.status_code, "body": body_text, "payload": upstream_payload})
                self._send_json(
                    {"error": {"message": f"upstream unavailable: HTTP {response.status_code}: {body_text[:1000]}", "type": "server_error"}},
                    status=502,
                )
                return
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", response.headers.get("Content-Type", "text/event-stream"))
                self.end_headers()
                first_chunk_logged = False
                try:
                    for chunk in response.iter_content(chunk_size=4096):
                        if not chunk:
                            continue
                        if not first_chunk_logged:
                            first_chunk_logged = True
                            log_api_event("openai_chat_first_chunk", {"model": model_name, "first_chunk_ms": _elapsed_ms(started_at)})
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        mark_model_activity(model_name, "openai_chat", "stream_chunk", log=False)
                except (BrokenPipeError, ConnectionResetError, requests.RequestException) as exc:
                    log_api_event("openai_chat_stream_interrupted", {"model": model_name, "error": str(exc)})
                finally:
                    response.close()
                    mark_model_activity(model_name, "openai_chat", "stream_closed")
                    log_api_event("openai_chat_total", {"model": model_name, "total_ms": _elapsed_ms(started_at), "stream": True})
                return
            try:
                data = response.json()
            except Exception as exc:
                log_api_event("openai_chat_upstream_invalid_json", {"error": str(exc), "payload": upstream_payload, "body": response.text[:4000]})
                self._send_json({"error": {"message": f"upstream invalid response: {exc}", "type": "server_error"}}, status=502)
                return
            log_api_event("openai_chat_total", {"model": model_name, "total_ms": _elapsed_ms(started_at), "stream": False})
            choice = (data.get("choices") or [{}])[0]
            final_payload = {
                "id": data.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
                "object": data.get("object") or "chat.completion",
                "created": int(data.get("created") or time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": choice.get("message") or {"role": "assistant", "content": ""},
                        "finish_reason": choice.get("finish_reason") or "stop",
                    }
                ],
            }
            if data.get("usage") is not None:
                final_payload["usage"] = data.get("usage")
            if data.get("system_fingerprint") is not None:
                final_payload["system_fingerprint"] = data.get("system_fingerprint")
            log_api_event("openai_chat_response", final_payload)
            mark_model_activity(model_name, "openai_chat", "response_done")
            self._send_json(final_payload)

        def _handle_ollama_generate(self):
            payload = self._read_json_body()
            log_api_event("ollama_generate_request", payload)
            catalog = load_catalog(catalog_path)
            model_name = resolve_catalog_model_name(str(payload.get("model") or "").strip(), catalog)
            if not model_name:
                self._send_json({"error": "model is required"}, status=400)
                return
            if self._reject_if_gpu_busy(model_name, catalog, api_style="ollama"):
                return
            mark_model_activity(model_name, "ollama_generate", "request_start")
            model_entry = next((item for item in catalog if item.model_id == model_name), None)
            prompt = str(payload.get("prompt") or "")
            images = payload.get("images") or []
            options = payload.get("options") or {}
            stream = bool(payload.get("stream"))
            system = str(payload.get("system") or "").strip()
            upstream_messages = []
            if system:
                upstream_messages.append({"role": "system", "content": system})
            upstream_messages.append(
                _ollama_message_to_openai({"role": "user", "content": prompt, "images": images})
            )
            if images and (model_entry is None or not _has_vision_runtime(model_entry)):
                self._send_json(
                    {
                        "error": (
                            f"model '{model_name}' is installed without multimodal projector support (mmproj). "
                            "Re-add or update the model so the matching mmproj GGUF is downloaded and configured."
                        )
                    },
                    status=400,
                )
                return
            upstream_payload = {
                "model": model_name,
                "messages": upstream_messages,
                "stream": False,
            }
            endpoint = "/v1/chat/completions"
            if "temperature" in options:
                upstream_payload["temperature"] = options["temperature"]
            if "num_ctx" in options:
                upstream_payload["max_tokens"] = int(options["num_ctx"])
            try:
                response = requests.post(
                    f"http://{host}:{int(args.public_port)}{endpoint}",
                    data=json.dumps(upstream_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    timeout=(10, 600),
                    stream=True,
                )
            except requests.RequestException as exc:
                log_api_event("ollama_generate_upstream_network_error", {"error": str(exc)})
                self._send_json({"error": f"upstream unavailable: {exc}"}, status=502)
                return
            if response.status_code >= 400:
                body_text = response.text[:4000]
                log_api_event("ollama_generate_upstream_error", {"status": response.status_code, "body": body_text, "payload": upstream_payload})
                self._send_json({"error": f"upstream unavailable: HTTP {response.status_code}: {body_text[:1000]}"}, status=502)
                return
            try:
                data = _collect_openai_sse_response(response)
            except Exception as exc:
                log_api_event("ollama_generate_upstream_invalid_json", {"status": response.status_code, "error": str(exc)})
                self._send_json({"error": f"upstream invalid response: {exc}"}, status=502)
                return
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            base = {
                "model": model_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "response": text,
                "done": True,
                "done_reason": "stop",
                "context": [],
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "prompt_eval_duration": 0,
                "eval_count": 0,
                "eval_duration": 0,
            }
            if stream:
                _stream_ollama_json_lines(self, [base])
                mark_model_activity(model_name, "ollama_generate", "response_done")
                return
            log_api_event("ollama_generate_response", base)
            mark_model_activity(model_name, "ollama_generate", "response_done")
            self._send_json(base)

        def _handle_ollama_embeddings(self):
            payload = self._read_json_body()
            log_api_event("ollama_embeddings_request", payload)
            model = str(payload.get("model") or "").strip()
            catalog = load_catalog(catalog_path)
            model = resolve_catalog_model_name(model, catalog)
            text_input = payload.get("input")
            if text_input is None:
                text_input = payload.get("prompt")
            if not model or text_input is None:
                self._send_json({"error": "model and input are required"}, status=400)
                return
            if self._reject_if_gpu_busy(model, catalog, api_style="ollama"):
                return
            mark_model_activity(model, "ollama_embeddings", "request_start")
            upstream_payload = {"model": model, "input": text_input}
            try:
                response = _proxy_request_to_public_api(
                    "POST",
                    "/v1/embeddings",
                    body=json.dumps(upstream_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    host=host,
                    port=int(args.public_port),
                )
                data = response.json()
            except requests.RequestException as exc:
                log_api_event("ollama_embeddings_upstream_network_error", {"error": str(exc)})
                self._send_json({"error": f"upstream unavailable: {exc}"}, status=502)
                return
            except ValueError as exc:
                log_api_event("ollama_embeddings_upstream_invalid_json", {"status": response.status_code if 'response' in locals() else None, "error": str(exc)})
                self._send_json({"error": f"upstream invalid response: {exc}"}, status=502)
                return
            if response.status_code >= 400:
                log_api_event("ollama_embeddings_upstream_error", {"status": response.status_code, "body": response.text[:4000], "payload": upstream_payload})
                self._send_json({"error": f"upstream unavailable: HTTP {response.status_code}: {response.text[:1000]}"}, status=502)
                return
            embeddings = data.get("data", [])
            if len(embeddings) == 1:
                mark_model_activity(model, "ollama_embeddings", "response_done")
                self._send_json({"embedding": embeddings[0].get("embedding", [])})
                return
            mark_model_activity(model, "ollama_embeddings", "response_done")
            self._send_json({"embeddings": [item.get("embedding", []) for item in embeddings]})

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        print(f"[!] Could not start ctx metadata server on http://{host}:{port}: {e}")
        print(f"    Check which process owns the port with: sudo ss -ltnp 'sport = :{port}'")
        return None

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[*] Ctx metadata API listening on http://{host}:{port}")
    return server


def debug_mode(args):
    """Start the debug API in the foreground and keep it alive until Ctrl+C."""
    debug_args = argparse.Namespace(**vars(args))
    server = None
    # Prefer connecting to a running manager via socket; if unavailable,
    # start a local debug API on a free port.
    if getattr(debug_args, "api_port", None) is None:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(SOCKET_PATH)
                print("[*] Connected to manager; forwarding debug session to manager.")
                # Keep alive until interrupted (tests patch time.sleep to raise KeyboardInterrupt)
                while True:
                    time.sleep(1)
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
        except KeyboardInterrupt:
            # User interrupted via Ctrl+C (or test mock) while socket was connected;
            # this is the successful case: we connected and were interrupted.
            return 0
        except Exception:
            # Manager not available — pick a free port and serve locally
            debug_args.api_port = _find_free_port()
            try:
                server = start_ctx_metadata_server(debug_args)
                if server is None:
                    return 1
                print(f"[*] Debug mode active on http://{debug_args.public_host}:{resolve_api_port(debug_args)}")
                print("[*] Use the /api/debug/* endpoints while this command is running.")
                server.serve_forever(poll_interval=0.5)
            except KeyboardInterrupt:
                print("\n[*] Debug mode interrupted, shutting down.")
            finally:
                try:
                    from llamacpp_stack.debug_manager import DEBUG_SESSION_MANAGER

                    DEBUG_SESSION_MANAGER.stop_session()
                except Exception:
                    pass
                if server is not None:
                    try:
                        server.shutdown()
                    except Exception:
                        pass
                    try:
                        server.server_close()
                    except Exception:
                        pass
            return 0
    # If api_port was provided, fall through and start a local server (same as above)
    try:
        server = start_ctx_metadata_server(debug_args)
        if server is None:
            return 1
        print(f"[*] Debug mode active on http://{debug_args.public_host}:{resolve_api_port(debug_args)}")
        print("[*] Use the /api/debug/* endpoints while this command is running.")
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n[*] Debug mode interrupted, shutting down.")
    finally:
        try:
            from llamacpp_stack.debug_manager import DEBUG_SESSION_MANAGER

            DEBUG_SESSION_MANAGER.stop_session()
        except Exception:
            pass
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass
    return 0

def list_models(args):
    try:
        is_owner = (os.getuid() == 0 or os.getuid() == os.stat(args.catalog.parent).st_uid)
    except:
        is_owner = False

    if not is_owner:
        try:
            output = run_manager_command("list", args)
            if output:
                print("\n" + output)
            return 0
        except RuntimeError as e:
            message = str(e)
            if "FileNotFoundError" in message or "No such file or directory" in message:
                _show_local_catalog_fallback(args, e)
                return 0
            raise e
        except Exception as e:
            _show_local_catalog_fallback(args, e)
            return 0

    output = render_models_table(load_catalog(args.catalog, _args_server_config_path(args)), args.public_host, args.public_port, get_effective_idle_ttl(args))
    if output:
        print("\n" + output)
    else:
        print("No models registered yet.")
    return 0


def _rollback_failed_run_publication(args, model_id: str, *, reason: str = "") -> None:
    """Remove a just-published transient run model after load/warmup failure."""
    try:
        catalog = load_catalog(args.catalog, _args_server_config_path(args))
        if not any(item.model_id == model_id for item in catalog):
            return
        remaining = [item for item in catalog if item.model_id != model_id]
        save_catalog(args.catalog, remaining)
        _emit_message(
            f"{model_id}: removed from catalog/config after failed run{f' ({reason})' if reason else ''}.",
            None,
        )
        try:
            apply_config_and_wait_absent(
                remaining,
                args.config,
                args.llama_server,
                args.start_port,
                model_id,
                args.public_host,
                args.public_port,
                progress_callback=None,
                server_defaults=resolve_llama_server_defaults(args),
                timeout=20.0,
            )
        except Exception as exc:
            _emit_message(
                f"{model_id}: catalog entry removed, but publication cleanup could not be verified ({type(exc).__name__}: {exc}).",
                None,
            )
    except Exception as exc:
        _emit_message(
            f"{model_id}: could not rollback failed run publication ({type(exc).__name__}: {exc}).",
            None,
        )


def run_command(args):
    """Handle `run` subcommand with support for multiple `-hf` values.

    Semantics:
    - If multiple HF entries are provided and `--speculative` is set, the
            first HF is treated as the master and the second as draft. Both are
            ensured/downloaded first, then a speculative pair entry is created
            that serves the master with the draft model wired as `--model-draft`.
    - Otherwise falls back to default single-model behavior.
    """
    raw_hf = getattr(args, "hf", None)
    hf_list = []
    if raw_hf is None:
        hf_list = []
    elif isinstance(raw_hf, (list, tuple)):
        for group in raw_hf:
            if isinstance(group, (list, tuple)):
                hf_list.extend(group)
            else:
                hf_list.append(group)
    else:
        hf_list = [raw_hf]

    if len(hf_list) >= 2 and bool(getattr(args, "speculative", False)):
        # Ensure master model (first HF) without forcing an immediate publish.
        master_args = argparse.Namespace(**vars(args))
        master_args.hf = hf_list[0]
        master_args.repo = None
        master_args.model_id = None
        master_args.speculative = False
        master_args.defer_publish = True
        # Propagate user's auto/skip intent to the ensured master model so
        # an explicit --auto requested by the caller will trigger auto-tuning
        # instead of being silently skipped for speculative setup.
        master_args.auto_ctx = bool(getattr(args, "auto_ctx", False))
        master_args.skip_ctx = bool(getattr(args, "skip_ctx", False))
        master_mid = ensure_model_available(master_args)

        # Ensure draft model (second HF) without immediate publish.
        draft_args = argparse.Namespace(**vars(args))
        draft_args.hf = hf_list[1]
        draft_args.repo = None
        draft_args.model_id = None
        draft_args.speculative = False
        draft_args.defer_publish = True
        # The draft model is only used as a speculative companion, so keep
        # it on the skip-ctx path regardless of caller auto/skip defaults.
        draft_args.auto_ctx = bool(getattr(args, "auto_ctx", False))
        draft_args.skip_ctx = True
        draft_mid = ensure_model_available(draft_args)

        # Create/publish the speculative pair using master as target model and
        # draft as --model-draft.
        spec_args = argparse.Namespace(**vars(args))
        spec_args.hf = None
        spec_args.repo = None
        spec_args.file = None
        spec_args.model_id = master_mid
        spec_args.speculative = True
        spec_args.spec_base_model_id = master_mid
        spec_args.spec_draft_model_id = draft_mid
        spec_args.auto_ctx = bool(getattr(args, "auto_ctx", False))
        spec_args.skip_ctx = True
        spec_mid = ensure_model_available(spec_args)

        # Any remaining HF entries: ensure they're downloaded as regular models
        for extra in hf_list[2:]:
            extra_args = argparse.Namespace(**vars(args))
            extra_args.hf = extra
            extra_args.repo = None
            extra_args.model_id = None
            extra_args.speculative = False
            ensure_model_available(extra_args)

        if args.no_chat:
            return 0
        chat_status = start_chat(spec_mid, args.public_host, args.public_port)
        if chat_status:
            _rollback_failed_run_publication(args, spec_mid, reason="warmup failed")
        return chat_status

    # Default single-model path
    effective_args = argparse.Namespace(**vars(args))
    if hf_list:
        effective_args.hf = hf_list[0]
    mid = ensure_model_available(effective_args)
    if args.no_chat:
        return 0
    return start_chat(mid, args.public_host, args.public_port)





def show_request_log(args):
    print(_tail_text_file(DEFAULT_REQUESTS_LOG_PATH, lines=int(args.lines)))
    return 0


def _collect_model_references(args) -> list[str]:
    refs: list[str] = []
    for raw in (getattr(args, "repo", None), getattr(args, "hf", None)):
        if isinstance(raw, list):
            for item in raw:
                value = str(item).strip()
                if value:
                    refs.append(value)
            continue
        if raw is None:
            continue
        value = str(raw).strip()
        if value:
            refs.append(value)
    unique_refs: list[str] = []
    for ref in refs:
        if ref not in unique_refs:
            unique_refs.append(ref)
    return unique_refs


def _clone_namespace(args) -> argparse.Namespace:
    return argparse.Namespace(**vars(args))


def _normalize_single_ref_args(args) -> argparse.Namespace:
    cloned = _clone_namespace(args)
    for field in ("repo", "hf"):
        value = getattr(cloned, field, None)
        if isinstance(value, list):
            if len(value) > 1:
                raise RuntimeError(f"Multiple values provided for {field}. Use batch mode with add/remove/update wrappers.")
            setattr(cloned, field, value[0] if value else None)
    return cloned


def add_models(args):
    refs = _collect_model_references(args)
    if refs:
        if getattr(args, "model_id", None):
            raise RuntimeError("Use either model references list or --model-id, not both.")
        for ref in refs:
            cloned = _clone_namespace(args)
            cloned.repo = ref
            cloned.hf = None
            cloned.model_id = None
            ensure_model_available(cloned)
        restart_service_to_free_vram(getattr(args, "service", DEFAULT_SERVICE_NAME))
        return 0
    res = ensure_model_available(_normalize_single_ref_args(args))
    restart_service_to_free_vram(getattr(args, "service", DEFAULT_SERVICE_NAME))
    return res and 0


def remove_models(args):
    effective_args = _clone_namespace(args)
    if not hasattr(effective_args, "delete_files"):
        effective_args.delete_files = True

    refs = _collect_model_references(effective_args)
    if refs:
        if getattr(effective_args, "model_id", None) or getattr(effective_args, "file", None):
            raise RuntimeError("Use either a references list or --model-id/--file, not both.")
        for ref in refs:
            cloned = _clone_namespace(effective_args)
            cloned.repo = ref
            cloned.hf = None
            cloned.model_id = None
            cloned.file = None
            remove_model(cloned)
        restart_service_to_free_vram(getattr(effective_args, "service", DEFAULT_SERVICE_NAME))
        return 0
    res = remove_model(_normalize_single_ref_args(effective_args))
    restart_service_to_free_vram(getattr(effective_args, "service", DEFAULT_SERVICE_NAME))
    return res and 0


def unload_models(args):
    effective_args = _clone_namespace(args)
    refs = _collect_model_references(effective_args)
    unload_all = False
    if refs:
        normalized_refs = [str(ref).strip().lower() for ref in refs]
        if "all" in normalized_refs:
            if len(refs) > 1:
                raise RuntimeError("Use either 'all' or specific model references, not both.")
            unload_all = True
    else:
        unload_all = True

    catalog = load_catalog(effective_args.catalog, _args_server_config_path(effective_args))
    if unload_all:
        target_ids = [model.model_id for model in catalog]
        remaining_catalog = []
    else:
        target_ids = []
        for ref in refs:
            cloned = _clone_namespace(effective_args)
            cloned.repo = ref
            cloned.hf = None
            cloned.model_id = None
            cloned.file = None
            model = resolve_catalog_model(
                catalog,
                target=getattr(cloned, "repo", None),
                repo_ref=getattr(cloned, "hf", None),
                model_id=getattr(cloned, "model_id", None),
                filename=getattr(cloned, "file", None),
            )
            if model.model_id not in target_ids:
                target_ids.append(model.model_id)
        remaining_catalog = [model for model in catalog if model.model_id not in target_ids]

    render_llamaswap_config(
        remaining_catalog,
        effective_args.config,
        effective_args.llama_server,
        effective_args.start_port,
        resolve_idle_ttl(effective_args),
        server_defaults=resolve_llama_server_defaults(effective_args),
    )
    _emit_message("Unloaded all models." if unload_all else f"Unloaded model(s): {', '.join(target_ids)}.", None)
    if wait_for_models_absent(target_ids if not unload_all else [], effective_args.public_host, effective_args.public_port):
        return 0
    raise RuntimeError("Unload request was sent, but the target model(s) are still visible in /v1/models.")


def update_models(args):
    refs = _collect_model_references(args)
    if refs:
        if getattr(args, "model_id", None) or getattr(args, "file", None):
            raise RuntimeError("Use either a references list or --model-id/--file, not both.")
        for ref in refs:
            cloned = _clone_namespace(args)
            cloned.repo = ref
            cloned.hf = None
            cloned.model_id = None
            cloned.file = None
            update_config(cloned)
        restart_service_to_free_vram(getattr(args, "service", DEFAULT_SERVICE_NAME))
        return 0
    res = update_config(_normalize_single_ref_args(args))
    restart_service_to_free_vram(getattr(args, "service", DEFAULT_SERVICE_NAME))
    return res and 0

def temporarily_unload_published_models(args, progress_callback = None, timeout = 45):
    _emit_message(
        "To probe ctx reliably, published models will be unloaded temporarily so they do not occupy VRAM.",
        progress_callback,
    )
    render_llamaswap_config(
        [],
        args.config,
        args.llama_server,
        args.start_port,
        resolve_idle_ttl(args),
        server_defaults=resolve_llama_server_defaults(args),
    )
    _emit_message("Temporary empty config written. Waiting for llama-swap --watch-config...", progress_callback)
    time.sleep(3.0)
    host = _normalize_client_host(args.public_host)
    url = f"http://{host}:{args.public_port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200 and not r.json().get("data", []):
                _emit_message("Published models unloaded for probing.", progress_callback)
                return True
        except Exception:
            pass
        time.sleep(1.5)
    _emit_message("Could not verify that models were fully unloaded, continuing with probes anyway.", progress_callback)
    return False

def restart_service_to_free_vram(service_name: str, progress_callback = None, settle_time = 3.0):
    _emit_message(f"Restarting {service_name} to free VRAM...", progress_callback)
    try:
        result = subprocess.run(
            ["systemctl", "restart", service_name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        _emit_message(f"{service_name} restarted successfully.", progress_callback)
        time.sleep(settle_time)
        return True
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or "").strip()
        if detail:
            _emit_message(f"Could not restart {service_name}: {detail}", progress_callback)
        else:
            _emit_message(f"Could not restart {service_name}: exit status {e.returncode}", progress_callback)
    except Exception as e:
        _emit_message(f"Could not restart {service_name}: {e}", progress_callback)
    return False

def restore_catalog_config(args, catalog, progress_callback = None, settle_time = 3.0, restart_service = False):
    render_llamaswap_config(
        catalog,
        args.config,
        args.llama_server,
        args.start_port,
        resolve_idle_ttl(args),
        server_defaults=resolve_llama_server_defaults(args),
    )
    _emit_message("Previous llama-swap config restored after the failed operation.", progress_callback)
    if restart_service:
        restart_service_to_free_vram(args.service, progress_callback=progress_callback, settle_time=settle_time)
    else:
        time.sleep(settle_time)


def remove_templates(args) -> None:
    """Remove all template-backed model entries from the catalog.

    Deletes any catalog entry whose model_id ends with '+template'.
    Saves the updated catalog and restores the swap configuration.
    """
    catalog = load_catalog(args.catalog, _args_server_config_path(args))
    original_count = len(catalog)
    removed: list[str] = []
    
    # Filter out all entries ending with '+template'
    filtered = [m for m in catalog if not m.model_id.endswith("+template")]
    
    for m in catalog:
        if m.model_id.endswith("+template"):
            removed.append(m.model_id)
    
    if not removed:
        print("No template-backed models found in catalog.")
        return
    
    save_catalog(args.catalog, filtered)
    print(f"Removed {len(removed)} template-backed models: {', '.join(removed)}")
    
    try:
        restore_catalog_config(args, filtered, progress_callback=None, restart_service=True)
    except Exception as e:
        print(f"Catalog updated but failed to restore config: {e}")


def refresh_templates(args) -> None:
    """Scan templates directory and add template-backed catalog entries.

    For each model in the catalog, if a template file exists in the
    templates folder, create a duplicate catalog entry with model_id
    '<base>+template' and a server_overrides key `chat_template_file`
    pointing to the template. After changes, save and restore config.
    Speculative models are not affected by this operation.
    """
    templates_dir = None
    if os.environ.get("LLAMACPP_TEMPLATES_DIR"):
        templates_dir = Path(os.environ.get("LLAMACPP_TEMPLATES_DIR")).expanduser()
    else:
        server_conf = Path(args.server_config) if getattr(args, "server_config", None) else Path(DEFAULT_SERVER_CONFIG_PATH)
        templates_dir = server_conf.expanduser().parent / "templates"

    if not templates_dir.exists():
        print(f"Templates folder not found: {templates_dir}")
        return

    catalog = load_catalog(args.catalog, _args_server_config_path(args))
    added: list[str] = []
    updated: list[str] = []
    existing_ids = {m.model_id for m in catalog}
    for base in list(catalog):
        # Skip speculative models (do not create template variants of drafts)
        # and models that already have a +template suffix
        if getattr(base, "speculative", False) or base.model_id.endswith("+template"):
            continue
        try:
            found = _find_chat_template_for_model(base.model_id, templates_dir)
        except Exception:
            found = None
        if not found:
            continue
        new_id = f"{base.model_id}+template"
        try:
            base_local_path = Path(base.local_path) if base.local_path else None
            if base_local_path is None or not base_local_path.exists():
                print(f"Skipping {base.model_id}: local model file not found.")
                continue

            # Create a minimal catalog entry: same local_path, only difference
            # is a chat template override. Do NOT create on-disk variant here;
            # independent files are created later when rendering swap YAML.
            duplicate = next((m for m in catalog if m.model_id == new_id), None)
            if duplicate is None:
                duplicate = ManagedModel(**asdict(base))
                duplicate.model_id = new_id
                catalog.append(duplicate)
                added.append(new_id)
                existing_ids.add(new_id)

            duplicate.filename = base.filename
            duplicate.local_path = str(base_local_path)
            duplicate.aliases = []
            duplicate.server_overrides = dict(duplicate.server_overrides or {})
            # Store the template file name (relative or absolute) so build command
            # can emit --chat-template-file. Keep the catalog lightweight.
            duplicate.server_overrides["chat_template_file"] = str(found)
            updated.append(new_id)
        except Exception:
            continue

    if not added and not updated:
        print("No new template-backed models found.")
        return

    save_catalog(args.catalog, catalog)
    if added:
        print(f"Added {len(added)} template-backed models: {', '.join(added)}")
    if updated and not added:
        print(f"Updated {len(updated)} template-backed models: {', '.join(updated)}")
    try:
        restore_catalog_config(args, catalog, progress_callback=None, restart_service=True)
    except Exception as e:
        print(f"Catalog updated but failed to restore config: {e}")

def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        try:
            sock.close()
        except Exception:
            pass

def create_llamacpp_trace_file(model_id: str, ctx_size: int) -> tuple[Path, object]:
    safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_id).strip("-") or "model"
    handle = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f"llamacpp-{safe_model}-ctx{ctx_size}-",
        suffix=".log",
        delete=False,
    )
    return Path(handle.name), handle

def _with_trace(reason: str, trace_path: Path | None) -> str:
    if trace_path is None:
        return reason
    return f"{reason} (trace: {trace_path})"


def _parse_probe_trace_metrics(trace_path: Path | None) -> ProbeTraceMetrics:
    metrics = ProbeTraceMetrics()
    if trace_path is None or not trace_path.exists():
        return metrics
    try:
        text = trace_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return metrics

    for match in re.finditer(r"CUDA(\d+)\s+model buffer size =\s+([0-9.]+)\s+MiB", text):
        metrics.model_buffers_mib[int(match.group(1))] = float(match.group(2))
    for match in re.finditer(r"CUDA(\d+)\s+KV buffer size =\s+([0-9.]+)\s+MiB", text):
        metrics.kv_buffers_mib[int(match.group(1))] = float(match.group(2))
    for match in re.finditer(r"CUDA(\d+)\s+compute buffer size =\s+([0-9.]+)\s+MiB", text):
        metrics.compute_buffers_mib[int(match.group(1))] = float(match.group(2))

    projector = re.search(r"clip_ctx:\s+CLIP using CUDA(\d+) backend", text)
    if projector:
        metrics.projector_gpu = int(projector.group(1))

    oom = re.search(r"allocating\s+([0-9.]+)\s+MiB on device (\d+): cudaMalloc failed", text)
    if oom:
        metrics.oom_requested_mib = float(oom.group(1))
        metrics.oom_gpu = int(oom.group(2))
    return metrics


def _query_gpu_free_memory_mib() -> dict[int, float]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return {}
    free_mib: dict[int, float] = {}
    for idx, line in enumerate(result.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            free_mib[idx] = float(line)
        except ValueError:
            continue
    return free_mib

def build_ctx_probe_prompt(ctx_size: int, multimodal: bool = False) -> str:
    reserve_tokens = 4096 if multimodal else 2048
    target_tokens = max(1024, int(ctx_size * 0.92) - reserve_tokens)
    header = (
        "Read the entire context carefully.\n"
        "Reply with exactly OK.\n"
        "The filler below is intentionally repetitive and should still be treated as part of the prompt.\n\n"
    )
    filler_chunk = " hello"
    prompt_body = header + (filler_chunk * target_tokens)
    return (
        "Read the full context carefully and reply with exactly OK.\n\n"
        f"{prompt_body}"
    )

def _build_probe_image_data_url() -> str:
    width, height = 256, 256
    rows = []
    for y in range(height):
        row = b"\x00"
        for x in range(width):
            row += bytes(((x * 255) // max(1, width - 1), (y * 255) // max(1, height - 1), ((x + y) * 255) // max(1, width + height - 2)))
        rows.append(row)
    raw = b"".join(rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", binascii.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _probe_request_payload(model: ManagedModel, prompt: str):
    if _has_vision_runtime(model):
        image_url = _build_probe_image_data_url()
        multimodal_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
    else:
        multimodal_messages = [{"role": "user", "content": prompt}]
    return [
        (
            "/v1/chat/completions",
            {
                "messages": multimodal_messages,
                "stream": False,
                "max_tokens": 10,
                "temperature": 0,
            },
        )]
    return [
        (
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "max_tokens": 1,
                "temperature": 0,
            },
        ),
        (
            "/v1/completions",
            {
                "prompt": prompt,
                "stream": False,
                "max_tokens": 1,
                "temperature": 0,
            },
        ),
    ]

def _stop_process(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _probe_runtime_env() -> dict[str, str] | None:
    env = os.environ.copy()
    lib_paths: list[str] = []
    cuda_root = env.get("LLAMACPP_CUDA_ROOT", "").strip()
    nccl_root = env.get("LLAMACPP_NCCL_ROOT", "").strip()
    if cuda_root:
        env["CUDA_PATH"] = cuda_root
        lib_paths.extend([f"{cuda_root}/lib64", f"{cuda_root}/lib"])
    if nccl_root:
        lib_paths.extend([f"{nccl_root}/lib64", f"{nccl_root}/lib"])
    existing = env.get("LD_LIBRARY_PATH", "")
    if not lib_paths:
        return env
    if existing:
        lib_paths.append(existing)
    env["LD_LIBRARY_PATH"] = ":".join(lib_paths)
    return env


def _spawn_validation_server(model: ManagedModel, llama_server: Path, ctx_size: int):
    local_path = Path(model.local_path) if model.local_path else None
    if local_path is None or not local_path.exists():
        raise RuntimeError("Model file is missing.")

    port = _find_free_port()
    trace_path, trace_handle = create_llamacpp_trace_file(model.model_id, ctx_size)
    probe_model = ManagedModel(**asdict(model))
    probe_model.ctx_size = int(ctx_size)
    cmd = build_llama_server_command(
        probe_model,
        llama_server,
        port=str(port),
        host="127.0.0.1",
        server_defaults=resolve_llama_server_defaults(),
    )

    proc = subprocess.Popen(
        cmd,
        stdout=trace_handle,
        stderr=subprocess.STDOUT,
        env=_probe_runtime_env(),
    )
    return proc, port, trace_path, trace_handle


def _wait_for_probe_server_ready(proc, port: int, trace_path: Path, timeout: int = 600) -> None:
    health_url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Validation server exited {proc.returncode}. trace: {trace_path}")
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"Validation server health timeout. trace: {trace_path}")



def wait_for_models_absent(model_ids, host, port, timeout=35):
    host = _normalize_client_host(host)
    url = f"http://{host}:{port}/v1/models"
    target_ids = {str(model_id).strip() for model_id in model_ids if str(model_id).strip()}
    deadline = time.time() + timeout
    label = "all models" if not target_ids else ", ".join(sorted(target_ids))
    spinner = Spinner(f"\033[36mWaiting for unload: {label}...\033[0m ")
    spinner.start()
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                data = r.json().get("data", [])
                published_ids = {str(model.get("id") or "").strip() for model in data if isinstance(model, dict)}
                if not target_ids and not published_ids:
                    spinner.stop()
                    print("\033[32mAll models are unloaded.\033[0m")
                    return True
                if target_ids and target_ids.isdisjoint(published_ids):
                    spinner.stop()
                    print(f"\033[32mModel(s) unloaded: {label}.\033[0m")
                    return True
        except Exception:
            pass
        time.sleep(1.5)
    spinner.stop()
    return False

def _parse_last_trace_value(trace_path: Path, pattern: str) -> int | None:
    try:
        text = trace_path.read_text(errors="ignore")
    except Exception:
        return None
    matches = re.findall(pattern, text)
    return int(matches[-1]) if matches else None


def _validate_request_runtime_tokens(trace_path: Path) -> tuple[int | None, int | None]:
    requested = _parse_last_trace_value(trace_path, r"task\.n_tokens = (\d+)")
    processed = _parse_last_trace_value(trace_path, r"prompt processing done, n_tokens = (\d+)")
    return requested, processed


def _validation_prompt_for_reps(repetitions: int, nonce: str) -> str:
    line = (
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
        "mike november oscar papa quebec romeo sierra tango uniform victor "
        "whiskey xray yankee zulu.\n"
    )
    header = "Read the entire context carefully and reply with exactly OK.\n\n"
    return header + f"nonce={nonce}\n" + (line * repetitions)


def _run_validation_probe_request(port: int, model: ManagedModel, prompt: str) -> requests.Response:
    body = {
        "model": model.model_id,
        "stream": False,
        "max_tokens": 1,
        "temperature": 0,
    }
    if _has_vision_runtime(model):
        body["messages"] = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _build_probe_image_data_url()}},
            ],
        }]
    else:
        body["messages"] = [{"role": "user", "content": prompt}]
    return requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=1800)


def validate_model_ctx_runtime(model: ManagedModel, llama_server: Path, ctx_size: int, progress_callback = None) -> dict:
    proc = None
    trace_handle = None
    trace_path = None
    try:
        proc, port, trace_path, trace_handle = _spawn_validation_server(model, llama_server, ctx_size)
        _emit_message(f"{model.model_id}: starting isolated validation server at ctx {ctx_size}...", progress_callback)
        _wait_for_probe_server_ready(proc, port, trace_path)

        reserve = 2048 if _has_vision_runtime(model) else 1024
        target_total = max(1024, ctx_size - reserve)
        repetitions = 256
        final_result = None
        for attempt in range(1, 7):
            prompt = _validation_prompt_for_reps(repetitions, f"{model.model_id}-{attempt}-{time.time_ns()}")
            response = _run_validation_probe_request(port, model, prompt)
            time.sleep(2)
            requested, processed = _validate_request_runtime_tokens(trace_path)
            if requested is None:
                raise RuntimeError(f"{model.model_id}: could not parse runtime token count. trace: {trace_path}")
            final_result = {
                "attempt": attempt,
                "repetitions": repetitions,
                "status_code": response.status_code,
                "requested_tokens": requested,
                "processed_tokens": processed,
                "response_preview": response.text[:240],
                "target_total_tokens": target_total,
            }
            _emit_message(
                f"{model.model_id}: validation attempt {attempt} -> requested={requested} target={target_total} status={response.status_code}",
                progress_callback,
            )
            if response.status_code == 200 and requested >= int(target_total * 0.95):
                final_result["result"] = "ok"
                final_result["trace_path"] = str(trace_path)
                return final_result
            scale = 0.995 if response.status_code == 200 else 0.97
            repetitions = int(repetitions * (target_total / max(1, requested)) * scale)
            if repetitions <= 0:
                repetitions = 1
        final_result = final_result or {}
        final_result["result"] = "inconclusive"
        final_result["trace_path"] = str(trace_path)
        return final_result
    finally:
        _stop_process(proc)
        if trace_handle is not None:
            try:
                trace_handle.close()
            except Exception:
                pass

def probe_model_ctx(model: ManagedModel, llama_server: Path, ctx_size: int, timeout: int = 600):
    local_path = Path(model.local_path) if model.local_path else None
    if local_path is None or not local_path.exists():
        return False, "missing-file"

    port = _find_free_port()
    trace_path, trace_handle = create_llamacpp_trace_file(model.model_id, ctx_size)
    probe_model = ManagedModel(**asdict(model))
    probe_model.ctx_size = int(ctx_size)
    cmd = build_llama_server_command(
        probe_model,
        llama_server,
        port=str(port),
        host="127.0.0.1",
        server_defaults=resolve_llama_server_defaults(),
    )

    proc = None
    loader = LoadingBar(f"\033[35;1mProbe {model.model_id} ctx={ctx_size}:\033[0m ")
    probe_metrics: dict[str, object] = {}
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=trace_handle,
            stderr=subprocess.STDOUT,
            env=_probe_runtime_env(),
        )
        loader.start()
        health_url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                return False, _with_trace(f"exit-{proc.returncode}", trace_path)
            try:
                r = requests.get(health_url, timeout=2)
                if r.status_code == 200:
                    prompt = build_ctx_probe_prompt(ctx_size, multimodal=_has_vision_runtime(model))
                    request_timeout = max(120, min(1800, ctx_size // 256))
                    last_reason = "prompt-no-endpoint"
                    for endpoint, payload in _probe_request_payload(model, prompt):
                        body = dict(payload)
                        body["model"] = model.model_id
                        request_started = time.perf_counter()
                        try:
                            probe = requests.post(
                                f"http://127.0.0.1:{port}{endpoint}",
                                json=body,
                                timeout=request_timeout,
                            )
                        except requests.Timeout:
                            elapsed_ms = max(0.0, (time.perf_counter() - request_started) * 1000.0)
                            probe_metrics = {
                                "probe_ctx": int(ctx_size),
                                "probe_endpoint": endpoint,
                                "probe_latency_ms": elapsed_ms,
                            }
                            last_reason = _with_trace(f"{endpoint}-timeout", trace_path)
                            continue
                        except Exception as e:
                            elapsed_ms = max(0.0, (time.perf_counter() - request_started) * 1000.0)
                            probe_metrics = {
                                "probe_ctx": int(ctx_size),
                                "probe_endpoint": endpoint,
                                "probe_latency_ms": elapsed_ms,
                            }
                            last_reason = _with_trace(f"{endpoint}-{e.__class__.__name__}", trace_path)
                            continue
                        elapsed_ms = max(0.0, (time.perf_counter() - request_started) * 1000.0)
                        probe_metrics = {
                            "probe_ctx": int(ctx_size),
                            "probe_endpoint": endpoint,
                            "probe_latency_ms": elapsed_ms,
                        }
                        if probe.status_code != 200:
                            detail = ""
                            try:
                                detail = probe.text.strip()
                            except Exception:
                                detail = ""
                            detail = detail[:120] if detail else ""
                            base_reason = f"{endpoint}-http-{probe.status_code}" + (f": {detail}" if detail else "")
                            last_reason = _with_trace(base_reason, trace_path)
                            continue
                        try:
                            body = probe.json()
                            choices = body.get("choices") or []
                            if not choices:
                                last_reason = _with_trace(f"{endpoint}-empty", trace_path)
                                continue
                            usage = body.get("usage") if isinstance(body, dict) else {}
                            prompt_tokens = _to_int_or_none((usage or {}).get("prompt_tokens") if isinstance(usage, dict) else None)
                            completion_tokens = _to_int_or_none((usage or {}).get("completion_tokens") if isinstance(usage, dict) else None)
                            total_tokens = _to_int_or_none((usage or {}).get("total_tokens") if isinstance(usage, dict) else None)
                            read_s = None
                            tokens_s = None
                            totals_s = None
                            speed_tps = None
                            elapsed_s = elapsed_ms / 1000.0 if elapsed_ms > 0 else None
                            if elapsed_s is not None:
                                if prompt_tokens is not None:
                                    read_s = prompt_tokens / elapsed_s
                                if completion_tokens is not None:
                                    tokens_s = completion_tokens / elapsed_s
                                if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
                                    total_tokens = int((prompt_tokens or 0) + (completion_tokens or 0))
                                if total_tokens is not None:
                                    totals_s = total_tokens / elapsed_s
                            speed_tps = totals_s
                            probe_metrics = {
                                "probe_ctx": int(ctx_size),
                                "probe_endpoint": endpoint,
                                "probe_latency_ms": elapsed_ms,
                                "probe_prompt_tokens": prompt_tokens,
                                "probe_completion_tokens": completion_tokens,
                                "probe_total_tokens": total_tokens,
                                "probe_read_s": read_s,
                                "probe_tokens_s": tokens_s,
                                "probe_totals_s": totals_s,
                                "probe_speed_tps": speed_tps,
                            }
                        except Exception:
                            last_reason = _with_trace(f"{endpoint}-invalid-json", trace_path)
                            continue
                        try:
                            trace_handle.flush()
                        except Exception:
                            pass
                        return True, _with_trace("ok", trace_path), probe_metrics
                    return False, last_reason, probe_metrics
            except Exception:
                pass
            time.sleep(2)
        return False, _with_trace("timeout", trace_path), probe_metrics
    finally:
        loader.stop()
        _stop_process(proc)
        try:
            trace_handle.close()
        except Exception:
            pass


def _estimate_ctx_ceiling(model: ManagedModel, calibration_ctx: int, metrics: ProbeTraceMetrics, free_vram_mib: dict[int, float]) -> int | None:
    if calibration_ctx <= 0 or not metrics.kv_buffers_mib or not free_vram_mib:
        return None
    candidates: list[int] = []
    for gpu_idx, free_mib in free_vram_mib.items():
        kv_mib = metrics.kv_buffers_mib.get(gpu_idx)
        if not kv_mib or kv_mib <= 0:
            continue
        fixed_mib = metrics.model_buffers_mib.get(gpu_idx, 0.0) + metrics.compute_buffers_mib.get(gpu_idx, 0.0)
        reserve_mib = max(1024.0, free_mib * 0.10)
        if _has_vision_runtime(model) and metrics.projector_gpu == gpu_idx:
            reserve_mib += 1536.0
        usable_mib = free_mib - fixed_mib - reserve_mib
        if usable_mib <= 0:
            continue
        kv_per_token_mib = kv_mib / calibration_ctx
        if kv_per_token_mib <= 0:
            continue
        candidates.append(int(usable_mib / kv_per_token_mib))
    if not candidates:
        return None
    return _align_ctx(max(1024, min(candidates)))

def _align_ctx(value: int) -> int:
    step = 1024
    return max(step, (value // step) * step)


def _normalize_probe_result(result: object) -> tuple[bool, str, dict[str, object]]:
    if isinstance(result, tuple):
        if len(result) >= 3:
            ok = bool(result[0])
            reason = str(result[1])
            payload = result[2] if isinstance(result[2], dict) else {}
            return ok, reason, payload
        if len(result) == 2:
            ok = bool(result[0])
            reason = str(result[1])
            return ok, reason, {}
    return False, "invalid-probe-result", {}


def _trace_path_from_reason(reason: str | None) -> Path | None:
    match = re.search(r"trace: ([^)]+)", reason or "")
    if not match:
        return None
    return Path(match.group(1))


def _trace_tail_from_reason(reason: str | None, *, lines: int = 25) -> str:
    trace_path = _trace_path_from_reason(reason)
    if trace_path is None:
        return ""
    if not trace_path.exists():
        return f"Trace file not found: {trace_path}"
    return _tail_text_file(trace_path, lines=lines)


def _ctx_gb_from_metrics(metrics: ProbeTraceMetrics) -> float | None:
    kv_buffers = getattr(metrics, "kv_buffers_mib", None)
    if not isinstance(kv_buffers, dict) or not kv_buffers:
        return None
    total_mib = 0.0
    for raw_value in kv_buffers.values():
        value = _to_float_or_none(raw_value)
        if value is None or value <= 0:
            continue
        total_mib += value
    if total_mib <= 0:
        return None
    return total_mib / 1024.0


def _ctx_gb_from_reason(reason: str | None) -> float | None:
    trace_path = _trace_path_from_reason(reason)
    if trace_path is None:
        return None
    metrics = _parse_probe_trace_metrics(trace_path)
    return _ctx_gb_from_metrics(metrics)


def _probe_fixed_ctx_metrics_once(model: ManagedModel, llama_server: Path, ctx_size: int) -> tuple[bool, str, dict[str, object]]:
    ok, reason, payload = _normalize_probe_result(probe_model_ctx(model, llama_server, int(ctx_size)))
    info = dict(payload or {})
    info["selected_ctx"] = int(ctx_size)

    speed_tps = _to_float_or_none(info.get("probe_speed_tps"))
    totals_s = _to_float_or_none(info.get("probe_totals_s"))
    if speed_tps is None:
        speed_tps = totals_s
        info["probe_speed_tps"] = speed_tps
    if totals_s is None:
        totals_s = speed_tps
        info["probe_totals_s"] = totals_s

    if "selected_ctx_gb" not in info:
        info["selected_ctx_gb"] = _ctx_gb_from_reason(reason)

    trace_path = _trace_path_from_reason(reason)
    if ok and trace_path is not None:
        try:
            trace_path.unlink()
        except FileNotFoundError:
            pass

    return ok, reason, info

def choose_auto_ctx(model: ManagedModel, llama_server: Path, progress_callback = None):
    max_ctx = get_model_context_size(model)
    if max_ctx is None:
        _emit_message(f"{model.model_id}: could not read GGUF max context, skipping.", progress_callback, timestamp=True)
        return None, "metadata-missing", {"max_ctx": None}

    start_ctx = min(8192, max_ctx)
    if start_ctx <= 0:
        _emit_message(f"{model.model_id}: invalid max context {max_ctx}, skipping.", progress_callback, timestamp=True)
        return None, "metadata-missing", {"max_ctx": max_ctx}

    _emit_message(
        f"{model.model_id}: GGUF max ctx {max_ctx}. Auto-fit will calibrate memory and probe boundaries.",
        progress_callback,
        timestamp=True,
    )
    _emit_message(f"{model.model_id}: each probe now sends a conservative long prompt sized for that ctx candidate.", progress_callback, timestamp=True)
    if _has_vision_runtime(model):
        _emit_message(f"{model.model_id}: vision runtime detected, each ctx probe will include a valid image plus text.", progress_callback, timestamp=True)

    probe_max_first = os.environ.get("LLAMACPP_AUTO_CTX_MAX_FIRST", "").lower() in ("1", "true", "yes", "on")

    # Detect whether a previous configured ctx value or prior probe metrics exist.
    prev_ctx_present = False
    try:
        prev_ctx_present = bool(int(model.ctx_size) and int(model.ctx_size) != DEFAULT_CTX_SIZE)
    except Exception:
        prev_ctx_present = False
    if model.ctx_probe_kv_gb is not None or model.ctx_probe_totals_s is not None or model.auto_ctx_failed:
        prev_ctx_present = True

    # Resolve a safely aligned hint from previous config, if any.
    max_probe_ctx = _align_ctx(max_ctx)
    known_ctx_hint = None
    try:
        known_ctx_hint = _align_ctx(int(model.ctx_size))
    except Exception:
        known_ctx_hint = None
    if known_ctx_hint is not None and (known_ctx_hint <= 0 or known_ctx_hint > max_probe_ctx):
        known_ctx_hint = None

    if not probe_max_first:
        free_vram_mib = _query_gpu_free_memory_mib()
        # If a previous cfg ctx exists, test it first before the conservative calibration probe.
        prior_ok = False
        if prev_ctx_present and known_ctx_hint is not None:
            _emit_message(f"{model.model_id}: testing previously configured ctx {known_ctx_hint} first.", progress_callback, timestamp=True)
            ok_hint, hint_reason, hint_metrics = _normalize_probe_result(probe_model_ctx(model, llama_server, known_ctx_hint))
            if ok_hint:
                prior_ok = True
                calibration_ctx = known_ctx_hint
                calibration_probe_metrics = hint_metrics
                calibration_trace = _trace_path_from_reason(hint_reason)
                metrics = _parse_probe_trace_metrics(calibration_trace)
                if calibration_trace is not None:
                    try:
                        calibration_trace.unlink()
                    except FileNotFoundError:
                        pass

                selected_probe_latency_ms = _to_float_or_none(calibration_probe_metrics.get("probe_latency_ms"))
                selected_probe_read_s = _to_float_or_none(calibration_probe_metrics.get("probe_read_s"))
                selected_probe_tokens_s = _to_float_or_none(calibration_probe_metrics.get("probe_tokens_s"))
                selected_probe_totals_s = _to_float_or_none(calibration_probe_metrics.get("probe_totals_s"))
                selected_probe_speed_tps = _to_float_or_none(calibration_probe_metrics.get("probe_speed_tps"))
                if selected_probe_speed_tps is None:
                    selected_probe_speed_tps = selected_probe_totals_s
                if selected_probe_totals_s is None:
                    selected_probe_totals_s = selected_probe_speed_tps
                selected_probe_prompt_tokens = _to_int_or_none(calibration_probe_metrics.get("probe_prompt_tokens"))
                selected_ctx_gb = _ctx_gb_from_metrics(metrics)

                if calibration_ctx >= max_probe_ctx:
                    return calibration_ctx, "selected", {
                        "max_ctx": max_ctx,
                        "calibration_ctx": calibration_ctx,
                        "estimated_ctx": calibration_ctx,
                        "first_failure": None,
                        "selected_ctx": calibration_ctx,
                        "probe_latency_ms": selected_probe_latency_ms,
                        "probe_read_s": selected_probe_read_s,
                        "probe_tokens_s": selected_probe_tokens_s,
                        "probe_totals_s": selected_probe_totals_s,
                        "probe_speed_tps": selected_probe_speed_tps,
                        "probe_prompt_tokens": selected_probe_prompt_tokens,
                        "selected_ctx_gb": selected_ctx_gb,
                    }
            else:
                _emit_message(f"{model.model_id}: previous ctx {known_ctx_hint} no longer works ({hint_reason}).", progress_callback, timestamp=True)

        if not prior_ok:
            calibration_ctx = start_ctx
            _emit_message(f"{model.model_id}: calibration probe at ctx {calibration_ctx}...", progress_callback, timestamp=True)
            ok, reason, calibration_probe_metrics = _normalize_probe_result(probe_model_ctx(model, llama_server, calibration_ctx))
            if not ok:
                _emit_message(f"{model.model_id}: failed even at calibration ctx {calibration_ctx} ({reason}), skipping.", progress_callback, timestamp=True)
                return None, "min-failed", {"max_ctx": max_ctx, "min_ctx": calibration_ctx, "reason": reason}

        if not prior_ok:
            calibration_trace = _trace_path_from_reason(reason)
            metrics = _parse_probe_trace_metrics(calibration_trace)
            if calibration_trace is not None:
                try:
                    calibration_trace.unlink()
                except FileNotFoundError:
                    pass

        selected_probe_latency_ms = _to_float_or_none(calibration_probe_metrics.get("probe_latency_ms"))
        selected_probe_read_s = _to_float_or_none(calibration_probe_metrics.get("probe_read_s"))
        selected_probe_tokens_s = _to_float_or_none(calibration_probe_metrics.get("probe_tokens_s"))
        selected_probe_totals_s = _to_float_or_none(calibration_probe_metrics.get("probe_totals_s"))
        selected_probe_speed_tps = _to_float_or_none(calibration_probe_metrics.get("probe_speed_tps"))
        if selected_probe_speed_tps is None:
            selected_probe_speed_tps = selected_probe_totals_s
        if selected_probe_totals_s is None:
            selected_probe_totals_s = selected_probe_speed_tps
        selected_probe_prompt_tokens = _to_int_or_none(calibration_probe_metrics.get("probe_prompt_tokens"))
        selected_ctx_gb = _ctx_gb_from_metrics(metrics)

        estimated_ctx = _estimate_ctx_ceiling(model, calibration_ctx, metrics, free_vram_mib) or calibration_ctx
        estimated_ctx = max(calibration_ctx, min(_align_ctx(max_ctx), estimated_ctx))
        _emit_message(f"{model.model_id}: estimated stable ctx ceiling from memory fit = {estimated_ctx}.", progress_callback, timestamp=True)

        low = calibration_ctx
        max_probe_ctx = _align_ctx(max_ctx)
        high = None
        last_success = calibration_ctx
        first_failure = None
        tested = {calibration_ctx}
        min_ctx_rechecked = False

        known_ctx_hint = _align_ctx(int(model.ctx_size))
        if known_ctx_hint <= calibration_ctx or known_ctx_hint > max_probe_ctx:
            known_ctx_hint = None

        candidate_order: list[int] = []
        if known_ctx_hint is not None:
            candidate_order.append(known_ctx_hint)
            _emit_message(
                f"{model.model_id}: refresh hint using previous cfg ctx {known_ctx_hint} as first upper probe.",
                progress_callback,
                timestamp=True,
            )
        if max_probe_ctx > calibration_ctx:
            candidate_order.append(max_probe_ctx)
        if estimated_ctx > calibration_ctx and estimated_ctx < max_probe_ctx:
            candidate_order.append(estimated_ctx)
            optimistic_ctx = _align_ctx(min(max_probe_ctx, max(estimated_ctx + 1024, int(estimated_ctx * 1.2))))
            if optimistic_ctx > estimated_ctx and optimistic_ctx < max_probe_ctx:
                candidate_order.append(optimistic_ctx)

        for candidate in candidate_order:
            if candidate in tested:
                continue
            tested.add(candidate)
            _emit_message(f"{model.model_id}: testing ctx {candidate}...", progress_callback, timestamp=True)
            ok, reason, probe_metrics = _normalize_probe_result(probe_model_ctx(model, llama_server, candidate))
            # If context probing fails due to n_ctx_seq limit, reduce batch/ubatch to allow larger context
            if not ok and "exceeds the available context size" in str(reason):
                _emit_message(f"{model.model_id}: ctx {candidate} failed due to batch memory; reducing batch_size to prioritize context...", progress_callback, timestamp=True)
                # Reduce batch parameters and retry once
                if not hasattr(model, 'server_overrides'):
                    model.server_overrides = {}
                original_batch = model.server_overrides.get("batch_size")
                original_ubatch = model.server_overrides.get("ubatch_size")
                original_parallel = model.server_overrides.get("parallel")
                
                # Aggressively reduce batch/ubatch to maximize context over throughput
                model.server_overrides["batch_size"] = 512
                model.server_overrides["ubatch_size"] = 256
                model.server_overrides["parallel"] = 1
                
                ok, reason, probe_metrics = _normalize_probe_result(probe_model_ctx(model, llama_server, candidate))
                if not ok:
                    # Restore originals if still failing
                    if original_batch is not None:
                        model.server_overrides["batch_size"] = original_batch
                    else:
                        model.server_overrides.pop("batch_size", None)
                    if original_ubatch is not None:
                        model.server_overrides["ubatch_size"] = original_ubatch
                    else:
                        model.server_overrides.pop("ubatch_size", None)
                    if original_parallel is not None:
                        model.server_overrides["parallel"] = original_parallel
                    else:
                        model.server_overrides.pop("parallel", None)
                    _emit_message(f"{model.model_id}: ctx {candidate} still failed after batch reduction ({reason}).", progress_callback, timestamp=True)
                else:
                    _emit_message(f"{model.model_id}: ctx {candidate} success after reducing batch_size.", progress_callback, timestamp=True)
            if ok:
                low = candidate
                last_success = candidate
                selected_probe_latency_ms = _to_float_or_none(probe_metrics.get("probe_latency_ms"))
                selected_probe_read_s = _to_float_or_none(probe_metrics.get("probe_read_s"))
                selected_probe_tokens_s = _to_float_or_none(probe_metrics.get("probe_tokens_s"))
                selected_probe_totals_s = _to_float_or_none(probe_metrics.get("probe_totals_s"))
                selected_probe_speed_tps = _to_float_or_none(probe_metrics.get("probe_speed_tps"))
                if selected_probe_speed_tps is None:
                    selected_probe_speed_tps = selected_probe_totals_s
                if selected_probe_totals_s is None:
                    selected_probe_totals_s = selected_probe_speed_tps
                selected_probe_prompt_tokens = _to_int_or_none(probe_metrics.get("probe_prompt_tokens"))
                selected_ctx_gb = _ctx_gb_from_reason(reason) or selected_ctx_gb
                if candidate == max_probe_ctx:
                    return last_success, "selected", {
                        "max_ctx": max_ctx,
                        "calibration_ctx": calibration_ctx,
                        "estimated_ctx": estimated_ctx,
                        "first_failure": first_failure,
                        "selected_ctx": last_success,
                        "probe_latency_ms": selected_probe_latency_ms,
                        "probe_read_s": selected_probe_read_s,
                        "probe_tokens_s": selected_probe_tokens_s,
                        "probe_totals_s": selected_probe_totals_s,
                        "probe_speed_tps": selected_probe_speed_tps,
                        "probe_prompt_tokens": selected_probe_prompt_tokens,
                        "selected_ctx_gb": selected_ctx_gb,
                    }
                continue
            if high is None or candidate < high:
                high = candidate
            first_failure = candidate
            _emit_message(f"{model.model_id}: ctx {candidate} failed ({reason}).", progress_callback, timestamp=True)
            if not min_ctx_rechecked and candidate > calibration_ctx:
                min_ctx_rechecked = True
                _emit_message(
                    f"{model.model_id}: guard probe at minimum ctx {calibration_ctx} after upper-bound failure...",
                    progress_callback,
                    timestamp=True,
                )
                min_ok, min_reason, _min_probe_metrics = _normalize_probe_result(probe_model_ctx(model, llama_server, calibration_ctx))
                if not min_ok:
                    _emit_message(
                        f"{model.model_id}: minimum ctx {calibration_ctx} failed on guard probe ({min_reason}).",
                        progress_callback,
                        timestamp=True,
                    )
                    return None, "min-failed", {
                        "max_ctx": max_ctx,
                        "min_ctx": calibration_ctx,
                        "reason": min_reason,
                        "guard_failed_after": candidate,
                    }
            if candidate != max_probe_ctx:
                break

        if high is None and low < max_probe_ctx:
            high = max_probe_ctx

        while high is not None and high - low >= 4096:
            midpoint = _align_ctx((low + high) // 2)
            if midpoint <= low or midpoint >= high or midpoint in tested:
                break
            tested.add(midpoint)
            _emit_message(f"{model.model_id}: refinement probe at {midpoint}...", progress_callback, timestamp=True)
            ok, reason, probe_metrics = _normalize_probe_result(probe_model_ctx(model, llama_server, midpoint))
            # If refinement fails due to n_ctx_seq limit, reduce batch/ubatch to allow larger context
            if not ok and "exceeds the available context size" in str(reason):
                _emit_message(f"{model.model_id}: ctx {midpoint} failed due to batch memory; reducing batch_size to prioritize context...", progress_callback, timestamp=True)
                # Reduce batch parameters and retry
                if not hasattr(model, 'server_overrides'):
                    model.server_overrides = {}
                original_batch = model.server_overrides.get("batch_size")
                original_ubatch = model.server_overrides.get("ubatch_size")
                original_parallel = model.server_overrides.get("parallel")
                
                # Aggressively reduce batch/ubatch to maximize context over throughput
                model.server_overrides["batch_size"] = 512
                model.server_overrides["ubatch_size"] = 256
                model.server_overrides["parallel"] = 1
                
                ok, reason, probe_metrics = _normalize_probe_result(probe_model_ctx(model, llama_server, midpoint))
                if not ok:
                    if original_batch is not None:
                        model.server_overrides["batch_size"] = original_batch
                    else:
                        model.server_overrides.pop("batch_size", None)
                    if original_ubatch is not None:
                        model.server_overrides["ubatch_size"] = original_ubatch
                    else:
                        model.server_overrides.pop("ubatch_size", None)
                    if original_parallel is not None:
                        model.server_overrides["parallel"] = original_parallel
                    else:
                        model.server_overrides.pop("parallel", None)
                    _emit_message(f"{model.model_id}: ctx {midpoint} still failed after batch reduction ({reason}).", progress_callback, timestamp=True)
                else:
                    _emit_message(f"{model.model_id}: ctx {midpoint} success after reducing batch_size.", progress_callback, timestamp=True)
            if ok:
                low = midpoint
                last_success = midpoint
                selected_probe_latency_ms = _to_float_or_none(probe_metrics.get("probe_latency_ms"))
                selected_probe_read_s = _to_float_or_none(probe_metrics.get("probe_read_s"))
                selected_probe_tokens_s = _to_float_or_none(probe_metrics.get("probe_tokens_s"))
                selected_probe_totals_s = _to_float_or_none(probe_metrics.get("probe_totals_s"))
                selected_probe_speed_tps = _to_float_or_none(probe_metrics.get("probe_speed_tps"))
                if selected_probe_speed_tps is None:
                    selected_probe_speed_tps = selected_probe_totals_s
                if selected_probe_totals_s is None:
                    selected_probe_totals_s = selected_probe_speed_tps
                selected_probe_prompt_tokens = _to_int_or_none(probe_metrics.get("probe_prompt_tokens"))
                selected_ctx_gb = _ctx_gb_from_reason(reason) or selected_ctx_gb
            else:
                high = midpoint
                first_failure = midpoint
                _emit_message(f"{model.model_id}: refinement ctx {midpoint} failed ({reason}).", progress_callback, timestamp=True)

        return last_success, "selected", {
            "max_ctx": max_ctx,
            "calibration_ctx": calibration_ctx,
            "estimated_ctx": estimated_ctx,
            "first_failure": first_failure,
            "selected_ctx": last_success,
            "probe_latency_ms": selected_probe_latency_ms,
            "probe_read_s": selected_probe_read_s,
            "probe_tokens_s": selected_probe_tokens_s,
            "probe_totals_s": selected_probe_totals_s,
            "probe_speed_tps": selected_probe_speed_tps,
            "probe_prompt_tokens": selected_probe_prompt_tokens,
            "selected_ctx_gb": selected_ctx_gb,
        }

    # probe_max_first branch
    free_vram_mib = _query_gpu_free_memory_mib()
    max_probe_ctx = _align_ctx(max_ctx)
    min_probe_ctx = _align_ctx(1024)

    prev_ctx_present = False
    try:
        prev_ctx_present = bool(int(model.ctx_size) and int(model.ctx_size) != DEFAULT_CTX_SIZE)
    except Exception:
        prev_ctx_present = False
    if model.ctx_probe_kv_gb is not None or model.ctx_probe_totals_s is not None or model.auto_ctx_failed:
        prev_ctx_present = True

    selected_probe_latency_ms = None
    selected_probe_read_s = None
    selected_probe_tokens_s = None
    selected_probe_totals_s = None
    selected_probe_speed_tps = None
    selected_probe_prompt_tokens = None
    selected_ctx_gb = None

    if not prev_ctx_present:
        _emit_message(f"{model.model_id}: no previous ctx probe found — testing max ctx {max_probe_ctx} first.", progress_callback)
        ok, reason, probe_metrics = _normalize_probe_result(probe_model_ctx(model, llama_server, max_probe_ctx))
        if ok:
            calibration_ctx = max_probe_ctx
            last_success = max_probe_ctx
            selected_probe_latency_ms = _to_float_or_none(probe_metrics.get("probe_latency_ms"))
            selected_probe_read_s = _to_float_or_none(probe_metrics.get("probe_read_s"))
            selected_probe_tokens_s = _to_float_or_none(probe_metrics.get("probe_tokens_s"))
            selected_probe_totals_s = _to_float_or_none(probe_metrics.get("probe_totals_s"))
            selected_probe_speed_tps = _to_float_or_none(probe_metrics.get("probe_speed_tps"))
            if selected_probe_speed_tps is None:
                selected_probe_speed_tps = selected_probe_totals_s
            if selected_probe_totals_s is None:
                selected_probe_totals_s = selected_probe_speed_tps
            selected_probe_prompt_tokens = _to_int_or_none(probe_metrics.get("probe_prompt_tokens"))
            trace = _trace_path_from_reason(reason)
            metrics = _parse_probe_trace_metrics(trace)
            selected_ctx_gb = _ctx_gb_from_metrics(metrics) or selected_ctx_gb
            if trace is not None:
                try:
                    trace.unlink()
                except FileNotFoundError:
                    pass
            return last_success, "selected", {
                "max_ctx": max_ctx,
                "calibration_ctx": calibration_ctx,
                "estimated_ctx": calibration_ctx,
                "first_failure": None,
                "selected_ctx": last_success,
                "probe_latency_ms": selected_probe_latency_ms,
                "probe_read_s": selected_probe_read_s,
                "probe_tokens_s": selected_probe_tokens_s,
                "probe_totals_s": selected_probe_totals_s,
                "probe_speed_tps": selected_probe_speed_tps,
                "probe_prompt_tokens": selected_probe_prompt_tokens,
                "selected_ctx_gb": selected_ctx_gb,
            }

        _emit_message(f"{model.model_id}: max ctx {max_probe_ctx} failed ({reason}). Trying minimum ctx {min_probe_ctx}...", progress_callback)
        ok_min, min_reason, min_metrics = _normalize_probe_result(probe_model_ctx(model, llama_server, min_probe_ctx))
        if not ok_min:
            _emit_message(f"{model.model_id}: minimum ctx {min_probe_ctx} failed ({min_reason}).", progress_callback)
            return None, "min-failed", {"max_ctx": max_ctx, "min_ctx": min_probe_ctx, "reason": min_reason}

        calibration_ctx = min_probe_ctx
        calibration_probe_metrics = min_metrics
        calibration_trace = _trace_path_from_reason(min_reason)
        metrics = _parse_probe_trace_metrics(calibration_trace)
        if calibration_trace is not None:
            try:
                calibration_trace.unlink()
            except FileNotFoundError:
                pass

        selected_probe_latency_ms = _to_float_or_none(calibration_probe_metrics.get("probe_latency_ms"))
        selected_probe_read_s = _to_float_or_none(calibration_probe_metrics.get("probe_read_s"))
        selected_probe_tokens_s = _to_float_or_none(calibration_probe_metrics.get("probe_tokens_s"))
        selected_probe_totals_s = _to_float_or_none(calibration_probe_metrics.get("probe_totals_s"))
        selected_probe_speed_tps = _to_float_or_none(calibration_probe_metrics.get("probe_speed_tps"))
        if selected_probe_speed_tps is None:
            selected_probe_speed_tps = selected_probe_totals_s
        if selected_probe_totals_s is None:
            selected_probe_totals_s = selected_probe_speed_tps
        selected_probe_prompt_tokens = _to_int_or_none(calibration_probe_metrics.get("probe_prompt_tokens"))
        selected_ctx_gb = _ctx_gb_from_metrics(metrics)

    prev_ctx_present = False
    try:
        prev_ctx_present = bool(int(model.ctx_size) and int(model.ctx_size) != DEFAULT_CTX_SIZE)
    except Exception:
        prev_ctx_present = False
    if model.ctx_probe_kv_gb is not None or model.ctx_probe_totals_s is not None or model.auto_ctx_failed:
        prev_ctx_present = True

    selected_probe_latency_ms = None
    selected_probe_read_s = None
    selected_probe_tokens_s = None
    selected_probe_totals_s = None
    selected_probe_speed_tps = None
    selected_probe_prompt_tokens = None
    selected_ctx_gb = None

    if not prev_ctx_present:
        _emit_message(f"{model.model_id}: no previous ctx probe found — testing max ctx {max_probe_ctx} first.", progress_callback)
        ok, reason, probe_metrics = _normalize_probe_result(probe_model_ctx(model, llama_server, max_probe_ctx))
        if ok:
            calibration_ctx = max_probe_ctx
            last_success = max_probe_ctx
            selected_probe_latency_ms = _to_float_or_none(probe_metrics.get("probe_latency_ms"))
            selected_probe_read_s = _to_float_or_none(probe_metrics.get("probe_read_s"))
            selected_probe_tokens_s = _to_float_or_none(probe_metrics.get("probe_tokens_s"))
            selected_probe_totals_s = _to_float_or_none(probe_metrics.get("probe_totals_s"))
            selected_probe_speed_tps = _to_float_or_none(probe_metrics.get("probe_speed_tps"))
            if selected_probe_speed_tps is None:
                selected_probe_speed_tps = selected_probe_totals_s
            if selected_probe_totals_s is None:
                selected_probe_totals_s = selected_probe_speed_tps
            selected_probe_prompt_tokens = _to_int_or_none(probe_metrics.get("probe_prompt_tokens"))
            trace = _trace_path_from_reason(reason)
            metrics = _parse_probe_trace_metrics(trace)
            selected_ctx_gb = _ctx_gb_from_metrics(metrics) or selected_ctx_gb
            if trace is not None:
                try:
                    trace.unlink()
                except FileNotFoundError:
                    pass
            return last_success, "selected", {
                "max_ctx": max_ctx,
                "calibration_ctx": calibration_ctx,
                "estimated_ctx": calibration_ctx,
                "first_failure": None,
                "selected_ctx": last_success,
                "probe_latency_ms": selected_probe_latency_ms,
                "probe_read_s": selected_probe_read_s,
                "probe_tokens_s": selected_probe_tokens_s,
                "probe_totals_s": selected_probe_totals_s,
                "probe_speed_tps": selected_probe_speed_tps,
                "probe_prompt_tokens": selected_probe_prompt_tokens,
                "selected_ctx_gb": selected_ctx_gb,
            }

        _emit_message(f"{model.model_id}: max ctx {max_probe_ctx} failed ({reason}). Trying minimum ctx {min_probe_ctx}...", progress_callback)
        ok_min, min_reason, min_metrics = _normalize_probe_result(probe_model_ctx(model, llama_server, min_probe_ctx))
        if not ok_min:
            _emit_message(f"{model.model_id}: minimum ctx {min_probe_ctx} failed ({min_reason}).", progress_callback)
            return None, "min-failed", {"max_ctx": max_ctx, "min_ctx": min_probe_ctx, "reason": min_reason}

        calibration_ctx = min_probe_ctx
        calibration_probe_metrics = min_metrics
        calibration_trace = _trace_path_from_reason(min_reason)
        metrics = _parse_probe_trace_metrics(calibration_trace)
        if calibration_trace is not None:
            try:
                calibration_trace.unlink()
            except FileNotFoundError:
                pass

        selected_probe_latency_ms = _to_float_or_none(calibration_probe_metrics.get("probe_latency_ms"))
        selected_probe_read_s = _to_float_or_none(calibration_probe_metrics.get("probe_read_s"))
        selected_probe_tokens_s = _to_float_or_none(calibration_probe_metrics.get("probe_tokens_s"))
        selected_probe_totals_s = _to_float_or_none(calibration_probe_metrics.get("probe_totals_s"))
        selected_probe_speed_tps = _to_float_or_none(calibration_probe_metrics.get("probe_speed_tps"))
        if selected_probe_speed_tps is None:
            selected_probe_speed_tps = selected_probe_totals_s
        if selected_probe_totals_s is None:
            selected_probe_totals_s = selected_probe_speed_tps
        selected_probe_prompt_tokens = _to_int_or_none(calibration_probe_metrics.get("probe_prompt_tokens"))
        selected_ctx_gb = _ctx_gb_from_metrics(metrics)

    else:
        known_ctx_hint = None
        try:
            known_ctx_hint = _align_ctx(int(model.ctx_size))
        except Exception:
            known_ctx_hint = None
        if known_ctx_hint is None or known_ctx_hint <= 0 or known_ctx_hint > max_probe_ctx:
            known_ctx_hint = None

        if known_ctx_hint is not None:
            _emit_message(f"{model.model_id}: testing previously configured ctx {known_ctx_hint} first.", progress_callback)
            ok_hint, hint_reason, hint_metrics = _normalize_probe_result(probe_model_ctx(model, llama_server, known_ctx_hint))
            if ok_hint:
                calibration_ctx = known_ctx_hint
                calibration_probe_metrics = hint_metrics
                calibration_trace = _trace_path_from_reason(hint_reason)
                metrics = _parse_probe_trace_metrics(calibration_trace)
                if calibration_trace is not None:
                    try:
                        calibration_trace.unlink()
                    except FileNotFoundError:
                        pass

                selected_probe_latency_ms = _to_float_or_none(calibration_probe_metrics.get("probe_latency_ms"))
                selected_probe_read_s = _to_float_or_none(calibration_probe_metrics.get("probe_read_s"))
                selected_probe_tokens_s = _to_float_or_none(calibration_probe_metrics.get("probe_tokens_s"))
                selected_probe_totals_s = _to_float_or_none(calibration_probe_metrics.get("probe_totals_s"))
                selected_probe_speed_tps = _to_float_or_none(calibration_probe_metrics.get("probe_speed_tps"))
                if selected_probe_speed_tps is None:
                    selected_probe_speed_tps = selected_probe_totals_s
                if selected_probe_totals_s is None:
                    selected_probe_totals_s = selected_probe_speed_tps
                selected_probe_prompt_tokens = _to_int_or_none(calibration_probe_metrics.get("probe_prompt_tokens"))
                selected_ctx_gb = _ctx_gb_from_metrics(metrics)

                if calibration_ctx >= max_probe_ctx:
                    return calibration_ctx, "selected", {
                        "max_ctx": max_ctx,
                        "calibration_ctx": calibration_ctx,
                        "estimated_ctx": calibration_ctx,
                        "first_failure": None,
                        "selected_ctx": calibration_ctx,
                        "probe_latency_ms": selected_probe_latency_ms,
                        "probe_read_s": selected_probe_read_s,
                        "probe_tokens_s": selected_probe_tokens_s,
                        "probe_totals_s": selected_probe_totals_s,
                        "probe_speed_tps": selected_probe_speed_tps,
                        "probe_prompt_tokens": selected_probe_prompt_tokens,
                        "selected_ctx_gb": selected_ctx_gb,
                    }
            else:
                _emit_message(f"{model.model_id}: previous ctx {known_ctx_hint} no longer works ({hint_reason}).", progress_callback)
                calibration_ctx = start_ctx
                calibration_probe_metrics = None
                metrics = ProbeTraceMetrics()

        else:
            calibration_ctx = start_ctx
            calibration_probe_metrics = None
            metrics = ProbeTraceMetrics()

    estimated_ctx = _estimate_ctx_ceiling(model, calibration_ctx, metrics, free_vram_mib) or calibration_ctx
    estimated_ctx = max(calibration_ctx, min(_align_ctx(max_ctx), estimated_ctx))
    _emit_message(f"{model.model_id}: estimated stable ctx ceiling from memory fit = {estimated_ctx}.", progress_callback)

    low = calibration_ctx
    last_success = calibration_ctx
    max_probe_ctx = _align_ctx(max_ctx)
    high = None
    first_failure = None
    tested = {calibration_ctx}
    min_ctx_rechecked = False

    known_ctx_hint = None
    try:
        known_ctx_hint = _align_ctx(int(model.ctx_size))
    except Exception:
        known_ctx_hint = None
    if known_ctx_hint is not None and (known_ctx_hint <= calibration_ctx or known_ctx_hint > max_probe_ctx):
        known_ctx_hint = None

    candidate_order: list[int] = []
    if known_ctx_hint is not None:
        candidate_order.append(known_ctx_hint)
        _emit_message(
            f"{model.model_id}: refresh hint using previous cfg ctx {known_ctx_hint} as first upper probe.",
            progress_callback,
        )
    if max_probe_ctx > calibration_ctx:
        candidate_order.append(max_probe_ctx)
    if estimated_ctx > calibration_ctx and estimated_ctx < max_probe_ctx:
        candidate_order.append(estimated_ctx)
        optimistic_ctx = _align_ctx(min(max_probe_ctx, max(estimated_ctx + 1024, int(estimated_ctx * 1.2))))
        if optimistic_ctx > estimated_ctx and optimistic_ctx < max_probe_ctx:
            candidate_order.append(optimistic_ctx)

    for candidate in candidate_order:
        if candidate in tested:
            continue
        tested.add(candidate)
        _emit_message(f"{model.model_id}: testing ctx {candidate}...", progress_callback)
        ok, reason, probe_metrics = _normalize_probe_result(probe_model_ctx(model, llama_server, candidate))
        if ok:
            low = candidate
            last_success = candidate
            selected_probe_latency_ms = _to_float_or_none(probe_metrics.get("probe_latency_ms"))
            selected_probe_read_s = _to_float_or_none(probe_metrics.get("probe_read_s"))
            selected_probe_tokens_s = _to_float_or_none(probe_metrics.get("probe_tokens_s"))
            selected_probe_totals_s = _to_float_or_none(probe_metrics.get("probe_totals_s"))
            selected_probe_speed_tps = _to_float_or_none(probe_metrics.get("probe_speed_tps"))
            if selected_probe_speed_tps is None:
                selected_probe_speed_tps = selected_probe_totals_s
            if selected_probe_totals_s is None:
                selected_probe_totals_s = selected_probe_speed_tps
            selected_probe_prompt_tokens = _to_int_or_none(probe_metrics.get("probe_prompt_tokens"))
            selected_ctx_gb = _ctx_gb_from_reason(reason) or selected_ctx_gb
            if candidate == max_probe_ctx:
                return last_success, "selected", {
                    "max_ctx": max_ctx,
                    "calibration_ctx": calibration_ctx,
                    "estimated_ctx": estimated_ctx,
                    "first_failure": first_failure,
                    "selected_ctx": last_success,
                    "probe_latency_ms": selected_probe_latency_ms,
                    "probe_read_s": selected_probe_read_s,
                    "probe_tokens_s": selected_probe_tokens_s,
                    "probe_totals_s": selected_probe_totals_s,
                    "probe_speed_tps": selected_probe_speed_tps,
                    "probe_prompt_tokens": selected_probe_prompt_tokens,
                    "selected_ctx_gb": selected_ctx_gb,
                }
            continue
        if high is None or candidate < high:
            high = candidate
        first_failure = candidate
        _emit_message(f"{model.model_id}: ctx {candidate} failed ({reason}).", progress_callback)
        if not min_ctx_rechecked and candidate > calibration_ctx:
            min_ctx_rechecked = True
            _emit_message(
                f"{model.model_id}: guard probe at minimum ctx {calibration_ctx} after upper-bound failure...",
                progress_callback,
            )
            min_ok, min_reason, _min_probe_metrics = _normalize_probe_result(probe_model_ctx(model, llama_server, calibration_ctx))
            if not min_ok:
                _emit_message(
                    f"{model.model_id}: minimum ctx {calibration_ctx} failed on guard probe ({min_reason}).",
                    progress_callback,
                )
                return None, "min-failed", {
                    "max_ctx": max_ctx,
                    "min_ctx": calibration_ctx,
                    "reason": min_reason,
                    "guard_failed_after": candidate,
                }
        if candidate != max_probe_ctx:
            break

    if high is None and low < max_probe_ctx:
        high = max_probe_ctx

    while high is not None and high - low >= 4096:
        midpoint = _align_ctx((low + high) // 2)
        if midpoint <= low or midpoint >= high or midpoint in tested:
            break
        tested.add(midpoint)
        _emit_message(f"{model.model_id}: refinement probe at {midpoint}...", progress_callback)
        ok, reason, probe_metrics = _normalize_probe_result(probe_model_ctx(model, llama_server, midpoint))
        if ok:
            low = midpoint
            last_success = midpoint
            selected_probe_latency_ms = _to_float_or_none(probe_metrics.get("probe_latency_ms"))
            selected_probe_read_s = _to_float_or_none(probe_metrics.get("probe_read_s"))
            selected_probe_tokens_s = _to_float_or_none(probe_metrics.get("probe_tokens_s"))
            selected_probe_totals_s = _to_float_or_none(probe_metrics.get("probe_totals_s"))
            selected_probe_speed_tps = _to_float_or_none(probe_metrics.get("probe_speed_tps"))
            if selected_probe_speed_tps is None:
                selected_probe_speed_tps = selected_probe_totals_s
            if selected_probe_totals_s is None:
                selected_probe_totals_s = selected_probe_speed_tps
            selected_probe_prompt_tokens = _to_int_or_none(probe_metrics.get("probe_prompt_tokens"))
            selected_ctx_gb = _ctx_gb_from_reason(reason) or selected_ctx_gb
        else:
            high = midpoint
            first_failure = midpoint
            _emit_message(f"{model.model_id}: refinement ctx {midpoint} failed ({reason}).", progress_callback)

    return last_success, "selected", {
        "max_ctx": max_ctx,
        "calibration_ctx": calibration_ctx,
        "estimated_ctx": estimated_ctx,
        "first_failure": first_failure,
        "selected_ctx": last_success,
        "probe_latency_ms": selected_probe_latency_ms,
        "probe_read_s": selected_probe_read_s,
        "probe_tokens_s": selected_probe_tokens_s,
        "probe_totals_s": selected_probe_totals_s,
        "probe_speed_tps": selected_probe_speed_tps,
        "probe_prompt_tokens": selected_probe_prompt_tokens,
        "selected_ctx_gb": selected_ctx_gb,
    }


def _materialize_validation_model(args, progress_callback = None) -> tuple[ManagedModel, Path | None]:
    catalog = load_catalog(args.catalog, _args_server_config_path(args))
    try:
        existing = resolve_catalog_model(
            catalog,
            target=args.repo,
            repo_ref=args.hf,
            model_id=args.model_id,
            filename=args.file,
        )
        return existing, None
    except RuntimeError:
        pass

    ref = args.repo or args.hf or args.model_id
    if not ref:
        raise RuntimeError("Model reference required.")

    repo_id, quant = parse_hf_input(ref)
    token = args.hf_token or os.environ.get("HF_TOKEN")
    api = HfApi()
    selected_file = choose_gguf_file(api, repo_id, quant, args.file, token)
    expected_sizes = _repo_sibling_sizes(api, repo_id, token)
    temp_root = Path(tempfile.mkdtemp(prefix="llamacpp-validate-"))
    repo_dir = temp_root / repo_id
    repo_dir.mkdir(parents=True, exist_ok=True)
    model_path = download_hf_file(
        repo_id=repo_id,
        filename=selected_file,
        token=token,
        target_dir=repo_dir,
        label=Path(selected_file).name,
        progress_callback=progress_callback,
        expected_size=expected_sizes.get(selected_file),
    )

    mmproj_filename = choose_mmproj_file(api, repo_id, token)
    mmproj_path = None
    if mmproj_filename:
        mmproj_path = download_hf_file(
            repo_id=repo_id,
            filename=mmproj_filename,
            token=token,
            target_dir=repo_dir,
            label=f"mmproj {Path(mmproj_filename).name}",
            progress_callback=progress_callback,
            expected_size=expected_sizes.get(mmproj_filename),
        )

    temp_model = ManagedModel(
        model_id=normalize_model_id(repo_id, quant, selected_file),
        repo_id=repo_id,
        quant=quant,
        filename=selected_file,
        local_path=str(model_path),
        mmproj_filename=mmproj_filename,
        mmproj_path=mmproj_path,
        ctx_size=int(args.ctx_size),
        n_gpu_layers=int(args.n_gpu_layers),
        tensor_split=normalize_tensor_split(args.tensor_split),
        host=args.host,
        jinja=not args.no_jinja,
        ttl=resolve_idle_ttl(args),
        description=args.description or f"{repo_id} / {selected_file}",
    )
    temp_model.tensor_split = preferred_tensor_split(temp_model, temp_model.tensor_split)
    return temp_model, temp_root


def validate_model(args, progress_callback = None):
    stable_catalog = load_catalog(args.catalog, _args_server_config_path(args))
    model, temp_root = _materialize_validation_model(args, progress_callback=progress_callback)
    try:
        temporarily_unload_published_models(args, progress_callback=progress_callback)
        if getattr(args, "ctx_override", None) is not None:
            model.ctx_size = int(args.ctx_override)
        elif getattr(args, "auto_ctx", False) or temp_root is not None:
            best_ctx, status, info = choose_auto_ctx(model, args.llama_server, progress_callback=progress_callback)
            if best_ctx is None:
                raise RuntimeError(f"Could not auto-fit ctx for {model.model_id}: {status} {info}")
            model.ctx_size = int(best_ctx)
            _emit_message(f"{model.model_id}: auto-fit selected ctx {model.ctx_size}.", progress_callback)

        result = validate_model_ctx_runtime(model, args.llama_server, int(model.ctx_size), progress_callback=progress_callback)
        print(json.dumps({
            "model_id": model.model_id,
            "repo_id": model.repo_id,
            "ctx": model.ctx_size,
            "multimodal": _has_vision_runtime(model),
            **result,
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        try:
            restore_catalog_config(args, stable_catalog, progress_callback=progress_callback, restart_service=True)
        except Exception:
            pass
        if temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)

def delete_downloaded_files(target_dir: Path, filenames: list[str]):
    removed = 0
    for filename in filenames:
        final_path = target_dir / filename
        part_path = final_path.with_name(final_path.name + ".part")
        for path in [final_path, part_path]:
            if path.exists():
                try:
                    path.unlink()
                    removed += 1
                except IsADirectoryError:
                    shutil.rmtree(path, ignore_errors=True)
                    removed += 1
                except FileNotFoundError:
                    pass
    return removed

def update_config(args, progress_callback = None):
    try:
        is_owner = (os.getuid() == 0 or os.getuid() == os.stat(args.catalog.parent).st_uid)
    except:
        is_owner = False

    if not is_owner:
        try:
            return run_manager_command("update", args)
        except RuntimeError:
            raise
        except Exception as e:
            if os.geteuid() != 0:
                raise manager_unavailable_error(e)
            _emit_message(
                f"Manager unavailable ({e}). Continuing with a local update as root.",
                progress_callback,
            )

    catalog = load_catalog(args.catalog, _args_server_config_path(args))
    target = getattr(args, "repo", None)
    repo_ref = getattr(args, "hf", None)
    model_id = getattr(args, "model_id", None)
    filename = getattr(args, "file", None)
    ctx_override = getattr(args, "ctx_override", None)
    ctx_override = int(ctx_override) if ctx_override is not None else None
    auto_ctx = bool(getattr(args, "auto_ctx", False))
    preserve_ctx = bool(getattr(args, "preserve_ctx", False))
    sync_gguf_ctx = bool(getattr(args, "sync_gguf_ctx", False))
    if preserve_ctx:
        sync_gguf_ctx = False
    if ctx_override is not None and auto_ctx:
        raise RuntimeError("Use either -ctx or --auto, not both.")

    if target or repo_ref or model_id or filename:
        selected_model = resolve_catalog_model(catalog, target=target, repo_ref=repo_ref, model_id=model_id, filename=filename)
        target_models = [selected_model]
    else:
        target_models = catalog

    deleted_models = 0
    auto_ctx_rows: list[dict[str, str]] = []
    if auto_ctx:
        probe_config_replaced = False
        if target_models:
            probe_config_replaced = True
            temporarily_unload_published_models(args, progress_callback=progress_callback)
        paired_spec_by_model: dict[str, tuple[str, str]] = {}
        paired_draft_model_ids: set[str] = set()
        for item in target_models:
            if not bool(getattr(item, "speculative", False)):
                continue
            spec_meta = item.spec_meta if isinstance(item.spec_meta, dict) else {}
            base_model_id = str(spec_meta.get("base_model_id") or item.spec_variant_of or "").strip()
            draft_model_id = str(spec_meta.get("draft_model_id") or "").strip()
            if base_model_id and draft_model_id:
                paired_spec_by_model[item.model_id] = (base_model_id, draft_model_id)
                paired_draft_model_ids.add(draft_model_id)

        ordered_models = [m for m in target_models if not bool(getattr(m, "speculative", False))]
        ordered_models.extend([m for m in target_models if bool(getattr(m, "speculative", False))])
        total_models = len(ordered_models)
        updated_ctx = 0
        missing_ctx = 0
        min_failed_models: list[ManagedModel] = []
        try:
            for idx, model in enumerate(ordered_models, start=1):
                if model.model_id in paired_draft_model_ids and model.model_id not in paired_spec_by_model:
                    _emit_message(
                        f"[{idx}/{total_models}] Skipping draft-only auto-ctx probe for paired speculative draft {model.model_id}.",
                        progress_callback,
                        timestamp=True,
                    )
                    auto_ctx_rows.append(
                        {
                            "MODEL": model.model_id,
                            "CFG_CTX": _display_cfg_ctx(model),
                            "API_CTX": _display_api_ctx(model),
                            "CTX_GB": _format_ctx_probe_gb(model.ctx_probe_kv_gb),
                            "READ/S": _format_ctx_probe_rate(model.ctx_probe_read_s),
                            "TOKENS/S": _format_ctx_probe_rate(model.ctx_probe_tokens_s),
                            "TOTALS/S": _format_ctx_probe_rate(model.ctx_probe_totals_s),
                            "LATENCY": _format_ctx_probe_latency(model.ctx_probe_latency_ms),
                            "STATUS": "paired-draft-skip",
                        }
                    )
                    continue

                paired_spec_info = paired_spec_by_model.get(model.model_id)
                if paired_spec_info is not None:
                    base_model_id, _draft_model_id = paired_spec_info
                    base_model = next((item for item in catalog if item.model_id == base_model_id), None)
                    if base_model is not None:
                        paired_ctx = int(getattr(base_model, "ctx_size", 0) or model.ctx_size or DEFAULT_CTX_SIZE)
                        # If the master entry still has the default ctx or a prior
                        # auto-ctx probe failed, run a full auto-probe on the master
                        # so we can obtain a concrete selected ctx to apply to the
                        # speculative pair. Previously we only captured metrics at
                        # the master ctx which left catalog ctx unchanged (0 updates).
                        if (paired_ctx == DEFAULT_CTX_SIZE) or bool(getattr(base_model, "auto_ctx_failed", False)):
                            _emit_message(
                                f"[{idx}/{total_models}] Master {base_model_id} appears unprobed or previously failed; running full auto-probe on master.",
                                progress_callback,
                                timestamp=True,
                            )
                            try:
                                best_base_ctx, base_status, base_info = choose_auto_ctx(base_model, args.llama_server, progress_callback=progress_callback)
                            except Exception as e:
                                best_base_ctx = None
                                base_status = "error"
                                base_info = {"reason": str(e)}
                            if best_base_ctx is not None:
                                if base_model.ctx_size != best_base_ctx:
                                    base_model.ctx_size = best_base_ctx
                                    updated_ctx += 1
                                base_model.auto_ctx_failed = False
                                base_model.auto_ctx_error = ""
                                apply_ctx_probe_metrics(base_model, base_info)
                                refresh_model_load_capabilities(base_model)
                                save_catalog(args.catalog, catalog)
                                paired_ctx = best_base_ctx
                            else:
                                # keep existing paired_ctx but record failure on master
                                base_err = base_info.get("reason") if isinstance(base_info, dict) else base_status
                                base_model.auto_ctx_failed = True
                                base_model.auto_ctx_error = str(base_err)
                                refresh_model_load_capabilities(base_model)
                                save_catalog(args.catalog, catalog)
                        if model.ctx_size != paired_ctx:
                            model.ctx_size = paired_ctx
                            updated_ctx += 1
                        model.auto_ctx_failed = False
                        model.auto_ctx_error = ""
                        _emit_message(
                            (
                                f"[{idx}/{total_models}] Probing paired speculative model {model.model_id} "
                                f"at master ctx {paired_ctx} (metrics only)."
                            ),
                            progress_callback,
                            timestamp=True,
                        )
                        probe_ok, probe_reason, probe_info = _probe_fixed_ctx_metrics_once(model, args.llama_server, paired_ctx)
                        if probe_ok:
                            apply_ctx_probe_metrics(model, probe_info)
                            status_label = "paired-metrics"
                            _emit_message(
                                f"{model.model_id}: paired metrics captured at ctx {paired_ctx}.",
                                progress_callback,
                                timestamp=True,
                            )
                        else:
                            clear_ctx_probe_metrics(model)
                            status_label = "paired-metrics-failed"
                            _emit_message(
                                (
                                    f"{model.model_id}: paired metrics probe failed at master ctx {paired_ctx} "
                                    f"({probe_reason})."
                                ),
                                progress_callback,
                                timestamp=True,
                            )
                        refresh_model_load_capabilities(model)
                        save_catalog(args.catalog, catalog)
                        auto_ctx_rows.append(
                            {
                                "MODEL": model.model_id,
                                "CFG_CTX": str(model.ctx_size),
                                "API_CTX": str(displayed_api_ctx(model, args)),
                                "CTX_GB": _format_ctx_probe_gb(model.ctx_probe_kv_gb),
                                "READ/S": _format_ctx_probe_rate(model.ctx_probe_read_s),
                                "TOKENS/S": _format_ctx_probe_rate(model.ctx_probe_tokens_s),
                                "TOTALS/S": _format_ctx_probe_rate(model.ctx_probe_totals_s),
                                "LATENCY": _format_ctx_probe_latency(model.ctx_probe_latency_ms),
                                "STATUS": status_label,
                            }
                        )
                        continue

                _emit_message(f"[{idx}/{total_models}] Probing {model.model_id}...", progress_callback, timestamp=True)
                best_ctx, status, info = choose_auto_ctx(model, args.llama_server, progress_callback=progress_callback)
                if best_ctx is None:
                    missing_ctx += 1
                    details = info if isinstance(info, dict) else {}
                    failure_reason = str(details.get("reason") or status)
                    model.auto_ctx_failed = True
                    model.auto_ctx_error = f"min-failed:{failure_reason}" if status == "min-failed" else failure_reason
                    clear_ctx_probe_metrics(model)
                    refresh_model_load_capabilities(model)
                    if status == "min-failed":
                        min_ctx = int(details.get("min_ctx") or model.ctx_size or DEFAULT_CTX_SIZE)
                        model.ctx_size = min_ctx
                        min_failed_models.append(model)
                        _emit_message(
                            f"{model.model_id}: failed even at minimum ctx {min_ctx} ({failure_reason}).",
                            progress_callback,
                            timestamp=True,
                        )
                    else:
                        _emit_message(
                            f"{model.model_id}: automatic ctx probing failed ({failure_reason}).",
                            progress_callback,
                            timestamp=True,
                        )
                    auto_ctx_rows.append({
                        "MODEL": model.model_id,
                        "CFG_CTX": _display_cfg_ctx(model),
                        "API_CTX": _display_api_ctx(model),
                        "CTX_GB": _format_ctx_probe_gb(None),
                        "READ/S": _format_ctx_probe_rate(None),
                        "TOKENS/S": _format_ctx_probe_rate(None),
                        "TOTALS/S": _format_ctx_probe_rate(None),
                        "LATENCY": _format_ctx_probe_latency(None),
                        "STATUS": status,
                    })
                    save_catalog(args.catalog, catalog)
                    continue
                if model.ctx_size != best_ctx:
                    model.ctx_size = best_ctx
                    updated_ctx += 1
                model.auto_ctx_failed = False
                model.auto_ctx_error = ""
                apply_ctx_probe_metrics(model, info)
                refresh_model_load_capabilities(model)
                save_catalog(args.catalog, catalog)
                _emit_message(f"{model.model_id}: selected cfg ctx {model.ctx_size}.", progress_callback, timestamp=True)
                auto_ctx_rows.append({
                    "MODEL": model.model_id,
                    "CFG_CTX": str(model.ctx_size),
                    "API_CTX": str(displayed_api_ctx(model, args)),
                    "CTX_GB": _format_ctx_probe_gb(model.ctx_probe_kv_gb),
                    "READ/S": _format_ctx_probe_rate(model.ctx_probe_read_s),
                    "TOKENS/S": _format_ctx_probe_rate(model.ctx_probe_tokens_s),
                    "TOTALS/S": _format_ctx_probe_rate(model.ctx_probe_totals_s),
                    "LATENCY": _format_ctx_probe_latency(model.ctx_probe_latency_ms),
                    "STATUS": "selected",
                })

            if min_failed_models:
                failed_ids = {model.model_id for model in min_failed_models}
                failed_labels = ", ".join(sorted(failed_ids))
                if _ask_confirmation(
                    (
                        "The following models did not load even at the minimum ctx and will remain unusable: "
                        f"{failed_labels}. Delete them now?"
                    ),
                    progress_callback=progress_callback,
                    default=True,
                ):
                    models_root = Path(getattr(args, "models_dir", DEFAULT_MODELS_DIR))
                    for failed_model in min_failed_models:
                        shared_repo = any(
                            item.repo_id == failed_model.repo_id and item.model_id not in failed_ids
                            for item in catalog
                        )
                        if shared_repo:
                            _emit_message(
                                (
                                    f"{failed_model.model_id}: keeping local files because another catalog entry "
                                    f"still uses {failed_model.repo_id}."
                                ),
                                progress_callback,
                            )
                            continue

                        repo_dir = models_root / failed_model.repo_id
                        if repo_dir.exists():
                            shutil.rmtree(repo_dir, ignore_errors=True)
                            _emit_message(f"{failed_model.model_id}: deleted local files under {repo_dir}.", progress_callback)
                        else:
                            _emit_message(f"{failed_model.model_id}: no local files found under {repo_dir}.", progress_callback)

                    catalog = [item for item in catalog if item.model_id not in failed_ids]
                    deleted_models = len(failed_ids)
                    for row in auto_ctx_rows:
                        if row.get("MODEL") in failed_ids:
                            row["CFG_CTX"] = "ERROR"
                            row["API_CTX"] = "ERROR"
                            row["STATUS"] = "deleted"
                    save_catalog(args.catalog, catalog)
                    _emit_message(
                        f"Removed {deleted_models} model(s) that failed the minimum ctx probe.",
                        progress_callback,
                    )
        except Exception:
            if probe_config_replaced:
                restore_catalog_config(args, catalog, progress_callback=progress_callback)
            raise
    elif ctx_override is not None:
        for model in target_models:
            if model.ctx_size != ctx_override:
                clear_ctx_probe_metrics(model)
            model.ctx_size = ctx_override
            model.auto_ctx_failed = False
            model.auto_ctx_error = ""
            refresh_model_load_capabilities(model)
        updated_ctx = len(target_models)
        missing_ctx = 0
        if len(target_models) == 1:
            _emit_message(f"Applied ctx override to {target_models[0].model_id}: {ctx_override}", progress_callback)
        else:
            _emit_message(f"Applied ctx override to all catalog models: {ctx_override}", progress_callback)
    else:
        if sync_gguf_ctx:
            previous_ctx_by_model = {model.model_id: model.ctx_size for model in target_models}
            updated_ctx, missing_ctx = sync_catalog_context_sizes(target_models)
            for model in target_models:
                if model.ctx_size != previous_ctx_by_model.get(model.model_id):
                    clear_ctx_probe_metrics(model)
                    refresh_model_load_capabilities(model)
        else:
            updated_ctx = 0
            missing_ctx = 0
    save_catalog(args.catalog, catalog)
    render_llamaswap_config(
        catalog,
        args.config,
        args.llama_server,
        args.start_port,
        resolve_idle_ttl(args),
        server_defaults=resolve_llama_server_defaults(args),
    )
    if auto_ctx:
        summary = f"Catalog ctx updated automatically: {updated_ctx} models changed, {missing_ctx} skipped."
        if deleted_models:
            summary = f"{summary} {deleted_models} deleted after failing the minimum ctx probe."
        _emit_message(summary, progress_callback)
        if auto_ctx_rows:
            _emit_message("Auto-ctx summary:\n" + render_auto_ctx_summary_table(auto_ctx_rows), progress_callback)
    elif ctx_override is not None:
        _emit_message(
            f"Catalog ctx updated manually: {updated_ctx} models set to {ctx_override}.",
            progress_callback,
        )
    else:
        if sync_gguf_ctx:
            _emit_message(
                f"Catalog synchronized from GGUF metadata: {updated_ctx} ctx values updated, {missing_ctx} unavailable.",
                progress_callback,
            )
        else:
            _emit_message(
                "Catalog context values preserved (CFG_CTX unchanged); regenerated llama-swap config from catalog.",
                progress_callback,
            )
    _emit_message("Config updated from catalog. Waiting for llama-swap --watch-config...", progress_callback)
    time.sleep(3.0)
    try:
        host = _normalize_client_host(args.public_host)
        r = requests.get(f"http://{host}:{args.public_port}/v1/models", timeout=2)
        if r.status_code == 200:
            _emit_message(
                f"Public API reachable on http://{host}:{args.public_port} ({len(r.json().get('data', []))} published models).",
                progress_callback,
            )
        else:
            _emit_message(
                f"Config updated, but public API responded with HTTP {r.status_code}.",
                progress_callback,
            )
    except Exception as e:
        _emit_message(
            f"Config updated, but could not verify public API on http://{_normalize_client_host(args.public_host)}:{args.public_port} ({e.__class__.__name__}).",
            progress_callback,
        )
    return "updated"

def _http_error_snippet(exc: requests.HTTPError) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    try:
        text = (response.text or "").strip()
    except Exception:
        text = ""
    if len(text) > 500:
        text = text[:500] + "..."
    return text or str(exc)


def warmup_model(model_id, host, port, timeout=600):
    host = _normalize_client_host(host)
    url = f"http://{host}:{port}/v1/chat/completions"
    print(f"\033[35;1mWarming model {model_id} before opening the chat...\033[0m")
    loader = LoadingBar("\033[35;1mLoading model:\033[0m ")
    loader.start()
    deadline = time.time() + timeout
    last_error = None
    try:
        while time.time() < deadline:
            try:
                r = requests.post(
                    url,
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": "."}],
                        "stream": False,
                        "max_tokens": 1,
                        "temperature": 0,
                    },
                    timeout=(10, 30),
                )
                r.raise_for_status()
                return
            except requests.HTTPError as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code not in {502, 503, 504}:
                    raise
                last_error = exc
            except Exception as exc:
                last_error = exc
            time.sleep(1.5)
        if last_error is not None:
            raise last_error
    finally:
        loader.stop()


def _print_warmup_failure(model_id: str, host: str, port: int, exc: Exception) -> None:
    print(f"\n\033[31;1mCould not warm model {model_id} through http://{host}:{port}.\033[0m")
    if isinstance(exc, requests.HTTPError):
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        print(f"HTTP status: {status_code or 'unknown'}")
        snippet = _http_error_snippet(exc)
        if snippet:
            print(f"Response: {snippet}")
        if status_code in {502, 503, 504}:
            print("The proxy was reachable but the backend llama-server failed to load or respond.")
    else:
        print(f"{type(exc).__name__}: {exc}")
    print(f"Recent request log: {DEFAULT_REQUESTS_LOG_PATH}")
    print("Run for details:")
    print("  llamacpp-superserver requests-log --lines 80")
    print(f"  sudo journalctl -u {SWAP_SERVICE_NAME} -u {MANAGER_SERVICE_NAME} -n 120 --no-pager")


def flush_stdin_buffer() -> None:
    try:
        if sys.stdin.isatty():
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass

def start_chat(model_id, host, port):
    host = _normalize_client_host(host)
    try:
        warmup_model(model_id, host, port)
    except Exception as exc:
        _print_warmup_failure(model_id, host, port, exc)
        return 1
    flush_stdin_buffer()
    print(f"\n\033[35;1m--- Chat: {model_id} ---\033[0m\nCommands: /exit, /clear, /help\n")
    url = f"http://{host}:{port}/v1/chat/completions"
    msgs = []
    while True:
        try:
            p = input("\033[32;1m>>> \033[0m").strip()
            if not p: continue
            if p.lower() in ("/exit", "/quit", "/bye"): break
            if p.lower() == "/clear": msgs = []; print("History cleared."); continue
            
            msgs.append({"role": "user", "content": p})
            spinner = Spinner("\033[35;1massistant:\033[0m ")
            spinner.start()
            
            res = ""; first_token = True
            response = None
            interrupted = False
            previous_sigint = signal.getsignal(signal.SIGINT)

            def _handle_sigint(signum, frame):
                nonlocal interrupted, response
                interrupted = True
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        pass

            signal.signal(signal.SIGINT, _handle_sigint)
            try:
                response = requests.post(
                    url,
                    json={"model": model_id, "messages": msgs, "stream": True, "temperature": 0.7},
                    stream=True,
                    timeout=(10, 10),
                )
                response.raise_for_status()
                for line in response.iter_lines(chunk_size=1, decode_unicode=False):
                    if interrupted:
                        break
                    if not line:
                        continue
                    decoded = line.decode("utf-8", errors="ignore").strip()
                    if not decoded.startswith("data: "):
                        continue
                    if decoded[6:].strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(decoded[6:].strip())
                        txt = chunk["choices"][0].get("delta", {}).get("content", "")
                        if txt:
                            if first_token:
                                spinner.stop()
                                first_token = False
                            sys.stdout.write(txt)
                            sys.stdout.flush()
                            res += txt
                    except Exception:
                        pass
                if first_token: spinner.stop()
                if interrupted:
                    sys.stdout.write("\n\033[33mGeneration interrupted.\033[0m\n\n")
                    sys.stdout.flush()
                    msgs.pop()
                else:
                    sys.stdout.write("\n\n")
                    msgs.append({"role": "assistant", "content": res})
            except Exception as e:
                spinner.stop()
                print(f"\n\033[31mError: {e}\033[0m")
                msgs.pop()
            finally:
                signal.signal(signal.SIGINT, previous_sigint)
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        pass
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break
    return 0

def daemon_mode(args):
    """Background manager listening on Unix socket."""
    _prepare_manager_socket_path(SOCKET_PATH)
    try:
        os.makedirs(os.path.dirname(SOCKET_PATH), exist_ok=True)
    except Exception as exc:
        raise RuntimeError(f"Could not create manager socket directory for {SOCKET_PATH}: {exc}") from exc
    ctx_metadata_server = start_ctx_metadata_server(args)
    unload_guard_thread = start_unexpected_unload_guard(args)
    
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(SOCKET_PATH)
    except Exception as exc:
        raise RuntimeError(f"Could not bind manager socket {SOCKET_PATH}: {exc}") from exc
    os.chmod(SOCKET_PATH, 0o666)
    server.listen(5)
    
    user_name = pwd.getpwuid(os.getuid()).pw_name
    print(f"[*] Manager listening on {SOCKET_PATH} (User: {user_name})")
    
    while True:
        conn, _ = server.accept()
        def handle(client_conn=conn):
            sock_in = None
            try:
                sock_in = client_conn.makefile("r", encoding="utf-8")
                data = sock_in.readline()
                if not data: return
                req = json.loads(data)

                def send_event(event):
                    client_conn.sendall((json.dumps(event) + "\n").encode())
                    if event.get("type") == "question":
                        raw = sock_in.readline()
                        if not raw:
                            return ""
                        reply = json.loads(raw)
                        return reply.get("answer", "")

                if req["command"] == "add":
                    mock_args = argparse.Namespace(**req["args"])
                    mock_args.catalog = Path(mock_args.catalog)
                    mock_args.config = Path(mock_args.config)
                    mock_args.models_dir = Path(mock_args.models_dir)
                    mock_args.llama_server = Path(mock_args.llama_server)
                    mock_args.server_config = Path(getattr(mock_args, "server_config", DEFAULT_SERVER_CONFIG_PATH))
                    model_id = ensure_model_available(mock_args, progress_callback=send_event)
                    send_event({"type": "done", "model_id": model_id})
                elif req["command"] == "list":
                    mock_args = argparse.Namespace(**req["args"])
                    mock_args.catalog = Path(mock_args.catalog)
                    mock_args.server_config = Path(getattr(mock_args, "server_config", DEFAULT_SERVER_CONFIG_PATH))
                    table = render_models_table(load_catalog(mock_args.catalog, mock_args.server_config), mock_args.public_host, mock_args.public_port)
                    send_event({"type": "done", "result": table})
                elif req["command"] == "update":
                    mock_args = argparse.Namespace(**req["args"])
                    mock_args.catalog = Path(mock_args.catalog)
                    mock_args.config = Path(mock_args.config)
                    mock_args.llama_server = Path(mock_args.llama_server)
                    mock_args.server_config = Path(getattr(mock_args, "server_config", DEFAULT_SERVER_CONFIG_PATH))
                    result = update_config(mock_args, progress_callback=send_event)
                    send_event({"type": "done", "result": result})
                elif req["command"] == "remove":
                    mock_args = argparse.Namespace(**req["args"])
                    mock_args.catalog = Path(mock_args.catalog)
                    mock_args.config = Path(mock_args.config)
                    mock_args.models_dir = Path(mock_args.models_dir)
                    mock_args.llama_server = Path(mock_args.llama_server)
                    mock_args.server_config = Path(getattr(mock_args, "server_config", DEFAULT_SERVER_CONFIG_PATH))
                    model_id = remove_model(mock_args, progress_callback=send_event)
                    send_event({"type": "done", "model_id": model_id})
                elif req["command"] == "unload":
                    mock_args = argparse.Namespace(**req["args"])
                    mock_args.catalog = Path(mock_args.catalog)
                    mock_args.config = Path(mock_args.config)
                    mock_args.llama_server = Path(mock_args.llama_server)
                    mock_args.server_config = Path(getattr(mock_args, "server_config", DEFAULT_SERVER_CONFIG_PATH))
                    result = unload_models(mock_args)
                    send_event({"type": "done", "result": result})
                elif req["command"] == "auto-performance":
                    from llamacpp_stack.auto_perf_runner import prepare_auto_perf_daemon_handler
                    prepare_auto_perf_daemon_handler(req, send_event, sock_in)
            except Exception as e:
                try:
                    conn.sendall((json.dumps({"type": "error", "message": str(e)}) + "\n").encode())
                except Exception:
                    pass
            finally:
                if sock_in is not None:
                    try:
                        sock_in.close()
                    except Exception:
                        pass
                client_conn.close()
        threading.Thread(target=handle).start()


def _prepare_manager_socket_path(socket_path: str) -> None:
    if not os.path.exists(socket_path):
        return

    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.5)
        probe.connect(socket_path)
    except OSError as exc:
        if exc.errno not in {errno.ECONNREFUSED, errno.ENOENT}:
            raise RuntimeError(f"Could not probe existing manager socket {socket_path}: {exc}") from exc
        try:
            os.remove(socket_path)
        except Exception as remove_exc:
            raise RuntimeError(f"Could not remove stale manager socket {socket_path}: {remove_exc}") from remove_exc
    else:
        raise RuntimeError(
            f"Manager socket {socket_path} is already in use by a running manager instance. "
            f"Stop it first (for example: sudo systemctl stop {MANAGER_SERVICE_NAME})."
        )
    finally:
        try:
            probe.close()
        except Exception:
            pass

def get_public_endpoint_status(host=DEFAULT_PUBLIC_HOST, port=DEFAULT_PUBLIC_PORT):
    base_url = f"http://{host}:{port}"
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::", "[::]"} else host
    models_url = f"http://{probe_host}:{port}/v1/models"
    via = "" if probe_host == host else f" via {probe_host}"
    try:
        r = requests.get(models_url, timeout=1.5)
        if r.status_code == 200:
            data = r.json().get("data", [])
            return f"reachable on {base_url}{via} ({len(data)} models listed)"
        return f"responding on {base_url}{via} with HTTP {r.status_code}"
    except Exception as e:
        return f"not reachable on {base_url}{via} ({e.__class__.__name__})"


def get_api_endpoint_status(host=DEFAULT_PUBLIC_HOST, port=None):
    if port is None:
        port = resolve_api_port()
    base_url = f"http://{host}:{port}"
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::", "[::]"} else host
    models_url = f"http://{probe_host}:{port}/v1/models"
    via = "" if probe_host == host else f" via {probe_host}"
    try:
        r = requests.get(models_url, timeout=1.5)
        if r.status_code == 200:
            data = r.json().get("data", [])
            return f"reachable on {base_url}{via} ({len(data)} catalog models listed)"
        return f"responding on {base_url}{via} with HTTP {r.status_code}"
    except Exception as e:
        return f"not reachable on {base_url}{via} ({e.__class__.__name__})"

def read_install_manifest():
    manifest_path = DEFAULT_CONFIG_PATH.parent / "install-manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def get_superserver_version() -> str:
    forced = os.environ.get("LLAMACPP_SUPERSERVER_VERSION", "").strip()
    if forced:
        return forced
    try:
        return version("llamacpp-superserver")
    except PackageNotFoundError:
        pass
    except Exception:
        pass
    # Fallback for local editable runs.
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        for raw in pyproject_path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"\s*version\s*=\s*\"([^\"]+)\"", raw)
            if match:
                return match.group(1)
    except Exception:
        pass
    return "0.0.0"


def render_superserver_banner() -> str:
    divider = "=" * 72
    return (
        f"{divider}\n"
        " _ _                                _\n"
        "| | | __ _ _ __ ___   __ _  ___ _ __| |_ _ __\n"
        "| | |/ _` | '_ ` _ \\ / _` |/ __| '__| __| '_ \\\n"
        "| | | (_| | | | | | | (_| | (__| |  | |_| |_) |\n"
        "|_|_|\\__,_|_| |_| |_|\\__,_|\\___|_|   \\__| .__/\n"
        "                                           |_|   \n"
        "              llama.cpp  SuperServer\n"
        f"         llamacpp-superserver v{get_superserver_version()}\n"
        f"{divider}"
    )


def _install_root_from_llama_server(llama_server_path: Path) -> Path:
    if "llama.cpp/build/bin" in str(llama_server_path):
        return llama_server_path.parent.parent.parent
    return llama_server_path.parent


def build_info_text(args = None) -> str:
    public_host = getattr(args, "public_host", DEFAULT_PUBLIC_HOST)
    public_port = int(getattr(args, "public_port", DEFAULT_PUBLIC_PORT))
    api_port = resolve_api_port(args)
    ui_url = f"http://{public_host}:{public_port}"
    api_url = f"http://{public_host}:{api_port}"

    manifest = read_install_manifest()
    llama_cpp_tag = str(manifest.get("llama_cpp_tag") or "unknown")
    llamaswap_tag = str(manifest.get("llamaswap_tag") or "unknown")

    install_mode = infer_install_mode()
    start_cmd, status_cmd, restart_cmd = service_commands_for_mode(install_mode)

    models_dir = Path(getattr(args, "models_dir", DEFAULT_MODELS_DIR))
    config_path = Path(getattr(args, "config", DEFAULT_CONFIG_PATH))
    catalog_path = Path(getattr(args, "catalog", DEFAULT_CATALOG_PATH))
    server_config_path = _args_server_config_path(args)
    llama_server_path = Path(getattr(args, "llama_server", DEFAULT_LLAMA_SERVER))
    install_root = _install_root_from_llama_server(llama_server_path)
    # Templates directory: can be overridden with LLAMACPP_TEMPLATES_DIR
    templates_env = os.environ.get("LLAMACPP_TEMPLATES_DIR")
    if templates_env:
        templates_dir = Path(templates_env).expanduser()
    else:
        templates_dir = Path(server_config_path).expanduser().parent / "templates"

    return (
        "Default endpoints:\n"
        f"  llama-swap UI/backend: {ui_url}\n"
        f"  Superserver API:       {api_url}\n"
        "Installed versions:\n"
        f"  llama.cpp:           {llama_cpp_tag}\n"
        f"  llama-swap:          {llamaswap_tag}\n"
        "Runtime info:\n"
        f"  Install root:        {install_root}\n"
        f"  Models dir:          {models_dir}\n"
        f"  llama-swap config:   {config_path}\n"
        f"  Catalog:             {catalog_path}\n"
        f"  App config:          {server_config_path}\n"
        f"  Templates dir:       {templates_dir}\n"
        f"  llama-server binary: {llama_server_path}\n"
        f"  Requests log:        {DEFAULT_REQUESTS_LOG_PATH}\n"
        f"  UI activity:         {ui_url}/ui/#/activity\n"
        f"  Idle TTL:            {resolve_idle_ttl(args)}s\n"
        "Service management:\n"
        f"  Install mode:        {install_mode}\n"
        f"  Start services:      {start_cmd}\n"
        f"  Status:              {status_cmd}\n"
        f"  Restart:             {restart_cmd}\n"
        "Config knobs:\n"
        f"  Global llama-server defaults: {server_config_path} -> llama_server_defaults\n"
        f"  Per-model overrides:          {catalog_path} -> server_overrides\n"
        f"  API_CTX factor:               {server_config_path} -> api_ctx_factor (default {DEFAULT_API_CTX_FACTOR})\n"
        "  Default llama-server flags:\n"
        "    --keep 512, --mirostat 2, --mirostat-ent 4.5, --mirostat-lr 0.1\n"
        "    --cache-type-k q8_0, --cache-type-v q8_0\n"
        f"    (Change these in {server_config_path}['llama_server_defaults'])\n"
        "  Main folders: install root, models dir, state/config paths above.\n"
        f"  API status:          {get_api_endpoint_status(public_host, api_port)}\n"
        f"  UI status:           {get_public_endpoint_status(public_host, public_port)}"
    )


def show_info(args):
    print(render_superserver_banner())
    print()
    print(build_info_text(args))
    return 0


def build_help_epilog():
    return (
        "Command guide:\n"
        "  add [repo ...] [-hf HF ...] [--auto|--skip-ctx]\n"
        "    Register/download one or more models into the catalog.\n"
        f"    Example: {CLI_COMMAND} add -hf Qwen/Qwen2.5-32B-Instruct-GGUF:Q4_K_M\n"
        "  run [repo|-hf HF] [--auto] [--no-chat]\n"
        "    Ensure model exists and start chat (or only preload with --no-chat).\n"
        f"    Example: {CLI_COMMAND} run -hf Qwen/Qwen2.5-32B-Instruct-GGUF:Q4_K_M --auto\n"
        f"    Pair example: {CLI_COMMAND} run -hf org/master:Q4 --speculative -hf org/draft:IQ1\n"
        "  remove [repo ...|-hf HF ...] [--keep-files]\n"
        "    Remove models from config/catalog and delete files by default.\n"
        f"    Example: {CLI_COMMAND} remove qwen2.5-32b-instruct-q4_k_m\n"
        "  rm [repo ...|-hf HF ...] [--keep-files]\n"
        "    Alias of remove (also deletes model files by default).\n"
        f"    Example: {CLI_COMMAND} rm qwen2.5-32b-instruct-q4_k_m\n"
        "  update [repo ...|-hf HF ...] [--auto|--preserve-ctx|--sync-gguf-ctx]\n"
        "    Refresh config and optionally re-probe ctx.\n"
        f"    Example: {CLI_COMMAND} update qwen2.5-32b-instruct-q4_k_m --auto\n"
        "  validate [repo|-hf HF] [--auto]\n"
        "    Probe/validate a model and context settings before serving.\n"
        f"    Example: {CLI_COMMAND} validate -hf Qwen/Qwen2.5-32B-Instruct-GGUF:Q4_K_M --auto\n"
        "  daemon\n"
        "    Start the manager daemon loop (socket/API lifecycle automation).\n"
        f"    Example: {CLI_COMMAND} daemon\n"
        "  list\n"
        "    Show configured models (ctx, memory estimate, speed hints).\n"
        f"    Example: {CLI_COMMAND} list\n"
        "  ps\n"
        "    Alias view for model table/state (same rendering as list).\n"
        f"    Example: {CLI_COMMAND} ps\n"
        "  requests [-n LINES]\n"
        "    Show recent API request log entries.\n"
        f"    Example: {CLI_COMMAND} requests -n 50 (or {CLI_COMMAND} requests -n 1)\n"
        "  info\n"
        "    Show endpoints, runtime paths, versions, service commands and status.\n"
        f"    Example: {CLI_COMMAND} info\n"
        f"  Help     Show options for any command.\n"
        f"    Example: {CLI_COMMAND} <command> -h\n"
        f"For endpoints/runtime/service/config details run: {CLI_COMMAND} info"
    )

class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def _detect_requested_subcommand(argv: list[str], available: set[str]) -> str | None:
    for token in argv:
        if token in available:
            return token
    return None


def build_cli_parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    parser = argparse.ArgumentParser(
        prog=CLI_COMMAND,
        description="Manage GGUF models for llama-swap + llama-server.",
        epilog=build_help_epilog(),
        formatter_class=HelpFormatter,
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--server-config", type=Path, default=DEFAULT_SERVER_CONFIG_PATH)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--llama-server", type=Path, default=DEFAULT_LLAMA_SERVER)
    parser.add_argument("--service", default=DEFAULT_SERVICE_NAME)
    parser.add_argument("--start-port", type=int, default=DEFAULT_START_PORT)
    parser.add_argument("--public-host", default=DEFAULT_PUBLIC_HOST)
    parser.add_argument("--public-port", type=int, default=DEFAULT_PUBLIC_PORT)
    parser.add_argument("--api-port", type=int, default=None)
    parser.add_argument("--idle-ttl", type=int, default=None)
    parser.add_argument("--api-ctx-factor", type=float, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    subparsers: dict[str, argparse.ArgumentParser] = {}
    
    p_add = sub.add_parser(
        "add",
        help="Register/download model(s)",
        description="Register/download one or more GGUF models into the catalog.",
    )
    subparsers["add"] = p_add
    p_add.set_defaults(func=add_models)
    p_add.add_argument("repo", nargs="*", help="HF repo[:QUANT] (accepts a list)")
    p_add.add_argument("-hf", "--hf", nargs="+", help="HF repo[:QUANT] list")
    p_add.add_argument("--file")
    p_add.add_argument("--model-id")
    p_add.add_argument("--ctx-size", default=DEFAULT_CTX_SIZE)
    p_add.add_argument(
        "--auto",
        "-auto",
        "--auto-ctx",
        "-auto-ctx",
        dest="auto_ctx",
        action="store_true",
        help="Force a fresh automatic ctx probe even if a fallback was already saved",
    )
    p_add.add_argument("--skip-ctx", action="store_true", help="Skip automatic ctx tuning and keep the default ctx size")
    p_add.add_argument("--n-gpu-layers", default=DEFAULT_N_GPU_LAYERS)
    p_add.add_argument("--tensor-split", default=default_tensor_split())
    p_add.add_argument("--host", default="127.0.0.1")
    p_add.add_argument("--no-jinja", action="store_true")
    p_add.add_argument("--force", action="store_true")
    p_add.add_argument("--hf-token")
    p_add.add_argument("--description")
    p_add.add_argument("--speculative", action="store_true", help="Create/download a speculative draft variant (speculative-<base_id>)")
    p_add.add_argument("--float16", "--f16", action="store_true", help="Use float16 precision (vLLM)")
    p_add.add_argument("--bfloat16", "--bf16", action="store_true", help="Use bfloat16 precision (vLLM)")
    p_add.add_argument("--float32", "--f32", action="store_true", help="Use float32 precision (vLLM)")
    p_add.add_argument("--defer-publish", action="store_true", help="Do not try to publish the model to the proxy immediately")
    
    p_run = sub.add_parser(
        "run",
        help="Run chat with a model",
        description="Ensure model availability and run chat unless --no-chat is used.",
    )
    subparsers["run"] = p_run
    p_run.set_defaults(func=run_command)
    p_run.add_argument("--no-chat", action="store_true")
    p_run.add_argument("-ctx", "--ctx", dest="ctx_override", type=int, help="Override ctx size for this run")
    p_run.add_argument("--speculative", action="store_true", help="Run against a speculative draft variant (speculative-<base_id>). With two -hf values, first is base/master and second is draft.")
    p_run.add_argument("--float16", "--f16", action="store_true", help="Use float16 precision (vLLM)")
    p_run.add_argument("--bfloat16", "--bf16", action="store_true", help="Use bfloat16 precision (vLLM)")
    p_run.add_argument("--float32", "--f32", action="store_true", help="Use float32 precision (vLLM)")
    p_run.add_argument("--gpu-memory-utilization", type=float, help="GPU memory utilization (0.0 to 1.0, vLLM)")

    p_remove = sub.add_parser(
        "remove",
        aliases=["rm"],
        help="Remove model(s) from catalog",
        description="Remove model entries from catalog/config; rm alias deletes files by default.",
    )
    subparsers["remove"] = p_remove
    subparsers["rm"] = p_remove
    p_remove.set_defaults(func=remove_models)
    p_remove.add_argument("repo", nargs="*", help="Model id or HF repo[:QUANT] (accepts a list)")
    p_remove.add_argument("-hf", "--hf", nargs="+", help="HF repo list")
    p_remove.add_argument("--file")
    p_remove.add_argument("--model-id")
    p_remove.add_argument(
        "--delete-files",
        dest="delete_files",
        action="store_true",
        default=True,
        help="Delete local model files from disk (default).",
    )
    p_remove.add_argument(
        "--keep-files",
        dest="delete_files",
        action="store_false",
        help="Keep local model files on disk.",
    )

    p_unload = sub.add_parser(
        "unload",
        help="Unload model(s) from llama-swap",
        description="Rewrite llama-swap config to unload one model or all currently published models without deleting files.",
    )
    subparsers["unload"] = p_unload
    p_unload.set_defaults(func=unload_models)
    p_unload.add_argument("repo", nargs="*", help="Model id, HF repo[:QUANT], or 'all' to unload everything")
    p_unload.add_argument("-hf", "--hf", nargs="+", help="HF repo list")
    p_unload.add_argument("--file")
    p_unload.add_argument("--model-id")

    p_update = sub.add_parser(
        "update",
        help="Refresh model configuration",
        description="Refresh model configuration and optionally re-probe context limits.",
    )
    subparsers["update"] = p_update
    p_update.set_defaults(func=update_models)
    p_update.add_argument("repo", nargs="*", help="Model id or HF repo[:QUANT] (accepts a list)")
    p_update.add_argument("-hf", "--hf", nargs="+", help="HF repo list")
    p_update.add_argument("--file")
    p_update.add_argument("--model-id")
    p_update.add_argument("-ctx", "--ctx", dest="ctx_override", type=int, help="Set ctx size for the selected model(s)")
    p_update.add_argument(
        "--auto",
        "-auto",
        "--auto-ctx",
        "-auto-ctx",
        dest="auto_ctx",
        action="store_true",
        help="Probe and set a practical ctx size per model automatically",
    )

    p_refresh = sub.add_parser(
        "refresh-templates",
        help="Scan templates folder and add template-backed model entries to catalog",
        description="Detect chat template files and create duplicate catalog entries (model_id+template) so models can be launched with or without templates.",
    )
    subparsers["refresh-templates"] = p_refresh
    p_refresh.set_defaults(func=lambda args: refresh_templates(args))
    
    p_remove_templates = sub.add_parser(
        "remove-templates",
        help="Remove all template-backed model entries from catalog",
        description="Delete all model entries whose ID ends with '+template' from the catalog.",
    )
    subparsers["remove-templates"] = p_remove_templates
    p_remove_templates.set_defaults(func=lambda args: remove_templates(args))
    p_update.add_argument(
        "--preserve-ctx",
        action="store_true",
        help="Keep existing CFG_CTX values while regenerating config",
    )
    p_update.add_argument("--speculative", action="store_true", help="Update/probe a speculative draft variant (speculative-<base_id>)")
    p_update.add_argument(
        "--sync-gguf-ctx",
        action="store_true",
        help="Overwrite CFG_CTX values from GGUF metadata",
    )
    p_update.add_argument("--defer-publish", action="store_true", help="Do not try to publish the model to the proxy immediately")
    p_validate = sub.add_parser(
        "validate",
        help="Probe/validate model ctx",
        description="Probe and validate context behavior before serving a model.",
    )
    subparsers["validate"] = p_validate
    p_validate.set_defaults(func=validate_model)
    p_validate.add_argument("repo", nargs="?", help="HF repo[:QUANT] or installed model id")
    p_validate.add_argument("-hf", "--hf", help="HF repo")
    p_validate.add_argument("--file")
    p_validate.add_argument("--model-id")
    p_validate.add_argument("-ctx", "--ctx", dest="ctx_override", type=int, help="Validate using this ctx directly")
    p_validate.add_argument("--ctx-size", default=DEFAULT_CTX_SIZE, help="Fallback ctx before auto-fit for temporary models")
    p_validate.add_argument(
        "--auto",
        "-auto",
        "--auto-ctx",
        "-auto-ctx",
        dest="auto_ctx",
        action="store_true",
        help="Force auto-fit before validating",
    )
    p_validate.add_argument("--n-gpu-layers", default=DEFAULT_N_GPU_LAYERS)
    p_validate.add_argument("--tensor-split", default=default_tensor_split())
    p_validate.add_argument("--host", default="127.0.0.1")
    p_validate.add_argument("--no-jinja", action="store_true")
    p_validate.add_argument("--hf-token")
    p_validate.add_argument("--description")
    p_validate.add_argument("--speculative", action="store_true", help="Validate a speculative draft variant (speculative-<base_id>)")
    
    p_daemon = sub.add_parser(
        "daemon",
        help="Start manager daemon",
        description="Start the manager daemon loop for background lifecycle automation.",
    )
    subparsers["daemon"] = p_daemon
    p_daemon.set_defaults(func=daemon_mode)
    
    p_debug = sub.add_parser(
        "debug",
        help="Start debug mode",
        description="Start the debug API in the foreground. The session is only available while this command is running.",
    )
    subparsers["debug"] = p_debug
    p_debug.set_defaults(func=debug_mode)
    
    for p in [p_run]:
        p.add_argument("repo", nargs="?", help="HF repo[:QUANT]")
        # Allow specifying multiple HF entries. Use append+nargs so the
        # user can pass either a space-separated group or repeat the flag.
        p.add_argument("-hf", "--hf", nargs="+", action="append", help="HF repo")
        p.add_argument("--file")
        p.add_argument("--model-id")
        p.add_argument("--ctx-size", default=DEFAULT_CTX_SIZE)
        p.add_argument(
            "--auto",
            "-auto",
            "--auto-ctx",
            "-auto-ctx",
            dest="auto_ctx",
            action="store_true",
            help="Force a fresh automatic ctx probe even if a fallback was already saved",
        )
        p.add_argument("--skip-ctx", action="store_true", help="Skip automatic ctx tuning and keep the default ctx size")
        p.add_argument("--n-gpu-layers", default=DEFAULT_N_GPU_LAYERS)
        p.add_argument("--tensor-split", default=default_tensor_split())
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--no-jinja", action="store_true")
        p.add_argument("--force", action="store_true")
        p.add_argument("--hf-token")
        p.add_argument("--description")

    p_auto_perf = sub.add_parser(
        "auto-performance",
        help="Run auto-tuner to find best performance config",
        description="Uses Optuna to find the best hardware configuration for a given model.",
    )
    subparsers["auto-performance"] = p_auto_perf
    
    def run_auto_perf_lazy(args):
        from llamacpp_stack.auto_perf_runner import run_auto_perf_command
        return run_auto_perf_command(args)
        
    p_auto_perf.set_defaults(func=run_auto_perf_lazy)
    p_auto_perf.add_argument("repo", nargs="?", help="Model id or HF repo[:QUANT]")
    p_auto_perf.add_argument("-hf", "--hf", help="HF repo")
    p_auto_perf.add_argument("--file")
    p_auto_perf.add_argument("--model-id")
    p_auto_perf.add_argument("--mock", action="store_true", help="Run with mock metrics without starting the real server")
    p_auto_perf.add_argument("--server-api", action="store_true", help="Benchmark server-side concurrent request handling instead of a single raw completion")
    p_auto_perf.add_argument("--load-concurrency", type=int, default=1, help="Concurrent requests to issue during server-api benchmarking")
    p_auto_perf.add_argument("--load-requests", type=int, default=1, help="Total requests to issue during server-api benchmarking")
    p_auto_perf.add_argument("--trials-per-phase", type=int, default=None, help="Number of trials to execute per optimization phase (default: auto-calculated based on GPU count)")
    p_auto_perf.add_argument("--unattended", action="store_true", help="Non-interactive tuning: answer Yes to tuning/refresh/phase prompts, but keep final catalog overwrite as No")
    p_auto_perf.add_argument("--assume-no", action="store_true", help="Non-interactive mode: answer 'No' to all follow-up prompts")
    p_auto_perf.add_argument("--no-prompt", action="store_true", help="Alias of --assume-no")

    p_list = sub.add_parser(
        "list",
        help="List configured models",
        description="Show configured models with runtime/ctx summary.",
    )
    subparsers["list"] = p_list
    p_list.set_defaults(func=list_models)
    p_ps = sub.add_parser(
        "ps",
        help="Alias of list",
        description="Alias of list for compatibility with process-style listing.",
    )
    subparsers["ps"] = p_ps
    p_ps.set_defaults(func=list_models)
    p_ps.set_defaults(func=list_models)
    p_requests = sub.add_parser(
        "requests",
        help="Show recent request logs",
        description="Show recent API request log lines.",
    )
    subparsers["requests"] = p_requests
    p_requests.add_argument("-n", "--lines", type=int, default=50)
    p_requests.set_defaults(func=show_request_log)

    p_info = sub.add_parser(
        "info",
        help="Show runtime/system information",
        description="Show endpoints, versions, runtime paths, service commands and config knobs.",
    )
    subparsers["info"] = p_info
    p_info.set_defaults(func=show_info)

    return parser, subparsers


def parse_cli_args(
    parser: argparse.ArgumentParser,
    subparsers: dict[str, argparse.ArgumentParser],
    argv: list[str] | None = None,
) -> argparse.Namespace:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if any(token in {"-info", "--info"} for token in argv_list):
        parser.error("Use the 'info' subcommand without dashes: llamacpp-superserver info")
    try:
        return parser.parse_args(argv_list)
    except SystemExit as exc:
        if exc.code == 2 and subparsers:
            command = _detect_requested_subcommand(argv_list, set(subparsers.keys()))
            if command:
                print(f"\nOptions for '{command}':", file=sys.stderr)
                subparsers[command].print_help(sys.stderr)
        raise


def main(argv: list[str] | None = None):
    parser, subparsers = build_cli_parser()
    args = parse_cli_args(parser, subparsers, argv=argv)
    persist_server_config(args)
    return args.func(args)

if __name__ == "__main__":
    try: sys.exit(main() or 0)
    except Exception as e: print(f"Error: {e}"); sys.exit(1)
