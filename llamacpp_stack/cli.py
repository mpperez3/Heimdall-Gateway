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
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

# Dependencies
try:
    import yaml
    import requests
    from huggingface_hub import HfApi, hf_hub_download
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
DEFAULT_SERVICE_NAME = os.environ.get("LLAMACPP_SERVICE_NAME", "llamaswap")
CLI_COMMAND = "llamacpp-superserver"
LEGACY_CLI_COMMAND = "llamacpp-server"
MANAGER_SERVICE_NAME = "llamacpp-superserver-manager"
SWAP_SERVICE_NAME = "llamacpp-superserver-swap"
DEFAULT_LLAMA_SERVER = _env_path("LLAMA_SERVER_BIN", "/opt/llamacpp-superserver/llama.cpp/build/bin/llama-server")

DEFAULT_CTX_SIZE = 8192
DEFAULT_N_GPU_LAYERS = 999
DEFAULT_IDLE_TTL = int(os.environ.get("LLAMACPP_IDLE_TTL", os.environ.get("LLAMACPP_DEFAULT_TTL", "300")))


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
CATALOG_CACHE: dict[str, tuple[int, int, list["ManagedModel"]]] = {}

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
    aliases: list[str] = field(default_factory=list)
    ctx_size: int = DEFAULT_CTX_SIZE
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS
    tensor_split: str = DEFAULT_TENSOR_SPLIT
    host: str = "127.0.0.1"
    jinja: bool = True
    ttl: int = DEFAULT_IDLE_TTL
    description: str = ""
    auto_ctx_failed: bool = False
    auto_ctx_error: str = ""
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


def normalize_server_overrides(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, object] = {}
    for raw_key, raw_val in value.items():
        key = str(raw_key).strip().lower().replace("-", "_")
        if not key:
            continue
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
        if key in {"ctx_size", "n_gpu_layers", "batch_size", "ubatch_size", "threads", "threads_batch", "fit_target"}:
            try:
                normalized[key] = int(raw_val)
            except (TypeError, ValueError):
                continue
            continue
        if key == "flash_attn":
            bool_val = _normalize_bool_flag(raw_val)
            if bool_val is not None:
                normalized[key] = bool_val
            continue
        if key == "tensor_split":
            normalized[key] = normalize_tensor_split(str(raw_val))
            continue
        if key in {"split_mode", "numa", "cache_type_k", "cache_type_v", "host"}:
            normalized[key] = str(raw_val).strip()
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


def load_catalog(path: Path) -> list[ManagedModel]:
    items, _ = load_catalog_with_diagnostics(path)
    return items


def load_catalog_with_diagnostics(path: Path) -> tuple[list[ManagedModel], str | None]:
    if not path.exists():
        CATALOG_CACHE.pop(str(path), None)
        return [], f"Catalog file not found: {path}"
    cache_key = str(path)
    try:
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
    except OSError as exc:
        CATALOG_CACHE.pop(cache_key, None)
        return [], f"Could not stat catalog {path}: {exc}"
    cached = CATALOG_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature[0] and cached[1] == signature[1]:
        return [replace(item) for item in cached[2]], None
    try:
        payload = json.loads(path.read_text("utf-8"))
    except Exception as exc:
        CATALOG_CACHE.pop(cache_key, None)
        return [], f"Could not read/parse catalog {path}: {exc}"
    if not isinstance(payload, list):
        CATALOG_CACHE.pop(cache_key, None)
        return [], f"Catalog {path} has invalid format (expected a JSON array)."
    try:
        items = [ManagedModel(**m) for m in payload]
    except Exception as exc:
        CATALOG_CACHE.pop(cache_key, None)
        return [], f"Catalog {path} has invalid entries: {exc}"
    changed = False
    for item in items:
        normalized = preferred_tensor_split(item, item.tensor_split)
        if normalized != item.tensor_split:
            item.tensor_split = normalized
            changed = True
        normalized_overrides = normalize_server_overrides(item.server_overrides)
        if normalized_overrides != item.server_overrides:
            item.server_overrides = normalized_overrides
            changed = True
    if changed:
        save_catalog(path, items)
    cached_items = [replace(item) for item in items]
    CATALOG_CACHE[cache_key] = (signature[0], signature[1], cached_items)
    return [replace(item) for item in cached_items], None

def save_catalog(path: Path, models: list[ManagedModel]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([asdict(m) for m in models], indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    try:
        stat = path.stat()
        CATALOG_CACHE[str(path)] = (stat.st_mtime_ns, stat.st_size, [replace(model) for model in models])
    except OSError:
        CATALOG_CACHE.pop(str(path), None)


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


def _append_quant_suffix_if_missing(base_id: str, quant: str | None) -> str:
    quant_token = _normalize_model_id_token(quant or "")
    if not quant_token:
        return base_id
    if re.search(rf"(^|[-._]){re.escape(quant_token)}($|[-._])", base_id):
        return base_id
    return _normalize_model_id_token(f"{base_id}-{quant_token}")


def normalize_model_id(repo_id, quant, filename):
    filename_seed = Path(filename).stem if filename else ""
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
    files = sorted(s.rfilename for s in info.siblings if s.rfilename and s.rfilename.lower().endswith(".gguf"))
    if not files: raise RuntimeError(f"No GGUF in {repo_id}.")
    if explicit_file:
        if explicit_file not in files: raise RuntimeError(f"File {explicit_file} not found.")
        return explicit_file
    if quant:
        ql = quant.lower()
        matches = [f for f in files if ql in f.lower()]
        if len(matches) == 1: return matches[0]
        if len(matches) > 1:
            # Handle sharded models: Pick first shard
            shards = sorted([f for f in matches if "-00001-of-" in f])
            if shards: return shards[0]
            
            exact = [f for f in matches if Path(f).stem.lower().endswith(ql)]
            if len(exact) == 1: return exact[0]
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

def render_llamaswap_config(catalog, path, server_path, start_port, idle_ttl=DEFAULT_IDLE_TTL):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "healthCheckTimeout": 600, "logLevel": "info", "logToStdout": "proxy", "startPort": start_port,
        "sendLoadingState": True, "includeAliasesInList": True, "models": {},
    }
    server_defaults = resolve_llama_server_defaults()
    for m in sorted(catalog, key=lambda x: x.model_id):
        cmd = build_llama_server_command(m, server_path, port="${PORT}", server_defaults=server_defaults)
        data["models"][m.model_id] = {"cmd": " \\\n  ".join(shell_quote(part) for part in cmd), "checkEndpoint": "/health", "ttl": int(idle_ttl)}
        if m.aliases: data["models"][m.model_id]["aliases"] = m.aliases
        if m.description: data["models"][m.model_id]["description"] = m.description
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f: yaml.safe_dump(data, f, sort_keys=False)
    tmp.replace(path)

def shell_quote(v):
    if re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", v):
        return v
    if re.fullmatch(r"[A-Za-z0-9_./:=,+-]+", v): return v
    return "'" + v.replace("'", "'\"'\"'") + "'"


def _load_server_config_payload(args = None) -> dict:
    if args is not None and getattr(args, "idle_ttl", None) is not None:
        pass
    server_config = DEFAULT_SERVER_CONFIG_PATH
    if args is not None and getattr(args, "server_config", None) is not None:
        server_config = Path(args.server_config)
    try:
        payload = json.loads(server_config.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def resolve_idle_ttl(args = None) -> int:
    if args is not None and getattr(args, "idle_ttl", None) is not None:
        return int(args.idle_ttl)
    value = _load_server_config_payload(args).get("idle_ttl")
    if value is not None:
        return int(value)
    return DEFAULT_IDLE_TTL


def resolve_llama_server_defaults(args = None) -> dict[str, object]:
    return normalize_server_overrides(_load_server_config_payload(args).get("llama_server_defaults"))


def _append_llama_server_flag(cmd: list[str], key: str, value: object) -> None:
    if key == "split_mode":
        cmd.extend(["--split-mode", str(value)])
    elif key == "flash_attn":
        bool_val = _normalize_bool_flag(value)
        if bool_val:
            cmd.append("--flash-attn")
    elif key == "batch_size":
        cmd.extend(["--batch-size", str(int(value))])
    elif key == "ubatch_size":
        cmd.extend(["--ubatch-size", str(int(value))])
    elif key == "threads":
        cmd.extend(["--threads", str(int(value))])
    elif key == "threads_batch":
        cmd.extend(["--threads-batch", str(int(value))])
    elif key == "numa":
        cmd.extend(["--numa", str(value)])
    elif key == "fit_target":
        cmd.extend(["--fit-target", str(int(value))])
    elif key == "cache_type_k":
        cmd.extend(["--cache-type-k", str(value)])
    elif key == "cache_type_v":
        cmd.extend(["--cache-type-v", str(value)])
    elif key == "mmap":
        bool_val = _normalize_bool_flag(value)
        if bool_val is False:
            cmd.append("--no-mmap")


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
) -> list[str]:
    effective = dict(normalize_server_overrides(server_defaults or {}))
    effective.update(normalize_server_overrides(model.server_overrides))
    ctx_size = int(effective.pop("ctx_size", model.ctx_size))
    n_gpu_layers = int(effective.pop("n_gpu_layers", model.n_gpu_layers))
    tensor_split = preferred_tensor_split(model, str(effective.pop("tensor_split", model.tensor_split)))
    resolved_host = str(effective.pop("host", host or model.host))
    cmd = [str(server_path), "--port", str(port)]
    if include_model_path:
        cmd.extend(["--model", str(model.local_path)])
    cmd.extend(["--ctx-size", str(ctx_size)])
    cmd.extend(["--n-gpu-layers", str(n_gpu_layers)])
    cmd.extend(["--tensor-split", tensor_split])
    cmd.extend(["--host", resolved_host])
    for key in (
        "split_mode",
        "flash_attn",
        "batch_size",
        "ubatch_size",
        "threads",
        "threads_batch",
        "numa",
        "fit_target",
        "cache_type_k",
        "cache_type_v",
        "mmap",
    ):
        if key in effective:
            _append_llama_server_flag(cmd, key, effective[key])
    if include_mmproj and model.mmproj_path:
        cmd.extend(["--mmproj", str(model.mmproj_path)])
    if include_jinja and model.jinja:
        cmd.append("--jinja")
    return cmd


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
    if getattr(args, "idle_ttl", None) is None and getattr(args, "api_port", None) is None:
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

def wait_for_model(model_id, host, port, timeout=35):
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

def apply_config_and_wait(catalog, config_path, llama_server, start_port, model_id, host, port, progress_callback = None, settle_time = 3.0, timeout = 45.0):
    stop_running_ollama_models(progress_callback=progress_callback)
    render_llamaswap_config(catalog, config_path, llama_server, start_port, resolve_idle_ttl())
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

def _emit_message(message: str, progress_callback = None):
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
                part_path.replace(dest_path)
                _emit_message(f"{filename} already complete.", progress_callback)
                return str(dest_path)

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
                    raise RuntimeError(event.get("message", "Unknown manager error."))
                else:
                    raise RuntimeError(f"Unexpected response from manager: {event}")

    raise RuntimeError("Manager connection closed unexpectedly.")


def manager_hint() -> str:
    user_mode_commands = (
        f"systemctl --user start {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}",
        f"systemctl --user status {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}",
    )
    system_mode_commands = (
        f"sudo systemctl start {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}",
        f"sudo systemctl status {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}",
    )
    return (
        "Could not connect to the background manager.\n"
        "Try one of these:\n"
        f"  {user_mode_commands[0]}\n"
        f"  {user_mode_commands[1]}\n"
        f"  {system_mode_commands[0]}\n"
        f"  {system_mode_commands[1]}\n"
        f"Socket path: {SOCKET_PATH}"
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

    if not matches:
        raise RuntimeError("Model not found in catalog.")
    if len(matches) > 1:
        options = ", ".join(sorted(m.model_id for m in matches))
        raise RuntimeError(f"Ambiguous model selection. Use --model-id or --file. Matches: {options}")
    return matches[0]

def ensure_model_available(args, progress_callback = None):
    # CLIENT MODE: Send to socket if not owner
    try:
        is_owner = (os.getuid() == os.stat(args.catalog.parent).st_uid)
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
    catalog = load_catalog(args.catalog)
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
    if existing and ctx_override is not None and (existing.ctx_size != ctx_override or existing.auto_ctx_failed or existing.auto_ctx_error):
        existing.ctx_size = ctx_override
        existing.auto_ctx_failed = False
        existing.auto_ctx_error = ""
        save_catalog(args.catalog, catalog)
        ctx_changed = True
        auto_ctx_failed = False
        auto_ctx_error = ""
        _emit_message(f"Applied ctx override for {existing.model_id}: {ctx_override}", progress_callback)
    elif existing and skip_ctx and (existing.ctx_size != default_ctx or existing.auto_ctx_failed or existing.auto_ctx_error):
        existing.ctx_size = default_ctx
        existing.auto_ctx_failed = False
        existing.auto_ctx_error = ""
        save_catalog(args.catalog, catalog)
        ctx_changed = True
        auto_ctx_failed = False
        auto_ctx_error = ""
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
        if existing.mmproj_filename != desired_mmproj_filename:
            existing.mmproj_filename = desired_mmproj_filename
            config_changed = True
            _emit_message(f"Applied mmproj filename for {existing.model_id}: {desired_mmproj_filename or 'none'}", progress_callback)
        if existing.mmproj_path != desired_mmproj_path:
            existing.mmproj_path = desired_mmproj_path
            config_changed = True
            _emit_message(f"Applied mmproj path for {existing.model_id}: {desired_mmproj_path or 'none'}", progress_callback)
        if config_changed:
            save_catalog(args.catalog, catalog)

    mid = args.model_id or (existing.model_id if existing else normalize_model_id(repo_id, quant, selected_file))
    files_ready = model_files_ready(target_dir, to_download, expected_sizes)
    completed_files, partial_files, missing_files = summarize_download_state(target_dir, to_download, expected_sizes)

    if existing and not args.force and files_ready and mmproj_ready and not force_auto_ctx:
        _emit_message("All required model files already exist locally. Skipping download.", progress_callback)
        if not ctx_changed and not config_changed and wait_for_model(existing.model_id, args.public_host, args.public_port, timeout=2):
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
            )
        except Exception:
            save_catalog(args.catalog, stable_catalog)
            restore_catalog_config(args, stable_catalog, progress_callback=progress_callback, restart_service=True)
            raise
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

    desired_ctx = ctx_override if ctx_override is not None else (existing.ctx_size if existing and existing.auto_ctx_failed else default_ctx)
    if ctx_override is not None:
        auto_ctx_failed = False
        auto_ctx_error = ""
        _emit_message(f"Using explicit ctx override {desired_ctx} for {mid}.", progress_callback)
    elif skip_ctx:
        auto_ctx_failed = False
        auto_ctx_error = ""
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
            _emit_message(f"Selected automatic ctx {desired_ctx} for {mid}.", progress_callback)
        elif status == "min-failed":
            if probe_config_replaced:
                save_catalog(args.catalog, stable_catalog)
                restore_catalog_config(args, stable_catalog, progress_callback=progress_callback, restart_service=True)
            desired_ctx = int(info.get("min_ctx") or default_ctx)
            auto_ctx_failed = True
            auto_ctx_error = info.get("reason") or status
            _emit_message(
                f"{mid} failed the automatic probe at the minimum ctx {desired_ctx} ({auto_ctx_error}).",
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
            _emit_message(
                f"Could not auto-tune ctx for {mid}. Keeping fallback ctx {desired_ctx} and disabling automatic re-probes for future runs.",
                progress_callback,
            )

    new_m = ManagedModel(model_id=mid, repo_id=repo_id, quant=quant, filename=selected_file, local_path=str(local_path), mmproj_filename=mmproj_filename, mmproj_path=mmproj_path, ctx_size=desired_ctx, n_gpu_layers=int(args.n_gpu_layers), tensor_split=args.tensor_split, host=args.host, jinja=not args.no_jinja, ttl=resolve_idle_ttl(args), description=args.description or f"{repo_id} / {selected_file}", auto_ctx_failed=auto_ctx_failed, auto_ctx_error=auto_ctx_error)
    new_cat = [m for m in catalog if m.model_id != mid] + [new_m]
    save_catalog(args.catalog, new_cat)
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
        )
    except Exception:
        save_catalog(args.catalog, stable_catalog)
        restore_catalog_config(args, stable_catalog, progress_callback=progress_callback, restart_service=True)
        raise
    return mid

def wait_for_model_absent(model_id, host, port, timeout=35):
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

def apply_config_and_wait_absent(catalog, config_path, llama_server, start_port, model_id, host, port, progress_callback = None, settle_time = 3.0, timeout = 45.0):
    render_llamaswap_config(catalog, config_path, llama_server, start_port, resolve_idle_ttl())
    _emit_message("Config updated. Waiting for llama-swap --watch-config...", progress_callback)
    time.sleep(settle_time)
    if wait_for_model_absent(model_id, host, port, timeout=timeout):
        return True
    raise RuntimeError(
        f"Model {model_id} is still published after updating {config_path}. "
        "Ensure llama-swap is running with --watch-config and is watching that config file."
    )

def remove_model(args, progress_callback = None):
    try:
        is_owner = (os.getuid() == os.stat(args.catalog.parent).st_uid)
    except:
        is_owner = False

    if not is_owner:
        try:
            return run_manager_command("remove", args)
        except RuntimeError as e:
            raise e
        except Exception as e:
            raise manager_unavailable_error(e)

    catalog = load_catalog(args.catalog)
    model = resolve_catalog_model(catalog, target=args.repo, repo_ref=args.hf, model_id=args.model_id, filename=args.file)
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
    )

    if args.delete_files:
        if any(m.repo_id == model.repo_id for m in remaining):
            _emit_message("Other catalog entries still use this repository. Keeping files on disk.", progress_callback)
        else:
            repo_dir = args.models_dir / model.repo_id
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
                _emit_message(f"Deleted local files under {repo_dir}.", progress_callback)
            else:
                _emit_message(f"No local files found under {repo_dir}.", progress_callback)

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
        if process is not None:
            running_model = model_by_path.get(process.get("model_path") or "")
            if running_model == model_id:
                continue
            if running_model:
                conflicts.append(f"{running_model} (pid {pid}, {used_mem} MiB)")
                continue
        conflicts.append(f"{_describe_pid(pid)} (pid {pid}, {used_mem} MiB)")
    if not conflicts:
        return None
    joined = "; ".join(conflicts[:4])
    if len(conflicts) > 4:
        joined += f"; +{len(conflicts) - 4} more"
    return (
        f"Cannot load model '{model_id}' because the GPU is already in use: {joined}. "
        "Wait for those workloads to finish or unload them first."
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
        gguf_ctx = get_model_context_size(model)
        process = process_by_model.get(_safe_realpath(model.local_path))
        published = model.model_id in published_models
        loaded = process is not None
        runtime, planned = classify_runtime(model, loaded, process["pid"] if process else None, gpu_process_map)
        rows.append({
            "MODEL_ID": model.model_id,
            "PUBLISHED": "yes" if published else "no",
            "LOADED": "yes" if loaded else "no",
            "RUNTIME": runtime,
            "GPU_PLAN": planned,
            "PID": str(process["pid"]) if process else "-",
            "MAX_CTX": str(gguf_ctx) if gguf_ctx is not None else "?",
            "CFG_CTX": str(model.ctx_size),
            "SIZE": _format_bytes(storage["size"]),
            "FILES": str(storage["file_count"]),
            "STATUS": storage["status"],
            "REPO": model.repo_id,
        })

    columns = [
        ("MODEL_ID", 999),
        ("PUBLISHED", 9),
        ("LOADED", 6),
        ("RUNTIME", 12),
        ("GPU_PLAN", 18),
        ("PID", 8),
        ("MAX_CTX", 8),
        ("CFG_CTX", 8),
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

def build_model_ctx_payload(model: ManagedModel) -> dict:
    max_ctx = get_model_context_size(model)
    return {
        "name": model.model_id,
        "model": model.model_id,
        "size": get_model_storage_info(model)["size"],
        "details": {
            "format": "gguf",
            "family": "llama.cpp",
            "parameter_size": model.quant or "",
            "configured_ctx": model.ctx_size,
            "max_ctx": max_ctx,
        },
    }


def build_openai_model_payload(model: ManagedModel) -> dict:
    max_ctx = get_model_context_size(model)
    return {
        "id": model.model_id,
        "object": "model",
        "created": 0,
        "owned_by": "llama-swap",
        "metadata": {
            "configured_context_length": model.ctx_size,
            "context_length": max_ctx,
            "vision": _has_vision_runtime(model),
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
    return bool(model.mmproj_path and Path(model.mmproj_path).exists())


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
    storage = get_model_storage_info(model)
    max_ctx = get_model_context_size(model)
    local_path = Path(model.local_path)
    modified = None
    try:
        modified = datetime.fromtimestamp(local_path.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        modified = datetime.now(timezone.utc).isoformat()
    details = {
        "parent_model": "",
        "format": "gguf",
        "family": _infer_model_family(model),
        "families": [_infer_model_family(model)],
        "parameter_size": _infer_parameter_size(model),
        "quantization_level": model.quant or "",
        "configured_context_length": model.ctx_size,
        "context_length": max_ctx,
        "vision": _has_vision_runtime(model),
    }
    payload = {
        "name": model.model_id,
        "model": model.model_id,
        "modified_at": modified,
        "size": storage["size"],
        "digest": _model_digest(model),
        "details": details,
        "model_info": {
            "llamacpp.configured_context_length": model.ctx_size,
            "llamacpp.context_length": max_ctx,
        },
    }
    if loaded:
        runtime, _planned = classify_runtime(model, True, process["pid"] if process else None, gpu_process_map or {})
        payload["expires_at"] = None
        payload["size_vram"] = storage["size"] if runtime in {"100%-gpu", "partial-gpu", "gpu-active"} else 0
    return payload


def model_name_aliases(model: ManagedModel) -> list[str]:
    aliases: list[str] = []
    if model.repo_id:
        aliases.append(f"hf.co/{model.repo_id}")
        if model.quant:
            aliases.append(f"hf.co/{model.repo_id}:{model.quant}")
    for alias in model.aliases:
        if alias not in aliases:
            aliases.append(alias)
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
            self.wfile.write(encoded)

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
                self._send_json({"version": "0.1.0-llamacpp-stack"})
                return
            if parsed.path == "/api/ctx":
                self._send_json({
                    "models": [
                        {
                            "name": model.model_id,
                            "configured_ctx": model.ctx_size,
                            "max_ctx": get_model_context_size(model),
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
            max_ctx = get_model_context_size(model)
            self._send_json({
                "license": "",
                "modelfile": f"FROM {model.model_id}\nPARAMETER num_ctx {model.ctx_size}\n",
                "parameters": f"num_ctx {model.ctx_size}",
                "template": "",
                "details": build_model_ctx_payload(model)["details"],
                "model_info": {
                    "llamacpp.configured_context_length": model.ctx_size,
                    "llamacpp.context_length": max_ctx,
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

def list_models(args):
    try:
        is_owner = (os.getuid() == os.stat(args.catalog.parent).st_uid)
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

    output = render_models_table(load_catalog(args.catalog), args.public_host, args.public_port, get_effective_idle_ttl(args))
    if output:
        print("\n" + output)
    else:
        print("No models registered yet.")
    return 0


def show_request_log(args):
    print(_tail_text_file(DEFAULT_REQUESTS_LOG_PATH, lines=int(args.lines)))
    return 0

def temporarily_unload_published_models(args, progress_callback = None, timeout = 45):
    _emit_message(
        "To probe ctx reliably, published models will be unloaded temporarily so they do not occupy VRAM.",
        progress_callback,
    )
    render_llamaswap_config([], args.config, args.llama_server, args.start_port, resolve_idle_ttl(args))
    _emit_message("Temporary empty config written. Waiting for llama-swap --watch-config...", progress_callback)
    time.sleep(3.0)
    url = f"http://{args.public_host}:{args.public_port}/v1/models"
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
    render_llamaswap_config(catalog, args.config, args.llama_server, args.start_port, resolve_idle_ttl(args))
    _emit_message("Previous llama-swap config restored after the failed operation.", progress_callback)
    if restart_service:
        restart_service_to_free_vram(args.service, progress_callback=progress_callback, settle_time=settle_time)
    else:
        time.sleep(settle_time)

def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]

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
                "max_tokens": 1,
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
                        try:
                            probe = requests.post(
                                f"http://127.0.0.1:{port}{endpoint}",
                                json=body,
                                timeout=request_timeout,
                            )
                        except requests.Timeout:
                            last_reason = _with_trace(f"{endpoint}-timeout", trace_path)
                            continue
                        except Exception as e:
                            last_reason = _with_trace(f"{endpoint}-{e.__class__.__name__}", trace_path)
                            continue
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
                        except Exception:
                            last_reason = _with_trace(f"{endpoint}-invalid-json", trace_path)
                            continue
                        try:
                            trace_handle.flush()
                        except Exception:
                            pass
                        return True, _with_trace("ok", trace_path)
                    return False, last_reason
            except Exception:
                pass
            time.sleep(2)
        return False, _with_trace("timeout", trace_path)
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

def choose_auto_ctx(model: ManagedModel, llama_server: Path, progress_callback = None):
    max_ctx = get_model_context_size(model)
    if max_ctx is None:
        _emit_message(f"{model.model_id}: could not read GGUF max context, skipping.", progress_callback)
        return None, "metadata-missing", {"max_ctx": None}

    start_ctx = min(8192, max_ctx)
    if start_ctx <= 0:
        _emit_message(f"{model.model_id}: invalid max context {max_ctx}, skipping.", progress_callback)
        return None, "metadata-missing", {"max_ctx": max_ctx}

    _emit_message(
        f"{model.model_id}: GGUF max ctx {max_ctx}. Auto-fit will calibrate memory once, probe the GGUF maximum, and then refine downward using memory-fit hints.",
        progress_callback,
    )
    _emit_message(f"{model.model_id}: each probe now sends a conservative long prompt sized for that ctx candidate.", progress_callback)
    if _has_vision_runtime(model):
        _emit_message(f"{model.model_id}: vision runtime detected, each ctx probe will include a valid image plus text.", progress_callback)

    free_vram_mib = _query_gpu_free_memory_mib()
    calibration_ctx = start_ctx
    _emit_message(f"{model.model_id}: calibration probe at ctx {calibration_ctx}...", progress_callback)
    ok, reason = probe_model_ctx(model, llama_server, calibration_ctx)
    if not ok:
        _emit_message(f"{model.model_id}: failed even at calibration ctx {calibration_ctx} ({reason}), skipping.", progress_callback)
        return None, "min-failed", {"max_ctx": max_ctx, "min_ctx": calibration_ctx, "reason": reason}

    calibration_trace = None
    match = re.search(r"trace: ([^)]+)", reason or "")
    if match:
        calibration_trace = Path(match.group(1))
    metrics = _parse_probe_trace_metrics(calibration_trace)
    if calibration_trace is not None:
        try:
            calibration_trace.unlink()
        except FileNotFoundError:
            pass
    estimated_ctx = _estimate_ctx_ceiling(model, calibration_ctx, metrics, free_vram_mib) or calibration_ctx
    estimated_ctx = max(calibration_ctx, min(_align_ctx(max_ctx), estimated_ctx))
    _emit_message(f"{model.model_id}: estimated stable ctx ceiling from memory fit = {estimated_ctx}.", progress_callback)

    low = calibration_ctx
    max_probe_ctx = _align_ctx(max_ctx)
    high = None
    last_success = calibration_ctx
    first_failure = None
    tested = {calibration_ctx}

    candidate_order: list[int] = []
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
        ok, reason = probe_model_ctx(model, llama_server, candidate)
        if ok:
            low = candidate
            last_success = candidate
            if candidate == max_probe_ctx:
                return last_success, "selected", {
                    "max_ctx": max_ctx,
                    "calibration_ctx": calibration_ctx,
                    "estimated_ctx": estimated_ctx,
                    "first_failure": first_failure,
                    "selected_ctx": last_success,
                }
            continue
        if high is None or candidate < high:
            high = candidate
        first_failure = candidate
        _emit_message(f"{model.model_id}: ctx {candidate} failed ({reason}).", progress_callback)
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
        ok, reason = probe_model_ctx(model, llama_server, midpoint)
        if ok:
            low = midpoint
            last_success = midpoint
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
    }


def _materialize_validation_model(args, progress_callback = None) -> tuple[ManagedModel, Path | None]:
    catalog = load_catalog(args.catalog)
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
    stable_catalog = load_catalog(args.catalog)
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
        is_owner = (os.getuid() == os.stat(args.catalog.parent).st_uid)
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

    catalog = load_catalog(args.catalog)
    target = getattr(args, "repo", None)
    repo_ref = getattr(args, "hf", None)
    model_id = getattr(args, "model_id", None)
    filename = getattr(args, "file", None)
    ctx_override = getattr(args, "ctx_override", None)
    ctx_override = int(ctx_override) if ctx_override is not None else None
    auto_ctx = bool(getattr(args, "auto_ctx", False))
    if ctx_override is not None and auto_ctx:
        raise RuntimeError("Use either -ctx or --auto, not both.")

    if target or repo_ref or model_id or filename:
        selected_model = resolve_catalog_model(catalog, target=target, repo_ref=repo_ref, model_id=model_id, filename=filename)
        target_models = [selected_model]
    else:
        target_models = catalog

    if auto_ctx:
        probe_config_replaced = False
        if target_models:
            probe_config_replaced = True
            temporarily_unload_published_models(args, progress_callback=progress_callback)
        total_models = len(target_models)
        updated_ctx = 0
        missing_ctx = 0
        try:
            for idx, model in enumerate(target_models, start=1):
                _emit_message(f"[{idx}/{total_models}] Probing {model.model_id}...", progress_callback)
                best_ctx, status, _info = choose_auto_ctx(model, args.llama_server, progress_callback=progress_callback)
                if best_ctx is None:
                    missing_ctx += 1
                    continue
                if model.ctx_size != best_ctx:
                    model.ctx_size = best_ctx
                    updated_ctx += 1
                model.auto_ctx_failed = False
                model.auto_ctx_error = ""
                save_catalog(args.catalog, catalog)
                _emit_message(f"{model.model_id}: selected cfg ctx {model.ctx_size}.", progress_callback)
        except Exception:
            if probe_config_replaced:
                restore_catalog_config(args, catalog, progress_callback=progress_callback)
            raise
    elif ctx_override is not None:
        for model in target_models:
            model.ctx_size = ctx_override
            model.auto_ctx_failed = False
            model.auto_ctx_error = ""
        updated_ctx = len(target_models)
        missing_ctx = 0
        if len(target_models) == 1:
            _emit_message(f"Applied ctx override to {target_models[0].model_id}: {ctx_override}", progress_callback)
        else:
            _emit_message(f"Applied ctx override to all catalog models: {ctx_override}", progress_callback)
    else:
        updated_ctx, missing_ctx = sync_catalog_context_sizes(target_models)
    save_catalog(args.catalog, catalog)
    render_llamaswap_config(catalog, args.config, args.llama_server, args.start_port, resolve_idle_ttl(args))
    if auto_ctx:
        _emit_message(
            f"Catalog ctx updated automatically: {updated_ctx} models changed, {missing_ctx} skipped.",
            progress_callback,
        )
    elif ctx_override is not None:
        _emit_message(
            f"Catalog ctx updated manually: {updated_ctx} models set to {ctx_override}.",
            progress_callback,
        )
    else:
        _emit_message(
            f"Catalog synchronized from GGUF metadata: {updated_ctx} ctx values updated, {missing_ctx} unavailable.",
            progress_callback,
        )
    _emit_message("Config updated from catalog. Waiting for llama-swap --watch-config...", progress_callback)
    time.sleep(3.0)
    try:
        r = requests.get(f"http://{args.public_host}:{args.public_port}/v1/models", timeout=2)
        if r.status_code == 200:
            _emit_message(
                f"Public API reachable on http://{args.public_host}:{args.public_port} ({len(r.json().get('data', []))} published models).",
                progress_callback,
            )
        else:
            _emit_message(
                f"Config updated, but public API responded with HTTP {r.status_code}.",
                progress_callback,
            )
    except Exception as e:
        _emit_message(
            f"Config updated, but could not verify public API on http://{args.public_host}:{args.public_port} ({e.__class__.__name__}).",
            progress_callback,
        )
    return "updated"

def warmup_model(model_id, host, port, timeout=600):
    url = f"http://{host}:{port}/v1/chat/completions"
    print(f"\033[35;1mWarming model {model_id} before opening the chat...\033[0m")
    loader = LoadingBar("\033[35;1mLoading model:\033[0m ")
    loader.start()
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
            timeout=timeout,
        )
        r.raise_for_status()
    finally:
        loader.stop()


def flush_stdin_buffer() -> None:
    try:
        if sys.stdin.isatty():
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass

def start_chat(model_id, host, port):
    warmup_model(model_id, host, port)
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
        def handle():
            sock_in = None
            try:
                sock_in = conn.makefile("r", encoding="utf-8")
                data = sock_in.readline()
                if not data: return
                req = json.loads(data)

                def send_event(event):
                    conn.sendall((json.dumps(event) + "\n").encode())
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
                    model_id = ensure_model_available(mock_args, progress_callback=send_event)
                    send_event({"type": "done", "model_id": model_id})
                elif req["command"] == "list":
                    mock_args = argparse.Namespace(**req["args"])
                    mock_args.catalog = Path(mock_args.catalog)
                    table = render_models_table(load_catalog(mock_args.catalog), mock_args.public_host, mock_args.public_port)
                    send_event({"type": "done", "result": table})
                elif req["command"] == "update":
                    mock_args = argparse.Namespace(**req["args"])
                    mock_args.catalog = Path(mock_args.catalog)
                    mock_args.config = Path(mock_args.config)
                    mock_args.llama_server = Path(mock_args.llama_server)
                    result = update_config(mock_args, progress_callback=send_event)
                    send_event({"type": "done", "result": result})
                elif req["command"] == "remove":
                    mock_args = argparse.Namespace(**req["args"])
                    mock_args.catalog = Path(mock_args.catalog)
                    mock_args.config = Path(mock_args.config)
                    mock_args.models_dir = Path(mock_args.models_dir)
                    mock_args.llama_server = Path(mock_args.llama_server)
                    model_id = remove_model(mock_args, progress_callback=send_event)
                    send_event({"type": "done", "model_id": model_id})
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
                conn.close()
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

def build_help_epilog():
    ui_url = f"http://{DEFAULT_PUBLIC_HOST}:{DEFAULT_PUBLIC_PORT}"
    api_url = f"http://{DEFAULT_PUBLIC_HOST}:{resolve_api_port()}"
    manifest = read_install_manifest()
    llama_cpp_tag = str(manifest.get("llama_cpp_tag") or "unknown")
    llamaswap_tag = str(manifest.get("llamaswap_tag") or "unknown")
    return (
        "Default endpoints:\n"
        f"  llama-swap UI/backend: {ui_url}\n"
        f"  Superserver API:       {api_url}\n"
        "Installed versions:\n"
        f"  llama.cpp:           {llama_cpp_tag}\n"
        f"  llama-swap:          {llamaswap_tag}\n"
        "Runtime info:\n"
        f"  Install root:        {DEFAULT_LLAMA_SERVER.parent.parent.parent if 'llama.cpp/build/bin' in str(DEFAULT_LLAMA_SERVER) else DEFAULT_LLAMA_SERVER.parent}\n"
        f"  Models dir:          {DEFAULT_MODELS_DIR}\n"
        f"  llama-swap config:   {DEFAULT_CONFIG_PATH}\n"
        f"  Catalog:             {DEFAULT_CATALOG_PATH}\n"
        f"  App config:          {DEFAULT_SERVER_CONFIG_PATH}\n"
        f"  llama-server binary: {DEFAULT_LLAMA_SERVER}\n"
        f"  UI activity:         {ui_url}/ui/#/activity\n"
        f"  Idle TTL:            {resolve_idle_ttl()}s\n"
        "Config knobs:\n"
        f"  Global llama-server defaults: {DEFAULT_SERVER_CONFIG_PATH} -> llama_server_defaults\n"
        f"  Per-model overrides:          {DEFAULT_CATALOG_PATH} -> server_overrides\n"
        "  Main folders: install root, models dir, state/config paths above.\n"
        f"  API status:          {get_api_endpoint_status()}\n"
        f"  UI status:           {get_public_endpoint_status()}"
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
    sub = parser.add_subparsers(dest="command", required=True)
    subparsers: dict[str, argparse.ArgumentParser] = {}
    
    p_add = sub.add_parser("add")
    subparsers["add"] = p_add
    p_add.set_defaults(func=lambda a: ensure_model_available(a) and 0)
    
    p_run = sub.add_parser("run")
    subparsers["run"] = p_run
    p_run.set_defaults(
        func=lambda a: (
            (ensure_model_available(a) and 0)
            if a.no_chat
            else start_chat(ensure_model_available(a), a.public_host, a.public_port)
        )
    )
    p_run.add_argument("--no-chat", action="store_true")
    p_run.add_argument("-ctx", "--ctx", dest="ctx_override", type=int, help="Override ctx size for this run")

    p_remove = sub.add_parser("remove")
    subparsers["remove"] = p_remove
    p_remove.set_defaults(func=lambda a: remove_model(a) and 0)
    p_remove.add_argument("repo", nargs="?", help="Model id or HF repo[:QUANT]")
    p_remove.add_argument("-hf", "--hf", help="HF repo")
    p_remove.add_argument("--file")
    p_remove.add_argument("--model-id")
    p_remove.add_argument("--delete-files", action="store_true")

    p_update = sub.add_parser("update")
    subparsers["update"] = p_update
    p_update.set_defaults(func=lambda a: update_config(a) and 0)
    p_update.add_argument("repo", nargs="?", help="Model id or HF repo[:QUANT]")
    p_update.add_argument("-hf", "--hf", help="HF repo")
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

    p_validate = sub.add_parser("validate")
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
    
    p_daemon = sub.add_parser("daemon")
    subparsers["daemon"] = p_daemon
    p_daemon.set_defaults(func=daemon_mode)
    
    for p in [p_add, p_run]:
        p.add_argument("repo", nargs="?", help="HF repo[:QUANT]")
        p.add_argument("-hf", "--hf", help="HF repo")
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

    p_list = sub.add_parser("list")
    subparsers["list"] = p_list
    p_list.set_defaults(func=list_models)
    p_ps = sub.add_parser("ps")
    subparsers["ps"] = p_ps
    p_ps.set_defaults(func=list_models)
    p_requests = sub.add_parser("requests")
    subparsers["requests"] = p_requests
    p_requests.add_argument("-n", "--lines", type=int, default=50)
    p_requests.set_defaults(func=show_request_log)

    return parser, subparsers


def parse_cli_args(
    parser: argparse.ArgumentParser,
    subparsers: dict[str, argparse.ArgumentParser],
    argv: list[str] | None = None,
) -> argparse.Namespace:
    argv_list = list(sys.argv[1:] if argv is None else argv)
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
