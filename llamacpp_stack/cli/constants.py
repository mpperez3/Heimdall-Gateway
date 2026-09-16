"""Shared constants extracted from llamacpp_stack/cli.py:132-190 + port/tensor helpers.

No top-level import of models/command_router to avoid cycles.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

from .env import _env_path, _env_path2, _env_value

PRODUCT_NAME = "LLM Server"
PRODUCT_SLUG = "llm-server"
SOCKET_PATH = _env_value(
    "LLM_SERVER_MANAGER_SOCKET",
    "HEIMDALL_GATEWAY_MANAGER_SOCKET",
    f"/run/{PRODUCT_SLUG}/manager.sock",
)
DEFAULT_MODELS_DIR = _env_path2("LLM_SERVER_MODELS", "HEIMDALL_GATEWAY_MODELS", f"/var/lib/{PRODUCT_SLUG}/models")
DEFAULT_CONFIG_PATH = _env_path2("LLM_SERVER_CONFIG", "HEIMDALL_GATEWAY_CONFIG", f"/var/lib/{PRODUCT_SLUG}/config.yaml")
DEFAULT_CATALOG_PATH = _env_path2("LLM_SERVER_CATALOG", "HEIMDALL_GATEWAY_CATALOG", f"/var/lib/{PRODUCT_SLUG}/catalog.json")
DEFAULT_SERVER_CONFIG_PATH = _env_path(
    "LLM_SERVER_SERVER_CONFIG",
    _env_value(
        "LLM_SERVER_SERVER_CONFIG",
        "HEIMDALL_GATEWAY_SERVER_CONFIG",
        f"/etc/{PRODUCT_SLUG}/conf.json"
        if os.geteuid() == 0
        else str(Path.home() / ".config" / PRODUCT_SLUG / "conf.json"),
    ),
)
ALTERNATE_SERVER_CONFIG_BASENAME = "conf.json"
DEFAULT_SERVICE_NAME = _env_value("LLM_SERVER_SERVICE_NAME", "HEIMDALL_GATEWAY_SERVICE_NAME", "llamaswap")
CLI_COMMAND = "llm-server"
LEGACY_CLI_COMMAND = "llamacpp-superserver"
MANAGER_SERVICE_NAME = "llm-server-manager"
SWAP_SERVICE_NAME = "llm-server-router"
DEFAULT_LLAMA_SERVER = _env_path("LLAMA_SERVER_BIN", f"/opt/{PRODUCT_SLUG}/llama.cpp/build/bin/llama-server")

DEFAULT_CTX_SIZE = 8192
REASONING_BUDGET_HALF_CONTEXT = "half_context"
DEFAULT_REASONING_VISIBLE_RESERVE = 1024
CHAT_TOOL_CONTINUE_REPAIR_THINKING_BUDGET_TOKENS = 512
MODEL_PROBE_REASONING_MAX_TOKENS = 128
try:
    DEFAULT_API_CTX_FACTOR = float(
        os.environ.get(
            "LLM_SERVER_API_CTX_FACTOR",
            os.environ.get("HEIMDALL_GATEWAY_API_CTX_FACTOR", os.environ.get("LLAMACPP_API_CTX_FACTOR", os.environ.get("LLAMACPP_CTX_DISPLAY_RATIO", "0.5"))),
        )
    )
except ValueError:
    DEFAULT_API_CTX_FACTOR = 0.5
DEFAULT_N_GPU_LAYERS = 999
DEFAULT_IDLE_TTL = int(_env_value("LLM_SERVER_IDLE_TTL", "HEIMDALL_GATEWAY_IDLE_TTL", os.environ.get("LLAMACPP_DEFAULT_TTL", "300")))
DEFAULT_MODEL_SWITCH_GRACE_S = int(_env_value("LLM_SERVER_MODEL_SWITCH_GRACE_S", "HEIMDALL_GATEWAY_MODEL_SWITCH_GRACE_S", "30"))
DEFAULT_MAX_CONCURRENT_PER_MODEL = int(
    _env_value("LLM_SERVER_MAX_CONCURRENT_PER_MODEL", "HEIMDALL_GATEWAY_MAX_CONCURRENT_PER_MODEL", "2")
)
LLAMASWAP_UPSTREAM_STATIC_BLOCKED_BASENAMES = {
    "sw.js",
    "service-worker.js",
    "favicon.ico",
    "manifest.json",
}
LLAMASWAP_UPSTREAM_STATIC_BLOCKED_EXTENSIONS = {
    ".css",
    ".js",
    ".map",
    ".ico",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".webp",
    ".woff",
    ".woff2",
    ".ttf",
}

_CUDA_DEVICE_COUNT_CACHE: tuple[float, int] = (0.0, -1)
_CUDA_DEVICE_COUNT_LOCK = threading.Lock()


def detect_cuda_device_count(*, ttl_s: float = 2.0) -> int:
    global _CUDA_DEVICE_COUNT_CACHE
    now = time.monotonic()
    with _CUDA_DEVICE_COUNT_LOCK:
        ts, count = _CUDA_DEVICE_COUNT_CACHE
        if count >= 0 and now - ts <= ttl_s:
            return count
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        count = len([line for line in result.stdout.splitlines() if line.strip()])
    except Exception:
        count = 0
    with _CUDA_DEVICE_COUNT_LOCK:
        _CUDA_DEVICE_COUNT_CACHE = (now, count)
    return count


def default_tensor_split() -> str:
    gpu_count = detect_cuda_device_count()
    if gpu_count <= 0:
        return "1"
    return ",".join(["1"] * gpu_count)


DEFAULT_TENSOR_SPLIT = default_tensor_split()
DEFAULT_START_PORT = 18080
DEFAULT_PUBLIC_HOST = _env_value("LLM_SERVER_PUBLIC_HOST", "HEIMDALL_GATEWAY_PUBLIC_HOST", "127.0.0.1")
DEFAULT_PUBLIC_PORT = int(_env_value("LLM_SERVER_PUBLIC_PORT", "HEIMDALL_GATEWAY_PUBLIC_PORT", "11437"))
DEFAULT_API_PORT = int(_env_value("LLM_SERVER_API_PORT", "HEIMDALL_GATEWAY_API_PORT", str(DEFAULT_PUBLIC_PORT - 1)))
DEFAULT_REQUESTS_LOG_PATH = _env_path(
    "LLM_SERVER_REQUESTS_LOG",
    f"/var/lib/{PRODUCT_SLUG}/api-requests.log"
    if os.geteuid() == 0
    else str(Path.home() / f".local/state/{PRODUCT_SLUG}/api-requests.log"),
)
DEFAULT_REQUESTS_LOG_MAX_BYTES = 10485760
DEFAULT_REQUESTS_LOG_RETAIN_DAYS = 3
DEFAULT_REQUESTS_LOG_COMPRESS = False
DEFAULT_LAST_CHAT_RESPONSE_LOG_PATH = Path(
    _env_value(
        "LLM_SERVER_LAST_CHAT_RESPONSE_LOG",
        "HEIMDALL_GATEWAY_LAST_CHAT_RESPONSE_LOG",
        str(DEFAULT_REQUESTS_LOG_PATH.with_name("last-chat-response.json")),
    )
).expanduser()
SYSTEM_REQUESTS_LOG_PATH = Path(f"/var/lib/{PRODUCT_SLUG}/api-requests.log")


__all__ = [
    "PRODUCT_NAME",
    "PRODUCT_SLUG",
    "SOCKET_PATH",
    "DEFAULT_MODELS_DIR",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_CATALOG_PATH",
    "DEFAULT_SERVER_CONFIG_PATH",
    "ALTERNATE_SERVER_CONFIG_BASENAME",
    "DEFAULT_SERVICE_NAME",
    "CLI_COMMAND",
    "LEGACY_CLI_COMMAND",
    "MANAGER_SERVICE_NAME",
    "SWAP_SERVICE_NAME",
    "DEFAULT_LLAMA_SERVER",
    "DEFAULT_CTX_SIZE",
    "REASONING_BUDGET_HALF_CONTEXT",
    "DEFAULT_REASONING_VISIBLE_RESERVE",
    "CHAT_TOOL_CONTINUE_REPAIR_THINKING_BUDGET_TOKENS",
    "MODEL_PROBE_REASONING_MAX_TOKENS",
    "DEFAULT_API_CTX_FACTOR",
    "DEFAULT_N_GPU_LAYERS",
    "DEFAULT_IDLE_TTL",
    "DEFAULT_MODEL_SWITCH_GRACE_S",
    "DEFAULT_MAX_CONCURRENT_PER_MODEL",
    "LLAMASWAP_UPSTREAM_STATIC_BLOCKED_BASENAMES",
    "LLAMASWAP_UPSTREAM_STATIC_BLOCKED_EXTENSIONS",
    "detect_cuda_device_count",
    "default_tensor_split",
    "DEFAULT_TENSOR_SPLIT",
    "DEFAULT_START_PORT",
    "DEFAULT_PUBLIC_HOST",
    "DEFAULT_PUBLIC_PORT",
    "DEFAULT_API_PORT",
    "DEFAULT_REQUESTS_LOG_PATH",
    "DEFAULT_LAST_CHAT_RESPONSE_LOG_PATH",
    "SYSTEM_REQUESTS_LOG_PATH",
]
