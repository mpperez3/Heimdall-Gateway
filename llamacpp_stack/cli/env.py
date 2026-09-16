"""Env helpers extracted from llamacpp_stack/cli.py:79-144 + _is_vllm_backend:171.

No top-level import of constants/models/command_router to avoid cycles.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_installed_env() -> None:
    candidates = [
        Path.home() / ".config/llm-server/llm-server.env",
        Path("/etc/llm-server/llm-server.env"),
        Path.home() / ".config/heimdall-gateway/heimdall-gateway.env",  # legacy fallback
        Path("/etc/heimdall-gateway/heimdall-gateway.env"),  # legacy fallback
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
    val = os.environ.get(name)
    if val:
        return Path(val).expanduser()
    if name.startswith("LLM_SERVER_"):
        heimdall = name.replace("LLM_SERVER_", "HEIMDALL_GATEWAY_", 1)  # legacy fallback
        if os.environ.get(heimdall):
            return Path(os.environ[heimdall]).expanduser()
        llamacpp = heimdall.replace("HEIMDALL_GATEWAY_", "LLAMACPP_", 1)  # legacy fallback
        if os.environ.get(llamacpp):
            return Path(os.environ[llamacpp]).expanduser()
        if heimdall == "HEIMDALL_GATEWAY_REQUESTS_LOG" and os.environ.get("LLAMACPP_REQUESTS_LOG"):  # legacy fallback
            return Path(os.environ["LLAMACPP_REQUESTS_LOG"]).expanduser()
    if name.startswith("HEIMDALL_GATEWAY_"):  # legacy fallback
        llamacpp = name.replace("HEIMDALL_GATEWAY_", "LLAMACPP_", 1)  # legacy fallback
        if os.environ.get(llamacpp):
            return Path(os.environ[llamacpp]).expanduser()
    return Path(default).expanduser()


def _env_value(primary: str, legacy: str | None = None, default: str = "") -> str:
    if os.environ.get(primary):
        return str(os.environ[primary])
    if legacy and os.environ.get(legacy):
        return str(os.environ[legacy])
    if legacy and legacy.startswith("HEIMDALL_GATEWAY_"):  # legacy fallback
        llamacpp_key = legacy.replace("HEIMDALL_GATEWAY_", "LLAMACPP_", 1)  # legacy fallback
        if os.environ.get(llamacpp_key):
            return str(os.environ[llamacpp_key])
        if legacy == "HEIMDALL_GATEWAY_API_CTX_FACTOR":  # legacy fallback
            if os.environ.get("LLAMACPP_API_CTX_FACTOR"):
                return str(os.environ["LLAMACPP_API_CTX_FACTOR"])
            if os.environ.get("LLAMACPP_CTX_DISPLAY_RATIO"):
                return str(os.environ["LLAMACPP_CTX_DISPLAY_RATIO"])
        if legacy == "HEIMDALL_GATEWAY_IDLE_TTL" and os.environ.get("LLAMACPP_DEFAULT_TTL"):  # legacy fallback
            return str(os.environ["LLAMACPP_DEFAULT_TTL"])
        if legacy == "HEIMDALL_GATEWAY_DEBUG" and os.environ.get("DEBUG_LLAMACPP"):  # legacy fallback
            return str(os.environ["DEBUG_LLAMACPP"])
    if primary.startswith("LLM_SERVER_") and legacy and legacy.startswith("HEIMDALL_GATEWAY_"):  # legacy fallback
        llamacpp_key = legacy.replace("HEIMDALL_GATEWAY_", "LLAMACPP_", 1)  # legacy fallback
        if os.environ.get(llamacpp_key):
            return str(os.environ[llamacpp_key])
        if legacy == "HEIMDALL_GATEWAY_DEBUG" and os.environ.get("DEBUG_LLAMACPP"):  # legacy fallback
            return str(os.environ["DEBUG_LLAMACPP"])
    return default


def _env_path2(primary: str, legacy: str | None, default: str) -> Path:
    return Path(_env_value(primary, legacy, default)).expanduser()


def _is_vllm_backend() -> bool:
    backend = _env_value("LLM_SERVER_BACKEND", "HEIMDALL_GATEWAY_BACKEND", "")  # legacy fallback
    if _env_value("LLM_SERVER_DEBUG", "HEIMDALL_GATEWAY_DEBUG", ""):  # legacy fallback
        print(f"DEBUG: _is_vllm_backend check. LLM_SERVER_BACKEND='{backend}'", file=sys.stderr)
    return backend == "vllm-beta"


__all__ = [
    "_load_installed_env",
    "_env_path",
    "_env_value",
    "_env_path2",
    "_is_vllm_backend",
]
