"""Env helpers extracted from llamacpp_stack/cli.py:79-144 + _is_vllm_backend:171.

No top-level import of constants/models/command_router to avoid cycles.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_installed_env() -> None:
    candidates = [
        Path.home() / ".config/heimdall-gateway/heimdall-gateway.env",
        Path("/etc/heimdall-gateway/heimdall-gateway.env"),
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


def _env_value(primary: str, legacy: str | None = None, default: str = "") -> str:
    if os.environ.get(primary):
        return str(os.environ[primary])
    if legacy and os.environ.get(legacy):
        return str(os.environ[legacy])
    return default


def _env_path2(primary: str, legacy: str | None, default: str) -> Path:
    return Path(_env_value(primary, legacy, default)).expanduser()


def _is_vllm_backend() -> bool:
    backend = _env_value("HEIMDALL_GATEWAY_BACKEND", "LLAMACPP_BACKEND", "")
    if _env_value("HEIMDALL_GATEWAY_DEBUG", "DEBUG_LLAMACPP", ""):
        print(f"DEBUG: _is_vllm_backend check. HEIMDALL_GATEWAY_BACKEND='{backend}'", file=sys.stderr)
    return backend == "vllm-beta"


__all__ = [
    "_load_installed_env",
    "_env_path",
    "_env_value",
    "_env_path2",
    "_is_vllm_backend",
]
