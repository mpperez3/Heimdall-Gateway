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
import hmac
import html
import ipaddress
import ssl
import json
import math
import os
import re
import shlex
import shutil
import binascii
import zlib
import subprocess
import sys
import time
import threading
import traceback
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
from dataclasses import asdict, dataclass, field, fields, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

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


def _json_loads_allow_comments(text: str, path_desc: str = "") -> object:
    """Parse JSON supporting `#` full-line comments. Returns parsed value or raises."""
    lines = text.split("\n")
    comment_offset = 0
    clean_lines: list[str] = []
    for i, line in enumerate(lines):
        if line.strip().startswith("#"):
            comment_offset += 1
            continue
        clean_lines.append(line)
    clean_text = "\n".join(clean_lines)
    # Strip trailing commas before } or ] (common mistake)
    clean_text = re.sub(r",\s*([}\]])", r"\1", clean_text)
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as exc:
        original_line = exc.lineno + comment_offset
        print(f"\n[!] Syntax error in {path_desc or 'config'} at line {original_line},"
              f" column {exc.colno}: {exc.msg}", file=sys.stderr)
        if original_line <= len(lines):
            context_line = lines[original_line - 1] if original_line > 0 else ""
            print(f"    {original_line}: {context_line}", file=sys.stderr)
        print(f"    Fix the error or delete the file to regenerate it from defaults.\n", file=sys.stderr)
        raise


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


PRODUCT_NAME = "Heimdall Gateway"
PRODUCT_SLUG = "heimdall-gateway"
SOCKET_PATH = _env_value("HEIMDALL_GATEWAY_MANAGER_SOCKET", "LLAMACPP_MANAGER_SOCKET", f"/run/{PRODUCT_SLUG}/manager.sock")
DEFAULT_MODELS_DIR = _env_path2("HEIMDALL_GATEWAY_MODELS", "LLAMACPP_MODELS", f"/var/lib/{PRODUCT_SLUG}/models")
DEFAULT_CONFIG_PATH = _env_path2("HEIMDALL_GATEWAY_CONFIG", "LLAMACPP_CONFIG", f"/var/lib/{PRODUCT_SLUG}/config.yaml")
DEFAULT_CATALOG_PATH = _env_path2("HEIMDALL_GATEWAY_CATALOG", "LLAMACPP_CATALOG", f"/var/lib/{PRODUCT_SLUG}/catalog.json")
DEFAULT_SERVER_CONFIG_PATH = _env_path(
    "HEIMDALL_GATEWAY_SERVER_CONFIG",
    _env_value(
        "HEIMDALL_GATEWAY_SERVER_CONFIG",
        "LLAMACPP_SERVER_CONFIG",
        f"/etc/{PRODUCT_SLUG}/conf.json"
    if os.geteuid() == 0
        else str(Path.home() / ".config" / PRODUCT_SLUG / "conf.json"),
    ),
)
ALTERNATE_SERVER_CONFIG_BASENAME = "conf.json"
DEFAULT_SERVICE_NAME = _env_value("HEIMDALL_GATEWAY_SERVICE_NAME", "LLAMACPP_SERVICE_NAME", "llamaswap")
CLI_COMMAND = "heimdall-gateway"
LEGACY_CLI_COMMAND = "llamacpp-superserver"
MANAGER_SERVICE_NAME = "heimdall-gateway-manager"
SWAP_SERVICE_NAME = "heimdall-gateway-router"
DEFAULT_LLAMA_SERVER = _env_path("LLAMA_SERVER_BIN", f"/opt/{PRODUCT_SLUG}/llama.cpp/build/bin/llama-server")
def _is_vllm_backend() -> bool:
    backend = _env_value("HEIMDALL_GATEWAY_BACKEND", "LLAMACPP_BACKEND", "")
    # Debug print to stderr so it shows up in logs even if stdout is captured
    if _env_value("HEIMDALL_GATEWAY_DEBUG", "DEBUG_LLAMACPP", ""):
        print(f"DEBUG: _is_vllm_backend check. HEIMDALL_GATEWAY_BACKEND='{backend}'", file=sys.stderr)
    return backend == "vllm-beta"

DEFAULT_CTX_SIZE = 8192
REASONING_BUDGET_HALF_CONTEXT = "half_context"
DEFAULT_REASONING_VISIBLE_RESERVE = 1024
CHAT_TOOL_CONTINUE_REPAIR_THINKING_BUDGET_TOKENS = 512
MODEL_PROBE_REASONING_MAX_TOKENS = 128
try:
    DEFAULT_API_CTX_FACTOR = float(
        os.environ.get("HEIMDALL_GATEWAY_API_CTX_FACTOR", os.environ.get("LLAMACPP_API_CTX_FACTOR", os.environ.get("LLAMACPP_CTX_DISPLAY_RATIO", "0.5")))
    )
except ValueError:
    DEFAULT_API_CTX_FACTOR = 0.5
DEFAULT_N_GPU_LAYERS = 999
DEFAULT_IDLE_TTL = int(_env_value("HEIMDALL_GATEWAY_IDLE_TTL", "LLAMACPP_IDLE_TTL", os.environ.get("LLAMACPP_DEFAULT_TTL", "300")))
DEFAULT_MODEL_SWITCH_GRACE_S = int(_env_value("HEIMDALL_GATEWAY_MODEL_SWITCH_GRACE_S", "LLAMACPP_MODEL_SWITCH_GRACE_S", "30"))
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
DEFAULT_PUBLIC_HOST = _env_value("HEIMDALL_GATEWAY_PUBLIC_HOST", "LLAMACPP_PUBLIC_HOST", "127.0.0.1")
DEFAULT_PUBLIC_PORT = int(_env_value("HEIMDALL_GATEWAY_PUBLIC_PORT", "LLAMACPP_PUBLIC_PORT", "11437"))
DEFAULT_API_PORT = int(_env_value("HEIMDALL_GATEWAY_API_PORT", "LLAMACPP_API_PORT", str(DEFAULT_PUBLIC_PORT - 1)))
DEFAULT_REQUESTS_LOG_PATH = _env_path(
    "HEIMDALL_GATEWAY_REQUESTS_LOG",
    "/var/lib/heimdall-gateway/api-requests.log"
    if os.geteuid() == 0
    else str(Path.home() / ".local/state/heimdall-gateway/api-requests.log"),
)
DEFAULT_LAST_CHAT_RESPONSE_LOG_PATH = Path(_env_value(
    "HEIMDALL_GATEWAY_LAST_CHAT_RESPONSE_LOG",
    "LLAMACPP_LAST_CHAT_RESPONSE_LOG",
    str(DEFAULT_REQUESTS_LOG_PATH.with_name("last-chat-response.json")),
)).expanduser()
SYSTEM_REQUESTS_LOG_PATH = Path("/var/lib/heimdall-gateway/api-requests.log")
MODEL_ACTIVITY_LOCK = threading.Lock()
MODEL_ACTIVITY: dict[str, dict[str, float | str]] = {}
LAST_ACTIVITY_MODEL_ID = ""

LLAMASWAP_CONFIG_HEADER = (
    "# Heimdall Gateway config.yaml\n"
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


def _summarize_responses_input_tool_items(payload: dict) -> dict[str, object]:
    raw_input = payload.get("input") if isinstance(payload, dict) else None
    if isinstance(raw_input, dict):
        raw_input = [raw_input]
    if not isinstance(raw_input, list):
        return {"function_calls": [], "function_call_outputs": []}
    calls = []
    outputs = []
    for item in raw_input:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type == "function_call":
            calls.append({
                "id": str(item.get("id") or "")[:120],
                "call_id": str(item.get("call_id") or "")[:120],
                "name": str(item.get("name") or "")[:160],
                "arguments_len": len(str(item.get("arguments") or "")),
            })
        elif item_type in {"function_call_output", "tool_result", "computer_call_output"}:
            text = item.get("output") or item.get("text") or item.get("content") or ""
            outputs.append({
                "call_id": str(item.get("call_id") or "")[:120],
                "status": str(item.get("status") or "")[:80],
                "output_len": len(str(text)),
                "output_preview": str(text)[:300],
            })
    return {"function_calls": calls[:20], "function_call_outputs": outputs[:20]}



def _text_from_message_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content") or item.get("output") or ""
                if isinstance(value, str):
                    parts.append(value)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _summarize_chat_tool_message_diagnostics(messages: object) -> dict[str, object]:
    if not isinstance(messages, list):
        return {"tool_message_count": 0, "matches": []}
    patterns = {
        "terminal_output_suppressed": "[terminal output suppressed]",
        "sudo": "sudo",
        "permission_denied": "permission denied",
        "password": "password",
        "not_allowed": "not allowed",
        "require_escalated": "require_escalated",
    }
    matches: list[dict[str, object]] = []
    tool_count = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        tool_count += 1
        text = _text_from_message_content(message.get("content"))
        lowered = text.lower()
        hit_names = [name for name, needle in patterns.items() if needle.lower() in lowered]
        if not hit_names:
            continue
        preview = text.replace("\r", "\n")
        preview = re.sub(r"\s+", " ", preview).strip()[:500]
        matches.append({
            "message_index": index,
            "tool_call_id": str(message.get("tool_call_id") or "")[:160],
            "name": str(message.get("name") or "")[:160],
            "content_len": len(text),
            "patterns": hit_names,
            "preview": preview,
        })
    return {"tool_message_count": tool_count, "matches": matches[:20]}


def _log_chat_stop_without_tools(
    request_id: str,
    model: str,
    upstream_model: str | None,
    *,
    stream: bool,
    content: str,
    reasoning_len: int,
    finish_reason: object,
    repair_rounds: int = 0,
) -> None:
    if str(finish_reason or "") != "stop":
        return
    preview = str(content or "").replace("\r", "\n")
    preview = re.sub(r"\s+", " ", preview).strip()[:500]
    log_api_event(
        "openai_chat_stop_without_tool_calls",
        {
            "request_id": request_id,
            "model": model,
            "upstream_model": upstream_model,
            "stream": stream,
            "visible_content_len": len(str(content or "")),
            "reasoning_len": int(reasoning_len or 0),
            "finish_reason": str(finish_reason or ""),
            "repair_rounds": repair_rounds,
            "visible_preview": preview,
        },
    )


def resolve_chat_last_response_log_config(args = None) -> dict[str, object]:
    raw = _load_server_config_payload(args).get("experimental")
    cfg = _normalize_experimental_config(raw).get("chat_last_response_log", {})
    if not isinstance(cfg, dict):
        cfg = dict(_default_experimental_config()["chat_last_response_log"])
    return cfg


def _truncate_debug_text(value: object, max_chars: int) -> tuple[str, bool]:
    text = str(value or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _tool_call_debug_summary(tool_calls: object, max_items: int = 20, max_arg_chars: int = 2000) -> list[dict[str, object]]:
    if not isinstance(tool_calls, list):
        return []
    out: list[dict[str, object]] = []
    for index, item in enumerate(tool_calls[:max_items]):
        if not isinstance(item, dict):
            out.append({"index": index, "type": type(item).__name__})
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else {}
        args, args_truncated = _truncate_debug_text(fn.get("arguments") if isinstance(fn, dict) else "", max_arg_chars)
        out.append(
            {
                "index": index,
                "id": str(item.get("id") or "")[:200],
                "type": str(item.get("type") or "")[:80],
                "name": str(fn.get("name") or item.get("name") or "")[:200] if isinstance(fn, dict) else str(item.get("name") or "")[:200],
                "arguments": args,
                "arguments_truncated": args_truncated,
                "arguments_len": len(str(fn.get("arguments") or "")) if isinstance(fn, dict) else 0,
            }
        )
    return out


def _write_chat_last_response_log(
    args,
    *,
    request_id: str,
    model: str,
    upstream_model: str | None,
    stream: bool,
    content: object,
    reasoning: object = "",
    reasoning_len: int | None = None,
    tool_calls: object = None,
    tool_call_chunks: int = 0,
    finish_reason: object = "",
    repair_rounds: int = 0,
) -> None:
    cfg = resolve_chat_last_response_log_config(args)
    if not bool(cfg.get("enabled")):
        return
    try:
        max_chars = max(0, int(cfg.get("max_chars", 20000)))
    except Exception:
        max_chars = 20000
    path_text = str(cfg.get("path") or "").strip()
    path = Path(path_text) if path_text else DEFAULT_LAST_CHAT_RESPONSE_LOG_PATH
    visible_text = str(content or "")
    visible_preview, visible_truncated = _truncate_debug_text(visible_text, max_chars)
    include_reasoning = bool(cfg.get("include_reasoning"))
    reasoning_text = str(reasoning or "")
    reasoning_preview, reasoning_truncated = _truncate_debug_text(reasoning_text, max_chars) if include_reasoning else ("", False)
    entry: dict[str, object] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model": model,
        "upstream_model": upstream_model,
        "stream": stream,
        "finish_reason": str(finish_reason or ""),
        "repair_rounds": int(repair_rounds or 0),
        "visible_content": visible_preview,
        "visible_content_len": len(visible_text),
        "visible_content_truncated": visible_truncated,
        "reasoning_len": int(reasoning_len if reasoning_len is not None else len(reasoning_text)),
        "reasoning_included": include_reasoning,
        "tool_call_chunks": int(tool_call_chunks or 0),
    }
    if include_reasoning:
        entry["reasoning"] = reasoning_preview
        entry["reasoning_truncated"] = reasoning_truncated
    if bool(cfg.get("include_tool_calls", True)):
        entry["tool_calls"] = _tool_call_debug_summary(tool_calls)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        log_api_event("openai_chat_last_response_logged", {"request_id": request_id, "path": str(path), "visible_content_len": len(visible_text), "visible_content_truncated": visible_truncated})
    except Exception as exc:
        log_api_event("openai_chat_last_response_log_error", {"request_id": request_id, "path": str(path), "error": str(exc)})


def _default_global_replicas_config() -> dict[str, object]:
    return {
        "enabled": False,
        "max": "auto",
        "placement": "exclusive_gpus",
        "safety_vram_mib": 2048,
    }


def _default_experimental_config() -> dict[str, object]:
    return {
        "chat_tool_continue_repair": {
            "enabled": False,
            "max_rounds": 1,
            "max_tokens": 2048,
            "stream_keepalive_seconds": 15,
            "visible_notice_after_seconds": 4,
            "trigger_prefixes": [
                "[terminal command",
                "[terminal_inline",
                "</terminal_inline>",
                "Voy a",
                "Empezando por",
            ],
            "prompt": (
                "Your previous assistant message ended without any tool_calls.\n"
                "You are in a tool-capable agent environment. If the next step requires reading files, editing files, running commands, searching, inspecting state, or using any external capability, you must call one of the available tools instead of describing the action in text.\n"
                "Do not answer with empty visible content. Do not answer with a sentence that only sets up an action and ends with a colon.\n"
                "Available tool names: {tool_names}."
            ),
            "truncated_tool_call_prompt": (
                "Your previous assistant message started a tool_call but it was truncated before the JSON arguments were complete.\n"
                "Retry now with exactly one complete, valid tool_call. Keep the arguments minimal and valid JSON. "
                "Do not stream or repeat partial arguments. Do not include explanatory text before the tool_call.\n"
                "Available tool names: {tool_names}."
            ),
            "include_failed_assistant_message": False,
            "loop_guard": {
                "enabled": True,
                "no_tool_call_max_chars": 0,
                "repeated_tail_min_chars": 3000,
                "repeated_tail_repetitions": 4,
            },
        },
        "chat_last_response_log": {
            "enabled": False,
            "path": "",
            "max_chars": 20000,
            "include_reasoning": False,
            "include_tool_calls": True,
        }
    }


def _normalize_chat_tool_continue_trigger_prefixes(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    prefixes: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in prefixes:
            prefixes.append(text)
    return prefixes


def _normalize_experimental_config(raw: object) -> dict[str, object]:
    cfg = _default_experimental_config()
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key == "chat_tool_continue_repair" and isinstance(value, dict):
                repair = dict(cfg["chat_tool_continue_repair"])
                repair.update(value)
                repair["enabled"] = _as_bool(repair.get("enabled"), False)
                try:
                    repair["max_rounds"] = max(0, int(repair.get("max_rounds", 1)))
                except Exception:
                    repair["max_rounds"] = 1
                try:
                    repair["max_tokens"] = max(0, int(repair.get("max_tokens", 2048)))
                except Exception:
                    repair["max_tokens"] = 2048
                try:
                    repair["stream_keepalive_seconds"] = max(1, int(repair.get("stream_keepalive_seconds", 15)))
                except Exception:
                    repair["stream_keepalive_seconds"] = 15
                try:
                    repair["visible_notice_after_seconds"] = max(0, int(repair.get("visible_notice_after_seconds", 4)))
                except Exception:
                    repair["visible_notice_after_seconds"] = 4
                repair["trigger_prefixes"] = _normalize_chat_tool_continue_trigger_prefixes(repair.get("trigger_prefixes"))
                for prompt_key in ("prompt", "truncated_tool_call_prompt"):
                    prompt_value = repair.get(prompt_key)
                    default_prompt = _default_experimental_config()["chat_tool_continue_repair"].get(prompt_key, "")
                    if not isinstance(prompt_value, str) or not prompt_value.strip():
                        repair[prompt_key] = default_prompt
                    else:
                        repair[prompt_key] = prompt_value
                repair["include_failed_assistant_message"] = _as_bool(repair.get("include_failed_assistant_message"), False)
                loop_guard = repair.get("loop_guard")
                default_loop_guard = _default_experimental_config()["chat_tool_continue_repair"]["loop_guard"]
                if not isinstance(loop_guard, dict):
                    loop_guard = dict(default_loop_guard)
                else:
                    merged_loop_guard = dict(default_loop_guard)
                    merged_loop_guard.update(loop_guard)
                    loop_guard = merged_loop_guard
                loop_guard["enabled"] = _as_bool(loop_guard.get("enabled"), True)
                for lk, dv in (
                    ("no_tool_call_max_chars", 0),
                    ("repeated_tail_min_chars", 3000),
                    ("repeated_tail_repetitions", 4),
                ):
                    try:
                        loop_guard[lk] = max(0, int(loop_guard.get(lk, dv)))
                    except Exception:
                        loop_guard[lk] = dv
                repair["loop_guard"] = loop_guard
                cfg["chat_tool_continue_repair"] = repair
            elif key == "chat_last_response_log" and isinstance(value, dict):
                response_log = dict(cfg["chat_last_response_log"])
                response_log.update(value)
                response_log["enabled"] = _as_bool(response_log.get("enabled"), False)
                response_log["path"] = str(response_log.get("path") or "").strip()
                try:
                    response_log["max_chars"] = max(0, int(response_log.get("max_chars", 20000)))
                except Exception:
                    response_log["max_chars"] = 20000
                response_log["include_reasoning"] = _as_bool(response_log.get("include_reasoning"), False)
                response_log["include_tool_calls"] = _as_bool(response_log.get("include_tool_calls"), True)
                cfg["chat_last_response_log"] = response_log
            elif key not in cfg:
                cfg[key] = value
    return cfg


def _default_api_auth_config() -> dict[str, object]:
    return {"enabled": False, "api_key": ""}


def _default_api_https_config() -> dict[str, object]:
    return {"enabled": False, "cert_file": "", "key_file": ""}


def _normalize_api_auth_config(raw: object) -> dict[str, object]:
    cfg = _default_api_auth_config()
    if isinstance(raw, dict):
        cfg.update({k: v for k, v in raw.items() if k in cfg})
    cfg["enabled"] = _as_bool(cfg.get("enabled"), False)
    cfg["api_key"] = str(cfg.get("api_key") or _env_value("HEIMDALL_GATEWAY_API_KEY", "LLAMACPP_API_KEY", "")).strip()
    return cfg


def _normalize_api_https_config(raw: object) -> dict[str, object]:
    cfg = _default_api_https_config()
    if isinstance(raw, dict):
        cfg.update({k: v for k, v in raw.items() if k in cfg})
    cfg["enabled"] = _as_bool(cfg.get("enabled"), False)
    cfg["cert_file"] = str(cfg.get("cert_file") or "").strip()
    cfg["key_file"] = str(cfg.get("key_file") or "").strip()
    return cfg


def normalize_server_config_payload(payload: dict[str, object]) -> tuple[dict[str, object], bool]:
    changed = False
    if not isinstance(payload, dict):
        payload = {}
        changed = True
    result = dict(payload)
    if "models" in result:
        # Legacy UI/per-model metadata used to live here. catalog.json is now
        # the only editable source for model definitions; config.yaml is
        # generated for llama-swap. Keep conf.json global-only.
        result.pop("models", None)
        changed = True
    raw_defaults = result.get("llama_server_defaults")
    if isinstance(raw_defaults, dict):
        normalized_defaults = normalize_server_overrides(raw_defaults)
        if _migrate_llama_server_defaults(normalized_defaults):
            changed = True
        if normalized_defaults != raw_defaults:
            result["llama_server_defaults"] = normalized_defaults
            changed = True
    else:
        normalized_defaults = {}
        _migrate_llama_server_defaults(normalized_defaults)
        result["llama_server_defaults"] = normalized_defaults
        changed = True
    family_defaults = _normalize_llama_server_family_defaults_config(result.get("llama_server_family_defaults"))
    if _migrate_llama_server_family_defaults(family_defaults):
        changed = True
    if result.get("llama_server_family_defaults") != family_defaults:
        result["llama_server_family_defaults"] = family_defaults
        changed = True
    vllm_config = _normalize_vllm_config(result.get("vllm"))
    if result.get("vllm") != vllm_config:
        result["vllm"] = vllm_config
        changed = True
    raw_replicas = result.get("replicas")
    if not isinstance(raw_replicas, dict):
        result["replicas"] = _default_global_replicas_config()
        changed = True
    else:
        replicas = _default_global_replicas_config()
        replicas.update(raw_replicas)
        placement = str(replicas.get("placement") or "exclusive_gpus").strip().lower().replace("-", "_")
        if placement not in {"exclusive_gpus", "pack_small_models"}:
            placement = "exclusive_gpus"
        replicas["placement"] = placement
        if replicas != raw_replicas:
            result["replicas"] = replicas
            changed = True
    auth_cfg = _normalize_api_auth_config(result.get("api_auth"))
    if result.get("api_auth") != auth_cfg:
        result["api_auth"] = auth_cfg
        changed = True
    https_cfg = _normalize_api_https_config(result.get("api_https"))
    if result.get("api_https") != https_cfg:
        result["api_https"] = https_cfg
        changed = True
    experimental_cfg = _normalize_experimental_config(result.get("experimental"))
    if result.get("experimental") != experimental_cfg:
        result["experimental"] = experimental_cfg
        changed = True
    if "api_ctx_factor" not in result:
        result["api_ctx_factor"] = 0.5
        changed = True
    if "idle_ttl" not in result:
        result["idle_ttl"] = DEFAULT_IDLE_TTL
        changed = True
    if "flatten_namespace_tools" not in result:
        result["flatten_namespace_tools"] = True
        changed = True
    before_meta = json.dumps(result.get("_meta", {}), sort_keys=True, default=str)
    _ensure_server_config_metadata(result)
    after_meta = json.dumps(result.get("_meta", {}), sort_keys=True, default=str)
    if before_meta != after_meta:
        changed = True
    return result, changed


def _load_bundled_llama_server_default_values() -> dict[str, object]:
    defaults_path = Path(__file__).resolve().parent / "bundle" / "llama_server_defaults.yaml"
    try:
        payload = yaml.safe_load(defaults_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    base = payload.get("default")
    if not isinstance(base, dict):
        base = {}
    selected: dict[str, object] = {}
    presets = payload.get("presets")
    if isinstance(presets, dict):
        gpu_count = detect_cuda_device_count()
        for key in (str(gpu_count), gpu_count):
            preset = presets.get(key)
            if isinstance(preset, dict):
                selected = dict(preset)
                break
    merged = dict(base)
    merged.update(selected)
    mtp_defaults = payload.get("mtp_defaults")
    if isinstance(mtp_defaults, dict):
        merged["mtp_defaults"] = dict(mtp_defaults)
    speculative_defaults = payload.get("speculative_defaults")
    if isinstance(speculative_defaults, dict):
        merged.setdefault("speculative_defaults", dict(speculative_defaults))
    return normalize_server_overrides(merged)


def _load_bundled_llama_server_family_default_values() -> dict[str, dict[str, object]]:
    defaults_path = Path(__file__).resolve().parent / "bundle" / "llama_server_defaults.yaml"
    try:
        payload = yaml.safe_load(defaults_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return _normalize_llama_server_family_defaults_config(payload.get("family_defaults"))


def _normalize_llama_server_family_defaults_config(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, dict[str, object]] = {}
    for raw_pattern, raw_defaults in value.items():
        pattern = str(raw_pattern or "").strip()
        if not pattern or not isinstance(raw_defaults, dict):
            continue
        normalized_defaults = normalize_server_overrides(raw_defaults)
        if normalized_defaults:
            normalized[pattern] = normalized_defaults
    return normalized


def _normalize_vllm_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip().lower().replace("-", "_")
        if not key:
            continue
        if isinstance(raw_value, dict):
            normalized[key] = _normalize_vllm_mapping(raw_value)
        else:
            normalized[key] = raw_value
    return normalized


def _normalize_vllm_config(value: object) -> dict[str, object]:
    raw = _normalize_vllm_mapping(value)
    defaults = raw.get("defaults")
    family_defaults = raw.get("family_defaults")
    return {
        "defaults": defaults if isinstance(defaults, dict) else {},
        "family_defaults": family_defaults if isinstance(family_defaults, dict) else {},
    }


def _migrate_llama_server_family_defaults(defaults: dict[str, dict[str, object]]) -> bool:
    if not isinstance(defaults, dict):
        return False
    changed = False
    bundled = _load_bundled_llama_server_family_default_values()
    # The old bundled Qwen profile hard-coded 2048.  It was an installer
    # default, not a user choice, and would override the new global dynamic
    # policy.  Remove only that exact legacy value; other explicit budgets
    # remain untouched.
    for pattern, current in defaults.items():
        if str(pattern).strip().casefold() == "qwen" and isinstance(current, dict):
            if current.get("reasoning_budget") == 2048:
                current.pop("reasoning_budget", None)
                changed = True
    for pattern, pattern_defaults in bundled.items():
        if pattern not in defaults:
            defaults[pattern] = dict(pattern_defaults)
            changed = True
            continue
        current = defaults.get(pattern)
        if not isinstance(current, dict):
            defaults[pattern] = dict(pattern_defaults)
            changed = True
            continue
        for key, value in pattern_defaults.items():
            if key not in current:
                current[key] = value
                changed = True
    return changed


def _migrate_llama_server_defaults(defaults: dict[str, object]) -> bool:
    if not isinstance(defaults, dict):
        return False
    changed = False

    # Retire old installer defaults only when they are still exactly the old
    # values.  Explicit user-tuned mirostat settings remain valid overrides.
    old_default_values = {
        "mirostat": 2,
        "mirostat_ent": 4.5,
        "mirostat_lr": 0.1,
        # Retired global defaults. They are either model-family specific
        # or not valid/safe for all llama.cpp builds. Per-model overrides
        # remain explicit. KV cache defaults are intentionally allowed: the
        # installer now manages cache_type_k/v globally as f16.
        "chat_template_kwargs": None,
        "mul_mat_q": None,
        "grp_attn_n": None,
    }
    for key, old_value in old_default_values.items():
        if key not in defaults:
            continue
        current = defaults.get(key)
        if old_value is None:
            same = True
        else:
            try:
                same = float(current) == float(old_value)
            except Exception:
                same = str(current).strip() == str(old_value)
        if same:
            defaults.pop(key, None)
            changed = True

    use_fitc_value = _normalize_bool_flag(defaults.get("use_fitc"))
    if use_fitc_value is False and "fit_target" in defaults:
        defaults.pop("fit_target", None)
        changed = True

    bundled = _load_bundled_llama_server_default_values()
    for key, value in bundled.items():
        if key not in defaults:
            defaults[key] = value
            changed = True
        elif key == "mtp_defaults" and isinstance(value, dict):
            target = defaults.get(key)
            if not isinstance(target, dict):
                defaults[key] = dict(value)
                changed = True
            else:
                for sub_key, sub_value in value.items():
                    if (
                        sub_key == "spec_draft_n_max"
                        and str(target.get(sub_key, "")).strip() == "2"
                        and str(sub_value).strip() == "3"
                    ):
                        target[sub_key] = sub_value
                        changed = True
                    elif sub_key not in target:
                        target[sub_key] = sub_value
                        changed = True
    return changed


def _ensure_server_config_metadata(payload: dict[str, object]) -> dict[str, object]:
    meta = payload.get("_meta")
    if not isinstance(meta, dict):
        meta = {}
    # _meta is documentation owned by Heimdall Gateway, not user configuration.
    # Do not store example config values here: they look like duplicated active
    # settings and can contradict the real top-level keys.
    meta["purpose"] = "Global Heimdall Gateway settings consumed by CLI/services."
    meta[
        "note"
    ] = "Active settings are top-level keys only. Model definitions live in catalog.json; config.yaml is generated for llama-swap."
    meta.pop("example", None)
    meta["security"] = "Set api_auth.enabled/api_auth.api_key for Bearer or X-API-Key auth. Set api_https.enabled with cert_file/key_file to serve the Heimdall Gateway API over HTTPS."
    meta["service_restart_help"] = {
        "system_mode": f"sudo systemctl restart {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}",
        "user_mode": f"systemctl --user restart {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}",
    }
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

def _service_is_active() -> bool:
    try:
        mode = infer_install_mode()
        if mode == "system":
            r = subprocess.run(
                ["systemctl", "is-active", "--quiet", MANAGER_SERVICE_NAME],
                capture_output=True, timeout=5,
            )
        elif mode == "user":
            r = subprocess.run(
                ["systemctl", "--user", "is-active", "--quiet", MANAGER_SERVICE_NAME],
                capture_output=True, timeout=5,
            )
        else:
            return False
        return r.returncode == 0
    except Exception:
        return False


def manager_unavailable_error(exc: Exception) -> RuntimeError:
    hint = manager_hint()
    if isinstance(exc, FileNotFoundError):
        if _service_is_active():
            mode = infer_install_mode()
            _, _, restart_cmd = service_commands_for_mode(mode)
            hint = (
                "The manager service is running but its socket file is missing.\n"
                "The socket was probably deleted externally.\n"
                "Restart the service to recreate it:\n"
                f"  {restart_cmd}"
            )
    return RuntimeError(
        f"Could not connect to manager: {exc}.\n{hint}"
    )


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
    backend: str = "llama.cpp"
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
    downloaded_at: str = ""
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
class ReplicaConfig:
    enabled: bool = False
    max: int = 1
    gpus_per_replica: int = 1
    placement: str = "exclusive_gpus"
    safety_vram_mib: int = 2048
    max_models_per_gpu: int = 2
    max_pack_fraction: float = 0.35
    sticky_ttl_s: int = 3600


@dataclass
class ReplicaRecord:
    base_model_id: str
    replica_model_id: str
    gpu_set: list[int] = field(default_factory=list)
    status: str = "cold"
    estimated_mib: float | None = None
    actual_mib: float | None = None
    gpu_actual_mib: dict[int, float] = field(default_factory=dict)
    pid: int | None = None
    port: int | None = None
    in_flight: int = 0
    last_used: float = 0.0
    blacklist_until: float = 0.0


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
    if not normalized:
        return default_tensor_split()
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if not parts:
        return default_tensor_split()
    # Preserve the number of entries exactly. A catalog value of 1,1,1 means
    # the user selected three visible GPUs; expanding it to all host GPUs is
    # destructive and causes config.yaml to diverge from catalog.json.
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
    return _env_value("HEIMDALL_GATEWAY_BACKEND", "LLAMACPP_BACKEND", "") == "vllm-beta"


def _normalize_model_backend(value: object, filename: object = "", local_path: object = "") -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    if str(filename or "").strip().lower() == "hf-native":
        return "vllm"
    local = Path(str(local_path or ""))
    if local.is_dir() and not str(filename or "").lower().endswith(".gguf"):
        return "vllm"
    if normalized in {"vllm", "vllm-beta"}:
        return "vllm"
    if normalized in {"llama.cpp", "llama-cpp", "llamacpp"}:
        return "llama.cpp"
    return "llama.cpp"


def _model_backend(model: ManagedModel) -> str:
    return _normalize_model_backend(
        getattr(model, "backend", None),
        getattr(model, "filename", None),
        getattr(model, "local_path", None),
    )


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
            Path.home() / ".local" / "opt" / PRODUCT_SLUG / "cuda" / "lib",
            Path.home() / ".local" / "opt" / PRODUCT_SLUG / "nccl" / "lib",
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
        if key in {"mul_mat_q", "use_fitc", "swa_full"}:
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
            "checkpoint_min_step",
            "checkpoint_every_n_tokens",
            "cache_ram",
            "n_cpu_moe",
            "top_k",
            "predict",
            "image_min_tokens",
        }:
            try:
                normalized[key] = int(raw_val)
            except (TypeError, ValueError):
                continue
            continue
        if key == "reasoning_budget":
            if isinstance(raw_val, str) and raw_val.strip().casefold() in {
                REASONING_BUDGET_HALF_CONTEXT,
                "half_ctx",
                "auto",
            }:
                normalized[key] = REASONING_BUDGET_HALF_CONTEXT
                continue
            try:
                normalized[key] = int(raw_val)
            except (TypeError, ValueError):
                continue
            continue
        if key in {"mirostat_ent", "mirostat_lr", "draft_p_min", "defrag_threshold", "top_p", "min_p", "repeat_penalty", "presence_penalty"}:
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
        if key in {"kv_offload", "cont_batching", "op_offload", "cpu_moe", "kv_unified", "cache_idle_slots", "direct_io", "swa_full", "cache_prompt"}:
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
        if key == "reasoning":
            bool_val = _normalize_bool_flag(raw_val)
            if bool_val is not None:
                normalized[key] = "on" if bool_val else "off"
            else:
                normalized[key] = str(raw_val).strip()
            continue
        if key in {"split_mode", "cache_type_k", "cache_type_v", "host", "model_draft", "hf_repo_draft", "reasoning_format", "reasoning_budget_message", "chat_template_file", "chat_template", "device", "chat_template_kwargs"}:
            if key in {"cache_type_k", "cache_type_v"}:
                normalized_cache_type = _normalize_cache_type_value(raw_val)
                if normalized_cache_type is not None:
                    normalized[key] = normalized_cache_type
                continue
            if key == "chat_template_kwargs":
                if isinstance(raw_val, dict):
                    normalized[key] = json.dumps(raw_val, ensure_ascii=False, separators=(",", ":"))
                else:
                    sval = str(raw_val).strip()
                    # Be forgiving with common config-style booleans inside this
                    # JSON-valued llama.cpp flag. llama-server expects valid JSON;
                    # strings such as {"preserve_thinking":off} make it exit.
                    sval = re.sub(r'(:\s*)(on)(\s*[,}])', lambda match: f'{match.group(1)}true{match.group(3)}', sval, flags=re.IGNORECASE)
                    sval = re.sub(r'(:\s*)(off)(\s*[,}])', lambda match: f'{match.group(1)}false{match.group(3)}', sval, flags=re.IGNORECASE)
                    normalized[key] = sval
            else:
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
        model_field_names = {f.name for f in fields(ManagedModel)}
        items = []
        for raw_item in payload:
            if not isinstance(raw_item, dict):
                raise TypeError(f"expected object entries, got {type(raw_item).__name__}")
            known = {k: v for k, v in raw_item.items() if k in model_field_names}
            extra = {k: v for k, v in raw_item.items() if k not in model_field_names}
            if extra:
                overrides = dict(known.get("server_overrides") or {})
                managed_aliases = {
                    k: v
                    for k, v in extra.items()
                    if "-" in str(k) and str(k).strip().lower().replace("-", "_") in model_field_names
                }
                # Unknown top-level catalog keys are treated as llama.cpp
                # server overrides. This keeps manual edits like
                # `"parallel": 2` backward-compatible and avoids rejecting
                # new llama.cpp flags before the dataclass schema knows them.
                # Exception: dash aliases of managed model fields, e.g.
                # `tensor-split`, are intentionally not migrated silently.
                # They are ignored here and reported by catalog-key warnings;
                # raw llama.cpp flags belong under server_overrides.
                for extra_key, extra_value in extra.items():
                    if extra_key in managed_aliases:
                        continue
                    overrides[extra_key] = extra_value
                if overrides:
                    known["server_overrides"] = overrides
            items.append(ManagedModel(**known))
    except Exception as exc:
        _clear_catalog_cache(path)
        return [], f"Catalog {path} has invalid entries: {exc}"
    changed = False
    # Load global defaults for normalization context.
    try:
        args = argparse.Namespace(server_config=server_config_path) if server_config_path is not None else None
        server_defaults = resolve_llama_server_defaults(args)
    except Exception:
        server_defaults = {}

    for item in items:
        normalized_backend = _normalize_model_backend(item.backend, item.filename, item.local_path)
        if normalized_backend != item.backend:
            item.backend = normalized_backend
            changed = True
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

        # Preserve catalog tensor_split exactly as the user wrote it.
        # Expanding 1,1,1 to all currently visible GPUs destroys manual
        # placement and makes update/reinstall appear to ignore catalog.json.
        normalized = normalize_tensor_split(item.tensor_split)
        if normalized != item.tensor_split:
            item.tensor_split = normalized
            changed = True
        raw_auto_performance = None
        if isinstance(item.server_overrides, dict):
            raw_auto_performance = item.server_overrides.get("auto_performance")
        normalized_overrides = normalize_server_overrides(item.server_overrides)
        override_tensor_split = normalized_overrides.pop("tensor_split", None)
        if override_tensor_split is not None:
            migrated_tensor_split = normalize_tensor_split(str(override_tensor_split))
            if item.tensor_split != migrated_tensor_split:
                item.tensor_split = migrated_tensor_split
            changed = True

        # Keep explicit per-model overrides from the catalog. Loading a catalog
        # must not silently remove a user edit such as {"parallel": 2} just
        # because it currently matches a global default; otherwise `update` can
        # appear to ignore catalog.json edits and can rewrite them away.
        normalized_overrides_preserved = dict(normalized_overrides)
        if isinstance(raw_auto_performance, dict):
            # Persistent auto-performance metadata belongs in catalog config,
            # but normalize_server_overrides intentionally strips it so it can
            # never be emitted as a llama-server flag.
            normalized_overrides_preserved["auto_performance"] = raw_auto_performance

        if normalized_overrides_preserved != item.server_overrides:
            item.server_overrides = normalized_overrides_preserved
            changed = True
    if changed:
        save_catalog(path, items)
    cached_items = [replace(item) for item in items]
    CATALOG_CACHE[cache_key] = (signature[0], signature[1], cached_items)
    return [replace(item) for item in cached_items], None


def _model_download_sort_timestamp(model: ManagedModel) -> float:
    raw = str(getattr(model, "downloaded_at", "") or "").strip()
    if raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except Exception:
            pass
    local_path = str(getattr(model, "local_path", "") or "").strip()
    if local_path:
        try:
            return Path(local_path).stat().st_mtime
        except Exception:
            pass
    return 0.0


def sort_catalog_for_json(models: list[ManagedModel]) -> list[ManagedModel]:
    indexed = list(enumerate(models))
    indexed.sort(key=lambda pair: (_model_download_sort_timestamp(pair[1]), -pair[0]), reverse=True)
    return [item for _, item in indexed]

def save_catalog(path: Path, models: list[ManagedModel]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    effective_idle_ttl = get_configured_idle_ttl(DEFAULT_CONFIG_PATH, resolve_idle_ttl())
    serialized: list[dict[str, object]] = []
    for model in sort_catalog_for_json(models):
        payload = asdict(model)
        if not str(payload.get("downloaded_at") or "").strip():
            ts = _model_download_sort_timestamp(model)
            if ts > 0:
                payload["downloaded_at"] = datetime.fromtimestamp(ts, timezone.utc).isoformat()
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

def _normalize_quant_code(value: str | None) -> str:
    token = str(value or "").strip().lower()
    token = re.sub(r"\.gguf$", "", token)
    token = re.sub(r"[-._]?\d{5}-of-\d{5}$", "", token)
    token = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
    return token


def choose_gguf_file(api, repo_id, quant, explicit_file, token):
    info = api.model_info(repo_id=repo_id, token=token)
    all_files = sorted(s.rfilename for s in info.siblings if s.rfilename and s.rfilename.lower().endswith(".gguf"))
    if not all_files:
        if _env_value("HEIMDALL_GATEWAY_DEBUG", "DEBUG_LLAMACPP", ""):
            print(f"DEBUG: No GGUF files found in {repo_id}; treating it as a native HuggingFace model.", file=sys.stderr)
        return None
    
    # Exclude mmproj files from main model search (they're selected separately)
    files = [f for f in all_files if "mmproj" not in f.lower()]
    if not files: files = all_files  # Fallback if all are mmproj
    
    if explicit_file:
        if explicit_file not in files: raise RuntimeError(f"File {explicit_file} not found.")
        return explicit_file
    
    if quant:
        ql = _normalize_quant_code(quant)
        
        # Extract the quantization token from the file basename.  Sharded GGUFs
        # commonly look like `BF16/model-BF16-00001-of-00002.gguf`; stripping the
        # shard suffix first is required, otherwise the extracted token becomes
        # `00002` and an explicit `:BF16` request falls through to the default
        # Q4_K_M selection.
        def extract_quant_code(filename):
            stem = Path(filename).stem  # Remove directory and .gguf
            stem = re.sub(r"[-._]?\d{5}-of-\d{5}$", "", stem, flags=re.IGNORECASE)
            if "/" in filename:
                parent = Path(filename).parent.name
                if parent and _normalize_quant_code(parent) == ql:
                    return ql
            # Prefer the longest known quant-like suffix.  Splitting only on '-'
            # loses UD-Q5_K_XL as UD_Q5_K_XL, so normalize separators instead.
            normalized_stem = _normalize_quant_code(stem)
            pieces = normalized_stem.split("_")
            best = ""
            for start in range(len(pieces)):
                candidate = "_".join(pieces[start:])
                # A quant code must start with q+digit, f+digit, bf+digit,
                # mxfp, iq+digit, or ud_ (avoid matching model name like
                # "qwen3..." where q is not followed by a digit).
                if not any(ch.isdigit() for ch in candidate):
                    continue
                if re.match(r"^(ud_)?(iq|q)\d", candidate):
                    if len(candidate) > len(best):
                        best = candidate
                elif re.match(r"^(ud_)?(bf)?f\d", candidate):
                    if len(candidate) > len(best):
                        best = candidate
                elif re.match(r"^(ud_)?mxfp", candidate):
                    if len(candidate) > len(best):
                        best = candidate
            return best or (pieces[-1] if pieces else "")
        
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


def _is_mtp_drafter_filename(filename: str) -> bool:
    path = str(filename or "").strip()
    if not path.lower().endswith(".gguf"):
        return False
    parts = Path(path).parts
    basename = Path(path).name.lower()
    parent_names = {part.lower() for part in parts[:-1]}
    return basename.startswith("mtp-") or ("mtp" in parent_names and "mtp" in basename)




def _is_integrated_mtp_model_filename(filename: str | None) -> bool:
    name = Path(str(filename or "")).name.lower()
    if not name.endswith(".gguf"):
        return False
    stem = re.sub(r"(?i)\.gguf$", "", name)
    return bool(re.search(r"(^|[-_.])mtp($|[-_.])", stem)) and not name.startswith("mtp-")


def _looks_like_integrated_mtp_model(
    repo_id: str | None = None,
    filename: str | None = None,
    model_id: str | None = None,
    local_path: str | Path | None = None,
) -> bool:
    """Detect GGUFs that embed MTP even when the selected filename omits MTP.

    Some repos are named `*-MTP-GGUF` while the main file itself is named only
    by model and quant, e.g. `Qwen3.6-27B-UD-Q5_K_XL.gguf`.  llama.cpp still
    needs the draft-mtp flags for these integrated MTP files.
    """
    if _is_integrated_mtp_model_filename(filename):
        return True
    if not str(filename or local_path or "").lower().endswith(".gguf"):
        return False
    haystack = " ".join(str(x or "") for x in (repo_id, model_id, local_path)).lower()
    return bool(re.search(r"(^|[/\-_.])mtp($|[/\-_.])", haystack))


def _should_probe_repo_for_mtp_drafter(repo_id: str, filename: str, model_id: str = "", draft_path: str = "") -> bool:
    haystack = " ".join([repo_id, filename, model_id, draft_path]).lower()
    if "mtp" in haystack or str(draft_path or "").strip():
        return True
    # Some Unsloth Gemma 4 GGUF repos ship an MTP drafter (`mtp-gemma-4-31B-it.gguf`)
    # while the repo and selected main model filenames do not contain "mtp".
    # Heimdall Gateway launches local GGUFs with --model, so it must explicitly
    # discover/download this drafter instead of relying on llama.cpp -hf magic.
    return "unsloth/gemma-4-31b-it" in haystack and str(filename or "").strip().lower().endswith(".gguf")


def _detect_mtp_drafter_file(api, repo_id: str, token: str | None, selected_file: str | None = None) -> str | None:
    try:
        info = api.model_info(repo_id=repo_id, token=token)
    except Exception:
        info = None
    selected = str(selected_file or "").strip()
    siblings = [s.rfilename for s in getattr(info, "siblings", []) or [] if getattr(s, "rfilename", None)] if info is not None else []
    if not siblings:
        try:
            for item in api.list_repo_tree(repo_id=repo_id, recursive=True, token=token):
                item_type = type(item).__name__.lower()
                if "folder" in item_type:
                    continue
                name = ""
                for attr in ("rfilename", "path", "name"):
                    value = getattr(item, attr, None)
                    if isinstance(value, str) and value:
                        name = value
                        break
                if name:
                    siblings.append(name)
        except Exception:
            return None
    candidates = sorted(f for f in siblings if f != selected and _is_mtp_drafter_filename(str(f)))
    if not candidates:
        return None

    # Prefer the model-card default: root-level `mtp-*.gguf`.  The MTP/
    # directory usually contains alternate precisions for explicit usage.
    root_candidates = [f for f in candidates if "/" not in f]
    if root_candidates:
        return sorted(root_candidates, key=lambda f: (len(Path(f).name), f.lower()))[0]
    return sorted(candidates, key=lambda f: (0 if Path(f).parent.name.lower() == "mtp" else 1, len(f), f.lower()))[0]


def _apply_mtp_server_overrides(
    overrides: dict[str, object] | None,
    mtp_local_path: str | Path | None,
    server_defaults: dict[str, object] | None = None,
) -> tuple[dict[str, object], bool]:
    result = dict(normalize_server_overrides(overrides or {}))
    changed = False
    mtp_cfg: dict[str, object] = {}
    if isinstance(server_defaults, dict):
        mtp_cfg_raw = server_defaults.get("mtp_defaults")
        if isinstance(mtp_cfg_raw, dict):
            for k, v in mtp_cfg_raw.items():
                if v is not None:
                    mtp_cfg[str(k).strip().lower().replace("-", "_")] = v
    desired: dict[str, object] = {
        "spec_type": "draft-mtp",
        "spec_draft_n_max": mtp_cfg.get("spec_draft_n_max", 2),
        "spec_draft_n_min": mtp_cfg.get("spec_draft_n_min", 0),
        "spec_draft_p_min": mtp_cfg.get("spec_draft_p_min", 0.75),
    }
    if mtp_local_path is not None and str(mtp_local_path).strip():
        desired["model_draft"] = str(mtp_local_path)
    if "image_min_tokens" in mtp_cfg:
        desired["image_min_tokens"] = mtp_cfg.get("image_min_tokens")
    else:
        desired["image_min_tokens"] = 1024
    if str(result.get("model_draft") or "").strip():
        # `draft` is a legacy alias for draft token count and can render as a
        # second --spec-draft-n-max/--draft-max.  MTP orchestration owns
        # spec_draft_n_max, so never keep both.
        if "draft" in result:
            result.pop("draft", None)
            changed = True
    for key, value in desired.items():
        current = result.get(key)
        if key == "model_draft" and str(current or "").strip() != str(value).strip():
            result[key] = value
            changed = True
        elif key == "spec_draft_n_max" and str(result.get(key, "")).strip() in {"2", "4", "16"} and str(value).strip() == "3":
            result[key] = value
            changed = True
        elif key not in result:
            result[key] = value
            changed = True
    return result, changed


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


def _as_bool(value: object, default: bool = False) -> bool:
    parsed = _normalize_bool_flag(value)
    return default if parsed is None else parsed


def resolve_global_replica_config(args = None) -> dict[str, object]:
    payload = _load_server_config_payload(args)
    raw = payload.get("replicas")
    return dict(raw) if isinstance(raw, dict) else {}


def get_model_replica_config(model: ManagedModel, global_raw: dict[str, object] | None = None, total_gpus: int | None = None) -> ReplicaConfig:
    explicit_global_config = global_raw is not None
    global_config: dict[str, object] = dict(global_raw if global_raw is not None else resolve_global_replica_config())
    raw: dict[str, object] = dict(global_config)
    model_raw: dict[str, object] = {}
    try:
        candidate = (model.server_overrides or {}).get("replicas")
        if isinstance(candidate, dict):
            model_raw = dict(candidate)
    except Exception:
        model_raw = {}
    # Activation is global-first. Old per-model replicas.enabled values are
    # only honored when there is no global enabled key, so stale per-model
    # enabled=false cannot silently override conf.json enabled=true.
    model_enabled = model_raw.pop("enabled", None)
    raw.update(model_raw)
    cfg = ReplicaConfig()
    if explicit_global_config and "enabled" in global_config:
        cfg.enabled = _as_bool(global_config.get("enabled"), False)
    elif model_enabled is not None:
        cfg.enabled = _as_bool(model_enabled, False)
    else:
        cfg.enabled = _as_bool(global_config.get("enabled"), False)
    try:
        cfg.gpus_per_replica = max(1, int(raw.get("gpus_per_replica", _infer_base_gpu_count(model))))
    except Exception:
        cfg.gpus_per_replica = _infer_base_gpu_count(model)
    detected_total_gpus = total_gpus if total_gpus is not None else detect_cuda_device_count()
    auto_max = max(1, detected_total_gpus // max(1, cfg.gpus_per_replica)) if detected_total_gpus > 0 else cfg.max
    raw_max = raw.get("max", "auto" if cfg.enabled else cfg.max)
    try:
        if isinstance(raw_max, str) and raw_max.strip().lower() in {"auto", "", "none"}:
            cfg.max = auto_max
        else:
            cfg.max = max(1, int(raw_max))
    except Exception:
        cfg.max = auto_max
    placement = str(raw.get("placement", cfg.placement) or cfg.placement).strip().lower().replace("-", "_")
    if placement in {"exclusive_gpus", "pack_small_models"}:
        cfg.placement = placement
    try:
        cfg.safety_vram_mib = max(0, int(raw.get("safety_vram_mib", cfg.safety_vram_mib)))
    except Exception:
        pass
    try:
        cfg.max_models_per_gpu = max(1, int(raw.get("max_models_per_gpu", cfg.max_models_per_gpu)))
    except Exception:
        pass
    try:
        cfg.max_pack_fraction = max(0.01, min(1.0, float(raw.get("max_pack_fraction", cfg.max_pack_fraction))))
    except Exception:
        pass
    try:
        cfg.sticky_ttl_s = max(60, int(raw.get("sticky_ttl_s", cfg.sticky_ttl_s)))
    except Exception:
        pass
    return cfg


def replica_model_id(base_model_id: str, index: int) -> str:
    return f"{base_model_id}__replica_{index}"


def is_replica_model_id(model_id: str) -> bool:
    return bool(re.search(r"__replica_\d+$", str(model_id or "")))


def replica_base_model_id(model_id: str) -> str:
    return re.sub(r"__replica_\d+$", "", str(model_id or ""))


def _infer_base_gpu_count(model: ManagedModel) -> int:
    try:
        raw = (model.server_overrides or {}).get("gpu_count")
        if raw is not None:
            return max(1, int(raw))
    except Exception:
        pass
    try:
        ts = str((model.server_overrides or {}).get("tensor_split") or model.tensor_split or "1")
        parts = [p for p in ts.split(",") if p.strip()]
        return max(1, len(parts))
    except Exception:
        return 1


def _replica_gpu_sets(model: ManagedModel, cfg: ReplicaConfig, total_gpus: int | None = None) -> list[list[int]]:
    if not cfg.enabled or cfg.max <= 0:
        return []
    total = total_gpus if total_gpus is not None else detect_cuda_device_count()
    if total <= 0:
        total = max(cfg.gpus_per_replica, _infer_base_gpu_count(model))
    gpr = max(1, cfg.gpus_per_replica or _infer_base_gpu_count(model))
    sets: list[list[int]] = []
    if cfg.placement == "exclusive_gpus":
        base_gpu_count = max(0, _infer_base_gpu_count(model))
        if total <= base_gpu_count:
            return []
        for start in range(base_gpu_count, total, gpr):
            gpu_set = list(range(start, min(start + gpr, total)))
            if len(gpu_set) == gpr:
                sets.append(gpu_set)
            if len(sets) >= cfg.max:
                break
    else:
        for idx in range(cfg.max):
            if gpr == 1:
                sets.append([idx % total])
            else:
                start = (idx * gpr) % total
                gpu_set = [(start + off) % total for off in range(gpr)]
                if len(set(gpu_set)) == gpr:
                    sets.append(gpu_set)
    return sets[: cfg.max]


def _replica_overrides(base: ManagedModel, gpu_set: list[int]) -> dict[str, object]:
    overrides = dict(base.server_overrides or {})
    overrides.pop("replicas", None)
    if gpu_set:
        visible_count = len(gpu_set)
        overrides.pop("tensor_split", None)
        overrides["__replica_tensor_split"] = ",".join(["1"] * visible_count)
        overrides["main_gpu"] = 0
    return overrides


def build_replica_model(base: ManagedModel, index: int, gpu_set: list[int]) -> ManagedModel:
    return replace(
        base,
        model_id=replica_model_id(base.model_id, index),
        aliases=[],
        description=f"internal replica {index} of {base.model_id}",
        server_overrides=_replica_overrides(base, gpu_set),
    )


def _command_with_cuda_visible_devices(cmd: list[str], gpu_set: list[int]) -> list[str]:
    if not gpu_set:
        return cmd
    return ["/usr/bin/env", f"CUDA_VISIBLE_DEVICES={','.join(str(g) for g in gpu_set)}", *cmd]


def iter_catalog_with_replicas(catalog: list[ManagedModel], global_replica_config: dict[str, object] | None = None) -> list[tuple[ManagedModel, str | None, list[int] | None]]:
    """Return (model, public_base_model_id, gpu_set). public_base_model_id is set for internal replicas."""
    result: list[tuple[ManagedModel, str | None, list[int]]] = []
    for model in catalog:
        result.append((model, None, None))
        cfg = get_model_replica_config(model, global_replica_config)
        if not cfg.enabled:
            continue
        for idx, gpu_set in enumerate(_replica_gpu_sets(model, cfg)):
            result.append((build_replica_model(model, idx, gpu_set), model.model_id, gpu_set))
    return result


def iter_catalog_base_models(catalog: list[ManagedModel]) -> list[tuple[ManagedModel, str | None, list[int] | None]]:
    """Return only public/base catalog models for the static llama-swap config."""
    return [(model, None, None) for model in catalog]


def summarize_configured_replicas(catalog: list[ManagedModel], global_replica_config: dict[str, object] | None = None) -> list[str]:
    lines: list[str] = []
    total_gpus = detect_cuda_device_count()
    for model in catalog:
        cfg = get_model_replica_config(model, global_replica_config, total_gpus=total_gpus)
        if not cfg.enabled:
            continue
        gpu_sets = _replica_gpu_sets(model, cfg, total_gpus=total_gpus)
        if gpu_sets:
            lines.append(
                f"{model.model_id}: {len(gpu_sets)} replica(s), gpus_per_replica={cfg.gpus_per_replica}, gpu_sets={gpu_sets}"
            )
        else:
            lines.append(
                f"{model.model_id}: replicas enabled but no GPU set generated (gpus_per_replica={cfg.gpus_per_replica}, max={cfg.max})"
            )
    return lines


def _get_model_size_mib(m: ManagedModel) -> float:
    """Estimate model size in MiB for GGUF files and HF snapshots."""
    try:
        path = Path(m.local_path)
        if path.is_dir():
            return sum(
                file_path.stat().st_size
                for file_path in path.rglob("*")
                if file_path.is_file()
            ) / (1024 * 1024)
        return float(path.stat().st_size) / (1024 * 1024)
    except Exception:
        return 0.0


def _is_embedding_model(m: ManagedModel) -> bool:
    """Heuristic to identify if a model is an embedding model."""
    low_id = m.model_id.lower()
    # Check common embedding model names/prefixes
    if any(k in low_id for k in ["embedding", "bge-", "gte-", "snowflake-", "nomic-"]):
        return True
    # Check overrides for --embedding flag
    for v in m.server_overrides.values():
        if isinstance(v, str) and "--embedding" in v:
            return True
    # Check load capabilities if populated
    if "embedding" in (m.load_capabilities or []):
        return True
    return False


def _is_small_model(m: ManagedModel) -> bool:
    """Heuristic to identify if a model is 'small' and thus packable with others."""
    # Heuristic: < 4GB is considered small and likely to fit alongside a main model.
    return _get_model_size_mib(m) < 4096.0


def _calculate_llama_swap_matrix(models_info: list[dict]) -> dict[str, object]:
    """
    Calculate the llama-swap matrix configuration to allow maximum concurrency.
    
    models_info is a list of dicts: {id, gpu_set, is_embedding, is_small, size_mib}
    """
    if not models_info:
        return {}

    # llama-swap's matrix parser expects `sets` to reference variable IDs, not
    # arbitrary model names. Real model IDs commonly contain dots, colons or
    # other characters that are not valid DSL atoms, so expose deterministic
    # short vars and use those vars consistently in sets and evict_costs.
    sorted_ids = sorted({str(m["id"]) for m in models_info if str(m.get("id") or "").strip()})
    id_to_var = {model_id: f"m{idx}" for idx, model_id in enumerate(sorted_ids)}
    vars_map = {var_id: model_id for model_id, var_id in id_to_var.items()}

    embeddings = [m["id"] for m in models_info if m["is_embedding"]]
    smalls = [m["id"] for m in models_info if m["is_small"] and not m["is_embedding"]]
    larges = [m for m in models_info if not m["is_small"] and not m["is_embedding"]]

    # Packables are models we assume can always run together or alongside a large model.
    packables = sorted(embeddings + smalls)
    
    # We want to group large models into sets that don't conflict on GPUs.
    # conflicting sets use a greedy approach to find maximal independent sets.
    large_groups: list[list[str]] = []
    
    # If a model has an empty gpu_set but is large, assume it uses all GPUs to be safe.
    # We approximate all GPUs by collecting all known GPU IDs from all models.
    all_known_gpus = set()
    for m in models_info:
        all_known_gpus.update(m["gpu_set"])
    
    # Normalize gpu_set for collision checking
    for m in larges:
        if not m["gpu_set"]:
            m["_effective_gpu_set"] = set(all_known_gpus)
        else:
            m["_effective_gpu_set"] = set(m["gpu_set"])
            
    sorted_larges = sorted(larges, key=lambda x: len(x["_effective_gpu_set"]), reverse=True)
    
    for l in sorted_larges:
        added = False
        l_gpu_set = l["_effective_gpu_set"]
        
        for group in large_groups:
            conflict = False
            for existing_id in group:
                existing = next(m for m in models_info if m["id"] == existing_id)
                # Check for GPU overlap
                existing_gpus = existing.get("_effective_gpu_set", set(existing["gpu_set"]))
                if not existing_gpus and not existing["is_small"] and not existing["is_embedding"]:
                    existing_gpus = set(all_known_gpus)
                if l_gpu_set & existing_gpus:
                    conflict = True
                    break
            if not conflict:
                group.append(l["id"])
                added = True
                break
        if not added:
            large_groups.append([l["id"]])

            
    # Final sets: each large group + all packables

    matrix_sets = {}
    for idx, group in enumerate(large_groups):
        group_vars = [id_to_var[model_id] for model_id in sorted(group + packables) if model_id in id_to_var]
        if group_vars:
            matrix_sets[f"group_{idx}"] = " & ".join(group_vars)

    # If there are no large models, just one set of packables
    if not large_groups and packables:
        packable_vars = [id_to_var[model_id] for model_id in packables if model_id in id_to_var]
        if packable_vars:
            matrix_sets["packables"] = " & ".join(packable_vars)

    # Evict costs based on size (harder to evict larger models)
    evict_costs = {id_to_var[m["id"]]: max(1, int(m["size_mib"])) for m in models_info if m["id"] in id_to_var}
    
    return {
        "vars": vars_map,
        "sets": matrix_sets,
        "evict_costs": evict_costs
    }


def render_llamaswap_config(
    catalog,
    path,
    server_path,
    start_port,
    idle_ttl=DEFAULT_IDLE_TTL,
    server_defaults: dict[str, object] | None = None,
    replica_defaults: dict[str, object] | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "healthCheckTimeout": 600,
        "logLevel": "info",
        "logToStdout": "proxy",
        "startPort": start_port,
        "sendLoadingState": False,
        "includeAliasesInList": True,
        "models": {},
    }
    resolved_defaults = normalize_server_overrides(server_defaults or resolve_llama_server_defaults())
    resolved_vllm_defaults = resolve_vllm_defaults()

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

    replica_group_members: list[str] = []
    models_info_for_matrix: list[dict] = []
    for m, public_base_model_id, replica_gpu_set in sorted(iter_catalog_base_models(catalog), key=lambda item: item[0].model_id):
        # Respect user's preference: do NOT create on-disk GGUF variants.
        # Use the original model file for rendering; duplicate catalog
        # entries will reference the same local_path. Note: internal replicas
        # intentionally reference the same GGUF but publish distinct model ids.
        use_model = m

        cmd = build_llama_server_command(
            use_model,
            server_path,
            port="${PORT}",
            server_defaults=resolved_defaults,
            vllm_defaults=resolved_vllm_defaults,
        )
        if public_base_model_id is not None:
            cmd = _command_with_cuda_visible_devices(cmd, replica_gpu_set)
            replica_group_members.append(m.model_id)
        
        # Collect info for the concurrency matrix
        models_info_for_matrix.append({
            "id": m.model_id,
            "gpu_set": replica_gpu_set if replica_gpu_set is not None else list(range(detect_cuda_device_count())),
            "is_embedding": _is_embedding_model(m),
            "is_small": _is_small_model(m),
            "size_mib": _get_model_size_mib(m)
        })

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
        if public_base_model_id is not None:
            data["models"][m.model_id]["metadata"] = {"internal_replica_of": public_base_model_id}
        if public_base_model_id is None and m.aliases:
            data["models"][m.model_id]["aliases"] = m.aliases
        if m.description:
            data["models"][m.model_id]["description"] = m.description
    
    # Generate the concurrency matrix
    matrix_config = _calculate_llama_swap_matrix(models_info_for_matrix)
    if matrix_config:
        data["matrix"] = matrix_config

    tmp = path.with_suffix(".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        f.write(LLAMASWAP_CONFIG_HEADER)
        yaml.safe_dump(data, f, sort_keys=False)
    tmp.replace(path)


def _model_info_for_matrix_entry(model_id: str, model_entry: dict, catalog_by_id: dict[str, ManagedModel]) -> dict:
    metadata = model_entry.get("metadata") if isinstance(model_entry, dict) else {}
    base_id = metadata.get("internal_replica_of") if isinstance(metadata, dict) else None
    model = catalog_by_id.get(str(base_id or model_id))
    cmd = str(model_entry.get("cmd") or "") if isinstance(model_entry, dict) else ""
    gpu_set: list[int] = []
    match = re.search(r"CUDA_VISIBLE_DEVICES=([0-9,]+)", cmd)
    if match:
        gpu_set = [int(part) for part in match.group(1).split(",") if part.strip().isdigit()]
    elif model is not None:
        gpu_set = list(range(max(0, detect_cuda_device_count())))
    return {
        "id": model_id,
        "gpu_set": gpu_set,
        "is_embedding": _is_embedding_model(model) if model is not None else False,
        "is_small": _is_small_model(model) if model is not None else False,
        "size_mib": _get_model_size_mib(model) if model is not None else 1,
    }


def _recalculate_llamaswap_matrix_from_config(data: dict, catalog: list[ManagedModel]) -> None:
    models = data.get("models")
    if not isinstance(models, dict):
        data.pop("matrix", None)
        return
    catalog_by_id = {model.model_id: model for model in catalog}
    infos = [_model_info_for_matrix_entry(str(model_id), entry, catalog_by_id) for model_id, entry in models.items() if isinstance(entry, dict)]
    matrix = _calculate_llama_swap_matrix(infos)
    if matrix:
        data["matrix"] = matrix
    else:
        data.pop("matrix", None)


def ensure_replica_route_in_llamaswap_config(
    base_model: ManagedModel,
    replica_index: int,
    gpu_set: list[int],
    catalog: list[ManagedModel],
    config_path: Path | str,
    server_path: Path | str,
    idle_ttl: int,
    server_defaults: dict[str, object] | None = None,
) -> str:
    """Create one internal replica route in config.yaml and let llama-swap --watch-config reload it."""
    path = Path(config_path)
    rid = replica_model_id(base_model.model_id, replica_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("healthCheckTimeout", 600)
    data.setdefault("logLevel", "info")
    data.setdefault("logToStdout", "proxy")
    data.setdefault("sendLoadingState", False)
    data.setdefault("includeAliasesInList", True)
    models = data.setdefault("models", {})
    if not isinstance(models, dict):
        models = {}
        data["models"] = models

    if rid not in models:
        replica = build_replica_model(base_model, replica_index, gpu_set)
        resolved_defaults = normalize_server_overrides(server_defaults or resolve_llama_server_defaults())
        cmd = build_llama_server_command(
            replica,
            Path(server_path),
            port="${PORT}",
            server_defaults=resolved_defaults,
            vllm_defaults=resolve_vllm_defaults(),
        )
        cmd = _command_with_cuda_visible_devices(cmd, gpu_set)
        models[rid] = {
            "cmd": " ".join(shell_quote(part) for part in cmd),
            "checkEndpoint": "/health",
            "ttl": int(idle_ttl),
            "metadata": {"internal_replica_of": base_model.model_id},
            "description": replica.description,
        }
        _recalculate_llamaswap_matrix_from_config(data, catalog)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.write(LLAMASWAP_CONFIG_HEADER)
            yaml.safe_dump(data, f, sort_keys=False)
        tmp.replace(path)
        log_api_event("replica_route_added", {"model": base_model.model_id, "replica": rid, "gpu_set": gpu_set, "config_path": str(path)})
    return rid


def wait_for_published_model_id(model_id: str, host: str, port: int, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if model_id in get_published_model_ids(host, port):
            return True
        time.sleep(0.15)
    return model_id in get_published_model_ids(host, port)


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
        configured_path = Path(args.server_config)
        candidate_paths.append(configured_path)
        # Also check for conf.json fallback in the same directory (installer
        # writes conf.json, but --server-config may point to a different file).
        conf_fallback = configured_path.with_name(ALTERNATE_SERVER_CONFIG_BASENAME)
        if conf_fallback != configured_path:
            candidate_paths.append(conf_fallback)
    else:
        default_path = Path(DEFAULT_SERVER_CONFIG_PATH)
        # Prefer new conf.json for new installs, fall back to legacy name
        candidate_paths.append(default_path.with_name(ALTERNATE_SERVER_CONFIG_BASENAME))
        candidate_paths.append(default_path)

    for p in candidate_paths:
        try:
            if not p.exists():
                continue
            payload = _json_loads_allow_comments(
                p.read_text(encoding="utf-8"),
                path_desc=str(p),
            )
            if isinstance(payload, dict):
                normalized, changed = normalize_server_config_payload(payload)
                if changed:
                    try:
                        p.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
                    except Exception:
                        pass
                return normalized
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
    payload = _load_server_config_payload(args)
    defaults = normalize_server_overrides(payload.get("llama_server_defaults"))
    family_defaults = _normalize_llama_server_family_defaults_config(payload.get("llama_server_family_defaults"))
    if family_defaults:
        defaults["__family_defaults"] = family_defaults
    return defaults


def resolve_vllm_defaults(args = None) -> dict[str, object]:
    payload = _load_server_config_payload(args)
    config = _normalize_vllm_config(payload.get("vllm"))
    bundled_path = Path(__file__).resolve().parent / "bundle" / "llama_server_defaults.yaml"
    bundled_vllm: dict[str, object] = {}
    try:
        bundled = yaml.safe_load(bundled_path.read_text(encoding="utf-8")) or {}
        bundled_vllm = _normalize_vllm_config(bundled.get("vllm")) if isinstance(bundled, dict) else {}
    except Exception:
        bundled_vllm = {}
    defaults = dict(bundled_vllm.get("defaults") or {})
    defaults.update(config.get("defaults") or {})
    family_defaults = {
        str(pattern): dict(values)
        for pattern, values in (bundled_vllm.get("family_defaults") or {}).items()
        if isinstance(values, dict)
    }
    for pattern, values in (config.get("family_defaults") or {}).items():
        if isinstance(values, dict):
            family_defaults.setdefault(str(pattern), {}).update(values)
    if isinstance(family_defaults, dict) and family_defaults:
        defaults["__family_defaults"] = family_defaults
    return defaults


def _args_server_config_path(args) -> Path | None:
    value = getattr(args, "server_config", None)
    if value is None:
        return None
    return Path(value)


def _llama_flag_name(key: str) -> str:
    return key if str(key).startswith("-") else f"--{str(key).replace('_', '-')}"


def _model_family_match_text(model: ManagedModel) -> str:
    parts: list[str] = []
    for attr in ("model_id", "repo_id", "filename", "local_path", "file", "quant"):
        try:
            value = getattr(model, attr, None)
        except Exception:
            value = None
        if value:
            parts.append(str(value))
    try:
        aliases = getattr(model, "aliases", None)
        if isinstance(aliases, list):
            parts.extend(str(item) for item in aliases if item)
    except Exception:
        pass
    try:
        parts.extend(str(item) for item in model_name_aliases(model) if item)
    except Exception:
        pass
    text = "\n".join(parts).lower()
    # Qwopus model IDs are Qwen-family fine tunes but do not contain the
    # literal substring "qwen". Let existing qwen family defaults apply.
    if "qwopus" in text:
        text += "\nqwen"
    return text


def _family_defaults_for_model(model: ManagedModel, family_defaults: object) -> dict[str, object]:
    if not isinstance(family_defaults, dict):
        return {}
    haystack = _model_family_match_text(model)
    matched: dict[str, object] = {}
    for raw_pattern, raw_defaults in family_defaults.items():
        pattern = str(raw_pattern or "").strip().lower()
        if not pattern or pattern not in haystack or not isinstance(raw_defaults, dict):
            continue
        matched.update(normalize_server_overrides(raw_defaults))
    return matched


def _vllm_family_defaults_for_model(model: ManagedModel, family_defaults: object) -> dict[str, object]:
    if not isinstance(family_defaults, dict):
        return {}
    haystack = _model_family_match_text(model)
    matched: dict[str, object] = {}
    for raw_pattern, raw_defaults in family_defaults.items():
        pattern = str(raw_pattern or "").strip().lower()
        if pattern and pattern in haystack and isinstance(raw_defaults, dict):
            matched.update(_normalize_vllm_mapping(raw_defaults))
    return matched


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _effective_llama_server_options(
    model: ManagedModel,
    server_defaults: dict[str, object] | None,
) -> dict[str, object]:
    """Merge global, family and model options using launch precedence."""
    effective = dict(normalize_server_overrides(server_defaults or {}))
    family_defaults = effective.pop("__family_defaults", None)
    effective.update(_family_defaults_for_model(model, family_defaults))
    effective.update(normalize_server_overrides(model.server_overrides))
    return effective


def resolve_request_reasoning_budget(
    payload: dict[str, object],
    model: ManagedModel,
    *,
    server_defaults: dict[str, object] | None = None,
) -> tuple[int | None, str]:
    """Return a safe request-level thinking budget for dynamic defaults.

    A configured ``half_context`` is a policy, not a literal CLI integer.
    llama.cpp counts reasoning and visible output against the same generation
    limit, so the policy is clamped to leave output room and to the active
    predict/request cap. Explicit client controls always win.
    """
    if not isinstance(payload, dict) or _model_backend(model) != "llama.cpp":
        return None, "unsupported_backend"
    # Numeric budgets are real client decisions; qualitative hints carry no
    # token budget and must never disable this starvation safety net.
    numeric_budget_keys = {
        "thinking_budget_tokens",
        "reasoning_budget_tokens",
    }
    if any(key in payload for key in numeric_budget_keys):
        return None, "explicit_client_control"

    effective = _effective_llama_server_options(model, server_defaults)
    policy = effective.get("reasoning_budget")
    if not (isinstance(policy, str) and policy.casefold() == REASONING_BUDGET_HALF_CONTEXT):
        return None, "fixed_or_unconfigured"
    if str(effective.get("reasoning") or "").strip().casefold() == "off":
        return None, "reasoning_disabled"

    request_limit = _positive_int(payload.get("max_tokens"))
    if request_limit is None:
        request_limit = _positive_int(payload.get("max_completion_tokens"))
    configured_predict = _positive_int(effective.get("predict"))
    limits = [limit for limit in (request_limit, configured_predict) if limit]
    generation_limit = min(limits) if limits else None
    context_size = displayed_configured_ctx(model)
    if context_size <= 0:
        return None, "unknown_context"
    half_context = max(0, context_size // 2)

    # Tiny one-message requests are capability probes used by several
    # clients. They have no room for hidden reasoning; neutralize thinking so
    # the probe gets a visible response instead of finish_reason=length.
    if request_looks_like_model_probe(payload) and request_limit is not None and request_limit <= MODEL_PROBE_REASONING_MAX_TOKENS:
        return 0, "model_probe"

    if generation_limit is None:
        return half_context, "half_context"
    reserve = min(DEFAULT_REASONING_VISIBLE_RESERVE, max(1, generation_limit // 8))
    budget = max(0, min(half_context, generation_limit - reserve))
    reason = "half_context_clamped_to_generation_limit" if budget < half_context else "half_context"
    return budget, reason


def resolve_vllm_options(
    model: ManagedModel,
    vllm_defaults: dict[str, object] | None = None,
) -> dict[str, object]:
    raw_defaults = _normalize_vllm_mapping(vllm_defaults or resolve_vllm_defaults())
    family_defaults = raw_defaults.pop("__family_defaults", None)
    options = dict(raw_defaults)
    options.update(_vllm_family_defaults_for_model(model, family_defaults))
    overrides = model.server_overrides if isinstance(model.server_overrides, dict) else {}
    model_vllm = overrides.get("vllm")
    if isinstance(model_vllm, dict):
        options.update(_normalize_vllm_mapping(model_vllm))

    if "max_model_len" not in options:
        options["max_model_len"] = int(getattr(model, "ctx_size", DEFAULT_CTX_SIZE) or DEFAULT_CTX_SIZE)
    if "tensor_parallel_size" not in options:
        parts = [part for part in str(getattr(model, "tensor_split", "1") or "1").split(",") if part.strip()]
        options["tensor_parallel_size"] = max(1, len(parts))
    if "dtype" not in options:
        legacy_dtype = next(
            (key for key in ("float16", "bfloat16", "float32") if overrides.get(key)),
            None,
        )
        if legacy_dtype:
            options["dtype"] = legacy_dtype
    return options


def _is_gemma4_model(model: ManagedModel) -> bool:
    try:
        haystack = _model_family_match_text(model)
    except Exception:
        haystack = ""
    return any(token in haystack for token in ("gemma-4", "gemma4"))


def _server_supports_or_unknown(server_path: Path | str | None, flag: str) -> bool:
    try:
        return server_path is None or server_supports_flag(server_path, flag)
    except Exception:
        # If help probing fails, prefer not to drop a configured flag silently.
        return True


def _append_llama_server_flag(cmd: list[str], key: str, value: object, server_path: Path | str | None = None) -> None:
    if value is None:
        cmd.append(_llama_flag_name(key))
        return
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
    elif key == "image_min_tokens":
        try:
            cmd.extend(["--image-min-tokens", str(int(value))])
        except (TypeError, ValueError):
            pass
    elif key == "model_draft":
        sval = str(value or "").strip()
        if sval and sval.lower() not in {"none", "null"}:
            cmd.extend(["--model-draft", sval])
    elif key == "hf_repo_draft":
        cmd.extend(["--hf-repo-draft", str(value)])
    elif key == "spec_type":
        sval = str(value or "").strip()
        if sval and _server_supports_or_unknown(server_path, "--spec-type"):
            cmd.extend(["--spec-type", sval])
    elif key == "spec_draft_n_max":
        try:
            flag = "--spec-draft-n-max" if _server_supports_or_unknown(server_path, "--spec-draft-n-max") else "--draft-max"
            cmd.extend([flag, str(int(value))])
        except (TypeError, ValueError):
            pass
    elif key == "spec_draft_n_min":
        try:
            cmd.extend(["--spec-draft-n-min", str(int(value))])
        except (TypeError, ValueError):
            pass
    elif key == "spec_draft_p_min":
        try:
            cmd.extend(["--spec-draft-p-min", str(value)])
        except (TypeError, ValueError):
            pass
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
    elif key == "draft_mtp":
        # New llama.cpp option: enable draft MTP handling (flag `--draft-mtp`).
        # Prefer emitting the flag only if the server supports it, but
        # fall back to emitting it unconditionally if help parsing fails.
        try:
            flag = "--draft-mtp"
            if server_supports_flag(server_path, flag) or server_path is None:
                cmd.append(flag)
        except Exception:
            try:
                cmd.append("--draft-mtp")
            except Exception:
                pass
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
        if _normalize_bool_flag(value) and _server_supports_or_unknown(server_path, "--mul-mat-q"):
            cmd.append("--mul-mat-q")
    elif key == "grp_attn_n":
        try:
            if _server_supports_or_unknown(server_path, "--grp-attn-n"):
                cmd.extend(["--grp-attn-n", str(int(value))])
        except (TypeError, ValueError):
            pass
    elif key == "parallel":
        try:
            if _server_supports_or_unknown(server_path, "--parallel"):
                cmd.extend(["--parallel", str(int(value))])
        except (TypeError, ValueError):
            pass
    elif key == "ctx_checkpoints":
        try:
            if _server_supports_or_unknown(server_path, "--ctx-checkpoints"):
                cmd.extend(["--ctx-checkpoints", str(int(value))])
        except (TypeError, ValueError):
            pass
    elif key in {"checkpoint_min_step", "checkpoint_every_n_tokens"}:
        try:
            if _server_supports_or_unknown(server_path, "--checkpoint-min-step"):
                cmd.extend(["--checkpoint-min-step", str(int(value))])
            elif _server_supports_or_unknown(server_path, "--checkpoint-every-n-tokens"):
                cmd.extend(["--checkpoint-every-n-tokens", str(int(value))])
        except (TypeError, ValueError):
            pass
    elif key == "cache_ram":
        try:
            if _server_supports_or_unknown(server_path, "--cache-ram"):
                cmd.extend(["--cache-ram", str(int(value))])
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
        if bool_val is True and _server_supports_or_unknown(server_path, "--kv-unified"):
            cmd.append("--kv-unified")
        elif bool_val is False and _server_supports_or_unknown(server_path, "--no-kv-unified"):
            cmd.append("--no-kv-unified")
    elif key == "cache_idle_slots":
        bool_val = _normalize_bool_flag(value)
        if bool_val is True and _server_supports_or_unknown(server_path, "--cache-idle-slots"):
            cmd.append("--cache-idle-slots")
        elif bool_val is False and _server_supports_or_unknown(server_path, "--no-cache-idle-slots"):
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
            if _server_supports_or_unknown(server_path, "--defrag-threshold"):
                cmd.extend(["--defrag-threshold", str(float(value))])
        except (TypeError, ValueError):
            pass
    elif key == "swa_full":
        bool_val = _normalize_bool_flag(value)
        if bool_val is True and _server_supports_or_unknown(server_path, "--swa-full"):
            cmd.append("--swa-full")
        elif bool_val is False and _server_supports_or_unknown(server_path, "--no-swa-full"):
            cmd.append("--no-swa-full")
    elif key == "top_k":
        try:
            if _server_supports_or_unknown(server_path, "--top-k"):
                cmd.extend(["--top-k", str(int(value))])
        except (TypeError, ValueError):
            pass
    elif key == "top_p":
        try:
            if _server_supports_or_unknown(server_path, "--top-p"):
                cmd.extend(["--top-p", str(float(value))])
        except (TypeError, ValueError):
            pass
    elif key == "min_p":
        try:
            if _server_supports_or_unknown(server_path, "--min-p"):
                cmd.extend(["--min-p", str(float(value))])
        except (TypeError, ValueError):
            pass
    elif key == "repeat_penalty":
        try:
            if _server_supports_or_unknown(server_path, "--repeat-penalty"):
                cmd.extend(["--repeat-penalty", str(float(value))])
        except (TypeError, ValueError):
            pass
    elif key == "presence_penalty":
        try:
            if _server_supports_or_unknown(server_path, "--presence-penalty"):
                cmd.extend(["--presence-penalty", str(float(value))])
        except (TypeError, ValueError):
            pass
    elif key == "predict":
        try:
            if _server_supports_or_unknown(server_path, "--predict"):
                cmd.extend(["--predict", str(int(value))])
        except (TypeError, ValueError):
            pass
    elif key == "reasoning":
        sval = str(value or "").strip()
        if sval and _server_supports_or_unknown(server_path, "--reasoning"):
            cmd.extend(["--reasoning", sval])
    elif key == "reasoning_budget":
        try:
            if _server_supports_or_unknown(server_path, "--reasoning-budget"):
                # -1 delegates the actual budget to the request-level
                # thinking_budget_tokens field. This is the launch-time
                # representation of the half_context policy.
                if isinstance(value, str) and value.strip().casefold() == REASONING_BUDGET_HALF_CONTEXT:
                    value = -1
                cmd.extend(["--reasoning-budget", str(int(value))])
        except (TypeError, ValueError):
            pass
    elif key == "reasoning_budget_message":
        sval = str(value or "").strip()
        if sval and _server_supports_or_unknown(server_path, "--reasoning-budget-message"):
            cmd.extend(["--reasoning-budget-message", sval])
    elif key == "chat_template_kwargs":
        sval = str(value or "").strip()
        if sval and _server_supports_or_unknown(server_path, "--chat-template-kwargs"):
            cmd.extend(["--chat-template-kwargs", sval])
    else:
        # Generic fallback: map unknown keys to --kebab-case and append value.
        try:
            flag = _llama_flag_name(str(key))
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




def _device_list_for_tensor_split(tensor_split: str, total_devices: int | None = None) -> str | None:
    parts = [part.strip() for part in str(tensor_split or "").split(",") if part.strip()]
    if not parts:
        return None
    # Use llama.cpp device names, not bare indices. Do not depend on
    # detect_cuda_device_count() here: in service environments it can disagree
    # with the devices that llama-server later sees, which previously let a
    # 3-way tensor_split launch while still touching 7 physical GPUs.
    return ",".join(f"CUDA{idx}" for idx in range(len(parts)))

def _vllm_flag_value(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def build_vllm_server_command(
    model: ManagedModel,
    *,
    port: str,
    host: str | None = None,
    vllm_defaults: dict[str, object] | None = None,
) -> list[str]:
    options = resolve_vllm_options(model, vllm_defaults)
    command = [
        os.environ.get("VLLM_SERVER_BIN", "vllm-server"),
        "--model",
        str(model.local_path),
        "--port",
        str(port),
        "--served-model-name",
        str(model.model_id),
    ]
    resolved_host = str(options.pop("host", host or model.host) or "127.0.0.1")
    if resolved_host:
        command.extend(["--host", resolved_host])

    reserved = {"model", "port", "served_model_name", "host"}
    for raw_key, value in options.items():
        key = str(raw_key).strip().lower().replace("-", "_")
        if not key or key in reserved or value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            # vLLM exposes both positive and negative argparse switches for
            # most boolean engine arguments. Emitting the explicit negative
            # form keeps a configured false value distinguishable from an
            # omitted default.
            command.append(flag if value else f"--no-{key.replace('_', '-')}")
            continue
        command.extend([flag, _vllm_flag_value(value)])
    return command


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
    vllm_defaults: dict[str, object] | None = None,
    extra_flags: list[str] | None = None,
) -> list[str]:
    if _model_backend(model) == "vllm":
        return build_vllm_server_command(
            model,
            port=port,
            host=host,
            vllm_defaults=vllm_defaults,
        )
    effective = dict(normalize_server_overrides(server_defaults or {}))
    family_defaults = effective.pop("__family_defaults", None)
    effective.update(_family_defaults_for_model(model, family_defaults))
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
    model_overrides = normalize_server_overrides(model.server_overrides)
    effective.update(model_overrides)
    if _is_gemma4_model(model) and "swa_full" not in model_overrides:
        # Gemma4 uses sliding-window attention.  A global --swa-full default
        # turns that into a full-size KV cache; at 262k ctx this can request
        # >100 GiB on a single GPU and make llama-server exit during load.
        # Keep an explicit per-model override for advanced manual testing.
        effective.pop("swa_full", None)
    # Replica orchestration config is consumed by the Heimdall Gateway proxy/config
    # renderer and must never be forwarded as a llama-server CLI flag.
    effective.pop("replicas", None)
    effective.pop("auto_performance", None)
    effective.pop("__family_defaults", None)
    # speculative_defaults is internal config for orchestration, not a
    # llama-server CLI flag.  Pop the whole dict so it does not reach the
    # generic fallback handler which would str() it and emit a broken flag.
    effective.pop("speculative_defaults", None)
    # mtp_defaults is internal config for MTP drafter orchestration; the
    # resolved individual keys (spec_draft_n_max, etc.) are set on the
    # model's server_overrides by _apply_mtp_server_overrides and emitted
    # as separate flags. Pop the whole dict to avoid --mtp-defaults {...}.
    effective.pop("mtp_defaults", None)
    # Sub-keys from speculative_defaults that are NOT valid llama-server
    # flags and must not leak into the command line.
    for _sk in ("enabled", "id_prefix", "allow_multiple_variants"):
        effective.pop(_sk, None)

    # Internal escape hatch for generated replicas: tensor_split must be relative
    # to CUDA_VISIBLE_DEVICES, not normalized against all host GPUs.
    replica_tensor_split = effective.pop("__replica_tensor_split", None)

    if _safe_runtime_enabled():
        # Crash-diagnostics mode: avoid llama.cpp features that are more likely
        # to hit backend-specific segfaults while preserving a usable baseline.
        # This is intentionally conservative and opt-in via env var.
        effective["flash_attn"] = False
        effective["fit"] = False
        effective["parallel"] = 1
        effective["cont_batching"] = False
        for risky_key in (
            "draft_mtp",
            "model_draft",
            "hf_repo_draft",
            "spec_type",
            "spec_draft_n_max",
            "spec_draft_n_min",
            "spec_draft_p_min",
            "draft",
            "draft_min",
            "draft_p_min",
            "ctx_size_draft",
            "n_gpu_layers_draft",
            "ctx_checkpoints",
        ):
            effective.pop(risky_key, None)

    if (
        str(effective.get("model_draft") or "").strip()
        and "draft" in effective
        and (str(effective.get("spec_type") or "").strip() == "draft-mtp" or "spec_draft_n_max" in effective)
    ):
        # Avoid rendering both MTP-specific --spec-draft-n-max and legacy
        # --draft/--draft-max aliases for the same drafter.
        effective.pop("draft", None)

    # Auto-enable draft-mtp for models whose id ends with '-mtp' unless
    # explicitly overridden by server_overrides. Downstream code emits
    # the corresponding `--draft-mtp` flag when present.
    try:
        mid = str(getattr(model, "model_id", "") or "").strip().lower()
        if (not _safe_runtime_enabled()) and mid.endswith("-mtp"):
            # normalized keys use underscores
            if "draft_mtp" not in effective and "draft-mtp" not in effective:
                effective["draft_mtp"] = True
    except Exception:
        pass

    # Resolve non-fit launch dimensions first so fit-mode can optionally move
    # context control from --ctx-size into -fitc.
    ctx_size = int(effective.pop("ctx_size", model.ctx_size))
    n_gpu_layers = int(effective.pop("n_gpu_layers", model.n_gpu_layers))
    cuda_visible_devices: list[int] | None = None
    if replica_tensor_split is not None:
        tensor_split = str(replica_tensor_split)
        effective.pop("tensor_split", None)
        if "device" not in effective:
            effective["device"] = ",".join(f"CUDA{idx}" for idx in range(len([p for p in tensor_split.split(",") if p.strip()])))
    else:
        # Server defaults and legacy server_overrides.tensor_split must not
        # override the model placement. load_catalog migrates the legacy
        # override into model.tensor_split, and this keeps command rendering
        # robust for in-memory objects too.
        effective.pop("tensor_split", None)
        raw_tensor_split = str(model.tensor_split)
        raw_parts = [part.strip() for part in raw_tensor_split.split(",") if part.strip()]
        explicit_device = "device" in effective
        if raw_parts:
            # The number of tensor_split entries is the desired visible GPU set
            # for this process. Enforce it with CUDA_VISIBLE_DEVICES as a hard
            # boundary instead of relying on llama.cpp --device or on host GPU
            # detection, both of which can still allow CUDA contexts on extra
            # GPUs in some deployments.
            tensor_split = raw_tensor_split
            if not explicit_device:
                cuda_visible_devices = list(range(len(raw_parts)))
        else:
            tensor_split = preferred_tensor_split(model, raw_tensor_split)
        inferred_device = _device_list_for_tensor_split(tensor_split)
        if inferred_device is not None and "device" not in effective:
            effective["device"] = inferred_device
    resolved_host = str(effective.pop("host", host or model.host))

    # Fit policy:
    # - use_fitc true  -> use -fitc for context, omit --ctx-size
    # - use_fitc false -> keep --ctx-size directly, omit automatic -fitc/-fitt
    # Legacy configs that explicitly set fit=on and do not define use_fitc keep
    # the old fitc behavior.
    have_autocontext = (getattr(model, "ctx_probe_kv_gb", None) is not None) or (getattr(model, "ctx_probe_read_s", None) is not None)
    use_fitc_raw = effective.pop("use_fitc", None)
    use_fitc = _normalize_bool_flag(use_fitc_raw)
    if use_fitc is None:
        if use_fitc_raw is None and "fit" in effective:
            legacy_fit = _normalize_bool_flag(effective.get("fit"))
            use_fitc = bool(legacy_fit) if legacy_fit is not None else bool(str(effective.get("fit") or "").strip())
        else:
            use_fitc = False
    fit_enabled = bool(use_fitc)

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
        effective.pop("fit", None)
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
    explicit_override_keys = set(model_overrides.keys())
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
        "spec_type",
        "spec_draft_n_max",
        "spec_draft_n_min",
        "spec_draft_p_min",
        "use_fitc",
        "draft",
        "draft_min",
        "draft_p_min",
        "ctx_size_draft",
        "n_gpu_layers_draft",
        "draft_mtp",
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
        "cache_prompt",
        "kv_offload",
        "cont_batching",
        "op_offload",
        "direct_io",
        "cpu_moe",
        "n_cpu_moe",
        "device",
        "defrag_threshold",
        "swa_full",
        "top_k",
        "top_p",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "predict",
        "reasoning",
        "reasoning_budget",
        "reasoning_budget_message",
    ):
        if key in effective:
            # Defaults generated by Heimdall Gateway should be conservative and must
            # not inject unsupported flags into older/different llama.cpp builds.
            # Explicit per-model catalog overrides are different: if the user
            # configured a flag, emit it and let llama.cpp decide.
            flag_probe_path = None if key in explicit_override_keys else server_path
            _append_llama_server_flag(cmd, key, effective[key], flag_probe_path)
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
    if cuda_visible_devices is not None:
        cmd = _command_with_cuda_visible_devices(cmd, cuda_visible_devices)
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
    path = Path(args.server_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Load existing values with conf.json fallback. If the file exists but has
    # invalid JSON, abort so the user can fix it rather than silently overwriting.
    if path.exists():
        try:
            raw = path.read_text(encoding="utf-8")
            if raw.strip():
                loaded = _json_loads_allow_comments(raw, path_desc=str(path))
                payload = loaded if isinstance(loaded, dict) else {}
            else:
                payload = {}
        except Exception:
            print(f"[!] {path.name} has invalid JSON. Not overwriting.\n"
                  f"    Fix the error or delete the file to regenerate from defaults.",
                  file=sys.stderr)
            return
    else:
        # First-time creation: seed with fallback values from conf.json
        payload = _load_server_config_payload(args)
    if getattr(args, "idle_ttl", None) is not None:
        payload["idle_ttl"] = int(args.idle_ttl)
    if getattr(args, "api_port", None) is not None:
        payload["api_port"] = int(args.api_port)
    if getattr(args, "api_ctx_factor", None) is not None:
        payload["api_ctx_factor"] = float(args.api_ctx_factor)
    if getattr(args, "flatten", None) is not None:
        payload["flatten_namespace_tools"] = bool(args.flatten)
    normalized, changed = normalize_server_config_payload(payload)
    if _rewrite_legacy_api_https_paths(normalized, path):
        changed = True
    if changed or not path.exists() or any(getattr(args, name, None) is not None for name in ("idle_ttl", "api_port", "api_ctx_factor", "flatten")):
        path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")




def _rewrite_legacy_api_https_paths(payload: dict[str, object], server_config_path: Path) -> bool:
    https = payload.get("api_https")
    if not isinstance(https, dict):
        return False
    changed = False
    certs_dir = server_config_path.parent / "certs"
    mapping = {
        "cert_file": ("superserver-api.crt", "heimdall-gateway-api.crt"),
        "key_file": ("superserver-api.key", "heimdall-gateway-api.key"),
    }
    for key, (legacy_name, new_name) in mapping.items():
        raw = str(https.get(key) or "").strip()
        if "llamacpp-superserver" not in raw and legacy_name not in raw:
            continue
        src = Path(raw).expanduser()
        dst = certs_dir / new_name
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass
        https[key] = str(dst)
        changed = True
    return changed

def _server_config_summary_for_print(text: str) -> dict[str, object]:
    try:
        payload = _json_loads_allow_comments(text) if text.strip() else {}
    except Exception as exc:
        return {"parse_error": str(exc)}
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}
    defaults = payload.get("llama_server_defaults")
    bad_default_keys: list[str] = []
    if isinstance(defaults, dict):
        bad_default_keys = sorted(str(k) for k in defaults if "-" in str(k))
    return {
        "has_models": "models" in payload,
        "has_replicas": isinstance(payload.get("replicas"), dict),
        "bad_default_keys": bad_default_keys,
    }


def migrate_server_config(args) -> int:
    if getattr(args, "server_config", None) is None:
        raise RuntimeError("No --server-config path configured")
    path = Path(args.server_config)
    before = path.read_text(encoding="utf-8") if path.exists() else ""
    before_summary = _server_config_summary_for_print(before)
    persist_server_config(args)
    after = path.read_text(encoding="utf-8") if path.exists() else ""
    after_summary = _server_config_summary_for_print(after)
    changed = before != after
    print(f"Server config migrated: {path} ({'changed' if changed else 'already current'})")
    print("Before:", json.dumps(before_summary, sort_keys=True))
    print("After: ", json.dumps(after_summary, sort_keys=True))
    return 0


def log_api_event(kind: str, payload: dict, log_path: Path = DEFAULT_REQUESTS_LOG_PATH) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        **payload,
    }
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    fallback_path = Path(_env_value("HEIMDALL_GATEWAY_REQUESTS_LOG_FALLBACK", "LLAMACPP_REQUESTS_LOG_FALLBACK", "/tmp/heimdall-gateway-api-requests.log"))
    
    errors = []
    for candidate in (Path(log_path), fallback_path):
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            with candidate.open("a", encoding="utf-8") as fh:
                fh.write(line)
            return
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            continue
            
    # If we got here, all attempts failed.
    # We print to stderr so it shows up in systemd journal, but don't crash the server.
    if _env_value("HEIMDALL_GATEWAY_DEBUG_LOGGING", "LLAMACPP_DEBUG_LOGGING", ""):
        print(f"[!] log_api_event failed to write to any candidate: {', '.join(errors)}", file=sys.stderr)


def _redact_large_log_value(value, *, max_string: int = 500):
    if isinstance(value, str):
        if value.startswith("data:") or len(value) > max_string:
            return f"<str len={len(value)} preview={value[:120]!r}>"
        return value
    if isinstance(value, list):
        return [_redact_large_log_value(item, max_string=max_string) for item in value[:20]]
    if isinstance(value, dict):
        return {str(key): _redact_large_log_value(item, max_string=max_string) for key, item in list(value.items())[:80]}
    return value


def _payload_image_stats(payload: object) -> dict[str, int]:
    stats = {"input_image_count": 0, "image_url_count": 0, "data_image_count": 0}

    def scan(value: object) -> None:
        if isinstance(value, dict):
            item_type = str(value.get("type") or "").strip()
            is_image_item = False
            if item_type == "input_image":
                stats["input_image_count"] += 1
                is_image_item = True
            elif item_type == "image_url":
                stats["image_url_count"] += 1
                is_image_item = True
            image_url = value.get("image_url")
            if item_type not in {"input_image", "image_url"} and isinstance(image_url, (str, dict)) and image_url:
                stats["image_url_count"] += 1
                is_image_item = True
            if value.get("data") and str(value.get("mime_type") or value.get("media_type") or "").startswith("image/"):
                stats["data_image_count"] += 1
                is_image_item = True
            if is_image_item:
                stats["total_image_count"] = stats.get("total_image_count", 0) + 1
            for item in value.values():
                scan(item)
            return
        if isinstance(value, list):
            for item in value:
                scan(item)

    scan(payload)
    stats.setdefault("total_image_count", 0)
    return stats


def _payload_contains_images(payload: object) -> bool:
    return _payload_image_stats(payload).get("total_image_count", 0) > 0


def _summarize_api_payload_for_log(payload: dict) -> dict:
    """Keep request diagnostics useful without dumping huge prompts/images."""
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}
    summary: dict[str, object] = {}
    for key in (
        "model",
        "stream",
        "max_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "thinking_budget_tokens",
        "temperature",
        "tool_choice",
        "parallel_tool_calls",
    ):
        if key in payload:
            summary[key] = _redact_large_log_value(payload.get(key))
    tools = payload.get("tools")
    if isinstance(tools, list):
        summary["tools_count"] = len(tools)
        summary["tool_types"] = [
            str(item.get("type") or "") if isinstance(item, dict) else type(item).__name__
            for item in tools[:20]
        ]
        tool_summaries: list[dict[str, object]] = []
        for item in tools[:20]:
            if not isinstance(item, dict):
                tool_summaries.append({"type": type(item).__name__})
                continue
            tool_type = str(item.get("type") or "")
            entry: dict[str, object] = {"type": tool_type}
            if item.get("name") is not None:
                entry["name"] = str(item.get("name") or "")[:120]
            function = item.get("function")
            if isinstance(function, dict):
                entry["function_name"] = str(function.get("name") or "")[:120]
            sub_tools = item.get("tools")
            if isinstance(sub_tools, list):
                entry["tools_count"] = len(sub_tools)
                entry["tool_names"] = [
                    str(st.get("name") or "")[:80] if isinstance(st, dict) else type(st).__name__
                    for st in sub_tools[:12]
                ]
            tool_summaries.append(entry)
        summary["tools_sample"] = tool_summaries
    image_stats = _payload_image_stats(payload)
    if image_stats.get("total_image_count", 0):
        summary["has_images"] = True
    summary.update(image_stats)
    raw_input = payload.get("input")
    if isinstance(raw_input, str):
        summary["input_type"] = "str"
        summary["input_len"] = len(raw_input)
        summary["input_preview"] = raw_input[:200]
    elif isinstance(raw_input, list):
        summary["input_type"] = "list"
        summary["input_items"] = len(raw_input)
        summary["input_item_types"] = [
            str(item.get("type") or "") if isinstance(item, dict) else type(item).__name__
            for item in raw_input[:20]
        ]
    elif raw_input is not None:
        summary["input_type"] = type(raw_input).__name__
    messages = payload.get("messages")
    if isinstance(messages, list):
        summary["messages_count"] = len(messages)
        summary["message_roles"] = [
            str(item.get("role") or "") if isinstance(item, dict) else type(item).__name__
            for item in messages[:20]
        ]
        if len(messages) > 20:
            summary["message_roles_tail"] = [
                str(item.get("role") or "") if isinstance(item, dict) else type(item).__name__
                for item in messages[-20:]
            ]
    return summary


def _elapsed_ms(started_at: float) -> int:
    return int((time.monotonic() - started_at) * 1000)



class ReplicaRouterState:
    def __init__(self):
        self.lock = threading.RLock()
        self.records: dict[str, ReplicaRecord] = {}
        self.base_in_flight: dict[str, int] = {}
        self.base_last_used: dict[str, float] = {}
        self.affinity: dict[str, tuple[str, float]] = {}
        self.response_to_replica: dict[str, tuple[str, float]] = {}
        self.gpu_snapshot: tuple[float, dict[int, dict[str, float]]] = (0.0, {})
        # A model load is triggered by the first request that targets an
        # unloaded process.  Keep an atomic claim so concurrent requests do
        # not all reach llama-swap and make it start/restart the same model.
        self.loading_claims: dict[str, float] = {}
        self.loading_claim_aliases: dict[str, str] = {}


    def prune(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self.lock:
            self.affinity = {k: v for k, v in self.affinity.items() if v[1] > now}
            self.response_to_replica = {k: v for k, v in self.response_to_replica.items() if v[1] > now}

    def bind_response(self, response_id: str, replica_id: str, ttl_s: int = 3600) -> None:
        if not response_id or not replica_id:
            return
        with self.lock:
            self.response_to_replica[response_id] = (replica_id, time.monotonic() + ttl_s)

    def response_replica(self, response_id: str) -> str | None:
        if not response_id:
            return None
        now = time.monotonic()
        with self.lock:
            value = self.response_to_replica.get(response_id)
            if not value:
                return None
            if value[1] <= now:
                self.response_to_replica.pop(response_id, None)
                return None
            return value[0]

    def request_started(self, replica_id: str) -> None:
        with self.lock:
            rec = self.records.get(replica_id)
            if rec:
                rec.in_flight += 1
                rec.last_used = time.monotonic()
                if rec.status == "cold":
                    rec.status = "loading"
            else:
                self.base_in_flight[replica_id] = self.base_in_flight.get(replica_id, 0) + 1
                self.base_last_used[replica_id] = time.monotonic()

    def request_finished(self, replica_id: str, *, ok: bool = True) -> None:
        with self.lock:
            self.loading_claims.pop(replica_id, None)
            claim_key = self.loading_claim_aliases.pop(replica_id, None)
            if claim_key:
                self.loading_claims.pop(claim_key, None)
            rec = self.records.get(replica_id)
            if rec:
                rec.in_flight = max(0, rec.in_flight - 1)
                rec.last_used = time.monotonic()
                rec.status = "ready" if ok else "error"
                if not ok:
                    rec.blacklist_until = time.monotonic() + 120.0
            else:
                self.base_in_flight[replica_id] = max(0, self.base_in_flight.get(replica_id, 0) - 1)
                self.base_last_used[replica_id] = time.monotonic()

    def base_load(self, model_id: str) -> int:
        with self.lock:
            return int(self.base_in_flight.get(model_id, 0))

    def claim_loading(self, model_id: str, catalog: list[ManagedModel], *, is_replica: bool = False, claim_key: str | None = None) -> bool:
        """Claim the first request that may cause *model_id* to load.

        The claim is only needed while no llama-server process exists.  It is
        deliberately separate from ``in_flight``: a loaded server may serve
        concurrent requests, while an unloaded server must receive exactly
        one loader request.
        """
        now = time.monotonic()
        claim_key = claim_key or model_id
        rec = self.records.get(model_id) if is_replica else None
        process_loaded = (rec is not None and rec.status == "ready") or (
            not is_replica and get_catalog_model_process(model_id, catalog) is not None
        )
        with self.lock:
            self.loading_claims = {
                key: deadline for key, deadline in self.loading_claims.items()
                if deadline > now
            }
            if process_loaded:
                self.loading_claims.pop(claim_key, None)
                return True
            deadline = self.loading_claims.get(claim_key)
            if deadline is not None and deadline > now:
                return False
            self.loading_claims[claim_key] = now + 900.0
            self.loading_claim_aliases[model_id] = claim_key
            return True

    def release_loading_claim(self, model_id: str) -> None:
        """Release a claim when preflight rejects the request before loading."""
        with self.lock:
            claim_key = self.loading_claim_aliases.pop(model_id, None) or model_id
            self.loading_claims.pop(model_id, None)
            self.loading_claims.pop(claim_key, None)


REPLICA_ROUTER_STATE = ReplicaRouterState()


class ConversationRequestToken:
    def __init__(self, key: str, model: str, generation: int):
        self.key = key
        self.model = model
        self.generation = generation


class ConversationSwitchState:
    """Tracks model changes per real conversation/agent, independent of model id.

    This is intentionally separate from replica affinity. Replica affinity is
    model-scoped; cancellation needs to know that the same conversation moved
    from Gemma to Qwen so old repair loops stop reloading Gemma.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.generation_by_key: dict[str, int] = {}
        self.active_model_by_key: dict[str, str] = {}
        self.in_flight_by_key: dict[str, int] = {}
        self.last_seen_by_key: dict[str, float] = {}

    def prune(self, ttl_s: float = 3600.0) -> None:
        now = time.monotonic()
        with self.lock:
            stale = [
                key for key, last_seen in self.last_seen_by_key.items()
                if self.in_flight_by_key.get(key, 0) <= 0 and now - last_seen > ttl_s
            ]
            for key in stale:
                self.generation_by_key.pop(key, None)
                self.active_model_by_key.pop(key, None)
                self.in_flight_by_key.pop(key, None)
                self.last_seen_by_key.pop(key, None)

    def start(self, key: str, model: str) -> tuple[ConversationRequestToken | None, str]:
        if not key:
            return None, ""
        now = time.monotonic()
        model = str(model or "")
        with self.lock:
            previous_model = self.active_model_by_key.get(key, "")
            generation = int(self.generation_by_key.get(key, 0))
            reason = ""
            if previous_model and previous_model != model:
                generation += 1
                reason = "model_changed"
            elif key not in self.generation_by_key:
                generation = 1
                reason = "new_conversation"
            self.generation_by_key[key] = generation
            self.active_model_by_key[key] = model
            self.in_flight_by_key[key] = max(0, int(self.in_flight_by_key.get(key, 0))) + 1
            self.last_seen_by_key[key] = now
            return ConversationRequestToken(key, model, generation), reason

    def finish(self, token: ConversationRequestToken | None) -> None:
        if token is None or not token.key:
            return
        with self.lock:
            self.in_flight_by_key[token.key] = max(0, int(self.in_flight_by_key.get(token.key, 0)) - 1)
            self.last_seen_by_key[token.key] = time.monotonic()

    def should_cancel(self, token: ConversationRequestToken | None) -> tuple[bool, str]:
        if token is None or not token.key:
            return False, ""
        with self.lock:
            generation = int(self.generation_by_key.get(token.key, 0))
            active_model = self.active_model_by_key.get(token.key, "")
            if generation != token.generation:
                return True, "conversation_generation_changed"
            if active_model and active_model != token.model:
                return True, "conversation_model_changed"
            return False, ""

    def snapshot(self, key: str) -> dict[str, object]:
        with self.lock:
            return {
                "key": key,
                "generation": self.generation_by_key.get(key, 0),
                "active_model": self.active_model_by_key.get(key, ""),
                "in_flight": self.in_flight_by_key.get(key, 0),
            }


CONVERSATION_SWITCH_STATE = ConversationSwitchState()


_CACHE_TYPE_ALIASES = {
    "q8": "q8_0",
    "8bit": "q8_0",
    "int8": "q8_0",
    "q4": "q4_0",
    "4bit": "q4_0",
    "int4": "q4_0",
    "fp16": "f16",
    "float16": "f16",
    "fp32": "f32",
    "float32": "f32",
}
_VALID_CACHE_TYPES = {
    "f32",
    "f16",
    "bf16",
    "q8_0",
    "q4_0",
    "q4_1",
    "iq4_nl",
    "q5_0",
    "q5_1",
}


def _normalize_cache_type_value(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    text = _CACHE_TYPE_ALIASES.get(text, text)
    if text in _VALID_CACHE_TYPES:
        return text
    return None



def _catalog_model_key_names() -> list[str]:
    return sorted(f.name for f in fields(ManagedModel))


def _catalog_known_top_level_key_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for name in _catalog_model_key_names():
        dashed = name.replace("_", "-")
        if dashed != name:
            aliases[dashed] = name
    return aliases


def catalog_key_warnings(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text("utf-8"))
    except Exception as exc:
        return [f"Could not inspect catalog keys in {path}: {exc}"]
    if not isinstance(payload, list):
        return []
    model_keys = set(_catalog_model_key_names())
    aliases = _catalog_known_top_level_key_aliases()
    warnings: list[str] = []
    for idx, raw_item in enumerate(payload):
        if not isinstance(raw_item, dict):
            continue
        model_id = str(raw_item.get("model_id") or f"entry#{idx}")
        for raw_key in raw_item:
            key = str(raw_key).strip()
            normalized = key.lower()
            if key in model_keys or key == "server_overrides":
                continue
            if normalized in aliases:
                warnings.append(
                    f"catalog.json model {model_id}: top-level key {key!r} is not a catalog model key. "
                    f"Use {aliases[normalized]!r} for the managed model field, or put raw llama.cpp flag "
                    f"{key!r} under server_overrides."
                )
            elif "-" in key and normalized.replace("-", "_") in model_keys:
                canonical = normalized.replace("-", "_")
                warnings.append(
                    f"catalog.json model {model_id}: top-level key {key!r} looks like managed field {canonical!r}. "
                    f"Use snake_case top-level keys; raw llama.cpp flags belong under server_overrides."
                )
    return warnings


def print_config_keys(args) -> int:
    catalog_keys = _catalog_model_key_names()
    server_default_keys = sorted(set(_load_bundled_llama_server_default_values().keys()) | set(resolve_llama_server_defaults(args).keys()))
    global_keys = sorted(k for k in normalize_server_config_payload({})[0].keys() if not k.startswith("_"))
    resolved_vllm = resolve_vllm_defaults(args)
    vllm_keys = sorted(key for key in resolved_vllm if key != "__family_defaults")
    family_vllm = resolved_vllm.get("__family_defaults")
    if isinstance(family_vllm, dict):
        for pattern, pattern_values in family_vllm.items():
            if isinstance(pattern_values, dict):
                vllm_keys.extend(f"family_defaults.{pattern}.{key}" for key in pattern_values)
    vllm_keys = sorted(set(vllm_keys))
    if getattr(args, "format", "text") == "json":
        print(json.dumps({
            "catalog_model_top_level_keys": catalog_keys,
            "conf_json_top_level_keys": global_keys,
            "llama_server_defaults_keys": server_default_keys,
            "vllm_keys": vllm_keys,
            "notes": [
                "Catalog model top-level keys use snake_case.",
                "Raw llama.cpp flags belong under server_overrides and may use dash or underscore spelling.",
            ],
        }, indent=2, ensure_ascii=False))
        return 0
    print("Catalog model top-level keys (snake_case):")
    for key in catalog_keys:
        print(f"  {key}")
    print("\nconf.json top-level keys:")
    for key in global_keys:
        print(f"  {key}")
    print("\nllama_server_defaults keys seen/defaulted:")
    for key in server_default_keys:
        print(f"  {key}")
    print("\nvLLM configuration keys:")
    for key in vllm_keys:
        print(f"  {key}")
    print("\nRule:")
    print("  - Use snake_case for managed catalog fields, e.g. tensor_split.")
    print("  - Put raw llama.cpp flags in server_overrides, e.g. {'batch-size': 1024}.")
    print("  - Put vLLM-specific flags in server_overrides.vllm.")
    return 0

def _server_config_validation_warnings(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    defaults = payload.get("llama_server_defaults")
    if not isinstance(defaults, dict):
        return []
    warnings: list[str] = []
    for key in ("cache_type_k", "cache_type_v", "cache-type-k", "cache-type-v"):
        if key not in defaults:
            continue
        raw = defaults.get(key)
        raw_text = str(raw or "").strip()
        normalized = _normalize_cache_type_value(raw)
        if normalized is None:
            warnings.append(
                f"Invalid llama_server_defaults.{key}={raw_text!r}; omitting this KV cache flag. "
                f"Use one of: {', '.join(sorted(_VALID_CACHE_TYPES))}."
            )
            continue
        if raw_text and normalized != raw_text:
            warnings.append(
                f"Non-canonical llama_server_defaults.{key}={raw_text!r}; normalized to {normalized!r}."
            )
    return warnings


def _headers_lower(headers) -> dict[str, str]:
    try:
        return {str(k).lower(): str(v) for k, v in headers.items()}
    except Exception:
        return {}


def _first_header(headers: dict[str, str], names: list[str]) -> str:
    for name in names:
        value = headers.get(name.lower())
        if value:
            return str(value).strip()
    return ""


def _payload_string(payload: dict, names: list[str]) -> str:
    for name in names:
        value = payload.get(name)
        if isinstance(value, (str, int, float)) and str(value).strip():
            return str(value).strip()
    return ""


def _message_text_fingerprint(payload: dict) -> str:
    pieces: list[str] = []
    messages = payload.get("messages")
    if isinstance(messages, list):
        for item in messages[:3]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "")
            content = item.get("content")
            if isinstance(content, str):
                text = content[:500]
            elif isinstance(content, list):
                text = " ".join(str(part.get("text") or "") for part in content if isinstance(part, dict))[:500]
            else:
                text = str(content or "")[:500]
            pieces.append(f"{role}:{text}")
    raw_input = payload.get("input")
    if isinstance(raw_input, str):
        pieces.append(raw_input[:800])
    elif isinstance(raw_input, list):
        pieces.append(json.dumps(_redact_large_log_value(raw_input[:3]), sort_keys=True, ensure_ascii=False)[:1000])
    return hashlib.sha256("\n".join(pieces).encode("utf-8", errors="ignore")).hexdigest()[:24]


def _agent_fingerprint_from_payload(payload: dict) -> str:
    pieces: list[str] = []
    messages = payload.get("messages")
    if isinstance(messages, list):
        for item in messages[:8]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").lower()
            if role not in {"system", "developer", "tool"}:
                continue
            content = item.get("content")
            if isinstance(content, str):
                text = content[:1200]
            elif isinstance(content, list):
                text = " ".join(str(part.get("text") or "") for part in content if isinstance(part, dict))[:1200]
            else:
                text = str(content or "")[:1200]
            if text.strip():
                pieces.append(f"{role}:{text}")
    if not pieces:
        return "default"
    return hashlib.sha256("\n".join(pieces).encode("utf-8", errors="ignore")).hexdigest()[:12]


def _request_session_and_agent(payload: dict, headers) -> tuple[str, str]:
    h = _headers_lower(headers)
    session = _first_header(h, [
        "thread-id", "session-id", "x-conversation-id", "x-session-id", "x-thread-id",
        "helicone-session-id", "x-opencode-session",
    ]) or _payload_string(payload, [
        "thread_id", "session_id", "conversation_id", "conversation", "user", "promptCacheKey", "prompt_cache_key",
    ])
    agent = _first_header(h, ["x-agent-id", "x-subagent-id", "x-opencode-agent", "x-codex-subagent"]) or _payload_string(payload, ["agent_id", "subagent_id"])
    if not agent:
        # With an explicit conversation/session, do not include user text in the
        # agent fallback: user text changes every turn and would break sticky
        # routing. OpenCode agents/subagents are instead distinguished by stable
        # system/developer/tool prompt prefixes when no explicit agent id exists.
        agent = _agent_fingerprint_from_payload(payload)
    return session, agent


def resolve_request_conversation_key(payload: dict, headers) -> str:
    """Return conversation/agent identity without model id for cancellation.

    Unlike replica affinity, this key is intentionally model-agnostic so a
    request that switches the same conversation from Gemma to Qwen can cancel
    old Gemma repair rounds. Fallback fingerprints are not reliable enough for
    cancellation because they change with prompt text.
    """
    session, agent = _request_session_and_agent(payload, headers)
    if not session:
        # Best-effort for clients that do not expose explicit conversation ids.
        # This is weaker than session/agent, but still model-agnostic and stable
        # enough for long-running same-conversation model switches because it
        # hashes the early message prefix rather than the current model.
        return f"fallback:{_message_text_fingerprint(payload)}:agent:{agent}"
    return f"session:{session}:agent:{agent}"


def resolve_request_affinity_key(model_name: str, payload: dict, headers) -> str:
    session, agent = _request_session_and_agent(payload, headers)
    previous_response_id = str(payload.get("previous_response_id") or "").strip()
    if previous_response_id:
        mapped = REPLICA_ROUTER_STATE.response_replica(previous_response_id)
        if mapped:
            return f"{model_name}:response:{previous_response_id}"
    if session:
        return f"{model_name}:session:{session}:agent:{agent}"
    return f"{model_name}:fallback:{_message_text_fingerprint(payload)}"


def is_fallback_affinity_key(affinity_key: str) -> bool:
    return ":fallback:" in str(affinity_key or "")


def _query_gpu_memory_snapshot_cached(ttl_s: float = 1.0) -> dict[int, dict[str, float]]:
    now = time.monotonic()
    with REPLICA_ROUTER_STATE.lock:
        ts, snap = REPLICA_ROUTER_STATE.gpu_snapshot
        if snap and now - ts <= ttl_s:
            return {k: dict(v) for k, v in snap.items()}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        snap: dict[int, dict[str, float]] = {}
        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4 and parts[0].isdigit():
                snap[int(parts[0])] = {"used_mib": float(parts[1]), "free_mib": float(parts[2]), "total_mib": float(parts[3])}
    except Exception:
        snap = {}
    with REPLICA_ROUTER_STATE.lock:
        REPLICA_ROUTER_STATE.gpu_snapshot = (now, {k: dict(v) for k, v in snap.items()})
    return snap


def estimate_model_runtime_mib(model: ManagedModel) -> float | None:
    try:
        size_mib = Path(model.local_path).stat().st_size / (1024 * 1024)
    except Exception:
        return None
    overrides = model.server_overrides or {}
    ctx = int(overrides.get("ctx_size", model.ctx_size) or model.ctx_size or 8192)
    parallel = int(overrides.get("parallel", 1) or 1)
    batch = int(overrides.get("batch_size", 2048) or 2048)
    ubatch = int(overrides.get("ubatch_size", 512) or 512)
    kv_mib = (ctx * parallel * (size_mib / 2048.0) * 0.5) / 1024.0
    compute_mib = (ubatch * 0.5) + (batch * 0.1)
    return (size_mib * 1.08) + kv_mib + compute_mib + 700.0


def _placement_fits(model: ManagedModel, cfg: ReplicaConfig, gpu_set: list[int], used_by_records: dict[int, int]) -> tuple[bool, float | None]:
    required = estimate_model_runtime_mib(model)
    if required is None:
        return False, None
    snap = _query_gpu_memory_snapshot_cached()
    if not snap:
        # Test/offline fallback: allow exclusive placement when GPU discovery is unavailable.
        return cfg.placement == "exclusive_gpus", required
    gpu_count = max(1, len(gpu_set))
    # estimate_model_runtime_mib returns a whole-process estimate. For a
    # tensor-split replica spread over N GPUs, comparing that total against
    # each GPU is overly conservative and prevents scale-out. Use a per-GPU
    # estimate plus a fixed overhead/safety cushion.
    required_per_gpu = (required / gpu_count) + 1024.0
    if cfg.placement == "pack_small_models":
        for gpu in gpu_set:
            total = snap.get(gpu, {}).get("total_mib", 0.0)
            if total > 0 and required_per_gpu > total * cfg.max_pack_fraction:
                log_api_event("replica_placement_reject", {"model": model.model_id, "gpu_set": gpu_set, "gpu": gpu, "reason": "pack_fraction", "required_total_mib": required, "required_per_gpu_mib": required_per_gpu, "total_mib": total, "max_pack_fraction": cfg.max_pack_fraction})
                return False, required
            if used_by_records.get(gpu, 0) >= cfg.max_models_per_gpu:
                log_api_event("replica_placement_reject", {"model": model.model_id, "gpu_set": gpu_set, "gpu": gpu, "reason": "max_models_per_gpu", "used": used_by_records.get(gpu, 0), "max_models_per_gpu": cfg.max_models_per_gpu})
                return False, required
    for gpu in gpu_set:
        free = snap.get(gpu, {}).get("free_mib", 0.0)
        if required_per_gpu + cfg.safety_vram_mib > free:
            log_api_event("replica_placement_reject", {"model": model.model_id, "gpu_set": gpu_set, "gpu": gpu, "reason": "vram", "required_total_mib": required, "required_per_gpu_mib": required_per_gpu, "safety_vram_mib": cfg.safety_vram_mib, "free_mib": free})
            return False, required
    return True, required


def select_replica_for_request(
    base_model: ManagedModel,
    payload: dict,
    headers,
    replica_defaults: dict[str, object] | None = None,
    published_model_ids: set[str] | None = None,
    catalog: list[ManagedModel] | None = None,
    config_path: Path | str | None = None,
    server_path: Path | str | None = None,
    idle_ttl: int | None = None,
    server_defaults: dict[str, object] | None = None,
    public_host: str = DEFAULT_PUBLIC_HOST,
    public_port: int = DEFAULT_PUBLIC_PORT,
) -> tuple[str, str, bool]:
    """Return (upstream_model_id, affinity_key, is_replica)."""
    cfg = get_model_replica_config(base_model, replica_defaults)
    if not cfg.enabled:
        return base_model.model_id, "", False
    now = time.monotonic()
    REPLICA_ROUTER_STATE.prune(now)
    previous_response_id = str(payload.get("previous_response_id") or "").strip()
    mapped_response_replica = REPLICA_ROUTER_STATE.response_replica(previous_response_id) if previous_response_id else None
    affinity_key = resolve_request_affinity_key(base_model.model_id, payload, headers)
    gpu_sets = _replica_gpu_sets(base_model, cfg)
    if not gpu_sets:
        log_api_event("replica_no_gpu_sets", {"model": base_model.model_id, "replicas_max": cfg.max, "gpus_per_replica": cfg.gpus_per_replica})
        return base_model.model_id, affinity_key, False
    replica_ids = [replica_model_id(base_model.model_id, idx) for idx in range(len(gpu_sets))]
    published = set(published_model_ids) if published_model_ids is not None else get_published_model_ids()
    published_replica_ids = [rid for rid in replica_ids if rid in published]
    dynamic_routes_enabled = catalog is not None and config_path is not None and server_path is not None
    if not published_replica_ids and not dynamic_routes_enabled:
        with REPLICA_ROUTER_STATE.lock:
            REPLICA_ROUTER_STATE.affinity[affinity_key] = (base_model.model_id, now + cfg.sticky_ttl_s)
        log_api_event(
            "replica_routes_missing",
            {
                "model": base_model.model_id,
                "replicas_max": cfg.max,
                "expected_replicas": replica_ids,
                "published_sample": sorted(list(published))[:20],
            },
        )
        return base_model.model_id, affinity_key, False
    with REPLICA_ROUTER_STATE.lock:
        for idx, rid in enumerate(replica_ids):
            rec = REPLICA_ROUTER_STATE.records.setdefault(
                rid,
                ReplicaRecord(base_model_id=base_model.model_id, replica_model_id=rid, gpu_set=gpu_sets[idx]),
            )
            rec.gpu_set = list(gpu_sets[idx])
        bound = REPLICA_ROUTER_STATE.affinity.get(affinity_key)
        if mapped_response_replica and mapped_response_replica in replica_ids:
            REPLICA_ROUTER_STATE.affinity[affinity_key] = (mapped_response_replica, now + cfg.sticky_ttl_s)
            return mapped_response_replica, affinity_key, True
        if bound and bound[1] > now and bound[0] in replica_ids:
            return bound[0], affinity_key, True
        if bound and bound[1] > now and bound[0] == base_model.model_id:
            return base_model.model_id, affinity_key, False
        if REPLICA_ROUTER_STATE.base_in_flight.get(base_model.model_id, 0) <= 0:
            REPLICA_ROUTER_STATE.affinity[affinity_key] = (base_model.model_id, now + cfg.sticky_ttl_s)
            log_api_event("replica_base_selected_idle", {"model": base_model.model_id, "affinity_key": affinity_key})
            return base_model.model_id, affinity_key, False
        candidates = [
            REPLICA_ROUTER_STATE.records[rid]
            for rid in replica_ids
            if REPLICA_ROUTER_STATE.records[rid].blacklist_until <= now
        ]
        # For a new conversation/agent with no sticky binding, prefer scaling
        # out to a cold replica if placement fits. Reusing an idle ready replica
        # first keeps latency low but prevents the intended multi-server spread
        # across available GPUs for independent conversations.
        used_by_records: dict[int, int] = {}
        for rec in REPLICA_ROUTER_STATE.records.values():
            if rec.status in {"ready", "loading"}:
                for gpu in rec.gpu_set:
                    used_by_records[gpu] = used_by_records.get(gpu, 0) + 1
        cold = [r for r in candidates if r.status in {"cold", "error"}]
    # Potentially slow VRAM checks happen outside the lock.
    for rec in sorted(cold, key=lambda r: r.replica_model_id):
        fits, required = _placement_fits(base_model, cfg, rec.gpu_set, used_by_records)
        if not fits:
            continue
        with REPLICA_ROUTER_STATE.lock:
            # Re-verify state in case another concurrent request claimed this replica
            # while we were performing the slow VRAM check outside the lock.
            if rec.status not in {"cold", "error"}:
                continue
            rec.estimated_mib = required
            rec.status = "loading"
            REPLICA_ROUTER_STATE.affinity[affinity_key] = (rec.replica_model_id, time.monotonic() + cfg.sticky_ttl_s)
        if dynamic_routes_enabled and rec.replica_model_id not in published:
            try:
                replica_index = replica_ids.index(rec.replica_model_id)
                ensure_replica_route_in_llamaswap_config(
                    base_model,
                    replica_index,
                    rec.gpu_set,
                    catalog or [base_model],
                    config_path,
                    server_path,
                    int(idle_ttl or DEFAULT_IDLE_TTL),
                    server_defaults=server_defaults,
                )
                if not wait_for_published_model_id(rec.replica_model_id, public_host, int(public_port), timeout_s=5.0):
                    raise RuntimeError(f"replica route {rec.replica_model_id} was written but not published by llama-swap")
            except Exception as exc:
                log_api_event("replica_route_add_failed", {"model": base_model.model_id, "replica": rec.replica_model_id, "error": str(exc)})
                with REPLICA_ROUTER_STATE.lock:
                    rec.status = "error"
                continue
        log_api_event("replica_selected_cold", {"model": base_model.model_id, "replica": rec.replica_model_id, "gpu_set": rec.gpu_set, "estimated_mib": required})
        return rec.replica_model_id, affinity_key, True
    with REPLICA_ROUTER_STATE.lock:
        candidates = [
            REPLICA_ROUTER_STATE.records[rid]
            for rid in replica_ids
            if REPLICA_ROUTER_STATE.records[rid].blacklist_until <= now
        ]
        ready_empty = [r for r in candidates if r.status == "ready" and r.in_flight == 0]
        if ready_empty:
            chosen = sorted(ready_empty, key=lambda r: r.last_used)[0]
            REPLICA_ROUTER_STATE.affinity[affinity_key] = (chosen.replica_model_id, now + cfg.sticky_ttl_s)
            log_api_event("replica_selected_ready_empty", {"model": base_model.model_id, "replica": chosen.replica_model_id})
            return chosen.replica_model_id, affinity_key, True
        if not candidates:
            log_api_event("replica_no_unblacklisted_routes", {"model": base_model.model_id, "replicas": replica_ids})
            return base_model.model_id, affinity_key, False
        chosen = sorted(candidates, key=lambda r: (r.in_flight, 1 if r.status == "loading" else 0, r.last_used))[0]
        REPLICA_ROUTER_STATE.affinity[affinity_key] = (chosen.replica_model_id, now + cfg.sticky_ttl_s)
        log_api_event("replica_selected_loaded", {"model": base_model.model_id, "replica": chosen.replica_model_id, "in_flight": chosen.in_flight, "status": chosen.status})
        return chosen.replica_model_id, affinity_key, True




def replica_trace_state_for_base(base_model_id: str) -> dict[str, object]:
    with REPLICA_ROUTER_STATE.lock:
        records = [
            {
                "replica": rec.replica_model_id,
                "status": rec.status,
                "in_flight": rec.in_flight,
                "gpu_set": list(rec.gpu_set),
                "pid": rec.pid,
                "port": rec.port,
                "actual_mib": rec.actual_mib,
                "blacklist_until": rec.blacklist_until,
            }
            for rec in REPLICA_ROUTER_STATE.records.values()
            if rec.base_model_id == base_model_id
        ]
        affinities = [
            {"key": key, "target": target, "expires_in_s": max(0.0, expires - time.monotonic())}
            for key, (target, expires) in REPLICA_ROUTER_STATE.affinity.items()
            if str(key).startswith(f"{base_model_id}:")
        ]
        return {
            "base_in_flight": REPLICA_ROUTER_STATE.base_in_flight.get(base_model_id, 0),
            "records": records,
            "affinities": affinities[:10],
        }

def public_model_id_for_response(model_id: str) -> str:
    return replica_base_model_id(model_id) if is_replica_model_id(model_id) else model_id


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


def recent_activity_blocking_model_switch(
    target_model_id: str,
    activity: dict[str, dict[str, float | str]],
    *,
    now: float,
    grace_s: int = DEFAULT_MODEL_SWITCH_GRACE_S,
) -> tuple[str, float, str] | None:
    """Return the most recent different model active inside the switch grace window."""
    target = str(target_model_id or "")
    best: tuple[str, float, str] | None = None
    for model_id, record in activity.items():
        if not model_id or model_id == target:
            continue
        try:
            ts = float(record.get("last_activity_monotonic", 0.0))
        except Exception:
            continue
        age = now - ts
        if age < 0 or age > grace_s:
            continue
        phase = str(record.get("last_phase") or "")
        if best is None or age < best[1]:
            best = (model_id, age, phase)
    return best


def request_looks_like_model_probe(payload: dict) -> bool:
    """Heuristic for client model probes that should not autoload/evict a model."""
    if not isinstance(payload, dict):
        return False
    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        return False
    if payload.get("tool_choice") not in (None, "", "none", "auto"):
        return False
    raw_input = payload.get("input")
    if isinstance(raw_input, list):
        return len(raw_input) <= 1
    messages = payload.get("messages")
    if isinstance(messages, list):
        return len(messages) <= 1 and not bool(payload.get("functions"))
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        return len(prompt.strip()) <= 128
    return False


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
    # Do not fight llama-swap's normal TTL unload near the end of the idle
    # window. The guard is for abrupt unloads shortly after activity, not for
    # keeping an idle model alive forever or reloading it seconds before TTL.
    ttl_tail_grace = min(30.0, max(5.0, idle_ttl * 0.1))
    if age >= max(0.0, idle_ttl - ttl_tail_grace):
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
    replica_defaults: dict[str, object] | None = None,
):
    stop_running_ollama_models(progress_callback=progress_callback)
    render_llamaswap_config(
        catalog,
        config_path,
        llama_server,
        start_port,
        resolve_idle_ttl(),
        server_defaults=server_defaults,
        replica_defaults=replica_defaults,
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
    return _env_value("HEIMDALL_GATEWAY_DOWNLOAD_BACKEND", "LLAMACPP_DOWNLOAD_BACKEND", "parallel").strip().lower()


def _should_use_hf_transfer() -> bool:
    raw = _env_value("HEIMDALL_GATEWAY_USE_HF_TRANSFER", "LLAMACPP_USE_HF_TRANSFER", "1").strip().lower()
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
    raw = _env_value("HEIMDALL_GATEWAY_DOWNLOAD_WORKERS", "LLAMACPP_DOWNLOAD_WORKERS", "8").strip()
    try:
        value = int(raw)
    except Exception:
        value = 8
    return max(2, min(value, 32))


def _download_hf_file_parallel(
    repo_id: str,
    filename: str,
    token: str | None,
    target_dir: Path,
    label: str | None = None,
    progress_callback = None,
    activity_guard = None,
) -> str | None:
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
                    if activity_guard is not None:
                        activity_guard()
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
    activity_guard = None,
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

    if activity_guard is not None:
        activity_guard()

    parallel_path = _download_hf_file_parallel(
        repo_id,
        filename,
        token,
        target_dir,
        label=label,
        progress_callback=progress_callback,
        activity_guard=activity_guard,
    )
    if parallel_path:
        return str(Path(parallel_path))

    if activity_guard is not None:
        activity_guard()

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
                    if activity_guard is not None:
                        activity_guard()
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
    explicit = _env_value("HEIMDALL_GATEWAY_INSTALL_MODE", "LLAMACPP_INSTALL_MODE", "").strip().lower()
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
        "/etc/heimdall-gateway",
        "/var/lib/heimdall-gateway",
        "/opt/heimdall-gateway",
    )
    config_values = [str(path.expanduser()) for path in config_paths]
    if any(any(marker in value for marker in system_markers) for value in config_values):
        return "system"

    socket_path = str(Path(SOCKET_PATH).expanduser())
    if "/run/heimdall-gateway" in socket_path:
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
            "Register models first with 'heimdall-gateway run <hf-repo[:quant]>' or 'add'."
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

def _active_llama_server_summary(catalog: list[ManagedModel] | None = None) -> str:
    processes = get_llama_server_processes()
    if not processes:
        return ""
    names: list[str] = []
    if catalog:
        by_path = {_safe_realpath(model.local_path): model.model_id for model in catalog}
        for proc in processes:
            name = by_path.get(proc.get("model_path") or "")
            if name:
                names.append(f"{name}(pid={proc.get('pid')})")
    if not names:
        names = [f"pid={proc.get('pid')}" for proc in processes]
    return ", ".join(names)


def _active_api_work_summary(*, recent_s: float = 30.0) -> str:
    """Return a short summary when the API wrapper is actively serving or just served traffic."""
    parts: list[str] = []
    now = time.monotonic()
    try:
        with REPLICA_ROUTER_STATE.lock:
            in_flight = {
                str(model_id): int(count)
                for model_id, count in REPLICA_ROUTER_STATE.base_in_flight.items()
                if int(count or 0) > 0
            }
            replica_in_flight = {
                str(model_id): int(rec.in_flight)
                for model_id, rec in REPLICA_ROUTER_STATE.records.items()
                if int(rec.in_flight or 0) > 0
            }
        for model_id, count in sorted({**in_flight, **replica_in_flight}.items()):
            parts.append(f"{model_id}(in_flight={count})")
    except Exception:
        pass

    try:
        activity, _last_model_id = get_model_activity_snapshot()
        for model_id, record in sorted(activity.items()):
            last = record.get("last_activity_monotonic")
            if not isinstance(last, (float, int)):
                continue
            age = now - float(last)
            if age < 0 or age > recent_s:
                continue
            phase = str(record.get("last_phase") or "")
            source = str(record.get("last_source") or "")
            if phase in {"request_start", "stream_chunk"}:
                parts.append(f"{model_id}({phase}, {age:.1f}s ago via {source})")
    except Exception:
        pass
    return ", ".join(parts)


def _active_download_blocker_summary(catalog: list[ManagedModel] | None = None) -> str:
    active_parts = []
    api_active = _active_api_work_summary()
    if api_active:
        active_parts.append(api_active)
    server_active = _active_llama_server_summary(catalog)
    if server_active:
        active_parts.append(server_active)
    return "; ".join(active_parts)


def _raise_if_download_unsafe_while_active(catalog: list[ManagedModel], *, force: bool, progress_callback = None) -> None:
    if force:
        return
    active = _active_download_blocker_summary(catalog)
    if not active:
        return
    message = (
        "Refusing to download/register missing model files while the API is busy or llama-server is active "
        f"({active}). Stop/unload active models first, or rerun with --force if you explicitly accept degraded service/reload risk."
    )
    _emit_message(message, progress_callback)
    raise RuntimeError(message)


def _explicit_hf_ref_quant(args) -> tuple[str, str] | None:
    raw = getattr(args, "repo", None) or getattr(args, "hf", None)
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    if not raw or getattr(args, "model_id", None):
        return None
    try:
        repo_id, quant = parse_hf_input(str(raw))
    except Exception:
        return None
    if not quant:
        return None
    return repo_id, quant


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
    backend = "llama.cpp"

    if requested_model is not None:
        explicit_ref = _explicit_hf_ref_quant(args)
        if explicit_ref is not None and not getattr(args, "file", None):
            ref_repo_id, ref_quant = explicit_ref
            try:
                hf_selected = choose_gguf_file(api, ref_repo_id, ref_quant, None, token)
            except Exception:
                hf_selected = None
            if hf_selected and requested_model.filename != hf_selected:
                _emit_message(
                    f"Catalog entry {requested_model.model_id} points to {requested_model.filename}, "
                    f"but {ref_repo_id}:{ref_quant} resolves to {hf_selected}. Ignoring stale catalog alias.",
                    progress_callback,
                )
                requested_model = None
        if requested_model is not None:
            existing = requested_model
            repo_id = existing.repo_id
            quant = existing.quant
            selected_file = existing.filename
            backend = _model_backend(existing)
            to_download = infer_shard_filenames(selected_file)
            if len(to_download) > 1:
                _emit_message(f"Using catalog model {existing.model_id} with {len(to_download)} shards.", progress_callback)
        else:
            existing = None
    if requested_model is None:
        if args.model_id:
            raise RuntimeError(f"Model {args.model_id} not found in catalog.")
        ref = args.repo or args.hf
        repo_id, quant = parse_hf_input(ref)
        selected_file = choose_gguf_file(api, repo_id, quant, args.file, token)

        if selected_file is None:
            # A repository without GGUF is a native HuggingFace model. Keep a
            # virtual filename in the catalog while storing the complete
            # snapshot below models_dir.
            backend = "vllm"
            selected_file = "hf-native"
            # snapshot_download owns the complete native repository transfer;
            # do not download individual siblings first.
            to_download = []
        else:
            backend = "llama.cpp"
            to_download = [selected_file]
        if "-00001-of-" in selected_file:
            _emit_message("Detected sharded model. Resolving all parts...", progress_callback)
            prefix = selected_file.split("-00001-of-")[0]
            info = api.model_info(repo_id=repo_id, token=token)
            repo_files = [s.rfilename for s in info.siblings if s.rfilename]
            to_download = sorted([f for f in repo_files if f.startswith(prefix) and "-of-" in f and f.lower().endswith(".gguf")])
            _emit_message(f"Found {len(to_download)} shards in the repository.", progress_callback)

        existing = next((m for m in catalog if m.repo_id == repo_id and m.filename == selected_file), None)

    expected_sizes = _repo_sibling_sizes(api, repo_id, token)

    native_snapshot_ready = bool(
        backend == "vllm"
        and selected_file == "hf-native"
        and existing
        and Path(existing.local_path).is_dir()
    )
    if native_snapshot_ready:
        to_download = []

    target_dir = args.models_dir / repo_id
    target_dir.mkdir(parents=True, exist_ok=True)
    mmproj_filename = existing.mmproj_filename if existing else None
    mmproj_path = existing.mmproj_path if existing else None
    if backend == "llama.cpp" and _looks_like_vision_model(existing if existing else ManagedModel(model_id=mid if 'mid' in locals() else "", repo_id=repo_id, quant=quant, filename=selected_file, local_path="", description=args.description or f"{repo_id} / {selected_file}")):
        try:
            mmproj_filename = choose_mmproj_file(api, repo_id, token)
        except Exception:
            mmproj_filename = existing.mmproj_filename if existing else None
        if mmproj_filename:
            mmproj_path = str(target_dir / mmproj_filename)
    mmproj_ready = bool(not mmproj_filename or (mmproj_path and Path(mmproj_path).exists()))
    mtp_filename = None
    mtp_path = None
    if selected_file != "hf-native":
        mtp_filename = _detect_mtp_drafter_file(api, repo_id, token, selected_file)
        if mtp_filename:
            mtp_path = str(target_dir / mtp_filename)
    mtp_ready = bool(not mtp_filename or (mtp_path and Path(mtp_path).exists()))
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
    elif existing and skip_ctx and (existing.auto_ctx_failed or existing.auto_ctx_error):
        # --skip-ctx means do not run auto-probing now; it must not silently
        # downgrade an already tuned model to the installer default.  Preserve
        # the existing ctx_size and only clear the failure state so the model can
        # be republished with other metadata/config changes such as mmproj.
        existing.auto_ctx_failed = False
        existing.auto_ctx_error = ""
        refresh_model_load_capabilities(existing)
        save_catalog(args.catalog, catalog)
        ctx_changed = True
        auto_ctx_failed = False
        auto_ctx_error = ""
        _emit_message(f"Preserving existing ctx for {existing.model_id}: {existing.ctx_size or default_ctx}", progress_callback)

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
        requested_tensor_split = getattr(args, "tensor_split", None)
        if requested_tensor_split is not None and existing.tensor_split != requested_tensor_split:
            existing.tensor_split = requested_tensor_split
            config_changed = True
            _emit_message(f"Applied tensor_split for {existing.model_id}: {requested_tensor_split}", progress_callback)
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
        if mtp_filename and mtp_path:
            updated_overrides, mtp_changed = _apply_mtp_server_overrides(existing.server_overrides, mtp_path, active_server_defaults)
            if mtp_changed or updated_overrides != normalize_server_overrides(existing.server_overrides):
                existing.server_overrides = updated_overrides
                config_changed = True
                _emit_message(
                    f"Detected MTP drafter for {existing.model_id}: {mtp_filename}; enabling draft-mtp.",
                    progress_callback,
                )
            elif str(normalize_server_overrides(existing.server_overrides).get("model_draft") or "").strip() == mtp_path:
                _emit_message(
                    f"MTP drafter already configured for {existing.model_id}: {mtp_filename}.",
                    progress_callback,
                )
        elif _looks_like_integrated_mtp_model(repo_id, selected_file, existing.model_id, existing.local_path):
            updated_overrides, mtp_changed = _apply_mtp_server_overrides(existing.server_overrides, None, active_server_defaults)
            if mtp_changed or updated_overrides != normalize_server_overrides(existing.server_overrides):
                existing.server_overrides = updated_overrides
                config_changed = True
                _emit_message(
                    f"Detected integrated MTP model for {existing.model_id}: {selected_file}; enabling draft-mtp.",
                    progress_callback,
                )
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

    if existing and not is_speculative_request and not args.force and files_ready and mmproj_ready and mtp_ready and not force_auto_ctx:
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
    elif existing and not mtp_ready:
        _emit_message("Catalog entry exists but MTP drafter is missing. Downloading the MTP drafter...", progress_callback)
    if len(to_download) > 1 and (completed_files or partial_files):
        remaining_files = partial_files + missing_files
        _emit_message(
            f"Resume status: {completed_files}/{len(to_download)} complete, "
            f"{partial_files} partial, {remaining_files} remaining.",
            progress_callback,
        )
    elif len(to_download) == 1 and partial_files:
        _emit_message("Resume status: single file partially downloaded, continuing from the saved offset.", progress_callback)

    needs_download = (
        (not files_ready)
        or (bool(mmproj_filename) and not mmproj_ready)
        or (bool(mtp_filename) and not mtp_ready)
        or (selected_file == "hf-native" and not native_snapshot_ready)
    )
    has_force = bool(getattr(args, "force", False))
    if needs_download:
        _raise_if_download_unsafe_while_active(catalog, force=has_force, progress_callback=progress_callback)
    download_activity_guard = None
    if needs_download and not has_force:
        download_activity_guard = lambda: _raise_if_download_unsafe_while_active(catalog, force=has_force, progress_callback=progress_callback)
    
    local_path = str(existing.local_path) if native_snapshot_ready and existing else ""
    total_files = len(to_download)
    for idx, f in enumerate(to_download, start=1):
        if f == "hf-native":
            continue
        if _download_file_state(target_dir, f, expected_sizes.get(f)) == "completed":
            if f == selected_file:
                local_path = str(target_dir / f)
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
            activity_guard=download_activity_guard,
        )
        if f == selected_file:
            local_path = loc
    if selected_file == "hf-native" and not native_snapshot_ready:
        _emit_message(f"Populating native HF repo {repo_id}...", progress_callback)
        local_path = snapshot_download(
            repo_id=repo_id,
            token=token,
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
    if not mmproj_filename and selected_file != "hf-native" and local_path:
        capability_probe = ManagedModel(
            model_id=mid,
            repo_id=repo_id,
            quant=quant,
            filename=selected_file,
            local_path=str(local_path),
            description=args.description or f"{repo_id} / {selected_file}",
        )
        detected_capabilities = refresh_model_load_capabilities(capability_probe)
        if _model_capabilities_imply_vision(capability_probe):
            try:
                detected_mmproj_filename = choose_mmproj_file(api, repo_id, token)
            except Exception:
                detected_mmproj_filename = None
            if detected_mmproj_filename:
                mmproj_filename = detected_mmproj_filename
                mmproj_path = str(target_dir / mmproj_filename)
                mmproj_ready = bool(Path(mmproj_path).exists())
                _emit_message(
                    f"Detected image-capable GGUF metadata ({', '.join(detected_capabilities)}); pairing mmproj {Path(mmproj_filename).name}.",
                    progress_callback,
                )
    if mmproj_filename:
        if _download_file_state(target_dir, mmproj_filename, expected_sizes.get(mmproj_filename)) == "completed":
            mmproj_path = str(target_dir / mmproj_filename)
        else:
            mmproj_label = f"mmproj {Path(mmproj_filename).name}"
            mmproj_loc = download_hf_file(
                repo_id=repo_id,
                filename=mmproj_filename,
                token=token,
                target_dir=target_dir,
                label=mmproj_label,
                progress_callback=progress_callback,
                expected_size=expected_sizes.get(mmproj_filename),
                activity_guard=download_activity_guard,
            )
            mmproj_path = mmproj_loc
    if mtp_filename:
        if _download_file_state(target_dir, mtp_filename, expected_sizes.get(mtp_filename)) == "completed":
            mtp_path = str(target_dir / mtp_filename)
        else:
            mtp_label = f"MTP {Path(mtp_filename).name}"
            mtp_loc = download_hf_file(
                repo_id=repo_id,
                filename=mtp_filename,
                token=token,
                target_dir=target_dir,
                label=mtp_label,
                progress_callback=progress_callback,
                expected_size=expected_sizes.get(mtp_filename),
                activity_guard=download_activity_guard,
            )
            mtp_path = mtp_loc

    probe_config_replaced = False
    desired_ctx = ctx_override if ctx_override is not None else (
        existing.ctx_size
        if existing and skip_ctx and existing.ctx_size
        else (existing.ctx_size if existing and (existing.auto_ctx_failed or existing.auto_ctx_error) else default_ctx)
    )
    if backend == "vllm" and ctx_override is None and not (existing and existing.ctx_size and skip_ctx):
        vllm_options = resolve_vllm_options(
            ManagedModel(
                model_id=mid,
                repo_id=repo_id,
                quant=quant,
                filename=selected_file,
                local_path=str(local_path),
                backend="vllm",
                ctx_size=default_ctx,
                tensor_split=args.tensor_split or default_tensor_split(),
                server_overrides=(existing.server_overrides if existing else {}),
            )
        )
        desired_ctx = int(vllm_options.get("max_model_len") or default_ctx)
        auto_ctx_failed = False
        auto_ctx_error = ""
        ctx_probe_read_s = None
        ctx_probe_tokens_s = None
        ctx_probe_totals_s = None
        ctx_probe_latency_ms = None
        ctx_probe_speed_tps = None
        ctx_probe_kv_gb = None
        ctx_probe_prompt_tokens = None
        _emit_message(f"Native HF model detected; using vLLM max_model_len {desired_ctx} without llama.cpp auto-ctx.", progress_callback)
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
    elif backend == "vllm":
        pass
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
        if existing and existing.ctx_size:
            _emit_message(f"Skipping automatic ctx tuning. Preserving existing ctx {desired_ctx} for {mid}.", progress_callback)
        else:
            _emit_message(f"Skipping automatic ctx tuning. Using default ctx {desired_ctx} for {mid}.", progress_callback)
    elif existing and (existing.auto_ctx_failed or existing.auto_ctx_error) and not force_auto_ctx:
        desired_ctx = existing.ctx_size or default_ctx
        auto_ctx_failed = bool(existing.auto_ctx_failed)
        auto_ctx_error = existing.auto_ctx_error or "previous-auto-ctx-failure"
        _emit_message(
            f"Previous auto ctx issue ({auto_ctx_error}). Using saved ctx {desired_ctx} without re-probing; use --force-auto-ctx to override.",
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
                "Re-run with `heimdall-gateway update --auto --model-id {}` after downloads finish.".format(mid),
                progress_callback,
            )
            # Keep desired_ctx as-is (fallback or default) and avoid probing.
            probe_config_replaced = False
        else:
            _emit_message(
                "Process: start at 8192, try a few larger values, keep a practical stable ctx, and avoid exhaustive slow probing.",
                progress_callback,
            )
            probe_server_overrides = normalize_server_overrides(existing.server_overrides) if existing else {}
            if mtp_filename and mtp_path:
                probe_server_overrides, _ = _apply_mtp_server_overrides(probe_server_overrides, mtp_path, active_server_defaults)
            elif _looks_like_integrated_mtp_model(repo_id, selected_file, mid, str(local_path)):
                probe_server_overrides, _ = _apply_mtp_server_overrides(probe_server_overrides, None, active_server_defaults)
            probe_model = ManagedModel(
                model_id=mid,
                repo_id=repo_id,
                quant=quant,
                filename=selected_file,
                local_path=str(local_path),
                backend=backend,
                mmproj_filename=mmproj_filename,
                mmproj_path=mmproj_path,
                ctx_size=default_ctx,
                n_gpu_layers=int(args.n_gpu_layers),
                tensor_split=args.tensor_split or default_tensor_split(),
                host=args.host,
                jinja=not args.no_jinja,
                ttl=resolve_idle_ttl(args),
                description=args.description or f"{repo_id} / {selected_file}",
                server_overrides=probe_server_overrides,
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
        backend=backend,
        mmproj_filename=mmproj_filename,
        mmproj_path=mmproj_path,
        ctx_size=desired_ctx,
        n_gpu_layers=int(args.n_gpu_layers),
        tensor_split=args.tensor_split or default_tensor_split(),
        host=args.host,
        jinja=not args.no_jinja,
        ttl=resolve_idle_ttl(args),
        description=args.description or f"{repo_id} / {selected_file}",
        downloaded_at=(existing.downloaded_at if existing and existing.downloaded_at else datetime.now(timezone.utc).isoformat()),
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
    if existing and existing.server_overrides and not is_speculative_request:
        new_m.server_overrides = dict(normalize_server_overrides(existing.server_overrides))
    if mtp_filename and mtp_path and not is_speculative_request:
        new_m.server_overrides, mtp_changed = _apply_mtp_server_overrides(new_m.server_overrides, mtp_path, active_server_defaults)
        if mtp_changed:
            _emit_message(
                f"Detected MTP drafter for {new_m.model_id}: {mtp_filename}; enabling draft-mtp.",
                progress_callback,
            )
    elif _looks_like_integrated_mtp_model(repo_id, selected_file, mid, local_path) and not is_speculative_request:
        new_m.server_overrides, mtp_changed = _apply_mtp_server_overrides(new_m.server_overrides, None, active_server_defaults)
        if mtp_changed:
            _emit_message(
                f"Detected integrated MTP model for {new_m.model_id}: {selected_file}; enabling draft-mtp.",
                progress_callback,
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
            if "draft" not in spec_overrides and str(spec_overrides.get("spec_type") or "") != "draft-mtp":
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
    replica_defaults: dict[str, object] | None = None,
):
    render_llamaswap_config(
        catalog,
        config_path,
        llama_server,
        start_port,
        resolve_idle_ttl(),
        server_defaults=server_defaults,
        replica_defaults=replica_defaults,
    )
    _emit_message("Config updated. Waiting for llama-swap --watch-config...", progress_callback)
    time.sleep(settle_time)
    if wait_for_model_absent(model_id, host, port, timeout=timeout):
        return True
    raise RuntimeError(
        f"Model {model_id} is still published after updating {config_path}. "
        "Ensure llama-swap is running with --watch-config and is watching that config file."
    )


def reload_model_runtime_from_catalog_config(
    model: ManagedModel,
    catalog: list[ManagedModel],
    args,
    host: str,
    port: int,
    *,
    progress_callback=None,
    unload_timeout: float = 45.0,
    reload_timeout: float = 45.0,
) -> bool:
    """Force llama-swap to drop a stale live process and publish it again.

    llama-swap publishes models from config.yaml, but an already-loaded
    llama-server process can survive a command change such as adding --mmproj.
    Temporarily removing just this model from the watched config makes
    llama-swap stop the old process; restoring the full catalog makes the next
    request load the model with the current command.
    """
    if model is None or not getattr(model, "model_id", None):
        return False
    model_id = model.model_id
    config_path = getattr(args, "config", None)
    llama_server = getattr(args, "llama_server", None)
    start_port = getattr(args, "start_port", None)
    if not config_path or not llama_server or start_port is None:
        return False
    server_defaults = resolve_llama_server_defaults(args)
    replica_defaults = resolve_global_replica_config(args)
    idle_ttl = resolve_idle_ttl(args)
    reduced_catalog = [item for item in catalog if item.model_id != model_id]
    try:
        log_api_event("model_runtime_reload_begin", {"model": model_id, "reason": "stale_runtime_flags"})
        render_llamaswap_config(
            reduced_catalog,
            config_path,
            llama_server,
            start_port,
            idle_ttl,
            server_defaults=server_defaults,
            replica_defaults=replica_defaults,
        )
        if not wait_for_model_absent([model_id], host, port, timeout=unload_timeout):
            log_api_event("model_runtime_reload_unload_timeout", {"model": model_id})
            return False
        render_llamaswap_config(
            catalog,
            config_path,
            llama_server,
            start_port,
            idle_ttl,
            server_defaults=server_defaults,
            replica_defaults=replica_defaults,
        )
        if not wait_for_model(model_id, host, port, timeout=reload_timeout):
            log_api_event("model_runtime_reload_restore_timeout", {"model": model_id})
            return False
        log_api_event("model_runtime_reload_done", {"model": model_id})
        return True
    except Exception as exc:
        log_api_event("model_runtime_reload_error", {"model": model_id, "error": str(exc), "traceback": traceback.format_exc(limit=6)})
        try:
            render_llamaswap_config(
                catalog,
                config_path,
                llama_server,
                start_port,
                idle_ttl,
                server_defaults=server_defaults,
                replica_defaults=replica_defaults,
            )
        except Exception:
            pass
        return False


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

    def add_path(raw_path: str) -> None:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            for nested in path.rglob("*"):
                if nested.is_file() and nested.name.lower().endswith(_MODEL_ARTIFACT_SUFFIXES):
                    refs.add(_safe_realpath(str(nested)))
            return
        refs.add(_safe_realpath(str(path)))

    for item in catalog:
        for candidate in _model_candidate_files(item):
            add_path(str(candidate))
        # Draft and projector paths can be supplied as per-model overrides and
        # are not necessarily duplicated in local_path/mmproj_path.
        overrides = getattr(item, "server_overrides", {}) or {}
        for key in ("model_draft", "mmproj", "mmproj_path"):
            value = str(overrides.get(key, "") or "").strip()
            if value:
                add_path(value)
    return refs


_MODEL_ARTIFACT_SUFFIXES = (
    ".gguf",
    ".gguf.part",
    ".safetensors",
    ".safetensors.part",
    ".bin",
    ".bin.part",
)


def _delete_path_with_permission_fallback(path: Path, root: Path | None = None) -> bool:
    """Delete a path, retrying with sudo only after a real permission failure.

    The root check is mandatory for cleanup commands so a malformed catalog or
    symlink cannot turn an orphan cleanup into an arbitrary file deletion.
    ``sudo`` is deliberately a last resort: normal Heimdall installs should
    have the model tree owned by the service account.
    """
    path = Path(path)
    if root is not None:
        try:
            path.resolve(strict=False).relative_to(Path(root).resolve(strict=False))
        except ValueError as exc:
            raise RuntimeError(f"Refusing to delete path outside models directory: {path}") from exc

    is_dir = path.is_dir() and not path.is_symlink()
    try:
        if is_dir:
            shutil.rmtree(path)
        else:
            path.unlink()
        return True
    except FileNotFoundError:
        return False
    except PermissionError as local_exc:
        sudo = shutil.which("sudo")
        if not sudo:
            raise RuntimeError(f"Could not delete {path}: permission denied and sudo is unavailable") from local_exc
        command = [sudo, "rm", "-rf" if is_dir else "-f", "--", str(path)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as sudo_exc:
            detail = (sudo_exc.stderr or sudo_exc.stdout or str(sudo_exc)).strip()
            raise RuntimeError(f"Could not delete {path} with sudo: {detail}") from sudo_exc
        return True


def find_orphan_model_files(catalog: list[ManagedModel], models_dir: Path) -> list[Path]:
    """Return model artifacts below ``models_dir`` absent from the catalog.

    Only known model artifact suffixes are considered. Configuration,
    tokenizer and documentation files are intentionally left untouched.
    """
    root = Path(models_dir).expanduser().resolve(strict=False)
    if not root.exists():
        return []
    referenced = _catalog_referenced_paths(catalog)
    orphans: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or not path.name.lower().endswith(_MODEL_ARTIFACT_SUFFIXES):
            continue
        if _safe_realpath(str(path)) not in referenced:
            orphans.append(path)
    return sorted(orphans, key=lambda item: str(item))


def remove_orphan_models(args, progress_callback=None) -> int:
    """List and optionally remove model artifacts not referenced by catalog."""
    try:
        is_owner = os.getuid() == 0 or os.getuid() == os.stat(args.catalog.parent).st_uid
    except OSError:
        is_owner = False
    if not is_owner:
        try:
            return run_manager_command("remove-orphans", args)
        except Exception as exc:
            raise manager_unavailable_error(exc)

    catalog = load_catalog(args.catalog, _args_server_config_path(args))
    orphans = find_orphan_model_files(catalog, args.models_dir)
    if not orphans:
        _emit_message(f"No orphan model artifacts found under {args.models_dir}.", progress_callback)
        return 0
    _emit_message(
        f"Found {len(orphans)} orphan model artifact(s) not referenced by catalog:",
        progress_callback,
    )
    for path in orphans:
        _emit_message(f"  {path}", progress_callback)

    if getattr(args, "dry_run", False):
        return 0
    if not getattr(args, "yes", False) and not _ask_confirmation(
        f"Delete these {len(orphans)} orphan artifact(s)?", progress_callback, default=False
    ):
        _emit_message("Orphan cleanup cancelled.", progress_callback)
        return 0

    deleted = 0
    for path in orphans:
        if _delete_path_with_permission_fallback(path, args.models_dir):
            deleted += 1
            _emit_message(f"Deleted orphan file {path}.", progress_callback)
            _prune_empty_dirs_under_root(path.parent, args.models_dir, progress_callback)
    _emit_message(f"Deleted {deleted} orphan model artifact(s).", progress_callback)
    return 0


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
            _delete_path_with_permission_fallback(candidate, models_dir)
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
                _delete_path_with_permission_fallback(repo_dir, models_dir)
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
                _delete_path_with_permission_fallback(file_path, models_dir)
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
        replica_defaults=resolve_global_replica_config(args),
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


def is_gemma4_sliding_window_long_context_model(model: ManagedModel) -> bool:
    local_path = Path(model.local_path) if model.local_path else None
    if local_path is None or not local_path.exists():
        haystack = " ".join(str(x or "") for x in (model.model_id, model.repo_id, model.filename)).lower()
        return "gemma-4" in haystack or "gemma4" in haystack
    try:
        saw_gemma4_ctx = False
        saw_sliding_window = False
        with local_path.open("rb") as fh:
            if fh.read(4) != b"GGUF":
                return False
            version_raw = fh.read(4)
            tensor_count_raw = fh.read(8)
            kv_count_raw = fh.read(8)
            if len(version_raw) != 4 or len(tensor_count_raw) != 8 or len(kv_count_raw) != 8:
                return False
            version = struct.unpack("<I", version_raw)[0]
            if version not in (2, 3):
                return False
            kv_count = struct.unpack("<Q", kv_count_raw)[0]
            for _ in range(kv_count):
                key = _read_gguf_string(fh)
                value_type_raw = fh.read(4)
                if len(value_type_raw) != 4:
                    return False
                value_type = struct.unpack("<I", value_type_raw)[0]
                value = _read_gguf_value(fh, value_type)
                lowered = key.lower()
                if lowered == "gemma4.context_length" and isinstance(value, (int, float)) and int(value) >= 131072:
                    saw_gemma4_ctx = True
                elif lowered == "gemma4.attention.sliding_window":
                    saw_sliding_window = True
                if saw_gemma4_ctx and saw_sliding_window:
                    return True
    except Exception:
        return False
    return False

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
            [
                "pgrep",
                "-af",
                "llama-server|vllm.entrypoints.openai.api_server|vllm serve|vllm-server",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
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
        try:
            argv = shlex.split(cmdline)
        except Exception:
            argv = cmdline.split()
        server_arg_index = next(
            (
                idx for idx, arg in enumerate(argv)
                if Path(str(arg)).name == "llama-server"
                or str(arg) in {"vllm-server", "vllm.entrypoints.openai.api_server"}
                or str(arg).endswith("vllm.entrypoints.openai.api_server")
            ),
            None,
        )
        if server_arg_index is None or "--port" not in argv[server_arg_index + 1:]:
            continue
        model_path = ""
        match = re.search(r"--model\s+(\S+)", cmdline)
        if match:
            model_path = match.group(1).strip("'\"")
        port = None
        port_match = re.search(r"--port\s+(\d+)", cmdline)
        if port_match:
            try:
                port = int(port_match.group(1))
            except Exception:
                port = None
        processes.append({"pid": pid, "cmdline": cmdline, "model_path": _safe_realpath(model_path), "port": port})
    return processes


def get_loaded_catalog_model_ids(catalog: list[ManagedModel]) -> set[str]:
    processes = get_llama_server_processes()
    process_by_model = {proc["model_path"]: proc for proc in processes}
    return {
        model.model_id
        for model in catalog
        if process_by_model.get(_safe_realpath(model.local_path)) is not None
    }


def get_catalog_model_process(model_id: str, catalog: list[ManagedModel]) -> dict | None:
    model = next((item for item in catalog if item.model_id == model_id), None)
    if model is None:
        return None
    process_by_model = {proc["model_path"]: proc for proc in get_llama_server_processes()}
    return process_by_model.get(_safe_realpath(model.local_path))


def get_gpu_uuid_index_map() -> dict[str, int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return {}
    mapping: dict[str, int] = {}
    for line in result.stdout.splitlines():
        chunks = [chunk.strip() for chunk in line.split(",")]
        if len(chunks) >= 2 and chunks[0].isdigit() and chunks[1]:
            mapping[chunks[1]] = int(chunks[0])
    return mapping


def get_gpu_process_memory_by_pid() -> dict[int, dict[int, float]]:
    uuid_to_index = get_gpu_uuid_index_map()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return {}
    by_pid: dict[int, dict[int, float]] = {}
    for line in result.stdout.splitlines():
        chunks = [chunk.strip() for chunk in line.split(",")]
        if len(chunks) < 3 or not chunks[0].isdigit():
            continue
        pid = int(chunks[0])
        gpu_uuid = chunks[1]
        gpu_index = uuid_to_index.get(gpu_uuid)
        if gpu_index is None:
            continue
        try:
            used_mib = float(str(chunks[2]).replace("MiB", "").strip())
        except Exception:
            continue
        by_pid.setdefault(pid, {})[gpu_index] = used_mib
    return by_pid


def get_gpu_process_map() -> dict[int, str]:
    # Backward-compatible summary: pid -> total MiB across GPUs.
    by_pid = get_gpu_process_memory_by_pid()
    return {pid: str(int(sum(gpus.values()))) for pid, gpus in by_pid.items()}


def _model_runtime_snapshot(
    model_id: str,
    catalog: list[ManagedModel],
    host=DEFAULT_PUBLIC_HOST,
    port=DEFAULT_PUBLIC_PORT,
    *,
    include_upstream_health: bool = False,
) -> dict:
    """Collect non-fatal diagnostics for a model that may have just unloaded/crashed."""
    snapshot: dict[str, object] = {"model": model_id}
    try:
        published = get_published_model_ids(host, port)
        snapshot["published"] = model_id in published
        snapshot["published_count"] = len(published)
    except Exception as exc:
        snapshot["published_error"] = str(exc)
    try:
        proc = get_catalog_model_process(model_id, catalog)
        snapshot["process"] = proc or None
        if proc and isinstance(proc.get("pid"), int):
            gpu_mem = get_gpu_process_map().get(proc["pid"])
            if gpu_mem is not None:
                snapshot["gpu_memory_mib"] = gpu_mem
    except Exception as exc:
        snapshot["process_error"] = str(exc)
    if include_upstream_health:
        try:
            health_url = f"http://{_normalize_client_host(host)}:{port}/upstream/{quote(model_id, safe='')}/health"
            started_at = time.monotonic()
            response = requests.get(health_url, timeout=(1.5, 3))
            snapshot["upstream_health_status"] = response.status_code
            snapshot["upstream_health_ms"] = _elapsed_ms(started_at)
            if response.status_code >= 400:
                snapshot["upstream_health_body"] = response.text[:1000]
        except Exception as exc:
            snapshot["upstream_health_error"] = str(exc)
    return snapshot


def log_model_runtime_snapshot(
    kind: str,
    model_id: str,
    catalog: list[ManagedModel],
    host=DEFAULT_PUBLIC_HOST,
    port=DEFAULT_PUBLIC_PORT,
    *,
    include_upstream_health: bool = False,
    **extra,
) -> None:
    payload = _model_runtime_snapshot(model_id, catalog, host, port, include_upstream_health=include_upstream_health)
    payload.update(extra)
    log_api_event(kind, payload)


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


def _parse_llama_device_indices(value: object) -> list[int]:
    indices: list[int] = []
    for match in re.finditer(r"(?:CUDA)?\s*([0-9]+)", str(value or ""), flags=re.IGNORECASE):
        try:
            gpu = int(match.group(1))
        except Exception:
            continue
        if gpu not in indices:
            indices.append(gpu)
    return indices


def model_launch_gpu_set(model: ManagedModel, total_gpus: int | None = None) -> list[int]:
    """Return the physical GPU indices this catalog model is expected to touch."""
    try:
        if int(model.n_gpu_layers) == 0:
            return []
    except Exception:
        pass
    overrides = model.server_overrides or {}
    explicit_devices = _parse_llama_device_indices(overrides.get("device"))
    if explicit_devices:
        return explicit_devices
    tensor_split = str(overrides.get("__replica_tensor_split") or model.tensor_split or "")
    parts = [part for part in tensor_split.split(",") if part.strip()]
    count = len(parts) or 1
    total = total_gpus if total_gpus is not None else detect_cuda_device_count()
    if total > 0:
        count = min(count, total)
    return list(range(max(0, count)))


def model_has_enough_free_vram_to_load(model: ManagedModel, *, safety_vram_mib: int = 2048) -> tuple[bool, dict[str, object]]:
    """Conservative preflight: allow coexistence only when the target estimate fits current free VRAM."""
    gpu_set = model_launch_gpu_set(model)
    if not gpu_set:
        return True, {"reason": "cpu_model", "gpu_set": []}
    required = estimate_model_runtime_mib(model)
    if required is None:
        return False, {"reason": "missing_estimate", "gpu_set": gpu_set}
    snap = _query_gpu_memory_snapshot_cached()
    if not snap:
        return False, {"reason": "missing_gpu_snapshot", "gpu_set": gpu_set, "required_total_mib": required}
    required_per_gpu = (required / max(1, len(gpu_set))) + 1024.0
    checks: list[dict[str, float | int | str]] = []
    for gpu in gpu_set:
        free = float(snap.get(gpu, {}).get("free_mib", 0.0))
        need = required_per_gpu + safety_vram_mib
        checks.append({"gpu": gpu, "free_mib": free, "required_mib": need})
        if need > free:
            return False, {
                "reason": "insufficient_vram",
                "gpu_set": gpu_set,
                "required_total_mib": required,
                "required_per_gpu_mib": required_per_gpu,
                "safety_vram_mib": safety_vram_mib,
                "checks": checks,
            }
    return True, {
        "reason": "fits",
        "gpu_set": gpu_set,
        "required_total_mib": required,
        "required_per_gpu_mib": required_per_gpu,
        "safety_vram_mib": safety_vram_mib,
        "checks": checks,
    }


def get_gpu_conflict_message(model_id: str, catalog: list[ManagedModel], host=DEFAULT_PUBLIC_HOST, port=DEFAULT_PUBLIC_PORT) -> str | None:
    """Generate a user-friendly error message for GPU conflicts.
    
    Uses only the local Heimdall Gateway installation for model management.
    If a GPU conflict is detected, suggests using 'heimdall-gateway unload' to free resources.
    """
    processes = get_llama_server_processes()
    process_by_pid = {proc["pid"]: proc for proc in processes}
    model_by_path = {_safe_realpath(model.local_path): model.model_id for model in catalog}
    target_process = get_catalog_model_process(model_id, catalog)
    if target_process is not None:
        return None
    target_model = next((model for model in catalog if model.model_id == model_id), None)
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
    if target_model is not None:
        fits, fit_info = model_has_enough_free_vram_to_load(target_model)
        if fits:
            log_api_event(
                "model_load_allowed_vram_available",
                {"model": model_id, "conflicts": conflicts[:4], **fit_info},
            )
            return None
        log_api_event(
            "model_load_vram_preflight_reject",
            {"model": model_id, "conflicts": conflicts[:4], **fit_info},
        )
    joined = "; ".join(conflicts[:4])
    if len(conflicts) > 4:
        joined += f"; +{len(conflicts) - 4} more"
    return (
        f"Cannot load model '{model_id}' because the GPU is already in use: {joined}. "
        "Use 'heimdall-gateway unload <model>' to free resources, or wait for those workloads to finish."
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




def build_openai_model_list_payload(model: ManagedModel) -> dict:
    # Keep /v1/models fast and side-effect free. Some clients call this during
    # startup and expect a quick OpenAI-compatible list; detailed metadata is
    # available through /api/show and /api/ctx.  Include lightweight capability
    # and context hints so OpenAI-compatible clients can avoid sending images
    # to text-only GGUF models and do not have to guess/cache the context
    # window for custom endpoints.
    load_capabilities = _normalize_load_capabilities(getattr(model, "load_capabilities", []))
    cfg_ctx = displayed_configured_ctx(model)
    api_ctx = displayed_api_ctx(model) if cfg_ctx > 0 else 0
    ctx_status = ctx_evaluation_status(model)
    context_length = cfg_ctx if cfg_ctx > 0 and ctx_status != "ERROR" else None
    api_context_length = api_ctx if api_ctx > 0 and ctx_status != "ERROR" else None
    # OpenAI's public Model object does not standardize context metadata, but
    # OpenAI-compatible agent clients commonly need it. Keep these aliases
    # deliberately flat and non-ambiguous:
    #   - Hermes reads context_length/max_model_len/max_input_tokens.
    #   - vLLM-style clients understand max_model_len.
    #   - Some clients recursively parse nested keys named `input`/`output` as
    #     pricing, and `max_tokens` is commonly treated as an output limit, so
    #     avoid OpenCode-style `limit.output` and top-level `max_tokens` here.
    return {
        "id": model.model_id,
        "object": "model",
        "created": 0,
        "owned_by": "llama-swap",
        "context_length": context_length,
        "context_window": context_length,
        "context_size": context_length,
        "max_context_length": context_length,
        "max_model_len": context_length,
        "max_input_tokens": context_length,
        "max_output_tokens": api_context_length,
        "max_completion_tokens": api_context_length,
        "metadata": {
            "vision": _has_vision_runtime(model),
            "load_capabilities": load_capabilities,
            "context_length": context_length,
            "context_window": context_length,
            "context_size": context_length,
            "max_context_length": context_length,
            "max_model_len": context_length,
            "max_input_tokens": context_length,
            "configured_context_length": context_length,
            "api_context_length": api_context_length,
            "max_output_tokens": api_context_length,
            "max_completion_tokens": api_context_length,
            "api_context_status": ctx_status,
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


def _model_capabilities_imply_vision(model: ManagedModel) -> bool:
    capabilities = _normalize_load_capabilities(getattr(model, "load_capabilities", []))
    return any(token in capabilities for token in ("image", "image-text-to-text", "image-to-text", "vision"))


def _looks_like_vision_model(model: ManagedModel) -> bool:
    if model.mmproj_path:
        return True
    if _model_capabilities_imply_vision(model):
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


def _has_configured_mmproj_runtime(model: ManagedModel) -> bool:
    # llama.cpp accepts actual image blocks only when the launched server has a
    # valid multimodal projector. Hub-level capabilities such as image-to-text
    # are metadata, not proof that this local GGUF runtime can consume images.
    return bool(model.mmproj_path and Path(model.mmproj_path).exists())


def _cmdline_mmproj_path(cmdline: str) -> str | None:
    try:
        argv = shlex.split(str(cmdline or ""))
    except Exception:
        argv = str(cmdline or "").split()
    for idx, arg in enumerate(argv):
        if arg == "--mmproj" and idx + 1 < len(argv):
            return argv[idx + 1]
        if str(arg).startswith("--mmproj="):
            return str(arg).split("=", 1)[1]
    return None


def _loaded_process_missing_configured_mmproj(model: ManagedModel, catalog: list[ManagedModel]) -> bool:
    """Return True when a currently loaded llama-server is stale for image input.

    The catalog/YAML can be updated to include ``--mmproj`` while llama-swap
    keeps an older already-loaded process alive.  In that state the model is
    advertised as vision-capable, but image requests still fail upstream with
    llama.cpp's opaque ``image input is not supported`` 500.
    """
    if not _has_configured_mmproj_runtime(model):
        return False
    try:
        proc = get_catalog_model_process(model.model_id, catalog)
    except Exception:
        return False
    if not proc:
        return False
    configured_mmproj = _safe_realpath(str(model.mmproj_path or ""))
    live_mmproj = _cmdline_mmproj_path(str(proc.get("cmdline") or ""))
    if not live_mmproj:
        return True
    return _safe_realpath(live_mmproj) != configured_mmproj


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


_PRECISION_MODEL_SUFFIX_RE = re.compile(
    r"(?i)(?:[-_.:](?:bf16|bfloat16|f16|fp16|float16|f32|fp32|float32))$"
)


def _strip_precision_model_suffix(value: str) -> str:
    return _PRECISION_MODEL_SUFFIX_RE.sub("", (value or "").strip())


def _precision_suffix_alias_candidates(name: str) -> list[str]:
    candidates: list[str] = []

    def _append(value: str) -> None:
        candidate = (value or "").strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    stripped = _strip_precision_model_suffix(name)
    if stripped != name:
        _append(stripped)
    if "/" in name:
        provider_prefix, remainder = name.split("/", 1)
        if provider_prefix and remainder:
            stripped_remainder = _strip_precision_model_suffix(remainder)
            if stripped_remainder != remainder:
                _append(stripped_remainder)
    return candidates


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



def _responses_tools_with_deferred_search(tools: object) -> list[dict]:
    """Prepare Responses tools for tool_search-capable backends.

    OpenAI Tool Search expects `{"type":"tool_search"}` in the tools list and
    `defer_loading: true` on functions or MCP server definitions that should be
    loaded lazily.  For namespaces, the namespace remains visible while each
    child function is marked deferred.
    """
    if not isinstance(tools, list):
        return []

    result: list[dict] = []
    has_tool_search = False
    for tool in tools:
        if not isinstance(tool, dict):
            result.append(tool)
            continue
        tool_type = str(tool.get("type") or "").strip()
        if tool_type == "tool_search":
            has_tool_search = True
            result.append(dict(tool))
            continue
        updated = dict(tool)
        if tool_type == "namespace":
            sub_tools = tool.get("tools")
            if isinstance(sub_tools, list):
                updated["tools"] = [
                    {**sub_tool, "defer_loading": True} if isinstance(sub_tool, dict) else sub_tool
                    for sub_tool in sub_tools
                ]
        elif tool_type in {"function", "mcp"}:
            updated["defer_loading"] = True
        result.append(updated)

    if result and not has_tool_search:
        result.append({"type": "tool_search"})
    return result


def _flatten_responses_tools(tools: object, flatten_enabled: bool = True) -> list[dict]:
    """Convert Responses-native tool types to standard function-type tools.

    Handles:
      - namespace: flatten sub-tools with "__" prefix (mcp__x__y)
      - custom:   rename to "function", wrap name/desc/params in function dict
      - web_search, computer_use_preview, etc.:  convert to generic function tool
    """
    if not flatten_enabled or not isinstance(tools, list):
        return tools if isinstance(tools, list) else []
    
    initial_count = len(tools)
    result: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            result.append(tool)
            continue
        tool_type = str(tool.get("type") or "").strip()
        if tool_type == "function":
            result.append(tool)
            continue
        if tool_type == "namespace":
            ns_name = str(tool.get("name") or "").strip()
            sub_tools = tool.get("tools")
            if ns_name and isinstance(sub_tools, list):
                for sub_tool in sub_tools:
                    if not isinstance(sub_tool, dict):
                        continue
                    sub_name = str(sub_tool.get("name") or "").strip()
                    if not sub_name:
                        continue
                    result.append({
                        "type": "function",
                        "function": {
                            "name": f"{ns_name}__{sub_name}",
                            "description": sub_tool.get("description", ""),
                            "parameters": sub_tool.get("parameters", {}),
                        },
                    })
                continue
            result.append(tool)
            continue
        if tool_type == "custom":
            result.append({
                "type": "function",
                "function": {
                    "name": str(tool.get("name") or "").strip() or "custom_tool",
                    "description": str(tool.get("description") or "").strip(),
                    "parameters": tool.get("parameters", {}),
                },
            })
            continue
        tool_name = str(tool.get("name") or tool_type or "unknown_tool").strip()
        tool_desc = str(tool.get("description") or f"Built-in tool: {tool_type}").strip()
        result.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_desc,
                "parameters": tool.get("parameters", {}),
            },
        })

    log_api_event("tool_flattening_done", {
        "initial_count": initial_count,
        "final_count": len(result),
        "flattened_names": [t.get("function", {}).get("name") for t in result if t.get("type") == "function"]
    })
    return result


def _responses_namespace_tool_map(tools: object) -> dict[str, dict[str, str]]:
    """Map flattened legacy function names back to Responses namespace calls."""
    mapping: dict[str, dict[str, str]] = {}
    if not isinstance(tools, list):
        return mapping
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if str(tool.get("type") or "").strip() != "namespace":
            continue
        namespace = str(tool.get("name") or "").strip()
        sub_tools = tool.get("tools")
        if not namespace or not isinstance(sub_tools, list):
            continue
        for sub_tool in sub_tools:
            if not isinstance(sub_tool, dict):
                continue
            name = str(sub_tool.get("name") or "").strip()
            if not name:
                continue
            mapping[f"{namespace}__{name}"] = {"namespace": namespace, "name": name}
    return mapping


def _responses_contains_failed_tool_output(text: object) -> bool:
    value = str(text or "")
    return "unsupported call:" in value or "unknown MCP server" in value


_IMAGE_DATA_URL_RE = re.compile(
    r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\r\n]+",
    flags=re.IGNORECASE,
)


def _extract_legacy_tool_output_parts(value: object) -> tuple[str, list[dict], int]:
    """Split Responses tool output into text plus structured image parts.

    Legacy Chat Completions can carry images as structured `image_url` content
    on a user message, but not as raw Python/JSON stringified tool output.  When
    a Responses tool returns screenshots or other multimodal data, extract those
    images so the caller can attach them as real image blocks for vision-capable
    backends.  The returned text never contains data URL/base64 payloads.
    """

    images: list[dict] = []
    image_count = 0

    def add_image_url(url: object) -> None:
        nonlocal image_count
        url_text = str(url or "").strip()
        if not url_text:
            return
        images.append({"type": "image_url", "image_url": {"url": url_text}})
        image_count += 1

    def sanitize(item: object) -> str:
        nonlocal image_count
        if item is None:
            return ""
        if isinstance(item, str):
            def repl(match: re.Match[str]) -> str:
                add_image_url(match.group(0))
                return f"[image attached from tool output #{image_count}]"

            return _IMAGE_DATA_URL_RE.sub(repl, item)
        if isinstance(item, bytes):
            return sanitize(item.decode("utf-8", errors="replace"))
        if isinstance(item, list):
            parts = [sanitize(part).strip() for part in item]
            return "\n".join(part for part in parts if part)
        if isinstance(item, dict):
            item_type = str(item.get("type") or "").strip()
            if item_type in {"input_image", "image_url"} or item.get("image_url"):
                url = item.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url")
                add_image_url(url)
                return f"[image attached from tool output #{image_count}]"
            if item.get("data") and str(item.get("mime_type") or item.get("media_type") or "").startswith("image/"):
                mime_type = str(item.get("mime_type") or item.get("media_type") or "image/png").strip()
                add_image_url(f"data:{mime_type};base64,{str(item.get('data') or '').strip()}")
                return f"[image attached from tool output #{image_count}]"
            if item_type in {"input_text", "output_text", "text"}:
                return sanitize(item.get("text") or "")
            # For other structured tool outputs, preserve JSON-ish content but
            # still extract any nested image data URLs.
            try:
                return sanitize(json.dumps(item, ensure_ascii=False))
            except Exception:
                return sanitize(str(item))
        return sanitize(str(item))

    return sanitize(value), images, image_count


def _responses_chat_tool_entry(name: str, description: object = "", parameters: object = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or "",
            "parameters": parameters if parameters is not None else {},
        },
    }


RESPONSES_INTERNAL_TOOL_SEARCH_NAME = "tool_search"
RESPONSES_INTERNAL_CALL_DEFERRED_TOOL_NAME = "call_deferred_tool"


@dataclass
class ResponsesDeferredTool:
    legacy_name: str
    responses_name: str
    namespace: str
    description: str
    parameters: object
    source_type: str = "function"

    def search_blob(self) -> str:
        return " ".join(
            str(part or "").lower()
            for part in (
                self.legacy_name,
                self.responses_name,
                self.namespace,
                self.description,
            )
        )

    def schema_payload(self) -> dict:
        payload = {
            "name": self.responses_name,
            "legacy_name": self.legacy_name,
            "description": self.description,
            "parameters": self.parameters if self.parameters is not None else {},
        }
        if self.namespace:
            payload["namespace"] = self.namespace
        return payload




@dataclass
class ToolCallRepairResult:
    translated: dict | None = None
    repaired: bool = False
    feedback: dict | None = None
    fatal: str | None = None

    @property
    def ok(self) -> bool:
        return self.translated is not None

    def feedback_message(self, call_id: str) -> dict | None:
        if not self.feedback:
            return None
        diagnostic_json = json.dumps(self.feedback, ensure_ascii=False, separators=(",", ":"))
        content = (
            "The previous call_deferred_tool call was not executed.\n"
            "Retry by calling call_deferred_tool again with the corrected namespace, name, and arguments. "
            "Do not answer the user yet unless you can complete the task without this tool.\n"
            "Use exactly this wrapper shape: "
            '{"namespace":"<namespace>","name":"<tool_name>","arguments":{...}}\n'
            "Diagnostic JSON:\n"
            f"{diagnostic_json}"
        )
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        }


class ResponsesToolRegistry:
    """KV-stable registry for Responses deferred tools.

    The legacy chat fallback must not mutate the upstream `tools` field between
    internal rounds, otherwise llama.cpp loses most of the benefit of prompt/KV
    reuse.  This registry keeps full deferred schemas in proxy memory, exposes a
    compact directory in messages, and uses two stable internal functions:
    `tool_search` and `call_deferred_tool`.
    """

    def __init__(self, eager_chat_tools: list[dict] | None = None, deferred_tools: list[ResponsesDeferredTool] | None = None):
        self.eager_chat_tools = list(eager_chat_tools or [])
        self.deferred_tools = list(deferred_tools or [])
        self.deferred_by_legacy_name = {tool.legacy_name: tool for tool in self.deferred_tools}
        self.deferred_by_response_key = {
            self._response_key(tool.namespace, tool.responses_name): tool
            for tool in self.deferred_tools
        }

    @staticmethod
    def _response_key(namespace: object, name: object) -> str:
        return f"{str(namespace or '').strip()}::{str(name or '').strip()}"

    @staticmethod
    def _function_fields(tool: dict) -> tuple[str, str, object]:
        function = tool.get("function")
        if isinstance(function, dict):
            return (
                str(function.get("name") or "").strip(),
                str(function.get("description") or tool.get("description") or "").strip(),
                function.get("parameters", tool.get("parameters", {})),
            )
        return (
            str(tool.get("name") or "").strip(),
            str(tool.get("description") or "").strip(),
            tool.get("parameters", {}),
        )

    @classmethod
    def from_responses_tools(
        cls,
        tools: object,
        *,
        flatten_namespace_tools: bool = True,
        default_defer_namespaces: bool = True,
    ) -> "ResponsesToolRegistry":
        if not isinstance(tools, list):
            return cls()
        eager: list[dict] = []
        deferred: list[ResponsesDeferredTool] = []

        def add_tool(*, namespace: str, name: str, description: object, parameters: object, defer: bool, source_type: str = "function") -> None:
            if not name:
                return
            legacy_name = f"{namespace}__{name}" if namespace and flatten_namespace_tools else name
            if defer:
                deferred.append(
                    ResponsesDeferredTool(
                        legacy_name=legacy_name,
                        responses_name=name,
                        namespace=namespace,
                        description=str(description or ""),
                        parameters=parameters if parameters is not None else {},
                        source_type=source_type,
                    )
                )
                return
            eager.append(_responses_chat_tool_entry(legacy_name, description, parameters if parameters is not None else {}))

        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_type = str(tool.get("type") or "").strip()
            if tool_type == "namespace" and flatten_namespace_tools:
                namespace = str(tool.get("name") or "").strip()
                if "defer_loading" in tool:
                    namespace_defer = bool(tool.get("defer_loading"))
                else:
                    namespace_defer = bool(default_defer_namespaces)
                sub_tools = tool.get("tools")
                if not namespace or not isinstance(sub_tools, list):
                    continue
                for sub_tool in sub_tools:
                    if not isinstance(sub_tool, dict):
                        continue
                    name, desc, params = cls._function_fields(sub_tool)
                    defer = bool(sub_tool.get("defer_loading")) if "defer_loading" in sub_tool else namespace_defer
                    add_tool(namespace=namespace, name=name, description=desc, parameters=params, defer=defer, source_type="namespace")
                continue
            if tool_type != "function":
                if tool_type == "mcp" and bool(tool.get("defer_loading")):
                    server_label = str(tool.get("server_label") or tool.get("name") or "").strip()
                    if server_label:
                        deferred.append(
                            ResponsesDeferredTool(
                                legacy_name=server_label,
                                responses_name=server_label,
                                namespace="",
                                description=str(tool.get("description") or f"MCP server {server_label}"),
                                parameters=tool.get("parameters", {}),
                                source_type="mcp",
                            )
                        )
                continue
            name, desc, params = cls._function_fields(tool)
            add_tool(namespace="", name=name, description=desc, parameters=params, defer=bool(tool.get("defer_loading")), source_type="function")
        return cls(eager, deferred)

    @property
    def has_deferred_tools(self) -> bool:
        return bool(self.deferred_tools)

    def directory_text(self, *, max_tools_per_namespace: int = 40) -> str:
        if not self.deferred_tools:
            return ""
        grouped: dict[str, list[ResponsesDeferredTool]] = {}
        for tool in self.deferred_tools:
            grouped.setdefault(tool.namespace or "global", []).append(tool)
        lines = [
            "Deferred tool directory:",
            "Use the internal tool_search function to load full schemas before using deferred tools.",
            "After a schema is loaded, invoke it with call_deferred_tool(namespace, name, arguments).",
        ]
        for namespace in sorted(grouped):
            tools = sorted(grouped[namespace], key=lambda item: item.responses_name)
            visible = tools[:max_tools_per_namespace]
            suffix = f" (+{len(tools) - len(visible)} more)" if len(tools) > len(visible) else ""
            names = ", ".join(tool.responses_name for tool in visible)
            lines.append(f"- {namespace}: {names}{suffix}")
        return "\n".join(lines)

    def _internal_tool_search_entry(self) -> dict:
        return _responses_chat_tool_entry(
            RESPONSES_INTERNAL_TOOL_SEARCH_NAME,
            "Search deferred tools by namespace, exact tool name, or task query. Returns full schemas as tool output.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Task or capability to search for."},
                    "namespaces": {"type": "array", "items": {"type": "string"}},
                    "tools": {"type": "array", "items": {"type": "string"}, "description": "Exact tool names or namespace.tool names."},
                },
                "required": [],
            },
        )

    def _internal_call_deferred_tool_entry(self) -> dict:
        return _responses_chat_tool_entry(
            RESPONSES_INTERNAL_CALL_DEFERRED_TOOL_NAME,
            "Invoke a deferred tool after loading its schema with tool_search. The proxy translates this to the real Responses tool call.",
            {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Namespace shown by tool_search, empty for global tools."},
                    "name": {"type": "string", "description": "Deferred tool name."},
                    "arguments": {"type": "object", "description": "Arguments for the deferred tool."},
                },
                "required": ["name", "arguments"],
            },
        )

    def chat_tools_with_internal_search(self) -> list[dict]:
        tools = [dict(tool) for tool in self.eager_chat_tools]
        if self.deferred_tools:
            tools.append(self._internal_tool_search_entry())
            tools.append(self._internal_call_deferred_tool_entry())
        return tools

    def search(self, request: object, *, max_results: int = 24) -> dict:
        if isinstance(request, str):
            request = {"query": request}
        request = request if isinstance(request, dict) else {}
        namespaces = {
            str(item or "").strip()
            for item in (request.get("namespaces") if isinstance(request.get("namespaces"), list) else [])
            if str(item or "").strip()
        }
        exact_tools = {
            str(item or "").strip()
            for item in (request.get("tools") if isinstance(request.get("tools"), list) else [])
            if str(item or "").strip()
        }
        query = str(request.get("query") or "").strip().lower()
        query_terms = [term for term in re.split(r"\W+", query) if term]

        matches: list[ResponsesDeferredTool] = []
        for tool in self.deferred_tools:
            if namespaces and tool.namespace not in namespaces:
                continue
            exact_names = {tool.responses_name, tool.legacy_name}
            if tool.namespace:
                exact_names.add(f"{tool.namespace}.{tool.responses_name}")
                exact_names.add(f"{tool.namespace}__{tool.responses_name}")
            is_match = False
            if exact_tools and exact_names.intersection(exact_tools):
                is_match = True
            if query_terms:
                blob = tool.search_blob()
                if all(term in blob for term in query_terms) or any(term in blob for term in query_terms):
                    is_match = True
            if namespaces and not exact_tools and not query_terms:
                is_match = True
            if is_match:
                matches.append(tool)

        deduped: list[ResponsesDeferredTool] = []
        seen: set[str] = set()
        for tool in matches:
            if tool.legacy_name in seen:
                continue
            seen.add(tool.legacy_name)
            deduped.append(tool)

        if not deduped:
            return {
                "status": "not_found",
                "message": "No deferred tools matched. Use a narrower namespace or exact tool name from the directory.",
                "tools": [],
            }
        if len(deduped) > max_results:
            return {
                "status": "too_many_matches",
                "message": f"Too many deferred tools matched ({len(deduped)}). Please narrow by namespace or exact tool names.",
                "matches": [tool.schema_payload() for tool in deduped[:max_results]],
            }
        return {"status": "ok", "tools": [tool.schema_payload() for tool in deduped]}

    def tool_search_output_message(self, call_id: str, arguments: object) -> dict:
        result = self.search(arguments)
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
        }

    @staticmethod
    def _normalize_chrome_evaluate_script_arguments(raw_args: object) -> object:
        if not isinstance(raw_args, dict):
            return raw_args
        function_text = raw_args.get("function")
        if not isinstance(function_text, str):
            return raw_args
        stripped = function_text.strip()
        if not stripped:
            return raw_args
        starts_like_function = (
            stripped.startswith("function")
            or stripped.startswith("async function")
            or stripped.startswith("()=>")
            or stripped.startswith("() =>")
            or stripped.startswith("async () =>")
            or stripped.startswith("async()=>")
            or stripped.startswith("async() =>")
        )
        if starts_like_function:
            return raw_args
        normalized = dict(raw_args)
        if (
            stripped.startswith("return ")
            or stripped.startswith("await ")
            or stripped.startswith("var ")
            or stripped.startswith("let ")
            or stripped.startswith("const ")
            or ";" in stripped
        ):
            normalized["function"] = f"() => {{ {stripped} }}"
            return normalized
        normalized["function"] = f"() => ({stripped})"
        return normalized

    def _deferred_tools_named(self, name: str) -> list[ResponsesDeferredTool]:
        matches: list[ResponsesDeferredTool] = []
        for tool in self.deferred_tools:
            exact_names = {tool.responses_name, tool.legacy_name}
            if tool.namespace:
                exact_names.add(f"{tool.namespace}.{tool.responses_name}")
                exact_names.add(f"{tool.namespace}__{tool.responses_name}")
            if name in exact_names and tool not in matches:
                matches.append(tool)
        return matches

    @staticmethod
    def _required_parameters(tool: ResponsesDeferredTool) -> list[str]:
        parameters = tool.parameters if isinstance(tool.parameters, dict) else {}
        required = parameters.get("required")
        if not isinstance(required, list):
            return []
        return [str(item) for item in required if str(item or "").strip()]

    @staticmethod
    def _argument_example_for_schema(parameters: object, required: list[str]) -> dict:
        params = parameters if isinstance(parameters, dict) else {}
        properties = params.get("properties") if isinstance(params.get("properties"), dict) else {}
        example: dict = {}
        for name in required:
            schema = properties.get(name) if isinstance(properties.get(name), dict) else {}
            schema_type = str(schema.get("type") or "").strip()
            if schema_type == "array":
                example[name] = ["..."]
            elif schema_type in {"integer", "number"}:
                example[name] = 0
            elif schema_type == "boolean":
                example[name] = True
            elif schema_type == "object":
                example[name] = {}
            else:
                example[name] = "..."
        return example

    @staticmethod
    def _candidate_payloads(tools: list[ResponsesDeferredTool]) -> list[dict]:
        return [
            {"namespace": tool.namespace, "name": tool.responses_name, "legacy_name": tool.legacy_name}
            for tool in tools[:12]
        ]

    def _feedback_payload(
        self,
        *,
        error: str,
        message: str,
        received: dict,
        tool: ResponsesDeferredTool | None = None,
        candidates: list[ResponsesDeferredTool] | None = None,
        missing: list[str] | None = None,
    ) -> dict:
        payload: dict = {
            "error": error,
            "message": message,
            "received": received,
        }
        if tool is not None:
            required = self._required_parameters(tool)
            example_args = self._argument_example_for_schema(tool.parameters, required)
            payload["expected"] = {
                "namespace": tool.namespace,
                "name": tool.responses_name,
                "arguments": example_args,
            }
            payload["example"] = {
                "namespace": tool.namespace,
                "name": tool.responses_name,
                "arguments": example_args,
            }
        if candidates:
            payload["candidates"] = self._candidate_payloads(candidates)
        if missing is not None:
            payload["missing"] = missing
        return payload

    def repair_deferred_tool_call(self, arguments: object, *, loaded_tools: list[dict] | None = None) -> ToolCallRepairResult:
        original_arguments = arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                return ToolCallRepairResult(feedback={
                    "error": "invalid_call_deferred_tool_arguments",
                    "message": "call_deferred_tool arguments must be valid JSON. Retry with {namespace, name, arguments}.",
                    "received": {"arguments": str(original_arguments or "")[:500]},
                    "example": {"namespace": "mcp__chrome_devtools", "name": "wait_for", "arguments": {"text": ["..."], "timeout": 5000}},
                })
        if not isinstance(arguments, dict):
            return ToolCallRepairResult(feedback={
                "error": "invalid_call_deferred_tool_arguments",
                "message": "call_deferred_tool arguments must be an object with name and arguments.",
                "received": {"arguments": str(original_arguments or "")[:500]},
                "example": {"namespace": "mcp__chrome_devtools", "name": "wait_for", "arguments": {"text": ["..."]}},
            })
        namespace = str(arguments.get("namespace") or "").strip()
        name = str(arguments.get("name") or "").strip()
        received = {"namespace": namespace, "name": name, "arguments": arguments.get("arguments", {})}
        if not name:
            return ToolCallRepairResult(feedback={
                "error": "missing_deferred_tool_name",
                "message": "call_deferred_tool requires a deferred tool name. Use tool_search first if you need to find the name.",
                "received": received,
                "example": {"namespace": "mcp__chrome_devtools", "name": "wait_for", "arguments": {"text": ["..."]}},
            })
        raw_args = arguments.get("arguments", {})
        if isinstance(raw_args, str):
            decoded_raw_args = _decode_tool_call_arguments(raw_args)
            repaired = decoded_raw_args != raw_args
        else:
            decoded_raw_args = raw_args if isinstance(raw_args, dict) else {}
            repaired = False
        if not namespace and isinstance(decoded_raw_args, dict):
            server_namespace = str(decoded_raw_args.get("server") or "").strip()
            if server_namespace:
                namespace = server_namespace
                repaired = True
        tool = (
            self.deferred_by_response_key.get(self._response_key(namespace, name))
            or self.deferred_by_legacy_name.get(name)
            or self.deferred_by_legacy_name.get(f"{namespace}__{name}" if namespace else name)
        )
        if tool is None and not namespace and loaded_tools:
            scoped_matches: list[ResponsesDeferredTool] = []
            for loaded in loaded_tools:
                if not isinstance(loaded, dict):
                    continue
                loaded_name = str(loaded.get("name") or loaded.get("responses_name") or "").strip()
                loaded_namespace = str(loaded.get("namespace") or "").strip()
                if loaded_name != name or not loaded_namespace:
                    continue
                scoped_tool = self.deferred_by_response_key.get(self._response_key(loaded_namespace, loaded_name))
                if scoped_tool is not None and scoped_tool not in scoped_matches:
                    scoped_matches.append(scoped_tool)
            if len(scoped_matches) == 1:
                tool = scoped_matches[0]
        if tool is None:
            candidates = self._deferred_tools_named(name)
            if not namespace and len(candidates) > 1:
                return ToolCallRepairResult(feedback=self._feedback_payload(
                    error="ambiguous_deferred_tool",
                    message="Multiple deferred tools match this name. Retry with an explicit namespace from candidates, or use tool_search with a namespace.",
                    received=received,
                    candidates=candidates,
                ))
            if candidates:
                return ToolCallRepairResult(feedback=self._feedback_payload(
                    error="deferred_tool_schema_not_loaded",
                    message="This deferred tool was not resolved in the current scope. Retry with namespace/name from candidates, or call tool_search for the schema first.",
                    received=received,
                    candidates=candidates,
                ))
            return ToolCallRepairResult(feedback=self._feedback_payload(
                error="deferred_tool_not_found",
                message="No deferred tool matched this name. Use tool_search to find the correct namespace and tool name, then retry call_deferred_tool.",
                received=received,
            ))
        if isinstance(decoded_raw_args, dict):
            normalized_args = dict(decoded_raw_args)
            if "server" in normalized_args:
                normalized_args.pop("server", None)
                repaired = True
        else:
            normalized_args = raw_args
        if tool.responses_name == "evaluate_script" and str(tool.namespace or "").startswith("mcp__chrome_devtools"):
            before = normalized_args
            normalized_args = self._normalize_chrome_evaluate_script_arguments(normalized_args)
            if normalized_args != before:
                repaired = True
        required = self._required_parameters(tool)
        if required and isinstance(normalized_args, dict):
            missing = [item for item in required if item not in normalized_args or normalized_args.get(item) in (None, "")]
            if missing:
                return ToolCallRepairResult(feedback=self._feedback_payload(
                    error="invalid_deferred_tool_arguments",
                    message=f"{tool.responses_name} requires {', '.join(missing)}. Retry with namespace and all required fields.",
                    received={"namespace": namespace, "name": name, "arguments": normalized_args},
                    tool=tool,
                    missing=missing,
                ))
        if isinstance(normalized_args, str):
            arg_text = normalized_args
        else:
            arg_text = json.dumps(normalized_args if normalized_args is not None else {}, ensure_ascii=False, separators=(",", ":"))
        return ToolCallRepairResult(
            translated={
                "legacy_name": tool.legacy_name,
                "responses_name": tool.responses_name,
                "namespace": tool.namespace,
                "arguments": arg_text,
            },
            repaired=repaired,
        )

    def translate_deferred_tool_call(self, arguments: object, *, loaded_tools: list[dict] | None = None) -> dict | None:
        return self.repair_deferred_tool_call(arguments, loaded_tools=loaded_tools).translated

    def deferred_tool_for_history(self, *, namespace: object = "", name: object = "") -> ResponsesDeferredTool | None:
        namespace_text = str(namespace or "").strip().rstrip("_")
        name_text = str(name or "").strip()
        if not name_text:
            return None
        return (
            self.deferred_by_response_key.get(self._response_key(namespace_text, name_text))
            or self.deferred_by_legacy_name.get(f"{namespace_text}__{name_text}" if namespace_text else name_text)
            or self.deferred_by_legacy_name.get(name_text)
        )


def _responses_tool_search_emulation_enabled(server_cfg: dict | None = None) -> bool:
    server_cfg = server_cfg or {}
    cfg = server_cfg.get("responses_tool_search_emulation")
    if isinstance(cfg, dict) and "enabled" in cfg:
        return bool(cfg.get("enabled"))
    raw = _env_value("HEIMDALL_GATEWAY_RESPONSES_TOOL_SEARCH_EMULATION", "LLAMACPP_RESPONSES_TOOL_SEARCH_EMULATION", "")
    if raw:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return True



def _loaded_deferred_tools_from_tool_search_messages(messages: object) -> list[dict]:
    loaded: list[dict] = []
    if not isinstance(messages, list):
        return loaded
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except Exception:
            continue
        tools = payload.get("tools") if isinstance(payload, dict) else None
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if isinstance(tool, dict):
                loaded.append(tool)
    return loaded

def _decode_tool_call_arguments(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value or "{}")
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


def _sanitize_responses_tool_arguments(name: object, namespace: object, arguments: object) -> tuple[str, bool]:
    """Normalize tool-call arguments before exposing them as Responses events.

    The Responses item already carries namespace separately. Passing namespace/server
    inside the function arguments makes Codex try to execute tools with parameters
    that real MCP/browser tools do not accept. This helper keeps the public
    function_call shape stable while removing proxy-only routing metadata.
    """
    if isinstance(arguments, str):
        original_text = arguments
    elif arguments is None:
        original_text = "{}"
    else:
        try:
            original_text = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            original_text = str(arguments)
    decoded = _decode_tool_call_arguments(arguments)
    repaired = False
    if not isinstance(decoded, dict):
        return original_text, False
    normalized = dict(decoded)
    for routing_key in ("namespace", "server"):
        if routing_key in normalized:
            normalized.pop(routing_key, None)
            repaired = True
    if (
        str(name or "").strip() == "evaluate_script"
        and str(namespace or "").strip().startswith("mcp__chrome_devtools")
    ):
        before = normalized
        normalized = ResponsesToolRegistry._normalize_chrome_evaluate_script_arguments(normalized)
        if normalized != before:
            repaired = True
    if repaired:
        return json.dumps(normalized, ensure_ascii=False, separators=(",", ":")), True
    return original_text, False


def _translate_internal_deferred_tool_calls_in_chat_response(
    data: dict,
    registry: ResponsesToolRegistry,
    *,
    loaded_schema_messages: list[dict] | None = None,
) -> tuple[dict, bool]:
    """Translate internal call_deferred_tool chat calls into real tool calls."""
    loaded_tools = _loaded_deferred_tools_from_tool_search_messages(loaded_schema_messages or [])
    changed = False
    new_data = dict(data)
    new_choices = []
    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            new_choices.append(choice)
            continue
        new_choice = dict(choice)
        message = dict(choice.get("message") or {})
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            translated_calls = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    translated_calls.append(tc)
                    continue
                function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                name = str(function.get("name") or "").strip()
                if name != RESPONSES_INTERNAL_CALL_DEFERRED_TOOL_NAME:
                    translated_calls.append(tc)
                    continue
                repair = registry.repair_deferred_tool_call(_decode_tool_call_arguments(function.get("arguments")), loaded_tools=loaded_tools)
                translated = repair.translated
                if translated is None:
                    continue
                if repair.repaired:
                    log_api_event(
                        "openai_responses_tool_repair_applied",
                        {
                            "internal_name": RESPONSES_INTERNAL_CALL_DEFERRED_TOOL_NAME,
                            "call_id": str(tc.get("id") or tc.get("call_id") or ""),
                            "namespace": translated.get("namespace"),
                            "name": translated.get("responses_name"),
                        },
                    )
                new_tc = dict(tc)
                new_function = dict(function)
                new_function["name"] = translated["responses_name"]
                new_function["arguments"] = translated["arguments"]
                if translated.get("namespace"):
                    new_function["namespace"] = translated["namespace"]
                    new_tc["namespace"] = translated["namespace"]
                new_tc["function"] = new_function
                translated_calls.append(new_tc)
                changed = True
            message["tool_calls"] = translated_calls
        new_choice["message"] = message
        new_choices.append(new_choice)
    new_data["choices"] = new_choices
    return new_data, changed


def _chat_response_internal_tool_repair_followup_messages(
    data: dict,
    registry: ResponsesToolRegistry,
    *,
    loaded_schema_messages: list[dict] | None = None,
) -> list[dict]:
    """Return internal assistant/tool feedback messages for unrepairable wrapper calls.

    This keeps the legacy Responses fallback from ending with an empty response
    when the local model emits call_deferred_tool in a shape the proxy cannot
    translate.  The feedback is appended to the internal chat history so the
    model can retry with corrected arguments without changing the stable tools
    array.
    """
    loaded_tools = _loaded_deferred_tools_from_tool_search_messages(loaded_schema_messages or [])
    tool_calls_for_history: list[dict] = []
    output_messages: list[dict] = []
    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = str(function.get("name") or "").strip()
            if name != RESPONSES_INTERNAL_CALL_DEFERRED_TOOL_NAME:
                continue
            decoded = _decode_tool_call_arguments(function.get("arguments"))
            if isinstance(decoded, dict) and str(decoded.get("name") or "").strip() == RESPONSES_INTERNAL_TOOL_SEARCH_NAME:
                continue
            repair = registry.repair_deferred_tool_call(decoded, loaded_tools=loaded_tools)
            if repair.ok:
                continue
            call_id = str(tc.get("id") or tc.get("call_id") or f"call_{uuid.uuid4().hex}")
            normalized_tc = dict(tc)
            normalized_tc["id"] = call_id
            normalized_tc["type"] = "function"
            normalized_tc["function"] = {
                "name": RESPONSES_INTERNAL_CALL_DEFERRED_TOOL_NAME,
                "arguments": json.dumps(decoded if isinstance(decoded, dict) else {}, ensure_ascii=False, separators=(",", ":")),
            }
            feedback_message = repair.feedback_message(call_id)
            if feedback_message is None:
                continue
            tool_calls_for_history.append(normalized_tc)
            output_messages.append(feedback_message)
    if not tool_calls_for_history:
        return []
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls_for_history,
        },
        *output_messages,
    ]


def _chat_response_internal_tool_search_followup_messages(data: dict, registry: ResponsesToolRegistry) -> list[dict]:
    tool_calls_for_history: list[dict] = []
    output_messages: list[dict] = []
    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            function = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = str(function.get("name") or "").strip()
            raw_arguments = function.get("arguments")
            search_arguments = raw_arguments
            if name == RESPONSES_INTERNAL_CALL_DEFERRED_TOOL_NAME:
                # Local models sometimes wrap the internal schema discovery call as
                # call_deferred_tool({name:"tool_search", arguments:{...}}). Treat
                # that as the same private discovery round instead of translating it
                # to a real client tool or dropping it as an unresolved deferred call.
                decoded = _decode_tool_call_arguments(raw_arguments)
                nested_name = str(decoded.get("name") or "").strip() if isinstance(decoded, dict) else ""
                if nested_name != RESPONSES_INTERNAL_TOOL_SEARCH_NAME:
                    continue
                search_arguments = decoded.get("arguments", {})
                name = RESPONSES_INTERNAL_TOOL_SEARCH_NAME
            if name != RESPONSES_INTERNAL_TOOL_SEARCH_NAME:
                continue
            call_id = str(tc.get("id") or tc.get("call_id") or f"call_{uuid.uuid4().hex}")
            decoded_search_arguments = _decode_tool_call_arguments(search_arguments)
            normalized_tc = dict(tc)
            normalized_tc["id"] = call_id
            normalized_tc["type"] = "function"
            normalized_tc["function"] = {
                "name": RESPONSES_INTERNAL_TOOL_SEARCH_NAME,
                "arguments": json.dumps(decoded_search_arguments, ensure_ascii=False, separators=(",", ":")),
            }
            tool_calls_for_history.append(normalized_tc)
            output_messages.append(registry.tool_search_output_message(call_id, decoded_search_arguments))
    if not tool_calls_for_history:
        return []
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls_for_history,
        },
        *output_messages,
    ]


def _chat_response_internal_tool_search_messages(data: dict, registry: ResponsesToolRegistry) -> list[dict]:
    return [
        message
        for message in _chat_response_internal_tool_search_followup_messages(data, registry)
        if message.get("role") == "tool"
    ]


def _combine_responses_usage(first: dict | None, second: dict | None) -> dict | None:
    first = _normalize_responses_usage(first) if first is not None else None
    second = _normalize_responses_usage(second) if second is not None else None
    if not first:
        return second
    if not second:
        return first
    return {
        "input_tokens": int(first.get("input_tokens") or 0) + int(second.get("input_tokens") or 0),
        "input_tokens_details": {"cached_tokens": int((first.get("input_tokens_details") or {}).get("cached_tokens") or 0) + int((second.get("input_tokens_details") or {}).get("cached_tokens") or 0)},
        "output_tokens": int(first.get("output_tokens") or 0) + int(second.get("output_tokens") or 0),
        "output_tokens_details": {"reasoning_tokens": int((first.get("output_tokens_details") or {}).get("reasoning_tokens") or 0) + int((second.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)},
        "total_tokens": int(first.get("total_tokens") or 0) + int(second.get("total_tokens") or 0),
    }


def _responses_tools_to_chat_tools(tools: object, flatten_namespace_tools: bool = True) -> list[dict]:
    """Convert Responses tools to legacy Chat Completions function tools.

    Top-level function tools are passed through. Namespace tools are flattened
    into legacy function names and translated back to Responses namespace calls
    on output. Built-ins/custom/freeform tools are intentionally not converted
    here because the local legacy backend cannot execute them.
    """
    if not isinstance(tools, list):
        return []
    result: list[dict] = []
    skipped: dict[str, int] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            skipped[type(tool).__name__] = skipped.get(type(tool).__name__, 0) + 1
            continue
        tool_type = str(tool.get("type") or "").strip()
        if tool_type == "namespace" and flatten_namespace_tools:
            ns_name = str(tool.get("name") or "").strip()
            sub_tools = tool.get("tools")
            if ns_name and isinstance(sub_tools, list):
                added = 0
                for sub_tool in sub_tools:
                    if not isinstance(sub_tool, dict):
                        continue
                    sub_name = str(sub_tool.get("name") or "").strip()
                    if not sub_name:
                        continue
                    result.append(_responses_chat_tool_entry(
                        f"{ns_name}__{sub_name}",
                        sub_tool.get("description", ""),
                        sub_tool.get("parameters", {}),
                    ))
                    added += 1
                if added:
                    continue
            skipped["namespace_empty"] = skipped.get("namespace_empty", 0) + 1
            continue
        if tool_type != "function":
            skipped[tool_type or "unknown"] = skipped.get(tool_type or "unknown", 0) + 1
            continue
        function = tool.get("function")
        if isinstance(function, dict):
            name = str(function.get("name") or "").strip()
            if not name:
                skipped["function_without_name"] = skipped.get("function_without_name", 0) + 1
                continue
            result.append(_responses_chat_tool_entry(name, function.get("description", ""), function.get("parameters", {})))
            continue
        name = str(tool.get("name") or "").strip()
        if not name:
            skipped["function_without_name"] = skipped.get("function_without_name", 0) + 1
            continue
        result.append(_responses_chat_tool_entry(name, tool.get("description", ""), tool.get("parameters", {})))
    if skipped:
        log_api_event("responses_legacy_chat_tools_skipped", {"skipped": skipped, "forwarded_count": len(result)})
    return result


def _responses_content_to_openai_content(content) -> str | list:
    """Convert Responses API content blocks into chat/completions content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts: list[dict] = []
    for item in content:
        if isinstance(item, str):
            if item:
                parts.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip()
        if item_type in {"input_text", "output_text", "text"}:
            text = str(item.get("text") or "").strip()
            if text:
                parts.append({"type": "text", "text": text})
            continue
        image_part = _normalize_openai_image_part(item)
        if image_part is not None:
            parts.append(image_part)
    if not parts:
        return ""
    if len(parts) == 1 and parts[0].get("type") == "text":
        return str(parts[0].get("text") or "")
    return parts


def _openai_chat_content_is_empty(content: object) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        return not content
    return False


def _merge_openai_chat_content(left: object, right: object) -> object:
    if _openai_chat_content_is_empty(left):
        return right
    if _openai_chat_content_is_empty(right):
        return left
    if isinstance(left, str) and isinstance(right, str):
        return f"{left}\n\n{right}"
    if isinstance(left, list) and isinstance(right, list):
        return [*left, *right]
    if isinstance(left, list) and isinstance(right, str):
        return [*left, {"type": "text", "text": right}]
    if isinstance(left, str) and isinstance(right, list):
        return [{"type": "text", "text": left}, *right]
    return f"{left}\n\n{right}"


def _append_responses_history_message(messages: list[dict], role: str, content: object) -> None:
    """Append a Responses history message in a llama.cpp-chat-safe shape.

    llama.cpp rejects histories that end with multiple assistant messages.  The
    Responses API can represent a single assistant turn as several output items
    (message, reasoning/function_call, final empty message), so the legacy
    fallback must compact adjacent assistant content and drop empty assistant
    placeholders before forwarding to /v1/chat/completions.
    """
    if role == "assistant" and _openai_chat_content_is_empty(content):
        log_api_event(
            "responses_legacy_chat_history_assistant_message_skipped",
            {"reason": "empty_assistant_placeholder"},
        )
        return
    if (
        role == "assistant"
        and messages
        and messages[-1].get("role") == "assistant"
        and not messages[-1].get("tool_calls")
        and "tool_call_id" not in messages[-1]
    ):
        messages[-1]["content"] = _merge_openai_chat_content(messages[-1].get("content"), content)
        log_api_event(
            "responses_legacy_chat_history_assistant_message_merged",
            {"reason": "adjacent_assistant_messages"},
        )
        return
    messages.append({"role": role, "content": content})


def _responses_input_to_openai_messages(
    payload: dict,
    allowed_tool_names: set[str] | None = None,
    tool_registry: ResponsesToolRegistry | None = None,
    allow_tool_output_images: bool = True,
) -> list[dict]:
    """Best-effort shim from modern /v1/responses input into chat messages.

    When translating Responses history to legacy Chat Completions, only keep
    tool-call history for tools that are actually being forwarded to the legacy
    backend.  Otherwise old failed MCP/namespace calls such as
    ``mcp__engram__mem_context`` can be injected back into the local model as
    normal chat context and trigger repeated "unsupported call" loops.
    """
    messages: list[dict] = []
    allowed_tool_call_ids: set[str] = set()
    blocked_tool_call_ids: set[str] = set()
    instructions = str(payload.get("instructions") or "").strip()
    if instructions:
        messages.append({"role": "system", "content": instructions})
    raw_input = payload.get("input")
    if isinstance(raw_input, str):
        if raw_input.strip():
            messages.append({"role": "user", "content": raw_input})
        return messages
    if isinstance(raw_input, dict):
        raw_input = [raw_input]
    if not isinstance(raw_input, list):
        return messages
    failed_tool_call_ids: set[str] = set()
    for item in raw_input:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip()
        if item_type in {"function_call_output", "computer_call_output", "tool_result"}:
            call_id = str(item.get("call_id") or "").strip()
            text = item.get("output") or item.get("text") or item.get("content") or ""
            if call_id and _responses_contains_failed_tool_output(text):
                failed_tool_call_ids.add(call_id)
    for item in raw_input:
        if isinstance(item, str):
            if item:
                messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip()
        if item_type in {"message", ""} or "role" in item:
            role = str(item.get("role") or "user").strip() or "user"
            if role in {"tool", "function"}:
                log_api_event("responses_legacy_chat_history_tool_message_skipped", {"role": role, "reason": "tool_history_not_forwarded_as_user"})
                continue
            _append_responses_history_message(messages, role, _responses_content_to_openai_content(item.get("content")))
            continue
        if item_type in {"input_text", "output_text", "text", "input_image"}:
            messages.append({"role": "user", "content": _responses_content_to_openai_content([item])})
            continue
        if item_type == "function_call":
            raw_name = str(item.get("name") or "").strip()
            namespace = str(item.get("namespace") or "").strip().rstrip("_")
            name = f"{namespace}__{raw_name}" if namespace and raw_name else raw_name
            arguments = str(item.get("arguments") or "{}")
            call_id = str(item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}")
            if name:
                history_deferred_tool = None
                if call_id in failed_tool_call_ids:
                    blocked_tool_call_ids.add(call_id)
                    log_api_event(
                        "responses_legacy_chat_history_tool_call_skipped",
                        {"name": name, "call_id": call_id, "reason": "previous_tool_output_failed"},
                    )
                    continue
                if allowed_tool_names is not None and name not in allowed_tool_names:
                    history_deferred_tool = tool_registry.deferred_tool_for_history(namespace=namespace, name=raw_name) if tool_registry is not None else None
                    if history_deferred_tool is None:
                        blocked_tool_call_ids.add(call_id)
                        log_api_event(
                            "responses_legacy_chat_history_tool_call_skipped",
                            {"name": name, "call_id": call_id, "reason": "not_in_forwarded_legacy_toolset"},
                        )
                        continue
                allowed_tool_call_ids.add(call_id)
                function_name = name
                function_arguments = arguments
                if history_deferred_tool is not None:
                    function_name = RESPONSES_INTERNAL_CALL_DEFERRED_TOOL_NAME
                    function_arguments = json.dumps(
                        {
                            "namespace": history_deferred_tool.namespace,
                            "name": history_deferred_tool.responses_name,
                            "arguments": _decode_tool_call_arguments(arguments),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    log_api_event(
                        "responses_legacy_chat_history_deferred_tool_call_mapped",
                        {
                            "name": name,
                            "call_id": call_id,
                            "internal_name": function_name,
                            "namespace": history_deferred_tool.namespace,
                            "responses_name": history_deferred_tool.responses_name,
                        },
                    )
                tool_call_entry = {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": function_name, "arguments": function_arguments},
                }
                if messages and messages[-1].get("role") == "assistant" and "tool_call_id" not in messages[-1]:
                    previous_tool_calls = messages[-1].setdefault("tool_calls", [])
                    if isinstance(previous_tool_calls, list):
                        previous_tool_calls.append(tool_call_entry)
                    else:
                        messages[-1]["tool_calls"] = [tool_call_entry]
                else:
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [tool_call_entry],
                    })
            continue
        if item_type in {"function_call_output", "computer_call_output", "tool_result"}:
            text = item.get("output") or item.get("text") or item.get("content") or ""
            call_id = str(item.get("call_id") or "").strip()
            if text:
                output_text, output_images, image_count = _extract_legacy_tool_output_parts(text)
                if image_count:
                    log_api_event(
                        "responses_legacy_chat_tool_output_images_extracted",
                        {"call_id": call_id, "images": image_count, "output_preview": output_text[:160]},
                    )
                if call_id:
                    if call_id in failed_tool_call_ids:
                        log_api_event(
                            "responses_legacy_chat_history_tool_output_skipped",
                            {
                                "call_id": call_id,
                                "reason": "previous_tool_output_failed",
                                "was_blocked_tool_call": call_id in blocked_tool_call_ids,
                                "output_preview": output_text[:160],
                            },
                        )
                        continue
                    if allowed_tool_names is not None and call_id not in allowed_tool_call_ids:
                        log_api_event(
                            "responses_legacy_chat_history_tool_output_skipped",
                            {
                                "call_id": call_id,
                                "reason": "matching_tool_call_not_forwarded",
                                "was_blocked_tool_call": call_id in blocked_tool_call_ids,
                                "output_preview": output_text[:160],
                            },
                        )
                        continue
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": output_text})
                    if output_images and allow_tool_output_images:
                        image_content = [{"type": "text", "text": f"Images returned by tool call {call_id}."}, *output_images]
                        messages.append({"role": "user", "content": image_content})
                    elif output_images:
                        log_api_event(
                            "responses_legacy_chat_tool_output_images_not_forwarded_nonvision",
                            {"call_id": call_id, "images": image_count, "output_preview": output_text[:160]},
                        )
                else:
                    if "unsupported call:" in output_text or "unknown MCP server" in output_text:
                        log_api_event(
                            "responses_legacy_chat_history_tool_output_skipped",
                            {"call_id": "", "reason": "orphan_unsupported_tool_output", "output_preview": output_text[:160]},
                        )
                        continue
                    if output_images and allow_tool_output_images:
                        image_content = [{"type": "text", "text": output_text or "Images returned by tool output."}, *output_images]
                        messages.append({"role": "user", "content": image_content})
                    elif output_images:
                        log_api_event(
                            "responses_legacy_chat_tool_output_images_not_forwarded_nonvision",
                            {"call_id": "", "images": image_count, "output_preview": output_text[:160]},
                        )
                        if output_text:
                            messages.append({"role": "user", "content": output_text})
                    elif output_text:
                        messages.append({"role": "user", "content": output_text})
    return messages


def _responses_payload_to_chat_payload(
    payload: dict,
    model_name: str,
    *,
    flatten_namespace_tools: bool = True,
    tool_registry: ResponsesToolRegistry | None = None,
    extra_messages: list[dict] | None = None,
    allow_tool_output_images: bool = True,
) -> dict:
    """Translate /v1/responses request fields supported by legacy chat backends."""
    chat_tools = (
        tool_registry.chat_tools_with_internal_search()
        if tool_registry is not None
        else _responses_tools_to_chat_tools(payload.get("tools"), flatten_namespace_tools=flatten_namespace_tools)
    )
    allowed_tool_names = {
        str(tool.get("function", {}).get("name") or "")
        for tool in chat_tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    stream_enabled = bool(payload.get("stream"))
    upstream_payload = {
        "model": model_name,
        "messages": _responses_input_to_openai_messages(
            payload,
            allowed_tool_names=allowed_tool_names,
            tool_registry=tool_registry,
            allow_tool_output_images=allow_tool_output_images,
        ),
        "stream": stream_enabled,
        # llama.cpp only reuses the slot KV/prompt prefix when this request
        # option is enabled. OpenCode does not send it, so the proxy must
        # preserve the cache by default while still honoring an explicit false.
        "cache_prompt": payload.get("cache_prompt", True),
    }
    if tool_registry is not None and tool_registry.has_deferred_tools:
        directory = tool_registry.directory_text()
        if directory:
            if upstream_payload["messages"] and upstream_payload["messages"][0].get("role") == "system":
                upstream_payload["messages"][0]["content"] = f"{upstream_payload['messages'][0].get('content')}\n\n{directory}"
            else:
                upstream_payload["messages"].insert(0, {"role": "system", "content": directory})
    if extra_messages:
        upstream_payload["messages"].extend(extra_messages)
    if stream_enabled:
        # Codex relies on Responses `response.completed.usage` for context/token
        # accounting.  The legacy Chat Completions stream only includes usage if
        # requested explicitly, so always ask llama.cpp for it when we are acting
        # as the Responses compatibility adapter.
        stream_options = payload.get("stream_options") if isinstance(payload.get("stream_options"), dict) else {}
        upstream_payload["stream_options"] = {**stream_options, "include_usage": True}
    passthrough_fields = {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "seed",
    }
    for field_name in passthrough_fields:
        if field_name in payload:
            upstream_payload[field_name] = payload[field_name]
    if "max_output_tokens" in payload:
        upstream_payload["max_tokens"] = payload["max_output_tokens"]
    elif "max_tokens" in payload:
        upstream_payload["max_tokens"] = payload["max_tokens"]
    if chat_tools:
        upstream_payload["tools"] = chat_tools
        if "tool_choice" in payload:
            upstream_payload["tool_choice"] = _responses_tool_choice_to_chat_tool_choice(payload["tool_choice"])

    return upstream_payload


def _responses_internal_round_max_tokens() -> int:
    raw = _env_value("HEIMDALL_GATEWAY_RESPONSES_INTERNAL_MAX_TOKENS", "LLAMACPP_RESPONSES_INTERNAL_MAX_TOKENS", "4096")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 4096
    return max(256, value)



def _responses_tool_choice_to_chat_tool_choice(tool_choice: object) -> object:
    if not isinstance(tool_choice, dict):
        return tool_choice
    choice_type = str(tool_choice.get("type") or "").strip()
    if choice_type == "function" and isinstance(tool_choice.get("function"), dict):
        return tool_choice
    if choice_type == "function":
        name = str(tool_choice.get("name") or "").strip()
        if name:
            return {"type": "function", "function": {"name": name}}
    return tool_choice

def _responses_raw_passthrough_enabled() -> bool:
    value = _env_value("HEIMDALL_GATEWAY_RESPONSES_RAW_PASSTHROUGH", "LLAMACPP_SUPERSERVER_RESPONSES_RAW_PASSTHROUGH", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_runtime_enabled() -> bool:
    value = _env_value("HEIMDALL_GATEWAY_SAFE_RUNTIME", "HEIMDALL_GATEWAY_SAFE_RUNTIME", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_responses_usage(usage: object) -> dict | None:
    """Normalize Chat Completions usage into Responses API usage shape."""
    if not isinstance(usage, dict):
        return None
    if any(key in usage for key in ("input_tokens", "output_tokens")):
        input_tokens = _to_int_or_none(usage.get("input_tokens")) or 0
        output_tokens = _to_int_or_none(usage.get("output_tokens")) or 0
        total_tokens = _to_int_or_none(usage.get("total_tokens"))
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        normalized = dict(usage)
        normalized["input_tokens"] = input_tokens
        normalized["output_tokens"] = output_tokens
        normalized["total_tokens"] = total_tokens
        normalized.setdefault("input_tokens_details", {"cached_tokens": 0})
        normalized.setdefault("output_tokens_details", {"reasoning_tokens": 0})
        return normalized

    prompt_tokens = _to_int_or_none(usage.get("prompt_tokens")) or 0
    completion_tokens = _to_int_or_none(usage.get("completion_tokens")) or 0
    total_tokens = _to_int_or_none(usage.get("total_tokens"))
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens
    prompt_details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    completion_details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
    return {
        "input_tokens": prompt_tokens,
        "input_tokens_details": {
            "cached_tokens": _to_int_or_none(prompt_details.get("cached_tokens")) or 0,
        },
        "output_tokens": completion_tokens,
        "output_tokens_details": {
            "reasoning_tokens": _to_int_or_none(completion_details.get("reasoning_tokens")) or 0,
        },
        "total_tokens": total_tokens,
    }


def _chat_response_to_responses_payload(data: dict, model_name: str, request_payload: dict | None = None) -> dict:
    request_payload = request_payload or {}
    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output_text = message.get("content") or ""
    reasoning_text = message.get("reasoning_content") or message.get("reasoning") or ""
    
    output_items = []
    if reasoning_text:
        output_items.append({
            "id": f"rs_{uuid.uuid4().hex}",
            "type": "reasoning",
            "status": "completed",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": reasoning_text}],
        })
    if output_text:
        output_items.append({
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": output_text, "annotations": []}],
        })
        
    for tc in message.get("tool_calls", []):
        func = tc.get("function", {})
        call_id = tc.get("call_id") or tc.get("id") or f"call_{uuid.uuid4().hex}"
        item_id = tc.get("responses_item_id") or tc.get("item_id") or (tc.get("id") if str(tc.get("id") or "").startswith("fc_") else f"fc_{uuid.uuid4().hex}")
        namespace = tc.get("namespace") or func.get("namespace")
        arguments, _ = _sanitize_responses_tool_arguments(func.get("name", ""), namespace, func.get("arguments", "{}"))
        output_item = {
            "id": item_id,
            "type": "function_call",
            "status": "completed",
            "call_id": call_id,
            "name": func.get("name", ""),
            "arguments": arguments,
        }
        if namespace:
            output_item["namespace"] = namespace
        output_items.append(output_item)
        
    payload = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": model_name,
        "output": output_items,
        "parallel_tool_calls": bool(request_payload.get("parallel_tool_calls", True)),
        "tool_choice": request_payload.get("tool_choice", "auto"),
        "tools": request_payload.get("tools") if isinstance(request_payload.get("tools"), list) else [],
        "output_text": output_text,
    }
    normalized_usage = _normalize_responses_usage(data.get("usage"))
    if normalized_usage is not None:
        payload["usage"] = normalized_usage
    return payload


def _responses_payload_has_output_items(payload: dict) -> bool:
    """Return True when a Responses payload has something meaningful to send.

    An empty message after internal repair feedback is not a valid completion:
    it means the local model stopped instead of retrying/correcting the tool
    call.  Function calls are valid even when output_text is empty.
    """
    output = payload.get("output")
    if not isinstance(output, list):
        return False
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call" and str(item.get("name") or "").strip():
            return True
        if item_type == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and str(part.get("text") or "").strip():
                    return True
        if item_type == "reasoning":
            for part in item.get("content") or []:
                if isinstance(part, dict) and str(part.get("text") or "").strip():
                    return True
    return False


def _responses_event(event_type: str, sequence_number: int, **fields) -> dict:
    payload = {"type": event_type, "sequence_number": sequence_number}
    payload.update(fields)
    return payload


def _write_sse_event(handler: BaseHTTPRequestHandler, event_name: str, event_payload: dict) -> None:
    handler.wfile.write(f"event: {event_name}\n".encode("utf-8"))
    handler.wfile.write(("data: " + json.dumps(event_payload, ensure_ascii=False) + "\n\n").encode("utf-8"))
    handler.wfile.flush()


def _start_responses_sse_stream(handler: BaseHTTPRequestHandler) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()


def _write_sse_comment(handler: BaseHTTPRequestHandler, comment: str) -> None:
    safe_comment = str(comment or "keepalive").replace("\n", " ")
    handler.wfile.write(f": {safe_comment}\n\n".encode("utf-8"))
    handler.wfile.flush()


def _response_in_progress(payload: dict) -> dict:
    return {**payload, "status": "in_progress", "output": []}


def _response_message_item(item_id: str, text: str, status: str = "completed") -> dict:
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _response_function_call_item(item_id: str, call_id: str, name: str, arguments: str, status: str = "completed", namespace: str | None = None) -> dict:
    item = {
        "id": item_id,
        "type": "function_call",
        "status": status,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }
    if namespace:
        item["namespace"] = namespace
    return item


def _response_reasoning_item(item_id: str, text: str, status: str = "completed") -> dict:
    return {
        "id": item_id,
        "type": "reasoning",
        "status": status,
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
    }


def _responses_payload_sse_events(payload: dict) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    seq = 0

    seq += 1
    events.append(("response.created", _responses_event("response.created", seq, response=_response_in_progress(payload))))

    for output_index, item in enumerate(payload.get("output") or []):
        item_type = item.get("type")
        if item_type == "reasoning":
            text = ""
            content = item.get("content") or []
            for part in content:
                if part.get("type") == "reasoning_text":
                    text += str(part.get("text") or "")
            item_id = str(item.get("id") or f"rs_{uuid.uuid4().hex}")
            seq += 1
            events.append(("response.output_item.added", _responses_event("response.output_item.added", seq, output_index=output_index, item=_response_reasoning_item(item_id, "", "in_progress"))))
            if text:
                seq += 1
                events.append(("response.reasoning_text.delta", _responses_event("response.reasoning_text.delta", seq, item_id=item_id, output_index=output_index, content_index=0, delta=text)))
            seq += 1
            events.append(("response.reasoning_text.done", _responses_event("response.reasoning_text.done", seq, item_id=item_id, output_index=output_index, content_index=0, text=text)))
            seq += 1
            events.append(("response.output_item.done", _responses_event("response.output_item.done", seq, output_index=output_index, item=_response_reasoning_item(item_id, text, "completed"))))
        elif item_type == "message":
            text = ""
            content = item.get("content") or []
            for part in content:
                if part.get("type") == "output_text":
                    text += str(part.get("text") or "")
            in_progress_item = _response_message_item(str(item.get("id") or f"msg_{uuid.uuid4().hex}"), "", "in_progress")
            seq += 1
            events.append(("response.output_item.added", _responses_event("response.output_item.added", seq, output_index=output_index, item=in_progress_item)))
            part = {"type": "output_text", "text": "", "annotations": []}
            seq += 1
            events.append(("response.content_part.added", _responses_event("response.content_part.added", seq, item_id=in_progress_item["id"], output_index=output_index, content_index=0, part=part)))
            if text:
                seq += 1
                events.append(("response.output_text.delta", _responses_event("response.output_text.delta", seq, item_id=in_progress_item["id"], output_index=output_index, content_index=0, delta=text, logprobs=[])))
            done_part = {"type": "output_text", "text": text, "annotations": []}
            seq += 1
            events.append(("response.output_text.done", _responses_event("response.output_text.done", seq, item_id=in_progress_item["id"], output_index=output_index, content_index=0, text=text, logprobs=[])))
            seq += 1
            events.append(("response.content_part.done", _responses_event("response.content_part.done", seq, item_id=in_progress_item["id"], output_index=output_index, content_index=0, part=done_part)))
            seq += 1
            events.append(("response.output_item.done", _responses_event("response.output_item.done", seq, output_index=output_index, item=_response_message_item(in_progress_item["id"], text, "completed"))))
        elif item_type == "function_call":
            item_id = str(item.get("id") or f"fc_{uuid.uuid4().hex}")
            call_id = str(item.get("call_id") or item_id)
            name = str(item.get("name") or "")
            namespace = str(item.get("namespace") or "").strip() or None
            arguments, _ = _sanitize_responses_tool_arguments(name, namespace, item.get("arguments") or "")
            started_item = _response_function_call_item(item_id, call_id, name, "", "in_progress", namespace)
            seq += 1
            events.append(("response.output_item.added", _responses_event("response.output_item.added", seq, output_index=output_index, item=started_item)))
            if arguments:
                seq += 1
                events.append(("response.function_call_arguments.delta", _responses_event("response.function_call_arguments.delta", seq, item_id=item_id, output_index=output_index, delta=arguments)))
            seq += 1
            done_payload = _responses_event("response.function_call_arguments.done", seq, item_id=item_id, output_index=output_index, name=name, arguments=arguments)
            if namespace:
                done_payload["namespace"] = namespace
            events.append(("response.function_call_arguments.done", done_payload))
            seq += 1
            events.append(("response.output_item.done", _responses_event("response.output_item.done", seq, output_index=output_index, item=_response_function_call_item(item_id, call_id, name, arguments, "completed", namespace))))

    seq += 1
    events.append(("response.completed", _responses_event("response.completed", seq, response=payload)))
    return events


def _write_responses_sse(handler: BaseHTTPRequestHandler, payload: dict) -> None:
    _start_responses_sse_stream(handler)
    _write_responses_sse_events(handler, payload)


def _write_responses_sse_events(handler: BaseHTTPRequestHandler, payload: dict) -> None:
    for event_name, event_payload in _responses_payload_sse_events(payload):
        _write_sse_event(handler, event_name, event_payload)
    handler.wfile.write(b"data: [DONE]\n\n")
    handler.wfile.flush()


def _write_responses_sse_error(handler: BaseHTTPRequestHandler, message: str, error_type: str = "server_error") -> None:
    seq = 1
    payload = {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "failed",
        "error": {"message": message, "type": error_type},
        "output": [],
    }
    _write_sse_event(handler, "response.failed", _responses_event("response.failed", seq, response=payload))
    handler.wfile.write(b"data: [DONE]\n\n")
    handler.wfile.flush()



_CHAT_TOOL_REPAIR_NOTICE_RE = re.compile(
    r"\s*(?:<think>\s*)?↻\s*Retrying tool call generation \(attempt \d+/\d+\);[^\n]*(?:\n\s*</think>)?",
    re.IGNORECASE,
)


def _strip_chat_tool_repair_notice_text(content: object) -> str:
    text = str(content or "")
    previous = None
    while previous != text:
        previous = text
        text = _CHAT_TOOL_REPAIR_NOTICE_RE.sub("\n", text)
    return text.strip()


def _sanitize_chat_tool_repair_notices_in_messages(messages: object) -> list[dict]:
    if not isinstance(messages, list):
        return []
    sanitized: list[dict] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        if copied.get("role") == "assistant" and "content" in copied:
            content = copied.get("content")
            if isinstance(content, str):
                copied["content"] = _strip_chat_tool_repair_notice_text(content)
            elif isinstance(content, list):
                new_parts = []
                for part in content:
                    if isinstance(part, dict):
                        part_copy = dict(part)
                        for key in ("text", "content", "output"):
                            if isinstance(part_copy.get(key), str):
                                part_copy[key] = _strip_chat_tool_repair_notice_text(part_copy.get(key))
                        new_parts.append(part_copy)
                    else:
                        new_parts.append(part)
                copied["content"] = new_parts
        sanitized.append(copied)
    return sanitized


def _merge_chat_message_content(left: object, right: object) -> object:
    if left in (None, ""):
        return right if right is not None else ""
    if right in (None, ""):
        return left
    if isinstance(left, list) or isinstance(right, list):
        merged: list[object] = []
        if isinstance(left, list):
            merged.extend(left)
        else:
            merged.append({"type": "text", "text": str(left)})
        if isinstance(right, list):
            merged.extend(right)
        else:
            merged.append({"type": "text", "text": str(right)})
        return merged
    return f"{left}\n{right}"


def _normalize_system_messages_for_llamacpp(messages: object) -> list[dict]:
    """Collapse OpenAI/OpenCode system messages into the first system item.

    Several clients, notably OpenCode, append system layers as separate
    messages. llama.cpp chat templates are not uniform: Qwen3.5's template
    accepts exactly one system message and raises if any later system item is
    encountered. Keep the first system position and preserve every layer's
    content in order, without changing user/assistant/tool messages.
    """
    if not isinstance(messages, list):
        return []
    normalized = [dict(item) for item in messages if isinstance(item, dict)]
    system_indexes = [
        index for index, item in enumerate(normalized)
        if str(item.get("role") or "") == "system"
    ]
    if len(system_indexes) < 2:
        return normalized
    first_index = system_indexes[0]
    first_system = dict(normalized[first_index])
    for index in system_indexes[1:]:
        left = first_system.get("content")
        right = normalized[index].get("content")
        if isinstance(left, str) and isinstance(right, str) and left and right:
            first_system["content"] = f"{left}\n\n{right}"
        else:
            first_system["content"] = _merge_chat_message_content(left, right)
    return [
        first_system if index == first_index else item
        for index, item in enumerate(normalized)
        if index == first_index or index not in set(system_indexes[1:])
    ]


def _normalize_trailing_assistant_messages_for_llamacpp(messages: object) -> list[dict]:
    """Merge a final run of plain assistant messages into one assistant message.

    llama.cpp rejects prompts ending in two or more assistant messages. Some
    clients use trailing assistant prefills after thinking-only continuations.
    Preserve the information but send a single final assistant message. Do not
    merge assistant messages that contain tool calls because those have protocol
    semantics and should fail loudly rather than be rewritten.
    """
    if not isinstance(messages, list):
        return []
    normalized = [dict(item) for item in messages if isinstance(item, dict)]
    tail_start = len(normalized)
    while tail_start > 0 and str(normalized[tail_start - 1].get("role") or "") == "assistant":
        tail_start -= 1
    if len(normalized) - tail_start < 2:
        return normalized
    tail = normalized[tail_start:]
    if any(item.get("tool_calls") for item in tail):
        return normalized
    merged = dict(tail[0])
    for item in tail[1:]:
        merged["content"] = _merge_chat_message_content(merged.get("content"), item.get("content"))
        for key, value in item.items():
            if key in {"role", "content"}:
                continue
            if key not in merged or merged.get(key) in (None, "", []):
                merged[key] = value
    return normalized[:tail_start] + [merged]


def resolve_chat_tool_continue_repair_config(args = None) -> dict[str, object]:
    raw = _load_server_config_payload(args).get("experimental")
    cfg = _normalize_experimental_config(raw).get("chat_tool_continue_repair", {})
    if not isinstance(cfg, dict):
        cfg = dict(_default_experimental_config()["chat_tool_continue_repair"])
    return cfg


def _chat_tool_names_from_payload(tools: object, limit: int = 50) -> list[str]:
    names: list[str] = []
    if not isinstance(tools, list):
        return names
    for item in tools:
        if not isinstance(item, dict):
            continue
        name = ""
        fn = item.get("function")
        if isinstance(fn, dict):
            name = str(fn.get("name") or "").strip()
        if not name:
            name = str(item.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def _valid_chat_tool_names_from_payload(tools: object) -> set[str]:
    return set(_chat_tool_names_from_payload(tools, limit=500))


def _normalize_tool_call_arguments_for_example(value: object, max_chars: int = 2000) -> str:
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
        text = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        # Keep original string for diagnostics, but invalid JSON cannot be used
        # as a formatting example for a future tool repair.
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


def _extract_chat_tool_call_examples_from_messages(
    messages: object,
    tools: object,
    *,
    limit: int = 3,
) -> list[dict[str, str]]:
    """Return recent valid assistant tool-call examples from request history.

    The examples are used only as text inside the repair prompt, never as
    protocol-level assistant/tool messages, so they cannot mutate conversation
    state or create invalid trailing assistant sequences.
    """
    if not isinstance(messages, list):
        return []
    valid_names = _valid_chat_tool_names_from_payload(tools)
    examples: list[dict[str, str]] = []
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue
        for tool_call in reversed(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") or {}
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            if not name or (valid_names and name not in valid_names):
                continue
            arguments = _normalize_tool_call_arguments_for_example(function.get("arguments") or "")
            if not arguments:
                continue
            example = {"tool_name": name, "arguments": arguments}
            if example not in examples:
                examples.append(example)
            if len(examples) >= limit:
                return examples
    return examples


class ChatToolCallExampleStore:
    def __init__(self, max_per_key: int = 4, ttl_s: float = 3600.0):
        self.lock = threading.RLock()
        self.max_per_key = max(1, int(max_per_key))
        self.ttl_s = max(60.0, float(ttl_s))
        self.examples_by_key: dict[str, list[dict[str, object]]] = {}

    def _prune_locked(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        stale_keys = []
        for key, examples in self.examples_by_key.items():
            fresh = [item for item in examples if now - float(item.get("seen_monotonic") or 0.0) <= self.ttl_s]
            if fresh:
                self.examples_by_key[key] = fresh[: self.max_per_key]
            else:
                stale_keys.append(key)
        for key in stale_keys:
            self.examples_by_key.pop(key, None)

    def remember(self, key: str, tool_calls: object, tools: object, *, request_id: str = "") -> None:
        if not key or not isinstance(tool_calls, list) or not tool_calls:
            return
        valid_names = _valid_chat_tool_names_from_payload(tools)
        now = time.monotonic()
        new_examples: list[dict[str, object]] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") or {}
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            if not name or (valid_names and name not in valid_names):
                continue
            arguments = _normalize_tool_call_arguments_for_example(function.get("arguments") or "")
            if not arguments:
                continue
            new_examples.append(
                {
                    "tool_name": name,
                    "arguments": arguments,
                    "request_id": request_id,
                    "seen_monotonic": now,
                }
            )
        if not new_examples:
            return
        with self.lock:
            self._prune_locked(now)
            existing = self.examples_by_key.setdefault(key, [])
            for example in new_examples:
                existing = [
                    old for old in existing
                    if not (
                        old.get("tool_name") == example.get("tool_name")
                        and old.get("arguments") == example.get("arguments")
                    )
                ]
                existing.insert(0, example)
            self.examples_by_key[key] = existing[: self.max_per_key]

    def get(self, keys: list[str], tools: object) -> dict[str, str] | None:
        valid_names = _valid_chat_tool_names_from_payload(tools)
        with self.lock:
            self._prune_locked()
            for key in keys:
                if not key:
                    continue
                for example in self.examples_by_key.get(key, []):
                    name = str(example.get("tool_name") or "").strip()
                    args = str(example.get("arguments") or "").strip()
                    if not name or not args:
                        continue
                    if valid_names and name not in valid_names:
                        continue
                    return {"tool_name": name, "arguments": args}
        return None


CHAT_TOOL_CALL_EXAMPLES = ChatToolCallExampleStore()


def _chat_tool_example_keys(model: str, upstream_model: str = "", conversation_key: str = "") -> list[str]:
    model_key = str(model or "").strip()
    upstream_key = str(upstream_model or "").strip()
    conversation = str(conversation_key or "").strip()
    keys: list[str] = []
    for base in (upstream_key, model_key):
        if not base:
            continue
        if conversation:
            keys.append(f"conversation:{base}:{conversation}")
        keys.append(f"model:{base}")
    return list(dict.fromkeys(keys))


def _strip_markup_edges_for_tool_heuristics(content: object) -> str:
    """Ignore XML/HTML-like wrappers appended or prepended by clients.

    OpenCode may append metadata such as ``<dcp-message-id>...</dcp-message-id>``
    after the model text. The UI can hide that metadata, making a response
    that visually ends in ``:`` appear not to match the repair heuristic.
    This only changes the copy used for detection; the original response is
    still returned unchanged.
    """
    text = str(content or "")
    paired_suffix = re.compile(r"\s*<([A-Za-z][A-Za-z0-9_.:-]*)\b[^<>]*>.*?</\1>\s*$", re.DOTALL)
    edge_tag = re.compile(r"^\s*</?[A-Za-z][A-Za-z0-9_.:-]*\b[^<>]*>\s*")
    trailing_tag = re.compile(r"\s*</?[A-Za-z][A-Za-z0-9_.:-]*\b[^<>]*>\s*$")
    for _ in range(8):
        updated = paired_suffix.sub("", text, count=1)
        updated = edge_tag.sub("", updated, count=1)
        updated = trailing_tag.sub("", updated, count=1)
        if updated == text:
            break
        text = updated
    return text.strip()


def _chat_tool_continue_trigger_reason(
    content: object,
    tool_calls: object,
    tools: object,
    trigger_prefixes: object = None,
) -> str:
    if not isinstance(tools, list) or not tools:
        return ""
    has_tool_calls = bool(tool_calls)
    if isinstance(tool_calls, list):
        has_tool_calls = len(tool_calls) > 0
    if has_tool_calls:
        return ""
    visible = _strip_chat_tool_repair_notice_text(content)
    if not visible:
        return "empty_visible_content"
    heuristic_visible = _strip_markup_edges_for_tool_heuristics(visible)
    if heuristic_visible.endswith(":"):
        return "visible_content_trailing_colon"
    visible_lines = [line.lstrip() for line in heuristic_visible.splitlines()]
    visible_line_starts = [line.casefold() for line in visible_lines]
    visible_start = heuristic_visible.lstrip().casefold()
    visible_folded = visible.casefold()
    # Hermes/OpenCode sometimes receives a model-imagined inline terminal tag
    # as visible text instead of a real tool_call. Treat this as a strong
    # syntactic failure signal even if the closing tag appears mid-line.
    if "[terminal_inline" in visible_folded or "<terminal_inline" in visible_folded or "</terminal_inline>" in visible_folded:
        return "visible_content_terminal_inline_markup"
    for prefix in _normalize_chat_tool_continue_trigger_prefixes(trigger_prefixes):
        folded_prefix = prefix.casefold()
        if visible_start.startswith(folded_prefix):
            return "visible_content_configured_prefix"
        if any(line.startswith(folded_prefix) for line in visible_line_starts if line):
            return "visible_content_configured_prefix_line"
    tool_names = {name.casefold() for name in _chat_tool_names_from_payload(tools, limit=200)}
    if re.search(r"```[\s\S]*?```", visible):
        shell_like = re.search(
            r"```(?:bash|sh|shell|zsh|python|python3)?\s*[\r\n]+"
            r"(?=[\s\S]*(?:\b(?:cd|python3?|bash|sh|zsh|sudo|docker|grep|find|cat|ls|curl|wget|git|sed|awk)\b|[/~.]|&&|\|\|))",
            visible,
            re.IGNORECASE,
        )
        action_like = re.search(
            r"\b(voy a|déjame|dejame|let me|i will|i'll|verificar|ejecut|buscar|leer|escribir|inspect|run|search|read|write)\b",
            visible,
            re.IGNORECASE,
        )
        if shell_like or action_like:
            return "visible_content_fenced_tool_like_block"
    for tool_name in tool_names:
        escaped = re.escape(tool_name)
        if re.search(rf"(^|[^\w.-]){escaped}\s*\(", visible_folded):
            return "visible_content_pseudo_tool_call"
    for line in visible_lines:
        if not line.startswith("["):
            continue
        match = re.match(r"^\[([A-Za-z_][A-Za-z0-9_.-]*)(?:\s|\]|=)", line)
        if match and match.group(1).casefold() in tool_names:
            return "visible_content_pseudo_tool_line"
    return ""



def _format_chat_tool_repair_prompt(prompt_template: object, tools: object, fallback: str) -> str:
    template = str(prompt_template or "").strip() or fallback
    tool_names = _chat_tool_names_from_payload(tools)
    tool_names_text = ", ".join(tool_names) if tool_names else "(tool names unavailable)"
    try:
        return template.format(tool_names=tool_names_text, tools=tool_names_text)
    except Exception:
        # Do not let a malformed user-configured template break live requests.
        return f"{template}\nAvailable tool names: {tool_names_text}."


def _format_chat_tool_call_example_for_prompt(example: object) -> str:
    if not isinstance(example, dict):
        return "No prior valid tool_call example is available. Use the provided tool schemas."
    name = str(example.get("tool_name") or "").strip()
    arguments = str(example.get("arguments") or "").strip()
    if not name or not arguments:
        return "No prior valid tool_call example is available. Use the provided tool schemas."
    return (
        "Recent valid tool_call example from this same environment, for formatting only:\n"
        f"Tool name: {name}\n"
        "Arguments JSON:\n"
        f"{arguments}\n\n"
        "Do not copy this exact example unless it matches the current intended action."
    )


def _format_failed_visible_response_for_prompt(assistant_message: object, max_chars: int = 4000) -> str:
    if not isinstance(assistant_message, dict):
        return "(unavailable)"
    content = _strip_chat_tool_repair_notice_text(assistant_message.get("content") or "")
    if not content:
        return "(empty visible response)"
    if len(content) > max_chars:
        return content[:max_chars] + "…"
    return content


def _chat_tool_continue_repair_messages(
    messages: list[dict],
    assistant_message: dict,
    tools: object,
    prompt_template: object = None,
    include_failed_assistant_message: bool = False,
    tool_call_example: object = None,
) -> list[dict]:
    repaired = _sanitize_chat_tool_repair_notices_in_messages(messages)
    if include_failed_assistant_message:
        content = _strip_chat_tool_repair_notice_text(assistant_message.get("content") or "")
        repaired.append({"role": "assistant", "content": content})
    default_prompt = _default_experimental_config()["chat_tool_continue_repair"]["prompt"]
    base_prompt = _format_chat_tool_repair_prompt(prompt_template, tools, default_prompt)
    failed_visible = _format_failed_visible_response_for_prompt(assistant_message)
    example_text = _format_chat_tool_call_example_for_prompt(tool_call_example)
    combined_prompt = (
        f"{base_prompt}\n\n"
        "Previous visible response that failed to produce a valid tool_call:\n"
        "-----\n"
        f"{failed_visible}\n"
        "-----\n\n"
        f"{example_text}\n\n"
        "Repair rule: infer the intended tool use from the failed visible response and emit exactly one real tool_call now. "
        "Do not write Markdown, pseudo tool syntax, XML tags, shell/code blocks, or explanatory prose."
    )
    repaired.append(
        {
            "role": "user",
            "content": combined_prompt,
        }
    )
    return repaired






def _chat_tool_truncated_trigger_reason(stream_state: dict[str, object]) -> str:
    finish_reason = str(stream_state.get("finish_reason") or "")
    try:
        tool_call_chunks = int(stream_state.get("tool_call_chunks") or 0)
    except Exception:
        tool_call_chunks = 0
    if tool_call_chunks > 0 and finish_reason == "length":
        return "tool_calls_truncated_length"
    return ""


def _chat_tool_truncated_repair_messages(messages: list[dict], tools: object, prompt_template: object = None) -> list[dict]:
    repaired = [dict(item) for item in messages if isinstance(item, dict)]
    default_prompt = _default_experimental_config()["chat_tool_continue_repair"]["truncated_tool_call_prompt"]
    repaired.append(
        {
            "role": "user",
            "content": _format_chat_tool_repair_prompt(prompt_template, tools, default_prompt),
        }
    )
    return repaired


def _send_openai_chat_sse_truncated_tool_error(
    handler: BaseHTTPRequestHandler,
    *,
    request_id: str,
    model: str,
    rounds: int,
) -> None:
    _send_openai_chat_sse_status(
        handler,
        request_id=request_id,
        model=model,
        content=(
            "\n⚠️ Tool call generation was truncated before valid JSON arguments were complete; "
            f"stopping after {rounds} repair attempt(s) instead of forwarding an incomplete tool call.\n"
        ),
    )
    handler.wfile.write(b"data: [DONE]\n\n")
    handler.wfile.flush()

def _chat_completion_state_from_payload(data: dict) -> dict[str, object]:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = str(message.get("content") or "")
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
    tool_calls = message.get("tool_calls") or []
    return {
        "choice": choice,
        "message": message,
        "content": content,
        "reasoning": reasoning,
        "tool_calls": tool_calls,
        "finish_reason": choice.get("finish_reason") or "",
    }


def _chat_completion_state_from_sse_lines(lines: list[bytes]) -> dict[str, object]:
    content_parts: list[str] = []
    reasoning_len = 0
    tool_call_chunks = 0
    finish_reason = ""
    tool_calls_by_index: dict[str, dict[str, object]] = {}
    for line in lines:
        if not line or not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        try:
            chunk_data = json.loads(line[6:].decode("utf-8", errors="ignore"))
        except Exception:
            continue
        choice0 = (chunk_data.get("choices") or [{}])[0]
        delta = choice0.get("delta", {}) or {}
        content = delta.get("content", "") or ""
        reasoning = delta.get("reasoning_content", "") or ""
        tool_calls = delta.get("tool_calls") or []
        if content:
            content_parts.append(str(content))
        if reasoning:
            reasoning_len += len(str(reasoning))
        if tool_calls:
            tool_call_chunks += len(tool_calls) if isinstance(tool_calls, list) else 1
            for tool_call in tool_calls if isinstance(tool_calls, list) else [tool_calls]:
                if not isinstance(tool_call, dict):
                    continue
                index_key = str(tool_call.get("index") if tool_call.get("index") is not None else tool_call.get("id") or "0")
                existing = tool_calls_by_index.setdefault(
                    index_key,
                    {
                        "id": tool_call.get("id") or f"call_{index_key}",
                        "type": tool_call.get("type") or "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if tool_call.get("id"):
                    existing["id"] = tool_call.get("id")
                if tool_call.get("type"):
                    existing["type"] = tool_call.get("type")
                function = tool_call.get("function") or {}
                if isinstance(function, dict):
                    existing_function = existing.setdefault("function", {"name": "", "arguments": ""})
                    if not isinstance(existing_function, dict):
                        existing_function = {"name": "", "arguments": ""}
                        existing["function"] = existing_function
                    if function.get("name"):
                        existing_function["name"] = str(function.get("name") or "")
                    if function.get("arguments"):
                        existing_function["arguments"] = str(existing_function.get("arguments") or "") + str(function.get("arguments") or "")
        if choice0.get("finish_reason"):
            finish_reason = str(choice0.get("finish_reason") or "")
    content = "".join(content_parts)
    tool_calls_out: list[dict] = []
    for key in sorted(tool_calls_by_index.keys(), key=lambda x: int(x) if x.isdigit() else x):
        tool_calls_out.append(tool_calls_by_index[key])
    if tool_call_chunks > 0 and not tool_calls_out:
        tool_calls_out = [{}] * tool_call_chunks
    return {
        "message": {"role": "assistant", "content": content, "tool_calls": tool_calls_out},
        "content": content,
        "reasoning": "",
        "reasoning_len": reasoning_len,
        "tool_calls": tool_calls_out,
        "tool_call_chunks": tool_call_chunks,
        "finish_reason": finish_reason,
    }


def _write_openai_chat_sse_lines(handler: BaseHTTPRequestHandler, lines: list[bytes], content_type: str = "text/event-stream") -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.end_headers()
    for line in lines:
        if line:
            handler.wfile.write(line + b"\n")
        else:
            handler.wfile.write(b"\n")
    handler.wfile.flush()



def _force_tool_choice_for_chat_repair(payload: dict[str, object]) -> dict[str, object]:
    repaired = dict(payload)
    current = repaired.get("tool_choice")
    if current in (None, "", "none", "auto"):
        repaired["tool_choice"] = "required"
    # The model already thought on the failed attempt; repair rounds only need
    # to emit the corrected call, not re-enter a full reasoning phase.
    if repaired.get("thinking_budget_tokens") is None and repaired.get("reasoning_budget_tokens") is None:
        repaired["thinking_budget_tokens"] = CHAT_TOOL_CONTINUE_REPAIR_THINKING_BUDGET_TOKENS
    return repaired

def _apply_chat_tool_continue_repair_token_cap(payload: dict, fallback_max_tokens: object) -> dict:
    capped = dict(payload)

    def _positive_int(value: object) -> int:
        try:
            parsed = int(value)
        except Exception:
            return 0
        return parsed if parsed > 0 else 0

    # Repair rounds should honor the caller's requested output budget when it
    # exists.  The configured value is only a fallback for clients that omit an
    # explicit max_tokens/n_predict.  Capping a large external request to the
    # repair default can truncate long but valid tool-call JSON arguments.
    effective_cap = _positive_int(capped.get("max_tokens"))
    if effective_cap <= 0:
        effective_cap = _positive_int(capped.get("n_predict"))
    if effective_cap <= 0:
        effective_cap = _positive_int(fallback_max_tokens)
    if effective_cap <= 0:
        return capped

    capped["max_tokens"] = effective_cap
    if "n_predict" in capped:
        capped["n_predict"] = effective_cap
    return capped


def _chat_tool_continue_loop_guard_reason(state: dict[str, object], loop_guard: dict[str, object] | None) -> str:
    if not isinstance(loop_guard, dict) or not _as_bool(loop_guard.get("enabled"), True):
        return ""
    if int(state.get("tool_call_chunks") or 0) > 0:
        return ""
    visible_len = int(state.get("visible_content_len") or 0)
    reasoning_len = int(state.get("reasoning_len") or 0)
    generated_chars = visible_len + reasoning_len
    try:
        no_tool_limit = max(0, int(loop_guard.get("no_tool_call_max_chars", 0)))
    except Exception:
        no_tool_limit = 12000
    try:
        budget_tokens = max(0, int(state.get("thinking_budget_tokens") or 0))
    except Exception:
        budget_tokens = 0
    if budget_tokens > 0:
        # Deep thinking legitimately consumes the injected reasoning budget
        # before the first tool call; state counts chars, the budget is tokens.
        no_tool_limit = max(no_tool_limit, budget_tokens * 4 + 4096)
    if no_tool_limit > 0 and generated_chars >= no_tool_limit:
        return "no_tool_call_generation_limit"
    tail = str(state.get("_loop_text_tail") or "")
    try:
        min_chars = max(0, int(loop_guard.get("repeated_tail_min_chars", 3000)))
    except Exception:
        min_chars = 3000
    try:
        repetitions = max(2, int(loop_guard.get("repeated_tail_repetitions", 4)))
    except Exception:
        repetitions = 4
    if min_chars <= 0 or len(tail) < min_chars:
        return ""
    normalized_tail = " ".join(tail.lower().split())
    if len(normalized_tail) < min_chars:
        return ""
    segment_len = min(512, max(120, min_chars // 6))
    segment = normalized_tail[-segment_len:]
    if not segment.strip():
        return ""
    if normalized_tail.count(segment) >= repetitions:
        return "repeated_tail"
    return ""


def _send_openai_chat_sse_status(
    handler: BaseHTTPRequestHandler,
    *,
    request_id: str,
    model: str,
    content: str,
) -> None:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }
    handler.wfile.write(("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8"))
    handler.wfile.flush()

def _send_openai_chat_sse_error(handler: BaseHTTPRequestHandler, message: str) -> None:
    payload = {"error": {"message": message, "type": "server_error"}}
    try:
        handler.wfile.write(("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8"))
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()
    except Exception:
        pass


def _buffer_openai_chat_sse_with_keepalive(
    handler: BaseHTTPRequestHandler,
    response: requests.Response,
    *,
    request_id: str,
    keepalive_seconds: int,
    write_lock: threading.Lock,
    visible_status: dict[str, object] | None = None,
    visible_notice_after_seconds: int | None = None,
    passthrough_visible_chars: int = 0,
    passthrough_tool_calls: bool = False,
    loop_guard: dict[str, object] | None = None,
    cancel_check=None,
    thinking_budget_tokens: int | None = None,
) -> tuple[list[bytes], bool, dict[str, object]]:
    buffered_lines: list[bytes] = []
    stop_heartbeat = threading.Event()
    suppress_visible_notice = threading.Event()
    started_at = time.monotonic()
    state: dict[str, object] = {
        "visible_content_len": 0,
        "reasoning_len": 0,
        "thinking_budget_tokens": max(0, int(thinking_budget_tokens or 0)),
        "tool_call_chunks": 0,
        "tool_argument_chars": 0,
        "tool_names": [],
        "finish_reason": "",
        "finish_reasons_seen": [],
        "passthrough_reason": "",
        "tool_call_indices": [],
        "tool_argument_lengths_by_index": {},
        "tool_argument_json_valid_by_index": {},
        "tool_argument_tail_by_index": {},
        "buffer_abort_reason": "",
        "_loop_text_tail": "",
    }
    tool_argument_parts_by_index: dict[str, list[str]] = {}

    def _update_tool_argument_diagnostics(index_key: str) -> None:
        joined = "".join(tool_argument_parts_by_index.get(index_key) or [])
        lengths = state.get("tool_argument_lengths_by_index")
        if not isinstance(lengths, dict):
            lengths = {}
            state["tool_argument_lengths_by_index"] = lengths
        valid = state.get("tool_argument_json_valid_by_index")
        if not isinstance(valid, dict):
            valid = {}
            state["tool_argument_json_valid_by_index"] = valid
        tails = state.get("tool_argument_tail_by_index")
        if not isinstance(tails, dict):
            tails = {}
            state["tool_argument_tail_by_index"] = tails
        lengths[index_key] = len(joined)
        try:
            json.loads(joined)
            valid[index_key] = True
        except Exception:
            valid[index_key] = False
        tails[index_key] = joined[-160:]

    def observe_line(line: bytes) -> None:
        if not line or not line.startswith(b"data: ") or line == b"data: [DONE]":
            return
        try:
            chunk_data = json.loads(line[6:].decode("utf-8", errors="ignore"))
        except Exception:
            return
        choice0 = (chunk_data.get("choices") or [{}])[0]
        delta = choice0.get("delta", {}) or {}
        content = delta.get("content", "") or ""
        reasoning = delta.get("reasoning_content", "")
        if reasoning in (None, ""):
            reasoning = delta.get("reasoning", "")
        if reasoning in (None, ""):
            reasoning = delta.get("thinking", "")
        tool_calls = delta.get("tool_calls") or []
        if content:
            content_text = str(content)
            state["visible_content_len"] = int(state.get("visible_content_len") or 0) + len(content_text)
            tail = str(state.get("_loop_text_tail") or "") + content_text
            state["_loop_text_tail"] = tail[-24000:]
        if reasoning:
            reasoning_text = str(reasoning)
            state["reasoning_len"] = int(state.get("reasoning_len") or 0) + len(reasoning_text)
            tail = str(state.get("_loop_text_tail") or "") + reasoning_text
            state["_loop_text_tail"] = tail[-24000:]
        if choice0.get("finish_reason"):
            finish_reason = str(choice0.get("finish_reason") or "")
            state["finish_reason"] = finish_reason
            seen = state.get("finish_reasons_seen")
            if not isinstance(seen, list):
                seen = []
                state["finish_reasons_seen"] = seen
            if finish_reason and finish_reason not in seen:
                seen.append(finish_reason)
        if tool_calls:
            state["tool_call_chunks"] = int(state.get("tool_call_chunks") or 0) + (len(tool_calls) if isinstance(tool_calls, list) else 1)
            names = state.get("tool_names")
            if not isinstance(names, list):
                names = []
                state["tool_names"] = names
            for tool_call in tool_calls if isinstance(tool_calls, list) else [tool_calls]:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") or {}
                if not isinstance(function, dict):
                    continue
                name = str(function.get("name") or "").strip()
                if name and name not in names:
                    names.append(name)
                index_key = str(tool_call.get("index") if tool_call.get("index") is not None else tool_call.get("id") or "0")
                indices = state.get("tool_call_indices")
                if not isinstance(indices, list):
                    indices = []
                    state["tool_call_indices"] = indices
                if index_key not in indices:
                    indices.append(index_key)
                arguments = function.get("arguments") or ""
                if arguments:
                    arg_text = str(arguments)
                    state["tool_argument_chars"] = int(state.get("tool_argument_chars") or 0) + len(arg_text)
                    tool_argument_parts_by_index.setdefault(index_key, []).append(arg_text)
                    _update_tool_argument_diagnostics(index_key)

    def _forward_reasoning_line(line: bytes) -> bool:
        # Streaming pure-reasoning deltas straight through keeps slow-thinking
        # models from tripping client stale-timeouts before the first token.
        if not line or not line.startswith(b"data: ") or line == b"data: [DONE]":
            return False
        try:
            chunk_data = json.loads(line[6:].decode("utf-8", errors="ignore"))
        except Exception:
            return False
        choice0 = (chunk_data.get("choices") or [{}])[0]
        delta = choice0.get("delta", {}) or {}
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thinking")
        if not reasoning or delta.get("content") or delta.get("tool_calls") or choice0.get("finish_reason"):
            return False
        with write_lock:
            write_sse_line(line)
            handler.wfile.flush()
        observe_line(line)
        return True

    def should_passthrough() -> bool:
        if int(state.get("tool_call_chunks") or 0) > 0:
            if passthrough_tool_calls:
                state["passthrough_reason"] = "tool_call_seen"
                return True
            return False
        if passthrough_visible_chars > 0 and int(state.get("visible_content_len") or 0) >= passthrough_visible_chars:
            state["passthrough_reason"] = "visible_content_threshold"
            return True
        return False

    def write_sse_line(line: bytes) -> None:
        handler.wfile.write(line + b"\n\n" if line else b"\n")

    def cancelled_reason() -> str:
        if cancel_check is None:
            return ""
        try:
            value = cancel_check()
        except Exception as exc:
            log_api_event("openai_chat_stream_cancel_check_error", {"request_id": request_id, "error": str(exc)})
            return ""
        if isinstance(value, tuple):
            return str(value[1] or "") if value[0] else ""
        return "cancelled" if value else ""

    def send_visible_notice() -> None:
        if not visible_status:
            return
        try:
            delay = max(0, int(visible_notice_after_seconds if visible_notice_after_seconds is not None else 0))
        except Exception:
            delay = 0
        if stop_heartbeat.wait(delay):
            return
        cancel_reason = cancelled_reason()
        if cancel_reason:
            state["buffer_abort_reason"] = cancel_reason
            stop_heartbeat.set()
            try:
                response.close()
            except Exception:
                pass
            return
        if suppress_visible_notice.is_set() or int(state.get("tool_call_chunks") or 0) > 0:
            tool_chunks = int(state.get("tool_call_chunks") or 0)
            suppress_reason = "tool_call_started" if tool_chunks > 0 else "passthrough_started"
            log_api_event(
                "openai_chat_tool_continue_repair_user_notice_suppressed",
                {
                    "request_id": request_id,
                    "model": visible_status.get("model") if visible_status else "",
                    "upstream_model": visible_status.get("upstream_model") if visible_status else "",
                    "round": visible_status.get("round") if visible_status else None,
                    "trigger_reason": visible_status.get("trigger_reason") if visible_status else "",
                    "reason": suppress_reason,
                    "passthrough_reason": state.get("passthrough_reason") or "",
                    "tool_call_chunks": tool_chunks,
                    "visible_content_len": state.get("visible_content_len") or 0,
                },
            )
            return
        try:
            with write_lock:
                notice = str(visible_status.get("content") or "").strip()
                if notice:
                    # Emit the status as thinking text, not as normal assistant text.
                    # Some clients hide <think> blocks, and we also strip this marker
                    # from future inbound history before forwarding to the model.
                    _send_openai_chat_sse_status(
                        handler,
                        request_id=str(visible_status.get("request_id") or request_id),
                        model=str(visible_status.get("model") or ""),
                        content=f"\n<think>\n{notice}\n</think>\n",
                    )
                log_api_event(
                    "openai_chat_tool_continue_repair_user_notice",
                    {
                        "request_id": request_id,
                        "model": visible_status.get("model") or "",
                        "upstream_model": visible_status.get("upstream_model") or "",
                        "round": visible_status.get("round"),
                        "max_rounds": visible_status.get("max_rounds"),
                        "trigger_reason": visible_status.get("trigger_reason") or "",
                        "delay_seconds": delay,
                        "elapsed_ms": int((time.monotonic() - started_at) * 1000),
                    },
                )
        except Exception:
            stop_heartbeat.set()
            try:
                response.close()
            except Exception:
                pass

    def send_pulse() -> None:
        while not stop_heartbeat.is_set():
            if stop_heartbeat.wait(max(1, keepalive_seconds)):
                break
            try:
                cancel_reason = cancelled_reason()
                if cancel_reason:
                    state["buffer_abort_reason"] = cancel_reason
                    stop_heartbeat.set()
                    try:
                        response.close()
                    except Exception:
                        pass
                    break
                with write_lock:
                    log_api_event(
                        "openai_chat_stream_repair_buffer_pulse",
                        {
                            "request_id": request_id,
                            "elapsed_ms": int((time.monotonic() - started_at) * 1000),
                            "visible_content_len": state.get("visible_content_len") or 0,
                            "reasoning_len": state.get("reasoning_len") or 0,
                            "tool_call_chunks": state.get("tool_call_chunks") or 0,
                            "tool_argument_chars": state.get("tool_argument_chars") or 0,
                            "finish_reason": state.get("finish_reason") or "",
                            "buffer_abort_reason": state.get("buffer_abort_reason") or "",
                        },
                    )
                    handler.wfile.write(b": chat-tool-continue-repair-buffering\n\n")
                    handler.wfile.flush()
            except Exception:
                stop_heartbeat.set()
                try:
                    response.close()
                except Exception:
                    pass
                break

    heartbeat_thread = threading.Thread(target=send_pulse, daemon=True)
    notice_thread = threading.Thread(target=send_visible_notice, daemon=True)
    heartbeat_thread.start()
    notice_thread.start()
    passthrough = False
    try:
        for line in response.iter_lines():
            cancel_reason = cancelled_reason()
            if cancel_reason:
                state["buffer_abort_reason"] = cancel_reason
                try:
                    response.close()
                except Exception:
                    pass
                break
            if stop_heartbeat.is_set():
                if state.get("buffer_abort_reason"):
                    break
                raise BrokenPipeError("client disconnected while buffering chat repair stream")
            if not _forward_reasoning_line(line):
                buffered_lines.append(line)
                observe_line(line)
            loop_reason = _chat_tool_continue_loop_guard_reason(state, loop_guard)
            if loop_reason:
                state["buffer_abort_reason"] = loop_reason
                state.pop("_loop_text_tail", None)
                try:
                    response.close()
                except Exception:
                    pass
                break
            if should_passthrough():
                passthrough = True
                suppress_visible_notice.set()
                state.pop("_loop_text_tail", None)
                with write_lock:
                    for buffered in buffered_lines:
                        write_sse_line(buffered)
                    handler.wfile.flush()
                buffered_lines = []
                log_api_event(
                    "openai_chat_stream_repair_passthrough",
                    {
                        "request_id": request_id,
                        "reason": state.get("passthrough_reason") or "",
                        "visible_content_len": state.get("visible_content_len") or 0,
                        "tool_call_chunks": state.get("tool_call_chunks") or 0,
                        "tool_argument_chars": state.get("tool_argument_chars") or 0,
                        "tool_names": state.get("tool_names") or [],
                        "finish_reasons_seen": state.get("finish_reasons_seen") or [],
                        "tool_call_indices": state.get("tool_call_indices") or [],
                        "tool_argument_lengths_by_index": state.get("tool_argument_lengths_by_index") or {},
                        "tool_argument_json_valid_by_index": state.get("tool_argument_json_valid_by_index") or {},
                        "tool_argument_tail_by_index": state.get("tool_argument_tail_by_index") or {},
                    },
                )
                break
        if passthrough:
            for line in response.iter_lines():
                cancel_reason = cancelled_reason()
                if cancel_reason:
                    state["buffer_abort_reason"] = cancel_reason
                    try:
                        response.close()
                    except Exception:
                        pass
                    break
                if stop_heartbeat.is_set():
                    if state.get("buffer_abort_reason"):
                        break
                    raise BrokenPipeError("client disconnected while passthrough streaming chat repair")
                observe_line(line)
                with write_lock:
                    write_sse_line(line)
                    handler.wfile.flush()
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1.0)
        notice_thread.join(timeout=1.0)
        state.pop("_loop_text_tail", None)
    return buffered_lines, passthrough, state


def resolve_catalog_model_name(raw_name: str, catalog: list[ManagedModel]) -> str:
    name = (raw_name or "").strip()
    if not name:
        return name

    def _match_catalog(candidate: str) -> str | None:
        exact = next((item.model_id for item in catalog if item.model_id == candidate), None)
        if exact:
            return exact
        alias_match = next((item.model_id for item in catalog if candidate in item.aliases), None)
        if alias_match:
            return alias_match
        for catalog_item in catalog:
            if candidate in model_name_aliases(catalog_item):
                return catalog_item.model_id
        try:
            repo_id, quant = parse_hf_input(candidate)
        except Exception:
            return None
        for catalog_item in catalog:
            if catalog_item.repo_id == repo_id and ((catalog_item.quant or "") == (quant or "")):
                return catalog_item.model_id
        return None

    if "/" in name:
        provider_prefix, remainder = name.split("/", 1)
        if provider_prefix and remainder:
            prefixed_match = _match_catalog(remainder)
            if prefixed_match:
                return prefixed_match
    direct_match = _match_catalog(name)
    if direct_match:
        return direct_match
    for candidate in _precision_suffix_alias_candidates(name):
        suffix_match = _match_catalog(candidate)
        if suffix_match:
            return suffix_match
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
    return requests.request(method, url, data=body, headers=proxy_headers, timeout=(60, 600), stream=False)


def _proxy_headers_safe(headers: dict) -> dict:
    return {
        k: v for k, v in headers.items()
        if k.lower() not in {"host", "connection", "content-length", "accept-encoding"}
    }


def is_llamaswap_upstream_static_autoload_path(method: str, path: str) -> bool:
    """Return True for browser static-asset GETs that would otherwise autoload /upstream models."""
    if str(method or "").upper() not in {"GET", "HEAD"}:
        return False
    parsed = urlparse(str(path or ""))
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 3 or parts[0] != "upstream":
        return False
    # /upstream/<model>/health is an intentional load/touch path used by the
    # manager; only block static browser/UI assets below the upstream prefix.
    remainder = "/" + "/".join(parts[2:])
    basename = parts[-1].lower()
    suffix = Path(basename).suffix.lower()
    return (
        basename in LLAMASWAP_UPSTREAM_STATIC_BLOCKED_BASENAMES
        or suffix in LLAMASWAP_UPSTREAM_STATIC_BLOCKED_EXTENSIONS
        or remainder.startswith("/assets/")
        or remainder.startswith("/static/")
    )


def _next_llamaswap_guard_backend_port(public_port: int) -> int:
    raw = os.environ.get("LLAMASWAP_GUARD_BACKEND_PORT")
    if raw:
        try:
            return int(raw)
        except Exception:
            pass
    candidate = int(public_port) + 10000
    if candidate <= 65535:
        return candidate
    return int(public_port) + 1000


def run_llamaswap_guard(args) -> int:
    """Run llama-swap behind a small guard proxy that blocks upstream static autoloads."""
    listen_host = str(getattr(args, "listen_host", None) or getattr(args, "public_host", None) or DEFAULT_PUBLIC_HOST)
    listen_port = int(getattr(args, "listen_port", None) or getattr(args, "public_port", None) or DEFAULT_PUBLIC_PORT)
    backend_host = "127.0.0.1"
    backend_port = int(getattr(args, "backend_port", None) or _next_llamaswap_guard_backend_port(listen_port))
    llamaswap_bin = Path(getattr(args, "llamaswap_bin", None) or os.environ.get("LLAMASWAP_BIN", "llama-swap"))
    config_path = Path(getattr(args, "config", None) or os.environ.get("HEIMDALL_GATEWAY_CONFIG", _env_value("HEIMDALL_GATEWAY_CONFIG", "LLAMACPP_CONFIG", str(DEFAULT_CONFIG_PATH))))
    child_cmd = [
        str(llamaswap_bin),
        "--config",
        str(config_path),
        "--listen",
        f"{backend_host}:{backend_port}",
        "--watch-config",
    ]
    child = subprocess.Popen(child_cmd)

    class GuardHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *values):
            return

        def _blocked(self) -> bool:
            if not is_llamaswap_upstream_static_autoload_path(self.command, self.path):
                return False
            log_api_event(
                "llamaswap_upstream_static_autoload_blocked",
                {"method": self.command, "path": self.path},
            )
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return True

        def _proxy(self):
            if self._blocked():
                return
            length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(length) if length > 0 else None
            target_url = f"http://{backend_host}:{backend_port}{self.path}"
            try:
                upstream = requests.request(
                    self.command,
                    target_url,
                    headers=_proxy_headers_safe(dict(self.headers)),
                    data=body,
                    stream=True,
                    timeout=(5, None),
                )
            except Exception as exc:
                message = f"llama-swap guard backend error: {exc}"
                log_api_event("llamaswap_guard_backend_error", {"method": self.command, "path": self.path, "error": str(exc)})
                data = message.encode("utf-8", errors="replace")
                self.send_response(502)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_response(upstream.status_code)
            for key, value in upstream.headers.items():
                lowered = key.lower()
                if lowered in {"connection", "transfer-encoding", "content-encoding", "content-length"}:
                    continue
                self.send_header(key, value)
            
            # Crucial fix: If upstream is chunked, we MUST either send chunked encoding ourselves 
            # or close the connection so the client knows when the response ends. 
            # The safest and easiest is to disable keep-alive for proxied streaming responses.
            if upstream.headers.get("Transfer-Encoding", "").lower() == "chunked":
                self.send_header("Connection", "close")
                
            if upstream.headers.get("Content-Length") is not None:
                content = upstream.content
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(content)
                return
            self.end_headers()
            if self.command == "HEAD":
                return
            for chunk in upstream.iter_content(chunk_size=None):
                if chunk:
                    self.wfile.write(chunk)
                    self.wfile.flush()

        def do_GET(self): self._proxy()
        def do_HEAD(self): self._proxy()
        def do_POST(self): self._proxy()
        def do_PUT(self): self._proxy()
        def do_PATCH(self): self._proxy()
        def do_DELETE(self): self._proxy()
        def do_OPTIONS(self): self._proxy()

    server = ThreadingHTTPServer((listen_host, listen_port), GuardHandler)
    server.daemon_threads = True
    stop_event = threading.Event()

    def _request_server_shutdown():
        try:
            server.shutdown()
        except Exception:
            pass

    def _shutdown(_signum=None, _frame=None):
        stop_event.set()
        # BaseServer.shutdown() must not run in the same thread that is inside
        # serve_forever(); signal handlers run on the main thread, so dispatch
        # the actual shutdown call to a helper thread to avoid systemd stop
        # hanging in "deactivating".
        threading.Thread(target=_request_server_shutdown, daemon=True).start()
        if child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    def _watch_child():
        rc = child.wait()
        if not stop_event.is_set():
            log_api_event("llamaswap_guard_child_exited", {"returncode": rc})
            _request_server_shutdown()

    threading.Thread(target=_watch_child, daemon=True).start()
    print(
        f"llama-swap guard listening on {listen_host}:{listen_port}; "
        f"backend {backend_host}:{backend_port}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        _shutdown()
        try:
            child.wait(timeout=10)
        except Exception:
            child.kill()
    return child.returncode or 0


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
                                {
                                    "model": model_id,
                                    "activity_age_seconds": age,
                                    "idle_ttl": idle_ttl,
                                    "last_activity": activity.get(model_id),
                                    "last_activity_model_id": last_activity_model_id,
                                },
                            )
                            log_model_runtime_snapshot(
                                "model_unload_observed_runtime",
                                model_id,
                                catalog,
                                host,
                                port,
                            )
                            continue
                        log_api_event(
                            "model_unexpected_unload",
                            {
                                "model": model_id,
                                "activity_age_seconds": age,
                                "idle_ttl": idle_ttl,
                                "last_activity": activity.get(model_id),
                                "last_activity_model_id": last_activity_model_id,
                            },
                        )
                        log_model_runtime_snapshot(
                            "model_unexpected_unload_runtime_before_reload",
                            model_id,
                            catalog,
                            host,
                            port,
                        )
                        if _touch_model_via_llamaswap(model_id, host, port):
                            state["loaded"] = get_loaded_catalog_model_ids(catalog)
                            log_model_runtime_snapshot(
                                "model_unexpected_unload_runtime_after_reload",
                                model_id,
                                catalog,
                                host,
                                port,
                            )
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


def _candidate_request_log_paths(explicit_path: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    
    # Check current user's local log
    user_log = Path.home() / ".local/state/heimdall-gateway/api-requests.log"
    if user_log not in candidates:
        candidates.append(user_log)
        
    # Check system-wide log
    if SYSTEM_REQUESTS_LOG_PATH not in candidates:
        candidates.append(SYSTEM_REQUESTS_LOG_PATH)
        
    # Check DEFAULT (which might be one of the above or from env)
    if DEFAULT_REQUESTS_LOG_PATH not in candidates:
        candidates.append(DEFAULT_REQUESTS_LOG_PATH)

    env_path = _env_value("HEIMDALL_GATEWAY_REQUESTS_LOG", "HEIMDALL_GATEWAY_REQUESTS_LOG", "")
    if env_path:
        path = Path(env_path).expanduser()
        if path not in candidates:
            candidates.insert(0, path)
    return candidates


def get_config_model_port_map(config_path: Path | str | None) -> dict[str, int]:
    if config_path is None:
        return {}
    try:
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    try:
        start_port = int(payload.get("startPort") or DEFAULT_START_PORT)
    except Exception:
        start_port = DEFAULT_START_PORT
    models = payload.get("models") or {}
    if not isinstance(models, dict):
        return {}
    return {str(model_id): start_port + idx for idx, model_id in enumerate(models.keys())}




def _find_llama_server_process_for_replica(
    model: ManagedModel,
    gpu_set: list[int],
    processes: list[dict],
    gpu_mem_by_pid: dict[int, dict[int, float]],
    preferred_port: int | None = None,
) -> dict | None:
    expected_path = _safe_realpath(model.local_path)
    expected_gpus = set(gpu_set)
    if preferred_port is not None:
        for proc in processes:
            if proc.get("port") != preferred_port:
                continue
            if proc.get("model_path") != expected_path:
                continue
            pid = int(proc.get("pid") or 0)
            actual_gpus = set(gpu_mem_by_pid.get(pid, {}).keys())
            if not expected_gpus or actual_gpus == expected_gpus:
                return proc
    for proc in processes:
        if proc.get("model_path") != expected_path:
            continue
        pid = int(proc.get("pid") or 0)
        actual_gpus = set(gpu_mem_by_pid.get(pid, {}).keys())
        if actual_gpus == expected_gpus:
            return proc
    return None

def sync_replica_runtime_state(
    catalog: list[ManagedModel],
    config_path: Path | str | None = None,
    global_replica_config: dict[str, object] | None = None,
) -> None:
    port_by_model = get_config_model_port_map(config_path)
    model_by_port = {port: model_id for model_id, port in port_by_model.items()}
    total_gpus = detect_cuda_device_count()
    processes = get_llama_server_processes()
    proc_by_port = {proc.get("port"): proc for proc in processes if proc.get("port") is not None}
    gpu_mem_by_pid = get_gpu_process_memory_by_pid()
    now = time.monotonic()
    with REPLICA_ROUTER_STATE.lock:
        for model in catalog:
            cfg = get_model_replica_config(model, global_replica_config, total_gpus=total_gpus)
            if not cfg.enabled:
                continue
            for idx, gpu_set in enumerate(_replica_gpu_sets(model, cfg, total_gpus=total_gpus)):
                rid = replica_model_id(model.model_id, idx)
                rec = REPLICA_ROUTER_STATE.records.setdefault(
                    rid,
                    ReplicaRecord(base_model_id=model.model_id, replica_model_id=rid, gpu_set=list(gpu_set)),
                )
                rec.gpu_set = list(gpu_set)
                configured_port = port_by_model.get(rid)
                proc = _find_llama_server_process_for_replica(
                    model,
                    list(gpu_set),
                    processes,
                    gpu_mem_by_pid,
                    preferred_port=configured_port,
                )
                if proc is not None:
                    pid = int(proc["pid"])
                    rec.pid = pid
                    rec.port = int(proc.get("port") or configured_port or 0) or configured_port
                    rec.gpu_actual_mib = dict(gpu_mem_by_pid.get(pid, {}))
                    rec.actual_mib = float(sum(rec.gpu_actual_mib.values())) if rec.gpu_actual_mib else None
                    if rec.status in {"cold", "loading", "error"}:
                        rec.status = "ready"
                    rec.last_used = rec.last_used or now
                else:
                    rec.pid = None
                    rec.port = configured_port
                    rec.gpu_actual_mib = {}
                    rec.actual_mib = None
                    if rec.status == "ready":
                        rec.status = "cold"


def replica_router_snapshot(catalog: list[ManagedModel], config_path: Path | str | None = None, args = None) -> dict[str, object]:
    global_replica_config = resolve_global_replica_config(args)
    sync_replica_runtime_state(catalog, config_path, global_replica_config)
    if args is not None:
        published = sorted(
            get_published_model_ids(
                host=getattr(args, "public_host", DEFAULT_PUBLIC_HOST),
                port=int(getattr(args, "public_port", DEFAULT_PUBLIC_PORT)),
            )
        )
    else:
        published = sorted(get_published_model_ids())
    configured = []
    skipped = []
    total_gpus = detect_cuda_device_count()
    for model in catalog:
        cfg = get_model_replica_config(model, global_replica_config, total_gpus=total_gpus)
        if not cfg.enabled:
            skipped.append({"model": model.model_id, "reason": "replicas_disabled", "tensor_split": model.tensor_split})
            continue
        gpu_sets = _replica_gpu_sets(model, cfg, total_gpus=total_gpus)
        if not gpu_sets:
            skipped.append({
                "model": model.model_id,
                "reason": "no_gpu_sets",
                "tensor_split": model.tensor_split,
                "max": cfg.max,
                "gpus_per_replica": cfg.gpus_per_replica,
                "total_gpus": total_gpus,
            })
            continue
        configured.append({
            "model": model.model_id,
            "enabled": cfg.enabled,
            "max": cfg.max,
            "gpus_per_replica": cfg.gpus_per_replica,
            "placement": cfg.placement,
            "tensor_split": model.tensor_split,
            "gpu_sets": gpu_sets,
            "replica_ids": [replica_model_id(model.model_id, idx) for idx in range(len(gpu_sets))],
        })
    with REPLICA_ROUTER_STATE.lock:
        records = [
            {
                "replica": rec.replica_model_id,
                "base_model": rec.base_model_id,
                "gpu_set": rec.gpu_set,
                "status": rec.status,
                "in_flight": rec.in_flight,
                "estimated_mib": rec.estimated_mib,
                "actual_mib": rec.actual_mib,
                "gpu_actual_mib": rec.gpu_actual_mib,
                "pid": rec.pid,
                "port": rec.port,
                "blacklist_until": rec.blacklist_until,
                "published": rec.replica_model_id in published,
            }
            for rec in sorted(REPLICA_ROUTER_STATE.records.values(), key=lambda r: r.replica_model_id)
        ]
        affinities = [
            {"key": key, "replica": value[0], "expires_in_s": max(0.0, value[1] - time.monotonic())}
            for key, value in sorted(REPLICA_ROUTER_STATE.affinity.items())
        ]
    diagnostics = {
        "server_config_path": str(_args_server_config_path(args)),
        "config_path": str(config_path) if config_path is not None else "",
        "catalog_path": str(getattr(args, "catalog", "")) if args is not None else "",
        "catalog_count": len(catalog),
        "total_gpus": total_gpus,
        "replicas_config": global_replica_config,
        "replicas_enabled_raw": global_replica_config.get("enabled") if isinstance(global_replica_config, dict) else None,
        "configured_count": len(configured),
        "skipped_sample": skipped[:20],
    }
    return {"diagnostics": diagnostics, "configured": configured, "records": records, "affinities": affinities, "published_model_ids": published}


def resolve_api_auth_config(args = None) -> dict[str, object]:
    return _normalize_api_auth_config(_load_server_config_payload(args).get("api_auth"))


def resolve_api_https_config(args = None) -> dict[str, object]:
    return _normalize_api_https_config(_load_server_config_payload(args).get("api_https"))


def _api_auth_matches(headers, expected_key: str) -> bool:
    if not expected_key:
        return False
    try:
        auth = str(headers.get("Authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
            if hmac.compare_digest(token, expected_key):
                return True
        api_key = str(headers.get("X-API-Key") or headers.get("x-api-key") or "").strip()
        if api_key and hmac.compare_digest(api_key, expected_key):
            return True
    except Exception:
        return False
    return False


def _is_loopback_client(host: str) -> bool:
    value = str(host or "").strip()
    if value in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _make_tls_context(https_cfg: dict[str, object]) -> ssl.SSLContext:
    cert_file = str(https_cfg.get("cert_file") or "").strip()
    key_file = str(https_cfg.get("key_file") or "").strip()
    if not cert_file or not key_file:
        raise RuntimeError("api_https.enabled=true requires api_https.cert_file and api_https.key_file")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=cert_file, keyfile=key_file)
    return ctx


def _should_redirect_plain_http_to_https(*, https_enabled: bool, client_host: str, connection: object) -> bool:
    if not https_enabled:
        return False
    if isinstance(connection, ssl.SSLSocket):
        return False
    # Local API clients such as Codex Desktop often use http://127.0.0.1
    # intentionally.  Keep loopback plaintext usable even when the network API
    # is HTTPS-enabled; remote plaintext clients are still redirected.
    if _is_loopback_client(client_host):
        return False
    return True


class HTTPSRedirectingThreadingHTTPServer(ThreadingHTTPServer):
    """Serve HTTPS and redirect plain HTTP clients on the same API port.

    A normal ssl.wrap_socket(server.socket) rejects HTTP plaintext before the
    request reaches BaseHTTPRequestHandler, causing a connection reset. This
    server peeks the first byte: TLS handshakes start with 0x16; plaintext HTTP
    methods start with ASCII letters. Plain HTTP gets a small 301 response to
    the same host/path with https://.
    """

    def __init__(self, server_address, RequestHandlerClass, https_cfg: dict[str, object]):
        super().__init__(server_address, RequestHandlerClass)
        self.tls_context = _make_tls_context(https_cfg)

    def get_request(self):
        while True:
            sock, addr = self.socket.accept()
            try:
                first = sock.recv(1, socket.MSG_PEEK)
            except Exception:
                sock.close()
                continue
            if first == b"\x16":
                try:
                    return self.tls_context.wrap_socket(sock, server_side=True), addr
                except ssl.SSLError:
                    sock.close()
                    continue
            return sock, addr


def _wrap_http_server_with_tls(server: ThreadingHTTPServer, https_cfg: dict[str, object]) -> None:
    # Kept for backwards-compatible tests/helpers. Production HTTPS uses
    # HTTPSRedirectingThreadingHTTPServer so plaintext HTTP can be redirected.
    server.socket = _make_tls_context(https_cfg).wrap_socket(server.socket, server_side=True)


def start_ctx_metadata_server(args):
    bind_host = args.public_host
    client_host = _normalize_client_host(args.public_host)
    port = resolve_api_port(args)
    catalog_path = Path(args.catalog)
    api_auth = resolve_api_auth_config(args)
    api_https = resolve_api_https_config(args)
    api_auth_enabled = bool(api_auth.get("enabled"))
    api_key = str(api_auth.get("api_key") or "").strip()

    class Handler(BaseHTTPRequestHandler):
        def _redirect_plain_http_to_https(self) -> bool:
            try:
                client_addr = self.client_address[0] if self.client_address else ""
                if not _should_redirect_plain_http_to_https(
                    https_enabled=bool(api_https.get("enabled")),
                    client_host=client_addr,
                    connection=self.connection,
                ):
                    return False
                host = self.headers.get("Host") or f"{client_host}:{port}"
                location = f"https://{host}{self.path or '/'}"
                log_api_event("api_http_redirect", {"path": self.path, "location": location, "client": client_addr})
                encoded = (
                    "HTTP/1.1 308 Permanent Redirect\r\n"
                    f"Location: {location}\r\n"
                    "Connection: close\r\n"
                    "Content-Length: 0\r\n"
                    "\r\n"
                ).encode("utf-8")
                self.connection.sendall(encoded)
            except Exception:
                pass
            return True

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


        def _send_html(self, body: str, status: int = 200):
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError, OSError):
                try:
                    self.connection.shutdown(2)
                except Exception:
                    pass

        def _is_local_request(self) -> bool:
            client = self.client_address[0] if self.client_address else ""
            return _is_loopback_client(client)

        def _handle_root(self, catalog: list[ManagedModel]):
            scheme = "https" if bool(api_https.get("enabled")) else "http"
            host = self.headers.get("Host") or f"{client_host}:{port}"
            base_url = f"{scheme}://{host}"
            if not self._is_local_request():
                self._send_json({
                    "service": "heimdall-gateway",
                    "api": "openai-compatible",
                    "models_endpoint": "/v1/models",
                    "auth_required": api_auth_enabled,
                })
                return
            server_config_path = str(getattr(args, "server_config", DEFAULT_SERVER_CONFIG_PATH))
            config_path = str(getattr(args, "config", ""))
            catalog_file = str(getattr(args, "catalog", ""))
            api_key_row = "disabled"
            curl_auth = ""
            if api_auth_enabled:
                escaped_key = html.escape(api_key)
                api_key_row = f"<code>{escaped_key}</code>" if api_key else "enabled but key is empty"
                curl_auth = f" -H 'Authorization: Bearer {escaped_key}'" if api_key else ""
            https_row = "disabled"
            if bool(api_https.get("enabled")):
                https_row = (
                    "enabled "
                    f"<br><small>cert: <code>{html.escape(str(api_https.get('cert_file') or ''))}</code></small>"
                    f"<br><small>key: <code>{html.escape(str(api_https.get('key_file') or ''))}</code></small>"
                )
            endpoints = [
                ("OpenAI models", "/v1/models"),
                ("Ollama tags", "/api/tags"),
                ("Loaded models", "/api/ps"),
                ("Context info", "/api/ctx"),
                ("Replica diagnostics", "/api/replicas"),
                ("Version", "/api/version"),
            ]
            endpoint_items = "".join(
                f"<li><a href='{html.escape(path)}'>{html.escape(label)}</a> <code>{html.escape(path)}</code></li>"
                for label, path in endpoints
            )
            escaped_base = html.escape(base_url)
            body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Heimdall Gateway API</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 980px; margin: 32px auto; padding: 0 18px; line-height: 1.45; }}
    code, pre {{ background: #f4f4f5; border-radius: 6px; padding: 2px 5px; }}
    pre {{ padding: 12px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #fafafa; }}
    .warn {{ color: #9a3412; }}
  </style>
</head>
<body>
  <h1>Heimdall Gateway API</h1>
  <p>Esta pagina muestra informacion sensible y solo se sirve completa a clientes locales/loopback.</p>
  <table>
    <tr><th>Base URL</th><td><code>{escaped_base}</code></td></tr>
    <tr><th>Modelos configurados</th><td>{len(catalog)}</td></tr>
    <tr><th>API key remota</th><td>{api_key_row}<br><small>El acceso local desde 127.0.0.1/::1 no requiere API key.</small></td></tr>
    <tr><th>HTTPS</th><td>{https_row}</td></tr>
    <tr><th>conf.json</th><td><code>{html.escape(server_config_path)}</code></td></tr>
    <tr><th>config.yaml</th><td><code>{html.escape(config_path)}</code></td></tr>
    <tr><th>catalog.json</th><td><code>{html.escape(catalog_file)}</code></td></tr>
  </table>
  <h2>Endpoints</h2>
  <ul>{endpoint_items}</ul>
  <h2>Ejemplos</h2>
  <pre>curl {curl_auth} {escaped_base}/v1/models</pre>
  <pre>curl {curl_auth} {escaped_base}/api/tags</pre>
  <p class="warn">Nota: un certificado self-signed debe ser confiado manualmente por clientes remotos si quieres evitar avisos TLS.</p>
</body>
</html>"""
            self._send_html(body)

        def log_message(self, format, *args):
            return

        def _check_auth(self, parsed) -> bool:
            if not api_auth_enabled:
                return True
            # Local loopback access is trusted so a local browser or health check
            # can inspect the API even when the network-facing API requires a key.
            client_host = self.client_address[0] if self.client_address else ""
            if _is_loopback_client(client_host):
                return True
            # Keep the cheap health/version endpoints available for service checks.
            if parsed.path in {"/api/version"}:
                return True
            if _api_auth_matches(self.headers, api_key):
                return True
            log_api_event("api_auth_failed", {"path": parsed.path, "client": self.client_address[0] if self.client_address else ""})
            self._send_json({"error": {"message": "missing or invalid API key", "type": "authentication_error"}}, status=401)
            return False

        def do_GET(self):
            parsed = urlparse(self.path)
            if self._redirect_plain_http_to_https():
                return
            if not self._check_auth(parsed):
                return
            catalog = load_catalog(catalog_path)
            if parsed.path in {"", "/"}:
                return self._handle_root(catalog)
            if parsed.path in {"/v1/models", "/models"}:
                self._send_json({"object": "list", "data": [build_openai_model_list_payload(model) for model in catalog]})
                return
            model_lookup_id = None
            for prefix in ("/v1/models/", "/models/"):
                if parsed.path.startswith(prefix):
                    model_lookup_id = unquote(parsed.path[len(prefix):]).strip()
                    break
            if model_lookup_id:
                wanted = model_lookup_id.casefold()
                wanted_bare = wanted.rsplit("/", 1)[-1]
                for model in catalog:
                    mid = str(getattr(model, "model_id", "") or "")
                    folded = mid.casefold()
                    if folded == wanted or folded.rsplit("/", 1)[-1] == wanted_bare:
                        self._send_json(build_openai_model_list_payload(model))
                        return
                self._send_json({"error": {"message": f"model not found: {model_lookup_id}", "type": "not_found_error"}}, status=404)
                return
            if parsed.path == "/api/tags":
                published_models = get_published_model_ids(client_host, int(args.public_port))
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
                published_models = get_published_model_ids(client_host, int(args.public_port))
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
                self._send_json({"version": f"{get_heimdall_gateway_version()}-heimdall-gateway"})
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
            if parsed.path == "/api/replicas":
                self._send_json(replica_router_snapshot(catalog, args.config, args))
                return
            if parsed.path == "/api/show":
                name = parse_qs(parsed.query).get("name", [""])[0]
                return self._handle_show(name, catalog)
            if parsed.path.startswith("/v1/"):
                return self._proxy_request("GET")
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self):
            parsed = urlparse(self.path)
            if self._redirect_plain_http_to_https():
                return
            if not self._check_auth(parsed):
                return
            if parsed.path == "/api/chat":
                return self._handle_ollama_chat()
            if parsed.path == "/api/generate":
                return self._handle_ollama_generate()
            if parsed.path in {"/api/embed", "/api/embeddings"}:
                return self._handle_ollama_embeddings()
            if parsed.path in {"/v1/responses", "/responses"}:
                return self._handle_openai_responses()
            if parsed.path in {"/v1/chat/completions", "/chat/completions"}:
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
            upstream_model_name = ""
            is_replica_request = False
            proxy_payload = {}
            if body and method == "POST":
                try:
                    proxy_payload = json.loads(body.decode("utf-8"))
                    
                    # Use explicit CLI flag if provided, otherwise fall back to server config
                    flatten_enabled = True
                    if getattr(args, "flatten", None) is not None:
                        flatten_enabled = bool(args.flatten)
                    else:
                        try:
                            _scfg = _load_server_config_payload(args)
                            flatten_enabled = bool(_scfg.get("flatten_namespace_tools", True))
                        except Exception:
                            pass

                    if isinstance(proxy_payload, dict) and isinstance(proxy_payload.get("tools"), list) and parsed.path.rstrip("/") in {"/v1/chat/completions", "/chat/completions"}:
                        try:
                            if flatten_enabled:
                                proxy_payload["tools"] = _flatten_responses_tools(proxy_payload["tools"])
                                body = json.dumps(proxy_payload).encode("utf-8")
                        except Exception:
                            pass
                    if isinstance(proxy_payload, dict) and proxy_payload.get("model"):
                        catalog = load_catalog(catalog_path)
                        activity_model = resolve_catalog_model_name(str(proxy_payload.get("model") or ""), catalog)
                        if activity_model:
                            mark_model_activity(activity_model, f"proxy:{parsed.path}", "request_start")
                            model_entry = next((item for item in catalog if item.model_id == activity_model), None)
                            if model_entry is not None:
                                replica_defaults = resolve_global_replica_config(args)
                                published_model_ids = get_published_model_ids(client_host, int(args.public_port))
                                sync_replica_runtime_state(catalog, args.config, replica_defaults)
                                upstream_model_name, _, is_replica_request = select_replica_for_request(
                                    model_entry,
                                    proxy_payload,
                                    self.headers,
                                    replica_defaults,
                                    published_model_ids,
                                    catalog=catalog,
                                    config_path=args.config,
                                    server_path=args.llama_server,
                                    idle_ttl=resolve_idle_ttl(args),
                                    server_defaults=resolve_llama_server_defaults(args),
                                    public_host=client_host,
                                    public_port=int(args.public_port),
                                )
                                if upstream_model_name:
                                    if is_replica_request:
                                        proxy_payload["model"] = upstream_model_name
                                        body = json.dumps(proxy_payload).encode("utf-8")
                                    REPLICA_ROUTER_STATE.request_started(upstream_model_name)
                except Exception:
                    pass
            log_api_event("proxy_request", {"method": method, "path": parsed.path, "query": parsed.query, "body": body.decode("utf-8", errors="replace")[:4000] if body else ""})
            try:
                response = _proxy_request_to_public_api(
                    method,
                    parsed.path + (("?" + parsed.query) if parsed.query else ""),
                    body=body,
                    headers={key: value for key, value in self.headers.items()},
                    host=client_host,
                    port=int(args.public_port),
                )
            except requests.RequestException as exc:
                log_api_event("proxy_error", {"method": method, "path": parsed.path, "error": str(exc)})
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                self._send_json({"error": f"upstream unavailable: {exc}"}, status=502)
                return

            content = response.content
            log_api_event("proxy_response", {"method": method, "path": parsed.path, "status": response.status_code, "body": content.decode('utf-8', errors='replace')[:4000]})
            if upstream_model_name:
                REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=response.status_code < 400)
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
            payload = payload if isinstance(payload, dict) else {}
            
            # Use explicit CLI flag if provided, otherwise fall back to server config
            flatten_enabled = True
            if getattr(args, "flatten", None) is not None:
                flatten_enabled = bool(args.flatten)
            else:
                try:
                    _scfg = _load_server_config_payload(args)
                    flatten_enabled = bool(_scfg.get("flatten_namespace_tools", True))
                except Exception:
                    pass

            if isinstance(payload.get("tools"), list) and urlparse(self.path).path in {"/v1/chat/completions", "/chat/completions"}:
                try:
                    if flatten_enabled:
                        payload["tools"] = _flatten_responses_tools(payload["tools"])
                except Exception:
                    pass
            return payload

        def _proxy_raw_response(self, response: requests.Response):
            content = response.content
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                lowered = key.lower()
                if lowered in {"content-length", "connection", "transfer-encoding", "content-encoding"}:
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _reject_if_gpu_busy(self, model_name: str, catalog: list[ManagedModel], *, api_style: str, payload: dict | None = None) -> bool:
            target_loaded = get_catalog_model_process(model_name, catalog) is not None
            if (not target_loaded) and request_looks_like_model_probe(payload or {}):
                activity, _last_activity_model_id = get_model_activity_snapshot()
                blocker = recent_activity_blocking_model_switch(
                    model_name,
                    activity,
                    now=time.monotonic(),
                    grace_s=DEFAULT_MODEL_SWITCH_GRACE_S,
                )
                if blocker is not None:
                    active_model, age_s, phase = blocker
                    message = (
                        f"Refusing to autoload probe for model '{model_name}' while model "
                        f"'{active_model}' was active {age_s:.1f}s ago ({phase})."
                    )
                    log_api_event(
                        "model_probe_autoload_blocked",
                        {
                            "model": model_name,
                            "active_model": active_model,
                            "activity_age_seconds": age_s,
                            "phase": phase,
                            "grace_s": DEFAULT_MODEL_SWITCH_GRACE_S,
                            "api_style": api_style,
                        },
                    )
                    REPLICA_ROUTER_STATE.release_loading_claim(model_name)
                    if api_style == "openai":
                        self._send_json({"error": {"message": message, "type": "server_error"}}, status=503)
                    else:
                        self._send_json({"error": message}, status=503)
                    return True
            gpu_conflict = get_gpu_conflict_message(model_name, catalog, client_host, int(args.public_port))
            if not gpu_conflict:
                return False
            log_api_event("model_load_blocked_gpu_busy", {"model": model_name, "message": gpu_conflict, "api_style": api_style})
            REPLICA_ROUTER_STATE.release_loading_claim(model_name)
            if api_style == "openai":
                self._send_json({"error": {"message": gpu_conflict, "type": "server_error"}}, status=503)
            else:
                self._send_json({"error": gpu_conflict}, status=503)
            return True

        def _reject_if_model_loading(self, upstream_model_name: str, catalog: list[ManagedModel], *, public_model_name: str, is_replica: bool, api_style: str) -> bool:
            if not upstream_model_name:
                return False
            if REPLICA_ROUTER_STATE.claim_loading(
                upstream_model_name,
                catalog,
                is_replica=is_replica,
                claim_key=public_model_name,
            ):
                return False
            message = (
                f"Model '{upstream_model_name}' is loading; the first request owns the load. "
                "Retry this request when that load completes."
            )
            log_api_event(
                "model_request_blocked_while_loading",
                {
                    "model": upstream_model_name,
                    "api_style": api_style,
                    "reason": "load_in_progress",
                },
            )
            if api_style == "openai":
                self._send_json(
                    {"error": {"message": message, "type": "model_loading", "code": "model_loading"}},
                    status=503,
                )
            else:
                self._send_json({"error": message, "code": "model_loading"}, status=503)
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
            model_entry = next((item for item in catalog if item.model_id == model_name), None)
            upstream_model_name = model_name
            is_replica_request = False
            affinity_key = ""
            if model_entry is not None:
                replica_defaults = resolve_global_replica_config(args)
                published_model_ids = get_published_model_ids(client_host, int(args.public_port))
                sync_replica_runtime_state(catalog, args.config, replica_defaults)
                upstream_model_name, affinity_key, is_replica_request = select_replica_for_request(
                    model_entry,
                    payload,
                    self.headers,
                    replica_defaults,
                    published_model_ids,
                    catalog=catalog,
                    config_path=args.config,
                    server_path=args.llama_server,
                    idle_ttl=resolve_idle_ttl(args),
                    server_defaults=resolve_llama_server_defaults(args),
                    public_host=client_host,
                    public_port=int(args.public_port),
                )
            if self._reject_if_model_loading(upstream_model_name, catalog, public_model_name=model_name, is_replica=is_replica_request, api_style="ollama"):
                return
            if (not is_replica_request) and self._reject_if_gpu_busy(model_name, catalog, api_style="ollama", payload=payload):
                return
            mark_model_activity(model_name, "ollama_chat", "request_start")
            if upstream_model_name:
                REPLICA_ROUTER_STATE.request_started(upstream_model_name)
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
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                self._send_json({"error": "messages or prompt is required"}, status=400)
                return
            if _messages_include_images(messages) and (model_entry is None or not _has_configured_mmproj_runtime(model_entry)):
                self._send_json(
                    {
                        "error": (
                            f"model '{model_name}' is installed without multimodal projector support (mmproj). "
                            "Re-add or update the model so the matching mmproj GGUF is downloaded and configured."
                        )
                    },
                    status=400,
                )
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                return
            stream = bool(payload.get("stream"))
            upstream_payload = {
                "model": upstream_model_name,
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
                        f"http://{client_host}:{int(args.public_port)}/v1/chat/completions",
                        data=json.dumps(upstream_payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        timeout=(60, 600),
                        stream=True,
                    )
                    log_api_event("ollama_chat_upstream_headers", {"model": model_name, "status": response.status_code, "wait_ms": _elapsed_ms(started_at), "stream": True})
                except requests.RequestException as exc:
                    log_api_event("ollama_chat_upstream_network_error", {"error": str(exc)})
                    if upstream_model_name:
                        REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                    self._send_json({"error": f"upstream unavailable: {exc}"}, status=502)
                    return
                if response.status_code >= 400:
                    body_text = response.text[:4000]
                    log_api_event("ollama_chat_upstream_error", {"status": response.status_code, "body": body_text, "payload": upstream_payload})
                    if upstream_model_name:
                        REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
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
                        if upstream_model_name:
                            REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=True)
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
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=True)
                log_api_event("ollama_chat_stream_done", done_payload)
                log_api_event("ollama_chat_total", {"model": model_name, "upstream_model": upstream_model_name, "total_ms": _elapsed_ms(started_at), "stream": True})
                return

            try:
                response = requests.post(
                    f"http://{client_host}:{int(args.public_port)}/v1/chat/completions",
                    data=json.dumps(upstream_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    timeout=(60, 600),
                    stream=False,
                )
                log_api_event("ollama_chat_upstream_headers", {"model": model_name, "status": response.status_code, "wait_ms": _elapsed_ms(started_at), "stream": False})
            except requests.RequestException as exc:
                log_api_event("ollama_chat_upstream_network_error", {"error": str(exc)})
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                self._send_json({"error": f"upstream unavailable: {exc}"}, status=502)
                return
            if response.status_code >= 400:
                body_text = response.text[:4000]
                log_api_event("ollama_chat_upstream_error", {"status": response.status_code, "body": body_text, "payload": upstream_payload})
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                self._send_json({"error": f"upstream unavailable: HTTP {response.status_code}: {body_text[:1000]}"}, status=502)
                return
            try:
                data = response.json()
            except Exception as exc:
                log_api_event("ollama_chat_upstream_invalid_json", {"status": response.status_code, "error": str(exc), "body": response.text[:4000]})
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                self._send_json({"error": f"upstream invalid response: {exc}"}, status=502)
                return
            if upstream_model_name:
                REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=True)
            log_api_event("ollama_chat_total", {"model": model_name, "upstream_model": upstream_model_name, "affinity_key": affinity_key, "total_ms": _elapsed_ms(started_at), "stream": False})
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
            request_id = f"chat_req_{uuid.uuid4().hex}"
            log_api_event("openai_chat_request", {"request_id": request_id, "payload": _summarize_api_payload_for_log(payload)})
            started_at = time.monotonic()
            catalog = load_catalog(catalog_path)
            model_name = resolve_catalog_model_name(str(payload.get("model") or "").strip(), catalog)
            if not model_name:
                self._send_json({"error": {"message": "model is required", "type": "invalid_request_error"}}, status=400)
                return
            model_entry = next((item for item in catalog if item.model_id == model_name), None)
            upstream_model_name = model_name
            is_replica_request = False
            affinity_key = ""
            if model_entry is not None:
                replica_defaults = resolve_global_replica_config(args)
                published_model_ids = get_published_model_ids(client_host, int(args.public_port))
                sync_replica_runtime_state(catalog, args.config, replica_defaults)
                upstream_model_name, affinity_key, is_replica_request = select_replica_for_request(
                    model_entry,
                    payload,
                    self.headers,
                    replica_defaults,
                    published_model_ids,
                    catalog=catalog,
                    config_path=args.config,
                    server_path=args.llama_server,
                    idle_ttl=resolve_idle_ttl(args),
                    server_defaults=resolve_llama_server_defaults(args),
                    public_host=client_host,
                    public_port=int(args.public_port),
                )
            conversation_key = resolve_request_conversation_key(payload, self.headers)
            conversation_token, conversation_start_reason = CONVERSATION_SWITCH_STATE.start(conversation_key, model_name)
            if conversation_start_reason == "model_changed":
                log_api_event(
                    "openai_chat_conversation_model_switch",
                    {
                        "request_id": request_id,
                        "model": model_name,
                        "conversation_key_hash": hashlib.sha256(conversation_key.encode("utf-8", errors="ignore")).hexdigest()[:16] if conversation_key else "",
                        "conversation_state": CONVERSATION_SWITCH_STATE.snapshot(conversation_key) if conversation_key else {},
                    },
                )
            if self._reject_if_model_loading(upstream_model_name, catalog, public_model_name=model_name, is_replica=is_replica_request, api_style="openai"):
                CONVERSATION_SWITCH_STATE.finish(conversation_token)
                return
            if (not is_replica_request) and self._reject_if_gpu_busy(model_name, catalog, api_style="openai", payload=payload):
                CONVERSATION_SWITCH_STATE.finish(conversation_token)
                return
            log_api_event(
                "openai_chat_route_decision",
                {
                    "request_id": request_id,
                    "model": model_name,
                    "upstream_model": upstream_model_name,
                    "is_replica": is_replica_request,
                    "affinity_key": affinity_key,
                    "conversation_key_hash": hashlib.sha256(conversation_key.encode("utf-8", errors="ignore")).hexdigest()[:16] if conversation_key else "",
                    "conversation_generation": conversation_token.generation if conversation_token else 0,
                    "router_state": replica_trace_state_for_base(model_name),
                },
            )
            log_model_runtime_snapshot(
                "openai_chat_runtime_before_upstream",
                upstream_model_name or model_name,
                catalog,
                client_host,
                int(args.public_port),
                include_upstream_health=True,
                request_id=request_id,
                public_model=model_name,
                is_replica=is_replica_request,
                affinity_key=affinity_key,
            )
            mark_model_activity(model_name, "openai_chat", "request_start")
            if upstream_model_name:
                REPLICA_ROUTER_STATE.request_started(upstream_model_name)
            raw_messages = payload.get("messages") or []
            if not isinstance(raw_messages, list) or not raw_messages:
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                CONVERSATION_SWITCH_STATE.finish(conversation_token)
                self._send_json({"error": {"message": "messages is required", "type": "invalid_request_error"}}, status=400)
                return
            messages = _sanitize_chat_tool_repair_notices_in_messages([_normalize_openai_message(item) for item in raw_messages if isinstance(item, dict)])
            normalized_system_messages = _normalize_system_messages_for_llamacpp(messages)
            if len(normalized_system_messages) != len(messages):
                log_api_event(
                    "openai_chat_system_messages_merged",
                    {
                        "request_id": request_id,
                        "model": model_name,
                        "before_count": len(messages),
                        "after_count": len(normalized_system_messages),
                        "system_count": sum(1 for item in messages if item.get("role") == "system"),
                    },
                )
            messages = normalized_system_messages
            normalized_messages = _normalize_trailing_assistant_messages_for_llamacpp(messages)
            if len(normalized_messages) != len(messages):
                log_api_event(
                    "openai_chat_trailing_assistant_messages_merged",
                    {
                        "request_id": request_id,
                        "model": model_name,
                        "before_count": len(messages),
                        "after_count": len(normalized_messages),
                        "tail_roles_before": [str(item.get("role") or "") for item in messages[-8:] if isinstance(item, dict)],
                    },
                )
            messages = normalized_messages
            tool_diag = _summarize_chat_tool_message_diagnostics(messages)
            if tool_diag.get("matches"):
                log_api_event("openai_chat_tool_message_diagnostics", {"request_id": request_id, "model": model_name, "matches": tool_diag.get("matches"), "tool_message_count": tool_diag.get("tool_message_count")})
            if not messages:
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                CONVERSATION_SWITCH_STATE.finish(conversation_token)
                self._send_json({"error": {"message": "messages is required", "type": "invalid_request_error"}}, status=400)
                return
            if _messages_include_images(messages) and (model_entry is None or not _has_configured_mmproj_runtime(model_entry)):
                error_message = (
                    f"model '{model_name}' is installed without multimodal projector support (mmproj). "
                    "Use a vision model or re-add/update this model so the matching mmproj GGUF is downloaded and configured."
                )
                log_api_event(
                    "openai_chat_image_rejected_nonvision",
                    {
                        "model": model_name,
                        "request_id": request_id,
                        "has_model_entry": model_entry is not None,
                        "vision": bool(model_entry and _has_configured_mmproj_runtime(model_entry)),
                        "payload": _summarize_api_payload_for_log(payload),
                    },
                )
                self._send_json(
                    {
                        "error": {
                            "message": error_message,
                            "type": "invalid_request_error",
                        }
                    },
                    status=400,
                )
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                CONVERSATION_SWITCH_STATE.finish(conversation_token)
                return
            upstream_payload = dict(payload)
            upstream_payload["model"] = upstream_model_name
            upstream_payload["messages"] = messages
            upstream_payload.setdefault("cache_prompt", True)
            # OpenAI-compatible clients use both spellings. llama.cpp uses
            # max_tokens for the generation cap; normalize before deriving a
            # reasoning budget so max_completion_tokens cannot accidentally
            # leave the model with an unbounded/fixed server cap.
            if "max_tokens" not in upstream_payload and "max_completion_tokens" in upstream_payload:
                upstream_payload["max_tokens"] = upstream_payload.get("max_completion_tokens")

            if model_entry is not None and _model_backend(model_entry) == "llama.cpp":
                active_server_defaults = resolve_llama_server_defaults(args)
                reasoning_budget, reasoning_budget_reason = resolve_request_reasoning_budget(
                    payload,
                    model_entry,
                    server_defaults=active_server_defaults,
                )
                if reasoning_budget is not None:
                    upstream_payload["thinking_budget_tokens"] = reasoning_budget
                    log_api_event(
                        "openai_reasoning_budget_applied",
                        {
                            "request_id": request_id,
                            "model": model_name,
                            "upstream_model": upstream_model_name,
                            "configured_context": displayed_configured_ctx(model_entry),
                            "half_context": displayed_configured_ctx(model_entry) // 2,
                            "applied_budget": reasoning_budget,
                            "request_max_tokens": payload.get("max_tokens", payload.get("max_completion_tokens")),
                            "configured_predict": _effective_llama_server_options(model_entry, active_server_defaults).get("predict"),
                            "reason": reasoning_budget_reason,
                        },
                    )
            
            # Use explicit CLI flag if provided, otherwise fall back to server config
            flatten_enabled = True
            if getattr(args, "flatten", None) is not None:
                flatten_enabled = bool(args.flatten)
            else:
                try:
                    _server_cfg = _load_server_config_payload(args)
                    flatten_enabled = bool(_server_cfg.get("flatten_namespace_tools", True))
                except Exception:
                    pass

            if flatten_enabled:
                tools_list = upstream_payload.get("tools")
                if isinstance(tools_list, list):
                    upstream_payload["tools"] = _flatten_responses_tools(tools_list)
            
            stream = bool(payload.get("stream"))
            repair_cfg = resolve_chat_tool_continue_repair_config(args)
            repair_enabled = bool(repair_cfg.get("enabled")) and bool(upstream_payload.get("tools"))
            try:
                repair_max_rounds = max(0, int(repair_cfg.get("max_rounds", 1)))
            except Exception:
                repair_max_rounds = 1
            try:
                repair_max_tokens = max(0, int(repair_cfg.get("max_tokens", 2048)))
            except Exception:
                repair_max_tokens = 2048
            try:
                repair_stream_keepalive_seconds = max(1, int(repair_cfg.get("stream_keepalive_seconds", 15)))
            except Exception:
                repair_stream_keepalive_seconds = 15
            try:
                repair_visible_notice_after_seconds = max(0, int(repair_cfg.get("visible_notice_after_seconds", 4)))
            except Exception:
                repair_visible_notice_after_seconds = 4
            repair_trigger_prefixes = _normalize_chat_tool_continue_trigger_prefixes(repair_cfg.get("trigger_prefixes"))
            repair_prompt_template = repair_cfg.get("prompt")
            repair_truncated_prompt_template = repair_cfg.get("truncated_tool_call_prompt")
            repair_include_failed_assistant_message = bool(repair_cfg.get("include_failed_assistant_message"))
            repair_loop_guard = repair_cfg.get("loop_guard") if isinstance(repair_cfg.get("loop_guard"), dict) else {}

            def _select_chat_tool_call_example_for_repair(current_messages_for_example: object) -> dict[str, str] | None:
                examples = _extract_chat_tool_call_examples_from_messages(
                    current_messages_for_example,
                    upstream_payload.get("tools"),
                    limit=1,
                )
                if examples:
                    return examples[0]
                return CHAT_TOOL_CALL_EXAMPLES.get(
                    _chat_tool_example_keys(model_name, upstream_model_name, conversation_key),
                    upstream_payload.get("tools"),
                )

            def _remember_successful_chat_tool_calls(tool_calls: object) -> None:
                keys = _chat_tool_example_keys(model_name, upstream_model_name, conversation_key)
                for key in keys:
                    CHAT_TOOL_CALL_EXAMPLES.remember(
                        key,
                        tool_calls,
                        upstream_payload.get("tools"),
                        request_id=request_id,
                    )

            def _conversation_cancel_reason() -> tuple[bool, str]:
                cancelled, reason = CONVERSATION_SWITCH_STATE.should_cancel(conversation_token)
                if cancelled:
                    log_api_event(
                        "openai_chat_conversation_request_cancelled",
                        {
                            "request_id": request_id,
                            "model": model_name,
                            "upstream_model": upstream_model_name,
                            "reason": reason,
                            "conversation_key_hash": hashlib.sha256(conversation_key.encode("utf-8", errors="ignore")).hexdigest()[:16] if conversation_key else "",
                            "conversation_generation": conversation_token.generation if conversation_token else 0,
                            "conversation_state": CONVERSATION_SWITCH_STATE.snapshot(conversation_key) if conversation_key else {},
                        },
                    )
                return cancelled, reason

            def _finish_openai_chat_request() -> None:
                CONVERSATION_SWITCH_STATE.finish(conversation_token)

            try:
                response = requests.post(
                    f"http://{client_host}:{int(args.public_port)}/v1/chat/completions",
                    data=json.dumps(upstream_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    timeout=(60, 600),
                    stream=stream,
                )
                log_api_event("openai_chat_upstream_headers", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "affinity_key": affinity_key, "status": response.status_code, "wait_ms": _elapsed_ms(started_at), "stream": stream})
            except requests.RequestException as exc:
                log_api_event("openai_chat_upstream_network_error", {"request_id": request_id, "error": str(exc), "payload": _summarize_api_payload_for_log(upstream_payload), "router_state": replica_trace_state_for_base(model_name)})
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                CONVERSATION_SWITCH_STATE.finish(conversation_token)
                self._send_json({"error": {"message": f"upstream unavailable: {exc}", "type": "server_error"}}, status=502)
                return
            if response.status_code >= 400:
                body_text = response.text[:4000]
                log_api_event("openai_chat_upstream_error", {"request_id": request_id, "status": response.status_code, "body": body_text, "payload": _summarize_api_payload_for_log(upstream_payload), "router_state": replica_trace_state_for_base(model_name)})
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                CONVERSATION_SWITCH_STATE.finish(conversation_token)
                self._send_json(
                    {"error": {"message": f"upstream unavailable: HTTP {response.status_code}: {body_text[:1000]}", "type": "server_error"}},
                    status=502,
                )
                return
            stream_repair_round_used = 0
            if stream:
                if repair_enabled and repair_max_rounds > 0:
                    self.send_response(200)
                    self.send_header("Content-Type", response.headers.get("Content-Type", "text/event-stream"))
                    self.end_headers()
                    repair_write_lock = threading.Lock()
                    current_response = response
                    current_payload = upstream_payload
                    current_messages = messages
                    final_buffered_lines: list[bytes] = []
                    final_stream_state: dict[str, object] = {}
                    stream_repair_round_used = 0
                    pending_visible_status: dict[str, object] | None = None
                    while True:
                        try:
                            buffered_lines, passthrough_done, passthrough_state = _buffer_openai_chat_sse_with_keepalive(
                                self,
                                current_response,
                                request_id=request_id,
                                keepalive_seconds=repair_stream_keepalive_seconds,
                                write_lock=repair_write_lock,
                                visible_status=pending_visible_status,
                                visible_notice_after_seconds=repair_visible_notice_after_seconds,
                                loop_guard=repair_loop_guard,
                                cancel_check=_conversation_cancel_reason,
                                thinking_budget_tokens=current_payload.get("thinking_budget_tokens"),
                            )
                            pending_visible_status = None
                            if passthrough_state.get("buffer_abort_reason"):
                                abort_reason = str(passthrough_state.get("buffer_abort_reason") or "")
                                is_conversation_cancel = abort_reason.startswith("conversation_")
                                log_api_event(
                                    "openai_chat_conversation_repair_cancelled" if is_conversation_cancel else "openai_chat_tool_continue_repair_loop_detected",
                                    {
                                        "request_id": request_id,
                                        "model": model_name,
                                        "upstream_model": upstream_model_name,
                                        "repair_round": stream_repair_round_used,
                                        "reason": abort_reason,
                                        "visible_content_len": int(passthrough_state.get("visible_content_len") or 0),
                                        "reasoning_len": int(passthrough_state.get("reasoning_len") or 0),
                                        "tool_call_chunks": int(passthrough_state.get("tool_call_chunks") or 0),
                                        "tool_argument_chars": int(passthrough_state.get("tool_argument_chars") or 0),
                                        "finish_reason": passthrough_state.get("finish_reason") or "",
                                        "elapsed_ms": _elapsed_ms(started_at),
                                    },
                                )
                                with repair_write_lock:
                                    if not is_conversation_cancel:
                                        _send_openai_chat_sse_status(
                                            self,
                                            request_id=request_id,
                                            model=model_name,
                                            content=(
                                                "\n<think>\n"
                                                "Tool-call repair loop detected; stopping this response instead of consuming more tokens.\n"
                                                "</think>\n"
                                            ),
                                        )
                                    self.wfile.write(b"data: [DONE]\n\n")
                                    self.wfile.flush()
                                mark_model_activity(model_name, "openai_chat", "stream_conversation_cancelled" if is_conversation_cancel else "stream_repair_loop_detected")
                                if upstream_model_name:
                                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                                _finish_openai_chat_request()
                                _write_chat_last_response_log(
                                    args,
                                    request_id=request_id,
                                    model=model_name,
                                    upstream_model=upstream_model_name,
                                    stream=True,
                                    content="",
                                    reasoning_len=int(passthrough_state.get("reasoning_len") or 0),
                                    tool_calls=[],
                                    tool_call_chunks=0,
                                    finish_reason=str(passthrough_state.get("buffer_abort_reason") or ""),
                                    repair_rounds=stream_repair_round_used,
                                )
                                return
                            if passthrough_done:
                                mark_model_activity(model_name, "openai_chat", "stream_passthrough_closed")
                                if upstream_model_name:
                                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=True)
                                _finish_openai_chat_request()
                                log_api_event("openai_chat_total", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "total_ms": _elapsed_ms(started_at), "stream": True, "visible_content_len": int(passthrough_state.get("visible_content_len") or 0), "reasoning_len": int(passthrough_state.get("reasoning_len") or 0), "tool_call_chunks": int(passthrough_state.get("tool_call_chunks") or 0), "tool_argument_chars": int(passthrough_state.get("tool_argument_chars") or 0), "tool_names": passthrough_state.get("tool_names") or [], "finish_reason": passthrough_state.get("finish_reason") or "passthrough", "finish_reasons_seen": passthrough_state.get("finish_reasons_seen") or [], "passthrough_reason": passthrough_state.get("passthrough_reason") or "", "tool_call_indices": passthrough_state.get("tool_call_indices") or [], "tool_argument_lengths_by_index": passthrough_state.get("tool_argument_lengths_by_index") or {}, "tool_argument_json_valid_by_index": passthrough_state.get("tool_argument_json_valid_by_index") or {}, "tool_argument_tail_by_index": passthrough_state.get("tool_argument_tail_by_index") or {}, "reasoning_only_final": False, "repair_rounds": stream_repair_round_used, "router_state": replica_trace_state_for_base(model_name)})
                                return
                        except (requests.RequestException, Exception) as exc:
                            log_api_event("openai_chat_stream_interrupted", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "error": str(exc), "router_state": replica_trace_state_for_base(model_name), "repair_round": stream_repair_round_used})
                            if upstream_model_name:
                                REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                            _finish_openai_chat_request()
                            _send_openai_chat_sse_error(self, f"upstream stream failed before repair decision: {exc}")
                            return
                        finally:
                            current_response.close()
                        stream_state = _chat_completion_state_from_sse_lines(buffered_lines)
                        trigger_reason = _chat_tool_continue_trigger_reason(
                            stream_state.get("content"),
                            stream_state.get("tool_calls"),
                            upstream_payload.get("tools"),
                            repair_trigger_prefixes,
                        )
                        truncated_trigger_reason = _chat_tool_truncated_trigger_reason(stream_state)
                        final_buffered_lines = buffered_lines
                        final_stream_state = stream_state
                        if not trigger_reason and not truncated_trigger_reason:
                            break
                        active_trigger_reason = truncated_trigger_reason or trigger_reason
                        if stream_repair_round_used >= repair_max_rounds:
                            log_api_event("openai_chat_tool_continue_repair_exhausted", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "rounds": stream_repair_round_used, "stream": True, "visible_content_len": len(str(stream_state.get("content") or "")), "reasoning_len": int(stream_state.get("reasoning_len") or 0), "tool_call_chunks": int(stream_state.get("tool_call_chunks") or 0), "finish_reason": stream_state.get("finish_reason") or "", "trigger_reason": active_trigger_reason})
                            if truncated_trigger_reason:
                                if upstream_model_name:
                                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                                _finish_openai_chat_request()
                                _send_openai_chat_sse_truncated_tool_error(self, request_id=request_id, model=model_name, rounds=stream_repair_round_used)
                                return
                            break
                        log_api_event("openai_chat_tool_continue_repair_triggered", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "stream": True, "visible_content_len": len(str(stream_state.get("content") or "")), "reasoning_len": int(stream_state.get("reasoning_len") or 0), "tool_call_chunks": int(stream_state.get("tool_call_chunks") or 0), "finish_reason": stream_state.get("finish_reason") or "", "trigger_reason": active_trigger_reason})
                        stream_repair_round_used += 1
                        repair_payload = _force_tool_choice_for_chat_repair(_apply_chat_tool_continue_repair_token_cap(upstream_payload, repair_max_tokens))
                        repair_payload["stream"] = True
                        if truncated_trigger_reason:
                            current_messages = _chat_tool_truncated_repair_messages(current_messages, upstream_payload.get("tools"), repair_truncated_prompt_template)
                        else:
                            current_messages = _chat_tool_continue_repair_messages(
                                current_messages,
                                stream_state.get("message") if isinstance(stream_state.get("message"), dict) else {"role": "assistant", "content": ""},
                                upstream_payload.get("tools"),
                                repair_prompt_template,
                                repair_include_failed_assistant_message,
                                _select_chat_tool_call_example_for_repair(current_messages),
                            )
                        repair_payload["messages"] = current_messages
                        pending_visible_status = {
                            "request_id": request_id,
                            "model": model_name,
                            "upstream_model": upstream_model_name,
                            "round": stream_repair_round_used,
                            "max_rounds": repair_max_rounds,
                            "trigger_reason": active_trigger_reason,
                            "content": (
                                f"\n↻ Retrying tool call generation (attempt {stream_repair_round_used}/{repair_max_rounds}); "
                                "the previous model turn did not produce a complete valid tool call. Waiting for the repaired tool call…\n"
                            ),
                        }
                        log_api_event("openai_chat_tool_continue_repair_round", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "round": stream_repair_round_used, "stream": True, "trigger_reason": active_trigger_reason, "visible_notice_after_seconds": repair_visible_notice_after_seconds})
                        cancelled, cancel_reason = _conversation_cancel_reason()
                        if cancelled:
                            if upstream_model_name:
                                REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                            _finish_openai_chat_request()
                            with repair_write_lock:
                                self.wfile.write(b"data: [DONE]\n\n")
                                self.wfile.flush()
                            return
                        try:
                            current_response = requests.post(
                                f"http://{client_host}:{int(args.public_port)}/v1/chat/completions",
                                data=json.dumps(repair_payload).encode("utf-8"),
                                headers={"Content-Type": "application/json"},
                                timeout=(60, 600),
                                stream=True,
                            )
                            current_content_type = current_response.headers.get("Content-Type", "text/event-stream")
                            log_api_event("openai_chat_upstream_headers", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "affinity_key": affinity_key, "status": current_response.status_code, "wait_ms": _elapsed_ms(started_at), "stream": True, "repair_round": stream_repair_round_used})
                        except requests.RequestException as exc:
                            log_api_event("openai_chat_upstream_network_error", {"request_id": request_id, "error": str(exc), "payload": _summarize_api_payload_for_log(repair_payload), "router_state": replica_trace_state_for_base(model_name), "repair_round": stream_repair_round_used})
                            if upstream_model_name:
                                REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                            _finish_openai_chat_request()
                            _send_openai_chat_sse_error(self, f"upstream unavailable during repair: {exc}")
                            return
                        if current_response.status_code >= 400:
                            body_text = current_response.text[:4000]
                            log_api_event("openai_chat_upstream_error", {"request_id": request_id, "status": current_response.status_code, "body": body_text, "payload": _summarize_api_payload_for_log(repair_payload), "router_state": replica_trace_state_for_base(model_name), "repair_round": stream_repair_round_used})
                            if upstream_model_name:
                                REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                            _finish_openai_chat_request()
                            _send_openai_chat_sse_error(self, f"upstream unavailable during repair: HTTP {current_response.status_code}: {body_text[:1000]}")
                            return
                    mark_model_activity(model_name, "openai_chat", "stream_closed")
                    if upstream_model_name:
                        REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=True)
                    _finish_openai_chat_request()
                    try:
                        with repair_write_lock:
                            for line in final_buffered_lines:
                                if line:
                                    self.wfile.write(line + b"\n")
                                else:
                                    self.wfile.write(b"\n")
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, ssl.SSLEOFError, OSError) as exc:
                        log_api_event("openai_chat_stream_client_disconnected", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "error": str(exc), "phase": "repair_buffer_flush"})
                    final_content = str(final_stream_state.get("content") or "")
                    final_reasoning_len = int(final_stream_state.get("reasoning_len") or 0)
                    final_tool_call_chunks = int(final_stream_state.get("tool_call_chunks") or 0)
                    if final_tool_call_chunks > 0:
                        _remember_successful_chat_tool_calls(final_stream_state.get("tool_calls") or [])
                    reasoning_only_final = bool(final_reasoning_len) and not final_content and not final_stream_state.get("tool_calls")
                    log_api_event("openai_chat_total", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "total_ms": _elapsed_ms(started_at), "stream": True, "visible_content_len": len(final_content), "reasoning_len": final_reasoning_len, "tool_call_chunks": final_tool_call_chunks, "finish_reason": final_stream_state.get("finish_reason") or "", "reasoning_only_final": reasoning_only_final, "repair_rounds": stream_repair_round_used, "router_state": replica_trace_state_for_base(model_name)})
                    _write_chat_last_response_log(args, request_id=request_id, model=model_name, upstream_model=upstream_model_name, stream=True, content=final_content, reasoning_len=final_reasoning_len, tool_calls=final_stream_state.get("tool_calls"), tool_call_chunks=final_tool_call_chunks, finish_reason=final_stream_state.get("finish_reason") or "", repair_rounds=stream_repair_round_used)
                    if final_tool_call_chunks == 0:
                        _log_chat_stop_without_tools(request_id, model_name, upstream_model_name, stream=True, content=final_content, reasoning_len=final_reasoning_len, finish_reason=final_stream_state.get("finish_reason") or "", repair_rounds=stream_repair_round_used)
                    if reasoning_only_final:
                        log_api_event("openai_chat_reasoning_only_final", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "reasoning_len": final_reasoning_len, "finish_reason": final_stream_state.get("finish_reason") or ""})
                    return
                self.send_response(200)
                self.send_header("Content-Type", response.headers.get("Content-Type", "text/event-stream"))
                self.end_headers()
                
                write_lock = threading.Lock()
                stop_heartbeat = threading.Event()

                def send_pulse():
                    while not stop_heartbeat.is_set():
                        if stop_heartbeat.wait(15):
                            break
                        try:
                            with write_lock:
                                log_api_event("openai_chat_stream_pulse", {"request_id": request_id})
                                pulse_payload = {
                                    "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}]
                                }
                                # Send SSE comment and empty delta as heartbeat
                                self.wfile.write(b": keep-alive\n\n")
                                self.wfile.write(("data: " + json.dumps(pulse_payload, ensure_ascii=False) + "\n\n").encode("utf-8"))
                                self.wfile.flush()
                        except Exception:
                            break

                heartbeat_thread = threading.Thread(target=send_pulse)
                heartbeat_thread.daemon = True
                heartbeat_thread.start()

                first_chunk_logged = False
                stream_visible_content_len = 0
                stream_reasoning_len = 0
                stream_tool_call_chunks = 0
                stream_finish_reason = ""
                stream_visible_content_parts: list[str] = []
                try:
                    for line in response.iter_lines():
                        with write_lock:
                            if not first_chunk_logged and line and line.startswith(b"data: "):
                                first_chunk_logged = True
                                log_api_event("openai_chat_first_chunk", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "first_chunk_ms": _elapsed_ms(started_at)})
                            
                            if line:
                                if line.startswith(b"data: ") and line != b"data: [DONE]":
                                    try:
                                        chunk_data = json.loads(line[6:].decode("utf-8", errors="ignore"))
                                        choice0 = (chunk_data.get("choices") or [{}])[0]
                                        delta = choice0.get("delta", {}) or {}
                                        content = delta.get("content", "") or ""
                                        reasoning = delta.get("reasoning_content", "") or ""
                                        tool_calls = delta.get("tool_calls") or []
                                        finish_reason = choice0.get("finish_reason")
                                        if finish_reason:
                                            stream_finish_reason = str(finish_reason)
                                        if content:
                                            stream_visible_content_len += len(str(content))
                                            stream_visible_content_parts.append(str(content))
                                        if reasoning:
                                            stream_reasoning_len += len(str(reasoning))
                                        if tool_calls:
                                            stream_tool_call_chunks += len(tool_calls) if isinstance(tool_calls, list) else 1
                                        if content or reasoning or tool_calls or finish_reason:
                                            log_api_event("openai_chat_stream_chunk_received", {
                                                "request_id": request_id,
                                                "content_len": len(str(content)),
                                                "reasoning_len": len(str(reasoning)),
                                                "tool_call_chunks": len(tool_calls) if isinstance(tool_calls, list) else (1 if tool_calls else 0),
                                                "finish_reason": finish_reason or "",
                                                "preview": (str(content) or str(reasoning))[:50]
                                            })
                                    except Exception:
                                        pass
                                self.wfile.write(line + b"\n")
                            else:
                                self.wfile.write(b"\n")
                            self.wfile.flush()
                        if line and line.startswith(b"data: "):
                            mark_model_activity(model_name, "openai_chat", "stream_chunk", log=False)
                except (BrokenPipeError, ConnectionResetError, requests.RequestException) as exc:
                    log_api_event("openai_chat_stream_interrupted", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "error": str(exc), "router_state": replica_trace_state_for_base(model_name)})
                finally:
                    stop_heartbeat.set()
                    heartbeat_thread.join(timeout=1.0)
                    response.close()
                    mark_model_activity(model_name, "openai_chat", "stream_closed")
                    if upstream_model_name:
                        REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=True)
                    _finish_openai_chat_request()
                    reasoning_only_final = stream_reasoning_len > 0 and stream_visible_content_len == 0 and stream_tool_call_chunks == 0
                    log_api_event("openai_chat_total", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "total_ms": _elapsed_ms(started_at), "stream": True, "visible_content_len": stream_visible_content_len, "reasoning_len": stream_reasoning_len, "tool_call_chunks": stream_tool_call_chunks, "finish_reason": stream_finish_reason, "reasoning_only_final": reasoning_only_final, "router_state": replica_trace_state_for_base(model_name)})
                    _write_chat_last_response_log(args, request_id=request_id, model=model_name, upstream_model=upstream_model_name, stream=True, content="".join(stream_visible_content_parts), reasoning_len=stream_reasoning_len, tool_calls=[{}] * stream_tool_call_chunks if stream_tool_call_chunks > 0 else [], tool_call_chunks=stream_tool_call_chunks, finish_reason=stream_finish_reason, repair_rounds=stream_repair_round_used)
                    if stream_tool_call_chunks == 0:
                        _log_chat_stop_without_tools(request_id, model_name, upstream_model_name, stream=True, content="".join(stream_visible_content_parts), reasoning_len=stream_reasoning_len, finish_reason=stream_finish_reason, repair_rounds=stream_repair_round_used)
                    if reasoning_only_final:
                        log_api_event("openai_chat_reasoning_only_final", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "reasoning_len": stream_reasoning_len, "finish_reason": stream_finish_reason})
                    if stream_repair_round_used:
                        exhausted_reason = _chat_tool_continue_trigger_reason("".join(stream_visible_content_parts), [] if stream_tool_call_chunks == 0 else [{}], upstream_payload.get("tools"), repair_trigger_prefixes)
                        if exhausted_reason:
                            log_api_event("openai_chat_tool_continue_repair_exhausted", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "rounds": stream_repair_round_used, "stream": True, "visible_content_len": stream_visible_content_len, "reasoning_len": stream_reasoning_len, "finish_reason": stream_finish_reason, "trigger_reason": exhausted_reason})
                return
            try:
                data = response.json()
            except Exception as exc:
                log_api_event("openai_chat_upstream_invalid_json", {"request_id": request_id, "error": str(exc), "payload": _summarize_api_payload_for_log(upstream_payload), "body": response.text[:4000], "router_state": replica_trace_state_for_base(model_name)})
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                _finish_openai_chat_request()
                self._send_json({"error": {"message": f"upstream invalid response: {exc}", "type": "server_error"}}, status=502)
                return
            repair_rounds_used = 0
            if repair_enabled and repair_max_rounds > 0:
                while repair_rounds_used < repair_max_rounds:
                    state = _chat_completion_state_from_payload(data)
                    trigger_reason = _chat_tool_continue_trigger_reason(
                        state.get("content"),
                        state.get("tool_calls"),
                        upstream_payload.get("tools"),
                        repair_trigger_prefixes,
                    )
                    if not trigger_reason:
                        break
                    repair_rounds_used += 1
                    log_api_event("openai_chat_tool_continue_repair_triggered", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "stream": False, "visible_content_len": len(str(state.get("content") or "")), "reasoning_len": len(str(state.get("reasoning") or "")), "finish_reason": state.get("finish_reason") or "", "trigger_reason": trigger_reason})
                    repair_payload = _force_tool_choice_for_chat_repair(_apply_chat_tool_continue_repair_token_cap(upstream_payload, repair_max_tokens))
                    repair_payload["stream"] = False
                    repair_payload["messages"] = _chat_tool_continue_repair_messages(
                        messages,
                        state.get("message") if isinstance(state.get("message"), dict) else {"role": "assistant", "content": ""},
                        upstream_payload.get("tools"),
                        repair_prompt_template,
                        repair_include_failed_assistant_message,
                        _select_chat_tool_call_example_for_repair(messages),
                    )
                    log_api_event("openai_chat_tool_continue_repair_round", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "round": repair_rounds_used, "stream": False, "trigger_reason": trigger_reason})
                    cancelled, cancel_reason = _conversation_cancel_reason()
                    if cancelled:
                        if upstream_model_name:
                            REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                        _finish_openai_chat_request()
                        self._send_json({"error": {"message": f"request cancelled because conversation changed model: {cancel_reason}", "type": "server_error"}}, status=499)
                        return
                    try:
                        repair_response = requests.post(
                            f"http://{client_host}:{int(args.public_port)}/v1/chat/completions",
                            data=json.dumps(repair_payload).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            timeout=(60, 600),
                            stream=False,
                        )
                        log_api_event("openai_chat_upstream_headers", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "affinity_key": affinity_key, "status": repair_response.status_code, "wait_ms": _elapsed_ms(started_at), "stream": False, "repair_round": repair_rounds_used})
                    except requests.RequestException as exc:
                        log_api_event("openai_chat_upstream_network_error", {"request_id": request_id, "error": str(exc), "payload": _summarize_api_payload_for_log(repair_payload), "router_state": replica_trace_state_for_base(model_name), "repair_round": repair_rounds_used})
                        if upstream_model_name:
                            REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                        _finish_openai_chat_request()
                        self._send_json({"error": {"message": f"upstream unavailable during repair: {exc}", "type": "server_error"}}, status=502)
                        return
                    if repair_response.status_code >= 400:
                        body_text = repair_response.text[:4000]
                        log_api_event("openai_chat_upstream_error", {"request_id": request_id, "status": repair_response.status_code, "body": body_text, "payload": _summarize_api_payload_for_log(repair_payload), "router_state": replica_trace_state_for_base(model_name), "repair_round": repair_rounds_used})
                        if upstream_model_name:
                            REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                        _finish_openai_chat_request()
                        self._send_json({"error": {"message": f"upstream unavailable during repair: HTTP {repair_response.status_code}: {body_text[:1000]}", "type": "server_error"}}, status=502)
                        return
                    try:
                        data = repair_response.json()
                    except Exception as exc:
                        log_api_event("openai_chat_upstream_invalid_json", {"request_id": request_id, "error": str(exc), "payload": _summarize_api_payload_for_log(repair_payload), "body": repair_response.text[:4000], "router_state": replica_trace_state_for_base(model_name), "repair_round": repair_rounds_used})
                        if upstream_model_name:
                            REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                        _finish_openai_chat_request()
                        self._send_json({"error": {"message": f"upstream invalid response during repair: {exc}", "type": "server_error"}}, status=502)
                        return
                final_state = _chat_completion_state_from_payload(data)
                exhausted_reason = _chat_tool_continue_trigger_reason(
                    final_state.get("content"),
                    final_state.get("tool_calls"),
                    upstream_payload.get("tools"),
                    repair_trigger_prefixes,
                )
                if exhausted_reason and repair_rounds_used >= repair_max_rounds:
                    log_api_event("openai_chat_tool_continue_repair_exhausted", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "rounds": repair_rounds_used, "stream": False, "visible_content_len": len(str(final_state.get("content") or "")), "reasoning_len": len(str(final_state.get("reasoning") or "")), "finish_reason": final_state.get("finish_reason") or "", "trigger_reason": exhausted_reason})
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            nonstream_content = str(message.get("content") or "")
            nonstream_reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
            nonstream_tool_calls = message.get("tool_calls") or []
            nonstream_reasoning_only = bool(nonstream_reasoning) and not nonstream_content and not nonstream_tool_calls
            if nonstream_tool_calls:
                _remember_successful_chat_tool_calls(nonstream_tool_calls)
            if upstream_model_name:
                REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=True)
            _finish_openai_chat_request()
            log_api_event("openai_chat_total", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "total_ms": _elapsed_ms(started_at), "stream": False, "visible_content_len": len(nonstream_content), "reasoning_len": len(nonstream_reasoning), "tool_calls_count": len(nonstream_tool_calls) if isinstance(nonstream_tool_calls, list) else (1 if nonstream_tool_calls else 0), "finish_reason": choice.get("finish_reason") or "", "reasoning_only_final": nonstream_reasoning_only, "router_state": replica_trace_state_for_base(model_name)})
            _write_chat_last_response_log(args, request_id=request_id, model=model_name, upstream_model=upstream_model_name, stream=False, content=nonstream_content, reasoning=nonstream_reasoning, tool_calls=nonstream_tool_calls, tool_call_chunks=len(nonstream_tool_calls) if isinstance(nonstream_tool_calls, list) else (1 if nonstream_tool_calls else 0), finish_reason=choice.get("finish_reason") or "", repair_rounds=repair_rounds_used)
            if not nonstream_tool_calls:
                _log_chat_stop_without_tools(request_id, model_name, upstream_model_name, stream=False, content=nonstream_content, reasoning_len=len(nonstream_reasoning), finish_reason=choice.get("finish_reason") or "", repair_rounds=repair_rounds_used)
            if nonstream_reasoning_only:
                log_api_event("openai_chat_reasoning_only_final", {"request_id": request_id, "model": model_name, "upstream_model": upstream_model_name, "reasoning_len": len(nonstream_reasoning), "finish_reason": choice.get("finish_reason") or ""})
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

        def _handle_openai_responses(self):
            payload = self._read_json_body()
            request_id = f"resp_req_{uuid.uuid4().hex}"
            _payload_summary = _summarize_api_payload_for_log(payload)
            _tool_item_summary = _summarize_responses_input_tool_items(payload)
            if _tool_item_summary.get("function_calls") or _tool_item_summary.get("function_call_outputs"):
                _payload_summary.update(_tool_item_summary)
            log_api_event(
                "openai_responses_request",
                {"request_id": request_id, "payload": _payload_summary},
            )
            started_at = time.monotonic()
            catalog = load_catalog(catalog_path)
            model_name = resolve_catalog_model_name(str(payload.get("model") or "").strip(), catalog)
            if not model_name:
                self._send_json({"error": {"message": "model is required", "type": "invalid_request_error"}}, status=400)
                return
            model_entry = next((item for item in catalog if item.model_id == model_name), None)
            upstream_model_name = model_name
            is_replica_request = False
            affinity_key = ""
            if model_entry is not None:
                replica_defaults = resolve_global_replica_config(args)
                published_model_ids = get_published_model_ids(client_host, int(args.public_port))
                sync_replica_runtime_state(catalog, args.config, replica_defaults)
                upstream_model_name, affinity_key, is_replica_request = select_replica_for_request(
                    model_entry,
                    payload,
                    self.headers,
                    replica_defaults,
                    published_model_ids,
                    catalog=catalog,
                    config_path=args.config,
                    server_path=args.llama_server,
                    idle_ttl=resolve_idle_ttl(args),
                    server_defaults=resolve_llama_server_defaults(args),
                    public_host=client_host,
                    public_port=int(args.public_port),
                )
            if self._reject_if_model_loading(upstream_model_name, catalog, public_model_name=model_name, is_replica=is_replica_request, api_style="openai"):
                return
            if (not is_replica_request) and self._reject_if_gpu_busy(model_name, catalog, api_style="openai", payload=payload):
                return
            mark_model_activity(model_name, f"openai_responses:{request_id}", "request_start")
            if upstream_model_name:
                REPLICA_ROUTER_STATE.request_started(upstream_model_name)
            log_model_runtime_snapshot(
                "openai_responses_runtime_before_upstream",
                model_name,
                catalog,
                client_host,
                int(args.public_port),
                request_id=request_id,
            )
            _server_cfg_resp = _load_server_config_payload(args)

            # Default path: this proxy is the Responses API compatibility
            # adapter. Do not blindly forward Codex Desktop's modern Responses
            # payload to llama.cpp: some builds expose/route /v1/responses only
            # partially and may crash on proprietary tools such as
            # computer_use_preview. Raw passthrough is opt-in for future
            # backends that are known to implement modern Responses safely.
            if _responses_raw_passthrough_enabled():
                _passthrough_payload = {**payload, "model": upstream_model_name}
                try:
                    # Use explicit CLI flag if provided, otherwise fall back to server config
                    flatten_enabled = True
                    if getattr(args, "flatten", None) is not None:
                        flatten_enabled = bool(args.flatten)
                    else:
                        flatten_enabled = bool(_server_cfg_resp.get("flatten_namespace_tools", True))
                    
                    pt_tools = _passthrough_payload.get("tools")
                    if isinstance(pt_tools, list):
                        # Raw Responses passthrough is for backends that understand
                        # Responses-native tools.  Do not explode namespaces into
                        # hundreds of legacy functions here; prepare them for
                        # tool_search/deferred loading instead.
                        _passthrough_payload["tools"] = _responses_tools_with_deferred_search(pt_tools)
                    
                    log_api_event("openai_responses_passthrough_payload", {
                        "model": model_name,
                        "request_id": request_id,
                        "tool_count": len(_passthrough_payload.get("tools", [])),
                        "payload": _summarize_api_payload_for_log(_passthrough_payload)
                    })
                except Exception:
                    log_api_event("flatten_tools_error", {"error": "failed to flatten tools in responses passthrough", "model": model_name})
                    pass
                try:
                    passthrough_response = requests.post(
                        f"http://{client_host}:{int(args.public_port)}/v1/responses",
                        data=json.dumps(_passthrough_payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        timeout=(60, 600),
                        stream=False,
                    )
                    log_api_event(
                        "openai_responses_passthrough_headers",
                        {
                            "model": model_name,
                            "upstream_model": upstream_model_name,
                            "affinity_key": affinity_key,
                            "request_id": request_id,
                            "status": passthrough_response.status_code,
                            "wait_ms": _elapsed_ms(started_at),
                        },
                    )
                    if passthrough_response.status_code not in {404, 405, 501}:
                        mark_model_activity(model_name, f"openai_responses:{request_id}", "passthrough_done")
                        if upstream_model_name:
                            REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=True)
                        self._proxy_raw_response(passthrough_response)
                        return
                    log_api_event(
                        "openai_responses_passthrough_unsupported",
                        {
                            "model": model_name,
                            "request_id": request_id,
                            "status": passthrough_response.status_code,
                            "body": passthrough_response.text[:1000],
                        },
                    )
                except requests.RequestException as exc:
                    log_api_event("openai_responses_passthrough_network_error", {"model": model_name, "request_id": request_id, "error": str(exc)})
                    log_model_runtime_snapshot(
                        "openai_responses_runtime_after_passthrough_network_error",
                        model_name,
                        catalog,
                        client_host,
                        int(args.public_port),
                        request_id=request_id,
                    )
            else:
                log_api_event("openai_responses_passthrough_skipped", {"model": model_name, "request_id": request_id, "reason": "disabled"})

            # Compatibility path for llama.cpp versions that only expose
            # /v1/chat/completions.  This path accepts modern Responses payloads
            # and translates tools to the legacy Chat Completions schema while
            # preserving the Responses facade on the way back to Codex.
            flatten_namespace_tools = bool(_server_cfg_resp.get("flatten_namespace_tools", True))
            tool_registry = (
                ResponsesToolRegistry.from_responses_tools(payload.get("tools"), flatten_namespace_tools=flatten_namespace_tools)
                if _responses_tool_search_emulation_enabled(_server_cfg_resp)
                else None
            )
            namespace_tool_map = _responses_namespace_tool_map(payload.get("tools")) if flatten_namespace_tools else {}
            allow_tool_output_images = bool(model_entry and _has_configured_mmproj_runtime(model_entry))
            upstream_payload = _responses_payload_to_chat_payload(
                payload,
                upstream_model_name,
                flatten_namespace_tools=flatten_namespace_tools,
                tool_registry=tool_registry,
                allow_tool_output_images=allow_tool_output_images,
            )
            allowed_legacy_tool_names = {
                str(tool.get("function", {}).get("name") or "")
                for tool in upstream_payload.get("tools", [])
                if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
            }
            log_api_event(
                "openai_responses_chat_fallback_payload",
                {
                    "model": model_name, 
                    "request_id": request_id, 
                    "tool_count": len(upstream_payload.get("tools", [])),
                    "allowed_tool_names": sorted(name for name in allowed_legacy_tool_names if name)[:80],
                    "payload": _summarize_api_payload_for_log(upstream_payload)
                },
            )
            messages = _normalize_system_messages_for_llamacpp(upstream_payload.get("messages") or [])
            normalized_messages = _normalize_trailing_assistant_messages_for_llamacpp(messages)
            if len(normalized_messages) != len(messages):
                log_api_event(
                    "openai_responses_trailing_assistant_messages_merged",
                    {
                        "request_id": request_id,
                        "model": model_name,
                        "before_count": len(messages),
                        "after_count": len(normalized_messages),
                        "tail_roles_before": [str(item.get("role") or "") for item in messages[-8:] if isinstance(item, dict)],
                    },
                )
                upstream_payload["messages"] = normalized_messages
                messages = normalized_messages
            if not messages:
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                self._send_json({"error": {"message": "input is required", "type": "invalid_request_error"}}, status=400)
                return
            messages_have_images = _messages_include_images(messages)
            if messages_have_images and (model_entry is None or not _has_configured_mmproj_runtime(model_entry)):
                error_message = (
                    f"model '{model_name}' is installed without multimodal projector support (mmproj). "
                    "Use a vision model or re-add/update this model so the matching mmproj GGUF is downloaded and configured."
                )
                log_api_event(
                    "openai_responses_image_rejected_nonvision",
                    {
                        "model": model_name,
                        "request_id": request_id,
                        "has_model_entry": model_entry is not None,
                        "vision": bool(model_entry and _has_configured_mmproj_runtime(model_entry)),
                        "payload": _summarize_api_payload_for_log(upstream_payload),
                    },
                )
                self._send_json(
                    {
                        "error": {
                            "message": error_message,
                            "type": "invalid_request_error",
                        }
                    },
                    status=400,
                )
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                return
            if messages_have_images and model_entry is not None and _loaded_process_missing_configured_mmproj(model_entry, catalog):
                proc = get_catalog_model_process(model_entry.model_id, catalog)
                log_api_event(
                    "openai_responses_image_stale_mmproj_runtime_reload",
                    {
                        "model": model_name,
                        "request_id": request_id,
                        "configured_mmproj": str(model_entry.mmproj_path or ""),
                        "process": proc,
                        "payload": _summarize_api_payload_for_log(upstream_payload),
                    },
                )
                reloaded = reload_model_runtime_from_catalog_config(
                    model_entry,
                    catalog,
                    args,
                    client_host,
                    int(args.public_port),
                    unload_timeout=45.0,
                    reload_timeout=45.0,
                )
                if not reloaded or _loaded_process_missing_configured_mmproj(model_entry, catalog):
                    error_message = (
                        f"model '{model_name}' is configured with mmproj but the currently loaded llama-server process was started without it. "
                        "Automatic reload failed; unload/restart the model so llama-swap reloads it with the configured --mmproj before sending image inputs."
                    )
                    log_api_event(
                        "openai_responses_image_rejected_stale_mmproj_runtime",
                        {
                            "model": model_name,
                            "request_id": request_id,
                            "configured_mmproj": str(model_entry.mmproj_path or ""),
                            "process": get_catalog_model_process(model_entry.model_id, catalog),
                            "payload": _summarize_api_payload_for_log(upstream_payload),
                        },
                    )
                    self._send_json(
                        {
                            "error": {
                                "message": error_message,
                                "type": "stale_runtime_error",
                            }
                        },
                        status=503,
                    )
                    if upstream_model_name:
                        REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                    return
                log_api_event(
                    "openai_responses_image_stale_mmproj_runtime_reloaded",
                    {"model": model_name, "request_id": request_id, "configured_mmproj": str(model_entry.mmproj_path or "")},
                )
            stream = bool(payload.get("stream"))
            chat_continue_repair_cfg = resolve_chat_tool_continue_repair_config(args)
            chat_continue_repair_enabled = bool(chat_continue_repair_cfg.get("enabled")) and bool(upstream_payload.get("tools"))
            try:
                chat_continue_repair_max_rounds = max(0, int(chat_continue_repair_cfg.get("max_rounds", 1)))
            except Exception:
                chat_continue_repair_max_rounds = 1
            chat_continue_repair_trigger_prefixes = _normalize_chat_tool_continue_trigger_prefixes(chat_continue_repair_cfg.get("trigger_prefixes"))
            chat_continue_repair_prompt_template = chat_continue_repair_cfg.get("prompt")
            chat_continue_repair_include_failed_assistant_message = bool(chat_continue_repair_cfg.get("include_failed_assistant_message"))
            if tool_registry is not None and tool_registry.has_deferred_tools:
                extra_messages: list[dict] = []
                accumulated_usage: dict | None = None
                rounds = 0
                tool_repair_rounds = 0
                empty_final_repair_rounds = 0
                seen_tool_repair_signatures: set[str] = set()
                max_internal_tool_search_rounds = 2
                max_internal_tool_repair_rounds = 2
                force_tool_choice_next_round = False
                sse_started = False

                def fail_internal_response(message: str, *, status: int = 502, error_type: str = "server_error", extra: dict | None = None) -> None:
                    if upstream_model_name:
                        REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                    if stream and sse_started:
                        _write_responses_sse_error(self, message, error_type)
                        return
                    payload_error = {"message": message, "type": error_type}
                    if extra:
                        payload_error.update(extra)
                    self._send_json({"error": payload_error}, status=status)

                if stream:
                    _start_responses_sse_stream(self)
                    sse_started = True
                    _write_sse_comment(self, "heimdall-gateway internal Responses fallback started")
                while True:
                    internal_payload = _responses_payload_to_chat_payload(
                        payload,
                        upstream_model_name,
                        flatten_namespace_tools=flatten_namespace_tools,
                        tool_registry=tool_registry,
                        extra_messages=extra_messages,
                        allow_tool_output_images=allow_tool_output_images,
                    )
                    if force_tool_choice_next_round:
                        internal_payload = _force_tool_choice_for_chat_repair(
                            _apply_chat_tool_continue_repair_token_cap(internal_payload, chat_continue_repair_cfg.get("max_tokens"))
                        )
                    # Keep tool discovery KV-stable and private.  For streaming
                    # clients we collect the final response and emit Responses SSE
                    # only after internal tool_search rounds have completed.
                    internal_payload["stream"] = False
                    if extra_messages and "max_tokens" not in internal_payload:
                        internal_payload["max_tokens"] = _responses_internal_round_max_tokens()
                        log_api_event(
                            "openai_responses_internal_round_max_tokens_applied",
                            {
                                "model": model_name,
                                "request_id": request_id,
                                "round": rounds,
                                "extra_messages": len(extra_messages),
                                "max_tokens": internal_payload["max_tokens"],
                            },
                        )
                    if stream and sse_started:
                        _write_sse_comment(self, f"internal round {rounds}; repair_rounds={tool_repair_rounds}; extra_messages={len(extra_messages)}")
                    log_api_event(
                        "openai_responses_tool_search_round_payload",
                        {
                            "model": model_name,
                            "request_id": request_id,
                            "round": rounds,
                            "tool_count": len(internal_payload.get("tools", [])),
                            "extra_messages": len(extra_messages),
                            "payload": _summarize_api_payload_for_log(internal_payload),
                        },
                    )
                    try:
                        response = requests.post(
                            f"http://{client_host}:{int(args.public_port)}/v1/chat/completions",
                            data=json.dumps(internal_payload).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            timeout=(60, 600),
                            stream=False,
                        )
                        log_api_event(
                            "openai_responses_tool_search_round_headers",
                            {
                                "model": model_name,
                                "request_id": request_id,
                                "round": rounds,
                                "status": response.status_code,
                                "wait_ms": _elapsed_ms(started_at),
                            },
                        )
                    except requests.RequestException as exc:
                        log_api_event(
                            "openai_responses_tool_search_round_network_error",
                            {"model": model_name, "request_id": request_id, "round": rounds, "error": str(exc)},
                        )
                        fail_internal_response(f"upstream unavailable: {exc}", status=502, error_type="server_error")
                        return
                    if response.status_code >= 400:
                        body_text = response.text
                        log_api_event(
                            "openai_responses_tool_search_round_upstream_error",
                            {
                                "model": model_name,
                                "request_id": request_id,
                                "round": rounds,
                                "status": response.status_code,
                                "body_preview": body_text[:2000],
                                "payload": _summarize_api_payload_for_log(internal_payload),
                            },
                        )
                        message = f"upstream HTTP {response.status_code}: {body_text[:1000]}" if body_text else f"upstream HTTP {response.status_code}"
                        fail_internal_response(message, status=response.status_code, error_type="upstream_error", extra={"upstream_status": response.status_code})
                        return
                    try:
                        data = response.json()
                    except Exception as exc:
                        log_api_event(
                            "openai_responses_tool_search_round_invalid_json",
                            {"model": model_name, "request_id": request_id, "round": rounds, "error": str(exc), "body_len": len(response.text)},
                        )
                        fail_internal_response(f"upstream invalid response: {exc}", status=502, error_type="server_error")
                        return

                    accumulated_usage = _combine_responses_usage(accumulated_usage, data.get("usage"))
                    followup_messages = _chat_response_internal_tool_search_followup_messages(data, tool_registry)
                    if followup_messages:
                        if rounds >= max_internal_tool_search_rounds:
                            log_api_event(
                                "openai_responses_tool_search_round_limit",
                                {"model": model_name, "request_id": request_id, "rounds": rounds, "followup_messages": len(followup_messages)},
                            )
                            fail_internal_response("internal tool_search round limit exceeded; narrow the deferred tool search", status=500, error_type="server_error")
                            return
                        extra_messages.extend(followup_messages)
                        if stream and sse_started:
                            _write_sse_comment(self, "internal tool_search result appended")
                        rounds += 1
                        continue

                    repair_followup_messages = _chat_response_internal_tool_repair_followup_messages(
                        data,
                        tool_registry,
                        loaded_schema_messages=extra_messages,
                    )
                    if repair_followup_messages:
                        repair_feedbacks = [
                            message for message in repair_followup_messages
                            if isinstance(message, dict) and message.get("role") == "tool"
                        ]
                        repair_signature = ""
                        if repair_feedbacks:
                            repair_signature = str(repair_feedbacks[0].get("content") or "")[:1000]
                        if tool_repair_rounds >= max_internal_tool_repair_rounds or repair_signature in seen_tool_repair_signatures:
                            log_api_event(
                                "openai_responses_tool_repair_limit",
                                {
                                    "model": model_name,
                                    "request_id": request_id,
                                    "repair_rounds": tool_repair_rounds,
                                    "repeated": repair_signature in seen_tool_repair_signatures,
                                    "feedback_preview": repair_signature[:500],
                                },
                            )
                            fail_internal_response(
                                "internal tool repair limit exceeded; model repeated an invalid tool call",
                                status=502,
                                error_type="upstream_tool_call_error",
                                extra={"feedback": repair_signature[:1000]},
                            )
                            return
                        if repair_signature:
                            seen_tool_repair_signatures.add(repair_signature)
                        log_api_event(
                            "openai_responses_tool_repair_feedback",
                            {
                                "model": model_name,
                                "request_id": request_id,
                                "repair_round": tool_repair_rounds,
                                "feedback_count": len(repair_feedbacks),
                                "feedback_preview": repair_signature[:500],
                            },
                        )
                        extra_messages.extend(repair_followup_messages)
                        if stream and sse_started:
                            _write_sse_comment(self, "internal tool repair feedback appended")
                        tool_repair_rounds += 1
                        continue

                    data, translated_internal_call = _translate_internal_deferred_tool_calls_in_chat_response(data, tool_registry, loaded_schema_messages=extra_messages)
                    if accumulated_usage is not None:
                        data["usage"] = accumulated_usage
                    final_payload = _chat_response_to_responses_payload(data, model_name, payload)
                    final_has_output = _responses_payload_has_output_items(final_payload)
                    if (
                        chat_continue_repair_enabled
                        and not final_has_output
                        and not translated_internal_call
                        and empty_final_repair_rounds < chat_continue_repair_max_rounds
                    ):
                        state = _chat_completion_state_from_payload(data)
                        trigger_reason = _chat_tool_continue_trigger_reason(
                            state.get("content") or "",
                            state.get("tool_calls") or [],
                            internal_payload.get("tools") or [],
                            chat_continue_repair_trigger_prefixes,
                        ) or "responses_empty_final"
                        current_messages = internal_payload.get("messages") if isinstance(internal_payload.get("messages"), list) else []
                        examples = _extract_chat_tool_call_examples_from_messages(current_messages, internal_payload.get("tools"), limit=1)
                        repair_messages = _chat_tool_continue_repair_messages(
                            current_messages,
                            state.get("message") if isinstance(state.get("message"), dict) else {"role": "assistant", "content": ""},
                            internal_payload.get("tools") or [],
                            chat_continue_repair_prompt_template,
                            chat_continue_repair_include_failed_assistant_message,
                            examples[0] if examples else None,
                        )
                        added_messages = repair_messages[len(current_messages):] if len(repair_messages) >= len(current_messages) else repair_messages
                        extra_messages.extend(added_messages)
                        force_tool_choice_next_round = True
                        log_api_event(
                            "openai_responses_tool_fix_triggered",
                            {
                                "model": model_name,
                                "request_id": request_id,
                                "round": rounds,
                                "repair_round": empty_final_repair_rounds,
                                "trigger_reason": trigger_reason,
                                "translated_internal_call": translated_internal_call,
                                "output_text_len": len(str(final_payload.get("output_text") or "")),
                                "output_items": len(final_payload.get("output") or []) if isinstance(final_payload.get("output"), list) else 0,
                                "usage": final_payload.get("usage"),
                            },
                        )
                        if stream and sse_started:
                            _write_sse_comment(self, f"internal empty tool response repair {empty_final_repair_rounds + 1}")
                        empty_final_repair_rounds += 1
                        continue
                    if chat_continue_repair_enabled and not final_has_output and not translated_internal_call and empty_final_repair_rounds >= chat_continue_repair_max_rounds:
                        log_api_event(
                            "openai_responses_tool_fix_exhausted",
                            {
                                "model": model_name,
                                "request_id": request_id,
                                "rounds": rounds,
                                "repair_rounds": empty_final_repair_rounds,
                                "usage": final_payload.get("usage"),
                            },
                        )
                    if (tool_repair_rounds > 0 or empty_final_repair_rounds > 0) and not final_has_output:
                        log_api_event(
                            "openai_responses_tool_repair_empty_final",
                            {
                                "model": model_name,
                                "request_id": request_id,
                                "rounds": rounds,
                                "repair_rounds": tool_repair_rounds,
                                "empty_final_repair_rounds": empty_final_repair_rounds,
                                "usage": final_payload.get("usage"),
                            },
                        )
                        fail_internal_response(
                            "model stopped after internal tool repair feedback without producing text or a valid tool call",
                            status=502,
                            error_type="upstream_tool_call_error",
                        )
                        return
                    if upstream_model_name:
                        REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=True)
                        REPLICA_ROUTER_STATE.bind_response(str(final_payload.get("id") or ""), upstream_model_name, get_model_replica_config(model_entry).sticky_ttl_s if model_entry else 3600)
                    log_api_event(
                        "openai_responses_tool_search_final",
                        {
                            "model": model_name,
                            "request_id": request_id,
                            "rounds": rounds,
                            "translated_internal_call": translated_internal_call,
                            "output_text_len": len(str(final_payload.get("output_text") or "")),
                            "usage": final_payload.get("usage"),
                        },
                    )
                    mark_model_activity(model_name, f"openai_responses:{request_id}", "response_done")
                    if stream:
                        if sse_started:
                            _write_responses_sse_events(self, final_payload)
                        else:
                            _write_responses_sse(self, final_payload)
                    else:
                        self._send_json(final_payload)
                    return
            try:
                response = requests.post(
                    f"http://{client_host}:{int(args.public_port)}/v1/chat/completions",
                    data=json.dumps(upstream_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    timeout=(60, 600),
                    stream=stream,
                )
                log_api_event(
                    "openai_responses_upstream_headers",
                    {
                        "model": model_name,
                        "request_id": request_id,
                        "status": response.status_code,
                        "wait_ms": _elapsed_ms(started_at),
                        "stream": stream,
                    },
                )
            except requests.RequestException as exc:
                log_api_event(
                    "openai_responses_upstream_network_error",
                    {
                        "model": model_name,
                        "request_id": request_id,
                        "error": str(exc),
                        "payload": _summarize_api_payload_for_log(upstream_payload),
                    },
                )
                log_model_runtime_snapshot(
                    "openai_responses_runtime_after_upstream_network_error",
                    model_name,
                    catalog,
                    client_host,
                    int(args.public_port),
                    request_id=request_id,
                )
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                self._send_json({"error": {"message": f"upstream unavailable: {exc}", "type": "server_error"}}, status=502)
                return

            if response.status_code >= 400:
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                body_text = response.text
                log_api_event(
                    "openai_responses_upstream_error",
                    {
                        "model": model_name,
                        "request_id": request_id,
                        "status": response.status_code,
                        "body_len": len(body_text),
                        "body_preview": body_text[:2000],
                        "payload": _summarize_api_payload_for_log(upstream_payload),
                    },
                )
                log_model_runtime_snapshot(
                    "openai_responses_runtime_after_upstream_http_error",
                    model_name,
                    catalog,
                    client_host,
                    int(args.public_port),
                    request_id=request_id,
                    upstream_status=response.status_code,
                )
                message = f"upstream HTTP {response.status_code}: {body_text[:1000]}" if body_text else f"upstream HTTP {response.status_code}"
                self._send_json(
                    {"error": {"message": message, "type": "upstream_error", "upstream_status": response.status_code}},
                    status=response.status_code,
                )
                return

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                
                write_lock = threading.Lock()
                first_chunk_sent = threading.Event()
                stop_heartbeat = threading.Event()
                
                generated_resp_id = f"resp_{uuid.uuid4().hex}"
                msg_item_id = f"msg_{uuid.uuid4().hex}"
                reasoning_item_id = f"rs_{uuid.uuid4().hex}"
                sequence_counter = [0]
                message_item_started = [False]
                reasoning_item_started = [False]
                
                def next_sequence() -> int:
                    sequence_counter[0] += 1
                    return sequence_counter[0]

                def stream_event(event_name: str, event_payload: dict) -> None:
                    _write_sse_event(self, event_name, event_payload)
                
                # We must track output_index carefully.
                # Do not reserve an index for text until text is actually emitted;
                # tool-only responses must have their function_call at output_index=0.
                output_index_counter = [0]
                msg_output_index: list[int | None] = [None]
                reasoning_output_index: list[int | None] = [None]

                def allocate_output_index() -> int:
                    output_index = output_index_counter[0]
                    output_index_counter[0] += 1
                    return output_index

                def send_pulse():
                    while not stop_heartbeat.is_set():
                        if stop_heartbeat.wait(15):
                            break
                        try:
                            with write_lock:
                                log_api_event("openai_responses_stream_pulse", {"request_id": request_id, "first_chunk_sent": first_chunk_sent.is_set()})
                                if not first_chunk_sent.is_set() or not message_item_started[0]:
                                    continue
                                
                                delta_payload = _responses_event(
                                    "response.output_text.delta",
                                    next_sequence(),
                                    item_id=msg_item_id,
                                    output_index=msg_output_index[0],
                                    content_index=0,
                                    delta="",
                                    logprobs=[],
                                )
                                stream_event("response.output_text.delta", delta_payload)
                        except Exception:
                            break

                heartbeat_thread = threading.Thread(target=send_pulse)
                heartbeat_thread.daemon = True
                heartbeat_thread.start()

                # Send response.created immediately
                with write_lock:
                    log_api_event("openai_responses_stream_start_immediate", {"request_id": request_id, "model": model_name})
                    base_payload = _chat_response_to_responses_payload({"choices": [{"message": {"role": "assistant", "content": ""}}]}, model_name, payload)
                    base_payload["id"] = generated_resp_id
                    initial_payload = _responses_event("response.created", next_sequence(), response=_response_in_progress(base_payload))
                    
                    stream_event("response.created", initial_payload)
                    first_chunk_sent.set()

                full_content = ""
                full_reasoning = ""
                latest_usage = None
                active_tool_calls = {} # tool_call_index -> {item_id, call_id, name, args_buf, output_index}
                in_loading_block = False

                try:
                    # Use decode_unicode=False to avoid 'requests' guessing wrong encoding
                    for line_bytes in response.iter_lines(decode_unicode=False):
                        if not line_bytes:
                            continue
                        
                        with write_lock:
                            if not line_bytes.startswith(b"data: "):
                                log_api_event("openai_responses_stream_raw_line", {"request_id": request_id, "line": str(line_bytes[:200])})
                                continue
                            
                            data_bytes = line_bytes[6:].strip()
                            if data_bytes == b"[DONE]":
                                log_api_event("openai_responses_stream_done_signal", {"request_id": request_id})
                                break
                            
                            try:
                                data_str = data_bytes.decode("utf-8", errors="replace")
                                chunk = json.loads(data_str)
                                usage = _normalize_responses_usage(chunk.get("usage"))
                                if isinstance(usage, dict):
                                    latest_usage = usage
                                    log_api_event("openai_responses_stream_usage_received", {"request_id": request_id, "usage": usage})
                                choices = chunk.get("choices")
                                if not isinstance(choices, list) or not choices:
                                    choices = [{}]
                                delta = choices[0].get("delta", {}) if isinstance(choices[0], dict) else {}
                                
                                content = str(delta.get("content") or "")
                                reasoning = ""
                                for field in ("reasoning_content", "reasoning"):
                                    val = delta.get(field)
                                    if val is not None:
                                        reasoning += str(val)
                                tool_calls = delta.get("tool_calls")
                                
                                if content or reasoning or tool_calls:
                                    log_api_event("openai_responses_stream_chunk_received", {
                                        "request_id": request_id,
                                        "has_content": bool(content),
                                        "content_len": len(content),
                                        "content_preview": content[:100],
                                        "has_reasoning": bool(reasoning),
                                        "reasoning_len": len(reasoning),
                                        "reasoning_preview": reasoning[:100],
                                        "has_tool_calls": bool(tool_calls)
                                    })

                                if reasoning:
                                    if not reasoning_item_started[0]:
                                        reasoning_output_index[0] = allocate_output_index()
                                        reasoning_item_started[0] = True
                                        stream_event(
                                            "response.output_item.added",
                                            _responses_event(
                                                "response.output_item.added",
                                                next_sequence(),
                                                output_index=reasoning_output_index[0],
                                                item=_response_reasoning_item(reasoning_item_id, "", "in_progress"),
                                            ),
                                        )
                                    full_reasoning += reasoning
                                    stream_event(
                                        "response.reasoning_text.delta",
                                        _responses_event(
                                            "response.reasoning_text.delta",
                                            next_sequence(),
                                            item_id=reasoning_item_id,
                                            output_index=reasoning_output_index[0],
                                            content_index=0,
                                            delta=reasoning,
                                        ),
                                    )

                                if content:
                                    # Filter out llama-swap loading animations
                                    if "━━━━━\nllama-swap loading model" in content or in_loading_block:
                                        in_loading_block = True
                                        if "━━━━━\n" in content and "llama-swap" not in content.split("━━━━━\n")[-1]:
                                            # Found the end of the block
                                            content = content.split("━━━━━\n")[-1]
                                            in_loading_block = False
                                        else:
                                            content = ""

                                if content:
                                    if not message_item_started[0]:
                                        msg_output_index[0] = allocate_output_index()
                                        message_item_started[0] = True
                                        stream_event(
                                            "response.output_item.added",
                                            _responses_event(
                                                "response.output_item.added",
                                                next_sequence(),
                                                output_index=msg_output_index[0],
                                                item=_response_message_item(msg_item_id, "", "in_progress"),
                                            ),
                                        )
                                        stream_event(
                                            "response.content_part.added",
                                            _responses_event(
                                                "response.content_part.added",
                                                next_sequence(),
                                                item_id=msg_item_id,
                                                output_index=msg_output_index[0],
                                                content_index=0,
                                                part={"type": "output_text", "text": "", "annotations": []},
                                            ),
                                        )
                                    full_content += content
                                    delta_payload = _responses_event(
                                        "response.output_text.delta",
                                        next_sequence(),
                                        item_id=msg_item_id,
                                        output_index=msg_output_index[0],
                                        content_index=0,
                                        delta=content,
                                        logprobs=[],
                                    )
                                    stream_event("response.output_text.delta", delta_payload)

                                if isinstance(tool_calls, list):
                                    for tc in tool_calls:
                                        idx = tc.get("index", 0)
                                        function_delta = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
                                        args_delta = str(function_delta.get("arguments") or "")
                                        name_delta = str(function_delta.get("name") or "")
                                        if idx not in active_tool_calls:
                                            active_tool_calls[idx] = {
                                                "item_id": f"fc_{uuid.uuid4().hex}",
                                                "call_id": tc.get("id") or f"call_{uuid.uuid4().hex}",
                                                "name": "",
                                                "responses_name": "",
                                                "namespace": "",
                                                "args_buf": "",
                                                "output_index": None,
                                                "started": False,
                                                "blocked": False,
                                            }
                                        tc_state = active_tool_calls[idx]
                                        if name_delta and not tc_state["name"]:
                                            tc_state["name"] = name_delta
                                            mapped_tool = namespace_tool_map.get(name_delta, {}) if isinstance(namespace_tool_map, dict) else {}
                                            tc_state["responses_name"] = str(mapped_tool.get("name") or name_delta)
                                            tc_state["namespace"] = str(mapped_tool.get("namespace") or "")
                                            if allowed_legacy_tool_names and name_delta not in allowed_legacy_tool_names:
                                                tc_state["blocked"] = True
                                                log_api_event(
                                                    "openai_responses_tool_call_blocked_not_forwarded",
                                                    {
                                                        "request_id": request_id,
                                                        "name": name_delta,
                                                        "call_id": tc_state["call_id"],
                                                        "allowed_tool_names": sorted(name for name in allowed_legacy_tool_names if name)[:80],
                                                    },
                                                )
                                        if args_delta:
                                            tc_state["args_buf"] += args_delta

                                        if (
                                            tc_state["name"]
                                            and not tc_state["blocked"]
                                            and not tc_state["started"]
                                        ):
                                            tc_state["output_index"] = allocate_output_index()
                                            tc_state["started"] = True
                                            stream_event(
                                                "response.output_item.added",
                                                _responses_event(
                                                    "response.output_item.added",
                                                    next_sequence(),
                                                    output_index=tc_state["output_index"],
                                                    item=_response_function_call_item(
                                                        tc_state["item_id"],
                                                        tc_state["call_id"],
                                                        str(tc_state.get("responses_name") or tc_state["name"] or ""),
                                                        "",
                                                        "in_progress",
                                                        str(tc_state.get("namespace") or "") or None,
                                                    ),
                                                ),
                                            )

                                mark_model_activity(model_name, f"openai_responses:{request_id}", "stream_chunk", log=False)
                            except Exception as exc:
                                log_api_event("openai_responses_stream_parse_error", {"request_id": request_id, "error": str(exc), "data_len": len(data_bytes)})
                                continue
                    
                    responses_stream_repair_rounds = 0
                    if chat_continue_repair_enabled and chat_continue_repair_max_rounds > 0:
                        visible_tool_calls_started = any(
                            bool(tc_info.get("started")) and not bool(tc_info.get("blocked"))
                            for tc_info in active_tool_calls.values()
                            if isinstance(tc_info, dict)
                        )
                        trigger_reason = _chat_tool_continue_trigger_reason(
                            full_content,
                            [{}] if visible_tool_calls_started else [],
                            upstream_payload.get("tools"),
                            chat_continue_repair_trigger_prefixes,
                        )
                        if trigger_reason:
                            responses_stream_repair_rounds = 1
                            log_api_event(
                                "openai_chat_tool_continue_repair_triggered",
                                {
                                    "request_id": request_id,
                                    "model": model_name,
                                    "upstream_model": upstream_model_name,
                                    "stream": True,
                                    "api": "responses",
                                    "visible_content_len": len(full_content),
                                    "reasoning_len": len(full_reasoning),
                                    "finish_reason": "stop",
                                    "trigger_reason": trigger_reason,
                                },
                            )
                            with write_lock:
                                _write_sse_comment(self, "internal chat tool continuation repair")
                            repair_payload = dict(upstream_payload)
                            repair_payload["stream"] = False
                            repair_payload["messages"] = _chat_tool_continue_repair_messages(
                                upstream_payload.get("messages") if isinstance(upstream_payload.get("messages"), list) else [],
                                {"role": "assistant", "content": full_content},
                                upstream_payload.get("tools"),
                                chat_continue_repair_prompt_template,
                                chat_continue_repair_include_failed_assistant_message,
                            )
                            log_api_event(
                                "openai_chat_tool_continue_repair_round",
                                {
                                    "request_id": request_id,
                                    "model": model_name,
                                    "upstream_model": upstream_model_name,
                                    "round": responses_stream_repair_rounds,
                                    "stream": True,
                                    "api": "responses",
                                    "trigger_reason": trigger_reason,
                                },
                            )
                            try:
                                repair_response = requests.post(
                                    f"http://{client_host}:{int(args.public_port)}/v1/chat/completions",
                                    data=json.dumps(repair_payload).encode("utf-8"),
                                    headers={"Content-Type": "application/json"},
                                    timeout=(60, 600),
                                    stream=False,
                                )
                                log_api_event(
                                    "openai_responses_upstream_headers",
                                    {
                                        "model": model_name,
                                        "request_id": request_id,
                                        "status": repair_response.status_code,
                                        "wait_ms": _elapsed_ms(started_at),
                                        "stream": False,
                                        "repair_round": responses_stream_repair_rounds,
                                    },
                                )
                            except requests.RequestException as exc:
                                log_api_event(
                                    "openai_chat_upstream_network_error",
                                    {
                                        "request_id": request_id,
                                        "error": str(exc),
                                        "payload": _summarize_api_payload_for_log(repair_payload),
                                        "router_state": replica_trace_state_for_base(model_name),
                                        "repair_round": responses_stream_repair_rounds,
                                        "api": "responses",
                                    },
                                )
                                repair_response = None
                            if repair_response is not None and repair_response.status_code < 400:
                                try:
                                    repair_data = repair_response.json()
                                except Exception as exc:
                                    log_api_event(
                                        "openai_chat_upstream_invalid_json",
                                        {
                                            "request_id": request_id,
                                            "error": str(exc),
                                            "payload": _summarize_api_payload_for_log(repair_payload),
                                            "body": repair_response.text[:4000],
                                            "router_state": replica_trace_state_for_base(model_name),
                                            "repair_round": responses_stream_repair_rounds,
                                            "api": "responses",
                                        },
                                    )
                                    repair_data = {}
                                repair_state = _chat_completion_state_from_payload(repair_data) if isinstance(repair_data, dict) else {}
                                repair_content = str(repair_state.get("content") or "")
                                repair_tool_calls = repair_state.get("tool_calls") or []
                                if repair_content:
                                    with write_lock:
                                        if not message_item_started[0]:
                                            msg_output_index[0] = allocate_output_index()
                                            message_item_started[0] = True
                                            stream_event(
                                                "response.output_item.added",
                                                _responses_event(
                                                    "response.output_item.added",
                                                    next_sequence(),
                                                    output_index=msg_output_index[0],
                                                    item=_response_message_item(msg_item_id, "", "in_progress"),
                                                ),
                                            )
                                            stream_event(
                                                "response.content_part.added",
                                                _responses_event(
                                                    "response.content_part.added",
                                                    next_sequence(),
                                                    item_id=msg_item_id,
                                                    output_index=msg_output_index[0],
                                                    content_index=0,
                                                    part={"type": "output_text", "text": "", "annotations": []},
                                                ),
                                            )
                                        full_content += repair_content
                                        stream_event(
                                            "response.output_text.delta",
                                            _responses_event(
                                                "response.output_text.delta",
                                                next_sequence(),
                                                item_id=msg_item_id,
                                                output_index=msg_output_index[0],
                                                content_index=0,
                                                delta=repair_content,
                                                logprobs=[],
                                            ),
                                        )
                                if isinstance(repair_tool_calls, list):
                                    with write_lock:
                                        for tc in repair_tool_calls:
                                            if not isinstance(tc, dict):
                                                continue
                                            function_payload = tc.get("function", {}) if isinstance(tc.get("function"), dict) else {}
                                            legacy_name = str(function_payload.get("name") or "").strip()
                                            if not legacy_name:
                                                continue
                                            mapped_tool = namespace_tool_map.get(legacy_name, {}) if isinstance(namespace_tool_map, dict) else {}
                                            responses_name = str(mapped_tool.get("name") or legacy_name)
                                            namespace = str(mapped_tool.get("namespace") or "") or None
                                            blocked = bool(allowed_legacy_tool_names and legacy_name not in allowed_legacy_tool_names)
                                            item_id = str(tc.get("responses_item_id") or tc.get("id") or f"fc_{uuid.uuid4().hex}")
                                            call_id = str(tc.get("call_id") or tc.get("id") or f"call_{uuid.uuid4().hex}")
                                            output_index = allocate_output_index()
                                            tc_state = {
                                                "item_id": item_id,
                                                "call_id": call_id,
                                                "name": legacy_name,
                                                "responses_name": responses_name,
                                                "namespace": namespace or "",
                                                "args_buf": str(function_payload.get("arguments") or ""),
                                                "output_index": output_index,
                                                "started": not blocked,
                                                "blocked": blocked,
                                            }
                                            active_tool_calls[f"repair_{len(active_tool_calls)}"] = tc_state
                                            if blocked:
                                                log_api_event(
                                                    "openai_responses_tool_call_blocked_not_forwarded",
                                                    {
                                                        "request_id": request_id,
                                                        "name": legacy_name,
                                                        "call_id": call_id,
                                                        "allowed_tool_names": sorted(name for name in allowed_legacy_tool_names if name)[:80],
                                                        "repair_round": responses_stream_repair_rounds,
                                                    },
                                                )
                                                continue
                                            stream_event(
                                                "response.output_item.added",
                                                _responses_event(
                                                    "response.output_item.added",
                                                    next_sequence(),
                                                    output_index=output_index,
                                                    item=_response_function_call_item(
                                                        item_id,
                                                        call_id,
                                                        responses_name,
                                                        "",
                                                        "in_progress",
                                                        namespace,
                                                    ),
                                                ),
                                            )
                                repaired_tool_calls_started = any(
                                    bool(tc_info.get("started")) and not bool(tc_info.get("blocked"))
                                    for tc_info in active_tool_calls.values()
                                    if isinstance(tc_info, dict)
                                )
                                exhausted_reason = _chat_tool_continue_trigger_reason(
                                    full_content,
                                    [{}] if repaired_tool_calls_started else [],
                                    upstream_payload.get("tools"),
                                    chat_continue_repair_trigger_prefixes,
                                )
                                if exhausted_reason:
                                    log_api_event(
                                        "openai_chat_tool_continue_repair_exhausted",
                                        {
                                            "request_id": request_id,
                                            "model": model_name,
                                            "upstream_model": upstream_model_name,
                                            "rounds": responses_stream_repair_rounds,
                                            "stream": True,
                                            "api": "responses",
                                            "visible_content_len": len(full_content),
                                            "reasoning_len": len(full_reasoning),
                                            "finish_reason": "stop",
                                            "trigger_reason": exhausted_reason,
                                        },
                                    )
                            elif repair_response is not None:
                                body_text = repair_response.text[:4000]
                                log_api_event(
                                    "openai_chat_upstream_error",
                                    {
                                        "request_id": request_id,
                                        "status": repair_response.status_code,
                                        "body": body_text,
                                        "payload": _summarize_api_payload_for_log(repair_payload),
                                        "router_state": replica_trace_state_for_base(model_name),
                                        "repair_round": responses_stream_repair_rounds,
                                        "api": "responses",
                                    },
                                )

                    # End of stream
                    with write_lock:
                        if full_reasoning and reasoning_item_started[0]:
                            stream_event(
                                "response.reasoning_text.done",
                                _responses_event(
                                    "response.reasoning_text.done",
                                    next_sequence(),
                                    item_id=reasoning_item_id,
                                    output_index=reasoning_output_index[0],
                                    content_index=0,
                                    text=full_reasoning,
                                ),
                            )
                            stream_event(
                                "response.output_item.done",
                                _responses_event(
                                    "response.output_item.done",
                                    next_sequence(),
                                    output_index=reasoning_output_index[0],
                                    item=_response_reasoning_item(reasoning_item_id, full_reasoning, "completed"),
                                ),
                            )

                        if full_content and message_item_started[0]:
                            stream_event(
                                "response.output_text.done",
                                _responses_event(
                                    "response.output_text.done",
                                    next_sequence(),
                                    item_id=msg_item_id,
                                    output_index=msg_output_index[0],
                                    content_index=0,
                                    text=full_content,
                                    logprobs=[],
                                ),
                            )
                            stream_event(
                                "response.content_part.done",
                                _responses_event(
                                    "response.content_part.done",
                                    next_sequence(),
                                    item_id=msg_item_id,
                                    output_index=msg_output_index[0],
                                    content_index=0,
                                    part={"type": "output_text", "text": full_content, "annotations": []},
                                ),
                            )
                            stream_event(
                                "response.output_item.done",
                                _responses_event(
                                    "response.output_item.done",
                                    next_sequence(),
                                    output_index=msg_output_index[0],
                                    item=_response_message_item(msg_item_id, full_content, "completed"),
                                ),
                            )
                        
                        # Pack everything into response.completed
                        tc_list = []
                        for tc_info in active_tool_calls.values():
                            if tc_info.get("blocked") or not tc_info.get("started"):
                                log_api_event(
                                    "openai_responses_tool_call_omitted_from_completed",
                                    {
                                        "request_id": request_id,
                                        "name": str(tc_info.get("name") or ""),
                                        "call_id": str(tc_info.get("call_id") or ""),
                                        "blocked": bool(tc_info.get("blocked")),
                                        "started": bool(tc_info.get("started")),
                                        "arguments_len": len(str(tc_info.get("args_buf") or "")),
                                    },
                                )
                                continue
                            responses_name = str(tc_info.get("responses_name") or tc_info["name"] or "")
                            namespace = str(tc_info.get("namespace") or "") or None
                            raw_arguments = str(tc_info.get("args_buf") or "")
                            final_arguments, arguments_sanitized = _sanitize_responses_tool_arguments(
                                responses_name,
                                namespace,
                                raw_arguments,
                            )
                            if arguments_sanitized:
                                log_api_event(
                                    "openai_responses_stream_tool_arguments_sanitized",
                                    {
                                        "request_id": request_id,
                                        "item_id": tc_info["item_id"],
                                        "call_id": tc_info["call_id"],
                                        "name": responses_name,
                                        "namespace": namespace or "",
                                        "raw_arguments_len": len(raw_arguments),
                                        "raw_arguments_preview": raw_arguments[:500],
                                        "final_arguments_len": len(final_arguments),
                                        "final_arguments_preview": final_arguments[:500],
                                    },
                                )
                            if final_arguments:
                                stream_event(
                                    "response.function_call_arguments.delta",
                                    _responses_event(
                                        "response.function_call_arguments.delta",
                                        next_sequence(),
                                        item_id=tc_info["item_id"],
                                        output_index=tc_info["output_index"],
                                        delta=final_arguments,
                                    ),
                                )
                            stream_event(
                                "response.function_call_arguments.done",
                                _responses_event(
                                    "response.function_call_arguments.done",
                                    next_sequence(),
                                    item_id=tc_info["item_id"],
                                    output_index=tc_info["output_index"],
                                    name=responses_name,
                                    arguments=final_arguments,
                                ),
                            )
                            stream_event(
                                "response.output_item.done",
                                _responses_event(
                                    "response.output_item.done",
                                    next_sequence(),
                                    output_index=tc_info["output_index"],
                                    item=_response_function_call_item(
                                        tc_info["item_id"],
                                        tc_info["call_id"],
                                        responses_name,
                                        final_arguments,
                                        "completed",
                                        namespace,
                                    ),
                                ),
                            )
                            log_api_event(
                                "openai_responses_stream_tool_call_completed",
                                {
                                    "request_id": request_id,
                                    "item_id": tc_info["item_id"],
                                    "call_id": tc_info["call_id"],
                                    "name": responses_name,
                                    "namespace": namespace or "",
                                    "legacy_name": str(tc_info["name"] or ""),
                                    "arguments_len": len(final_arguments),
                                    "arguments_preview": final_arguments[:500],
                                    "output_index": tc_info["output_index"],
                                },
                            )
                            tc_entry = {
                                "id": tc_info["call_id"],
                                "responses_item_id": tc_info["item_id"],
                                "type": "function",
                                "function": {
                                    "name": responses_name,
                                    "arguments": final_arguments,
                                },
                            }
                            if namespace:
                                tc_entry["namespace"] = namespace
                                tc_entry["function"]["namespace"] = namespace
                            tc_list.append(tc_entry)
                            
                        final_message = {"role": "assistant", "content": full_content}
                        if full_reasoning:
                            final_message["reasoning_content"] = full_reasoning
                        if tc_list:
                            final_message["tool_calls"] = tc_list
                            
                        final_chat_payload = {"choices": [{"message": final_message}]}
                        if isinstance(latest_usage, dict):
                            final_chat_payload["usage"] = latest_usage
                        final_payload = _chat_response_to_responses_payload(final_chat_payload, model_name, payload)
                        final_payload["id"] = generated_resp_id
                        completed_payload = _responses_event("response.completed", next_sequence(), response=final_payload)
                        
                        stream_event("response.completed", completed_payload)
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        log_api_event("openai_responses_stream_completed_sent", {"request_id": request_id, "output_text_len": len(full_content), "tool_calls": len(tc_list), "usage": final_payload.get("usage")})

                    
                    if upstream_model_name:
                        REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=True)
                    mark_model_activity(model_name, f"openai_responses:{request_id}", "response_done")
                except Exception as exc:
                    log_api_event(
                        "openai_responses_stream_error",
                        {
                            "request_id": request_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(limit=8),
                        },
                    )
                    if upstream_model_name:
                        REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                finally:
                    stop_heartbeat.set()
                    heartbeat_thread.join(timeout=1.0)
                    response.close()
                return

            try:
                data = response.json()
            except Exception as exc:
                log_api_event(
                    "openai_responses_upstream_invalid_json",
                    {
                        "model": model_name,
                        "request_id": request_id,
                        "error": str(exc),
                        "payload": _summarize_api_payload_for_log(upstream_payload),
                        "body_len": len(response.text),
                    },
                )
                log_model_runtime_snapshot(
                    "openai_responses_runtime_after_upstream_invalid_json",
                    model_name,
                    catalog,
                    client_host,
                    int(args.public_port),
                    request_id=request_id,
                )
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                self._send_json({"error": {"message": f"upstream invalid response: {exc}", "type": "server_error"}}, status=502)
                return
            final_payload = _chat_response_to_responses_payload(data, model_name, payload)
            if upstream_model_name:
                REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=True)
                REPLICA_ROUTER_STATE.bind_response(str(final_payload.get("id") or ""), upstream_model_name, get_model_replica_config(model_entry).sticky_ttl_s if model_entry else 3600)
            log_api_event(
                "openai_responses_response",
                {
                    "model": model_name,
                    "request_id": request_id,
                    "output_text_len": len(str(final_payload.get("output_text") or "")),
                    "usage": final_payload.get("usage"),
                },
            )
            mark_model_activity(model_name, f"openai_responses:{request_id}", "response_done")
            if bool(payload.get("stream")):
                _write_responses_sse(self, final_payload)
            else:
                self._send_json(final_payload)

        def _handle_ollama_generate(self):
            payload = self._read_json_body()
            log_api_event("ollama_generate_request", payload)
            started_at = time.monotonic()
            catalog = load_catalog(catalog_path)
            model_name = resolve_catalog_model_name(str(payload.get("model") or "").strip(), catalog)
            if not model_name:
                self._send_json({"error": "model is required"}, status=400)
                return
            model_entry = next((item for item in catalog if item.model_id == model_name), None)
            upstream_model_name = model_name
            is_replica_request = False
            affinity_key = ""
            if model_entry is not None:
                replica_defaults = resolve_global_replica_config(args)
                published_model_ids = get_published_model_ids(client_host, int(args.public_port))
                sync_replica_runtime_state(catalog, args.config, replica_defaults)
                upstream_model_name, affinity_key, is_replica_request = select_replica_for_request(
                    model_entry,
                    payload,
                    self.headers,
                    replica_defaults,
                    published_model_ids,
                    catalog=catalog,
                    config_path=args.config,
                    server_path=args.llama_server,
                    idle_ttl=resolve_idle_ttl(args),
                    server_defaults=resolve_llama_server_defaults(args),
                    public_host=client_host,
                    public_port=int(args.public_port),
                )
            if self._reject_if_model_loading(upstream_model_name, catalog, public_model_name=model_name, is_replica=is_replica_request, api_style="ollama"):
                return
            if (not is_replica_request) and self._reject_if_gpu_busy(model_name, catalog, api_style="ollama", payload=payload):
                return
            mark_model_activity(model_name, "ollama_generate", "request_start")
            if upstream_model_name:
                REPLICA_ROUTER_STATE.request_started(upstream_model_name)
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
            if images and (model_entry is None or not _has_configured_mmproj_runtime(model_entry)):
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
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
                "model": upstream_model_name,
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
                    f"http://{client_host}:{int(args.public_port)}{endpoint}",
                    data=json.dumps(upstream_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    timeout=(60, 600),
                    stream=True,
                )
            except requests.RequestException as exc:
                log_api_event("ollama_generate_upstream_network_error", {"error": str(exc)})
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                self._send_json({"error": f"upstream unavailable: {exc}"}, status=502)
                return
            if response.status_code >= 400:
                body_text = response.text[:4000]
                log_api_event("ollama_generate_upstream_error", {"status": response.status_code, "body": body_text, "payload": upstream_payload})
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                self._send_json({"error": f"upstream unavailable: HTTP {response.status_code}: {body_text[:1000]}"}, status=502)
                return
            try:
                data = _collect_openai_sse_response(response)
            except Exception as exc:
                log_api_event("ollama_generate_upstream_invalid_json", {"status": response.status_code, "error": str(exc)})
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                self._send_json({"error": f"upstream invalid response: {exc}"}, status=502)
                return
            if upstream_model_name:
                REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=True)
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            base = {
                "model": model_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "response": text,
                "done": True,
                "done_reason": "stop",
                "context": [],
                "total_duration": int(_elapsed_ms(started_at) * 1_000_000),
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
            started_at = time.monotonic()
            model_name_raw = str(payload.get("model") or "").strip()
            catalog = load_catalog(catalog_path)
            model_name = resolve_catalog_model_name(model_name_raw, catalog)
            text_input = payload.get("input")
            if text_input is None:
                text_input = payload.get("prompt")
            if not model_name or text_input is None:
                self._send_json({"error": "model and input are required"}, status=400)
                return
            model_entry = next((item for item in catalog if item.model_id == model_name), None)
            upstream_model_name = model_name
            is_replica_request = False
            affinity_key = ""
            if model_entry is not None:
                replica_defaults = resolve_global_replica_config(args)
                published_model_ids = get_published_model_ids(client_host, int(args.public_port))
                sync_replica_runtime_state(catalog, args.config, replica_defaults)
                upstream_model_name, affinity_key, is_replica_request = select_replica_for_request(
                    model_entry,
                    payload,
                    self.headers,
                    replica_defaults,
                    published_model_ids,
                    catalog=catalog,
                    config_path=args.config,
                    server_path=args.llama_server,
                    idle_ttl=resolve_idle_ttl(args),
                    server_defaults=resolve_llama_server_defaults(args),
                    public_host=client_host,
                    public_port=int(args.public_port),
                )
            if self._reject_if_model_loading(upstream_model_name, catalog, public_model_name=model_name, is_replica=is_replica_request, api_style="ollama"):
                return
            if (not is_replica_request) and self._reject_if_gpu_busy(model_name, catalog, api_style="ollama", payload=payload):
                return
            mark_model_activity(model_name, "ollama_embeddings", "request_start")
            if upstream_model_name:
                REPLICA_ROUTER_STATE.request_started(upstream_model_name)
            upstream_payload = {"model": upstream_model_name, "input": text_input}
            try:
                response = _proxy_request_to_public_api(
                    "POST",
                    "/v1/embeddings",
                    body=json.dumps(upstream_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    host=client_host,
                    port=int(args.public_port),
                )
                data = response.json()
            except requests.RequestException as exc:
                log_api_event("ollama_embeddings_upstream_network_error", {"error": str(exc)})
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                self._send_json({"error": f"upstream unavailable: {exc}"}, status=502)
                return
            except ValueError as exc:
                log_api_event("ollama_embeddings_upstream_invalid_json", {"status": response.status_code if 'response' in locals() else None, "error": str(exc)})
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                self._send_json({"error": f"upstream invalid response: {exc}"}, status=502)
                return
            if response.status_code >= 400:
                log_api_event("ollama_embeddings_upstream_error", {"status": response.status_code, "body": response.text[:4000], "payload": upstream_payload})
                if upstream_model_name:
                    REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=False)
                self._send_json({"error": f"upstream unavailable: HTTP {response.status_code}: {response.text[:1000]}"}, status=502)
                return
            if upstream_model_name:
                REPLICA_ROUTER_STATE.request_finished(upstream_model_name, ok=True)
            embeddings = data.get("data", [])
            if len(embeddings) == 1:
                mark_model_activity(model_name, "ollama_embeddings", "response_done")
                self._send_json({"embedding": embeddings[0].get("embedding", [])})
                return
            mark_model_activity(model_name, "ollama_embeddings", "response_done")
            self._send_json({"embeddings": [item.get("embedding", []) for item in embeddings]})


    try:
        if bool(api_https.get("enabled")):
            server = HTTPSRedirectingThreadingHTTPServer((bind_host, port), Handler, api_https)
        else:
            server = ThreadingHTTPServer((bind_host, port), Handler)
    except Exception as e:
        scheme = "https" if bool(api_https.get("enabled")) else "http"
        print(f"[!] Could not start ctx metadata server on {scheme}://{bind_host}:{port}: {e}")
        print(f"    Check which process owns the port with: sudo ss -ltnp 'sport = :{port}'")
        return None

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    scheme = "https" if bool(api_https.get("enabled")) else "http"
    auth_note = " with API key auth" if api_auth_enabled else ""
    print(f"[*] Ctx metadata API listening on {scheme}://{bind_host}:{port}{auth_note}")
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
    # run is an explicit user action — force through the active-model guard
    # so the user isn't blocked by other running models.
    effective_args.force = True
    mid = ensure_model_available(effective_args)
    if args.no_chat:
        return 0
    return start_chat(mid, args.public_host, args.public_port)





def show_request_log(args):
    explicit_path = Path(args.path).expanduser() if getattr(args, "path", None) else None
    missing: list[str] = []
    for path in _candidate_request_log_paths(explicit_path):
        if path.exists():
            print(f"-- {path}")
            print(_tail_text_file(path, lines=int(args.lines)))
            return 0
        missing.append(str(path))
    print("No request log found. Checked:")
    for path in missing:
        print(f"  {path}")
    print("If the service runs under systemd/root, try:")
    print(f"  sudo {CLI_COMMAND} requests --path {SYSTEM_REQUESTS_LOG_PATH} --lines {int(args.lines)}")
    return 0


def show_logs(args):
    lines = int(getattr(args, "lines", 200))
    print("== API request log ==")
    explicit_path = Path(args.path).expanduser() if getattr(args, "path", None) else None
    found_request_log = False
    missing: list[str] = []
    for path in _candidate_request_log_paths(explicit_path):
        if path.exists():
            print(f"-- {path}")
            print(_tail_text_file(path, lines=lines))
            found_request_log = True
            break
        missing.append(str(path))
    if not found_request_log:
        print("No API request log found. Checked:")
        for path in missing:
            print(f"  {path}")
    print()
    print("== systemd journals ==")
    journal_cmd = [
        "journalctl",
        "-u",
        SWAP_SERVICE_NAME,
        "-u",
        MANAGER_SERVICE_NAME,
        "-n",
        str(lines),
        "--no-pager",
    ]
    if getattr(args, "since", None):
        journal_cmd.extend(["--since", str(args.since)])
    print("Command:")
    prefix = "sudo " if os.geteuid() != 0 else ""
    print(f"  {prefix}{' '.join(journal_cmd)}")
    if not getattr(args, "journal", False):
        print("Run with --journal to execute journalctl from this command.")
        return 0
    try:
        result = subprocess.run(journal_cmd, check=False, capture_output=True, text=True, timeout=30)
    except Exception as exc:
        print(f"Could not run journalctl: {exc}")
        if os.geteuid() != 0:
            print("Try the sudo command shown above.")
        return 1
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    return 0 if result.returncode == 0 else result.returncode


def show_hacks(args):
    print("llamacpp-stack llama.cpp modifications / risky knobs")
    print()
    print("Source patches applied during source builds:")
    print("  - none")
    print()
    print("Potentially aggressive CUDA CMake flags when supported by source/config:")
    print("  - GGML_CUDA_FORCE_MMQ")
    print("  - GGML_CUDA_GRAPHS")
    print("  - GGML_CUDA_FA_ALL_QUANTS")
    print("    Disable for rebuild with: HEIMDALL_GATEWAY_DISABLE_AGGRESSIVE_CUDA=1")
    print()
    print("Runtime knobs most relevant to native segfault isolation:")
    print("  - flash_attn")
    print("  - fit / fit_target / fitc")
    print("  - parallel > 1")
    print("  - cont_batching")
    print("  - draft/speculative/MTP")
    print("  - ctx_checkpoints")
    print("    Disable conservatively with: HEIMDALL_GATEWAY_SAFE_RUNTIME=1, then regenerate config/restart.")
    print()
    print("Active env in this CLI process:")
    for key in (
        "HEIMDALL_GATEWAY_DISABLE_AGGRESSIVE_CUDA",
        "HEIMDALL_GATEWAY_SAFE_RUNTIME",
        "HEIMDALL_GATEWAY_REQUESTS_LOG",
    ):
        print(f"  {key}={os.environ.get(key, '')!r}")
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
        replica_defaults=resolve_global_replica_config(effective_args),
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
        replica_defaults=resolve_global_replica_config(args),
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
    cmd = ["systemctl", "restart", service_name]
    if os.geteuid() != 0:
        cmd = ["systemctl", "--user", "restart", service_name]
    try:
        result = subprocess.run(
            cmd,
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
        replica_defaults=resolve_global_replica_config(args),
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
    if _env_value("HEIMDALL_GATEWAY_TEMPLATES_DIR", "LLAMACPP_TEMPLATES_DIR", ""):
        templates_dir = Path(_env_value("HEIMDALL_GATEWAY_TEMPLATES_DIR", "LLAMACPP_TEMPLATES_DIR", "")).expanduser()
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
    cuda_root = _env_value("HEIMDALL_GATEWAY_CUDA_ROOT", "LLAMACPP_CUDA_ROOT", "").strip()
    nccl_root = _env_value("HEIMDALL_GATEWAY_NCCL_ROOT", "LLAMACPP_NCCL_ROOT", "").strip()
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

    if is_gemma4_sliding_window_long_context_model(model):
        selected_ctx = _align_ctx(max_ctx)
        _emit_message(
            f"{model.model_id}: Gemma4 sliding-window GGUF detected; trusting metadata max ctx {selected_ctx} instead of conservative prompt probing.",
            progress_callback,
            timestamp=True,
        )
        return selected_ctx, "metadata-selected", {
            "max_ctx": max_ctx,
            "calibration_ctx": None,
            "estimated_ctx": selected_ctx,
            "first_failure": None,
            "selected_ctx": selected_ctx,
            "probe_latency_ms": None,
            "probe_read_s": None,
            "probe_tokens_s": None,
            "probe_totals_s": None,
            "probe_speed_tps": None,
            "probe_prompt_tokens": None,
            "selected_ctx_gb": None,
        }

    _emit_message(
        f"{model.model_id}: GGUF max ctx {max_ctx}. Auto-fit will calibrate memory and probe boundaries.",
        progress_callback,
        timestamp=True,
    )
    _emit_message(f"{model.model_id}: each probe now sends a conservative long prompt sized for that ctx candidate.", progress_callback, timestamp=True)
    if _has_vision_runtime(model):
        _emit_message(f"{model.model_id}: vision runtime detected, each ctx probe will include a valid image plus text.", progress_callback, timestamp=True)

    probe_max_first = _env_value("HEIMDALL_GATEWAY_AUTO_CTX_MAX_FIRST", "LLAMACPP_AUTO_CTX_MAX_FIRST", "").lower() in ("1", "true", "yes", "on")

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
                _emit_message(f"{model.model_id}: ctx {candidate} failed due to batch/parallel memory; reducing batch_size and parallel=1 to prioritize context...", progress_callback, timestamp=True)
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
                    _emit_message(f"{model.model_id}: ctx {candidate} still failed after batch/parallel reduction ({reason}).", progress_callback, timestamp=True)
                else:
                    _emit_message(f"{model.model_id}: ctx {candidate} success after reducing batch_size and parallel=1.", progress_callback, timestamp=True)
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
                _emit_message(f"{model.model_id}: ctx {midpoint} failed due to batch/parallel memory; reducing batch_size and parallel=1 to prioritize context...", progress_callback, timestamp=True)
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
                    _emit_message(f"{model.model_id}: ctx {midpoint} still failed after batch/parallel reduction ({reason}).", progress_callback, timestamp=True)
                else:
                    _emit_message(f"{model.model_id}: ctx {midpoint} success after reducing batch_size and parallel=1.", progress_callback, timestamp=True)
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
        tensor_split=normalize_tensor_split(args.tensor_split or default_tensor_split()),
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
                    if _delete_path_with_permission_fallback(path, target_dir):
                        removed += 1
                except FileNotFoundError:
                    pass
    return removed


def ensure_catalog_mtp_drafters(
    models: list[ManagedModel],
    args,
    server_defaults: dict[str, object] | None,
    progress_callback=None,
) -> int:
    """Backfill MTP drafter files/flags for already-installed local GGUF models.

    llama.cpp can auto-discover a repo-root MTP drafter when launched with
    `-hf`, but Heimdall Gateway launches downloaded GGUFs with `--model`.  Existing
    installs therefore need an explicit local drafter file and `--model-draft`.
    """
    token = getattr(args, "hf_token", None) or os.environ.get("HF_TOKEN")
    api = HfApi()
    changed = 0
    size_cache: dict[str, dict[str, int | None]] = {}
    for model in models:
        repo_id = str(getattr(model, "repo_id", "") or "").strip()
        filename = str(getattr(model, "filename", "") or "").strip()
        overrides = normalize_server_overrides(getattr(model, "server_overrides", {}) or {})
        draft_path = str(overrides.get("model_draft") or "").strip()
        if (
            not repo_id
            or not filename
            or filename == "hf-native"
            or not _should_probe_repo_for_mtp_drafter(repo_id, filename, str(getattr(model, "model_id", "") or ""), draft_path)
        ):
            continue
        if draft_path and Path(draft_path).exists():
            updated, mtp_changed = _apply_mtp_server_overrides(overrides, draft_path, server_defaults)
            if mtp_changed or updated != overrides:
                model.server_overrides = updated
                changed += 1
            continue
        try:
            mtp_filename = _detect_mtp_drafter_file(api, repo_id, token, filename)
        except Exception as exc:
            log_api_event("mtp_drafter_detect_failed", {"model": model.model_id, "repo_id": repo_id, "error": str(exc)})
            mtp_filename = None
        if not mtp_filename:
            if _looks_like_integrated_mtp_model(repo_id, filename, str(getattr(model, "model_id", "") or ""), getattr(model, "local_path", "")):
                updated, mtp_changed = _apply_mtp_server_overrides(overrides, None, server_defaults)
                if mtp_changed or updated != overrides:
                    model.server_overrides = updated
                    changed += 1
                    _emit_message(
                        f"Detected integrated MTP model for {model.model_id}: {filename}; enabling draft-mtp.",
                        progress_callback,
                    )
            continue
        target_dir = Path(model.local_path).parent if model.local_path else Path(getattr(args, "models_dir", DEFAULT_MODELS_DIR)) / repo_id
        target_dir.mkdir(parents=True, exist_ok=True)
        local_path = target_dir / mtp_filename
        if not local_path.exists():
            try:
                if repo_id not in size_cache:
                    size_cache[repo_id] = _repo_sibling_sizes(api, repo_id, token)
                download_hf_file(
                    repo_id=repo_id,
                    filename=mtp_filename,
                    token=token,
                    target_dir=target_dir,
                    label=f"MTP drafter {Path(mtp_filename).name}",
                    progress_callback=progress_callback,
                    expected_size=size_cache.get(repo_id, {}).get(mtp_filename),
                )
            except Exception as exc:
                log_api_event("mtp_drafter_download_failed", {"model": model.model_id, "repo_id": repo_id, "filename": mtp_filename, "error": str(exc)})
                continue
        updated, mtp_changed = _apply_mtp_server_overrides(overrides, local_path, server_defaults)
        if mtp_changed or updated != overrides:
            model.server_overrides = updated
            changed += 1
            _emit_message(
                f"Detected MTP drafter for {model.model_id}: {mtp_filename}; enabling draft-mtp.",
                progress_callback,
            )
    return changed


def update_config(args, progress_callback = None):
    # Always migrate/canonicalize the global server config on update, including
    # when update is executed inside the manager daemon rather than via main().
    persist_server_config(args)
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

    catalog, catalog_diag = load_catalog_with_diagnostics(args.catalog, _args_server_config_path(args))
    if catalog_diag:
        raise RuntimeError(f"Refusing to update from an invalid catalog: {catalog_diag}")
    for warning in catalog_key_warnings(args.catalog):
        _emit_message(f"WARNING: {warning}", progress_callback)
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
    mtp_updated = ensure_catalog_mtp_drafters(target_models, args, resolve_llama_server_defaults(args), progress_callback=progress_callback)
    if mtp_updated:
        _emit_message(f"MTP drafter configuration updated for {mtp_updated} model(s).", progress_callback)
    save_catalog(args.catalog, catalog)
    replica_defaults = resolve_global_replica_config(args)
    render_llamaswap_config(
        catalog,
        args.config,
        args.llama_server,
        args.start_port,
        resolve_idle_ttl(args),
        server_defaults=resolve_llama_server_defaults(args),
        replica_defaults=replica_defaults,
    )
    replica_summary = summarize_configured_replicas(catalog, replica_defaults)
    if replica_summary:
        _emit_message("Replicas configured:\n" + "\n".join(f"- {line}" for line in replica_summary), progress_callback)
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


def sync_config_from_server_config_for_startup(args) -> str:
    """Render llama-swap config from current conf/catalog during service startup.

    This intentionally avoids the full `update` command behavior that waits for
    the public llama-swap API. On systemd restart the router may not be up yet,
    but conf.json changes still need to materialize into config.yaml before the
    router starts watching it.
    """
    raw_server_config = _load_server_config_payload(args)
    warnings = _server_config_validation_warnings(raw_server_config)
    for warning in warnings:
        print(f"[!] Heimdall Gateway config warning: {warning}", flush=True)
        log_api_event("server_config_validation_warning", {"warning": warning, "server_config": str(_args_server_config_path(args))})
    persist_server_config(args)
    catalog, catalog_diag = load_catalog_with_diagnostics(args.catalog, _args_server_config_path(args))
    if catalog_diag:
        raise RuntimeError(f"Refusing startup sync from an invalid catalog: {catalog_diag}")
    replica_defaults = resolve_global_replica_config(args)
    render_llamaswap_config(
        catalog,
        args.config,
        args.llama_server,
        args.start_port,
        resolve_idle_ttl(args),
        server_defaults=resolve_llama_server_defaults(args),
        replica_defaults=replica_defaults,
    )
    return "synced"

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
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)
                response_text = str(getattr(response, "text", "") or "").lower()
                matrix_reloading = status_code == 500 and "matrix is shutting down" in response_text
                if status_code not in {502, 503, 504} and not matrix_reloading:
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
    print("  heimdall-gateway requests --lines 80")
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



def _file_signature(path: Path | str | None) -> tuple[int, int] | None:
    if path is None:
        return None
    try:
        st = Path(path).stat()
        return (int(st.st_mtime_ns), int(st.st_size))
    except Exception:
        return None


def start_catalog_auto_update_watch(args, *, poll_s: float = 2.0, debounce_s: float = 1.0, stop_event: threading.Event | None = None):
    """Watch catalog/server config edits and regenerate llama-swap config automatically."""
    catalog_path = Path(args.catalog)
    server_config_path = Path(getattr(args, "server_config", DEFAULT_SERVER_CONFIG_PATH))
    watched = [catalog_path, server_config_path]
    state = {str(path): _file_signature(path) for path in watched}
    pending_changed: set[str] = set()

    def loop():
        while stop_event is None or not stop_event.is_set():
            try:
                changed: list[str] = []
                for path in watched:
                    key = str(path)
                    sig = _file_signature(path)
                    if sig is not None and state.get(key) is not None and sig != state.get(key):
                        changed.append(key)
                    state[key] = sig
                if changed:
                    pending_changed.update(changed)
                    log_api_event("auto_update_change_detected", {"paths": changed})
                if pending_changed:
                    active_summary = _active_download_blocker_summary()
                    if active_summary:
                        log_api_event(
                            "auto_update_deferred_model_active",
                            {"paths": sorted(pending_changed), "active": active_summary},
                        )
                    else:
                        if stop_event is not None and stop_event.wait(max(0.1, debounce_s)):
                            break
                        if stop_event is None:
                            time.sleep(max(0.1, debounce_s))
                        # Refresh signatures after debounce so a partial write is less likely.
                        for path in watched:
                            state[str(path)] = _file_signature(path)
                        update_paths = sorted(pending_changed)
                        pending_changed.clear()
                        started = time.monotonic()
                        try:
                            update_config(args)
                            # update_config may canonicalize catalog.json/conf.json itself.
                            # Refresh signatures after the write so the watcher does not
                            # trigger an infinite self-update loop.
                            for path in watched:
                                state[str(path)] = _file_signature(path)
                            log_api_event("auto_update_completed", {"paths": update_paths, "elapsed_ms": _elapsed_ms(started)})
                        except Exception as exc:
                            pending_changed.update(update_paths)
                            for path in watched:
                                state[str(path)] = _file_signature(path)
                            log_api_event("auto_update_failed", {"paths": update_paths, "elapsed_ms": _elapsed_ms(started), "error": str(exc)})
                            print(f"[!] Automatic config update after file change failed: {exc}")
            except Exception as exc:
                log_api_event("auto_update_watch_error", {"error": str(exc)})
            if stop_event is not None:
                stop_event.wait(max(0.01, poll_s))
            else:
                time.sleep(max(0.01, poll_s))

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread

def daemon_mode(args):
    """Background manager listening on Unix socket."""
    # Ensure llama-swap configuration is up-to-date with manual edits to
    # catalog.json or conf.json. This is render-only and does not wait for
    # llama-swap/API because the router service may still be starting.
    try:
        print(f"[*] Using Heimdall Gateway server config: {_args_server_config_path(args)}", flush=True)
        print("[*] Syncing Heimdall Gateway config on startup...", flush=True)
        sync_config_from_server_config_for_startup(args)
        log_api_event("startup_config_sync_done", {"config": str(args.config), "catalog": str(args.catalog)})
    except Exception as exc:
        log_api_event("startup_config_sync_failed", {"config": str(getattr(args, "config", "")), "catalog": str(getattr(args, "catalog", "")), "error": str(exc)})
        print(f"[!] Warning: Startup configuration sync failed: {exc}", flush=True)

    _prepare_manager_socket_path(SOCKET_PATH)
    try:
        os.makedirs(os.path.dirname(SOCKET_PATH), exist_ok=True)
    except Exception as exc:
        raise RuntimeError(f"Could not create manager socket directory for {SOCKET_PATH}: {exc}") from exc
    ctx_metadata_server = start_ctx_metadata_server(args)
    if ctx_metadata_server is None:
        raise RuntimeError(f"Could not start Heimdall Gateway API on {args.public_host}:{resolve_api_port(args)}")
    unload_guard_thread = start_unexpected_unload_guard(args)
    auto_update_thread = start_catalog_auto_update_watch(args)
    
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
                elif req["command"] == "remove-orphans":
                    mock_args = argparse.Namespace(**req["args"])
                    mock_args.catalog = Path(mock_args.catalog)
                    mock_args.models_dir = Path(mock_args.models_dir)
                    mock_args.server_config = Path(getattr(mock_args, "server_config", DEFAULT_SERVER_CONFIG_PATH))
                    result = remove_orphan_models(mock_args, progress_callback=send_event)
                    send_event({"type": "done", "result": result})
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
        r = requests.get(models_url, timeout=1.5, verify=False)
        if r.status_code == 200:
            data = r.json().get("data", [])
            return f"reachable on {r.url}{via} ({len(data)} models listed)"
        return f"responding on {r.url}{via} with HTTP {r.status_code}"
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
        # verify=False is needed because we often use self-signed certs for localhost
        r = requests.get(models_url, timeout=1.5, verify=False)
        if r.status_code == 200:
            data = r.json().get("data", [])
            return f"reachable on {r.url}{via} ({len(data)} catalog models listed)"
        return f"responding on {r.url}{via} with HTTP {r.status_code}"
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


def get_heimdall_gateway_version() -> str:
    forced = _env_value("HEIMDALL_GATEWAY_VERSION", "LLAMACPP_SUPERSERVER_VERSION", "").strip()
    if forced:
        return forced
    try:
        return version("heimdall-gateway")
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


def render_heimdall_gateway_banner() -> str:
    divider = "=" * 72
    return (
        f"{divider}\n"
        " _ _                                _\n"
        "| | | __ _ _ __ ___   __ _  ___ _ __| |_ _ __\n"
        "| | |/ _` | '_ ` _ \\ / _` |/ __| '__| __| '_ \\\n"
        "| | | (_| | | | | | | (_| | (__| |  | |_| |_) |\n"
        "|_|_|\\__,_|_| |_| |_|\\__,_|\\___|_|   \\__| .__/\n"
        "                                           |_|   \n"
        "              Heimdall Gateway\n"
        f"         heimdall-gateway v{get_heimdall_gateway_version()}\n"
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
    # Templates directory: can be overridden with HEIMDALL_GATEWAY_TEMPLATES_DIR
    templates_env = _env_value("HEIMDALL_GATEWAY_TEMPLATES_DIR", "LLAMACPP_TEMPLATES_DIR", "")
    if templates_env:
        templates_dir = Path(templates_env).expanduser()
    else:
        templates_dir = Path(server_config_path).expanduser().parent / "templates"

    llama_defaults = resolve_llama_server_defaults(args)
    default_keep = llama_defaults.get("keep", 20000)
    default_cache_k = llama_defaults.get("cache_type_k", llama_defaults.get("cache-type-k", ""))
    default_cache_v = llama_defaults.get("cache_type_v", llama_defaults.get("cache-type-v", ""))
    default_parallel = llama_defaults.get("parallel", 1)
    default_batch = llama_defaults.get("batch_size", llama_defaults.get("batch-size", 4096))
    default_ubatch = llama_defaults.get("ubatch_size", llama_defaults.get("ubatch-size", 2048))
    default_swa_full = _as_bool(llama_defaults.get("swa_full", llama_defaults.get("swa-full")), False)
    default_flags_line = f"    --keep {default_keep}"
    if default_cache_k not in (None, ""):
        default_flags_line += f", --cache-type-k {default_cache_k}"
    if default_cache_v not in (None, ""):
        default_flags_line += f", --cache-type-v {default_cache_v}"
    default_flags_line += f", --parallel {default_parallel}"
    default_perf_line = f"    --batch-size {default_batch}, --ubatch-size {default_ubatch}"
    if default_swa_full:
        default_perf_line += ", --swa-full"

    return (
        "Default endpoints:\n"
        f"  llama-swap UI/backend: {ui_url}\n"
        f"  Heimdall Gateway API:       {api_url}\n"
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
        f"  UI activity:         {ui_url}/ui/#/activity\n"
        f"  Idle TTL:            {resolve_idle_ttl(args)}s\n"
        "Service management:\n"
        f"  Install mode:        {install_mode}\n"
        f"  Start services:      {start_cmd}\n"
        f"  Status:              {status_cmd}\n"
        f"  Restart:             {restart_cmd}\n"
        "Logs & Diagnostics:\n"
        f"  API Requests Log:    {DEFAULT_REQUESTS_LOG_PATH}\n"
        f"  Manager Service:     journalctl -u {MANAGER_SERVICE_NAME} -n 100 --no-pager\n"
        f"  Swap Backend:        journalctl -u {SWAP_SERVICE_NAME} -n 100 --no-pager\n"
        "Config knobs:\n"
        f"  Global llama-server defaults: {server_config_path} -> llama_server_defaults\n"
        f"  Per-model overrides:          {catalog_path} -> server_overrides\n"
        f"  API_CTX factor:               {server_config_path} -> api_ctx_factor (default {DEFAULT_API_CTX_FACTOR})\n"
        "  Default llama-server flags:\n"
        f"{default_flags_line}\n"
        f"{default_perf_line}\n"
        f"    use_fitc=false uses --ctx-size directly; use_fitc=true uses -fitc.\n"
        f"    (Change these in {server_config_path}['llama_server_defaults'])\n"
        "  Main folders: install root, models dir, state/config paths above.\n"
        f"  API status:          {get_api_endpoint_status(public_host, api_port)}\n"
        f"  UI status:           {get_public_endpoint_status(public_host, public_port)}"
    )


def show_info(args):
    print(render_heimdall_gateway_banner())
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
        "  logs [-n LINES] [--journal]\n"
        "    Show request log and journalctl command/output for crash diagnostics.\n"
        f"    Example: {CLI_COMMAND} logs -n 200 --journal\n"
        "  hacks\n"
        "    List source patches, aggressive build flags and runtime safe-mode knobs.\n"
        f"    Example: {CLI_COMMAND} hacks\n"
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
    parser.add_argument("--flatten", action=argparse.BooleanOptionalAction, default=None, help="Flatten Responses-native tools (namespaces) to standard function tools.")
    public_command_metavar = (
        "{add,run,remove,rm,unload,update,config-migrate,refresh-templates,"
        "remove-templates,validate,daemon,debug,auto-performance,list,ps,"
        "requests,logs,hacks,info}"
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar=public_command_metavar)
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
    p_add.add_argument("--tensor-split", default=None)
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

    p_orphans = sub.add_parser(
        "remove-orphans",
        aliases=["clean-orphans", "orphans"],
        help="Remove model artifacts not referenced by the catalog",
        description=(
            "List model artifacts under the models directory that are not in the catalog. "
            "The command asks for confirmation before deleting them; use --dry-run to only inspect."
        ),
    )
    for alias in ("remove-orphans", "clean-orphans", "orphans"):
        subparsers[alias] = p_orphans
    p_orphans.set_defaults(func=remove_orphan_models)
    p_orphans.add_argument("--dry-run", action="store_true", help="Only list orphan artifacts")
    p_orphans.add_argument("--yes", action="store_true", help="Delete without an interactive confirmation")

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

    p_config_migrate = sub.add_parser(
        "config-migrate",
        help="Migrate/canonicalize the global conf.json",
        description="Rewrite the selected --server-config file with current global keys (replicas, api_ctx_factor, metadata) and canonical llama.cpp defaults.",
    )
    subparsers["config-migrate"] = p_config_migrate
    p_config_migrate.set_defaults(func=migrate_server_config)

    p_config_keys = sub.add_parser(
        "config-keys",
        help="List valid configuration/catalog keys",
        description="Print valid Heimdall Gateway catalog/conf keys and explain where raw llama.cpp flags belong.",
    )
    subparsers["config-keys"] = p_config_keys
    p_config_keys.set_defaults(func=print_config_keys)
    p_config_keys.add_argument("--format", choices=("text", "json"), default="text")

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
    p_validate.add_argument("--tensor-split", default=None)
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

    p_swap_guard = sub.add_parser(
        "llama-swap-guard",
        help=argparse.SUPPRESS,
        description="Internal guard proxy for llama-swap.",
    )
    subparsers["llama-swap-guard"] = p_swap_guard
    p_swap_guard.set_defaults(func=run_llamaswap_guard)
    p_swap_guard.add_argument("--llamaswap-bin", type=Path, default=None)
    p_swap_guard.add_argument("--listen-host", default=None)
    p_swap_guard.add_argument("--listen-port", type=int, default=None)
    p_swap_guard.add_argument("--backend-port", type=int, default=None)
    try:
        sub._choices_actions = [action for action in sub._choices_actions if action.dest != "llama-swap-guard"]
    except Exception:
        pass
    
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
        p.add_argument("--tensor-split", default=None)
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
    p_requests.add_argument("--path", type=Path, help="Read a specific API request log path")
    p_requests.set_defaults(func=show_request_log)

    p_logs = sub.add_parser(
        "logs",
        help="Show request logs and service journal hints",
        description="Show API request logs and optionally collect systemd journal entries for llama-swap/manager crash diagnostics.",
    )
    subparsers["logs"] = p_logs
    p_logs.add_argument("-n", "--lines", type=int, default=200)
    p_logs.add_argument("--path", type=Path, help="Read a specific API request log path")
    p_logs.add_argument("--since", help="journalctl --since value, e.g. '10 minutes ago'")
    p_logs.add_argument("--journal", action="store_true", help="Run journalctl and include service logs")
    p_logs.set_defaults(func=show_logs)

    p_hacks = sub.add_parser(
        "hacks",
        help="List llama.cpp source/build/runtime modifications",
        description="List stack-managed llama.cpp source patches, aggressive CUDA build flags and runtime safe-mode knobs.",
    )
    subparsers["hacks"] = p_hacks
    p_hacks.set_defaults(func=show_hacks)

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
        parser.error("Use the 'info' subcommand without dashes: heimdall-gateway info")
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
    if getattr(args, "func", None) is not migrate_server_config:
        persist_server_config(args)
    return args.func(args)

if __name__ == "__main__":
    try: sys.exit(main() or 0)
    except Exception as e: print(f"Error: {e}"); sys.exit(1)
