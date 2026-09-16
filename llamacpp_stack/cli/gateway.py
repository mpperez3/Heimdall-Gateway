"""LLM Server HTTP/dedup/Responses/tool-repair extracted from cli.py.

Preserves ports 11435/11436, DEDUP_STATE wiring, Responses proxy, tool-repair,
and manager_hint/run_manager_command. No top-level import of replica/server_commands
to avoid cycles; lazy inside functions where needed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlparse

try:
    import yaml  # type: ignore
    import requests  # type: ignore
except ImportError:
    yaml = None  # type: ignore
    requests = None  # type: ignore

try:
    from llamacpp_stack.dedup import DedupState, TeeBroadcast, canonical_body, fingerprint, principal_hash_for_api_key  # type: ignore
except Exception:  # pragma: no cover
    DedupState = None  # type: ignore
    TeeBroadcast = None  # type: ignore
    canonical_body = None  # type: ignore
    fingerprint = None  # type: ignore
    principal_hash_for_api_key = None  # type: ignore

from .constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_LLAMA_SERVER,
    DEFAULT_PUBLIC_HOST,
    DEFAULT_PUBLIC_PORT,
    DEFAULT_REQUESTS_LOG_PATH,
    MANAGER_SERVICE_NAME,
    SWAP_SERVICE_NAME,
    SOCKET_PATH,
    CHAT_TOOL_CONTINUE_REPAIR_THINKING_BUDGET_TOKENS,
)
from .env import _env_value

try:
    from .raw_log import log_raw_request  # type: ignore
except Exception:  # pragma: no cover
    log_raw_request = None  # type: ignore

# ---------------------------------------------------------------------------
# DEDUP_STATE singleton (preserves cli.py:71-76 semantics)
# ---------------------------------------------------------------------------
DEDUP_STATE = None  # type: ignore
try:
    if DedupState is not None:  # type: ignore[truthy-function]
        DEDUP_STATE = DedupState(max_entries=2000)  # type: ignore[call-arg]
except Exception:  # pragma: no cover
    DEDUP_STATE = None

_DEDUP_STREAM_TEES: dict[str, object] = {}

# ---------------------------------------------------------------------------
# Helpers that gateway needs but lives in cli file - lazy fallback imports
# ---------------------------------------------------------------------------

def _as_bool(value: object, default: bool = False) -> bool:
    try:
        mod = _get_cli_file()
        if mod is not None and hasattr(mod, "_as_bool"):
            return mod._as_bool(value, default)  # type: ignore
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off"}:
            return False
    return default

def _normalize_bool_flag(value: object):  # type: ignore[no-untyped-def]
    try:
        mod = _get_cli_file()
        if mod is not None and hasattr(mod, "_normalize_bool_flag"):
            return mod._normalize_bool_flag(value)  # type: ignore
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        n = value.strip().lower()
        if n in {"1", "true", "yes", "on"}:
            return True
        if n in {"0", "false", "no", "off"}:
            return False
    return None

def _is_loopback_client(host: str) -> bool:
    try:
        mod = _get_cli_file()
        if mod is not None and hasattr(mod, "_is_loopback_client"):
            return bool(mod._is_loopback_client(host))  # type: ignore
    except Exception:
        pass
    h = str(host or "").strip()
    return h in {"127.0.0.1", "::1", "localhost"} or h.startswith("127.")

def _load_server_config_payload(args=None):  # type: ignore[no-untyped-def]
    try:
        mod = _get_cli_file()
        if mod is not None and hasattr(mod, "_load_server_config_payload"):
            return mod._load_server_config_payload(args)  # type: ignore
    except Exception:
        pass
    return {}

def log_api_event(event: str, data: dict | None = None, log_path=None):  # type: ignore[no-untyped-def]
    try:
        mod = _get_cli_file()
        if mod is not None and hasattr(mod, "log_api_event"):
            if log_path is not None:
                return mod.log_api_event(event, data, log_path)  # type: ignore
            return mod.log_api_event(event, data)  # type: ignore
    except Exception:
        pass

_cli_file_mod_cache = None

def _get_cli_file():  # type: ignore[no-untyped-def]
    global _cli_file_mod_cache
    if _cli_file_mod_cache is not None:
        return _cli_file_mod_cache
    try:
        import importlib.util
        import sys
        if "llamacpp_stack._cli_file" in sys.modules:
            _cli_file_mod_cache = sys.modules["llamacpp_stack._cli_file"]
            return _cli_file_mod_cache
        path = Path(__file__).parent.parent / "cli.py"
        if not path.exists():
            return None
        spec = importlib.util.spec_from_file_location("llamacpp_stack._cli_file_gw", path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        # avoid double exec if already loaded
        if "llamacpp_stack._cli_file" in sys.modules:
            _cli_file_mod_cache = sys.modules["llamacpp_stack._cli_file"]
            return _cli_file_mod_cache
        sys.modules["llamacpp_stack._cli_file_gw"] = mod
        spec.loader.exec_module(mod)
        _cli_file_mod_cache = mod
        return mod
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Dedup config
# ---------------------------------------------------------------------------

def _default_dedup_inflight_config() -> dict[str, object]:
    return {
        "enabled": False,
        "ttl_s": 600,
        "max_wait_ms": 2000,
        "max_entries": 2000,
        "tee_buffer_lines": 1024,
        "tee_buffer_bytes": 2097152,
        "grace_s": 30,
    }


def _normalize_dedup_inflight_config(raw: object) -> tuple[dict[str, object], bool]:
    global DEDUP_STATE
    defaults = _default_dedup_inflight_config()
    if not isinstance(raw, dict):
        try:
            if DedupState is not None:  # type: ignore[truthy-function]
                max_e = int(defaults.get("max_entries", 2000))  # type: ignore[arg-type]
                if DEDUP_STATE is None:
                    DEDUP_STATE = DedupState(max_entries=max_e)  # type: ignore[call-arg]
        except Exception:
            pass
        return dict(defaults), True
    normalized: dict[str, object] = {}
    changed = False
    if "enabled" not in raw:
        normalized["enabled"] = defaults["enabled"]
        changed = True
    else:
        raw_enabled = raw.get("enabled")
        parsed = _normalize_bool_flag(raw_enabled)
        if parsed is None:
            normalized["enabled"] = defaults["enabled"]
            changed = True
        else:
            normalized["enabled"] = parsed
            if isinstance(raw_enabled, bool):
                if parsed != raw_enabled:
                    changed = True
            else:
                changed = True

    def _clamp_int(key: str, default: int, min_v: int, max_v: int) -> None:
        nonlocal changed
        if key not in raw:
            normalized[key] = default
            changed = True
            return
        raw_val = raw.get(key)
        try:
            if isinstance(raw_val, bool):
                raise ValueError("bool not allowed")
            if isinstance(raw_val, str):
                iv = int(raw_val.strip())
            else:
                iv = int(raw_val)  # type: ignore[arg-type]
        except Exception:
            normalized[key] = default
            changed = True
            return
        clamped = max(min_v, min(max_v, iv))
        normalized[key] = clamped
        if clamped != iv:
            changed = True
        if not isinstance(raw_val, int) or isinstance(raw_val, bool):
            changed = True
        elif raw_val != clamped:
            changed = True

    _clamp_int("ttl_s", 600, 0, 86400)
    _clamp_int("max_wait_ms", 2000, 0, 30000)
    _clamp_int("max_entries", 2000, 1, 100000)
    _clamp_int("tee_buffer_lines", 1024, 1, 100000)
    _clamp_int("tee_buffer_bytes", 2097152, 1024, 104857600)
    _clamp_int("grace_s", 30, 0, 86400)
    for k in defaults:
        if k not in normalized:
            normalized[k] = defaults[k]
            changed = True
    try:
        if DedupState is not None:  # type: ignore[truthy-function]
            max_e = int(normalized.get("max_entries", 2000))  # type: ignore[arg-type]
            if DEDUP_STATE is None:
                DEDUP_STATE = DedupState(max_entries=max_e)  # type: ignore[call-arg]
            elif getattr(DEDUP_STATE, "max_entries", None) != max_e:
                try:
                    DEDUP_STATE.max_entries = max_e  # type: ignore
                except Exception:
                    pass
    except Exception:
        pass
    return normalized, changed


def _dedup_should_bypass(args=None) -> str | None:  # type: ignore[no-untyped-def]
    try:
        cfg = _load_server_config_payload(args)
        exp = cfg.get("experimental") if isinstance(cfg.get("experimental"), dict) else {}
        dedup_cfg = exp.get("dedup_inflight") if isinstance(exp, dict) else None
        if not isinstance(dedup_cfg, dict):
            dedup_cfg = _default_dedup_inflight_config()
        else:
            dedup_cfg, _ = _normalize_dedup_inflight_config(dedup_cfg)
        if not bool(dedup_cfg.get("enabled")):
            return "dedup_bypass_disabled"
        if DEDUP_STATE is None or DedupState is None:
            return "dedup_bypass_no_state"
        return None
    except Exception:
        return "dedup_bypass_error"


def _dedup_extract_principal(handler_self) -> str:  # type: ignore[no-untyped-def]
    try:
        headers = getattr(handler_self, "headers", {}) or {}
        api_key = None
        try:
            auth = str(headers.get("Authorization") or headers.get("authorization") or "").strip()
            if auth.lower().startswith("bearer "):
                api_key = auth[7:].strip()
        except Exception:
            pass
        if not api_key:
            try:
                api_key = str(headers.get("X-Api-Key") or headers.get("X-API-Key") or headers.get("x-api-key") or "").strip()
            except Exception:
                pass
        if not api_key:
            try:
                client_host = handler_self.client_address[0] if getattr(handler_self, "client_address", None) else ""
                if _is_loopback_client(client_host):
                    return "anonymous"
            except Exception:
                pass
            return "anonymous" if not api_key else principal_hash_for_api_key(api_key)  # type: ignore
        if principal_hash_for_api_key is not None:  # type: ignore
            return principal_hash_for_api_key(api_key)  # type: ignore
        return "anonymous"
    except Exception:
        return "anonymous"


def _dedup_build_fingerprint(method: str, path: str, principal_hash: str, stream_flag: bool, body_dict: dict | None) -> str:  # type: ignore[no-untyped-def]
    try:
        if canonical_body is not None and fingerprint is not None:  # type: ignore
            cbody = canonical_body(body_dict)  # type: ignore
            return fingerprint(method, path, principal_hash, stream_flag, cbody)  # type: ignore
    except Exception:
        pass
    try:
        cbody = json.dumps(body_dict or {}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except Exception:
        cbody = b"{}"
    m = (method or "POST").upper()
    p = (path or "/").split("?", 1)[0]
    inner = hashlib.sha256(cbody).hexdigest()
    outer = f"{m}\n{p}\n{principal_hash}\n{str(bool(stream_flag))}\n{inner}"
    return hashlib.sha256(outer.encode("utf-8")).hexdigest()


def _dedup_send_cached_response(handler_self, cached_result: dict, hit_type: str) -> None:  # type: ignore[no-untyped-def]
    result = copy.deepcopy(cached_result)
    status = int(result.get("status", 200))
    headers = result.get("headers") or {}
    body = result.get("body", b"")
    if isinstance(body, str):
        body = body.encode("utf-8")
    if not isinstance(body, (bytes, bytearray)):
        body = str(body).encode("utf-8")
    ctype = "application/json"
    try:
        for k, v in (headers or {}).items():
            if str(k).lower() == "content-type":
                ctype = str(v)
                break
    except Exception:
        pass
    handler_self.send_response(status)
    handler_self.send_header("Content-Type", ctype)
    handler_self.send_header("Content-Length", str(len(body)))
    hdr_val = "hit" if hit_type == "hit" else "shared"
    try:
        handler_self.send_header("X-LLM-Server-Dedup", hdr_val)
    except Exception:
        pass
    try:
        for k, v in (headers or {}).items():
            lk = str(k).lower()
            if lk in {"content-type", "content-length", "connection", "transfer-encoding", "content-encoding", "x-llm-server-dedup", "x-heimdall-dedup"}:  # x-heimdall-dedup compat fallback
                continue
            try:
                handler_self.send_header(k, str(v))
            except Exception:
                continue
    except Exception:
        pass
    handler_self.end_headers()
    try:
        handler_self.wfile.write(body)
        handler_self.wfile.flush()
    except Exception:
        pass


def _dedup_send_json_with_header(handler_self, payload: dict, status: int = 200, dedup_header: str = "miss") -> None:  # type: ignore[no-untyped-def]
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler_self.send_response(status)
    handler_self.send_header("Content-Type", "application/json; charset=utf-8")
    handler_self.send_header("Content-Length", str(len(encoded)))
    try:
        handler_self.send_header("X-LLM-Server-Dedup", dedup_header)  # compat: old x-heimdall-dedup accepted inbound, normalized to new outbound
    except Exception:
        pass
    handler_self.end_headers()
    try:
        handler_self.wfile.write(encoded)
    except (BrokenPipeError, ConnectionResetError, OSError):
        try:
            handler_self.connection.shutdown(2)
        except Exception:
            pass


def _dedup_log_graced_hit(key: str, leader_id: str, similar_id: str, grace_ms: int) -> None:
    try:
        log_api_event("dedup_graced_hit", {"leader_id": leader_id, "similar_id": similar_id, "key_prefix8": key[:8], "grace_ms": grace_ms})
    except Exception:
        pass

def _dedup_log_similar_wait(key: str, leader_id: str, similar_id: str) -> None:
    try:
        log_api_event("dedup_similar_wait", {"leader_id": leader_id, "similar_id": similar_id, "key_prefix8": key[:8]})
    except Exception:
        pass

def _dedup_log_grace_expired(key: str, grace_s: int) -> None:
    try:
        log_api_event("dedup_grace_expired", {"key_prefix8": key[:8], "grace_s": grace_s})
    except Exception:
        pass

def _dedup_streaming_finalize(key: str, collected: list[bytes], dedup_cfg: dict, principal_hash: str, stream_flag: bool, truncated: bool, total_bytes: int, tee: object | None = None) -> None:  # type: ignore[no-untyped-def]
    try:
        if tee is not None and hasattr(tee, "close"):
            tee.close()  # type: ignore
    except Exception:
        pass
    grace_s = float(dedup_cfg.get("grace_s", 30) or 30) if isinstance(dedup_cfg, dict) else 30
    if truncated:
        try:
            if DEDUP_STATE is not None:
                DEDUP_STATE.complete_ok(key, {"status": 200, "headers": {}, "body": b""})  # type: ignore
                DEDUP_STATE.forget(key, grace_s=grace_s)  # type: ignore
        except Exception:
            pass
        log_api_event("dedup_stream_not_cached_truncated", {"key_prefix8": key[:8], "principal_hash8": principal_hash[:8], "stream": stream_flag, "bytes": total_bytes})
        return
    if not collected:
        try:
            if DEDUP_STATE is not None:
                DEDUP_STATE.forget(key, grace_s=grace_s)  # type: ignore
        except Exception:
            pass
        return
    try:
        max_bytes = int(dedup_cfg.get("tee_buffer_bytes", 2097152) or 2097152)
        body_bytes = b"\n".join(collected)
        if len(body_bytes) <= max_bytes:
            result = {"status": 200, "headers": {"Content-Type": "text/event-stream"}, "body": body_bytes}
            if DEDUP_STATE is not None:
                ttl_s = float(dedup_cfg.get("ttl_s", 600) or 600)
                if ttl_s > 0:
                    DEDUP_STATE.put_cached(key, result, ttl_s)  # type: ignore
                    DEDUP_STATE.complete_ok(key, result)  # type: ignore
                else:
                    DEDUP_STATE.complete_ok(key, result)  # type: ignore
                    DEDUP_STATE.forget(key, grace_s=grace_s)  # type: ignore
            log_api_event("dedup_stream_cached", {"key_prefix8": key[:8], "principal_hash8": principal_hash[:8], "stream": stream_flag, "bytes": len(body_bytes)})
        else:
            if DEDUP_STATE is not None:
                DEDUP_STATE.complete_ok(key, {"status": 200, "headers": {}, "body": b""})  # type: ignore
                DEDUP_STATE.forget(key, grace_s=grace_s)  # type: ignore
            log_api_event("dedup_stream_not_cached_truncated", {"key_prefix8": key[:8], "principal_hash8": principal_hash[:8], "stream": stream_flag, "bytes": len(body_bytes)})
    except Exception:
        try:
            if DEDUP_STATE is not None:
                DEDUP_STATE.forget(key, grace_s=grace_s)  # type: ignore
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Diagnostics / chat helpers
# ---------------------------------------------------------------------------

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


def _log_chat_stop_without_tools(request_id: str, model: str, upstream_model: str | None, *, stream: bool, content: str, reasoning_len: int, finish_reason: object, repair_rounds: int = 0) -> None:
    if str(finish_reason or "") != "stop":
        return
    preview = str(content or "").replace("\r", "\n")
    preview = re.sub(r"\s+", " ", preview).strip()[:500]
    log_api_event("openai_chat_stop_without_tool_calls", {"request_id": request_id, "model": model, "upstream_model": upstream_model, "stream": stream, "visible_content_len": len(str(content or "")), "reasoning_len": int(reasoning_len or 0), "finish_reason": str(finish_reason or ""), "repair_rounds": repair_rounds, "visible_preview": preview})


def resolve_chat_last_response_log_config(args=None) -> dict[str, object]:  # type: ignore[no-untyped-def]
    raw = _load_server_config_payload(args).get("experimental")
    cfg = _normalize_experimental_config(raw).get("chat_last_response_log", {})  # type: ignore
    if not isinstance(cfg, dict):
        cfg = dict(_default_experimental_config()["chat_last_response_log"])  # type: ignore
    return cfg  # type: ignore


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
        args, args_truncated = _truncate_debug_text(fn.get("arguments") if isinstance(fn, dict) else "", max_arg_chars)  # type: ignore
        out.append({"index": index, "id": str(item.get("id") or "")[:200], "type": str(item.get("type") or "")[:80], "name": str(fn.get("name") or item.get("name") or "")[:200] if isinstance(fn, dict) else str(item.get("name") or "")[:200], "arguments": args, "arguments_truncated": args_truncated, "arguments_len": len(str(fn.get("arguments") or "")) if isinstance(fn, dict) else 0})
    return out


def _write_chat_last_response_log(args, *, request_id: str, model: str, upstream_model: str | None, stream: bool, content: object, reasoning: object = "", reasoning_len: int | None = None, tool_calls: object = None, tool_call_chunks: int = 0, finish_reason: object = "", repair_rounds: int = 0) -> None:  # type: ignore[no-untyped-def]
    cfg = resolve_chat_last_response_log_config(args)
    if not bool(cfg.get("enabled")):
        return
    try:
        max_chars = max(0, int(cfg.get("max_chars", 20000)))  # type: ignore
    except Exception:
        max_chars = 20000
    path_text = str(cfg.get("path") or "").strip()
    path = Path(path_text) if path_text else Path(str(DEFAULT_REQUESTS_LOG_PATH)).with_name("last-chat-response.json")
    visible_text = str(content or "")
    visible_preview, visible_truncated = _truncate_debug_text(visible_text, max_chars)
    include_reasoning = bool(cfg.get("include_reasoning"))
    reasoning_text = str(reasoning or "")
    reasoning_preview, reasoning_truncated = _truncate_debug_text(reasoning_text, max_chars) if include_reasoning else ("", False)
    entry: dict[str, object] = {"ts": datetime.now(timezone.utc).isoformat(), "request_id": request_id, "model": model, "upstream_model": upstream_model, "stream": stream, "finish_reason": str(finish_reason or ""), "repair_rounds": int(repair_rounds or 0), "visible_content": visible_preview, "visible_content_len": len(visible_text), "visible_content_truncated": visible_truncated, "reasoning_len": int(reasoning_len if reasoning_len is not None else len(reasoning_text)), "reasoning_included": include_reasoning, "tool_call_chunks": int(tool_call_chunks or 0)}
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

# ---------------------------------------------------------------------------
# Experimental config (needed by resolve_chat_last_response_log_config)
# ---------------------------------------------------------------------------

def _default_experimental_config() -> dict[str, object]:
    return {
        "dedup_inflight": _default_dedup_inflight_config(),
        "chat_tool_continue_repair": {"enabled": False, "max_rounds": 1, "max_tokens": 2048, "stream_keepalive_seconds": 15, "visible_notice_after_seconds": 4, "trigger_prefixes": ["[terminal command", "[terminal_inline", "</terminal_inline>", "Voy a", "Empezando por"], "prompt": "Your previous assistant message ended without any tool_calls.\nYou are in a tool-capable agent environment. If the next step requires reading files, editing files, running commands, searching, inspecting state, or using any external capability, you must call one of the available tools instead of describing the action in text.\nDo not answer with empty visible content. Do not answer with a sentence that only sets up an action and ends with a colon.\nAvailable tool names: {tool_names}.", "truncated_tool_call_prompt": "Your previous assistant message started a tool_call but it was truncated before the JSON arguments were complete.\nRetry now with exactly one complete, valid tool_call. Keep the arguments minimal and valid JSON. Do not stream or repeat partial arguments. Do not include explanatory text before the tool_call.\nAvailable tool names: {tool_names}.", "include_failed_assistant_message": False, "loop_guard": {"enabled": True, "no_tool_call_max_chars": 0, "repeated_tail_min_chars": 3000, "repeated_tail_repetitions": 4}},
        "chat_last_response_log": {"enabled": False, "path": "", "max_chars": 20000, "include_reasoning": False, "include_tool_calls": True},
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
            if key == "dedup_inflight":
                normalized, _ = _normalize_dedup_inflight_config(value)
                cfg["dedup_inflight"] = normalized
            elif key == "chat_tool_continue_repair" and isinstance(value, dict):
                repair = dict(cfg["chat_tool_continue_repair"])  # type: ignore
                repair.update(value)  # type: ignore
                repair["enabled"] = _as_bool(repair.get("enabled"), False)  # type: ignore
                try:
                    repair["max_rounds"] = max(0, int(repair.get("max_rounds", 1)))  # type: ignore
                except Exception:
                    repair["max_rounds"] = 1  # type: ignore
                try:
                    repair["max_tokens"] = max(0, int(repair.get("max_tokens", 2048)))  # type: ignore
                except Exception:
                    repair["max_tokens"] = 2048  # type: ignore
                try:
                    repair["stream_keepalive_seconds"] = max(1, int(repair.get("stream_keepalive_seconds", 15)))  # type: ignore
                except Exception:
                    repair["stream_keepalive_seconds"] = 15  # type: ignore
                try:
                    repair["visible_notice_after_seconds"] = max(0, int(repair.get("visible_notice_after_seconds", 4)))  # type: ignore
                except Exception:
                    repair["visible_notice_after_seconds"] = 4  # type: ignore
                repair["trigger_prefixes"] = _normalize_chat_tool_continue_trigger_prefixes(repair.get("trigger_prefixes"))  # type: ignore
                for prompt_key in ("prompt", "truncated_tool_call_prompt"):
                    prompt_value = repair.get(prompt_key)  # type: ignore
                    default_prompt = _default_experimental_config()["chat_tool_continue_repair"].get(prompt_key, "")  # type: ignore
                    if not isinstance(prompt_value, str) or not prompt_value.strip():
                        repair[prompt_key] = default_prompt  # type: ignore
                    else:
                        repair[prompt_key] = prompt_value  # type: ignore
                repair["include_failed_assistant_message"] = _as_bool(repair.get("include_failed_assistant_message"), False)  # type: ignore
                loop_guard = repair.get("loop_guard")  # type: ignore
                default_loop_guard = _default_experimental_config()["chat_tool_continue_repair"]["loop_guard"]  # type: ignore
                if not isinstance(loop_guard, dict):
                    loop_guard = dict(default_loop_guard)  # type: ignore
                else:
                    merged = dict(default_loop_guard)  # type: ignore
                    merged.update(loop_guard)  # type: ignore
                    loop_guard = merged  # type: ignore
                loop_guard["enabled"] = _as_bool(loop_guard.get("enabled"), True)  # type: ignore
                for lk, dv in (("no_tool_call_max_chars", 0), ("repeated_tail_min_chars", 3000), ("repeated_tail_repetitions", 4)):
                    try:
                        loop_guard[lk] = max(0, int(loop_guard.get(lk, dv)))  # type: ignore
                    except Exception:
                        loop_guard[lk] = dv  # type: ignore
                repair["loop_guard"] = loop_guard  # type: ignore
                cfg["chat_tool_continue_repair"] = repair  # type: ignore
            elif key == "chat_last_response_log" and isinstance(value, dict):
                rl = dict(cfg["chat_last_response_log"])  # type: ignore
                rl.update(value)  # type: ignore
                rl["enabled"] = _as_bool(rl.get("enabled"), False)  # type: ignore
                rl["path"] = str(rl.get("path") or "").strip()  # type: ignore
                try:
                    rl["max_chars"] = max(0, int(rl.get("max_chars", 20000)))  # type: ignore
                except Exception:
                    rl["max_chars"] = 20000  # type: ignore
                rl["include_reasoning"] = _as_bool(rl.get("include_reasoning"), False)  # type: ignore
                rl["include_tool_calls"] = _as_bool(rl.get("include_tool_calls"), True)  # type: ignore
                cfg["chat_last_response_log"] = rl  # type: ignore
            elif key not in cfg:
                cfg[key] = value  # type: ignore
    if "dedup_inflight" not in cfg or not isinstance(cfg.get("dedup_inflight"), dict):
        cfg["dedup_inflight"], _ = _normalize_dedup_inflight_config(None)
    else:
        cfg["dedup_inflight"], _ = _normalize_dedup_inflight_config(cfg.get("dedup_inflight"))
    return cfg

# ---------------------------------------------------------------------------
# manager_hint / run_manager_command (preserve semantics, no cycle)
# ---------------------------------------------------------------------------

def infer_install_mode() -> str:
    try:
        mod = _get_cli_file()
        if mod is not None and hasattr(mod, "infer_install_mode"):
            return mod.infer_install_mode()  # type: ignore
    except Exception:
        pass
    explicit = _env_value("HEIMDALL_GATEWAY_INSTALL_MODE", "LLAMACPP_INSTALL_MODE", "").strip().lower()
    if explicit in {"system", "user"}:
        return explicit
    home = Path.home().resolve()
    def _is_home(p: Path) -> bool:
        try:
            pr = p.expanduser().resolve()
            return pr == home or home in pr.parents
        except Exception:
            return False
    if any(_is_home(Path(p)) for p in [str(DEFAULT_CONFIG_PATH), str(Path.home() / ".local/state/heimdall-gateway/catalog.json")]):
        return "user"
    return "system" if os.geteuid() == 0 else "user"


def service_commands_for_mode(mode: str) -> tuple[str, str, str]:
    if mode == "system":
        return (f"sudo systemctl start {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}", f"sudo systemctl status {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}", f"sudo systemctl restart {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}")
    return (f"systemctl --user start {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}", f"systemctl --user status {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}", f"systemctl --user restart {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}")


def manager_hint() -> str:
    mode = infer_install_mode()
    start_cmd, status_cmd, _ = service_commands_for_mode(mode)
    return f"Could not connect to the background manager.\nDetected install mode: {mode}.\nTry:\n  {start_cmd}\n  {status_cmd}\nSocket path: {SOCKET_PATH}"


def run_manager_command(command: str, args):  # type: ignore[no-untyped-def]
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(SOCKET_PATH)
        req = {"command": command, "args": vars(args)}
        req["args"] = {k: str(v) if isinstance(v, Path) else v for k, v in req["args"].items() if k != "func"}
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
                    try:
                        mod = _get_cli_file()
                        if mod is not None and hasattr(mod, "_render_download_progress"):
                            mod._render_download_progress(event["label"], int(event["downloaded"]), int(event["total"]) if event["total"] is not None else None, float(event["speed_bps"]), done=bool(event.get("done")))  # type: ignore
                    except Exception:
                        pass
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


def get_public_endpoint_status(host=DEFAULT_PUBLIC_HOST, port=DEFAULT_PUBLIC_PORT):  # type: ignore[no-untyped-def]
    base_url = f"http://{host}:{port}"
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::", "[::]"} else host
    models_url = f"http://{probe_host}:{port}/v1/models"
    via = "" if probe_host == host else f" via {probe_host}"
    try:
        if requests is None:
            return f"not reachable on {base_url}{via} (requests missing)"
        r = requests.get(models_url, timeout=1.5, verify=False)
        if r.status_code == 200:
            data = r.json().get("data", [])
            return f"reachable on {r.url}{via} ({len(data)} models listed)"
        return f"responding on {r.url}{via} with HTTP {r.status_code}"
    except Exception as e:
        return f"not reachable on {base_url}{via} ({e.__class__.__name__})"

# ---------------------------------------------------------------------------
# HTTP handlers (ThreadingHTTPServer + GuardHandler) - minimal wiring
# ---------------------------------------------------------------------------

def _proxy_headers_safe(headers: dict) -> dict:  # type: ignore[no-untyped-def]
    return {k: v for k, v in headers.items() if k.lower() not in {"host", "connection", "content-length", "accept-encoding"}}


def _proxy_request_to_public_api(method: str, path: str, *, body: bytes | None = None, headers: dict[str, str] | None = None, host: str = DEFAULT_PUBLIC_HOST, port: int = DEFAULT_PUBLIC_PORT):  # type: ignore[no-untyped-def]
    url = f"http://{host}:{port}{path}"
    proxy_headers = {}
    if headers:
        for k, v in headers.items():
            if k.lower() in {"host", "content-length", "connection"}:
                continue
            proxy_headers[k] = v
    if requests is None:
        raise RuntimeError("requests missing")
    return requests.request(method, url, data=body, headers=proxy_headers, timeout=(60, 600), stream=False)


def is_llamaswap_upstream_static_autoload_path(method: str, path: str) -> bool:
    if str(method or "").upper() not in {"GET", "HEAD"}:
        return False
    parsed = urlparse(str(path or ""))
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 3 or parts[0] != "upstream":
        return False
    remainder = "/" + "/".join(parts[2:])
    basename = parts[-1].lower()
    suffix = Path(basename).suffix.lower()
    from .constants import LLAMASWAP_UPSTREAM_STATIC_BLOCKED_BASENAMES, LLAMASWAP_UPSTREAM_STATIC_BLOCKED_EXTENSIONS  # noqa: WPS433

    return basename in LLAMASWAP_UPSTREAM_STATIC_BLOCKED_BASENAMES or suffix in LLAMASWAP_UPSTREAM_STATIC_BLOCKED_EXTENSIONS or remainder.startswith("/assets/") or remainder.startswith("/static/")


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


def run_llamaswap_guard(args) -> int:  # type: ignore[no-untyped-def]
    listen_host = str(getattr(args, "listen_host", None) or getattr(args, "public_host", None) or DEFAULT_PUBLIC_HOST)
    listen_port = int(getattr(args, "listen_port", None) or getattr(args, "public_port", None) or DEFAULT_PUBLIC_PORT)
    backend_host = "127.0.0.1"
    backend_port = int(getattr(args, "backend_port", None) or _next_llamaswap_guard_backend_port(listen_port))
    llamaswap_bin = Path(getattr(args, "llamaswap_bin", None) or os.environ.get("LLAMASWAP_BIN", "llama-swap"))
    config_path = Path(getattr(args, "config", None) or os.environ.get("HEIMDALL_GATEWAY_CONFIG", _env_value("HEIMDALL_GATEWAY_CONFIG", "LLAMACPP_CONFIG", str(DEFAULT_CONFIG_PATH))))
    child_cmd = [str(llamaswap_bin), "--config", str(config_path), "--listen", f"{backend_host}:{backend_port}", "--watch-config"]
    import subprocess
    import signal
    child = subprocess.Popen(child_cmd)

    class GuardHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, fmt, *values):  # type: ignore[no-untyped-def]
            return
        def _blocked(self) -> bool:  # type: ignore[no-untyped-def]
            if not is_llamaswap_upstream_static_autoload_path(self.command, self.path):
                return False
            log_api_event("llamaswap_upstream_static_autoload_blocked", {"method": self.command, "path": self.path})
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return True
        def _proxy(self):  # type: ignore[no-untyped-def]
            if self._blocked():
                return
            length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(length) if length > 0 else None
            try:
                if self.command == "POST":
                    _raw = body if body is not None else b""
                    if log_raw_request is not None:
                        log_raw_request(_raw)
                    else:
                        try:
                            from llamacpp_stack.cli.raw_log import log_raw_request as _lr  # type: ignore

                            _lr(_raw)
                        except Exception:
                            pass
            except Exception:
                pass
            target_url = f"http://{backend_host}:{backend_port}{self.path}"
            try:
                if requests is None:
                    raise RuntimeError("requests missing")
                upstream = requests.request(self.command, target_url, headers=_proxy_headers_safe(dict(self.headers)), data=body, stream=True, timeout=(5, None))
            except Exception as exc:
                msg = f"llama-swap guard backend error: {exc}"
                log_api_event("llamaswap_guard_backend_error", {"method": self.command, "path": self.path, "error": str(exc)})
                _bundle_guard = None
                _bundle_ref_guard = None
                try:
                    import uuid as _uuid_guard
                    import json as _json_guard

                    try:
                        _bundle_ref_guard = (
                            self.headers.get("X-Request-ID")
                            or self.headers.get("X-Correlation-ID")
                            or _uuid_guard.uuid4().hex[:8]
                        )
                    except Exception:
                        _bundle_ref_guard = _uuid_guard.uuid4().hex[:8]
                    try:
                        from llamacpp_stack.cli.crash_bundle import collect_engine_crash_bundle as _collect_guard  # type: ignore
                    except Exception:
                        _collect_guard = None  # type: ignore
                    if _collect_guard is not None:
                        try:
                            _bundle_guard = _collect_guard(
                                "llama-server",
                                port=int(backend_port),
                                pid=None,
                                cmdline=None,
                                returncode=None,
                                timeout_s=8.0,
                            )
                        except Exception as _be:
                            _bundle_guard = {"engine": "llama-server", "journal_router_tail": f"unavailable: {type(_be).__name__}", "journal_manager_tail": f"unavailable: {type(_be).__name__}", "nvidia_smi": f"unavailable: {type(_be).__name__}", "probed_model": "", "pid": None, "port": int(backend_port), "cmdline": "", "returncode": None}
                    else:
                        _bundle_guard = {"engine": "llama-server", "journal_router_tail": "unavailable: ImportError", "journal_manager_tail": "unavailable: ImportError", "nvidia_smi": "unavailable: ImportError", "probed_model": "", "pid": None, "port": int(backend_port), "cmdline": "", "returncode": None}
                    try:
                        log_api_event(
                            "proxy_error_with_bundle",
                            {"method": self.command, "path": self.path, "error": str(exc)[:1000], "bundle": _bundle_guard, "bundle_ref": _bundle_ref_guard},
                        )
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    import json as _json_guard2

                    _err_short_g = f"upstream unavailable: {exc}"[:200]
                    _eng_g = (_bundle_guard.get("engine") if isinstance(_bundle_guard, dict) else "llama-server") or "llama-server"
                    _hint_g = "uv run heimdall-gateway logs --lines 200 --journal"
                    _ref_g = _bundle_ref_guard or __import__("uuid").uuid4().hex[:8]
                    payload_g = {"error": _err_short_g, "engine": _eng_g, "hint": _hint_g, "bundle_ref": _ref_g}
                    data_g = _json_guard2.dumps(payload_g).encode("utf-8", errors="replace")
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data_g)))
                    self.end_headers()
                    self.wfile.write(data_g)
                    return
                except Exception:
                    pass
                data = msg.encode("utf-8", errors="replace")
                self.send_response(502)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_response(upstream.status_code)
            for k, v in upstream.headers.items():
                if k.lower() in {"connection", "transfer-encoding", "content-encoding", "content-length"}:
                    continue
                self.send_header(k, v)
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
        def do_GET(self): self._proxy()  # type: ignore[no-untyped-def]
        def do_HEAD(self): self._proxy()  # type: ignore[no-untyped-def]
        def do_POST(self): self._proxy()  # type: ignore[no-untyped-def]
        def do_PUT(self): self._proxy()  # type: ignore[no-untyped-def]
        def do_PATCH(self): self._proxy()  # type: ignore[no-untyped-def]
        def do_DELETE(self): self._proxy()  # type: ignore[no-untyped-def]
        def do_OPTIONS(self): self._proxy()  # type: ignore[no-untyped-def]

    server = ThreadingHTTPServer((listen_host, listen_port), GuardHandler)
    server.daemon_threads = True
    stop_event = threading.Event()

    def _request_server_shutdown():  # type: ignore[no-untyped-def]
        try:
            server.shutdown()
        except Exception:
            pass

    def _shutdown(_signum=None, _frame=None):  # type: ignore[no-untyped-def]
        stop_event.set()
        threading.Thread(target=_request_server_shutdown, daemon=True).start()
        if child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    def _watch_child():  # type: ignore[no-untyped-def]
        rc = child.wait()
        if not stop_event.is_set():
            log_api_event("llamaswap_guard_child_exited", {"returncode": rc})
            try:
                from llamacpp_stack.cli.crash_bundle import collect_engine_crash_bundle as _collect_gc2  # type: ignore

                _bundle_gc2 = _collect_gc2("llama-swap", port=int(backend_port), pid=getattr(child, "pid", None), cmdline=None, returncode=rc, timeout_s=8.0)
            except Exception as _be:
                _bundle_gc2 = {"engine": "llama-swap", "pid": getattr(child, "pid", None), "port": int(backend_port), "cmdline": f"unavailable: {type(_be).__name__}", "returncode": rc, "journal_router_tail": f"unavailable: {type(_be).__name__}", "journal_manager_tail": f"unavailable: {type(_be).__name__}", "nvidia_smi": f"unavailable: {type(_be).__name__}", "probed_model": ""}
            try:
                log_api_event("llamaswap_guard_child_exited_with_bundle", {"returncode": rc, "bundle": _bundle_gc2})
            except Exception:
                pass
            _request_server_shutdown()

    threading.Thread(target=_watch_child, daemon=True).start()
    print(f"llama-swap guard listening on {listen_host}:{listen_port}; backend {backend_host}:{backend_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        _shutdown()
        try:
            child.wait(timeout=10)
        except Exception:
            child.kill()
    return child.returncode or 0

# Re-export Responses/tool-repair symbols by delegating to cli file for fidelity
# (single source of truth remains cli.py until fully extracted; tests import via
# llamacpp_stack.cli which fallback will resolve to gateway then cli file)
def __getattr__(name: str):  # type: ignore[no-untyped-def]
    mod = _get_cli_file()
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [  # type: ignore
    "DEDUP_STATE", "_DEDUP_STREAM_TEES",
    "_default_dedup_inflight_config", "_normalize_dedup_inflight_config",
    "_dedup_should_bypass", "_dedup_extract_principal", "_dedup_build_fingerprint",
    "_dedup_send_cached_response", "_dedup_send_json_with_header", "_dedup_streaming_finalize",
    "_log_chat_stop_without_tools", "_tool_call_debug_summary",
    "resolve_chat_last_response_log_config", "_write_chat_last_response_log",
    "_normalize_experimental_config", "_default_experimental_config",
    "manager_hint", "run_manager_command", "infer_install_mode", "service_commands_for_mode",
    "get_public_endpoint_status", "_proxy_request_to_public_api", "_proxy_headers_safe",
    "is_llamaswap_upstream_static_autoload_path", "run_llamaswap_guard", "GuardHandler",
    "ThreadingHTTPServer",
]
