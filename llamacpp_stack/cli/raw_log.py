"""Raw ring buffer log for last 10 requests - api-raw-requests.log.

Single responsibility: store raw request bodies without parsing.
Reuses path resolution precedence from T2 (explicit > env > conf > DEFAULT).
Falls back to /tmp/heimdall-gateway-api-raw-requests.log on primary failure.
Ring keeps exactly last 10 lines, atomic tmp+rename, never throws.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .constants import DEFAULT_REQUESTS_LOG_PATH  # noqa: F401 - only top-level constants allowed


def _get_env_value(primary: str, legacy: str | None = None, default: str = "") -> str:
    """Lazy import of _env_value to avoid circular deps; only constants at top-level."""
    try:
        from .env import _env_value as _ev  # type: ignore

        return _ev(primary, legacy, default)
    except Exception:
        import os

        if os.environ.get(primary):
            return str(os.environ[primary])
        if legacy and os.environ.get(legacy):
            return str(os.environ[legacy])
        return default

RAW_BASENAME = "api-raw-requests.log"
RAW_MAX_CHARS = 1048576  # 1 MiB per request cap
RAW_RING_SIZE = 10
RAW_FALLBACK_PATH = Path("/tmp/heimdall-gateway-api-raw-requests.log")


def _resolve_requests_log_base_path(explicit: Path | None) -> Path:
    """Replicate _resolve_requests_log_base_path precedence: explicit > env > conf > default."""
    if explicit is not None:
        return Path(explicit).expanduser()
    env_path = _get_env_value("HEIMDALL_GATEWAY_REQUESTS_LOG_PATH", "LLAMACPP_REQUESTS_LOG_PATH", "")
    if env_path.strip():
        return Path(env_path.strip()).expanduser()
    # conf.json logging.requests_log.path
    try:
        # lazy to avoid circular
        from llamacpp_stack._cli_impl import _effective_requests_log_config  # type: ignore

        cfg = _effective_requests_log_config()
        p = str(cfg.get("path") or "").strip()
        if p:
            return Path(p).expanduser()
    except Exception:
        pass
    # fallback try direct file read (if _cli_impl not available yet)
    try:
        import json

        for candidate in (
            Path.home() / ".config/heimdall-gateway/conf.json",
            Path("/etc/heimdall-gateway/conf.json"),
        ):
            if candidate.exists():
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                logging_cfg = payload.get("logging") if isinstance(payload.get("logging"), dict) else {}
                req_cfg = logging_cfg.get("requests_log") if isinstance(logging_cfg, dict) else None
                if isinstance(req_cfg, dict):
                    cp = str(req_cfg.get("path") or "").strip()
                    if cp:
                        return Path(cp).expanduser()
                break
    except Exception:
        pass
    return DEFAULT_REQUESTS_LOG_PATH


def _raw_path_for_base(base: Path) -> Path:
    """Derive raw log path as dir(base)/api-raw-requests.log.

    If base is a directory (exists and is_dir) or looks like a dir (no suffix and not file-like),
    use base dir directly. Otherwise use parent.
    """
    try:
        # If caller passed a directory explicitly (env pointing to dir)
        if base.exists() and base.is_dir():
            return base / RAW_BASENAME
        # Heuristic: if path has no suffix and was env/conf dir intention, treat as dir if ends with /
        # Simpler: if base.suffix == "" and not base.name.endswith(".log"):
        # Use parent / RAW_BASENAME for file bases (most common: .../api-requests.log)
        if base.suffix == "" and "." not in base.name:
            # ambiguous dir-like without extension, treat as directory
            return base / RAW_BASENAME
    except Exception:
        pass
    return base.parent / RAW_BASENAME


def _sanitize_one_line(text: str) -> str:
    """Keep one line per request: escape newlines.

    Replaces \\r\\n, \\n, \\r with literal \\n/\\r to preserve 1 line per request
    while keeping maximum fidelity vs re-serializing. Truncation handled separately.
    """
    # Replace CRLF first to avoid double escaping
    text = text.replace("\r\n", "\\n")
    text = text.replace("\n", "\\n")
    text = text.replace("\r", "\\r")
    return text


def _write_ring(target: Path, sanitized: str) -> bool:
    """Write sanitized line to ring buffer (last 10). Atomic tmp+rename. Returns True on success."""
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        # propagate as failure so fallback can be tried; record via caller
        raise OSError(f"mkdir {target.parent}: {exc}") from exc
    try:
        existing: list[str] = []
        if target.exists():
            try:
                # read with replace to handle any encoding issues
                data = target.read_text(encoding="utf-8", errors="replace")
                existing = data.splitlines()
            except Exception:
                existing = []
        # append and keep last 10
        existing.append(sanitized)
        if len(existing) > RAW_RING_SIZE:
            existing = existing[-RAW_RING_SIZE:]
        # atomic write via tmp
        tmp = target.with_name(f".{target.name}.tmp")
        try:
            tmp.write_text("\n".join(existing) + "\n", encoding="utf-8")
            os.replace(tmp, target)
        except Exception:
            # cleanup tmp on failure
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            raise
        return True
    except Exception as exc:
        raise OSError(f"write_ring {target}: {exc}") from exc


def log_raw_request(raw_body: bytes | str | None, *, log_path: Path | str | None = None) -> None:
    """Append raw request body to api-raw-requests.log ring (last 10), best-effort, never throws.

    - Decodes bytes utf-8 errors=replace.
    - Truncates to 1 MiB chars.
    - Sanitizes newlines to keep 1 line per request (\\n escaped).
    - Ring buffer: keep last 10 lines, discard older.
    - Resolves base path via explicit > env > conf > default, then derives raw path as dir/base/api-raw-requests.log.
    - Fallback to /tmp/heimdall-gateway-api-raw-requests.log if primary not writable.
    - Never raises; on total failure prints to stderr only if HEIMDALL_GATEWAY_DEBUG_LOGGING set.
    """
    try:
        # 1. Decode
        if raw_body is None:
            text = ""
        elif isinstance(raw_body, bytes):
            text = raw_body.decode("utf-8", errors="replace")
        else:
            text = str(raw_body)

        # 2. Truncate to 1 MiB chars (after decode, before sanitization to keep cap on stored chars)
        if len(text) > RAW_MAX_CHARS:
            text = text[:RAW_MAX_CHARS]

        # 3. Sanitize to one line
        sanitized = _sanitize_one_line(text)

        # Ensure sanitized still within cap (escaping \n adds one char per newline, negligible)
        if len(sanitized) > RAW_MAX_CHARS:
            sanitized = sanitized[:RAW_MAX_CHARS]

        # 4. Resolve primary raw target
        explicit_base: Path | None = None
        if log_path is not None:
            try:
                explicit_base = Path(log_path).expanduser()  # type: ignore[arg-type]
            except Exception:
                explicit_base = None

        try:
            base = _resolve_requests_log_base_path(explicit_base)
        except Exception:
            base = DEFAULT_REQUESTS_LOG_PATH
        primary_target = _raw_path_for_base(base)

        # Also derive fallback raw target
        try:
            fallback_base = Path(
                _get_env_value(
                    "HEIMDALL_GATEWAY_REQUESTS_LOG_FALLBACK",
                    "LLAMACPP_REQUESTS_LOG_FALLBACK",
                    "/tmp/heimdall-gateway-api-requests.log",
                )
            ).expanduser()
            fallback_target = _raw_path_for_base(fallback_base)
            # Ensure canonical fallback is /tmp/heimdall-gateway-api-raw-requests.log if fallback_base is default tmp path
            if fallback_target == Path("/tmp/api-raw-requests.log"):
                fallback_target = RAW_FALLBACK_PATH
        except Exception:
            fallback_target = RAW_FALLBACK_PATH

        # If fallback would be same as primary, just use primary
        errors: list[str] = []

        def _try(target: Path) -> bool:
            try:
                return _write_ring(target, sanitized)
            except Exception as exc:
                errors.append(f"{target}: {exc}")
                return False

        if _try(primary_target):
            return
        if primary_target != fallback_target:
            if _try(fallback_target):
                return
        # also try canonical RAW_FALLBACK if not already tried
        if fallback_target != RAW_FALLBACK_PATH and primary_target != RAW_FALLBACK_PATH:
            if _try(RAW_FALLBACK_PATH):
                return

        if _get_env_value("HEIMDALL_GATEWAY_DEBUG_LOGGING", "LLAMACPP_DEBUG_LOGGING", ""):
            print(f"[!] log_raw_request failed to write to any candidate: {', '.join(errors)}", file=sys.stderr)
    except Exception:
        # Never throw; debug only
        if _get_env_value("HEIMDALL_GATEWAY_DEBUG_LOGGING", "LLAMACPP_DEBUG_LOGGING", ""):
            import traceback

            print(f"[!] log_raw_request unexpected error: {traceback.format_exc()}", file=sys.stderr)
        return


__all__ = ["log_raw_request", "RAW_BASENAME", "RAW_MAX_CHARS", "RAW_RING_SIZE", "RAW_FALLBACK_PATH"]
