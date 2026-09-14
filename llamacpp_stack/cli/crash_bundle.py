"""Centralized crash-bundle collector for all engines.

Never throws. Caps: journal 12000, nvidia 2000, HTTP 4000 (handled at call-site).
Global timeout <= timeout_s (default 8s), per-subprocess 3s.

Only top-level import is .constants to avoid cycles; everything else lazy.
"""
from __future__ import annotations

import subprocess
import time
import re
from pathlib import Path
from typing import Any

from .constants import DEFAULT_REQUESTS_LOG_PATH  # noqa: F401 - allowed top-level

# Caps per spec
JOURNAL_CAP = 12000
NVIDIA_CAP = 2000
HTTP_BUNDLE_CAP = 4000  # used at call-site when embedding bundle in HTTP
JOURNAL_TIMEOUT = 3.0
NVIDIA_TIMEOUT = 3.0

VALID_ENGINES = {"llama-server", "beellama", "buun", "vllm", "exllama", "llama-swap", "manager"}
ENGINE_ALIASES = {
    "llama-server-beellama": "beellama",
    "llama-server-buun": "buun",
}


def _truncate(text: str, cap: int) -> str:
    if len(text) > cap:
        return text[:cap]
    return text


def _unavailable(exc: BaseException) -> str:
    return f"unavailable: {type(exc).__name__}"


def _detect_engine(engine_hint: str | None, cmdline: str | None) -> str:
    hint = (engine_hint or "").strip().lower()
    cl = (cmdline or "").lower()
    if hint in VALID_ENGINES:
        if hint == "beellama":
            return "beellama"
        if hint == "buun":
            return "buun"
        if hint in {"vllm", "exllama", "llama-swap", "manager", "llama-server"}:
            if hint == "manager":
                return "manager"
            if "buun" in cl:
                return "buun"
            if "beellama" in cl:
                return "beellama"
            if "vllm" in cl:
                return "vllm"
            if "exllama" in cl:
                return "exllama"
            if "llama-swap" in cl:
                return "llama-swap"
            return hint
    if "buun" in cl:
        return "buun"
    if "beellama" in cl:
        return "beellama"
    if "vllm" in cl:
        return "vllm"
    if "exllama" in cl:
        return "exllama"
    if "llama-swap" in cl:
        return "llama-swap"
    if "vllm.entrypoints" in cl or "vllm serve" in cl:
        return "vllm"
    if hint and "buun" in hint:
        return "buun"
    if hint and "beellama" in hint:
        return "beellama"
    if hint and "vllm" in hint:
        return "vllm"
    if hint and "exllama" in hint:
        return "exllama"
    if hint and "llama-swap" in hint:
        return "llama-swap"
    return "llama-server"


def _run_journal(unit: str, timeout: float) -> str:
    try:
        result = subprocess.run(
            ["journalctl", "--no-pager", "-n", "80", "-u", unit],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout if result.stdout else ""
        # If stderr and no stdout, include stderr
        if not out and result.stderr:
            out = result.stderr
        return _truncate(out, JOURNAL_CAP)
    except BaseException as exc:  # noqa: BLE001
        return _unavailable(exc)


def _run_nvidia(timeout: float) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout if result.stdout else ""
        if not out and result.stderr:
            out = result.stderr
        return _truncate(out, NVIDIA_CAP)
    except BaseException as exc:  # noqa: BLE001
        return _unavailable(exc)


def _resolve_cmdline(*, pid: int | None, port: int | None, cmdline_hint: str | None) -> str | None:
    if cmdline_hint:
        return str(cmdline_hint)
    # Try get_llama_server_processes lazy
    try:
        from llamacpp_stack._cli_impl import get_llama_server_processes  # type: ignore
    except Exception:
        try:
            from llamacpp_stack.cli import get_llama_server_processes as _g  # type: ignore

            get_llama_server_processes = _g  # type: ignore
        except Exception:
            get_llama_server_processes = None  # type: ignore

    if get_llama_server_processes is not None:
        try:
            procs = get_llama_server_processes()
            for p in procs:
                if pid is not None and p.get("pid") == pid:
                    return str(p.get("cmdline") or "")
                if port is not None and p.get("port") == port:
                    return str(p.get("cmdline") or "")
        except BaseException:
            pass
    # Fallback ps -p <pid> -o args=
    if pid is not None:
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "args="],
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            out = (result.stdout or "").strip()
            if out:
                return out
        except BaseException:
            pass
    return cmdline_hint


def _extract_probed_model(cmdline: str | None) -> str:
    if not cmdline:
        return ""
    try:
        m = re.search(r"--model\s+(\S+)", cmdline)
        if m:
            return m.group(1).strip("'\"")
    except Exception:
        pass
    return ""


def collect_engine_crash_bundle(
    engine: str,
    *,
    port: int | None = None,
    pid: int | None = None,
    cmdline: str | None = None,
    returncode: int | None = None,
    timeout_s: float = 8.0,
) -> dict[str, Any]:
    """Collect diagnostic bundle. Never throws.

    Returns dict with 9 keys: engine,pid,port,cmdline,returncode,
    journal_router_tail,journal_manager_tail,nvidia_smi,probed_model.
    Caps journal 12000, nvidia 2000. Global timeout <= timeout_s.
    """
    start = time.monotonic()
    deadline = start + float(timeout_s or 8.0)
    try:
        # Resolve cmdline first (may need subprocess but cheap)
        try:
            resolved_cmdline = _resolve_cmdline(pid=pid, port=port, cmdline_hint=cmdline)
        except BaseException as exc:  # noqa: BLE001
            resolved_cmdline = _unavailable(exc)

        try:
            detected = _detect_engine(engine, resolved_cmdline if isinstance(resolved_cmdline, str) else None)
        except BaseException:
            detected = "llama-server"

        # Helper to compute remaining timeout for each subprocess
        def _remaining(default: float) -> float:
            rem = deadline - time.monotonic()
            if rem <= 0.1:
                return 0.1
            return min(default, rem)

        # Journal router
        remaining = _remaining(JOURNAL_TIMEOUT)
        try:
            if time.monotonic() >= deadline:
                journal_router = _unavailable(TimeoutError("global timeout"))
            else:
                journal_router = _run_journal("heimdall-gateway-router", timeout=remaining)
        except BaseException as exc:  # noqa: BLE001
            journal_router = _unavailable(exc)

        remaining = _remaining(JOURNAL_TIMEOUT)
        try:
            if time.monotonic() >= deadline:
                journal_manager = _unavailable(TimeoutError("global timeout"))
            else:
                journal_manager = _run_journal("heimdall-gateway-manager", timeout=remaining)
        except BaseException as exc:  # noqa: BLE001
            journal_manager = _unavailable(exc)

        remaining = _remaining(NVIDIA_TIMEOUT)
        try:
            if time.monotonic() >= deadline:
                nvidia_smi = _unavailable(TimeoutError("global timeout"))
            else:
                nvidia_smi = _run_nvidia(timeout=remaining)
        except BaseException as exc:  # noqa: BLE001
            nvidia_smi = _unavailable(exc)

        try:
            probed = _extract_probed_model(resolved_cmdline if isinstance(resolved_cmdline, str) else None)
        except BaseException as exc:  # noqa: BLE001
            probed = _unavailable(exc)

        # Normalize resolved_cmdline to string or unavailable marker
        if resolved_cmdline is None:
            cmdline_out = ""
        elif isinstance(resolved_cmdline, str):
            # Ensure not overly huge? no cap specified for cmdline, keep as is but guard large
            cmdline_out = resolved_cmdline
        else:
            cmdline_out = str(resolved_cmdline)

        return {
            "engine": detected,
            "pid": pid,
            "port": port,
            "cmdline": cmdline_out,
            "returncode": returncode,
            "journal_router_tail": journal_router,
            "journal_manager_tail": journal_manager,
            "nvidia_smi": nvidia_smi,
            "probed_model": probed,
        }
    except BaseException as exc:  # noqa: BLE001
        # Absolute never-throw guarantee
        try:
            eng = _detect_engine(engine, cmdline)
        except Exception:
            eng = "llama-server"
        return {
            "engine": eng,
            "pid": pid,
            "port": port,
            "cmdline": (cmdline or _unavailable(exc)),
            "returncode": returncode,
            "journal_router_tail": _unavailable(exc),
            "journal_manager_tail": _unavailable(exc),
            "nvidia_smi": _unavailable(exc),
            "probed_model": "",
        }
