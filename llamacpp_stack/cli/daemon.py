"""LLM Server daemon/auto-watch extracted from cli.py.

Preserves:
- sync_config_from_server_config_for_startup (render-only, no API wait)
- start_catalog_auto_update_watch (poll_s=2.0 debounce_s=1.0, defer if download active)
- daemon_mode (manager socket lifecycle, no systemctl restart/daemon-reload)
- _prepare_manager_socket_path, read_install_manifest, build_info_text, show_info
- update_config hot-reload via --watch-config remains in cli.py (render + sleep 3s + probe)

No top-level import of main cli module to avoid cycles; lazy via _get_cli_file().
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import pwd
import re
import socket
import sys
import threading
import time
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    import requests  # type: ignore  # pyright: ignore[reportMissingModuleSource]
    import yaml  # type: ignore  # pyright: ignore[reportMissingModuleSource]
except ImportError:  # pragma: no cover
    requests = None  # type: ignore
    yaml = None  # type: ignore  # pyright: ignore[reportAssignmentType]

from .constants import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_IDLE_TTL,
    DEFAULT_LLAMA_SERVER,
    DEFAULT_PUBLIC_HOST,
    DEFAULT_PUBLIC_PORT,
    DEFAULT_REQUESTS_LOG_PATH,
    DEFAULT_SERVER_CONFIG_PATH,
    MANAGER_SERVICE_NAME,
    SWAP_SERVICE_NAME,
    SOCKET_PATH,
)
from .env import _env_value

# ---------------------------------------------------------------------------
# Lazy CLI file helpers (mirror gateway.py pattern)
# ---------------------------------------------------------------------------
_cli_file_mod_cache = None


def _get_cli_file():  # type: ignore[no-untyped-def]
    global _cli_file_mod_cache
    if _cli_file_mod_cache is not None:
        return _cli_file_mod_cache
    try:
        import importlib.util
        import sys

        path = Path(__file__).parent.parent / "cli.py"
        if not path.exists():
            return None
        # reuse already-loaded _cli_file if present (from cli/__init__.py)
        if "llamacpp_stack._cli_file" in sys.modules:
            _cli_file_mod_cache = sys.modules["llamacpp_stack._cli_file"]
            return _cli_file_mod_cache
        spec = importlib.util.spec_from_file_location("llamacpp_stack._cli_file_daemon", path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["llamacpp_stack._cli_file_daemon"] = mod
        spec.loader.exec_module(mod)
        _cli_file_mod_cache = mod
        return mod
    except Exception:
        return None


def _fallback_attr(name: str, default=None):  # type: ignore[no-untyped-def]
    try:
        mod = _get_cli_file()
        if mod is not None and hasattr(mod, name):
            return getattr(mod, name)
    except Exception:
        pass
    return default


def log_api_event(event: str, data: dict[str, object] | None = None, log_path=None):  # type: ignore[no-untyped-def]
    try:
        mod = _get_cli_file()
        if mod is not None and hasattr(mod, "log_api_event"):
            if log_path is not None:
                return mod.log_api_event(event, data, log_path)  # type: ignore
            return mod.log_api_event(event, data)  # type: ignore
    except Exception:
        pass


def _elapsed_ms(started_at: float) -> int:
    fn = _fallback_attr("_elapsed_ms")
    if fn is not None:
        try:
            return int(fn(started_at))  # type: ignore
        except Exception:
            pass
    return int((time.monotonic() - started_at) * 1000)


def _active_download_blocker_summary(catalog=None):  # type: ignore[no-untyped-def]
    fn = _fallback_attr("_active_download_blocker_summary")
    if fn is not None:
        try:
            return fn(catalog) if catalog is not None else fn()  # type: ignore
        except TypeError:
            try:
                return fn()  # type: ignore
            except Exception:
                pass
        except Exception:
            pass
    return ""


def _args_server_config_path(args) -> Path | None:  # type: ignore[no-untyped-def]
    fn = _fallback_attr("_args_server_config_path")
    if fn is not None:
        try:
            return fn(args)  # type: ignore
        except Exception:
            pass
    val = getattr(args, "server_config", None)
    if val is None:
        return None
    return Path(val)


def _load_server_config_payload(args=None):  # type: ignore[no-untyped-def]
    fn = _fallback_attr("_load_server_config_payload")
    if fn is not None:
        try:
            return fn(args)  # type: ignore
        except Exception:
            pass
    return {}


def _server_config_validation_warnings(payload):  # type: ignore[no-untyped-def]
    fn = _fallback_attr("_server_config_validation_warnings")
    if fn is not None:
        try:
            return fn(payload)  # type: ignore
        except Exception:
            pass
    return []


def persist_server_config(args) -> None:  # type: ignore[no-untyped-def]
    fn = _fallback_attr("persist_server_config")
    if fn is not None:
        try:
            return fn(args)  # type: ignore
        except Exception:
            pass


def load_catalog_with_diagnostics(path: Path, server_config_path=None):  # type: ignore[no-untyped-def]
    fn = _fallback_attr("load_catalog_with_diagnostics")
    if fn is not None:
        try:
            return fn(path, server_config_path)  # type: ignore
        except Exception:
            pass
    return [], None


def resolve_global_replica_config(args=None):  # type: ignore[no-untyped-def]
    fn = _fallback_attr("resolve_global_replica_config")
    if fn is not None:
        try:
            return fn(args)  # type: ignore
        except Exception:
            pass
    return {}


def render_llamaswap_config(*args, **kwargs):  # type: ignore[no-untyped-def]
    fn = _fallback_attr("render_llamaswap_config")
    if fn is not None:
        try:
            return fn(*args, **kwargs)  # type: ignore
        except Exception:
            pass


def resolve_idle_ttl(args=None) -> int:  # type: ignore[no-untyped-def]
    fn = _fallback_attr("resolve_idle_ttl")
    if fn is not None:
        try:
            return int(fn(args))  # type: ignore
        except Exception:
            pass
    if args is not None and getattr(args, "idle_ttl", None) is not None:
        try:
            return int(args.idle_ttl)  # type: ignore
        except Exception:
            pass
    return int(DEFAULT_IDLE_TTL)


def resolve_llama_server_defaults(args=None):  # type: ignore[no-untyped-def]
    fn = _fallback_attr("resolve_llama_server_defaults")
    if fn is not None:
        try:
            return fn(args)  # type: ignore
        except Exception:
            pass
    return {}


def _normalize_client_host(host):  # type: ignore[no-untyped-def]
    fn = _fallback_attr("_normalize_client_host")
    if fn is not None:
        try:
            return fn(host)  # type: ignore
        except Exception:
            pass
    return str(host or "127.0.0.1")


def _update_config(args, progress_callback=None):  # type: ignore[no-untyped-def]
    fn = _fallback_attr("update_config")
    if fn is not None:
        try:
            return fn(args, progress_callback=progress_callback)  # type: ignore
        except TypeError as exc:
            if "progress_callback" in str(exc) or "unexpected keyword" in str(exc):
                try:
                    return fn(args)  # type: ignore
                except Exception:
                    raise
            raise
        except Exception:
            raise
    raise RuntimeError("update_config not available")


def _get_start_ctx_metadata_server():  # type: ignore[no-untyped-def]
    return _fallback_attr("start_ctx_metadata_server")


def _get_start_unexpected_unload_guard():  # type: ignore[no-untyped-def]
    return _fallback_attr("start_unexpected_unload_guard")


def _get_infer_install_mode():  # type: ignore[no-untyped-def]
    fn = _fallback_attr("infer_install_mode")
    if fn is not None:
        return fn
    # fallback to gateway
    try:
        from .gateway import infer_install_mode as gw_infer  # type: ignore

        return gw_infer
    except Exception:
        pass
    return None


def _get_service_commands_for_mode():  # type: ignore[no-untyped-def]
    fn = _fallback_attr("service_commands_for_mode")
    if fn is not None:
        return fn
    try:
        from .gateway import service_commands_for_mode as gw_svc  # type: ignore

        return gw_svc
    except Exception:
        pass
    return None


def _get_public_endpoint_status():  # type: ignore[no-untyped-def]
    fn = _fallback_attr("get_public_endpoint_status")
    if fn is not None:
        return fn
    try:
        from .gateway import get_public_endpoint_status as gw_pub  # type: ignore

        return gw_pub
    except Exception:
        pass
    return None


def _get_api_endpoint_status():  # type: ignore[no-untyped-def]
    fn = _fallback_attr("get_api_endpoint_status")
    if fn is not None:
        return fn
    return None


def _get_as_bool():  # type: ignore[no-untyped-def]
    return _fallback_attr("_as_bool")


# ---------------------------------------------------------------------------
# Core daemon helpers (verbatim from cli.py)
# ---------------------------------------------------------------------------

def _file_signature(path: Path | str | None) -> tuple[int, int] | None:
    if path is None:
        return None
    try:
        st = Path(path).stat()
        return (int(st.st_mtime_ns), int(st.st_size))
    except Exception:
        return None


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
        print(f"[!] LLM Server config warning: {warning}", flush=True)
        log_api_event("server_config_validation_warning", {"warning": warning, "server_config": str(_args_server_config_path(args))})
    persist_server_config(args)
    catalog, catalog_diag = load_catalog_with_diagnostics(args.catalog, _args_server_config_path(args))
    if catalog_diag:
        raise RuntimeError(f"Refusing startup sync from an invalid catalog: {catalog_diag}")
    for _entry in catalog:
        _so = getattr(_entry, "server_overrides", {}) or {}
        if isinstance(_so, dict) and str(_so.get("engine") or "").strip().lower() == "exllama":
            _mid = str(getattr(_entry, "model_id", "") or "unknown")
            warnings.warn(f"engine exllama deprecated, use buun for EXL3 ({_mid})", DeprecationWarning, stacklevel=2)
            print(f"[!] DeprecationWarning: engine exllama deprecated, use buun for EXL3 ({_mid})", file=sys.stderr, flush=True)
            break
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
                            _update_config(args)
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
    forced = (os.environ.get("LLM_SERVER_VERSION", "").strip() or _env_value("HEIMDALL_GATEWAY_VERSION", "LLAMACPP_SUPERSERVER_VERSION", "").strip())
    if forced:
        return forced
    for _pkg in ("llm-server", "heimdall-gateway"):
        try:
            return version(_pkg)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    # Fallback for local editable runs.
    pyproject_path = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
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
                 "              LLM Server\n"
        f"         llm-server v{get_heimdall_gateway_version()}\n"
        f"{divider}"
    )


def _install_root_from_llama_server(llama_server_path: Path) -> Path:
    if "llama.cpp/build/bin" in str(llama_server_path):
        return llama_server_path.parent.parent.parent
    return llama_server_path.parent


def build_info_text(args=None) -> str:
    public_host = getattr(args, "public_host", DEFAULT_PUBLIC_HOST) if args is not None else DEFAULT_PUBLIC_HOST
    public_port = int(getattr(args, "public_port", DEFAULT_PUBLIC_PORT) if args is not None else DEFAULT_PUBLIC_PORT)
    # resolve_api_port lazy
    api_port_fn = _fallback_attr("resolve_api_port")
    if api_port_fn is not None:
        try:
            api_port = int(api_port_fn(args))  # type: ignore
        except Exception:
            api_port = int(DEFAULT_PUBLIC_PORT) - 1
    else:
        api_port = int(DEFAULT_PUBLIC_PORT) - 1
    ui_url = f"http://{public_host}:{public_port}"
    api_url = f"http://{public_host}:{api_port}"

    manifest = read_install_manifest()
    llama_cpp_tag = str(manifest.get("llama_cpp_tag") or "unknown")
    llamaswap_tag = str(manifest.get("llamaswap_tag") or "unknown")

    infer_fn = _get_infer_install_mode()
    install_mode = infer_fn() if infer_fn is not None else "user"  # type: ignore
    svc_fn = _get_service_commands_for_mode()
    if svc_fn is not None:
        start_cmd, status_cmd, restart_cmd = svc_fn(install_mode)  # type: ignore
    else:
        if install_mode == "system":
            start_cmd = f"sudo systemctl start {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}"
            status_cmd = f"sudo systemctl status {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}"
            restart_cmd = f"sudo systemctl restart {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}"
        else:
            start_cmd = f"systemctl --user start {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}"
            status_cmd = f"systemctl --user status {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}"
            restart_cmd = f"systemctl --user restart {MANAGER_SERVICE_NAME} {SWAP_SERVICE_NAME}"

    models_dir = Path(getattr(args, "models_dir", DEFAULT_CONFIG_PATH.parent / "models")) if args is not None else Path(DEFAULT_CONFIG_PATH.parent / "models")
    # args may have models_dir as Path already; fallback constants handle
    if args is not None and hasattr(args, "models_dir"):
        try:
            models_dir = Path(getattr(args, "models_dir"))
        except Exception:
            pass
    config_path = Path(getattr(args, "config", DEFAULT_CONFIG_PATH)) if args is not None else Path(DEFAULT_CONFIG_PATH)
    catalog_path = Path(getattr(args, "catalog", DEFAULT_CATALOG_PATH)) if args is not None else Path(DEFAULT_CATALOG_PATH)
    server_config_path = _args_server_config_path(args) if args is not None else Path(DEFAULT_SERVER_CONFIG_PATH)
    llama_server_path = Path(getattr(args, "llama_server", DEFAULT_LLAMA_SERVER)) if args is not None else Path(DEFAULT_LLAMA_SERVER)
    install_root = _install_root_from_llama_server(llama_server_path)
    # Templates directory: can be overridden with HEIMDALL_GATEWAY_TEMPLATES_DIR
    templates_env = _env_value("HEIMDALL_GATEWAY_TEMPLATES_DIR", "LLAMACPP_TEMPLATES_DIR", "")
    if templates_env:
        templates_dir = Path(templates_env).expanduser()
    else:
        try:
            base = Path(server_config_path).expanduser().parent if server_config_path is not None else Path(DEFAULT_SERVER_CONFIG_PATH).parent
        except Exception:
            base = Path(DEFAULT_SERVER_CONFIG_PATH).parent
        templates_dir = base / "templates"

    llama_defaults = resolve_llama_server_defaults(args)
    default_keep = llama_defaults.get("keep", 20000)
    default_cache_k = llama_defaults.get("cache_type_k", llama_defaults.get("cache-type-k", ""))
    default_cache_v = llama_defaults.get("cache_type_v", llama_defaults.get("cache-type-v", ""))
    default_parallel = llama_defaults.get("parallel", 1)
    default_batch = llama_defaults.get("batch_size", llama_defaults.get("batch-size", 4096))
    default_ubatch = llama_defaults.get("ubatch_size", llama_defaults.get("ubatch-size", 2048))
    as_bool_fn = _get_as_bool()
    if as_bool_fn is not None:
        try:
            default_swa_full = bool(as_bool_fn(llama_defaults.get("swa_full", llama_defaults.get("swa-full")), False))  # type: ignore
        except Exception:
            default_swa_full = False
    else:
        default_swa_full = bool(llama_defaults.get("swa_full", False))
    default_flags_line = f"    --keep {default_keep}"
    if default_cache_k not in (None, ""):
        default_flags_line += f", --cache-type-k {default_cache_k}"
    if default_cache_v not in (None, ""):
        default_flags_line += f", --cache-type-v {default_cache_v}"
    default_flags_line += f", --parallel {default_parallel}"
    default_perf_line = f"    --batch-size {default_batch}, --ubatch-size {default_ubatch}"
    if default_swa_full:
        default_perf_line += ", --swa-full"

    # Resolve idle ttl
    try:
        idle_ttl_val = resolve_idle_ttl(args)
    except Exception:
        idle_ttl_val = DEFAULT_IDLE_TTL

    # endpoint statuses
    pub_fn = _get_public_endpoint_status()
    api_fn = _get_api_endpoint_status()
    try:
        api_status = api_fn(public_host, api_port) if api_fn is not None else "unknown"  # type: ignore
    except Exception:
        api_status = "unknown"
    try:
        ui_status = pub_fn(public_host, public_port) if pub_fn is not None else "unknown"  # type: ignore
    except Exception:
        ui_status = "unknown"

    return (
        "Default endpoints:\n"
        f"  llama-swap UI/backend: {ui_url}\n"
        f"  LLM Server API:       {api_url}\n"
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
        f"  Idle TTL:            {idle_ttl_val}s\n"
        "Service management:\n"
        f"  Install mode:        {install_mode}\n"
        f"  Start services:      {start_cmd}\n"
        f"  Status:              {status_cmd}\n"
        f"  Restart:             {restart_cmd}\n"
        "Logs & Diagnostics:\n"
        f"  API Requests Log:    {DEFAULT_REQUESTS_LOG_PATH.parent}/api-requests.log.YYYY-MM-DD (rotado diario, 3 días)\n"
        f"  Manager Service:     journalctl -u {MANAGER_SERVICE_NAME} -n 100 --no-pager\n"
        f"  Swap Backend:        journalctl -u {SWAP_SERVICE_NAME} -n 100 --no-pager\n"
        "Config knobs:\n"
        f"  Global llama-server defaults: {server_config_path} -> llama_server_defaults\n"
        f"  Per-model overrides:          {catalog_path} -> server_overrides\n"
        f"  API_CTX factor:               {server_config_path} -> api_ctx_factor (default {DEFAULT_IDLE_TTL if False else 0.5})\n"
        "  Default llama-server flags:\n"
        f"{default_flags_line}\n"
        f"{default_perf_line}\n"
        f"    use_fitc=false uses --ctx-size directly; use_fitc=true uses -fitc.\n"
        f"    (Change these in {server_config_path}['llama_server_defaults'])\n"
        "  Main folders: install root, models dir, state/config paths above.\n"
        f"  API status:          {api_status}\n"
        f"  UI status:           {ui_status}"
    )


def show_info(args):
    print(render_heimdall_gateway_banner())
    print()
    print(build_info_text(args))
    return 0


def daemon_mode(args):
    """Background manager listening on Unix socket."""
    # Ensure llama-swap configuration is up-to-date with manual edits to
    # catalog.json or conf.json. This is render-only and does not wait for
    # llama-swap/API because the router service may still be starting.
    try:
        print(f"[*] Using LLM Server server config: {_args_server_config_path(args)}", flush=True)
        print("[*] Syncing LLM Server config on startup...", flush=True)
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
    # Lazy imports for servers (remain in cli.py to avoid duplication)
    ctx_fn = _get_start_ctx_metadata_server()
    guard_fn = _get_start_unexpected_unload_guard()
    ctx_metadata_server = ctx_fn(args) if ctx_fn is not None else None  # type: ignore
    if ctx_metadata_server is None:
        # Use same error semantics as cli.py
        ap_fn = _fallback_attr("resolve_api_port")
        try:
            api_p = ap_fn(args) if ap_fn is not None else 11435  # type: ignore
        except Exception:
            api_p = 11435
        raise RuntimeError(f"Could not start LLM Server API on {args.public_host}:{api_p}")
    unload_guard_thread = guard_fn(args) if guard_fn is not None else None  # type: ignore
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
                if not data:
                    return
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
                    # lazy
                    fn = _fallback_attr("ensure_model_available")
                    if fn is None:
                        raise RuntimeError("ensure_model_available not available")
                    model_id = fn(mock_args, progress_callback=send_event)  # type: ignore
                    send_event({"type": "done", "model_id": model_id})
                elif req["command"] == "list":
                    mock_args = argparse.Namespace(**req["args"])
                    mock_args.catalog = Path(mock_args.catalog)
                    mock_args.server_config = Path(getattr(mock_args, "server_config", DEFAULT_SERVER_CONFIG_PATH))
                    fn_cat = _fallback_attr("load_catalog")
                    fn_render = _fallback_attr("render_models_table")
                    if fn_cat is None or fn_render is None:
                        raise RuntimeError("list helpers not available")
                    table = fn_render(fn_cat(mock_args.catalog, mock_args.server_config), mock_args.public_host, mock_args.public_port)  # type: ignore
                    send_event({"type": "done", "result": table})
                elif req["command"] == "update":
                    mock_args = argparse.Namespace(**req["args"])
                    mock_args.catalog = Path(mock_args.catalog)
                    mock_args.config = Path(mock_args.config)
                    mock_args.llama_server = Path(mock_args.llama_server)
                    mock_args.server_config = Path(getattr(mock_args, "server_config", DEFAULT_SERVER_CONFIG_PATH))
                    result = _update_config(mock_args, progress_callback=send_event)
                    send_event({"type": "done", "result": result})
                elif req["command"] == "remove":
                    mock_args = argparse.Namespace(**req["args"])
                    mock_args.catalog = Path(mock_args.catalog)
                    mock_args.config = Path(mock_args.config)
                    mock_args.models_dir = Path(mock_args.models_dir)
                    mock_args.llama_server = Path(mock_args.llama_server)
                    mock_args.server_config = Path(getattr(mock_args, "server_config", DEFAULT_SERVER_CONFIG_PATH))
                    fn = _fallback_attr("remove_model")
                    if fn is None:
                        raise RuntimeError("remove_model not available")
                    model_id = fn(mock_args, progress_callback=send_event)  # type: ignore
                    send_event({"type": "done", "model_id": model_id})
                elif req["command"] == "remove-orphans":
                    mock_args = argparse.Namespace(**req["args"])
                    mock_args.catalog = Path(mock_args.catalog)
                    mock_args.models_dir = Path(mock_args.models_dir)
                    mock_args.server_config = Path(getattr(mock_args, "server_config", DEFAULT_SERVER_CONFIG_PATH))
                    fn = _fallback_attr("remove_orphan_models")
                    if fn is None:
                        raise RuntimeError("remove_orphan_models not available")
                    result = fn(mock_args, progress_callback=send_event)  # type: ignore
                    send_event({"type": "done", "result": result})
                elif req["command"] == "unload":
                    mock_args = argparse.Namespace(**req["args"])
                    mock_args.catalog = Path(mock_args.catalog)
                    mock_args.config = Path(mock_args.config)
                    mock_args.llama_server = Path(mock_args.llama_server)
                    mock_args.server_config = Path(getattr(mock_args, "server_config", DEFAULT_SERVER_CONFIG_PATH))
                    fn = _fallback_attr("unload_models")
                    if fn is None:
                        raise RuntimeError("unload_models not available")
                    result = fn(mock_args)  # type: ignore
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


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    mod = _get_cli_file()
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [  # type: ignore
    "_file_signature",
    "sync_config_from_server_config_for_startup",
    "start_catalog_auto_update_watch",
    "_prepare_manager_socket_path",
    "read_install_manifest",
    "get_heimdall_gateway_version",
    "render_heimdall_gateway_banner",
    "build_info_text",
    "show_info",
    "daemon_mode",
]
