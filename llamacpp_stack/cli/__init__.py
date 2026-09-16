"""LLM Server cli package - re-exports for compat.

Both `from llamacpp_stack.cli import ManagedModel` and
`from llamacpp_stack.cli.models import ManagedModel` must work.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from .constants import (
    ALTERNATE_SERVER_CONFIG_BASENAME,
    CHAT_TOOL_CONTINUE_REPAIR_THINKING_BUDGET_TOKENS,
    CLI_COMMAND,
    DEFAULT_API_CTX_FACTOR,
    DEFAULT_API_PORT,
    DEFAULT_CATALOG_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_CTX_SIZE,
    DEFAULT_IDLE_TTL,
    DEFAULT_LAST_CHAT_RESPONSE_LOG_PATH,
    DEFAULT_LLAMA_SERVER,
    DEFAULT_MAX_CONCURRENT_PER_MODEL,
    DEFAULT_MODELS_DIR,
    DEFAULT_MODEL_SWITCH_GRACE_S,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_PUBLIC_HOST,
    DEFAULT_PUBLIC_PORT,
    DEFAULT_REASONING_VISIBLE_RESERVE,
    DEFAULT_REQUESTS_LOG_PATH,
    DEFAULT_SERVER_CONFIG_PATH,
    DEFAULT_SERVICE_NAME,
    DEFAULT_START_PORT,
    DEFAULT_TENSOR_SPLIT,
    LEGACY_CLI_COMMAND,
    LLAMASWAP_UPSTREAM_STATIC_BLOCKED_BASENAMES,
    LLAMASWAP_UPSTREAM_STATIC_BLOCKED_EXTENSIONS,
    MANAGER_SERVICE_NAME,
    MODEL_PROBE_REASONING_MAX_TOKENS,
    PRODUCT_NAME,
    PRODUCT_SLUG,
    REASONING_BUDGET_HALF_CONTEXT,
    SOCKET_PATH,
    SWAP_SERVICE_NAME,
    SYSTEM_REQUESTS_LOG_PATH,
    DEFAULT_TENSOR_SPLIT as _DEFAULT_TENSOR_SPLIT,  # pyright: ignore[reportUnusedImport]
    detect_cuda_device_count,
    default_tensor_split,
)
from .env import _env_path, _env_path2, _env_value, _is_vllm_backend, _load_installed_env
from .models import LoadingBar, ManagedModel, ProbeTraceMetrics, ReplicaConfig, ReplicaRecord, Spinner

_PARSER_EXPORTS = ("HelpFormatter", "_detect_requested_subcommand", "build_help_epilog", "build_cli_parser", "parse_cli_args")
_parser_mod_cache = None
_GATEWAY_EXPORTS = (
    "DEDUP_STATE", "_DEDUP_STREAM_TEES",
    "_default_dedup_inflight_config", "_normalize_dedup_inflight_config",
    "_dedup_should_bypass", "_dedup_extract_principal", "_dedup_build_fingerprint",
    "_dedup_send_cached_response", "_dedup_send_json_with_header", "_dedup_streaming_finalize",
    "_log_chat_stop_without_tools", "_tool_call_debug_summary",
    "resolve_chat_last_response_log_config", "_normalize_experimental_config", "_default_experimental_config",
    "run_manager_command", "manager_hint", "infer_install_mode", "service_commands_for_mode",
    "get_public_endpoint_status", "_proxy_request_to_public_api", "_proxy_headers_safe",
    "is_llamaswap_upstream_static_autoload_path", "run_llamaswap_guard",
)
_gateway_mod_cache = None
_REPLICA_EXPORTS = (
    "LLAMASWAP_CONFIG_HEADER",
    "resolve_global_replica_config",
    "get_model_replica_config",
    "replica_model_id",
    "is_replica_model_id",
    "replica_base_model_id",
    "build_replica_model",
    "iter_catalog_with_replicas",
    "iter_catalog_base_models",
    "summarize_configured_replicas",
    "_calculate_llama_swap_matrix",
    "render_llamaswap_config",
    "ensure_replica_route_in_llamaswap_config",
    "ensure_replica_route",
    "shell_quote",
)
_SERVER_EXPORTS = (
    "get_server_supported_flags",
    "server_supports_flag",
    "normalize_server_overrides",
    "normalize_tensor_split",
    "_append_llama_server_flag",
    "build_vllm_server_command",
    "build_llama_server_command",
    "resolve_request_reasoning_budget",
    "resolve_api_ctx_factor",
    "resolve_llama_server_defaults",
    "resolve_vllm_defaults",
    "resolve_vllm_options",
)
_DAEMON_EXPORTS = (
    "_file_signature",
    "sync_config_from_server_config_for_startup",
    "start_catalog_auto_update_watch",
    "_prepare_manager_socket_path",
    "read_install_manifest",
    "get_heimdall_gateway_version",
    "render_heimdall_gateway_banner",
    "_install_root_from_llama_server",
    "build_info_text",
    "show_info",
    "daemon_mode",
    "get_public_endpoint_status",
    "get_api_endpoint_status",
)
_replica_mod_cache = None
_server_mod_cache = None
_daemon_mod_cache = None


def _get_gateway_mod():
    global _gateway_mod_cache
    if _gateway_mod_cache is not None:
        return _gateway_mod_cache
    try:
        import importlib
        mod = importlib.import_module("llamacpp_stack.cli.gateway")
        _gateway_mod_cache = mod
        return mod
    except Exception:
        return None


def _get_replica_mod():
    global _replica_mod_cache
    if _replica_mod_cache is not None:
        return _replica_mod_cache
    try:
        import importlib
        mod = importlib.import_module("llamacpp_stack.cli.replica")
        _replica_mod_cache = mod
        return mod
    except Exception:
        return None


def _get_server_mod():
    global _server_mod_cache
    if _server_mod_cache is not None:
        return _server_mod_cache
    try:
        import importlib
        mod = importlib.import_module("llamacpp_stack.cli.server_commands")
        _server_mod_cache = mod
        return mod
    except Exception:
        return None


def _get_daemon_mod():
    global _daemon_mod_cache
    if _daemon_mod_cache is not None:
        return _daemon_mod_cache
    try:
        import importlib
        mod = importlib.import_module("llamacpp_stack.cli.daemon")
        _daemon_mod_cache = mod
        return mod
    except Exception:
        return None


def _get_parser_mod():
    global _parser_mod_cache
    if _parser_mod_cache is not None:
        return _parser_mod_cache
    try:
        import importlib

        mod = importlib.import_module("llamacpp_stack.cli.parser")
        _parser_mod_cache = mod
        return mod
    except Exception:
        return None

__all__ = [  # pyright: ignore[reportUnsupportedDunderAll]
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
    "DEFAULT_TENSOR_SPLIT",
    "DEFAULT_START_PORT",
    "DEFAULT_PUBLIC_HOST",
    "DEFAULT_PUBLIC_PORT",
    "DEFAULT_API_PORT",
    "DEFAULT_REQUESTS_LOG_PATH",
    "DEFAULT_LAST_CHAT_RESPONSE_LOG_PATH",
    "SYSTEM_REQUESTS_LOG_PATH",
    "detect_cuda_device_count",
    "default_tensor_split",
    "_load_installed_env",
    "_env_path",
    "_env_value",
    "_env_path2",
    "_is_vllm_backend",
    "Spinner",
    "LoadingBar",
    "ManagedModel",
    "ReplicaConfig",
    "ReplicaRecord",
    "ProbeTraceMetrics",
    "HelpFormatter",
    "_detect_requested_subcommand",
    "build_help_epilog",
    "build_cli_parser",
    "parse_cli_args",
]

_cli_file_mod_cache = None


def _get_cli_file():
    global _cli_file_mod_cache
    if _cli_file_mod_cache is not None:
        return _cli_file_mod_cache
    file_path = Path(__file__).parent.parent / "cli.py"
    if not file_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("llamacpp_stack._cli_file", file_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["llamacpp_stack._cli_file"] = mod
    spec.loader.exec_module(mod)
    _cli_file_mod_cache = mod
    return mod


class _CliPackageModule(types.ModuleType):
    def __getattr__(self, name: str):  # type: ignore[override]
        if name in globals():
            return globals()[name]
        if name in _PARSER_EXPORTS:
            pmod = _get_parser_mod()
            if pmod is not None and hasattr(pmod, name):
                return getattr(pmod, name)
        if name in _GATEWAY_EXPORTS:
            gmod = _get_gateway_mod()
            if gmod is not None and hasattr(gmod, name):
                return getattr(gmod, name)
        if name in _REPLICA_EXPORTS:
            rmod = _get_replica_mod()
            if rmod is not None and hasattr(rmod, name):
                return getattr(rmod, name)
        if name in _SERVER_EXPORTS:
            smod = _get_server_mod()
            if smod is not None and hasattr(smod, name):
                return getattr(smod, name)
        if name in _DAEMON_EXPORTS:
            dmod = _get_daemon_mod()
            if dmod is not None and hasattr(dmod, name):
                return getattr(dmod, name)
        for _getter in (_get_replica_mod, _get_server_mod, _get_gateway_mod, _get_daemon_mod, _get_parser_mod):
            try:
                _m = _getter()
                if _m is not None and hasattr(_m, name):
                    return getattr(_m, name)
            except Exception:
                continue
        try:
            import importlib as _ilc2

            _cmod2 = _ilc2.import_module("llamacpp_stack.cli.constants")
            if hasattr(_cmod2, name):
                return getattr(_cmod2, name)
        except Exception:
            pass
        mod = _get_cli_file()
        if mod is not None and hasattr(mod, name):
            return getattr(mod, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __delattr__(self, name: str) -> None:  # type: ignore[override]
        try:
            super().__delattr__(name)
        except AttributeError:
            pass
        for _mod_key in list(sys.modules.keys()):
            if _mod_key.startswith("llamacpp_stack._cli_file") or _mod_key.startswith("llamacpp_stack._cli_impl"):
                try:
                    _m = sys.modules[_mod_key]
                    if name in getattr(_m, "__dict__", {}):
                        delattr(_m, name)
                    elif hasattr(_m, name):
                        try:
                            delattr(_m, name)
                        except Exception:
                            pass
                except Exception:
                    pass
        for _getter in (_get_gateway_mod, _get_daemon_mod, _get_replica_mod, _get_server_mod, _get_parser_mod):
            try:
                _m = _getter()
                if _m is not None and name in getattr(_m, "__dict__", {}):
                    delattr(_m, name)
                elif _m is not None and hasattr(_m, name):
                    try:
                        delattr(_m, name)
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            import llamacpp_stack.cli.constants as _cmod_del
            if name in getattr(_cmod_del, "__dict__", {}):
                delattr(_cmod_del, name)
            elif hasattr(_cmod_del, name):
                try:
                    delattr(_cmod_del, name)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            _impl_del = __import__("importlib").import_module("llamacpp_stack._cli_impl")
            if name in getattr(_impl_del, "__dict__", {}):
                delattr(_impl_del, name)
            elif hasattr(_impl_del, name):
                try:
                    delattr(_impl_del, name)
                except Exception:
                    pass
        except Exception:
            pass
        mod = _get_cli_file()
        if mod is not None and name in getattr(mod, "__dict__", {}):
            try:
                delattr(mod, name)
            except Exception:
                pass

    def __setattr__(self, name: str, value) -> None:  # type: ignore[override]
        super().__setattr__(name, value)
        for _mod_key in list(sys.modules.keys()):
            if _mod_key.startswith("llamacpp_stack._cli_file"):
                try:
                    _m = sys.modules[_mod_key]
                    if hasattr(_m, name):
                        setattr(_m, name, value)
                except Exception:
                    pass
        mod = _get_cli_file()
        if mod is not None:
            try:
                setattr(mod, name, value)
            except Exception:
                pass
        if name in _GATEWAY_EXPORTS or name in ("_load_server_config_payload", "_is_loopback_client", "_as_bool", "_normalize_bool_flag"):
            try:
                gmod = _get_gateway_mod()
                if gmod is not None:
                    setattr(gmod, name, value)
            except Exception:
                pass
            try:
                mod2 = _get_cli_file()
                if mod2 is not None:
                    setattr(mod2, name, value)
            except Exception:
                pass
        if name in _DAEMON_EXPORTS or name in ("_elapsed_ms", "_active_download_blocker_summary", "_args_server_config_path", "persist_server_config", "load_catalog_with_diagnostics", "resolve_global_replica_config", "render_llamaswap_config", "resolve_idle_ttl", "resolve_llama_server_defaults", "_normalize_client_host", "update_config", "start_ctx_metadata_server", "start_unexpected_unload_guard"):
            try:
                dmod = _get_daemon_mod()
                if dmod is not None:
                    setattr(dmod, name, value)
            except Exception:
                pass
            try:
                mod2 = _get_cli_file()
                if mod2 is not None:
                    setattr(mod2, name, value)
            except Exception:
                pass
        if name in ("detect_cuda_device_count", "default_tensor_split", "DEDUP_STATE", "_normalize_dedup_inflight_config", "_dedup_should_bypass"):
            try:
                import llamacpp_stack.cli.constants as const_mod
                if hasattr(const_mod, name):
                    setattr(const_mod, name, value)
            except Exception:
                pass
        try:
            import llamacpp_stack.cli.constants as const_mod2
            if hasattr(const_mod2, name):
                setattr(const_mod2, name, value)
        except Exception:
            pass
        try:
            gmod2 = _get_gateway_mod()
            if gmod2 is not None and hasattr(gmod2, name):
                setattr(gmod2, name, value)
        except Exception:
            pass
        try:
            dmod2 = _get_daemon_mod()
            if dmod2 is not None and hasattr(dmod2, name):
                setattr(dmod2, name, value)
        except Exception:
            pass
        try:
            rmod2 = _get_replica_mod()
            if rmod2 is not None and hasattr(rmod2, name):
                setattr(rmod2, name, value)
        except Exception:
            pass
        try:
            smod2 = _get_server_mod()
            if smod2 is not None and hasattr(smod2, name):
                setattr(smod2, name, value)
        except Exception:
            pass
        try:
            import importlib as _il2
            _impl = _il2.import_module("llamacpp_stack._cli_impl")
            setattr(_impl, name, value)
        except Exception:
            pass
        try:
            import importlib as _il3
            _shim = _il3.import_module("llamacpp_stack._cli_impl")
            setattr(_shim, name, value)
        except Exception:
            pass
        for _mod_key in list(sys.modules.keys()):
            if _mod_key.startswith("llamacpp_stack._cli_file"):
                try:
                    _m = sys.modules[_mod_key]
                    setattr(_m, name, value)
                except Exception:
                    pass
            if _mod_key.startswith("llamacpp_stack._cli_impl"):
                try:
                    _m = sys.modules[_mod_key]
                    setattr(_m, name, value)
                except Exception:
                    pass


def __getattr__(name: str):  # PEP562 fallback for non-class path
    if name in globals():
        return globals()[name]
    if name in _PARSER_EXPORTS:
        pmod = _get_parser_mod()
        if pmod is not None and hasattr(pmod, name):
            return getattr(pmod, name)
    if name in _GATEWAY_EXPORTS:
        gmod = _get_gateway_mod()
        if gmod is not None and hasattr(gmod, name):
            return getattr(gmod, name)
    if name in _REPLICA_EXPORTS:
        rmod = _get_replica_mod()
        if rmod is not None and hasattr(rmod, name):
            return getattr(rmod, name)
    if name in _SERVER_EXPORTS:
        smod = _get_server_mod()
        if smod is not None and hasattr(smod, name):
            return getattr(smod, name)
    if name in _DAEMON_EXPORTS:
        dmod = _get_daemon_mod()
        if dmod is not None and hasattr(dmod, name):
            return getattr(dmod, name)
    for _getter in (_get_replica_mod, _get_server_mod, _get_gateway_mod, _get_daemon_mod, _get_parser_mod):
        try:
            _m = _getter()
            if _m is not None and hasattr(_m, name):
                return getattr(_m, name)
        except Exception:
            continue
    try:
        import importlib as _ilc

        _cmod = _ilc.import_module("llamacpp_stack.cli.constants")
        if hasattr(_cmod, name):
            return getattr(_cmod, name)
    except Exception:
        pass
    try:
        import importlib as _ilm

        _mm = _ilm.import_module("llamacpp_stack.cli.models")
        if hasattr(_mm, name):
            return getattr(_mm, name)
    except Exception:
        pass
    mod = _get_cli_file()
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


sys.modules[__name__].__class__ = _CliPackageModule

try:
    _self_mod = sys.modules[__name__]
    for _exp, _getter in (
        (_PARSER_EXPORTS, _get_parser_mod),
        (_GATEWAY_EXPORTS, _get_gateway_mod),
        (_REPLICA_EXPORTS, _get_replica_mod),
        (_SERVER_EXPORTS, _get_server_mod),
        (_DAEMON_EXPORTS, _get_daemon_mod),
    ):
        try:
            _m = _getter()
        except Exception:
            continue
        if _m is None:
            continue
        for _n in _exp:
            if _n in _self_mod.__dict__:
                continue
            try:
                _v = getattr(_m, _n)
            except Exception:
                continue
            try:
                object.__setattr__(_self_mod, _n, _v)
            except Exception:
                try:
                    _self_mod.__dict__[_n] = _v
                except Exception:
                    pass
    try:
        import importlib as _il_fallback
        _impl_fallback = _il_fallback.import_module("llamacpp_stack._cli_impl")
        _sub_names: set[str] = set()
        for _getter in (_get_replica_mod, _get_server_mod, _get_gateway_mod, _get_daemon_mod, _get_parser_mod):
            try:
                _m = _getter()
                if _m is not None:
                    _sub_names.update(dir(_m))
            except Exception:
                pass
        try:
            import llamacpp_stack.cli.constants as _cmod_check
            _sub_names.update(dir(_cmod_check))
        except Exception:
            pass
        for _n in dir(_impl_fallback):
            if _n.startswith("__"):
                continue
            if _n in _self_mod.__dict__:
                continue
            if _n in _sub_names:
                continue
            try:
                _v = getattr(_impl_fallback, _n)
            except Exception:
                continue
            if isinstance(_v, type(sys)):
                continue
            try:
                object.__setattr__(_self_mod, _n, _v)
            except Exception:
                try:
                    _self_mod.__dict__[_n] = _v
                except Exception:
                    pass
    except Exception:
        pass
except Exception:
    pass
