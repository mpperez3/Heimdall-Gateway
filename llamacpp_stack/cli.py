#!/usr/bin/env python3
"""
Heimdall Gateway cli shim - re-exports from cli/* subpackage.

All business logic lives in llamacpp_stack/cli/*.py (env, constants, models,
parser, replica, server_commands, gateway, daemon) and the remaining legacy
implementation in llamacpp_stack/_cli_impl.py. This file contains no duplicated
business logic and stays <400 lines for T7.
"""

from __future__ import annotations

import importlib
import sys

# ---------------------------------------------------------------------------
# Re-export public symbols from modularized subpackage
# (explicit imports avoid star pollution and keep type checkers happy)
# ---------------------------------------------------------------------------

from llamacpp_stack.cli.constants import (  # noqa: F401
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
    detect_cuda_device_count,
    default_tensor_split,
)

from llamacpp_stack.cli.env import (  # noqa: F401
    _env_path,
    _env_path2,
    _env_value,
    _is_vllm_backend,
    _load_installed_env,
)

from llamacpp_stack.cli.models import (  # noqa: F401
    LoadingBar,
    ManagedModel,
    ProbeTraceMetrics,
    ReplicaConfig,
    ReplicaRecord,
    Spinner,
)

from llamacpp_stack.cli.parser import (  # noqa: F401
    HelpFormatter,
    _detect_requested_subcommand,
    build_cli_parser,
    build_help_epilog,
    parse_cli_args,
)

from llamacpp_stack.cli.replica import (  # noqa: F401
    LLAMASWAP_CONFIG_HEADER,
    _calculate_llama_swap_matrix,
    build_replica_model,
    ensure_replica_route_in_llamaswap_config,
    get_model_replica_config,
    is_replica_model_id,
    iter_catalog_base_models,
    iter_catalog_with_replicas,
    render_llamaswap_config,
    replica_base_model_id,
    replica_model_id,
    resolve_global_replica_config,
    shell_quote,
    summarize_configured_replicas,
)

from llamacpp_stack.cli.server_commands import (  # noqa: F401
    _append_llama_server_flag,
    build_llama_server_command,
    build_vllm_server_command,
    get_server_supported_flags,
    normalize_server_overrides,
    normalize_tensor_split,
    resolve_api_ctx_factor,
    resolve_llama_server_defaults,
    resolve_request_reasoning_budget,
    resolve_vllm_defaults,
    resolve_vllm_options,
    server_supports_flag,
)

from llamacpp_stack.cli.gateway import (  # noqa: F401
    DEDUP_STATE,
    _DEDUP_STREAM_TEES,
    _dedup_build_fingerprint,
    _dedup_extract_principal,
    _dedup_send_cached_response,
    _dedup_send_json_with_header,
    _dedup_should_bypass,
    _dedup_streaming_finalize,
    _default_dedup_inflight_config,
    _default_experimental_config,
    _log_chat_stop_without_tools,
    _normalize_dedup_inflight_config,
    _normalize_experimental_config,
    _tool_call_debug_summary,
    get_public_endpoint_status,
    infer_install_mode,
    is_llamaswap_upstream_static_autoload_path,
    manager_hint,
    resolve_chat_last_response_log_config,
    run_llamaswap_guard,
    run_manager_command,
    service_commands_for_mode,
)

from llamacpp_stack.cli.daemon import (  # noqa: F401
    _file_signature,
    _install_root_from_llama_server,
    _prepare_manager_socket_path,
    build_info_text,
    daemon_mode,
    get_heimdall_gateway_version,
    read_install_manifest,
    render_heimdall_gateway_banner,
    show_info,
    start_catalog_auto_update_watch,
    sync_config_from_server_config_for_startup,
)
try:
    from llamacpp_stack.cli.gateway import get_api_endpoint_status  # noqa: F401
except ImportError:
    pass

# Legacy fallback: any symbol not yet modularized lives in _cli_impl
_cli_impl_cache = None


def _get_cli_impl():
    global _cli_impl_cache
    if _cli_impl_cache is not None:
        return _cli_impl_cache
    try:
        mod = importlib.import_module("llamacpp_stack._cli_impl")
        _cli_impl_cache = mod
        return mod
    except Exception:
        return None


def __getattr__(name: str):  # PEP 562
    # First, try already-imported globals (should have been handled by import system)
    if name in globals():
        return globals()[name]
    mod = _get_cli_impl()
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    keys = set(globals().keys())
    mod = _get_cli_impl()
    if mod is not None:
        keys.update(k for k in dir(mod) if not k.startswith("_"))
    return sorted(keys)


# Minimal __all__ for wildcard imports; kept in sync with package __all__
__all__ = [  # noqa: F822
    "PRODUCT_NAME", "PRODUCT_SLUG", "SOCKET_PATH",
    "DEFAULT_MODELS_DIR", "DEFAULT_CONFIG_PATH", "DEFAULT_CATALOG_PATH",
    "DEFAULT_SERVER_CONFIG_PATH", "ALTERNATE_SERVER_CONFIG_BASENAME",
    "DEFAULT_SERVICE_NAME", "CLI_COMMAND", "LEGACY_CLI_COMMAND",
    "MANAGER_SERVICE_NAME", "SWAP_SERVICE_NAME", "DEFAULT_LLAMA_SERVER",
    "DEFAULT_CTX_SIZE", "REASONING_BUDGET_HALF_CONTEXT",
    "DEFAULT_REASONING_VISIBLE_RESERVE",
    "CHAT_TOOL_CONTINUE_REPAIR_THINKING_BUDGET_TOKENS",
    "MODEL_PROBE_REASONING_MAX_TOKENS", "DEFAULT_API_CTX_FACTOR",
    "DEFAULT_N_GPU_LAYERS", "DEFAULT_IDLE_TTL",
    "DEFAULT_MODEL_SWITCH_GRACE_S", "DEFAULT_MAX_CONCURRENT_PER_MODEL",
    "LLAMASWAP_UPSTREAM_STATIC_BLOCKED_BASENAMES",
    "LLAMASWAP_UPSTREAM_STATIC_BLOCKED_EXTENSIONS",
    "DEFAULT_TENSOR_SPLIT", "DEFAULT_START_PORT", "DEFAULT_PUBLIC_HOST",
    "DEFAULT_PUBLIC_PORT", "DEFAULT_API_PORT", "DEFAULT_REQUESTS_LOG_PATH",
    "DEFAULT_LAST_CHAT_RESPONSE_LOG_PATH", "SYSTEM_REQUESTS_LOG_PATH",
    "detect_cuda_device_count", "default_tensor_split",
    "_load_installed_env", "_env_path", "_env_value", "_env_path2", "_is_vllm_backend",
    "Spinner", "LoadingBar", "ManagedModel", "ReplicaConfig", "ReplicaRecord", "ProbeTraceMetrics",
    "HelpFormatter", "_detect_requested_subcommand", "build_help_epilog", "build_cli_parser", "parse_cli_args",
    "LLAMASWAP_CONFIG_HEADER", "resolve_global_replica_config", "get_model_replica_config",
    "replica_model_id", "is_replica_model_id", "replica_base_model_id", "build_replica_model",
    "iter_catalog_with_replicas", "iter_catalog_base_models", "summarize_configured_replicas",
    "_calculate_llama_swap_matrix", "render_llamaswap_config", "ensure_replica_route_in_llamaswap_config", "shell_quote",
    "get_server_supported_flags", "server_supports_flag", "normalize_server_overrides", "normalize_tensor_split",
    "_append_llama_server_flag", "build_vllm_server_command", "build_llama_server_command",
    "resolve_request_reasoning_budget", "resolve_api_ctx_factor", "resolve_llama_server_defaults",
    "resolve_vllm_defaults", "resolve_vllm_options",
    "DEDUP_STATE", "_DEDUP_STREAM_TEES", "_default_dedup_inflight_config", "_normalize_dedup_inflight_config",
    "_dedup_should_bypass", "_dedup_extract_principal", "_dedup_build_fingerprint",
    "_dedup_send_cached_response", "_dedup_send_json_with_header", "_dedup_streaming_finalize",
    "_log_chat_stop_without_tools", "_tool_call_debug_summary", "resolve_chat_last_response_log_config",
    "_normalize_experimental_config", "_default_experimental_config",
    "run_manager_command", "manager_hint", "get_public_endpoint_status", "infer_install_mode", "service_commands_for_mode",
    "is_llamaswap_upstream_static_autoload_path", "run_llamaswap_guard",
    "_file_signature", "sync_config_from_server_config_for_startup", "start_catalog_auto_update_watch",
    "_prepare_manager_socket_path", "read_install_manifest", "get_heimdall_gateway_version",
    "render_heimdall_gateway_banner", "_install_root_from_llama_server", "build_info_text", "show_info",
    "daemon_mode", "get_api_endpoint_status",
]


# Ensure heap for main entry point delegation
def _delegate_main(argv=None):
    mod = _get_cli_impl()
    if mod is not None and hasattr(mod, "main"):
        return mod.main(argv)
    raise RuntimeError("legacy main not available")


# Expose main for llamacpp_api_install entry
try:
    _impl = _get_cli_impl()
    if _impl is not None and hasattr(_impl, "main"):
        main = _impl.main  # type: ignore[attr-defined]
except Exception:
    pass
