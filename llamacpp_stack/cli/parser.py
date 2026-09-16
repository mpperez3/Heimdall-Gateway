"""CLI parser extracted from llamacpp_stack/cli.py:19373-19835.

Contains HelpFormatter, build_help_epilog, _detect_requested_subcommand,
build_cli_parser, parse_cli_args. Handler funcs are lazy-imported inside
build_cli_parser to avoid top-level circular imports (cli.py <-> parser.py).

Preserves all subparsers, help strings, aliases, suppressed help,
required=True, lazy imports for auto-performance/debug exactly line-by-line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .constants import (
    CLI_COMMAND,
    DEFAULT_CATALOG_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_CTX_SIZE,
    DEFAULT_LLAMA_SERVER,
    DEFAULT_MODELS_DIR,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_PUBLIC_HOST,
    DEFAULT_PUBLIC_PORT,
    DEFAULT_SERVER_CONFIG_PATH,
    DEFAULT_SERVICE_NAME,
    DEFAULT_START_PORT,
)


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
        "  update [--check|--dry-run] [--only DEP ...] [--yes] [--force] [--list]\n"
        "    Update registered dependencies (llama.cpp, llama-swap, vLLM, heimdall-gateway, ...).\n"
        f"    Example: {CLI_COMMAND} update --check\n"
        f"    Example: {CLI_COMMAND} update --only vllm --yes\n"
        f"    Example: {CLI_COMMAND} update --list\n"
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
    # Lazy imports to avoid circular top-level dependency (cli.py imports parser)
    # Use dynamic lookup via _cli_file to keep pyright cycle-free.
    import sys as _sys

    _cli_mod = _sys.modules.get("llamacpp_stack._cli_file")
    if _cli_mod is None:
        try:
            import importlib as _il

            _pkg = _il.import_module("llamacpp_stack.cli")
            _getter = getattr(_pkg, "_get_cli_file", None)
            if callable(_getter):
                _cli_mod = _getter()
        except Exception:
            pass
    if _cli_mod is None:
        _cli_mod = _sys.modules.get("llamacpp_stack._cli_file")
    assert _cli_mod is not None, "cli file module not loaded"
    add_models = getattr(_cli_mod, "add_models")
    daemon_mode = getattr(_cli_mod, "daemon_mode")
    debug_mode = getattr(_cli_mod, "debug_mode")
    list_models = getattr(_cli_mod, "list_models")
    migrate_server_config = getattr(_cli_mod, "migrate_server_config")
    print_config_keys = getattr(_cli_mod, "print_config_keys")
    refresh_templates = getattr(_cli_mod, "refresh_templates")
    remove_models = getattr(_cli_mod, "remove_models")
    remove_orphan_models = getattr(_cli_mod, "remove_orphan_models")
    remove_templates = getattr(_cli_mod, "remove_templates")
    run_command = getattr(_cli_mod, "run_command")
    run_llamaswap_guard = getattr(_cli_mod, "run_llamaswap_guard")
    show_hacks = getattr(_cli_mod, "show_hacks")
    show_info = getattr(_cli_mod, "show_info")
    show_logs = getattr(_cli_mod, "show_logs")
    show_request_log = getattr(_cli_mod, "show_request_log")
    unload_models = getattr(_cli_mod, "unload_models")
    update_models = getattr(_cli_mod, "update_models")
    validate_model = getattr(_cli_mod, "validate_model")

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
        help="Refresh model configuration or update dependencies",
        description=(
            "Refresh model configuration and optionally re-probe context limits.\n"
            "When invoked with --check/--dry-run/--only/--list or --deps, updates registered "
            "dependencies (llama.cpp, llama-swap, vLLM, heimdall-gateway, and any future tech "
            "registered via llamacpp_stack.dependencies.register_dependency) instead of model configs."
        ),
    )
    subparsers["update"] = p_update
    p_update.set_defaults(func=update_models)
    p_update.add_argument("repo", nargs="*", help="Model id or HF repo[:QUANT] (accepts a list). When --deps/--only is used, repo values are treated as dependency names (llama.cpp, llama-swap, vllm, heimdall-gateway).")
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
    dep_group = p_update.add_argument_group("dependency update options (extensible via dependencies.register_dependency)")
    dep_group.add_argument("--check", action="store_true", help="Check for dependency updates without installing (implies --dry-run)")
    dep_group.add_argument("--dry-run", action="store_true", help="Show what would be updated without changing the system")
    dep_group.add_argument("--yes", "-y", action="store_true", help="Assume yes to confirmation prompts")
    dep_group.add_argument("--force", action="store_true", help="Force re-install even if already at latest version")
    dep_group.add_argument("--only", nargs="+", help="Only update the named dependencies (repeatable, comma-separated). Choices: llama.cpp, llama-swap, vllm, heimdall-gateway (aliases accepted)")
    dep_group.add_argument("--list", dest="list_deps", action="store_true", help="List all registered updatable dependencies and exit")
    dep_group.add_argument("--deps", action="store_true", help="Explicitly run in dependency-update mode (required when a dependency name collides with a model id)")
    dep_group.add_argument("--verbose", "-v", action="store_true", help="Verbose output during dependency updates")
    dep_group.add_argument("--llama-cpp-ref", dest="llama_cpp_ref", help="For llama.cpp updates: specific git commit/tag/ref to build (implies source build, same as install --llama-cpp-ref). If omitted in interactive TTY, you will be asked latest vs commit.")

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
        description="Print valid LLM Server catalog/conf keys and explain where raw llama.cpp flags belong.",
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
        description="Internal guard proxy for llama-swap (llm-server).",
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


__all__ = [
    "HelpFormatter",
    "_detect_requested_subcommand",
    "build_help_epilog",
    "build_cli_parser",
    "parse_cli_args",
]
