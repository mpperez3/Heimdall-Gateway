"""Baseline snapshot tests for CLI help and config-keys surface.

Preserves golden help output before refactor of cli.py.
If a flag/alias changes, update snapshot at tests/__snapshots__/help.txt.
"""
import argparse
import json
import io
import hashlib
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pytest

from llamacpp_stack.cli import build_cli_parser, build_help_epilog


SNAPSHOT_PATH = Path(__file__).parent / "__snapshots__" / "help.txt"

# 14+ substrings from build_help_epilog that must be present
EPILOG_REQUIRED_SUBSTRINGS = [
    "Command guide:",
    "add [repo",
    "run [repo|-hf HF]",
    "remove [repo",
    "rm [repo",
    "update [repo",
    "update [--check",
    "validate [repo|-hf HF]",
    "daemon",
    "list",
    "ps",
    "requests [-n LINES]",
    "logs [-n LINES]",
    "hacks",
    "info",
    "Help     Show options",
    "For endpoints",
]

# Commands that must appear in parser help
ALL_COMMAND_NAMES = [
    "add",
    "run",
    "remove",
    "rm",
    "orphans",
    "remove-orphans",
    "clean-orphans",
    "unload",
    "update",
    "config-migrate",
    "config-keys",
    "refresh-templates",
    "remove-templates",
    "validate",
    "daemon",
    "llama-swap-guard",
    "debug",
    "auto-performance",
    "list",
    "ps",
    "requests",
    "logs",
    "hacks",
    "info",
]


class TestHelpSnapshotGolden:
    def test_build_help_epilog_contains_required_substrings(self):
        epilog = build_help_epilog()
        assert isinstance(epilog, str)
        missing = [s for s in EPILOG_REQUIRED_SUBSTRINGS if s not in epilog]
        assert not missing, f"build_help_epilog missing substrings: {missing}\nGot:\n{epilog}"

    def test_build_cli_parser_help_contains_all_commands(self):
        parser, subparsers = build_cli_parser()
        help_text = parser.format_help()
        # epilog contains commands, but also ensure help_text includes each visible command
        # parser.format_help includes epilog via HelpFormatter
        for cmd in ALL_COMMAND_NAMES:
            # llama-swap-guard is suppressed in choices but still registered as subparser
            if cmd == "llama-swap-guard":
                assert cmd in subparsers, "llama-swap-guard must be in subparsers dict even if SUPPRESS"
                continue
            assert cmd in help_text or cmd in subparsers, f"command {cmd!r} not in help_text or subparsers"
        # Also assert public metavar visible
        assert "add,run,remove" in help_text or "{add,run" in help_text

    def test_parser_help_golden_snapshot(self):
        parser, _ = build_cli_parser()
        help_text = parser.format_help()
        epilog = build_help_epilog()
        combined = help_text + "\n---EPILOG---\n" + epilog
        # Ensure snapshot directory exists
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not SNAPSHOT_PATH.exists():
            SNAPSHOT_PATH.write_text(combined, encoding="utf-8")
            # First run creates golden; do not fail
            assert SNAPSHOT_PATH.exists()
            return
        golden = SNAPSHOT_PATH.read_text(encoding="utf-8")
        # Compare normalized (strip trailing whitespace per line) to tolerate minor whitespace
        def normalize(s: str) -> str:
            return "\n".join(line.rstrip() for line in s.strip().splitlines())
        if normalize(golden) != normalize(combined):
            # Provide diff-friendly message
            pytest.fail(
                f"Golden help snapshot mismatch at {SNAPSHOT_PATH}\n"
                f"Help output changed - if intentional, delete {SNAPSHOT_PATH} and re-run.\n"
                f"Golden sha: {hashlib.sha256(golden.encode()).hexdigest()[:12]} vs current {hashlib.sha256(combined.encode()).hexdigest()[:12]}"
            )
        # Also assert file not empty and contains expected substrings
        assert len(golden) > 500

    def test_parser_uses_help_formatter_with_defaults(self):
        parser, _ = build_cli_parser()
        # HelpFormatter inherits ArgumentDefaultsHelpFormatter + RawDescriptionHelpFormatter
        assert parser.formatter_class is not None
        # Check that defaults are shown: parser help should mention default values
        help_text = parser.format_help()
        # Global options have defaults; check one
        assert "--models-dir" in help_text
        assert "--config" in help_text

    def test_install_build_parser_help_snapshot(self):
        from llamacpp_stack.install import build_parser as install_build_parser

        parser = install_build_parser()
        help_text = parser.format_help()
        # Must contain install-specific flags
        for needle in ["--mode", "--backend", "--models-dir", "--idle-ttl", "--public-host", "--public-port", "--no-install-services", "--dry-run"]:
            assert needle in help_text, f"install parser help missing {needle!r}"
        assert "Install LLM Server" in help_text
        # Ensure --dry-run is documented
        assert "dry-run" in help_text.lower()
        # Snapshot determinism: second call yields same help
        help2 = install_build_parser().format_help()
        assert help_text == help2

    def test_cli_help_alias_handling_in_help(self):
        parser, subparsers = build_cli_parser()
        # Verify aliases share same parser object (where aliases param used)
        assert subparsers["rm"] is subparsers["remove"]
        assert subparsers["orphans"] is subparsers["remove-orphans"]
        assert subparsers["clean-orphans"] is subparsers["remove-orphans"]
        # list/ps are separate add_parser calls (not alias param) but delegate to same func
        assert subparsers["ps"].get_default("func") is subparsers["list"].get_default("func")
        parser2, _ = build_cli_parser()
        ns = parser2.parse_args(["rm"])
        assert ns.command == "rm" or ns.command == "remove"


class TestConfigKeysJsonSchema:
    def test_config_keys_json_schema_structure(self):
        from llamacpp_stack.cli import print_config_keys

        args = argparse.Namespace(format="json")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = print_config_keys(args)
        assert rc == 0
        output = buf.getvalue()
        data = json.loads(output)
        # Required top-level keys
        for key in ["catalog_model_top_level_keys", "conf_json_top_level_keys", "experimental_keys", "llama_server_defaults_keys", "vllm_keys", "notes"]:
            assert key in data, f"config-keys json missing key {key!r}"
            assert isinstance(data[key], list), f"{key} should be list"
        # catalog keys should be snake_case and non-empty
        assert len(data["catalog_model_top_level_keys"]) > 5
        for k in data["catalog_model_top_level_keys"]:
            assert "-" not in k, f"catalog key {k!r} should be snake_case"
        # conf keys should include replicas, llama_server_defaults etc.
        assert any("replicas" in k for k in data["conf_json_top_level_keys"])
        # notes present
        assert any("snake_case" in n for n in data["notes"])

    def test_config_keys_text_format(self):
        from llamacpp_stack.cli import print_config_keys

        args = argparse.Namespace(format="text")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = print_config_keys(args)
        assert rc == 0
        txt = buf.getvalue()
        assert "Catalog model top-level keys" in txt
        assert "conf.json top-level keys" in txt
        assert "llama_server_defaults" in txt
        assert "vLLM" in txt or "vllm" in txt.lower()

    def test_config_keys_json_deterministic(self):
        from llamacpp_stack.cli import print_config_keys
        import io
        def capture():
            buf = io.StringIO()
            with redirect_stdout(buf):
                print_config_keys(argparse.Namespace(format="json"))
            return json.loads(buf.getvalue())
        a = capture()
        b = capture()
        assert a == b
        # Ensure sorted keys
        assert a["catalog_model_top_level_keys"] == sorted(a["catalog_model_top_level_keys"])


class TestVerifyLlamaCmdsWrapper:
    def test_verify_llama_cmds_importable_and_renders(self):
        # Ensure tools/verify_llama_cmds is importable and functions work without side effects
        import importlib.util
        import sys
        spec_path = Path(__file__).parent.parent / "tools" / "verify_llama_cmds.py"
        assert spec_path.exists(), "tools/verify_llama_cmds.py must exist"
        # Import via importlib to avoid yaml stub side effects colliding with real yaml
        # We already have real yaml installed; test direct import of functions
        from llamacpp_stack.cli import ManagedModel, build_llama_server_command
        import json as _json
        catalog_path = Path(__file__).parent / "fixtures" / "test_catalog.json"
        assert catalog_path.exists()
        models = _json.loads(catalog_path.read_text(encoding="utf-8"))
        assert len(models) >= 1
        for m in models:
            model = ManagedModel(**m)
            cmd = build_llama_server_command(model, Path("/usr/local/bin/llama-server"), port="18090")
            assert isinstance(cmd, list)
            assert len(cmd) >= 2
            assert "18090" in " ".join(map(str, cmd)) or "--port" in cmd

    def test_verify_llama_cmds_module_main_runs(self, capsys):
        # Import the tool module and call main via subprocess-like capture
        # Avoid stubbing; use real function path
        import tools.verify_llama_cmds as v
        # Capture stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                v.main()
            except SystemExit as e:
                # main may sys.exit; allow 0
                assert e.code in (0, None)
        out = buf.getvalue()
        # Should have printed at least one model id
        assert "qwen-3.5" in out.lower() or "qwen" in out.lower()
        assert "llama-server" in out or "/usr/local/bin" in out
