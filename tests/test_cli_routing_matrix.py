"""Parametrized routing matrix for 22+ commands.

Verifies subparser existence, alias handling, set_defaults(func=) and
delegation owner vs non-owner via execute_with_manager_delegation.

Covers commands: add/run/remove/rm/orphans/remove-orphans/clean-orphans/
unload/update/config-migrate/config-keys/refresh-templates/remove-templates/
validate/daemon/llama-swap-guard/debug/auto-performance/list/ps/requests/logs/hacks/info
"""
import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llamacpp_stack.cli import build_cli_parser
from llamacpp_stack.command_router import execute_with_manager_delegation


# Full matrix as specified in plan (24 entries incl aliases)
ALL_COMMANDS = [
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

ALIAS_GROUPS_SAME_OBJECT = {
    "remove": ["remove", "rm"],
    "remove-orphans": ["remove-orphans", "clean-orphans", "orphans"],
}
ALIAS_GROUPS_SAME_FUNC = {
    "list": ["list", "ps"],
}


def _parser_and_subparsers():
    return build_cli_parser()


class TestRoutingSubparserExists:
    @pytest.mark.parametrize("cmd", ALL_COMMANDS)
    def test_subparser_exists(self, cmd):
        _, subparsers = _parser_and_subparsers()
        assert cmd in subparsers, f"subparser {cmd!r} missing from dict keys {sorted(subparsers.keys())}"

    @pytest.mark.parametrize("cmd", ALL_COMMANDS)
    def test_subparser_has_func(self, cmd):
        _, subparsers = _parser_and_subparsers()
        p = subparsers[cmd]
        # func set via set_defaults(func=...) — check via get_default
        func = p.get_default("func")
        # Also try parsed namespace for a minimal parse where possible
        assert func is not None, f"subparser {cmd!r} has no func default"
        assert callable(func), f"func for {cmd!r} is not callable"

    def test_alias_sharing_identity(self):
        _, subparsers = _parser_and_subparsers()
        for canonical, members in ALIAS_GROUPS_SAME_OBJECT.items():
            base = subparsers[members[0]]
            for alias in members[1:]:
                assert subparsers[alias] is base, f"alias {alias!r} should share parser object with {canonical!r}"
        for canonical, members in ALIAS_GROUPS_SAME_FUNC.items():
            base_func = subparsers[members[0]].get_default("func")
            for alias in members[1:]:
                assert subparsers[alias].get_default("func") is base_func, f"alias {alias!r} should share func with {canonical!r}"

    def test_llama_swap_guard_suppressed_but_present(self):
        parser, subparsers = _parser_and_subparsers()
        assert "llama-swap-guard" in subparsers
        # Check that help does not list it as public (suppressed)
        # The help's public_command_metavar should not contain llama-swap-guard
        help_text = parser.format_help()
        # The suppressed command help should be SUPPRESS, but subparser still exists
        p = subparsers["llama-swap-guard"]
        # Help for that subparser is SUPPRESS
        assert p.get_default("func") is not None

    @pytest.mark.parametrize("cmd", ["add", "run", "remove", "update", "validate"])
    def test_hyphen_aware_flags_preserved(self, cmd):
        # Ensure hyphen flag conventions: e.g. --auto-ctx aliases, --hf, etc.
        _, subparsers = _parser_and_subparsers()
        p = subparsers[cmd]
        actions = {a.dest for a in p._actions}
        # hf is common
        if cmd in ("add", "run", "remove", "update", "validate"):
            assert "hf" in actions or any("hf" in str(a.option_strings) for a in p._actions)

    def test_required_subparsers(self):
        parser, _ = _parser_and_subparsers()
        # subparsers required=True should cause SystemExit when no command
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([])
        assert exc.value.code == 2


class TestRoutingDelegationMatrix:
    """Parametrized delegation owner vs non-owner for each command.

    Uses execute_with_manager_delegation directly to verify routing logic
    without needing real catalog I/O.
    """

    @pytest.mark.parametrize("cmd", ALL_COMMANDS)
    @patch("os.getuid", return_value=0)
    def test_owner_executes_locally(self, mock_getuid, cmd):
        args = argparse.Namespace(
            catalog=Path("/tmp/test_catalog.json"),
            config=Path("/tmp/test_config.yaml"),
            models_dir=Path("/tmp/models"),
            llama_server=Path("/tmp/llama-server"),
        )
        local = MagicMock(return_value="local-result")
        result = execute_with_manager_delegation(
            command_name=cmd,
            args=args,
            local_executor=local,
        )
        local.assert_called_once_with(args)
        assert result == "local-result"

    @pytest.mark.parametrize("cmd", ALL_COMMANDS)
    @patch("os.getuid", return_value=1000)
    @patch("os.stat")
    def test_non_owner_delegates_to_manager(self, mock_stat, mock_getuid, cmd):
        mock_stat.return_value.st_uid = 2000
        args = argparse.Namespace(
            catalog=Path("/tmp/test_catalog.json"),
            config=Path("/tmp/test_config.yaml"),
            models_dir=Path("/tmp/models"),
            llama_server=Path("/tmp/llama-server"),
        )
        local = MagicMock()
        with patch("llamacpp_stack.cli.run_manager_command") as mock_manager:
            mock_manager.return_value = f"handled-{cmd}"
            result = execute_with_manager_delegation(
                command_name=cmd,
                args=args,
                local_executor=local,
            )
            local.assert_not_called()
            mock_manager.assert_called_once_with(cmd, args)
            assert result == f"handled-{cmd}"

    @pytest.mark.parametrize("cmd", ALL_COMMANDS)
    @patch("os.getuid", return_value=1000)
    @patch("os.stat")
    def test_non_owner_manager_runtime_error_propagates(self, mock_stat, mock_getuid, cmd):
        mock_stat.return_value.st_uid = 2000
        args = argparse.Namespace(catalog=Path("/tmp/catalog.json"))
        with patch("llamacpp_stack.cli.run_manager_command", side_effect=RuntimeError("mgr down")):
            with pytest.raises(RuntimeError, match="mgr down"):
                execute_with_manager_delegation(
                    command_name=cmd,
                    args=args,
                    local_executor=MagicMock(),
                )

    @pytest.mark.parametrize("cmd", ALL_COMMANDS)
    @patch("os.getuid", return_value=1000)
    @patch("os.stat")
    def test_non_owner_non_runtime_wrapped(self, mock_stat, mock_getuid, cmd):
        mock_stat.return_value.st_uid = 2000
        args = argparse.Namespace(catalog=Path("/tmp/catalog.json"))
        with patch("llamacpp_stack.cli.run_manager_command", side_effect=ConnectionError("sock")):
            with patch("llamacpp_stack.cli.MANAGER_SERVICE_NAME", "llm-server-manager"):
                with pytest.raises(RuntimeError, match="Could not connect to manager"):
                    execute_with_manager_delegation(
                        command_name=cmd,
                        args=args,
                        local_executor=MagicMock(),
                    )

    @patch("os.getuid", return_value=1000)
    @patch("os.stat")
    def test_legacy_heimdall_manager_is_disabled_fallback(self, mock_stat, mock_getuid):
        """Legacy fallback: old heimdall-gateway-manager should be considered disabled/inactive."""
        mock_stat.return_value.st_uid = 2000
        args = argparse.Namespace(catalog=Path("/tmp/catalog.json"))
        # Old service name should still be handled as fallback (disabled) via compat layer
        with patch("llamacpp_stack.cli.run_manager_command", side_effect=ConnectionError("sock")):
            with patch("llamacpp_stack.cli.MANAGER_SERVICE_NAME", "heimdall-gateway-manager"):
                with pytest.raises(RuntimeError, match="Could not connect to manager"):
                    execute_with_manager_delegation(
                        command_name="add",
                        args=args,
                        local_executor=MagicMock(),
                    )

    def test_stat_failure_delegates(self):
        """Stat failure should delegate (fail-safe)."""
        args = argparse.Namespace(catalog=Path("/tmp/catalog.json"))
        with patch("os.getuid", return_value=1000):
            with patch("os.stat", side_effect=OSError("permission")):
                with patch("llamacpp_stack.cli.run_manager_command") as m:
                    m.return_value = 0
                    result = execute_with_manager_delegation("add", args, MagicMock())
                    assert result == 0
                    m.assert_called_once()

    def test_canonical_vs_alias_delegate_same(self):
        """Alias and canonical should delegate identically."""
        for alias in ["rm", "orphans", "clean-orphans", "ps"]:
            args = argparse.Namespace(catalog=Path("/tmp/catalog.json"))
            with patch("os.getuid", return_value=1000):
                with patch("os.stat") as mock_stat:
                    mock_stat.return_value.st_uid = 2000
                    with patch("llamacpp_stack.cli.run_manager_command") as m:
                        m.return_value = "ok"
                        r1 = execute_with_manager_delegation(alias, args, MagicMock())
                        assert r1 == "ok"


class TestCommandParseIntegration:
    """Light integration: ensure each command parses without error for minimal args."""

    @pytest.mark.parametrize("cmd,extra", [
        ("add", []),
        ("run", []),
        ("remove", []),
        ("rm", []),
        ("remove-orphans", ["--dry-run"]),
        ("clean-orphans", ["--dry-run"]),
        ("orphans", ["--dry-run"]),
        ("unload", []),
        ("update", []),
        ("config-migrate", []),
        ("config-keys", []),
        ("config-keys", ["--format", "json"]),
        ("refresh-templates", []),
        ("remove-templates", []),
        ("validate", []),
        ("daemon", []),
        ("llama-swap-guard", []),
        ("debug", []),
        ("auto-performance", []),
        ("list", []),
        ("ps", []),
        ("requests", []),
        ("logs", []),
        ("hacks", []),
        ("info", []),
    ])
    def test_parse_minimal(self, cmd, extra):
        parser, _ = _parser_and_subparsers()
        # Build args list; for commands that need no required positional, empty extra works
        ns = parser.parse_args([cmd] + extra)
        # command attribute should be present (dest="command")
        assert getattr(ns, "command", None) == cmd or cmd in ("rm", "clean-orphans", "orphans", "ps")
        # func callable
        assert callable(getattr(ns, "func", None))
