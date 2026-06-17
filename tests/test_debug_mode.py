from pathlib import Path
from unittest.mock import patch

from llamacpp_stack.cli import (
    ManagedModel,
    _disable_debug_gate,
    _enable_debug_gate,
    _is_debug_gate_active,
    build_cli_parser,
    build_llama_server_command,
)
from llamacpp_stack.debug_manager import DEBUG_SESSION_MANAGER, build_debug_model_name, build_optimized_model_name, parse_debug_flags


class _DummyProcess:
    def __init__(self, pid: int, alive: bool = True):
        self.pid = pid
        self._alive = alive

    def poll(self):
        return None if self._alive else 1

    def terminate(self):
        self._alive = False

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._alive = False


def test_parse_debug_flags_accepts_dict_and_string():
    tokens, normalized = parse_debug_flags({"n_gpu_layers": 42, "flash_attn": "on", "tensor_split": "1,1"})

    assert normalized["n_gpu_layers"] == 42
    assert "--n-gpu-layers" in tokens
    assert "42" in tokens
    assert "--flash-attn" in tokens
    assert "on" in tokens

    string_tokens, string_normalized = parse_debug_flags("--foo bar --baz=qux")
    assert string_tokens == ["--foo", "bar", "--baz=qux"]
    assert string_normalized["raw_flags"] == "--foo bar --baz=qux"


def test_build_model_names_include_expected_suffixes():
    debug_name = build_debug_model_name("foo-model", "1234567890abcdef")
    assert "foo-model" in debug_name
    assert "DEBUG" in debug_name

    optimized_name = build_optimized_model_name("foo-model")
    assert "foo-model" in optimized_name
    assert "Optimised" in optimized_name


def test_build_llama_server_command_appends_generic_and_extra_flags():
    model = ManagedModel(
        model_id="foo",
        repo_id="org/foo",
        quant=None,
        filename="foo.gguf",
        local_path="/models/foo.gguf",
        ctx_size=8192,
        server_overrides={"tensor_split": "1,1", "custom_flag": "enabled"},
    )

    cmd = build_llama_server_command(
        model,
        Path("/bin/llama-server"),
        port="12345",
        extra_flags=["--debug-mode", "true"],
    )
    joined = " ".join(cmd)

    assert "--tensor-split 1,1" in joined
    assert "--custom-flag enabled" in joined
    assert joined.endswith("--debug-mode true")


def test_debug_metrics_snapshot_idle_is_structured():
    snapshot = DEBUG_SESSION_MANAGER.get_metrics_snapshot()

    assert snapshot["active"] is False
    assert snapshot["status"] == "idle"
    assert snapshot["gpu"]["devices"] == []
    assert snapshot["trace"] is None or snapshot["trace"]["metrics"] == {}


def test_debug_session_clears_when_process_exits():
    previous = DEBUG_SESSION_MANAGER._session
    try:
        DEBUG_SESSION_MANAGER._session = type(
            "Record",
            (),
            {
                "session_id": "dead-session",
                "base_model_id": "foo",
                "debug_model_id": "foo [DEBUG]",
                "process": _DummyProcess(123, alive=False),
                "trace_path": None,
                "trace_handle": None,
                "catalog_path": None,
            },
        )()

        assert DEBUG_SESSION_MANAGER.get_session() is None
        assert DEBUG_SESSION_MANAGER._session is None
    finally:
        DEBUG_SESSION_MANAGER._session = previous


def test_debug_metrics_snapshot_includes_current_config():
    previous = DEBUG_SESSION_MANAGER._session
    try:
        DEBUG_SESSION_MANAGER._session = type(
            "Record",
            (),
            {
                "session_id": "active-session",
                "base_model_id": "foo",
                "debug_model_id": "foo [DEBUG]",
                "port": 12345,
                "ctx_size": 8192,
                "n_gpu_layers": 42,
                "tensor_split": "1,1",
                "description": "debug model",
                "process": _DummyProcess(321, alive=True),
                "trace_path": None,
                "trace_handle": None,
                "command": ["/bin/llama-server", "--debug"],
                "flags": {"n_gpu_layers": 42},
                "extra_tokens": ["--debug"],
                "started_at": 0.0,
                "catalog_path": None,
            },
        )()

        snapshot = DEBUG_SESSION_MANAGER.get_metrics_snapshot()

        assert snapshot["active"] is True
        assert snapshot["current_config"]["ctx_size"] == 8192
        assert snapshot["current_config"]["n_gpu_layers"] == 42
        assert snapshot["current_config"]["tensor_split"] == "1,1"
        assert snapshot["current_config"]["command"] == ["/bin/llama-server", "--debug"]
    finally:
        DEBUG_SESSION_MANAGER._session = previous


def test_websocket_accept_key_matches_rfc_example():
    accept_key = DEBUG_SESSION_MANAGER.websocket_accept_key("dGhlIHNhbXBsZSBub25jZQ==")

    assert accept_key == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_cli_parser_registers_debug_subcommand():
    parser, _ = build_cli_parser()
    args = parser.parse_args(["debug"])

    assert args.command == "debug"
    assert callable(args.func)


def test_debug_gate_toggle():
    _disable_debug_gate()

    assert _is_debug_gate_active() is False
    assert _enable_debug_gate("tester", ttl_s=5.0) is True
    assert _is_debug_gate_active() is True
    assert _disable_debug_gate("tester") is True
    assert _is_debug_gate_active() is False


def test_debug_mode_connects_to_manager_socket():
    """Test that debug_mode connects to manager socket instead of launching new server."""
    parser, _ = build_cli_parser()
    args = parser.parse_args(["debug"])

    # Mock socket to simulate manager daemon connection
    mock_socket = type("MockSocket", (), {
        "connect": lambda self, path: None,
        "sendall": lambda self, data: None,
        "close": lambda self: None,
    })()

    with patch("llamacpp_stack.cli.socket.socket", return_value=mock_socket):
        with patch("llamacpp_stack.cli.time.sleep", side_effect=KeyboardInterrupt):
            result = args.func(args)
            
            # Should return 0 (success) even with KeyboardInterrupt (Ctrl+C)
            assert result == 0


def test_get_server_output_returns_idle_when_no_session():
    previous = DEBUG_SESSION_MANAGER._session
    try:
        DEBUG_SESSION_MANAGER._session = None
        result = DEBUG_SESSION_MANAGER.get_server_output(lines=50)
        assert result["status"] == "idle"
        assert result["output"] == ""
    finally:
        DEBUG_SESSION_MANAGER._session = previous


def test_restart_server_returns_error_when_no_session():
    previous = DEBUG_SESSION_MANAGER._session
    try:
        DEBUG_SESSION_MANAGER._session = None
        result = DEBUG_SESSION_MANAGER.restart_server(timeout_s=30)
        assert result["status"] == "error"
        assert "No active debug session" in result["message"]
    finally:
        DEBUG_SESSION_MANAGER._session = previous


def test_get_server_output_previous_empty_when_no_restart():
    previous = DEBUG_SESSION_MANAGER._session
    try:
        DEBUG_SESSION_MANAGER._session = type(
            "Record",
            (),
            {
                "session_id": "active-session",
                "debug_model_id": "foo [DEBUG]",
                "previous_server_output": "",  # No restart yet
                "process": _DummyProcess(123, alive=True),
            },
        )()

        result = DEBUG_SESSION_MANAGER.get_server_output_previous(lines=50)
        assert result["status"] == "running"
        assert result["previous_output"] == ""
        assert "No previous output" in result.get("note", "")
    finally:
        DEBUG_SESSION_MANAGER._session = previous


def test_get_server_output_previous_returns_saved_output():
    previous = DEBUG_SESSION_MANAGER._session
    try:
        saved_output = "line1\nline2\nline3\nline4\nline5"
        DEBUG_SESSION_MANAGER._session = type(
            "Record",
            (),
            {
                "session_id": "active-session",
                "debug_model_id": "foo [DEBUG]",
                "previous_server_output": saved_output,
                "process": _DummyProcess(123, alive=True),
            },
        )()

        result = DEBUG_SESSION_MANAGER.get_server_output_previous(lines=3)
        assert result["status"] == "running"
        assert "line3" in result["previous_output"]
        assert "line4" in result["previous_output"]
        assert "line5" in result["previous_output"]
        assert "Yes, this is output before last restart" in result.get("from_restart", "")
    finally:
        DEBUG_SESSION_MANAGER._session = previous
