from llamacpp_stack.cli import (
    _chat_response_to_responses_payload,
    _summarize_api_payload_for_log,
    _responses_raw_passthrough_enabled,
    _responses_payload_to_chat_payload,
    _responses_internal_round_max_tokens,
    _responses_namespace_tool_map,
    _responses_tools_to_chat_tools,
    _responses_tools_with_deferred_search,
    ResponsesToolRegistry,
    _responses_tool_choice_to_chat_tool_choice,
    _responses_input_to_openai_messages,
    _write_responses_sse,
    _responses_payload_sse_events,
    _responses_payload_has_output_items,
    _translate_internal_deferred_tool_calls_in_chat_response,
    _chat_response_internal_tool_search_followup_messages,
    _chat_response_internal_tool_repair_followup_messages,
    _chat_tool_continue_repair_messages,
    _chat_tool_continue_trigger_reason,
    _apply_chat_tool_continue_repair_token_cap,
    _force_tool_choice_for_chat_repair,
    _send_openai_chat_sse_status,
    _buffer_openai_chat_sse_with_keepalive,
    _sanitize_responses_tool_arguments,
    _summarize_chat_tool_message_diagnostics,
    _strip_chat_tool_repair_notice_text,
    _sanitize_chat_tool_repair_notices_in_messages,
    _normalize_experimental_config,
)
import inspect
import io
import json
import threading
import llamacpp_stack.cli as cli


def test_chat_last_response_log_config_normalizes_defaults_and_overrides():
    normalized = _normalize_experimental_config({
        "chat_last_response_log": {
            "enabled": True,
            "path": " /tmp/last.json ",
            "max_chars": "123",
            "include_reasoning": True,
            "include_tool_calls": False,
        }
    })

    cfg = normalized["chat_last_response_log"]
    assert cfg["enabled"] is True
    assert cfg["path"] == "/tmp/last.json"
    assert cfg["max_chars"] == 123
    assert cfg["include_reasoning"] is True
    assert cfg["include_tool_calls"] is False


def test_chat_last_response_log_writes_bounded_snapshot(monkeypatch, tmp_path):
    target = tmp_path / "last-chat-response.json"
    monkeypatch.setattr(
        cli,
        "resolve_chat_last_response_log_config",
        lambda args=None: {
            "enabled": True,
            "path": str(target),
            "max_chars": 5,
            "include_reasoning": True,
            "include_tool_calls": True,
        },
    )
    monkeypatch.setattr(cli, "log_api_event", lambda *args, **kwargs: None)

    cli._write_chat_last_response_log(
        None,
        request_id="chat_req_test",
        model="public-model",
        upstream_model="internal-model",
        stream=True,
        content="abcdefghi",
        reasoning="razonamiento largo",
        tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": "{\"cmd\":\"ls\"}"}}],
        tool_call_chunks=1,
        finish_reason="tool_calls",
        repair_rounds=1,
    )

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["request_id"] == "chat_req_test"
    assert payload["visible_content"] == "abcde"
    assert payload["visible_content_len"] == 9
    assert payload["visible_content_truncated"] is True
    assert payload["reasoning"] == "razon"
    assert payload["reasoning_included"] is True
    assert payload["tool_calls"][0]["name"] == "terminal"


def _feedback_json_from_tool_content(content: str) -> dict:
    marker = "Diagnostic JSON:"
    assert marker in content
    return json.loads(content.split(marker, 1)[1].strip())


def _sample_chat_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "description": "Runs a command",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
            },
        }
    ]



def test_chat_tool_message_diagnostics_flags_suppressed_sudo_output():
    summary = _summarize_chat_tool_message_diagnostics([
        {"role": "user", "content": "hi"},
        {
            "role": "tool",
            "tool_call_id": "call_terminal",
            "name": "terminal",
            "content": "sudo systemctl restart demo\n[terminal output suppressed]",
        },
    ])

    assert summary["tool_message_count"] == 1
    assert summary["matches"][0]["tool_call_id"] == "call_terminal"
    assert "terminal_output_suppressed" in summary["matches"][0]["patterns"]
    assert "sudo" in summary["matches"][0]["patterns"]
    assert "[terminal output suppressed]" in summary["matches"][0]["preview"]

def test_chat_tool_continue_triggers_on_empty_trailing_colon_or_configured_prefix():
    tools = _sample_chat_tools()

    assert _chat_tool_continue_trigger_reason("", [], tools) == "empty_visible_content"
    assert _chat_tool_continue_trigger_reason("   ", [], tools) == "empty_visible_content"
    assert _chat_tool_continue_trigger_reason("Déjame editar:", [], tools) == "visible_content_trailing_colon"
    assert _chat_tool_continue_trigger_reason("Déjame editar el archivo", [], tools) == ""
    assert _chat_tool_continue_trigger_reason("Let me edit the file", [], tools) == ""
    assert _chat_tool_continue_trigger_reason("Voy a ejecutar el comando", [], tools, ["Voy a"]) == "visible_content_configured_prefix"
    assert _chat_tool_continue_trigger_reason("  [terminal command] sudo systemctl restart", [], tools, ["[terminal command"]) == "visible_content_configured_prefix"
    assert _chat_tool_continue_trigger_reason("voy a ejecutar el comando", [], tools, ["Voy a"]) == "visible_content_configured_prefix"
    assert _chat_tool_continue_trigger_reason("Te explico: voy a ejecutar el comando", [], tools, ["Voy a"]) == ""
    assert _chat_tool_continue_trigger_reason("Voy a revisar:\n[terminal command=\"sg docker -c 'docker compose ls'\" timeout=10]", [], tools, ["[terminal command"]) == "visible_content_configured_prefix_line"
    tools_with_search = tools + [{"type": "function", "function": {"name": "search_files"}}]
    assert _chat_tool_continue_trigger_reason("[search_files output_mode=\"files_only\" path=\"/tmp\"]", [], tools_with_search) == "visible_content_pseudo_tool_line"
    assert _chat_tool_continue_trigger_reason("Voy a revisar:\n[search_files output_mode=\"files_only\" path=\"/tmp\"]", [], tools_with_search) == "visible_content_pseudo_tool_line"
    assert _chat_tool_continue_trigger_reason("[unknown_tool foo=\"bar\"]", [], tools_with_search) == ""




def test_chat_tool_continue_repair_notice_is_stripped_from_visible_content_and_history():
    notice = "↻ Retrying tool call generation (attempt 1/4); the previous model turn did not produce a valid tool call. Waiting for the repaired tool call…"
    contaminated = f"Voy a escribir la documentación por partes. Primera sección:\n\n{notice}"

    assert _strip_chat_tool_repair_notice_text(contaminated) == "Voy a escribir la documentación por partes. Primera sección:"
    assert _chat_tool_continue_trigger_reason(contaminated, [], _sample_chat_tools()) == "visible_content_trailing_colon"

    sanitized = _sanitize_chat_tool_repair_notices_in_messages([
        {"role": "assistant", "content": contaminated},
        {"role": "user", "content": "continua"},
    ])
    assert sanitized[0]["content"] == "Voy a escribir la documentación por partes. Primera sección:"
    assert sanitized[1]["content"] == "continua"


def test_chat_tool_continue_repair_think_notice_is_stripped():
    notice = "<think>\n↻ Retrying tool call generation (attempt 2/4); the previous model turn did not produce a valid tool call. Waiting for the repaired tool call…\n</think>"
    contaminated = f"Let me use a different approach:\n{notice}"

    assert _strip_chat_tool_repair_notice_text(contaminated) == "Let me use a different approach:"
    assert _chat_tool_continue_trigger_reason(contaminated, [], _sample_chat_tools()) == "visible_content_trailing_colon"

def test_chat_tool_continue_does_not_trigger_without_tools_or_with_tool_calls():
    tools = _sample_chat_tools()

    assert _chat_tool_continue_trigger_reason("", [], []) == ""
    assert _chat_tool_continue_trigger_reason(":", [{"id": "call_1"}], tools) == ""


def test_chat_tool_continue_repair_message_lists_available_tools():
    repaired = _chat_tool_continue_repair_messages(
        [{"role": "user", "content": "hazlo"}],
        {"role": "assistant", "content": "Déjame editar:"},
        _sample_chat_tools(),
    )

    assert repaired[-2] == {"role": "assistant", "content": "Déjame editar:"}
    assert repaired[-1]["role"] == "user"
    assert "Your previous assistant message ended without any tool_calls" in repaired[-1]["content"]
    assert "you must call one of the available tools" in repaired[-1]["content"]
    assert "exec_command" in repaired[-1]["content"]




def test_chat_tool_continue_repair_status_notice_is_visible_sse_delta():
    class Handler:
        def __init__(self):
            self.wfile = io.BytesIO()

    handler = Handler()
    _send_openai_chat_sse_status(
        handler,
        request_id="chat_req_test",
        model="local/model",
        content="\n↻ Retrying tool call generation (attempt 1/4)…\n",
    )

    raw = handler.wfile.getvalue().decode("utf-8")
    assert raw.startswith("data: ")
    payload = json.loads(raw.split("data: ", 1)[1])
    assert payload["choices"][0]["delta"]["content"].startswith("\n↻ Retrying")
    assert payload["choices"][0]["finish_reason"] is None



def test_chat_tool_continue_repair_buffer_suppresses_fast_visible_notice():
    class Handler:
        def __init__(self):
            self.wfile = io.BytesIO()

    class Response:
        def iter_lines(self):
            yield b"data: [DONE]"

        def close(self):
            pass

    handler = Handler()
    lines, passthrough, state = _buffer_openai_chat_sse_with_keepalive(
        handler,
        Response(),
        request_id="chat_req_test",
        keepalive_seconds=30,
        write_lock=threading.Lock(),
        visible_status={"request_id": "chat_req_test", "model": "local/model", "content": "visible repair"},
        visible_notice_after_seconds=4,
    )

    assert lines == [b"data: [DONE]"]
    assert passthrough is False
    assert handler.wfile.getvalue() == b""


def test_chat_tool_continue_repair_buffer_can_emit_immediate_visible_notice():
    class Handler:
        def __init__(self):
            self.wfile = io.BytesIO()

    class Response:
        def iter_lines(self):
            yield b"data: [DONE]"

        def close(self):
            pass

    handler = Handler()
    _buffer_openai_chat_sse_with_keepalive(
        handler,
        Response(),
        request_id="chat_req_test",
        keepalive_seconds=30,
        write_lock=threading.Lock(),
        visible_status={"request_id": "chat_req_test", "model": "local/model", "content": "visible repair"},
        visible_notice_after_seconds=0,
    )

    assert b"visible repair" in handler.wfile.getvalue()



def test_chat_tool_continue_repair_buffer_passthroughs_when_tool_call_seen():
    class Handler:
        def __init__(self):
            self.wfile = io.BytesIO()

    class Response:
        def iter_lines(self):
            yield b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1"}]}}]}'
            yield b"data: [DONE]"

        def close(self):
            pass

    handler = Handler()
    lines, passthrough, state = _buffer_openai_chat_sse_with_keepalive(
        handler,
        Response(),
        request_id="chat_req_test",
        keepalive_seconds=30,
        write_lock=threading.Lock(),
        passthrough_tool_calls=True,
    )

    assert lines == []
    assert passthrough is True
    assert state["passthrough_reason"] == "tool_call_seen"
    assert b"call_1" in handler.wfile.getvalue()


def test_chat_tool_continue_repair_buffer_passthroughs_long_visible_text():
    class Handler:
        def __init__(self):
            self.wfile = io.BytesIO()

    class Response:
        def iter_lines(self):
            yield b'data: {"choices":[{"delta":{"content":"hello world"}}]}'
            __import__("time").sleep(1.05)
            yield b"data: [DONE]"

        def close(self):
            pass

    handler = Handler()
    lines, passthrough, state = _buffer_openai_chat_sse_with_keepalive(
        handler,
        Response(),
        request_id="chat_req_test",
        keepalive_seconds=30,
        write_lock=threading.Lock(),
        passthrough_visible_chars=5,
    )

    assert lines == []
    assert passthrough is True
    assert state["passthrough_reason"] == "visible_content_threshold"
    assert b"hello world" in handler.wfile.getvalue()


def test_chat_tool_continue_repair_buffer_does_not_passthrough_visible_text_by_default():
    class Handler:
        def __init__(self):
            self.wfile = io.BytesIO()

    class Response:
        def iter_lines(self):
            yield b'data: {"choices":[{"delta":{"content":"[terminal command=\\\\\\"sg docker -c test\\\\\\"]"}}]}'
            yield b"data: [DONE]"

        def close(self):
            pass

    handler = Handler()
    lines, passthrough, state = _buffer_openai_chat_sse_with_keepalive(
        handler,
        Response(),
        request_id="chat_req_test",
        keepalive_seconds=30,
        write_lock=threading.Lock(),
    )

    assert passthrough is False
    assert len(lines) == 2
    assert state["visible_content_len"] > 0
    assert handler.wfile.getvalue() == b""





def test_chat_tool_continue_repair_notice_suppression_reports_visible_passthrough(monkeypatch):
    events = []
    monkeypatch.setattr(cli, "log_api_event", lambda kind, payload: events.append((kind, payload)))

    class Handler:
        def __init__(self):
            self.wfile = io.BytesIO()

    class Response:
        def iter_lines(self):
            yield b'data: {"choices":[{"delta":{"content":"hello world"}}]}'
            __import__("time").sleep(1.05)
            yield b"data: [DONE]"

        def close(self):
            pass

    _buffer_openai_chat_sse_with_keepalive(
        Handler(),
        Response(),
        request_id="chat_req_test",
        keepalive_seconds=30,
        write_lock=threading.Lock(),
        visible_status={
            "request_id": "chat_req_test",
            "model": "local/model",
            "upstream_model": "local/model",
            "round": 1,
            "trigger_reason": "visible_content_trailing_colon",
            "content": "retrying",
        },
        visible_notice_after_seconds=1,
        passthrough_visible_chars=5,
    )

    suppressed = [payload for kind, payload in events if kind == "openai_chat_tool_continue_repair_user_notice_suppressed"]
    assert suppressed
    assert suppressed[-1]["reason"] == "passthrough_started"
    assert suppressed[-1]["passthrough_reason"] == "visible_content_threshold"
    assert suppressed[-1]["tool_call_chunks"] == 0

def test_chat_tool_continue_repair_passthrough_tracks_tool_argument_metrics_to_finish():
    class Handler:
        def __init__(self):
            self.wfile = io.BytesIO()

    class Response:
        def iter_lines(self):
            chunks = [
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "write_file", "arguments": "{\"path\":"}}]}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"x.md\"}"}}]}}]},
                {"choices": [{"finish_reason": "tool_calls", "delta": {}}]},
            ]
            for chunk in chunks:
                yield ("data: " + json.dumps(chunk)).encode("utf-8")
            yield b"data: [DONE]"

        def close(self):
            pass

    handler = Handler()
    lines, passthrough, state = _buffer_openai_chat_sse_with_keepalive(
        handler,
        Response(),
        request_id="chat_req_test",
        keepalive_seconds=30,
        write_lock=threading.Lock(),
        passthrough_tool_calls=True,
    )

    assert lines == []
    assert passthrough is True
    assert state["passthrough_reason"] == "tool_call_seen"
    assert state["finish_reason"] == "tool_calls"
    assert state["tool_names"] == ["write_file"]
    assert state["tool_argument_chars"] > 0



def test_chat_tool_continue_repair_buffers_tool_calls_by_default_to_catch_length():
    class Handler:
        def __init__(self):
            self.wfile = io.BytesIO()

    class Response:
        def iter_lines(self):
            chunks = [
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "terminal", "arguments": "{\"cmd\":"}}]}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"ls"}}]}}]},
                {"choices": [{"finish_reason": "length", "delta": {}}]},
            ]
            for chunk in chunks:
                yield ("data: " + json.dumps(chunk)).encode("utf-8")
            yield b"data: [DONE]"

        def close(self):
            pass

    handler = Handler()
    lines, passthrough, state = _buffer_openai_chat_sse_with_keepalive(
        handler,
        Response(),
        request_id="chat_req_test",
        keepalive_seconds=30,
        write_lock=threading.Lock(),
    )

    assert passthrough is False
    assert len(lines) == 4
    assert handler.wfile.getvalue() == b""
    assert state["finish_reason"] == "length"
    assert state["tool_call_chunks"] > 0
    assert state["tool_argument_json_valid_by_index"]["0"] is False


def test_chat_tool_continue_repair_loop_guard_aborts_reasoning_without_tool_calls():
    class Handler:
        def __init__(self):
            self.wfile = io.BytesIO()

    class Response:
        def __init__(self):
            self.closed = False

        def iter_lines(self):
            chunks = [
                {"choices": [{"delta": {"reasoning_content": "thinking "}}]},
                {"choices": [{"delta": {"reasoning_content": "still thinking "}}]},
                {"choices": [{"delta": {"reasoning_content": "more hidden text"}}]},
                {"choices": [{"finish_reason": "length", "delta": {}}]},
            ]
            for chunk in chunks:
                yield ("data: " + json.dumps(chunk)).encode("utf-8")
            yield b"data: [DONE]"

        def close(self):
            self.closed = True

    response = Response()
    handler = Handler()
    lines, passthrough, state = _buffer_openai_chat_sse_with_keepalive(
        handler,
        response,
        request_id="chat_req_test",
        keepalive_seconds=30,
        write_lock=threading.Lock(),
        loop_guard={
            "enabled": True,
            "no_tool_call_max_chars": 10,
            "repeated_tail_min_chars": 0,
            "repeated_tail_repetitions": 4,
        },
    )

    assert passthrough is False
    assert state["buffer_abort_reason"] == "no_tool_call_generation_limit"
    assert state["reasoning_len"] >= 10
    assert response.closed is True
    assert handler.wfile.getvalue() == b""
    assert lines


def test_chat_tool_continue_repair_loop_guard_allows_large_tool_arguments():
    class Handler:
        def __init__(self):
            self.wfile = io.BytesIO()

    class Response:
        def iter_lines(self):
            long_args = "x" * 200
            chunks = [
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "write_file", "arguments": long_args}}]}}]},
                {"choices": [{"finish_reason": "tool_calls", "delta": {}}]},
            ]
            for chunk in chunks:
                yield ("data: " + json.dumps(chunk)).encode("utf-8")
            yield b"data: [DONE]"

        def close(self):
            pass

    handler = Handler()
    lines, passthrough, state = _buffer_openai_chat_sse_with_keepalive(
        handler,
        Response(),
        request_id="chat_req_test",
        keepalive_seconds=30,
        write_lock=threading.Lock(),
        loop_guard={
            "enabled": True,
            "no_tool_call_max_chars": 10,
            "repeated_tail_min_chars": 0,
            "repeated_tail_repetitions": 4,
        },
    )

    assert passthrough is False
    assert state["buffer_abort_reason"] == ""
    assert state["tool_call_chunks"] > 0
    assert state["tool_argument_chars"] == 200
    assert len(lines) == 3


def test_chat_tool_continue_repair_forces_required_tool_choice_when_auto():
    payload = {"model": "local/model", "tool_choice": "auto", "tools": _sample_chat_tools()}

    repaired = _force_tool_choice_for_chat_repair(payload)

    assert repaired["tool_choice"] == "required"
    assert payload["tool_choice"] == "auto"


def test_chat_tool_continue_repair_preserves_explicit_tool_choice():
    explicit = {"type": "function", "function": {"name": "exec_command"}}
    payload = {"model": "local/model", "tool_choice": explicit, "tools": _sample_chat_tools()}

    repaired = _force_tool_choice_for_chat_repair(payload)

    assert repaired["tool_choice"] == explicit

def test_chat_tool_continue_repair_token_cap_limits_internal_rounds():
    payload = {"model": "local/model", "stream": True, "max_tokens": 65536, "n_predict": 65536}

    capped = _apply_chat_tool_continue_repair_token_cap(payload, 2048)

    assert capped["max_tokens"] == 2048
    assert capped["n_predict"] == 2048
    assert payload["max_tokens"] == 65536



def test_responses_payload_does_not_convert_builtin_tools_for_legacy_chat_fallback():
    payload = {
        "model": "openai/test-model",
        "instructions": "Be concise.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        ],
        "tools": [{"type": "computer_use_preview"}],
        "parallel_tool_calls": True,
        "max_output_tokens": 32,
        "temperature": 0.2,
    }

    chat_payload = _responses_payload_to_chat_payload(payload, "test-model")

    assert chat_payload["model"] == "test-model"
    assert chat_payload["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "hello"},
    ]
    assert chat_payload["stream"] is False
    assert chat_payload["max_tokens"] == 32
    assert chat_payload["temperature"] == 0.2
    assert "tools" not in chat_payload
    assert "parallel_tool_calls" not in chat_payload


def test_responses_internal_round_max_tokens_has_safe_default(monkeypatch):
    monkeypatch.delenv("HEIMDALL_GATEWAY_RESPONSES_INTERNAL_MAX_TOKENS", raising=False)
    assert _responses_internal_round_max_tokens() == 4096

    monkeypatch.setenv("HEIMDALL_GATEWAY_RESPONSES_INTERNAL_MAX_TOKENS", "64")
    assert _responses_internal_round_max_tokens() == 256

    monkeypatch.setenv("HEIMDALL_GATEWAY_RESPONSES_INTERNAL_MAX_TOKENS", "8192")
    assert _responses_internal_round_max_tokens() == 8192


def test_responses_top_level_function_tools_are_wrapped_for_chat_completions():
    tools = [
        {
            "type": "function",
            "name": "exec_command",
            "description": "Runs a command",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        }
    ]

    chat_tools = _responses_tools_to_chat_tools(tools)

    assert chat_tools == [
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "description": "Runs a command",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
            },
        }
    ]



def test_responses_tools_with_deferred_search_marks_namespaces_and_functions():
    tools = [
        {
            "type": "namespace",
            "name": "browser",
            "description": "Browser tools",
            "tools": [
                {"type": "function", "name": "click", "parameters": {}},
                {"type": "function", "name": "snapshot", "defer_loading": False, "parameters": {}},
            ],
        },
        {"type": "function", "name": "exec_command", "parameters": {}},
        {"type": "mcp", "server_label": "github"},
    ]

    prepared = _responses_tools_with_deferred_search(tools)

    namespace = prepared[0]
    assert namespace["type"] == "namespace"
    assert all(tool["defer_loading"] is True for tool in namespace["tools"])
    assert prepared[1]["defer_loading"] is True
    assert prepared[2]["defer_loading"] is True
    assert prepared[-1] == {"type": "tool_search"}


def test_responses_tools_with_deferred_search_does_not_duplicate_tool_search():
    prepared = _responses_tools_with_deferred_search([
        {"type": "function", "name": "exec_command", "parameters": {}},
        {"type": "tool_search", "execution": "client"},
    ])

    assert [tool["type"] for tool in prepared].count("tool_search") == 1
    assert prepared[0]["defer_loading"] is True
    assert prepared[1] == {"type": "tool_search", "execution": "client"}

def test_responses_namespace_tools_are_flattened_for_legacy_chat_fallback():
    tools = [
        {
            "type": "namespace",
            "name": "mcp__codex_apps__github",
            "tools": [{"name": "_search", "description": "Search", "parameters": {}}],
        },
        {
            "type": "namespace",
            "name": "mcp__chrome_devtools",
            "tools": [{"name": "new_page", "description": "Open", "parameters": {}}],
        },
        {
            "type": "namespace",
            "name": "mcp__chrome_devtools_1",
            "tools": [{"name": "new_page", "description": "Open duplicate", "parameters": {}}],
        },
        {"type": "function", "name": "exec_command", "parameters": {}},
    ]

    chat_tools = _responses_tools_to_chat_tools(tools)

    assert chat_tools == [
        {
            "type": "function",
            "function": {"name": "mcp__codex_apps__github___search", "description": "Search", "parameters": {}},
        },
        {
            "type": "function",
            "function": {"name": "mcp__chrome_devtools__new_page", "description": "Open", "parameters": {}},
        },
        {
            "type": "function",
            "function": {"name": "mcp__chrome_devtools_1__new_page", "description": "Open duplicate", "parameters": {}},
        },
        {
            "type": "function",
            "function": {"name": "exec_command", "description": "", "parameters": {}},
        },
    ]
    assert _responses_namespace_tool_map(tools) == {
        "mcp__codex_apps__github___search": {"namespace": "mcp__codex_apps__github", "name": "_search"},
        "mcp__chrome_devtools__new_page": {"namespace": "mcp__chrome_devtools", "name": "new_page"},
        "mcp__chrome_devtools_1__new_page": {"namespace": "mcp__chrome_devtools_1", "name": "new_page"},
    }


def test_responses_tool_choice_function_is_wrapped_for_chat_completions():
    assert _responses_tool_choice_to_chat_tool_choice({"type": "function", "name": "lookup"}) == {
        "type": "function",
        "function": {"name": "lookup"},
    }

    payload = _responses_payload_to_chat_payload(
        {
            "model": "m",
            "input": "call lookup",
            "tools": [{"type": "function", "name": "lookup", "parameters": {}}],
            "tool_choice": {"type": "function", "name": "lookup"},
        },
        "m",
    )

    assert payload["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}


def test_responses_function_call_history_maps_to_chat_tool_messages():
    chat_payload = _responses_payload_to_chat_payload(
        {
            "model": "m",
            "tools": [{"type": "function", "name": "exec_command", "parameters": {}}],
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "run ls"}]},
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": '{"cmd":"ls"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "file.txt",
                },
            ],
        },
        "m",
    )

    assert chat_payload["messages"] == [
        {"role": "user", "content": "run ls"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "exec_command", "arguments": '{"cmd":"ls"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "file.txt"},
    ]


def test_responses_history_skips_empty_assistant_placeholders():
    messages = _responses_input_to_openai_messages(
        {
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "message", "role": "assistant", "content": []},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
                {"type": "message", "role": "assistant", "content": []},
            ]
        }
    )

    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "answer"},
    ]


def test_responses_history_merges_adjacent_assistant_messages_for_llamacpp():
    messages = _responses_input_to_openai_messages(
        {
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "part one"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "part two"}]},
            ]
        }
    )

    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "part one\n\npart two"},
    ]


def test_responses_history_attaches_function_call_to_previous_assistant_content():
    messages = _responses_input_to_openai_messages(
        {
            "tools": [{"type": "function", "name": "exec_command", "parameters": {}}],
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "run"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "I will run it."}]},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": "{\"cmd\":\"ls\"}",
                },
            ],
        },
        allowed_tool_names={"exec_command"},
    )

    assert messages == [
        {"role": "user", "content": "run"},
        {
            "role": "assistant",
            "content": "I will run it.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "exec_command", "arguments": "{\"cmd\":\"ls\"}"},
                }
            ],
        },
    ]


def test_responses_namespace_function_call_history_maps_to_flat_allowlisted_legacy_tool():
    chat_payload = _responses_payload_to_chat_payload(
        {
            "model": "m",
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__engram",
                    "tools": [{"name": "mem_context", "parameters": {}}],
                }
            ],
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "continue"}]},
                {
                    "type": "function_call",
                    "call_id": "call_mem",
                    "namespace": "mcp__engram",
                    "name": "mem_context",
                    "arguments": '{"limit":1}',
                },
                {"type": "function_call_output", "call_id": "call_mem", "output": "ok"},
            ],
        },
        "m",
    )

    assert chat_payload["messages"][1]["tool_calls"][0]["function"] == {
        "name": "mcp__engram__mem_context",
        "arguments": '{"limit":1}',
    }
    assert chat_payload["messages"][2] == {"role": "tool", "tool_call_id": "call_mem", "content": "ok"}


def test_chat_response_to_responses_payload_preserves_tool_namespace():
    payload = _chat_response_to_responses_payload(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "responses_item_id": "fc_1",
                                "type": "function",
                                "namespace": "mcp__engram",
                                "function": {"name": "mem_context", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        },
        "m",
        {"tools": []},
    )

    assert payload["output"] == [
        {
            "id": "fc_1",
            "type": "function_call",
            "status": "completed",
            "call_id": "call_1",
            "name": "mem_context",
            "arguments": "{}",
            "namespace": "mcp__engram",
        }
    ]


def test_chat_response_to_responses_payload_strips_tool_routing_arguments():
    payload = _chat_response_to_responses_payload(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "responses_item_id": "fc_1",
                                "type": "function",
                                "namespace": "mcp__chrome_devtools",
                                "function": {
                                    "name": "take_snapshot",
                                    "arguments": '{"namespace":"mcp__chrome_devtools","server":"mcp__chrome_devtools","verbose":false}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
        "m",
        {"tools": []},
    )

    assert payload["output"][0]["namespace"] == "mcp__chrome_devtools"
    assert json.loads(payload["output"][0]["arguments"]) == {"verbose": False}


def test_responses_function_call_history_filters_unforwarded_mcp_tools():
    chat_payload = _responses_payload_to_chat_payload(
        {
            "model": "m",
            "tools": [{"type": "function", "name": "exec_command", "parameters": {}}],
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "continue"}]},
                {
                    "type": "function_call",
                    "call_id": "call_bad",
                    "name": "mcp__engram__mem_context",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_bad",
                    "output": "unsupported call: mcp__engram__mem_context",
                },
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "now answer"}]},
            ],
        },
        "m",
    )

    assert chat_payload["messages"] == [
        {"role": "user", "content": "continue"},
        {"role": "user", "content": "now answer"},
    ]


def test_responses_tool_role_history_is_not_degraded_to_user_message():
    chat_payload = _responses_payload_to_chat_payload(
        {
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "continue"}]},
                {"type": "message", "role": "tool", "content": "unsupported call: mcp__serena__get_symbols_overview"},
            ],
        },
        "m",
    )

    assert chat_payload["messages"] == [{"role": "user", "content": "continue"}]


def test_responses_payload_accepts_string_input():
    chat_payload = _responses_payload_to_chat_payload(
        {"model": "m", "input": "Say hi", "stream": True},
        "m",
    )

    assert chat_payload["messages"] == [{"role": "user", "content": "Say hi"}]
    assert chat_payload["stream"] is True


def test_responses_stream_requests_usage_from_legacy_chat_backend():
    chat_payload = _responses_payload_to_chat_payload(
        {"model": "m", "input": "Say hi", "stream": True},
        "m",
    )

    assert chat_payload["stream_options"] == {"include_usage": True}


def test_responses_stream_preserves_existing_stream_options_when_requesting_usage():
    chat_payload = _responses_payload_to_chat_payload(
        {"model": "m", "input": "Say hi", "stream": True, "stream_options": {"foo": "bar"}},
        "m",
    )

    assert chat_payload["stream_options"] == {"foo": "bar", "include_usage": True}



def test_responses_payload_summary_flags_images_without_logging_data():
    image_data = "a" * 2000
    payload = {
        "model": "m",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is this?"},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image_data}"},
                ],
            }
        ],
    }

    summary = _summarize_api_payload_for_log(payload)

    assert summary["has_images"] is True
    assert summary["input_type"] == "list"
    assert summary["input_image_count"] == 1
    assert summary["total_image_count"] == 1
    assert image_data not in json.dumps(summary)


def test_sanitize_responses_tool_arguments_strips_namespace_and_server():
    arguments, repaired = _sanitize_responses_tool_arguments(
        "take_snapshot",
        "mcp__chrome_devtools",
        '{"namespace":"mcp__chrome_devtools","server":"mcp__chrome_devtools","verbose":false}',
    )

    assert repaired is True
    assert json.loads(arguments) == {"verbose": False}


def test_sanitize_responses_tool_arguments_normalizes_chrome_evaluate_script_body():
    arguments, repaired = _sanitize_responses_tool_arguments(
        "evaluate_script",
        "mcp__chrome_devtools",
        '{"function":"var x = 1; return x;"}',
    )

    assert repaired is True
    assert json.loads(arguments)["function"] == "() => { var x = 1; return x; }"


def test_sanitize_responses_tool_arguments_normalizes_chrome_evaluate_script_iife():
    arguments, repaired = _sanitize_responses_tool_arguments(
        "evaluate_script",
        "mcp__chrome_devtools",
        '{"function":"(() => document.title)()"}',
    )

    assert repaired is True
    assert json.loads(arguments)["function"] == "() => ((() => document.title)())"


def test_responses_payload_to_chat_payload_preserves_image_url_part():
    payload = {
        "model": "m",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is this?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,abc"},
                ],
            }
        ],
    }

    chat_payload = _responses_payload_to_chat_payload(payload, "m")

    assert chat_payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }
    ]


def test_responses_legacy_tool_output_extracts_structured_screenshot_as_image_part():
    image_data = "a" * 4096
    chat_payload = _responses_payload_to_chat_payload(
        {
            "model": "m",
            "tools": [{"type": "function", "name": "take_screenshot", "parameters": {}}],
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "inspect page"}]},
                {
                    "type": "function_call",
                    "call_id": "call_screen",
                    "name": "take_screenshot",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_screen",
                    "output": [
                        {"type": "input_text", "text": "Took a screenshot of the current page's viewport."},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{image_data}"},
                    ],
                },
            ],
        },
        "m",
    )

    tool_message = chat_payload["messages"][2]
    assert tool_message["role"] == "tool"
    assert "Took a screenshot" in tool_message["content"]
    assert "data:image" not in tool_message["content"]
    assert image_data not in tool_message["content"]

    image_message = chat_payload["messages"][3]
    assert image_message["role"] == "user"
    assert image_message["content"][0]["type"] == "text"
    assert image_message["content"][1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_data}"},
    }


def test_responses_legacy_tool_output_extracts_inline_base64_data_url_as_image_part():
    image_data = "b" * 4096
    chat_payload = _responses_payload_to_chat_payload(
        {
            "model": "m",
            "tools": [{"type": "function", "name": "capture", "parameters": {}}],
            "input": [
                {"type": "function_call", "call_id": "call_capture", "name": "capture", "arguments": "{}"},
                {
                    "type": "function_call_output",
                    "call_id": "call_capture",
                    "output": f"before data:image/png;base64,{image_data} after",
                },
            ],
        },
        "m",
    )

    tool_content = chat_payload["messages"][1]["content"]
    assert tool_content.startswith("before ")
    assert tool_content.endswith(" after")
    assert "[image attached" in tool_content
    assert "data:image" not in tool_content
    assert image_data not in tool_content

    image_message = chat_payload["messages"][2]
    assert image_message["role"] == "user"
    assert image_message["content"][1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_data}"},
    }

def test_raw_responses_passthrough_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("HEIMDALL_GATEWAY_RESPONSES_RAW_PASSTHROUGH", raising=False)

    assert _responses_raw_passthrough_enabled() is False

    monkeypatch.setenv("HEIMDALL_GATEWAY_RESPONSES_RAW_PASSTHROUGH", "1")

    assert _responses_raw_passthrough_enabled() is True


def test_responses_handler_does_not_preflatten_responses_tools_before_adapter():
    source = inspect.getsource(cli.start_ctx_metadata_server)
    read_body_source = source[source.index("def _read_json_body"):source.index("def _proxy_raw_response")]
    proxy_source = source[source.index("def _proxy_request"):source.index("def _read_json_body")]

    assert '"/v1/responses"' not in read_body_source
    assert '"/responses"' not in read_body_source
    assert '"/v1/responses"' not in proxy_source
    assert '"/responses"' not in proxy_source


def test_responses_stream_blocks_tool_calls_not_in_forwarded_legacy_toolset():
    source = inspect.getsource(cli.start_ctx_metadata_server)
    responses_source = source[source.index("def _handle_openai_responses"):]

    assert "allowed_legacy_tool_names" in responses_source
    assert "openai_responses_tool_call_blocked_not_forwarded" in responses_source
    assert "openai_responses_tool_call_omitted_from_completed" in responses_source


def test_responses_stream_completed_payload_includes_upstream_usage_when_available():
    source = inspect.getsource(cli.start_ctx_metadata_server)
    responses_source = source[source.index("def _handle_openai_responses"):]

    assert "latest_usage = None" in responses_source
    assert "openai_responses_stream_usage_received" in responses_source
    assert 'final_chat_payload["usage"] = latest_usage' in responses_source


def test_responses_handler_runtime_snapshots_use_defined_client_host():
    source = inspect.getsource(cli.start_ctx_metadata_server)
    responses_source = source[source.index("def _handle_openai_responses"):]

    assert "\n                host,\n" not in responses_source
    assert "openai_responses_runtime_before_upstream" in responses_source
    assert "client_host,\n                int(args.public_port)" in responses_source


def test_chat_response_is_wrapped_as_responses_payload():
    request_payload = {
        "tools": [{"type": "computer_use_preview"}],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    payload = _chat_response_to_responses_payload(
        {
            "choices": [{"message": {"role": "assistant", "content": "hola"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
        "model-a",
        request_payload,
    )

    assert payload["object"] == "response"
    assert payload["status"] == "completed"
    assert payload["model"] == "model-a"
    assert payload["output_text"] == "hola"
    assert payload["output"][0]["content"][0] == {
        "type": "output_text",
        "text": "hola",
        "annotations": [],
    }
    assert payload["usage"] == {
        "input_tokens": 2,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 1,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 3,
    }
    assert payload["tools"] == [{"type": "computer_use_preview"}]
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is True



def test_responses_usage_shape_is_preserved_when_already_responses_style():
    payload = _chat_response_to_responses_payload(
        {
            "choices": [{"message": {"role": "assistant", "content": "hola"}}],
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 1},
                "total_tokens": 13,
            },
        },
        "model-a",
        {},
    )

    assert payload["usage"]["input_tokens"] == 10
    assert payload["usage"]["output_tokens"] == 3
    assert payload["usage"]["total_tokens"] == 13


def _parse_sse_events(raw: str):
    events = []
    current_event = None
    current_data = []
    for line in raw.splitlines():
        if not line:
            if current_event or current_data:
                data_text = "\n".join(current_data)
                events.append((current_event, data_text))
                current_event = None
                current_data = []
            continue
        if line.startswith("event: "):
            current_event = line.removeprefix("event: ")
        elif line.startswith("data: "):
            current_data.append(line.removeprefix("data: "))
    return events


def test_responses_sse_completed_event_wraps_response_object():
    class DummyHandler:
        def __init__(self):
            self.wfile = io.BytesIO()
            self.status = None
            self.headers = []

        def send_response(self, status):
            self.status = status

        def send_header(self, key, value):
            self.headers.append((key, value))

        def end_headers(self):
            pass

    handler = DummyHandler()
    payload = {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "m",
        "output": [],
        "output_text": "hola",
    }

    _write_responses_sse(handler, payload)

    events = _parse_sse_events(handler.wfile.getvalue().decode("utf-8"))
    created = json.loads(next(data for event, data in events if event == "response.created"))
    completed = json.loads(next(data for event, data in events if event == "response.completed"))
    assert created["type"] == "response.created"
    assert created["response"]["id"] == "resp_test"
    assert created["response"]["status"] == "in_progress"
    assert completed["type"] == "response.completed"
    assert completed["response"]["id"] == "resp_test"
    assert completed["response"]["status"] == "completed"
    assert completed["sequence_number"] > created["sequence_number"]
    assert events[-1] == (None, "[DONE]")


def test_chat_response_tool_calls_become_responses_function_call_items():
    payload = _chat_response_to_responses_payload(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"q":"hola"}'},
                            }
                        ],
                    }
                }
            ]
        },
        "model-a",
        {},
    )

    item = payload["output"][0]
    assert item["type"] == "function_call"
    assert item["id"].startswith("fc_")
    assert item["call_id"] == "call_123"
    assert item["name"] == "lookup"
    assert item["arguments"] == '{"q":"hola"}'


def test_responses_sse_function_call_stream_events_are_openai_compatible():
    class DummyHandler:
        def __init__(self):
            self.wfile = io.BytesIO()
            self.status = None
            self.headers = []

        def send_response(self, status):
            self.status = status

        def send_header(self, key, value):
            self.headers.append((key, value))

        def end_headers(self):
            pass

    handler = DummyHandler()
    payload = {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "m",
        "output": [
            {
                "id": "fc_test",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_test",
                "name": "lookup",
                "arguments": '{"q":"hola"}',
            }
        ],
        "output_text": "",
    }

    _write_responses_sse(handler, payload)

    parsed = [
        (event, json.loads(data))
        for event, data in _parse_sse_events(handler.wfile.getvalue().decode("utf-8"))
        if data != "[DONE]"
    ]
    event_names = [event for event, _ in parsed]
    assert event_names == [
        "response.created",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    added = parsed[1][1]
    assert added["item"] == {
        "id": "fc_test",
        "type": "function_call",
        "status": "in_progress",
        "call_id": "call_test",
        "name": "lookup",
        "arguments": "",
    }
    assert parsed[2][1]["delta"] == '{"q":"hola"}'
    assert parsed[3][1]["arguments"] == '{"q":"hola"}'
    assert parsed[4][1]["item"]["status"] == "completed"
    assert parsed[5][1]["response"]["output"][0]["call_id"] == "call_test"

def test_chat_response_keeps_reasoning_content_separate_from_output_text():
    payload = _chat_response_to_responses_payload(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "intermediate reasoning text",
                    }
                }
            ]
        },
        "model-a",
        {},
    )

    assert payload["output_text"] == ""
    assert payload["output"][0]["type"] == "reasoning"
    assert payload["output"][0]["content"][0] == {
        "type": "reasoning_text",
        "text": "intermediate reasoning text",
    }


def test_responses_sse_reasoning_events_are_not_output_text_events():
    class DummyHandler:
        def __init__(self):
            self.wfile = io.BytesIO()
            self.status = None
            self.headers = []

        def send_response(self, status):
            self.status = status

        def send_header(self, key, value):
            self.headers.append((key, value))

        def end_headers(self):
            pass

    handler = DummyHandler()
    payload = {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "m",
        "output": [
            {
                "id": "rs_test",
                "type": "reasoning",
                "status": "completed",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "thinking..."}],
            },
            {
                "id": "msg_test",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "visible", "annotations": []}],
            },
        ],
        "output_text": "visible",
    }

    _write_responses_sse(handler, payload)

    parsed = [
        (event, json.loads(data))
        for event, data in _parse_sse_events(handler.wfile.getvalue().decode("utf-8"))
        if data != "[DONE]"
    ]
    event_names = [event for event, _ in parsed]
    assert "response.reasoning_text.delta" in event_names
    assert "response.reasoning_text.done" in event_names
    reasoning_delta = next(data for event, data in parsed if event == "response.reasoning_text.delta")
    assert reasoning_delta["delta"] == "thinking..."
    text_delta = next(data for event, data in parsed if event == "response.output_text.delta")
    assert text_delta["delta"] == "visible"
    assert text_delta["delta"] != reasoning_delta["delta"]


def _parse_sse_events(raw: str):
    events = []
    current_event = None
    current_data = []
    for line in raw.splitlines():
        if not line:
            if current_event or current_data:
                data_text = "\n".join(current_data)
                events.append((current_event, data_text))
                current_event = None
                current_data = []
            continue
        if line.startswith("event: "):
            current_event = line.removeprefix("event: ")
        elif line.startswith("data: "):
            current_data.append(line.removeprefix("data: "))
    return events


def test_responses_sse_completed_event_wraps_response_object():
    class DummyHandler:
        def __init__(self):
            self.wfile = io.BytesIO()
            self.status = None
            self.headers = []

        def send_response(self, status):
            self.status = status

        def send_header(self, key, value):
            self.headers.append((key, value))

        def end_headers(self):
            pass

    handler = DummyHandler()
    payload = {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "m",
        "output": [],
        "output_text": "hola",
    }

    _write_responses_sse(handler, payload)

    events = _parse_sse_events(handler.wfile.getvalue().decode("utf-8"))
    created = json.loads(next(data for event, data in events if event == "response.created"))
    completed = json.loads(next(data for event, data in events if event == "response.completed"))
    assert created["type"] == "response.created"
    assert created["response"]["id"] == "resp_test"
    assert created["response"]["status"] == "in_progress"
    assert completed["type"] == "response.completed"
    assert completed["response"]["id"] == "resp_test"
    assert completed["response"]["status"] == "completed"
    assert completed["sequence_number"] > created["sequence_number"]
    assert events[-1] == (None, "[DONE]")


def test_responses_sse_function_call_stream_events_are_openai_compatible():
    class DummyHandler:
        def __init__(self):
            self.wfile = io.BytesIO()
            self.status = None
            self.headers = []

        def send_response(self, status):
            self.status = status

        def send_header(self, key, value):
            self.headers.append((key, value))

        def end_headers(self):
            pass

    handler = DummyHandler()
    payload = {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "m",
        "output": [
            {
                "id": "fc_test",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_test",
                "name": "lookup",
                "arguments": '{"q":"hola"}',
            }
        ],
        "output_text": "",
    }

    _write_responses_sse(handler, payload)

    parsed = [
        (event, json.loads(data))
        for event, data in _parse_sse_events(handler.wfile.getvalue().decode("utf-8"))
        if data != "[DONE]"
    ]
    event_names = [event for event, _ in parsed]
    assert event_names == [
        "response.created",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    added = parsed[1][1]
    assert added["item"] == {
        "id": "fc_test",
        "type": "function_call",
        "status": "in_progress",
        "call_id": "call_test",
        "name": "lookup",
        "arguments": "",
    }
    assert parsed[2][1]["delta"] == '{"q":"hola"}'
    assert parsed[3][1]["arguments"] == '{"q":"hola"}'
    assert parsed[4][1]["item"]["status"] == "completed"
    assert parsed[5][1]["response"]["output"][0]["call_id"] == "call_test"


def _sample_deferred_tools():
    return [
        {
            "type": "namespace",
            "name": "mcp__chrome_devtools",
            "description": "Chrome debugging tools",
            "defer_loading": True,
            "tools": [
                {"name": "list_pages", "description": "List pages", "parameters": {"type": "object", "properties": {}}},
                {"name": "get_console_messages", "description": "Read console errors", "parameters": {"type": "object", "properties": {"level": {"type": "string"}}}},
                {"name": "click", "description": "Click a selector", "defer_loading": False, "parameters": {"type": "object", "properties": {"selector": {"type": "string"}}}},
            ],
        },
        {"type": "function", "name": "exec_command", "description": "Run shell command", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}},
    ]


def test_responses_tool_registry_splits_eager_and_deferred_kv_stable():
    registry = ResponsesToolRegistry.from_responses_tools(_sample_deferred_tools())

    eager_names = [tool["function"]["name"] for tool in registry.eager_chat_tools]
    assert "exec_command" in eager_names
    assert "mcp__chrome_devtools__click" in eager_names
    assert "mcp__chrome_devtools__get_console_messages" not in eager_names
    assert "mcp__chrome_devtools__get_console_messages" in registry.deferred_by_legacy_name

    stable_tools = registry.chat_tools_with_internal_search()
    stable_names = [tool["function"]["name"] for tool in stable_tools]
    assert stable_names == eager_names + ["tool_search", "call_deferred_tool"]


def test_responses_tool_registry_search_returns_schema_without_mutating_tools():
    registry = ResponsesToolRegistry.from_responses_tools(_sample_deferred_tools())
    before = json.dumps(registry.chat_tools_with_internal_search(), sort_keys=True)

    result = registry.search({"query": "console errors", "namespaces": ["mcp__chrome_devtools"]})

    after = json.dumps(registry.chat_tools_with_internal_search(), sort_keys=True)
    assert before == after
    assert result["status"] == "ok"
    names = [item["name"] for item in result["tools"]]
    assert "get_console_messages" in names
    assert result["tools"][0]["parameters"]


def test_responses_tool_registry_broad_search_requests_narrowing():
    tools = [
        {
            "type": "namespace",
            "name": "ns",
            "defer_loading": True,
            "tools": [
                {"name": f"tool_{idx}", "description": "same broad description", "parameters": {}}
                for idx in range(30)
            ],
        }
    ]
    registry = ResponsesToolRegistry.from_responses_tools(tools)

    result = registry.search({"query": "same"}, max_results=24)

    assert result["status"] == "too_many_matches"
    assert len(result["matches"]) == 24
    assert "narrow" in result["message"].lower()


def test_responses_tool_registry_translates_call_deferred_tool_to_real_call():
    registry = ResponsesToolRegistry.from_responses_tools(_sample_deferred_tools())
    translated = registry.translate_deferred_tool_call({
        "namespace": "mcp__chrome_devtools",
        "name": "get_console_messages",
        "arguments": {"level": "error"},
    })

    assert translated == {
        "legacy_name": "mcp__chrome_devtools__get_console_messages",
        "responses_name": "get_console_messages",
        "namespace": "mcp__chrome_devtools",
        "arguments": '{"level":"error"}',
    }


def test_kv_stable_payload_keeps_tool_set_same_after_tool_search_output():
    payload = {
        "model": "m",
        "input": "inspect console",
        "tools": _sample_deferred_tools(),
    }
    registry = ResponsesToolRegistry.from_responses_tools(payload["tools"])
    first = _responses_payload_to_chat_payload(payload, "m", tool_registry=registry)
    second = _responses_payload_to_chat_payload(
        payload,
        "m",
        tool_registry=registry,
        extra_messages=[registry.tool_search_output_message("call_search", {"query": "console"})],
    )

    assert first["tools"] == second["tools"]
    assert len(second["messages"]) == len(first["messages"]) + 1
    assert second["messages"][-1]["role"] == "tool"
    assert second["messages"][-1]["tool_call_id"] == "call_search"
    assert "get_console_messages" in second["messages"][-1]["content"]


def test_translate_internal_call_deferred_tool_chat_response_hides_wrapper_name():
    registry = ResponsesToolRegistry.from_responses_tools(_sample_deferred_tools())
    data = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "call_deferred_tool",
                                "arguments": json.dumps({
                                    "namespace": "mcp__chrome_devtools",
                                    "name": "get_console_messages",
                                    "arguments": {"level": "error"},
                                }),
                            },
                        }
                    ]
                }
            }
        ]
    }

    translated, changed = _translate_internal_deferred_tool_calls_in_chat_response(data, registry)

    assert changed is True
    tool_call = translated["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_console_messages"
    assert tool_call["function"]["namespace"] == "mcp__chrome_devtools"
    assert tool_call["function"]["arguments"] == '{"level":"error"}'





def test_unresolved_deferred_tool_generates_internal_feedback_message():
    registry = ResponsesToolRegistry.from_responses_tools(_sample_deferred_tools())
    data = {
        "choices": [{"message": {"tool_calls": [{
            "id": "call_missing",
            "type": "function",
            "function": {
                "name": "call_deferred_tool",
                "arguments": json.dumps({"name": "missing_tool", "arguments": {"x": 1}}),
            },
        }]}}]
    }

    followups = _chat_response_internal_tool_repair_followup_messages(data, registry)

    assert len(followups) == 2
    assert followups[0]["role"] == "assistant"
    assert followups[0]["tool_calls"][0]["function"]["name"] == "call_deferred_tool"
    assert followups[1]["role"] == "tool"
    assert followups[1]["tool_call_id"] == "call_missing"
    assert "The previous call_deferred_tool call was not executed." in followups[1]["content"]
    assert "Retry by calling call_deferred_tool again" in followups[1]["content"]
    feedback = _feedback_json_from_tool_content(followups[1]["content"])
    assert feedback["error"] == "deferred_tool_not_found"
    assert feedback["received"]["name"] == "missing_tool"
    assert "tool_search" in feedback["message"]


def test_ambiguous_deferred_tool_without_namespace_generates_candidates_feedback():
    tools = [
        {"type": "namespace", "name": "mcp__chrome_devtools", "tools": [{"name": "evaluate_script", "parameters": {}}]},
        {"type": "namespace", "name": "mcp__chrome_devtools_1", "tools": [{"name": "evaluate_script", "parameters": {}}]},
    ]
    registry = ResponsesToolRegistry.from_responses_tools(tools)
    data = {
        "choices": [{"message": {"tool_calls": [{
            "id": "call_ambiguous",
            "type": "function",
            "function": {
                "name": "call_deferred_tool",
                "arguments": json.dumps({"name": "evaluate_script", "arguments": {"function": "() => 1"}}),
            },
        }]}}]
    }

    followups = _chat_response_internal_tool_repair_followup_messages(data, registry)

    feedback = _feedback_json_from_tool_content(followups[1]["content"])
    assert feedback["error"] == "ambiguous_deferred_tool"
    assert {item["namespace"] for item in feedback["candidates"]} == {"mcp__chrome_devtools", "mcp__chrome_devtools_1"}


def test_missing_required_deferred_tool_argument_generates_schema_feedback():
    tools = [
        {
            "type": "namespace",
            "name": "mcp__chrome_devtools_1",
            "tools": [{
                "name": "wait_for",
                "description": "Wait for text",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "array", "items": {"type": "string"}}, "timeout": {"type": "integer"}},
                    "required": ["text"],
                },
            }],
        }
    ]
    registry = ResponsesToolRegistry.from_responses_tools(tools)
    data = {
        "choices": [{"message": {"tool_calls": [{
            "id": "call_wait",
            "type": "function",
            "function": {
                "name": "call_deferred_tool",
                "arguments": json.dumps({"name": "wait_for", "arguments": {"server": "mcp__chrome_devtools_1", "timeout": 5000}}),
            },
        }]}}]
    }

    followups = _chat_response_internal_tool_repair_followup_messages(data, registry)

    feedback = _feedback_json_from_tool_content(followups[1]["content"])
    assert feedback["error"] == "invalid_deferred_tool_arguments"
    assert feedback["missing"] == ["text"]
    assert feedback["expected"]["arguments"]["text"] == ["..."]
    assert feedback["example"]["namespace"] == "mcp__chrome_devtools_1"


def test_responses_payload_has_output_items_rejects_empty_final_after_repair():
    assert not _responses_payload_has_output_items({"output": []})
    assert not _responses_payload_has_output_items({"output": [{"type": "message", "content": []}]})
    assert _responses_payload_has_output_items({
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": "ok"}],
        }]
    })
    assert _responses_payload_has_output_items({
        "output": [{
            "type": "function_call",
            "name": "click",
            "arguments": "{}",
        }]
    })


def test_translate_call_deferred_tool_without_namespace_uses_loaded_tool_search_scope():
    tools = [
        {
            "type": "namespace",
            "name": "mcp__chrome_devtools",
            "tools": [
                {"name": "evaluate_script", "description": "Evaluate JS", "parameters": {"type": "object", "properties": {"function": {"type": "string"}}}},
            ],
        },
        {
            "type": "namespace",
            "name": "mcp__chrome_devtools_1",
            "tools": [
                {"name": "evaluate_script", "description": "Evaluate JS", "parameters": {"type": "object", "properties": {"function": {"type": "string"}}}},
            ],
        },
    ]
    registry = ResponsesToolRegistry.from_responses_tools(tools)
    search_output = registry.tool_search_output_message(
        "call_search",
        {"namespaces": ["mcp__chrome_devtools"], "query": "evaluate script"},
    )
    data = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_eval",
                            "type": "function",
                            "function": {
                                "name": "call_deferred_tool",
                                "arguments": json.dumps({
                                    "name": "evaluate_script",
                                    "arguments": {"function": "return 1"},
                                }),
                            },
                        }
                    ]
                }
            }
        ]
    }

    translated, changed = _translate_internal_deferred_tool_calls_in_chat_response(data, registry, loaded_schema_messages=[search_output])

    assert changed is True
    tool_call = translated["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "evaluate_script"
    assert tool_call["function"]["namespace"] == "mcp__chrome_devtools"
    assert tool_call["function"]["arguments"] == '{"function":"() => { return 1 }"}'



def test_translate_call_deferred_tool_uses_server_argument_as_namespace_and_strips_it():
    tools = [
        {
            "type": "namespace",
            "name": "mcp__chrome_devtools_1",
            "tools": [
                {"name": "wait_for", "description": "Wait", "parameters": {"type": "object", "properties": {"text": {"type": "array", "items": {"type": "string"}}}}},
            ],
        }
    ]
    registry = ResponsesToolRegistry.from_responses_tools(tools)

    translated = registry.translate_deferred_tool_call({
        "name": "wait_for",
        "arguments": {"server": "mcp__chrome_devtools_1", "timeout": 5000},
    })

    assert translated is not None
    assert translated["namespace"] == "mcp__chrome_devtools_1"
    assert translated["responses_name"] == "wait_for"
    assert json.loads(translated["arguments"]) == {"timeout": 5000}


def test_translate_chrome_evaluate_script_wraps_return_body_as_function_declaration():
    tools = [
        {
            "type": "namespace",
            "name": "mcp__chrome_devtools_1",
            "tools": [
                {"name": "evaluate_script", "description": "Evaluate JS", "parameters": {"type": "object", "properties": {"function": {"type": "string"}}}},
            ],
        }
    ]
    registry = ResponsesToolRegistry.from_responses_tools(tools)

    translated = registry.translate_deferred_tool_call({
        "namespace": "mcp__chrome_devtools_1",
        "name": "evaluate_script",
        "arguments": {"function": "return new Promise(resolve => setTimeout(resolve, 4000))"},
    })

    assert translated is not None
    args = json.loads(translated["arguments"])
    assert args == {"function": "() => { return new Promise(resolve => setTimeout(resolve, 4000)) }"}


def test_internal_tool_search_followup_accepts_nested_call_deferred_tool_search():
    registry = ResponsesToolRegistry.from_responses_tools(_sample_deferred_tools())
    data = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_search_nested",
                            "type": "function",
                            "function": {
                                "name": "call_deferred_tool",
                                "arguments": json.dumps({
                                    "name": "tool_search",
                                    "arguments": {
                                        "namespaces": ["mcp__chrome_devtools"],
                                        "tools": ["get_console_messages"],
                                    },
                                }),
                            },
                        }
                    ]
                }
            }
        ]
    }

    followups = _chat_response_internal_tool_search_followup_messages(data, registry)

    assert len(followups) == 2
    assert followups[0]["tool_calls"][0]["function"]["name"] == "tool_search"
    assert followups[0]["tool_calls"][0]["function"]["arguments"] == json.dumps(
        {"namespaces": ["mcp__chrome_devtools"], "tools": ["get_console_messages"]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert followups[1]["role"] == "tool"
    assert followups[1]["tool_call_id"] == "call_search_nested"
    assert "get_console_messages" in followups[1]["content"]

def test_responses_handler_contains_kv_stable_tool_search_loop():
    source = inspect.getsource(cli.start_ctx_metadata_server)
    responses_source = source[source.index("def _handle_openai_responses"):]

    assert "ResponsesToolRegistry.from_responses_tools" in responses_source
    assert "openai_responses_tool_search_round_payload" in responses_source
    assert "internal_payload[\"stream\"] = False" in responses_source
    assert "_chat_response_internal_tool_search_followup_messages" in responses_source
    assert "_translate_internal_deferred_tool_calls_in_chat_response" in responses_source
    assert "_chat_response_internal_tool_repair_followup_messages" in responses_source
    assert "openai_responses_tool_repair_feedback" in responses_source
    assert "openai_responses_tool_repair_empty_final" in responses_source
    assert "openai_responses_internal_round_max_tokens_applied" in responses_source
    assert "_responses_internal_round_max_tokens()" in responses_source
    assert "_start_responses_sse_stream" in responses_source
    assert "_write_sse_comment" in responses_source
    assert "internal chat tool continuation repair" in responses_source
    assert responses_source.index("internal chat tool continuation repair") < responses_source.index("self.wfile.write(b\"data: [DONE]\\n\\n\")")


def test_responses_payload_sse_events_preserve_function_call_namespace():
    payload = {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "m",
        "output": [
            {
                "id": "fc_test",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_test",
                "name": "get_console_messages",
                "namespace": "mcp__chrome_devtools",
                "arguments": "{}",
            }
        ],
    }

    events = _responses_payload_sse_events(payload)
    function_events = [event for name, event in events if name in {"response.output_item.added", "response.function_call_arguments.done", "response.output_item.done"}]

    assert function_events[0]["item"]["namespace"] == "mcp__chrome_devtools"
    assert function_events[1]["namespace"] == "mcp__chrome_devtools"
    assert function_events[2]["item"]["namespace"] == "mcp__chrome_devtools"


def test_responses_tool_registry_defers_namespaces_by_default_for_local_fallback():
    tools = [
        {
            "type": "namespace",
            "name": "mcp__chrome_devtools",
            "tools": [
                {"name": "list_pages", "description": "List pages", "parameters": {}},
                {"name": "click", "description": "Click", "defer_loading": False, "parameters": {}},
            ],
        },
        {"type": "function", "name": "exec_command", "parameters": {}},
    ]

    registry = ResponsesToolRegistry.from_responses_tools(tools)

    eager_names = [tool["function"]["name"] for tool in registry.eager_chat_tools]
    assert "exec_command" in eager_names
    assert "mcp__chrome_devtools__click" in eager_names
    assert "mcp__chrome_devtools__list_pages" not in eager_names
    assert "mcp__chrome_devtools__list_pages" in registry.deferred_by_legacy_name


def test_responses_payload_with_namespaces_uses_internal_tools_instead_of_flattening_all():
    tools = [
        {"type": "function", "name": "exec_command", "parameters": {}},
        {"type": "namespace", "name": "ns", "tools": [{"name": f"tool_{idx}", "parameters": {}} for idx in range(30)]},
    ]
    registry = ResponsesToolRegistry.from_responses_tools(tools)

    payload = _responses_payload_to_chat_payload({"model": "m", "input": "hi", "tools": tools}, "m", tool_registry=registry)
    tool_names = [tool["function"]["name"] for tool in payload["tools"]]

    assert tool_names == ["exec_command", "tool_search", "call_deferred_tool"]
    assert "Deferred tool directory" in payload["messages"][0]["content"]




def test_responses_history_merges_assistant_text_with_following_deferred_tool_call():
    tools = [
        {
            "type": "namespace",
            "name": "mcp__chrome_devtools",
            "tools": [
                {"name": "new_page", "description": "Open page", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}},
            ],
        }
    ]
    registry = ResponsesToolRegistry.from_responses_tools(tools)
    chat_payload = _responses_payload_to_chat_payload(
        {
            "model": "m",
            "tools": tools,
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "open gemini"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Voy a abrir Gemini."}]},
                {"type": "function_call", "call_id": "call_np", "name": "new_page", "namespace": "mcp__chrome_devtools", "arguments": '{"url":"https://gemini.google.com/"}'},
                {"type": "function_call_output", "call_id": "call_np", "output": "opened page"},
            ],
        },
        "m",
        tool_registry=registry,
    )

    assistant_messages = [message for message in chat_payload["messages"] if message.get("role") == "assistant"]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["content"] == "Voy a abrir Gemini."
    assert assistant_messages[0]["tool_calls"][0]["function"]["name"] == "call_deferred_tool"
    assert {"role": "tool", "tool_call_id": "call_np", "content": "opened page"} in chat_payload["messages"]

def test_responses_payload_preserves_deferred_namespace_tool_history_as_internal_call():
    tools = [
        {
            "type": "namespace",
            "name": "mcp__chrome_devtools",
            "tools": [
                {"name": "new_page", "description": "Open page", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}},
            ],
        }
    ]
    registry = ResponsesToolRegistry.from_responses_tools(tools)
    payload = {
        "model": "m",
        "tools": tools,
        "input": [
            {"type": "function_call", "call_id": "call_np", "name": "new_page", "namespace": "mcp__chrome_devtools", "arguments": '{"url":"https://example.com"}'},
            {"type": "function_call_output", "call_id": "call_np", "output": "opened page"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "continue"}]},
        ],
    }

    chat_payload = _responses_payload_to_chat_payload(payload, "m", tool_registry=registry)

    assistant_msg = next(message for message in chat_payload["messages"] if message.get("role") == "assistant")
    assert assistant_msg["role"] == "assistant"
    tool_call = assistant_msg["tool_calls"][0]
    assert tool_call["id"] == "call_np"
    assert tool_call["function"]["name"] == "call_deferred_tool"
    args = json.loads(tool_call["function"]["arguments"])
    assert args == {
        "namespace": "mcp__chrome_devtools",
        "name": "new_page",
        "arguments": {"url": "https://example.com"},
    }
    assert {"role": "tool", "tool_call_id": "call_np", "content": "opened page"} in chat_payload["messages"]






def test_looks_like_vision_model_uses_detected_load_capabilities():
    model = cli.ManagedModel(
        model_id="plain-name",
        repo_id="org/plain-name",
        quant="Q4",
        filename="plain.gguf",
        local_path="/tmp/plain.gguf",
        load_capabilities=["text-to-text", "image-to-text"],
    )

    assert cli._looks_like_vision_model(model) is True

def test_image_input_runtime_requires_configured_mmproj_even_with_image_capability():
    model = cli.ManagedModel(
        model_id="qwen-with-hub-image-capability",
        repo_id="repo",
        quant="Q4",
        filename="model.gguf",
        local_path="/tmp/model.gguf",
        load_capabilities=["text-to-text", "image-to-text"],
        mmproj_path=None,
    )

    assert cli._has_vision_runtime(model) is True
    assert cli._has_configured_mmproj_runtime(model) is False



def test_live_mmproj_runtime_detects_stale_loaded_process_without_mmproj(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"
    model_path.write_bytes(b"model")
    mmproj_path.write_bytes(b"proj")
    model = cli.ManagedModel(
        model_id="qwen-vision",
        repo_id="repo",
        quant="Q4",
        filename="model.gguf",
        local_path=str(model_path),
        load_capabilities=["image-to-text"],
        mmproj_path=str(mmproj_path),
    )
    monkeypatch.setattr(
        cli,
        "get_catalog_model_process",
        lambda model_id, catalog: {
            "pid": 123,
            "cmdline": f"/opt/llama-server --port 18080 --model {model_path}",
            "model_path": str(model_path),
            "port": 18080,
        },
    )

    assert cli._loaded_process_missing_configured_mmproj(model, [model]) is True


def test_live_mmproj_runtime_accepts_matching_loaded_process(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    mmproj_path = tmp_path / "mmproj.gguf"
    model_path.write_bytes(b"model")
    mmproj_path.write_bytes(b"proj")
    model = cli.ManagedModel(
        model_id="qwen-vision",
        repo_id="repo",
        quant="Q4",
        filename="model.gguf",
        local_path=str(model_path),
        load_capabilities=["image-to-text"],
        mmproj_path=str(mmproj_path),
    )
    monkeypatch.setattr(
        cli,
        "get_catalog_model_process",
        lambda model_id, catalog: {
            "pid": 123,
            "cmdline": f"/opt/llama-server --port 18080 --model {model_path} --mmproj {mmproj_path}",
            "model_path": str(model_path),
            "port": 18080,
        },
    )

    assert cli._loaded_process_missing_configured_mmproj(model, [model]) is False



def test_reload_stale_model_runtime_temporarily_removes_then_restores_config(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"model")
    model = cli.ManagedModel(
        model_id="qwen-vision",
        repo_id="repo",
        quant="Q4",
        filename="model.gguf",
        local_path=str(model_path),
    )
    calls = []

    def fake_render(catalog, config_path, llama_server, start_port, idle_ttl, **kwargs):
        calls.append([item.model_id for item in catalog])

    monkeypatch.setattr(cli, "render_llamaswap_config", fake_render)
    monkeypatch.setattr(cli, "wait_for_model_absent", lambda ids, host, port, timeout=30: calls[-1] == [])
    monkeypatch.setattr(cli, "wait_for_model", lambda model_id, host, port, timeout=45: calls[-1] == [model_id])
    monkeypatch.setattr(cli, "resolve_idle_ttl", lambda args=None: 300)
    monkeypatch.setattr(cli, "resolve_llama_server_defaults", lambda args: {"parallel": 1})
    monkeypatch.setattr(cli, "resolve_global_replica_config", lambda args: {"enabled": False})

    args = type("Args", (), {
        "config": str(tmp_path / "config.yaml"),
        "llama_server": str(tmp_path / "llama-server"),
        "start_port": 18080,
        "public_host": "127.0.0.1",
        "public_port": 11436,
    })()

    assert cli.reload_model_runtime_from_catalog_config(model, [model], args, "127.0.0.1", 11436) is True
    assert calls == [[], ["qwen-vision"]]

def test_responses_tool_output_images_are_not_forwarded_to_nonvision_models():
    image_data = "a" * 2048
    payload = {
        "model": "m",
        "tools": [{"type": "function", "name": "take_screenshot", "parameters": {}}],
        "input": [
            {"type": "function_call", "call_id": "call_img", "name": "take_screenshot", "arguments": "{}"},
            {
                "type": "function_call_output",
                "call_id": "call_img",
                "output": [
                    {"type": "input_text", "text": "Took a screenshot."},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image_data}"},
                ],
            },
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "continue"}]},
        ],
    }

    chat_payload = _responses_payload_to_chat_payload(payload, "m", allow_tool_output_images=False)

    dumped = json.dumps(chat_payload)
    assert "data:image" not in dumped
    assert image_data not in dumped
    assert "[image attached" in dumped
    assert all(not isinstance(message.get("content"), list) or not any(part.get("type") == "image_url" for part in message.get("content") if isinstance(part, dict)) for message in chat_payload["messages"])


def test_responses_tool_output_images_are_forwarded_when_vision_allowed():
    image_data = "b" * 2048
    payload = {
        "model": "m",
        "tools": [{"type": "function", "name": "take_screenshot", "parameters": {}}],
        "input": [
            {"type": "function_call", "call_id": "call_img", "name": "take_screenshot", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_img", "output": f"before data:image/png;base64,{image_data} after"},
        ],
    }

    chat_payload = _responses_payload_to_chat_payload(payload, "m", allow_tool_output_images=True)

    assert any(isinstance(message.get("content"), list) and any(part.get("type") == "image_url" for part in message.get("content") if isinstance(part, dict)) for message in chat_payload["messages"])
