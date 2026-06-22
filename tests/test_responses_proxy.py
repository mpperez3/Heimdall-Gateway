from llamacpp_stack.cli import (
    _chat_response_to_responses_payload,
    _responses_raw_passthrough_enabled,
    _responses_payload_to_chat_payload,
    _responses_namespace_tool_map,
    _responses_tools_to_chat_tools,
    _responses_tool_choice_to_chat_tool_choice,
    _write_responses_sse,
)
import inspect
import io
import json
import llamacpp_stack.cli as cli


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


def test_responses_namespace_tools_are_flattened_internally_for_legacy_chat_fallback():
    tools = [
        {
            "type": "namespace",
            "name": "mcp__codex_apps__github",
            "tools": [
                {"name": "_search", "description": "Search", "parameters": {}},
                {"name": "_fetch", "description": "Fetch", "parameters": {}},
            ],
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
            "function": {"name": "mcp__codex_apps__github___fetch", "description": "Fetch", "parameters": {}},
        },
        {
            "type": "function",
            "function": {"name": "exec_command", "description": "", "parameters": {}},
        },
    ]
    assert _responses_namespace_tool_map(tools) == {
        "mcp__codex_apps__github___search": {"namespace": "mcp__codex_apps__github", "name": "_search"},
        "mcp__codex_apps__github___fetch": {"namespace": "mcp__codex_apps__github", "name": "_fetch"},
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


def test_responses_namespace_function_call_history_maps_to_flat_legacy_tool():
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


def test_raw_responses_passthrough_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LLAMACPP_SUPERSERVER_RESPONSES_RAW_PASSTHROUGH", raising=False)

    assert _responses_raw_passthrough_enabled() is False

    monkeypatch.setenv("LLAMACPP_SUPERSERVER_RESPONSES_RAW_PASSTHROUGH", "1")

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
    assert payload["usage"]["total_tokens"] == 3
    assert payload["tools"] == [{"type": "computer_use_preview"}]
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is True


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
