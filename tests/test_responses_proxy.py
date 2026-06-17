from llamacpp_stack.cli import (
    _chat_response_to_responses_payload,
    _responses_raw_passthrough_enabled,
    _responses_payload_to_chat_payload,
)


def test_responses_payload_keeps_modern_tools_out_of_legacy_chat_fallback():
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

    assert chat_payload == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "hello"},
        ],
        "stream": False,
        "max_tokens": 32,
        "temperature": 0.2,
    }
    # The legacy chat fallback cannot execute Responses-native tools, so they
    # must not be forwarded to /v1/chat/completions. The real /v1/responses
    # passthrough path sends the original payload unchanged.
    assert "tools" not in chat_payload
    assert "parallel_tool_calls" not in chat_payload


def test_responses_payload_accepts_string_input():
    chat_payload = _responses_payload_to_chat_payload(
        {"model": "m", "input": "Say hi", "stream": True},
        "m",
    )

    assert chat_payload["messages"] == [{"role": "user", "content": "Say hi"}]
    assert chat_payload["stream"] is False


def test_raw_responses_passthrough_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LLAMACPP_SUPERSERVER_RESPONSES_RAW_PASSTHROUGH", raising=False)

    assert _responses_raw_passthrough_enabled() is False

    monkeypatch.setenv("LLAMACPP_SUPERSERVER_RESPONSES_RAW_PASSTHROUGH", "1")

    assert _responses_raw_passthrough_enabled() is True


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
