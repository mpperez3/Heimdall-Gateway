from llamacpp_stack.cli import _normalize_system_messages_for_llamacpp


def _27_repro_messages():
    msgs = [{"role": "system", "content": "system base"}]
    for i in range(20):
        msgs.append({"role": "user", "content": f"user {i}"})
    msgs.append({"role": "developer", "content": "developer late instructions"})
    for i in range(5):
        msgs.append({"role": "user", "content": f"tail user {i}"})
    assert len(msgs) == 27
    assert msgs[0]["role"] == "system"
    assert msgs[21]["role"] == "developer"
    return msgs


def test_normalize_merges_system_and_developer_late_to_front():
    messages = _27_repro_messages()
    normalized = _normalize_system_messages_for_llamacpp(messages)
    assert len(normalized) == 26
    assert normalized[0]["role"] == "system"
    assert "system base" in normalized[0]["content"]
    assert "developer late instructions" in normalized[0]["content"]
    assert all(m["role"] not in ("system", "developer") for m in normalized[1:])
    roles_tail = [m["role"] for m in normalized[1:5]]
    assert roles_tail == ["user", "user", "user", "user"]


def test_normalize_single_developer_late_moves_to_front():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "developer", "content": "dev instructions"},
        {"role": "user", "content": "follow up"},
    ]
    normalized = _normalize_system_messages_for_llamacpp(messages)
    assert normalized[0]["role"] == "developer"
    assert normalized[0]["content"] == "dev instructions"
    assert [m["role"] for m in normalized] == ["developer", "user", "user"]


def test_normalize_multiple_developers_collapse():
    messages = [
        {"role": "system", "content": "s1"},
        {"role": "user", "content": "u"},
        {"role": "developer", "content": "d1"},
        {"role": "developer", "content": "d2"},
    ]
    normalized = _normalize_system_messages_for_llamacpp(messages)
    assert len(normalized) == 2
    assert normalized[0]["content"] == "s1\n\nd1\n\nd2"
    assert normalized[1]["role"] == "user"


def test_normalize_large_481_without_developer_unchanged():
    messages = [{"role": "system", "content": "sys"}]
    for i in range(480):
        messages.append({"role": "user", "content": f"msg {i}"})
    assert len(messages) == 481
    normalized = _normalize_system_messages_for_llamacpp(messages)
    assert normalized == messages
    assert normalized[0]["role"] == "system"
    assert len(normalized) == 481


def test_normalize_481_with_system_at_front_no_move_needed():
    messages = [{"role": "system", "content": "only system"}]
    messages.append({"role": "user", "content": "hello"})
    normalized = _normalize_system_messages_for_llamacpp(messages)
    assert normalized == messages


def test_normalize_preserves_order_of_non_privileged():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u1"},
        {"role": "developer", "content": "d"},
        {"role": "tool", "content": "t"},
    ]
    normalized = _normalize_system_messages_for_llamacpp(messages)
    assert normalized[0]["content"] == "s\n\nd"
    assert [m["role"] for m in normalized[1:]] == ["assistant", "user", "tool"]


def test_normalize_no_privileged_returns_same():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    normalized = _normalize_system_messages_for_llamacpp(messages)
    assert normalized == messages


def test_normalize_repro_27_no_jinja_exception_condition():
    messages = _27_repro_messages()
    normalized = _normalize_system_messages_for_llamacpp(messages)
    privileged_after_first = [
        i for i, m in enumerate(normalized[1:], start=1) if m["role"] in ("system", "developer")
    ]
    assert privileged_after_first == [], "Jinja Qwen exige system/developer solo al inicio"


def test_normalize_list_content_merge():
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "user", "content": "u"},
        {"role": "developer", "content": [{"type": "text", "text": "dev"}]},
    ]
    normalized = _normalize_system_messages_for_llamacpp(messages)
    assert len(normalized) == 2
    assert isinstance(normalized[0]["content"], list)
    assert len(normalized[0]["content"]) == 2
