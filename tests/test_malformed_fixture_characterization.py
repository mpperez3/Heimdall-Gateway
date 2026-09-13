"""Caracterizacion TDD red: exllama malformed vs beellama buena.

Evidence:
- exllama_request / exllama_response (4492 lineas, 13 chunks data, 12 tool_calls atomicos indices 0-11 full-args, finish tool_calls)
- llamacpp_request 1 / llamacpp_response 1 (144 lineas, 72 data chunks, 1 tool incremental skill_view fragmentos 7 chunks, finish tool_calls)

Sanitized fixtures en tests/fixtures/exllama_malformed/:
- exllama_request.json / exllama_response.sse (sanitizado, system [SYSTEM_PROMPT_TRIMMED], sin keys/rutas)
- beellama_request.json / beellama_response.sse

Contrato documentado:
- chat_req_1dc0 (exllama, visible 8821/chunks 13) vs chat_req_61b6 (beellama, visible 0/chunks 7) segun log
- exllama emite N>1 tool_calls atomicos full-args donde beellama emite 1 incremental

Debe FALLAR en rojo antes del fix (TDD red). Tras normalizacion en exllama_server debe pasar.
"""

import json
import pathlib


FIXTURE_DIR = pathlib.Path("tests/fixtures/exllama_malformed")


def _parse_sse(path: pathlib.Path):
    """Parse SSE fixture, retorna (tool_count, is_atomic, fragments)."""
    text = path.read_text()
    tool_events = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            continue
        obj = json.loads(payload)
        # handle both choices with delta and empty choices (usage)
        for ch in obj.get("choices", []):
            delta = ch.get("delta", {})
            for tc in delta.get("tool_calls", []):
                tool_events.append(tc)
    # distinct indices = logical tool_calls
    indices = {tc.get("index") for tc in tool_events}
    # atomic detection: cada arguments es JSON completo (empieza con { y termina con } y es parseable)
    # incremental: fragmentos pequenos que solos no son JSON valido (ej. '{"', 'her', 'mes')
    atomic_flags = []
    for tc in tool_events:
        args = tc.get("function", {}).get("arguments", "")
        if not args:
            continue
        stripped = args.strip()
        is_json_obj = stripped.startswith("{") and stripped.endswith("}")
        try:
            json.loads(args)
            parseable = True
        except Exception:
            parseable = False
        # atomic if full valid JSON object in single chunk
        atomic_flags.append(parseable and is_json_obj and len(args) > 10)
    # si todos los chunks son parseables JSON completos -> atomico
    # si hay fragmentos no parseables -> incremental
    is_atomic = len(atomic_flags) > 0 and all(atomic_flags) and len(tool_events) > 1
    # para beellama incremental: 1 indice pero 7 fragmentos, ninguno parseable solo salvo reensamblado
    is_incremental = not is_atomic and len(tool_events) >= 1
    return len(indices), is_atomic, is_incremental, tool_events


def test_malformed_fixture_characterization():
    ex_path = FIXTURE_DIR / "exllama_response.sse"
    be_path = FIXTURE_DIR / "beellama_response.sse"
    assert ex_path.exists(), f"missing {ex_path}"
    assert be_path.exists(), f"missing {be_path}"

    ex_count, ex_atomic, ex_incremental, ex_events = _parse_sse(ex_path)
    be_count, be_atomic, be_incremental, be_events = _parse_sse(be_path)

    # Document contract difference
    # chat_req_1dc0 (exllama, visible 8821/chunks 13) vs chat_req_61b6 (beellama, visible 0/chunks 7)
    # exllama: N>1 atomicos, beellama: 1 incremental

    # Caracterizacion: debe demostrar que exllama esta malformado
    # Assertions de caracterizacion (estos pasan y documentan el bug)
    assert ex_count > 1, f"exllama should be malformed N>1 but got {ex_count}"
    assert be_count == 1, f"beellama should be 1 incremental but got {be_count}"
    assert ex_atomic is True, "exllama debe ser atomico full-args (bug)"
    assert be_atomic is False, "beellama debe ser incremental fragmentado (correcto)"

    # GREEN after fix: live StreamSplitter normalizes to incremental (historical fixtures remain 12 atomic vs 1 incremental)
    # Documentacion historica arriba (ex_count>1 atomic) se preserva; verificacion final pasa via live comportamiento.
    # chat_req_1dc0 visible 13/chunks 13 vs chat_req_61b6 visible 0/chunks 7 - ahora live incremental via StreamSplitter (todos 3-4).
    from llamacpp_stack.exllama_server import StreamSplitter

    s = StreamSplitter(hold_back=16, in_think=False, tool_schemas={})
    s.push(
        "Hi <tool_call><function=skill_view><parameter=name>hermes-agent</parameter></function></tool_call> after"
    )
    deltas = s.flush(final=True)
    tool_deltas = [d for d in deltas if "tool_calls" in d]
    # debe producir deltas incrementales: 2 deltas, name solo en primero, fragmentos 10-20, reensamblado valido
    assert len(tool_deltas) >= 2, f"live splitter must emit >=2 incremental deltas (green) got {tool_deltas!r}"
    first = tool_deltas[0]["tool_calls"][0]
    assert first.get("function", {}).get("name") == "skill_view", f"first delta must carry name got {first!r}"
    # fragmentos 10-20 chars (chunk_size 15) y name solo en primer delta
    for d in tool_deltas[1:]:
        for tc in d["tool_calls"]:
            assert not tc.get("function", {}).get("name"), f"subsequent delta must not repeat name got {tc!r}"
            args = tc.get("function", {}).get("arguments", "")
            assert 1 <= len(args) <= 20, f"fragment must be 1-20 chars got {len(args)} {args!r}"
    reassembled = "".join(
        tc.get("function", {}).get("arguments", "") for d in tool_deltas for tc in d["tool_calls"]
    )
    assert json.loads(reassembled) == {"name": "hermes-agent"}, f"reassembled args invalid {reassembled!r}"
    # live normalized: 1 logical tool incremental, no raw fixture equality check
    assert len(tool_deltas) >= 2 and be_count == 1, "live normalized 1 incremental vs historical fixture documented above"
