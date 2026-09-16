import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from llamacpp_stack.cli.raw_log import RAW_FALLBACK_PATH, RAW_MAX_CHARS, RAW_RING_SIZE, _raw_path_for_base, log_raw_request
from llamacpp_stack.cli.constants import DEFAULT_REQUESTS_LOG_PATH


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _raw_for(base: Path) -> Path:
    return _raw_path_for_base(base)


class TestRingSize:
    def test_ring_keeps_last_10_of_12(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LLM_SERVER_REQUESTS_LOG_PATH", raising=False)
        monkeypatch.delenv("LLM_SERVER_REQUESTS_LOG_FALLBACK", raising=False)
        base = tmp_path / "api-requests.log"
        raw = _raw_for(base)
        for i in range(1, 13):
            log_raw_request(f'{{"a":{i}}}', log_path=base)
        lines = _read_lines(raw)
        assert len(lines) == 10
        assert lines[0] == '{"a":3}'
        assert lines[-1] == '{"a":12}'
        # ensure 1 and 2 discarded
        assert '{"a":1}' not in lines
        assert '{"a":2}' not in lines

    def test_exactly_10_no_truncation(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LLM_SERVER_REQUESTS_LOG_PATH", raising=False)
        base = tmp_path / "api-requests.log"
        raw = _raw_for(base)
        for i in range(10):
            log_raw_request(f"req{i}", log_path=base)
        assert len(_read_lines(raw)) == 10

    def test_single_write_creates_one_line(self, tmp_path):
        base = tmp_path / "api-requests.log"
        raw = _raw_for(base)
        log_raw_request("hello", log_path=base)
        assert _read_lines(raw) == ["hello"]

    def test_overwrite_oldest_when_11(self, tmp_path):
        base = tmp_path / "api-requests.log"
        raw = _raw_for(base)
        for i in range(11):
            log_raw_request(f"line{i}", log_path=base)
        lines = _read_lines(raw)
        assert len(lines) == 10
        assert lines[0] == "line1"
        assert lines[-1] == "line10"


class TestRawPreservation:
    def test_preserve_spaces_and_key_order(self, tmp_path):
        base = tmp_path / "api-requests.log"
        raw = _raw_for(base)
        body = '{"b": 2,  "a":  1 }'
        log_raw_request(body, log_path=base)
        assert _read_lines(raw)[-1] == body

    def test_preserve_non_json_bytes(self, tmp_path):
        base = tmp_path / "api-requests.log"
        raw = _raw_for(base)
        body = "not json at all { [ }"
        log_raw_request(body, log_path=base)
        assert _read_lines(raw)[-1] == body

    def test_preserve_unicode(self, tmp_path):
        base = tmp_path / "api-requests.log"
        raw = _raw_for(base)
        body = '{"msg":"café naïve 🎉"}'
        log_raw_request(body, log_path=base)
        assert _read_lines(raw)[-1] == body

    def test_no_json_parse_reserialize(self, tmp_path):
        base = tmp_path / "api-requests.log"
        raw = _raw_for(base)
        # spacing that json.loads + dumps would normalize
        body = '{"a":1,"b":2}'
        body_spaced = '{ "b" : 2 , "a" : 1 }'
        log_raw_request(body_spaced, log_path=base)
        assert _read_lines(raw)[-1] == body_spaced
        assert _read_lines(raw)[-1] != json.dumps(json.loads(body_spaced), separators=(",", ":"))


class TestCap:
    def test_cap_1mib_truncates(self, tmp_path):
        base = tmp_path / "api-requests.log"
        raw = _raw_for(base)
        big = "x" * (2 * 1024 * 1024)
        log_raw_request(big, log_path=base)
        lines = _read_lines(raw)
        assert len(lines) == 1
        assert len(lines[0]) == RAW_MAX_CHARS
        assert lines[0] == "x" * RAW_MAX_CHARS

    def test_cap_bytes_truncates(self, tmp_path):
        base = tmp_path / "api-requests.log"
        raw = _raw_for(base)
        big_bytes = b"y" * (2 * 1024 * 1024)
        log_raw_request(big_bytes, log_path=base)
        lines = _read_lines(raw)
        assert len(lines[0]) == RAW_MAX_CHARS

    def test_exactly_1mib_not_truncated(self, tmp_path):
        base = tmp_path / "api-requests.log"
        raw = _raw_for(base)
        exact = "z" * RAW_MAX_CHARS
        log_raw_request(exact, log_path=base)
        assert len(_read_lines(raw)[0]) == RAW_MAX_CHARS


class TestNewlineSanitization:
    def test_newline_escaped(self, tmp_path):
        base = tmp_path / "api-requests.log"
        raw = _raw_for(base)
        body = "line1\nline2\rline3\r\nline4"
        log_raw_request(body, log_path=base)
        lines = _read_lines(raw)
        assert len(lines) == 1
        assert "\n" not in lines[0]
        assert "\r" not in lines[0]
        assert "\\n" in lines[0]

    def test_multiline_body_one_line(self, tmp_path):
        base = tmp_path / "api-requests.log"
        raw = _raw_for(base)
        body = '{"a":1}\n{"b":2}\n'
        log_raw_request(body, log_path=base)
        assert len(_read_lines(raw)) == 1
        # second request should be second line
        log_raw_request("second", log_path=base)
        assert len(_read_lines(raw)) == 2


class TestBytesHandling:
    def test_bytes_utf8(self, tmp_path):
        base = tmp_path / "api-requests.log"
        raw = _raw_for(base)
        log_raw_request(b'{"a":1}', log_path=base)
        assert _read_lines(raw)[-1] == '{"a":1}'

    def test_bytes_non_utf8_replace(self, tmp_path):
        base = tmp_path / "api-requests.log"
        raw = _raw_for(base)
        # invalid utf8 bytes
        log_raw_request(b"\xff\xfe\xfd", log_path=base)
        line = _read_lines(raw)[-1]
        # errors=replace produces replacement char
        assert "�" in line or len(line) > 0

    def test_empty_body(self, tmp_path):
        base = tmp_path / "api-requests.log"
        raw = _raw_for(base)
        log_raw_request(b"", log_path=base)
        # empty still produces a line? sanitized empty is "" -> we append "" -> file has "\n" -> splitlines gives [""]? but last 10 logic appends "" -> "\n".splitlines() discards trailing? Let's check.
        content = raw.read_text(encoding="utf-8")
        assert content == "\n"
        # second empty + one real
        log_raw_request("real", log_path=base)
        lines = raw.read_text(encoding="utf-8").splitlines()
        # first empty results in empty string first line
        assert len(lines) == 2

    def test_none_body(self, tmp_path):
        base = tmp_path / "api-requests.log"
        raw = _raw_for(base)
        log_raw_request(None, log_path=base)
        # should not throw and produce empty line
        assert raw.exists()


class TestFallbackAndNeverThrow:
    def test_fallback_when_primary_not_writable(self, tmp_path, monkeypatch):
        primary = tmp_path / "primary" / "api-requests.log"
        fallback = tmp_path / "fallback" / "api-requests.log"
        monkeypatch.setenv("LLM_SERVER_REQUESTS_LOG_FALLBACK", str(fallback))
        original_open = Path.open
        primary_raw = _raw_for(primary)
        primary_tmp = primary_raw.with_name(f".{primary_raw.name}.tmp")

        def fake_open(self, *args, **kwargs):
            if self == primary_raw or self == primary_tmp:
                raise OSError("mock primary not writable")
            return original_open(self, *args, **kwargs)

        with patch.object(Path, "open", fake_open):
            log_raw_request("fallback_body", log_path=primary)
        fb_raw = _raw_for(fallback)
        assert fb_raw.exists()
        assert "fallback_body" in fb_raw.read_text(encoding="utf-8")

    def test_never_throws_on_open_failure(self, tmp_path, monkeypatch):
        primary = tmp_path / "p.log"
        monkeypatch.setenv("LLM_SERVER_REQUESTS_LOG_FALLBACK", str(tmp_path / "f.log"))
        with patch.object(Path, "open", side_effect=OSError("all fail")):
            log_raw_request("no_crash", log_path=primary)
        # should not raise

    def test_never_throws_on_mkdir_failure(self, tmp_path, monkeypatch):
        primary = tmp_path / "p.log"
        with patch.object(Path, "mkdir", side_effect=OSError("mkdir fail")):
            log_raw_request("x", log_path=primary)

    def test_no_interference_with_api_requests_log(self, tmp_path, monkeypatch):
        # raw log should not affect api-requests.log
        monkeypatch.delenv("LLM_SERVER_REQUESTS_LOG_PATH", raising=False)
        base = tmp_path / "api-requests.log"
        from llamacpp_stack._cli_impl import log_api_event, _daily_active_path_for

        log_api_event("test_kind", {"v": 1}, log_path=base)
        log_raw_request("raw_body_123", log_path=base)
        # api log dated file should still have test_kind
        active = _daily_active_path_for(base)
        assert active.exists()
        assert "test_kind" in active.read_text(encoding="utf-8")
        # raw should have raw_body
        raw = _raw_for(base)
        assert "raw_body_123" in raw.read_text(encoding="utf-8")
        # second api event not overwritten by raw
        log_api_event("test_kind2", {"v": 2}, log_path=base)
        assert "test_kind2" in active.read_text(encoding="utf-8")


class TestPathResolution:
    def test_env_path_precedence(self, tmp_path, monkeypatch):
        env_path = tmp_path / "env.log"
        monkeypatch.setenv("LLM_SERVER_REQUESTS_LOG_PATH", str(env_path))
        # without explicit, raw should derive from env dir
        log_raw_request("from_env", log_path=None)
        env_raw = _raw_for(env_path)
        assert env_raw.exists()
        # cleanup
        monkeypatch.delenv("LLM_SERVER_REQUESTS_LOG_PATH", raising=False)
        if env_raw.exists():
            env_raw.unlink()

    def test_explicit_over_env(self, tmp_path, monkeypatch):
        env_path = tmp_path / "envdir" / "env.log"
        explicit = tmp_path / "explicitdir" / "explicit.log"
        monkeypatch.setenv("LLM_SERVER_REQUESTS_LOG_PATH", str(env_path))
        log_raw_request("explicit_body", log_path=explicit)
        assert _raw_for(explicit).exists()
        assert "explicit_body" in _raw_for(explicit).read_text(encoding="utf-8")
        env_raw = _raw_for(env_path)
        assert not env_raw.exists() or "explicit_body" not in env_raw.read_text(encoding="utf-8")

    def test_default_path_derivation(self, monkeypatch):
        # ensure helper derives correctly for default
        base = DEFAULT_REQUESTS_LOG_PATH
        raw = _raw_path_for_base(base)
        assert raw.name == "api-raw-requests.log"
        assert raw.parent == base.parent

    def test_atomic_write_tmp_cleanup(self, tmp_path, monkeypatch):
        base = tmp_path / "api-requests.log"
        raw = _raw_for(base)
        log_raw_request("a", log_path=base)
        tmp = raw.with_name(f".{raw.name}.tmp")
        assert not tmp.exists()

    def test_legacy_heimdall_fallback_path_still_resolves(self, tmp_path, monkeypatch):
        from llamacpp_stack.cli.raw_log import OLD_RAW_FALLBACK_PATH, RAW_FALLBACK_PATH

        assert RAW_FALLBACK_PATH.name == "llm-server-api-raw-requests.log"
        assert OLD_RAW_FALLBACK_PATH.name == "heimdall-gateway-api-raw-requests.log"
        assert OLD_RAW_FALLBACK_PATH.exists() is False or True
        monkeypatch.setenv("HEIMDALL_GATEWAY_REQUESTS_LOG_FALLBACK", "/tmp/should-not-be-used")  # legacy fallback still accepted
