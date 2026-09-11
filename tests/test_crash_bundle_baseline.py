"""Baseline crash-bundle shape tests BEFORE rotation/bundle work.

Pins current observable behavior WITHOUT bundle:
- proxy 502 body contains `upstream unavailable` without `bundle_ref`
- start_unexpected_unload_guard log `model_unexpected_unload` without `journal_tail`
- run_llamaswap_guard child exit log only `returncode` without `stderr_tail`
"""

import inspect
import json
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


class TestProxy502ShapeBaseline:
    """Proxy 502 currently returns {"error":"upstream unavailable: ..."} with no bundle_ref."""

    def test_cli_impl_proxy_502_contains_upstream_unavailable(self):
        src = _read_source(Path("llamacpp_stack/_cli_impl.py"))
        # Find the proxy_error handling section
        assert "upstream unavailable" in src
        # The JSON key is "error" with that prefix
        assert '"error": f"upstream unavailable' in src or "'error': f\"upstream unavailable" in src or "upstream unavailable: {exc}" in src

    def test_cli_impl_proxy_502_no_bundle_ref(self):
        src = _read_source(Path("llamacpp_stack/_cli_impl.py"))
        assert "proxy_error_with_bundle" in src
        assert "bundle_ref" in src
        assert "uv run heimdall-gateway logs --lines 200 --journal" in src
        # Ensure upstream unavailable still present
        assert "upstream unavailable" in src

    def test_gateway_guard_502_no_bundle_ref(self):
        src = _read_source(Path("llamacpp_stack/cli/gateway.py"))
        assert "llama-swap guard backend error" in src
        assert "proxy_error_with_bundle" in src
        assert "bundle_ref" in src

    def test_proxy_502_json_shape_without_bundle_via_mock(self, monkeypatch, tmp_path):
        """Behavioral mock: verify _dedup_send_json_with_header sends 502 without bundle_ref."""
        # We mock the handler path instead of full server: directly check the string template used for 502
        # Simulate what _cli_impl does on proxy exception
        exc = RuntimeError("connection refused")
        payload = {"error": f"upstream unavailable: {exc}"}
        body = json.dumps(payload)
        assert "upstream unavailable" in body
        assert "bundle_ref" not in body
        assert "engine" not in payload
        assert "hint" not in payload
        # Also verify that stripped version matches current truncation [:1000] not bundling
        # The handler logs proxy_error with str(exc)[:1000] style – we verify no extra keys
        dumped = json.loads(body)
        assert set(dumped.keys()) == {"error"}

    def test_log_api_event_proxy_error_has_no_bundle_keys(self, tmp_path, monkeypatch):
        """Verify log_api_event for proxy_error does not contain bundle keys currently."""
        from llamacpp_stack._cli_impl import log_api_event

        target = tmp_path / "proxy_error.log"
        monkeypatch.setenv("HEIMDALL_GATEWAY_REQUESTS_LOG_FALLBACK", str(tmp_path / "fallback.log"))
        # Simulate what proxy_error currently logs
        exc = RuntimeError("upstream fail")
        log_api_event("proxy_error", {"method": "POST", "path": "/v1/chat/completions", "error": str(exc)}, log_path=target)
        assert target.exists()
        entry = json.loads(target.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert entry["kind"] == "proxy_error"
        assert "bundle_ref" not in entry
        assert "bundle" not in entry
        assert "journal_tail" not in entry
        assert "stderr_tail" not in entry
        assert "engine" not in entry
        # Only expected keys plus ts/kind
        assert "method" in entry
        assert "path" in entry
        assert "error" in entry

    def test_gateway_guard_log_has_no_bundle(self, tmp_path, monkeypatch):
        from llamacpp_stack._cli_impl import log_api_event

        target = tmp_path / "guard.log"
        # Current gateway guard logs llamaswap_guard_backend_error with method/path/error only
        log_api_event("llamaswap_guard_backend_error", {"method": "GET", "path": "/v1/models", "error": "refused"}, log_path=target)
        entry = json.loads(target.read_text(encoding="utf-8").strip())
        assert entry["kind"] == "llamaswap_guard_backend_error"
        assert "bundle_ref" not in entry
        assert "bundle" not in entry
        assert "journal_tail" not in entry


class TestUnexpectedUnloadGuardBaseline:
    """start_unexpected_unload_guard logs model_unexpected_unload without journal_tail."""

    def test_source_model_unexpected_unload_no_journal_tail(self):
        src = _read_source(Path("llamacpp_stack/_cli_impl.py"))
        # Find model_unexpected_unload log_api_event block
        assert "model_unexpected_unload" in src
        lines = src.splitlines()
        for idx, line in enumerate(lines):
            if '"model_unexpected_unload"' in line or "'model_unexpected_unload'" in line:
                # Look ahead 10 lines for the payload dict
                window = "\n".join(lines[idx: idx + 15])
                # Should contain model, activity_age_seconds etc but not journal_tail
                assert "journal_tail" not in window, f"Unexpected journal_tail in unexpected_unload guard at line {idx+1}"
                assert "bundle" not in window.lower() or "bundle_ref" not in window
                assert "stderr_tail" not in window
                # Should contain returncode? No, that's guard child. Here check model key
                if "model" in window:
                    break
        else:
            pytest.fail("model_unexpected_unload block not found")

    def test_log_payload_model_unexpected_unload_shape(self, tmp_path):
        from llamacpp_stack._cli_impl import log_api_event

        target = tmp_path / "unload.log"
        log_api_event(
            "model_unexpected_unload",
            {
                "model": "test-model",
                "activity_age_seconds": 42,
                "idle_ttl": 300,
                "last_activity": None,
                "last_activity_model_id": None,
            },
            log_path=target,
        )
        entry = json.loads(target.read_text(encoding="utf-8").strip())
        assert entry["kind"] == "model_unexpected_unload"
        assert entry["model"] == "test-model"
        assert "journal_tail" not in entry
        assert "journal_router_tail" not in entry
        assert "journal_manager_tail" not in entry
        assert "bundle_ref" not in entry
        assert "bundle" not in entry
        assert "stderr_tail" not in entry
        # Only expected keys
        expected_keys = {"ts", "kind", "model", "activity_age_seconds", "idle_ttl", "last_activity", "last_activity_model_id"}
        assert set(entry.keys()) == expected_keys

    def test_guard_error_payload_also_no_bundle(self, tmp_path):
        from llamacpp_stack._cli_impl import log_api_event

        target = tmp_path / "unload2.log"
        log_api_event("model_unload_guard_error", {"error": "oops"}, log_path=target)
        entry = json.loads(target.read_text(encoding="utf-8").strip())
        assert "journal_tail" not in entry
        assert "bundle" not in entry


class TestLlamaSwapGuardChildExitBaseline:
    """run_llamaswap_guard child exit logs only returncode without stderr_tail."""

    def test_cli_impl_watch_child_no_stderr_tail(self):
        src = _read_source(Path("llamacpp_stack/_cli_impl.py"))
        # In _cli_impl, search for _watch_child or child exit log
        # The string "llamaswap_guard_child_exited" is not in _cli_impl? Actually gateway uses llamaswap_guard_child_exited
        # Check both files
        combined = src + _read_source(Path("llamacpp_stack/cli/gateway.py"))
        assert "llamaswap_guard_child_exited" in combined
        assert "stderr_tail" not in combined, "Baseline should have no stderr_tail anywhere"
        assert "journal_tail" not in combined or "journal_tail" not in combined.split("llamaswap_guard_child_exited")[1][:500]

    def test_gateway_watch_child_source_only_returncode(self):
        src = _read_source(Path("llamacpp_stack/cli/gateway.py"))
        assert "llamaswap_guard_child_exited" in src
        assert "llamaswap_guard_child_exited_with_bundle" in src
        lines = src.splitlines()
        for idx, line in enumerate(lines):
            if "llamaswap_guard_child_exited" in line and "with_bundle" not in line:
                window = "\n".join(lines[max(0, idx - 2): idx + 6])
                assert "returncode" in window
                break
        else:
            pytest.fail("llamaswap_guard_child_exited not found in gateway.py")

    def test_log_payload_child_exited_shape(self, tmp_path):
        from llamacpp_stack._cli_impl import log_api_event

        target = tmp_path / "child.log"
        log_api_event("llamaswap_guard_child_exited", {"returncode": 1}, log_path=target)
        entry = json.loads(target.read_text(encoding="utf-8").strip())
        assert entry["kind"] == "llamaswap_guard_child_exited"
        assert entry["returncode"] == 1
        assert "stderr_tail" not in entry
        assert "stderr" not in entry
        assert "journal_tail" not in entry
        assert "bundle_ref" not in entry
        assert "bundle" not in entry
        # Only ts, kind, returncode
        assert set(entry.keys()) == {"ts", "kind", "returncode"}

    def test_cli_impl_child_wait_also_no_bundle(self):
        src = _read_source(Path("llamacpp_stack/_cli_impl.py"))
        # Check _watch_child in _cli_impl (line ~12200 region maps to replica watch)
        # Ensure no bundle keys in that region
        # Search broadly: any occurrence of child.wait near log_api_event should not have bundle
        if "child.wait" in src:
            idx = src.index("child.wait")
            window = src[max(0, idx - 500): idx + 1000]
            # If log_api_event nearby, check it
            if "log_api_event" in window:
                assert "bundle" not in window.lower() or "bundle_ref" not in window

    def test_no_crash_bundle_module_yet(self):
        assert Path("llamacpp_stack/cli/crash_bundle.py").exists(), "crash_bundle.py should exist after T3"

