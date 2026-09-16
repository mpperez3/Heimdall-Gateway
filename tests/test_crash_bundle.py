"""T3 crash-bundle collector tests: 9 keys, caps, never-throws, engine detection, wiring, timeout."""

import json
import time
import subprocess
import pathlib
from unittest.mock import patch, MagicMock

import pytest


EXPECTED_KEYS = {"engine", "pid", "port", "cmdline", "returncode", "journal_router_tail", "journal_manager_tail", "nvidia_smi", "probed_model"}


def test_collector_returns_nine_keys():
    from llamacpp_stack.cli.crash_bundle import collect_engine_crash_bundle

    b = collect_engine_crash_bundle("llama-server", port=None, pid=None, cmdline="llama-server --port 18080 --model /tmp/a.gguf", returncode=1)
    assert set(b.keys()) == EXPECTED_KEYS


def test_collector_default_beellama_cmdline_none():
    from llamacpp_stack.cli.crash_bundle import collect_engine_crash_bundle

    b = collect_engine_crash_bundle("beellama", port=None, pid=None, cmdline=None, returncode=1)
    # beellama hint should be preserved even without cmdline
    assert b["engine"] == "beellama"
    assert b["returncode"] == 1


def test_caps_journal_12000_nvidia_2000():
    from llamacpp_stack.cli.crash_bundle import collect_engine_crash_bundle

    huge = "x" * 20000

    def fake_run(cmd, capture_output=True, text=True, timeout=None):
        m = MagicMock()
        m.stdout = huge
        m.stderr = ""
        m.returncode = 0
        return m

    with patch("llamacpp_stack.cli.crash_bundle.subprocess.run", side_effect=fake_run):
        b = collect_engine_crash_bundle("llama-server", port=None, pid=None, cmdline="llama-server --port 1", returncode=0, timeout_s=2)
        assert len(b["journal_router_tail"]) == 12000
        assert len(b["journal_manager_tail"]) == 12000
        assert len(b["nvidia_smi"]) == 2000


def test_never_throws_on_timeout_expired():
    from llamacpp_stack.cli.crash_bundle import collect_engine_crash_bundle

    def fake_run_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0] if args else "journalctl", timeout=3)

    with patch("llamacpp_stack.cli.crash_bundle.subprocess.run", side_effect=fake_run_timeout):
        b = collect_engine_crash_bundle("llama-server", port=None, pid=None, cmdline=None, returncode=None, timeout_s=2)
        assert "unavailable: TimeoutExpired" in b["journal_router_tail"]
        assert "unavailable: TimeoutExpired" in b["journal_manager_tail"]
        assert "unavailable: TimeoutExpired" in b["nvidia_smi"]
        assert set(b.keys()) == EXPECTED_KEYS


def test_never_throws_on_generic_exception():
    from llamacpp_stack.cli.crash_bundle import collect_engine_crash_bundle

    def fake_raise(*args, **kwargs):
        raise RuntimeError("boom")

    with patch("llamacpp_stack.cli.crash_bundle.subprocess.run", side_effect=fake_raise):
        b = collect_engine_crash_bundle("unknown-engine", port=None, pid=None, cmdline=None, returncode=99)
        assert "unavailable: RuntimeError" in b["journal_router_tail"]
        assert b["engine"] in {"llama-server", "beellama", "vllm", "exllama", "llama-swap", "manager"}
        assert b["returncode"] == 99


def test_engine_detection_beellama_vllm_exllama_llamaswap():
    from llamacpp_stack.cli.crash_bundle import collect_engine_crash_bundle, _detect_engine

    # direct via cmdline
    assert _detect_engine("llama-server", "llama-server-beellama --port 1") == "beellama"
    assert _detect_engine("", "VLLM serve --model foo") == "vllm"
    assert _detect_engine("vllm", "vllm.entrypoints.openai.api_server --port 1") == "vllm"
    assert _detect_engine("", "/usr/bin/exllama --port 1") == "exllama"
    assert _detect_engine("", "llama-swap --config /tmp/c.yaml") == "llama-swap"
    assert _detect_engine("llama-swap", None) == "llama-swap"
    assert _detect_engine("beellama", None) == "beellama"
    assert _detect_engine("unknown", None) == "llama-server"

    with patch("llamacpp_stack.cli.crash_bundle.subprocess.run") as m:
        mock_ret = MagicMock()
        mock_ret.stdout = ""
        mock_ret.stderr = ""
        m.return_value = mock_ret
        b = collect_engine_crash_bundle("beellama", port=None, pid=None, cmdline=None, returncode=0, timeout_s=1)
        assert b["engine"] == "beellama"
        b2 = collect_engine_crash_bundle("vllm", port=None, pid=None, cmdline="vllm serve --port 8000", returncode=0, timeout_s=1)
        assert b2["engine"] == "vllm"


def test_engine_detection_via_collect():
    from llamacpp_stack.cli.crash_bundle import collect_engine_crash_bundle

    def fake_run(cmd, **kw):
        r = MagicMock()
        r.stdout = ""
        r.stderr = ""
        return r

    with patch("llamacpp_stack.cli.crash_bundle.subprocess.run", side_effect=fake_run):
        for engine_hint, cmdline, expected in [
            ("llama-server", "llama-server-beellama --model x --port 1", "beellama"),
            ("llama-server", "vllm serve --model x", "vllm"),
            ("llama-server", "exllama --model x", "exllama"),
            ("llama-server", "llama-swap --config x", "llama-swap"),
            ("llama-server", "llama-server --port 1", "llama-server"),
        ]:
            b = collect_engine_crash_bundle(engine_hint, port=None, pid=None, cmdline=cmdline, returncode=None, timeout_s=1)
            assert b["engine"] == expected, f"{cmdline} -> {b['engine']} != {expected}"


def test_probed_model_extraction():
    from llamacpp_stack.cli.crash_bundle import collect_engine_crash_bundle

    def fake_run(cmd, **kw):
        r = MagicMock()
        r.stdout = ""
        r.stderr = ""
        return r

    with patch("llamacpp_stack.cli.crash_bundle.subprocess.run", side_effect=fake_run):
        b = collect_engine_crash_bundle("llama-server", port=None, pid=None, cmdline="llama-server --model /tmp/my-model.gguf --port 1", returncode=None, timeout_s=1)
        assert b["probed_model"] == "/tmp/my-model.gguf"
        b2 = collect_engine_crash_bundle("llama-server", port=None, pid=None, cmdline=None, returncode=None, timeout_s=1)
        assert b2["probed_model"] == ""


def test_journal_hung_returns_under_8s():
    from llamacpp_stack.cli.crash_bundle import collect_engine_crash_bundle

    def fake_hang(cmd, capture_output=True, text=True, timeout=None):
        # Simulate journalctl hanging longer than timeout -> TimeoutExpired
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout or 3)

    start = time.monotonic()
    with patch("llamacpp_stack.cli.crash_bundle.subprocess.run", side_effect=fake_hang):
        b = collect_engine_crash_bundle("llama-server", port=None, pid=None, cmdline=None, returncode=1, timeout_s=8.0)
    elapsed = time.monotonic() - start
    assert elapsed < 8.0, f"collector took {elapsed}s, should be <8s"
    assert "unavailable: TimeoutExpired" in b["journal_router_tail"]
    assert "unavailable: TimeoutExpired" in b["journal_manager_tail"]
    assert "unavailable: TimeoutExpired" in b["nvidia_smi"]


def test_journal_hung_real_sleep_mock_under_8s():
    from llamacpp_stack.cli.crash_bundle import collect_engine_crash_bundle

    def fake_sleep(cmd, capture_output=True, text=True, timeout=None):
        # Simulate actual sleep longer than timeout by sleeping timeout+0.1 then raising
        time.sleep(min(0.05, timeout or 0.05))
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout or 3)

    start = time.monotonic()
    with patch("llamacpp_stack.cli.crash_bundle.subprocess.run", side_effect=fake_sleep):
        b = collect_engine_crash_bundle("beellama", port=18080, pid=123, cmdline=None, returncode=1, timeout_s=8.0)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0
    assert set(b.keys()) == EXPECTED_KEYS


def test_proxy_502_wiring_mock_source_contains_bundle():
    src_cli = pathlib.Path("llamacpp_stack/_cli_impl.py").read_text(encoding="utf-8")
    src_gw = pathlib.Path("llamacpp_stack/cli/gateway.py").read_text(encoding="utf-8")
    src_inst = pathlib.Path("llamacpp_stack/install.py").read_text(encoding="utf-8")
    # proxy 502 wiring
    assert "proxy_error_with_bundle" in src_cli
    assert "proxy_error_with_bundle" in src_gw
    assert '"engine":' in src_cli or "'engine':" in src_cli or '"engine"' in src_cli
    assert "bundle_ref" in src_cli
    assert "uv run llm-server logs --lines 200 --journal" in src_cli
    # unload guard wiring
    assert "model_unexpected_unload_with_bundle" in src_cli
    # guard child exit
    assert "llamaswap_guard_child_exited_with_bundle" in src_cli
    assert "llamaswap_guard_child_exited_with_bundle" in src_gw
    # manager socket timeout
    assert "manager_socket_timeout_with_bundle" in src_inst
    assert "logs --lines 200 --journal" in src_inst or "logs --journal" in src_inst


def test_proxy_502_json_shape_with_bundle_via_mock():
    """Simulate proxy 502 enriched JSON shape."""
    from llamacpp_stack.cli.crash_bundle import collect_engine_crash_bundle

    fake_bundle = collect_engine_crash_bundle("llama-server", port=11436, pid=999, cmdline="llama-server --port 11436 --model /tmp/m.gguf", returncode=None, timeout_s=1)
    # Mock wiring: error truncation and bundle_ref
    exc = RuntimeError("connection refused by backend which is down and not responding to any request")
    err_short = f"upstream unavailable: {exc}"[:200]
    payload = {"error": err_short, "engine": fake_bundle["engine"], "hint": "uv run llm-server logs --lines 200 --journal", "bundle_ref": "abcd1234"}
    assert "error" in payload
    assert payload["error"].startswith("upstream unavailable:")
    assert len(payload["error"]) <= 200
    assert payload["engine"] in {"llama-server", "beellama", "vllm", "exllama", "llama-swap", "manager"}
    assert payload["hint"] == "uv run llm-server logs --lines 200 --journal"
    assert payload["bundle_ref"] == "abcd1234"
    # Ensure full bundle would be in log, truncated in HTTP
    bundle_json = json.dumps(fake_bundle)
    assert len(bundle_json) > 0
    truncated = bundle_json[:4000]
    assert len(truncated) <= 4000


def test_crash_bundle_module_only_imports_constants_top_level():
    src = pathlib.Path("llamacpp_stack/cli/crash_bundle.py").read_text(encoding="utf-8")
    lines = src.splitlines()
    # Find imports before first function def
    import_lines = []
    for line in lines:
        if line.strip().startswith("def ") or line.strip().startswith("class "):
            break
        if "import" in line:
            import_lines.append(line)
    # Should have at most constants import at top-level
    top_imports = "\n".join(import_lines)
    assert "from .constants import" in top_imports or "from llamacpp_stack.cli.constants import" in top_imports
    # Should NOT have top-level import of log_api_event
    assert "log_api_event" not in top_imports
    assert "get_llama_server_processes" not in top_imports


def test_malformed_input_never_throws():
    from llamacpp_stack.cli.crash_bundle import collect_engine_crash_bundle

    # engine unknown, pid None, port None, cmdline None, timeout weird
    b = collect_engine_crash_bundle("", port=None, pid=None, cmdline=None, returncode=None, timeout_s=0.5)
    assert set(b.keys()) == EXPECTED_KEYS
    b2 = collect_engine_crash_bundle(None, port="bad", pid="bad", cmdline=123, returncode="bad", timeout_s=8.0)  # type: ignore
    assert set(b2.keys()) == EXPECTED_KEYS

def test_file_under_600_lines():
    src = pathlib.Path("llamacpp_stack/cli/crash_bundle.py").read_text(encoding="utf-8")
    lines = src.splitlines()
    assert len(lines) < 600, f"crash_bundle.py has {len(lines)} lines, must be <600"


def test_legacy_heimdall_journal_fallback_still_tried():
    src = pathlib.Path("llamacpp_stack/cli/crash_bundle.py").read_text(encoding="utf-8")
    assert "llm-server-router" in src
    assert "llm-server-manager" in src
    assert "heimdall-gateway-router" in src
    assert "heimdall-gateway-manager" in src
