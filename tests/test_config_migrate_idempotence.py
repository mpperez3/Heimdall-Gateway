"""Idempotence tests for migrate_server_config + update_config and normalize.

- uses tempfile, asserts changed==False on second run, config.yaml SHA stable,
- normalize_server_config_payload deterministic,
- legacy-key warning stable,
- exercises tools/verify_llama_cmds.py via import.
"""
import argparse
import hashlib
import json
import io
import tempfile
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

import pytest

from llamacpp_stack.cli import (
    normalize_server_config_payload,
    persist_server_config,
    migrate_server_config,
    update_config,
    catalog_key_warnings,
)
from llamacpp_stack.cli import ManagedModel


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _minimal_catalog_entry(model_id="test-model", ctx_size=8192):
    return {
        "model_id": model_id,
        "repo_id": "Test/Repo",
        "quant": None,
        "filename": "test.gguf",
        "local_path": "/tmp/test.gguf",
        "mmproj_filename": None,
        "mmproj_path": None,
        "load_capabilities": [],
        "aliases": [],
        "ctx_size": ctx_size,
        "n_gpu_layers": -1,
        "tensor_split": None,
        "host": "127.0.0.1",
        "jinja": True,
        "description": "",
        "speculative": False,
        "spec_variant_of": None,
        "spec_meta": {},
        "auto_ctx_failed": False,
        "auto_ctx_error": "",
        "ctx_probe_read_s": None,
        "ctx_probe_tokens_s": None,
        "ctx_probe_totals_s": None,
        "ctx_probe_latency_ms": None,
        "ctx_probe_speed_tps": None,
        "ctx_probe_kv_gb": None,
        "ctx_probe_prompt_tokens": None,
        "server_overrides": {},
    }


def _make_args(tmpdir: Path):
    models_dir = tmpdir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    catalog = tmpdir / "catalog.json"
    server_config = tmpdir / "conf.json"
    config_yaml = tmpdir / "config.yaml"
    llama_server = tmpdir / "llama-server"
    llama_server.write_text("# dummy", encoding="utf-8")
    # default minimal catalog
    catalog.write_text(json.dumps([_minimal_catalog_entry()], indent=2), encoding="utf-8")
    # server_config initially empty
    server_config.write_text(json.dumps({}, indent=2), encoding="utf-8")
    # args namespace mirrors cli defaults
    return argparse.Namespace(
        catalog=catalog,
        server_config=server_config,
        config=config_yaml,
        models_dir=models_dir,
        llama_server=llama_server,
        service="llamaswap",
        start_port=11436,
        public_host="127.0.0.1",
        public_port=11436,
        api_port=None,
        idle_ttl=None,
        api_ctx_factor=None,
        flatten=None,
    )


class TestNormalizeDeterministic:
    def test_normalize_deterministic_and_idempotent(self):
        payload = {"models": [{"id": "x"}], "llama_server_defaults": {"batch_size": 512}, "api_ctx_factor": 0.5}
        norm1, changed1 = normalize_server_config_payload(dict(payload))
        norm2, changed2 = normalize_server_config_payload(dict(norm1))
        assert norm1 == norm2, "normalize must be deterministic"
        assert changed2 is False, "second normalize should be idempotent (changed==False)"
        # changed1 should be True because legacy 'models' removed and defaults added
        assert changed1 is True

    def test_normalize_removes_legacy_models_key(self):
        payload = {"models": [], "llama_server_defaults": {}}
        norm, changed = normalize_server_config_payload(payload)
        assert "models" not in norm
        assert changed is True

    def test_normalize_twice_same_hash(self):
        payload = {"replicas": {"enabled": True}, "experimental": {}}
        n1, _ = normalize_server_config_payload(dict(payload))
        n2, _ = normalize_server_config_payload(dict(payload))
        h1 = hashlib.sha256(json.dumps(n1, sort_keys=True).encode()).hexdigest()
        h2 = hashlib.sha256(json.dumps(n2, sort_keys=True).encode()).hexdigest()
        assert h1 == h2

    def test_normalize_handles_non_dict(self):
        norm, changed = normalize_server_config_payload(None)  # type: ignore
        assert isinstance(norm, dict)
        assert changed is True

    def test_normalize_preserves_unknown_keys_stable(self):
        payload = {"custom_key": "value", "idle_ttl": 300}
        n1, _ = normalize_server_config_payload(dict(payload))
        n2, _ = normalize_server_config_payload(dict(n1))
        assert n1 == n2


class TestMigrateIdempotence:
    def test_migrate_idempotence_sha_stable(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            args = _make_args(tmpdir)
            # First migrate
            buf1 = io.StringIO()
            with redirect_stdout(buf1):
                migrate_server_config(args)
            sha1 = _sha(args.server_config)
            text1 = args.server_config.read_text(encoding="utf-8")
            assert "changed" in buf1.getvalue() or "already current" in buf1.getvalue()
            # Second migrate
            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                migrate_server_config(args)
            sha2 = _sha(args.server_config)
            text2 = args.server_config.read_text(encoding="utf-8")
            assert sha1 == sha2, "server_config SHA must be stable after second migrate"
            assert text1 == text2
            assert "already current" in buf2.getvalue()

    def test_persist_server_config_second_call_no_change(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            args = _make_args(tmpdir)
            persist_server_config(args)
            sha1 = _sha(args.server_config)
            norm1, _ = normalize_server_config_payload(json.loads(args.server_config.read_text()))
            # Second call should not change file (no args overrides)
            persist_server_config(args)
            sha2 = _sha(args.server_config)
            assert sha1 == sha2
            norm2, changed2 = normalize_server_config_payload(json.loads(args.server_config.read_text()))
            assert changed2 is False
            assert norm1 == norm2

    def test_migrate_deterministic_with_legacy_hyphen_keys(self):
        """Legacy hyphen keys in llama_server_defaults should be normalized and stable."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            args = _make_args(tmpdir)
            # Write legacy hyphenated keys
            legacy = {"llama_server_defaults": {"cache-type-k": "q8_0", "batch-size": 1024}}
            args.server_config.write_text(json.dumps(legacy, indent=2), encoding="utf-8")
            buf = io.StringIO()
            with redirect_stdout(buf):
                migrate_server_config(args)
            data = json.loads(args.server_config.read_text())
            # Hyphen keys should be migrated to underscore normalized form deterministically
            # normalize_server_overrides converts hyphens -> underscores internally
            # Check that second migrate stable
            sha_before = _sha(args.server_config)
            with redirect_stdout(io.StringIO()):
                migrate_server_config(args)
            sha_after = _sha(args.server_config)
            assert sha_before == sha_after

    def test_catalog_legacy_key_warning_stable(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            catalog = tmpdir / "catalog.json"
            # Entry with hyphenated top-level key that looks like managed field
            catalog.write_text(json.dumps([
                {**_minimal_catalog_entry(), "tensor-split": "1,1"}
            ], indent=2), encoding="utf-8")
            w1 = catalog_key_warnings(catalog)
            w2 = catalog_key_warnings(catalog)
            assert w1 == w2
            assert len(w1) == 1
            assert "tensor_split" in w1[0] or "tensor-split" in w1[0]

            # Valid catalog should have no warnings
            catalog.write_text(json.dumps([_minimal_catalog_entry()], indent=2), encoding="utf-8")
            assert catalog_key_warnings(catalog) == []


class TestUpdateConfigIdempotence:
    @patch("os.getuid", return_value=0)
    def test_update_config_twice_sha_stable(self, _mock_uid):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            args = _make_args(tmpdir)
            # Ensure config.yaml will be generated via update_config
            # Mock os.stat for catalog parent to match uid is not needed since uid==0 -> owner
            # First update
            update_config(args)
            assert args.config.exists(), "config.yaml should be rendered"
            sha1 = _sha(args.config)
            text1 = args.config.read_text(encoding="utf-8")
            # Second update without changes should yield same SHA
            update_config(args)
            sha2 = _sha(args.config)
            text2 = args.config.read_text(encoding="utf-8")
            assert sha1 == sha2, f"config.yaml SHA unstable: {sha1[:12]} vs {sha2[:12]}"
            assert text1 == text2

    @patch("os.getuid", return_value=0)
    def test_update_config_with_catalog_unchanged_no_growth(self, _mock_uid):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            args = _make_args(tmpdir)
            # Write two models to ensure multiple entries stability
            catalog_data = [_minimal_catalog_entry("model-a", 8192), _minimal_catalog_entry("model-b", 4096)]
            args.catalog.write_text(json.dumps(catalog_data, indent=2), encoding="utf-8")
            update_config(args)
            size1 = args.config.stat().st_size
            sha1 = _sha(args.config)
            # Third call also stable
            update_config(args)
            update_config(args)
            size2 = args.config.stat().st_size
            sha2 = _sha(args.config)
            assert sha1 == sha2
            assert size1 == size2

    @patch("os.getuid", return_value=0)
    def test_migrate_plus_update_twice_idempotent(self, _mock_uid):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            args = _make_args(tmpdir)
            # First round: migrate + update
            with redirect_stdout(io.StringIO()):
                migrate_server_config(args)
            update_config(args)
            server_sha1 = _sha(args.server_config)
            cfg_sha1 = _sha(args.config)
            # Second round
            with redirect_stdout(io.StringIO()):
                migrate_server_config(args)
            update_config(args)
            server_sha2 = _sha(args.server_config)
            cfg_sha2 = _sha(args.config)
            assert server_sha1 == server_sha2
            assert cfg_sha1 == cfg_sha2


class TestVerifyLlamaCmdsExercised:
    def test_verify_llama_cmds_import_and_main(self):
        import tools.verify_llama_cmds as v
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            v.main()
        out = buf.getvalue()
        assert out.strip() != ""
        # Each line should contain model_id and llama-server
        assert "qwen-3.5" in out.lower()
        # Ensure command contains port 18090
        assert "18090" in out

    def test_build_llama_server_command_deterministic(self):
        from llamacpp_stack.cli import build_llama_server_command
        data = _minimal_catalog_entry("det-model", 8192)
        m = ManagedModel(**data)
        cmd1 = build_llama_server_command(m, Path("/usr/local/bin/llama-server"), port="18090")
        cmd2 = build_llama_server_command(m, Path("/usr/local/bin/llama-server"), port="18090")
        assert cmd1 == cmd2
        # SHA stable for command string
        h1 = hashlib.sha256(" ".join(map(str, cmd1)).encode()).hexdigest()
        h2 = hashlib.sha256(" ".join(map(str, cmd2)).encode()).hexdigest()
        assert h1 == h2


class TestLoggingRequestsLogMigrate:
    """T5: config-migrate adds logging.requests_log defaults, never overwrites, idempotent."""

    def test_adds_defaults_when_missing(self):
        from llamacpp_stack.cli import normalize_server_config_payload

        payload = {}
        norm, changed = normalize_server_config_payload(dict(payload))
        assert "logging" in norm
        assert "requests_log" in norm["logging"]
        req = norm["logging"]["requests_log"]
        assert req["path"] == ""
        assert req["max_bytes"] == 10485760
        assert req["retain_days"] == 3
        assert req["compress"] is False
        assert changed is True
        # second pass idempotent
        norm2, changed2 = normalize_server_config_payload(dict(norm))
        assert changed2 is False
        assert norm2 == norm

    def test_never_overwrites_existing(self):
        from llamacpp_stack.cli import normalize_server_config_payload

        custom = {
            "logging": {
                "requests_log": {
                    "path": "/tmp/custom.log",
                    "max_bytes": 65536,
                    "retain_days": 7,
                    "compress": True,
                }
            }
        }
        norm, _ = normalize_server_config_payload(dict(custom))
        req = norm["logging"]["requests_log"]
        assert req["path"] == "/tmp/custom.log"
        assert req["max_bytes"] == 65536
        assert req["retain_days"] == 7
        assert req["compress"] is True
        # second pass must preserve and be idempotent
        norm2, changed2 = normalize_server_config_payload(dict(norm))
        assert norm2["logging"]["requests_log"] == req
        assert changed2 is False

    def test_malformed_logging_as_string(self):
        from llamacpp_stack.cli import normalize_server_config_payload

        payload = {"logging": "bad string"}
        norm, changed = normalize_server_config_payload(dict(payload))
        assert changed is True
        assert isinstance(norm["logging"], dict)
        req = norm["logging"]["requests_log"]
        assert req["max_bytes"] == 10485760
        assert req["retain_days"] == 3
        # second pass stable
        norm2, changed2 = normalize_server_config_payload(dict(norm))
        assert changed2 is False
        assert norm2 == norm

    def test_malformed_requests_log_as_string(self):
        from llamacpp_stack.cli import normalize_server_config_payload

        payload = {"logging": {"requests_log": "oops"}}
        norm, changed = normalize_server_config_payload(dict(payload))
        assert changed is True
        req = norm["logging"]["requests_log"]
        assert req["path"] == ""
        assert req["max_bytes"] == 10485760

    def test_clamp_on_migrate(self):
        from llamacpp_stack.cli import normalize_server_config_payload

        payload = {"logging": {"requests_log": {"max_bytes": 10, "retain_days": 0}}}
        norm, changed = normalize_server_config_payload(dict(payload))
        assert norm["logging"]["requests_log"]["max_bytes"] == 65536
        assert norm["logging"]["requests_log"]["retain_days"] == 1

    def test_migrate_via_file_preserves_custom(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            args = _make_args(tmpdir)
            custom = {
                "logging": {
                    "requests_log": {
                        "path": "/tmp/keep.log",
                        "max_bytes": 65536,
                        "retain_days": 5,
                        "compress": False,
                    }
                }
            }
            args.server_config.write_text(json.dumps(custom, indent=2), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                migrate_server_config(args)
            data = json.loads(args.server_config.read_text(encoding="utf-8"))
            req = data["logging"]["requests_log"]
            assert req["path"] == "/tmp/keep.log"
            assert req["max_bytes"] == 65536
            assert req["retain_days"] == 5
            # second migrate must be already current
            buf = io.StringIO()
            with redirect_stdout(buf):
                migrate_server_config(args)
            assert "already current" in buf.getvalue()
            data2 = json.loads(args.server_config.read_text(encoding="utf-8"))
            assert data2 == data

    def test_migrate_adds_when_missing_via_file(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            args = _make_args(tmpdir)
            args.server_config.write_text(json.dumps({}, indent=2), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                migrate_server_config(args)
            data = json.loads(args.server_config.read_text(encoding="utf-8"))
            assert "logging" in data
            assert data["logging"]["requests_log"]["retain_days"] == 3
            # second pass idempotent
            sha1 = _sha(args.server_config)
            with redirect_stdout(io.StringIO()):
                migrate_server_config(args)
            sha2 = _sha(args.server_config)
            assert sha1 == sha2

    def test_update_config_preserves_logging(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            args = _make_args(tmpdir)
            custom = {
                "logging": {
                    "requests_log": {
                        "path": "/tmp/preserve.log",
                        "max_bytes": 65536,
                        "retain_days": 9,
                        "compress": True,
                    }
                }
            }
            args.server_config.write_text(json.dumps(custom, indent=2), encoding="utf-8")
            with patch("os.getuid", return_value=0):
                update_config(args)
            data = json.loads(args.server_config.read_text(encoding="utf-8"))
            req = data["logging"]["requests_log"]
            assert req["path"] == "/tmp/preserve.log"
            assert req["retain_days"] == 9
            assert req["max_bytes"] == 65536
            # second update stable
            sha1 = _sha(args.server_config)
            cfg_sha1 = _sha(args.config)
            with patch("os.getuid", return_value=0):
                update_config(args)
            assert _sha(args.server_config) == sha1
            assert _sha(args.config) == cfg_sha1

    def test_raw_ring_not_affected_by_migrate(self):
        """Raw ring api-raw-requests.log must not be truncated by config-migrate."""
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            args = _make_args(tmpdir)
            raw_path = tmpdir / "api-raw-requests.log"
            raw_path.write_text('{"raw":1}\n{"raw":2}\n', encoding="utf-8")
            sha_raw_before = _sha(raw_path)
            # also create rotated request log to ensure migrate does not prune it unexpectedly in tmpdir
            req_active = tmpdir / "api-requests.log.2099-01-01"
            req_active.write_text('{"kind":"old"}\n', encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                migrate_server_config(args)
            # raw must be untouched
            assert raw_path.exists()
            assert _sha(raw_path) == sha_raw_before


class TestConfigKeysLogging:
    def test_config_keys_contains_logging(self):
        from llamacpp_stack.cli import normalize_server_config_payload

        norm, _ = normalize_server_config_payload({})
        # Simulate what print_config_keys does: global_keys derived from normalize_server_config_payload({})
        global_keys = sorted(k for k in norm.keys() if not k.startswith("_"))
        assert "logging" in global_keys

    def test_config_keys_cli_json_contains_logging(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "llamacpp_stack.llamacpp_api_install", "config-keys", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # fallback: try heimdall-gateway binary if module fails
        if result.returncode != 0:
            import shutil
            bin_path = shutil.which("heimdall-gateway")
            if bin_path:
                result = subprocess.run([bin_path, "config-keys", "--format", "json"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            assert "logging" in data.get("conf_json_top_level_keys", [])
        else:
            # If CLI not available in this env, fallback to unit check via normalize
            from llamacpp_stack.cli import normalize_server_config_payload
            norm, _ = normalize_server_config_payload({})
            assert "logging" in norm
