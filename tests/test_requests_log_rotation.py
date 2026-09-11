import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from llamacpp_stack._cli_impl import (
    _daily_active_path_for,
    _default_requests_log_config,
    _effective_requests_log_config,
    _normalize_requests_log_config,
    log_api_event,
)
from llamacpp_stack.cli.constants import DEFAULT_REQUESTS_LOG_PATH


def _active_for(base: Path) -> Path:
    return _daily_active_path_for(base)


class TestNormalizeRequestsLog:
    def test_defaults(self):
        cfg, changed = _normalize_requests_log_config(None)
        assert cfg["path"] == ""
        assert cfg["max_bytes"] == 10485760
        assert cfg["retain_days"] == 3
        assert cfg["compress"] is False

    def test_clamp_max_bytes(self):
        cfg, _ = _normalize_requests_log_config({"max_bytes": 10})
        assert cfg["max_bytes"] == 65536
        cfg2, _ = _normalize_requests_log_config({"max_bytes": 9999999999})
        assert cfg2["max_bytes"] == 1073741824
        cfg3, _ = _normalize_requests_log_config({"max_bytes": "65536"})
        assert cfg3["max_bytes"] == 65536
        cfg4, _ = _normalize_requests_log_config({"max_bytes": -100})
        assert cfg4["max_bytes"] == 65536

    def test_clamp_retain_days(self):
        cfg, _ = _normalize_requests_log_config({"retain_days": 0})
        assert cfg["retain_days"] == 1
        cfg2, _ = _normalize_requests_log_config({"retain_days": 100})
        assert cfg2["retain_days"] == 30
        cfg3, _ = _normalize_requests_log_config({"retain_days": "0"})
        assert cfg3["retain_days"] == 1
        cfg4, _ = _normalize_requests_log_config({"retain_days": "-5"})
        assert cfg4["retain_days"] == 1

    def test_compress_bool(self):
        cfg, _ = _normalize_requests_log_config({"compress": True})
        assert cfg["compress"] is True
        cfg2, _ = _normalize_requests_log_config({"compress": "true"})
        assert cfg2["compress"] is True
        cfg3, _ = _normalize_requests_log_config({"compress": 1})
        assert cfg3["compress"] is True

    def test_path_trim(self):
        cfg, _ = _normalize_requests_log_config({"path": "  /tmp/foo.log  "})
        assert cfg["path"] == "/tmp/foo.log"
        cfg2, _ = _normalize_requests_log_config({"path": ""})
        assert cfg2["path"] == ""

    def test_env_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HEIMDALL_GATEWAY_REQUESTS_LOG_MAX_BYTES", "65536")
        monkeypatch.setenv("HEIMDALL_GATEWAY_REQUESTS_LOG_RETAIN_DAYS", "1")
        cfg = _effective_requests_log_config()
        assert cfg["max_bytes"] == 65536
        assert cfg["retain_days"] == 1
        monkeypatch.delenv("HEIMDALL_GATEWAY_REQUESTS_LOG_MAX_BYTES", raising=False)
        monkeypatch.delenv("HEIMDALL_GATEWAY_REQUESTS_LOG_RETAIN_DAYS", raising=False)
        # clamp via env
        monkeypatch.setenv("HEIMDALL_GATEWAY_REQUESTS_LOG_MAX_BYTES", "10")
        cfg2 = _effective_requests_log_config()
        assert cfg2["max_bytes"] == 65536
        monkeypatch.setenv("HEIMDALL_GATEWAY_REQUESTS_LOG_RETAIN_DAYS", "0")
        cfg3 = _effective_requests_log_config()
        assert cfg3["retain_days"] == 1


class TestDailyRotation:
    def test_creates_dated_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HEIMDALL_GATEWAY_REQUESTS_LOG_PATH", raising=False)
        monkeypatch.delenv("HEIMDALL_GATEWAY_REQUESTS_LOG_FALLBACK", raising=False)
        base = tmp_path / "api-requests.log"
        log_api_event("rotation_test", {"x": 1}, log_path=base)
        active = _active_for(base)
        assert active.exists()
        # legacy symlink should exist and point to active
        assert base.exists()
        # active should contain JSONL
        lines = active.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1
        entry = json.loads(lines[-1])
        assert entry["kind"] == "rotation_test"
        # fallback dated should be same logic but not needed here
        # Ensure no direct write without date after change except via symlink
        # The legacy file should be symlink or copy with same content
        assert base.read_text(encoding="utf-8").strip().splitlines()[-1] == lines[-1]

    def test_custom_basename(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HEIMDALL_GATEWAY_REQUESTS_LOG_PATH", raising=False)
        base = tmp_path / "my-custom.log"
        log_api_event("custom", {"v": 1}, log_path=base)
        active = _active_for(base)
        assert active.exists()
        assert active.name == "my-custom.log." + active.name.split(".")[-3] + "-" + active.name.split(".")[-2] + "-" + active.name.split(".")[-1] or active.name.startswith("my-custom.log.")
        assert "my-custom.log." in active.name


class TestSplitPart:
    def test_split_when_exceeds_max_bytes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HEIMDALL_GATEWAY_REQUESTS_LOG_MAX_BYTES", "65536")
        monkeypatch.setenv("HEIMDALL_GATEWAY_REQUESTS_LOG_RETAIN_DAYS", "3")
        base = tmp_path / "api-requests.log"
        active = _active_for(base)
        # pre-fill active with size ~ 65400
        active.parent.mkdir(parents=True, exist_ok=True)
        filler = "x" * 65000
        active.write_text(filler + "\n", encoding="utf-8")
        sz_before = active.stat().st_size
        assert sz_before < 65536
        assert sz_before > 60000
        # next log line will exceed 65536, should go to part1
        log_api_event("split_test", {"data": "y" * 500}, log_path=base)
        part1 = Path(str(active) + ".part1")
        # either active not grown beyond max, or part created
        assert part1.exists() or active.stat().st_size <= 65536
        if part1.exists():
            content = part1.read_text(encoding="utf-8")
            assert "split_test" in content
        else:
            # if not part, then active should contain split_test and size limited
            content = active.read_text(encoding="utf-8")
            assert "split_test" in content
        # write another large that exceeds part1 too -> part2
        # fill part1 to near limit if exists
        if part1.exists():
            # make part1 also full
            part1.write_text("z" * 65000 + "\n", encoding="utf-8")
            log_api_event("split_test2", {"data": "y" * 500}, log_path=base)
            part2 = Path(str(active) + ".part2")
            # should create part2
            assert part2.exists() or part1.stat().st_size <= 65536


class TestPrune:
    def test_prune_keeps_3_days(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HEIMDALL_GATEWAY_REQUESTS_LOG_PATH", raising=False)
        monkeypatch.setenv("HEIMDALL_GATEWAY_REQUESTS_LOG_RETAIN_DAYS", "3")
        monkeypatch.setenv("HEIMDALL_GATEWAY_REQUESTS_LOG_MAX_BYTES", "10485760")
        base = tmp_path / "api-requests.log"
        base.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        # create 5 fake dated files with old mtimes
        for days_ago in [0, 1, 2, 3, 4, 5]:
            import datetime
            d = (datetime.datetime.now().astimezone() - datetime.timedelta(days=days_ago)).date().isoformat()
            p = tmp_path / f"api-requests.log.{d}"
            p.write_text(f"old {days_ago}\n", encoding="utf-8")
            if days_ago <= 3:
                mtime = now - days_ago * 86400 + 3600
            else:
                mtime = now - days_ago * 86400 - 100
            os.utime(p, (mtime, mtime))
            if days_ago == 5:
                part = Path(str(p) + ".part1")
                part.write_text("part old\n", encoding="utf-8")
                os.utime(part, (mtime, mtime))
        # trigger prune via log_api_event
        log_api_event("prune_trigger", {"x": 1}, log_path=base)
        # after prune, files older than 3 days (4,5) should be gone
        remaining = list(tmp_path.glob("api-requests.log.*"))
        # check that oldest (5 days ago) is pruned
        import datetime
        d5 = (datetime.datetime.now().astimezone() - datetime.timedelta(days=5)).date().isoformat()
        assert not (tmp_path / f"api-requests.log.{d5}").exists()
        assert not (tmp_path / f"api-requests.log.{d5}.part1").exists()
        d4 = (datetime.datetime.now().astimezone() - datetime.timedelta(days=4)).date().isoformat()
        assert not (tmp_path / f"api-requests.log.{d4}").exists()
        # 0-3 days should remain
        for days_ago in [0, 1, 2, 3]:
            d = (datetime.datetime.now().astimezone() - datetime.timedelta(days=days_ago)).date().isoformat()
            p = tmp_path / f"api-requests.log.{d}"
            # 0 days is today's active, should exist
            if days_ago == 0:
                assert p.exists()
            else:
                # prune retains up to 3 days, so 3 should still exist
                assert p.exists(), f"{p} should remain"

    def test_retain_days_clamp_zero(self, tmp_path, monkeypatch):
        # retain_days=0 should clamp to 1, so prune with 0 should behave like 1
        # If clamp broken, prune would delete everything older than 0 days (all)
        monkeypatch.setenv("HEIMDALL_GATEWAY_REQUESTS_LOG_RETAIN_DAYS", "0")
        cfg = _effective_requests_log_config()
        assert cfg["retain_days"] == 1
        # also test normalize directly
        cfg2, _ = _normalize_requests_log_config({"retain_days": 0})
        assert cfg2["retain_days"] == 1

    def test_prune_not_break_write(self, tmp_path, monkeypatch):
        # ensure prune failure does not break write (never throws)
        base = tmp_path / "api-requests.log"
        # make directory read-only to cause prune unlink failure? Instead patch unlink to raise
        with patch.object(Path, "unlink", side_effect=OSError("mock unlink fail")):
            log_api_event("no_throw", {"a": 1}, log_path=base)
            active = _active_for(base)
            assert active.exists()


class TestFallbackAndNeverThrows:
    def test_fallback_when_primary_not_writable(self, tmp_path, monkeypatch):
        primary = tmp_path / "primary" / "api-requests.log"
        fallback = tmp_path / "fallback" / "api-requests.log"
        monkeypatch.setenv("HEIMDALL_GATEWAY_REQUESTS_LOG_FALLBACK", str(fallback))
        original_open = Path.open

        def fake_open(self, *args, **kwargs):
            if self == primary:
                raise OSError("mock primary not writable")
            return original_open(self, *args, **kwargs)

        with patch.object(Path, "open", fake_open):
            log_api_event("fallback_test", {"x": 123}, log_path=primary)
        # fallback dated should exist, primary should not (except maybe symlink not created)
        assert not primary.exists() or primary.is_symlink() is False or not primary.resolve().exists()
        # check fallback dated
        fb_active = _active_for(fallback)
        assert fb_active.exists()
        lines = fb_active.read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[-1])
        assert entry["kind"] == "fallback_test"

    def test_never_throws(self, tmp_path, monkeypatch):
        primary = tmp_path / "p.log"
        fallback = tmp_path / "f.log"
        monkeypatch.setenv("HEIMDALL_GATEWAY_REQUESTS_LOG_FALLBACK", str(fallback))
        with patch.object(Path, "open", side_effect=OSError("all fail")):
            log_api_event("no_crash", {"z": 1}, log_path=primary)
        # should not raise, and not create files (or empty)
        # just ensure no exception propagated

    def test_env_path_precedence(self, tmp_path, monkeypatch):
        # env path should take precedence over conf path
        env_path = tmp_path / "env.log"
        monkeypatch.setenv("HEIMDALL_GATEWAY_REQUESTS_LOG_PATH", str(env_path))
        # simulate conf path via mocking _load_server_config_payload to return logging path
        fake_conf_path = tmp_path / "conf.log"
        with patch("llamacpp_stack._cli_impl._load_server_config_payload", return_value={"logging": {"requests_log": {"path": str(fake_conf_path), "max_bytes": 10485760, "retain_days": 3, "compress": False}}}):
            # calling without explicit should use env_path, not conf
            log_api_event("env_precedence", {"a": 1}, log_path=None)
            env_active = _active_for(env_path)
            conf_active = _active_for(fake_conf_path)
            assert env_active.exists()
            # conf should not have been written
            assert not conf_active.exists()
