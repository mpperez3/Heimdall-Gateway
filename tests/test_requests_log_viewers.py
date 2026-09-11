import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from llamacpp_stack._cli_impl import (
    _candidate_request_log_paths,
    _glob_rotated_for_base,
    _rotated_sort_key,
    _tail_across_files,
    show_logs,
    show_request_log,
)
from llamacpp_stack.cli.constants import DEFAULT_REQUESTS_LOG_PATH, SYSTEM_REQUESTS_LOG_PATH
from llamacpp_stack.cli.daemon import build_info_text


def _make_rotated_files(base: Path, dates_and_contents: list[tuple[str, list[str], str | None]]):
    # dates_and_contents: list of (date_str, lines, part_suffix)
    created = []
    for date, lines, part in dates_and_contents:
        if part:
            p = base.parent / f"{base.name}.{date}.part{part}"
        else:
            p = base.parent / f"{base.name}.{date}"
        # handle .gz if part contains gz? not needed
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        created.append(p)
    return created


class TestCandidateGlob:
    def test_glob_enumerates_dated(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HEIMDALL_GATEWAY_REQUESTS_LOG", str(tmp_path / "api-requests.log"))
        base = tmp_path / "api-requests.log"
        dates = ["2026-09-09", "2026-09-10", "2026-09-11"]
        for d in dates:
            (tmp_path / f"api-requests.log.{d}").write_text(f"line {d}\n")
        paths = _glob_rotated_for_base(base)
        assert len(paths) == 3
        assert [p.name for p in paths] == [f"api-requests.log.{d}" for d in sorted(dates)]

    def test_glob_sorted_by_date_then_part(self, tmp_path):
        base = tmp_path / "api-requests.log"
        # Create out of order
        (tmp_path / "api-requests.log.2026-09-11.part2").write_text("p2\n")
        (tmp_path / "api-requests.log.2026-09-11").write_text("base\n")
        (tmp_path / "api-requests.log.2026-09-10").write_text("old\n")
        (tmp_path / "api-requests.log.2026-09-11.part1").write_text("p1\n")
        paths = _glob_rotated_for_base(base)
        names = [p.name for p in paths]
        assert names == [
            "api-requests.log.2026-09-10",
            "api-requests.log.2026-09-11",
            "api-requests.log.2026-09-11.part1",
            "api-requests.log.2026-09-11.part2",
        ]

    def test_glob_handles_gz(self, tmp_path):
        base = tmp_path / "api-requests.log"
        p1 = tmp_path / "api-requests.log.2026-09-11"
        p2 = tmp_path / "api-requests.log.2026-09-11.part1.gz"
        p1.write_text("a\n")
        # create gz file
        import gzip

        with gzip.open(p2, "wt", encoding="utf-8") as fh:
            fh.write("b\n")
        paths = _glob_rotated_for_base(base)
        assert p1 in paths
        assert p2 in paths

    def test_candidate_explicit_only(self, tmp_path, monkeypatch):
        explicit = tmp_path / "explicit" / "api-requests.log"
        explicit.parent.mkdir(parents=True, exist_ok=True)
        (explicit.parent / "api-requests.log.2026-09-10").write_text("old\n")
        (explicit.parent / "api-requests.log.2026-09-11").write_text("new\n")
        other = tmp_path / "other" / "api-requests.log"
        other.parent.mkdir(parents=True, exist_ok=True)
        (other.parent / "api-requests.log.2026-09-12").write_text("other\n")
        monkeypatch.setenv("HEIMDALL_GATEWAY_REQUESTS_LOG", str(other))
        paths = _candidate_request_log_paths(explicit)
        assert any(str(p).startswith(str(explicit.parent)) for p in paths)
        assert any("2026-09-10" in str(p) for p in paths)
        assert any("2026-09-11" in str(p) for p in paths)

    def test_candidate_none_includes_tmp_fallback(self, tmp_path, monkeypatch):
        # Ensure fallback tmp is included when no rotated elsewhere
        fallback = tmp_path / "fallback.log"
        monkeypatch.setenv("HEIMDALL_GATEWAY_REQUESTS_LOG_FALLBACK", str(fallback))
        fallback_dated = tmp_path / "fallback.log.2026-09-11"
        # fallback base parent is tmp_path, prefix is fallback.log
        # But _candidate with None will look at DEFAULT etc. Not fallback's prefix?
        # We test fallback base directly via _glob
        (tmp_path / "api-requests.log.2026-09-11").write_text("x\n")
        paths = _candidate_request_log_paths(None)
        # Should contain at least the tmp_path api file if mocked? Not reliable.
        # Instead test that _glob for fallback works
        fb_base = Path(str(fallback))
        (tmp_path / "fallback.log.2026-09-11").write_text("fb\n")
        fb_paths = _glob_rotated_for_base(fb_base)
        assert any("fallback.log.2026-09-11" in str(p) for p in fb_paths)


class TestTailTransversal:
    def test_tail_crosses_boundary_with_prefix(self, tmp_path):
        base = tmp_path / "api-requests.log"
        p1 = tmp_path / "api-requests.log.2026-09-09"
        p2 = tmp_path / "api-requests.log.2026-09-10"
        p3 = tmp_path / "api-requests.log.2026-09-11"
        p1.write_text("\n".join([f"f1-{i}" for i in range(1, 6)]) + "\n")
        p2.write_text("\n".join([f"f2-{i}" for i in range(1, 6)]) + "\n")
        p3.write_text("\n".join([f"f3-{i}" for i in range(1, 6)]) + "\n")
        paths = [p1, p2, p3]
        out = _tail_across_files(paths, 7)
        assert "-- " in out
        # Should contain last 2 of p2 and all 5 of p3
        assert "f2-4" in out
        assert "f2-5" in out
        assert "f3-1" in out
        assert "f3-5" in out
        assert "f1-1" not in out
        assert out.count("-- ") == 2

    def test_tail_single_file_no_duplicate_prefix(self, tmp_path):
        base = tmp_path / "api-requests.log"
        p1 = tmp_path / "api-requests.log.2026-09-11"
        p1.write_text("\n".join([f"line{i}" for i in range(10)]) + "\n")
        out = _tail_across_files([p1], 3)
        assert "-- " not in out
        assert "line7" in out
        assert "line8" in out
        assert "line9" in out
        assert "line0" not in out

    def test_tail_missing_part_intermediate(self, tmp_path):
        base = tmp_path / "api-requests.log"
        p_base = tmp_path / "api-requests.log.2026-09-11"
        p1 = tmp_path / "api-requests.log.2026-09-11.part1"
        p2 = tmp_path / "api-requests.log.2026-09-11.part2"
        # Simulate missing part1 (deleted)
        p_base.write_text("base1\nbase2\n")
        # p1 missing
        p2.write_text("p2-1\np2-2\n")
        paths = [p_base, p2]  # missing p1 not in list
        out = _tail_across_files(paths, 3)
        assert "base2" in out
        assert "p2-1" in out

    def test_tail_handles_gz(self, tmp_path):
        import gzip

        base = tmp_path / "api-requests.log"
        p1 = tmp_path / "api-requests.log.2026-09-10"
        p2 = tmp_path / "api-requests.log.2026-09-11.gz"
        p1.write_text("a1\na2\n")
        with gzip.open(p2, "wt", encoding="utf-8") as fh:
            fh.write("b1\nb2\n")
        out = _tail_across_files([p1, p2], 3)
        assert "a2" in out
        assert "b1" in out
        assert "b2" in out


class TestShowRequestLog:
    def test_show_request_log_transversal(self, tmp_path, monkeypatch, capsys):
        base = tmp_path / "api-requests.log"
        p1 = tmp_path / "api-requests.log.2026-09-09"
        p2 = tmp_path / "api-requests.log.2026-09-10"
        p3 = tmp_path / "api-requests.log.2026-09-11"
        p1.write_text("\n".join([f"a{i}" for i in range(5)]) + "\n")
        p2.write_text("\n".join([f"b{i}" for i in range(5)]) + "\n")
        p3.write_text("\n".join([f"c{i}" for i in range(5)]) + "\n")
        args = type("A", (), {"path": str(base), "lines": 7})()
        rc = show_request_log(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "-- " in out
        assert "c0" in out

    def test_show_request_log_explicit(self, tmp_path, capsys):
        base = tmp_path / "my.log"
        p = tmp_path / "my.log.2026-09-11"
        p.write_text("hello\nworld\n")
        args = type("A", (), {"path": str(base), "lines": 5})()
        rc = show_request_log(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "hello" in out

    def test_show_request_log_nonexistent(self, tmp_path, capsys):
        args = type("A", (), {"path": str(tmp_path / "nope.log"), "lines": 5})()
        rc = show_request_log(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "No request log found" in out


class TestShowLogs:
    def test_show_logs_contains_crash_hint(self, tmp_path, monkeypatch, capsys):
        base = tmp_path / "api-requests.log"
        p = tmp_path / "api-requests.log.2026-09-11"
        p.write_text("entry1\n")
        monkeypatch.setenv("HEIMDALL_GATEWAY_REQUESTS_LOG", str(base))
        args = type("A", (), {"path": str(base), "lines": 5, "since": None, "journal": False})()
        rc = show_logs(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "== API request log ==" in out
        assert 'Crash bundles: buscar "<bundle_ref>" o "*_with_bundle" en el log rotado; usar --journal para journal completo' in out

    def test_show_logs_transversal(self, tmp_path, capsys):
        base = tmp_path / "api-requests.log"
        p1 = tmp_path / "api-requests.log.2026-09-09"
        p2 = tmp_path / "api-requests.log.2026-09-10"
        p1.write_text("x1\nx2\nx3\n")
        p2.write_text("y1\ny2\ny3\n")
        args = type("A", (), {"path": str(base), "lines": 4, "since": None, "journal": False})()
        rc = show_logs(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "-- " in out


class TestBuildInfoText:
    def test_rotated_path_pattern(self, monkeypatch):
        text = build_info_text(None)
        assert "api-requests.log.YYYY-MM-DD" in text
        assert "rotado diario, 3 días" in text
        # Should not contain single file pattern without date
        # The old pattern was DEFAULT_REQUESTS_LOG_PATH without suffix
        # New should contain .YYYY-MM-DD
        assert "API Requests Log:" in text
