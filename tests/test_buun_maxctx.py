import json
from pathlib import Path

import pytest


def test_buun_bench_exists():
    bench = Path(".omo/evidence/exl3-buun-maxctx-bench.json")
    best = Path(".omo/evidence/exl3-buun-maxctx-best.json")
    assert bench.exists(), "bench json missing — run helper --bench"
    assert best.exists(), "best json missing"
    data = json.loads(bench.read_text(encoding="utf-8"))
    assert "results" in data
    assert len(data["results"]) == 72
    ok = [r for r in data["results"] if r["load"] == "OK"]
    assert len(ok) >= 1
    assert data["summary"]["max_ctx_stable"] in [16384, 32768, 65536, 98304]
    assert data["summary"]["balanced_entry"]["predicted_per_second"] is not None


def test_buun_best_ranking():
    best = json.loads(Path(".omo/evidence/exl3-buun-maxctx-best.json").read_text(encoding="utf-8"))
    assert "MAX_CTX" in best and "BALANCED" in best
    assert best["MAX_CTX"]["ctx"] is not None
    assert best["BALANCED"]["predicted_per_second"] is not None
    assert "heimdall-gateway update" in best["MAX_CTX"]["update_cmd"]
    assert "heimdall-gateway update" in best["BALANCED"]["update_cmd"]
    assert best["MAX_CTX"]["vram_total_mib"] <= 24069
    assert "engine buun" in best["MAX_CTX"]["update_cmd"]


def test_buun_helper_bench_cli():
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "tests/fixtures/buun_maxctx_bench.py", "--bench"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    assert "wrote" in r.stdout
    assert "max_ctx" in r.stdout.lower()
