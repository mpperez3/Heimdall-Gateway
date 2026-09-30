import contextlib
import io
import shlex
from pathlib import Path
from unittest import mock

import pytest
import yaml

from llamacpp_stack import _cli_impl, cli
from llamacpp_stack.cli import replica as replica_mod
from llamacpp_stack.cli.server_commands import ENGINE_DIR_NAMES, _engine_binary_anchor

ROOT = "/home/x/.local/opt/llm-server"


def test_anchor_is_parent_for_top_level_launcher():
    assert _engine_binary_anchor(Path(ROOT) / "llama-server") == Path(ROOT)


@pytest.mark.parametrize("engine", ENGINE_DIR_NAMES)
def test_anchor_climbs_to_root_for_engine_specific_binary(engine):
    server_path = Path(ROOT) / engine / "bin" / f"llama-server-{engine}"
    assert _engine_binary_anchor(server_path) == Path(ROOT)


def test_anchor_is_parent_for_unrelated_path():
    server_path = Path("/usr/local/bin/llama-server")
    assert _engine_binary_anchor(server_path) == Path("/usr/local/bin")


def test_anchor_ignores_bin_without_engine_dir_name():
    server_path = Path("/opt/llama.cpp/build/bin/llama-server")
    assert _engine_binary_anchor(server_path) == Path("/opt/llama.cpp/build/bin")


def _model(engine):
    return cli.ManagedModel(
        model_id="test",
        repo_id="local/Qwen3.8-27B-EXL3-3.5bpw",
        quant=None,
        filename="model.safetensors",
        local_path="/models/Qwen3.8-27B-EXL3-3.5bpw",
        ctx_size=8192,
        tensor_split="",
        server_overrides={"engine": engine},
    )


def test_command_resolves_engine_bin_from_top_level_launcher(monkeypatch):
    monkeypatch.setattr(cli, "_server_supports_or_unknown", lambda _server_path, _flag: True)
    cmd = cli.build_llama_server_command(
        _model("buun"), Path(ROOT) / "llama-server", port="12345"
    )
    assert cmd[0] == f"{ROOT}/buun/bin/llama-server-buun"


def test_command_resolves_engine_bin_from_engine_specific_path(monkeypatch):
    """Regression: anchoring on `parent` produced <root>/beellama/bin/buun/bin/..."""
    monkeypatch.setattr(cli, "_server_supports_or_unknown", lambda _server_path, _flag: True)
    cmd = cli.build_llama_server_command(
        _model("buun"),
        Path(ROOT) / "beellama" / "bin" / "llama-server-beellama",
        port="12345",
    )
    assert cmd[0] == f"{ROOT}/buun/bin/llama-server-buun"


def test_command_keeps_engine_specific_path_untouched(monkeypatch):
    monkeypatch.setattr(cli, "_server_supports_or_unknown", lambda _server_path, _flag: True)
    cmd = cli.build_llama_server_command(
        _model("beellama"),
        Path(ROOT) / "beellama" / "bin" / "llama-server-beellama",
        port="12345",
    )
    assert cmd[0] == f"{ROOT}/beellama/bin/llama-server-beellama"


def test_command_without_engine_uses_server_path_as_is(monkeypatch):
    monkeypatch.setattr(cli, "_server_supports_or_unknown", lambda _server_path, _flag: True)
    cmd = cli.build_llama_server_command(
        cli.ManagedModel(
            model_id="plain",
            repo_id="local/gguf",
            quant=None,
            filename="model.gguf",
            local_path="/models/model.gguf",
            ctx_size=4096,
            tensor_split="",
        ),
        Path(ROOT) / "llama-server",
        port="12345",
    )
    assert cmd[0] == f"{ROOT}/llama-server"


def _executable_token(cmd: str) -> str:
    parts = shlex.split(cmd)
    idx = 1 if parts and parts[0].endswith("/env") else 0
    while idx < len(parts) and "=" in parts[idx] and not parts[idx].startswith("-"):
        idx += 1
    return parts[idx]


def _engine_model(engine: str = "buun"):
    return cli.ManagedModel(
        model_id="qwen3.8-27b-exl3-3.5bpw",
        repo_id="local/Qwen3.8-27B-EXL3-3.5bpw",
        quant=None,
        filename="model.safetensors",
        local_path="/models/Qwen3.8-27B-EXL3-3.5bpw",
        tensor_split="1",
        server_overrides={"engine": engine},
    )


def _install_tree(root: Path) -> None:
    for engine in ENGINE_DIR_NAMES:
        bindir = root / engine / "bin"
        bindir.mkdir(parents=True, exist_ok=True)
        (bindir / f"llama-server-{engine}").write_text("#!/bin/sh\n", encoding="utf-8")
    (root / "llama-server").write_text("#!/bin/sh\n", encoding="utf-8")


def _server_path_for(root: Path, shape: str) -> Path:
    if shape == "top-level":
        return root / "llama-server"
    return root / "beellama" / "bin" / "llama-server-beellama"


def _run_site(mod, kind: str, patch_target: str, tmp_path: Path, server_path: Path) -> tuple[str, str]:
    config = tmp_path / "config.yaml"
    model = _engine_model()
    buf = io.StringIO()
    with mock.patch(patch_target, return_value=1), contextlib.redirect_stdout(buf):
        if kind == "render":
            mod.render_llamaswap_config([model], config, server_path, 18080, idle_ttl=10)
        else:
            mod.ensure_replica_route_in_llamaswap_config(
                model, 0, [0], [model], config, server_path, 10
            )
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    entry = next(iter(data["models"].values()))
    return entry["cmd"], buf.getvalue()


SITES = [
    (
        "_cli_impl.render_llamaswap_config",
        _cli_impl,
        "render",
        "llamacpp_stack._cli_impl.detect_cuda_device_count",
    ),
    (
        "_cli_impl.ensure_replica_route_in_llamaswap_config",
        _cli_impl,
        "ensure",
        "llamacpp_stack._cli_impl.detect_cuda_device_count",
    ),
    (
        "cli.replica.render_llamaswap_config",
        replica_mod,
        "render",
        "llamacpp_stack.cli.detect_cuda_device_count",
    ),
    (
        "cli.replica.ensure_replica_route_in_llamaswap_config",
        replica_mod,
        "ensure",
        "llamacpp_stack.cli.detect_cuda_device_count",
    ),
]
SITE_IDS = [site[0] for site in SITES]
SHAPES = ["top-level", "engine-specific"]


@pytest.mark.parametrize("label,mod,kind,patch_target", SITES, ids=SITE_IDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_site_anchors_engine_binary(label, mod, kind, patch_target, shape, tmp_path):
    """Regression: every engine-bin site must anchor on the install root, not on `parent`.

    With the naive `parent` base the candidate was `<root>/beellama/bin/buun/bin/..`,
    which never exists, so the `.exists()` guard silently missed and the site logged
    nothing. Assert the site itself resolves and adopts the anchored path.
    """
    root = tmp_path / "llm-server"
    _install_tree(root)
    cmd, out = _run_site(mod, kind, patch_target, tmp_path, _server_path_for(root, shape))
    exe = _executable_token(cmd)
    assert exe == str(root / "buun" / "bin" / "llama-server-buun")
    assert "beellama/bin/buun" not in exe
    assert "[buun]" in out
    assert str(root / "buun" / "bin" / "llama-server-buun") in out
    assert "beellama/bin/buun" not in out


@pytest.mark.parametrize("label,mod,kind,patch_target", SITES, ids=SITE_IDS)
@pytest.mark.parametrize("shape", SHAPES)
def test_site_falls_back_when_engine_binary_missing(
    label, mod, kind, patch_target, shape, tmp_path
):
    """The `.exists()` guard must miss, fall back to server_path, and not raise."""
    root = tmp_path / "llm-server"
    root.mkdir(parents=True, exist_ok=True)
    (root / "llama-server").write_text("#!/bin/sh\n", encoding="utf-8")
    cmd, out = _run_site(mod, kind, patch_target, tmp_path, _server_path_for(root, shape))
    assert "[buun]" not in out
    assert "beellama/bin/buun" not in _executable_token(cmd)


def test_engine_binary_base_shims_delegate_to_shared_helper(tmp_path):
    root = tmp_path / "llm-server"
    engine_specific = root / "beellama" / "bin" / "llama-server-beellama"
    top_level = root / "llama-server"
    for shim in (_cli_impl._engine_binary_base, replica_mod._engine_binary_base):
        assert shim(engine_specific) == root
        assert shim(top_level) == root
        assert shim(engine_specific) == _engine_binary_anchor(engine_specific)
