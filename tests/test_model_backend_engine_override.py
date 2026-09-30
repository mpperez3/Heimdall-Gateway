"""Regression tests for ``_model_backend`` honouring an explicit engine.

A catalog model whose weights are a *directory* of sharded safetensors is
classified ``vllm`` by the directory heuristic, so the renderer emitted
``vllm-server``. When ``server_overrides.engine`` names a llama.cpp-family
engine, the explicit declaration must win.
"""

import pytest

from llamacpp_stack import _cli_impl as impl
from llamacpp_stack.cli.server_commands import ENGINE_DIR_NAMES


def _model(directory, *, backend="llama.cpp", overrides=None, filename="Qwen3.8-27B-EXL3"):
    return impl.ManagedModel(
        model_id="exl3-model",
        repo_id="org/exl3-model",
        quant=None,
        filename=filename,
        local_path=str(directory),
        backend=backend,
        server_overrides=overrides if isinstance(overrides, dict) else {},
    )


def _directory(tmp_path):
    directory = tmp_path / "exl3-model"
    directory.mkdir()
    return directory


@pytest.mark.parametrize("engine", ENGINE_DIR_NAMES)
def test_explicit_llama_cpp_engine_wins_over_directory_heuristic(tmp_path, engine):
    model = _model(_directory(tmp_path), backend="", overrides={"engine": engine})
    assert impl._normalize_model_backend(model.backend, model.filename, model.local_path) == "vllm"
    assert impl._model_backend(model) == "llama.cpp"


def test_engine_overrides_persisted_vllm_backend(tmp_path):
    model = _model(_directory(tmp_path), backend="vllm", overrides={"engine": "buun"})
    assert impl._normalize_model_backend(model.backend, model.filename, model.local_path) == "vllm"
    assert impl._model_backend(model) == "llama.cpp"


def test_engine_is_case_and_whitespace_insensitive(tmp_path):
    model = _model(_directory(tmp_path), backend="", overrides={"engine": " BUUN "})
    assert impl._normalize_model_backend(model.backend, model.filename, model.local_path) == "vllm"
    assert impl._model_backend(model) == "llama.cpp"


@pytest.mark.parametrize(
    "overrides",
    [
        None,
        {},
        {"engine": ""},
        {"engine": None},
        {"engine": "vllm"},
        {"engine": "totally-unknown"},
        "not-a-dict",
        ["buun"],
    ],
)
def test_non_llamacpp_engine_declarations_are_unchanged(tmp_path, overrides):
    model = _model(_directory(tmp_path), backend="")
    if overrides is not None:
        setattr(model, "server_overrides", overrides)
    assert impl._normalize_model_backend(model.backend, model.filename, model.local_path) == "vllm"
    assert impl._model_backend(model) == "vllm"


def test_missing_server_overrides_attribute_uses_normalize(tmp_path):
    model = _model(_directory(tmp_path), backend="")
    del model.server_overrides
    assert impl._model_backend(model) == "vllm"


def test_gguf_model_without_engine_is_unchanged(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")
    plain = _model(gguf, filename="model.gguf")
    assert impl._model_backend(plain) == "llama.cpp"
    assert impl._model_backend(_model(gguf, filename="model.gguf", overrides={"engine": "buun"})) == "llama.cpp"
