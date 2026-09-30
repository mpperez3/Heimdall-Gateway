"""Regression tests for ``_normalize_model_backend`` directory handling.

A sharded/safetensors model is stored as a *directory* on disk. The explicit
``backend`` field must win over the directory heuristic, otherwise an EXL3
model registered with ``backend: "llama.cpp"`` renders as ``vllm-server``.
"""

from llamacpp_stack import _cli_impl as impl


def test_explicit_llama_cpp_backend_survives_directory_local_path(tmp_path):
    directory = tmp_path / "Qwen3.8-27B-EXL3-3.5bpw"
    directory.mkdir()
    assert (
        impl._normalize_model_backend("llama.cpp", "Qwen3.8-27B-EXL3-3.5bpw", str(directory))
        == "llama.cpp"
    )


def test_unspecified_backend_on_directory_still_falls_back_to_vllm(tmp_path):
    directory = tmp_path / "hf-native-repo"
    directory.mkdir()
    assert impl._normalize_model_backend("", "some-model", str(directory)) == "vllm"


def test_explicit_vllm_backend_on_directory_is_vllm(tmp_path):
    directory = tmp_path / "unsloth-qwen-nvfp4"
    directory.mkdir()
    assert impl._normalize_model_backend("vllm", "hf-native", str(directory)) == "vllm"


def test_engine_names_win_on_directory(tmp_path):
    directory = tmp_path / "exl3-model"
    directory.mkdir()
    assert impl._normalize_model_backend("buun", "m", str(directory)) == "buun"
    assert impl._normalize_model_backend("exllama", "m", str(directory)) == "exllama"
    assert impl._normalize_model_backend("hf-native", "hf-native", str(directory)) == "vllm"


def test_gguf_file_and_unknown_value_fallbacks_unchanged(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"gguf")
    assert impl._normalize_model_backend("llama.cpp", "model.gguf", str(gguf)) == "llama.cpp"
    assert impl._normalize_model_backend("totally-unknown", "m", str(gguf)) == "llama.cpp"
