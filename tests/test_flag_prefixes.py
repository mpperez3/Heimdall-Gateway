import pytest

from pathlib import Path

from llamacpp_stack import cli


def test_fit_flags_single_dash():
    cmd = []
    cli._append_llama_server_flag(cmd, "fit", True)
    assert "-fit" in cmd
    assert "--fit" not in cmd

    cmd = []
    cli._append_llama_server_flag(cmd, "fitt", 2048)
    assert "-fitt" in cmd
    assert "--fitt" not in cmd

    cmd = []
    cli._append_llama_server_flag(cmd, "fitc", True)
    assert "-fitc" in cmd
    assert "--fitc" not in cmd


def test_normalize_server_overrides_preserves_fit_keys():
    normalized = cli.normalize_server_overrides({"fit": "off", "fitt": "1024", "fitc": "131072"})
    assert normalized["fit"] is False
    assert normalized["fitt"] == 1024
    assert normalized["fitc"] == 131072


def test_normalize_server_overrides_preserves_long_context_cache_keys():
    normalized = cli.normalize_server_overrides(
        {
            "cache-ram": "32768",
            "ctx-checkpoints": "32",
            "checkpoint-every-n-tokens": "1024",
            "chat-template-kwargs": {"preserve_thinking": True},
        }
    )

    assert normalized["cache_ram"] == 32768
    assert normalized["ctx_checkpoints"] == 32
    assert normalized["checkpoint_every_n_tokens"] == 1024
    assert normalized["chat_template_kwargs"] == '{"preserve_thinking":true}'


def test_build_command_emits_long_context_cache_flags(monkeypatch):
    monkeypatch.setattr(cli, "_server_supports_or_unknown", lambda _server_path, _flag: True)
    model = cli.ManagedModel(
        model_id="test",
        repo_id="test/repo",
        quant=None,
        filename="model.gguf",
        local_path="/tmp/model.gguf",
        ctx_size=4096,
        tensor_split="1",
        server_overrides={
            "cache_ram": 32768,
            "ctx_checkpoints": 32,
            "checkpoint_every_n_tokens": 1024,
            "chat_template_kwargs": {"preserve_thinking": True},
        },
    )

    cmd = cli.build_llama_server_command(model, Path("/bin/llama-server"), port="12345")
    joined = " ".join(cmd)

    assert "--cache-ram 32768" in joined
    assert "--ctx-checkpoints 32" in joined
    assert "--checkpoint-min-step 1024" in joined
    assert '--chat-template-kwargs {"preserve_thinking":true}' in joined


def test_qwen_mtp_vision_inherits_global_context_checkpoints(monkeypatch):
    monkeypatch.setattr(cli, "_server_supports_or_unknown", lambda _path, _flag: True)
    model = cli.ManagedModel(
        model_id="qwen3.6-27b-ud-q5_k_xl",
        repo_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        quant="UD-Q5_K_XL",
        filename="Qwen3.6-27B-UD-Q5_K_XL.gguf",
        local_path="/tmp/model.gguf",
        mmproj_path="/tmp/mmproj-BF16.gguf",
        load_capabilities=["image", "image-text-to-text"],
        ctx_size=262144,
        tensor_split="1,1",
        server_overrides={"spec_type": "draft-mtp"},
    )

    cmd = cli.build_llama_server_command(
        model, Path("/bin/llama-server"), port="12345",
        server_defaults={"ctx_checkpoints": 32, "cache_ram": 32768},
    )

    assert cmd[cmd.index("--ctx-checkpoints") + 1] == "32"
    assert cmd[cmd.index("--cache-ram") + 1] == "32768"


def test_qwen_mtp_vision_respects_explicit_checkpoint_override(monkeypatch):
    monkeypatch.setattr(cli, "_server_supports_or_unknown", lambda _path, _flag: True)
    model = cli.ManagedModel(
        model_id="qwen3.6-27b-ud-q5_k_xl",
        repo_id="unsloth/Qwen3.6-27B-MTP-GGUF",
        quant="UD-Q5_K_XL",
        filename="Qwen3.6-27B-UD-Q5_K_XL.gguf",
        local_path="/tmp/model.gguf",
        mmproj_path="/tmp/mmproj-BF16.gguf",
        load_capabilities=["image"],
        ctx_size=262144,
        tensor_split="1,1",
        server_overrides={
            "spec_type": "draft-mtp",
            "ctx_checkpoints": 32,
            "cache_ram": 32768,
        },
    )

    cmd = cli.build_llama_server_command(
        model, Path("/bin/llama-server"), port="12345",
        server_defaults={"ctx_checkpoints": 32, "cache_ram": 32768},
    )

    assert cmd[cmd.index("--ctx-checkpoints") + 1] == "32"
    assert cmd[cmd.index("--cache-ram") + 1] == "32768"


def test_normalize_server_config_migrates_new_defaults_and_removes_legacy_mirostat(monkeypatch):
    monkeypatch.setattr(cli, "detect_cuda_device_count", lambda: 2)

    normalized, changed = cli.normalize_server_config_payload(
        {
            "llama_server_defaults": {
                "mirostat": 2,
                "mirostat_ent": 4.5,
                "mirostat_lr": 0.1,
                "mtp_defaults": {"spec_draft_n_max": 2},
            }
        }
    )

    assert changed is True
    defaults = normalized["llama_server_defaults"]
    assert "mirostat" not in defaults
    assert "mirostat_ent" not in defaults
    assert "mirostat_lr" not in defaults
    assert defaults["cache_ram"] == 32768
    assert defaults["ctx_checkpoints"] == 32
    assert defaults["checkpoint_min_step"] == 1024
    assert "chat_template_kwargs" not in defaults
    assert defaults["top_k"] == 20
    assert defaults["top_p"] == 0.95
    assert defaults["min_p"] == 0.0
    assert defaults["repeat_penalty"] == 1.0
    assert defaults["presence_penalty"] == 0.0
    assert defaults["tensor_split"] == "1,1"
    assert defaults["mtp_defaults"]["spec_draft_n_max"] == 3




def test_normalize_server_config_keeps_managed_kv_cache_fp16_defaults(monkeypatch):
    monkeypatch.setattr(cli, "detect_cuda_device_count", lambda: 2)

    normalized, changed = cli.normalize_server_config_payload(
        {
            "llama_server_defaults": {
                "cache_type_k": "f16",
                "cache_type_v": "f16",
                "chat_template_kwargs": '{"preserve_thinking":true}',
                "mul_mat_q": True,
                "grp_attn_n": 16,
            }
        }
    )

    assert changed is True
    defaults = normalized["llama_server_defaults"]
    assert defaults["cache_type_k"] == "f16"
    assert defaults["cache_type_v"] == "f16"
    assert "chat_template_kwargs" not in defaults
    assert "mul_mat_q" not in defaults
    assert "grp_attn_n" not in defaults
    assert defaults["top_k"] == 20
    assert defaults["top_p"] == 0.95
    assert defaults["min_p"] == 0.0
    assert defaults["repeat_penalty"] == 1.0
    assert defaults["presence_penalty"] == 0.0


def test_build_command_emits_cache_type_for_gguf_defaults(monkeypatch):
    monkeypatch.setattr(cli, "_server_supports_or_unknown", lambda _server_path, _flag: True)
    model = cli.ManagedModel(
        model_id="test",
        repo_id="test/repo",
        quant="Q4_K_M",
        filename="model.gguf",
        local_path="/tmp/model.gguf",
        ctx_size=4096,
        tensor_split="1",
    )

    cmd = cli.build_llama_server_command(
        model,
        Path("/bin/llama-server"),
        port="12345",
        server_defaults={"cache_type_k": "f16", "cache_type_v": "f16"},
    )
    joined = " ".join(cmd)

    assert "--cache-type-k f16" in joined
    assert "--cache-type-v f16" in joined

def test_normalize_server_overrides_drops_tuning_internals_but_keeps_draft_layers_sentinel():
    normalized = cli.normalize_server_overrides({
        "gpu_set": [0, 1],
        "gpu_set_idx": 12,
        "main_gpu_raw": 2,
        "tensor_split_strategy": "equal",
        "ts_strategy": "even",
        "n_gpu_layers_draft": "all",
        "batch_size": 256,
    })

    assert normalized == {"n_gpu_layers_draft": "all", "batch_size": 256}


def test_normalize_server_overrides_drops_auto_performance_metadata():
    normalized = cli.normalize_server_overrides({
        "auto_performance": {"baselines": {"CORE:abc": {"score": 1.0}}},
        "batch_size": 512,
    })
    assert "auto_performance" not in normalized
    assert normalized["batch_size"] == 512


def test_load_catalog_preserves_auto_performance_metadata(tmp_path):
    catalog = tmp_path / "catalog.json"
    cache_payload = {"baselines": {"CORE:abc": {"role": "baseline", "score": 1.0}}}
    catalog.write_text(
        __import__("json").dumps([
            {
                "model_id": "m",
                "repo_id": "org/repo",
                "quant": None,
                "filename": "model.gguf",
                "local_path": "/tmp/model.gguf",
                "server_overrides": {
                    "auto_performance": cache_payload,
                    "batch_size": 512,
                },
            }
        ]),
        encoding="utf-8",
    )

    loaded = cli.load_catalog(catalog)

    assert loaded[0].server_overrides["auto_performance"] == cache_payload
    assert "auto_performance" in __import__("json").loads(catalog.read_text("utf-8"))[0]["server_overrides"]


def test_normalize_server_overrides_omits_numa_when_none():
    # numa=None should be omitted from normalized overrides (not converted to string "None")
    normalized = cli.normalize_server_overrides({"numa": None})
    assert "numa" not in normalized
    
    # numa with string "none" should also be omitted
    normalized = cli.normalize_server_overrides({"numa": "none"})
    assert "numa" not in normalized
    
    # numa with valid values should be preserved
    normalized = cli.normalize_server_overrides({"numa": "distribute"})
    assert normalized["numa"] == "distribute"
    
    normalized = cli.normalize_server_overrides({"numa": "isolate"})
    assert normalized["numa"] == "isolate"


def test_normalize_server_overrides_preserves_direct_io_and_emits_flags():
    normalized = cli.normalize_server_overrides({"direct_io": True})
    assert normalized["direct_io"] is True

    model = cli.ManagedModel(
        model_id="test",
        repo_id="test/repo",
        quant=None,
        filename="model.gguf",
        local_path="/tmp/model.gguf",
        mmproj_filename=None,
        mmproj_path=None,
        load_capabilities=[],
        aliases=[],
        ctx_size=4096,
        n_gpu_layers=999,
        tensor_split="1",
        host="127.0.0.1",
        jinja=False,
        description="",
        speculative=False,
        spec_variant_of=None,
        spec_meta={},
        auto_ctx_failed=False,
        auto_ctx_error="",
        ctx_probe_read_s=None,
        ctx_probe_tokens_s=None,
        ctx_probe_totals_s=None,
        ctx_probe_latency_ms=None,
        ctx_probe_speed_tps=None,
        ctx_probe_kv_gb=None,
        ctx_probe_prompt_tokens=None,
        server_overrides={"direct_io": True},
    )

    cmd = cli.build_llama_server_command(model, Path("/tmp/llama-server"), port="18090")
    assert "--direct-io" in cmd


def test_build_llama_server_command_clamps_unsupported_ubatch_size():
    model = cli.ManagedModel(
        model_id="test",
        repo_id="test/repo",
        quant=None,
        filename="model.gguf",
        local_path="/tmp/model.gguf",
        mmproj_filename=None,
        mmproj_path=None,
        load_capabilities=[],
        aliases=[],
        ctx_size=4096,
        n_gpu_layers=999,
        tensor_split="1",
        host="127.0.0.1",
        jinja=False,
        description="",
        speculative=False,
        spec_variant_of=None,
        spec_meta={},
        auto_ctx_failed=False,
        auto_ctx_error="",
        ctx_probe_read_s=None,
        ctx_probe_tokens_s=None,
        ctx_probe_totals_s=None,
        ctx_probe_latency_ms=None,
        ctx_probe_speed_tps=None,
        ctx_probe_kv_gb=None,
        ctx_probe_prompt_tokens=None,
        server_overrides={
            "batch_size": 1024,
            "ubatch_size": 128,
        },
    )

    cmd = cli.build_llama_server_command(model, Path("/tmp/llama-server"), port="18090")
    ubatch_idx = cmd.index("--ubatch-size")
    assert cmd[ubatch_idx + 1] == "256"


def test_emitted_flags_are_supported_by_local_llama_server():
    server_bin = Path(__file__).resolve().parents[1] / "llama.cpp-source" / "build" / "bin" / "llama-server"
    if not server_bin.exists():
        pytest.skip("local llama-server binary not available")

    supported = cli.get_server_supported_flags(server_bin)
    assert supported, "expected help output to expose supported flags"

    model = cli.ManagedModel(
        model_id="test",
        repo_id="test/repo",
        quant=None,
        filename="model.gguf",
        local_path="/tmp/model.gguf",
        mmproj_filename=None,
        mmproj_path=None,
        load_capabilities=[],
        aliases=[],
        ctx_size=4096,
        n_gpu_layers=999,
        tensor_split="1,1",
        host="127.0.0.1",
        jinja=False,
        description="",
        speculative=False,
        spec_variant_of=None,
        spec_meta={},
        auto_ctx_failed=False,
        auto_ctx_error="",
        ctx_probe_read_s=None,
        ctx_probe_tokens_s=None,
        ctx_probe_totals_s=None,
        ctx_probe_latency_ms=None,
        ctx_probe_speed_tps=None,
        ctx_probe_kv_gb=None,
        ctx_probe_prompt_tokens=None,
        server_overrides={
            "direct_io": True,
            "fit": False,
            "batch_size": 1024,
            "ubatch_size": 256,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "kv_offload": True,
            "op_offload": False,
            "cont_batching": True,
        },
    )

    cmd = cli.build_llama_server_command(model, server_bin, port="18090")
    emitted_flags = {token for token in cmd if token.startswith("-")}
    assert emitted_flags <= supported


def test_chat_template_kwargs_normalizes_config_style_booleans():
    from llamacpp_stack.cli import normalize_server_overrides

    normalized = normalize_server_overrides({"chat_template_kwargs": '{"preserve_thinking":off}'})

    assert normalized["chat_template_kwargs"] == '{"preserve_thinking":false}'


def test_normalize_server_overrides_normalizes_cache_type_aliases():
    normalized = cli.normalize_server_overrides({"cache_type_k": "Q8", "cache_type_v": "fp16"})

    assert normalized["cache_type_k"] == "q8_0"
    assert normalized["cache_type_v"] == "f16"


def test_normalize_server_overrides_omits_invalid_cache_type_and_reports_warning():
    normalized = cli.normalize_server_overrides({"cache_type_k": "banana", "cache_type_v": "q8_0"})
    warnings = cli._server_config_validation_warnings(
        {"llama_server_defaults": {"cache_type_k": "banana", "cache_type_v": "Q8"}}
    )

    assert "cache_type_k" not in normalized
    assert normalized["cache_type_v"] == "q8_0"
    assert any("Invalid llama_server_defaults.cache_type_k" in warning for warning in warnings)
    assert any("normalized to 'q8_0'" in warning for warning in warnings)
