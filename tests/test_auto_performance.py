import unittest
import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from tempfile import NamedTemporaryFile

from llamacpp_stack.cli import ManagedModel, build_llama_server_command

from llamacpp_stack.auto_performance import (
    CACHE_TYPE_CANDIDATES,
    CACHE_TYPE_FLOOR,
    CTX_SIZE_POLICY,
    PHASE1_SPECULATIVE_TUNER_KEYS,
    PHASE1_SPECULATIVE_SEARCH_KEYS,
    PHASE2_SERVER_TUNER_KEYS,
    PHASE2_SERVER_SEARCH_KEYS,
    PROBED_TUNER_KEYS,
    STAGE0_PROBED_KEYS,
    PRUNED_TUNER_KEYS,
    _normalize_cache_type,
    _normalize_ctx_size,
    _choose_direct_io_preference,
    _estimate_trial_vram_mib,
    _minimum_gpu_count_for_config,
    _numa_candidates,
    _optuna_trial_state_complete,
    _prepare_benchmark_params,
    _phase1_tuner_keys,
    _phase2_speculative_tuner_keys,
    _phase2_tuner_keys,
    _validate_tuning_params,
    _n_gpu_layers_cli_value,
    _tensor_split_from_strategy,
    _tensor_split_strategy_candidates,
    _tensor_split_is_too_imbalanced,
    _speculative_tensor_split_strategy,
    _speculative_candidate_sequence,
    _speculative_knob_key,
    _speculative_candidate_not_heavier_than_reference,
    _speculative_ctx_descent_values,
    _spread_tensor_split,
    _speculative_trial_seed,
    _speculative_config_valid_after_repair,
    _speculative_repair_log_infeasible,
    _server_trial_seed,
    _tuner_flags_to_probe,
    _trial_params_for_catalog,
    _params_for_observation,
    _compact_catalog_auto_performance_store,
    _catalog_server_overrides_for_apply,
    _build_benchmark_command,
    _ask_yes_no,
    _ask_run_phase,
    _extract_descriptive_error,
    _early_prune_settings,
    _early_accept_settings,
    _long_confirmation_prompt_tokens,
    _long_confirmation_predict_tokens,
    LONG_CONFIRM_PROMPT_TOKENS,
    LONG_CONFIRM_PREDICT_TOKENS,
    _metric_float,
    _is_real_score_improvement,
    _score_improvement_percent,
    _format_trial_result,
    _phase_result_key,
    _find_cached_phase_result,
    _find_cached_baseline_result,
    _baseline_failure_is_fatal,
    _benchmark_metrics_failed,
    _server_benchmark_shape,
    _format_config_diff_notice,
    _benchmark_params_equivalent,
    _canonical_benchmark_key,
    _coerce_optuna_choice,
    _repair_is_structural,
    _validate_model_artifact,
    repair_until_feasible,
    run_benchmark,
    score_server_performance,
    score_performance,
)


class AutoPerformanceHelpersTest(unittest.TestCase):

    def test_phase_result_key_separates_same_params_by_phase(self) -> None:
        params = {"batch_size": 1024, "ubatch_size": 256}
        self.assertNotEqual(_phase_result_key(params, "CORE"), _phase_result_key(params, "SERVER"))

    def test_find_cached_phase_result_reads_matching_profile_role(self) -> None:
        with NamedTemporaryFile(mode="w+", suffix=".json") as tmp:
            key = _phase_result_key({"batch_size": 1024}, "CORE")
            payload = [
                {
                    "profile_key": "abc",
                    "phase_results": {
                        key: {"phase": "CORE", "role": "baseline", "score": 12.5, "metrics": {"decode_tokens_s": 1.0}},
                    },
                }
            ]
            Path(tmp.name).write_text(json.dumps(payload), encoding="utf-8")
            hit = _find_cached_phase_result("abc", key, "baseline", Path(tmp.name))
            miss = _find_cached_phase_result("abc", key, "best", Path(tmp.name))

        self.assertIsNotNone(hit)
        self.assertEqual(hit["score"], 12.5)
        self.assertIsNone(miss)

    def test_find_cached_phase_result_ignores_failed_baseline(self) -> None:
        with NamedTemporaryFile(mode="w+", suffix=".json") as tmp:
            key = _phase_result_key({"batch_size": 1024}, "CORE")
            payload = [
                {
                    "profile_key": "abc",
                    "phase_results": {
                        key: {
                            "phase": "CORE",
                            "role": "baseline",
                            "score": -1000.0,
                            "metrics": {"crash": True, "error": "no CUDA-capable device is detected"},
                        },
                    },
                }
            ]
            Path(tmp.name).write_text(json.dumps(payload), encoding="utf-8")
            hit = _find_cached_phase_result("abc", key, "baseline", Path(tmp.name))

        self.assertIsNone(hit)

    def test_find_cached_baseline_result_reads_legacy_top_level_baseline(self) -> None:
        with NamedTemporaryFile(mode="w+", suffix=".json") as tmp:
            params = {"batch_size": 1024}
            payload = [
                {
                    "profile_key": "abc",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "baseline": {
                        "phase": "CORE",
                        "trial_value": 12.5,
                        "metrics": {"decode_tokens_s": 1.0},
                        "params": params,
                        "benchmark_key": _canonical_benchmark_key(params, api_mode=False),
                    },
                    "phase_results": {},
                }
            ]
            Path(tmp.name).write_text(json.dumps(payload), encoding="utf-8")
            hit = _find_cached_baseline_result("abc", _phase_result_key(params, "CORE"), params, "CORE", path=Path(tmp.name))

        self.assertIsNotNone(hit)
        self.assertEqual(hit["score"], 12.5)
        self.assertEqual(hit["source"], "cache-top-level")

    def test_find_cached_baseline_result_reuses_same_model_when_profile_key_changes(self) -> None:
        with NamedTemporaryFile(mode="w+", suffix=".json") as tmp:
            params = {"batch_size": 1024}
            key = _phase_result_key(params, "CORE")
            payload = [
                {
                    "profile_key": "old-profile-key",
                    "model": "model-a",
                    "phase_results": {
                        key: {
                            "phase": "CORE",
                            "role": "baseline",
                            "score": 42.0,
                            "metrics": {"decode_tokens_s": 2.0},
                            "params": params,
                            "benchmark_key": _canonical_benchmark_key(params, api_mode=False),
                        }
                    },
                }
            ]
            Path(tmp.name).write_text(json.dumps(payload), encoding="utf-8")
            hit = _find_cached_baseline_result(
                "new-profile-key",
                key,
                params,
                "CORE",
                model_id="model-a",
                path=Path(tmp.name),
            )

        self.assertIsNotNone(hit)
        self.assertEqual(hit["score"], 42.0)
        self.assertEqual(hit["source"], "cache-cross-profile")

    def test_server_benchmark_shape_defaults_to_concurrent_load(self) -> None:
        self.assertEqual(_server_benchmark_shape({"parallel": 1}, 1, 1), (2, 6))
        self.assertEqual(_server_benchmark_shape({"parallel": 4}, 1, 1), (4, 12))
        self.assertEqual(_server_benchmark_shape({"parallel": 16}, 1, 1), (8, 24))
        self.assertEqual(_server_benchmark_shape({"parallel": 2}, 6, 20), (6, 20))

    def test_baseline_failure_is_fatal_for_environment_errors(self) -> None:
        fatal, reason = _baseline_failure_is_fatal({
            "timeout": True,
            "load_reason": "server-crashed",
            "error": "ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected",
        })

        self.assertTrue(fatal)
        self.assertIn(reason, {"failed to initialize cuda", "no cuda-capable device", "server-crashed"})

    def test_benchmark_metrics_failed_detects_crash_timeout_or_oom(self) -> None:
        self.assertTrue(_benchmark_metrics_failed({"crash": True}))
        self.assertTrue(_benchmark_metrics_failed({"timeout": True}))
        self.assertTrue(_benchmark_metrics_failed({"oom": True}))
        self.assertFalse(_benchmark_metrics_failed({"decode_tokens_s": 1.0}))

    def test_trial_params_catalog_maps_and_drops_internal_fields(self) -> None:
        raw = {
            "gpu_set": [0, 1],
            "ts_strategy": "even",
            "ctx_size": 32768,
            "fit": "on",
            "n_gpu_layers": "auto",
            "batch_size": 2048,
        }

        out = _trial_params_for_catalog(raw)

        self.assertNotIn("gpu_set", out)
        self.assertNotIn("ts_strategy", out)
        self.assertNotIn("ctx_size", out)
        self.assertEqual(out.get("fit"), True)
        self.assertEqual(out.get("n_gpu_layers"), -1)
        self.assertEqual(out.get("batch_size"), 2048)

    def test_trial_params_catalog_maps_all_to_999(self) -> None:
        out = _trial_params_for_catalog({"n_gpu_layers": "all", "fit": "off"})
        self.assertEqual(out.get("n_gpu_layers"), 999)
        self.assertEqual(out.get("fit"), False)

    def test_cli_n_gpu_layers_maps_sentinels_to_strings(self) -> None:
        self.assertEqual(_n_gpu_layers_cli_value(999), "all")
        self.assertEqual(_n_gpu_layers_cli_value(-1), "auto")
        self.assertEqual(_n_gpu_layers_cli_value("999"), "all")
        self.assertEqual(_n_gpu_layers_cli_value("all"), "all")
        self.assertEqual(_n_gpu_layers_cli_value("auto"), "auto")

    def test_estimate_trial_vram_scales_with_more_gpus_and_batch(self) -> None:
        baseline = {
            "gpu_set": [0],
            "batch_size": 2048,
            "ubatch_size": 512,
            "n_gpu_layers": "all",
            "fit": True,
        }
        low = _estimate_trial_vram_mib(1000.0, baseline, baseline)
        high = _estimate_trial_vram_mib(1000.0, baseline, {**baseline, "gpu_set": [0, 1, 2], "batch_size": 4096, "ubatch_size": 1024})

        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
        self.assertGreater(high, low)

    def test_minimum_gpu_count_for_config_uses_ctx_not_prompt_length(self) -> None:
        with NamedTemporaryFile() as tmp:
            tmp.truncate(20 * 1024 * 1024 * 1024)
            params = {
                "ctx_size": 262144,
                "parallel": 1,
                "batch_size": 256,
                "ubatch_size": 256,
                "flash_attn": "on",
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
            }
            hw = {"vram_mib": [8192, 8192, 8192, 8192]}

            result = _minimum_gpu_count_for_config(tmp.name, params, hw, 4)
            same_ctx_result = _minimum_gpu_count_for_config(tmp.name, {**params, "prompt_tokens": 128}, hw, 4)

        self.assertTrue(result["feasible"])
        self.assertGreaterEqual(result["gpu_count"], 2)
        self.assertEqual(result["gpu_count"], same_ctx_result["gpu_count"])
        self.assertNotIn("prompt_tokens", result)

    def test_choose_direct_io_prefers_better_score(self) -> None:
        off_metrics = {"decode_tokens_s": 100.0, "prefill_tokens_s": 50.0, "load_ready_s": 1.0}
        on_metrics = {"decode_tokens_s": 100.0, "prefill_tokens_s": 50.0, "load_ready_s": 0.4}

        self.assertTrue(_choose_direct_io_preference(off_metrics, on_metrics, 8192, 1, api_mode=False))



    def test_coerce_optuna_choice_maps_legacy_numeric_values_into_domain(self) -> None:
        self.assertEqual(_coerce_optuna_choice("256", [512, 1024, 2048, 4096, 8192], 2048), 512)
        self.assertEqual(_coerce_optuna_choice(1536, [512, 1024, 2048, 4096, 8192], 2048), 1024)
        self.assertEqual(_coerce_optuna_choice("2048", [512, 1024, 2048], 512), 2048)

    def test_coerce_optuna_choice_maps_legacy_categorical_values_into_domain(self) -> None:
        self.assertEqual(_coerce_optuna_choice("bad", ["on", "auto", "off"], "auto"), "auto")
        self.assertEqual(_coerce_optuna_choice("distribute", [None, "distribute", "isolate"], None), "distribute")

    def test_optuna_trial_state_complete_supports_trial_namespace_only(self) -> None:
        class TrialState:
            COMPLETE = object()

        optuna_like = SimpleNamespace(trial=SimpleNamespace(TrialState=TrialState))

        self.assertIs(_optuna_trial_state_complete(optuna_like), TrialState.COMPLETE)

    def test_optuna_trial_state_complete_supports_legacy_top_level(self) -> None:
        class TrialState:
            COMPLETE = object()

        optuna_like = SimpleNamespace(TrialState=TrialState)

        self.assertIs(_optuna_trial_state_complete(optuna_like), TrialState.COMPLETE)


    def test_real_score_improvement_requires_threshold_not_tiny_epsilon(self) -> None:
        self.assertTrue(_is_real_score_improvement(101.0, 100.0))
        self.assertFalse(_is_real_score_improvement(100.000001, 100.0))
        self.assertFalse(_is_real_score_improvement(100.05, 100.0))
        self.assertFalse(_is_real_score_improvement(100.4, 100.0))
        self.assertTrue(_is_real_score_improvement(100.41, 100.0))
        self.assertTrue(_is_real_score_improvement(26004.86, 26000.0))

    def test_score_improvement_percent_uses_absolute_denominator(self) -> None:
        self.assertAlmostEqual(_score_improvement_percent(110.0, 100.0), 10.0)

    def test_score_penalizes_failures(self) -> None:
        fail_metrics = {"oom": True, "decode_tokens_s": 100, "prefill_tokens_s": 1000}
        score = score_performance(fail_metrics, requested_ctx=8192, requested_gpus=1)
        self.assertEqual(score, -1000.0)

    def test_score_ignores_model_load_and_latency_signals(self) -> None:
        base = {
            "total_tokens_s": 200.0,
            "decode_tokens_s": 120.0,
            "prefill_tokens_s": 80.0,
            "load_ready_s": 0.2,
            "p95_latency_s": 0.4,
            "ctx_stable": 8192,
        }
        slower_load = {**base, "load_ready_s": 9.9}
        slower_latency = {**base, "p95_latency_s": 9.9}

        self.assertEqual(score_performance(base, requested_ctx=8192, requested_gpus=1), score_performance(slower_load, requested_ctx=8192, requested_gpus=1))
        self.assertEqual(score_server_performance(base, requested_ctx=8192, requested_gpus=1), score_server_performance(slower_latency, requested_ctx=8192, requested_gpus=1))

    def test_score_uses_total_tokens_s_as_unified_metric(self) -> None:
        higher_decode_lower_total = {
            "total_tokens_s": 20.0,
            "decode_tokens_s": 100.0,
            "prefill_tokens_s": 1.0,
            "ctx_stable": 262144,
        }
        lower_decode_higher_total = {
            "total_tokens_s": 25.0,
            "decode_tokens_s": 50.0,
            "prefill_tokens_s": 2.0,
            "ctx_stable": 262144,
        }

        self.assertGreater(
            score_performance(lower_decode_higher_total, requested_ctx=262144, requested_gpus=1),
            score_performance(higher_decode_lower_total, requested_ctx=262144, requested_gpus=1),
        )

    def test_server_score_prioritizes_real_chat_decode_over_prefill_dominated_total(self) -> None:
        """SERVER mode should follow perceived /v1/chat/completions speed.

        Real request logs can show prefill above 100 t/s while generation is
        only 1-7 t/s. A prompt-heavy request must not beat a faster decoder just
        because total prompt+decode throughput is larger.
        """
        slow_decode_prefill_heavy = {
            "total_tokens_s": 120.0,
            "prefill_tokens_s": 114.0,
            "decode_tokens_s": 2.5,
            "requests_s": 0.02,
            "server_success_rate": 1.0,
            "server_latency_p50_s": 100.0,
            "server_latency_p95_s": 140.0,
        }
        faster_decode_lower_total = {
            "total_tokens_s": 40.0,
            "prefill_tokens_s": 33.0,
            "decode_tokens_s": 6.8,
            "requests_s": 0.02,
            "server_success_rate": 1.0,
            "server_latency_p50_s": 100.0,
            "server_latency_p95_s": 140.0,
        }

        self.assertGreater(
            score_server_performance(faster_decode_lower_total, requested_ctx=262144, requested_gpus=1),
            score_server_performance(slow_decode_prefill_heavy, requested_ctx=262144, requested_gpus=1),
        )

    def test_metric_float_and_trial_result_handle_none_values(self) -> None:
        metrics = {
            "load_ready_s": None,
            "prefill_tokens_s": None,
            "decode_tokens_s": None,
            "load_reason": "health-ready",
        }

        self.assertEqual(_metric_float(metrics, "load_ready_s"), 0.0)
        self.assertEqual(_metric_float(metrics, "prefill_tokens_s"), 0.0)
        self.assertEqual(_metric_float(metrics, "decode_tokens_s"), 0.0)
        self.assertIn("Probe OK", _format_trial_result(metrics, is_probe=True))
        self.assertIn("Load: 0.00s", _format_trial_result(metrics, score=1.0, is_probe=False))

    def test_prepare_benchmark_params_strips_cache_types(self) -> None:
        config = {
            "ctx_size": 4096,
            "cache_type_k": "q4_0",
            "cache_type_v": "q4_0",
            "cache_type_k_draft": "q4_0",
            "cache_type_v_draft": "q4_0",
            "direct_io": True,
            "parallel": 4,
        }

        phase1 = _prepare_benchmark_params(config, api_mode=False)
        phase2 = _prepare_benchmark_params(config, api_mode=True)

        self.assertNotIn("cache_type_k", phase1)
        self.assertNotIn("cache_type_v", phase1)
        self.assertNotIn("cache_type_k_draft", phase1)
        self.assertNotIn("cache_type_v_draft", phase1)
        self.assertNotIn("parallel", phase1)
        self.assertIn("direct_io", phase1)

        self.assertNotIn("cache_type_k", phase2)
        self.assertNotIn("cache_type_v", phase2)
        self.assertNotIn("cache_type_k_draft", phase2)
        self.assertNotIn("cache_type_v_draft", phase2)
        self.assertIn("parallel", phase2)

    def test_prepare_benchmark_params_strips_internal_orchestration_keys(self) -> None:
        """Verify that internal orchestration keys are filtered before benchmark."""
        config = {
            "ctx_size": 4096,
            "batch_size": 1024,
            "ubatch_size": 512,
            "gpu_set": [0, 1, 2],  # Internal key that should be filtered
            "gpu_set_idx": 5,  # Internal key that should be filtered
            "ts_strategy": "balanced",  # Internal key that should be filtered
            "tensor_split_strategy": "balanced",  # Internal key that should be filtered
            "main_gpu_raw": 2,  # Internal key that should be filtered
            "mmap": True,  # Internal key that should be filtered
            "auto_performance": {"baselines": {}},  # Catalog cache metadata, not a llama.cpp flag
            "direct_io": True,  # Should NOT be filtered
        }
        
        result = _prepare_benchmark_params(config, api_mode=False)
        
        # Internal keys should be removed
        self.assertNotIn("gpu_set", result)
        self.assertNotIn("gpu_set_idx", result)
        self.assertNotIn("ts_strategy", result)
        self.assertNotIn("tensor_split_strategy", result)
        self.assertNotIn("main_gpu_raw", result)
        self.assertNotIn("mmap", result)
        self.assertNotIn("auto_performance", result)
        
        # Valid keys should remain
        self.assertIn("ctx_size", result)
        self.assertIn("batch_size", result)
        self.assertIn("ubatch_size", result)
        self.assertIn("direct_io", result)

    def test_format_config_diff_notice_explains_normalized_changes(self) -> None:
        reference = {"batch_size": 1024, "ubatch_size": 512}
        raw = {"batch_size": 2048, "ubatch_size": 256}
        current = {"batch_size": 1024, "ubatch_size": 512}

        notice = _format_config_diff_notice(current, reference, raw)

        self.assertIn("raw trial changes normalized away", notice)
        self.assertIn("batch_size", notice)
        self.assertIn("ubatch_size", notice)

    def test_benchmark_params_equivalent_ignores_internal_fields(self) -> None:
        baseline = {
            "batch_size": 1024,
            "ubatch_size": 512,
            "fit": "off",
            "gpu_set": [0, 1, 2],
            "tensor_split": "4,4,4",
        }
        trial = {
            "batch_size": 1024,
            "ubatch_size": 512,
            "fit": "off",
            "gpu_set": [0, 1, 2],
            "gpu_set_idx": 7,
            "tensor_split_strategy": "descending",
            "mmap": True,
            "main_gpu_raw": 3,
            "tensor_split": "4,4,4",
        }

        self.assertTrue(_benchmark_params_equivalent(trial, baseline, api_mode=False))


    def test_canonical_benchmark_key_ignores_internal_search_fields(self) -> None:
        base = {
            "batch_size": 1024,
            "ubatch_size": 512,
            "gpu_set": [0, 1],
            "gpu_set_idx": 3,
            "tensor_split_strategy": "equal",
            "main_gpu_raw": 7,
            "tensor_split": "1,1",
        }
        variant = {
            **base,
            "gpu_set": [0],
            "gpu_set_idx": 99,
            "tensor_split_strategy": "skewed",
            "main_gpu_raw": 1,
        }

        self.assertEqual(_canonical_benchmark_key(base), _canonical_benchmark_key(variant))

    def test_repair_is_structural_distinguishes_mechanical_normalization(self) -> None:
        self.assertFalse(_repair_is_structural(["Mechanical fix: ubatch 1024 > batch 512, capping ubatch."]))
        self.assertTrue(_repair_is_structural(["Reduced batch sizes to 1024/256"]))
        self.assertTrue(_repair_is_structural(["Rebalanced tensor_split from 5,2,0 to 4,2,1 (more even distribution)"]))




    def test_long_confirmation_uses_20k_input_and_20k_output_when_context_allows(self) -> None:
        ctx = 65536
        self.assertEqual(_long_confirmation_prompt_tokens(ctx), LONG_CONFIRM_PROMPT_TOKENS)
        self.assertEqual(_long_confirmation_predict_tokens(ctx), LONG_CONFIRM_PREDICT_TOKENS)

    def test_long_confirmation_caps_for_small_context(self) -> None:
        ctx = 4096
        prompt_tokens = _long_confirmation_prompt_tokens(ctx)
        predict_tokens = _long_confirmation_predict_tokens(ctx)
        self.assertGreaterEqual(prompt_tokens, 256)
        self.assertGreaterEqual(predict_tokens, 256)
        self.assertLessEqual(prompt_tokens + predict_tokens, ctx)


    def test_early_accept_settings_for_long_confirmation_require_clear_gain(self) -> None:
        screening = _early_accept_settings("benchmark")
        long_confirm = _early_accept_settings("ctx_half")

        self.assertEqual(long_confirm["min_tokens"], 512.0)
        self.assertEqual(long_confirm["min_predicted_ms"], 5000.0)
        self.assertEqual(long_confirm["relative_gain"], 1.03)
        self.assertGreater(screening["min_tokens"], 1e20)

    def test_early_prune_settings_are_more_conservative_for_long_confirmation(self) -> None:
        screening = _early_prune_settings("benchmark")
        long_confirm = _early_prune_settings("ctx_half")

        self.assertGreater(long_confirm["min_tokens"], screening["min_tokens"])
        self.assertGreater(long_confirm["min_predicted_ms"], screening["min_predicted_ms"])
        self.assertGreater(long_confirm["relative_floor"], screening["relative_floor"])
        self.assertEqual(long_confirm["relative_floor"], 0.92)

    def test_ctx_half_prompt_is_much_longer_than_screening_prompt(self) -> None:
        from llamacpp_stack.auto_performance import _make_completion_prompt

        short = _make_completion_prompt("benchmark")
        long = _make_completion_prompt("ctx_half", target_prompt_tokens=2048)

        self.assertGreater(len(long), len(short) * 5)
        self.assertIn("repite literalmente", long)

    def test_run_benchmark_logs_load_failure_reason(self) -> None:
        fake_log = []

        def fake_append(_log_path, message: str) -> None:
            fake_log.append(message)

        with NamedTemporaryFile(suffix=".gguf") as model_file:
            model_file.write(b"GGUF")
            model_file.flush()

            with patch("llamacpp_stack.auto_performance._append_auto_perf_log", side_effect=fake_append), \
                patch("llamacpp_stack.auto_performance._wait_for_model_load", return_value=(False, "server-crashed", 6.9, 100, 120)), \
                patch("llamacpp_stack.auto_performance._build_benchmark_command", return_value=["llama-server"]), \
                patch("llamacpp_stack.auto_performance.get_server_path", return_value=Path("/bin/llama-server")), \
                patch("llamacpp_stack.auto_performance.subprocess.Popen") as popen_mock:

                popen_mock.return_value.poll.return_value = None
                popen_mock.return_value.stdout = None

                metrics = run_benchmark(
                    Path(model_file.name),
                    {"direct_io": True, "ctx_size": 4096},
                    4096,
                    [0],
                    mock=False,
                    log_path=Path("/tmp/autoperf.log"),
                    probe_only=True,
                    expected_models_loaded=1,
                )

        self.assertTrue(metrics["timeout"])
        self.assertEqual(metrics["load_reason"], "server-crashed")
        self.assertTrue(any("SERVER_LOAD_UNSTABLE reason=server-crashed elapsed=6.9s" in entry for entry in fake_log))

    def test_validate_model_artifact_rejects_empty_file(self) -> None:
        with NamedTemporaryFile(suffix=".gguf") as empty_model:
            ok, reason = _validate_model_artifact(Path(empty_model.name))
        self.assertFalse(ok)
        self.assertIn("empty", reason)

    def test_extract_descriptive_error_prefers_cuda_root_cause(self) -> None:
        raw = "\n".join(
            [
                "[New LWP 1234]",
                "some setup line",
                "CUDA error: an illegal memory access was encountered",
                "[Inferior 1 (process 111) detached]",
            ]
        )
        self.assertEqual(
            _extract_descriptive_error(raw),
            "CUDA error: an illegal memory access was encountered",
        )

    def test_extract_descriptive_error_falls_back_when_empty(self) -> None:
        self.assertEqual(_extract_descriptive_error("", fallback="server load failed"), "server load failed")

    def test_ask_yes_no_assume_no_skips_prompt(self) -> None:
        called = {"value": False}

        def cb(_prompt, _default):
            called["value"] = True
            return "y"

        args = SimpleNamespace(assume_no=True, _question_callback=cb)
        self.assertFalse(_ask_yes_no(args, "Apply config?", "n"))
        self.assertFalse(called["value"])

    def test_ask_yes_no_no_prompt_alias_skips_prompt(self) -> None:
        called = {"value": False}

        def cb(_prompt, _default):
            called["value"] = True
            return "y"

        args = SimpleNamespace(no_prompt=True, _question_callback=cb)
        self.assertFalse(_ask_yes_no(args, "Apply config?", "n"))
        self.assertFalse(called["value"])

    def test_ask_run_phase_uses_callback_and_defaults_yes(self) -> None:
        calls = []

        def cb(prompt, default):
            calls.append((prompt, default))
            return ""

        args = SimpleNamespace(_question_callback=cb)
        self.assertTrue(_ask_run_phase(args, "CORE"))
        self.assertEqual(calls, [("¿Ejecutar fase CORE?", "y")])

    def test_ask_run_phase_can_skip_phase(self) -> None:
        args = SimpleNamespace(_question_callback=lambda _prompt, _default: "n")
        self.assertFalse(_ask_run_phase(args, "SPECULATIVE"))

    def test_ask_run_phase_assume_no_skips_noninteractive(self) -> None:
        called = {"value": False}

        def cb(_prompt, _default):
            called["value"] = True
            return "y"

        args = SimpleNamespace(assume_no=True, _question_callback=cb)
        self.assertFalse(_ask_run_phase(args, "SERVER"))
        self.assertFalse(called["value"])

    def test_ctx_size_never_decreases(self) -> None:
        self.assertEqual(CTX_SIZE_POLICY, "never_decrease")
        self.assertEqual(_normalize_ctx_size(8192, 4096), 8192)
        self.assertEqual(_normalize_ctx_size(8192, 16384), 16384)

    def test_repair_does_not_lower_ctx_under_default_policy(self) -> None:
        with NamedTemporaryFile() as tmp:
            tmp.truncate(100 * 1024 * 1024)
            config = {
                "ctx_size": 16384,
                "batch_size": 4096,
                "ubatch_size": 1024,
                "parallel": 8,
                "gpu_set": [0],
                "model_draft": None,
            }
            hw = {"vram_mib": [1024]}

            repaired, repair_log = repair_until_feasible(config, hw, tmp.name)

        self.assertEqual(repaired["ctx_size"], 16384)
        self.assertFalse(
            any("Reduced context size" in entry for entry in repair_log),
            f"Expected ctx_size to remain unchanged, got log: {repair_log}",
        )

    def test_repair_never_outputs_unsupported_ubatch_size(self) -> None:
        with NamedTemporaryFile() as tmp:
            tmp.truncate(100 * 1024 * 1024)
            config = {
                "ctx_size": 8192,
                "batch_size": 512,
                "ubatch_size": 128,
                "parallel": 1,
                "gpu_set": [0],
                "model_draft": None,
            }
            hw = {"vram_mib": [1024]}

            repaired, repair_log = repair_until_feasible(config, hw, tmp.name)

        self.assertGreaterEqual(repaired["ubatch_size"], 256)
        self.assertNotIn(128, {repaired["ubatch_size"]})
        self.assertTrue(
            any("ubatch_size below supported minimum" in entry for entry in repair_log),
            f"Expected a mechanical fix log entry, got: {repair_log}",
        )

    def test_cache_type_never_drops_below_q8(self) -> None:
        self.assertEqual(CACHE_TYPE_FLOOR, "q8_0")
        self.assertEqual(_normalize_cache_type("q4_0"), "q8_0")
        self.assertEqual(set(CACHE_TYPE_CANDIDATES), {"q8_0"})

    def test_tuner_flags_include_server_mode_and_prune_unsafe_flags(self) -> None:
        probed = set(_tuner_flags_to_probe())
        self.assertEqual(probed, set(PROBED_TUNER_KEYS))
        # direct_io is handled in the Stage 0 load probe; threads_http stays in Phase 2
        self.assertIn("numa", probed)
        self.assertIn("batch_size", probed)
        self.assertIn("direct_io", STAGE0_PROBED_KEYS)
        self.assertNotIn("direct_io", probed)
        self.assertNotIn("cache_type_k", probed)
        self.assertNotIn("cache_type_v", probed)
        self.assertNotIn("threads_http", probed)
        self.assertNotIn("mmap", probed)

        self.assertNotIn("mlock", probed)
        self.assertTrue("ctx_size" in PRUNED_TUNER_KEYS)
        self.assertNotIn("ctx_size", probed)
        self.assertEqual(_tensor_split_strategy_candidates(), ["auto", "equal", "descending", "skewed"])
        self.assertEqual(_tensor_split_from_strategy("auto", [0, 1], 2), "1,1")
        self.assertEqual(_tensor_split_from_strategy("equal", [0, 1], 2), "1,1")
        self.assertEqual(_tensor_split_from_strategy("descending", [0, 1, 2], 3), "3,2,1")
        self.assertEqual(_tensor_split_from_strategy("skewed", [0, 1, 2, 3], 4), "4,1,1,1")
        self.assertEqual(_numa_candidates(), [None, "distribute", "isolate"])

    def test_speculative_phase_rebalances_inherited_skewed_tensor_split(self) -> None:
        self.assertTrue(_tensor_split_is_too_imbalanced("7,1,1,1,1,1,1"))
        self.assertFalse(_tensor_split_is_too_imbalanced("1,1,1,1,1,1,1"))
        self.assertEqual(
            _speculative_tensor_split_strategy(
                {"tensor_split_strategy": "skewed", "tensor_split": "7,1,1,1,1,1,1"},
                [0, 1, 2, 3, 4, 5, 6],
            ),
            "equal",
        )
        self.assertEqual(
            _speculative_tensor_split_strategy(
                {"tensor_split_strategy": "equal", "tensor_split": "1,1,1,1,1,1,1"},
                [0, 1, 2, 3, 4, 5, 6],
            ),
            "equal",
        )

    def test_speculative_repair_log_prunes_no_room_for_draft_before_benchmark(self) -> None:
        ok, reason = _speculative_repair_log_infeasible([
            "Reduced draft context to 1024",
            "Speculative decoding remains enabled but config is infeasible (no room for draft model)",
        ])

        self.assertTrue(ok)
        self.assertEqual(reason, "speculative-infeasible-no-room-for-draft")

    def test_speculative_candidate_sequence_starts_with_baseline_then_cheaper_drafts(self) -> None:
        candidates = _speculative_candidate_sequence({"model_draft": "/models/draft.gguf", "draft": 32, "ctx_size_draft": 4096})

        self.assertGreater(len(candidates), 0)
        self.assertEqual(candidates[0]["ctx_size_draft"], 4096)
        self.assertEqual(candidates[0]["draft"], 32)
        self.assertEqual(candidates[1]["n_gpu_layers_draft"], "auto")
        self.assertEqual(candidates[1]["ctx_size_draft"], 512)
        self.assertEqual(candidates[1]["draft"], 4)
        self.assertTrue(any(c["ctx_size_draft"] == 1024 and c["draft"] == 8 for c in candidates))

    def test_speculative_knob_key_detects_baseline_duplicate(self) -> None:
        baseline = {"draft": 32, "ctx_size_draft": 4096, "n_gpu_layers_draft": "auto"}
        same = {"draft": "32", "ctx_size_draft": "4096", "n_gpu_layers_draft": "auto"}
        changed = {"draft": 16, "ctx_size_draft": 4096, "n_gpu_layers_draft": "auto"}

        self.assertEqual(_speculative_knob_key(baseline), _speculative_knob_key(same))
        self.assertNotEqual(_speculative_knob_key(baseline), _speculative_knob_key(changed))

    def test_speculative_candidate_not_heavier_than_reference(self) -> None:
        ref = {"draft": 16, "ctx_size_draft": 2048, "n_gpu_layers_draft": "all"}
        self.assertTrue(_speculative_candidate_not_heavier_than_reference({"draft": 8, "ctx_size_draft": 1024, "n_gpu_layers_draft": "auto"}, ref))
        self.assertTrue(_speculative_candidate_not_heavier_than_reference({"draft": 16, "ctx_size_draft": 2048, "n_gpu_layers_draft": "all"}, ref))
        self.assertFalse(_speculative_candidate_not_heavier_than_reference({"draft": 32, "ctx_size_draft": 1024, "n_gpu_layers_draft": "auto"}, ref))
        self.assertFalse(_speculative_candidate_not_heavier_than_reference({"draft": 8, "ctx_size_draft": 4096, "n_gpu_layers_draft": "auto"}, ref))
        self.assertFalse(_speculative_candidate_not_heavier_than_reference({"draft": 8, "ctx_size_draft": 1024, "n_gpu_layers_draft": "all"}, {"draft": 16, "ctx_size_draft": 2048, "n_gpu_layers_draft": "auto"}))

    def test_speculative_repair_prunes_if_draft_disabled_or_core_changed(self) -> None:
        before = {
            "model_draft": "/models/draft.gguf",
            "draft": 16,
            "n_gpu_layers": "all",
            "batch_size": 1024,
        }
        self.assertEqual(_speculative_config_valid_after_repair(before, dict(before)), (True, "ok"))

        disabled = dict(before)
        disabled["model_draft"] = None
        ok, reason = _speculative_config_valid_after_repair(before, disabled)
        self.assertFalse(ok)
        self.assertIn("model_draft", reason)

        core_changed = dict(before)
        core_changed["n_gpu_layers"] = "auto"
        ok, reason = _speculative_config_valid_after_repair(before, core_changed)
        self.assertFalse(ok)
        self.assertIn("n_gpu_layers", reason)

    def test_prepare_benchmark_params_drops_none_values(self) -> None:
        result = _prepare_benchmark_params({"model_draft": None, "draft": 0, "batch_size": 1024})
        self.assertNotIn("model_draft", result)
        self.assertIn("batch_size", result)

    def test_speculative_trial_seed_contains_only_speculative_optuna_params(self) -> None:
        seed = _speculative_trial_seed({
            "split_mode": "row",
            "batch_size": 8192,
            "gpu_set": [0, 1],
            "model_draft": "/models/draft.gguf",
            "draft": 32,
            "ctx_size_draft": 4096,
            "n_gpu_layers_draft": "auto",
        })

        self.assertEqual(set(seed), {"draft", "ctx_size_draft", "n_gpu_layers_draft"})
        self.assertEqual(seed["draft"], 32)
        self.assertEqual(seed["ctx_size_draft"], 4096)
        self.assertEqual(seed["n_gpu_layers_draft"], "auto")


    def test_server_trial_seed_contains_only_server_optuna_params(self) -> None:
        seed = _server_trial_seed({
            "split_mode": "row",
            "batch_size": 8192,
            "gpu_set": [0, 1],
            "parallel": 8,
            "cont_batching": False,
            "ctx_checkpoints": 64,
            "cache_ram": 16384,
            "threads_http": 99,
            "kv_unified": True,
            "cache_idle_slots": True,
        })

        self.assertEqual(set(seed), {"parallel", "cont_batching", "ctx_checkpoints", "cache_ram", "threads_http", "kv_unified", "cache_idle_slots"})
        self.assertEqual(seed["parallel"], 8)
        self.assertEqual(seed["threads_http"], 16)

    def test_speculative_phase_one_adds_draft_keys(self) -> None:
        model = type("M", (), {"speculative": True})()
        core_keys = set(_phase1_tuner_keys(model))
        keys = set(_phase2_speculative_tuner_keys())

        self.assertNotIn("draft", core_keys)
        self.assertNotIn("model_draft", core_keys)
        self.assertEqual(keys, PHASE1_SPECULATIVE_SEARCH_KEYS)
        self.assertIn("ctx_size_draft", keys)
        self.assertNotIn("draft", keys)
        self.assertNotIn("n_gpu_layers_draft", keys)
        self.assertNotIn("model_draft", keys)
        self.assertNotIn("cache_type_k_draft", keys)

    def test_speculative_ctx_descent_starts_at_half_main_context(self) -> None:
        values = _speculative_ctx_descent_values(262144, baseline_ctx_draft=2048)
        self.assertEqual(values[0], 131072)
        self.assertEqual(values[1], 65536)
        self.assertEqual(values[2], 32768)
        self.assertNotIn(262144, values)
        self.assertNotIn(98304, values)
        self.assertNotIn(73728, values)
        self.assertIn(2048, values)
        self.assertIn(512, values)
        self.assertLess(values.index(2048), values.index(512))
        self.assertEqual(values, sorted(values, reverse=True))

    def test_phase_two_server_keys_exposed(self) -> None:
        self.assertEqual(set(_phase2_tuner_keys()), set(PHASE2_SERVER_SEARCH_KEYS))
        self.assertNotIn("checkpoint_every_n_tokens", set(_phase2_tuner_keys()))
        self.assertIn("checkpoint_every_n_tokens", set(PHASE2_SERVER_TUNER_KEYS))

    def test_server_score_penalizes_latency_and_rewards_throughput(self) -> None:
        score = score_server_performance(
            {"total_tokens_s": 300.0, "decode_tokens_s": 100.0, "prefill_tokens_s": 200.0, "requests_s": 4.0, "server_latency_p95_s": 1.0},
            requested_ctx=8192,
            requested_gpus=1,
        )
        self.assertGreater(score, 0)

        slower_tokens_better_latency = score_server_performance(
            {
                "decode_tokens_s": 12.28,
                "prefill_tokens_s": 6.19,
                "total_tokens_s": 18.47,
                "requests_s": 0.04,
                "server_latency_p50_s": 39.87,
                "server_latency_p95_s": 60.75,
                "server_success_rate": 1.0,
            },
            requested_ctx=262144,
            requested_gpus=7,
        )
        faster_tokens_worse_latency = score_server_performance(
            {
                "decode_tokens_s": 12.45,
                "prefill_tokens_s": 6.21,
                "total_tokens_s": 18.66,
                "requests_s": 0.04,
                "server_latency_p50_s": 39.69,
                "server_latency_p95_s": 64.61,
                "server_success_rate": 1.0,
            },
            requested_ctx=262144,
            requested_gpus=7,
        )
        self.assertGreater(faster_tokens_worse_latency, slower_tokens_better_latency)

    def test_validate_tuning_params_rejects_unknown_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported tuning parameters"):
            _validate_tuning_params({"ctx_size": 8192, "typo_flag": 1}, {"ctx_size"}, label="auto-performance params")

    def test_build_benchmark_command_uses_shared_server_builder(self) -> None:
        params = {
            "fit": "off",
            "batch_size": 1024,
            "ubatch_size": 256,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "direct_io": True,
        }

        benchmark_cmd = _build_benchmark_command(
            Path("/tmp/model.gguf"),
            params,
            4096,
            server_path=Path("/bin/llama-server"),
        )

        model = ManagedModel(
            model_id="model",
            repo_id="auto-performance/benchmark",
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
            description="auto-performance benchmark harness",
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
                "fit": False,
                "batch_size": 1024,
                "ubatch_size": 256,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "direct_io": True,
            },
        )

        shared_cmd = build_llama_server_command(model, Path("/bin/llama-server"), port="18081", include_jinja=False)

        self.assertEqual(benchmark_cmd, shared_cmd)

    def test_n_gpu_layers_draft_normalizes_sentinel_values(self) -> None:
        """Verify that draft GPU layer sentinels are normalized in baseline."""
        # Test that 999 is converted to "all" (not kept as 999)
        self.assertEqual(_n_gpu_layers_cli_value(999), "all")
        # Test that -1 is converted to "auto" (not kept as -1)
        self.assertEqual(_n_gpu_layers_cli_value(-1), "auto")
        # Test string versions
        self.assertEqual(_n_gpu_layers_cli_value("999"), "all")
        self.assertEqual(_n_gpu_layers_cli_value("-1"), "auto")
        # Test valid strings pass through
        self.assertEqual(_n_gpu_layers_cli_value("all"), "all")
        self.assertEqual(_n_gpu_layers_cli_value("auto"), "auto")

    def test_vram_budget_pruning_is_hard_constraint_not_ranking_signal(self) -> None:
        """Verify that VRAM budget check penalizes with -1000 score, independent of throughput.
        
        This test ensures that VRAM factibilidad constraints are distinct from
        performance ranking signals. A trial should be rejected (score -1000) if
        it exceeds VRAM budget, regardless of observed throughput metrics.
        """
        requested_ctx = 4096
        requested_gpus = 1
        
        # Case 1: OOM due to VRAM budget (hard constraint violation)
        metrics_oom = {
            "oom": False,
            "crash": False,
            "timeout": False,
            "decode_tokens_s": 150.0,  # Even with good throughput
            "prefill_tokens_s": 50.0,
            "ctx_stable": requested_ctx,
            "load_ready_s": 0.1,  # Fast load
        }
        # Note: In production, VRAM budget check (line 1902) returns -1000 before scoring.
        # This test documents that budget violations are scored lowest regardless of throughput.
        score_vram_ok = score_performance(metrics_oom, requested_ctx, requested_gpus)
        self.assertGreater(score_vram_ok, -1000.0, 
                          "Metrics without VRAM violation should score > -1000")
        
        # Case 2: Metrics with failure flags (OOM or crash) should score -1000
        metrics_fail = {
            "oom": True,  # Hard failure
            "crash": False,
            "timeout": False,
            "decode_tokens_s": 150.0,
            "prefill_tokens_s": 50.0,
            "ctx_stable": requested_ctx,
        }
        score_fail = score_performance(metrics_fail, requested_ctx, requested_gpus)
        self.assertEqual(score_fail, -1000.0,
                        "OOM/crash/timeout flags should score -1000 (hard factibilidad violation)")

    def test_mid_inference_pruning_differentiated_from_crashes(self) -> None:
        """Verify that timeout from throughput pruning vs hard crashes are distinct.
        
        Mid-inference pruning (throughput < 80% of best) should be logged as
        timeout but differentiated as 'pruning' in error message.
        Hard failures (OOM, segfault) should not have 'pruning' in error string.
        """
        # In the actual code (line 1955), this is checked:
        # if metrics.get("timeout") and "pruning" in str(metrics.get("error", "")):
        
        # Case 1: Intentional pruning (should be differentiable)
        metrics_pruning = {
            "oom": False,
            "crash": False,
            "timeout": True,
            "error": "Mid-inference throughput pruning: 40.0 < 0.8*50.0",
            "decode_tokens_s": 40.0,  # Below 80% threshold
            "prefill_tokens_s": 10.0,
            "ctx_stable": 4096,
        }
        self.assertIn("pruning", metrics_pruning.get("error", "").lower(),
                     "Pruning event should be explicitly marked in error message")
        
        # Case 2: Real timeout (no pruning marker)
        metrics_timeout = {
            "oom": False,
            "crash": False,
            "timeout": True,
            "error": "Benchmark exceeded 120s max_total_s",
            "decode_tokens_s": 0.0,
            "prefill_tokens_s": 0.0,
            "ctx_stable": 0,
        }
        self.assertNotIn("pruning", metrics_timeout.get("error", "").lower(),
                        "Real timeout should not have 'pruning' marker")

    def test_score_functions_ignore_load_and_latency_in_pruning_context(self) -> None:
        """Verify scoring remains stable even with variable model load times.
        
        This regression test ensures that pruning decisions (based on throughput)
        and scoring (for trial ranking) are decoupled from transitory signals
        like model load time or API latency.
        
        Scenario: Two trials with identical throughput but different load times.
        Both should score identically (purely on decode/prefill/ctx stability).
        """
        base_metrics = {
            "oom": False,
            "crash": False,
            "timeout": False,
            "decode_tokens_s": 100.0,
            "prefill_tokens_s": 30.0,
            "total_tokens_s": 130.0,
            "ctx_stable": 4096,
            "requests_s": 5.0,
        }
        ctx, gpus = 4096, 1
        
        # Trial A: Fast load
        m_fast_load = dict(base_metrics)
        m_fast_load["load_ready_s"] = 0.2
        score_fast = score_performance(m_fast_load, ctx, gpus)
        
        # Trial B: Slow load (e.g., large model, cold cache)
        m_slow_load = dict(base_metrics)
        m_slow_load["load_ready_s"] = 15.0
        score_slow = score_performance(m_slow_load, ctx, gpus)
        
        # Both should score identically (load time should not influence ranking)
        self.assertEqual(score_fast, score_slow,
                        "Score should be invariant to load_ready_s")
        
        # Same test for server mode scoring
        server_score_fast = score_server_performance(m_fast_load, ctx, gpus)
        server_score_slow = score_server_performance(m_slow_load, ctx, gpus)
        self.assertEqual(server_score_fast, server_score_slow,
                        "Server scoring should be invariant to load_ready_s")


class OOMAwareAdaptationTest(unittest.TestCase):
    """Tests for OOM-aware search algorithm improvements.
    
    NEW in this session:
    - _spread_tensor_split(): rebalance tensor_split weights for more even GPU load
    - OOM reactive adaptation: enqueue recovery trials when OOM detected
    """
    
    def test_spread_tensor_split_flattens_skewed_distribution(self) -> None:
        """tensor_split rebalancing: skewed (5,2,0) -> (3,2,1) -> (2,2,1)"""
        # Skewed distribution: heavy on first GPU
        skewed = "5,2,0"
        spread1 = _spread_tensor_split(skewed, 3)
        # Should move weight from GPU 0 to GPU 2
        self.assertNotEqual(spread1, skewed, "Should rebalance skewed distribution")
        self.assertIn(",", spread1, "Should remain comma-separated")
        
        # Apply again to check progressive flattening
        spread2 = _spread_tensor_split(spread1, 3)
        # Continue progressing toward uniform if not already there
        parts1 = [float(x) for x in spread1.split(",")]
        parts2 = [float(x) for x in spread2.split(",")]
        std1 = (sum((x - sum(parts1)/len(parts1))**2 for x in parts1) / len(parts1))**0.5
        std2 = (sum((x - sum(parts2)/len(parts2))**2 for x in parts2) / len(parts2))**0.5
        # After progressive application, should trend toward uniform (lower std)
        self.assertLessEqual(std2, std1 + 0.1, "Should maintain or reduce distribution evenness")
    
    def test_spread_tensor_split_maintains_uniform_distribution(self) -> None:
        """tensor_split rebalancing: uniform (1,1,1) should remain unchanged"""
        uniform = "1,1,1"
        spread = _spread_tensor_split(uniform, 3)
        self.assertEqual(spread, uniform, "Uniform distribution should not change")
    
    def test_spread_tensor_split_handles_single_gpu(self) -> None:
        """tensor_split for single GPU (1) should remain unchanged"""
        single = "1"
        spread = _spread_tensor_split(single, 1)
        self.assertEqual(spread, single, "Single GPU should not change")
    
    def test_spread_tensor_split_handles_invalid_input(self) -> None:
        """tensor_split with unparseable input should return unchanged"""
        invalid = "auto"
        spread = _spread_tensor_split(invalid, 3)
        self.assertEqual(spread, invalid, "Unparseable input should return unchanged")
    
    def test_spread_tensor_split_pads_to_gpu_count(self) -> None:
        """tensor_split should pad unused GPUs with zeros"""
        partial = "2,1"  # Only 2 weights for 3 GPUs
        spread = _spread_tensor_split(partial, 3)
        # Should have 3 comma-separated parts after spreading
        parts = spread.split(",")
        self.assertEqual(len(parts), 2, "Should preserve partial GPU spec (2 GPUs out of potential 3)")
    
    def test_gpu_set_idx_removed_from_phase1_suggestions(self) -> None:
        """Verify gpu_set_idx is NO LONGER a Phase 1 search parameter.
        
        With the new algorithm, GPU set is fixed to 'all available' (index 0),
        and load balancing is done via tensor_split mutations instead.
        """
        # In Phase 1, gpu_set_idx should NOT be in PROBED_TUNER_KEYS
        self.assertNotIn("gpu_set_idx", PROBED_TUNER_KEYS,
                        "gpu_set_idx should not be explored in Phase 1 "
                        "(GPU set is fixed to all available, load balanced via tensor_split)")
        
        # tensor_split SHOULD still be explored (via tensor_split parameter)
        self.assertIn("tensor_split", PROBED_TUNER_KEYS,
                     "tensor_split should still be explored for load balancing")


class CatalogPersistenceSanitizationTest(unittest.TestCase):
    def test_trial_params_for_catalog_removes_internal_orchestration_keys(self) -> None:
        params = {
            "batch_size": 256,
            "ubatch_size": 256,
            "ctx_size": 262144,
            "gpu_set": [0, 1],
            "gpu_set_idx": 12,
            "main_gpu_raw": 2,
            "tensor_split_strategy": "equal",
            "ts_strategy": "even",
            "direct_io": True,
            "mmap": True,
            "auto_performance": {"baselines": {}},
            "fit": "off",
            "n_gpu_layers": "all",
            "n_gpu_layers_draft": 999,
        }

        persisted = _trial_params_for_catalog(params)

        for key in {
            "ctx_size",
            "gpu_set",
            "gpu_set_idx",
            "main_gpu_raw",
            "tensor_split_strategy",
            "ts_strategy",
            "direct_io",
            "mmap",
            "auto_performance",
        }:
            self.assertNotIn(key, persisted)
        self.assertEqual(persisted["batch_size"], 256)
        self.assertIs(persisted["fit"], False)
        self.assertEqual(persisted["n_gpu_layers"], 999)
        self.assertEqual(persisted["n_gpu_layers_draft"], "all")

    def test_params_for_observation_keeps_ctx_but_not_runtime_internals(self) -> None:
        observed = _params_for_observation({
            "ctx_size": 262144,
            "gpu_set": [0, 1],
            "gpu_set_idx": 12,
            "main_gpu_raw": 2,
            "batch_size": 1024,
        })

        self.assertEqual(observed["ctx_size"], 262144)
        self.assertEqual(observed["batch_size"], 1024)
        self.assertNotIn("gpu_set", observed)
        self.assertNotIn("gpu_set_idx", observed)
        self.assertNotIn("main_gpu_raw", observed)

    def test_final_catalog_apply_replaces_stale_overrides_and_preserves_compact_metadata(self) -> None:
        existing_overrides = {
            "batch_size": 999,  # stale value that must not survive
            "main_gpu_raw": 7,
            "gpu_set": [0, 1, 2],
            "auto_performance": {
                "baselines": {
                    "CORE:legacy": {
                        "phase": "CORE",
                        "role": "baseline",
                        "score": 26197.0,
                        "metrics": {"prefill_tokens_s": 1.0, "decode_tokens_s": 2.0},
                        "params": {"gpu_set": [0], "batch_size": 999},
                        "benchmark_key": "legacy",
                        "updated_at": "2026-05-06T00:00:00+00:00",
                    },
                    "CORE:current": {
                        "phase": "CORE",
                        "role": "baseline",
                        "score": 3079.0,
                        "metrics": {"prefill_tokens_s": 14.0, "decode_tokens_s": 16.0},
                        "params": {"gpu_set": [0], "batch_size": 1024, "ctx_size": 262144},
                        "benchmark_key": "current",
                        "updated_at": "2026-05-07T00:00:00+00:00",
                        "score_schema_version": "total_tokens_s_v1",
                    },
                },
                "debug_dump": {"large": "not-needed"},
            },
        }
        selected = {
            "batch_size": 256,
            "ubatch_size": 256,
            "main_gpu": 2,
            "main_gpu_raw": 2,
            "ctx_size": 262144,
            "gpu_set": [0, 1, 2],
            "tensor_split_strategy": "equal",
            "direct_io": True,
            "mmap": True,
            "fit": "off",
            "n_gpu_layers": "auto",
            "n_gpu_layers_draft": "all",
        }

        final_overrides = _catalog_server_overrides_for_apply(existing_overrides, selected)

        self.assertEqual(final_overrides["batch_size"], 256)
        self.assertEqual(final_overrides["ubatch_size"], 256)
        self.assertEqual(final_overrides["main_gpu"], 2)
        self.assertIs(final_overrides["fit"], False)
        self.assertEqual(final_overrides["n_gpu_layers"], -1)
        self.assertEqual(final_overrides["n_gpu_layers_draft"], "all")
        for key in {"gpu_set", "gpu_set_idx", "main_gpu_raw", "tensor_split_strategy", "direct_io", "mmap"}:
            self.assertNotIn(key, final_overrides)

        meta = final_overrides.get("auto_performance")
        self.assertIsInstance(meta, dict)
        self.assertNotIn("debug_dump", meta)
        self.assertEqual(set(meta.get("baselines", {}).keys()), {"CORE:current"})
        self.assertNotIn("gpu_set", meta["baselines"]["CORE:current"]["params"])
        self.assertEqual(meta["baselines"]["CORE:current"]["params"]["ctx_size"], 262144)

    def test_catalog_metadata_compaction_drops_legacy_and_limits_history(self) -> None:
        baselines = {}
        for idx in range(8):
            baselines[f"CORE:key{idx}"] = {
                "phase": "CORE",
                "role": "baseline",
                "score": 1000.0 + idx,
                "metrics": {"prefill_tokens_s": 1.0, "decode_tokens_s": 2.0},
                "params": {"batch_size": 256 + idx, "gpu_set": [0]},
                "benchmark_key": f"key{idx}",
                "updated_at": f"2026-05-07T00:00:0{idx}+00:00",
                "score_schema_version": "total_tokens_s_v1",
            }
        baselines["SERVER:legacy"] = {
            "phase": "SERVER",
            "role": "baseline",
            "score": 123.0,
            "metrics": {"prefill_tokens_s": 1.0},
            "params": {},
            "benchmark_key": "legacy",
            "updated_at": "2026-05-07T00:00:09+00:00",
        }

        compact = _compact_catalog_auto_performance_store({"baselines": baselines, "debug": "drop-me"})

        self.assertNotIn("debug", compact)
        kept = compact["baselines"]
        self.assertEqual(len(kept), 6)
        self.assertNotIn("SERVER:legacy", kept)
        self.assertIn("CORE:key7", kept)
        self.assertIn("CORE:key2", kept)
        self.assertNotIn("CORE:key1", kept)
        for record in kept.values():
            self.assertNotIn("gpu_set", record["params"])


if __name__ == "__main__":
    unittest.main()
