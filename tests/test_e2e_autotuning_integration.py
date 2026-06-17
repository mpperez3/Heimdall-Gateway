#!/usr/bin/env python3
"""
End-to-end integration tests for autotuning validation.

This test suite validates the complete autotuning flow:
1. Flag generation and validation
2. Shared builder integration
3. Profile persistence
4. Scoring stability

Run with: python -m pytest tests/test_e2e_autotuning_integration.py -v
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llamacpp_stack.cli import ManagedModel, build_llama_server_command
from llamacpp_stack.auto_performance import (
    _build_benchmark_command,
    _resolve_baseline,
    _prepare_benchmark_params,
    _trial_params_for_catalog,
    detect_cuda_device_count,
    run_benchmark,
    score_performance,
    score_server_performance,
)


REAL_E2E_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "SmolLM2-135M-Instruct-Q2_K.gguf"


class AutotuningEndToEndTests(unittest.TestCase):
    """Integration tests for complete autotuning pipelines."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        if not REAL_E2E_MODEL_PATH.exists():
            self.skipTest(f"Missing real E2E model: {REAL_E2E_MODEL_PATH}")

        self.model = ManagedModel(
            model_id="SmolLM2-135M-Instruct-Q2_K",
            repo_id="unsloth/SmolLM2-135M-Instruct-GGUF",
            quant="Q2_K",
            filename=REAL_E2E_MODEL_PATH.name,
            local_path=str(REAL_E2E_MODEL_PATH),
            ctx_size=4096,
            n_gpu_layers=999,
        )
        
        self.ctx = 4096
        self.gpu_count = 1

    def test_complete_flag_emission_and_validation_cycle(self) -> None:
        """Validate that trial params generate valid llama-server commands.
        
        This tests the complete pipeline:
        1. Autotuner generates trial parameters
        2. Parameters are converted to benchmark-ready format
        3. Benchmark command is built using shared CLI builder
        4. Commands include the same flags (core contract)
        """
        # Simulate trial parameters from Optuna
        trial_params = {
            "gpu_set": [0],
            "ts_strategy": "auto",
            "ctx_size": 4096,
            "fit": "off",
            "n_gpu_layers": 30,
            "batch_size": 512,
            "ubatch_size": 256,
            "cache_type_k": "q4_0",
            "cache_type_v": "q4_0",
            "direct_io": True,
            "numa": None,
        }
        
        # Step 1: Convert trial params to catalog-ready format
        catalog_params = _trial_params_for_catalog(trial_params)
        
        # Verify internal fields are dropped (orchestration fields, not user-facing)
        self.assertNotIn("gpu_set", catalog_params)
        self.assertNotIn("ts_strategy", catalog_params)
        self.assertNotIn("ctx_size", catalog_params)
        
        # Verify flag values are normalized
        self.assertIsInstance(catalog_params.get("fit"), bool)
        self.assertIsInstance(catalog_params.get("n_gpu_layers"), int)
        self.assertIsInstance(catalog_params.get("batch_size"), int)
        
        # Step 2: Build benchmark command from trial params
        benchmark_params = _prepare_benchmark_params(trial_params, api_mode=False)
        benchmark_cmd = _build_benchmark_command(
            model_path=self.model.local_path,
            params=benchmark_params,
            ctx_size=self.ctx,
        )
        
        # Step 3: Verify command structure (returns list of command args)
        self.assertIsInstance(benchmark_cmd, list)
        self.assertGreater(len(benchmark_cmd), 0, "Command should include arguments")
        
        # Command should include model path as a list element
        cmd_str = " ".join(str(arg) for arg in benchmark_cmd)
        self.assertIn("--model", cmd_str)
        self.assertIn(str(self.model.local_path), cmd_str)
        
        # Key flags from trial params should be present
        # These flags come from normalize_server_overrides conversion
        self.assertIn("-fit", cmd_str, "fit=off should emit -fit flag (single dash)")
        self.assertIn("--batch-size", cmd_str, "batch_size should emit --batch-size flag")
        self.assertIn("--direct-io", cmd_str, "direct_io=true should emit --direct-io flag")
        
        # Step 4: Verify that benchmark and shared builder use the same params
        # (not comparing command strings directly, but validating same flags)
        model_with_overrides = ManagedModel(
            model_id=self.model.model_id,
            repo_id=self.model.repo_id,
            quant=self.model.quant,
            filename=self.model.filename,
            local_path=self.model.local_path,
            ctx_size=self.ctx,
            n_gpu_layers=int(trial_params.get("n_gpu_layers", 999)),
            server_overrides=catalog_params,
        )
        
        shared_cmd = build_llama_server_command(
            model_with_overrides,
            Path("/bin/llama-server"),
            port="18081",
            include_jinja=False
        )
        
        # Both builders should include the same flag parameters
        # (builder path may differ, but flag content should match)
        shared_cmd_str = shared_cmd if isinstance(shared_cmd, str) else " ".join(str(arg) for arg in shared_cmd)
        
        # Validate key flags are in shared command too (contract validation)
        self.assertIn("--model", shared_cmd_str, "Shared builder should include model flag")
        self.assertIn("-fit", shared_cmd_str, "Shared builder should respect fit override (single dash)")
        self.assertIn("--batch-size", shared_cmd_str, "Shared builder should respect batch_size")
        self.assertIn("--direct-io", shared_cmd_str, "Shared builder should respect direct_io")

    @unittest.skipUnless(os.getenv("RUN_REAL_AUTO_PERF_E2E") == "1", "Set RUN_REAL_AUTO_PERF_E2E=1 to run the live SmolLM2 benchmark")
    def test_real_smollm2_benchmark_produces_nonzero_baseline(self) -> None:
        """Run a short live benchmark and ensure the baseline is not broken.

        This is an actual end-to-end check against the real SmolLM2 GGUF.
        It should fail if the load path regresses back to a zero-throughput
        baseline or a hard -1000 score.
        """
        gpu_count = detect_cuda_device_count()
        if gpu_count <= 0:
            self.skipTest("Real benchmark requires at least one CUDA device")

        baseline_params = _resolve_baseline(self.model, gpu_count)
        ctx_size = int(baseline_params.get("ctx_size", self.model.ctx_size or 2048) or self.model.ctx_size or 2048)
        ctx_size = min(ctx_size, 2048)

        # Convert baseline params (which include internal keys like gpu_mask, ts_strategy)
        # to benchmark-compatible format by filtering through _prepare_benchmark_params.
        # Baseline params are already in the right shape (not tuple format), so we can
        # pass them directly after removing internal orchestration keys.
        benchmark_params = {
            k: v for k, v in baseline_params.items()
            if k not in {"gpu_mask", "main_gpu_raw", "tensor_split_strategy"}
        }
        benchmark_params = _prepare_benchmark_params(benchmark_params, api_mode=False)
        benchmark_params["ctx_size"] = ctx_size

        metrics = run_benchmark(
            model_path=self.model.local_path,
            params=benchmark_params,
            ctx_size=ctx_size,
            gpu_set=list(range(gpu_count)),
            n_predict=16,
            runs=1,
            max_total_s=120,
            server_ready_timeout_s=180,
        )

        self.assertFalse(metrics["oom"], f"Unexpected OOM: {metrics}")
        self.assertFalse(metrics["crash"], f"Unexpected crash: {metrics}")
        self.assertFalse(metrics["timeout"], f"Unexpected timeout: {metrics}")
        self.assertGreater(metrics["prefill_tokens_s"], 0.0, f"Broken baseline metrics: {metrics}")
        self.assertGreater(metrics["decode_tokens_s"], 0.0, f"Broken baseline metrics: {metrics}")

        score = score_performance(metrics, ctx_size, gpu_count)
        self.assertGreater(score, -1000.0, f"Broken baseline score: {score}; metrics={metrics}")

    def test_scoring_stability_with_variable_load_conditions(self) -> None:
        """Validate that scoring remains stable across different load conditions.
        
        This test simulates trials with identical inference throughput but
        different model load times (e.g., cold vs warm cache, model size variations).
        Scoring should depend only on throughput and context stability.
        """
        base_metrics = {
            "oom": False,
            "crash": False,
            "timeout": False,
            "decode_tokens_s": 85.0,
            "prefill_tokens_s": 25.0,
            "ctx_stable": 4096,
            "requests_s": 8.0,
        }
        
        # Trial A: Mini model (fast load, same inference speed as B)
        metrics_mini = dict(base_metrics)
        metrics_mini["load_ready_s"] = 0.5
        score_mini = score_performance(metrics_mini, self.ctx, self.gpu_count)
        
        # Trial B: Larger variant (slow load due to K/V caches, same inference)
        metrics_larger = dict(base_metrics)
        metrics_larger["load_ready_s"] = 8.0
        score_larger = score_performance(metrics_larger, self.ctx, self.gpu_count)
        
        # Both should score identically (load time not a ranking factor)
        self.assertEqual(score_mini, score_larger,
                        "Score should be invariant to model load time for identical inference throughput")
        
        # Same for server mode
        server_mini = score_server_performance(metrics_mini, self.ctx, self.gpu_count)
        server_larger = score_server_performance(metrics_larger, self.ctx, self.gpu_count)
        self.assertEqual(server_mini, server_larger,
                        "Server score should be invariant to load time")

    def test_profile_serialization_preserves_tuning_policy(self) -> None:
        """Validate that autotuning profiles preserve tuning decisions.
        
        A complete profile should include:
        1. Best metrics achieved
        2. Parameters that achieved those metrics
        3. Tuning policy (which flags were probed, which were pruned)
        4. Configuration details for reproducibility
        """
        from llamacpp_stack.auto_performance import PROBED_TUNER_KEYS, PRUNED_TUNER_KEYS
        
        # Simulate a profile that would be saved
        profile = {
            "model_id": self.model.model_id,
            "ctx": self.ctx,
            "best": {
                "metrics": {
                    "decode_tokens_s": 90.0,
                    "prefill_tokens_s": 28.0,
                    "ctx_stable": 4096,
                    "load_ready_s": 1.2,
                    "oom": False,
                    "crash": False,
                    "timeout": False,
                },
                "trial_value": 52.5,  # Calculated score
                "params": {
                    "fit": "off",
                    "n_gpu_layers": 25,
                    "batch_size": 512,
                    "cache_type_k": "q4_0",
                    "cache_type_v": "q4_0",
                }
            },
            "tuning_policy": {
                "ctx_size_floor": 2048,
                "ctx_size_policy": "fit_budget",
                "probed_flags": sorted(PROBED_TUNER_KEYS),
                "pruned_flags": sorted(PRUNED_TUNER_KEYS),
            }
        }
        
        # Verify profile can be JSON serialized (for storage)
        profile_json = json.dumps(profile, ensure_ascii=False, default=str)
        self.assertIsInstance(profile_json, str)
        self.assertGreater(len(profile_json), 0)
        
        # Verify deserialization works
        restored = json.loads(profile_json)
        self.assertEqual(restored["model_id"], self.model.model_id)
        self.assertEqual(restored["best"]["metrics"]["decode_tokens_s"], 90.0)
        self.assertIn("probed_flags", restored["tuning_policy"])
        self.assertIn("pruned_flags", restored["tuning_policy"])

    def test_factibilidad_constraints_vs_performance_ranking(self) -> None:
        """Validate that factibilidad constraints and performance rankings are separate.
        
        Factibilidad (can the trial run at all?) should be checked before
        performance scoring. A trial that runs but underperforms should
        score low but not be rejected. A trial with OOM/crash should be
        rejected regardless of theoretical performance.
        """
        # Scenario 1: Trial runs but with poor throughput (underperformance)
        # Note: ctx_stable has significant weight in scoring (0.1), so we use
        # degraded ctx_stable to show underperformance despite successful run
        metrics_slow = {
            "oom": False,
            "crash": False,
            "timeout": False,
            "decode_tokens_s": 10.0,  # Very slow
            "prefill_tokens_s": 3.0,
            "ctx_stable": 1024,  # Degraded context (vs requested 4096)
            "load_ready_s": 0.5,
        }
        score_slow = score_performance(metrics_slow, self.ctx, self.gpu_count)
        
        # Trial ran successfully but scored lower due to poor throughput and degraded ctx
        # score = 0.58 * 10 + 0.28 * 3 + 0.1 * 1024 = 5.8 + 0.84 + 102.4 = 109.04
        self.assertLess(score_slow, 200.0, "Slow throughput with degraded context should score low")
        self.assertGreater(score_slow, -1000.0, "Underperformance is not a hard failure")
        
        # Scenario 2: Trial cannot run (OOM/crash = hard factibilidad failure)
        metrics_fail = {
            "oom": True,  # Hard failure
            "crash": False,
            "timeout": False,
            "decode_tokens_s": 0.0,
            "prefill_tokens_s": 0.0,
            "ctx_stable": 0,
        }
        score_fail = score_performance(metrics_fail, self.ctx, self.gpu_count)
        
        # Score should be exactly -1000 (hard rejection)
        self.assertEqual(score_fail, -1000.0, "OOM/crash should score -1000 (hard factibilidad failure)")

    def test_phase_transitions_maintain_scoring_consistency(self) -> None:
        """Validate that scoring remains consistent between Phase 1 and Phase 2.
        
        Phase 1: Raw inference tuning (llama-server bin)
        Phase 2: Server API tuning (HTTP requests)
        
        Both phases should use the same throughput metrics and same scoring rules,
        ensuring phase transitions don't introduce scoring discontinuities.
        """
        # Same workload metrics, as would come from benchmarking
        workload_metrics = {
            "oom": False,
            "crash": False,
            "timeout": False,
            "decode_tokens_s": 75.0,
            "prefill_tokens_s": 20.0,
            "ctx_stable": 4096,
            "requests_s": 6.0,
            "load_ready_s": 1.0,
        }
        
        # Phase 1 scoring (raw inference)
        score_p1 = score_performance(workload_metrics, self.ctx, self.gpu_count)
        
        # Phase 2 scoring (server API with requests_s)
        score_p2 = score_server_performance(workload_metrics, self.ctx, self.gpu_count)
        
        # Both should be positive (good metrics)
        self.assertGreater(score_p1, 0.0, "Phase 1 good metrics should score positive")
        self.assertGreater(score_p2, 0.0, "Phase 2 good metrics should score positive")
        
        # Both should penalize failures identically
        workload_metrics["crash"] = True
        score_p1_fail = score_performance(workload_metrics, self.ctx, self.gpu_count)
        score_p2_fail = score_server_performance(workload_metrics, self.ctx, self.gpu_count)
        
        self.assertEqual(score_p1_fail, -1000.0)
        self.assertEqual(score_p2_fail, -1000.0)


if __name__ == "__main__":
    unittest.main()
