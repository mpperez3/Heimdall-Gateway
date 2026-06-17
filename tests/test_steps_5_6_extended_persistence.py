#!/usr/bin/env python3
"""
Extended test coverage for Steps 5-6: Repair Determinism, VRAM Estimation, and Persistence.

These tests validate:
- Step 5: Repair logic determinism and VRAM estimation accuracy
- Step 6: Profile persistence and reproducibility
"""

import json
import unittest
from pathlib import Path

from llamacpp_stack.auto_performance import (
    _estimate_trial_vram_mib,
    _estimate_vram_breakdown,
    repair_until_feasible,
    VRAM_ZONES,
)


REAL_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "SmolLM2-135M-Instruct-Q2_K.gguf"


class Step5RepairAndVramTests(unittest.TestCase):
    """Step 5: Extended tests for repair logic and VRAM estimation."""

    @classmethod
    def setUpClass(cls) -> None:
        if not REAL_MODEL_PATH.exists():
            raise unittest.SkipTest(f"Missing real model artifact: {REAL_MODEL_PATH}")

    def test_repair_determinism_same_config_same_result(self) -> None:
        """Validate that repair_until_feasible is deterministic.
        
        Running repair on the same config twice should produce identical results.
        This ensures reproducibility and prevents flaky trial selection.
        """
        config = {
            "ctx_size": 8192,
            "batch_size": 2048,
            "ubatch_size": 512,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "parallel": 8,
            "gpu_set": [0, 1],
            "model_draft": None,
        }
        
        hw = {
            "vram_mib": [20480, 20480],  # 2x RTX 3090 (20GB each)
        }
        
        model_path = str(REAL_MODEL_PATH)
        
        # Run repair twice
        repair1, log1 = repair_until_feasible(config, hw, model_path)
        repair2, log2 = repair_until_feasible(config, hw, model_path)
        
        # Should be identical
        self.assertEqual(repair1, repair2, "Repair should be deterministic")
        self.assertEqual(log1, log2, "Repair logs should be identical")
        
        # Should not infinitely repair
        self.assertLessEqual(len(log1), 15, "Repair should max out at 15 iterations")

    def test_repair_respects_gpu_set_budget(self) -> None:
        """Validate that repair respects the VRAM budget for selected GPUs.
        
        A config that might need repair should produce a result that fits
        within the amber zone (95% of budget) for safe operation.
        """
        config = {
            "ctx_size": 16384,
            "batch_size": 4096,
            "ubatch_size": 1024,
            "cache_type_k": "f32",
            "cache_type_v": "f32",
            "parallel": 16,
            "gpu_set": [0],  # Single GPU with limited budget
            "model_draft": None,
        }
        
        hw = {
            "vram_mib": [8192],  # 8GB GPU
        }
        
        model_path = str(REAL_MODEL_PATH)
        
        # Repair should always return a valid config
        repaired, repair_log = repair_until_feasible(config, hw, model_path)
        
        # Verify repaired config fits in amber zone (safe zone)
        breakdown = _estimate_vram_breakdown(model_path, repaired, hw)
        total_needed = breakdown["total"]
        budget = breakdown["budget"]
        
        # Repaired config should fit safely (even if no repair was needed)
        self.assertLess(total_needed, budget * VRAM_ZONES["amber"], 
                       "Repaired config should fit in amber zone (95% of budget or less)")

    def test_vram_estimation_consistency_across_scales(self) -> None:
        """Validate that VRAM estimation scales predictably with context.
        
        Doubling context should increase VRAM estimate predictably
        (proportional scaling for KV cache component).
        """
        baseline_params = {
            "ctx_size": 2048,
            "batch_size": 512,
            "ubatch_size": 128,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
        }
        
        trial_1x = dict(baseline_params)
        trial_2x = dict(baseline_params)
        trial_2x["ctx_size"] = 4096
        
        base_vram = 4096.0  # Base 4GB estimate
        
        vram_1x = _estimate_trial_vram_mib(base_vram, baseline_params, trial_1x)
        vram_2x = _estimate_trial_vram_mib(base_vram, baseline_params, trial_2x)
        
        # 2x context should increase VRAM estimate by ~2x (KV cache dominates)
        ratio = vram_2x / vram_1x
        self.assertGreater(ratio, 1.8, "Doubling context should increase VRAM by ~2x")
        self.assertLess(ratio, 2.5, "Ratio should be reasonable (not super-linear)")

    def test_vram_estimation_batch_size_scaling(self) -> None:
        """Validate that batch size scaling is reflected in VRAM estimates.
        
        Larger batch sizes should increase compute/temporary buffer allocation.
        """
        baseline_params = {
            "ctx_size": 4096,
            "batch_size": 512,
            "ubatch_size": 128,
        }
        
        trial_small_batch = dict(baseline_params)
        trial_large_batch = dict(baseline_params)
        trial_large_batch["batch_size"] = 2048
        
        base_vram = 4096.0
        
        vram_small = _estimate_trial_vram_mib(base_vram, baseline_params, trial_small_batch)
        vram_large = _estimate_trial_vram_mib(base_vram, baseline_params, trial_large_batch)
        
        # Large batch should use more VRAM (10% penalty per trial_params)
        self.assertGreater(vram_large, vram_small,
                          "Larger batch size should increase VRAM estimate")


class Step6PersistenceTests(unittest.TestCase):
    """Step 6: Tests for profile persistence and reproducibility."""

    def test_profile_json_serialization_completeness(self) -> None:
        """Validate that profiles can be fully serialized and deserialized.
        
        Profile must retain all information needed for reproducibility:
        - Best metrics achieved
        - Parameters that achieved those metrics
        - Tuning policy (probed/pruned flags)
        - Hardware fingerprint / context
        """
        from llamacpp_stack.auto_performance import PROBED_TUNER_KEYS, PRUNED_TUNER_KEYS
        
        original_profile = {
            "profile_key": "test-tinystories-4096",
            "model_id": "test-tinystories-mini",
            "ctx": 4096,
            "best": {
                "metrics": {
                    "decode_tokens_s": 150.0,
                    "prefill_tokens_s": 50.0,
                    "ctx_stable": 4096,
                    "load_ready_s": 0.5,
                    "oom": False,
                    "crash": False,
                    "timeout": False,
                },
                "trial_value": 85.2,  # Score
                "params": {
                    "fit": False,
                    "n_gpu_layers": 25,
                    "batch_size": 1024,
                    "ubatch_size": 256,
                    "cache_type_k": "q8_0",
                    "cache_type_v": "q8_0",
                    "direct_io": True,
                }
            },
            "tuning_policy": {
                "ctx_size_floor": 2048,
                "ctx_size_policy": "fit_budget",
                "probed_flags": sorted(PROBED_TUNER_KEYS),
                "pruned_flags": sorted(PRUNED_TUNER_KEYS),
            },
            "timestamp": "2026-05-03T12:00:00Z",
        }
        
        # Serialize to JSON
        profile_json = json.dumps(original_profile, ensure_ascii=False, default=str)
        self.assertIsInstance(profile_json, str)
        self.assertGreater(len(profile_json), 100)
        
        # Deserialize
        restored_profile = json.loads(profile_json)
        
        # Validate all required fields are present
        self.assertEqual(restored_profile["model_id"], original_profile["model_id"])
        self.assertEqual(restored_profile["ctx"], original_profile["ctx"])
        self.assertEqual(restored_profile["best"]["metrics"]["decode_tokens_s"], 150.0)
        self.assertEqual(restored_profile["best"]["trial_value"], 85.2)
        
        # Validate tuning policy is complete
        self.assertIn("probed_flags", restored_profile["tuning_policy"])
        self.assertIn("pruned_flags", restored_profile["tuning_policy"])
        self.assertEqual(len(restored_profile["tuning_policy"]["probed_flags"]), 
                        len(PROBED_TUNER_KEYS))
        self.assertEqual(len(restored_profile["tuning_policy"]["pruned_flags"]), 
                        len(PRUNED_TUNER_KEYS))

    def test_profile_reproducibility_with_same_params(self) -> None:
        """Validate that running with saved params produces same results structure.
        
        This test doesn't actually run benchmarks (would be too slow),
        but validates that the profile structure supports reproducibility.
        """
        saved_params = {
            "fit": False,
            "n_gpu_layers": 30,
            "batch_size": 512,
            "ubatch_size": 256,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "direct_io": True,
        }
        
        # Simulate "running again" with same params
        restorable = dict(saved_params)
        restorable_json = json.dumps(restorable, ensure_ascii=False)
        restored_params = json.loads(restorable_json)
        
        # Params should be identical
        self.assertEqual(restored_params, saved_params,
                        "Restored params should match saved params exactly")
        
        # This proves the flow: save → JSON → restore preserves precision

    def test_multi_profile_catalog_management(self) -> None:
        """Validate that multiple profiles can be stored and retrieved correctly.
        
        A profile catalog (list of profiles) should support adding new profiles
        without losing existing ones, and handle updates correctly.
        """
        profile1 = {
            "profile_key": "model-a-4096",
            "model_id": "model-a",
            "ctx": 4096,
            "best": {"metrics": {"decode_tokens_s": 100.0}},
        }
        
        profile2 = {
            "profile_key": "model-b-8192",
            "model_id": "model-b",
            "ctx": 8192,
            "best": {"metrics": {"decode_tokens_s": 150.0}},
        }
        
        profile1_updated = {
            "profile_key": "model-a-4096",
            "model_id": "model-a",
            "ctx": 4096,
            "best": {"metrics": {"decode_tokens_s": 110.0}},  # Improved
        }
        
        # Simulate catalog as a list
        catalog = []
        
        # Add profile 1
        catalog = [p for p in catalog if p.get("profile_key") != profile1.get("profile_key")]
        catalog.append(profile1)
        self.assertEqual(len(catalog), 1)
        
        # Add profile 2
        catalog = [p for p in catalog if p.get("profile_key") != profile2.get("profile_key")]
        catalog.append(profile2)
        self.assertEqual(len(catalog), 2)
        
        # Update profile 1
        catalog = [p for p in catalog if p.get("profile_key") != profile1_updated.get("profile_key")]
        catalog.append(profile1_updated)
        self.assertEqual(len(catalog), 2, "Catalog size should stay at 2 after update")
        
        # Verify updated value
        profile1_retrieved = next((p for p in catalog if p.get("profile_key") == "model-a-4096"), None)
        self.assertIsNotNone(profile1_retrieved)
        self.assertEqual(profile1_retrieved["best"]["metrics"]["decode_tokens_s"], 110.0,
                        "Updated profile should have new metrics")


class IntegrationCompletionTests(unittest.TestCase):
    """Integration tests validating all 6 steps work together."""

    def test_complete_pipeline_repair_estimate_persist(self) -> None:
        """Validate the complete pipeline: estimate → repair → persist.
        
        This test ties together all components:
        1. Initial VRAM estimation
        2. Repair if needed
        3. Profile serialization for persistence
        """
        initial_config = {
            "ctx_size": 8192,
            "batch_size": 2048,
            "ubatch_size": 512,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "parallel": 8,
            "gpu_set": [0],
        }
        
        hw = {"vram_mib": [12288]}  # 12GB GPU
        model_path = str(REAL_MODEL_PATH)
        
        # Step 1: Estimate initial VRAM
        initial_breakdown = _estimate_vram_breakdown(model_path, initial_config, hw)
        initial_total = initial_breakdown["total"]
        budget = initial_breakdown["budget"]
        
        # Step 2: Repair if needed
        if initial_total > budget * VRAM_ZONES["amber"]:
            repaired_config, repair_log = repair_until_feasible(initial_config, hw, model_path)
            self.assertGreater(len(repair_log), 0, "Repair should be logged")
        else:
            repaired_config = initial_config
            repair_log = ["No repair needed"]
        
        # Step 3: Verify repaired config fits
        repaired_breakdown = _estimate_vram_breakdown(model_path, repaired_config, hw)
        repaired_total = repaired_breakdown["total"]
        self.assertLess(repaired_total, budget * VRAM_ZONES["amber"],
                       "Repaired config should fit in amber zone")
        
        # Step 4: Create and serialize profile
        profile = {
            "config": repaired_config,
            "repair_log": repair_log,
            "vram_breakdown": repaired_breakdown,
        }
        
        profile_json = json.dumps(profile, ensure_ascii=False, default=str)
        restored_profile = json.loads(profile_json)
        
        # Step 5: Verify restored profile is usable
        self.assertEqual(restored_profile["config"]["ctx_size"], repaired_config["ctx_size"])
        self.assertEqual(len(restored_profile["repair_log"]), len(repair_log))


if __name__ == "__main__":
    unittest.main()
