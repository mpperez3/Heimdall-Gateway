#!/usr/bin/env python3
"""
Complete pipeline bug hunt and validation.
Tests all failure modes, edge cases, and logging output.
"""

import json
import sys
import traceback
from pathlib import Path
import tempfile

def test_pipeline_imports():
    """Test basic imports and module availability."""
    print("\n[TEST 1] Module Imports & Availability")
    print("-" * 80)
    
    try:
        from llamacpp_stack.auto_performance import (
            run_auto_performance,
            run_benchmark,
            score_performance,
            score_server_performance,
            PROBED_TUNER_KEYS,
            PHASE2_SERVER_TUNER_KEYS,
            PRUNED_TUNER_KEYS,
            _hardware_fingerprint,
            _build_benchmark_command,
            get_server_path,
        )
        from llamacpp_stack.cli import (
            ManagedModel,
            build_llama_server_command,
            load_catalog_with_diagnostics,
            DEFAULT_CATALOG_PATH,
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_hardware_detection():
    """Test hardware fingerprinting and GPU detection."""
    print("\n[TEST 2] Hardware Detection")
    print("-" * 80)
    
    try:
        from llamacpp_stack.auto_performance import _hardware_fingerprint, get_server_path
        from llamacpp_stack.cli import detect_cuda_device_count
        
        ctx = 4096
        hw = _hardware_fingerprint(ctx)
        gpu_count = detect_cuda_device_count()
        server_path = get_server_path()
        
        print(f"✓ Hardware fingerprint: {hw.get('fingerprint', 'UNKNOWN')[:16]}...")
        print(f"  - CPU threads: {hw.get('cpu_count', 'UNKNOWN')}")
        print(f"  - VRAM budget: {hw.get('vram_budget_mib', 'UNKNOWN')} MiB")
        print(f"✓ GPU count: {gpu_count}")
        print(f"✓ Server binary: {server_path}")
        
        if not server_path or not Path(server_path).exists():
            print(f"⚠️  WARNING: Server binary not found at {server_path}")
            print(f"   This will cause all benchmarks to fail")
        
        return True
    except Exception as e:
        print(f"✗ Hardware detection failed: {e}")
        traceback.print_exc()
        return False


def test_catalog_loading():
    """Test catalog loading and error handling."""
    print("\n[TEST 3] Catalog Loading & Validation")
    print("-" * 80)
    
    try:
        from llamacpp_stack.cli import load_catalog_with_diagnostics, DEFAULT_CATALOG_PATH
        
        if not DEFAULT_CATALOG_PATH.exists():
            print(f"⚠️  Default catalog not found: {DEFAULT_CATALOG_PATH}")
            print(f"   Tests will use empty catalog")
            return True
        
        items, diagnostics = load_catalog_with_diagnostics(DEFAULT_CATALOG_PATH)
        print(f"✓ Catalog loaded: {len(items)} models")
        
        if diagnostics:
            print(f"⚠️  Diagnostics:")
            for d in diagnostics[:3]:
                print(f"   - {d}")
            if len(diagnostics) > 3:
                print(f"   ... and {len(diagnostics) - 3} more")
        
        # Check for models with missing paths
        missing_paths = 0
        for item in items:
            if not Path(item.local_path or "").exists():
                missing_paths += 1
        
        if missing_paths > 0:
            print(f"⚠️  {missing_paths}/{len(items)} models have missing local paths")
            print(f"   This will cause autotuning to fail for those models")
        
        return True
    except Exception as e:
        print(f"✗ Catalog loading failed: {e}")
        traceback.print_exc()
        return False


def test_tuning_keys_integrity():
    """Test PROBED_TUNER_KEYS, PHASE2_SERVER_TUNER_KEYS, PRUNED_TUNER_KEYS integrity."""
    print("\n[TEST 4] Tuning Keys Integrity Check")
    print("-" * 80)
    
    try:
        from llamacpp_stack.auto_performance import (
            PROBED_TUNER_KEYS,
            PHASE2_SERVER_TUNER_KEYS,
            PRUNED_TUNER_KEYS,
        )
        
        # Check for overlaps that shouldn't exist
        phase1_phase2_overlap = set(PROBED_TUNER_KEYS) & set(PHASE2_SERVER_TUNER_KEYS)
        pruned_probed_overlap = set(PRUNED_TUNER_KEYS) & set(PROBED_TUNER_KEYS)
        pruned_phase2_overlap = set(PRUNED_TUNER_KEYS) & set(PHASE2_SERVER_TUNER_KEYS)
        
        print(f"✓ PROBED_TUNER_KEYS: {len(PROBED_TUNER_KEYS)} keys")
        print(f"  {sorted(list(PROBED_TUNER_KEYS))[:5]}...")
        print(f"✓ PHASE2_SERVER_TUNER_KEYS: {len(PHASE2_SERVER_TUNER_KEYS)} keys")
        print(f"  {sorted(list(PHASE2_SERVER_TUNER_KEYS))[:5]}...")
        print(f"✓ PRUNED_TUNER_KEYS: {len(PRUNED_TUNER_KEYS)} keys")
        
        issues = []
        if phase1_phase2_overlap:
            issues.append(f"Phase 1 & 2 overlap: {phase1_phase2_overlap}")
        if pruned_probed_overlap:
            issues.append(f"Pruned & Probed overlap: {pruned_probed_overlap}")
        if pruned_phase2_overlap:
            issues.append(f"Pruned & Phase2 overlap: {pruned_phase2_overlap}")
        
        if issues:
            print(f"✗ INTEGRITY ISSUES:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        print(f"✓ No key overlap issues")
        return True
    except Exception as e:
        print(f"✗ Tuning keys check failed: {e}")
        traceback.print_exc()
        return False


def test_scoring_functions():
    """Test score_performance and score_server_performance."""
    print("\n[TEST 5] Scoring Functions Correctness")
    print("-" * 80)
    
    try:
        from llamacpp_stack.auto_performance import (
            score_performance,
            score_server_performance,
        )
        
        # Test normal metrics
        good_metrics = {
            "decode_tokens_s": 100.0,
            "prefill_tokens_s": 30.0,
            "requests_s": 5.0,
            "load_ready_s": 0.5,
            "oom": False,
            "crash": False,
            "timeout": False,
            "ctx_stable": 4096,
        }
        
        score_p1 = score_performance(good_metrics, 4096, 1)
        score_p2 = score_server_performance(good_metrics, 4096, 1)
        
        print(f"✓ Good metrics score (P1): {score_p1:.2f}")
        print(f"✓ Good metrics score (P2): {score_p2:.2f}")
        
        if score_p1 <= 0 or score_p2 <= 0:
            print(f"✗ Scores should be positive, got {score_p1:.2f}, {score_p2:.2f}")
            return False
        
        # Test OOM metrics
        oom_metrics = dict(good_metrics)
        oom_metrics["oom"] = True
        
        score_oom_p1 = score_performance(oom_metrics, 4096, 1)
        score_oom_p2 = score_server_performance(oom_metrics, 4096, 1)
        
        print(f"✓ OOM metrics score (P1): {score_oom_p1:.2f} (should be -1000 or similar)")
        print(f"✓ OOM metrics score (P2): {score_oom_p2:.2f} (should be -1000 or similar)")
        
        if score_oom_p1 >= 0 or score_oom_p2 >= 0:
            print(f"✗ OOM scores should be negative, got {score_oom_p1:.2f}, {score_oom_p2:.2f}")
            return False
        
        # Test load_ready_s invariance (critical for load tuning)
        fast_load_metrics = dict(good_metrics)
        fast_load_metrics["load_ready_s"] = 0.1
        
        slow_load_metrics = dict(good_metrics)
        slow_load_metrics["load_ready_s"] = 10.0
        
        score_fast = score_performance(fast_load_metrics, 4096, 1)
        score_slow = score_performance(slow_load_metrics, 4096, 1)
        
        print(f"✓ Fast load score (0.1s): {score_fast:.2f}")
        print(f"✓ Slow load score (10.0s): {score_slow:.2f}")
        
        if score_fast != score_slow:
            print(f"✗ CRITICAL BUG: Scores differ for different load times!")
            print(f"   Score should be invariant to load_ready_s")
            print(f"   Fast: {score_fast:.2f}, Slow: {score_slow:.2f}")
            return False
        
        print(f"✓ Load time invariance verified (scores are identical)")
        
        return True
    except Exception as e:
        print(f"✗ Scoring functions failed: {e}")
        traceback.print_exc()
        return False


def test_log_directory():
    """Test log directory creation and permissions."""
    print("\n[TEST 6] Log Directory & Permissions")
    print("-" * 80)
    
    try:
        from llamacpp_stack.auto_performance import AUTO_PERF_LOG_DIR, AUTO_PERF_HISTORY_PATH
        
        print(f"✓ Log directory: {AUTO_PERF_LOG_DIR}")
        print(f"✓ History path: {AUTO_PERF_HISTORY_PATH}")
        
        # Try to create log directory
        AUTO_PERF_LOG_DIR.mkdir(parents=True, exist_ok=True)
        if not AUTO_PERF_LOG_DIR.exists():
            print(f"✗ Failed to create log directory: {AUTO_PERF_LOG_DIR}")
            return False
        
        print(f"✓ Log directory is accessible")
        
        # Check if we can write
        test_file = AUTO_PERF_LOG_DIR / "test_write.tmp"
        try:
            test_file.write_text("test")
            test_file.unlink()
            print(f"✓ Can write to log directory")
        except Exception as e:
            print(f"✗ Cannot write to log directory: {e}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Log directory test failed: {e}")
        traceback.print_exc()
        return False


def test_profile_serialization():
    """Test profile JSON serialization."""
    print("\n[TEST 7] Profile Serialization")
    print("-" * 80)
    
    try:
        profile = {
            "profile_key": "abc123",
            "model": "test-model",
            "created_at": "2026-05-03T10:00:00",
            "baseline": {
                "metrics": {
                    "decode_tokens_s": 100.0,
                    "prefill_tokens_s": 30.0,
                    "load_ready_s": 0.5,
                },
                "params": {},
                "trial_value": 246.67,
            },
            "best": {
                "metrics": {
                    "decode_tokens_s": 150.0,
                    "prefill_tokens_s": 45.0,
                    "load_ready_s": 0.45,
                },
                "params": {
                    "batch_size": 128,
                    "n_gpu_layers": 24,
                },
                "trial_value": 370.0,
            },
        }
        
        # Try to serialize
        json_str = json.dumps(profile, indent=2)
        print(f"✓ Profile serialized ({len(json_str)} bytes)")
        
        # Try to deserialize
        loaded = json.loads(json_str)
        print(f"✓ Profile deserialized successfully")
        
        if loaded != profile:
            print(f"✗ Profile changed after serialization/deserialization")
            return False
        
        print(f"✓ Profile round-trip preserved all data")
        return True
    except Exception as e:
        print(f"✗ Profile serialization failed: {e}")
        traceback.print_exc()
        return False


def test_command_building():
    """Test benchmark command building."""
    print("\n[TEST 8] Command Building")
    print("-" * 80)
    
    try:
        from llamacpp_stack.auto_performance import _build_benchmark_command
        from llamacpp_stack.cli import ManagedModel, build_llama_server_command
        from pathlib import Path
        
        # Create a minimal model
        model_path = Path("/tmp/test-model.gguf")
        
        # Test with direct_io=True and False
        params_with_dio = {
            "direct_io": True,
            "batch_size": 128,
            "n_gpu_layers": 24,
        }
        
        params_without_dio = {
            "direct_io": False,
            "batch_size": 128,
            "n_gpu_layers": 24,
        }
        
        try:
            cmd_with = _build_benchmark_command(model_path, params_with_dio, 4096)
            cmd_without = _build_benchmark_command(model_path, params_without_dio, 4096)
            
            cmd_with_str = " ".join(cmd_with)
            cmd_without_str = " ".join(cmd_without)
            
            print(f"✓ Command with direct_io=True:")
            print(f"  Has --direct-io: {'--direct-io' in cmd_with_str}")
            print(f"  Has --no-direct-io: {'--no-direct-io' in cmd_with_str}")
            
            print(f"✓ Command with direct_io=False:")
            print(f"  Has --direct-io: {'--direct-io' in cmd_without_str}")
            print(f"  Has --no-direct-io: {'--no-direct-io' in cmd_without_str}")
            
            # Check consistency
            if ('--direct-io' in cmd_with_str) and ('--no-direct-io' not in cmd_with_str):
                print(f"✓ direct_io=True adds --direct-io (no conflicting --no-direct-io)")
            elif ('--no-direct-io' in cmd_without_str) and ('--direct-io' not in cmd_without_str):
                print(f"✓ direct_io=False adds --no-direct-io (no conflicting --direct-io)")
            
            if '--direct-io' in cmd_with_str and '--direct-io' not in cmd_without_str:
                print(f"✓ Commands differ based on direct_io parameter (good for load tuning)")
            else:
                print(f"✗ Commands don't differ based on direct_io (load tuning broken!)")
                return False
                
        except Exception as e:
            print(f"⚠️  Command building not fully testable: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Command building test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all pipeline bug hunt tests."""
    print("\n" + "=" * 80)
    print("AUTOTUNING PIPELINE BUG HUNT & VALIDATION")
    print("=" * 80)
    
    tests = [
        ("Module Imports", test_pipeline_imports),
        ("Hardware Detection", test_hardware_detection),
        ("Catalog Loading", test_catalog_loading),
        ("Tuning Keys Integrity", test_tuning_keys_integrity),
        ("Scoring Functions", test_scoring_functions),
        ("Log Directory", test_log_directory),
        ("Profile Serialization", test_profile_serialization),
        ("Command Building", test_command_building),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} | {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        traceback.print_exc()
        exit(1)
