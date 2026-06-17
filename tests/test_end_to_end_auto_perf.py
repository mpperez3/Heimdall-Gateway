#!/usr/bin/env python3
"""
End-to-end test of run_auto_performance with mock benchmark.
This simulates a complete tuning pipeline to catch runtime errors
that unit tests might miss.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llamacpp_stack.auto_performance import (
    run_auto_performance,
    run_benchmark,
)


def test_auto_performance_end_to_end():
    """Test auto-performance tuning pipeline with mock models and benchmarks."""
    
    # Create a mock catalog entry
    mock_model = MagicMock()
    mock_model.model_id = "test-model"
    mock_model.repo_id = "test/repo"
    mock_model.quant = "q8_0"
    mock_model.ctx_size = 8192
    mock_model.local_path = "/tmp/test_model.gguf"
    mock_model.server_overrides = {}
    mock_model.speculative = False
    mock_model.spec_meta = {}
    
    # Create mock args
    mock_args = MagicMock()
    mock_args.catalog = None
    mock_args.repo = None
    mock_args.hf = None
    mock_args.model_id = mock_model.model_id
    mock_args.file = None
    mock_args.server_api = False
    mock_args.load_concurrency = 1
    mock_args.load_requests = 1
    mock_args.auto_perf_log = None
    mock_args.mock = True  # Use mock benchmarking
    mock_args._question_callback = lambda prompt, default: "n"  # Don't apply to catalog
    
    # Patch dependencies
    with patch("llamacpp_stack.auto_performance.load_catalog_with_diagnostics") as mock_catalog, \
         patch("llamacpp_stack.auto_performance.resolve_catalog_model") as mock_resolve, \
         patch("llamacpp_stack.auto_performance.read_gguf_metadata") as mock_gguf, \
         patch("llamacpp_stack.auto_performance.detect_cuda_device_count") as mock_gpu_count, \
         patch("llamacpp_stack.auto_performance._ensure_optuna") as mock_optuna, \
         patch("llamacpp_stack.auto_performance.os.environ.get") as mock_env, \
         patch("llamacpp_stack.auto_performance._resolve_catalog_path") as mock_cat_path, \
         patch("llamacpp_stack.auto_performance.save_catalog") as mock_save_cat:
        
        # Setup mocks
        mock_catalog.return_value = ([mock_model], None)
        mock_resolve.return_value = mock_model
        mock_gguf.return_value = {
            "architecture": "llama",
            "trained_ctx": 8192,
            "layers": 32,
            "is_moe": False
        }
        mock_gpu_count.return_value = 2
        
        # Mock optuna
        class MockTrial:
            def __init__(self):
                self.number = 0
                self._user_attrs = {}
            
            def suggest_int(self, name, low, high):
                return low if name == "gpu_set_idx" else high // 2
            
            def suggest_categorical(self, name, options):
                return options[0] if options else None
            
            def set_user_attr(self, key, value):
                self._user_attrs[key] = value
            
            @property
            def value(self):
                return 100.0
            
            @property
            def user_attrs(self):
                return self._user_attrs
        
        class MockStudy:
            def __init__(self):
                self.trials = []
                self.best_trial = MockTrial()
            
            def enqueue_trial(self, params):
                pass
            
            def optimize(self, objective, n_trials, timeout=None):
                for i in range(min(n_trials, 2)):  # Limit to 2 trials for speed
                    trial = MockTrial()
                    trial.number = i
                    self.trials.append(trial)
                    try:
                        result = objective(trial)
                        trial._value = result
                    except Exception as e:
                        print(f"Trial {i} error: {e}")
                        raise
                if self.trials:
                    self.best_trial = self.trials[-1]
        
        mock_optuna_module = MagicMock()
        mock_optuna_module.create_study.return_value = MockStudy()
        mock_optuna_module.logging.set_verbosity = MagicMock()
        mock_optuna_module.TrialPruned = Exception
        mock_optuna.return_value = mock_optuna_module
        
        mock_env.return_value = None
        mock_cat_path.return_value = Path("/tmp/test_catalog.json")
        
        # Run the pipeline
        print("\n" + "="*70)
        print("RUNNING END-TO-END AUTO-PERFORMANCE TEST")
        print("="*70)
        
        try:
            result = run_auto_performance(mock_args)
            print("\n✅ End-to-end test PASSED")
            print(f"Return code: {result}")
            
            # Check for any exceptions during trial execution
            if result == 0:
                print("✅ Auto-performance completed successfully")
                return True
            else:
                print(f"⚠️  Unexpected return code: {result}")
                return False
        except Exception as e:
            print(f"\n❌ End-to-end test FAILED with error:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = test_auto_performance_end_to_end()
    sys.exit(0 if success else 1)
