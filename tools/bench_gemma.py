#!/usr/bin/env python3
import requests
import json
import time
import sys

# Target remote server
BASE_URL = "http://192.168.110.50:11435"
MODEL_ID = "speculative-gemma-4-31b-it-claude-opus-distill.q8_0"

def wait_for_operation(target_state="idle"):
    print(f"Monitoring progress for {MODEL_ID}...")
    while True:
        try:
            resp = requests.get(f"{BASE_URL}/api/debug/health", timeout=5)
            health = resp.json()
            
            op = health.get("operation_state", "idle")
            progress = health.get("operation_progress", 0)
            status = health.get("status", "unknown")
            
            if op:
                print(f"  [Progress] {op}: {progress}%")
            else:
                print(f"  [Status] {status}")
            
            if op == target_state or (not op and target_state == "idle"):
                print("Operation finished.")
                return health
            
            if health.get("last_error"):
                print(f"Error observed: {health['last_error']}")
                return health
                
            time.sleep(2)
        except Exception as e:
            print(f"Error polling health: {e}")
            time.sleep(2)

def benchmark_config(flags, name):
    print(f"\n--- Benchmarking: {name} ---")
    print(f"Loading config with flags: {json.dumps(flags)}")
    
    # Use the new async loading
    try:
        resp = requests.post(
            f"{BASE_URL}/api/debug/load-config",
            json={
                "model_id": MODEL_ID,
                "flags": flags,
                "async": True,
                "force": True
            },
            timeout=10
        )
        print(f"Response: {resp.json()}")
    except Exception as e:
        print(f"Failed to trigger load: {e}")
        return
    
    # Wait for ready
    health = wait_for_operation("idle")
    if health.get("status") != "active":
        print("Failed to activate session.")
        return

    # Run a simple benchmark request
    print("Running inference benchmark...")
    # Note: In a real scenario, we'd call the /v1/completions or similar endpoint
    # Here we just check the metrics snapshot
    try:
        resp = requests.get(f"{BASE_URL}/api/debug/gpu-status")
        metrics = resp.json()
        print(f"Current Metrics: {json.dumps(metrics, indent=2)}")
    except Exception as e:
        print(f"Failed to get metrics: {e}")

def main():
    print(f"Starting Gemma Speculative Optimization Benchmark on {BASE_URL}")
    
    # Test 1: Baseline (Default)
    benchmark_config({}, "Baseline")
    
    # Test 2: Optimized KV Cache (q8_0) + Flash Attention + Safe Draft Ctx
    benchmark_config({
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "flash_attn": "on",
        "n_gpu_layers": 999,
        "ctx_size_draft": 8192,  # Reduce from default to avoid OOM
        "n_gpu_layers_draft": 999,
        "cache_type_k_draft": "q8_0", # New proposed flag for draft cache
        "cache_type_v_draft": "q8_0"
    }, "Quantized KV + Flash Attention + Safe Draft")
    
    # Test 3: Maximum Efficiency (q4_0 KV Cache)
    benchmark_config({
        "cache_type_k": "q4_0",
        "cache_type_v": "q4_0",
        "flash_attn": "on",
        "n_gpu_layers": 999,
        "ctx_size_draft": 4096,
        "n_gpu_layers_draft": 999,
        "cache_type_k_draft": "q4_0",
        "cache_type_v_draft": "q4_0"
    }, "Aggressive Quantization (q4_0)")

if __name__ == "__main__":
    main()
