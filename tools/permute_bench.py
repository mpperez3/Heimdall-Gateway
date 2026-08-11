#!/usr/bin/env python3
import os
import requests
import json
import time
import sys

BASE_URL = os.environ.get("HEIMDALL_GATEWAY_URL", "http://127.0.0.1:11435").rstrip("/")
MODEL_ID = "speculative-gemma-4-31b-it-claude-opus-distill.q8_0"

# USER'S REVISED BASELINE (Q8, 256k Context)
USER_BASELINE = {
    "keep": 512,
    "mirostat": 2,
    "mirostat_ent": 4.5,
    "mirostat_lr": 0.1,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
    "tensor_split": "1,1,1,1,1,1,1", # User wants 1,1,1,1,1,1,1
    "batch_size": 4096,
    "ubatch_size": 2048,
    "threads": 16,
    "threads_batch": 8,
    "numa": "distribute",
    "fit_target": 1536,
    "flash_attn": "on",
    "spec_draft_n_max": 16,
    "fit_ctx": 8192,
    "ctx_size": 262144, # 256k
    "parallel": 8
}

def wait_for_ready(timeout=600): # Increased to 10 mins for 256k allocation
    start = time.time()
    print("  - Waiting for model to be ready (256k allocation takes time)...", end="", flush=True)
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{BASE_URL}/api/debug/health", timeout=10)
            health = resp.json()
            status = health.get("status")
            op = health.get("operation_state")
            
            if status == "active" and not op:
                print(" ✅ Ready!")
                return health.get("session", {})
            
            if health.get("last_error"):
                print(f" ❌ Error: {health['last_error']}")
                return None
            
            print(".", end="", flush=True)
            time.sleep(10)
        except Exception:
            print("?", end="", flush=True)
            time.sleep(10)
    print(" ❌ Timeout!")
    return None

def load_and_bench(flags, name):
    print(f"\n🚀 Testing: {name}")
    try:
        # 1. Trigger Async Load
        print(f"  - Loading 256k context configuration...")
        resp = requests.post(
            f"{BASE_URL}/api/debug/load-config",
            json={"model_id": MODEL_ID, "flags": flags, "async": True, "force": True},
            timeout=30
        )
        
        # 2. Wait for it to be active
        session = wait_for_ready()
        if not session:
            return None
        
        port = session.get("port")
        
        # 3. Measure Performance
        # Test 1: Short prompt (Generation speed)
        print("  - Measuring generation speed...")
        bench_resp = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Explain relativity."}],
                "max_tokens": 64,
                "temperature": 0
            },
            timeout=180
        )
        data = bench_resp.json()
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        duration = bench_resp.elapsed.total_seconds()
        tps = tokens / duration if duration > 0 else 0
        print(f"    TPS: {tps:.2f}")
        
        return tps
    except Exception as e:
        print(f"💥 Exception: {e}")
        return None

def main():
    results = {}
    
    # Test 1: User Baseline
    results["User Baseline (256k Q8)"] = load_and_bench(USER_BASELINE, "User Baseline")
    
    # Test 2: Optimized Long-Ctx
    # - Reduce spec-draft-n-max to 8 (better hit rate/less penalty)
    # - Increase threads-batch to 16
    # - Reduce ubatch-size to 1024 (better BW saturation)
    opt_flags = USER_BASELINE.copy()
    opt_flags.update({
        "spec_draft_n_max": 8,
        "threads_batch": 16,
        "ubatch_size": 1024,
        "batch_size": 8192,
        "defrag_threshold": 0.1
    })
    results["Opt Long-Ctx (256k Q8)"] = load_and_bench(opt_flags, "Optimized Long-Ctx")

    print("\n\n🏆 FINAL COMPARISON (256k Q8) 🏆")
    print("-" * 45)
    for name, tps in results.items():
        if tps:
            print(f"| {name:28} | {tps:6.2f} t/s |")
        else:
            print(f"| {name:28} | FAILED     |")
    print("-" * 45)

if __name__ == "__main__":
    main()
