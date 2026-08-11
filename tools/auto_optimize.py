import os
import requests
import time
import json
import sys

# Remote Debug Server
MANAGER_URL = os.environ.get("HEIMDALL_GATEWAY_URL", "http://127.0.0.1:11435").rstrip("/")
MODEL_ID = "speculative-gemma-4-31b-it-claude-opus-distill.q8_0"

# Exclude Device 6 entirely from tensor split as it was OOMing on compute buffers
TENSOR_SPLIT = "1,1,1,1,1,1,0" 

def wait_for_health():
    print("⏳ Waiting for model to be healthy...")
    start_time = time.time()
    while time.time() - start_time < 600: # 10 min max
        try:
            resp = requests.get(f"{MANAGER_URL}/api/debug/health", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status")
                if status == "active":
                    print("✅ Model is healthy!")
                    return True
                elif status == "failed":
                    print(f"❌ Model failed to load: {data.get('error') or data.get('last_error')}")
                    return False
                
                progress = data.get("operation_progress", 0)
                print(f"   Status: {status} ({progress}%)...")
            else:
                 print(f"   Health status code: {resp.status_code}")
        except Exception as e:
            # print(f"   (Health check error: {e})")
            pass
        time.sleep(5)
    print("❌ Timeout waiting for health")
    return False

def run_test(config):
    print(f"\n🚀 Testing config: {config}")
    
    # 1. Unload previous
    try:
        requests.post(f"{MANAGER_URL}/api/debug/unload", timeout=5)
    except:
        pass
    time.sleep(5)
    
    # 2. Load with new config
    load_payload = {
        "model_id": MODEL_ID,
        "flags": {
            **config,
            "n_gpu_layers": 100,
            "tensor_split": TENSOR_SPLIT,
        },
        "async": True,
        "force": True
    }
    resp = requests.post(f"{MANAGER_URL}/api/debug/load-config", json=load_payload)
    if resp.status_code != 200:
        print(f"❌ Failed to trigger load: {resp.text}")
        return None
    
    # 3. Wait for startup
    if not wait_for_health():
        return None
    
    # 4. Run benchmark
    print("📊 Running benchmark...")
    bench_payload = {
        "message": "Write a 200-word essay about the importance of open-source AI.",
        "max_tokens": 150,
        "n_runs": 3
    }
    try:
        resp = requests.post(f"{MANAGER_URL}/api/debug/run-benchmark", json=bench_payload, timeout=240)
        if resp.status_code == 200:
            result = resp.json()
            print(f"✨ Result: {result.get('average_tps')} TPS")
            return result
        else:
            print(f"❌ Benchmark failed: {resp.text}")
            return None
    except Exception as e:
        print(f"❌ Benchmark error: {e}")
        return None

def main():
    final_results = []
    
    to_test = [
        # Baseline with fixed split
        {"spec_draft_n_max": 0, "ubatch_size": 512, "flash_attn": "off"},
        
        # Performance incremental
        {"spec_draft_n_max": 0, "ubatch_size": 512, "flash_attn": "on"},
        {"spec_draft_n_max": 8, "ubatch_size": 1024, "flash_attn": "on"},
        {"spec_draft_n_max": 16, "ubatch_size": 2048, "flash_attn": "on", "cache_type_k": "q4_0"},
        {"spec_draft_n_max": 8, "ubatch_size": 2048, "flash_attn": "on", "threads_batch": 32},
    ]

    for config in to_test:
        try:
            res = run_test(config)
            if res and res.get("average_tps") is not None:
                record = {
                    "config": config,
                    "tps": res.get("average_tps"),
                    "details": res
                }
                final_results.append(record)
                with open("bench_results_v7.json", "w") as f:
                    json.dump(final_results, f, indent=2)
            else:
                print(f"⚠️ Test failed for {config}")
        except Exception as e:
            print(f"❌ Critical error: {e}")
            
    if final_results:
        print("\n🏆 RESULTS SUMMARY:")
        for r in sorted(final_results, key=lambda x: x['tps'], reverse=True):
            print(f"TPS: {r['tps']:.2f} | {r['config']}")

if __name__ == "__main__":
    main()
