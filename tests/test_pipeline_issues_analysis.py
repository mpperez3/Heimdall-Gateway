#!/usr/bin/env python3
"""
Deep analysis of pipeline bugs, edge cases, and problems.
"""

import json
from pathlib import Path

def analyze_pipeline_issues():
    """Analyze and report issues found in the autotuning pipeline."""
    print("\n" + "=" * 100)
    print("AUTOTUNING PIPELINE DEEP ANALYSIS: BUGS & EDGE CASES")
    print("=" * 100)
    
    issues = []
    
    # =========================================================================
    # ISSUE 1: Hardware fingerprint returns UNKNOWN for CPU count and VRAM
    # =========================================================================
    print("\n[ISSUE 1] Hardware Fingerprint Incomplete Information")
    print("-" * 100)
    
    from llamacpp_stack.auto_performance import _hardware_fingerprint
    
    hw = _hardware_fingerprint(4096)
    print(f"Hardware fingerprint result:")
    print(f"  - fingerprint: {hw.get('fingerprint', 'UNKNOWN')[:32]}...")
    print(f"  - cpu_count: {hw.get('cpu_count', 'UNKNOWN')}")
    print(f"  - vram_budget_mib: {hw.get('vram_budget_mib', 'UNKNOWN')}")
    
    if hw.get('cpu_count') == 'UNKNOWN' or hw.get('vram_budget_mib') == 'UNKNOWN':
        print(f"⚠️  PROBLEM: Hardware information missing!")
        print(f"   Impact: Cannot validate VRAM constraints or CPU threading options")
        print(f"   This means the system doesn't know how much headroom it has for config adjustment")
        issues.append({
            "severity": "HIGH",
            "component": "Hardware Detection",
            "issue": "CPU count and VRAM budget return UNKNOWN",
            "impact": "Cannot validate hardware constraints or provide meaningful limits",
            "fix": "Implement proper CPU and VRAM detection in _hardware_fingerprint()"
        })
    
    # =========================================================================
    # ISSUE 2: Error messages ambiguous - doesn't distinguish causes clearly
    # =========================================================================
    print("\n[ISSUE 2] Error Message Clarity & User Guidance")
    print("-" * 100)
    
    print("Current error handling scenarios:")
    print("  - timeout: User sees 'Server failed to become ready' (could be OOM, crash, or actual timeout)")
    print("  - crash: User sees 'Workload measurement failed' (vague, no root cause)")
    print("  - oom: Detected in crash scenario by looking for 'out of memory' in stderr")
    print("")
    
    problems = [
        ("No clear distinction between model load failure vs workload failure",
         "User can't tell if config failed to load model or failed during measurements"),
        ("Load timeout (300s!) same message as health timeout",
         "User doesn't know if their model actually loaded or timed out waiting"),
        ("Repair mechanism not logged clearly",
         "When repair adjusts config, user sees original params in output, not repaired ones"),
        ("Mid-inference pruning doesn't explain why test was pruned",
         "User sees '✂️ Trial pruned early' but doesn't know it was low throughput vs OOM risk"),
    ]
    
    for prob, impact in problems:
        print(f"  ⚠️  {prob}")
        print(f"      → {impact}")
    
    issues.append({
        "severity": "MEDIUM",
        "component": "Logging & User Feedback",
        "issue": "Error messages are ambiguous and don't provide root cause analysis",
        "impact": "Users can't understand why their configs fail or how to fix them",
        "fix": "Add detailed logging for each failure mode with clear explanations"
    })
    
    # =========================================================================
    # ISSUE 3: Configuration applied but not clearly communicated
    # =========================================================================
    print("\n[ISSUE 3] Configuration Application Clarity")
    print("-" * 100)
    
    print("At the end of tuning, system asks:")
    print("  1. '¿Quieres ejecutar ahora la fase 2 de servidor?' (raw inference only)")
    print("  2. '¿Aplicar esta configuracion al modelo?' (apply to catalog)")
    print("")
    print("PROBLEM: Unclear what happens if user says 'no'")
    print("  - Config is saved to profiles.json (good)")
    print("  - Config is NOT applied to catalog (expected?)")
    print("  - Config is NOT actively used (user needs to load it manually?)")
    print("  - User has no clear guidance on next steps")
    print("")
    
    issues.append({
        "severity": "MEDIUM",
        "component": "User Interface / Configuration Application",
        "issue": "Unclear what happens when user declines to apply config to catalog",
        "impact": "Users may think config is lost or wonder how to use tuned config",
        "fix": "Add clear messaging about where config was saved and how to apply it"
    })
    
    # =========================================================================
    # ISSUE 4: Profile uniqueness key should include context size
    # =========================================================================
    print("\n[ISSUE 4] Profile Keying Strategy")
    print("-" * 100)
    
    from llamacpp_stack.auto_performance import AUTO_PERF_PROFILES_PATH
    import hashlib
    
    profile_key_payload = {
        "model": "test-model",
        "quant": "q8_0",
        "llama_cpp_version": "0.2.48",
        "hardware_fingerprint": "Linux-x86_64-409",
    }
    
    # Note: Context size is NOT in the key!
    profile_key = hashlib.sha256(json.dumps(profile_key_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    
    print(f"Current profile key includes: {list(profile_key_payload.keys())}")
    print(f"PROBLEM: Context size is NOT included in profile key!")
    print(f"  - If you tune at ctx=4096, save profile key X")
    print(f"  - Then tune same model at ctx=8192, it overwrites the same profile key!")
    print(f"  - You lose the ctx=4096 configuration")
    print("")
    
    issues.append({
        "severity": "HIGH",
        "component": "Profile Management",
        "issue": "Profile key doesn't include context size - different contexts overwrite each other",
        "impact": "Can only keep one profile per (model, quant, llama_cpp_version, hardware) - lose configs for other contexts",
        "fix": "Add ctx_size to profile_key_payload before hashing"
    })
    
    # =========================================================================
    # ISSUE 5: No tracking of when config was last tuned
    # =========================================================================
    print("\n[ISSUE 5] Stale Configuration Detection")
    print("-" * 100)
    
    print("PROBLEM: No indication of when configuration was tuned")
    print("  - Scenario: Tune model on config A (hardware=X, llama_cpp version Y.1)")
    print("  - Later: Update llama.cpp to version Y.2 (performance characteristics may change)")
    print("  - System: Still uses old profile, doesn't warn that config is stale")
    print("")
    
    issues.append({
        "severity": "MEDIUM",
        "component": "Profile Staleness",
        "issue": "No detection of stale configs when hardware/llama_cpp changes",
        "impact": "Users may be using outdated tuning that no longer applies to current environment",
        "fix": "Track tuning date and warn if > 30 days old, add version change detection"
    })
    
    # =========================================================================
    # ISSUE 6: Repair mechanism not visible to user
    # =========================================================================
    print("\n[ISSUE 6] Repair Mechanism Opacity")
    print("-" * 100)
    
    print("PROBLEM: When repair adjusts config, user doesn't see what was changed")
    print("  Current: repair_until_feasible() adjusts params in-place")
    print("     User: Sees trial fail with original params")
    print("     Log: TRIAL_RESULT {metrics, params} - shows REPAIRED params")
    print("     Output: Shows repaired metrics (decode_tokens_s, etc.)")
    print("     User confusion: 'Why did my 8192 batch_size produce these tokens/s? That's wrong!'")
    print("")
    
    issues.append({
        "severity": "MEDIUM",
        "component": "Repair & Adjustment Tracking",
        "issue": "When repair adjusts config, changes not clearly communicated to user",
        "impact": "User sees metrics that don't match their specified parameters",
        "fix": "Log repair adjustments clearly, show delta between requested and actual config"
    })
    
    # =========================================================================
    # ISSUE 7: No resource cleanup on crash
    # =========================================================================
    print("\n[ISSUE 7] Resource Leak: Processes Not Cleaned Up on Crash")
    print("-" * 100)
    
    print("PROBLEM: server_proc might not be terminated on benchmark failure")
    print("  Scenario:")
    print("    1. run_benchmark() starts llama-server subprocess")
    print("    2. Health check timeout occurs")
    print("    3. Function returns early (cleanup skipped)")
    print("    4. Subprocess still running, consuming GPU VRAM")
    print("")
    print("Code location: run_benchmark() has try/except but no finally block for cleanup")
    print("")
    
    issues.append({
        "severity": "HIGH",
        "component": "Resource Management",
        "issue": "Server process may not be terminated on early return/timeout",
        "impact": "GPU VRAM leaks when benchmarks timeout, subsequent trials OOM",
        "fix": "Wrap server_proc lifecycle in try/finally to ensure cleanup"
    })
    
    # =========================================================================
    # ISSUE 8: No validation that model file actually exists and is readable
    # =========================================================================
    print("\n[ISSUE 8] Input Validation: Model File Validation")
    print("-" * 100)
    
    print("PROBLEM: No check that model file is valid before starting 20+ trials")
    print("  Current: run_auto_performance() immediately attempts benchmarks")
    print("  Better: Validate model exists, is readable, has reasonable size before starting")
    print("  Impact: User wastes time if model path is wrong, doesn't find out until first trial fails")
    print("")
    
    issues.append({
        "severity": "MEDIUM",
        "component": "Input Validation",
        "issue": "No pre-flight check that model file is valid",
        "impact": "User may wait for first benchmark to fail before realizing model path is wrong",
        "fix": "Add validation step at start of run_auto_performance()"
    })
    
    # =========================================================================
    # ISSUE 9: Scoring function can return 0 or negative when things are bad
    # =========================================================================
    print("\n[ISSUE 9] Scoring Edge Case: OOM returns -1000, not clearly explained")
    print("-" * 100)
    
    print("PROBLEM: Scoring returns -1000 for any failure, no differentiation")
    print("  - Full OOM: Score -1000")
    print("  - Crash: Score -1000")
    print("  - Timeout: Score -1000")
    print("  User perspective: All bad configs are equal, but actually:")
    print("    - OOM means config is too large (fixable by repair)")
    print("    - Timeout means model takes > 300s to load (might need different strategy)")
    print("    - Crash means unknown problem (could be any parameter combo)")
    print("")
    
    issues.append({
        "severity": "MEDIUM",
        "component": "Scoring & Differentiation",
        "issue": "All failure modes score -1000, loses diagnostic information",
        "impact": "Repair mechanism can't prioritize fixes based on failure cause",
        "fix": "Return different penalty scores for different failure modes"
    })
    
    # =========================================================================
    # ISSUE 10: No phase transition validation
    # =========================================================================
    print("\n[ISSUE 10] Phase Transition Validation")
    print("-" * 100)
    
    print("PROBLEM: Phase 2 inherits Phase 1 best config without validation")
    print("  - Phase 1: Finds best params for raw inference")
    print("  - Phase 2: Uses those params, adds server-specific flags")
    print("  - Issue: Phase 2 trials fail due to Phase 1 params being incompatible with server")
    print("    Example: Phase 1 chose n_gpu_layers='auto', Phase 2 can't apply '8' parallel slots to 'auto'")
    print("")
    
    issues.append({
        "severity": "MEDIUM",
        "component": "Phase Transition",
        "issue": "Phase 1 config not validated to work with Phase 2 parameters",
        "impact": "Phase 2 may fail immediately when trying to inherit Phase 1 config",
        "fix": "Run quick validation trial with Phase 2 before starting optimization"
    })
    
    # =========================================================================
    # Print Summary
    # =========================================================================
    print("\n" + "=" * 100)
    print("ISSUE SUMMARY")
    print("=" * 100)
    
    high_severity = [i for i in issues if i["severity"] == "HIGH"]
    medium_severity = [i for i in issues if i["severity"] == "MEDIUM"]
    
    print(f"\nHIGH SEVERITY: {len(high_severity)}")
    for i, issue in enumerate(high_severity, 1):
        print(f"  {i}. {issue['component']}: {issue['issue']}")
        print(f"     Impact: {issue['impact']}")
    
    print(f"\nMEDIUM SEVERITY: {len(medium_severity)}")
    for i, issue in enumerate(medium_severity, 1):
        print(f"  {i}. {issue['component']}: {issue['issue']}")
        print(f"     Impact: {issue['impact']}")
    
    print(f"\nTOTAL ISSUES FOUND: {len(issues)}")
    
    # =========================================================================
    # Recommendations
    # =========================================================================
    print("\n" + "=" * 100)
    print("PRIORITY FIXES")
    print("=" * 100)
    
    print(f"""
1. [CRITICAL] Resource Cleanup (Issue #7)
   - Impact: GPU memory leaks on timeout
   - Fix: Add try/finally to server_proc lifecycle
   
2. [HIGH] Profile Key Includes Context Size (Issue #4)
   - Impact: Lose configs when tuning different contexts
   - Fix: Add ctx_size to profile_key_payload
   
3. [MEDIUM] Better Error Messages (Issue #2)
   - Impact: Users can't diagnose failures
   - Fix: Log root cause for each failure type
   
4. [MEDIUM] Repair Mechanism Visibility (Issue #6)
   - Impact: User confusion about actual vs intended config
   - Fix: Log adjusted parameters clearly
   
5. [MEDIUM] Model File Validation (Issue #8)
   - Impact: Wasted time on invalid models
   - Fix: Check file exists and is readable before starting
""")
    
    return True

if __name__ == "__main__":
    try:
        success = analyze_pipeline_issues()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
