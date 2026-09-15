"""Server command building extracted from llamacpp_stack/cli.py."""
from __future__ import annotations
import json
import os
import re
import subprocess
from pathlib import Path
from .constants import (
    DEFAULT_LLAMA_SERVER,
    PRODUCT_SLUG,
    REASONING_BUDGET_HALF_CONTEXT,
    default_tensor_split,
)
_CACHE_TYPE_ALIASES = {
    "q8": "q8_0", "8bit": "q8_0", "int8": "q8_0", "q4": "q4_0", "4bit": "q4_0", "int4": "q4_0",
    "fp16": "f16", "float16": "f16", "fp32": "f32", "float32": "f32",
}
_VALID_CACHE_TYPES = {"f32","f16","bf16","q8_0","q4_0","q4_1","iq4_nl","q5_0","q5_1","kvarn2","kvarn3","kvarn4","kvarn5","kvarn6","kvarn8","karn2","karn3","karn4","karn5","karn6","karn8","turbo8","turbo4","turbo3","turbo2","turbo3_tcq","turbo2_tcq","turbo1_tcq","vbr"}
_TURBO_CACHE_TYPES = {"turbo8","turbo4","turbo3","turbo2","turbo3_tcq","turbo2_tcq","turbo1_tcq","vbr"}
def _normalize_cache_type_value(value: object) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    text = _CACHE_TYPE_ALIASES.get(text, text)
    return text if text in _VALID_CACHE_TYPES else None
def _normalize_bool_flag(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        n = value.strip().lower()
        if n in {"1","true","yes","on"}:
            return True
        if n in {"0","false","no","off"}:
            return False
    return None
def _get_cli_file():
    try:
        import importlib.util, sys as _sys
        from pathlib import Path as _P
        mod = _sys.modules.get("llamacpp_stack._cli_file")
        if mod is not None:
            return mod
        fp = _P(__file__).parent.parent / "cli.py"
        if not fp.exists():
            return None
        spec = importlib.util.spec_from_file_location("llamacpp_stack._cli_file", fp)
        if spec is None or spec.loader is None:
            return None
        m = importlib.util.module_from_spec(spec)
        _sys.modules["llamacpp_stack._cli_file"] = m
        spec.loader.exec_module(m)
        return m
    except Exception:
        return None
def normalize_tensor_split(value: str | None) -> str:
    try:
        cf = _get_cli_file()
        if cf is not None and hasattr(cf, "normalize_tensor_split"):
            return cf.normalize_tensor_split(value)  # type: ignore
    except Exception:
        pass
    normalized = (value or "").strip()
    if not normalized:
        return default_tensor_split()
    parts = [p.strip() for p in normalized.split(",") if p.strip()]
    return ",".join(parts) if parts else default_tensor_split()
_SERVER_FLAG_CACHE: dict[str, set[str]] = {}
_VLLM_FLAG_CACHE: set[str] | None = None
_VLLM_HELP_TEXT: str | None = None
def _vllm_help_env() -> dict[str, str]:
    env = os.environ.copy()
    try:
        from pathlib import Path as _P
        vllm_bin = os.environ.get("VLLM_SERVER_BIN", "vllm-server")
        cand_paths = [_P(vllm_bin)] if _P(vllm_bin).exists() else []
        try:
            from .constants import PRODUCT_SLUG as _SLUG
            cand_paths.extend([_P.home() / ".local" / "opt" / _SLUG / "cuda" / "lib", _P.home() / ".local" / "opt" / _SLUG / "nccl" / "lib"])
        except Exception:
            pass
        existing = env.get("LD_LIBRARY_PATH", "")
        parts = [str(d) for d in cand_paths if d.exists()]
        if existing:
            parts.append(existing)
        if parts:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(parts))
    except Exception:
        pass
    return env
def get_vllm_supported_flags() -> set[str]:
    global _VLLM_FLAG_CACHE, _VLLM_HELP_TEXT
    if _VLLM_FLAG_CACHE is not None:
        return _VLLM_FLAG_CACHE
    flags: set[str] = set()
    text = ""
    try:
        vllm_bin = os.environ.get("VLLM_SERVER_BIN", "vllm-server")
        proc = subprocess.run([vllm_bin, "--help"], capture_output=True, text=True, timeout=8, env=_vllm_help_env())
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except Exception:
        _VLLM_FLAG_CACHE = set()
        _VLLM_HELP_TEXT = ""
        return set()
    _VLLM_HELP_TEXT = text
    for m in re.findall(r"(--[A-Za-z0-9-]+)", text):
        flags.add(m)
    _VLLM_FLAG_CACHE = flags
    return flags
def vllm_server_supports_flag(flag: str) -> bool:
    try:
        return flag in get_vllm_supported_flags()
    except Exception:
        return False
def get_vllm_help_text() -> str:
    if _VLLM_HELP_TEXT is not None:
        return _VLLM_HELP_TEXT
    get_vllm_supported_flags()
    return _VLLM_HELP_TEXT or ""
def _server_help_env(server_path: Path | str | None) -> dict[str, str]:
    env = os.environ.copy()
    try:
        p = Path(server_path) if server_path is not None else Path(DEFAULT_LLAMA_SERVER)
        resolved = p.resolve()
        lib_dirs = [p.parent, resolved.parent, p.parent.parent/"lib", p.parent.parent/"lib64", p.parent/"cuda"/"lib", p.parent.parent/"cuda"/"lib", p.parent/"nccl"/"lib", p.parent.parent/"nccl"/"lib", p.parent.parent/"build"/"bin", Path.home()/".local"/"opt"/PRODUCT_SLUG/"cuda"/"lib", Path.home()/".local"/"opt"/PRODUCT_SLUG/"nccl"/"lib"]
        existing = env.get("LD_LIBRARY_PATH","")
        parts = [str(d) for d in lib_dirs if d.exists()]
        if existing:
            parts.append(existing)
        if parts:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(parts))
    except Exception:
        pass
    return env
def get_server_supported_flags(server_path: Path | str | None) -> set[str]:
    try:
        p = Path(server_path) if server_path is not None else Path(DEFAULT_LLAMA_SERVER)
        key = str(p)
    except Exception:
        key = str(server_path or "")
    if key in _SERVER_FLAG_CACHE:
        return _SERVER_FLAG_CACHE[key]
    flags: set[str] = set()
    try:
        proc = subprocess.run([key, "--help"], capture_output=True, text=True, timeout=8, env=_server_help_env(p))
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except Exception:
        _SERVER_FLAG_CACHE[key] = set()
        return set()
    for m in re.findall(r"(--[A-Za-z0-9-]+)", text):
        flags.add(m)
    for m in re.findall(r"(?<!-)(-[A-Za-z][A-Za-z0-9-]+)\b", text):
        flags.add(m)
    _SERVER_FLAG_CACHE[key] = flags
    return flags
def server_supports_flag(server_path: Path | str | None, flag: str) -> bool:
    try:
        return flag in get_server_supported_flags(server_path)
    except Exception:
        return False
def normalize_server_overrides(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, object] = {}
    internal_keys = {"gpu_set","gpu_set_idx","ts_strategy","tensor_split_strategy","main_gpu_raw","auto_performance"}
    for raw_key, raw_val in value.items():
        key = str(raw_key).strip().lower().replace("-", "_")
        if not key or key in internal_keys:
            continue
        if key == "speculative_defaults":
            if isinstance(raw_val, dict):
                nested: dict[str, object] = {}
                for nk, nv in raw_val.items():
                    nn = str(nk).strip().lower().replace("-", "_")
                    if not nn:
                        continue
                    if nn == "draft_max":
                        nn = "draft"
                    nested[nn] = nv
                normalized[key] = nested
            continue
        if key == "draft_max":
            key = "draft"
        if key == "gpu_layers":
            key = "n_gpu_layers"
            if isinstance(raw_val, str) and raw_val.strip().lower() == "all":
                continue
        if key == "split_mode" and str(raw_val).strip().lower() == "layer":
            continue
        if key == "mmap":
            bv = _normalize_bool_flag(raw_val)
            if bv is True:
                continue
            if bv is not None:
                normalized[key] = bv
            continue
        if key in {"mul_mat_q","use_fitc","swa_full"}:
            bv = _normalize_bool_flag(raw_val)
            if bv is not None:
                normalized[key] = bv
            continue
        if key == "fit":
            bv = _normalize_bool_flag(raw_val)
            if bv is not None:
                normalized[key] = bv
                continue
            if isinstance(raw_val, str) and raw_val.strip():
                normalized[key] = raw_val.strip()
            continue
        if key in {"fitt","fitc"}:
            if isinstance(raw_val, bool):
                normalized[key] = raw_val
                continue
            try:
                normalized[key] = int(raw_val)
            except (TypeError, ValueError):
                try:
                    normalized[key] = int(float(raw_val))
                except (TypeError, ValueError):
                    continue
            continue
        if key == "n_gpu_layers_draft":
            if isinstance(raw_val, str) and raw_val.strip().lower() in {"all","auto"}:
                normalized[key] = raw_val.strip().lower()
                continue
            try:
                normalized[key] = int(raw_val)
            except (TypeError, ValueError):
                continue
            continue
        if key in {"ctx_size","n_gpu_layers","batch_size","ubatch_size","threads","threads_batch","fit_target","keep","mirostat","draft","draft_min","ctx_size_draft","grp_attn_n","parallel","main_gpu","ctx_checkpoints","checkpoint_min_step","checkpoint_every_n_tokens","cache_ram","n_cpu_moe","top_k","predict","image_min_tokens"}:
            try:
                normalized[key] = int(raw_val)
            except (TypeError, ValueError):
                continue
            continue
        if key == "reasoning_budget":
            if isinstance(raw_val, str) and raw_val.strip().casefold() in {REASONING_BUDGET_HALF_CONTEXT,"half_ctx","auto"}:
                normalized[key] = REASONING_BUDGET_HALF_CONTEXT
                continue
            try:
                normalized[key] = int(raw_val)
            except (TypeError, ValueError):
                continue
            continue
        if key in {"mirostat_ent","mirostat_lr","draft_p_min","defrag_threshold","top_p","min_p","repeat_penalty","presence_penalty"}:
            try:
                normalized[key] = float(raw_val)
            except (TypeError, ValueError):
                continue
            continue
        if key == "flash_attn":
            if isinstance(raw_val, str):
                normalized[key] = raw_val.strip()
            else:
                bv = _normalize_bool_flag(raw_val)
                if bv is not None:
                    normalized[key] = bv
            continue
        if key in {"kv_offload","cont_batching","op_offload","cpu_moe","kv_unified","cache_idle_slots","direct_io","swa_full","cache_prompt"}:
            bv = _normalize_bool_flag(raw_val)
            if bv is not None:
                normalized[key] = bv
            continue
        if key == "tensor_split":
            normalized[key] = normalize_tensor_split(str(raw_val))
            continue
        if key == "numa":
            if raw_val is None or (isinstance(raw_val, str) and raw_val.strip().lower() == "none"):
                continue
            normalized[key] = str(raw_val).strip()
            continue
        if key == "reasoning":
            bv = _normalize_bool_flag(raw_val)
            normalized[key] = ("on" if bv else "off") if bv is not None else str(raw_val).strip()
            continue
        if key in {"cache_type","cache_type_k","cache_type_v"}:
            v = _normalize_cache_type_value(raw_val)
            if v is not None:
                normalized[key] = v
            continue
        if key in {"vbr_floor","vbr_entry","vbr_codec","vbr_vram","vbr_vram_budget","vbr_min_bits","vbr_min_bpv"}:
            sval = str(raw_val or "").strip()
            if sval:
                normalized[key] = sval.lower() if key in {"vbr_floor","vbr_entry"} else sval
            continue
        if key in {"split_mode","host","model_draft","hf_repo_draft","reasoning_format","reasoning_budget_message","chat_template_file","chat_template","device","chat_template_kwargs"}:
            if key == "chat_template_kwargs":
                if isinstance(raw_val, dict):
                    normalized[key] = json.dumps(raw_val, ensure_ascii=False, separators=(",",":"))
                else:
                    sval = str(raw_val).strip()
                    sval = re.sub(r'(:\s*)(on)(\s*[,}])', lambda m: f'{m.group(1)}true{m.group(3)}', sval, flags=re.IGNORECASE)
                    sval = re.sub(r'(:\s*)(off)(\s*[,}])', lambda m: f'{m.group(1)}false{m.group(3)}', sval, flags=re.IGNORECASE)
                    normalized[key] = sval
            else:
                normalized[key] = str(raw_val).strip()
            continue
        try:
            normalized[key] = raw_val
        except Exception:
            normalized[key] = str(raw_val)
    return normalized
def _llama_flag_name(key: str) -> str:
    return key if str(key).startswith("-") else f"--{str(key).replace('_','-')}"
def _positive_int(value: object) -> int | None:
    try:
        p = int(value)
    except (TypeError, ValueError):
        return None
    return p if p > 0 else None
def _server_supports_or_unknown(server_path: Path | str | None, flag: str) -> bool:
    try:
        import llamacpp_stack.cli as _cli_pkg
        fn = getattr(_cli_pkg, "_server_supports_or_unknown", None)
        if fn is not None and fn is not _server_supports_or_unknown:
            try:
                return fn(server_path, flag)
            except Exception:
                pass
    except Exception:
        pass
    try:
        cf = _get_cli_file()
        if cf is not None:
            fn = getattr(cf, "_server_supports_or_unknown", None)
            if fn is not None and fn is not _server_supports_or_unknown:
                try:
                    return fn(server_path, flag)
                except Exception:
                    pass
    except Exception:
        pass
    try:
        return server_path is None or server_supports_flag(server_path, flag)
    except Exception:
        return True
def _device_list_for_tensor_split(tensor_split: str, total_devices: int | None = None) -> str | None:
    parts = [p.strip() for p in str(tensor_split or "").split(",") if p.strip()]
    return ",".join(f"CUDA{idx}" for idx in range(len(parts))) if parts else None
def _vllm_flag_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",",":")) if isinstance(value, (dict, list)) else str(value)
def _load_server_config_payload(args=None) -> dict:
    cf = _get_cli_file()
    if cf is not None and hasattr(cf, "_load_server_config_payload"):
        try:
            return cf._load_server_config_payload(args)  # type: ignore
        except Exception:
            pass
    from .constants import DEFAULT_SERVER_CONFIG_PATH, ALTERNATE_SERVER_CONFIG_BASENAME
    cands: list[Path] = []
    if args is not None and getattr(args, "server_config", None) is not None:
        p = Path(args.server_config)
        cands.append(p)
        q = p.with_name(ALTERNATE_SERVER_CONFIG_BASENAME)
        if q != p:
            cands.append(q)
    else:
        d = Path(DEFAULT_SERVER_CONFIG_PATH)
        cands.append(d.with_name(ALTERNATE_SERVER_CONFIG_BASENAME))
        cands.append(d)
    for p in cands:
        try:
            if not p.exists():
                continue
            txt = p.read_text(encoding="utf-8")
            import re as _re
            lines = txt.split("\n")
            clean = [ln for ln in lines if not ln.strip().startswith("#")]
            ct = "\n".join(clean)
            ct = _re.sub(r",\s*([}\]])", r"\1", ct)
            payload = json.loads(ct)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return {}
def resolve_api_ctx_factor(args=None) -> float:
    from .constants import DEFAULT_API_CTX_FACTOR
    try:
        payload = _load_server_config_payload(args)
    except Exception:
        return float(DEFAULT_API_CTX_FACTOR)
    v = payload.get("api_ctx_factor")
    if v is None:
        v = payload.get("ctx_display_ratio")
    if v is None:
        v = payload.get("ctx_display_percent")
    if v is None:
        return float(DEFAULT_API_CTX_FACTOR)
    try:
        if isinstance(v, str):
            s = v.strip()
            if s.endswith("%"):
                return max(0.0, min(1.0, float(s[:-1])/100.0))
            f = float(s)
        else:
            f = float(v)
        if f > 1.0:
            f /= 100.0
        return max(0.0, min(1.0, f))
    except Exception:
        return float(DEFAULT_API_CTX_FACTOR)
resolve_ctx_display_ratio = resolve_api_ctx_factor
def resolve_llama_server_defaults(args=None) -> dict[str, object]:
    payload = _load_server_config_payload(args)
    defaults = normalize_server_overrides(payload.get("llama_server_defaults"))
    try:
        cf = _get_cli_file()
        fn = getattr(cf, "_normalize_llama_server_family_defaults_config", None)
        fd = fn(payload.get("llama_server_family_defaults")) if fn else {}
    except Exception:
        fd = {}
    if fd:
        defaults["__family_defaults"] = fd
    return defaults
def resolve_vllm_defaults(args=None) -> dict[str, object]:
    payload = _load_server_config_payload(args)
    try:
        cf = _get_cli_file()
        fn = getattr(cf, "_normalize_vllm_config", None)
        cfg = fn(payload.get("vllm")) if fn else {"defaults":{},"family_defaults":{}}
    except Exception:
        cfg = {"defaults":{},"family_defaults":{}}
    bundled_path = Path(__file__).resolve().parent.parent / "bundle" / "llama_server_defaults.yaml"
    bundled_vllm: dict[str, object] = {}
    try:
        import yaml  # type: ignore
        bundled = yaml.safe_load(bundled_path.read_text(encoding="utf-8")) or {}
        if isinstance(bundled, dict):
            cf = _get_cli_file()
            fn2 = getattr(cf, "_normalize_vllm_config", None)
            bundled_vllm = fn2(bundled.get("vllm")) if fn2 else {}
    except Exception:
        bundled_vllm = {}
    defaults = dict(bundled_vllm.get("defaults") or {})
    defaults.update(cfg.get("defaults") or {})
    fam = {str(k): dict(v) for k, v in (bundled_vllm.get("family_defaults") or {}).items() if isinstance(v, dict)}
    for k, v in (cfg.get("family_defaults") or {}).items():
        if isinstance(v, dict):
            fam.setdefault(str(k), {}).update(v)
    if fam:
        defaults["__family_defaults"] = fam
    return defaults
def resolve_vllm_options(model, vllm_defaults: dict[str, object] | None = None) -> dict[str, object]:
    try:
        cf = _get_cli_file()
        fn = getattr(cf, "_normalize_vllm_mapping", None)
        raw = fn(vllm_defaults or resolve_vllm_defaults()) if fn else dict(vllm_defaults or resolve_vllm_defaults())
    except Exception:
        raw = dict(vllm_defaults or resolve_vllm_defaults())
    fam = raw.pop("__family_defaults", None)
    opts = dict(raw)
    try:
        cf = _get_cli_file()
        fn2 = getattr(cf, "_vllm_family_defaults_for_model", None)
        if fn2:
            opts.update(fn2(model, fam))
    except Exception:
        pass
    overrides = model.server_overrides if isinstance(model.server_overrides, dict) else {}
    mv = overrides.get("vllm")
    if isinstance(mv, dict):
        try:
            cf = _get_cli_file()
            fn3 = getattr(cf, "_normalize_vllm_mapping", None)
            opts.update(fn3(mv) if fn3 else dict(mv))
        except Exception:
            opts.update(dict(mv))
    from .constants import DEFAULT_CTX_SIZE
    if "max_model_len" not in opts:
        opts["max_model_len"] = int(getattr(model, "ctx_size", DEFAULT_CTX_SIZE) or DEFAULT_CTX_SIZE)
    if "tensor_parallel_size" not in opts:
        parts = [p for p in str(getattr(model, "tensor_split","1") or "1").split(",") if p.strip()]
        opts["tensor_parallel_size"] = max(1, len(parts))
    if "dtype" not in opts:
        ld = next((k for k in ("float16","bfloat16","float32") if overrides.get(k)), None)
        if ld:
            opts["dtype"] = ld
    return opts
def resolve_request_reasoning_budget(payload: dict[str, object], model, *, server_defaults: dict[str, object] | None = None) -> tuple[int | None, str]:
    try:
        cf = _get_cli_file()
        fn = getattr(cf, "_model_backend", None)
        backend = fn(model) if fn else "llama.cpp"
    except Exception:
        backend = "llama.cpp"
    if not isinstance(payload, dict) or backend != "llama.cpp":
        return None, "unsupported_backend"
    if any(k in payload for k in {"thinking_budget_tokens","reasoning_budget_tokens"}):
        return None, "explicit_client_control"
    # effective options via lazy
    try:
        cf = _get_cli_file()
        fn_eff = getattr(cf, "_effective_llama_server_options", None)
        if fn_eff:
            effective = fn_eff(model, server_defaults)
        else:
            effective = dict(normalize_server_overrides(server_defaults or {}))
            try:
                fn_fam = getattr(cf, "_family_defaults_for_model", None)
                fd = effective.pop("__family_defaults", None)
                if fn_fam:
                    effective.update(fn_fam(model, fd))
            except Exception:
                pass
            effective.update(normalize_server_overrides(getattr(model, "server_overrides", {}) or {}))
    except Exception:
        effective = dict(normalize_server_overrides(server_defaults or {}))
        effective.update(normalize_server_overrides(getattr(model, "server_overrides", {}) or {}))
    policy = effective.get("reasoning_budget")
    if not (isinstance(policy, str) and policy.casefold() == REASONING_BUDGET_HALF_CONTEXT):
        return None, "fixed_or_unconfigured"
    if str(effective.get("reasoning") or "").strip().casefold() == "off":
        return None, "reasoning_disabled"
    rl = _positive_int(payload.get("max_tokens"))
    if rl is None:
        rl = _positive_int(payload.get("max_completion_tokens"))
    cp = _positive_int(effective.get("predict"))
    lims = [x for x in (rl, cp) if x]
    gen_lim = min(lims) if lims else None
    try:
        cf = _get_cli_file()
        fn2 = getattr(cf, "displayed_configured_ctx", None)
        ctx = fn2(model) if fn2 else int(getattr(model,"ctx_size",0) or 0)
    except Exception:
        ctx = int(getattr(model,"ctx_size",0) or 0)
    if ctx <= 0:
        return None, "unknown_context"
    half = max(0, ctx // 2)
    try:
        cf = _get_cli_file()
        fn3 = getattr(cf, "request_looks_like_model_probe", None)
        is_probe = fn3(payload) if fn3 else False
    except Exception:
        is_probe = False
    from .constants import MODEL_PROBE_REASONING_MAX_TOKENS, DEFAULT_REASONING_VISIBLE_RESERVE
    if is_probe and rl is not None and rl <= MODEL_PROBE_REASONING_MAX_TOKENS:
        return 0, "model_probe"
    if gen_lim is None:
        return half, "half_context"
    reserve = min(DEFAULT_REASONING_VISIBLE_RESERVE, max(1, gen_lim // 8))
    budget = max(0, min(half, gen_lim - reserve))
    reason = "half_context_clamped_to_generation_limit" if budget < half else "half_context"
    return budget, reason
def _append_llama_server_flag(cmd: list[str], key: str, value: object, server_path: Path | str | None = None) -> None:
    if value is None:
        cmd.append(_llama_flag_name(key)); return
    if key == "split_mode":
        cmd.extend(["--split-mode", str(value)]); return
    if key == "flash_attn":
        vs=None
        if isinstance(value, bool): vs="on" if value else "off"
        else:
            s=str(value).strip()
            if s:
                low=s.lower()
                vs="on" if low in {"1","true","yes","on"} else "off" if low in {"0","false","no","off"} else s
        if vs is not None: cmd.extend(["--flash-attn", vs])
        return
    if key == "batch_size": cmd.extend(["--batch-size", str(int(value))]); return
    if key == "ubatch_size": cmd.extend(["--ubatch-size", str(max(256, int(value)))]); return
    if key in {"threads","threads_batch"}:
        flag="--threads" if key=="threads" else "--threads-batch"
        try:
            if isinstance(value, str):
                pc=os.cpu_count() or 1
                low=value.strip().lower()
                v=max(1, pc//2) if low=="physical" else pc*2 if low=="logical" else int(value)
            else: v=int(value)
            cmd.extend([flag, str(v)])
        except Exception: pass
        return
    if key == "main_gpu": cmd.extend(["--main-gpu", str(int(value))]); return
    if key == "numa":
        if value is None: return
        s=str(value).strip()
        if s and s.lower()!="none": cmd.extend(["--numa", s])
        return
    if key == "reasoning_format":
        s=str(value).strip()
        if s: cmd.extend(["--reasoning-format", s])
        return
    if key == "fit_target": cmd.extend(["--fit-target", str(int(value))]); return
    if key == "image_min_tokens":
        try: cmd.extend(["--image-min-tokens", str(int(value))])
        except: pass
        return
    if key == "model_draft":
        s=str(value or "").strip()
        if s and s.lower() not in {"none","null"}: cmd.extend(["--model-draft", s])
        return
    if key in {"spec_draft_model","spec-draft-model"}:
        s=str(value or "").strip()
        if s and s.lower() not in {"none","null"}: cmd.extend(["--spec-draft-model", s])
        return
    if key in {"spec_draft_ngl","spec-draft-ngl"}:
        s=str(value or "").strip()
        if s: cmd.extend(["--spec-draft-ngl", s])
        return
    if key == "hf_repo_draft": cmd.extend(["--hf-repo-draft", str(value)]); return
    if key == "spec_type":
        s=str(value or "").strip()
        if s and _server_supports_or_unknown(server_path, "--spec-type"): cmd.extend(["--spec-type", s])
        return
    if key == "spec_draft_n_max":
        try:
            f="--spec-draft-n-max" if _server_supports_or_unknown(server_path, "--spec-draft-n-max") else "--draft-max"
            cmd.extend([f, str(int(value))])
        except: pass
        return
    if key == "spec_draft_n_min":
        try: cmd.extend(["--spec-draft-n-min", str(int(value))])
        except: pass
        return
    if key == "spec_draft_p_min":
        try: cmd.extend(["--spec-draft-p-min", str(value)])
        except: pass
        return
    if key == "draft":
        f=next((x for x in ("--spec-draft-n-max","--draft-max","--draft","--draft-n") if server_supports_flag(server_path, x)), "--draft-max")
        cmd.extend([f, str(int(value))]); return
    if key in {"draft_min","draft_p_min"}: return
    if key == "ctx_size_draft": cmd.extend(["--ctx-size-draft", str(int(value))]); return
    if key == "n_gpu_layers_draft":
        try:
            if isinstance(value, str) and value.strip().lower()=="all": v=999
            elif isinstance(value, str) and value.strip().lower()=="auto": v=-1
            else: v=int(value)
            cmd.extend(["--n-gpu-layers-draft", str(v)])
        except: pass
        return
    if key == "keep": cmd.extend(["--keep", str(int(value))]); return
    if key == "mirostat": cmd.extend(["--mirostat", str(int(value))]); return
    if key == "mirostat_ent": cmd.extend(["--mirostat-ent", str(float(value))]); return
    if key == "mirostat_lr": cmd.extend(["--mirostat-lr", str(float(value))]); return
    if key == "cache_type_k": cmd.extend(["--cache-type-k", str(value)]); return
    if key == "cache_type_v": cmd.extend(["--cache-type-v", str(value)]); return
    if key == "fit":
        vs=None
        if isinstance(value, bool): vs="on" if value else "off"
        else:
            s=str(value).strip()
            if s: vs="on" if s.lower() in {"1","true","yes","on"} else "off" if s.lower() in {"0","false","no","off"} else s
        if vs is not None: cmd.extend(["-fit", vs])
        return
    if key == "fitt":
        try: cmd.extend(["-fitt", str(int(value))])
        except: pass
        return
    if key == "fitc":
        try:
            if isinstance(value, bool):
                if value: cmd.extend(["-fitc", str(4096)])
            else:
                s=str(value).strip()
                if s:
                    try: cmd.extend(["-fitc", str(int(s))])
                    except:
                        try: cmd.extend(["-fitc", str(int(float(s)))])
                        except: cmd.extend(["-fitc", str(4096)])
        except: cmd.extend(["-fitc", str(4096)])
        return
    if key == "draft_mtp":
        try:
            if server_supports_flag(server_path, "--draft-mtp") or server_path is None: cmd.append("--draft-mtp")
        except:
            try: cmd.append("--draft-mtp")
            except: pass
        return
    if key == "mmap":
        bv=_normalize_bool_flag(value)
        if bv is False: cmd.append("--no-mmap")
        return
    if key in {"float16","bfloat16","float32"}:
        if _normalize_bool_flag(value): cmd.append(f"--{key}")
        return
    if key == "gpu_memory_utilization" and value is not None: cmd.extend(["--gpu-memory-utilization", str(float(value))]); return
    # generic supported checks via tables
    _bool_flags={"mul_mat_q":"--mul-mat-q","cpu_moe":"--cpu-moe"}
    if key in _bool_flags:
        if _normalize_bool_flag(value) and _server_supports_or_unknown(server_path, _bool_flags[key]): cmd.append(_bool_flags[key])
        return
    _bool_pair={"kv_offload":("--kv-offload","--no-kv-offload"),"cont_batching":("--cont-batching","--no-cont-batching"),"op_offload":("--op-offload","--no-op-offload"),"direct_io":("--direct-io","--no-direct-io"),"kv_unified":("--kv-unified","--no-kv-unified"),"cache_idle_slots":("--cache-idle-slots","--no-cache-idle-slots"),"swa_full":("--swa-full","--no-swa-full")}
    if key in _bool_pair:
        bv=_normalize_bool_flag(value)
        on,off=_bool_pair[key]
        if bv is True and _server_supports_or_unknown(server_path, on): cmd.append(on)
        elif bv is False and _server_supports_or_unknown(server_path, off): cmd.append(off)
        return
    _int_sup={"grp_attn_n":"--grp-attn-n","parallel":"--parallel","ctx_checkpoints":"--ctx-checkpoints","cache_ram":"--cache-ram","n_cpu_moe":"--n-cpu-moe","top_k":"--top-k","predict":"--predict"}
    if key in _int_sup:
        try:
            if _server_supports_or_unknown(server_path, _int_sup[key]): cmd.extend([_int_sup[key], str(int(value))])
        except: pass
        return
    if key in {"checkpoint_min_step","checkpoint_every_n_tokens"}:
        try:
            if _server_supports_or_unknown(server_path, "--checkpoint-min-step"): cmd.extend(["--checkpoint-min-step", str(int(value))])
            elif _server_supports_or_unknown(server_path, "--checkpoint-every-n-tokens"): cmd.extend(["--checkpoint-every-n-tokens", str(int(value))])
        except: pass
        return
    if key == "device":
        s=str(value).strip()
        if s: cmd.extend(["--device", s])
        return
    _float_sup={"defrag_threshold":"--defrag-threshold","top_p":"--top-p","min_p":"--min-p","repeat_penalty":"--repeat-penalty","presence_penalty":"--presence-penalty"}
    if key in _float_sup:
        try:
            if _server_supports_or_unknown(server_path, _float_sup[key]): cmd.extend([_float_sup[key], str(float(value))])
        except: pass
        return
    if key == "reasoning":
        s=str(value or "").strip()
        if s and _server_supports_or_unknown(server_path, "--reasoning"): cmd.extend(["--reasoning", s])
        return
    if key == "reasoning_budget":
        try:
            if _server_supports_or_unknown(server_path, "--reasoning-budget"):
                if isinstance(value, str) and value.strip().casefold()==REASONING_BUDGET_HALF_CONTEXT: value=-1
                cmd.extend(["--reasoning-budget", str(int(value))])
        except: pass
        return
    if key == "reasoning_budget_message":
        s=str(value or "").strip()
        if s and _server_supports_or_unknown(server_path, "--reasoning-budget-message"): cmd.extend(["--reasoning-budget-message", s])
        return
    if key == "chat_template_kwargs":
        s=str(value or "").strip()
        if s and _server_supports_or_unknown(server_path, "--chat-template-kwargs"): cmd.extend(["--chat-template-kwargs", s])
        return
    if key == "cache_quant":
        s=str(value or "").strip()
        if s: cmd.extend(["--cache_quant", s])
        return
    if key == "grid_size":
        try: cmd.extend(["--grid_size", str(int(value))])
        except: cmd.extend(["--grid_size", str(value)])
        return
    # generic fallback
    try:
        f=_llama_flag_name(str(key))
        if isinstance(value, bool):
            if value: cmd.append(f)
            else: cmd.extend([f, "false"])
        elif value is None: cmd.append(f)
        elif isinstance(value, (list,tuple)):
            for v in value: cmd.extend([f, str(v)])
        else: cmd.extend([f, str(value)])
    except: pass
def build_vllm_server_command(model, *, port: str, host: str | None = None, vllm_defaults: dict[str, object] | None = None) -> list[str]:
    # vLLM TODO: emulate legacy /completion like exllama_server.py legacy_completion so
    # vLLM y exl3 y futuros motores permitan http://127.0.0.1:11436/upstream/<model>/completion
    # parity with llama.cpp. Add POST /completion + /v1/completions in vllm-server wrapper,
    # mapping prompt->messages and reusing vLLM generate, returning llama.cpp compat JSON.
    options = resolve_vllm_options(model, vllm_defaults)
    command = [os.environ.get("VLLM_SERVER_BIN","vllm-server"), "--model", str(model.local_path), "--port", str(port), "--served-model-name", str(model.model_id)]
    resolved_host = str(options.pop("host", host or model.host) or "127.0.0.1")
    if resolved_host:
        command.extend(["--host", resolved_host])
    reserved = {"model","port","served_model_name","host"}
    for rk, val in options.items():
        k = str(rk).strip().lower().replace("-", "_")
        if not k or k in reserved or val is None:
            continue
        if k == "per_request_spec_decode_metrics":
            if not vllm_server_supports_flag("--per-request-spec-decode-metrics"):
                continue
            vs = str(val).strip().lower()
            if vs not in {"detailed", "summary", "none"}:
                vs = "summary"
            command.extend(["--per-request-spec-decode-metrics", vs])
            continue
        flag = f"--{k.replace('_','-')}"
        if isinstance(val, bool):
            command.append(flag if val else f"--no-{k.replace('_','-')}")
            continue
        command.extend([flag, _vllm_flag_value(val)])
    return command
def build_llama_server_command(model, server_path: Path, *, port: str, host: str | None = None, include_model_path: bool = True, include_mmproj: bool = True, include_jinja: bool = True, server_defaults: dict[str, object] | None = None, vllm_defaults: dict[str, object] | None = None, extra_flags: list[str] | None = None) -> list[str]:
    try:
        cf = _get_cli_file()
        fn_backend = getattr(cf, "_model_backend", None)
        backend = fn_backend(model) if fn_backend else "llama.cpp"
    except Exception:
        backend = "llama.cpp"
    if backend == "vllm":
        return build_vllm_server_command(model, port=port, host=host, vllm_defaults=vllm_defaults)
    effective = dict(normalize_server_overrides(server_defaults or {}))
    fam = effective.pop("__family_defaults", None)
    try:
        cf = _get_cli_file()
        fn = getattr(cf, "_family_defaults_for_model", None)
        if fn:
            effective.update(fn(model, fam))
    except Exception:
        pass
    try:
        if getattr(model,"speculative", False) and isinstance(server_defaults, dict):
            sd = server_defaults.get("speculative_defaults")
            if isinstance(sd, dict):
                effective.update(normalize_server_overrides(sd))
                for rk in ("fit","fitt","fitc"):
                    if rk not in effective and rk in sd:
                        effective[rk] = sd[rk]
    except Exception:
        pass
    model_overrides = normalize_server_overrides(getattr(model,"server_overrides",{}) or {})
    effective.update(model_overrides)
    try:
        cf = _get_cli_file()
        fn_mf = getattr(cf, "_model_family_match_text", None)
        hs = fn_mf(model) if fn_mf else ""
        is_gemma4 = any(t in hs for t in ("gemma-4","gemma4"))
    except Exception:
        is_gemma4 = False
    if is_gemma4 and "swa_full" not in model_overrides:
        effective.pop("swa_full", None)
    effective.pop("replicas", None); effective.pop("placement", None)
    _engine = str(effective.pop("engine", None) or "").strip().lower().replace("_","-")
    if _engine in {"buun-beta"}:
        _engine = "buun"
    effective.pop("auto_performance", None); effective.pop("__family_defaults", None)
    effective.pop("speculative_defaults", None); effective.pop("mtp_defaults", None)
    for _sk in ("enabled","id_prefix","allow_multiple_variants"):
        effective.pop(_sk, None)
    if _engine not in {"exllama","exllamav3","exllama-v3","exllama3"}:
        effective.pop("cache_quant", None)
        effective.pop("grid_size", None)
    if _engine in {"exllama","exllamav3","exllama-v3","exllama3"}:
        effective.pop("cache_type_k", None)
        effective.pop("cache_type_v", None)
        effective.pop("cache_type_k_draft", None)
        effective.pop("cache_type_v_draft", None)
    if _engine == "buun":
        _ct = str(effective.get("cache_type") or "").strip().lower()
        if _ct in _TURBO_CACHE_TYPES:
            effective.pop("cache_type_k", None)
            effective.pop("cache_type_v", None)
            effective.pop("cache_type_k_draft", None)
            effective.pop("cache_type_v_draft", None)
        if _ct != "vbr":
            for _vk in ("vbr_floor","vbr_entry","vbr_codec","vbr_vram","vbr_vram_budget","vbr_min_bits","vbr_min_bpv"):
                effective.pop(_vk, None)
    replica_tensor_split = effective.pop("__replica_tensor_split", None)
    try:
        cf = _get_cli_file()
        fn_safe = getattr(cf, "_safe_runtime_enabled", None)
        safe_enabled = fn_safe() if fn_safe else False
    except Exception:
        safe_enabled = False
    if safe_enabled:
        effective["flash_attn"]=False; effective["fit"]=False; effective["parallel"]=1; effective["cont_batching"]=False
        for rk in ("draft_mtp","model_draft","hf_repo_draft","spec_type","spec_draft_n_max","spec_draft_n_min","spec_draft_p_min","draft","draft_min","draft_p_min","ctx_size_draft","n_gpu_layers_draft","ctx_checkpoints"):
            effective.pop(rk, None)
    if str(effective.get("model_draft") or "").strip() and "draft" in effective and (str(effective.get("spec_type") or "").strip()=="draft-mtp" or "spec_draft_n_max" in effective):
        effective.pop("draft", None)
    try:
        mid = str(getattr(model,"model_id","") or "").strip().lower()
        if (not safe_enabled) and mid.endswith("-mtp") and "draft_mtp" not in effective and "draft-mtp" not in effective:
            effective["draft_mtp"]=True
    except Exception:
        pass
    ctx_size = int(effective.pop("ctx_size", model.ctx_size))
    n_gpu_layers = int(effective.pop("n_gpu_layers", model.n_gpu_layers))
    cuda_visible_devices: list[int] | None = None
    if replica_tensor_split is not None:
        tensor_split = str(replica_tensor_split)
        effective.pop("tensor_split", None)
        if "device" not in effective:
            effective["device"] = ",".join(f"CUDA{idx}" for idx in range(len([p for p in tensor_split.split(",") if p.strip()])))
    else:
        effective.pop("tensor_split", None)
        raw_ts = str(model.tensor_split)
        parts = [p.strip() for p in raw_ts.split(",") if p.strip()]
        explicit_device = "device" in effective
        if parts:
            tensor_split = raw_ts
            if not explicit_device:
                cuda_visible_devices = list(range(len(parts)))
        else:
            try:
                cf = _get_cli_file()
                fn_pref = getattr(cf, "preferred_tensor_split", None)
                tensor_split = fn_pref(model, raw_ts) if fn_pref else raw_ts
            except Exception:
                tensor_split = raw_ts
        inferred = _device_list_for_tensor_split(tensor_split)
        if inferred is not None and "device" not in effective:
            effective["device"] = inferred
    resolved_host = str(effective.pop("host", host or model.host))
    try:
        have_auto = (getattr(model,"ctx_probe_kv_gb",None) is not None) or (getattr(model,"ctx_probe_read_s",None) is not None)
    except Exception:
        have_auto = False
    use_fitc_raw = effective.pop("use_fitc", None)
    use_fitc = _normalize_bool_flag(use_fitc_raw)
    if use_fitc is None:
        if use_fitc_raw is None and "fit" in effective:
            legacy = _normalize_bool_flag(effective.get("fit"))
            use_fitc = bool(legacy) if legacy is not None else bool(str(effective.get("fit") or "").strip())
        else:
            use_fitc=False
    fit_enabled = bool(use_fitc)
    if fit_enabled:
        effective["fit"]=True
        if (not have_auto) and ("fitt" not in effective):
            effective["fitt"]=int(effective.get("fitt",1024))
        fv = effective.get("fitc")
        parsed: int | None = None
        if isinstance(fv, bool):
            parsed=None
        elif fv is not None:
            try:
                parsed=int(fv)
            except Exception:
                try:
                    parsed=int(float(str(fv).strip()))
                except Exception:
                    parsed=None
        if parsed is None or parsed<=0:
            effective["fitc"]=ctx_size
        else:
            effective["fitc"]=parsed
    else:
        effective.pop("fit",None); effective.pop("fitc",None); effective.pop("fitt",None)
    try:
        cf = _get_cli_file()
        fn_v = getattr(cf, "_is_vllm_backend", None)
        is_vllm = fn_v() if fn_v else False
    except Exception:
        is_vllm=False
    if is_vllm:
        vllm_bin=os.environ.get("VLLM_SERVER_BIN","vllm-server")
        vc=[vllm_bin,"--model",str(model.local_path),"--port",str(port)]
        vmap={"gpu_memory_utilization":"--gpu-memory-utilization","tensor_parallel_size":"--tensor-parallel-size","max_model_len":"--max-model-len","block_size":"--block-size","dtype":"--dtype","device":"--device","enable_chunked_prefill":"--enable-chunked-prefill"}
        for k,v in effective.items():
            if k in vmap:
                vc.extend([vmap[k], str(v)])
            elif k.startswith("--"):
                vc.extend([k, str(v)])
        if resolved_host and resolved_host not in {"0.0.0.0","::","[::]"}:
            vc.extend(["--host", resolved_host])
        return vc
    effective_server_path = server_path
    s_path_str = str(server_path)
    if _engine == "buun":
        if "llama-server-buun" in s_path_str:
            effective_server_path = s_path_str
        else:
            cand = Path(s_path_str).parent / "buun" / "bin" / "llama-server-buun"
            effective_server_path = str(cand)
    elif _engine == "beellama":
        if "llama-server-beellama" in s_path_str:
            effective_server_path = s_path_str
        else:
            cand = Path(s_path_str).parent / "beellama" / "bin" / "llama-server-beellama"
            effective_server_path = str(cand)
    elif _engine == "exllama":
        if "llama-server-exllama" in s_path_str:
            effective_server_path = s_path_str
        else:
            cand = Path(s_path_str).parent / "exllama" / "bin" / "llama-server-exllama"
            effective_server_path = str(cand)
    cmd=[str(effective_server_path),"--port",str(port)]
    if include_model_path:
        cmd.extend(["--model", str(model.local_path)])
    if not fit_enabled:
        cmd.extend(["--ctx-size", str(ctx_size)])
    cmd.extend(["--n-gpu-layers", str(n_gpu_layers)])
    cmd.extend(["--tensor-split", tensor_split])
    cmd.extend(["--host", resolved_host])
    explicit_keys=set(model_overrides.keys())
    for key in ("split_mode","flash_attn","reasoning_format","batch_size","ubatch_size","threads","threads_batch","main_gpu","numa","fit_target","model_draft","hf_repo_draft","spec_type","spec_draft_n_max","spec_draft_n_min","spec_draft_p_min","use_fitc","draft","draft_min","draft_p_min","ctx_size_draft","n_gpu_layers_draft","draft_mtp","fit","fitt","fitc","keep","mirostat","mirostat_ent","mirostat_lr","cache_type_k","cache_type_v","mmap","mul_mat_q","grp_attn_n","parallel","ctx_checkpoints","cache_ram","cache_prompt","kv_offload","cont_batching","op_offload","direct_io","cpu_moe","n_cpu_moe","device","defrag_threshold","swa_full","top_k","top_p","min_p","repeat_penalty","presence_penalty","predict","reasoning","reasoning_budget","reasoning_budget_message"):
        if key in effective:
            probe = None if key in explicit_keys else server_path
            _append_llama_server_flag(cmd, key, effective[key], probe)
            try:
                effective.pop(key, None)
            except Exception:
                pass
    for ek, ev in list(effective.items()):
        _append_llama_server_flag(cmd, ek, ev, server_path)
    if include_mmproj and model.mmproj_path:
        cmd.extend(["--mmproj", str(model.mmproj_path)])
    if include_jinja and model.jinja:
        cmd.append("--jinja")
    tmpl = effective.get("chat_template_file") or effective.get("chat_template")
    if tmpl:
        cmd.extend(["--chat-template-file", str(tmpl)])
    if extra_flags:
        cmd.extend(list(extra_flags))
    if cuda_visible_devices is not None:
        try:
            from llamacpp_stack.cli.replica import _command_with_cuda_visible_devices as _wrap
            cmd = _wrap(cmd, cuda_visible_devices)
        except Exception:
            try:
                cf=_get_cli_file()
                fn=getattr(cf,"_command_with_cuda_visible_devices",None)
                cmd = fn(cmd, cuda_visible_devices) if fn else ["/usr/bin/env", f"CUDA_VISIBLE_DEVICES={','.join(str(g) for g in cuda_visible_devices)}", *cmd]
            except Exception:
                cmd = ["/usr/bin/env", f"CUDA_VISIBLE_DEVICES={','.join(str(g) for g in cuda_visible_devices)}", *cmd]
    return cmd
__all__=["get_server_supported_flags","server_supports_flag","get_vllm_supported_flags","vllm_server_supports_flag","get_vllm_help_text","normalize_server_overrides","normalize_tensor_split","_normalize_bool_flag","resolve_api_ctx_factor","resolve_llama_server_defaults","resolve_vllm_defaults","resolve_vllm_options","resolve_request_reasoning_budget","_append_llama_server_flag","build_vllm_server_command","build_llama_server_command"]
