"""Lazy mmproj ("vision on demand") support for the llama-swap config.

A model that has a projector configured (``model.mmproj_path``) normally gets
``--mmproj`` baked into every launch, so the vision tower is resident even for
pure-text traffic. For big multimodal models (for example the native
safetensors EXL3 Qwen builds) that projector is expensive in VRAM.

``mmproj_mode`` declares the policy per model, via ``server_overrides``:

``always`` (default)
    Legacy behaviour: ``--mmproj`` is always emitted.
``lazy``
    The base route is rendered WITHOUT the projector and a sibling internal
    route ``<model_id>__vision`` is published WITH it. Image requests are
    routed to that sibling; llama-swap evicts it on idle TTL. The text-only
    base therefore never pays for the projector.
``off``
    Never emit ``--mmproj`` (useful to pin a broken/unwanted projector).

The sibling route reuses the internal-replica machinery (same
``metadata.internal_replica_of`` convention, same matrix/evict-cost handling),
so it is cheap to add and gets the cheaper 0.5x eviction cost of a replica.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from .constants import DEFAULT_SERVER_CONFIG_PATH
from .models import ManagedModel

MMPROJ_MODE_ALWAYS = "always"
MMPROJ_MODE_LAZY = "lazy"
MMPROJ_MODE_OFF = "off"
MMPROJ_MODES: tuple[str, ...] = (MMPROJ_MODE_ALWAYS, MMPROJ_MODE_LAZY, MMPROJ_MODE_OFF)

VISION_MODEL_SUFFIX = "__vision"
_VISION_MODEL_RE = re.compile(re.escape(VISION_MODEL_SUFFIX) + r"$")

# Override keys that must never reach llama-server as flags.
MMPROJ_OVERRIDE_KEYS = ("mmproj_mode",)

__all__ = [
    "MMPROJ_MODES",
    "MMPROJ_MODE_ALWAYS",
    "MMPROJ_MODE_LAZY",
    "MMPROJ_MODE_OFF",
    "MMPROJ_OVERRIDE_KEYS",
    "VISION_MODEL_SUFFIX",
    "build_vision_model",
    "default_mmproj_config",
    "get_model_mmproj_mode",
    "is_vision_model_id",
    "model_has_mmproj",
    "model_lazily_loads_mmproj",
    "normalize_mmproj_config",
    "normalize_mmproj_mode",
    "resolve_effective_mmproj_config",
    "resolve_render_include_mmproj",
    "vision_base_model_id",
    "vision_model_id",
    "vision_route_ttl",
]


def normalize_mmproj_mode(value: object, default: str = MMPROJ_MODE_ALWAYS) -> str:
    """Coerce a user-supplied mmproj mode into one of MMPROJ_MODES."""
    if isinstance(value, bool):
        # ``mmproj_mode: true`` reads as "lazy" for humans, "always" for code.
        return MMPROJ_MODE_LAZY if value else MMPROJ_MODE_OFF
    if value is None:
        text = ""
    elif isinstance(value, (int, float)):
        text = str(value)
    else:
        text = str(value)
    text = text.strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"lazy", "on_demand", "ondemand", "demand", "vision_on_demand"}:
        return MMPROJ_MODE_LAZY
    if text in {"off", "none", "never", "disabled", "disable", "false", "no"}:
        return MMPROJ_MODE_OFF
    if text in {"always", "eager", "on", "true", "yes", "enabled", "enable"}:
        return MMPROJ_MODE_ALWAYS
    fallback = str(default or MMPROJ_MODE_ALWAYS).strip().lower()
    return fallback if fallback in MMPROJ_MODES else MMPROJ_MODE_ALWAYS


def get_model_mmproj_mode(
    model: ManagedModel,
    mmproj_config: dict[str, object] | None = None,
) -> str:
    """Resolve the effective mmproj mode for ``model``.

    Per-model ``server_overrides.mmproj_mode`` wins; otherwise the global
    ``mmproj.default_mode`` from conf.json applies (``always`` by default, so
    existing installs are untouched).
    """
    raw = None
    try:
        overrides = getattr(model, "server_overrides", None) or {}
        raw = overrides.get("mmproj_mode")
    except Exception:
        raw = None
    if raw is None:
        cfg = mmproj_config if isinstance(mmproj_config, dict) else resolve_effective_mmproj_config()
        raw = (cfg or {}).get("default_mode")
    return normalize_mmproj_mode(raw)


def model_has_mmproj(model: ManagedModel) -> bool:
    """True when a projector path is configured (a GGUF file or a native dir)."""
    try:
        path = str(getattr(model, "mmproj_path", "") or "")
    except Exception:
        return False
    return bool(path.strip())


def model_lazily_loads_mmproj(
    model: ManagedModel,
    mmproj_config: dict[str, object] | None = None,
) -> bool:
    """True when this model should render a text-only base plus a vision sibling."""
    if not model_has_mmproj(model):
        return False
    if get_model_mmproj_mode(model, mmproj_config) != MMPROJ_MODE_LAZY:
        return False
    try:
        backend = str(getattr(model, "backend", "") or "").strip().lower()
    except Exception:
        backend = ""
    if backend == "vllm":
        # vLLM has no --mmproj and never returns early into this render path.
        return False
    try:
        overrides = getattr(model, "server_overrides", None) or {}
        engine = str(overrides.get("engine") or "").strip().lower().replace("_", "-")
    except Exception:
        engine = ""
    if engine in {"exllamav3", "exllama-v3", "exllama3", "exllama"}:
        # The exllama engine drops image_url parts before they reach the backend.
        return False
    return True


def vision_model_id(base_model_id: str) -> str:
    return f"{base_model_id}{VISION_MODEL_SUFFIX}"


def is_vision_model_id(model_id: str) -> bool:
    return bool(_VISION_MODEL_RE.search(str(model_id or "")))


def vision_base_model_id(model_id: str) -> str:
    return _VISION_MODEL_RE.sub("", str(model_id or ""))


def build_vision_model(base: ManagedModel) -> ManagedModel:
    """Sibling of ``base`` whose command carries ``--mmproj``.

    The projector is forced on regardless of the base's ``mmproj_mode`` so the
    vision route always renders a usable vision server.
    """
    overrides = dict(getattr(base, "server_overrides", None) or {})
    overrides["mmproj_mode"] = MMPROJ_MODE_ALWAYS
    overrides.pop("replicas", None)
    return replace(
        base,
        model_id=vision_model_id(base.model_id),
        aliases=[],
        description=f"lazy mmproj vision route of {base.model_id}",
        server_overrides=overrides,
    )


def resolve_render_include_mmproj(
    model: ManagedModel,
    mmproj_config: dict[str, object] | None = None,
) -> bool:
    """Whether the llama-swap ``cmd`` for ``model`` should carry ``--mmproj``.

    False for a lazy base (its vision sibling carries it instead) and for
    ``mmproj_mode: off``.
    """
    if not model_has_mmproj(model):
        return False
    return get_model_mmproj_mode(model, mmproj_config) == MMPROJ_MODE_ALWAYS


def default_mmproj_config() -> dict[str, object]:
    """Default ``mmproj`` block for conf.json."""
    return {
        "default_mode": MMPROJ_MODE_ALWAYS,
        "prefer_vision_route": True,
        "allow_co_resident": False,
        "vision_sticky_ttl_s": 1800,
        "route_publish_timeout_s": 90.0,
        "unload_timeout_s": 45.0,
        "reload_timeout_s": 45.0,
    }


def normalize_mmproj_config(raw: object) -> tuple[dict[str, object], bool]:
    """Fill in absent ``mmproj`` keys, clamp the rest. Returns (config, changed).

    Never overwrites an operator-provided value, so it is idempotent and safe to
    run on every config read.
    """
    defaults = default_mmproj_config()
    if not isinstance(raw, dict):
        # Absent -> nothing to migrate. Present but malformed -> rewrite it.
        return dict(defaults), raw is not None
    out: dict[str, object] = {}
    changed = False

    mode_raw = raw.get("default_mode", raw.get("default-mode"))
    if mode_raw is None:
        out["default_mode"] = defaults["default_mode"]
    else:
        out["default_mode"] = normalize_mmproj_mode(mode_raw)

    def _bool(key: str) -> None:
        nonlocal changed
        value = raw.get(key)
        if value is None:
            out[key] = defaults[key]
            return
        parsed = _coerce_bool(value, bool(defaults[key]))
        if parsed != value:
            changed = True
        out[key] = parsed

    _bool("prefer_vision_route")
    _bool("allow_co_resident")

    def _float(key: str, low: float, high: float) -> None:
        nonlocal changed
        value = raw.get(key)
        if value is None:
            out[key] = defaults[key]
            return
        try:
            parsed = float(value)
        except Exception:
            out[key] = defaults[key]
            changed = True
            return
        clamped = max(low, min(high, parsed))
        if clamped != value:
            changed = True
        out[key] = clamped

    _float("route_publish_timeout_s", 1.0, 900.0)
    _float("unload_timeout_s", 1.0, 900.0)
    _float("reload_timeout_s", 1.0, 900.0)

    value = raw.get("vision_sticky_ttl_s")
    if value is None:
        out["vision_sticky_ttl_s"] = defaults["vision_sticky_ttl_s"]
    else:
        try:
            parsed = int(value)
        except Exception:
            parsed = int(defaults["vision_sticky_ttl_s"])
            changed = True
        if parsed != value:
            changed = True
        out["vision_sticky_ttl_s"] = max(60, min(86400, parsed))

    for key, default_value in defaults.items():
        if key not in out:
            out[key] = default_value
            changed = True
    return out, changed


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def resolve_effective_mmproj_config(args: object | None = None) -> dict[str, object]:
    """Read the ``mmproj`` block out of the loaded server config (never raises)."""
    payload: object = None
    try:
        from llamacpp_stack.cli.server_commands import _load_server_config_payload  # type: ignore

        payload = _load_server_config_payload(args)
    except Exception:
        payload = None
    raw = payload.get("mmproj") if isinstance(payload, dict) else None
    normalized, _changed = normalize_mmproj_config(raw)
    return normalized


_MMPROJ_CONFIG_MEMO: dict[str, object] = {}


def cached_mmproj_config(args: object | None = None) -> dict[str, object]:
    """``resolve_effective_mmproj_config`` memoized on the config file identity.

    Rendering touches every catalog model, and the config loader re-reads and
    re-normalizes conf.json on each call.
    """
    try:
        configured = getattr(args, "server_config", None)
        path = Path(configured) if configured else Path(DEFAULT_SERVER_CONFIG_PATH)
        stat = path.stat()
        key = f"{path}:{stat.st_mtime_ns}:{stat.st_size}"
    except Exception:
        key = "unresolved"
    cached = _MMPROJ_CONFIG_MEMO.get("key")
    if cached == key:
        value = _MMPROJ_CONFIG_MEMO.get("value")
        if isinstance(value, dict):
            return value
    value = resolve_effective_mmproj_config(args)
    _MMPROJ_CONFIG_MEMO["key"] = key
    _MMPROJ_CONFIG_MEMO["value"] = value
    return value


def vision_route_ttl(mmproj_config: dict[str, object] | None = None) -> int | None:
    """llama-swap ttl for the vision sibling route (idle minutes, or None)."""
    cfg = mmproj_config if isinstance(mmproj_config, dict) else resolve_effective_mmproj_config()
    try:
        sticky = int((cfg or {}).get("vision_sticky_ttl_s", 1800))
    except Exception:
        return None
    if sticky <= 0:
        return None
    return max(1, int(round(sticky / 60.0)))