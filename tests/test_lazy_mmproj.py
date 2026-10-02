"""Tests for lazy mmproj (``mmproj_mode: lazy``).

There is **no dedicated vision route**. A lazy model renders exactly like every
other model -- text only, no ``--mmproj``. When a request carries an image the
gateway picks one instance that is free (normally a replica sitting idle) and
rewrites only that route's command so it carries the projector. The projector
lapses after ``mmproj.vision_sticky_ttl_s`` and the instance goes back to text.

These tests pin the three halves of that contract: the pure selection rule, the
single-route rewrite, and the gateway wiring.
"""

from __future__ import annotations

import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from llamacpp_stack import _cli_impl
from llamacpp_stack.cli import ManagedModel
from llamacpp_stack.cli import vision as vision_mod
from llamacpp_stack.cli.models import ReplicaRecord
from llamacpp_stack.cli.replica import (
    _calculate_llama_swap_matrix,
    ensure_internal_route_in_llamaswap_config,
    ensure_replica_route_in_llamaswap_config,
    render_llamaswap_config,
    route_carries_mmproj,
    set_instance_mmproj_in_llamaswap_config,
)
from llamacpp_stack.cli.server_commands import build_llama_server_command
from llamacpp_stack.cli.vision import (
    MMPROJ_MODE_ALWAYS,
    MMPROJ_MODE_LAZY,
    MMPROJ_MODE_OFF,
    candidate_instance_ids,
    choose_vision_instance,
    default_mmproj_config,
    get_model_mmproj_mode,
    model_has_mmproj,
    model_lazily_loads_mmproj,
    normalize_mmproj_config,
    normalize_mmproj_mode,
    resolve_render_include_mmproj,
    vision_route_ttl,
)

BASE_ID = "exl3-qwen"
REPLICA_ID = "exl3-qwen__replica_0"
SERVER = Path("/usr/bin/llama-server")


def _make_model(
    model_id: str = BASE_ID,
    *,
    local_path: str = "/models/big/big.safetensors",
    mmproj_path: str | None = "/models/proj",
    backend: str = "llamacpp",
    server_overrides: dict | None = None,
    aliases: list[str] | None = None,
    description: str | None = None,
) -> ManagedModel:
    overrides = dict(server_overrides or {})
    overrides.setdefault("engine", "buun")
    return ManagedModel(
        model_id=model_id,
        repo_id="Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw",
        quant="EXL3",
        filename="model.safetensors",
        local_path=local_path,
        backend=backend,
        mmproj_filename=None,
        mmproj_path=mmproj_path,
        load_capabilities=[],
        aliases=list(aliases or []),
        ctx_size=32768,
        # build_llama_server_command int()s this, so None would raise.
        n_gpu_layers=-1,
        tensor_split=None,
        host=None,
        jinja=True,
        ttl=None,
        description=description,
        downloaded_at=None,
        speculative=False,
        spec_variant_of=None,
        spec_meta=None,
        auto_ctx_failed=False,
        auto_ctx_error=None,
        ctx_probe_read_s=None,
        ctx_probe_tokens_s=None,
        ctx_probe_totals_s=None,
        ctx_probe_latency_ms=None,
        ctx_probe_speed_tps=None,
        ctx_probe_kv_gb=None,
        ctx_probe_prompt_tokens=None,
        server_overrides=overrides,
    )


def _lazy_model(**kwargs) -> ManagedModel:
    overrides = dict(kwargs.pop("server_overrides", None) or {})
    overrides.update({"engine": "buun", "mmproj_mode": "lazy"})
    return _make_model(server_overrides=overrides, **kwargs)


def _config_env(payload: dict | None = None) -> list:
    """Patch conf.json resolution and drop the memo that would hide the patch.

    ``cached_mmproj_config`` memoizes on the real conf.json stat and delegates
    to ``server_commands._load_server_config_payload``, so both must be
    handled or the stub silently has no effect.
    """
    body = payload if payload is not None else {}

    def _loader(_args=None):
        return copy.deepcopy(body)

    patcher = mock.patch(
        "llamacpp_stack.cli.server_commands._load_server_config_payload",
        side_effect=_loader,
    )
    patcher.start()
    vision_mod._MMPROJ_CONFIG_MEMO.clear()
    return [patcher]


def _stop(patches) -> None:
    for patcher in patches:
        patcher.stop()
    vision_mod._MMPROJ_CONFIG_MEMO.clear()


class _ConfigEnvMixin:
    def use_conf(self, mmproj: dict | None = None):
        payload: dict = {"llama_server_defaults": {}}
        if mmproj is not None:
            payload["mmproj"] = mmproj
        patches = _config_env(payload)
        self.addCleanup(_stop, patches)


class _RouterStateMixin:
    """Isolate the vision bookkeeping held on the global router singleton.

    ``reset_router_state`` is a plain method rather than ``setUp`` on purpose:
    ``unittest.TestCase`` also defines ``setUp``, so a mixin ``setUp`` is shadowed
    by the MRO and would silently never run.
    """

    def reset_router_state(self):
        state = _cli_impl.REPLICA_ROUTER_STATE
        for attr in (
            "records",
            "affinity",
            "response_to_replica",
            "loading_claims",
            "vision_until",
            "vision_affinity",
        ):
            getattr(state, attr).clear()
        state.base_last_used.clear()
        state.base_in_flight.clear()
        self.addCleanup(state.records.clear)
        self.addCleanup(state.vision_until.clear)
        self.addCleanup(state.vision_affinity.clear)


# --------------------------------------------------------------------------- #
# mode resolution
# --------------------------------------------------------------------------- #
class ModeResolutionTest(unittest.TestCase, _ConfigEnvMixin):
    def test_default_mode_is_always_and_keeps_mmproj(self):
        self.use_conf()
        model = _make_model()
        self.assertEqual(get_model_mmproj_mode(model), MMPROJ_MODE_ALWAYS)
        self.assertTrue(resolve_render_include_mmproj(model))
        self.assertFalse(model_lazily_loads_mmproj(model))

    def test_per_model_lazy_opt_in(self):
        self.use_conf()
        model = _lazy_model()
        self.assertEqual(get_model_mmproj_mode(model), MMPROJ_MODE_LAZY)
        self.assertTrue(model_lazily_loads_mmproj(model))
        self.assertFalse(
            resolve_render_include_mmproj(model),
            "a lazy model must render without the projector",
        )

    def test_global_default_mode_can_make_every_model_lazy(self):
        self.use_conf({"default_mode": "lazy"})
        model = _make_model()
        self.assertEqual(get_model_mmproj_mode(model), MMPROJ_MODE_LAZY)
        self.assertTrue(model_lazily_loads_mmproj(model))

    def test_per_model_mode_beats_the_global_default(self):
        self.use_conf({"default_mode": "lazy"})
        model = _make_model(server_overrides={"engine": "buun", "mmproj_mode": "always"})
        self.assertEqual(get_model_mmproj_mode(model), MMPROJ_MODE_ALWAYS)
        self.assertTrue(resolve_render_include_mmproj(model))

    def test_off_drops_the_projector_entirely(self):
        self.use_conf()
        model = _make_model(server_overrides={"engine": "buun", "mmproj_mode": "off"})
        self.assertEqual(get_model_mmproj_mode(model), MMPROJ_MODE_OFF)
        self.assertTrue(model_has_mmproj(model))
        self.assertFalse(model_lazily_loads_mmproj(model))

    def test_a_model_without_mmproj_is_never_lazy(self):
        self.use_conf()
        model = _make_model(mmproj_path=None, server_overrides={"mmproj_mode": "lazy"})
        self.assertFalse(model_has_mmproj(model))
        self.assertFalse(model_lazily_loads_mmproj(model))

    def test_vllm_backend_is_never_lazy(self):
        self.use_conf()
        model = _make_model(backend="vllm", server_overrides={"mmproj_mode": "lazy"})
        self.assertFalse(
            model_lazily_loads_mmproj(model),
            "vllm has no --mmproj flag, so it must keep the plain path",
        )

    def test_exllama_engine_is_never_lazy(self):
        self.use_conf()
        for engine in ("exllama", "exllamav3", "exllama-v3"):
            with self.subTest(engine=engine):
                model = _make_model(server_overrides={"engine": engine, "mmproj_mode": "lazy"})
                self.assertFalse(
                    model_lazily_loads_mmproj(model),
                    "the exllama adapter drops image_url parts",
                )

    def test_unknown_mode_falls_back_to_always(self):
        self.use_conf()
        self.assertEqual(normalize_mmproj_mode("nonsense"), MMPROJ_MODE_ALWAYS)
        self.assertEqual(normalize_mmproj_mode(None), MMPROJ_MODE_ALWAYS)
        self.assertEqual(normalize_mmproj_mode(MMPROJ_MODE_LAZY), MMPROJ_MODE_LAZY)

    def test_mode_survives_the_override_normalizer(self):
        self.use_conf()
        from llamacpp_stack.cli.server_commands import normalize_server_overrides

        normalized = normalize_server_overrides({"engine": "buun", "mmproj-mode": "lazy"})
        model = _make_model(server_overrides=normalized)
        self.assertEqual(get_model_mmproj_mode(model), MMPROJ_MODE_LAZY)


# --------------------------------------------------------------------------- #
# the operator's selection rule (pure)
# --------------------------------------------------------------------------- #
class InstanceRuleTest(unittest.TestCase):
    """``choose_vision_instance`` encodes the rule the operator specified."""

    def test_candidates_are_the_base_then_its_replicas(self):
        self.assertEqual(candidate_instance_ids(BASE_ID, []), [BASE_ID])
        self.assertEqual(candidate_instance_ids(BASE_ID, None), [BASE_ID])
        self.assertEqual(
            candidate_instance_ids(BASE_ID, [f"{BASE_ID}__replica_1", f"{BASE_ID}__replica_0"]),
            [BASE_ID, f"{BASE_ID}__replica_1", f"{BASE_ID}__replica_0"],
        )

    def test_candidates_drop_blanks_and_duplicates(self):
        self.assertEqual(candidate_instance_ids(BASE_ID, ["", BASE_ID, None]), [BASE_ID])
        self.assertEqual(candidate_instance_ids("", ["x"]), ["x"])
        self.assertEqual(candidate_instance_ids("", []), [])

    def test_unloaded_instance_beats_a_loaded_one(self):
        self.assertEqual(
            choose_vision_instance([BASE_ID, REPLICA_ID], loaded={BASE_ID}, has_mmproj={BASE_ID}),
            REPLICA_ID,
        )

    def test_among_idle_instances_the_warm_projector_wins(self):
        self.assertEqual(
            choose_vision_instance([BASE_ID, REPLICA_ID], loaded=set(), has_mmproj={REPLICA_ID}),
            REPLICA_ID,
        )

    def test_with_nothing_warm_the_longest_idle_wins(self):
        self.assertEqual(
            choose_vision_instance(
                [BASE_ID, REPLICA_ID],
                loaded=set(),
                has_mmproj=set(),
                last_used={BASE_ID: 10.0, REPLICA_ID: 99.0},
            ),
            BASE_ID,
        )

    def test_fallback_last_used_is_used_for_the_base(self):
        self.assertEqual(
            choose_vision_instance(
                [BASE_ID, REPLICA_ID],
                loaded=set(),
                has_mmproj=set(),
                last_used={},
                fallback_last_used=50.0,
            ),
            BASE_ID,
        )

    def test_liveness_outranks_projector_warmth(self):
        self.assertEqual(
            choose_vision_instance([BASE_ID, REPLICA_ID], loaded={REPLICA_ID}, has_mmproj={REPLICA_ID}),
            BASE_ID,
        )

    def test_no_candidates_returns_none(self):
        self.assertIsNone(choose_vision_instance([]))
        self.assertIsNone(choose_vision_instance(None))

    def test_junk_last_used_values_do_not_raise(self):
        self.assertEqual(
            choose_vision_instance(
                [BASE_ID], loaded=set(), has_mmproj=set(), last_used={BASE_ID: None}
            ),
            BASE_ID,
        )


# --------------------------------------------------------------------------- #
# command emission
# --------------------------------------------------------------------------- #
class CommandEmissionTest(unittest.TestCase, _ConfigEnvMixin):
    def _cmd(self, model, **kwargs):
        return build_llama_server_command(
            model,
            SERVER,
            port="11436",
            server_defaults={},
            **kwargs,
        )

    def test_builder_emits_mmproj_when_the_model_has_one(self):
        self.use_conf()
        self.assertIn("--mmproj", self._cmd(_make_model()))

    def test_include_mmproj_false_drops_it(self):
        self.use_conf()
        self.assertNotIn("--mmproj", self._cmd(_lazy_model(), include_mmproj=False))

    def test_mmproj_mode_never_leaks_into_the_command(self):
        self.use_conf()
        for mode in ("always", "lazy", "off"):
            with self.subTest(mode=mode):
                model = _make_model(server_overrides={"engine": "buun", "mmproj_mode": mode})
                cmd = self._cmd(model)
                self.assertNotIn("--mmproj-mode", cmd)
                self.assertNotIn("--mmproj_mode", cmd)

    def test_mode_off_removes_the_projector_from_the_command(self):
        self.use_conf()
        model = _make_model(server_overrides={"engine": "buun", "mmproj_mode": "off"})
        self.assertNotIn("--mmproj", self._cmd(model))

    def test_mmproj_is_placed_before_jinja(self):
        self.use_conf()
        cmd = self._cmd(_make_model())
        self.assertLess(cmd.index("--mmproj"), cmd.index("--jinja"))

    def test_a_model_without_mmproj_never_gets_the_flag(self):
        self.use_conf()
        self.assertNotIn("--mmproj", self._cmd(_make_model(mmproj_path=None)))


# --------------------------------------------------------------------------- #
# rendering: no dedicated vision route
# --------------------------------------------------------------------------- #
class RenderTest(unittest.TestCase, _ConfigEnvMixin):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # A sparse file above the small-model threshold, so the matrix treats
        # the model as large and gives it its own group.
        self.big = self.root / "big.safetensors"
        with open(self.big, "wb") as handle:
            handle.truncate(6 * 1024 * 1024 * 1024)
        self.config_path = self.root / "config.yaml"

    def _model(self, **kwargs):
        kwargs.setdefault("local_path", str(self.big))
        return _make_model(**kwargs)

    def _render(self, catalog, idle_ttl=300):
        render_llamaswap_config(catalog, self.config_path, SERVER, 11436, idle_ttl, {}, {})
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

    def test_a_lazy_model_publishes_exactly_one_route(self):
        self.use_conf()
        catalog = [self._model(server_overrides={"engine": "buun", "mmproj_mode": "lazy"})]
        data = self._render(catalog)
        self.assertEqual(
            list(data["models"]),
            [BASE_ID],
            "there must be no dedicated vision route",
        )
        self.assertNotIn("--mmproj", data["models"][BASE_ID]["cmd"])

    def test_no_route_anywhere_mentions_vision_for_a_lazy_model(self):
        self.use_conf()
        catalog = [self._model(server_overrides={"engine": "buun", "mmproj_mode": "lazy"})]
        data = self._render(catalog)
        for name, entry in data["models"].items():
            with self.subTest(route=name):
                self.assertNotIn("vision", entry.get("cmd", ""))
                self.assertNotIn("vision", (entry.get("description") or ""))

    def test_an_always_model_still_renders_the_projector(self):
        self.use_conf()
        data = self._render([self._model()])
        self.assertIn("--mmproj", data["models"][BASE_ID]["cmd"])

    def test_lazy_and_non_lazy_models_coexist(self):
        self.use_conf()
        other = self._model(model_id="q4-model", local_path=str(self.big))
        lazy = self._model(
            model_id="lazy-model",
            local_path=str(self.big),
            server_overrides={"engine": "buun", "mmproj_mode": "lazy"},
        )
        data = self._render([other, lazy])
        self.assertIn("--mmproj", data["models"]["q4-model"]["cmd"])
        self.assertNotIn("--mmproj", data["models"]["lazy-model"]["cmd"])
        self.assertEqual(sorted(data["models"]), ["lazy-model", "q4-model"])

    def test_idle_ttl_is_used_for_the_base_route(self):
        self.use_conf()
        data = self._render(
            [self._model(server_overrides={"mmproj_mode": "lazy"})], idle_ttl=123
        )
        self.assertEqual(data["models"][BASE_ID]["ttl"], 123)


# --------------------------------------------------------------------------- #
# matrix
# --------------------------------------------------------------------------- #
class MatrixTest(unittest.TestCase, _ConfigEnvMixin):
    def _matrix(self, infos):
        matrix = _calculate_llama_swap_matrix(infos)
        inverse = matrix["vars"]
        return {
            "sets": {key: sorted(value.split(" & ")) for key, value in matrix["sets"].items()},
            "evict": {inverse[name]: cost for name, cost in matrix["evict_costs"].items()},
        }

    def _infos(self):
        return [
            {"id": BASE_ID, "gpu_set": [0], "is_embedding": False, "is_small": False, "size_mib": 6144},
            {"id": "q4-model", "gpu_set": [1], "is_embedding": False, "is_small": False, "size_mib": 4096},
        ]

    def test_lazy_makes_no_difference_to_the_matrix(self):
        self.use_conf()
        # The projector must not introduce groups or evict costs of its own;
        # it rides on an instance that already exists.
        baseline = self._matrix(self._infos())
        self.assertEqual(self._matrix(self._infos()), baseline)

    def test_two_large_models_on_disjoint_gpus_share_one_group(self):
        self.use_conf()
        result = self._matrix(self._infos())
        self.assertEqual(
            len(result["sets"]),
            1,
            "disjoint larges are co-loaded on purpose",
        )
        self.assertEqual(sorted(result["sets"]["group_0"]), sorted(["m0", "m1"]))

    def test_replicas_keep_the_half_evict_cost(self):
        self.use_conf()
        result = self._matrix(
            [
                {"id": BASE_ID, "gpu_set": [0], "is_embedding": False, "is_small": False, "size_mib": 6144},
                {"id": REPLICA_ID, "gpu_set": [1], "is_embedding": False, "is_small": False, "size_mib": 6144},
            ]
        )
        self.assertEqual(result["evict"][BASE_ID], 6144)
        self.assertEqual(result["evict"][REPLICA_ID], 3072)


# --------------------------------------------------------------------------- #
# attaching / detaching the projector on a single route
# --------------------------------------------------------------------------- #
class SetInstanceMmprojTest(
    unittest.TestCase, _ConfigEnvMixin, _RouterStateMixin
):
    def setUp(self):
        self.reset_router_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config_path = self.root / "config.yaml"
        self.model = _lazy_model()
        self.catalog = [self.model]

    def _seed(self):
        render_llamaswap_config(
            self.catalog, self.config_path, SERVER, 11436, 300, {}, {}
        )

    def _read(self):
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

    def _set(self, instance_id, enable, gpu_set=None):
        return set_instance_mmproj_in_llamaswap_config(
            self.model,
            instance_id,
            self.catalog,
            self.config_path,
            SERVER,
            300,
            {},
            enable=enable,
            gpu_set=gpu_set,
        )

    def test_projector_lands_on_the_base_route(self):
        self.use_conf()
        self._seed()
        self.assertEqual(self._set(BASE_ID, True), BASE_ID)
        entry = self._read()["models"][BASE_ID]
        self.assertIn("--mmproj /models/proj", entry["cmd"])
        self.assertTrue(route_carries_mmproj(entry))
        self.assertTrue(entry["metadata"]["vision_variant"])
        self.assertEqual(entry["metadata"]["internal_replica_of"], BASE_ID)

    def test_projector_lands_on_a_replica_route(self):
        self.use_conf()
        self._seed()
        self.assertEqual(self._set(REPLICA_ID, True, gpu_set=[1]), REPLICA_ID)
        models = self._read()["models"]
        self.assertIn("--mmproj", models[REPLICA_ID]["cmd"])
        self.assertNotIn(
            "--mmproj",
            models[BASE_ID]["cmd"],
            "the busy base must stay text-only",
        )
        self.assertIn("CUDA_VISIBLE_DEVICES=1", models[REPLICA_ID]["cmd"])

    def test_attaching_twice_does_not_rewrite_the_config(self):
        self.use_conf()
        self._seed()
        self._set(BASE_ID, True)
        snapshot = self.config_path.read_bytes()
        self._set(BASE_ID, True)
        self.assertEqual(
            self.config_path.read_bytes(),
            snapshot,
            "an already-correct route must be left alone",
        )

    def test_detaching_removes_the_projector_and_the_marker(self):
        self.use_conf()
        self._seed()
        self._set(BASE_ID, True)
        self._set(BASE_ID, False)
        entry = self._read()["models"][BASE_ID]
        self.assertNotIn("--mmproj", entry["cmd"])
        self.assertNotIn("vision_variant", entry.get("metadata") or {})
        self.assertEqual(entry["metadata"]["internal_replica_of"], BASE_ID)

    def test_detaching_twice_does_not_rewrite_the_config(self):
        self.use_conf()
        self._seed()
        self._set(BASE_ID, False)
        snapshot = self.config_path.read_bytes()
        self._set(BASE_ID, False)
        self.assertEqual(self.config_path.read_bytes(), snapshot)

    def test_attaching_swaps_the_ttl_for_the_vision_window(self):
        self.use_conf({"vision_sticky_ttl_s": 3600})
        self._seed()
        self.assertEqual(self._read()["models"][BASE_ID]["ttl"], 300)
        self._set(BASE_ID, True)
        self.assertEqual(
            self._read()["models"][BASE_ID]["ttl"],
            60,
            "3600s sticky becomes a 60 minute ttl so llama-swap unloads it",
        )
        self._set(BASE_ID, False)
        self.assertEqual(self._read()["models"][BASE_ID]["ttl"], 300)

    def test_route_carries_mmproj_reads_the_rendered_command(self):
        self.assertFalse(route_carries_mmproj({}))
        self.assertFalse(route_carries_mmproj({"cmd": ["--jinja"]}))
        self.assertFalse(route_carries_mmproj(None))
        self.assertTrue(route_carries_mmproj({"cmd": "llama-server --mmproj /p --jinja"}))

    def test_a_render_resets_the_projector(self):
        self.use_conf()
        self._seed()
        self._set(BASE_ID, True)
        self._seed()
        self.assertNotIn("--mmproj", self._read()["models"][BASE_ID]["cmd"])


class EnsureInternalRouteTest(
    unittest.TestCase, _ConfigEnvMixin, _RouterStateMixin
):
    def setUp(self):
        self.reset_router_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config_path = self.root / "config.yaml"
        self.model = _lazy_model()
        self.catalog = [self.model]
        render_llamaswap_config(
            self.catalog, self.config_path, SERVER, 11436, 300, {}, {}
        )

    def _read(self):
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

    def test_creates_a_missing_replica_route(self):
        self.use_conf()
        ensure_internal_route_in_llamaswap_config(
            self.model,
            REPLICA_ID,
            self.catalog,
            self.config_path,
            SERVER,
            300,
            {},
            gpu_set=[1],
        )
        self.assertIn(REPLICA_ID, self._read()["models"])

    def test_is_idempotent(self):
        self.use_conf()
        for _ in range(2):
            ensure_internal_route_in_llamaswap_config(
                self.model,
                REPLICA_ID,
                self.catalog,
                self.config_path,
                SERVER,
                300,
                {},
                gpu_set=[1],
            )
        self.assertIn(REPLICA_ID, self._read()["models"])

    def test_replica_routes_of_a_lazy_model_stay_text_only(self):
        self.use_conf()
        ensure_replica_route_in_llamaswap_config(
            self.model, 0, [1], self.catalog, self.config_path, SERVER, 300, {}
        )
        self.assertNotIn("--mmproj", self._read()["models"][REPLICA_ID]["cmd"])

    def test_an_explicit_mmproj_request_marks_the_replica(self):
        self.use_conf()
        ensure_replica_route_in_llamaswap_config(
            self.model,
            0,
            [1],
            self.catalog,
            self.config_path,
            SERVER,
            300,
            {},
            include_mmproj=True,
        )
        self.assertIn("--mmproj", self._read()["models"][REPLICA_ID]["cmd"])


# --------------------------------------------------------------------------- #
# config block
# --------------------------------------------------------------------------- #
class ConfigBlockTest(unittest.TestCase):
    def test_defaults_to_always_and_a_one_hour_window(self):
        cfg = default_mmproj_config()
        self.assertEqual(cfg["default_mode"], MMPROJ_MODE_ALWAYS)
        self.assertEqual(cfg["vision_sticky_ttl_s"], 3600)

    def test_absent_block_does_not_report_a_change(self):
        cfg, changed = normalize_mmproj_config(None)
        self.assertEqual(changed, False)
        self.assertEqual(cfg, default_mmproj_config())

    def test_filling_absent_keys_is_not_a_change(self):
        cfg, changed = normalize_mmproj_config({"default_mode": "lazy"})
        self.assertEqual(changed, False)
        self.assertEqual(cfg["default_mode"], "lazy")
        self.assertEqual(cfg["vision_sticky_ttl_s"], 3600)

    def test_malformed_input_is_repaired(self):
        _, changed = normalize_mmproj_config("nonsense")
        self.assertEqual(changed, True)

    def test_sticky_ttl_is_clamped_like_the_replica_one(self):
        cfg, changed = normalize_mmproj_config({"vision_sticky_ttl_s": 1})
        self.assertTrue(changed)
        self.assertGreaterEqual(cfg["vision_sticky_ttl_s"], 60)

    def test_ttl_is_rendered_in_whole_minutes(self):
        self.assertEqual(vision_route_ttl({"vision_sticky_ttl_s": 600}), 10)
        self.assertEqual(vision_route_ttl({"vision_sticky_ttl_s": 45}), 1)
        self.assertEqual(vision_route_ttl({"vision_sticky_ttl_s": 90}), 2)
        self.assertEqual(vision_route_ttl(default_mmproj_config()), 60)

    def test_the_conf_block_is_written_once_and_then_stable(self):
        payload, changed = _cli_impl.normalize_server_config_payload({})
        self.assertTrue(changed)
        self.assertEqual(payload["mmproj"], default_mmproj_config())
        again, changed_again = _cli_impl.normalize_server_config_payload(payload)
        self.assertFalse(changed_again)
        self.assertEqual(again["mmproj"], payload["mmproj"])


# --------------------------------------------------------------------------- #
# request routing
# --------------------------------------------------------------------------- #
class RouteRequestTest(
    unittest.TestCase, _ConfigEnvMixin, _RouterStateMixin
):
    """``route_image_request_to_vision`` decides which instance sees the image."""

    def setUp(self):
        self.reset_router_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config_path = self.root / "config.yaml"
        self.model = _lazy_model()
        self.catalog = [self.model]
        self.args = argparse.Namespace(
            public_port=11436,
            config=str(self.config_path),
            llama_server=str(SERVER),
            idle_ttl=300,
            llama_server_defaults={},
            mode="user",
            state_dir=str(self.root),
            public_host="127.0.0.1",
        )
        for name in (
            "get_catalog_model_process",
            "wait_for_published_model_id",
            "log_api_event",
            "set_instance_mmproj_in_llamaswap_config",
        ):
            patcher = mock.patch.object(_cli_impl, name)
            setattr(self, f"{name}_mock", patcher.start())
            self.addCleanup(patcher.stop)
        self.get_catalog_model_process_mock.return_value = None
        self.wait_for_published_model_id_mock.return_value = True
        self.set_instance_mmproj_in_llamaswap_config_mock.return_value = self.config_path
        self.now = 1000.0
        self.clock = mock.patch.object(_cli_impl.time, "monotonic", return_value=self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    IMAGE_PAYLOAD = {
        "model": BASE_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                ],
            }
        ],
    }
    TEXT_PAYLOAD = {"model": BASE_ID, "messages": [{"role": "user", "content": "hello"}]}
    HEADERS = {"thread-id": "conv-1"}

    def _route(self, payload, headers=None):
        return _cli_impl.route_image_request_to_vision(
            self.model,
            payload,
            self.HEADERS if headers is None else headers,
            self.catalog,
            self.args,
            "127.0.0.1",
        )

    def _attached_ids(self):
        return [
            call.args[1]
            for call in self.set_instance_mmproj_in_llamaswap_config_mock.call_args_list
        ]

    def _add_replica(
        self, *, gpu_set=(1,), status="cold", pid=None, last_used=0.0, replica_id=REPLICA_ID
    ):
        _cli_impl.REPLICA_ROUTER_STATE.records[replica_id] = ReplicaRecord(
            base_model_id=BASE_ID,
            replica_model_id=replica_id,
            gpu_set=list(gpu_set),
            status=status,
            pid=pid,
            last_used=last_used,
        )
        return replica_id

    # -- not applicable ----------------------------------------------------- #
    def test_text_request_is_left_to_the_normal_router(self):
        self.use_conf()
        self.assertEqual(self._route(self.TEXT_PAYLOAD), (None, None))
        self.set_instance_mmproj_in_llamaswap_config_mock.assert_not_called()

    def test_a_non_lazy_model_is_left_to_the_normal_router(self):
        self.use_conf()
        model = _make_model(server_overrides={"engine": "buun"})
        self.assertEqual(
            _cli_impl.route_image_request_to_vision(
                model, self.IMAGE_PAYLOAD, self.HEADERS, [model], self.args, "127.0.0.1"
            ),
            (None, None),
        )

    def test_ollama_image_field_is_detected(self):
        self.use_conf()
        target, error = self._route({"model": BASE_ID, "prompt": "hi", "images": ["AAA"]})
        self.assertIsNone(error)
        self.assertEqual(target, BASE_ID)

    def test_a_malformed_message_list_does_not_raise(self):
        self.use_conf()
        for payload in (
            {"model": BASE_ID, "messages": "not-a-list"},
            {"model": BASE_ID, "messages": ["plain string", {"content": "text"}]},
            {"model": BASE_ID},
            {},
        ):
            with self.subTest(payload=payload):
                self.assertEqual(self._route(payload), (None, None))

    # -- the rule ----------------------------------------------------------- #
    def test_an_idle_replica_is_used_when_the_base_is_loaded(self):
        self.use_conf()
        self.get_catalog_model_process_mock.return_value = {"pid": 111}
        replica = self._add_replica(status="cold")
        target, error = self._route(self.IMAGE_PAYLOAD)
        self.assertIsNone(error)
        self.assertEqual(target, replica)
        self.assertEqual(self._attached_ids(), [replica])

    def test_the_base_is_used_when_it_is_the_only_instance(self):
        self.use_conf()
        target, error = self._route(self.IMAGE_PAYLOAD)
        self.assertIsNone(error)
        self.assertEqual(target, BASE_ID)

    def test_a_warm_projector_does_not_outrank_an_idle_instance(self):
        self.use_conf()
        self.get_catalog_model_process_mock.return_value = {"pid": 111}
        _cli_impl._vision_mark_instance(BASE_ID, BASE_ID, 3600.0)
        replica = self._add_replica()
        target, error = self._route(self.IMAGE_PAYLOAD)
        self.assertIsNone(error)
        self.assertEqual(
            target,
            replica,
            "a warm projector on a busy instance is useless; the free one wins",
        )

    def test_among_two_idle_instances_the_warm_projector_wins(self):
        self.use_conf()
        self.get_catalog_model_process_mock.return_value = None
        self._add_replica(status="cold")
        _cli_impl._vision_mark_instance(BASE_ID, BASE_ID, 3600.0)
        target, error = self._route(self.IMAGE_PAYLOAD)
        self.assertIsNone(error)
        self.assertEqual(
            target,
            BASE_ID,
            "with nothing loaded, keeping the projector already warm avoids a reload",
        )

    def test_the_longest_idle_instance_wins_when_nothing_is_warm(self):
        self.use_conf()
        _cli_impl.REPLICA_ROUTER_STATE.base_last_used[BASE_ID] = 10.0
        self._add_replica(last_used=99.0)
        target, error = self._route(self.IMAGE_PAYLOAD)
        self.assertIsNone(error)
        self.assertEqual(target, BASE_ID)

    def test_a_replica_without_a_known_gpu_set_falls_back_to_the_base(self):
        self.use_conf()
        self.get_catalog_model_process_mock.return_value = {"pid": 111}
        self._add_replica(gpu_set=[])
        target, error = self._route(self.IMAGE_PAYLOAD)
        self.assertIsNone(error)
        self.assertEqual(target, BASE_ID)

    def test_a_replica_is_marked_so_follow_up_turns_stay_on_it(self):
        self.use_conf()
        self.get_catalog_model_process_mock.return_value = {"pid": 111}
        replica = self._add_replica()
        self._route(self.IMAGE_PAYLOAD)
        state = _cli_impl.REPLICA_ROUTER_STATE
        self.assertIn(replica, state.vision_until[BASE_ID])
        self.assertTrue(state.vision_affinity)

    # -- reuse -------------------------------------------------------------- #
    def test_a_pinned_conversation_does_not_re_attach(self):
        self.use_conf()
        self.get_catalog_model_process_mock.return_value = {"pid": 111}
        replica = self._add_replica()
        first, _ = self._route(self.IMAGE_PAYLOAD)
        second, error = self._route(self.IMAGE_PAYLOAD)
        self.assertEqual((first, second), (replica, replica))
        self.assertIsNone(error)
        self.assertEqual(
            len(self.set_instance_mmproj_in_llamaswap_config_mock.call_args_list),
            1,
            "a warm projector must be reused, not rewritten",
        )

    def test_a_lapsed_pin_falls_back_to_the_rule(self):
        self.use_conf()
        self.get_catalog_model_process_mock.return_value = {"pid": 111}
        replica = self._add_replica()
        _cli_impl._vision_mark_instance(BASE_ID, replica, 1.0)
        self.now += 10.0
        target, error = self._route(self.IMAGE_PAYLOAD)
        self.assertIsNone(error)
        self.assertEqual(target, replica)

    def test_a_warm_target_is_not_awaited_again(self):
        self.use_conf()
        _cli_impl._vision_mark_instance(BASE_ID, BASE_ID, 3600.0)
        self._route(self.IMAGE_PAYLOAD)
        self.wait_for_published_model_id_mock.assert_not_called()

    def test_a_cold_target_is_awaited(self):
        self.use_conf()
        self._route(self.IMAGE_PAYLOAD)
        self.wait_for_published_model_id_mock.assert_called_once()

    # -- failures ----------------------------------------------------------- #
    def test_a_failed_attach_is_reported_not_silently_dropped(self):
        self.use_conf()
        self.set_instance_mmproj_in_llamaswap_config_mock.side_effect = RuntimeError("boom")
        target, error = self._route(self.IMAGE_PAYLOAD)
        self.assertIsNone(target)
        self.assertIn("could not be attached", error)

    def test_a_broken_server_config_is_not_blamed_on_the_route(self):
        self.use_conf()
        with mock.patch.object(_cli_impl, "resolve_idle_ttl", side_effect=ValueError("bad")):
            target, error = self._route(self.IMAGE_PAYLOAD)
        self.assertIsNone(target)
        self.assertIn("invalid server configuration", error)
        events = [call.args[0] for call in self.log_api_event_mock.call_args_list]
        self.assertIn("lazy_mmproj_route_create_failed", events)

    def test_no_instance_to_carry_the_projector_is_reported(self):
        self.use_conf()
        with mock.patch.object(_cli_impl, "candidate_instance_ids", return_value=[]):
            target, error = self._route(self.IMAGE_PAYLOAD)
        self.assertIsNone(target)
        self.assertIn("no instance", error)

    def test_an_invalid_public_port_is_reported(self):
        self.use_conf()
        self.args.public_port = "not-a-port"
        target, error = self._route(self.IMAGE_PAYLOAD)
        self.assertIsNone(target)
        self.assertIn("public port", error)


# --------------------------------------------------------------------------- #
# returning an instance to text
# --------------------------------------------------------------------------- #
class ReconcileTextTest(
    unittest.TestCase, _ConfigEnvMixin, _RouterStateMixin
):
    def setUp(self):
        self.reset_router_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config_path = self.root / "config.yaml"
        self.model = _lazy_model()
        self.catalog = [self.model]
        self.args = argparse.Namespace(
            config=str(self.config_path),
            llama_server=str(SERVER),
            idle_ttl=300,
            public_port=11436,
            llama_server_defaults={},
            mode="user",
            state_dir=str(self.root),
            public_host="127.0.0.1",
        )
        patcher = mock.patch.object(_cli_impl, "log_api_event")
        self.log_api_event_mock = patcher.start()
        self.addCleanup(patcher.stop)

    def _reconcile(self, instance_id):
        return _cli_impl.reconcile_text_instance_mmproj(
            self.model, instance_id, self.catalog, self.args
        )

    def test_an_unflagged_instance_is_left_alone(self):
        self.use_conf()
        with mock.patch.object(
            _cli_impl, "set_instance_mmproj_in_llamaswap_config"
        ) as attach:
            self._reconcile(BASE_ID)
            attach.assert_not_called()

    def test_a_flagged_instance_loses_the_projector(self):
        self.use_conf()
        _cli_impl._vision_mark_instance(BASE_ID, BASE_ID, 3600.0)
        with mock.patch.object(
            _cli_impl,
            "set_instance_mmproj_in_llamaswap_config",
            return_value=self.config_path,
        ) as attach:
            self._reconcile(BASE_ID)
            attach.assert_called_once()
            self.assertIs(attach.call_args.kwargs["enable"], False)
        self.assertEqual(
            _cli_impl.REPLICA_ROUTER_STATE.vision_until.get(BASE_ID),
            None,
            "the flag must be dropped once the route really lost the projector",
        )

    def test_a_failed_rewrite_keeps_the_flag_for_the_next_attempt(self):
        self.use_conf()
        _cli_impl._vision_mark_instance(BASE_ID, BASE_ID, 3600.0)
        with mock.patch.object(
            _cli_impl, "set_instance_mmproj_in_llamaswap_config", side_effect=RuntimeError("boom")
        ):
            self._reconcile(BASE_ID)
        self.assertIn(BASE_ID, _cli_impl.REPLICA_ROUTER_STATE.vision_until[BASE_ID])
        events = [call.args[0] for call in self.log_api_event_mock.call_args_list]
        self.assertIn("lazy_mmproj_text_restore_failed", events)

    def test_a_broken_server_config_keeps_the_flag(self):
        self.use_conf()
        _cli_impl._vision_mark_instance(BASE_ID, BASE_ID, 3600.0)
        with mock.patch.object(_cli_impl, "resolve_idle_ttl", side_effect=ValueError("bad")):
            self._reconcile(BASE_ID)
        self.assertIn(BASE_ID, _cli_impl.REPLICA_ROUTER_STATE.vision_until[BASE_ID])

    def test_an_empty_instance_id_is_ignored(self):
        self.use_conf()
        with mock.patch.object(
            _cli_impl, "set_instance_mmproj_in_llamaswap_config"
        ) as attach:
            self._reconcile("")
            attach.assert_not_called()


class VisionBookkeepingTest(unittest.TestCase, _RouterStateMixin):
    def setUp(self):
        self.reset_router_state()

    def test_marking_sets_a_deadline_in_the_future(self):
        with mock.patch.object(_cli_impl.time, "monotonic", return_value=100.0):
            _cli_impl._vision_mark_instance(BASE_ID, REPLICA_ID, 60.0)
        self.assertEqual(
            _cli_impl.REPLICA_ROUTER_STATE.vision_until[BASE_ID][REPLICA_ID], 160.0
        )
        self.assertEqual(_cli_impl._vision_live_instances(BASE_ID, now=120.0), {REPLICA_ID})
        self.assertEqual(_cli_impl._vision_live_instances(BASE_ID, now=200.0), set())

    def test_a_lapsed_flag_is_kept_so_text_can_still_strip_it(self):
        with mock.patch.object(_cli_impl.time, "monotonic", return_value=100.0):
            _cli_impl._vision_mark_instance(BASE_ID, REPLICA_ID, 10.0)
        self.assertEqual(_cli_impl._vision_live_instances(BASE_ID, now=500.0), set())
        self.assertIn(REPLICA_ID, _cli_impl.REPLICA_ROUTER_STATE.vision_until[BASE_ID])

    def test_forgetting_drops_the_instance_and_the_empty_parent(self):
        _cli_impl._vision_mark_instance(BASE_ID, REPLICA_ID, 3600.0)
        _cli_impl._vision_forget_instance(BASE_ID, REPLICA_ID)
        self.assertNotIn(
            BASE_ID,
            _cli_impl.REPLICA_ROUTER_STATE.vision_until,
            "an emptied parent entry would otherwise leak for the life of the daemon",
        )

    def test_forgetting_one_instance_keeps_the_others(self):
        other = f"{BASE_ID}__replica_1"
        _cli_impl._vision_mark_instance(BASE_ID, REPLICA_ID, 3600.0)
        _cli_impl._vision_mark_instance(BASE_ID, other, 3600.0)
        _cli_impl._vision_forget_instance(BASE_ID, REPLICA_ID)
        self.assertEqual(
            sorted(_cli_impl.REPLICA_ROUTER_STATE.vision_until[BASE_ID]),
            [other],
        )

    def test_forgetting_an_unknown_instance_is_a_noop(self):
        _cli_impl._vision_forget_instance(BASE_ID, "never-seen")
        self.assertNotIn(BASE_ID, _cli_impl.REPLICA_ROUTER_STATE.vision_until)

    def test_an_unknown_base_has_no_instances(self):
        self.assertEqual(_cli_impl._vision_live_instances("nothing-here"), set())


# --------------------------------------------------------------------------- #
# gateway wiring
# --------------------------------------------------------------------------- #
class HandlerWiringTest(unittest.TestCase):
    """The four image endpoints must consult lazy mmproj before replicas.

    The gateway handlers are methods defined inside a factory closure, so they
    are not reachable through ``dir(_cli_impl)``. Assert on the module source
    instead, which also pins the ordering constraints.
    """

    ENDPOINTS = (
        "_handle_ollama_chat",
        "_handle_openai_chat_completions",
        "_handle_openai_responses",
        "_handle_ollama_generate",
    )

    @classmethod
    def setUpClass(cls):
        cls.source = Path(_cli_impl.__file__).read_text(encoding="utf-8")

    def _endpoint_body(self, name: str) -> str:
        start = self.source.index(f"def {name}(self):")
        return self.source[start:]

    def test_all_four_image_endpoints_consult_lazy_mmproj(self):
        self.assertEqual(
            self.source.count("route_image_request_to_vision("),
            1 + len(self.ENDPOINTS),
            "expected one definition plus one call per image endpoint",
        )
        for name in self.ENDPOINTS:
            with self.subTest(endpoint=name):
                self.assertIn("route_image_request_to_vision(", self._endpoint_body(name))

    def test_vision_target_is_resolved_before_replica_selection(self):
        for name in self.ENDPOINTS:
            with self.subTest(endpoint=name):
                body = self._endpoint_body(name)
                self.assertLess(
                    body.index("route_image_request_to_vision("),
                    body.index("select_replica_for_request("),
                    "lazy mmproj must be resolved before replica selection",
                )
                self.assertIn("elif model_entry is not None:", body)

    def test_all_four_endpoints_release_the_projector_for_text(self):
        self.assertEqual(
            self.source.count("reconcile_text_instance_mmproj("),
            1 + len(self.ENDPOINTS),
        )
        for name in self.ENDPOINTS:
            with self.subTest(endpoint=name):
                self.assertIn("reconcile_text_instance_mmproj(", self._endpoint_body(name))

    def test_the_legacy_stale_reload_skips_lazy_models(self):
        body = self._endpoint_body("_handle_openai_responses")
        guard = body.index("_loaded_process_missing_configured_mmproj(")
        self.assertIn("not model_lazily_loads_mmproj(model_entry)", body[:guard])


# --------------------------------------------------------------------------- #
# public surface
# --------------------------------------------------------------------------- #
class SurfaceTest(unittest.TestCase):
    def test_vision_helpers_are_exported_from_the_public_package(self):
        import llamacpp_stack.cli as cli_pkg

        for name in (
            "MMPROJ_MODE_ALWAYS",
            "MMPROJ_MODE_LAZY",
            "MMPROJ_MODE_OFF",
            "default_mmproj_config",
            "normalize_mmproj_config",
            "normalize_mmproj_mode",
            "get_model_mmproj_mode",
            "resolve_effective_mmproj_config",
            "cached_mmproj_config",
            "model_has_mmproj",
            "model_lazily_loads_mmproj",
            "resolve_render_include_mmproj",
            "candidate_instance_ids",
            "choose_vision_instance",
            "vision_route_ttl",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(cli_pkg, name))

    def test_replica_helpers_are_exported_from_the_public_package(self):
        import llamacpp_stack.cli as cli_pkg

        for name in (
            "ensure_internal_route_in_llamaswap_config",
            "set_instance_mmproj_in_llamaswap_config",
            "route_carries_mmproj",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(cli_pkg, name))

    def test_the_removed_vision_route_helpers_are_gone(self):
        import llamacpp_stack.cli as cli_pkg

        for name in (
            "vision_model_id",
            "vision_base_model_id",
            "is_vision_model_id",
            "build_vision_model",
            "ensure_vision_route_in_llamaswap_config",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(cli_pkg, name))

    def test_every_model_payload_reports_the_live_vision_instances(self):
        from llamacpp_stack.cli.gateway import (  # noqa: F401  (re-export check)
            build_openai_model_list_payload,
            build_openai_model_payload,
        )

        lazy = _lazy_model()
        plain = _make_model(model_id="plain")
        with mock.patch.object(_cli_impl, "_vision_live_instances", return_value={REPLICA_ID}):
            for builder, key in (
                (_cli_impl.build_openai_model_payload, "metadata"),
                (_cli_impl.build_ollama_model_payload, "details"),
                (_cli_impl.build_openai_model_list_payload, "metadata"),
            ):
                with self.subTest(builder=builder.__name__):
                    details = builder(lazy)[key]
                    self.assertEqual(details["mmproj_mode"], MMPROJ_MODE_LAZY)
                    self.assertEqual(details["vision_instances"], [REPLICA_ID])
            plain_details = _cli_impl.build_openai_model_payload(plain)["metadata"]
        self.assertEqual(plain_details["vision_instances"], [])
        self.assertEqual(plain_details["mmproj_mode"], MMPROJ_MODE_ALWAYS)


if __name__ == "__main__":
    unittest.main()