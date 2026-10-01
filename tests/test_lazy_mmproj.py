"""Tests for lazy mmproj (``mmproj_mode: lazy``).

A lazy model renders a text-only base route plus a ``__vision`` sibling that
carries ``--mmproj``. Image-bearing requests are pointed at the sibling, or at
an in-place reload of an already-loaded base when the sibling cannot be
published.
"""

from __future__ import annotations

import argparse
import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from llamacpp_stack import _cli_impl
from llamacpp_stack.cli import ManagedModel
from llamacpp_stack.cli import vision as vision_mod
from llamacpp_stack.cli.replica import (
    _calculate_llama_swap_matrix,
    ensure_vision_route_in_llamaswap_config,
    render_llamaswap_config,
)
from llamacpp_stack.cli.server_commands import build_llama_server_command
from llamacpp_stack.cli.vision import (
    MMPROJ_MODE_ALWAYS,
    MMPROJ_MODE_LAZY,
    MMPROJ_MODE_OFF,
    build_vision_model,
    default_mmproj_config,
    get_model_mmproj_mode,
    is_vision_model_id,
    model_has_mmproj,
    model_lazily_loads_mmproj,
    normalize_mmproj_config,
    normalize_mmproj_mode,
    resolve_render_include_mmproj,
    vision_base_model_id,
    vision_model_id,
    vision_route_ttl,
)

BASE_ID = "exl3-qwen"
VISION_ID = "exl3-qwen__vision"


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


def _config_env(payload: dict | None = None):
    """Patch conf.json resolution and drop the vision config memo.

    ``cached_mmproj_config`` memoizes on the real conf.json stat and delegates
    to ``server_commands._load_server_config_payload``, so both must be handled.
    """
    body = payload if payload is not None else {}

    def _loader(_args=None):
        return copy.deepcopy(body)

    patches = [
        mock.patch(
            "llamacpp_stack.cli.server_commands._load_server_config_payload",
            side_effect=_loader,
        )
    ]
    for patcher in patches:
        patcher.start()
    vision_mod._MMPROJ_CONFIG_MEMO.clear()
    return patches


def _stop(patches):
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


class ModeResolutionTest(unittest.TestCase, _ConfigEnvMixin):
    def test_default_mode_is_always_and_keeps_mmproj(self):
        model = _make_model()
        self.assertEqual(get_model_mmproj_mode(model), MMPROJ_MODE_ALWAYS)
        self.assertTrue(resolve_render_include_mmproj(model))
        self.assertFalse(model_lazily_loads_mmproj(model))

    def test_per_model_lazy_opt_in(self):
        model = _make_model(server_overrides={"engine": "buun", "mmproj_mode": "lazy"})
        self.assertEqual(get_model_mmproj_mode(model), MMPROJ_MODE_LAZY)
        self.assertTrue(model_lazily_loads_mmproj(model))
        # The text-only base must not carry the projector.
        self.assertFalse(resolve_render_include_mmproj(model))

    def test_global_default_mode_applies_without_per_model_key(self):
        self.use_conf({"default_mode": "lazy"})
        model = _make_model()
        self.assertEqual(get_model_mmproj_mode(model), MMPROJ_MODE_LAZY)
        self.assertTrue(model_lazily_loads_mmproj(model))

    def test_per_model_mode_wins_over_global_default(self):
        self.use_conf({"default_mode": "lazy"})
        model = _make_model(server_overrides={"engine": "buun", "mmproj_mode": "always"})
        self.assertEqual(get_model_mmproj_mode(model), MMPROJ_MODE_ALWAYS)
        self.assertFalse(model_lazily_loads_mmproj(model))

    def test_mode_off_drops_mmproj_without_a_sibling(self):
        model = _make_model(server_overrides={"engine": "buun", "mmproj_mode": "off"})
        self.assertEqual(get_model_mmproj_mode(model), MMPROJ_MODE_OFF)
        self.assertFalse(model_lazily_loads_mmproj(model))
        self.assertFalse(resolve_render_include_mmproj(model))

    def test_unknown_mode_falls_back_to_always(self):
        self.assertEqual(normalize_mmproj_mode("sometimes"), MMPROJ_MODE_ALWAYS)
        self.assertEqual(normalize_mmproj_mode(None), MMPROJ_MODE_ALWAYS)
        self.assertEqual(normalize_mmproj_mode("LAZY"), MMPROJ_MODE_LAZY)

    def test_model_without_mmproj_is_never_lazy(self):
        model = _make_model(
            mmproj_path=None,
            server_overrides={"engine": "buun", "mmproj_mode": "lazy"},
        )
        self.assertFalse(model_has_mmproj(model))
        self.assertFalse(model_lazily_loads_mmproj(model))
        self.assertFalse(resolve_render_include_mmproj(model))

    def test_vllm_and_exllama_are_excluded_from_lazy(self):
        vllm = _make_model(
            backend="vllm",
            server_overrides={"engine": "buun", "mmproj_mode": "lazy"},
        )
        self.assertFalse(model_lazily_loads_mmproj(vllm))
        for engine in ("exllama", "exllamav3", "exllama-v3", "exllama3"):
            exllama = _make_model(
                server_overrides={"engine": engine, "mmproj_mode": "lazy"},
            )
            with self.subTest(engine=engine):
                self.assertFalse(model_lazily_loads_mmproj(exllama))

    def test_normalize_mmproj_config_is_idempotent_and_fills_absent_keys(self):
        defaults = default_mmproj_config()
        first, changed_first = normalize_mmproj_config(None)
        self.assertEqual(first, defaults)
        self.assertFalse(changed_first)

        partial = {"default_mode": "lazy"}
        second, changed_second = normalize_mmproj_config(partial)
        # Absent keys are filled in but that alone is not a migration.
        self.assertFalse(changed_second)
        self.assertEqual(second["default_mode"], "lazy")
        self.assertEqual(second["prefer_vision_route"], defaults["prefer_vision_route"])

        third, changed_third = normalize_mmproj_config(second)
        self.assertEqual(third, second)
        self.assertFalse(changed_third)

    def test_normalize_mmproj_config_rejects_malformed_input(self):
        normalized, changed = normalize_mmproj_config("nope")
        self.assertTrue(changed)
        self.assertEqual(normalized, default_mmproj_config())

    def test_vision_route_ttl_maps_sticky_seconds_to_minutes(self):
        self.assertEqual(vision_route_ttl({"vision_sticky_ttl_s": 600}), 10)
        self.assertEqual(vision_route_ttl({"vision_sticky_ttl_s": 1800}), 30)
        self.assertEqual(vision_route_ttl({"vision_sticky_ttl_s": 45}), 1)
        self.assertIsNone(vision_route_ttl({"vision_sticky_ttl_s": 0}))
        self.assertEqual(vision_route_ttl({}), 30)
        # normalize_mmproj_config clamps 0 up to the 60s floor, which is 1 minute.
        clamped, _ = normalize_mmproj_config({"vision_sticky_ttl_s": 0})
        self.assertEqual(clamped["vision_sticky_ttl_s"], 60)
        self.assertEqual(vision_route_ttl(clamped), 1)

    def test_vision_model_id_helpers_round_trip(self):
        self.assertEqual(vision_model_id(BASE_ID), VISION_ID)
        self.assertTrue(is_vision_model_id(VISION_ID))
        self.assertFalse(is_vision_model_id(BASE_ID))
        self.assertEqual(vision_base_model_id(VISION_ID), BASE_ID)

    def test_build_vision_model_forces_mmproj_and_drops_replicas(self):
        base = _make_model(
            aliases=["qwen"],
            server_overrides={
                "engine": "buun",
                "mmproj_mode": "lazy",
                "replicas": {"enabled": True},
            },
        )
        vision = build_vision_model(base)
        self.assertEqual(vision.model_id, VISION_ID)
        self.assertEqual(vision.aliases, [])
        self.assertEqual(vision.mmproj_path, base.mmproj_path)
        self.assertEqual(get_model_mmproj_mode(vision), MMPROJ_MODE_ALWAYS)
        self.assertTrue(resolve_render_include_mmproj(vision))
        self.assertNotIn("replicas", vision.server_overrides)
        # The base must not be mutated.
        self.assertEqual(base.aliases, ["qwen"])
        self.assertEqual(base.server_overrides["mmproj_mode"], "lazy")


class CommandEmissionTest(unittest.TestCase, _ConfigEnvMixin):
    def setUp(self):
        self.use_conf()

    def _cmd(self, model, **kwargs):
        return build_llama_server_command(
            model,
            Path("/usr/bin/llama-server"),
            port="11434",
            **kwargs,
        )

    def test_mmproj_mode_is_never_emitted_as_a_flag(self):
        model = _make_model(server_overrides={"engine": "buun", "mmproj_mode": "lazy"})
        cmd = self._cmd(model)
        self.assertNotIn("--mmproj-mode", cmd)
        self.assertNotIn("--mmproj_mode", cmd)
        self.assertFalse(any(part.startswith("--mmproj=") for part in cmd))

    def test_include_mmproj_is_the_only_render_time_switch(self):
        # The builder itself keeps emitting --mmproj for a lazy base; dropping it
        # is the render layer's job via include_mmproj=False.
        model = _make_model(server_overrides={"engine": "buun", "mmproj_mode": "lazy"})
        self.assertIn("--mmproj", self._cmd(model))
        self.assertNotIn("--mmproj", self._cmd(model, include_mmproj=False))
        vision = build_vision_model(model)
        self.assertIn("--mmproj", self._cmd(vision, include_mmproj=True))

    def test_mode_off_removes_mmproj_from_the_command(self):
        model = _make_model(server_overrides={"engine": "buun", "mmproj_mode": "off"})
        self.assertNotIn("--mmproj", self._cmd(model))

    def test_default_still_emits_mmproj(self):
        model = _make_model()
        cmd = self._cmd(model)
        self.assertEqual(cmd[cmd.index("--mmproj") + 1], "/models/proj")


class RenderTest(unittest.TestCase, _ConfigEnvMixin):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # A sparse file above the 4 GiB "small model" threshold so the matrix
        # treats the model as a large one that needs its own group.
        self.big = self.root / "big.safetensors"
        with open(self.big, "wb") as handle:
            handle.truncate(6 * 1024 * 1024 * 1024)
        self.config_path = self.root / "config.yaml"

    def _model(self, **kwargs):
        kwargs.setdefault("local_path", str(self.big))
        return _make_model(**kwargs)

    def _render(self, catalog, idle_ttl=300):
        render_llamaswap_config(
            catalog,
            self.config_path,
            Path("/usr/bin/llama-server"),
            11436,
            idle_ttl,
            {},
            {},
        )
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

    def test_lazy_render_splits_base_and_vision_sibling(self):
        self.use_conf()
        catalog = [self._model(server_overrides={"engine": "buun", "mmproj_mode": "lazy"})]
        data = self._render(catalog)
        models = data["models"]
        self.assertIn(BASE_ID, models)
        self.assertIn(VISION_ID, models)

        base_cmd = models[BASE_ID]["cmd"]
        self.assertNotIn("--mmproj", base_cmd)
        self.assertEqual(models[BASE_ID]["ttl"], 300)

        vision = models[VISION_ID]
        self.assertIn("--mmproj /models/proj", vision["cmd"])
        self.assertEqual(vision["ttl"], 30)  # 1800s sticky
        self.assertEqual(vision["metadata"]["internal_replica_of"], BASE_ID)
        self.assertTrue(vision["metadata"]["vision_variant"])
        self.assertEqual(vision["checkEndpoint"], "/health")
        # The projector flag must stay in front of the jinja flag.
        self.assertLess(vision["cmd"].index("--mmproj"), vision["cmd"].index("--jinja"))

    def test_lazy_render_honours_vision_sticky_ttl_override(self):
        self.use_conf({"vision_sticky_ttl_s": 600})
        catalog = [self._model(server_overrides={"engine": "buun", "mmproj_mode": "lazy"})]
        data = self._render(catalog)
        self.assertEqual(data["models"][VISION_ID]["ttl"], 10)

    def test_non_lazy_render_is_unchanged(self):
        self.use_conf()
        catalog = [self._model()]
        data = self._render(catalog)
        self.assertEqual(sorted(data["models"]), [BASE_ID])
        self.assertIn("--mmproj /models/proj", data["models"][BASE_ID]["cmd"])

    def test_mode_off_render_has_no_mmproj_and_no_sibling(self):
        self.use_conf()
        catalog = [self._model(server_overrides={"engine": "buun", "mmproj_mode": "off"})]
        data = self._render(catalog)
        self.assertEqual(sorted(data["models"]), [BASE_ID])
        self.assertNotIn("--mmproj", data["models"][BASE_ID]["cmd"])

    def test_lazy_and_non_lazy_models_coexist(self):
        self.use_conf()
        catalog = [
            self._model(server_overrides={"engine": "buun", "mmproj_mode": "lazy"}),
            self._model(
                model_id="plain-gguf",
                mmproj_path="/models/plain-mmproj.gguf",
                local_path=str(self.big),
            ),
        ]
        data = self._render(catalog)
        self.assertEqual(
            sorted(data["models"]),
            [BASE_ID, VISION_ID, "plain-gguf"],
        )
        self.assertNotIn("--mmproj", data["models"][BASE_ID]["cmd"])
        self.assertIn("--mmproj /models/proj", data["models"][VISION_ID]["cmd"])
        self.assertIn("--mmproj /models/plain-mmproj.gguf", data["models"]["plain-gguf"]["cmd"])

    def test_global_lazy_default_marks_every_projector_model(self):
        self.use_conf({"default_mode": "lazy"})
        catalog = [self._model()]
        data = self._render(catalog)
        self.assertNotIn("--mmproj", data["models"][BASE_ID]["cmd"])
        self.assertIn("--mmproj", data["models"][VISION_ID]["cmd"])

    def test_render_is_deterministic(self):
        self.use_conf()
        catalog = [self._model(server_overrides={"engine": "buun", "mmproj_mode": "lazy"})]
        self._render(catalog)
        first = self.config_path.read_bytes()
        self._render(catalog)
        self.assertEqual(first, self.config_path.read_bytes())


class MatrixTest(unittest.TestCase, _ConfigEnvMixin):
    def test_vision_sibling_is_mutually_exclusive_by_default(self):
        infos = [
            {"id": BASE_ID, "gpu_set": [0], "is_embedding": False, "is_small": False, "size_mib": 9216},
            {"id": VISION_ID, "gpu_set": [0], "is_embedding": False, "is_small": False, "size_mib": 9216},
        ]
        matrix = _calculate_llama_swap_matrix(infos)
        self.assertEqual(len(matrix["sets"]), 2)
        costs = {
            model_id: matrix["evict_costs"][var]
            for var, model_id in matrix["vars"].items()
        }
        self.assertEqual(costs[BASE_ID], 9216 * 1.0)
        self.assertEqual(costs[VISION_ID], 9216 * 0.5)
        # Mutually exclusive: no group may hold both variants at once.
        for group in matrix["sets"].values():
            self.assertNotEqual({VISION_ID, BASE_ID}, set(group.split(" & ")))

    def test_allow_co_resident_merges_the_pair(self):
        infos = [
            {"id": BASE_ID, "gpu_set": [0], "is_embedding": False, "is_small": False, "size_mib": 9216},
            {"id": VISION_ID, "gpu_set": [0], "is_embedding": False, "is_small": False, "size_mib": 9216},
        ]
        matrix = _calculate_llama_swap_matrix(infos, co_resident_variants=True)
        self.assertEqual(len(matrix["sets"]), 1)

    def test_disjoint_gpu_sets_merge_without_the_opt_in(self):
        infos = [
            {"id": BASE_ID, "gpu_set": [0], "is_embedding": False, "is_small": False, "size_mib": 9216},
            {"id": VISION_ID, "gpu_set": [1], "is_embedding": False, "is_small": False, "size_mib": 9216},
        ]
        self.assertEqual(len(_calculate_llama_swap_matrix(infos)["sets"]), 1)

    def test_replica_grouping_semantics_are_unchanged(self):
        infos = [
            {"id": BASE_ID, "gpu_set": [0], "is_embedding": False, "is_small": False, "size_mib": 9216},
            {"id": "exl3-qwen__replica_0", "gpu_set": [0], "is_embedding": False, "is_small": False, "size_mib": 9216},
        ]
        self.assertEqual(len(_calculate_llama_swap_matrix(infos)["sets"]), 2)


class EnsureVisionRouteTest(unittest.TestCase, _ConfigEnvMixin):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.big = self.root / "big.safetensors"
        with open(self.big, "wb") as handle:
            handle.truncate(6 * 1024 * 1024 * 1024)
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(
            yaml.safe_dump({"models": {}}, sort_keys=False), encoding="utf-8"
        )

    def _model(self, **kwargs):
        kwargs.setdefault("local_path", str(self.big))
        return _make_model(**kwargs)

    def _ensure(self, catalog):
        return ensure_vision_route_in_llamaswap_config(
            catalog[0],
            catalog,
            self.config_path,
            Path("/usr/bin/llama-server"),
            300,
            {},
        )

    def test_route_is_created_then_left_untouched(self):
        self.use_conf()
        catalog = [self._model(server_overrides={"engine": "buun", "mmproj_mode": "lazy"})]
        self.assertEqual(self._ensure(catalog), VISION_ID)
        after_first = self.config_path.read_bytes()
        data = yaml.safe_load(after_first.decode("utf-8"))
        self.assertIn(VISION_ID, data["models"])
        self.assertIn("--mmproj /models/proj", data["models"][VISION_ID]["cmd"])
        self.assertEqual(data["models"][VISION_ID]["ttl"], 30)
        self.assertTrue(data["models"][VISION_ID]["metadata"]["vision_variant"])
        self.assertIn("matrix", data)

        # Idempotent: a second call must not rewrite the watched config.
        self.assertEqual(self._ensure(catalog), VISION_ID)
        self.assertEqual(after_first, self.config_path.read_bytes())


class RouteRequestTest(unittest.TestCase, _ConfigEnvMixin):
    """``route_image_request_to_vision`` decides where an image request goes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(
            yaml.safe_dump({"models": {}}, sort_keys=False), encoding="utf-8"
        )
        self.model = _make_model(
            server_overrides={"engine": "buun", "mmproj_mode": "lazy"}
        )
        self.catalog = [self.model]
        self.args = argparse.Namespace(
            public_port=11436,
            config=str(self.config_path),
            llama_server="/usr/bin/llama-server",
            idle_ttl=300,
            llama_server_defaults={},
            mode="user",
            state_dir=str(self.root),
            public_host="127.0.0.1",
        )
        state = _cli_impl.REPLICA_ROUTER_STATE
        state.affinity.clear()
        state.response_to_replica.clear()
        state.loading_claims.clear()
        self.addCleanup(state.affinity.clear)
        self.addCleanup(state.response_to_replica.clear)

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
    TEXT_PAYLOAD = {
        "model": BASE_ID,
        "messages": [{"role": "user", "content": "hello"}],
    }
    HEADERS = {"thread-id": "conv-1"}

    def _route(self, payload, **overrides):
        kwargs = dict(
            published=set(),
            ensure_ok=True,
            wait_ok=True,
            loaded=False,
            reload_ok=True,
        )
        kwargs.update(overrides)
        ensure_calls: list[str] = []
        reload_calls: list[str] = []

        def _ensure(model, *args, **kwargs2):
            ensure_calls.append(model.model_id)
            if kwargs["ensure_ok"]:
                self.config_path.write_text(
                    yaml.safe_dump({"models": {}}, sort_keys=False), encoding="utf-8"
                )
                with open(self.config_path, "a", encoding="utf-8") as handle:
                    handle.write(
                        f"# {VISION_ID} {kwargs['published'] and 'present' or 'new'}\n"
                    )
            return VISION_ID

        def _reload(model, *args, **kwargs2):
            reload_calls.append(str(kwargs2.get("reason")))
            return kwargs["reload_ok"]

        with mock.patch.object(
            _cli_impl, "get_published_model_ids", return_value=set(kwargs["published"])
        ), mock.patch.object(
            _cli_impl, "ensure_vision_route_in_llamaswap_config", side_effect=_ensure
        ), mock.patch.object(
            _cli_impl, "wait_for_published_model_id", return_value=kwargs["wait_ok"]
        ), mock.patch.object(
            _cli_impl, "get_catalog_model_process",
            return_value={"pid": 4242, "cmdline": "", "port": 11436, "model_path": ""}
            if kwargs["loaded"]
            else None,
        ), mock.patch.object(
            _cli_impl, "reload_model_runtime_from_catalog_config", side_effect=_reload
        ), mock.patch.object(
            _cli_impl, "log_api_event", return_value=None
        ):
            target, error = _cli_impl.route_image_request_to_vision(
                self.model, payload, self.HEADERS, self.catalog, self.args, "127.0.0.1"
            )
        return target, error, ensure_calls, reload_calls

    def test_text_request_is_not_applicable(self):
        target, error, ensures, reloads = self._route(self.TEXT_PAYLOAD)
        self.assertIsNone(target)
        self.assertIsNone(error)
        self.assertEqual(ensures, [])
        self.assertEqual(reloads, [])

    def test_non_lazy_model_is_not_applicable(self):
        self.use_conf()
        model = _make_model(server_overrides={"engine": "buun"})
        with mock.patch.object(
            _cli_impl, "get_published_model_ids", return_value=set()
        ), mock.patch.object(
            _cli_impl, "ensure_vision_route_in_llamaswap_config"
        ) as ensure, mock.patch.object(_cli_impl, "log_api_event", return_value=None):
            result = _cli_impl.route_image_request_to_vision(
                model, self.IMAGE_PAYLOAD, self.HEADERS, [model], self.args, "127.0.0.1"
            )
        self.assertEqual(result, (None, None))
        ensure.assert_not_called()

    def test_ollama_top_level_images_count_as_images(self):
        self.use_conf()
        payload = {"model": BASE_ID, "messages": [{"role": "user", "content": "hi"}],
                   "images": ["AAA"]}
        target, error, ensures, _ = self._route(payload)
        self.assertEqual(target, VISION_ID)
        self.assertIsNone(error)
        self.assertEqual(ensures, [BASE_ID])

    def test_cold_request_publishes_the_sibling(self):
        self.use_conf()
        target, error, ensures, reloads = self._route(self.IMAGE_PAYLOAD)
        self.assertEqual(target, VISION_ID)
        self.assertIsNone(error)
        self.assertEqual(ensures, [BASE_ID])
        self.assertEqual(reloads, [])

    def test_already_published_sibling_avoids_a_config_rewrite(self):
        self.use_conf()
        target, error, ensures, reloads = self._route(
            self.IMAGE_PAYLOAD, published={VISION_ID}
        )
        self.assertEqual(target, VISION_ID)
        self.assertIsNone(error)
        self.assertEqual(ensures, [])
        self.assertEqual(reloads, [])

    def test_second_request_reuses_the_affinity_pin(self):
        self.use_conf()
        first_target, _error, _ensures, _reloads = self._route(self.IMAGE_PAYLOAD)
        self.assertEqual(first_target, VISION_ID)
        pinned = dict(_cli_impl.REPLICA_ROUTER_STATE.affinity)
        self.assertTrue(pinned)

        target, error, ensures, reloads = self._route(
            self.IMAGE_PAYLOAD, published={VISION_ID}
        )
        self.assertEqual(target, VISION_ID)
        self.assertIsNone(error)
        self.assertEqual(ensures, [])
        self.assertEqual(reloads, [])

    def test_falls_back_to_reloading_a_loaded_base(self):
        self.use_conf()
        target, error, ensures, reloads = self._route(
            self.IMAGE_PAYLOAD, wait_ok=False, loaded=True
        )
        self.assertEqual(target, BASE_ID)
        self.assertIsNone(error)
        self.assertEqual(ensures, [BASE_ID])
        self.assertEqual(reloads, ["lazy_mmproj_vision_upgrade"])

    def test_prefer_vision_route_disabled_uses_the_reload_fallback(self):
        self.use_conf({"prefer_vision_route": False})
        target, error, ensures, reloads = self._route(self.IMAGE_PAYLOAD, loaded=True)
        self.assertEqual(target, BASE_ID)
        self.assertIsNone(error)
        self.assertEqual(ensures, [])
        self.assertEqual(reloads, ["lazy_mmproj_vision_upgrade"])

    def test_unavailable_route_without_a_loaded_base_is_an_error(self):
        self.use_conf()
        target, error, ensures, reloads = self._route(
            self.IMAGE_PAYLOAD, wait_ok=False, loaded=False
        )
        self.assertIsNone(target)
        self.assertIsNotNone(error)
        self.assertIn("vision route could not be published", error)
        self.assertEqual(ensures, [BASE_ID])
        self.assertEqual(reloads, [])

    def test_failed_reload_is_an_error(self):
        self.use_conf()
        target, error, _ensures, reloads = self._route(
            self.IMAGE_PAYLOAD, wait_ok=False, loaded=True, reload_ok=False
        )
        self.assertIsNone(target)
        self.assertIn("failed", error)
        self.assertEqual(reloads, ["lazy_mmproj_vision_upgrade"])

    def test_invalid_public_port_is_an_error(self):
        self.use_conf()
        self.args.public_port = "not-a-port"
        target, error = _cli_impl.route_image_request_to_vision(
            self.model, self.IMAGE_PAYLOAD, self.HEADERS, self.catalog, self.args, "127.0.0.1"
        )
        self.assertIsNone(target)
        self.assertIn("invalid public port", error)


class ForcedMmprojTest(unittest.TestCase, _ConfigEnvMixin):
    def setUp(self):
        self.use_conf()

    def test_with_forced_mmproj_pins_the_mode_on(self):
        base = _make_model(server_overrides={"engine": "buun", "mmproj_mode": "lazy"})
        forced = _cli_impl._with_forced_mmproj(base)
        self.assertEqual(get_model_mmproj_mode(forced), MMPROJ_MODE_ALWAYS)
        self.assertTrue(resolve_render_include_mmproj(forced))
        # The original catalog entry is untouched so the on-disk file keeps lazy.
        self.assertEqual(get_model_mmproj_mode(base), MMPROJ_MODE_LAZY)

    def test_reload_helper_accepts_force_mmproj_kwarg(self):
        import inspect

        params = inspect.signature(
            _cli_impl.reload_model_runtime_from_catalog_config
        ).parameters
        self.assertIn("force_mmproj", params)
        self.assertIn("reason", params)
        self.assertFalse(params["force_mmproj"].default)


class HandlerWiringTest(unittest.TestCase):
    """The four image endpoints must consult the lazy route before replicas.

    The gateway handlers are methods defined inside a factory closure, so they
    are not reachable through ``dir(_cli_impl)``. Assert on the module source
    instead, which also pins the ordering constraint (vision resolution must run
    before the replica router, and the legacy stale-reload must skip lazy
    models whose base process never carries a projector by design).
    """

    ENDPOINTS = (
        "_handle_ollama_chat",
        "_handle_openai_chat_completions",
        "_handle_openai_responses",
        "_handle_ollama_generate",
    )

    @classmethod
    def setUpClass(cls):
        cls.source = (
            Path(_cli_impl.__file__).read_text(encoding="utf-8")
        )

    def _endpoint_body(self, name: str) -> str:
        marker = f"def {name}(self):"
        start = self.source.index(marker)
        return self.source[start:]

    def test_all_four_image_endpoints_call_the_lazy_route(self):
        self.assertEqual(
            self.source.count("route_image_request_to_vision("),
            1 + len(self.ENDPOINTS),
            "expected one definition plus one call per image endpoint",
        )
        for name in self.ENDPOINTS:
            with self.subTest(endpoint=name):
                body = self._endpoint_body(name)
                self.assertIn("route_image_request_to_vision(", body)

    def test_vision_target_bypasses_replica_selection(self):
        for name in self.ENDPOINTS:
            with self.subTest(endpoint=name):
                body = self._endpoint_body(name)
                resolve = body.index("route_image_request_to_vision(")
                replica = body.index("select_replica_for_request(")
                self.assertLess(
                    resolve,
                    replica,
                    "the lazy route must be resolved before replica selection",
                )
                self.assertIn("elif model_entry is not None:", body)


class SurfaceTest(unittest.TestCase):
    def test_vision_helpers_are_exported_from_the_public_package(self):
        import llamacpp_stack.cli as cli_pkg

        for name in (
            "vision_model_id",
            "vision_base_model_id",
            "is_vision_model_id",
            "build_vision_model",
            "get_model_mmproj_mode",
            "model_lazily_loads_mmproj",
            "resolve_render_include_mmproj",
            "normalize_mmproj_config",
            "default_mmproj_config",
            "MMPROJ_MODE_LAZY",
            "MMPROJ_MODE_ALWAYS",
            "MMPROJ_MODE_OFF",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(cli_pkg, name), f"{name} missing from cli package")
                self.assertTrue(hasattr(vision_mod, name), f"{name} missing from vision module")

        # Names the gateway resolves at runtime must be bound in _cli_impl.
        for name in (
            "get_model_mmproj_mode",
            "model_lazily_loads_mmproj",
            "resolve_render_include_mmproj",
            "MMPROJ_MODE_LAZY",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(_cli_impl, name), f"{name} missing from _cli_impl")

    def test_ensure_vision_route_is_rebound_in_cli_impl(self):
        for name in (
            "ensure_vision_route_in_llamaswap_config",
            "is_vision_model_id",
            "vision_base_model_id",
            "vision_model_id",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(_cli_impl, name))

    def test_config_migrate_adds_the_mmproj_block_idempotently(self):
        first, changed_first = _cli_impl.normalize_server_config_payload({})
        self.assertIn("mmproj", first)
        self.assertEqual(first["mmproj"], default_mmproj_config())
        self.assertTrue(changed_first)
        second, changed_second = _cli_impl.normalize_server_config_payload(first)
        self.assertEqual(second["mmproj"], first["mmproj"])
        self.assertFalse(changed_second)


if __name__ == "__main__":
    unittest.main()