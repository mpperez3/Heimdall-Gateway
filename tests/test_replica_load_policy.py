"""Replica-vs-new-model load policy: a load never competes with a load in flight.

The gateway's only admission question used to be "does this fit if it is alone on
the GPU". Two ~20GB models on one 24GB card both pass that, so llama-swap's solver
evicts whichever finished first, and every alternation costs a cold load plus a
full KV re-prefill. These tests pin the four halves of the fix:

* ``blocking_conflicting_load`` -- the pure first-come-first-served decision,
* ``_reject_if_gpu_busy`` -- the 503 that enforces it,
* ``select_replica_for_request`` -- new requests ride the loaded instance instead
  of spreading or cold-starting a second one,
* the reaper -- VRAM comes back through llama-swap's unload API, and *never* by
  rewriting ``config.yaml`` (llama-swap watches it, so a write evicts everything).
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path
from unittest import mock

import yaml
from dataclasses import replace

from llamacpp_stack import _cli_impl
from llamacpp_stack.cli import daemon as daemon_mod
from llamacpp_stack.cli import replica as R
from llamacpp_stack.cli import ManagedModel, normalize_server_config_payload
from llamacpp_stack.cli.models import ReplicaConfig, ReplicaRecord
from llamacpp_stack.cli.replica import get_model_replica_config

BASE_ID = "base-model"
OTHER_ID = "other-model"
REPLICA_ID = f"{BASE_ID}__replica_0"
SERVER = Path("/usr/bin/llama-server")


def _make_model(
    model_id: str = BASE_ID,
    *,
    server_overrides: dict | None = None,
    tensor_split: str | None = "1",
) -> ManagedModel:
    return ManagedModel(
        model_id=model_id,
        repo_id="org/model",
        quant="Q4",
        filename="model.gguf",
        local_path="/models/model.gguf",
        backend="llamacpp",
        mmproj_filename=None,
        mmproj_path=None,
        load_capabilities=[],
        aliases=[],
        ctx_size=32768,
        # build_llama_server_command int()s this, so None would raise.
        n_gpu_layers=-1,
        tensor_split=tensor_split,
        host=None,
        jinja=True,
        ttl=None,
        description=None,
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
        server_overrides=dict(server_overrides or {}),
    )


def _write_matrix(path: Path, sets: dict[str, list[str]]) -> Path:
    """Render a llama-swap matrix where ``sets`` maps a set name to its model ids.

    Matrix set names are llama-swap boolean expressions over vars, not lists, so
    this builds ``{"vars": {...}, "sets": {name: "m0 & m1"}}`` the way the real
    renderer does.
    """
    var_to_model: dict[str, str] = {}
    rendered: dict[str, str] = {}
    for index, (set_name, model_ids) in enumerate(sets.items()):
        tokens = []
        for offset, model_id in enumerate(model_ids):
            token = f"m{index * 100 + offset}"
            var_to_model[token] = model_id
            tokens.append(token)
        rendered[set_name] = " & ".join(tokens)
    path.write_text(
        yaml.safe_dump({"matrix": {"vars": var_to_model, "sets": rendered}}),
        encoding="utf-8",
    )
    return path


def _claim(state, key: str, age_s: float, *, now: float) -> None:
    state.loading_claims[key] = now - age_s + _cli_impl.LOADING_CLAIM_TTL_S


def _ready_replica(state, *, replica_id: str = REPLICA_ID, base_id: str = BASE_ID,
                   gpu_set=(1,), in_flight: int = 0, last_used: float = 0.0,
                   status: str = "ready") -> None:
    state.records[replica_id] = ReplicaRecord(
        base_model_id=base_id,
        replica_model_id=replica_id,
        gpu_set=list(gpu_set),
        status=status,
        in_flight=in_flight,
        last_used=last_used,
    )


class _RouterStateMixin:
    """Isolate the router singleton.

    ``reset_router_state`` is a plain method, not ``setUp``: ``unittest.TestCase``
    also defines ``setUp``, so a mixin ``setUp`` is shadowed by the MRO and would
    silently never run, leaking singleton state between tests.
    """

    def reset_router_state(self):
        state = _cli_impl.REPLICA_ROUTER_STATE
        for attr in ("records", "affinity", "response_to_replica", "loading_claims",
                     "loading_claim_aliases", "gpu_demand"):
            getattr(state, attr).clear()
        state.base_in_flight.clear()
        state.base_last_used.clear()
        state._health_cache.clear()
        for clear in (state.records.clear, state.affinity.clear, state.response_to_replica.clear,
                      state.loading_claims.clear, state.loading_claim_aliases.clear,
                      state.gpu_demand.clear,
                      state._health_cache.clear, state.base_in_flight.clear, state.base_last_used.clear):
            self.addCleanup(clear)


# --------------------------------------------------------------------------- #
# blocking_conflicting_load
# --------------------------------------------------------------------------- #
class BlockingConflictingLoadTest(unittest.TestCase, _RouterStateMixin):
    NOW = 10_000.0

    def setUp(self):
        self.reset_router_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / "config.yaml"
        self.state = _cli_impl.REPLICA_ROUTER_STATE

    def _block(self, model_id: str = BASE_ID, *, config: Path | None = None):
        return _cli_impl.blocking_conflicting_load(
            model_id, [], self.config if config is None else config, now=self.NOW
        )

    def test_no_claims_means_no_blocker(self):
        self.assertIsNone(self._block())

    def test_a_claim_on_the_models_own_base_is_not_a_blocker(self):
        _claim(self.state, BASE_ID, 1.0, now=self.NOW)
        self.assertIsNone(
            self._block(),
            "a model must never block its own reload -- it already owns the claim",
        )

    def test_a_base_and_its_replica_claims_collapse_to_one_unit(self):
        _claim(self.state, f"{BASE_ID}__replica_0", 1.0, now=self.NOW)
        self.assertIsNone(
            self._block(),
            "A__replica_0 is not a conflicting model, it is A's own second instance",
        )

    def test_an_overlapping_foreign_claim_blocks(self):
        _write_matrix(self.config, {"pair_0": [OTHER_ID, BASE_ID]})
        _claim(self.state, OTHER_ID, 5.0, now=self.NOW)
        blocker = self._block()
        self.assertIsNotNone(blocker)
        self.assertEqual(blocker[0], OTHER_ID)
        self.assertAlmostEqual(blocker[1], 5.0, places=3)

    def test_a_disjoint_foreign_claim_does_not_block(self):
        _write_matrix(self.config, {"pair_0": [OTHER_ID], "pair_1": [BASE_ID]})
        _claim(self.state, OTHER_ID, 5.0, now=self.NOW)
        self.assertIsNone(
            self._block(),
            "two models llama-swap can hold together must be allowed to load together",
        )

    def test_an_expired_claim_is_ignored(self):
        _write_matrix(self.config, {"pair_0": [OTHER_ID, BASE_ID]})
        expired = self.NOW - _cli_impl.LOADING_CLAIM_TTL_S - 1.0
        self.state.loading_claims[OTHER_ID] = expired + _cli_impl.LOADING_CLAIM_TTL_S
        self.assertIsNone(self._block())

    def test_the_oldest_conflicting_claim_is_reported(self):
        _write_matrix(self.config, {"pair_0": [OTHER_ID, "third", BASE_ID]})
        _claim(self.state, OTHER_ID, 1.0, now=self.NOW)
        _claim(self.state, "third", 30.0, now=self.NOW)
        blocker = self._block()
        self.assertEqual(blocker[0], "third", "the oldest claim is the one still holding the GPU")
        self.assertAlmostEqual(blocker[1], 30.0, places=3)

    def test_a_disjoint_newer_claim_never_outranks_an_overlapping_older_one(self):
        _write_matrix(self.config, {"pair_0": ["far", BASE_ID], "pair_1": ["near"]})
        _claim(self.state, "far", 300.0, now=self.NOW)
        _claim(self.state, "near", 1.0, now=self.NOW)
        blocker = self._block()
        self.assertEqual(blocker[0], "far")
        self.assertAlmostEqual(blocker[1], 300.0, places=3)

    def test_a_missing_config_is_a_conflict(self):
        _claim(self.state, OTHER_ID, 1.0, now=self.NOW)
        self.config.unlink(missing_ok=True)
        blocker = self._block()
        self.assertIsNotNone(blocker, "unprovable concurrency must be treated as conflict")
        self.assertEqual(blocker[0], OTHER_ID)

    def test_a_model_absent_from_the_matrix_is_a_conflict(self):
        _write_matrix(self.config, {"pair_0": [BASE_ID]})
        _claim(self.state, OTHER_ID, 1.0, now=self.NOW)
        self.assertIsNotNone(self._block())

    def test_an_unreadable_matrix_is_a_conflict(self):
        _write_matrix(self.config, {"pair_0": [OTHER_ID, BASE_ID]})
        _claim(self.state, OTHER_ID, 1.0, now=self.NOW)
        self.config.write_text(": : not yaml [", encoding="utf-8")
        self.assertIsNotNone(self._block())

    def test_a_broken_matrix_reader_degrades_to_conflict(self):
        _write_matrix(self.config, {"pair_0": [OTHER_ID, BASE_ID]})
        _claim(self.state, OTHER_ID, 1.0, now=self.NOW)
        with mock.patch.object(
            _cli_impl, "_matrix_group_membership", side_effect=RuntimeError("boom")
        ):
            self.assertIsNotNone(self._block())

    def test_the_helper_logs_nothing(self):
        _write_matrix(self.config, {"pair_0": [OTHER_ID, BASE_ID]})
        _claim(self.state, OTHER_ID, 1.0, now=self.NOW)
        with mock.patch.object(_cli_impl, "log_api_event") as log_mock:
            self._block()
        log_mock.assert_not_called()


class MatrixConflictTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = Path(self.tmp.name) / "config.yaml"

    def test_an_absent_matrix_conflicts(self):
        self.assertTrue(_cli_impl._matrix_models_conflict("a", "b", self.config))

    def test_one_shared_set_name_conflicts(self):
        _write_matrix(self.config, {"s0": ["a", "b"]})
        self.assertTrue(_cli_impl._matrix_models_conflict("a", "b", self.config))

    def test_disjoint_sets_do_not_conflict(self):
        _write_matrix(self.config, {"s0": ["a"], "s1": ["b"]})
        self.assertFalse(_cli_impl._matrix_models_conflict("a", "b", self.config))


# --------------------------------------------------------------------------- #
# _reject_if_gpu_busy
# --------------------------------------------------------------------------- #
class RejectIfGpuBusyTest(unittest.TestCase, _RouterStateMixin):
    """The gate as the request path sees it.

    ``_reject_if_gpu_busy`` is a method nested inside ``start_ctx_metadata_server``'s
    closure, so it is not reachable through ``dir(_cli_impl)``. Its source is lifted
    out of the real module and exec'd against a stub ``self``, which keeps the test
    pinned to the shipped body instead of a copy of it.
    """

    NOW = 10_000.0

    @classmethod
    def setUpClass(cls):
        cls.source = Path(_cli_impl.__file__).read_text(encoding="utf-8")
        cls.handler_source = inspect.getsource(_cli_impl.start_ctx_metadata_server)

    def _bind(self, namespace: dict):
        """Return ``_reject_if_gpu_busy`` bound to a stub ``self`` from *namespace*."""
        start = self.handler_source.index("def _reject_if_gpu_busy(self,")
        end = self.handler_source.index("def _reject_if_model_loading(self,", start)
        body = textwrap.dedent(self.handler_source[start:end])
        scope = dict(namespace)
        exec(compile(body, "<_reject_if_gpu_busy>", "exec"), scope)
        return scope["_reject_if_gpu_busy"]

    def setUp(self):
        self.reset_router_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / "config.yaml"
        _write_matrix(self.config, {"pair_0": [OTHER_ID, BASE_ID]})
        self.state = _cli_impl.REPLICA_ROUTER_STATE
        self.args = argparse.Namespace(config=self.config, public_port=11436)
        self.sent: list[tuple[dict, int]] = []
        self.events: list[tuple[str, dict]] = []
        stub = mock.Mock()
        stub._send_json = lambda payload, status=200: self.sent.append((payload, status))
        self.stub = stub
        self.log_mock = mock.Mock(
            side_effect=lambda event, data=None: self.events.append((event, data or {}))
        )

    def setUp(self):
        self.reset_router_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / "config.yaml"
        _write_matrix(self.config, {"pair_0": [OTHER_ID, BASE_ID]})
        self.state = _cli_impl.REPLICA_ROUTER_STATE
        self.args = argparse.Namespace(config=self.config, public_port=11436)
        self.sent: list[tuple[dict, int]] = []
        self.events: list[tuple[str, dict]] = []
        stub = mock.Mock()
        stub._send_json = lambda payload, status=200: self.sent.append((payload, status))
        self.stub = stub
        self.log_mock = mock.Mock(
            side_effect=lambda event, data=None: self.events.append((event, data or {}))
        )
        self.handler_source = inspect.getsource(_cli_impl.start_ctx_metadata_server)

    def _call(self, model_name: str = "wants-gpu-1", *, api_style: str = "openai"):
        with mock.patch.object(_cli_impl.time, "monotonic", return_value=self.NOW):
            return self._method()(self.stub, model_name, [], api_style=api_style)

    def _method(self):
        namespace = {
            "args": self.args,
            "client_host": "127.0.0.1",
            "ManagedModel": ManagedModel,
            "log_api_event": self.log_mock,
            "REPLICA_ROUTER_STATE": self.state,
            "blocking_conflicting_load": _cli_impl.blocking_conflicting_load,
            "plan_runtime_placement": _cli_impl.plan_runtime_placement,
            "retire_idle_occupants": _cli_impl.retire_idle_occupants,
            "apply_repin": _cli_impl.apply_repin,
            "get_catalog_model_process": mock.Mock(return_value=None),
            "get_gpu_conflict_message": mock.Mock(return_value=None),
            "get_model_activity_snapshot": mock.Mock(return_value=({}, None)),
            "recent_activity_blocking_model_switch": mock.Mock(return_value=None),
            "request_looks_like_model_probe": mock.Mock(return_value=False),
            "time": _cli_impl.time,
            "_resolve_model_probe_autoload_config": mock.Mock(return_value={"enabled": False}),
        }
        return self._bind(namespace)

    def _call(self, model_name: str = BASE_ID, *, api_style: str = "openai"):
        with mock.patch.object(_cli_impl.time, "monotonic", return_value=self.NOW):
            return self._method()(self.stub, model_name, [], api_style=api_style)

    def test_a_conflicting_load_is_refused_with_503_model_loading(self):
        _claim(self.state, OTHER_ID, 4.0, now=self.NOW)
        # The request itself owns a claim by the time this runs.
        _claim(self.state, BASE_ID, 0.0, now=self.NOW)
        self.assertTrue(self._call())
        self.assertEqual(len(self.sent), 1)
        payload, status = self.sent[0]
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["code"], "model_loading")
        self.assertEqual(payload["error"]["type"], "model_loading")
        self.assertIn(OTHER_ID, payload["error"]["message"])

    def test_the_ollama_shape_is_used_when_asked(self):
        _claim(self.state, OTHER_ID, 4.0, now=self.NOW)
        self.assertTrue(self._call(api_style="ollama"))
        payload, status = self.sent[0]
        self.assertEqual(status, 503)
        self.assertEqual(payload["code"], "model_loading")
        self.assertNotIn("type", payload["error"])

    def test_a_refused_request_releases_its_own_claim(self):
        _claim(self.state, OTHER_ID, 4.0, now=self.NOW)
        _claim(self.state, BASE_ID, 0.0, now=self.NOW)
        self.assertTrue(self._call())
        self.assertNotIn(
            BASE_ID,
            self.state.loading_claims,
            "a refused request must not leave a claim that blocks the next one",
        )
        self.assertIn(OTHER_ID, self.state.loading_claims)

    def test_the_refusal_is_logged_with_the_blocker(self):
        _claim(self.state, OTHER_ID, 4.0, now=self.NOW)
        self._call()
        events = dict(self.events)
        self.assertIn("model_load_blocked_by_concurrent_load", events)
        self.assertEqual(events["model_load_blocked_by_concurrent_load"]["blocker"], OTHER_ID)
        self.assertIn("model", events["model_load_blocked_by_concurrent_load"])
        self.assertIn("waited_s", events["model_load_blocked_by_concurrent_load"])

    def test_without_a_blocker_the_request_is_allowed(self):
        _write_matrix(self.config, {"pair_0": [OTHER_ID], "pair_1": [BASE_ID]})
        _claim(self.state, OTHER_ID, 4.0, now=self.NOW)
        _claim(self.state, BASE_ID, 0.0, now=self.NOW)
        self.assertFalse(self._call())
        self.assertEqual(self.sent, [], "the 503 path must not run")
        self.assertNotIn("model_load_blocked_by_concurrent_load", dict(self.events))

    def test_the_gate_runs_before_the_gpu_capacity_check(self):
        # get_gpu_conflict_message answers "does it fit alone"; the gate answers
        # "does it fit without evicting a load in flight". Ordering is load-bearing.
        body = self.handler_source[
            self.handler_source.index("def _reject_if_gpu_busy(self,"):
            self.handler_source.index("def _reject_if_model_loading(self,")
        ]
        self.assertLess(
            body.index("blocking_conflicting_load("),
            body.index("get_gpu_conflict_message("),
        )


# --------------------------------------------------------------------------- #
# select_replica_for_request
# --------------------------------------------------------------------------- #
class SelectReplicaGuardsTest(unittest.TestCase, _RouterStateMixin):
    NOW = 10_000.0

    def setUp(self):
        self.reset_router_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / "config.yaml"
        self.state = _cli_impl.REPLICA_ROUTER_STATE
        self.model = _make_model(
            server_overrides={
                "replicas": {
                    "enabled": True,
                    "max": 2,
                    "gpus_per_replica": 1,
                    "prefer_base_over_replica": False,
                    "idle_grace_s": 0,
                }
            }
        )
        self.catalog = [self.model]
        self.log_events: list[tuple[str, dict]] = []
        self.now = self.NOW
        # _placement_fits returns False for a model whose file cannot be stat'ed,
        # so cold scale-out would be untestable with a fictional path.
        self.model_path = self.root / "model.gguf"
        self.model_path.write_bytes(b"x" * (2 * 1024 * 1024))
        self.model.local_path = str(self.model_path)

        self.cuda = mock.patch.object(_cli_impl, "detect_cuda_device_count", return_value=3)
        self.cuda.start()
        self.addCleanup(self.cuda.stop)
        self.published = mock.patch.object(
            _cli_impl, "get_published_model_ids", return_value={BASE_ID, REPLICA_ID}
        )
        self.published.start()
        self.addCleanup(self.published.stop)
        self.ensure = mock.patch.object(
            _cli_impl, "ensure_replica_route_in_llamaswap_config", return_value=self.config
        )
        self.ensure_mock = self.ensure.start()
        self.addCleanup(self.ensure.stop)
        self.await_pub = mock.patch.object(
            _cli_impl, "wait_for_published_model_id", return_value=True
        )
        self.await_pub.start()
        self.addCleanup(self.await_pub.stop)
        self.gpu_mem = mock.patch.object(
            _cli_impl,
            "_query_gpu_memory_snapshot_cached",
            return_value={0: {"free_mib": 40000, "total_mib": 40000},
                          1: {"free_mib": 40000, "total_mib": 40000},
                          2: {"free_mib": 40000, "total_mib": 40000}},
        )
        self.gpu_mem.start()
        self.addCleanup(self.gpu_mem.stop)
        self.log = mock.patch.object(
            _cli_impl, "log_api_event",
            side_effect=lambda event, data=None: self.log_events.append((event, data or {})),
        )
        self.log.start()
        self.addCleanup(self.log.stop)
        self.clock = mock.patch.object(_cli_impl.time, "monotonic", side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def _select(self, thread_id: str, payload: dict | None = None):
        return _cli_impl.select_replica_for_request(
            self.model,
            payload or {"messages": [{"role": "user", "content": "hi"}]},
            {"thread-id": thread_id},
            catalog=self.catalog,
            config_path=self.config,
            server_path=SERVER,
        )

    def _event_names(self) -> list[str]:
        return [name for name, _ in self.log_events]

    def _conflicting_claim(self, *, age_s: float = 2.0):
        _write_matrix(self.config, {"pair_0": [OTHER_ID, BASE_ID]})
        _claim(self.state, OTHER_ID, age_s, now=self.now)

    # -- the spread branch -------------------------------------------------- #
    def test_with_prefer_base_a_new_request_stays_on_the_base(self):
        _ready_replica(self.state, status="ready", last_used=0.0)
        self.state.base_last_used[BASE_ID] = self.now - 5.0
        self.model.server_overrides["replicas"]["prefer_base_over_replica"] = True
        selected, _key, is_replica = self._select("c1")
        self.assertEqual((selected, is_replica), (BASE_ID, False))
        self.assertNotIn("replica_spread_new_conversation", self._event_names())

    def test_with_a_blocker_a_new_request_stays_on_the_base(self):
        _ready_replica(self.state, status="ready", last_used=0.0)
        self.state.base_last_used[BASE_ID] = self.now - 5.0
        self._conflicting_claim()
        selected, _key, is_replica = self._select("c2")
        self.assertEqual((selected, is_replica), (BASE_ID, False))
        self.ensure_mock.assert_not_called()

    def test_without_either_gate_the_spread_still_happens(self):
        # Guards the guards: if the gate were unconditional, replicas would never
        # be used at all and the affinity machinery would be dead code.
        _ready_replica(self.state, status="ready", last_used=0.0)
        self.state.base_last_used[BASE_ID] = self.now - 5.0
        selected, _key, is_replica = self._select("c3")
        self.assertEqual((selected, is_replica), (REPLICA_ID, True))
        self.assertIn("replica_spread_new_conversation", self._event_names())

    # -- affinity is untouched ---------------------------------------------- #
    def test_a_live_affinity_survives_a_blocker(self):
        _ready_replica(self.state, status="ready", last_used=self.now - 50.0)
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        headers = {"thread-id": "sticky-conv"}
        affinity_key = _cli_impl.resolve_request_affinity_key(BASE_ID, payload, headers)
        pinned = (REPLICA_ID, self.now + 900.0)
        self.state.affinity[affinity_key] = pinned
        self._conflicting_claim()
        selected, key, is_replica = _cli_impl.select_replica_for_request(
            self.model,
            payload,
            headers,
            catalog=self.catalog,
            config_path=self.config,
            server_path=SERVER,
        )
        self.assertEqual((selected, is_replica), (REPLICA_ID, True))
        self.assertEqual(key, affinity_key)
        self.assertEqual(
            self.state.affinity[affinity_key][0], pinned[0],
            "a load window must never move a conversation off its live owner",
        )

    # -- the cold scale-out loop -------------------------------------------- #
    def test_the_grace_window_blocks_cold_scale_out_after_a_recent_base_use(self):
        _ready_replica(self.state, status="cold")
        self.state.base_last_used[BASE_ID] = self.now - 30.0
        self.model.server_overrides["replicas"]["idle_grace_s"] = 600
        self._select("c4")
        self.ensure_mock.assert_not_called()
        self.assertIn("replica_scale_out_deferred", self._event_names())

    def test_the_grace_window_does_not_apply_to_a_never_used_base(self):
        _ready_replica(self.state, status="cold")
        self.state.base_last_used.pop(BASE_ID, None)
        self.model.server_overrides["replicas"]["idle_grace_s"] = 600
        selected, _key, is_replica = self._select("c5")
        self.assertEqual(
            (selected, is_replica), (REPLICA_ID, True),
            "a brand-new model must always be allowed to start",
        )

    def test_the_grace_window_expires(self):
        _ready_replica(self.state, status="cold")
        self.state.base_last_used[BASE_ID] = self.now - 601.0
        self.model.server_overrides["replicas"]["idle_grace_s"] = 600
        selected, _key, is_replica = self._select("c6")
        self.assertEqual((selected, is_replica), (REPLICA_ID, True))

    def test_a_zero_grace_window_disables_the_gate(self):
        _ready_replica(self.state, status="cold")
        self.state.base_last_used[BASE_ID] = self.now - 1.0
        self.model.server_overrides["replicas"]["idle_grace_s"] = 0
        selected, _key, is_replica = self._select("c7")
        self.assertEqual((selected, is_replica), (REPLICA_ID, True))
        self.assertNotIn("replica_scale_out_deferred", self._event_names())

    def test_a_blocker_blocks_cold_scale_out_and_names_the_blocker(self):
        _ready_replica(self.state, status="cold")
        self.state.base_last_used.pop(BASE_ID, None)
        self._conflicting_claim()
        selected, _key, is_replica = self._select("c8")
        self.assertEqual((selected, is_replica), (BASE_ID, False))
        self.ensure_mock.assert_not_called()
        deferred = [d for name, d in self.log_events if name == "replica_scale_out_deferred"]
        self.assertEqual(len(deferred), 1)
        self.assertEqual(deferred[0]["blocker"], OTHER_ID)
        self.assertIn("grace_s", deferred[0])
        self.assertIn("base_last_used", deferred[0])

    # -- config plumbing ---------------------------------------------------- #
    def test_the_three_keys_are_read_and_clamped(self):
        cfg = get_model_replica_config(
            self.model,
            {"enabled": True, "prefer_base_over_replica": False, "idle_grace_s": -5, "max_idle_s": "90"},
        )
        self.assertFalse(cfg.prefer_base_over_replica)
        self.assertEqual(cfg.idle_grace_s, 0)
        self.assertEqual(cfg.max_idle_s, 90)

    def test_the_three_keys_default_from_the_global_config(self):
        # A model with no per-model replicas block, so only the global config speaks.
        bare = _make_model()
        defaults = _cli_impl._default_global_replicas_config()
        cfg = get_model_replica_config(bare, dict(defaults))
        self.assertFalse(cfg.prefer_base_over_replica)
        self.assertEqual((cfg.idle_grace_s, cfg.max_idle_s), (600, 1800))


# --------------------------------------------------------------------------- #
# the reaper
# --------------------------------------------------------------------------- #
class ReplicaReaperTest(unittest.TestCase, _RouterStateMixin):
    NOW = 10_000.0

    def setUp(self):
        self.reset_router_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / "config.yaml"
        _write_matrix(self.config, {"pair_0": [OTHER_ID]})
        self.catalog = self.root / "catalog.json"
        self.catalog.write_text(json.dumps([]), encoding="utf-8")
        self.state = _cli_impl.REPLICA_ROUTER_STATE
        self.model = _make_model()
        self.replica_defaults = {"enabled": True, "max": 2, "gpus_per_replica": 1}
        self.args = argparse.Namespace(
            catalog=self.catalog,
            config=self.config,
            public_host="127.0.0.1",
            public_port=11436,
        )
        self.unloads: list[tuple[str, str, int]] = []
        self.events: list[tuple[str, dict]] = []
        self.unload = mock.patch.object(
            daemon_mod, "unload_replica_via_llamaswap", side_effect=self._unload
        )
        self.unload_mock = self.unload.start()
        self.addCleanup(self.unload.stop)
        self.log = mock.patch.object(
            daemon_mod, "log_api_event",
            side_effect=lambda event, data=None: self.events.append((event, data or {})),
        )
        self.log.start()
        self.addCleanup(self.log.stop)
        self.catalog_loader = mock.patch.object(
            daemon_mod, "load_catalog_with_diagnostics", return_value=([self.model], None)
        )
        self.catalog_loader.start()
        self.addCleanup(self.catalog_loader.stop)
        self.replica_defaults_patch = mock.patch.object(
            daemon_mod, "resolve_global_replica_config", return_value=self.replica_defaults
        )
        self.replica_defaults_patch.start()
        self.addCleanup(self.replica_defaults_patch.stop)
        self.clock = mock.patch.object(_cli_impl.time, "monotonic", return_value=self.NOW)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def _unload(self, replica_id: str, host: str, port: int, **kwargs) -> bool:
        self.unloads.append((replica_id, host, port))
        return True

    def _ready(self, health_state: str = "ready", **kwargs):
        _ready_replica(self.state, **kwargs)

        class _Health:
            state = health_state

        self.health = mock.patch.object(
            type(self.state), "health_of", lambda _self, _target, **_kw: _Health()
        )
        self.health.start()
        self.addCleanup(self.health.stop)

    def _tick(self) -> list[str]:
        return daemon_mod.run_replica_reaper_tick(self.args, now=self.NOW)

    def _retired(self) -> list[dict]:
        return [d for name, d in self.events if name == "replica_retired"]

    def test_a_replica_past_max_idle_is_retired(self):
        self._ready(last_used=self.NOW - 1801.0)
        self.assertEqual(self._tick(), [REPLICA_ID])
        self.assertEqual(self.unloads, [(REPLICA_ID, "127.0.0.1", 11436)])

    def test_a_replica_inside_max_idle_is_kept(self):
        self._ready(last_used=self.NOW - 10.0)
        self.assertEqual(self._tick(), [])
        self.assertEqual(self.unloads, [])

    def test_a_busy_replica_is_never_retired(self):
        self._ready(in_flight=1, last_used=self.NOW - 99999.0)
        self.assertEqual(self._tick(), [])
        self.assertEqual(self.unloads, [])

    def test_a_cold_replica_is_not_retired(self):
        # Nothing to free: the route has no process, so an unload call would be
        # noise and the "retired" event would be a lie.
        self._ready(health_state="cold", last_used=self.NOW - 99999.0)
        self.assertEqual(self._tick(), [])
        self.assertEqual(self.unloads, [])

    def test_a_replica_is_retired_when_a_conflicting_model_is_loading(self):
        self._ready(last_used=self.NOW - 5.0)
        _write_matrix(self.config, {"pair_0": [OTHER_ID, BASE_ID]})
        _claim(self.state, OTHER_ID, 3.0, now=self.NOW)
        self.assertEqual(self._tick(), [REPLICA_ID])
        self.assertEqual([d["reason"] for d in self._retired()], ["gpu_needed"])

    def test_an_idle_retirement_reports_the_idle_seconds(self):
        self._ready(last_used=self.NOW - 2000.0)
        self._tick()
        retired = self._retired()
        self.assertEqual(retired[0]["replica"], REPLICA_ID)
        self.assertEqual(retired[0]["reason"], "idle")
        self.assertEqual(retired[0]["idle_s"], 2000.0)

    def test_max_idle_s_zero_disables_the_backstop(self):
        self.replica_defaults["max_idle_s"] = 0
        self._ready(last_used=self.NOW - 99999.0)
        self.assertEqual(self._tick(), [])
        self.assertEqual(self.unloads, [])
        self.assertEqual(self.events, [], "an unconfigured reaper must be silent")

    def test_a_failed_unload_is_not_reported_as_retired(self):
        self._ready(last_used=self.NOW - 99999.0)
        self.unload_mock.side_effect = lambda *a, **k: False
        self.assertEqual(self._tick(), [])
        self.assertEqual(self._retired(), [])

    # -- the config-write prohibition --------------------------------------- #
    def test_retirement_never_writes_the_llama_swap_config(self):
        before = self.config.read_bytes()
        before_mtime = self.config.stat().st_mtime_ns
        self._ready(last_used=self.NOW - 99999.0)
        self.assertEqual(self._tick(), [REPLICA_ID])
        self.assertEqual(self.config.read_bytes(), before)
        self.assertEqual(self.config.stat().st_mtime_ns, before_mtime)
        self.assertEqual(self.catalog.read_text(encoding="utf-8"), "[]")

    def test_the_reaper_source_references_no_config_writer(self):
        # llama-swap runs --watch-config, so ANY config.yaml write tears down every
        # loaded model. A retirement path that rendered config would manufacture
        # the exact ping-pong this feature removes, so the ban is checked at the
        # source: the reaper may only reach for the unload endpoint.
        source = Path(daemon_mod.__file__).read_text(encoding="utf-8")
        body = source[source.index("def unload_replica_via_llamaswap("):]
        body = body[: body.index("def _prepare_manager_socket_path(")]
        for name in (
            "drop_internal_route_from_llamaswap_config",
            "ensure_replica_route_in_llamaswap_config",
            "ensure_replica_route",
            "render_llamaswap_config",
            "set_instance_mmproj_in_llamaswap_config",
            "save_catalog",
            "persist_server_config",
            "update_config",
            "sync_config_from_server_config_for_startup",
        ):
            self.assertNotIn(name, body, f"the reaper must never call {name}")

    def test_the_unload_helper_targets_the_documented_endpoint(self):
        self.unload.stop()
        response = mock.Mock(status_code=200)
        post = mock.Mock(return_value=response)
        with mock.patch.object(daemon_mod.requests, "post", post):
            ok = daemon_mod.unload_replica_via_llamaswap(REPLICA_ID, "127.0.0.1", 11436)
        self.assertTrue(ok)
        self.assertEqual(
            post.call_args.args[0],
            f"http://127.0.0.1:11436/api/models/unload/{REPLICA_ID}",
        )

    def test_the_unload_helper_swallows_a_transport_error_into_the_log(self):
        self.unload.stop()
        post = mock.Mock(side_effect=OSError("connection refused"))
        with mock.patch.object(daemon_mod.requests, "post", post):
            self.assertFalse(daemon_mod.unload_replica_via_llamaswap(REPLICA_ID, "127.0.0.1", 11436))
        self.assertIn("replica_unload_failed", dict(self.events))

    def test_the_unload_helper_reports_a_rejection(self):
        self.unload.stop()
        post = mock.Mock(return_value=mock.Mock(status_code=404))
        with mock.patch.object(daemon_mod.requests, "post", post):
            self.assertFalse(daemon_mod.unload_replica_via_llamaswap(REPLICA_ID, "127.0.0.1", 11436))
        self.assertIn("replica_unload_failed", dict(self.events))


class ReaperThreadTest(unittest.TestCase):
    def setUp(self):
        self.calls: list[int] = []

    def test_the_thread_polls_and_stops(self):
        done = threading.Event()

        def tick(args, *, now=None):
            self.calls.append(1)
            done.set()
            return []

        with mock.patch.object(daemon_mod, "run_replica_reaper_tick", tick):
            stop = threading.Event()
            thread = daemon_mod.start_replica_reaper(
                argparse.Namespace(), poll_s=0.01, stop_event=stop
            )
            self.assertTrue(done.wait(5.0))
            stop.set()
            thread.join(5.0)
            self.assertFalse(thread.is_alive())
            self.assertTrue(thread.daemon)

    def test_a_failing_tick_is_logged_and_the_loop_survives(self):
        seen = threading.Event()
        state = {"n": 0}

        def tick(args, *, now=None):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("boom")
            seen.set()
            return []

        with mock.patch.object(daemon_mod, "run_replica_reaper_tick", tick), \
             mock.patch.object(daemon_mod, "log_api_event") as log_mock:
            stop = threading.Event()
            thread = daemon_mod.start_replica_reaper(
                argparse.Namespace(), poll_s=0.01, stop_event=stop
            )
            try:
                self.assertTrue(seen.wait(5.0), "one bad tick must not kill the reaper")
                self.assertIn(
                    "replica_reaper_error",
                    [call.args[0] for call in log_mock.call_args_list],
                )
            finally:
                stop.set()
                thread.join(5.0)

    def test_daemon_mode_launches_the_reaper(self):
        source = Path(daemon_mod.__file__).read_text(encoding="utf-8")
        body = source[source.index("def daemon_mode(args):"):]
        self.assertIn("start_replica_reaper(args)", body)


# --------------------------------------------------------------------------- #
# config normalisation
# --------------------------------------------------------------------------- #
class ReplicasConfigNormalisationTest(unittest.TestCase):
    KEYS = {"prefer_base_over_replica": False, "idle_grace_s": 600, "max_idle_s": 1800}

    def test_a_replicas_dict_without_the_new_keys_is_filled(self):
        norm, changed = normalize_server_config_payload({"replicas": {"enabled": True}})
        self.assertTrue(changed)
        for key, value in self.KEYS.items():
            self.assertEqual(norm["replicas"][key], value)

    def test_a_second_pass_is_idempotent(self):
        norm, _ = normalize_server_config_payload({"replicas": {"enabled": True}})
        norm2, changed2 = normalize_server_config_payload(copy.deepcopy(norm))
        self.assertIs(changed2, False)
        self.assertEqual(norm2, norm)

    def test_existing_values_are_never_overwritten(self):
        norm, _ = normalize_server_config_payload(
            {"replicas": {"prefer_base_over_replica": True, "idle_grace_s": 0, "max_idle_s": 60}}
        )
        self.assertIs(norm["replicas"]["prefer_base_over_replica"], True)
        self.assertEqual(norm["replicas"]["idle_grace_s"], 0)
        self.assertEqual(norm["replicas"]["max_idle_s"], 60)

    def test_negative_seconds_are_clamped_to_zero(self):
        norm, _ = normalize_server_config_payload(
            {"replicas": {"idle_grace_s": -30, "max_idle_s": -1}}
        )
        self.assertEqual(norm["replicas"]["idle_grace_s"], 0)
        self.assertEqual(norm["replicas"]["max_idle_s"], 0)

    def test_a_string_seconds_value_is_coerced_once(self):
        norm, changed = normalize_server_config_payload({"replicas": {"idle_grace_s": "120"}})
        self.assertEqual(norm["replicas"]["idle_grace_s"], 120)
        self.assertTrue(changed)
        _norm2, changed2 = normalize_server_config_payload(copy.deepcopy(norm))
        self.assertIs(changed2, False)

    def test_an_uncoercible_value_falls_back_to_the_default(self):
        norm, _ = normalize_server_config_payload({"replicas": {"max_idle_s": "soon"}})
        self.assertEqual(norm["replicas"]["max_idle_s"], 1800)

    def test_both_default_dicts_agree(self):
        # install.py carries a verbatim duplicate; a fresh install and an upgraded
        # one must produce the same conf.json shape.
        import llamacpp_stack.install as install_mod

        self.assertEqual(
            _cli_impl._default_global_replicas_config(),
            install_mod._default_global_replicas_config(),
        )
        for key in self.KEYS:
            self.assertIn(key, _cli_impl._default_global_replicas_config())


class BusyVictimAdmissionTest(unittest.TestCase):
    """A load must never be admitted when it would cancel a request in flight.

    ``get_gpu_conflict_message`` used to identify running processes by their
    weights path. A base and its replicas load the same file, so every instance
    of a model collapsed onto one id: an idle replica could never appear in the
    conflict set, the admission decision was blind to it, and the only victim it
    could ever name was the base -- the one actually serving.
    """

    TARGET = OTHER_ID
    RID = REPLICA_ID

    def setUp(self):
        _cli_impl.REPLICA_ROUTER_STATE.records.clear()
        _cli_impl.REPLICA_ROUTER_STATE.affinity.clear()
        _cli_impl.REPLICA_ROUTER_STATE.base_in_flight.clear()
        _cli_impl.REPLICA_ROUTER_STATE.response_to_replica.clear()
        for attr in ("vision_until", "vision_affinity"):
            getattr(_cli_impl.REPLICA_ROUTER_STATE, attr, {}).clear()

        self.catalog = [
            replace(_make_model(BASE_ID), local_path="/models/base"),
            replace(_make_model(OTHER_ID), local_path="/models/other"),
        ]
        self.procs = [
            {"pid": 4242, "cmdline": "/bin/llama-server --model /models/base",
             "port": 18097, "model_path": "/models/base"},
            # Same weights file as the base: the replica is what the path map
            # cannot tell apart.
            {"pid": 4343, "cmdline": "/bin/llama-server --model /models/base",
             "port": 18098, "model_path": "/models/base"},
        ]
        self.gpu_map = {4242: 21000, 4343: 20954}

    def _record_replica(self, *, pid: int = 4343, gpu_set=(1,), status: str = "ready"):
        _cli_impl.REPLICA_ROUTER_STATE.records[self.RID] = ReplicaRecord(
            base_model_id=BASE_ID, replica_model_id=self.RID, gpu_set=list(gpu_set),
            status=status, pid=pid, port=18098, in_flight=0,
        )

    def _run(self, *, in_flight: int, target_groups=None, victim_groups=None, fits: bool = True):
        events: dict[str, list] = {}
        groups = {
            self.TARGET: set(target_groups or {"group_T"}),
            self.RID: set(victim_groups or {"group_R"}),
        }

        def cap(kind, payload=None, **_kw):
            events.setdefault(kind, []).append(payload or {})
            return None

        _cli_impl.REPLICA_ROUTER_STATE.records[self.RID].in_flight = in_flight
        with mock.patch.object(_cli_impl, "get_llama_server_processes", return_value=self.procs), \
             mock.patch.object(_cli_impl, "get_gpu_process_map", return_value=self.gpu_map), \
             mock.patch.object(_cli_impl, "get_catalog_model_process", return_value=None), \
             mock.patch.object(_cli_impl, "_matrix_group_membership", return_value=groups), \
             mock.patch.object(_cli_impl, "model_has_enough_vram_capacity",
                               return_value=(fits, {"reason": "fits_capacity"})), \
             mock.patch.object(_cli_impl, "log_api_event", side_effect=cap):
            message = _cli_impl.get_gpu_conflict_message(
                self.TARGET, self.catalog, "127.0.0.1", 11436, Path("/tmp/none/config.yaml")
            )
        return message, events

    def test_an_idle_replica_is_named_as_the_victim_instead_of_the_base(self):
        self._record_replica()
        message, events = self._run(in_flight=0)
        self.assertIsNone(message)
        allowed = events["model_load_allowed_matrix_evict"][0]
        will_evict = allowed["will_evict"]
        # The replica used to collapse into the base's id, so an idle replica was
        # never a candidate victim. Both instances must now be distinguishable.
        self.assertIn(self.RID, will_evict)
        self.assertIn(BASE_ID, will_evict)
        self.assertEqual(len(will_evict), 2)

    def test_the_replica_fixture_shares_its_base_weights_path(self):
        self._record_replica()
        # The property that made the old path->id map blind: both instances of a
        # model run the identical file, so path alone cannot name the replica.
        self.assertEqual(
            _cli_impl._safe_realpath(self.procs[0]["model_path"]),
            _cli_impl._safe_realpath(self.procs[1]["model_path"]),
        )

    def test_a_busy_victim_is_refused_instead_of_having_its_request_cancelled(self):
        self._record_replica()
        message, events = self._run(in_flight=2)
        self.assertIsNotNone(message)
        self.assertIn("would cancel a request", message)
        self.assertIn(self.RID, message)
        self.assertNotIn("model_load_allowed_matrix_evict", events)
        busy = events["model_load_blocked_busy_victim"][0]
        self.assertEqual(busy["busy_victims"][0]["instance"], self.RID)
        self.assertEqual(busy["busy_victims"][0]["in_flight"], 2)

    def test_the_refusal_names_the_busy_instance_and_leaves_the_idle_one_alone(self):
        self._record_replica()
        _cli_impl.REPLICA_ROUTER_STATE.records[self.RID].in_flight = 0
        _cli_impl.REPLICA_ROUTER_STATE.base_in_flight[BASE_ID] = 3
        message, events = self._run(in_flight=0)
        self.assertIsNotNone(message)
        self.assertIn(BASE_ID, message)
        self.assertIn("3 request(s) in flight", message)

    def test_a_model_never_conflicts_with_its_own_family(self):
        self._record_replica()
        _cli_impl.REPLICA_ROUTER_STATE.records[self.RID].in_flight = 4
        message, events = self._run(
            in_flight=0, target_groups={"group_R"}, victim_groups={"group_R"}
        )
        # The replica belongs to the base, so for the base's own load it is not a
        # conflict at all -- otherwise reloading would always look blocked.
        self.assertNotIn("would cancel", message or "")

    def test_instance_in_flight_reads_replica_records_and_base_counters(self):
        self._record_replica()
        _cli_impl.REPLICA_ROUTER_STATE.records[self.RID].in_flight = 5
        self.assertEqual(_cli_impl.instance_in_flight(self.RID), 5)
        _cli_impl.REPLICA_ROUTER_STATE.base_in_flight[BASE_ID] = 2
        self.assertEqual(_cli_impl.instance_in_flight(BASE_ID), 2)
        self.assertEqual(_cli_impl.instance_in_flight("never-heard-of-it"), 0)
        self.assertEqual(_cli_impl.instance_in_flight(""), 0)


class CrossModelPlacementTest(unittest.TestCase):
    """Two ~20GB models must not both be pinned to GPU 0.

    `build_llama_server_command` turns a single-GPU `tensor_split` of "1" into
    `CUDA_VISIBLE_DEVICES=0`, so every model was pinned to the same card and they
    evicted each other forever. Placement is therefore decided at render time,
    largest first round-robin, and a replica never lands on its base's GPU.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _sparse(self, name: str, mib: int) -> str:
        path = self.root / name
        with open(path, "wb") as handle:
            handle.truncate(mib * 1024 * 1024)
        return str(path)

    def _models(self):
        heavy_a = _make_model("heavy-a", tensor_split="1")
        heavy_b = _make_model("heavy-b", tensor_split="1")
        small = _make_model("small", tensor_split="1")
        return [
            replace(heavy_a, local_path=self._sparse("heavy-a.safetensors", 16304)),
            replace(heavy_b, local_path=self._sparse("heavy-b.safetensors", 14650)),
            replace(small, local_path=self._sparse("small.safetensors", 500)),
        ]

    def test_the_two_heaviest_models_land_on_different_gpus(self):
        catalog = self._models()
        assignment = R.assign_model_gpu_sets(catalog, 2)
        self.assertEqual(assignment["heavy-a"], [0])
        self.assertEqual(assignment["heavy-b"], [1])

    def test_small_models_are_left_alone_so_they_can_still_pack(self):
        assignment = R.assign_model_gpu_sets(self._models(), 2)
        self.assertNotIn("small", assignment)

    def test_a_single_gpu_host_gets_no_assignment(self):
        self.assertEqual(R.assign_model_gpu_sets(self._models(), 1), {})

    def test_placement_is_stable_across_calls(self):
        catalog = self._models()
        self.assertEqual(
            R.assign_model_gpu_sets(catalog, 2), R.assign_model_gpu_sets(list(catalog), 2)
        )

    def test_a_replica_never_lands_on_its_bases_gpu(self):
        model = self._models()[0]
        cfg = R.ReplicaConfig(enabled=True, max=2, gpus_per_replica=1, placement="exclusive_gpus")
        with_base_on_gpu1 = R._replica_gpu_sets(model, cfg, 2, base_gpu_set=[1])
        for gpu_set in with_base_on_gpu1:
            self.assertNotIn(1, gpu_set)
        self.assertTrue(with_base_on_gpu1)

    def test_render_pins_the_two_bases_to_different_gpus_with_one_prefix_each(self):
        catalog = self._models()
        out = self.root / "config.yaml"
        R.render_llamaswap_config(
            catalog, out, SERVER, 18097, 18000,
            server_defaults={}, replica_defaults={},
        )
        rendered = yaml.safe_load(out.read_text())
        seen = {}
        for model_id in ("heavy-a", "heavy-b"):
            cmd = str(rendered["models"][model_id]["cmd"])
            self.assertEqual(cmd.count("CUDA_VISIBLE_DEVICES="), 1, model_id)
            seen[model_id] = cmd.split("CUDA_VISIBLE_DEVICES=")[1].split()[0]
        self.assertEqual(seen["heavy-a"], "0")
        self.assertEqual(seen["heavy-b"], "1")

    def test_render_keeps_the_builder_prefix_on_a_model_placement_does_not_cover(self):
        # A model spanning both GPUs is excluded from the assignment (only
        # single-GPU models get one), so render must leave the tensor_split-derived
        # prefix alone instead of stripping it.
        wide = replace(
            _make_model("wide", tensor_split="1,1"),
            local_path=self._sparse("wide.safetensors", 16304),
        )
        out = self.root / "config.yaml"
        with mock.patch("llamacpp_stack.cli.replica.detect_cuda_device_count", return_value=2):
            R.render_llamaswap_config(
                [wide], out, SERVER, 18097, 18000,
                server_defaults={}, replica_defaults={},
            )
        cmd = str(yaml.safe_load(out.read_text())["models"]["wide"]["cmd"])
        self.assertEqual(cmd.count("CUDA_VISIBLE_DEVICES="), 1)
        self.assertEqual(cmd.split("CUDA_VISIBLE_DEVICES=")[1].split()[0], "0,1")

    def test_render_declares_two_disjoint_gpu_models_as_co_loadable(self):
        catalog = self._models()
        out = self.root / "config.yaml"
        R.render_llamaswap_config(
            catalog, out, SERVER, 18097, 18000,
            server_defaults={}, replica_defaults={},
        )
        matrix = yaml.safe_load(out.read_text())["matrix"]
        var_of = {mid: var for var, mid in (matrix.get("vars") or {}).items()}
        together = [
            name for name, expr in (matrix.get("sets") or {}).items()
            if var_of.get("heavy-a") in str(expr).split(" & ")
            and var_of.get("heavy-b") in str(expr).split(" & ")
        ]
        self.assertTrue(
            together,
            "two large models on disjoint GPUs must share a matrix set or they evict each other",
        )

    def test_the_memo_returns_a_copy_so_callers_cannot_corrupt_it(self):
        catalog = self._models()
        first = R.cached_model_gpu_sets(catalog, 2)
        first["heavy-a"] = [1]
        self.assertEqual(R.cached_model_gpu_sets(catalog, 2)["heavy-a"], [0])


class PreflightUsesTheAssignedCardTest(unittest.TestCase):
    """The VRAM preflight must measure the card the model will launch on.

    `model_launch_gpu_set` answers [0] for every single-GPU model, so measuring
    against it refuses a model assigned to an idle GPU1 while GPU0 is full.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.catalog = [
            replace(_make_model("on-gpu-0", tensor_split="1"), local_path=self._sparse("a.safetensors", 14650)),
            replace(_make_model("on-gpu-1", tensor_split="1"), local_path=self._sparse("b.safetensors", 14650)),
        ]
        self.assignment = R.assign_model_gpu_sets(self.catalog, 2)
        self.addCleanup(R._MODEL_GPU_ASSIGNMENT_MEMO.clear)

    def _sparse(self, name: str, mib: int) -> str:
        path = self.root / name
        with open(path, "wb") as handle:
            handle.truncate(mib * 1024 * 1024)
        return str(path)

    def _model(self, model_id: str):
        return next(m for m in self.catalog if m.model_id == model_id)

    def test_the_two_models_are_assigned_different_cards(self):
        self.assertEqual(self.assignment, {"on-gpu-0": [0], "on-gpu-1": [1]})

    def test_it_measures_the_assigned_card_not_the_launch_default(self):
        self.assertEqual(_cli_impl.model_launch_gpu_set(self._model("on-gpu-1")), [0])
        snapshot = {0: {"free_mib": 100.0}, 1: {"free_mib": 24000.0}}
        with (
            mock.patch.object(_cli_impl, "_query_gpu_memory_snapshot_cached", return_value=snapshot),
            mock.patch.object(_cli_impl, "estimate_model_runtime_mib", return_value=18000.0),
        ):
            fits, info = _cli_impl.model_has_enough_free_vram_to_load(
                self._model("on-gpu-1"), gpu_set=self.assignment["on-gpu-1"]
            )
        self.assertTrue(fits, info)
        self.assertEqual(info["checks"][0]["gpu"], 1)

    def test_a_full_assigned_card_is_still_refused(self):
        snapshot = {0: {"free_mib": 24000.0}, 1: {"free_mib": 100.0}}
        with (
            mock.patch.object(_cli_impl, "_query_gpu_memory_snapshot_cached", return_value=snapshot),
            mock.patch.object(_cli_impl, "estimate_model_runtime_mib", return_value=18000.0),
        ):
            fits, info = _cli_impl.model_has_enough_free_vram_to_load(
                self._model("on-gpu-1"), gpu_set=self.assignment["on-gpu-1"]
            )
        self.assertFalse(fits)
        self.assertEqual(info["reason"], "insufficient_vram")


class ReplicaStaysOffReservedGpusTest(unittest.TestCase):
    """A replica must not take the only card another model's base was assigned.

    Replicas are cheaper to evict (0.5x), so one landing on a neighbour's card
    wins the eviction, then the neighbour's request takes the card back: the two
    trade the GPU on every turn.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.cfg = ReplicaConfig(enabled=True, max=2, gpus_per_replica=1, placement="exclusive_gpus")
        self.catalog = [
            replace(_make_model("heavy-a", tensor_split="1"), local_path=self._sparse("a.safetensors", 16304)),
            replace(_make_model("heavy-b", tensor_split="1"), local_path=self._sparse("b.safetensors", 14650)),
        ]
        self.assignment = R.assign_model_gpu_sets(self.catalog, 2)

    def _sparse(self, name: str, mib: int) -> str:
        path = self.root / name
        with open(path, "wb") as handle:
            handle.truncate(mib * 1024 * 1024)
        return str(path)

    def _model(self, model_id: str):
        return next(m for m in self.catalog if m.model_id == model_id)

    def _sets(self, model_id: str) -> list[list[int]]:
        return R._replica_gpu_sets(
            self._model(model_id), self.cfg, 2,
            base_gpu_set=self.assignment.get(model_id),
            reserved_gpu_set=R.reserved_base_gpus(self.catalog, model_id, 2),
        )

    def test_no_replica_is_offered_when_every_card_is_a_base(self):
        self.assertEqual(self._sets("heavy-a"), [])
        self.assertEqual(self._sets("heavy-b"), [])

    def test_reserved_base_gpus_exclude_only_the_other_models(self):
        self.assertEqual(R.reserved_base_gpus(self.catalog, "heavy-a", 2), {1})
        self.assertEqual(R.reserved_base_gpus(self.catalog, "heavy-b", 2), {0})

    def test_a_third_gpu_would_still_host_a_replica(self):
        self.assertEqual(
            R._replica_gpu_sets(
                self._model("heavy-a"), self.cfg, 3,
                base_gpu_set=self.assignment["heavy-a"],
                reserved_gpu_set=R.reserved_base_gpus(self.catalog, "heavy-a", 3),
            ),
            [[2]],
        )

    def test_reserved_gpus_never_shrink_the_pool_below_the_base_own_set(self):
        sets = R._replica_gpu_sets(
            self._model("heavy-a"), self.cfg, 4,
            base_gpu_set=[0, 1],
            reserved_gpu_set=R.reserved_base_gpus(self.catalog, "heavy-a", 4),
        )
        for gpu_set in sets:
            self.assertFalse({0, 1} & set(gpu_set))


class GpuDemandTest(unittest.TestCase, _RouterStateMixin):
    """A refused load must be able to reclaim the card an idle replica holds."""

    def setUp(self):
        self.reset_router_state()
        self.state = _cli_impl.REPLICA_ROUTER_STATE

    def test_a_note_marks_exactly_the_given_gpus(self):
        self.state.note_gpu_demand([2], now=100.0)
        self.assertEqual(self.state.demanded_gpus(now=100.0), {2})

    def test_a_note_expires(self):
        self.state.note_gpu_demand([2], ttl_s=30.0, now=100.0)
        self.assertEqual(self.state.demanded_gpus(now=129.0), {2})
        self.assertEqual(self.state.demanded_gpus(now=131.0), set())

    def test_repeated_notes_take_the_later_deadline(self):
        self.state.note_gpu_demand([1], ttl_s=10.0, now=100.0)
        self.state.note_gpu_demand([1], ttl_s=100.0, now=110.0)
        self.assertEqual(self.state.demanded_gpus(now=150.0), {1})

    def test_a_demanded_gpu_is_only_kept_if_it_was_already_wanted(self):
        self.state.note_gpu_demand([1], ttl_s=100.0, now=100.0)
        self.state.note_gpu_demand([0], ttl_s=10.0, now=100.0)
        self.assertEqual(self.state.demanded_gpus(now=105.0), {0, 1})

    def test_an_empty_or_broken_gpu_set_is_ignored(self):
        self.state.note_gpu_demand([], now=100.0)
        self.state.note_gpu_demand(None, now=100.0)
        self.state.note_gpu_demand(["not-a-gpu"], now=100.0)
        self.assertEqual(self.state.demanded_gpus(now=100.0), set())

    def test_prune_drops_expired_demands(self):
        self.state.note_gpu_demand([3], ttl_s=5.0, now=100.0)
        self.state.prune(now=200.0)
        self.assertEqual(self.state.gpu_demand, {})


class DemandDrivenRetirementTest(ReplicaReaperTest):
    """An idle replica on a GPU somebody was refused must be retired."""

    def setUp(self):
        super().setUp()
        self.plan_args = self.args

    def test_it_retires_the_squatter_without_waiting_for_max_idle_s(self):
        self._ready(replica_id=REPLICA_ID, gpu_set=(1,), last_used=self.NOW - 5.0)
        self.state.note_gpu_demand([1], now=self.NOW)
        retired = daemon_mod.run_replica_reaper_tick(self.plan_args, now=self.NOW)
        self.assertEqual(retired, [REPLICA_ID])
        self.assertEqual(self.unloads, [(REPLICA_ID, "127.0.0.1", 11436)])
        reasons = [data.get("reason") for event, data in self.events if event == "replica_retired"]
        self.assertEqual(reasons, ["gpu_needed"])

    def test_a_demand_on_another_gpu_leaves_the_replica_alone(self):
        self._ready(replica_id=REPLICA_ID, gpu_set=(1,), last_used=self.NOW - 5.0)
        self.state.note_gpu_demand([0], now=self.NOW)
        self.assertEqual(daemon_mod.run_replica_reaper_tick(self.plan_args, now=self.NOW), [])
        self.assertEqual(self.unloads, [])

    def test_an_expired_demand_leaves_the_replica_alone(self):
        self._ready(replica_id=REPLICA_ID, gpu_set=(1,), last_used=self.NOW - 5.0)
        self.state.note_gpu_demand([1], ttl_s=_cli_impl.GPU_DEMAND_TTL_S, now=self.NOW)
        late = self.NOW + _cli_impl.GPU_DEMAND_TTL_S + 1.0
        self.assertEqual(daemon_mod.run_replica_reaper_tick(self.plan_args, now=late), [])
        self.assertEqual(self.unloads, [])

    def test_a_replica_serving_a_request_is_never_reclaimed(self):
        self._ready(replica_id=REPLICA_ID, gpu_set=(1,), last_used=self.NOW - 5.0, in_flight=1)
        self.state.note_gpu_demand([1], now=self.NOW)
        self.assertEqual(daemon_mod.run_replica_reaper_tick(self.plan_args, now=self.NOW), [])
        self.assertEqual(self.unloads, [])

    def test_an_absent_replica_is_not_announced_as_retired(self):
        self._ready(health_state="cold", replica_id=REPLICA_ID, gpu_set=(1,))
        self.state.note_gpu_demand([1], now=self.NOW)
        self.assertEqual(daemon_mod.run_replica_reaper_tick(self.plan_args, now=self.NOW), [])
        self.assertEqual(self.unloads, [])


class RefusalRegistersDemandTest(unittest.TestCase, _RouterStateMixin):
    """A refused load must advertise the GPUs it needs, so the reaper can act."""

    CONFLICT = "the GPU is already in use: something else"
    NOW = 10_000.0

    _bind = RejectIfGpuBusyTest._bind
    NOW = 10_000.0

    _bind = RejectIfGpuBusyTest._bind

    def setUp(self):
        self.reset_router_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / "config.yaml"
        _write_matrix(self.config, {"pair_0": [OTHER_ID, BASE_ID]})
        self.state = _cli_impl.REPLICA_ROUTER_STATE
        self.args = argparse.Namespace(config=self.config, public_port=11436)
        self.sent: list[tuple[dict, int]] = []
        self.events: list[tuple[str, dict]] = []
        stub = mock.Mock()
        stub._send_json = lambda payload, status=200: self.sent.append((payload, status))
        self.stub = stub
        self.log_mock = mock.Mock(
            side_effect=lambda event, data=None: self.events.append((event, data or {}))
        )
        self.handler_source = inspect.getsource(_cli_impl.start_ctx_metadata_server)

    def _call(self, model_name: str = "wants-gpu-1", *, api_style: str = "openai"):
        with mock.patch.object(_cli_impl.time, "monotonic", return_value=self.NOW):
            return self._method()(self.stub, model_name, [], api_style=api_style)

    def _method(self):
        namespace = {
            "args": self.args,
            "client_host": "127.0.0.1",
            "ManagedModel": ManagedModel,
            "log_api_event": self.log_mock,
            "REPLICA_ROUTER_STATE": self.state,
            "blocking_conflicting_load": _cli_impl.blocking_conflicting_load,
            "plan_runtime_placement": _cli_impl.plan_runtime_placement,
            "retire_idle_occupants": _cli_impl.retire_idle_occupants,
            "apply_repin": _cli_impl.apply_repin,
            "get_catalog_model_process": mock.Mock(return_value=None),
            "get_gpu_conflict_message": mock.Mock(return_value=self.CONFLICT),
            "cached_model_gpu_sets": mock.Mock(return_value={"wants-gpu-1": [1]}),
            "get_model_activity_snapshot": mock.Mock(return_value=({}, None)),
            "recent_activity_blocking_model_switch": mock.Mock(return_value=None),
            "request_looks_like_model_probe": mock.Mock(return_value=False),
            "time": _cli_impl.time,
            "_resolve_model_probe_autoload_config": mock.Mock(return_value={"enabled": False}),
        }
        self.gpu_sets_mock = namespace["cached_model_gpu_sets"]
        return self._bind(namespace)

    def test_a_refused_load_marks_the_requested_gpu(self):
        self.assertTrue(self._call("wants-gpu-1"))
        self.assertEqual(self.state.demanded_gpus(now=self.NOW), {1})

    def test_the_demand_uses_the_assigned_gpu_not_the_occupied_one(self):
        self.assertTrue(self._call("wants-gpu-1"))
        self.gpu_sets_mock.assert_called_once()

    def test_an_allowed_load_marks_nothing(self):
        method = self._method()
        method.__globals__["get_gpu_conflict_message"] = mock.Mock(return_value=None)
        with mock.patch.object(_cli_impl.time, "monotonic", return_value=self.NOW):
            self.assertFalse(method(self.stub, "wants-gpu-1", [], api_style="openai"))
        self.assertEqual(self.state.demanded_gpus(now=self.NOW), set())

    def test_an_unknown_model_marks_nothing_but_is_still_refused(self):
        self.assertTrue(self._call("never-heard-of-it"))
        self.assertEqual(self.state.demanded_gpus(now=self.NOW), set())


if __name__ == "__main__":
    unittest.main()