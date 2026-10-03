"""Runtime GPU placement: the planner, its hysteresis, and the safe config writer.

The GPU a route launches on is baked into its command by ``assign_model_gpu_sets``,
so "put this model on the free card" is not a routing decision llama-swap can make:
either the route's command is rewritten or the current occupant is retired. These
tests pin that decision down without touching a GPU.
"""

import argparse
import contextlib
import inspect
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path
from unittest import mock

import yaml

from llamacpp_stack import _cli_impl
from llamacpp_stack.cli import replica as R
from llamacpp_stack.cli.models import ReplicaConfig

TARGET_ID = "wants-gpu-1"
OCCUPANT_ID = "other-model"
REPLICA_ID = f"{TARGET_ID}__replica_0"
SERVER = Path("/usr/bin/llama-server")

OCCUPANT_PID = 4242
OCCUPANT_MIB = 18218.0
NEEDED_MIB = 20045.0
ALT_FREE_MIB = 24069.0
SMALL_FREE_MIB = 9000.0


def _monotonic() -> float:
    """Overrides are stamped with time.monotonic, so tests must start from the real
    clock or every live assertion reads as already expired."""
    return _cli_impl.time.monotonic()


def _make_model(model_id: str, local_path: str, tensor_split: str = "1"):
    return R.ManagedModel(
        model_id=model_id,
        repo_id="org/model",
        quant="EXL3_3.00bpw",
        filename="model.safetensors",
        local_path=local_path,
        backend="llamacpp",
        mmproj_filename=None,
        mmproj_path=None,
        load_capabilities=[],
        aliases=[],
        ctx_size=262144,
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
    )


class _RouterStateMixin:
    """The router state is a process-lifetime singleton; keep tests independent."""

    def _clear_placement_state(self):
        state = _cli_impl.REPLICA_ROUTER_STATE
        with state.lock:
            state.gpu_override.clear()
            state.gpu_override_written_at.clear()
            state.gpu_demand.clear()
            state.base_in_flight.clear()
            state.records.clear()
        _cli_impl._PLACEMENT_WRITE_LOG.clear()

    def reset_placement_state(self):
        # The cleanup must be the non-registering variant: a self-registering one
        # re-arms itself on every cleanup and never finishes.
        self._clear_placement_state()
        self.addCleanup(self._clear_placement_state)


class PlannerTest(_RouterStateMixin, unittest.TestCase):
    def setUp(self):
        self.reset_placement_state()
        self.catalog = [_make_model(TARGET_ID, "/models/target.safetensors")]
        self.state = _cli_impl.REPLICA_ROUTER_STATE

    @contextlib.contextmanager
    def _planning(
        self,
        *,
        enabled: bool = True,
        repin: bool = False,
        loaded: bool = False,
        replica_sets=(),
        pinned=(1,),
        occupants=None,
        in_flight: int = 1,
        alt=(0,),
        alt_free: float = ALT_FREE_MIB,
        fits_alt: bool = False,
        claim: bool = True,
    ):
        if occupants is None:
            occupants = {pinned[0]: [(OCCUPANT_ID, OCCUPANT_PID, OCCUPANT_MIB)]} if pinned else {}
        stack = contextlib.ExitStack()
        patches = {
            "_runtime_placement_enabled": enabled,
            "_placement_repin_enabled": repin,
            "get_catalog_model_process": mock.Mock(
                return_value={"pid": 1} if loaded else None
            ),
            "get_model_replica_config": mock.Mock(
                return_value=ReplicaConfig(enabled=bool(replica_sets), max=2)
            ),
            "resolve_global_replica_config": mock.Mock(return_value={}),
            "_replica_gpu_sets": mock.Mock(return_value=list(replica_sets)),
            "effective_model_gpu_set": mock.Mock(return_value=list(pinned)),
            "_gpu_occupants": mock.Mock(return_value=dict(occupants)),
            "estimate_model_runtime_mib": mock.Mock(return_value=NEEDED_MIB),
            "instance_in_flight": mock.Mock(return_value=in_flight),
            "best_alternative_gpu": mock.Mock(return_value=(list(alt) if alt else None, alt_free)),
            "model_has_enough_free_vram_to_load": mock.Mock(return_value=(fits_alt, {"reason": "stub"})),
            "_claim_placement_write": mock.Mock(return_value=claim),
        }
        with stack:
            for name, value in patches.items():
                if callable(value):
                    stack.enter_context(mock.patch.object(_cli_impl, name, value))
                else:
                    stack.enter_context(
                        mock.patch.object(_cli_impl, name, mock.Mock(return_value=value))
                    )
            yield stack

    def _plan(self):
        return _cli_impl.plan_runtime_placement(TARGET_ID, self.catalog)

    def test_a_busy_occupant_is_rejected_and_the_message_names_its_card(self):
        with self._planning(in_flight=2, alt=None, alt_free=0.0):
            plan = self._plan()
        self.assertEqual(plan.action, "reject")
        self.assertEqual(plan.pinned, [1])
        self.assertEqual(plan.occupant, (OCCUPANT_ID, OCCUPANT_PID, int(OCCUPANT_MIB)))
        self.assertIn("pinned GPU 1", plan.message)
        self.assertIn(OCCUPANT_ID, plan.message)
        self.assertIn(str(OCCUPANT_PID), plan.message)
        self.assertIn("2 requests in flight", plan.message)

    def test_an_already_loaded_model_is_never_replanned(self):
        with self._planning(loaded=True, repin=True, fits_alt=True, in_flight=0):
            plan = self._plan()
        self.assertEqual(plan.action, "forward")

    def test_an_idle_occupant_with_nowhere_else_to_go_is_retired(self):
        with self._planning(in_flight=0, alt=None, alt_free=0.0):
            plan = self._plan()
        self.assertEqual(plan.action, "retire_then_forward")
        self.assertIn(f"Unloaded idle '{OCCUPANT_ID}'", plan.message)

    def test_a_free_card_that_fits_is_used_instead_of_unloading_the_occupant(self):
        with self._planning(in_flight=0, repin=True, fits_alt=True):
            plan = self._plan()
        self.assertEqual(plan.action, "repin_then_forward")
        self.assertEqual(plan.alt_gpu_set, [0])
        self.assertIn("Moved the model to GPU 0", plan.message)
        self.assertIn(f"instead of unloading '{OCCUPANT_ID}'", plan.message)

    def test_repinning_is_refused_unless_the_kill_switch_allows_it(self):
        with self._planning(in_flight=0, repin=False, fits_alt=True):
            plan = self._plan()
        self.assertEqual(plan.action, "retire_then_forward")

    def test_replica_mode_never_repins_a_base(self):
        with self._planning(replica_sets=[[0]], in_flight=0, repin=True, fits_alt=True):
            plan = self._plan()
        self.assertEqual(plan.action, "forward")

    def test_the_replica_guard_is_given_the_static_assignment(self):
        seen = {}

        def spy(model, cfg, total_gpus=None, *, base_gpu_set=None, reserved_gpu_set=None):
            seen["base_gpu_set"] = base_gpu_set
            seen["reserved_gpu_set"] = reserved_gpu_set
            return []

        with self._planning(replica_sets=[[0]], in_flight=0):
            with mock.patch.object(_cli_impl, "_replica_gpu_sets", spy), mock.patch.object(
                _cli_impl,
                "cached_model_gpu_sets",
                mock.Mock(return_value={TARGET_ID: [1], OCCUPANT_ID: [1]}),
            ), mock.patch.object(_cli_impl, "reserved_base_gpus", mock.Mock(return_value={0, 1})):
                plan = self._plan()
        self.assertEqual(seen["base_gpu_set"], [1])
        self.assertEqual(seen["reserved_gpu_set"], {0, 1})
        self.assertNotEqual(plan.action, "forward")

    def test_the_planner_is_off_unless_the_kill_switch_enables_it(self):
        with self._planning(enabled=False, in_flight=0, repin=True, fits_alt=True):
            plan = self._plan()
        self.assertEqual(plan.action, "forward")

    def test_the_shortfall_is_named_when_no_card_is_big_enough(self):
        with self._planning(in_flight=1, alt=(0,), alt_free=SMALL_FREE_MIB, fits_alt=False):
            plan = self._plan()
        self.assertEqual(plan.action, "reject")
        self.assertEqual(plan.alt_gpu_set, [0])
        self.assertEqual(plan.alt_free_mib, SMALL_FREE_MIB)
        expected = int(NEEDED_MIB + 1024 + 2048 - SMALL_FREE_MIB)
        self.assertIn(f"short by {expected} MiB", plan.message)
        self.assertIn("GPU 0 has 9000 MiB free", plan.message)

    def test_an_idle_pin_is_left_alone(self):
        with self._planning(occupants={}):
            plan = self._plan()
        self.assertEqual(plan.action, "forward")
        self.assertEqual(plan.pinned, [1])

    def test_the_capacity_check_sees_the_pinned_card_not_the_launch_default(self):
        """model_launch_gpu_set answers [0] for every single-GPU model; trusting it
        measures the wrong card and refuses a model whose own card is empty."""
        seen = []

        def _spy(model, **kwargs):
            seen.append(kwargs.get("gpu_set"))
            return True, {"reason": "stub"}

        with self._planning(in_flight=0, repin=True, alt=(0,)) as stack:
            stack.enter_context(
                mock.patch.object(
                    _cli_impl, "model_has_enough_free_vram_to_load", side_effect=_spy
                )
            )
            stack.enter_context(
                mock.patch.object(
                    _cli_impl,
                    "model_launch_gpu_set",
                    side_effect=AssertionError("launch default must not decide placement"),
                )
            )
            plan = self._plan()
        self.assertEqual(plan.action, "repin_then_forward")
        self.assertEqual(seen, [[0]], "the alternative card is the one that gets measured")


class HysteresisTest(_RouterStateMixin, unittest.TestCase):
    def setUp(self):
        self.reset_placement_state()
        self.state = _cli_impl.REPLICA_ROUTER_STATE
        self.catalog = [_make_model(TARGET_ID, "/models/target.safetensors")]

    def test_a_live_override_outranks_the_static_map(self):
        with mock.patch.object(
            _cli_impl, "cached_model_gpu_sets", return_value={TARGET_ID: [1]}
        ):
            self.assertEqual(_cli_impl.effective_model_gpu_set(TARGET_ID, self.catalog), [1])
            self.assertTrue(self.state.claim_gpu_override(TARGET_ID, [0]))
            self.assertEqual(_cli_impl.effective_model_gpu_set(TARGET_ID, self.catalog), [0])

    def test_an_expired_override_falls_back_to_the_static_map(self):
        now = _monotonic()
        with mock.patch.object(
            _cli_impl, "cached_model_gpu_sets", return_value={TARGET_ID: [1]}
        ):
            self.state.claim_gpu_override(TARGET_ID, [0], ttl_s=10.0, now=now)
            self.assertEqual(_cli_impl.effective_model_gpu_set(TARGET_ID, self.catalog), [0])
            self.assertIsNone(self.state.live_gpu_override(TARGET_ID, now=now + 11.0))
            with mock.patch.object(_cli_impl.time, "monotonic", return_value=now + 11.0):
                self.assertEqual(_cli_impl.effective_model_gpu_set(TARGET_ID, self.catalog), [1])

    def test_prune_drops_an_expired_override(self):
        now = _monotonic()
        self.state.claim_gpu_override(TARGET_ID, [0], ttl_s=10.0, now=now)
        self.state.prune(now=now + 11.0)
        self.assertEqual(self.state.gpu_override, {})
        self.assertEqual(self.state.gpu_override_written_at, {})

    def test_a_second_override_while_one_is_live_is_refused(self):
        now = _monotonic()
        self.assertTrue(self.state.claim_gpu_override(TARGET_ID, [0], now=now))
        self.assertFalse(self.state.claim_gpu_override(TARGET_ID, [1], now=now + 1.0))

    def test_a_second_write_inside_the_minimum_interval_is_refused(self):
        now = _monotonic()
        self.assertTrue(self.state.claim_gpu_override(TARGET_ID, [0], ttl_s=1.0, now=now))
        self.assertFalse(
            self.state.claim_gpu_override(TARGET_ID, [1], now=now + 10.0),
            "the TTL is already spent, so only the minimum interval can refuse this",
        )

    def test_the_hourly_budget_caps_config_rewrites(self):
        clock = _monotonic()
        models = [f"{TARGET_ID}-{index}" for index in range(_cli_impl.PLACEMENT_MAX_WRITES_PER_HOUR + 1)]
        with mock.patch.object(_cli_impl.time, "monotonic", side_effect=lambda: clock):
            for model_id in models[:-1]:
                self.assertTrue(_cli_impl._claim_placement_write(model_id, [0]), model_id)
            self.assertFalse(
                _cli_impl._claim_placement_write(models[-1], [0]),
                "the budget is global, not per model",
            )
            clock += 3601.0
            self.assertTrue(
                _cli_impl._claim_placement_write(models[-1], [0]),
                "entries older than an hour must stop counting against the budget",
            )


class SafeWriterTest(_RouterStateMixin, unittest.TestCase):
    def setUp(self):
        self.reset_placement_state()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config_path = self.root / "config.yaml"
        # Above the small-model threshold, so the matrix gives it its own group.
        self.big = self.root / "big.safetensors"
        with open(self.big, "wb") as handle:
            handle.truncate(6 * 1024 * 1024 * 1024)
        self.model = _make_model(TARGET_ID, str(self.big))
        self.other = _make_model(OCCUPANT_ID, str(self.root / "other.safetensors"))
        self.catalog = [self.model, self.other]
        R.render_llamaswap_config(self.catalog, self.config_path, SERVER, 11436, 300, {}, {})

    def _ensure(self, gpu_set=None):
        R.ensure_internal_route_in_llamaswap_config(
            self.model,
            TARGET_ID,
            self.catalog,
            self.config_path,
            SERVER,
            300,
            {},
            gpu_set=gpu_set,
        )

    def _read(self):
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

    def test_a_corrupt_config_is_never_overwritten(self):
        """A failed read used to become ``{}`` and publish a one-route catalog that
        the manager cannot repair, because the full render only runs at startup."""
        cases = {
            "truncated_yaml": "models:\n  a: [1, 2\n",
            "a_yaml_list": "- one\n- two\n",
            "an_empty_file": "",
            "models_not_a_mapping": "models:\n  - one\n",
        }
        for name, body in cases.items():
            with self.subTest(case=name):
                self.config_path.write_text(body, encoding="utf-8")
                before = self.config_path.read_bytes()
                with self.assertRaises(R.LlamaSwapConfigReadError):
                    R._read_config_strict(self.config_path)
                with self.assertRaises(R.LlamaSwapConfigReadError):
                    self._ensure()
                self.assertEqual(self.config_path.read_bytes(), before)

    def test_a_missing_config_reads_as_empty_rather_than_raising(self):
        self.assertEqual(R._read_config_strict(self.root / "absent.yaml"), {})

    def test_a_gpu_only_change_reaches_the_file(self):
        """The old early-exit compared the projector state alone, so a CVD-only
        rewrite of a route was silently dropped."""
        with mock.patch.object(
            R, "cached_model_gpu_sets", return_value={TARGET_ID: [1], OCCUPANT_ID: [1]}
        ):
            R.render_llamaswap_config(self.catalog, self.config_path, SERVER, 11436, 300, {}, {})
            self.assertIn("CUDA_VISIBLE_DEVICES=1", self._read()["models"][TARGET_ID]["cmd"])
            self._ensure(gpu_set=[0])
            after = self._read()["models"][TARGET_ID]["cmd"]
        self.assertIn("CUDA_VISIBLE_DEVICES=0", after)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=1", after)

    def test_a_base_route_can_be_moved_to_another_card(self):
        with mock.patch.object(R, "cached_model_gpu_sets", return_value={TARGET_ID: [1]}):
            self._ensure(gpu_set=[0])
        cmd = self._read()["models"][TARGET_ID]["cmd"]
        self.assertIn("CUDA_VISIBLE_DEVICES=0", cmd)

    def test_the_moved_pair_becomes_co_loadable(self):
        with mock.patch.object(
            R,
            "cached_model_gpu_sets",
            return_value={TARGET_ID: [1], OCCUPANT_ID: [1]},
        ):
            self._ensure(gpu_set=[0])
        membership = _cli_impl._matrix_group_membership(self.config_path)
        self.assertTrue(
            set(membership.get(TARGET_ID) or ()) & set(membership.get(OCCUPANT_ID) or ()),
            f"a model sharing a set name is co-loadable; got {membership}",
        )

    def test_concurrent_writers_never_shrink_or_corrupt_the_model_set(self):
        errors = []
        barrier = threading.Barrier(2)

        def _hammer(fn):
            try:
                barrier.wait()
                for _ in range(50):
                    fn()
            except Exception as exc:  # noqa: BLE001 - surfaced by the assert below
                errors.append(exc)

        def _render():
            R.render_llamaswap_config(
                self.catalog, self.config_path, SERVER, 11436, 300, {}, {}
            )

        threads = [
            threading.Thread(target=_hammer, args=(_render,)),
            threading.Thread(target=_hammer, args=(lambda: self._ensure(gpu_set=[0]),)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        data = self._read()
        self.assertEqual(set(self.catalog_ids()), set(data["models"]))
        self.assertEqual(list(self.root.glob("config.yaml.*.tmp")), [])
        self.assertFalse((self.root / "config.tmp").exists())

    def catalog_ids(self):
        return {model.model_id for model in self.catalog}


class CompletionsAdmissionTest(unittest.TestCase):
    def test_the_completions_route_has_the_same_admission_gate(self):
        """/v1/completions resolves, picks a replica and forwards without ever
        calling _reject_if_gpu_busy, so it can trigger the very eviction the
        planner exists to avoid."""
        source = inspect.getsource(_cli_impl.start_ctx_metadata_server)
        start = source.index("def _proxy_request(self,")
        # It is the last method of the nested Handler, so the body runs to the end.
        body = textwrap.dedent(source[start:])
        self.assertIn("_reject_if_gpu_busy", body)


class PlacementConfigTest(_RouterStateMixin, unittest.TestCase):
    def setUp(self):
        self.reset_placement_state()
        self.args = argparse.Namespace(config="/dev/null")

    def _with_experimental(self, payload):
        return mock.patch.object(
            _cli_impl, "_load_server_config_payload", return_value=payload
        )

    def test_it_defaults_to_off(self):
        with self._with_experimental({}):
            self.assertEqual(
                _cli_impl._resolve_runtime_placement_config(self.args),
                {"enabled": False, "mode": "off"},
            )

    def test_disabling_forces_the_mode_back_to_off(self):
        with self._with_experimental(
            {"experimental": {"runtime_placement": {"enabled": False, "mode": "repin"}}}
        ):
            self.assertFalse(_cli_impl._placement_repin_enabled(self.args))

    def test_an_unknown_mode_is_treated_as_off(self):
        with self._with_experimental(
            {"experimental": {"runtime_placement": {"enabled": True, "mode": "yolo"}}}
        ):
            self.assertFalse(_cli_impl._runtime_placement_enabled(self.args))

    def test_repin_is_only_on_in_repin_mode(self):
        payload = {"experimental": {"runtime_placement": {"enabled": True, "mode": "retire_only"}}}
        with self._with_experimental(payload):
            self.assertTrue(_cli_impl._runtime_placement_enabled(self.args))
            self.assertFalse(_cli_impl._placement_repin_enabled(self.args))
        payload["experimental"]["runtime_placement"]["mode"] = "repin"
        with self._with_experimental(payload):
            self.assertTrue(_cli_impl._placement_repin_enabled(self.args))

    def test_a_missing_server_config_leaves_it_off(self):
        with mock.patch.object(
            _cli_impl, "_load_server_config_payload", side_effect=RuntimeError("boom")
        ):
            self.assertFalse(_cli_impl._runtime_placement_enabled(self.args))


class ConflictScopeTest(unittest.TestCase):
    """The conflict list must only cover the cards the launch will actually use.

    Regression: a base model pinned to GPU 1 was refused with "the GPU is already
    in use: <model on GPU 0>" while its own card sat empty, so two models could
    never be resident at once even with a whole card free.
    """

    @staticmethod
    def _conflict_source() -> str:
        source = inspect.getsource(_cli_impl.get_gpu_conflict_message)
        body = source[source.index("gpu_process_map = get_gpu_process_map()") :]
        return body[: body.index("if not conflicts:")]

    def test_the_per_gpu_map_is_not_replica_only(self):
        self.assertNotIn(
            "if replica_gpu_set is not None:",
            self._conflict_source(),
            "the per-GPU map was gated on replica_gpu_set, so base models saw no GPU "
            "per pid and every live process became a conflict",
        )

    def test_conflicts_are_scoped_to_the_effective_gpu_set(self):
        body = self._conflict_source()
        self.assertIn("conflict_scope = target_gpu_set", body)
        self.assertNotIn(
            "gpus.isdisjoint(set(replica_gpu_set))",
            body,
            "scoping by replica_gpu_set leaves base models unscoped",
        )


if __name__ == "__main__":
    unittest.main()