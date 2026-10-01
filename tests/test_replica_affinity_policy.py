"""Tests for the replica affinity policy primitives.

Two independent failure modes are covered here:

* **Attribution** -- only evidence about the *target process* may demote a
  target. A malformed request, a client disconnect or a transient 5xx must
  leave the record untouched.
* **Ownership** -- a conversation is bound to one target and saturation is
  never a reason to move it. See ``replica_policy`` for the rationale.

The first bug this file pins down is the historical one: ``request_finished``
used to treat any ``ok=False`` as a target fault, so a single aborted request
poisoned a healthy replica for 120s. Because ``select_replica_for_request``
reads ``status != "ready"`` as "stuck", that one abort evacuated every
conversation bound to the replica onto a cold GPU and forced a full ~18 GiB
reload.
"""

import http.client
import json
import os
import socket
import time
import unittest

from llamacpp_stack import _cli_impl as cli
from llamacpp_stack.cli.models import ReplicaRecord
from llamacpp_stack.cli.replica_policy import (
    TRANSFER_REASONS,
    AffinityDecision,
    FaultKind,
    TargetHealth,
    TargetState,
    TransferReason,
    fault_from_exception,
    fault_from_status,
    is_target_fatal,
    normalize_affinity_config,
    probe_target_health,
)


def _remote_disconnected() -> BaseException:
    """Build the exception llama.cpp surfaced when a peer vanished mid-stream.

    ``urllib`` wraps the underlying ``http.client.RemoteDisconnected`` when a
    connection closes without a response, so reproduce the same class name.
    RemoteDisconnected is itself a subclass of ConnectionResetError + BadStatusLine.
    """

    class RemoteDisconnected(http.client.RemoteDisconnected):
        pass

    return RemoteDisconnected("Remote end closed connection without response")


class _RecordFixture(unittest.TestCase):
    """Builds an isolated router state with a single ready replica."""

    TARGET = "repo-q4__replica_0"
    BASE = "repo-q4"

    def setUp(self) -> None:
        self.state = cli.ReplicaRouterState()
        self.state.records[self.TARGET] = ReplicaRecord(
            base_model_id=self.BASE,
            replica_model_id=self.TARGET,
            gpu_set=(1,),
            status="ready",
            pid=4242,
            port=18098,
        )

    def record(self) -> ReplicaRecord:
        return self.state.records[self.TARGET]


class TestFaultClassification(unittest.TestCase):
    """Statuses describe the *request*; only process evidence describes the target."""

    def test_no_status_is_target_fatal(self) -> None:
        for code in (200, 400, 404, 408, 425, 429, 500, 502, 503, 504):
            with self.subTest(code=code):
                self.assertFalse(is_target_fatal(fault_from_status(code)))

    def test_status_buckets(self) -> None:
        self.assertEqual(fault_from_status(400), FaultKind.CLIENT_REQUEST)
        self.assertEqual(fault_from_status(404), FaultKind.CLIENT_REQUEST)
        self.assertEqual(fault_from_status(429), FaultKind.UPSTREAM_TRANSIENT)
        self.assertEqual(fault_from_status(502), FaultKind.UPSTREAM_TRANSIENT)
        self.assertEqual(fault_from_status(503), FaultKind.UPSTREAM_TRANSIENT)
        # No response at all: unknown, therefore retriable rather than fatal.
        self.assertEqual(fault_from_status(None), FaultKind.UPSTREAM_TRANSIENT)

    def test_client_abort_classified_from_exception(self) -> None:
        self.assertEqual(fault_from_exception(_remote_disconnected()), FaultKind.CLIENT_ABORT)
        self.assertEqual(fault_from_exception(ConnectionResetError(104, "Connection reset by peer")), FaultKind.CLIENT_ABORT)
        self.assertEqual(fault_from_exception(BrokenPipeError(32, "Broken pipe")), FaultKind.CLIENT_ABORT)
        self.assertEqual(fault_from_exception(ConnectionAbortedError()), FaultKind.CLIENT_ABORT)

    def test_abort_by_message_when_class_is_generic(self) -> None:
        # Some transports raise a plain OSError/Exception with the same text.
        self.assertEqual(fault_from_exception(OSError("Connection reset by peer")), FaultKind.CLIENT_ABORT)
        self.assertEqual(
            fault_from_exception(RuntimeError("Remote end closed connection without response")),
            FaultKind.CLIENT_ABORT,
        )

    def test_timeouts_and_content_errors(self) -> None:
        self.assertEqual(fault_from_exception(TimeoutError("timed out")), FaultKind.UPSTREAM_TRANSIENT)
        self.assertEqual(fault_from_exception(socket.timeout("timed out")), FaultKind.UPSTREAM_TRANSIENT)
        self.assertEqual(fault_from_exception(json.JSONDecodeError("x", "y", 0)), FaultKind.CONTENT)
        self.assertEqual(fault_from_exception(ValueError("bad json")), FaultKind.CONTENT)

    def test_unknown_labels_are_never_fatal(self) -> None:
        # An unclassified call site must fail *safe*: it may not poison a target.
        self.assertFalse(is_target_fatal(None))
        self.assertFalse(is_target_fatal("some-future-fault"))
        self.assertFalse(is_target_fatal(object()))
        self.assertFalse(is_target_fatal(FaultKind.UPSTREAM_TRANSIENT))
        self.assertFalse(is_target_fatal(FaultKind.CLIENT_ABORT))
        self.assertTrue(is_target_fatal(FaultKind.TARGET_FATAL))
        self.assertTrue(is_target_fatal("target_fatal"))


class TestTransferReasonVocabulary(unittest.TestCase):
    """The enum is the enforcement mechanism: saturation must be inexpressible."""

    def test_no_saturation_or_spillover_reasons_exist(self) -> None:
        names = {member.name.lower() for member in TransferReason}
        names |= {member.value.lower() for member in TransferReason}
        for banned in ("busy", "saturated", "saturation", "spillover", "load_balance", "rebalance"):
            self.assertFalse(
                any(banned in name for name in names),
                f"TransferReason must not encode '{banned}': saturation is never a reason to move a conversation",
            )

    def test_only_hard_reasons_count_as_transfers(self) -> None:
        self.assertEqual(
            set(TRANSFER_REASONS),
            {TransferReason.EVACUATE_TARGET_DEAD, TransferReason.EVACUATE_NO_ROUTE},
        )


class TestTargetHealth(unittest.TestCase):
    def test_saturation_is_admission_not_health(self) -> None:
        busy = TargetHealth(target="t", state=TargetState.READY, in_flight=1, capacity=1)
        self.assertTrue(busy.saturated)
        # Saturated targets are still perfectly servable -- that is the whole point.
        self.assertTrue(busy.servable)
        self.assertEqual(busy.state, TargetState.READY)

    def test_dead_target_is_not_servable(self) -> None:
        dead = TargetHealth(target="t", state=TargetState.DEAD, in_flight=0, confirmed_dead=True)
        self.assertFalse(dead.servable)
        self.assertFalse(dead.saturated)

    def test_blacklisted_target_is_not_servable(self) -> None:
        blocked = TargetHealth(target="t", state=TargetState.SUSPECT, in_flight=0, blacklist_until=time.monotonic() + 60)
        self.assertTrue(blocked.blacklisted)

    def test_expired_blacklist_is_not_blacklisted(self) -> None:
        expired = TargetHealth(target="t", state=TargetState.READY, in_flight=0, blacklist_until=time.monotonic() - 1)
        self.assertFalse(expired.blacklisted)

    def test_suspect_is_servable_until_probed(self) -> None:
        suspect = TargetHealth(target="t", state=TargetState.SUSPECT, in_flight=0)
        self.assertTrue(suspect.servable, "suspicion must never demote; only a probe may")

    def test_describe_is_loggable(self) -> None:
        described = TargetHealth(target="repo-q4__replica_0", state=TargetState.READY, in_flight=2, capacity=1).describe()
        self.assertEqual(described["target"], "repo-q4__replica_0")
        self.assertEqual(described["state"], "ready")
        self.assertTrue(described["saturated"])
        self.assertTrue(described["servable"])


class TestRequestFinishedAttribution(_RecordFixture):
    """`ok=False` alone must not demote a target."""

    def test_client_abort_keeps_target_ready(self) -> None:
        for exc in (
            _remote_disconnected(),
            ConnectionResetError(104, "Connection reset by peer"),
            BrokenPipeError(32, "Broken pipe"),
        ):
            with self.subTest(exc=type(exc).__name__):
                self.state.request_finished(
                    self.TARGET, ok=False, fault=fault_from_exception(exc)
                )
                rec = self.record()
                self.assertEqual(rec.status, "ready")
                self.assertEqual(rec.blacklist_until, 0.0)
                self.assertEqual(rec.fatal_faults, 0)

    def test_replay_of_the_incident_sequence_never_demotes(self) -> None:
        """18:36:55 aborted two upstream calls; the replica stayed healthy.

        Three back-to-back requests must not be able to flip the record out of
        ``ready`` no matter which flavour of failure they hit.
        """
        for fault in (
            fault_from_exception(_remote_disconnected()),
            fault_from_exception(ConnectionResetError(104, "Connection reset by peer")),
            fault_from_status(502),
            fault_from_status(400),
            fault_from_status(429),
            None,
            "totally-unclassified",
        ):
            with self.subTest(fault=fault):
                self.state.request_started(self.TARGET)
                self.state.request_finished(self.TARGET, ok=False, fault=fault)
                self.assertEqual(self.record().status, "ready")
                self.assertEqual(self.record().blacklist_until, 0.0)
                self.assertEqual(self.record().in_flight, 0)

    def test_malformed_request_does_not_blacklist(self) -> None:
        # 400 "messages or prompt is required" is a client bug, not a target bug.
        self.state.request_started(self.TARGET)
        self.state.request_finished(self.TARGET, ok=False, fault=fault_from_status(400))
        self.assertEqual(self.record().status, "ready")

    def test_target_fatal_demotes_and_blacklists(self) -> None:
        self.state.request_started(self.TARGET)
        self.state.request_finished(self.TARGET, ok=False, fault=FaultKind.TARGET_FATAL)
        rec = self.record()
        self.assertEqual(rec.status, "error")
        self.assertGreater(rec.blacklist_until, time.monotonic())
        self.assertEqual(rec.fatal_faults, 1)
        self.assertEqual(rec.in_flight, 0)

    def test_success_is_a_liveness_proof(self) -> None:
        self.record().status = "error"
        self.record().blacklist_until = time.monotonic() + 120.0
        self.record().fatal_faults = 3
        self.state.request_finished(self.TARGET, ok=True)
        rec = self.record()
        self.assertEqual(rec.status, "ready")
        self.assertEqual(rec.blacklist_until, 0.0)
        self.assertEqual(rec.fatal_faults, 0)

    def test_slot_is_always_released(self) -> None:
        for _ in range(3):
            self.state.request_started(self.TARGET)
        self.assertEqual(self.record().in_flight, 3)
        self.state.request_finished(self.TARGET, ok=False, fault=FaultKind.CLIENT_ABORT)
        self.assertEqual(self.record().in_flight, 2)

    def test_slot_never_goes_negative(self) -> None:
        self.state.request_finished(self.TARGET, ok=False, fault=FaultKind.CLIENT_ABORT)
        self.assertEqual(self.record().in_flight, 0)

    def test_base_in_flight_path_is_unchanged(self) -> None:
        self.state.request_started(self.BASE)
        self.assertEqual(self.state.base_load(self.BASE), 1)
        self.state.request_finished(self.BASE, ok=False, fault=FaultKind.CLIENT_ABORT)
        self.assertEqual(self.state.base_load(self.BASE), 0)
        # An untracked id must never raise.
        self.state.request_finished("never-seen", ok=False, fault=FaultKind.TARGET_FATAL)


class TestSaturatedOwnerDoesNotEvacuate(_RecordFixture):
    """The regression that caused four GPU flips per hour.

    ``_find_affinity_spillover_candidate`` treated ``in_flight >= 1`` as a reason
    to move a bound conversation to another GPU. With ``--parallel 1`` any
    *other* client loading the owner moved this conversation, which then had to
    pay a cold reload (~5.5 min measured). Commit 1 removes the only trigger
    that could fire here: saturation is not a fault, so it never marks the
    target unusable.
    """

    def test_foreign_load_does_not_demote_the_owner(self) -> None:
        # A third party occupies the owner (already warm, so status stays "ready").
        self.state.request_started(self.TARGET)
        self.assertEqual(self.record().in_flight, 1)

        # Our conversation's own request completes as a client abort.
        self.state.request_finished(self.TARGET, ok=False, fault=FaultKind.CLIENT_ABORT)

        rec = self.record()
        self.assertEqual(rec.status, "ready", "a client abort must not rewrite a healthy target's state")
        self.assertEqual(rec.blacklist_until, 0.0)


def _always(status):
    return lambda port, timeout_s: status


class PolicyOwnershipTest(unittest.TestCase):
    """The chokepoint must make "move because busy" inexpressible."""

    def setUp(self) -> None:
        self.state = cli.ReplicaRouterState()
        self.cfg = normalize_affinity_config(None)
        self.key = "qwen3.8-27b-EXL3:fallback:653536f123fc:aef2512b8a6b"
        self.base = "qwen3.8-27b-EXL3"
        self.replica = "qwen3.8-27b-EXL3__replica_0"

    def bind_initial(self, target, now=1000.0):
        return self.state.bind(self.key, target, TransferReason.INITIAL_BIND, ttl_s=3600.0, cfg=self.cfg, now=now)

    def test_initial_bind_records_ownership(self) -> None:
        self.bind_initial(self.replica)
        self.assertEqual(self.state.owner_of(self.key, now=1000.0), self.replica)

    def test_sticky_refresh_never_changes_owner(self) -> None:
        self.bind_initial(self.replica)
        for step in range(5):
            self.state.request_started(self.replica)
            result = self.state.bind(
                self.key, self.base, TransferReason.STICKY_REFRESH, ttl_s=3600.0, cfg=self.cfg, now=1000.0 + step
            )
            self.assertEqual(result.target, self.replica, "a refresh must keep the owner")
            self.assertEqual(self.state.owner_of(self.key, now=1000.0), self.replica)
            self.state.request_finished(self.replica)

    def test_saturated_owner_waits_instead_of_moving(self) -> None:
        self.bind_initial(self.replica)
        decision, reason = self.state.policy.decide(
            self.replica,
            TargetHealth(target=self.replica, state=TargetState.READY, in_flight=1, capacity=1),
            has_alternatives=True,
            cfg=self.cfg,
        )
        self.assertIs(decision, AffinityDecision.WAIT)
        self.assertIsNone(reason, "a saturated owner has no transfer reason")

    def test_saturation_can_never_be_expressed_as_a_transfer(self) -> None:
        self.bind_initial(self.replica)
        for reason in (TransferReason.STICKY_REFRESH, TransferReason.RESPONSES_CHAIN, TransferReason.INITIAL_BIND):
            result = self.state.bind(self.key, self.base, reason, ttl_s=3600.0, cfg=self.cfg, now=1001.0)
            self.assertEqual(result.target, self.replica, f"{reason} must not move ownership")
            self.assertEqual(result.refused_by, "undeclared_reason")
        self.assertEqual(self.state.owner_of(self.key, now=1000.0), self.replica)

    def test_suspect_target_is_not_evacuated_by_default(self) -> None:
        self.bind_initial(self.replica)
        health = TargetHealth(target=self.replica, state=TargetState.ERROR, blacklist_until=time.monotonic() + 120)
        decision, _ = self.state.policy.decide(self.replica, health, has_alternatives=True, cfg=self.cfg)
        self.assertIs(decision, AffinityDecision.STICKY)

    def test_min_dwell_brake_blocks_suspected_transfers(self) -> None:
        self.bind_initial(self.replica, now=1000.0)
        result = self.state.bind(
            self.key, self.base, TransferReason.EVACUATE_NO_ROUTE, ttl_s=3600.0, cfg=self.cfg, now=1010.0
        )
        self.assertEqual(result.target, self.replica, "dwell brake should hold the owner")
        self.assertEqual(result.refused_by, "min_dwell_s")

    def test_confirmed_death_ignores_dwell_and_moves(self) -> None:
        self.bind_initial(self.replica, now=1000.0)
        result = self.state.bind(
            self.key, self.base, TransferReason.EVACUATE_TARGET_DEAD, ttl_s=3600.0, cfg=self.cfg, now=1001.0
        )
        self.assertEqual(result.target, self.base)
        self.assertEqual(self.state.owner_of(self.key, now=1000.0), self.base)

    def test_transfer_budget_is_per_binding_not_per_target(self) -> None:
        """A ping-pong between two targets must not reset the budget."""
        self.bind_initial(self.replica, now=1000.0)
        cfg = normalize_affinity_config({"min_dwell_s": 0, "max_transfers": 1, "evacuate_cooldown_s": 0})
        first = self.state.bind(self.key, self.base, TransferReason.EVACUATE_NO_ROUTE, ttl_s=3600.0, cfg=cfg, now=2000.0)
        self.assertEqual(first.target, self.base)
        second = self.state.bind(self.key, self.replica, TransferReason.EVACUATE_NO_ROUTE, ttl_s=3600.0, cfg=cfg, now=3000.0)
        self.assertEqual(second.target, self.base, "budget exhaustion must stop the ping-pong")
        self.assertEqual(second.refused_by, "max_transfers")

    def test_evacuate_cooldown_brake_blocks_returning_to_previous_target(self) -> None:
        self.bind_initial(self.replica, now=1000.0)
        cfg = normalize_affinity_config({"min_dwell_s": 0, "evacuate_cooldown_s": 600})
        self.state.bind(self.key, self.base, TransferReason.EVACUATE_NO_ROUTE, ttl_s=3600.0, cfg=cfg, now=2000.0)
        back = self.state.bind(self.key, self.replica, TransferReason.EVACUATE_NO_ROUTE, ttl_s=3600.0, cfg=cfg, now=2100.0)
        self.assertEqual(back.target, self.base)
        self.assertEqual(back.refused_by, "evacuate_cooldown_s")

    def test_exhausted_hard_budget_degrades_instead_of_deadlocking(self) -> None:
        self.bind_initial(self.replica, now=1000.0)
        cfg = normalize_affinity_config({"max_hard_transfers": 1})
        first = self.state.bind(self.key, self.base, TransferReason.EVACUATE_TARGET_DEAD, ttl_s=3600.0, cfg=cfg, now=1001.0)
        self.assertEqual(first.target, self.base)
        dead_again = self.state.bind(self.key, self.replica, TransferReason.EVACUATE_TARGET_DEAD, ttl_s=3600.0, cfg=cfg, now=1002.0)
        self.assertIsNone(dead_again.target, "must serve unbound rather than pin to a dead target")
        self.assertIs(dead_again.decision, AffinityDecision.UNBOUND)
        self.assertIsNone(self.state.owner_of(self.key, now=1000.0))


class ProbeTest(unittest.TestCase):
    def test_dead_pid_is_proof_of_death(self) -> None:
        health = probe_target_health("t", pid=2**22 - 1, port=0, probe=_always(200))
        self.assertTrue(health.confirmed_dead)
        self.assertEqual(health.detail, "pid_not_alive")
        self.assertFalse(health.servable)

    def test_unreachable_port_is_proof_of_death(self) -> None:
        health = probe_target_health("t", pid=0, port=1, probe=_always(None))
        self.assertTrue(health.confirmed_dead)
        self.assertEqual(health.detail, "health_unreachable")

    def test_503_means_loading_not_broken(self) -> None:
        health = probe_target_health("t", pid=0, port=1, probe=_always(503))
        self.assertIs(health.state, TargetState.LOADING)
        self.assertFalse(health.confirmed_dead)
        self.assertTrue(health.servable, "a loading target is still the right owner")

    def test_200_is_ready(self) -> None:
        health = probe_target_health("t", pid=0, port=1, probe=_always(200))
        self.assertIs(health.state, TargetState.READY)
        self.assertFalse(health.saturated)

    def test_odd_status_is_suspect_not_dead(self) -> None:
        health = probe_target_health("t", pid=0, port=1, probe=_always(418))
        self.assertIs(health.state, TargetState.SUSPECT)
        self.assertFalse(health.confirmed_dead)
        self.assertTrue(health.servable, "only a probe may convict; suspicion must not evict")

    def test_saturation_is_reported_separately_from_health(self) -> None:
        health = probe_target_health("t", pid=0, port=1, in_flight=1, capacity=1, probe=_always(200))
        self.assertTrue(health.saturated)
        self.assertTrue(health.servable)
        self.assertFalse(health.suspected)


class IncidentReplayTest(unittest.TestCase):
    """Deterministic replay of the 18:36 window that caused the flips.

    The owner replica is warm on GPU 1 and the conversation is bound to it. Two
    client aborts land, a third party keeps the owner loaded, and three
    sequential requests arrive. Under the old code the aborts marked the replica
    ``error`` for 120s, which read as "stuck", which evicted every bound
    conversation onto the cold base. Here nothing may move.
    """

    BASE = "qwen3.8-27b-EXL3"
    REPLICA = "qwen3.8-27b-EXL3__replica_0"
    KEY = "qwen3.8-27b-EXL3:fallback:653536f123fc:aef2512b8a6b"

    def setUp(self) -> None:
        self.state = cli.ReplicaRouterState()
        self.cfg = normalize_affinity_config(None)
        self.state.records[self.REPLICA] = ReplicaRecord(
            base_model_id=self.BASE,
            replica_model_id=self.REPLICA,
            gpu_set=[1],
            status="ready",
            pid=os.getpid(),
            port=1,
        )

    @staticmethod
    def alive_probe(port: int, timeout_s: float) -> int:
        return 200

    def test_the_incident_window_produces_zero_transfers(self) -> None:
        self.state.bind(self.KEY, self.REPLICA, TransferReason.INITIAL_BIND, ttl_s=3600.0, cfg=self.cfg, now=1000.0)
        for _ in range(2):
            self.state.request_started(self.REPLICA)
            self.state.request_finished(self.REPLICA, ok=False, fault=fault_from_exception(_remote_disconnected()))
        for _ in range(3):
            health = self.state.health_of(self.REPLICA, cfg=self.cfg, probe=self.alive_probe)
            decision, reason = self.state.policy.decide(
                self.state.owner_of(self.KEY), health, has_alternatives=True, cfg=self.cfg
            )
            target = self.state.bind(
                self.KEY,
                self.REPLICA if decision is not AffinityDecision.EVACUATE else self.BASE,
                reason or TransferReason.STICKY_REFRESH,
                ttl_s=3600.0,
                cfg=self.cfg,
                now=1001.0,
            ).target
            self.assertEqual(target, self.REPLICA, "a healthy owner must never be abandoned")
        self.assertEqual(self.state.records[self.REPLICA].status, "ready")
        self.assertEqual(self.state.records[self.REPLICA].blacklist_until, 0.0)

    def test_only_confirmed_death_moves_the_conversation(self) -> None:
        self.state.bind(self.KEY, self.REPLICA, TransferReason.INITIAL_BIND, ttl_s=3600.0, cfg=self.cfg, now=1000.0)
        self.state.records[self.REPLICA].pid = 2**22 - 1
        health = self.state.health_of(self.REPLICA, cfg=self.cfg, probe=self.alive_probe)
        self.assertTrue(health.confirmed_dead)
        decision, reason = self.state.policy.decide(self.REPLICA, health, has_alternatives=True, cfg=self.cfg)
        self.assertIs(decision, AffinityDecision.EVACUATE)
        self.assertIs(reason, TransferReason.EVACUATE_TARGET_DEAD)

    def test_health_probe_is_rate_limited(self) -> None:
        calls: list[int] = []

        def counting_probe(port: int, timeout_s: float) -> int:
            calls.append(port)
            return 200

        self.state.health_of(self.REPLICA, cfg=self.cfg, probe=counting_probe)
        second = self.state.health_of(self.REPLICA, cfg=self.cfg, probe=counting_probe)
        self.assertIs(second.state, TargetState.READY)
        self.assertEqual(len(calls), 1, "a warm owner must not be probed on every request")


class AffinityConfigNormalisationTest(unittest.TestCase):
    def test_garbage_falls_back_to_safe_defaults(self) -> None:
        cfg = normalize_affinity_config(
            {"min_dwell_s": "nonsense", "max_transfers": None, "saturated_target": "chaos"}
        )
        self.assertEqual(cfg.min_dwell_s, normalize_affinity_config(None).min_dwell_s)
        self.assertEqual(cfg.saturated_target, "retry")
        self.assertGreaterEqual(cfg.max_transfers, 0)

    def test_negative_values_are_clamped(self) -> None:
        cfg = normalize_affinity_config({"min_dwell_s": -99, "max_hard_transfers": 0, "queue_max_wait_ms": -5})
        self.assertEqual(cfg.min_dwell_s, 0.0)
        self.assertEqual(cfg.max_hard_transfers, 1, "there must always be at least one escape from a dead target")
        self.assertEqual(cfg.queue_max_wait_ms, 0)

    def test_queue_mode_is_accepted(self) -> None:
        self.assertEqual(normalize_affinity_config({"saturated_target": "queue"}).saturated_target, "queue")


if __name__ == "__main__":
    unittest.main()