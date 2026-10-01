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
import socket
import time
import unittest

from llamacpp_stack import _cli_impl as cli
from llamacpp_stack.cli.models import ReplicaRecord
from llamacpp_stack.cli.replica_policy import (
    TRANSFER_REASONS,
    FaultKind,
    TargetHealth,
    TargetState,
    TransferReason,
    fault_from_exception,
    fault_from_status,
    is_target_fatal,
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


if __name__ == "__main__":
    unittest.main()