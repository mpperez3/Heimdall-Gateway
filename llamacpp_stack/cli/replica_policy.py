"""Replica routing policy: fault attribution and conversation affinity.

Two concerns live here and must not be mixed again:

* **Fault attribution** -- deciding whether a failed request says anything
  about the *health of the upstream target* or only about the request.
* **Affinity ownership** -- deciding which upstream target a conversation
  belongs to, and when (if ever) that ownership may move.

The historical bug this module exists to prevent: ``request_finished`` treated
every non-ok exit path as a target fault, so a malformed client request or a
client-side disconnect marked a healthy replica ``error`` + blacklisted it for
120s.  The replica router reads ``status != "ready"`` as "stuck" and evacuates
every conversation bound to it, which moved a conversation to another GPU and
forced a full ~18 GiB reload (~5.5 min observed) for a fault that had never
touched the target at all.

Design rules encoded here:

1. Only :attr:`FaultKind.TARGET_FATAL` may demote a target.  Everything else
   releases the slot and leaves target state alone.
2. No HTTP status or transport exception maps to ``TARGET_FATAL``.  Statuses
   and transport errors describe *the request*; only process evidence (pid
   liveness, health probe) describes *the target*.
3. Saturation is never a reason to move a conversation.  A warm target that
   makes you wait is cheaper than a cold target that reloads.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass


class FaultKind(str, enum.Enum):
    """Why a request did not complete, attributed to the right layer."""

    NONE = "none"
    #: The client hung up, cancelled, or we could not write the response.
    CLIENT_ABORT = "client_abort"
    #: Malformed/invalid client input (4xx, validation failure).
    CLIENT_REQUEST = "client_request"
    #: Upstream returned something we could not use (bad JSON, tool repair).
    CONTENT = "content"
    #: 5xx, timeout, connection reset/refused -- retriable, not evidence of death.
    UPSTREAM_TRANSIENT = "upstream_transient"
    #: The target process itself is unusable.  Requires process evidence.
    TARGET_FATAL = "target_fatal"


#: The only fault kind allowed to demote a target to ``error`` + blacklist.
TARGET_FATAL_KINDS = frozenset({FaultKind.TARGET_FATAL})


def is_target_fatal(fault: FaultKind | str | None) -> bool:
    """True only for faults that legitimately say the *target* is unusable."""
    if fault is None:
        return False
    if isinstance(fault, FaultKind):
        return fault in TARGET_FATAL_KINDS
    try:
        return FaultKind(str(fault)) in TARGET_FATAL_KINDS
    except ValueError:
        # Unknown fault labels must never be able to blacklist a target.
        return False


def fault_from_status(status_code: int | None) -> FaultKind:
    """Attribute an upstream HTTP status to a layer.

    Deliberately conservative: no status yields ``TARGET_FATAL``.  A 502 or 503
    is a *symptom*; whether the process died is decided by a liveness probe, not
    by the status line.  Mapping 5xx to a target fault is what turned a single
    transient upstream error into a 120s blacklist.
    """
    code = int(status_code or 0)
    if code <= 0:
        return FaultKind.UPSTREAM_TRANSIENT
    if code < 400:
        return FaultKind.NONE
    if code in (408, 425, 429):
        return FaultKind.UPSTREAM_TRANSIENT
    if code < 500:
        return FaultKind.CLIENT_REQUEST
    return FaultKind.UPSTREAM_TRANSIENT


#: Transport exception type names that mean *we* lost the client, not the target.
_CLIENT_GONE_EXC_NAMES = frozenset(
    {
        "BrokenPipeError",
        "ConnectionResetError",
        "ConnectionAbortedError",
    }
)

_CLIENT_GONE_MESSAGE_HINTS = (
    "broken pipe",
    "connection reset by peer",
    "connection aborted",
    "remote end closed connection without response",
    "remotedisconnected",
    "client disconnected",
    "connection closed by peer",
)


def fault_from_exception(exc: BaseException | None) -> FaultKind:
    """Attribute a transport/protocol exception to a layer.

    ``RemoteDisconnected`` and ``ConnectionResetError`` are inherently ambiguous
    -- the upstream may have died, or the client may have hung up mid-stream.
    They are attributed to the *request*, never to the target, and the target's
    fate is settled by a liveness probe.  This is the single most important
    classification in this module: in the incident that motivated it, both
    observed errors were client aborts against a perfectly healthy replica.
    """
    if exc is None:
        return FaultKind.NONE
    name = type(exc).__name__
    if name in _CLIENT_GONE_EXC_NAMES:
        return FaultKind.CLIENT_ABORT
    text = str(exc).lower()
    if any(hint in text for hint in _CLIENT_GONE_MESSAGE_HINTS):
        return FaultKind.CLIENT_ABORT
    if name in {"Timeout", "ReadTimeout", "ConnectTimeout", "TimeoutException"}:
        return FaultKind.UPSTREAM_TRANSIENT
    if name in {"JSONDecodeError", "ValueError"}:
        return FaultKind.CONTENT
    # ConnectionError / SSLError / OSError and anything unrecognised: retriable.
    return FaultKind.UPSTREAM_TRANSIENT


class TransferReason(str, enum.Enum):
    """Closed set of reasons an affinity binding may be (re)written.

    Every write to the affinity map must name one of these.  The enum is the
    guardrail that keeps a future routing branch from silently moving a
    conversation: there is deliberately no ``busy``/``saturation``/
    ``spillover`` member, so "move because the target is busy" is not
    expressible, and a test asserts the absence of those names.
    """

    #: No previous binding (new conversation) or binding expired.
    INITIAL_BIND = "initial_bind"
    #: TTL refresh of an unchanged target.  Never changes ownership.
    STICKY_REFRESH = "sticky_refresh"
    #: Client sent previous_response_id pointing at a specific replica.
    RESPONSES_CHAIN = "responses_chain"
    #: Target liveness probe confirmed the process is gone.
    EVACUATE_TARGET_DEAD = "evacuate_target_dead"
    #: No route remains that can serve the request at all.
    EVACUATE_NO_ROUTE = "evacuate_no_route"
    #: Affinity disabled or unusable; serve without binding.
    DEGRADED_UNBOUND = "degraded_unbound"


#: Reasons that represent an ownership transfer (as opposed to a bind/refresh).
TRANSFER_REASONS = frozenset(
    {
        TransferReason.EVACUATE_TARGET_DEAD,
        TransferReason.EVACUATE_NO_ROUTE,
    }
)


class TargetState(str, enum.Enum):
    """Coarse target health, from measured process evidence."""

    #: No process yet; a request here will trigger a load.
    COLD = "cold"
    #: Process present and serving.
    READY = "ready"
    #: Load in progress.
    LOADING = "loading"
    #: Process confirmed gone (pid dead / health probe failed).
    DEAD = "dead"
    #: Believed unhealthy but not confirmed; requires a probe to demote.
    SUSPECT = "suspect"
    #: A fatal fault was reported; needs a probe to confirm.
    ERROR = "error"


@dataclass
class TargetHealth:
    """Measured health of one upstream target (base or replica).

    ``saturated`` is deliberately *not* part of health: saturation is an
    admission-control fact, and mixing it into health is what allowed a busy
    target to be treated as broken.
    """

    target: str
    state: TargetState = TargetState.COLD
    in_flight: int = 0
    capacity: int = 1
    #: True when process evidence (pid + health probe) confirms the target is gone.
    confirmed_dead: bool = False
    blacklist_until: float = 0.0
    detail: str = ""

    @property
    def servable(self) -> bool:
        """Can this target take the request (now or after a load)?

        Note that ``SUSPECT`` and ``ERROR`` are servable on purpose: they are
        unconfirmed suspicions, and only a liveness probe may turn them into
        ``DEAD``.  Demoting on suspicion is exactly what caused the incident.
        """
        if self.confirmed_dead or self.state is TargetState.DEAD:
            return False
        return self.state in {
            TargetState.COLD,
            TargetState.READY,
            TargetState.LOADING,
            TargetState.SUSPECT,
            TargetState.ERROR,
        }

    @property
    def blacklisted(self) -> bool:
        """Inside an active quarantine window.  Not servable until it expires."""
        return self.blacklist_until > time.monotonic()

    @property
    def saturated(self) -> bool:
        """At capacity.  Never a reason to move a conversation."""
        if self.capacity <= 0:
            return False
        return self.in_flight >= self.capacity

    def describe(self) -> dict[str, object]:
        return {
            "target": self.target,
            "state": self.state.value,
            "in_flight": self.in_flight,
            "capacity": self.capacity,
            "saturated": self.saturated,
            "servable": self.servable,
            "blacklisted": self.blacklisted,
            "confirmed_dead": self.confirmed_dead,
            "detail": self.detail,
        }


__all__ = [
    "FaultKind",
    "TARGET_FATAL_KINDS",
    "is_target_fatal",
    "fault_from_status",
    "fault_from_exception",
    "TransferReason",
    "TRANSFER_REASONS",
    "TargetState",
    "TargetHealth",
]
