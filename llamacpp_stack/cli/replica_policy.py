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
import os
import time
from collections.abc import Callable
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
    def suspected(self) -> bool:
        """Believed unhealthy, but not proven.  Never grounds for eviction alone."""
        return self.blacklisted or self.state in {TargetState.SUSPECT, TargetState.ERROR}

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
            "suspected": self.suspected,
            "confirmed_dead": self.confirmed_dead,
            "detail": self.detail,
        }


class AffinityDecision(str, enum.Enum):
    """The single answer to "where does this conversation go?".

    Ordered and mutually exclusive; ``decide()`` returns the first match. There
    is deliberately no decision that means "move it somewhere else because that
    would be more convenient".
    """

    #: No owner yet: pick freely, then record ownership.
    BIND = "bind"
    #: Serve on the owner.  Never writes a new target.
    STICKY = "sticky"
    #: Owner is healthy but full: wait (queue) or ask the client to retry.
    WAIT = "wait"
    #: Owner is impossible or provably dead: move, subject to the brakes.
    EVACUATE = "evacuate"
    #: Brakes exhausted: serve wherever, without recording ownership.
    UNBOUND = "unbound"


@dataclass
class AffinityConfig:
    """Tunables for conversation ownership (``experimental.affinity_policy``)."""

    #: Minimum dwell on a target before an *unproven* transfer may move it.
    min_dwell_s: float = 120.0
    #: Unproven transfers allowed per binding before degrading to UNBOUND.
    max_transfers: int = 2
    #: Proven-death transfers allowed per binding.  Higher than the unproven cap
    #: on purpose: a dead target must always be escapable.
    max_hard_transfers: int = 5
    #: Refuse to re-bind to a target we just evacuated away from.
    evacuate_cooldown_s: float = 300.0
    #: ``retry`` -> 429 + Retry-After.  ``queue`` -> wait up to queue_max_wait_ms.
    saturated_target: str = "retry"
    queue_max_wait_ms: int = 0
    queue_max_depth: int = 0
    #: Off by default.  When true, an *unproven* suspect target may still be
    #: left behind (the legacy ``affinity_spillover.enabled`` behaviour).
    allow_suspected_transfers: bool = False
    #: Liveness probe cadence per target and per-attempt timeout.
    probe_interval_s: float = 10.0
    probe_timeout_s: float = 1.5


def normalize_affinity_config(raw: object) -> AffinityConfig:
    """Build an :class:`AffinityConfig` from a possibly-wrong config blob."""
    cfg = AffinityConfig()
    if not isinstance(raw, dict):
        return cfg

    def _num(key: str, default: float, minimum: float = 0.0) -> float:
        try:
            return max(minimum, float(raw.get(key, default)))
        except (TypeError, ValueError):
            return default

    def _int(key: str, default: int, minimum: int = 0) -> int:
        try:
            return max(minimum, int(raw.get(key, default)))
        except (TypeError, ValueError):
            return default

    cfg.min_dwell_s = _num("min_dwell_s", cfg.min_dwell_s)
    cfg.max_transfers = _int("max_transfers", cfg.max_transfers)
    cfg.max_hard_transfers = _int("max_hard_transfers", cfg.max_hard_transfers, minimum=1)
    cfg.evacuate_cooldown_s = _num("evacuate_cooldown_s", cfg.evacuate_cooldown_s)
    mode = str(raw.get("saturated_target", cfg.saturated_target) or "").strip().lower()
    cfg.saturated_target = mode if mode in {"retry", "queue"} else "retry"
    cfg.queue_max_wait_ms = _int("queue_max_wait_ms", cfg.queue_max_wait_ms)
    cfg.queue_max_depth = _int("queue_max_depth", cfg.queue_max_depth)
    cfg.allow_suspected_transfers = bool(raw.get("allow_suspected_transfers", cfg.allow_suspected_transfers))
    cfg.probe_interval_s = _num("probe_interval_s", cfg.probe_interval_s, minimum=0.1)
    cfg.probe_timeout_s = _num("probe_timeout_s", cfg.probe_timeout_s, minimum=0.05)
    return cfg


@dataclass
class TransferBudget:
    """Per-binding transfer accounting, keyed by affinity key.

    The counters live on the *binding*, not on the target, so a ping-pong
    between two targets cannot reset the budget by alternating.
    """

    bound_at: float = 0.0
    last_transfer_at: float = 0.0
    last_target: str = ""
    suspected_transfers: int = 0
    hard_transfers: int = 0


@dataclass
class BindResult:
    """Outcome of an affinity write attempt."""

    target: str | None
    decision: AffinityDecision
    reason: TransferReason
    bound: bool
    #: Name of the brake that refused a transfer, for logging.
    refused_by: str = ""


class AffinityPolicy:
    """Decides conversation ownership; holds no mutable state of its own.

    All branching lives here so the routing code can only express *what it
    observed*, never invent a new reason to move a conversation.
    """

    def decide(
        self,
        bound_target: str | None,
        health: TargetHealth,
        *,
        has_alternatives: bool,
        cfg: AffinityConfig,
    ) -> tuple[AffinityDecision, TransferReason | None]:
        """Classify a request against its current owner. First match wins."""
        if not bound_target:
            return AffinityDecision.BIND, TransferReason.INITIAL_BIND
        if health.confirmed_dead:
            return AffinityDecision.EVACUATE, TransferReason.EVACUATE_TARGET_DEAD
        if not health.servable:
            return AffinityDecision.EVACUATE, TransferReason.EVACUATE_NO_ROUTE
        if health.suspected:
            # Unproven. A probe decides, not a timer and not a busy counter.
            if has_alternatives and cfg.allow_suspected_transfers:
                return AffinityDecision.EVACUATE, TransferReason.EVACUATE_NO_ROUTE
            return AffinityDecision.STICKY, None
        if health.saturated:
            return AffinityDecision.WAIT, None
        return AffinityDecision.STICKY, None

    def allow_transfer(
        self,
        budget: TransferBudget,
        reason: TransferReason,
        new_target: str,
        cfg: AffinityConfig,
        now: float,
    ) -> tuple[bool, str]:
        """Apply the anti-thrash brakes. Returns ``(allowed, brake_name)``."""
        hard = reason is TransferReason.EVACUATE_TARGET_DEAD
        if hard:
            # Death overrides dwell and cooldown: an unusable target must
            # always be escapable, so only the generous hard budget applies.
            return (budget.hard_transfers < cfg.max_hard_transfers, "max_hard_transfers")
        if now - budget.bound_at < cfg.min_dwell_s:
            return False, "min_dwell_s"
        if budget.suspected_transfers >= cfg.max_transfers:
            return False, "max_transfers"
        if (
            budget.last_target
            and new_target == budget.last_target
            and now - budget.last_transfer_at < cfg.evacuate_cooldown_s
        ):
            return False, "evacuate_cooldown_s"
        return True, ""

    def record_transfer(self, budget: TransferBudget, reason: TransferReason, previous: str, now: float) -> None:
        budget.last_target = previous
        budget.last_transfer_at = now
        if reason is TransferReason.EVACUATE_TARGET_DEAD:
            budget.hard_transfers += 1
        else:
            budget.suspected_transfers += 1


def _default_http_probe(port: int, timeout_s: float) -> int | None:
    """GET /health on *port*; returns the status code or None if unreachable."""
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", int(port), timeout=float(timeout_s))
    try:
        conn.request("GET", "/health")
        response = conn.getresponse()
        response.read()
        return int(response.status)
    except Exception:
        return None
    finally:
        conn.close()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def probe_target_health(
    target: str,
    *,
    pid: int = 0,
    port: int = 0,
    state: TargetState = TargetState.READY,
    in_flight: int = 0,
    capacity: int = 1,
    blacklist_until: float = 0.0,
    probe: Callable[[int, float], int | None] | None = None,
    timeout_s: float = 1.5,
) -> TargetHealth:
    """Measure a target instead of inferring its health from a request outcome.

    A dead pid or an unreachable port is proof of death.  HTTP 503 from
    llama-server means "still loading", which is not a fault.  Anything else is
    reported as :attr:`TargetState.SUSPECT` so the caller decides, never the
    transport.
    """
    probe = probe or _default_http_probe
    health = TargetHealth(
        target=target,
        state=state,
        in_flight=in_flight,
        capacity=capacity,
        blacklist_until=blacklist_until,
    )
    if pid > 0 and not _pid_alive(pid):
        health.state = TargetState.DEAD
        health.confirmed_dead = True
        health.detail = "pid_not_alive"
        return health
    if port > 0:
        status = probe(int(port), float(timeout_s))
        if status is None:
            health.state = TargetState.DEAD
            health.confirmed_dead = True
            health.detail = "health_unreachable"
        elif status == 200:
            health.state = TargetState.READY
        elif status == 503:
            # llama-server answers 503 while the weights are still loading.
            health.state = TargetState.LOADING
        else:
            health.state = TargetState.SUSPECT
            health.detail = f"health_status_{status}"
    return health


__all__ = [
    "AffinityConfig",
    "AffinityDecision",
    "AffinityPolicy",
    "BindResult",
    "FaultKind",
    "TARGET_FATAL_KINDS",
    "TargetHealth",
    "TargetState",
    "TRANSFER_REASONS",
    "TransferBudget",
    "TransferReason",
    "fault_from_exception",
    "fault_from_status",
    "is_target_fatal",
    "normalize_affinity_config",
    "probe_target_health",
]
