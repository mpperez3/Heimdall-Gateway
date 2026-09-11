"""llamacpp_stack.dedup — per-process singleflight + TTL cache + SSE tee-broadcast.

Scope: only :11435 POST dedup (GuardHandler/:11436 out of scope).
Fingerprint: SHA256(METHOD\\nPATH_sin_query\\nprincipal_hash\\nstream_flag\\nsha256(canonical)).
Buckets split on stream:true vs false. Only 2xx cached. Sync ThreadingHTTPServer
with RLock; lock never held during upstream I/O. Stdlib only.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

_CANONICAL_FIELDS: tuple[str, ...] = (
    "model", "messages", "prompt", "input", "temperature", "top_p", "top_k",
    "seed", "max_tokens", "stop", "presence_penalty", "frequency_penalty",
    "logit_bias", "response_format", "tools", "tool_choice", "n", "logprobs",
    "stream", "stream_options", "encoding_format", "dimensions",
)


def canonical_body(body: dict | None) -> bytes:
    """Canonical JSON bytes (allowlist, sort_keys, separators, ensure_ascii=False).

    None/non-dict -> b'{}'. Preserves list order. Drops non-allowlisted keys
    (Authorization etc are headers, not in body). Never raises.
    """
    if not isinstance(body, dict):
        return b"{}"
    filtered: dict[str, Any] = {k: body[k] for k in _CANONICAL_FIELDS if k in body}
    try:
        dumped = json.dumps(filtered, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError, OverflowError):
        safe: dict[str, Any] = {}
        for k, v in filtered.items():
            try:
                json.dumps(v, ensure_ascii=False)
                safe[k] = v
            except Exception:
                safe[k] = str(v)
        dumped = json.dumps(safe, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return dumped.encode("utf-8")


def principal_hash_for_api_key(api_key_or_none: str | None) -> str:
    """sha256(api_key) hex or 'anonymous' if empty/None."""
    if not api_key_or_none:
        return "anonymous"
    key = str(api_key_or_none)
    if not key:
        return "anonymous"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def principal_hash(api_key_or_sub: str | None) -> str:
    """Alias for principal_hash_for_api_key — isolates buckets per principal."""
    return principal_hash_for_api_key(api_key_or_sub)


def fingerprint(method: str, path: str, principal_hash_str: str, stream_flag: bool, canonical_bytes: bytes) -> str:
    """SHA256(METHOD\\nPATH_sin_query\\nprincipal_hash\\nstr(stream_flag)\\nsha256(canonical))."""
    m = (method or "POST").upper()
    p = (path or "/").split("?", 1)[0]
    if not isinstance(canonical_bytes, (bytes, bytearray)):
        canonical_bytes = str(canonical_bytes).encode("utf-8")
    inner = hashlib.sha256(bytes(canonical_bytes)).hexdigest()
    outer = f"{m}\n{p}\n{principal_hash_str}\n{str(stream_flag)}\n{inner}"
    return hashlib.sha256(outer.encode("utf-8")).hexdigest()


@dataclass
class Entry:
    """Dedup entry: event signalled on completion; result/exception; TTL."""

    event: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None
    exception: BaseException | None = None
    expires_at: float = 0.0
    stream: bool = False
    principal: str = ""


class DedupState:
    """RLock + dict[hash, Entry] singleflight + TTL cache.

    max_entries fail-open: if full, new hashes bypass dedup (leader with no shared state).
    Lock only covers lookup/insert/eviction, never I/O.
    """

    def __init__(self, max_entries: int = 2000) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, Entry] = {}
        self.max_entries: int = max_entries

    def get_or_create_inflight(self, key: str) -> tuple[bool, threading.Event | None, dict | None]:
        """-> (is_leader, event, cached_result_or_none).

        Cache hit -> (False, event, deepcopy(result)).
        Inflight  -> (False, event, None) follower waits.
        Full/bypass -> (True, None, None).
        New leader -> (True, new_event, None).
        """
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                if entry.result is not None and entry.expires_at > now:
                    return (False, entry.event, copy.deepcopy(entry.result))
                if entry.result is not None and entry.expires_at <= now:
                    self._entries.pop(key, None)
                    entry = None
                elif entry.exception is not None:
                    if entry.event.is_set():
                        self._entries.pop(key, None)
                        entry = None
                    else:
                        return (False, entry.event, None)
                elif entry.result is None and entry.exception is None:
                    return (False, entry.event, None)
            if len(self._entries) >= self.max_entries:
                return (True, None, None)
            new_entry = Entry(event=threading.Event(), result=None, exception=None, expires_at=0.0)
            self._entries[key] = new_entry
            return (True, new_entry.event, None)

    def complete_ok(self, key: str, result: dict) -> None:
        """Store deepcopy(result) and wake followers (no TTL yet)."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.result = copy.deepcopy(result)
            entry.exception = None
            entry.event.set()

    def complete_err(self, key: str, exc: BaseException) -> None:
        """Store exception and wake followers; evicted on next get_or_create."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.exception = exc
            entry.result = None
            entry.event.set()

    def get_cached(self, key: str) -> dict | None:
        """Deepcopy of cached result if valid, else None (evicts expired)."""
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.result is None:
                return None
            if entry.expires_at <= now:
                self._entries.pop(key, None)
                return None
            if entry.exception is not None:
                return None
            return copy.deepcopy(entry.result)

    def put_cached(self, key: str, result: dict, ttl_s: float) -> bool:
        """Store result with TTL. False if bypassed (max_entries full)."""
        now = time.monotonic()
        expires = now + float(ttl_s) if ttl_s > 0 else now
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.result = copy.deepcopy(result)
                entry.exception = None
                entry.expires_at = expires
                entry.event.set()
                return True
            if len(self._entries) >= self.max_entries:
                return False
            ev = threading.Event()
            ev.set()
            self._entries[key] = Entry(event=ev, result=copy.deepcopy(result), exception=None, expires_at=expires)
            return True

    def evict_expired(self) -> int:
        now = time.monotonic()
        evicted = 0
        with self._lock:
            for k in [k for k, e in self._entries.items() if e.result is not None and e.expires_at <= now]:
                self._entries.pop(k, None)
                evicted += 1
        return evicted

    def forget(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def __len__(self) -> int:  # pragma: no cover
        with self._lock:
            return len(self._entries)

    def _get_entry(self, key: str) -> Entry | None:  # pragma: no cover
        with self._lock:
            return self._entries.get(key)


_DONE_SENTINEL = object()


class TeeBroadcast:
    """Leader append() fans out to followers; followers subscribe_replay_then_live().

    Replay uses deque(maxlen=tee_buffer_lines). Per-follower bounded queue
    (maxsize=tee_buffer_lines) + tee_buffer_bytes cap. Slow followers dropped
    with DONE sentinel. Thread-safe via RLock.
    """

    def __init__(self, tee_buffer_lines: int = 1024, tee_buffer_bytes: int = 2097152) -> None:
        self._lock = threading.RLock()
        self._tee_buffer_lines = max(1, int(tee_buffer_lines))
        self._tee_buffer_bytes = max(1, int(tee_buffer_bytes))
        self._buffer: collections.deque[str | bytes] = collections.deque(maxlen=self._tee_buffer_lines)
        self._buffer_bytes: int = 0
        self._followers: dict[int, queue.Queue[Any]] = {}
        self._next_id: int = 0
        self._closed: bool = False
        self._follower_bytes: dict[int, int] = {}

    def append(self, line: str | bytes) -> None:
        """Append line to deque and fan-out; drop slow followers with DONE."""
        line_len = len(line) if isinstance(line, bytes) else len(line.encode("utf-8"))
        to_drop: list[int] = []
        with self._lock:
            if self._closed:
                return
            if len(self._buffer) == self._buffer.maxlen and len(self._buffer) > 0:
                oldest = self._buffer[0]
                self._buffer_bytes -= len(oldest) if isinstance(oldest, bytes) else len(oldest.encode("utf-8"))
            self._buffer.append(line)
            self._buffer_bytes += line_len
            while self._buffer_bytes > self._tee_buffer_bytes and len(self._buffer) > 1:
                oldest = self._buffer.popleft()
                self._buffer_bytes -= len(oldest) if isinstance(oldest, bytes) else len(oldest.encode("utf-8"))
            for fid, q in list(self._followers.items()):
                queued = self._follower_bytes.get(fid, 0)
                if q.qsize() >= self._tee_buffer_lines or queued + line_len > self._tee_buffer_bytes:
                    to_drop.append(fid)
                    continue
                try:
                    q.put_nowait(line)
                    self._follower_bytes[fid] = queued + line_len
                except queue.Full:
                    to_drop.append(fid)
            for fid in to_drop:
                q = self._followers.pop(fid, None)
                self._follower_bytes.pop(fid, None)
                if q is not None:
                    try:
                        q.put_nowait(_DONE_SENTINEL)
                    except queue.Full:
                        pass

    def close(self) -> None:
        """Close and signal DONE to all followers."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for _, q in list(self._followers.items()):
                try:
                    q.put_nowait(_DONE_SENTINEL)
                except queue.Full:
                    pass
            self._followers.clear()
            self._follower_bytes.clear()

    def subscribe_replay_then_live(self):  # type: ignore[no-untyped-def]
        """Yield deque replay then live lines until close/DONE (generator)."""
        q: queue.Queue[Any] = queue.Queue(maxsize=self._tee_buffer_lines)
        with self._lock:
            replay = list(self._buffer)
            closed = self._closed
            if not closed:
                fid = self._next_id
                self._next_id += 1
                self._followers[fid] = q
                self._follower_bytes[fid] = 0
            else:
                fid = -1
        for item in replay:
            yield item
        if closed:
            yield _DONE_SENTINEL
            return
        while True:
            try:
                item = q.get(timeout=30)
            except queue.Empty:
                with self._lock:
                    if self._closed:
                        break
                continue
            if item is _DONE_SENTINEL:
                yield item
                break
            try:
                ilen = len(item) if isinstance(item, bytes) else len(item.encode("utf-8"))
            except Exception:
                ilen = 0
            with self._lock:
                if fid in self._follower_bytes:
                    self._follower_bytes[fid] = max(0, self._follower_bytes[fid] - ilen)
            yield item
            with self._lock:
                if self._closed and q.empty():
                    break


__all__ = [
    "canonical_body", "fingerprint", "principal_hash_for_api_key",
    "principal_hash", "Entry", "DedupState", "TeeBroadcast",
]
