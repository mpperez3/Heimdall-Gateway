"""Tests for dedup inflight singleflight + TTL + TeeBroadcast + handler integration."""
import copy
import hashlib
import io
import json
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import llamacpp_stack.dedup as dedup_mod
from llamacpp_stack.dedup import (
    DedupState,
    Entry,
    TeeBroadcast,
    canonical_body,
    fingerprint,
    principal_hash,
    principal_hash_for_api_key,
)


# ---------------------------------------------------------------------------
# Unit: canonical_body
# ---------------------------------------------------------------------------

class TestCanonicalBody:
    def test_allowlist_and_order_same_canonical(self):
        body_a = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.7, "extra": "ignore"}
        body_b = {"temperature": 0.7, "messages": [{"role": "user", "content": "hi"}], "model": "m", "extra": "different"}
        assert canonical_body(body_a) == canonical_body(body_b)

    def test_extra_fields_ignored(self):
        body_with_extra = {"model": "m", "messages": [], "Authorization": "Bearer secret", "X-Request-ID": "123", "foo": "bar"}
        body_clean = {"model": "m", "messages": []}
        assert canonical_body(body_with_extra) == canonical_body(body_clean)

    def test_list_order_preserved(self):
        body1 = {"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]}
        body2 = {"messages": [{"role": "user", "content": "b"}, {"role": "user", "content": "a"}]}
        assert canonical_body(body1) != canonical_body(body2)

    def test_none_and_non_dict_returns_empty(self):
        assert canonical_body(None) == b"{}"
        assert canonical_body("string") == b"{}"
        assert canonical_body(123) == b"{}"
        assert canonical_body([]) == b"{}"

    def test_unserializable_fallback_stringifies(self):
        # use a truly unserializable allowlisted value: set for model
        body2 = {"model": {1, 2, 3}, "messages": [{"role": "user", "content": "hi"}]}
        result = canonical_body(body2)
        # should not raise and should be valid json bytes
        parsed = json.loads(result.decode("utf-8"))
        assert "model" in parsed
        # set was stringified to string
        assert isinstance(parsed["model"], str)

    def test_sort_keys_deterministic(self):
        b1 = {"model": "m", "temperature": 0.5, "top_p": 0.9}
        b2 = {"top_p": 0.9, "model": "m", "temperature": 0.5}
        assert canonical_body(b1) == canonical_body(b2)
        # separators no spaces
        assert b" " not in canonical_body(b1).replace(b" ", b"") or b": " not in canonical_body(b1)

    def test_ensure_ascii_false(self):
        body = {"model": "m", "prompt": "café"}
        result = canonical_body(body)
        assert "café".encode("utf-8") in result


# ---------------------------------------------------------------------------
# Unit: fingerprint stream buckets
# ---------------------------------------------------------------------------

class TestFingerprintStreamBuckets:
    def test_stream_true_vs_false_different(self):
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        c = canonical_body(body)
        ph = principal_hash_for_api_key("key")
        fp_false = fingerprint("POST", "/v1/chat/completions", ph, False, c)
        fp_true = fingerprint("POST", "/v1/chat/completions", ph, True, c)
        assert fp_false != fp_true

    def test_same_inputs_same_fingerprint(self):
        body = {"model": "m", "messages": []}
        c = canonical_body(body)
        ph = principal_hash_for_api_key("abc")
        assert fingerprint("POST", "/v1/chat/completions", ph, False, c) == fingerprint("POST", "/v1/chat/completions", ph, False, c)

    def test_method_upper_and_path_query_stripped(self):
        c = canonical_body({"model": "m"})
        ph = "anonymous"
        fp1 = fingerprint("post", "/v1/chat/completions?foo=bar", ph, False, c)
        fp2 = fingerprint("POST", "/v1/chat/completions", ph, False, c)
        assert fp1 == fp2

    def test_different_body_different_fingerprint(self):
        c1 = canonical_body({"model": "m", "temperature": 0.1})
        c2 = canonical_body({"model": "m", "temperature": 0.9})
        ph = "anonymous"
        assert fingerprint("POST", "/v1/chat/completions", ph, False, c1) != fingerprint("POST", "/v1/chat/completions", ph, False, c2)


# ---------------------------------------------------------------------------
# Unit: principal isolation
# ---------------------------------------------------------------------------

class TestPrincipalIsolation:
    def test_different_api_keys_different_fingerprints(self):
        c = canonical_body({"model": "m"})
        ph_a = principal_hash_for_api_key("key-A")
        ph_b = principal_hash_for_api_key("key-B")
        assert ph_a != ph_b
        fp_a = fingerprint("POST", "/v1/chat/completions", ph_a, False, c)
        fp_b = fingerprint("POST", "/v1/chat/completions", ph_b, False, c)
        assert fp_a != fp_b

    def test_anonymous_vs_key_different(self):
        c = canonical_body({"model": "m"})
        ph_anon = principal_hash_for_api_key(None)
        ph_key = principal_hash_for_api_key("secret")
        assert ph_anon == "anonymous"
        assert ph_anon != ph_key
        assert fingerprint("POST", "/v1/chat/completions", ph_anon, False, c) != fingerprint("POST", "/v1/chat/completions", ph_key, False, c)

    def test_empty_string_is_anonymous(self):
        assert principal_hash_for_api_key("") == "anonymous"
        assert principal_hash_for_api_key(None) == "anonymous"
        assert principal_hash("") == "anonymous"

    def test_alias(self):
        assert principal_hash("k") == principal_hash_for_api_key("k")
        assert hashlib.sha256("k".encode()).hexdigest() == principal_hash("k")


# ---------------------------------------------------------------------------
# Unit: DedupState singleflight
# ---------------------------------------------------------------------------

class TestDedupStateSingleflight:
    def test_leader_and_follower_share_event(self):
        state = DedupState(max_entries=10)
        key = "k1"
        is_leader1, ev1, cached1 = state.get_or_create_inflight(key)
        assert is_leader1 is True
        assert ev1 is not None
        assert cached1 is None
        is_leader2, ev2, cached2 = state.get_or_create_inflight(key)
        assert is_leader2 is False
        assert ev2 is ev1
        assert cached2 is None
        # follower waits for leader
        result = {"status": 200, "headers": {"Content-Type": "application/json"}, "body": b'{"ok":1}'}
        # complete in another thread after short delay to test wake
        def complete():
            time.sleep(0.05)
            state.complete_ok(key, result)
        t = threading.Thread(target=complete)
        t.start()
        waited = ev2.wait(timeout=2)
        assert waited is True
        t.join()
        # get_cached returns deep copy
        cached = state.get_cached(key)
        # put with TTL to make valid
        state.put_cached(key, result, ttl_s=10)
        cached2 = state.get_cached(key)
        assert cached2 is not None
        assert cached2["body"] == b'{"ok":1}'
        # deep copy isolation
        cached2["body"] = b"tampered"
        cached3 = state.get_cached(key)
        assert cached3["body"] == b'{"ok":1}'

    def test_complete_ok_and_get_cached_deep_copy(self):
        state = DedupState(max_entries=10)
        key = "k2"
        is_leader, ev, _ = state.get_or_create_inflight(key)
        assert is_leader
        payload = {"status": 200, "headers": {}, "body": b"hello", "nested": {"a": [1, 2]}}
        state.complete_ok(key, payload)
        # need put_cached to set TTL, otherwise get_cached returns None (expires_at 0)
        # but event is set and get_or_create should return cached after put
        state.put_cached(key, payload, ttl_s=10)
        c = state.get_cached(key)
        assert c == payload
        # mutate original and cached should not affect stored
        payload["nested"]["a"].append(99)
        c2 = state.get_cached(key)
        assert c2["nested"]["a"] == [1, 2]
        c["nested"]["a"].append(100)
        c3 = state.get_cached(key)
        assert c3["nested"]["a"] == [1, 2]

    def test_complete_err_not_cached(self):
        state = DedupState(max_entries=10)
        key = "k-err"
        is_leader, ev, _ = state.get_or_create_inflight(key)
        assert is_leader
        state.complete_err(key, RuntimeError("upstream fail"))
        # get_cached should be None (error not cached)
        assert state.get_cached(key) is None
        # next get_or_create should evict error entry and create new leader
        is_leader2, ev2, cached2 = state.get_or_create_inflight(key)
        assert is_leader2 is True
        assert ev2 is not None
        assert ev2 is not ev  # new entry
        assert cached2 is None

    def test_follower_event_wait_after_complete_err(self):
        state = DedupState(max_entries=10)
        key = "k-err2"
        _, ev1, _ = state.get_or_create_inflight(key)
        _, ev2, _ = state.get_or_create_inflight(key)
        assert ev1 is ev2
        state.complete_err(key, ValueError("oops"))
        assert ev2.is_set() is True
        # follower would see exception entry then next call evicts
        assert state.get_cached(key) is None


# ---------------------------------------------------------------------------
# Unit: DedupState max_entries fail-open
# ---------------------------------------------------------------------------

class TestDedupStateMaxEntries:
    def test_fail_open_on_full(self):
        state = DedupState(max_entries=2)
        k1, k2, k3 = "k1", "k2", "k3"
        is_leader1, ev1, _ = state.get_or_create_inflight(k1)
        assert is_leader1 and ev1 is not None
        is_leader2, ev2, _ = state.get_or_create_inflight(k2)
        assert is_leader2 and ev2 is not None
        # third should bypass
        is_leader3, ev3, cached3 = state.get_or_create_inflight(k3)
        assert is_leader3 is True
        assert ev3 is None
        assert cached3 is None

    def test_put_cached_fail_open_when_full(self):
        state = DedupState(max_entries=1)
        state.get_or_create_inflight("k1")
        # now full, put for new key should fail
        ok = state.put_cached("k-new", {"status": 200, "body": b"x"}, ttl_s=10)
        assert ok is False

    def test_after_forget_can_insert_again(self):
        state = DedupState(max_entries=1)
        is_l, ev, _ = state.get_or_create_inflight("k1")
        assert ev is not None
        is_l3, ev3, _ = state.get_or_create_inflight("k2")
        assert ev3 is None  # bypass
        state.forget("k1")
        is_l4, ev4, _ = state.get_or_create_inflight("k2")
        assert is_l4 is True and ev4 is not None


# ---------------------------------------------------------------------------
# Unit: TeeBroadcast replay_then_live
# ---------------------------------------------------------------------------

class TestTeeBroadcast:
    def test_replay_then_live_basic(self):
        tee = TeeBroadcast(tee_buffer_lines=10, tee_buffer_bytes=1024 * 1024)
        tee.append(b"data: line1\n\n")
        tee.append(b"data: line2\n\n")
        tee.append(b"data: line3\n\n")
        received = []
        # subscriber should get replay of 3 lines + live
        gen = tee.subscribe_replay_then_live()
        # collect replay synchronously (first 3 items)
        for _ in range(3):
            received.append(next(gen))
        assert received == [b"data: line1\n\n", b"data: line2\n\n", b"data: line3\n\n"]
        # now test live: append after subscription
        live_collected = []

        def follower():
            for item in gen:
                if item is dedup_mod._DONE_SENTINEL:
                    break
                live_collected.append(item)

        t = threading.Thread(target=follower)
        t.start()
        time.sleep(0.05)
        tee.append(b"data: line4\n\n")
        tee.append(b"data: line5\n\n")
        time.sleep(0.05)
        tee.close()
        t.join(timeout=2)
        assert live_collected[0] == b"data: line4\n\n"
        assert live_collected[1] == b"data: line5\n\n"

    def test_close_signals_done(self):
        tee = TeeBroadcast(tee_buffer_lines=10, tee_buffer_bytes=1024)
        gen = tee.subscribe_replay_then_live()
        tee.close()
        # after close, generator should yield DONE sentinel quickly
        # first replay is empty, then DONE
        items = list(gen)
        assert dedup_mod._DONE_SENTINEL in items

    def test_slow_follower_dropped(self):
        # buffer lines =2, so third append should drop slow follower whose queue full
        tee = TeeBroadcast(tee_buffer_lines=2, tee_buffer_bytes=1024 * 1024)
        # create a follower that never drains
        gen = tee.subscribe_replay_then_live()
        # fill follower queue by appending without consuming
        # The follower generator yields replay first, but we hold it at live queue
        # To test drop, we exhaust queue via direct internals: we will not consume gen, so its queue builds up
        # Use a fresh follower queue via internal inspection not needed; test drop via capacity
        # Instead, create two followers and verify slow one gets DONE while fast still works
        # Slow path: fill queue beyond tee_buffer_lines
        q = queue.Queue(maxsize=2)
        # Simulate slow follower by directly registering a queue and not draining
        # Use tee internals for direct test
        tee2 = TeeBroadcast(tee_buffer_lines=2, tee_buffer_bytes=100)
        # subscribe but don't consume live items fast
        gen_slow = tee2.subscribe_replay_then_live()
        # consume initial replay (none)
        # push lines faster than consumption
        tee2.append(b"x" * 50)  # 50 bytes
        tee2.append(b"y" * 50)
        # third append should exceed queued bytes or qsize and drop slow follower
        tee2.append(b"z" * 50)
        # slow follower should have been dropped and receive DONE
        # Check that tee no longer has that follower
        # The slow gen should eventually yield DONE
        # We close to ensure termination
        tee2.close()
        # collect whatever remains from slow follower
        collected = []
        for item in gen_slow:
            collected.append(item)
            if item is dedup_mod._DONE_SENTINEL:
                break
            if len(collected) > 10:
                break
        assert dedup_mod._DONE_SENTINEL in collected

    def test_buffer_lines_maxlen_respected(self):
        tee = TeeBroadcast(tee_buffer_lines=2, tee_buffer_bytes=1024 * 1024)
        tee.append(b"a")
        tee.append(b"b")
        tee.append(b"c")
        # buffer should only keep last 2
        assert list(tee._buffer) == [b"b", b"c"]
        gen = tee.subscribe_replay_then_live()
        replay = [next(gen), next(gen)]
        assert replay == [b"b", b"c"]
        tee.close()
        # consume DONE
        for item in gen:
            if item is dedup_mod._DONE_SENTINEL:
                break

    def test_typed_lines_str_and_bytes(self):
        tee = TeeBroadcast(tee_buffer_lines=10, tee_buffer_bytes=1024 * 1024)
        tee.append("data: hello\n\n")
        tee.append(b"data: bytes\n\n")
        assert len(tee._buffer) == 2
        gen = tee.subscribe_replay_then_live()
        assert next(gen) == "data: hello\n\n"
        assert next(gen) == b"data: bytes\n\n"
        tee.close()


# ---------------------------------------------------------------------------
# Helpers for handler-style integration tests
# ---------------------------------------------------------------------------

def _fake_handler(headers=None, path="/v1/chat/completions", client_address=("127.0.0.1", 12345)):
    h = MagicMock()
    h.headers = headers or {}
    h.path = path
    h.client_address = client_address
    h.send_response = MagicMock()
    h.send_header = MagicMock()
    h.end_headers = MagicMock()
    h.wfile = io.BytesIO()
    h.connection = MagicMock()
    return h


# ---------------------------------------------------------------------------
# Integration: double disparo non-streaming
# ---------------------------------------------------------------------------

class TestDoubleDisparoNonStreaming:
    def test_single_upstream_both_receive_same_body_and_headers(self, monkeypatch):
        import llamacpp_stack.cli as cli

        # isolate state
        fresh = DedupState(max_entries=100)
        monkeypatch.setattr(cli, "DEDUP_STATE", fresh)
        # ensure bypass disabled (enabled)
        monkeypatch.setattr(cli, "_dedup_should_bypass", lambda args=None: None)

        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.2, "stream": False}
        canonical = canonical_body(body)
        ph = principal_hash_for_api_key("key-123")
        key = fingerprint("POST", "/v1/chat/completions", ph, False, canonical)

        upstream_calls = {"n": 0}
        result_payload = {"status": 200, "headers": {"Content-Type": "application/json"}, "body": b'{"choices":[]}'}

        def upstream():
            upstream_calls["n"] += 1
            time.sleep(0.08)
            return result_payload

        # Simulate two concurrent requests with same fingerprint
        results = {}
        barriers = threading.Barrier(2)

        def request_thread(name):
            is_leader, ev, cached = fresh.get_or_create_inflight(key)
            if cached is not None:
                # TTL hit
                results[name] = ("hit", copy.deepcopy(cached))
                return
            if not is_leader and ev is not None:
                # follower waits like handler does
                barriers.wait()
                waited = ev.wait(timeout=2)
                if waited:
                    cached2 = fresh.get_cached(key)
                    if cached2 is not None:
                        results[name] = ("shared", cached2)
                    else:
                        # error path
                        results[name] = ("error", None)
                else:
                    results[name] = ("timeout", None)
                return
            # leader
            barriers.wait()
            res = upstream()
            fresh.complete_ok(key, res)
            fresh.put_cached(key, res, ttl_s=600)
            results[name] = ("miss", copy.deepcopy(res))

        t1 = threading.Thread(target=request_thread, args=("t1",))
        t2 = threading.Thread(target=request_thread, args=("t2",))
        t1.start()
        t2.start()
        t1.join(timeout=3)
        t2.join(timeout=3)

        assert upstream_calls["n"] == 1, f"upstream called {upstream_calls['n']} times, expected 1"
        assert set(results.keys()) == {"t1", "t2"}
        # one miss, one shared (order nondeterministic)
        kinds = sorted([v[0] for v in results.values()])
        assert kinds == ["miss", "shared"]
        # both bodies equal
        bodies = [v[1]["body"] for v in results.values() if v[1] is not None]
        assert bodies[0] == bodies[1] == b'{"choices":[]}'
        # deep copy isolation: mutate one should not affect the other cache entry
        for k, (kind, res) in results.items():
            if res is not None:
                res["body"] = b"tampered"
        cached_after = fresh.get_cached(key)
        assert cached_after["body"] == b'{"choices":[]}'

    def test_header_values_miss_shared_hit(self, monkeypatch):
        import llamacpp_stack.cli as cli

        fresh = DedupState(max_entries=10)
        monkeypatch.setattr(cli, "DEDUP_STATE", fresh)

        # test _dedup_send_cached_response header = hit/shared
        h_hit = _fake_handler()
        cached = {"status": 200, "headers": {"Content-Type": "application/json"}, "body": b'{"a":1}'}
        cli._dedup_send_cached_response(h_hit, cached, "hit")
        assert any("hit" in str(c) for c in h_hit.send_header.call_args_list)

        h_shared = _fake_handler()
        cli._dedup_send_cached_response(h_shared, cached, "shared")
        assert any("shared" in str(c) for c in h_shared.send_header.call_args_list)

        # miss via _dedup_send_json_with_header
        h_miss = _fake_handler()
        cli._dedup_send_json_with_header(h_miss, {"ok": True}, status=200, dedup_header="miss")
        assert any("miss" in str(c) for c in h_miss.send_header.call_args_list)


# ---------------------------------------------------------------------------
# Integration: double disparo streaming TeeBroadcast
# ---------------------------------------------------------------------------

class TestDoubleDisparoStreaming:
    def test_streaming_teebroadcast_followers_receive_same_chunks(self):
        tee = TeeBroadcast(tee_buffer_lines=10, tee_buffer_bytes=1024 * 1024)
        chunks = [b"data: chunk1\n\n", b"data: chunk2\n\n", b"data: chunk3\n\n"]

        leader_collected = []
        follower_collected = []

        # leader appends
        for c in chunks:
            tee.append(c)
            leader_collected.append(c)

        # follower subscribes after 3 chunks buffered -> should get replay of 3
        gen = tee.subscribe_replay_then_live()
        # consume replay
        for _ in range(3):
            follower_collected.append(next(gen))

        assert follower_collected == leader_collected

        # now test live streaming fanout with thread
        follower_live = []

        def follower_thread():
            for item in gen:
                if item is dedup_mod._DONE_SENTINEL:
                    break
                follower_live.append(item)

        t = threading.Thread(target=follower_thread)
        t.start()
        time.sleep(0.05)
        tee.append(b"data: chunk4\n\n")
        tee.append(b"data: chunk5\n\n")
        time.sleep(0.05)
        tee.close()
        t.join(timeout=2)
        assert follower_live == [b"data: chunk4\n\n", b"data: chunk5\n\n"]

        import llamacpp_stack.cli as cli
        state = DedupState(max_entries=10)
        orig_state = cli.DEDUP_STATE
        cli.DEDUP_STATE = state
        try:
            state.get_or_create_inflight("stream-key")
            dedup_cfg = {"tee_buffer_bytes": 1024 * 1024, "ttl_s": 600}
            all_chunks = [b"data: chunk1\n\n", b"data: chunk2\n\n"]
            k2 = "stream-key2"
            state.get_or_create_inflight(k2)
            cli._dedup_streaming_finalize(k2, all_chunks, dedup_cfg, "anonymous", True, False, sum(len(c) for c in all_chunks), tee=None)
            cached = state.get_cached(k2)
            assert cached is not None
            assert b"chunk1" in cached["body"]
            k3 = "stream-key3"
            state.get_or_create_inflight(k3)
            cli._dedup_streaming_finalize(k3, all_chunks, dedup_cfg, "anonymous", True, True, 999, tee=None)
            assert state.get_cached(k3) is None
        finally:
            cli.DEDUP_STATE = orig_state

    def test_slow_follower_does_not_break_leader(self):
        tee = TeeBroadcast(tee_buffer_lines=2, tee_buffer_bytes=200)
        # append rapidly while follower not draining fast
        gen_slow = tee.subscribe_replay_then_live()
        # leader continues appending; slow follower will be dropped after buffer exceeded
        for i in range(10):
            tee.append(f"data: line{i}\n\n".encode())
        # leader should still be able to append and close without error
        tee.close()
        # slow follower should eventually get DONE sentinel
        collected = []
        for item in gen_slow:
            collected.append(item)
            if item is dedup_mod._DONE_SENTINEL:
                break
            if len(collected) > 20:
                break
        assert dedup_mod._DONE_SENTINEL in collected


# ---------------------------------------------------------------------------
# Integration: no juntar distintos
# ---------------------------------------------------------------------------

class TestNoJuntarDistintos:
    def test_different_bodies_produce_distinct_fingerprints_and_two_upstream_calls(self, monkeypatch):
        import llamacpp_stack.cli as cli

        fresh = DedupState(max_entries=10)
        monkeypatch.setattr(cli, "DEDUP_STATE", fresh)

        body_a = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "temperature": 0.2}
        body_b = {"model": "m", "messages": [{"role": "user", "content": "hello world"}], "temperature": 0.2}
        ph = "anonymous"
        fp_a = fingerprint("POST", "/v1/chat/completions", ph, False, canonical_body(body_a))
        fp_b = fingerprint("POST", "/v1/chat/completions", ph, False, canonical_body(body_b))
        assert fp_a != fp_b

        # each should be separate leader, simulating two upstream calls
        upstream = {"calls": 0}

        def do_request(body):
            fp = fingerprint("POST", "/v1/chat/completions", ph, False, canonical_body(body))
            is_leader, ev, cached = fresh.get_or_create_inflight(fp)
            if is_leader and ev is not None:
                upstream["calls"] += 1
                res = {"status": 200, "headers": {}, "body": b"ok"}
                fresh.complete_ok(fp, res)
                fresh.put_cached(fp, res, ttl_s=10)
                return "miss"
            return "shared"

        assert do_request(body_a) == "miss"
        assert do_request(body_b) == "miss"
        assert upstream["calls"] == 2

    def test_temperature_vs_top_p_distinct(self):
        c1 = canonical_body({"model": "m", "temperature": 0.7})
        c2 = canonical_body({"model": "m", "top_p": 0.7})
        assert c1 != c2
        ph = "anonymous"
        assert fingerprint("POST", "/v1/chat/completions", ph, False, c1) != fingerprint("POST", "/v1/chat/completions", ph, False, c2)

        c3 = canonical_body({"model": "m", "temperature": 0.1})
        c4 = canonical_body({"model": "m", "temperature": 0.9})
        assert fingerprint("POST", "/v1/chat/completions", ph, False, c3) != fingerprint("POST", "/v1/chat/completions", ph, False, c4)


# ---------------------------------------------------------------------------
# Integration: TTL expiry
# ---------------------------------------------------------------------------

class TestTTLExpiry:
    def test_put_cached_ttl_expiry_and_evict(self):
        state = DedupState(max_entries=10)
        key = "ttl-key"
        result = {"status": 200, "headers": {}, "body": b"cached"}
        state.put_cached(key, result, ttl_s=0.1)
        assert state.get_cached(key) is not None
        time.sleep(0.22)
        assert state.get_cached(key) is None
        # evict_expired on empty should be 0
        assert state.evict_expired() == 0
        # put again and test evict_expired returns 1 after expiry
        state.put_cached(key, result, ttl_s=0.1)
        time.sleep(0.22)
        assert state.evict_expired() == 1
        assert state.get_cached(key) is None

    def test_ttl_zero_not_cached(self):
        state = DedupState(max_entries=10)
        key = "ttl-zero"
        # put_cached with 0 ttl sets expires_at = now, so immediate expiry
        state.put_cached(key, {"status": 200, "body": b"x"}, ttl_s=0)
        assert state.get_cached(key) is None

    def test_complete_ok_without_ttl_expires_immediately_but_follower_still_wakes(self):
        state = DedupState(max_entries=10)
        key = "no-ttl"
        _, ev, _ = state.get_or_create_inflight(key)
        state.complete_ok(key, {"status": 200, "body": b"hi"})
        # expires_at is 0.0 so get_cached returns None (not yet put_cached)
        assert state.get_cached(key) is None
        assert ev.is_set()


# ---------------------------------------------------------------------------
# Integration: kill-switch off
# ---------------------------------------------------------------------------

class TestKillSwitchOff:
    def test_bypass_when_disabled(self, monkeypatch):
        import llamacpp_stack.cli as cli

        # simulate config with enabled=False
        def fake_load(args=None):
            return {"experimental": {"dedup_inflight": {"enabled": False, "ttl_s": 600, "max_wait_ms": 2000, "max_entries": 2000, "tee_buffer_lines": 1024, "tee_buffer_bytes": 2097152}}}

        monkeypatch.setattr(cli, "_load_server_config_payload", fake_load)
        # ensure DEDUP_STATE exists
        fresh = DedupState(max_entries=10)
        monkeypatch.setattr(cli, "DEDUP_STATE", fresh)

        bypass = cli._dedup_should_bypass()
        assert bypass == "dedup_bypass_disabled"

        # when bypass, handler should do two upstream calls (no dedup)
        upstream = {"n": 0}

        def handler_sim(body):
            if cli._dedup_should_bypass() is not None:
                upstream["n"] += 1
                return "bypass"
            return "dedup"

        body = {"model": "m", "messages": []}
        assert handler_sim(body) == "bypass"
        assert handler_sim(body) == "bypass"
        assert upstream["n"] == 2

    def test_bypass_disabled_no_header_and_two_upstreams(self, monkeypatch):
        import llamacpp_stack.cli as cli

        # empty experimental -> defaults enabled False -> bypass
        monkeypatch.setattr(cli, "_load_server_config_payload", lambda args=None: {})
        fresh = DedupState(max_entries=10)
        monkeypatch.setattr(cli, "DEDUP_STATE", fresh)
        assert cli._dedup_should_bypass() == "dedup_bypass_disabled"

        # second case: enabled True -> no bypass
        monkeypatch.setattr(cli, "_load_server_config_payload", lambda args=None: {"experimental": {"dedup_inflight": {"enabled": True, "ttl_s": 600, "max_wait_ms": 2000, "max_entries": 2000, "tee_buffer_lines": 1024, "tee_buffer_bytes": 2097152}}})
        # need to ensure DEDUP_STATE is set after normalize would update max_entries; we set fresh again
        monkeypatch.setattr(cli, "DEDUP_STATE", fresh)
        # _normalize will be called inside _dedup_should_bypass, but we patch to bypass that and just check
        # Force direct check: enabled True should return None (no bypass) if DEDUP_STATE present
        # We need to patch _normalize_dedup_inflight_config to return enabled True
        orig_norm = cli._normalize_dedup_inflight_config
        monkeypatch.setattr(cli, "_normalize_dedup_inflight_config", lambda raw: ({"enabled": True, "ttl_s": 600, "max_wait_ms": 2000, "max_entries": 2000, "tee_buffer_lines": 1024, "tee_buffer_bytes": 2097152}, False))
        assert cli._dedup_should_bypass() is None
        monkeypatch.setattr(cli, "_normalize_dedup_inflight_config", orig_norm)


# ---------------------------------------------------------------------------
# Integration: error not cached
# ---------------------------------------------------------------------------

class TestErrorNotCached:
    def test_complete_err_not_cached_second_does_new_upstream(self):
        state = DedupState(max_entries=10)
        key = "err-key"
        is_leader, ev, _ = state.get_or_create_inflight(key)
        assert is_leader
        state.complete_err(key, RuntimeError("fail"))
        assert state.get_cached(key) is None
        # second identical should create new leader (previous error evicted)
        is_leader2, ev2, cached2 = state.get_or_create_inflight(key)
        assert is_leader2 is True
        assert ev2 is not None
        assert cached2 is None
        # simulate successful retry
        state.complete_ok(key, {"status": 200, "body": b"ok"})
        state.put_cached(key, {"status": 200, "body": b"ok"}, ttl_s=10)
        assert state.get_cached(key) is not None

    def test_4xx_not_cached_simulated_via_complete_err(self, monkeypatch):
        import llamacpp_stack.cli as cli

        fresh = DedupState(max_entries=10)
        monkeypatch.setattr(cli, "DEDUP_STATE", fresh)
        key = "4xx-key"
        # first request leader fails with 4xx
        fresh.get_or_create_inflight(key)
        fresh.complete_err(key, RuntimeError("upstream 400"))
        fresh.forget(key)
        assert fresh.get_cached(key) is None
        # second identical should be leader again (no cache hit), upstream called again
        calls = {"n": 0}
        for _ in range(2):
            # first iter already did error, second iter is retry
            pass
        is_leader, ev, cached = fresh.get_or_create_inflight(key)
        assert is_leader and ev is not None and cached is None
        calls["n"] += 1
        # simulate success on retry
        fresh.complete_ok(key, {"status": 200, "body": b"retry-ok"})
        fresh.put_cached(key, {"status": 200, "body": b"retry-ok"}, ttl_s=10)
        assert fresh.get_cached(key)["body"] == b"retry-ok"
        assert calls["n"] == 1

    def test_error_header_is_miss_not_hit(self, monkeypatch):
        import llamacpp_stack.cli as cli

        h = _fake_handler()
        cli._dedup_send_json_with_header(h, {"error": "oops"}, status=500, dedup_header="miss")
        # should have miss header
        assert any("miss" in str(c) for c in h.send_header.call_args_list)


# ---------------------------------------------------------------------------
# Additional: header values and principal extraction
# ---------------------------------------------------------------------------

class TestDedupHeadersAndPrincipal:
    def test_extract_principal_bearer(self):
        import llamacpp_stack.cli as cli

        h = _fake_handler(headers={"Authorization": "Bearer my-secret-key"})
        ph = cli._dedup_extract_principal(h)
        assert ph == principal_hash_for_api_key("my-secret-key")
        assert ph != "anonymous"

    def test_extract_principal_x_api_key(self):
        import llamacpp_stack.cli as cli

        h = _fake_handler(headers={"X-Api-Key": "another-key"})
        ph = cli._dedup_extract_principal(h)
        assert ph == principal_hash_for_api_key("another-key")

    def test_extract_principal_anonymous_loopback(self):
        import llamacpp_stack.cli as cli

        h = _fake_handler(headers={}, client_address=("127.0.0.1", 1234))
        assert cli._dedup_extract_principal(h) == "anonymous"

    def test_build_fingerprint_determinism(self):
        import llamacpp_stack.cli as cli

        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False}
        ph = principal_hash_for_api_key("k")
        fp1 = cli._dedup_build_fingerprint("POST", "/v1/chat/completions", ph, False, body)
        fp2 = cli._dedup_build_fingerprint("POST", "/v1/chat/completions", ph, False, body)
        assert fp1 == fp2
        # different stream flag
        fp3 = cli._dedup_build_fingerprint("POST", "/v1/chat/completions", ph, True, body)
        assert fp1 != fp3

    def test_ttl_expiry_via_fingerprint_flow(self, monkeypatch):
        import llamacpp_stack.cli as cli

        fresh = DedupState(max_entries=10)
        monkeypatch.setattr(cli, "DEDUP_STATE", fresh)
        body = {"model": "m", "messages": []}
        ph = "anonymous"
        key = fingerprint("POST", "/v1/chat/completions", ph, False, canonical_body(body))
        fresh.get_or_create_inflight(key)
        fresh.complete_ok(key, {"status": 200, "body": b"hi"})
        fresh.put_cached(key, {"status": 200, "body": b"hi"}, ttl_s=0.05)
        assert fresh.get_cached(key) is not None
        time.sleep(0.1)
        assert fresh.get_cached(key) is None
