"""Server module code to be embedded in exllama_install.py _write_server_module().

Based on MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw serve_openai.py (MIT license).
Adapted for Heimdall Gateway: --ctx-size alias, default host 127.0.0.1,
parse_known_args to ignore llama.cpp flags, default draft_model=mtp.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, threading, uuid
from aiohttp import web

gen_lock = threading.Lock()
stats_lock = threading.Lock()
stats = {
    "prompt_tokens_total": 0,
    "completion_tokens_total": 0,
    "context_length": None,
    # last job measured timings for /slots (no fabrication)
    "last_prefill_ms": None,
    "last_decode_ms": None,
    "last_prompt_n": None,
    "last_predicted_n": None,
}

# print_timing emulation (beellama parity) — incremental task counter
_slot_task_counter = 0
_slot_task_lock = threading.Lock()
_graphs_reused = 0
_graphs_reused_lock = threading.Lock()


def _log_both(msg: str) -> None:
    """Write msg to BOTH stdout and stderr for llama-swap logToStdout=both capture.

    llama-swap v251 Model Logs captures upstream stdout+stderr via logToStdout knob.
    EXL3 must emit to both so Model Logs tab shows llama_model_loader / slot
    print_timing / draft acceptance regardless of logToStdout setting.
    """
    try:
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _emit_print_timing(prompt_n: int, predicted_n: int, prompt_ms: float, predicted_ms: float,
                       draft_n, draft_n_accepted) -> None:
    """Emit beellama-like `slot print_timing` lines to BOTH stdout+stderr for llama-swap capture.

    Format matches llama.cpp:5d/8.2f/6.2f widths, includes prompt eval, eval, total,
    graphs reused (fixed 0/incremental since EXL3 has no graph cache), and draft acceptance.
    Real values: prompt_n/prompt_ms, predicted_n/predicted_ms, draft_n/draft_n_accepted.
    Writes to both stdout and stderr so Model Logs appears with logToStdout=both/proxy/upstream.
    """
    global _slot_task_counter, _graphs_reused
    try:
        with _slot_task_lock:
            _slot_task_counter += 1
            task_id = int(_slot_task_counter)
        slot_id = 0
        # graphs reused — no EXL3 equivalent, use counter 0 (or incremental fixed)
        with _graphs_reused_lock:
            graphs = int(_graphs_reused)
            # optionally increment for next call to show progression (keep 0 as safe fixed)
            # _graphs_reused += 1

        total_ms = float(prompt_ms or 0) + float(predicted_ms or 0)
        total_n = int(prompt_n or 0) + int(predicted_n or 0)

        # per-token rates
        if prompt_n and prompt_ms and prompt_ms > 0:
            p_ms_per = float(prompt_ms) / float(prompt_n)
            p_tps = float(prompt_n) / (float(prompt_ms) / 1000.0)
        else:
            p_ms_per = 0.0
            p_tps = 0.0
        if predicted_n and predicted_ms and predicted_ms > 0:
            e_ms_per = float(predicted_ms) / float(predicted_n)
            e_tps = float(predicted_n) / (float(predicted_ms) / 1000.0)
        else:
            e_ms_per = 0.0
            e_tps = 0.0

        lines = []
        # exact widths: %8.2f ms / %5d tokens (%6.2f ms per token, %8.2f tokens per second)
        lines.append(
            f"slot print_timing: id {slot_id:2d} | task {task_id:4d} | prompt eval time = {float(prompt_ms):8.2f} ms / {int(prompt_n):5d} tokens ({p_ms_per:6.2f} ms per token, {p_tps:8.2f} tokens per second)"
        )
        lines.append(
            f"slot print_timing: id {slot_id:2d} | task {task_id:4d} |        eval time = {float(predicted_ms):8.2f} ms / {int(predicted_n):5d} tokens ({e_ms_per:6.2f} ms per token, {e_tps:8.2f} tokens per second)"
        )
        lines.append(
            f"slot print_timing: id {slot_id:2d} | task {task_id:4d} |       total time = {total_ms:8.2f} ms / {total_n:5d} tokens"
        )
        lines.append(
            f"slot print_timing: id {slot_id:2d} | task {task_id:4d} |    graphs reused = {graphs:9d}"
        )
        if draft_n is not None and int(draft_n) > 0:
            try:
                dn = int(draft_n)
                da = int(draft_n_accepted) if draft_n_accepted is not None else 0
                ratio = (da / dn) if dn else 0.0
                # mean len approximated as dn / predicted_n if predicted_n else 0
                mean_len = (float(dn) / float(predicted_n)) if predicted_n else 0.0
                lines.append(
                    f"slot print_timing: id {slot_id:2d} | task {task_id:4d} | draft acceptance = {ratio:.5f} ({da:4d} accepted / {dn:4d} generated)"
                )
            except Exception:
                pass
        for ln in lines:
            _log_both(ln)
    except Exception:
        # never break generation on logging failure
        pass

def _bump_stats(prompt=0, completion=0):
    if prompt <= 0 and completion <= 0:
        return
    with stats_lock:
        if prompt > 0:
            stats["prompt_tokens_total"] += int(prompt)
        if completion > 0:
            stats["completion_tokens_total"] += int(completion)

def _result_new_tokens(r):
    ids = r.get("token_ids") if isinstance(r, dict) else None
    if ids is None:
        return 0
    try:
        return int(ids.shape[-1])
    except Exception:
        return 0


def _extract_draft_stats(job, generator):
    try:
        has_draft = False
        try:
            gd = generator.__dict__.get("draft_model", None) if hasattr(generator, "__dict__") else getattr(generator, "draft_model", None)
            mtp = generator.__dict__.get("mtp_draft", False) if hasattr(generator, "__dict__") else getattr(generator, "mtp_draft", False)
            ndt = generator.__dict__.get("num_draft_tokens", 0) if hasattr(generator, "__dict__") else getattr(generator, "num_draft_tokens", 0)
            ngram = generator.__dict__.get("ngram_match_min", None) if hasattr(generator, "__dict__") else getattr(generator, "ngram_match_min", None)
            has_draft = bool(
                gd is not None
                or bool(mtp)
                or int(ndt or 0) > 0
                or ngram
            )
        except Exception:
            pass
        accepted = int(getattr(job, "accepted_draft_tokens", 0) or 0)
        rejected = int(getattr(job, "rejected_draft_tokens", 0) or 0)
        draft_n = accepted + rejected
        if has_draft:
            return draft_n, accepted
        if draft_n > 0:
            return draft_n, accepted
        return None, None
    except Exception:
        return None, None


def _extract_cached_tokens(job):
    try:
        if hasattr(job, "cached_pages") and hasattr(job, "cached_tokens"):
            try:
                from exllamav3.generator.job import PAGE_SIZE as _PAGE_SIZE
            except Exception:
                _PAGE_SIZE = 256
            try:
                pages = int(getattr(job, "cached_pages", 0) or 0)
                toks = int(getattr(job, "cached_tokens", 0) or 0)
                seqs = getattr(job, "sequences", None)
                denom = len(seqs) if seqs else 1
                if denom <= 0:
                    denom = 1
                total = (pages * _PAGE_SIZE + toks) // denom
                if total > 0:
                    return total
            except Exception:
                pass
        return None
    except Exception:
        return None

TOOL_CALL_OPEN = "\u003ctool_call\u003e"
TOOL_CALL_CLOSE = "\u003c/tool_call\u003e"
HOLD_BACK = 16

# Gateway capa 2 metrics (shared with command_router gateway)
_METRICS_LOCK = threading.Lock()
_METRICS: dict[str, int] = {
    "leakage_marker_emitted_total": 0,
    "thinking_truncated_total": 0,
    "resolve_in_think_mismatch": 0,
}

_NEWLINE_RE = re.compile(r"\n{4,}")
_OUTPUT_JSON_RE = re.compile(r'\{\s*"output"\s*:')
_OUTPUT_KEY_RE = re.compile(r'"output"\s*:')
_USERUSER_CONCAT_RE = re.compile(r'(?:user){2,}', flags=re.IGNORECASE)
_USER_LINES_RE = re.compile(r'(?:^|\n)[ \t]*user[ \t]*(?:\n[ \t]*user[ \t]*)+', flags=re.IGNORECASE | re.MULTILINE)

def _sanitize_tool_output_artifacts(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    if '"output"' in text:
        text = _OUTPUT_JSON_RE.sub('[output_stripped]:', text)
        text = _OUTPUT_KEY_RE.sub('[output_stripped]:', text)
    if 'user' in text.lower():
        text = _USERUSER_CONCAT_RE.sub('user', text)
        text = _USER_LINES_RE.sub('\nuser\n', text)
    return _NEWLINE_RE.sub("\n\n\n", text)

def _inc_metric(name: str, amount: int = 1) -> None:
    if name not in _METRICS:
        return
    with _METRICS_LOCK:
        _METRICS[name] += int(amount)

def _get_metrics() -> dict[str, int]:
    with _METRICS_LOCK:
        return dict(_METRICS)

def _dynamic_hold_back(user_stops: list[str] | None = None) -> int:
    candidates = [
        MARKER_IM_START,
        MARKER_IM_END,
        "\u003c/think\u003e",
        TOOL_CALL_OPEN,
        TOOL_CALL_CLOSE,
    ]
    max_len = max((len(m) for m in candidates), default=16)
    if user_stops:
        for s in user_stops:
            if s:
                max_len = max(max_len, len(s))
    return max_len

# ---------------------------------------------------------------------------
# T2 single documented decision for decode_special_tokens (ONE place)
# ---------------------------------------------------------------------------
# Job(decode_special_tokens=False): special tokens like <|im_start|> /
# <|im_end|> are NOT decoded into `text` — they are handled as
# stop_conditions (Job stops and does not emit them).  Setting True would
# leak raw markers into both streamed deltas and final content, requiring
# stripping everywhere.  We choose False and keep stripping as
# defence-in-depth (covers the HOLD_BACK=16 synthetic leak and any
# tokenizer edge where a marker still appears mid-text).
# Reference: exllamav3/generator/job.py:59 (param) / 231-249 (stop decode).
DECODE_SPECIAL_TOKENS = False

MARKER_IM_START = "<|im_start|>"
MARKER_IM_END = "<|im_end|>"


def strip_markers(text: str) -> str:
    truncated = text.split(MARKER_IM_END)[0].split(MARKER_IM_START)[0]
    if len(truncated) < len(text):
        _inc_metric("leakage_marker_emitted_total")
    return truncated


def _earliest_marker_pos(text: str) -> int:
    """Return earliest index of either complete marker, or -1."""
    a = text.find(MARKER_IM_START)
    b = text.find(MARKER_IM_END)
    if a < 0:
        return b
    if b < 0:
        return a
    return a if a < b else b


def _earliest_user_stop_pos(text: str, stops: list[str] | None) -> int:
    """Return earliest index of any user stop string, or -1."""
    if not stops:
        return -1
    earliest = -1
    for s in stops:
        if not s:
            continue
        p = text.find(s)
        if p >= 0 and (earliest < 0 or p < earliest):
            earliest = p
    return earliest


def strip_user_stops(text: str, stops: list[str] | None) -> str:
    """Truncate at first user stop occurrence."""
    if not stops:
        return text
    pos = _earliest_user_stop_pos(text, stops)
    if pos >= 0:
        return text[:pos]
    return text


def _resolve_enable_thinking(reasoning) -> bool:
    """Normalize reasoning param to enable_thinking bool (matches generate_full logic)."""
    reasoning_norm = str(reasoning or "").strip().lower() if isinstance(reasoning, str) else reasoning
    if reasoning_norm in ("off", "none", "false", "0"):
        return False
    elif reasoning_norm in ("low", "medium", "high", "on", "true", "1", "", None):
        return True
    else:
        return bool(reasoning) if isinstance(reasoning, bool) else True


def _resolve_preserve_thinking_extra(extra_kwargs: dict, reasoning) -> None:
    """Inject preserve_thinking default false for Qwen if not provided (bundle family_defaults.qwen).

    Mutates extra_kwargs in place. Matches spec default false per llamacpp_stack/bundle/llama_server_defaults.yaml:42-47.
    """
    if "preserve_thinking" not in extra_kwargs:
        extra_kwargs["preserve_thinking"] = False


def resolve_in_think_initial(tokenizer, input_ids, enable_thinking: bool) -> bool:
    """Derive in_think from enable_thinking + template suffix (handles divergence).

    T3 QA failure clause: if 27B/small template differs in generation prefix without
    think, log suffix real and adjust per case instead of hardcode. We decode the
    rendered prompt and inspect its tail.
    """
    try:
        # exllamav3 tokenizer decode: decode(tensor) or decode(list)
        prompt_text = None
        if hasattr(tokenizer, "decode"):
            try:
                # input_ids is shape (1, n) tensor
                prompt_text = tokenizer.decode(input_ids[0] if hasattr(input_ids, "__getitem__") else input_ids)
                if isinstance(prompt_text, list):
                    prompt_text = prompt_text[0] if prompt_text else ""
            except Exception:
                prompt_text = None
        if not isinstance(prompt_text, str) and hasattr(tokenizer, "hf_chat_template"):
            # fallback: try to render via tokenizer internals is already done
            prompt_text = None
        if isinstance(prompt_text, str):
            # Template when enable_thinking=False ends with "<think>\\n\\n</think>\\n\\n"
            # When True ends with "<think>\\n"
            tail = prompt_text[-80:]
            has_empty_think = "<think>\n\n</think>" in tail
            has_open_think = tail.rstrip().endswith("<think>")
            # Log suffix for divergence detection (both stdout+stderr for Model Logs)
            _log_both(f"[exllama_server] enable_thinking={enable_thinking} prompt_tail={tail!r} has_empty_think={has_empty_think} has_open_think={has_open_think}")
            if enable_thinking and has_empty_think:
                # Divergence: template rendered empty think even though enable true
                return False
            if not enable_thinking and has_open_think and not has_empty_think:
                return True
            return bool(enable_thinking)
    except Exception as e:
        _log_both(f"[exllama_server] resolve_in_think fallback {e!r} enable={enable_thinking}")
    return bool(enable_thinking)

def build_model(argv, use_draft=True):
    from exllamav3 import model_init, Generator
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, add_draft_model_args=use_draft)
    args = parser.parse_args(argv)
    if use_draft:
        model, config, cache, tokenizer, draft_model, draft_config, draft_cache = \
            model_init.init(args, progress=True)
        generator = Generator(
            model, cache, tokenizer,
            draft_model=draft_model, draft_cache=draft_cache,
        )
    else:
        model, config, cache, tokenizer = model_init.init(args, progress=True)
        generator = Generator(model, cache, tokenizer)
    return generator, tokenizer

def _is_empty_user(content: object) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return content.strip() == ""
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                t = item.get("text") or item.get("content") or ""
                if isinstance(t, str) and t.strip():
                    return False
            elif isinstance(item, str) and item.strip():
                return False
        return True
    return str(content).strip() == ""

def normalize_messages(messages):
    out = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        texts.append(str(part.get("text") or ""))
                    elif part.get("type") == "image_url":
                        pass
                    elif isinstance(part.get("text"), str):
                        texts.append(part.get("text"))
            m["content"] = "\n".join(texts)
        # collapse \n{4,} -> \n\n\n + sanitize {"output": / useruser artifacts (capa 2b parity)
        if isinstance(m.get("content"), str):
            m["content"] = _sanitize_tool_output_artifacts(_NEWLINE_RE.sub("\n\n\n", m["content"]))  # type: ignore
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function") or {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                fn["arguments"] = args
                calls.append({"function": fn})
            m["tool_calls"] = calls
        # collapse consecutive empty user
        is_user = str(m.get("role") or "") == "user"
        if is_user and _is_empty_user(m.get("content")):
            if out and str(out[-1].get("role") or "") == "user" and _is_empty_user(out[-1].get("content")):
                continue
        out.append(m)
    return out

class StreamSplitter:
    THINK_CLOSE = "\u003c/think\u003e"
    TOOL_OPEN = TOOL_CALL_OPEN
    TOOL_CLOSE = TOOL_CALL_CLOSE

    def __init__(self, hold_back: int = HOLD_BACK, tool_schemas=None, in_think: bool = True, user_stops=None):
        self.hold_back = int(hold_back)
        self.tool_schemas = tool_schemas or {}
        self.pending: str = ""
        self.in_think: bool = bool(in_think)
        self.call_idx: int = 0
        self.calls_emitted: bool = False
        self.user_stops: list[str] = [s for s in (user_stops or []) if s]

    def push(self, chunk: str) -> None:
        if chunk:
            self.pending += chunk

    def flush(self, final: bool = False) -> list[dict]:
        out: list[dict] = []
        pending = self.pending
        in_think = self.in_think
        call_idx = self.call_idx
        calls_emitted = self.calls_emitted
        schemas = self.tool_schemas
        hb = self.hold_back

        def _strip(s: str) -> str:
            return s.split(MARKER_IM_END)[0].split(MARKER_IM_START)[0]

        def _strip_all(s: str) -> str:
            return strip_user_stops(_strip(s), self.user_stops)

        while True:
            if in_think:
                close = pending.find(self.THINK_CLOSE)
                if close >= 0:
                    head, pending = pending[:close], pending[close + len(self.THINK_CLOSE):]
                    head = _strip_all(head)
                    if head.strip():
                        out.append({"reasoning_content": head.lstrip("\n")})
                    in_think = False
                    continue
                mpos = _earliest_marker_pos(pending)
                upos = _earliest_user_stop_pos(pending, self.user_stops)
                # combined earliest for final and hold_back
                cpos = mpos if mpos >= 0 and (upos < 0 or mpos < upos) else upos
                if final and cpos >= 0:
                    pending = pending[:cpos]
                    stripped = _strip_all(pending)
                    if stripped.strip():
                        out.append({"reasoning_content": stripped})
                    pending = ""
                    break
                if cpos >= 0 and cpos < max(0, len(pending) - hb):
                    truncated = _strip_all(pending[:cpos])
                    if truncated.strip():
                        out.append({"reasoning_content": truncated})
                    pending = ""
                    break
                cut = len(pending) if final else max(0, len(pending) - hb)
                piece = pending[:cut]
                stripped = _strip_all(piece)
                if len(stripped) < len(piece):
                    if stripped.strip():
                        out.append({"reasoning_content": stripped})
                    pending = ""
                    break
                # also handle user stop inside piece that was held but now emittable via cut
                spos = _earliest_user_stop_pos(piece, self.user_stops)
                if spos >= 0:
                    truncated = _strip_all(piece[:spos])
                    if truncated.strip():
                        out.append({"reasoning_content": truncated})
                    pending = ""
                    break
                if stripped.strip():
                    out.append({"reasoning_content": stripped})
                pending = pending[cut:]
                break
            if TOOL_CALL_OPEN in pending:
                head, rest = pending.split(TOOL_CALL_OPEN, 1)
                orig_head = head
                head = _strip_all(head)
                marker_in_head = len(head) < len(orig_head)
                user_in_head = _earliest_user_stop_pos(head, self.user_stops) >= 0 if not marker_in_head else False
                # also check original head for user stop before stripping
                if not marker_in_head:
                    upos_h = _earliest_user_stop_pos(orig_head, self.user_stops)
                    if upos_h >= 0:
                        head = _strip_all(orig_head[:upos_h])
                        if head.strip() or (final and head):
                            out.append({"content": head})
                        pending = ""
                        break
                if marker_in_head:
                    if head.strip() or (final and head):
                        out.append({"content": head})
                    pending = ""
                    break
                if head.strip() or (final and head):
                    out.append({"content": head})
                if TOOL_CALL_CLOSE in rest:
                    block, pending = rest.split(TOOL_CALL_CLOSE, 1)
                    block_stripped = _strip_all(block)
                    marker_in_block = len(block_stripped) < len(block)
                    if not marker_in_block:
                        upos_b = _earliest_user_stop_pos(block, self.user_stops)
                        if upos_b >= 0:
                            block_stripped = _strip_all(block[:upos_b])
                            pending = ""
                            marker_in_block = True
                    if marker_in_block:
                        pending = ""
                    _, calls = parse_tool_calls(TOOL_CALL_OPEN + block_stripped + TOOL_CALL_CLOSE, schemas)
                    inc_deltas, call_idx = _tool_calls_to_incremental_deltas(calls, call_idx)
                    out.extend(inc_deltas)
                    if inc_deltas:
                        calls_emitted = True
                    if marker_in_block:
                        break
                    m2 = _earliest_marker_pos(pending)
                    u2 = _earliest_user_stop_pos(pending, self.user_stops)
                    c2 = m2 if m2 >= 0 and (u2 < 0 or m2 < u2) else u2
                    if c2 >= 0 and (final or c2 < max(0, len(pending) - hb)):
                        pending = pending[:c2]
                        continue
                    continue
                if final and "\u003cfunction=" in rest:
                    rest_stripped = _strip_all(rest)
                    _, calls = parse_tool_calls(TOOL_CALL_OPEN + rest_stripped, schemas)
                    inc_deltas, call_idx = _tool_calls_to_incremental_deltas(calls, call_idx)
                    out.extend(inc_deltas)
                    if inc_deltas:
                        calls_emitted = True
                    pending = ""
                else:
                    mpos2 = _earliest_marker_pos(rest)
                    upos2 = _earliest_user_stop_pos(rest, self.user_stops)
                    cpos2 = mpos2 if mpos2 >= 0 and (upos2 < 0 or mpos2 < upos2) else upos2
                    if cpos2 >= 0 and (final or (len(head) + len(TOOL_CALL_OPEN) + cpos2) < max(0, len(pending) - hb)):
                        pending = TOOL_CALL_OPEN + rest[:cpos2]
                    else:
                        pending = TOOL_CALL_OPEN + rest
                break
            mpos = _earliest_marker_pos(pending)
            upos = _earliest_user_stop_pos(pending, self.user_stops)
            cpos = mpos if mpos >= 0 and (upos < 0 or mpos < upos) else upos
            if final and cpos >= 0:
                truncated = _strip_all(pending[:cpos])
                if truncated:
                    out.append({"content": truncated})
                pending = ""
                break
            if cpos >= 0 and cpos < max(0, len(pending) - hb):
                truncated = _strip_all(pending[:cpos])
                if truncated:
                    out.append({"content": truncated})
                pending = ""
                break
            cut = len(pending) if final else max(0, len(pending) - hb)
            piece = pending[:cut]
            stripped = _strip_all(piece)
            if len(stripped) < len(piece):
                if stripped:
                    out.append({"content": stripped})
                pending = ""
                break
            # user stop inside piece not at marker level
            spos = _earliest_user_stop_pos(piece, self.user_stops)
            if spos >= 0:
                truncated = _strip_all(piece[:spos])
                if truncated:
                    out.append({"content": truncated})
                pending = ""
                break
            if stripped:
                out.append({"content": stripped})
            pending = pending[cut:]
            break

        self.pending = pending
        self.in_think = in_think
        self.call_idx = call_idx
        self.calls_emitted = calls_emitted
        return out

    def has_pending(self) -> bool:
        return bool(self.pending)


def split_stream_chunk(pending: str, in_think: bool, hold_back: int = HOLD_BACK, final: bool = False, tool_schemas=None, user_stops=None) -> tuple[list[dict], str, bool]:
    s = StreamSplitter(hold_back=hold_back, tool_schemas=tool_schemas, in_think=in_think, user_stops=user_stops)
    s.pending = pending
    deltas = s.flush(final=final)
    return deltas, s.pending, s.in_think


def split_reasoning(text, enable_thinking: bool, user_stops: list[str] | None = None):
    """Split think/content with enable_thinking awareness and user-stop truncation.

    enable_thinking must be explicit bool (legacy None removed in capa 3).
    enable_thinking=False -> reasoning always empty, visible is content after
    any </think> (discards reasoning prefix). enable_thinking=True -> still
    in think when no </think> present, so entire text is reasoning.
    User stops and markers are stripped from the returned parts.
    """
    if not enable_thinking:
        # thinking disabled: reasoning empty, content is after close if present
        close = text.find("\u003c/think\u003e")
        if close >= 0:
            content = text[close + len("\u003c/think\u003e"):]
        elif text.lstrip().startswith("\u003cthink\u003e"):
            content = text.lstrip()[len("\u003cthink\u003e"):].lstrip()
            # if still contains a close after prefix, take after it
            c2 = content.find("\u003c/think\u003e")
            if c2 >= 0:
                content = content[c2 + len("\u003c/think\u003e"):]
        else:
            content = text
        content = strip_user_stops(strip_markers(content), user_stops)
        # strip leading newlines but keep internal
        return "", content.strip("\n").strip() if content.strip() else content.strip("\n")
    # enable_thinking == True
    close = text.find("\u003c/think\u003e")
    if close >= 0:
        reasoning = text[:close]
        content = text[close + len("\u003c/think\u003e"):]
        reasoning = reasoning.lstrip().removeprefix("\u003cthink\u003e")
        reasoning = strip_user_stops(strip_markers(reasoning), user_stops).strip()
        content = strip_user_stops(strip_markers(content), user_stops).strip("\n").strip()
        # .strip() for reasoning already, content keep single strip for newlines then general
        # ensure content retains but without surrounding markers/stops
        return reasoning, content
    if text.lstrip().startswith("\u003cthink\u003e"):
        reasoning = text.lstrip()[len("\u003cthink\u003e"):].strip()
        reasoning = strip_user_stops(strip_markers(reasoning), user_stops).strip()
        return reasoning, ""
    # No close and no prefix: still inside thinking (prompt ended with <think>)
    reasoning = strip_user_stops(strip_markers(text), user_stops).strip()
    return reasoning, ""

def build_tool_schemas(tools):
    schemas = {}
    for t in tools or []:
        fn = (t or {}).get("function") or {}
        name = fn.get("name")
        props = ((fn.get("parameters") or {}).get("properties")) or {}
        if name and isinstance(props, dict):
            schemas[name] = {k: v.get("type") for k, v in props.items()
                             if isinstance(v, dict)}
    return schemas

def _coerce_value(value, jtype):
    v = value.strip()
    if not v:
        return value
    try:
        if jtype == "integer":
            return int(v)
        if jtype == "number":
            try:
                return int(v)
            except ValueError:
                return float(v)
        if jtype == "boolean":
            if v.lower() == "true": return True
            if v.lower() == "false": return False
        if jtype == "array":
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        if jtype == "object":
            parsed = json.loads(v)
            if isinstance(parsed, dict):
                return parsed
    except (ValueError, json.JSONDecodeError):
        pass
    return value

def coerce_tool_args(args, fn_schema):
    if not fn_schema:
        return args
    out = {}
    for k, v in args.items():
        t = fn_schema.get(k)
        types = t if isinstance(t, list) else [t]
        for tt in types:
            if isinstance(tt, str) and tt in ("integer", "number", "boolean",
                                              "array", "object"):
                cv = _coerce_value(v, tt)
                if not isinstance(cv, str):
                    v = cv
                    break
        out[k] = v
    return out

def _fragment_args_incremental(args_json: str, chunk_size: int = 15) -> list[str]:
    """Split arguments JSON into incremental fragments 10-20 chars (default 15) for beellama parity.

    Beellama emits 7 fragments for `{\"name\": \"hermes-agent\"}`: `{"`, `\"name\":\"`, `her`, `mes`, `-agent`, `"`, `}`.
    We mirror by chunking the JSON string into fixed-size pieces (15) which yields
    similar incremental behaviour and satisfies tests requiring 1 < len <=20 per fragment
    and name only in first delta.
    """
    if not args_json:
        return [""]
    size = max(10, min(20, int(chunk_size)))
    return [args_json[i:i+size] for i in range(0, len(args_json), size)] if args_json else [""]


def _tool_calls_to_incremental_deltas(calls: list[dict], start_idx: int) -> tuple[list[dict], int]:
    """Convert atomic calls to incremental deltas: first delta per index with name+id+first fragment, rest fragments.

    Returns (deltas, next_idx) where deltas is list of {"tool_calls": [...]} dicts.
    """
    deltas: list[dict] = []
    idx = int(start_idx)
    for c in calls:
        args_str = c.get("function", {}).get("arguments", "")
        if not isinstance(args_str, str):
            args_str = json.dumps(args_str) if args_str is not None else ""
        fragments = _fragment_args_incremental(args_str, chunk_size=15)
        # first delta carries name/id/type
        first_frag = fragments[0] if fragments else ""
        deltas.append({
            "tool_calls": [{
                "index": idx,
                "id": c.get("id"),
                "type": c.get("type", "function"),
                "function": {"name": c["function"]["name"], "arguments": first_frag},
            }]
        })
        for frag in fragments[1:]:
            deltas.append({
                "tool_calls": [{
                    "index": idx,
                    "function": {"arguments": frag},
                }]
            })
        idx += 1
    return deltas, idx


def parse_tool_calls(text, tool_schemas=None):
    calls = []
    content = text

    def parse_block(block):
        fm = re.search(r"\u003cfunction=([^\u003e]+)\u003e", block)
        if not fm:
            return None
        name = fm.group(1).strip()
        args = {}
        for pm in re.finditer(r"\u003cparameter=([^\u003e]+)\u003e\n?(.*?)\n?\u003c/parameter\u003e",
                              block[fm.end():], flags=re.S):
            args[pm.group(1).strip()] = pm.group(2)
        if tool_schemas:
            args = coerce_tool_args(args, tool_schemas.get(name))
        return {
            "id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }

    while True:
        i = content.find(TOOL_CALL_OPEN)
        if i < 0:
            break
        j = content.find(TOOL_CALL_CLOSE, i)
        if j < 0:
            call = parse_block(content[i + len(TOOL_CALL_OPEN):])
            if call:
                calls.append(call)
            content = content[:i]
            break
        call = parse_block(content[i + len(TOOL_CALL_OPEN):j])
        if call:
            calls.append(call)
        content = content[:i] + content[j + len(TOOL_CALL_CLOSE):]
    return content, calls

def tool_choice_directive(tool_choice, tools):
    if tool_choice in (None, "auto"):
        return tools, None
    if tool_choice == "none":
        return None, None
    names = [t["function"]["name"] for t in (tools or [])
             if isinstance(t, dict) and t.get("type") == "function"]
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name")
        return tools, (f"You must call the function `{name}` now. Reply ONLY with "
                       f"the \u003ctool_call\u003e block for `{name}` and nothing else.")
    if tool_choice == "required":
        one_of = " or ".join(f"`{n}`" for n in names)
        return tools, (f"You must call one of the available functions ({one_of}) "
                       "now. Reply ONLY with the \u003ctool_call\u003e block and nothing else.")
    return tools, None

def generate_full(generator, tokenizer, messages, max_tokens, temperature,
                  top_p, top_k, seed, tools, tool_choice=None, stop=None,
                  on_text=None, reasoning=None, chat_template_kwargs=None):
    schemas = build_tool_schemas(tools)
    tools_rendered, directive = tool_choice_directive(tool_choice, tools)
    if directive:
        messages = list(messages)
        if messages and messages[0].get("role") == "system":
            first = dict(messages[0])
            first["content"] = (first.get("content") or "").rstrip() + "\n\n" + directive
            messages[0] = first
        else:
            messages = [{"role": "system", "content": directive}] + messages
    # reasoning: low/medium/high compatible with llama.cpp (llamacpp_stack/cli.py:3050 half_context)
    # off disables thinking, low/medium/high all enable it (Qwen3.8 always thinks)
    enable_thinking = _resolve_enable_thinking(reasoning)
    extra_kwargs = dict(chat_template_kwargs) if isinstance(chat_template_kwargs, dict) else {}
    _resolve_preserve_thinking_extra(extra_kwargs, reasoning)
    try:
        input_ids = tokenizer.hf_chat_template(
            messages, add_generation_prompt=True, enable_thinking=enable_thinking,
            tools=tools_rendered, **extra_kwargs)
    except Exception as e:
        raise RuntimeError(f"prompt template error: {e}") from e
    prompt_toks = int(input_ids.shape[-1])
    from exllamav3.generator.sampler.presets import ComboSampler
    from exllamav3 import Job
    forced_choice = tool_choice not in (None, "auto", "none")
    reason = "max_new_tokens"
    text = ""
    ctx_fixed = 262144
    try:
        ctx_val = int(stats.get("context_length") or 0)
        ctx_size = ctx_val if ctx_val > 0 else ctx_fixed
    except Exception:
        ctx_size = ctx_fixed
    thinking_budget = 0
    internal_max_tokens = int(max_tokens)
    if enable_thinking:
        probe = False
        try:
            if int(max_tokens) <= 128 and isinstance(messages, list) and len(messages) == 1:
                c = messages[0].get("content") if isinstance(messages[0], dict) else ""
                if isinstance(c, str) and len(c.strip()) < 32:
                    probe = True
        except Exception:
            probe = False
        if not probe:
            half = ctx_size // 2
            thinking_budget = half
            remaining = ctx_size - prompt_toks
            if remaining < internal_max_tokens:
                remaining = internal_max_tokens
            if internal_max_tokens + thinking_budget > remaining:
                thinking_budget = max(0, remaining - internal_max_tokens)
            internal_max_tokens = internal_max_tokens + thinking_budget
        _log_both(f"[exllama_server] thinking_budget={thinking_budget} internal_max={internal_max_tokens} max_tokens={max_tokens} ctx={ctx_size} prompt={prompt_toks} probe={probe}")
    in_think_initial = resolve_in_think_initial(tokenizer, input_ids, enable_thinking)
    if in_think_initial != bool(enable_thinking):
        _inc_metric("resolve_in_think_mismatch")
    hb_dynamic = _dynamic_hold_back(stop)
    _log_both(f"[exllama_server] in_think_initial={in_think_initial} hold_back={hb_dynamic} enable_thinking={enable_thinking}")
    _log_both(f"[exllama_server] metrics resolve_in_think_mismatch={_get_metrics().get('resolve_in_think_mismatch', 0)}")

    prefill_ms = 0.0
    decode_ms = 0.0
    _t_gen_start = time.perf_counter()
    _t_prefill_end = None

    def run_once():
        nonlocal text, reason, prefill_ms, decode_ms, _t_prefill_end
        text = ""
        reason = "max_new_tokens"
        sampler = ComboSampler(temperature=temperature, top_k=top_k, top_p=top_p)
        stop_extra = [s for s in (stop or []) if s]
        eos = tokenizer.eos_token_id
        sc = ["\u003c/im_end\u003e"]
        if eos is not None:
            sc.append(eos)
            if eos != 151643:
                sc.append(151643)
            if eos != 151645:
                sc.append(151645)
        sc.append("\u003c/im_start\u003e")
        sc.extend(stop_extra)
        seen = set()
        stop_conditions = [x for x in sc if x not in seen and not seen.add(x)]
        job = Job(input_ids=input_ids, max_new_tokens=internal_max_tokens,
                  stop_conditions=stop_conditions,
                  sampler=sampler, seed=seed,
                  decode_special_tokens=DECODE_SPECIAL_TOKENS)
        prefill_seen = 0
        _t_prefill_end_local = None
        with gen_lock:
            generator.enqueue(job)
            while generator.num_remaining_jobs():
                for r in generator.iterate():
                    if r.get("stage") == "prefill":
                        curr = int(r.get("curr_progress") or 0)
                        if curr > prefill_seen:
                            _bump_stats(prompt=curr - prefill_seen)
                            prefill_seen = curr
                    elif _result_new_tokens(r):
                        if _t_prefill_end_local is None:
                            _t_prefill_end_local = time.perf_counter()
                        _bump_stats(completion=_result_new_tokens(r))
                    chunk = r.get("text", "")
                    if chunk:
                        text += chunk
                        if on_text is not None:
                            on_text(chunk)
                    if r.get("eos"):
                        reason = r.get("eos_reason", reason)
            if prefill_seen < prompt_toks:
                _bump_stats(prompt=prompt_toks - prefill_seen)
        _t_end = time.perf_counter()
        if _t_prefill_end_local is not None:
            prefill_ms = (_t_prefill_end_local - _t_gen_start) * 1000
            decode_ms = (_t_end - _t_prefill_end_local) * 1000
        else:
            prefill_ms = (_t_end - _t_gen_start) * 1000
            decode_ms = 0.0
        return job

    def _split_via_stream(raw_text: str):
        s = StreamSplitter(hold_back=hb_dynamic, tool_schemas=schemas, in_think=in_think_initial, user_stops=stop)
        s.push(raw_text)
        deltas = s.flush(final=True)
        reasoning_local = "".join(d.get("reasoning_content", "") for d in deltas)
        content_local = "".join(d.get("content", "") for d in deltas)
        calls_local: list[dict] = []
        for d in deltas:
            if "tool_calls" in d:
                calls_local.extend(d["tool_calls"])
        return reasoning_local, content_local, calls_local

    job = run_once()
    reasoning, content, calls = _split_via_stream(text)
    if forced_choice and not calls:
        temperature = 0.0
        _t_gen_start = time.perf_counter()
        job = run_once()
        reasoning, content, calls = _split_via_stream(text)
    seq = job.sequences[0]
    out_toks = int(seq.sequence_ids.seq_len - prompt_toks)
    if calls:
        finish = "tool_calls"
    else:
        raw_finish = {"max_new_tokens": "length", "eos": "stop",
                  "stop_condition": "stop", "banned": "content_filter"}.get(
                      reason, "stop")
        if raw_finish == "length" and enable_thinking:
            if not content.strip():
                finish = "stop"
            else:
                finish = "length"
                if "</think>" not in text:
                    _inc_metric("thinking_truncated_total")
            if raw_finish == "length" and "</think>" not in text:
                _log_both(f"[exllama_server] thinking_truncated thinking_budget={thinking_budget} internal_max={internal_max_tokens} in_think_initial={in_think_initial}")
        else:
            finish = raw_finish
    draft_n, draft_n_accepted = _extract_draft_stats(job, generator)
    cached_real = _extract_cached_tokens(job)
    with stats_lock:
        stats["last_prefill_ms"] = float(prefill_ms)
        stats["last_decode_ms"] = float(decode_ms)
        stats["last_prompt_n"] = int(prompt_toks)
        stats["last_predicted_n"] = int(out_toks)
    _emit_print_timing(prompt_toks, out_toks, prefill_ms, decode_ms, draft_n, draft_n_accepted)
    return text, calls, finish, prompt_toks, out_toks, reasoning, content, prefill_ms, decode_ms, draft_n, draft_n_accepted, cached_real

async def models(request):
    ctx = stats.get("context_length")
    return web.json_response({"object": "list", "data": [{
        "id": "qwen3.8-27b-exl3-3.5bpw",
        "object": "model",
        "owned_by": "exl3",
        **({"max_model_len": ctx} if ctx else {}),
    }]})

async def health(request):
    with stats_lock:
        busy = gen_lock.locked()
        pt = int(stats.get("prompt_tokens_total") or 0)
        ct = int(stats.get("completion_tokens_total") or 0)
        ctx = stats.get("context_length")
        return web.json_response({
            "ok": True,
            "status": "ok" if not busy else "loading",
            "busy": busy,
            "backend": "exl3",
            "model": "qwen3.8-27b-exl3-3.5bpw",
            "n_ctx": ctx,
            "context_length": ctx,
            "prompt_tokens_total": pt,
            "completion_tokens_total": ct,
            "total_tokens": pt + ct,
            "slots": [{
                "id": 0,
                "n_ctx": ctx or 0,
                "is_processing": busy,
                "prompt_tokens": pt,
                "generated_tokens": ct,
            }],
            "total_slots": 1,
            "idle_slots": 0 if busy else 1,
        })

async def metrics(request):
    with stats_lock:
        pt = int(stats.get("prompt_tokens_total") or 0)
        ct = int(stats.get("completion_tokens_total") or 0)
        ctx = stats.get("context_length")
    m = _get_metrics()
    return web.json_response({
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
        "prompt": pt,
        "generated": ct,
        "prefill_tokens": pt,
        "decode_tokens": ct,
        "context_length": ctx,
        "leakage_marker_emitted_total": int(m.get("leakage_marker_emitted_total", 0)),
        "thinking_truncated_total": int(m.get("thinking_truncated_total", 0)),
        "resolve_in_think_mismatch": int(m.get("resolve_in_think_mismatch", 0)),
    })

def parse_request(body):
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return None, "`messages` (list) is required"
    max_tokens = int(body.get("max_tokens") or
                     body.get("max_completion_tokens") or 1024)
    temperature = float(body.get("temperature", 0.6))
    top_p = float(body.get("top_p", 0.95))
    top_k = int(body.get("top_k", 20))
    seed = body.get("seed")
    tools = body.get("tools") or None
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    elif not isinstance(stop, list):
        stop = None
    # reasoning: low/medium/high/off compatible with llama.cpp (jinja)
    reasoning = body.get("reasoning")
    if reasoning is None:
        reasoning = body.get("reasoning_budget")
    if reasoning is None and isinstance(body.get("chat_template_kwargs"), dict):
        reasoning = body["chat_template_kwargs"].get("reasoning")
    chat_template_kwargs = body.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = None
    # image preprocessing compat: body.get("image_min_tokens") ignored (text-only)
    return dict(
        messages=normalize_messages(messages),
        max_tokens=max_tokens, temperature=temperature,
        top_p=top_p, top_k=top_k,
        seed=int(seed) if seed is not None else None,
        tools=tools,
        tool_choice=body.get("tool_choice"),
        stop=stop,
        stream=bool(body.get("stream", False)),
        model_id=body.get("model", "qwen3.8-27b-exl3-3.5bpw"),
        reasoning=reasoning,
        chat_template_kwargs=chat_template_kwargs,
    ), None

def parse_completion_request(body):
    """Parse legacy /completion body. Minimal llama.cpp compat.

    Accepts: prompt (str, required), n_predict/n/max_tokens, temperature, top_p, top_k,
    seed, stop, stream, model, reasoning/chat_template_kwargs.
    Maps prompt -> single user message for generate_full.
    """
    prompt = body.get("prompt")
    if prompt is None:
        prompt = body.get("input")
    if not isinstance(prompt, str) or not prompt:
        return None, "`prompt` (string) is required"
    n_predict = body.get("n_predict")
    if n_predict is None:
        n_predict = body.get("n")
    if n_predict is None:
        n_predict = body.get("max_tokens")
    if n_predict is None:
        n_predict = body.get("max_completion_tokens")
    if n_predict is None:
        n_predict = body.get("num_predict", 256)
    try:
        max_tokens = int(n_predict)
    except Exception:
        max_tokens = 256
    if max_tokens < 0:
        max_tokens = 1024
    if max_tokens == 0:
        max_tokens = 256
    temperature = float(body.get("temperature", 0.6))
    top_p = float(body.get("top_p", 0.95))
    top_k = int(body.get("top_k", 20))
    seed = body.get("seed")
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    elif not isinstance(stop, list):
        stop = None
    reasoning = body.get("reasoning")
    if reasoning is None:
        reasoning = body.get("reasoning_budget")
    if reasoning is None and isinstance(body.get("chat_template_kwargs"), dict):
        reasoning = body["chat_template_kwargs"].get("reasoning")
    chat_template_kwargs = body.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = None
    messages = [{"role": "user", "content": prompt}]
    return dict(
        prompt=prompt,
        messages=normalize_messages(messages),
        max_tokens=max_tokens, temperature=temperature,
        top_p=top_p, top_k=top_k,
        seed=int(seed) if seed is not None else None,
        stop=stop,
        stream=bool(body.get("stream", False)),
        model_id=body.get("model", "qwen3.8-27b-exl3-3.5bpw"),
        reasoning=reasoning,
        chat_template_kwargs=chat_template_kwargs,
    ), None


async def legacy_completion(request):
    """POST /completion and POST /v1/completions compat - emulates llama.cpp.

    Accepts prompt + n_predict + sampling params, maps prompt to user message,
    calls generate_full, returns llama.cpp-like JSON with content, tokens_*, timings.
    Supports stream=true via SSE (content chunks).
    """
    app = request.app
    generator, tokenizer = app["generator"], app["tokenizer"]
    try:
        body = await request.json()
    except web.HTTPRequestEntityTooLarge:
        return web.json_response(
            {"error": {"message": f"request body exceeds {request.app['max_body_mb']} MiB limit",
                       "type": "invalid_request_error",
                       "code": "request_entity_too_large"}},
            status=413)
    except Exception:
        return web.json_response({"error": {"message": "invalid JSON"}}, status=400)
    req, err = parse_completion_request(body)
    if err:
        return web.json_response({"error": {"message": err}}, status=400)

    import asyncio
    if not req["stream"]:
        try:
            result = await asyncio.to_thread(
                generate_full, generator, tokenizer, req["messages"],
                req["max_tokens"], req["temperature"], req["top_p"], req["top_k"],
                req["seed"], None, None, req["stop"],
                None, req.get("reasoning"), req.get("chat_template_kwargs"))
            if len(result) == 12:
                text, calls, finish, ptoks, otoks, reasoning, content, prefill_ms, decode_ms, draft_n, draft_n_accepted, cached_real = result
            else:
                text, calls, finish, ptoks, otoks, reasoning, content, prefill_ms, decode_ms = result
                draft_n = draft_n_accepted = cached_real = None
        except AssertionError as e:
            return web.json_response(
                {"error": {"message": f"context/cache: {e}", "type": "invalid_request_error"}},
                status=400)
        except Exception as e:
            return web.json_response(
                {"error": {"message": f"generation error: {e}", "type": "server_error"}},
                status=500)
        prompt_tps = ptoks / (prefill_ms / 1000) if prefill_ms > 0 else 0
        gen_tps = otoks / (decode_ms / 1000) if decode_ms > 0 else 0
        timings = {
            "prompt_n": ptoks,
            "predicted_n": otoks,
            "prompt_ms": round(prefill_ms, 2),
            "predicted_ms": round(decode_ms, 2),
            "prompt_per_second": round(prompt_tps, 2),
            "predicted_per_second": round(gen_tps, 2),
        }
        if draft_n is not None:
            timings["draft_n"] = int(draft_n)
            timings["draft_n_accepted"] = int(draft_n_accepted) if draft_n_accepted is not None else 0
        if cached_real is not None:
            timings["cache_n"] = int(cached_real)
        resp_body = {
            "content": content,
            "tokens_predicted": otoks,
            "tokens_evaluated": ptoks,
            "tokens_cached": int(cached_real) if cached_real is not None else 0,
            "truncated": finish == "length",
            "stop": finish == "stop",
            "stopped_eos": finish == "stop",
            "stopped_word": False,
            "stopped_limit": finish == "length",
            "stopping_word": "",
            "has_new_line": content.endswith("\n") if content else False,
            "model": req["model_id"],
            "timings": timings,
            "tokens": [ptoks, otoks],
            "prompt": req["prompt"],
        }
        resp_body["finish_reason"] = finish
        if reasoning:
            resp_body["reasoning_content"] = reasoning
        resp = web.json_response(resp_body)
        resp.headers["X-Prompt-Tokens"] = str(ptoks)
        resp.headers["X-Completion-Tokens"] = str(otoks)
        return resp

    resp = web.StreamResponse(headers={
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
        "Connection": "keep-alive"})
    await resp.prepare(request)
    model_id = req["model_id"]

    async def run():
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()

        def on_text(chunk):
            loop.call_soon_threadsafe(queue.put_nowait, ("delta", chunk))

        def worker():
            try:
                result = generate_full(
                    generator, tokenizer, req["messages"], req["max_tokens"],
                    req["temperature"], req["top_p"], req["top_k"],
                    req["seed"], None, None, req["stop"],
                    on_text=on_text,
                    reasoning=req.get("reasoning"), chat_template_kwargs=req.get("chat_template_kwargs"))
                if len(result) == 12:
                    _, calls, finish, ptoks, otoks, reasoning, content, prefill_ms, decode_ms, draft_n, draft_n_accepted, cached_real = result
                else:
                    _, calls, finish, ptoks, otoks, reasoning, content, prefill_ms, decode_ms = result
                    draft_n = draft_n_accepted = cached_real = None
                loop.call_soon_threadsafe(queue.put_nowait,
                                          ("done", (finish, content, reasoning, ptoks, otoks, prefill_ms, decode_ms, draft_n, draft_n_accepted, cached_real)))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
        loop.run_in_executor(None, worker)

        _enable_thinking_stream = _resolve_enable_thinking(req.get("reasoning"))
        hb_stream = _dynamic_hold_back(req.get("stop"))
        _in_think_stream = _enable_thinking_stream
        try:
            _tmp_extra = dict(req.get("chat_template_kwargs")) if isinstance(req.get("chat_template_kwargs"), dict) else {}
            _resolve_preserve_thinking_extra(_tmp_extra, req.get("reasoning"))
            _tmp_tools, _ = tool_choice_directive(None, None)
            _tmp_ids = tokenizer.hf_chat_template(req["messages"], add_generation_prompt=True, enable_thinking=_enable_thinking_stream, tools=_tmp_tools, **_tmp_extra)
            _in_think_stream = resolve_in_think_initial(tokenizer, _tmp_ids, _enable_thinking_stream)
        except Exception:
            _in_think_stream = _enable_thinking_stream
        if _in_think_stream != bool(_enable_thinking_stream):
            _inc_metric("resolve_in_think_mismatch")
        splitter = StreamSplitter(hold_back=hb_stream, tool_schemas={}, in_think=_in_think_stream, user_stops=req.get("stop"))

        async def flush_pending(final=False):
            deltas = splitter.flush(final=final)
            for d in deltas:
                if "reasoning_content" in d:
                    await resp.write(f"data: {json.dumps({'content': d['reasoning_content'], 'stop': False})}\n\n".encode())
                elif "content" in d:
                    await resp.write(f"data: {json.dumps({'content': d['content'], 'stop': False})}\n\n".encode())

        while True:
            kind, payload = await queue.get()
            if kind == "error":
                await resp.write(f'data: {json.dumps({"error": {"message": payload}})}\n\n'.encode())
                break
            if kind == "delta":
                splitter.push(payload)
                await flush_pending()
            elif kind == "done":
                await flush_pending(final=True)
                finish, content, reasoning, ptoks, otoks, prefill_ms, decode_ms, draft_n, draft_n_accepted, cached_real = payload
                prompt_tps = ptoks / (prefill_ms / 1000) if prefill_ms > 0 else 0
                gen_tps = otoks / (decode_ms / 1000) if decode_ms > 0 else 0
                timings = {
                    "prompt_n": ptoks, "predicted_n": otoks,
                    "prompt_ms": round(prefill_ms, 2), "predicted_ms": round(decode_ms, 2),
                    "prompt_per_second": round(prompt_tps, 2), "predicted_per_second": round(gen_tps, 2),
                }
                if draft_n is not None:
                    timings["draft_n"] = int(draft_n)
                    timings["draft_n_accepted"] = int(draft_n_accepted) if draft_n_accepted is not None else 0
                if cached_real is not None:
                    timings["cache_n"] = int(cached_real)
                final_obj = {
                    "content": "",
                    "tokens_predicted": otoks, "tokens_evaluated": ptoks,
                    "tokens_cached": int(cached_real) if cached_real is not None else 0,
                    "truncated": finish == "length", "stop": True,
                    "stopped_eos": finish == "stop", "stopped_limit": finish == "length",
                    "timings": timings, "model": model_id, "stop": True,
                }
                await resp.write(f"data: {json.dumps(final_obj)}\n\n".encode())
                await resp.write(b"data: [DONE]\n\n")
                break
        await resp.write_eof()
    try:
        await run()
    except ConnectionResetError:
        pass
    return resp


async def chat_completions(request):
    app = request.app
    generator, tokenizer = app["generator"], app["tokenizer"]
    try:
        body = await request.json()
    except web.HTTPRequestEntityTooLarge:
        return web.json_response(
            {"error": {"message": f"request body exceeds {request.app['max_body_mb']} MiB limit",
                       "type": "invalid_request_error",
                       "code": "request_entity_too_large"}},
            status=413)
    except Exception:
        return web.json_response({"error": {"message": "invalid JSON"}}, status=400)
    req, err = parse_request(body)
    if err:
        return web.json_response({"error": {"message": err}}, status=400)

    import asyncio
    if not req["stream"]:
        try:
            result = await asyncio.to_thread(
                generate_full, generator, tokenizer, req["messages"],
                req["max_tokens"], req["temperature"], req["top_p"], req["top_k"],
                req["seed"], req["tools"], req["tool_choice"], req["stop"],
                None, req.get("reasoning"), req.get("chat_template_kwargs"))
            if len(result) == 12:
                text, calls, finish, ptoks, otoks, reasoning, content, prefill_ms, decode_ms, draft_n, draft_n_accepted, cached_real = result
            else:
                text, calls, finish, ptoks, otoks, reasoning, content, prefill_ms, decode_ms = result
                draft_n = draft_n_accepted = cached_real = None
        except AssertionError as e:
            return web.json_response(
                {"error": {"message": f"context/cache: {e}", "type": "invalid_request_error"}},
                status=400)
        except Exception as e:
            return web.json_response(
                {"error": {"message": f"generation error: {e}", "type": "server_error"}},
                status=500)
        msg = {"role": "assistant", "content": content or None}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if calls:
            msg["tool_calls"] = calls
        prompt_tps = ptoks / (prefill_ms / 1000) if prefill_ms > 0 else 0
        gen_tps = otoks / (decode_ms / 1000) if decode_ms > 0 else 0
        reasoning_toks = len((reasoning or "").split()) if reasoning else 0
        usage = {
            "prompt_tokens": ptoks, "completion_tokens": otoks, "total_tokens": ptoks + otoks,
            "completion_tokens_details": {"reasoning_tokens": reasoning_toks, "visible_tokens": otoks - reasoning_toks},
        }
        if cached_real is not None:
            usage["prompt_tokens_details"] = {"cached_tokens": int(cached_real)}
        timings = {
            "prompt_n": ptoks,
            "predicted_n": otoks,
            "prompt_ms": round(prefill_ms, 2),
            "predicted_ms": round(decode_ms, 2),
            "prompt_per_second": round(prompt_tps, 2),
            "predicted_per_second": round(gen_tps, 2),
        }
        if draft_n is not None:
            timings["draft_n"] = int(draft_n)
            timings["draft_n_accepted"] = int(draft_n_accepted) if draft_n_accepted is not None else 0
        if cached_real is not None:
            timings["cache_n"] = int(cached_real)
        # Align finish_reason with actually emitted tool_calls (do not hide length)
        had_tool = bool(calls)
        if had_tool:
            effective_finish = "tool_calls"
        else:
            if finish == "tool_calls":
                effective_finish = "stop"
            else:
                effective_finish = finish
        resp = web.json_response({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion", "created": int(time.time()),
            "model": req["model_id"],
            "choices": [{"index": 0, "message": msg, "finish_reason": effective_finish}],
            "usage": usage,
            "timings": timings,
            "system_fingerprint": "exl3-mtp",
        })
        resp.headers["X-Prompt-Tokens"] = str(ptoks)
        resp.headers["X-Completion-Tokens"] = str(otoks)
        resp.headers["X-Total-Tokens"] = str(ptoks + otoks)
        return resp

    resp = web.StreamResponse(headers={
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
        "Connection": "keep-alive"})
    await resp.prepare(request)
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model_id = req["model_id"]
    req_schemas = build_tool_schemas(req["tools"])

    async def run():
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()

        def on_text(chunk):
            loop.call_soon_threadsafe(queue.put_nowait, ("delta", chunk))

        forced_choice = req["tool_choice"] not in (None, "auto", "none")

        def worker():
            try:
                result = generate_full(
                    generator, tokenizer, req["messages"], req["max_tokens"],
                    req["temperature"], req["top_p"], req["top_k"],
                    req["seed"], req["tools"], req["tool_choice"], req["stop"],
                    on_text=on_text,
                    reasoning=req.get("reasoning"), chat_template_kwargs=req.get("chat_template_kwargs"))
                if len(result) == 12:
                    _, calls, finish, ptoks, otoks, reasoning, content, prefill_ms, decode_ms, draft_n, draft_n_accepted, cached_real = result
                else:
                    _, calls, finish, ptoks, otoks, reasoning, content, prefill_ms, decode_ms = result
                    draft_n = draft_n_accepted = cached_real = None
                loop.call_soon_threadsafe(queue.put_nowait,
                                          ("done", (calls, finish, reasoning, content, ptoks, otoks, prefill_ms, decode_ms, draft_n, draft_n_accepted, cached_real)))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
        loop.run_in_executor(None, worker)

        async def send(delta, finish=None):
            obj = {"id": cid, "object": "chat.completion.chunk",
                   "created": int(time.time()), "model": model_id,
                   "choices": [{"index": 0, "delta": delta,
                                "finish_reason": finish}]}
            await resp.write(f"data: {json.dumps(obj)}\n\n".encode())

        finish = None
        _enable_thinking_stream = _resolve_enable_thinking(req.get("reasoning"))
        hb_stream = _dynamic_hold_back(req.get("stop"))
        _in_think_stream = _enable_thinking_stream
        try:
            _tmp_extra2 = dict(req.get("chat_template_kwargs")) if isinstance(req.get("chat_template_kwargs"), dict) else {}
            _resolve_preserve_thinking_extra(_tmp_extra2, req.get("reasoning"))
            _tmp_tools2, _ = tool_choice_directive(req.get("tool_choice"), req.get("tools"))
            _tmp_ids2 = tokenizer.hf_chat_template(req["messages"], add_generation_prompt=True, enable_thinking=_enable_thinking_stream, tools=_tmp_tools2, **_tmp_extra2)
            _in_think_stream = resolve_in_think_initial(tokenizer, _tmp_ids2, _enable_thinking_stream)
        except Exception:
            _in_think_stream = _enable_thinking_stream
        if _in_think_stream != bool(_enable_thinking_stream):
            _inc_metric("resolve_in_think_mismatch")
        _log_both(f"[exllama_server] stream in_think_initial={_in_think_stream} hold_back={hb_stream} enable={_enable_thinking_stream}")
        _log_both(f"[exllama_server] metrics resolve_in_think_mismatch={_get_metrics().get('resolve_in_think_mismatch', 0)}")
        splitter = StreamSplitter(hold_back=hb_stream, tool_schemas=req_schemas, in_think=_in_think_stream, user_stops=req.get("stop"))

        async def flush_pending(final=False):
            deltas = splitter.flush(final=final)
            for d in deltas:
                if "reasoning_content" in d:
                    await send({"reasoning_content": d["reasoning_content"]})
                elif "content" in d:
                    await send({"content": d["content"]})
                elif "tool_calls" in d:
                    await send({"tool_calls": d["tool_calls"]})

        while True:
            kind, payload = await queue.get()
            if kind == "error":
                await resp.write(
                    f'data: {json.dumps({"error": {"message": payload}})}\n\n'.encode())
                break
            if kind == "delta":
                splitter.push(payload)
                await flush_pending()
            elif kind == "done":
                if len(payload) == 11:
                    calls, finish, reasoning, content, ptoks, otoks, prefill_ms, decode_ms, draft_n, draft_n_accepted, cached_real = payload
                else:
                    calls, finish, reasoning, content, ptoks, otoks, prefill_ms, decode_ms = payload
                    draft_n = draft_n_accepted = cached_real = None
                await flush_pending(final=True)
                if not splitter.calls_emitted and calls:
                    inc_deltas, next_idx = _tool_calls_to_incremental_deltas(calls, splitter.call_idx)
                    for d in inc_deltas:
                        await send({"tool_calls": d["tool_calls"]})
                    splitter.call_idx = next_idx
                    splitter.calls_emitted = True
                had_tool_delta = bool(splitter.calls_emitted)
                if had_tool_delta:
                    effective_finish = "tool_calls"
                else:
                    if finish == "tool_calls":
                        effective_finish = "stop"
                    elif finish == "length":
                        effective_finish = "length"
                    else:
                        effective_finish = finish if finish in ("stop", "length", "content_filter") else "stop"
                finish = effective_finish
                prompt_tps = ptoks / (prefill_ms / 1000) if prefill_ms > 0 else 0
                gen_tps = otoks / (decode_ms / 1000) if decode_ms > 0 else 0
                usage = {
                    "prompt_tokens": ptoks, "completion_tokens": otoks,
                    "total_tokens": ptoks + otoks,
                }
                if cached_real is not None:
                    usage["prompt_tokens_details"] = {"cached_tokens": int(cached_real)}
                timings = {
                    "prompt_n": ptoks,
                    "predicted_n": otoks,
                    "prompt_ms": round(prefill_ms, 2),
                    "predicted_ms": round(decode_ms, 2),
                    "prompt_per_second": round(prompt_tps, 2),
                    "predicted_per_second": round(gen_tps, 2),
                }
                if draft_n is not None:
                    timings["draft_n"] = int(draft_n)
                    timings["draft_n_accepted"] = int(draft_n_accepted) if draft_n_accepted is not None else 0
                if cached_real is not None:
                    timings["cache_n"] = int(cached_real)
                final_obj = {
                    "id": cid, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    "usage": usage,
                    "timings": timings,
                }
                await resp.write(f"data: {json.dumps(final_obj)}\n\n".encode())
                await resp.write(b"data: [DONE]\n\n")
                break
        await resp.write_eof()
    try:
        await run()
    except ConnectionResetError:
        pass
    return resp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-dm", "--draft_model", default="mtp")
    ap.add_argument("-gs", "--grid_size", type=int, default=110)
    ap.add_argument("-cs", "--cache_size", type=int, default=65536)
    ap.add_argument("--ctx-size", type=int, default=None,
                    help="Alias for --cache_size (llama-swap compatibility)")
    ap.add_argument("-cq", "--cache_quant", type=str, default="4",
                    help="KV cache quantization bits (default: 4 = 4-bit, reduces VRAM)")
    ap.add_argument("-p", "--port", type=int, default=8080)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("-ccs", "--cpu_cache_size", type=float, default=0.0)
    ap.add_argument("--max_body_mb", type=int, default=64)
    ap.add_argument("--tensor_parallel", "--tensor-parallel",
                    action="store_true", default=False,
                    help="Enable tensor-parallel across all available GPUs")
    args, _ = ap.parse_known_args()

    if args.ctx_size is not None:
        args.cache_size = args.ctx_size

    _draft = args.draft_model.lower()
    use_mtp = _draft == "mtp"
    use_draft = _draft not in ("none", "", "-")
    argv = ["-m", args.model, "-cs", str(args.cache_size)]
    if not args.tensor_parallel:
        argv += ["-gs", str(args.grid_size)]
    if use_mtp:
        argv += ["-mtp"]
    elif use_draft:
        argv += ["-dm", args.draft_model]
    if args.cache_quant:
        argv += ["-cq", args.cache_quant]
    if args.cpu_cache_size:
        argv += ["-ccs", str(args.cpu_cache_size)]
    if args.tensor_parallel:
        argv += ["-tp"]

    _log_both(f" == loading {args.model}"
           + (" + MTP head" if use_mtp else
              (f" + draft {args.draft_model}" if use_draft else " (no draft)"))
           + " ...")
    _log_both(f" -- Config: ctx-size={args.cache_size} cache_quant={args.cache_quant} grid_size={args.grid_size}"
           f" tensor_parallel={args.tensor_parallel} cpu_cache={args.cpu_cache_size}"
           f" host={args.host}:{args.port} max_body={args.max_body_mb}MiB")
    _log_both(f" -- Loading {args.model}")
    _log_both(f" -- Loading tokenizer...")
    generator, tokenizer = build_model(argv, use_draft=use_draft)
    stats["context_length"] = int(args.cache_size)
    _log_both(f" -- n_ctx={args.cache_size} n_parallel=1 cache_quant={args.cache_quant} grid={args.grid_size}")
    _log_both(" == model ready; accepting requests")
    _log_both(f"llama_model_loader: loaded {args.model} n_ctx={args.cache_size} n_parallel=1")

    app = web.Application(client_max_size=args.max_body_mb * 1024 * 1024)
    app["generator"] = generator
    app["tokenizer"] = tokenizer
    app["max_body_mb"] = args.max_body_mb
    async def slots(request):
        with stats_lock:
            busy = gen_lock.locked()
            pt = int(stats.get("prompt_tokens_total") or 0)
            ct = int(stats.get("completion_tokens_total") or 0)
            ctx = stats.get("context_length") or 0
            last_prefill = stats.get("last_prefill_ms")
            last_decode = stats.get("last_decode_ms")
            last_prompt_n = stats.get("last_prompt_n")
            last_pred = stats.get("last_predicted_n")
        slot = {
            "id": 0, "n_ctx": ctx, "is_processing": busy,
            "prompt_tokens": pt, "generated_tokens": ct,
            "n_prompt_tokens": pt, "n_generated": ct, "n_tokens": pt + ct,
            "prefill_tokens": pt, "decode_tokens": ct,
            "total_tokens": pt + ct, "tokens_evaluated": pt, "tokens_generated": ct,
        }
        if last_prefill is not None and last_decode is not None:
            slot["t_prompt_ms"] = round(float(last_prefill), 2)
            slot["t_eval_ms"] = round(float(last_decode), 2)
            slot["t_ms"] = round(float(last_prefill) + float(last_decode), 2)
            if last_prompt_n is not None:
                slot["prompt_n"] = int(last_prompt_n)
            if last_pred is not None:
                slot["predicted_n"] = int(last_pred)
        return web.json_response([slot])

    async def props(request):
        with stats_lock:
            ctx = stats.get("context_length") or 0
        return web.json_response({
            "total_slots": 1, "default_generation_settings": {"n_ctx": ctx, "n_predict": -1},
            "n_ctx": ctx, "model": "qwen3.8-27b-exl3-3.5bpw",
        })

    async def index(request):
        return web.json_response({
            "model": "qwen3.8-27b-exl3-3.5bpw",
            "status": "ok",
            "endpoints": ["/health", "/metrics", "/slots", "/v1/chat/completions", "/completion"],
        })

    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    app.router.add_get("/slots", slots)
    app.router.add_get("/props", props)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/", index)
    try:
        app.router.add_get("", index)
    except Exception:
        pass
    app.router.add_get("/upstream", index)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_post("/completion", legacy_completion)
    app.router.add_post("/v1/completions", legacy_completion)
    app.router.add_post("/v1/completion", legacy_completion)
    web.run_app(app, host=args.host, port=args.port, print=None)

if __name__ == "__main__":
    main()
