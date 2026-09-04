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
}

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

TOOL_CALL_OPEN = "\u003ctool_call\u003e"
TOOL_CALL_CLOSE = "\u003c/tool_call\u003e"
HOLD_BACK = 16

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

def normalize_messages(messages):
    out = []
    for m in messages:
        m = dict(m)
        # OpenAI image content: list of {type:text|image_url} -> join text, drop images (EXL3 3.5bpw text-only)
        content = m.get("content")
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        texts.append(str(part.get("text") or ""))
                    elif part.get("type") == "image_url":
                        # image_min_tokens/mmproj not supported for text-only EXL3, ignore
                        pass
                    elif isinstance(part.get("text"), str):
                        texts.append(part.get("text"))
            m["content"] = "\n".join(texts)
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
        out.append(m)
    return out

def split_reasoning(text):
    close = text.find("\u003c/think\u003e")
    if close >= 0:
        reasoning = text[:close]
        content = text[close + len("\u003c/think\u003e"):]
        return reasoning.lstrip().removeprefix("\u003cthink\u003e").strip(), content.strip("\n")
    if text.lstrip().startswith("\u003cthink\u003e"):
        return text.lstrip()[len("\u003cthink\u003e"):].strip(), ""
    return "", text

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
    reasoning_norm = str(reasoning or "").strip().lower() if isinstance(reasoning, str) else reasoning
    if reasoning_norm in ("off", "none", "false", "0"):
        enable_thinking = False
    elif reasoning_norm in ("low", "medium", "high", "on", "true", "1", "", None):
        enable_thinking = True
    else:
        enable_thinking = bool(reasoning) if isinstance(reasoning, bool) else True
    extra_kwargs = dict(chat_template_kwargs) if isinstance(chat_template_kwargs, dict) else {}
    # preserve_thinking false is default for Qwen (llamacpp_stack/bundle/llama_server_defaults.yaml)
    if "preserve_thinking" not in extra_kwargs and reasoning_norm in ("low", "medium", "high"):
        extra_kwargs["preserve_thinking"] = False
    input_ids = tokenizer.hf_chat_template(
        messages, add_generation_prompt=True, enable_thinking=enable_thinking,
        tools=tools_rendered, **extra_kwargs)
    prompt_toks = int(input_ids.shape[-1])
    from exllamav3.generator.sampler.presets import ComboSampler
    from exllamav3 import Job
    forced_choice = tool_choice not in (None, "auto", "none")
    reason = "max_new_tokens"
    text = ""

    def run_once():
        nonlocal text, reason
        text = ""
        reason = "max_new_tokens"
        sampler = ComboSampler(temperature=temperature, top_k=top_k, top_p=top_p)
        stop_conditions = ["\u003c/im_end\u003e", tokenizer.eos_token_id] + (stop or [])
        job = Job(input_ids=input_ids, max_new_tokens=max_tokens,
                  stop_conditions=stop_conditions,
                  sampler=sampler, seed=seed)
        prefill_seen = 0
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
        return job

    job = run_once()
    if forced_choice and not parse_tool_calls(text, schemas)[1]:
        temperature = 0.0
        job = run_once()
    seq = job.sequences[0]
    out_toks = int(seq.sequence_ids.seq_len - prompt_toks)
    content, calls = parse_tool_calls(text, schemas)
    if calls:
        finish = "tool_calls"
    else:
        finish = {"max_new_tokens": "length", "eos": "stop",
                  "stop_condition": "stop", "banned": "content_filter"}.get(
                      reason, "stop")
    reasoning, content = split_reasoning(content)
    return text, calls, finish, prompt_toks, out_toks, reasoning, content

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
        # llama-server compatible fields for llama-swap (Cached/Prompt/Generated/Prefill/Decode)
        n_past = (pt + ct) % (ctx or 1) if ctx else 0
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
            "cached_tokens": ctx or 0,
            "slots": [{
                "id": 0,
                "n_ctx": ctx or 0,
                "n_past": n_past,
                "is_processing": busy,
                "prompt_tokens": pt,
                "generated_tokens": ct,
                "cached_tokens": n_past,
            }],
            "total_slots": 1,
            "idle_slots": 0 if busy else 1,
        })

async def metrics(request):
    with stats_lock:
        pt = int(stats.get("prompt_tokens_total") or 0)
        ct = int(stats.get("completion_tokens_total") or 0)
        ctx = stats.get("context_length")
    # llama-swap expects llama-server style metrics (Cached/Prompt/Generated/Prefill/Decode)
    # We map prompt→Prefill+Prompt, completion→Decode+Generated, context→Cached
    return web.json_response({
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
        "cached_tokens": ctx or 0,
        "prompt": pt,
        "generated": ct,
        "prefill_tokens": pt,
        "decode_tokens": ct,
        "cached": ctx or 0,
        "context_length": ctx,
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
            text, calls, finish, ptoks, otoks, reasoning, content = await asyncio.to_thread(
                generate_full, generator, tokenizer, req["messages"],
                req["max_tokens"], req["temperature"], req["top_p"], req["top_k"],
                req["seed"], req["tools"], req["tool_choice"], req["stop"],
                None, req.get("reasoning"), req.get("chat_template_kwargs"))
        except AssertionError as e:
            return web.json_response(
                {"error": {"message": f"context/cache: {e}", "type": "invalid_request_error"}},
                status=400)
        msg = {"role": "assistant", "content": content or None}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if calls:
            msg["tool_calls"] = calls
        resp = web.json_response({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion", "created": int(time.time()),
            "model": req["model_id"],
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": ptoks, "completion_tokens": otoks,
                      "total_tokens": ptoks + otoks},
        })
        # llama-swap token tracking via headers (Cached/Prompt/Generated/Prefill/Decode)
        resp.headers["X-Prompt-Tokens"] = str(ptoks)
        resp.headers["X-Completion-Tokens"] = str(otoks)
        resp.headers["X-Total-Tokens"] = str(ptoks + otoks)
        resp.headers["X-Cached-Tokens"] = str(stats.get("context_length") or 0)
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
                text, calls, finish, ptoks, otoks, reasoning, content = generate_full(
                    generator, tokenizer, req["messages"], req["max_tokens"],
                    req["temperature"], req["top_p"], req["top_k"],
                    req["seed"], req["tools"], req["tool_choice"], req["stop"],
                    on_text=None if forced_choice else on_text,
                    reasoning=req.get("reasoning"), chat_template_kwargs=req.get("chat_template_kwargs"))
                loop.call_soon_threadsafe(queue.put_nowait,
                                          ("done", (calls, finish, reasoning, content)))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
        loop.run_in_executor(None, worker)

        async def send(delta, finish=None):
            obj = {"id": cid, "object": "chat.completion.chunk",
                   "created": int(time.time()), "model": model_id,
                   "choices": [{"index": 0, "delta": delta,
                                "finish_reason": finish}]}
            await resp.write(f"data: {json.dumps(obj)}\n\n".encode())

        pending, finish, calls_emitted = "", None, False
        call_idx = [0]
        in_think = [True]
        THINK_CLOSE = "\u003c/think\u003e"

        async def send_call(c):
            nonlocal calls_emitted
            calls_emitted = True
            await send({"tool_calls": [dict(c, index=call_idx[0])]})
            call_idx[0] += 1

        async def flush_pending(final=False):
            nonlocal pending
            while True:
                if in_think[0]:
                    close = pending.find(THINK_CLOSE)
                    if close >= 0:
                        head, pending = pending[:close], pending[close + len(THINK_CLOSE):]
                        if head.strip():
                            await send({"reasoning_content": head.lstrip("\n")})
                        in_think[0] = False
                        continue
                    cut = len(pending) if final else max(0, len(pending) - HOLD_BACK)
                    piece = pending[:cut]
                    if piece.strip():
                        await send({"reasoning_content": piece})
                    pending = pending[cut:]
                    return
                if TOOL_CALL_OPEN in pending:
                    head, rest = pending.split(TOOL_CALL_OPEN, 1)
                    if head.strip() or (final and head):
                        await send({"content": head})
                    if TOOL_CALL_CLOSE in rest:
                        block, pending = rest.split(TOOL_CALL_CLOSE, 1)
                        _, calls = parse_tool_calls(
                            TOOL_CALL_OPEN + block + TOOL_CALL_CLOSE,
                            req_schemas)
                        for c in calls:
                            await send_call(c)
                        continue
                    if final and "\u003cfunction=" in rest:
                        _, calls = parse_tool_calls(TOOL_CALL_OPEN + rest,
                                                    req_schemas)
                        for c in calls:
                            await send_call(c)
                        pending = ""
                    else:
                        pending = TOOL_CALL_OPEN + rest
                    return
                cut = len(pending) if final else max(0, len(pending) - HOLD_BACK)
                await send({"content": pending[:cut]})
                pending = pending[cut:]
                return

        while True:
            kind, payload = await queue.get()
            if kind == "error":
                await resp.write(
                    f'data: {json.dumps({"error": {"message": payload}})}\n\n'.encode())
                break
            if kind == "delta":
                pending += payload
                await flush_pending()
            elif kind == "done":
                calls, finish, reasoning, content = payload
                await flush_pending(final=True)
                if forced_choice:
                    if reasoning:
                        await send({"reasoning_content": reasoning})
                    if content:
                        await send({"content": content})
                if not calls_emitted and calls:
                    for c in calls:
                        await send_call(c)
                await send({}, finish=finish)
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

    print(f" == loading {args.model}"
          + (" + MTP head" if use_mtp else
             (f" + draft {args.draft_model}" if use_draft else " (no draft)"))
          + " ...", flush=True)
    generator, tokenizer = build_model(argv, use_draft=use_draft)
    stats["context_length"] = int(args.cache_size)
    print(" == model ready; accepting requests", flush=True)

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
            n_past = (pt + ct) % ctx if ctx else 0
        # llama-swap UI expects Prefill/Decode (llama-server slots: n_prompt_tokens, n_generated, t_prompt_ms, t_eval_ms)
        return web.json_response([{
            "id": 0, "n_ctx": ctx, "n_past": n_past, "is_processing": busy,
            "prompt_tokens": pt, "generated_tokens": ct, "cached_tokens": n_past,
            "n_prompt_tokens": pt, "n_generated": ct, "n_tokens": pt + ct,
            "prefill_tokens": pt, "decode_tokens": ct,
            "total_tokens": pt + ct, "tokens_evaluated": pt, "tokens_generated": ct,
        }])

    async def props(request):
        with stats_lock:
            ctx = stats.get("context_length") or 0
        return web.json_response({
            "total_slots": 1, "default_generation_settings": {"n_ctx": ctx, "n_predict": -1},
            "n_ctx": ctx, "model": "qwen3.8-27b-exl3-3.5bpw",
        })

    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    app.router.add_get("/slots", slots)
    app.router.add_get("/props", props)
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/v1/chat/completions", chat_completions)
    web.run_app(app, host=args.host, port=args.port, print=None)

if __name__ == "__main__":
    main()
