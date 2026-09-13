"""Generic manager delegation framework for catalog commands.

This module provides a reusable pattern for commands that must execute in
consistent ownership context:
- Ownership checks (uid vs catalog directory owner)
- Manager delegation for non-owners
- Local execution for owners
- Graceful fallback on permission errors

USAGE PATTERN:
    def my_command(args):
        return execute_with_manager_delegation(
            command_name="my-command",
            args=args,
            local_executor=_execute_locally,
            progress_callback=None
        )
    
    def _execute_locally(args):
        # Your local implementation here
        return 0

This ensures any command, current or future, always executes in the correct
context without developers needing to reimplement ownership checks or manager
integration logic.
"""
import os
import argparse
import json
import re
import threading
from pathlib import Path
from typing import Callable, Any, Union

# ---------------------------------------------------------------------------
# Gateway sanitization (capa 2): normalize_messages + metrics
# Spec: colapsar user vacíos consecutivos y \n{4,} -> \n\n\n,
#       preserve_thinking False, forward thinking_budget,
#       metrics leakage_marker_emitted_total / thinking_truncated_total / resolve_in_think_mismatch
#       log thinking_budget/internal_max/in_think_initial + api_raw_requests.log ring
# ---------------------------------------------------------------------------
_GATEWAY_METRICS_LOCK = threading.Lock()
_GATEWAY_METRICS: dict[str, int] = {
    "leakage_marker_emitted_total": 0,
    "thinking_truncated_total": 0,
    "resolve_in_think_mismatch": 0,
}

_NEWLINE_RE = re.compile(r"\n{4,}")
_MARKER_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>")
# Generic artifact sanitization (capa 2b): evita que {"output": llegue a content
# y colapsa useruser literales que priman degeneración (59× user + output JSON).
_OUTPUT_JSON_RE = re.compile(r'\{\s*"output"\s*:')
_OUTPUT_KEY_RE = re.compile(r'"output"\s*:')
_USERUSER_CONCAT_RE = re.compile(r'(?:user){2,}', flags=re.IGNORECASE)
# líneas consecutivas "user" (con o sin espacios/nuevas líneas) -> una sola
_USER_LINES_RE = re.compile(r'(?:^|\n)[ \t]*user[ \t]*(?:\n[ \t]*user[ \t]*)+', flags=re.IGNORECASE | re.MULTILINE)

def _sanitize_text_newlines(text: str) -> str:
    if not isinstance(text, str):
        return str(text or "")
    return _NEWLINE_RE.sub("\n\n\n", text)

def _sanitize_tool_output_artifacts(text: str) -> str:
    """Elimina {"output": y colapsa useruser/repeticiones literales.

    - {"output": y "output": son marcadores de tool output serializado que
      nunca deberían viajar como user content; se reemplazan por
      [output_stripped] para no primar al modelo.
    - useruser / user\\nuser\\nuser literales son artefactos de transcripts
      corruptos que degeneran Qwen; se colapsan a un único 'user'.
    Genérico y determinista, no reordena mensajes.
    """
    if not isinstance(text, str) or not text:
        return text
    # output JSON fisura
    if '"output"' in text:
        text = _OUTPUT_JSON_RE.sub('[output_stripped]:', text)
        # por si quedó "output": sin llave (p.ej. output ya sin {)
        text = _OUTPUT_KEY_RE.sub('[output_stripped]:', text)
    # useruser concatenado sin espacios (useruseruser...)
    if 'user' in text.lower():
        text = _USERUSER_CONCAT_RE.sub('user', text)
        text = _USER_LINES_RE.sub('\nuser\n', text)
    return text

def _is_empty_user_content(content: object) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return content.strip() == ""
    if isinstance(content, list):
        # list of parts - empty if no text parts with non-empty text
        for item in content:
            if isinstance(item, dict):
                t = item.get("text") or item.get("content") or ""
                if isinstance(t, str) and t.strip():
                    return False
                # image parts count as non-empty? treat as non-empty to avoid collapsing
                if item.get("type") in {"image_url", "input_image"}:
                    return False
            elif isinstance(item, str) and item.strip():
                return False
        return True
    return str(content).strip() == ""

def _sanitize_content_field(content: object) -> object:
    def _sanitize_str(s: str) -> str:
        return _sanitize_text_newlines(_sanitize_tool_output_artifacts(_sanitize_text_newlines(s)))
    if isinstance(content, str):
        return _sanitize_str(content)
    if isinstance(content, list):
        out: list[object] = []
        for item in content:
            if isinstance(item, dict):
                copy = dict(item)
                for key in ("text", "content", "output"):
                    if isinstance(copy.get(key), str):
                        copy[key] = _sanitize_str(copy[key])
                out.append(copy)
            elif isinstance(item, str):
                out.append(_sanitize_str(item))
            else:
                out.append(item)
        return out
    return content

def normalize_messages(messages: list[dict] | None) -> list[dict]:
    """Gateway sanitization: collapse consecutive empty user + \\n{4,} -> \\n\\n\\n.

    Preserves tool_calls etc. Must not break dedup: deterministic, no reordering
    beyond collapsing empties.
    """
    if not isinstance(messages, list):
        return []
    out: list[dict] = []
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        m = dict(raw)
        # sanitize content newlines
        if "content" in m:
            m["content"] = _sanitize_content_field(m.get("content"))
        # image content list -> join text handled elsewhere; here we keep list form
        # collapse consecutive empty user
        is_user = str(m.get("role") or "") == "user"
        is_empty = _is_empty_user_content(m.get("content"))
        if is_user and is_empty:
            if out and str(out[-1].get("role") or "") == "user" and _is_empty_user_content(out[-1].get("content")):
                # collapse: skip this duplicate empty user
                continue
        out.append(m)
    return out

def _ensure_preserve_thinking_false(payload: dict) -> None:
    """Enforce chat_template_kwargs preserve_thinking=False (gateway layer)."""
    if not isinstance(payload, dict):
        return
    ctk = payload.get("chat_template_kwargs")
    # handle string JSON case (llama-server flag style)
    if isinstance(ctk, str):
        try:
            parsed = json.loads(ctk) if ctk.strip().startswith("{") else {}
            if not isinstance(parsed, dict):
                parsed = {}
            parsed["preserve_thinking"] = False
            payload["chat_template_kwargs"] = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
            return
        except Exception:
            payload["chat_template_kwargs"] = json.dumps({"preserve_thinking": False}, ensure_ascii=False)
            return
    if not isinstance(ctk, dict):
        ctk = {}
    ctk["preserve_thinking"] = False
    payload["chat_template_kwargs"] = ctk
    # also handle top-level reasoning ensure we don't override existing but enforce template
    # payload may have chat_template_kwargs as dict -> set

def _forward_thinking_budget(payload: dict) -> None:
    """Ensure thinking_budget_tokens is preserved/forwarded if present; no-op if absent."""
    # gateway must not drop thinking_budget_tokens that was resolved via resolve_request_reasoning_budget
    # This is a no-op keeper to make intent explicit; actual budget is set by caller.
    if not isinstance(payload, dict):
        return
    # normalize alternative key reasoning_budget_tokens -> thinking_budget_tokens
    if "reasoning_budget_tokens" in payload and "thinking_budget_tokens" not in payload:
        try:
            payload["thinking_budget_tokens"] = int(payload.get("reasoning_budget_tokens"))  # type: ignore
        except Exception:
            pass

def sanitize_gateway_payload(payload: dict, messages: list[dict] | None = None) -> list[dict]:
    """Gateway capa 2 sanitization entry: messages + preserve_thinking + thinking_budget."""
    msgs = messages if messages is not None else payload.get("messages") if isinstance(payload, dict) else None
    normalized = normalize_messages(msgs if isinstance(msgs, list) else [])
    if isinstance(payload, dict):
        payload["messages"] = normalized
        _ensure_preserve_thinking_false(payload)
        _forward_thinking_budget(payload)
    return normalized

def inc_gateway_metric(name: str, amount: int = 1) -> None:
    if name not in _GATEWAY_METRICS:
        return
    with _GATEWAY_METRICS_LOCK:
        _GATEWAY_METRICS[name] += int(amount)

def get_gateway_metrics() -> dict[str, int]:
    with _GATEWAY_METRICS_LOCK:
        return dict(_GATEWAY_METRICS)

def reset_gateway_metrics() -> None:
    with _GATEWAY_METRICS_LOCK:
        for k in _GATEWAY_METRICS:
            _GATEWAY_METRICS[k] = 0


def is_catalog_owner(catalog_path: Path) -> bool:
    """Check if current process owns the catalog parent directory.
    
    Returns True if:
    - Process uid is 0 (root)
    - Process uid matches catalog parent directory owner
    - Stat fails (fail-safe to False, allowing delegation)
    
    Args:
        catalog_path: Path to catalog file
    
    Returns:
        True if owner, False otherwise
    """
    try:
        return os.getuid() == 0 or os.getuid() == os.stat(catalog_path.parent).st_uid
    except Exception:
        return False


def manager_unavailable_error(exc: Exception, service_name: str) -> RuntimeError:
    """Format manager unavailable error with helpful context.
    
    Args:
        exc: Original exception from manager
        service_name: Name of manager service (e.g., "heimdall-gateway-manager")
    
    Returns:
        RuntimeError with actionable message
    """
    from llamacpp_stack.cli import manager_hint
    return RuntimeError(
        f"Could not connect to manager: {exc}.\n{manager_hint()}"
    )


def execute_with_manager_delegation(
    command_name: str,
    args: argparse.Namespace,
    local_executor: Callable[[argparse.Namespace], Any],
    progress_callback: Callable | None = None,
) -> Union[int, str]:
    """Execute a catalog command with automatic manager delegation.
    
    Routes command execution based on catalog ownership:
    - If owner (uid 0 or matches catalog parent owner):
        Execute local_executor() directly
    - If non-owner:
        Delegate to manager via Unix socket
    - If stat fails (permission denied):
        Safely delegate to manager
    
    Example:
        def run_my_command(args):
            return execute_with_manager_delegation(
                command_name="my-command",
                args=args,
                local_executor=_run_my_command_locally,
                progress_callback=None
            )
        
        def _run_my_command_locally(args):
            # Your implementation
            return 0
    
    Args:
        command_name: Name of command for manager delegation (e.g., "add", "remove")
        args: Parsed CLI arguments including catalog path
        local_executor: Function to call if owner (receives args, returns result)
        progress_callback: Optional callback for progress updates
    
    Returns:
        Result from local_executor or manager response
    
    Raises:
        RuntimeError: If manager is unavailable and non-owner
    """
    # Ownership check: can we write to catalog?
    if is_catalog_owner(args.catalog):
        # Owner: execute locally
        return local_executor(args)
    
    # Non-owner: delegate to manager
    try:
        from llamacpp_stack.cli import run_manager_command, MANAGER_SERVICE_NAME
        
        return run_manager_command(command_name, args)
    except RuntimeError:
        raise
    except Exception as e:
        from llamacpp_stack.cli import MANAGER_SERVICE_NAME
        raise manager_unavailable_error(e, MANAGER_SERVICE_NAME)


def prepare_daemon_handler_for_command(
    command_name: str,
    req: dict[str, Any],
    send_event: Callable,
    sock_in: Any,
    daemon_executor: Callable,
    executor_needs_callbacks: bool = False,
) -> Union[int, str]:
    """Prepare and execute a catalog command in daemon context.
    
    Reconstructs args from daemon request JSON, applies path conversions,
    and runs the command executor in manager context.
    
    Used by daemon_mode() to handle commands delegated by non-owners.
    
    Args:
        command_name: Name of command (e.g., "add", "remove")
        req: Request dict with "command" and "args" keys
        send_event: Callback to emit events to client
        sock_in: Socket file for receiving question answers
        daemon_executor: Function to execute in daemon context.
                        If executor_needs_callbacks=False (default):
                            Receives (mock_args) only
                        If executor_needs_callbacks=True:
                            Receives (mock_args, send_event, sock_in)
                        Should return result int/str
        executor_needs_callbacks: Whether executor expects (send_event, sock_in) args
    
    Returns:
        Result from daemon_executor
    
    Raises:
        RuntimeError: If command execution fails
    """
    # Reconstruct args from daemon request (JSON)
    mock_args = argparse.Namespace(**req["args"])
    mock_args.catalog = Path(mock_args.catalog)
    mock_args.config = Path(mock_args.config)
    mock_args.models_dir = Path(mock_args.models_dir)
    mock_args.llama_server = Path(mock_args.llama_server)
    
    from llamacpp_stack.cli import DEFAULT_SERVER_CONFIG_PATH
    mock_args.server_config = Path(getattr(mock_args, "server_config", DEFAULT_SERVER_CONFIG_PATH))
    
    # Run executor in daemon context
    if executor_needs_callbacks:
        return daemon_executor(mock_args, send_event, sock_in)
    else:
        return daemon_executor(mock_args)


def create_interactive_callback(send_event: Callable, sock_in: Any) -> Callable:
    """Create a question callback that sends events through manager socket.
    
    Useful for commands that need to ask user questions in manager-delegated mode.
    Injects this callback into args._question_callback so your command can:
    
    Example:
        args._question_callback = create_interactive_callback(send_event, sock_in)
        
        # Later in your command:
        answer = args._question_callback("What to do?", "n")
    
    Args:
        send_event: Callback to emit events to client
        sock_in: Socket file for receiving question answers
    
    Returns:
        Callable that takes (prompt, default) and returns answer string
    """
    def _ask(prompt: str, default: str = "n") -> str:
        """Send question event and wait for answer from client."""
        return send_event({"type": "question", "prompt": prompt, "default": default}) or ""
    
    return _ask
