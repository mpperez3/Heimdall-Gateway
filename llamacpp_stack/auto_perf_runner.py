"""Auto-performance command runner with manager delegation support.

This module encapsulates the routing logic for auto-performance:
- Ownership checks (uid vs catalog owner)
- Manager delegation for non-owners
- Local execution for owners
- Manager event handling (question callback injection)

This ensures auto-performance always executes in the correct context,
preventing permission mismatches on multi-user or service-based setups.
"""
import os
import argparse
from pathlib import Path
from typing import Any, Union
from contextlib import redirect_stdout, redirect_stderr
import io


class _EventStream(io.TextIOBase):
    """Forward text written by the daemon into manager message events."""

    def __init__(self, send_event, *, prefix: str = ""):
        self._send_event = send_event
        self._prefix = prefix
        self._buffer = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                self._send_event({"type": "message", "message": f"{self._prefix}{line}"})
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._send_event({"type": "message", "message": f"{self._prefix}{self._buffer.rstrip()}"})
        self._buffer = ""


def manager_unavailable_error(exc: Exception) -> RuntimeError:
    """Format manager unavailable error with helpful context."""
    from llamacpp_stack.cli import MANAGER_SERVICE_NAME, manager_hint
    return RuntimeError(
        f"Could not connect to manager: {exc}.\n{manager_hint()}"
    )


def run_auto_perf_command(args: argparse.Namespace) -> Union[int, str]:
    """Route auto-performance execution: manager or local.
    
    Checks ownership of catalog parent directory:
    - If owner (uid 0 or matches st_uid): execute locally
    - If non-owner: delegate to manager via Unix socket
    - If stat fails: treat as non-owner (fail-safe to manager)
    
    Args:
        args: Parsed CLI arguments including repo, hf, model_id, file, catalog, etc.
    
    Returns:
        0 on success, or model_id/result string from manager/local execution.
    
    Raises:
        RuntimeError: If manager is unavailable or command fails.
    """
    # Ownership check: are we authorized to write to catalog?
    try:
        is_owner = (os.getuid() == 0 or os.getuid() == os.stat(args.catalog.parent).st_uid)
    except Exception:
        is_owner = False

    if not is_owner:
        # Non-owner: delegate to manager via Unix socket
        try:
            from llamacpp_stack.cli import run_manager_command
            return run_manager_command("auto-performance", args)
        except RuntimeError:
            raise
        except Exception as e:
            raise manager_unavailable_error(e)

    # Owner: execute locally
    from llamacpp_stack.auto_performance import run_auto_performance

    return run_auto_performance(args)


def prepare_auto_perf_daemon_handler(
    req: dict[str, Any], send_event: callable, sock_in: Any
) -> Union[int, str]:
    """Prepare and execute auto-performance in daemon context.
    
    Reconstructs args from daemon request, injects question callback
    for manager event-based prompts, and runs tuner in manager context.
    
    Args:
        req: Request dict with "command" = "auto-performance" and "args" dict
        send_event: Callback to emit events ({"type": ..., ...}) to client
        sock_in: Socket file for receiving question answers
    
    Returns:
        0 on success or result from tuner.
    
    Raises:
        RuntimeError: If auto-performance execution fails.
    """
    from llamacpp_stack.command_router import prepare_daemon_handler_for_command
    
    def _daemon_executor_auto_perf(args, send_event, sock_in):
        """Auto-performance needs callbacks for interactive prompts."""
        from llamacpp_stack.auto_performance import run_auto_performance
        from llamacpp_stack.command_router import create_interactive_callback

        send_event({
            "type": "message",
            "message": "auto-performance: received by manager; next: resolve baseline and start tuning",
        })
        
        # Inject callback so tuner can ask questions via manager events  
        setattr(args, "_question_callback", create_interactive_callback(send_event, sock_in))

        # The tuner prints baseline/trial progress with plain `print()`, so the
        # daemon must forward stdout/stderr as manager events or the client will
        # appear to hang after the initial acceptance message.
        stdout_stream = _EventStream(send_event)
        stderr_stream = _EventStream(send_event, prefix="[stderr] ")

        # Execute in manager context while streaming prints back to the client.
        with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
            result = run_auto_performance(args)

        stdout_stream.flush()
        stderr_stream.flush()
        send_event({"type": "done", "result": result})
        return result
    
    return prepare_daemon_handler_for_command(
        command_name="auto-performance",
        req=req,
        send_event=send_event,
        sock_in=sock_in,
        daemon_executor=_daemon_executor_auto_perf,
        executor_needs_callbacks=True,
    )
