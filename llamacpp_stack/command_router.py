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
from pathlib import Path
from typing import Callable, Any, Union


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
        service_name: Name of manager service (e.g., "llamacpp-superserver-manager")
    
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
