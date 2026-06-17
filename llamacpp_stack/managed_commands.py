"""Example and reference for using command_router pattern with catalog commands.

This module demonstrates the recommended pattern for any catalog command,
current or future, using the generic command_router framework.

PATTERN:
    Each CLI command should delegate via execute_with_manager_delegation(),
    which automatically routes to manager or local execution based on ownership.

MIGRATION PATH (low-risk):
    1. Create a "_execute_locally(args)" function with your logic
    2. Wrap it with execute_with_manager_delegation()
    3. The router handles ownership checks, manager delegation, and errors
    4. Your code is simpler, cleaner, and guaranteed correct

EXAMPLES BELOW show how add_models, remove_models, and update_models
should be implemented to use the framework. These can be gradually rolled
out without breaking existing supported commands.
"""
import argparse
from pathlib import Path
from typing import Union
from llamacpp_stack.command_router import execute_with_manager_delegation


# ============================================================================
# EXAMPLE 1: How add_models SHOULD be implemented
# ============================================================================

def add_models_with_framework(args: argparse.Namespace) -> int:
    """Add model(s) - refactored example using command_router framework.
    
    This demonstrates the ideal pattern: one line that delegates,
    and a local executor that your actual logic lives in.
    """
    return execute_with_manager_delegation(
        command_name="add",
        args=args,
        local_executor=_add_models_locally,
    )


def _add_models_locally(args: argparse.Namespace) -> int:
    """Local add implementation - moved OUT of command routing.
    
    This function contains only the actual business logic.
    Ownership checks and manager delegation are handled by the router.
    """
    # Import here to avoid circular deps
    from llamacpp_stack.cli import (
        _collect_model_references,
        _clone_namespace,
        _normalize_single_ref_args,
        ensure_model_available,
        restart_service_to_free_vram,
    )
    
    refs = _collect_model_references(args)
    if refs:
        if getattr(args, "model_id", None):
            raise RuntimeError("Use either model references list or --model-id, not both.")
        for ref in refs:
            cloned = _clone_namespace(args)
            cloned.repo = ref
            cloned.hf = None
            cloned.model_id = None
            ensure_model_available(cloned)
        restart_service_to_free_vram(args.service)
        return 0
    
    res = ensure_model_available(_normalize_single_ref_args(args))
    restart_service_to_free_vram(args.service)
    return res and 0


# ============================================================================
# EXAMPLE 2: How remove_models SHOULD be implemented
# ============================================================================

def remove_models_with_framework(args: argparse.Namespace) -> int:
    """Remove model(s) - refactored example using command_router framework."""
    return execute_with_manager_delegation(
        command_name="remove",
        args=args,
        local_executor=_remove_models_locally,
    )


def _remove_models_locally(args: argparse.Namespace) -> int:
    """Local remove implementation."""
    from llamacpp_stack.cli import (
        _collect_model_references,
        _clone_namespace,
        _normalize_single_ref_args,
        remove_model,
        restart_service_to_free_vram,
    )
    
    effective_args = _clone_namespace(args)
    if not hasattr(effective_args, "delete_files"):
        effective_args.delete_files = True

    refs = _collect_model_references(effective_args)
    if refs:
        if getattr(effective_args, "model_id", None) or getattr(effective_args, "file", None):
            raise RuntimeError("Use either a references list or --model-id/--file, not both.")
        for ref in refs:
            cloned = _clone_namespace(effective_args)
            cloned.repo = ref
            cloned.hf = None
            cloned.model_id = None
            cloned.file = None
            remove_model(cloned)
        restart_service_to_free_vram(effective_args.service)
        return 0
    
    res = remove_model(_normalize_single_ref_args(effective_args))
    restart_service_to_free_vram(effective_args.service)
    return res and 0


# ============================================================================
# EXAMPLE 3: How update_config SHOULD be implemented
# ============================================================================

def update_models_with_framework(args: argparse.Namespace) -> int:
    """Update model config - refactored example using command_router framework."""
    return execute_with_manager_delegation(
        command_name="update",
        args=args,
        local_executor=_update_models_locally,
    )


def _update_models_locally(args: argparse.Namespace) -> int:
    """Local update implementation."""
    from llamacpp_stack.cli import update_config
    update_config(args)
    return 0


# ============================================================================
# DAEMON HANDLERS - How these commands execute in daemon context
# ============================================================================

def daemon_handler_for_add(req: dict, send_event, sock_in) -> int:
    """Daemon handler for 'add' command delegated by non-owner."""
    from llamacpp_stack.command_router import prepare_daemon_handler_for_command
    
    return prepare_daemon_handler_for_command(
        command_name="add",
        req=req,
        send_event=send_event,
        sock_in=sock_in,
        daemon_executor=_add_models_locally,
        executor_needs_callbacks=False,
    )


def daemon_handler_for_remove(req: dict, send_event, sock_in) -> int:
    """Daemon handler for 'remove' command delegated by non-owner."""
    from llamacpp_stack.command_router import prepare_daemon_handler_for_command
    
    return prepare_daemon_handler_for_command(
        command_name="remove",
        req=req,
        send_event=send_event,
        sock_in=sock_in,
        daemon_executor=_remove_models_locally,
        executor_needs_callbacks=False,
    )


def daemon_handler_for_update(req: dict, send_event, sock_in) -> int:
    """Daemon handler for 'update' command delegated by non-owner."""
    from llamacpp_stack.command_router import prepare_daemon_handler_for_command
    
    return prepare_daemon_handler_for_command(
        command_name="update",
        req=req,
        send_event=send_event,
        sock_in=sock_in,
        daemon_executor=_update_models_locally,
        executor_needs_callbacks=False,
    )
