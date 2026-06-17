"""Tests for command_router framework - generic manager delegation pattern."""
import unittest
from unittest.mock import MagicMock, patch, call
import argparse
import os
import io
from contextlib import redirect_stdout
from pathlib import Path


class CommandRouterOwnershipTest(unittest.TestCase):
    """Test ownership detection for catalog commands."""

    @patch("os.getuid")
    @patch("os.stat")
    def test_is_owner_when_uid_zero(self, mock_stat, mock_getuid):
        """Process with uid 0 (root) is always owner."""
        from llamacpp_stack.command_router import is_catalog_owner
        
        mock_getuid.return_value = 0
        result = is_catalog_owner(Path("/tmp/catalog.json"))
        
        self.assertTrue(result)
        # stat should not be called for root
        mock_stat.assert_not_called()

    @patch("os.getuid")
    @patch("os.stat")
    def test_is_owner_when_uid_matches_catalog_owner(self, mock_stat, mock_getuid):
        """Process with uid matching catalog parent owner is owner."""
        from llamacpp_stack.command_router import is_catalog_owner
        
        mock_getuid.return_value = 1000
        mock_stat.return_value.st_uid = 1000
        
        result = is_catalog_owner(Path("/tmp/catalog.json"))
        
        self.assertTrue(result)
        mock_stat.assert_called_once()

    @patch("os.getuid")
    @patch("os.stat")
    def test_is_not_owner_when_uid_differs(self, mock_stat, mock_getuid):
        """Process with different uid from catalog owner is not owner."""
        from llamacpp_stack.command_router import is_catalog_owner
        
        mock_getuid.return_value = 1000
        mock_stat.return_value.st_uid = 2000  # Different owner
        
        result = is_catalog_owner(Path("/tmp/catalog.json"))
        
        self.assertFalse(result)

    @patch("os.getuid")
    @patch("os.stat")
    def test_is_not_owner_when_stat_fails(self, mock_stat, mock_getuid):
        """Stat failure returns False (fail-safe allows delegation)."""
        from llamacpp_stack.command_router import is_catalog_owner
        
        mock_getuid.return_value = 1000
        mock_stat.side_effect = OSError("Permission denied")
        
        result = is_catalog_owner(Path("/tmp/catalog.json"))
        
        self.assertFalse(result)


class CommandRouterDelegationTest(unittest.TestCase):
    """Test manager delegation routing."""

    def setUp(self):
        self.mock_args = argparse.Namespace(
            catalog=Path("/tmp/test_catalog.json"),
            config=Path("/tmp/test_config.json"),
            models_dir=Path("/tmp/models"),
            llama_server=Path("/tmp/llama-server"),
        )

    @patch("os.getuid")
    @patch("os.stat")
    def test_execute_locally_when_owner(self, mock_stat, mock_getuid):
        """Owner should execute local_executor directly."""
        from llamacpp_stack.command_router import execute_with_manager_delegation
        
        mock_getuid.return_value = 0
        local_executor = MagicMock(return_value=0)
        
        result = execute_with_manager_delegation(
            command_name="test-command",
            args=self.mock_args,
            local_executor=local_executor,
        )
        
        # Should call local executor
        local_executor.assert_called_once_with(self.mock_args)
        self.assertEqual(result, 0)

    @patch("os.getuid")
    @patch("os.stat")
    def test_delegate_to_manager_when_non_owner(self, mock_stat, mock_getuid):
        """Non-owner should delegate to manager."""
        from llamacpp_stack.command_router import execute_with_manager_delegation
        
        mock_getuid.return_value = 1000
        mock_stat.return_value.st_uid = 2000
        local_executor = MagicMock()
        
        with patch("llamacpp_stack.cli.run_manager_command") as mock_manager:
            mock_manager.return_value = "model-id"
            
            result = execute_with_manager_delegation(
                command_name="test-command",
                args=self.mock_args,
                local_executor=local_executor,
            )
            
            # Should NOT call local executor
            local_executor.assert_not_called()
            # Should call manager
            mock_manager.assert_called_once_with("test-command", self.mock_args)
            self.assertEqual(result, "model-id")

    @patch("os.getuid")
    @patch("os.stat")
    def test_reraise_runtime_error_from_manager(self, mock_stat, mock_getuid):
        """RuntimeError from manager should be re-raised."""
        from llamacpp_stack.command_router import execute_with_manager_delegation
        
        mock_getuid.return_value = 1000
        mock_stat.return_value.st_uid = 2000
        local_executor = MagicMock()
        
        with patch("llamacpp_stack.cli.run_manager_command") as mock_manager:
            mock_manager.side_effect = RuntimeError("Manager failed")
            
            with self.assertRaises(RuntimeError) as ctx:
                execute_with_manager_delegation(
                    command_name="test-command",
                    args=self.mock_args,
                    local_executor=local_executor,
                )
            self.assertIn("Manager failed", str(ctx.exception))

    @patch("os.getuid")
    @patch("os.stat")
    def test_wrap_non_runtime_error_from_manager(self, mock_stat, mock_getuid):
        """Non-RuntimeError from manager should be wrapped."""
        from llamacpp_stack.command_router import execute_with_manager_delegation
        
        mock_getuid.return_value = 1000
        mock_stat.return_value.st_uid = 2000
        local_executor = MagicMock()
        
        with patch("llamacpp_stack.cli.run_manager_command") as mock_manager:
            with patch("llamacpp_stack.cli.MANAGER_SERVICE_NAME", "test-service"):
                mock_manager.side_effect = ConnectionError("Socket closed")
                
                with self.assertRaises(RuntimeError) as ctx:
                    execute_with_manager_delegation(
                        command_name="test-command",
                        args=self.mock_args,
                        local_executor=local_executor,
                    )
                self.assertIn("Could not connect to manager", str(ctx.exception))
                self.assertIn("test-service", str(ctx.exception))


class CommandRouterPreservesResultTest(unittest.TestCase):
    """Test that various result types are preserved through delegation."""

    def setUp(self):
        self.mock_args = argparse.Namespace(
            catalog=Path("/tmp/test_catalog.json"),
            config=Path("/tmp/test_config.json"),
            models_dir=Path("/tmp/models"),
            llama_server=Path("/tmp/llama-server"),
        )

    @patch("os.getuid")
    def test_preserve_int_result(self, mock_getuid):
        """Integer results should be preserved."""
        from llamacpp_stack.command_router import execute_with_manager_delegation
        
        mock_getuid.return_value = 0
        local_executor = MagicMock(return_value=42)
        
        result = execute_with_manager_delegation(
            command_name="test",
            args=self.mock_args,
            local_executor=local_executor,
        )
        
        self.assertEqual(result, 42)

    @patch("os.getuid")
    def test_preserve_string_result(self, mock_getuid):
        """String results should be preserved."""
        from llamacpp_stack.command_router import execute_with_manager_delegation
        
        mock_getuid.return_value = 0
        local_executor = MagicMock(return_value="model-123")
        
        result = execute_with_manager_delegation(
            command_name="test",
            args=self.mock_args,
            local_executor=local_executor,
        )
        
        self.assertEqual(result, "model-123")

    @patch("os.getuid")
    @patch("os.stat")
    def test_preserve_dict_result_from_manager(self, mock_stat, mock_getuid):
        """Dict/complex results from manager should be preserved."""
        from llamacpp_stack.command_router import execute_with_manager_delegation
        
        mock_getuid.return_value = 1000
        mock_stat.return_value.st_uid = 2000
        
        expected_result = {"status": "ok", "data": [1, 2, 3]}
        
        with patch("llamacpp_stack.cli.run_manager_command") as mock_manager:
            mock_manager.return_value = expected_result
            
            result = execute_with_manager_delegation(
                command_name="test",
                args=self.mock_args,
                local_executor=MagicMock(),
            )
            
            self.assertEqual(result, expected_result)


class DaemonHandlerPrepareTest(unittest.TestCase):
    """Test daemon command handler preparation."""

    @patch("llamacpp_stack.cli.DEFAULT_SERVER_CONFIG_PATH", "/etc/server.json")
    def test_prepare_daemon_handler_reconstructs_paths(self):
        """Daemon handler should reconstruct Path objects from JSON strings."""
        from llamacpp_stack.command_router import prepare_daemon_handler_for_command
        
        req = {
            "command": "test",
            "args": {
                "catalog": "/tmp/catalog.json",
                "config": "/tmp/config.json",
                "models_dir": "/tmp/models",
                "llama_server": "/tmp/llama-server",
                "repo": "test-repo",
                "service": "llamacpp",
            }
        }
        send_event = MagicMock()
        sock_in = MagicMock()
        executor = MagicMock(return_value=0)
        
        result = prepare_daemon_handler_for_command(
            command_name="test",
            req=req,
            send_event=send_event,
            sock_in=sock_in,
            daemon_executor=executor,
        )
        
        # Verify executor was called with reconstructed args
        executor.assert_called_once()
        args_received = executor.call_args[0][0]
        
        self.assertEqual(args_received.catalog, Path("/tmp/catalog.json"))
        self.assertEqual(args_received.config, Path("/tmp/config.json"))
        self.assertEqual(args_received.models_dir, Path("/tmp/models"))
        self.assertEqual(args_received.llama_server, Path("/tmp/llama-server"))
        self.assertEqual(args_received.repo, "test-repo")

    def test_prepare_daemon_handler_passes_event_callback_when_needs_callbacks(self):
        """Daemon handler should pass callbacks when executor_needs_callbacks=True."""
        from llamacpp_stack.command_router import prepare_daemon_handler_for_command
        
        req = {
            "command": "test",
            "args": {
                "catalog": "/tmp/catalog.json",
                "config": "/tmp/config.json",
                "models_dir": "/tmp/models",
                "llama_server": "/tmp/llama-server",
            }
        }
        send_event = MagicMock()
        sock_in = MagicMock()
        executor = MagicMock(return_value=0)
        
        prepare_daemon_handler_for_command(
            command_name="test",
            req=req,
            send_event=send_event,
            sock_in=sock_in,
            daemon_executor=executor,
            executor_needs_callbacks=True,
        )
        
        # Verify send_event and sock_in were passed to executor
        self.assertEqual(executor.call_args[0][1], send_event)
        self.assertEqual(executor.call_args[0][2], sock_in)

    def test_prepare_daemon_handler_no_callbacks_when_not_needed(self):
        """Daemon handler should NOT pass callbacks when executor_needs_callbacks=False."""
        from llamacpp_stack.command_router import prepare_daemon_handler_for_command
        
        req = {
            "command": "test",
            "args": {
                "catalog": "/tmp/catalog.json",
                "config": "/tmp/config.json",
                "models_dir": "/tmp/models",
                "llama_server": "/tmp/llama-server",
            }
        }
        send_event = MagicMock()
        sock_in = MagicMock()
        executor = MagicMock(return_value=0)
        
        prepare_daemon_handler_for_command(
            command_name="test",
            req=req,
            send_event=send_event,
            sock_in=sock_in,
            daemon_executor=executor,
            executor_needs_callbacks=False,  # Default
        )
        
        # Executor should only receive args (one positional argument)
        self.assertEqual(len(executor.call_args[0]), 1)
        # Should NOT receive send_event or sock_in
        self.assertNotIn(send_event, executor.call_args[0])
        self.assertNotIn(sock_in, executor.call_args[0])


class InteractiveCallbackTest(unittest.TestCase):
    """Test interactive question callback creation."""

    def test_create_interactive_callback(self):
        """Interactive callback should send event and return answer."""
        from llamacpp_stack.command_router import create_interactive_callback
        
        send_event = MagicMock(return_value="yes")
        sock_in = MagicMock()
        
        callback = create_interactive_callback(send_event, sock_in)
        
        result = callback("Should I continue?", "n")
        
        self.assertEqual(result, "yes")
        send_event.assert_called_once_with({
            "type": "question",
            "prompt": "Should I continue?",
            "default": "n"
        })
        
        
class ManagerCommandOutputTest(unittest.TestCase):
    def test_auto_performance_prints_immediate_next_step_message(self):
        from llamacpp_stack.cli import run_manager_command
        import json

        args = argparse.Namespace(repo="model", catalog=Path("/tmp/catalog.json"))
        sock = MagicMock()
        sock.__enter__.return_value = sock
        sock.__exit__.return_value = False
        sock.makefile.return_value.__enter__.return_value = sock.makefile.return_value
        sock.makefile.return_value.readline.side_effect = [json.dumps({"type": "done", "result": 0}) + "\n", ""]

        with patch("socket.socket", return_value=sock):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                result = run_manager_command("auto-performance", args)

        output = buffer.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("Request sent to background manager", output)
        self.assertIn("auto-performance ─ request queued; next: manager resolves baseline and starts tuning", output)

    def test_callback_handles_none_response(self):
        """Callback should handle None response from send_event."""
        from llamacpp_stack.command_router import create_interactive_callback
        
        send_event = MagicMock(return_value=None)
        sock_in = MagicMock()
        
        callback = create_interactive_callback(send_event, sock_in)
        result = callback("Question?", "n")
        
        self.assertEqual(result, "")


class PassThroughArgsTest(unittest.TestCase):
    """Test that args are passed unchanged through router."""

    def setUp(self):
        self.mock_args = argparse.Namespace(
            catalog=Path("/tmp/catalog.json"),
            config=Path("/tmp/config.json"),
            models_dir=Path("/tmp/models"),
            llama_server=Path("/tmp/llama-server"),
            custom_field="custom_value",
            nested={"key": "value"},
        )

    @patch("os.getuid")
    def test_args_passed_unchanged_to_local_executor(self, mock_getuid):
        """Args should be passed unchanged to local executor."""
        from llamacpp_stack.command_router import execute_with_manager_delegation
        
        mock_getuid.return_value = 0
        local_executor = MagicMock(return_value=0)
        
        execute_with_manager_delegation(
            command_name="test",
            args=self.mock_args,
            local_executor=local_executor,
        )
        
        # Verify exact args object was passed
        called_args = local_executor.call_args[0][0]
        self.assertIs(called_args, self.mock_args)
        self.assertEqual(called_args.custom_field, "custom_value")
        self.assertEqual(called_args.nested["key"], "value")

    @patch("os.getuid")
    @patch("os.stat")
    def test_args_passed_unchanged_to_manager(self, mock_stat, mock_getuid):
        """Args should be passed unchanged to manager."""
        from llamacpp_stack.command_router import execute_with_manager_delegation
        
        mock_getuid.return_value = 1000
        mock_stat.return_value.st_uid = 2000
        
        with patch("llamacpp_stack.cli.run_manager_command") as mock_manager:
            mock_manager.return_value = 0
            
            execute_with_manager_delegation(
                command_name="test",
                args=self.mock_args,
                local_executor=MagicMock(),
            )
            
            # Verify exact args object was passed to manager
            called_args = mock_manager.call_args[0][1]
            self.assertIs(called_args, self.mock_args)


if __name__ == "__main__":
    unittest.main()
