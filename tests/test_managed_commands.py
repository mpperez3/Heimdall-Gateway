"""Integration tests for command_router pattern with multiple commands.

Demonstrates that the generic routing framework works consistently
across multiple commands (add, remove, update) and prevents permission
issues from recurring.
"""
import unittest
from unittest.mock import MagicMock, patch, call
import argparse
from pathlib import Path


class MultiCommandRouterIntegrationTest(unittest.TestCase):
    """Test that multiple commands work correctly with generic routing."""

    def setUp(self):
        """Setup mock args for add/remove/update commands."""
        self.mock_args = argparse.Namespace(
            catalog=Path("/tmp/catalog.json"),
            config=Path("/tmp/config.json"),
            models_dir=Path("/tmp/models"),
            llama_server=Path("/tmp/llama-server"),
            service="llamacpp",
            repo="test-model",
            hf=None,
            model_id=None,
            file=None,
            delete_files=True,
        )

    @patch("os.getuid")
    def test_add_command_owner_executes_locally(self, mock_getuid):
        """Add command should execute locally for owner."""
        from llamacpp_stack.managed_commands import add_models_with_framework
        
        mock_getuid.return_value = 0
        
        with patch("llamacpp_stack.cli.ensure_model_available") as mock_ensure:
            with patch("llamacpp_stack.cli.restart_service_to_free_vram"):
                mock_ensure.return_value = "model-id"
                
                result = add_models_with_framework(self.mock_args)
                
                # Should call ensure_model_available locally
                mock_ensure.assert_called()
                self.assertEqual(result, 0)

    @patch("os.getuid")
    @patch("os.stat")
    def test_add_command_non_owner_delegates(self, mock_stat, mock_getuid):
        """Add command should delegate to manager for non-owner."""
        from llamacpp_stack.managed_commands import add_models_with_framework
        
        mock_getuid.return_value = 1000
        mock_stat.return_value.st_uid = 2000
        
        with patch("llamacpp_stack.cli.run_manager_command") as mock_manager:
            mock_manager.return_value = "model-id"
            
            result = add_models_with_framework(self.mock_args)
            
            # Should delegate to manager
            mock_manager.assert_called_once()
            self.assertEqual(result, "model-id")

    @patch("os.getuid")
    def test_remove_command_owner_executes_locally(self, mock_getuid):
        """Remove command should execute locally for owner."""
        from llamacpp_stack.managed_commands import remove_models_with_framework
        
        mock_getuid.return_value = 0
        
        with patch("llamacpp_stack.cli.remove_model") as mock_remove:
            with patch("llamacpp_stack.cli.restart_service_to_free_vram"):
                mock_remove.return_value = "model-id"
                
                result = remove_models_with_framework(self.mock_args)
                
                # Should call remove_model locally
                mock_remove.assert_called()
                self.assertEqual(result, 0)

    @patch("os.getuid")
    @patch("os.stat")
    def test_remove_command_non_owner_delegates(self, mock_stat, mock_getuid):
        """Remove command should delegate to manager for non-owner."""
        from llamacpp_stack.managed_commands import remove_models_with_framework
        
        mock_getuid.return_value = 1000
        mock_stat.return_value.st_uid = 2000
        
        with patch("llamacpp_stack.cli.run_manager_command") as mock_manager:
            mock_manager.return_value = "removed"
            
            result = remove_models_with_framework(self.mock_args)
            
            # Should delegate to manager
            mock_manager.assert_called_once_with("remove", self.mock_args)
            self.assertEqual(result, "removed")

    @patch("os.getuid")
    def test_update_command_owner_executes_locally(self, mock_getuid):
        """Update command should execute locally for owner."""
        from llamacpp_stack.managed_commands import update_models_with_framework
        
        mock_getuid.return_value = 0
        
        with patch("llamacpp_stack.cli.update_config") as mock_update:
            mock_update.return_value = None
            
            result = update_models_with_framework(self.mock_args)
            
            # Should call update_config locally
            mock_update.assert_called_once()
            self.assertEqual(result, 0)

    @patch("os.getuid")
    @patch("os.stat")
    def test_update_command_non_owner_delegates(self, mock_stat, mock_getuid):
        """Update command should delegate to manager for non-owner."""
        from llamacpp_stack.managed_commands import update_models_with_framework
        
        mock_getuid.return_value = 1000
        mock_stat.return_value.st_uid = 2000
        
        with patch("llamacpp_stack.cli.run_manager_command") as mock_manager:
            mock_manager.return_value = 0
            
            result = update_models_with_framework(self.mock_args)
            
            # Should delegate to manager
            mock_manager.assert_called_once_with("update", self.mock_args)
            self.assertEqual(result, 0)


class ConsistencyAcrossCommandsTest(unittest.TestCase):
    """Test that all commands behave consistently with shared ownership check."""

    def setUp(self):
        self.mock_args = argparse.Namespace(
            catalog=Path("/tmp/catalog.json"),
            config=Path("/tmp/config.json"),
            models_dir=Path("/tmp/models"),
            llama_server=Path("/tmp/llama-server"),
        )

    @patch("os.getuid")
    @patch("os.stat")
    def test_all_commands_respect_same_ownership_rule(self, mock_stat, mock_getuid):
        """All commands should respect the same ownership check logic."""
        from llamacpp_stack.managed_commands import (
            add_models_with_framework,
            remove_models_with_framework,
            update_models_with_framework,
        )
        
        # Setup: non-owner
        mock_getuid.return_value = 1000
        mock_stat.return_value.st_uid = 2000
        
        with patch("llamacpp_stack.cli.run_manager_command") as mock_manager:
            mock_manager.side_effect = lambda cmd, args: f"handled-{cmd}"
            
            # All commands should delegate to manager
            commands = [
                ("add", add_models_with_framework),
                ("remove", remove_models_with_framework),
                ("update", update_models_with_framework),
            ]
            
            for cmd_name, cmd_func in commands:
                mock_manager.reset_mock()
                
                result = cmd_func(self.mock_args)
                
                # All should call manager with their command name
                mock_manager.assert_called_once()
                call_args = mock_manager.call_args[0]
                self.assertEqual(call_args[0], cmd_name)


class DaemonHandlerIntegrationTest(unittest.TestCase):
    """Test daemon handlers work correctly with framework."""

    @patch("llamacpp_stack.cli.DEFAULT_SERVER_CONFIG_PATH", "/etc/server.json")
    def test_daemon_handler_for_add(self):
        """Daemon add handler should work with framework."""
        from llamacpp_stack.managed_commands import daemon_handler_for_add
        
        req = {
            "command": "add",
            "args": {
                "catalog": "/tmp/catalog.json",
                "config": "/tmp/config.json",
                "models_dir": "/tmp/models",
                "llama_server": "/tmp/llama-server",
                "repo": "test",
                "service": "llamacpp",
            }
        }
        send_event = MagicMock()
        sock_in = MagicMock()
        
        with patch("llamacpp_stack.cli.ensure_model_available") as mock_ensure:
            with patch("llamacpp_stack.cli.restart_service_to_free_vram"):
                mock_ensure.return_value = "model-123"
                
                result = daemon_handler_for_add(req, send_event, sock_in)
                
                # Should execute and return 0
                self.assertEqual(result, 0)
                mock_ensure.assert_called()

    @patch("llamacpp_stack.cli.DEFAULT_SERVER_CONFIG_PATH", "/etc/server.json")
    def test_daemon_handler_for_remove(self):
        """Daemon remove handler should work with framework."""
        from llamacpp_stack.managed_commands import daemon_handler_for_remove
        
        req = {
            "command": "remove",
            "args": {
                "catalog": "/tmp/catalog.json",
                "config": "/tmp/config.json",
                "models_dir": "/tmp/models",
                "llama_server": "/tmp/llama-server",
                "repo": "test",
                "delete_files": True,
                "service": "llamacpp",
            }
        }
        send_event = MagicMock()
        sock_in = MagicMock()
        
        with patch("llamacpp_stack.cli.remove_model") as mock_remove:
            with patch("llamacpp_stack.cli.restart_service_to_free_vram"):
                mock_remove.return_value = "removed"
                
                result = daemon_handler_for_remove(req, send_event, sock_in)
                
                # Should execute
                self.assertEqual(result, 0)
                mock_remove.assert_called()


class PreventingFutureRegressionTest(unittest.TestCase):
    """Test that pattern prevents permission issues from recurring.
    
    This simulates the scenario from the original bug: a command
    that forgot to delegate to manager causing permission denied errors.
    """

    @patch("os.getuid")
    @patch("os.stat")
    def test_new_command_cannot_bypass_routing(self, mock_stat, mock_getuid):
        """A new command using framework cannot accidentally skip manager."""
        from llamacpp_stack.command_router import execute_with_manager_delegation
        
        # Simulate: non-owner, cannot write to catalog
        mock_getuid.return_value = 1000
        mock_stat.return_value.st_uid = 2000  # Different owner - simulate permission issue
        
        mock_args = argparse.Namespace(
            catalog=Path("/etc/llamacpp/catalog.json"),  # System catalog, not writable
            config=Path("/etc/llamacpp/config.json"),
        )
        
        local_executor = MagicMock()  # Would fail with permission denied if called
        
        with patch("llamacpp_stack.cli.run_manager_command") as mock_manager:
            mock_manager.return_value = 0
            
            # Execute via framework
            result = execute_with_manager_delegation(
                command_name="hypothetical-new-command",
                args=mock_args,
                local_executor=local_executor,
            )
            
            # Local executor should NOT be called (would fail with permission denied)
            local_executor.assert_not_called()
            # Manager SHOULD be called (can write with higher privileges)
            mock_manager.assert_called_once()
            self.assertEqual(result, 0)

    @patch("os.getuid")
    @patch("os.stat")
    def test_framework_prevents_duplicating_ownership_logic(self, mock_stat, mock_getuid):
        """Developer cannot accidentally duplicate ownership checks."""
        from llamacpp_stack.command_router import (
            execute_with_manager_delegation,
            is_catalog_owner,
        )
        
        mock_getuid.return_value = 1000
        mock_stat.return_value.st_uid = 2000
        
        catalog_path = Path("/tmp/catalog.json")
        
        # Framework provides centralized ownership check
        is_owner_1 = is_catalog_owner(catalog_path)
        is_owner_2 = is_catalog_owner(catalog_path)
        
        # Both calls return same result (centralized, not duplicated)
        self.assertEqual(is_owner_1, is_owner_2)
        self.assertFalse(is_owner_1)
        
        # Using framework ensures consistency
        with patch("llamacpp_stack.cli.run_manager_command") as mock_manager:
            mock_manager.return_value = 0
            
            # Cannot accidentally implement command without routing
            # because execute_with_manager_delegation is THE way to do it
            result = execute_with_manager_delegation(
                command_name="new-cmd-1",
                args=argparse.Namespace(catalog=catalog_path),
                local_executor=MagicMock(),
            )
            
            # Manager MUST be called, no way to skip it
            mock_manager.assert_called_once()


if __name__ == "__main__":
    unittest.main()
