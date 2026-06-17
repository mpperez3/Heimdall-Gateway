"""Tests for auto-performance runner with manager delegation and ownership checks."""
import unittest
from unittest.mock import MagicMock, patch, call
import argparse
import os
from pathlib import Path


class AutoPerfRunnerTest(unittest.TestCase):
    """Test auto-performance runner routing: owner vs manager modes."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_args = argparse.Namespace(
            repo="test-model",
            hf=None,
            model_id=None,
            file=None,
            catalog=Path("/tmp/test_catalog.json"),
            config=Path("/tmp/test_config.json"),
            models_dir=Path("/tmp/models"),
            llama_server=Path("/tmp/llama-server"),
            server_config=Path("/tmp/server_config.json"),
            server_api=False,
            mock=True,
            load_concurrency=1,
            load_requests=1,
        )

    @patch("os.stat")
    @patch("os.getuid")
    def test_ownership_check_is_owner_uid_zero(self, mock_getuid, mock_stat):
        """When uid is 0 (root), should execute locally without manager delegation."""
        mock_getuid.return_value = 0
        
        from llamacpp_stack.auto_perf_runner import run_auto_perf_command
        
        with patch("llamacpp_stack.auto_performance.run_auto_performance") as mock_run_auto:
            mock_run_auto.return_value = 0
            result = run_auto_perf_command(self.mock_args)
            
            # Should call local run_auto_performance
            mock_run_auto.assert_called_once_with(self.mock_args)
            self.assertEqual(result, 0)

    @patch("os.stat")
    @patch("os.getuid")
    def test_ownership_check_is_owner_catalog_owner(self, mock_getuid, mock_stat):
        """When uid matches catalog parent owner, should execute locally."""
        # Current user owns catalog parent
        mock_getuid.return_value = 1000
        mock_stat.return_value.st_uid = 1000
        
        from llamacpp_stack.auto_perf_runner import run_auto_perf_command
        
        with patch("llamacpp_stack.auto_performance.run_auto_performance") as mock_run_auto:
            mock_run_auto.return_value = 0
            result = run_auto_perf_command(self.mock_args)
            
            # Should call local run_auto_performance
            mock_run_auto.assert_called_once_with(self.mock_args)
            self.assertEqual(result, 0)

    @patch("os.stat")
    @patch("os.getuid")
    def test_ownership_check_not_owner_delegates_to_manager(self, mock_getuid, mock_stat):
        """When uid doesn't match catalog owner, should delegate to manager."""
        # Different user
        mock_getuid.return_value = 1000
        mock_stat.return_value.st_uid = 2000  # Different owner
        
        from llamacpp_stack.auto_perf_runner import run_auto_perf_command
        
        with patch("llamacpp_stack.cli.run_manager_command") as mock_manager:
            mock_manager.return_value = "model-id"
            result = run_auto_perf_command(self.mock_args)
            
            # Should call manager with "auto-performance" command
            mock_manager.assert_called_once_with("auto-performance", self.mock_args)
            self.assertEqual(result, "model-id")

    @patch("os.stat")
    @patch("os.getuid")
    def test_ownership_check_raises_on_manager_error(self, mock_getuid, mock_stat):
        """RuntimeError from manager should be re-raised."""
        mock_getuid.return_value = 1000
        mock_stat.return_value.st_uid = 2000
        
        from llamacpp_stack.auto_perf_runner import run_auto_perf_command
        
        with patch("llamacpp_stack.cli.run_manager_command") as mock_manager:
            mock_manager.side_effect = RuntimeError("Manager unavailable")
            
            with self.assertRaises(RuntimeError) as ctx:
                run_auto_perf_command(self.mock_args)
            self.assertIn("Manager unavailable", str(ctx.exception))

    @patch("os.stat")
    @patch("os.getuid")
    def test_ownership_check_converts_exception_to_manager_unavailable(self, mock_getuid, mock_stat):
        """Non-RuntimeError from manager should be wrapped as manager unavailable."""
        mock_getuid.return_value = 1000
        mock_stat.return_value.st_uid = 2000
        
        from llamacpp_stack.auto_perf_runner import run_auto_perf_command
        
        with patch("llamacpp_stack.cli.run_manager_command") as mock_manager:
            mock_manager.side_effect = ConnectionError("Socket closed")
            
            with self.assertRaises(RuntimeError):
                run_auto_perf_command(self.mock_args)

    @patch("os.stat")
    @patch("os.getuid")
    def test_ownership_check_graceful_on_stat_error(self, mock_getuid, mock_stat):
        """When stat() fails, should treat as non-owner and delegate to manager."""
        mock_getuid.return_value = 1000
        mock_stat.side_effect = OSError("stat failed")
        
        from llamacpp_stack.auto_perf_runner import run_auto_perf_command
        
        with patch("llamacpp_stack.cli.run_manager_command") as mock_manager:
            mock_manager.return_value = "model-id"
            result = run_auto_perf_command(self.mock_args)
            
            # Should treat as non-owner and delegate
            mock_manager.assert_called_once()

    @patch("os.stat")
    @patch("os.getuid")
    def test_question_callback_injection_for_manager(self, mock_getuid, mock_stat):
        """When delegating to manager, args should not be modified."""
        mock_getuid.return_value = 0  # Owner, but we'll test args are passed through
        
        from llamacpp_stack.auto_perf_runner import run_auto_perf_command
        
        with patch("llamacpp_stack.auto_performance.run_auto_performance") as mock_run_auto:
            mock_run_auto.return_value = 0
            result = run_auto_perf_command(self.mock_args)
            
            # Verify the exact args object was passed
            called_args = mock_run_auto.call_args[0][0]
            self.assertIs(called_args, self.mock_args)
            
            # No _question_callback should be set in local mode
            self.assertFalse(hasattr(called_args, "_question_callback") and callable(called_args._question_callback))

    def test_daemon_handler_emits_start_message(self):
        from llamacpp_stack.auto_perf_runner import prepare_auto_perf_daemon_handler

        req = {
            "command": "auto-performance",
            "args": {
                "repo": "test-model",
                "catalog": "/tmp/test_catalog.json",
                "config": "/tmp/test_config.json",
                "models_dir": "/tmp/models",
                "llama_server": "/tmp/llama-server",
                "server_config": "/tmp/server_config.json",
            },
        }
        send_event = MagicMock(return_value=None)

        with patch("llamacpp_stack.auto_performance.run_auto_performance", return_value=0) as mock_run_auto:
            result = prepare_auto_perf_daemon_handler(req, send_event, sock_in=object())

        self.assertEqual(result, 0)
        self.assertTrue(send_event.call_args_list)
        self.assertEqual(send_event.call_args_list[0][0][0]["type"], "message")
        self.assertIn("next: resolve baseline and start tuning", send_event.call_args_list[0][0][0]["message"])
        mock_run_auto.assert_called_once()

    def test_daemon_handler_streams_stdout_as_messages(self):
        from llamacpp_stack.auto_perf_runner import prepare_auto_perf_daemon_handler

        req = {
            "command": "auto-performance",
            "args": {
                "repo": "test-model",
                "catalog": "/tmp/test_catalog.json",
                "config": "/tmp/test_config.json",
                "models_dir": "/tmp/models",
                "llama_server": "/tmp/llama-server",
                "server_config": "/tmp/server_config.json",
            },
        }
        send_event = MagicMock(return_value=None)

        def _run_auto_perf(_args):
            print("Baseline ready")
            print("[Trial 1/3] Testing configuration...")
            return 0

        with patch("llamacpp_stack.auto_performance.run_auto_performance", side_effect=_run_auto_perf):
            result = prepare_auto_perf_daemon_handler(req, send_event, sock_in=object())

        self.assertEqual(result, 0)
        messages = [call_args[0][0]["message"] for call_args in send_event.call_args_list if call_args[0][0]["type"] == "message"]
        self.assertTrue(any("Baseline ready" in message for message in messages))
        self.assertTrue(any("[Trial 1/3] Testing configuration..." in message for message in messages))


class AskYesNoTest(unittest.TestCase):
    """Test the _ask_yes_no helper for prompts."""

    def test_ask_yes_no_with_manager_callback(self):
        """Should use callback when provided and callable."""
        from llamacpp_stack.auto_performance import _ask_yes_no
        
        args = argparse.Namespace()
        args._question_callback = MagicMock(return_value="y")
        
        result = _ask_yes_no(args, "Test prompt?", "n")
        
        self.assertTrue(result)
        args._question_callback.assert_called_once_with("Test prompt?", "n")

    def test_ask_yes_no_with_manager_callback_returns_no(self):
        """Should handle 'n' response from callback."""
        from llamacpp_stack.auto_performance import _ask_yes_no
        
        args = argparse.Namespace()
        args._question_callback = MagicMock(return_value="n")
        
        result = _ask_yes_no(args, "Test prompt?", "y")
        
        self.assertFalse(result)

    def test_ask_yes_no_with_manager_callback_exception_falls_back(self):
        """Should fall back to default if callback raises exception."""
        from llamacpp_stack.auto_performance import _ask_yes_no
        
        args = argparse.Namespace()
        args._question_callback = MagicMock(side_effect=RuntimeError("Callback error"))
        
        # Should fall back to default "n"
        result = _ask_yes_no(args, "Test prompt?", "n")
        
        self.assertFalse(result)

    def test_ask_yes_no_defaults_to_yes(self):
        """Should use default 'y' when no callback and empty answer."""
        from llamacpp_stack.auto_performance import _ask_yes_no
        
        args = argparse.Namespace()
        
        with patch("builtins.input", return_value=""):
            result = _ask_yes_no(args, "Test prompt?", "y")
            
        self.assertTrue(result)

    def test_ask_yes_no_defaults_to_no(self):
        """Should use default 'n' when no callback and empty answer."""
        from llamacpp_stack.auto_performance import _ask_yes_no
        
        args = argparse.Namespace()
        
        with patch("builtins.input", return_value=""):
            result = _ask_yes_no(args, "Test prompt?", "n")
            
        self.assertFalse(result)

    def test_ask_yes_no_explicit_yes_response(self):
        """Should accept various yes responses."""
        from llamacpp_stack.auto_performance import _ask_yes_no
        
        args = argparse.Namespace()
        
        for response in ["y", "Y", "yes", "YES", "s", "S", "si", "SI"]:
            with patch("builtins.input", return_value=response):
                result = _ask_yes_no(args, "Test?", "n")
                self.assertTrue(result, f"Failed for response: {response}")

    def test_ask_yes_no_explicit_no_response(self):
        """Should accept various no responses."""
        from llamacpp_stack.auto_performance import _ask_yes_no
        
        args = argparse.Namespace()
        
        for response in ["n", "N", "no", "NO", "nope"]:
            with patch("builtins.input", return_value=response):
                result = _ask_yes_no(args, "Test?", "y")
                self.assertFalse(result, f"Failed for response: {response}")

    def test_ask_yes_no_eof_uses_default(self):
        """Should handle EOF (Ctrl+D) and use default."""
        from llamacpp_stack.auto_performance import _ask_yes_no
        
        args = argparse.Namespace()
        
        with patch("builtins.input", side_effect=EOFError()):
            result = _ask_yes_no(args, "Test?", "n")
            
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
