import argparse
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from llamacpp_stack import uninstall as uninstall_mod
from llamacpp_stack.install import InstallLayout


def _make_layout(tmp: Path, mode: str = "user") -> InstallLayout:
    if mode == "user":
        home = tmp / "home"
        install_root = home / ".local/opt/heimdall-gateway"
        state_dir = home / ".local/state/heimdall-gateway"
        config_dir = home / ".config/heimdall-gateway"
        run_dir = home / ".local/run/heimdall-gateway"
        bin_dir = home / ".local/bin"
    else:
        install_root = tmp / "opt/heimdall-gateway"
        state_dir = tmp / "var/lib/heimdall-gateway"
        config_dir = tmp / "etc/heimdall-gateway"
        run_dir = tmp / "run/heimdall-gateway"
        bin_dir = tmp / "usr/local/bin"
    models_dir = tmp / "models"
    for p in (install_root, state_dir, config_dir, run_dir, bin_dir, models_dir):
        p.mkdir(parents=True, exist_ok=True)
    return InstallLayout(
        mode=mode,
        state_dir=state_dir,
        bin_dir=bin_dir,
        install_root=install_root,
        models_dir=models_dir,
        config_dir=config_dir,
        run_dir=run_dir,
        service_user="martin" if mode == "user" else "llamaswap",
        service_group="martin" if mode == "user" else "llamaswap",
        public_host="127.0.0.1",
        public_port=11436,
        manager_socket=run_dir / "manager.sock",
        python_root=install_root / "python",
        runtime_venv=install_root / "venv",
        cuda_root=install_root / "cuda",
        nccl_root=install_root / "nccl",
        backend="auto",
    )


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        mode=None,
        yes=False,
        dry_run=False,
        remove_models=False,
        keep_models=False,
        public_host="127.0.0.1",
        public_port=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class UninstallModelsPreservationTest(unittest.TestCase):
    def test_models_outside_state_dir_are_kept(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        try:
            tmp_path = Path(tmp.name)
            layout = _make_layout(tmp_path)
            marker = layout.state_dir / "catalog.json"
            marker.write_text("[]", encoding="utf-8")
            model_file = layout.models_dir / "model.gguf"
            model_file.write_text("gguf", encoding="utf-8")

            with mock.patch.object(uninstall_mod, "uninstall_systemd_units"), \
                 mock.patch.object(uninstall_mod, "_ufw_remove_rules"), \
                 mock.patch.object(uninstall_mod, "_remove_service_user"), \
                 mock.patch.object(uninstall_mod, "_remove_uv_tool"), \
                 mock.patch.object(uninstall_mod, "_layout_for_mode", return_value=layout), \
                 mock.patch.object(uninstall_mod, "existing_models_dir", return_value=layout.models_dir), \
                 mock.patch.object(uninstall_mod.sys, "stdin") as stdin_mock:
                stdin_mock.isatty.return_value = True
                with mock.patch("builtins.input", return_value="y"), redirect_stdout(io.StringIO()):
                    rc = uninstall_mod.uninstall_stack(_args(mode="user"))

            self.assertEqual(rc, 0)
            self.assertFalse(layout.install_root.exists())
            self.assertFalse(layout.config_dir.exists())
            self.assertFalse(layout.run_dir.exists())
            self.assertFalse(marker.exists())
            # Models must survive
            self.assertTrue(model_file.exists())
            self.assertTrue(layout.models_dir.exists())
        finally:
            tmp.cleanup()

    def test_models_inside_state_dir_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            layout = _make_layout(tmp_path)
            # point the layout's models dir inside the state dir
            inner_models = layout.state_dir / "models"
            inner_models.mkdir(parents=True, exist_ok=True)
            layout.models_dir = inner_models
            (inner_models / "model.gguf").write_text("gguf", encoding="utf-8")
            (layout.state_dir / "catalog.json").write_text("[]", encoding="utf-8")

            with mock.patch.object(uninstall_mod, "uninstall_systemd_units"), \
                 mock.patch.object(uninstall_mod, "_ufw_remove_rules"), \
                 mock.patch.object(uninstall_mod, "_remove_service_user"), \
                 mock.patch.object(uninstall_mod, "_remove_uv_tool"), \
                 mock.patch.object(uninstall_mod, "_layout_for_mode", return_value=layout), \
                 mock.patch.object(uninstall_mod, "existing_models_dir", return_value=inner_models), \
                 mock.patch.object(uninstall_mod.sys, "stdin") as stdin_mock:
                stdin_mock.isatty.return_value = True
                with mock.patch("builtins.input", return_value="y"), redirect_stdout(io.StringIO()):
                    rc = uninstall_mod.uninstall_stack(_args(mode="user"))

            self.assertEqual(rc, 0)
            self.assertTrue((inner_models / "model.gguf").exists())
            self.assertFalse((layout.state_dir / "catalog.json").exists())

    def test_remove_models_flag_deletes_models(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        try:
            tmp_path = Path(tmp.name)
            layout = _make_layout(tmp_path)
            model_file = layout.models_dir / "model.gguf"
            model_file.write_text("gguf", encoding="utf-8")

            with mock.patch.object(uninstall_mod, "uninstall_systemd_units"), \
                 mock.patch.object(uninstall_mod, "_ufw_remove_rules"), \
                 mock.patch.object(uninstall_mod, "_remove_service_user"), \
                 mock.patch.object(uninstall_mod, "_remove_uv_tool"), \
                 mock.patch.object(uninstall_mod, "_layout_for_mode", return_value=layout), \
                 mock.patch.object(uninstall_mod, "existing_models_dir", return_value=layout.models_dir), \
                 mock.patch.object(uninstall_mod.sys, "stdin") as stdin_mock:
                stdin_mock.isatty.return_value = True
                with mock.patch("builtins.input", return_value="y"), redirect_stdout(io.StringIO()):
                    rc = uninstall_mod.uninstall_stack(_args(mode="user", remove_models=True))

            self.assertEqual(rc, 0)
            self.assertFalse(model_file.exists())
            self.assertFalse(layout.models_dir.exists())
        finally:
            tmp.cleanup()


class UninstallConfirmationTest(unittest.TestCase):
    def _base_layout(self) -> InstallLayout:
        with tempfile.TemporaryDirectory() as tmp:
            self._tmp = Path(tmp)
            return _make_layout(self._tmp)

    def test_aborts_when_user_says_no(self) -> None:
        layout = self._base_layout()
        with mock.patch.object(uninstall_mod, "uninstall_systemd_units"), \
             mock.patch.object(uninstall_mod, "_layout_for_mode", return_value=layout), \
             mock.patch.object(uninstall_mod, "existing_models_dir", return_value=layout.models_dir), \
             mock.patch.object(uninstall_mod.sys, "stdin") as stdin_mock, \
             redirect_stdout(io.StringIO()) as out:
            stdin_mock.isatty.return_value = True
            with mock.patch("builtins.input", return_value="n"):
                rc = uninstall_mod.uninstall_stack(_args(mode="user"))
        self.assertEqual(rc, 1)
        self.assertIn("Aborted", out.getvalue())

    def test_aborts_on_eof(self) -> None:
        layout = self._base_layout()
        with mock.patch.object(uninstall_mod, "uninstall_systemd_units"), \
             mock.patch.object(uninstall_mod, "_layout_for_mode", return_value=layout), \
             mock.patch.object(uninstall_mod, "existing_models_dir", return_value=layout.models_dir), \
             mock.patch.object(uninstall_mod.sys, "stdin") as stdin_mock:
            stdin_mock.isatty.return_value = True
            with mock.patch("builtins.input", side_effect=EOFError), redirect_stdout(io.StringIO()):
                rc = uninstall_mod.uninstall_stack(_args(mode="user"))
        self.assertEqual(rc, 1)

    def test_non_tty_requires_yes(self) -> None:
        layout = self._base_layout()
        with mock.patch.object(uninstall_mod, "uninstall_systemd_units"), \
             mock.patch.object(uninstall_mod, "_layout_for_mode", return_value=layout), \
             mock.patch.object(uninstall_mod, "existing_models_dir", return_value=layout.models_dir), \
             mock.patch.object(uninstall_mod.sys, "stdin") as stdin_mock, \
             redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            stdin_mock.isatty.return_value = False
            rc = uninstall_mod.uninstall_stack(_args(mode="user"))
        self.assertEqual(rc, 1)
        self.assertIn("--yes", err.getvalue())

    def test_yes_flag_skips_prompt(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        try:
            tmp_path = Path(tmp.name)
            layout = _make_layout(tmp_path)
            model_file = layout.models_dir / "model.gguf"
            model_file.write_text("gguf", encoding="utf-8")
            with mock.patch.object(uninstall_mod, "uninstall_systemd_units"), \
                 mock.patch.object(uninstall_mod, "_ufw_remove_rules"), \
                 mock.patch.object(uninstall_mod, "_remove_service_user"), \
                 mock.patch.object(uninstall_mod, "_remove_uv_tool"), \
                 mock.patch.object(uninstall_mod, "_layout_for_mode", return_value=layout), \
                 mock.patch.object(uninstall_mod, "existing_models_dir", return_value=layout.models_dir), \
                 mock.patch("builtins.input", side_effect=AssertionError("prompt should be skipped")), \
                 redirect_stdout(io.StringIO()):
                rc = uninstall_mod.uninstall_stack(_args(mode="user", yes=True))
            self.assertEqual(rc, 0)
            self.assertTrue(model_file.exists())
        finally:
            tmp.cleanup()

    def test_confirmation_message_lists_kept_models(self) -> None:
        layout = self._base_layout()
        with mock.patch.object(uninstall_mod, "uninstall_systemd_units"), \
             mock.patch.object(uninstall_mod, "_layout_for_mode", return_value=layout), \
             mock.patch.object(uninstall_mod, "existing_models_dir", return_value=layout.models_dir), \
             mock.patch.object(uninstall_mod.sys, "stdin") as stdin_mock, \
             redirect_stdout(io.StringIO()) as out:
            stdin_mock.isatty.return_value = True
            with mock.patch("builtins.input", return_value="n"):
                uninstall_mod.uninstall_stack(_args(mode="user"))
        msg = out.getvalue()
        self.assertIn("Models will be KEPT at", msg)
        self.assertIn(str(layout.models_dir), msg)
        self.assertIn("rm -rf", msg)
        self.assertIn(str(layout.state_dir), msg)


class UninstallDetectionTest(unittest.TestCase):
    def test_requested_mode_wins(self) -> None:
        self.assertEqual(uninstall_mod._detect_modes_to_uninstall("system"), ["system"])
        self.assertEqual(uninstall_mod._detect_modes_to_uninstall("user"), ["user"])

    def _dummy_layout(self, mode: str) -> InstallLayout:
        return InstallLayout(
            mode=mode,
            state_dir=Path("/nonexistent/state"),
            bin_dir=Path("/nonexistent/bin"),
            install_root=Path("/nonexistent/opt"),
            models_dir=Path("/nonexistent/models"),
            config_dir=Path("/nonexistent/config"),
            run_dir=Path("/nonexistent/run"),
            service_user="u",
            service_group="g",
            public_host="127.0.0.1",
            public_port=11436,
            manager_socket=Path("/nonexistent/run/m.sock"),
            python_root=Path("/nonexistent/opt/python"),
            runtime_venv=Path("/nonexistent/opt/venv"),
            cuda_root=Path("/nonexistent/opt/cuda"),
            nccl_root=Path("/nonexistent/opt/nccl"),
            backend="auto",
        )

    def test_detects_mode_with_config_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            (home / ".config/heimdall-gateway").mkdir(parents=True)
            with mock.patch("pathlib.Path.home", return_value=home), \
                 mock.patch.object(uninstall_mod, "existing_public_host", return_value=None), \
                 mock.patch.object(uninstall_mod, "choose_layout", side_effect=lambda mode, *_a, **_k: self._dummy_layout(mode)):
                self.assertEqual(uninstall_mod._detect_modes_to_uninstall(None), ["user"])

    def test_fallback_to_detected_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home.mkdir(parents=True)
            with mock.patch("pathlib.Path.home", return_value=home), \
                 mock.patch.object(uninstall_mod, "existing_public_host", return_value=None), \
                 mock.patch.object(uninstall_mod, "choose_layout", side_effect=lambda mode, *_a, **_k: self._dummy_layout(mode)), \
                 mock.patch.object(uninstall_mod, "detect_existing_mode", return_value="system"), \
                 mock.patch.object(uninstall_mod, "_systemd_unit_exists", return_value=False), \
                 mock.patch.object(uninstall_mod, "_system_install_present", return_value=False):
                self.assertEqual(uninstall_mod._detect_modes_to_uninstall(None), ["system"])


class UninstallDryRunTest(unittest.TestCase):
    def test_dry_run_removes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            layout = _make_layout(tmp_path)
            (layout.state_dir / "catalog.json").write_text("[]", encoding="utf-8")
            model_file = layout.models_dir / "model.gguf"
            model_file.write_text("gguf", encoding="utf-8")
            with mock.patch.object(uninstall_mod, "uninstall_systemd_units"), \
                 mock.patch.object(uninstall_mod, "_layout_for_mode", return_value=layout), \
                 mock.patch.object(uninstall_mod, "existing_models_dir", return_value=layout.models_dir), \
                 redirect_stdout(io.StringIO()) as out:
                rc = uninstall_mod.uninstall_stack(_args(mode="user", dry_run=True))
            self.assertEqual(rc, 0)
            self.assertIn("Dry-run complete. No changes were made.", out.getvalue())
            self.assertTrue(layout.install_root.exists())
            self.assertTrue(layout.config_dir.exists())
            self.assertTrue((layout.state_dir / "catalog.json").exists())
            self.assertTrue(model_file.exists())

    def test_dry_run_detects_uv_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            layout = _make_layout(tmp_path)
            with mock.patch.object(uninstall_mod, "uninstall_systemd_units"), \
                 mock.patch.object(uninstall_mod, "_layout_for_mode", return_value=layout), \
                 mock.patch.object(uninstall_mod, "existing_models_dir", return_value=layout.models_dir), \
                 mock.patch.object(uninstall_mod, "_uv_tool_installed", return_value=True), \
                 redirect_stdout(io.StringIO()) as out:
                rc = uninstall_mod.uninstall_stack(_args(mode="user", dry_run=True))
            self.assertEqual(rc, 0)
            self.assertIn("uv tool uninstall heimdall-gateway", out.getvalue())


class UninstallHelpersTest(unittest.TestCase):
    def test_remove_dir_preserve_models_keeps_subtree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "state"
            models = root / "models"
            models.mkdir(parents=True)
            (root / "catalog.json").write_text("[]", encoding="utf-8")
            (root / "api-requests.log").write_text("log", encoding="utf-8")
            (models / "model.gguf").write_text("gguf", encoding="utf-8")

            uninstall_mod._remove_dir_preserve_models(root, models, dry_run=False, use_sudo=False)

            self.assertTrue((models / "model.gguf").exists())
            self.assertFalse((root / "catalog.json").exists())
            self.assertFalse((root / "api-requests.log").exists())

    def test_looks_like_model_file(self) -> None:
        self.assertTrue(uninstall_mod._looks_like_model_file("model.gguf"))
        self.assertTrue(uninstall_mod._looks_like_model_file("model-Q4.gguf"))
        self.assertTrue(uninstall_mod._looks_like_model_file("consolidated.00001-of-00002.bin"))
        self.assertTrue(uninstall_mod._looks_like_model_file("model.safetensors"))
        self.assertFalse(uninstall_mod._looks_like_model_file("catalog.json"))
        self.assertFalse(uninstall_mod._looks_like_model_file("api-requests.log"))
        self.assertTrue(uninstall_mod._looks_like_model_file("consolidated.bin"))
        self.assertFalse(uninstall_mod._looks_like_model_file("app.bin"))

    def test_legacy_targets_skip_absent_and_keep_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            legacy_state = home / ".local/state/llamacpp-superserver"
            legacy_state.mkdir(parents=True)
            (legacy_state / "model.gguf").write_text("gguf", encoding="utf-8")
            (legacy_state / "catalog.json").write_text("[]", encoding="utf-8")

            with mock.patch("pathlib.Path.home", return_value=home), redirect_stdout(io.StringIO()) as out:
                targets = uninstall_mod._legacy_removal_targets("user", preserve_models=True)
            # state dir with models is kept (not in targets)
            paths = [str(p) for p, _ in targets]
            self.assertNotIn(str(legacy_state), paths)
            self.assertIn("contains model artifacts", out.getvalue())

            with mock.patch("pathlib.Path.home", return_value=home), redirect_stdout(io.StringIO()) as out2:
                targets = uninstall_mod._legacy_removal_targets("user", preserve_models=False)
            paths = [str(p) for p, _ in targets]
            self.assertIn(str(legacy_state), paths)


if __name__ == "__main__":
    unittest.main()
