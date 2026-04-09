import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from llamacpp_stack.cli import (
    ManagedModel,
    ensure_model_available,
    list_running_ollama_models,
    load_catalog,
    render_llamaswap_config,
    resolve_idle_ttl,
    save_catalog,
    should_reload_after_unexpected_unload,
    stop_running_ollama_models,
)
from llamacpp_stack.install import (
    build_parser,
    choose_llamacpp_linux_asset,
    choose_llamaswap_asset,
    locate_cuda_root_for_python,
    normalize_python_cuda_layout,
    detect_cuda_toolkit_package,
    maybe_install_cuda_toolkit_via_uv,
    desired_models_dir_owner,
    detect_cuda_toolkit,
    derive_models_dir,
    InstallLayout,
    parse_ollama_models_from_systemctl,
)


class InstallHelpersTest(unittest.TestCase):
    def test_parse_ollama_models_from_systemctl(self) -> None:
        sample = 'Environment="OLLAMA_MODELS=/var/ollama_models/"\nEnvironment="OLLAMA_HOST=0.0.0.0"\n'
        self.assertEqual(parse_ollama_models_from_systemctl(sample), "/var/ollama_models/")

    def test_derive_models_dir_from_ollama(self) -> None:
        self.assertEqual(derive_models_dir(Path("/var/ollama_models"), "system"), Path("/var/llamacpp_models"))

    def test_choose_llamaswap_asset(self) -> None:
        release = {"assets": [{"name": "llama-swap_199_linux_amd64.tar.gz", "browser_download_url": "x"}]}
        self.assertEqual(choose_llamaswap_asset(release)["name"], "llama-swap_199_linux_amd64.tar.gz")

    def test_choose_llamacpp_linux_asset(self) -> None:
        release = {"assets": [{"name": "llama-b8705-bin-ubuntu-x64.tar.gz", "browser_download_url": "x"}]}
        self.assertEqual(choose_llamacpp_linux_asset(release)["name"], "llama-b8705-bin-ubuntu-x64.tar.gz")

    def test_detect_cuda_toolkit_uses_nvcc_lookup(self) -> None:
        with mock.patch("llamacpp_stack.install.locate_nvcc", return_value="/usr/local/cuda/bin/nvcc"):
            self.assertTrue(detect_cuda_toolkit())

    def test_detect_cuda_toolkit_package_prefers_available_candidate(self) -> None:
        def fake_run(cmd, **kwargs):
            package = cmd[-1]
            if package == "cuda-toolkit-12-4":
                return mock.Mock(returncode=0, stdout="Candidate: 12.4.1-1\n")
            return mock.Mock(returncode=0, stdout="Candidate: (none)\n")

        with (
            mock.patch("llamacpp_stack.install.shutil.which", return_value="/usr/bin/apt-cache"),
            mock.patch("llamacpp_stack.install.subprocess.run", side_effect=fake_run),
        ):
            self.assertEqual(detect_cuda_toolkit_package(), "cuda-toolkit-12-4")

    def test_maybe_install_cuda_toolkit_via_uv_invokes_uv_pip(self) -> None:
        with (
            mock.patch("llamacpp_stack.install.prompt_bool", return_value=True),
            mock.patch("llamacpp_stack.install.subprocess.run") as run_mock,
            mock.patch("llamacpp_stack.install.locate_nvcc", return_value=None),
            mock.patch("llamacpp_stack.install.locate_nvcc_for_python", return_value="/tmp/nvcc"),
            mock.patch("llamacpp_stack.install.locate_cuda_root_for_python", return_value=Path("/tmp/cuda")),
        ):
            self.assertTrue(maybe_install_cuda_toolkit_via_uv("/usr/bin/python3", dry_run=False))
        run_mock.assert_called_once_with(
            ["uv", "pip", "install", "--python", "/usr/bin/python3", "cuda-toolkit[all]"],
            check=True,
        )

    def test_locate_cuda_root_for_python_from_venv_nvcc_path(self) -> None:
        with mock.patch(
            "llamacpp_stack.install.locate_nvcc_for_python",
            return_value="/tmp/venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc",
        ):
            with mock.patch.object(Path, "exists", autospec=True) as exists_mock, mock.patch.object(
                Path, "glob", autospec=True
            ) as glob_mock:
                def fake_exists(path_self):
                    return str(path_self).endswith("/include")

                def fake_glob(path_self, pattern):
                    if str(path_self).endswith("/lib") and pattern == "libcudart.so*":
                        return iter([Path("/tmp/venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13")])
                    return iter([])

                exists_mock.side_effect = fake_exists
                glob_mock.side_effect = fake_glob
                self.assertEqual(
                    locate_cuda_root_for_python("/usr/bin/python3"),
                    Path("/tmp/venv/lib/python3.12/site-packages/nvidia/cu13"),
                )

    def test_normalize_python_cuda_layout_creates_expected_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lib_dir = root / "lib"
            lib_dir.mkdir(parents=True)
            (lib_dir / "libcudart.so.13").write_text("", encoding="utf-8")
            (lib_dir / "libcublas.so.13").write_text("", encoding="utf-8")
            changed = normalize_python_cuda_layout(root)
            self.assertTrue(changed)
            self.assertTrue((root / "lib64").is_symlink())
            self.assertEqual((root / "lib64").resolve(), lib_dir.resolve())
            self.assertTrue((lib_dir / "libcudart.so").is_symlink())
            self.assertEqual((lib_dir / "libcudart.so").resolve(), (lib_dir / "libcudart.so.13").resolve())
            self.assertTrue((lib_dir / "libcublas.so").is_symlink())
            self.assertEqual((lib_dir / "libcublas.so").resolve(), (lib_dir / "libcublas.so.13").resolve())

    def test_parser_prefers_cuda_build_by_default(self) -> None:
        args = build_parser().parse_args([])
        self.assertTrue(args.prefer_source_cuda)
        self.assertTrue(args.prefer_binary)
        args = build_parser().parse_args(["--no-prefer-source-cuda", "--no-prefer-binary"])
        self.assertFalse(args.prefer_source_cuda)
        self.assertFalse(args.prefer_binary)

    def test_desired_models_dir_owner_uses_service_identity_for_system_mode(self) -> None:
        layout = InstallLayout(
            mode="system",
            state_dir=Path("/var/lib/llamacpp"),
            bin_dir=Path("/opt/llamacpp-stack/bin"),
            install_root=Path("/opt/llamacpp-stack"),
            cuda_root=Path("/opt/llamacpp-stack/cuda"),
            models_dir=Path("/var/llamacpp_models"),
            config_dir=Path("/etc/llamacpp"),
            run_dir=Path("/run/llamacpp"),
            service_user="ollama",
            service_group="ollama",
            public_host="127.0.0.1",
            public_port=11435,
            manager_socket=Path("/run/llamacpp/manager.sock"),
            python_root=Path("/opt/llamacpp-stack/python"),
            runtime_venv=Path("/opt/llamacpp-stack/venv"),
        )
        self.assertEqual(desired_models_dir_owner(layout), ("ollama", "ollama"))

    def test_existing_model_updates_runtime_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            repo_dir = models_dir / "org/repo"
            repo_dir.mkdir(parents=True)
            model_path = repo_dir / "model-q4.gguf"
            model_path.write_bytes(b"gguf")
            catalog_path = root / "catalog.json"
            config_path = root / "config.yaml"
            save_catalog(
                catalog_path,
                [
                    ManagedModel(
                        model_id="repo-q4",
                        repo_id="org/repo",
                        quant="Q4",
                        filename="model-q4.gguf",
                        local_path=str(model_path),
                        ttl=300,
                        n_gpu_layers=0,
                        tensor_split="1",
                        description="old",
                    )
                ],
            )
            args = Namespace(
                repo="org/repo:Q4",
                hf=None,
                model_id=None,
                file=None,
                catalog=catalog_path,
                models_dir=models_dir,
                force=False,
                ctx_override=None,
                auto_ctx=False,
                skip_ctx=True,
                ctx_size=8192,
                config=config_path,
                llama_server=root / "llama-server",
                start_port=18080,
                public_host="127.0.0.1",
                public_port=11435,
                n_gpu_layers=99,
                tensor_split="3,1",
                host="0.0.0.0",
                idle_ttl=10,
                no_jinja=True,
                hf_token=None,
                description="new",
                service="llamaswap",
            )
            with (
                mock.patch("llamacpp_stack.cli.wait_for_model", return_value=False),
                mock.patch("llamacpp_stack.cli.apply_config_and_wait", return_value=True),
            ):
                result = ensure_model_available(args)

            self.assertEqual(result, "repo-q4")
            model = load_catalog(catalog_path)[0]
            self.assertEqual(model.ttl, 300)
            self.assertEqual(model.n_gpu_layers, 99)
            self.assertEqual(model.tensor_split, "3,1")
            self.assertEqual(model.host, "0.0.0.0")
            self.assertFalse(model.jinja)
            self.assertEqual(model.description, "new")

    def test_render_config_uses_global_idle_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            render_llamaswap_config(
                [
                    ManagedModel(
                        model_id="repo-q4",
                        repo_id="org/repo",
                        quant="Q4",
                        filename="model-q4.gguf",
                        local_path="/tmp/model-q4.gguf",
                        ttl=999,
                    )
                ],
                config_path,
                root / "llama-server",
                18080,
                idle_ttl=10,
            )
            self.assertIn("ttl: 10", config_path.read_text(encoding="utf-8"))

    def test_resolve_idle_ttl_reads_server_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_config = root / "llamacpp-server.json"
            server_config.write_text('{"idle_ttl": 42}\n', encoding="utf-8")
            args = Namespace(server_config=server_config, idle_ttl=None)
            self.assertEqual(resolve_idle_ttl(args), 42)

    def test_unload_guard_reloads_only_inside_activity_window(self) -> None:
        activity = {"model-a": {"last_activity_monotonic": 100.0}}
        self.assertEqual(
            should_reload_after_unexpected_unload(
                "model-a",
                activity,
                "model-a",
                now=120.0,
                idle_ttl=300,
            ),
            (True, 20.0),
        )
        self.assertEqual(
            should_reload_after_unexpected_unload(
                "model-a",
                activity,
                "model-a",
                now=401.0,
                idle_ttl=300,
            ),
            (False, 301.0),
        )
        self.assertEqual(
            should_reload_after_unexpected_unload(
                "model-a",
                activity,
                "model-b",
                now=120.0,
                idle_ttl=300,
            ),
            (False, 20.0),
        )

    def test_list_running_ollama_models_parses_ps_output(self) -> None:
        completed = mock.Mock(returncode=0, stdout="NAME                ID              SIZE    PROCESSOR\nqwen2.5:7b         abc123          5 GB    100% GPU\nphi4-mini:latest   def456          2 GB    100% GPU\n")
        with (
            mock.patch("llamacpp_stack.cli.shutil.which", return_value="/usr/bin/ollama"),
            mock.patch("llamacpp_stack.cli.subprocess.run", return_value=completed),
        ):
            self.assertEqual(list_running_ollama_models(), ["qwen2.5:7b", "phi4-mini:latest"])

    def test_stop_running_ollama_models_stops_all_detected_models(self) -> None:
        ps_result = mock.Mock(returncode=0, stdout="NAME                ID              SIZE    PROCESSOR\nqwen2.5:7b         abc123          5 GB    100% GPU\nphi4-mini:latest   def456          2 GB    100% GPU\n")
        stop_result = mock.Mock(returncode=0, stdout="", stderr="")
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[:2] == ["ollama", "ps"]:
                return ps_result
            if cmd[:2] == ["ollama", "stop"]:
                return stop_result
            raise AssertionError(f"Unexpected command: {cmd}")

        with (
            mock.patch("llamacpp_stack.cli.shutil.which", return_value="/usr/bin/ollama"),
            mock.patch("llamacpp_stack.cli.subprocess.run", side_effect=fake_run),
        ):
            stopped = stop_running_ollama_models()

        self.assertEqual(stopped, ["qwen2.5:7b", "phi4-mini:latest"])
        self.assertEqual(
            calls,
            [
                ["ollama", "ps"],
                ["ollama", "stop", "qwen2.5:7b"],
                ["ollama", "stop", "phi4-mini:latest"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
