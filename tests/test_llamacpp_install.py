import os
import errno
import json
import socket
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import yaml

from llamacpp_stack.cli import (
    ManagedModel,
    _prepare_manager_socket_path,
    _probe_runtime_env,
    build_llama_server_command,
    build_help_epilog,
    ensure_model_available,
    get_gpu_conflict_message,
    list_models,
    list_running_ollama_models,
    load_catalog,
    normalize_server_overrides,
    render_llamaswap_config,
    resolve_llama_server_defaults,
    resolve_idle_ttl,
    save_catalog,
    should_reload_after_unexpected_unload,
    stop_running_ollama_models,
    update_config,
)
from llamacpp_stack.install import (
    CLI_COMMAND,
    DEFAULT_SERVICE_USER,
    ELEVATED_INSTALL_ENV,
    MANAGER_SERVICE_NAME,
    SERVER_CONFIG_BASENAME,
    SWAP_SERVICE_NAME,
    _export_nvcc_path,
    determine_build_jobs,
    build_parser,
    build_llama_cpp_from_source,
    choose_default_swap_port,
    choose_layout,
    choose_llamacpp_linux_asset,
    choose_llamaswap_asset,
    detect_existing_llama_cpp_mode,
    locate_cuda_root_for_python,
    locate_nccl_root_for_python,
    maybe_install_nccl_via_uv,
    normalize_python_cuda_layout,
    detect_cuda_toolkit_package,
    maybe_install_cuda_toolkit_via_uv,
    desired_models_dir_owner,
    detect_cuda_toolkit,
    derive_models_dir,
    InstallLayout,
    maybe_reexec_system_install,
    maybe_migrate_existing_install,
    _link_stable_binary,
    _is_self_referential_symlink,
    _resolve_existing_stable_target,
    parse_ollama_models_from_systemctl,
    print_install_summary,
    maybe_rerun_auto_ctx,
    prompt_choice,
    render_manager_wrapper,
    restart_systemd_units,
    stop_systemd_units,
    wait_for_manager_socket,
    resolve_public_host,
    resolve_uv_executable,
    resolve_install_mode,
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

    def test_link_stable_binary_keeps_existing_file_when_target_matches_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llama-server"
            path.write_text("bin", encoding="utf-8")
            result = _link_stable_binary(path, path, dry_run=False)
            self.assertEqual(result, path)
            self.assertTrue(path.exists())
            self.assertFalse(path.is_symlink())

    def test_link_stable_binary_rejects_self_referential_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llama-server"
            path.symlink_to(path)
            with self.assertRaises(RuntimeError) as ctx:
                _link_stable_binary(path, path, dry_run=False)
            self.assertIn("self-referential symlink", str(ctx.exception))

    def test_link_stable_binary_repairs_self_loop_when_target_is_real_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stable = root / "llama-server"
            stable.symlink_to(stable)
            real = root / "llama-b8770-bin.d" / "bin" / "llama-server"
            real.parent.mkdir(parents=True)
            real.write_text("bin", encoding="utf-8")
            result = _link_stable_binary(real, stable, dry_run=False)
            self.assertEqual(result, stable)
            self.assertTrue(stable.is_symlink())
            self.assertEqual(stable.resolve(), real.resolve())

    def test_resolve_existing_stable_target_prefers_latest_extracted_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stable = root / "llama-server"
            older = root / "llama-b8700-bin.d" / "bin" / "llama-server"
            newer = root / "llama-b8710-bin.d" / "bin" / "llama-server"
            older.parent.mkdir(parents=True)
            newer.parent.mkdir(parents=True)
            older.write_text("older", encoding="utf-8")
            newer.write_text("newer", encoding="utf-8")

            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))

            resolved = _resolve_existing_stable_target(root, stable, "llama-server")
            self.assertEqual(resolved, newer)

    def test_is_self_referential_symlink_detects_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llama-server"
            path.symlink_to(path)
            self.assertTrue(_is_self_referential_symlink(path))

    def test_is_self_referential_symlink_returns_false_for_regular_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "real-llama-server"
            target.write_text("bin", encoding="utf-8")
            link = root / "llama-server"
            link.symlink_to(target)
            self.assertFalse(_is_self_referential_symlink(link))

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
            mock.patch("llamacpp_stack.install.locate_nccl_root_for_python", return_value=Path("/tmp/nccl")),
            mock.patch("llamacpp_stack.install.resolve_uv_executable", return_value="/tmp/bootstrap/bin/uv"),
        ):
            self.assertTrue(maybe_install_cuda_toolkit_via_uv("/usr/bin/python3", dry_run=False))
        self.assertEqual(
            run_mock.call_args_list[0],
            mock.call(["/tmp/bootstrap/bin/uv", "pip", "install", "--python", "/usr/bin/python3", "cuda-toolkit[all]"], check=True),
        )
        self.assertEqual(
            run_mock.call_args_list[1],
            mock.call(["/tmp/bootstrap/bin/uv", "pip", "install", "--python", "/usr/bin/python3", "nvidia-nccl-cu12"], check=True),
        )

    def test_maybe_install_nccl_via_uv_invokes_uv_pip(self) -> None:
        with (
            mock.patch("llamacpp_stack.install.subprocess.run") as run_mock,
            mock.patch("llamacpp_stack.install.locate_nccl_root_for_python", return_value=Path("/tmp/nccl")),
            mock.patch("llamacpp_stack.install.resolve_uv_executable", return_value="/tmp/bootstrap/bin/uv"),
        ):
            self.assertTrue(maybe_install_nccl_via_uv("/usr/bin/python3", dry_run=False))
        run_mock.assert_called_once_with(
            ["/tmp/bootstrap/bin/uv", "pip", "install", "--python", "/usr/bin/python3", "nvidia-nccl-cu12"],
            check=True,
        )

    def test_locate_nccl_root_for_python_from_site_packages(self) -> None:
        with mock.patch(
            "llamacpp_stack.install.subprocess.run",
            return_value=mock.Mock(stdout="/tmp/venv/lib/python3.12/site-packages/nvidia/nccl_cu12\n"),
        ):
            self.assertEqual(
                locate_nccl_root_for_python("/usr/bin/python3"),
                Path("/tmp/venv/lib/python3.12/site-packages/nvidia/nccl_cu12"),
            )

    def test_resolve_uv_executable_falls_back_to_bootstrap_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            python_bin = Path(tmp) / "bin" / "python"
            uv_bin = Path(tmp) / "bin" / "uv"
            python_bin.parent.mkdir(parents=True)
            python_bin.write_text("", encoding="utf-8")
            uv_bin.write_text("", encoding="utf-8")
            with (
                mock.patch("llamacpp_stack.install.shutil.which", return_value=None),
                mock.patch("llamacpp_stack.install.sys.executable", str(python_bin)),
            ):
                self.assertEqual(resolve_uv_executable(), str(uv_bin))

    def test_resolve_uv_executable_prefers_bootstrap_env_var(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            uv_bin = Path(tmp) / "uv"
            uv_bin.write_text("", encoding="utf-8")
            with (
                mock.patch.dict("os.environ", {"LLAMACPP_BOOTSTRAP_UV": str(uv_bin)}, clear=True),
                mock.patch("llamacpp_stack.install.shutil.which", return_value=None),
            ):
                self.assertEqual(resolve_uv_executable(), str(uv_bin))

    def test_export_nvcc_path_sets_cudacxx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nvcc = Path(tmp) / "bin" / "nvcc"
            nvcc.parent.mkdir(parents=True)
            nvcc.write_text("", encoding="utf-8")
            with mock.patch.dict("os.environ", {"PATH": "/usr/bin"}, clear=True):
                self.assertTrue(_export_nvcc_path(str(nvcc)))
                self.assertEqual(Path(os.environ["CUDACXX"]), nvcc.resolve())
                self.assertTrue(os.environ["PATH"].startswith(str(nvcc.parent.resolve())))

    def test_build_llama_cpp_from_source_passes_cuda_compiler(self) -> None:
        release = {"tag_name": "b9999"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_dir = root / "llama.cpp-b9999"
            build_dir = src_dir / "build"
            archive = root / "b9999.tar.gz"
            nvcc = root / "cuda" / "bin" / "nvcc"
            cuda_root = root / "cuda"
            nccl_root = root / "nccl"
            nvcc.parent.mkdir(parents=True)
            nvcc.write_text("", encoding="utf-8")
            (cuda_root / "include").mkdir(parents=True)
            (cuda_root / "lib64").mkdir(parents=True)
            (nccl_root / "include").mkdir(parents=True)
            (nccl_root / "lib").mkdir(parents=True)
            (nccl_root / "lib" / "libnccl.so.2").write_text("", encoding="utf-8")
            commands: list[list[str]] = []

            def fake_run(cmd, cwd=None, env=None):
                commands.append(cmd)

            with (
                mock.patch("llamacpp_stack.install._download"),
                mock.patch("llamacpp_stack.install._extract_tarball", side_effect=lambda archive_path, dest: src_dir.mkdir(parents=True, exist_ok=True)),
                mock.patch("llamacpp_stack.install.shutil.rmtree"),
                mock.patch("llamacpp_stack.install.shutil.which", side_effect=lambda name: "/usr/bin/ninja" if name == "ninja" else None),
                mock.patch("llamacpp_stack.install.source_tree_supports_flag", return_value=True),
                mock.patch("llamacpp_stack.install.detect_cuda_arch", return_value="86"),
                mock.patch("llamacpp_stack.install.locate_nvcc", return_value=str(nvcc)),
                mock.patch("llamacpp_stack.install.locate_cuda_root_for_python", return_value=cuda_root),
                mock.patch("llamacpp_stack.install.locate_nccl_root_for_python", return_value=nccl_root),
                mock.patch("llamacpp_stack.install.normalize_python_cuda_layout"),
                mock.patch("llamacpp_stack.install.determine_build_jobs", return_value=12),
                mock.patch("llamacpp_stack.install._run", side_effect=fake_run),
                mock.patch.dict("os.environ", {}, clear=True),
            ):
                result = build_llama_cpp_from_source(release, root, False, False, sys.executable, enable_cuda=True)

            self.assertEqual(result, build_dir / "bin/llama-server")
            self.assertGreaterEqual(len(commands), 2)
            cmake_cmd = commands[0]
            build_cmd = commands[1]
            self.assertIn(f"-DCMAKE_CUDA_COMPILER={nvcc.resolve()}", cmake_cmd)
            self.assertIn(f"-DCUDAToolkit_ROOT={cuda_root}", cmake_cmd)
            self.assertIn(f"-DNCCL_INCLUDE_DIR={nccl_root / 'include'}", cmake_cmd)
            self.assertIn(f"-DNCCL_LIBRARY={nccl_root / 'lib' / 'libnccl.so.2'}", cmake_cmd)
            self.assertEqual(build_cmd[-2:], ["-j", "12"])

    def test_determine_build_jobs_uses_all_available_cpus(self) -> None:
        with mock.patch("llamacpp_stack.install.os.cpu_count", return_value=32):
            self.assertEqual(determine_build_jobs(), 32)

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
        self.assertIsNone(args.public_host)
        args = build_parser().parse_args(["--no-prefer-source-cuda", "--no-prefer-binary"])
        self.assertFalse(args.prefer_source_cuda)
        self.assertFalse(args.prefer_binary)

    def test_resolve_llama_cpp_mode_prompt_is_human_readable(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("build from source", help_text.lower())

    def test_prompt_choice_renders_multiline_options(self) -> None:
        prompts: list[str] = []

        def fake_input(prompt: str) -> str:
            prompts.append(prompt)
            return ""

        with (
            mock.patch("sys.stdin.isatty", return_value=True),
            mock.patch("builtins.input", side_effect=fake_input),
        ):
            value = prompt_choice(
                "How should llama.cpp be installed?",
                [
                    ("source", "build locally from source"),
                    ("prebuilt", "download a precompiled binary"),
                    ("native", "use a system-wide llama.cpp"),
                ],
                default="source",
            )
        self.assertEqual(value, "source")
        rendered = prompts[0]
        self.assertIn("How should llama.cpp be installed?\n", rendered)
        self.assertIn("  1. Source [default]\n", rendered)
        self.assertIn("     build locally from source\n", rendered)
        self.assertIn("  2. Prebuilt\n", rendered)
        self.assertIn("     download a precompiled binary\n", rendered)
        self.assertIn("  3. Native\n", rendered)
        self.assertIn("     use a system-wide llama.cpp\n", rendered)
        self.assertTrue(rendered.endswith("Choice [source]: "))

    def test_prompt_choice_accepts_numeric_selection(self) -> None:
        with (
            mock.patch("sys.stdin.isatty", return_value=True),
            mock.patch("builtins.input", return_value="2"),
        ):
            value = prompt_choice(
                "How should llama.cpp be installed?",
                [
                    ("source", "build locally from source"),
                    ("prebuilt", "download a precompiled binary"),
                    ("native", "use a system-wide llama.cpp"),
                ],
                default="source",
            )
        self.assertEqual(value, "prebuilt")

    def test_resolve_install_mode_prompts_even_when_existing_install_is_detected(self) -> None:
        with (
            mock.patch("llamacpp_stack.install.detect_existing_mode", return_value="user"),
            mock.patch("sys.stdin.isatty", return_value=True),
            mock.patch("llamacpp_stack.install.prompt_bool", return_value=True),
        ):
            self.assertEqual(resolve_install_mode(None), "system")

    def test_resolve_public_host_can_expose_all_interfaces(self) -> None:
        with (
            mock.patch("sys.stdin.isatty", return_value=True),
            mock.patch("llamacpp_stack.install.prompt_bool", return_value=True),
        ):
            self.assertEqual(resolve_public_host("127.0.0.1"), "0.0.0.0")

    def test_resolve_public_host_keeps_explicit_value(self) -> None:
        self.assertEqual(resolve_public_host("192.168.110.50"), "192.168.110.50")

    def test_maybe_migrate_existing_install_uninstalls_previous_mode_and_keeps_models(self) -> None:
        with (
            mock.patch("llamacpp_stack.install.detect_existing_mode", return_value="user"),
            mock.patch("llamacpp_stack.uninstall.uninstall_stack") as uninstall_mock,
        ):
            maybe_migrate_existing_install("system", "127.0.0.1", 11436, False)
        uninstall_args = uninstall_mock.call_args.args[0]
        self.assertEqual(uninstall_args.mode, "user")
        self.assertTrue(uninstall_args.keep_models)

    def test_maybe_migrate_existing_install_skips_when_mode_matches(self) -> None:
        with (
            mock.patch("llamacpp_stack.install.detect_existing_mode", return_value="system"),
            mock.patch("llamacpp_stack.uninstall.uninstall_stack") as uninstall_mock,
        ):
            maybe_migrate_existing_install("system", "127.0.0.1", 11436, False)
        uninstall_mock.assert_not_called()

    def test_maybe_reexec_system_install_uses_sudo_and_resolved_args(self) -> None:
        args = Namespace(
            public_host="127.0.0.1",
            public_port=11436,
            idle_ttl=300,
            enable_tls=False,
            prefer_source_cuda=True,
            prefer_binary=True,
            install_services=True,
            update_binaries=False,
            dry_run=False,
        )
        with (
            mock.patch("llamacpp_stack.install.os.geteuid", return_value=1000),
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch("llamacpp_stack.install.subprocess.run") as run_mock,
        ):
            result = maybe_reexec_system_install(args, "system", "source", Path("/var/llamacpp_models"))
        self.assertEqual(result, 0)
        cmd = run_mock.call_args.args[0]
        self.assertEqual(cmd[:3], ["sudo", "-E", sys.executable])
        self.assertIn("--mode", cmd)
        self.assertIn("system", cmd)
        self.assertIn("--llama-cpp-mode", cmd)
        self.assertIn("source", cmd)
        self.assertIn("--models-dir", cmd)
        self.assertIn("/var/llamacpp_models", cmd)
        self.assertIn("--no-update-binaries", cmd)
        env = run_mock.call_args.kwargs["env"]
        self.assertEqual(env[ELEVATED_INSTALL_ENV], "1")

    def test_maybe_reexec_system_install_skips_when_already_root(self) -> None:
        args = Namespace(
            public_host="127.0.0.1",
            public_port=None,
            idle_ttl=300,
            enable_tls=False,
            prefer_source_cuda=True,
            prefer_binary=True,
            install_services=True,
            update_binaries=None,
            dry_run=False,
        )
        with mock.patch("llamacpp_stack.install.os.geteuid", return_value=0):
            self.assertIsNone(maybe_reexec_system_install(args, "system", "source", Path("/var/llamacpp_models")))

    def test_build_help_epilog_mentions_paths_and_override_locations(self) -> None:
        help_text = build_help_epilog()
        self.assertIn("Models dir", help_text)
        self.assertIn("llama_server_defaults", help_text)
        self.assertIn("server_overrides", help_text)
        self.assertIn("Superserver API", help_text)
        self.assertIn("llama-swap UI/backend", help_text)
        self.assertNotIn("Wrapper API", help_text)

    def test_choose_default_swap_port_prefers_ollama_plus_two(self) -> None:
        with (
            mock.patch("llamacpp_stack.install.existing_public_port", return_value=None),
            mock.patch("llamacpp_stack.install.detect_ollama_port", return_value=11434),
            mock.patch("llamacpp_stack.install._port_is_free", return_value=True),
        ):
            self.assertEqual(choose_default_swap_port("127.0.0.1", "system", None), 11436)

    def test_choose_layout_system_uses_llamaswap_identity_and_global_bin(self) -> None:
        with (
            mock.patch("llamacpp_stack.install.detect_existing_mode", return_value="system"),
            mock.patch("llamacpp_stack.install.existing_public_port", return_value=11436),
            mock.patch("llamacpp_stack.install.detect_ollama_models_dir", return_value=Path("/var/lib/ollama/models")),
        ):
            layout = choose_layout("system", "127.0.0.1", None, None)
        self.assertEqual(layout.service_user, DEFAULT_SERVICE_USER)
        self.assertEqual(layout.service_group, DEFAULT_SERVICE_USER)
        self.assertEqual(layout.bin_dir, Path("/usr/local/bin"))
        self.assertEqual(layout.public_port, 11436)

    def test_detect_existing_llama_cpp_mode_maps_manifest_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = InstallLayout(
                mode="system",
                state_dir=Path(tmp),
                bin_dir=Path("/usr/local/bin"),
                install_root=Path("/opt/llamacpp-superserver"),
                models_dir=Path("/var/lib/llamacpp-superserver/models"),
                config_dir=Path("/etc/llamacpp-superserver"),
                run_dir=Path("/run/llamacpp-superserver"),
                service_user=DEFAULT_SERVICE_USER,
                service_group=DEFAULT_SERVICE_USER,
                public_host="127.0.0.1",
                public_port=11436,
                manager_socket=Path("/run/llamacpp-superserver/manager.sock"),
                python_root=Path("/opt/llamacpp-superserver/python"),
                runtime_venv=Path("/opt/llamacpp-superserver/venv"),
                cuda_root=Path("/opt/llamacpp-superserver/cuda"),
            )
            (layout.state_dir / "install-manifest.json").write_text('{"llama_cpp_strategy":"binary"}\n', encoding="utf-8")
            self.assertEqual(detect_existing_llama_cpp_mode(layout), "prebuilt")

    def test_render_manager_wrapper_is_resilient_to_missing_llama_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = InstallLayout(
                mode="system",
                state_dir=root / "state",
                bin_dir=root / "bin",
                install_root=root / "install",
                models_dir=root / "models",
                config_dir=root / "config",
                run_dir=root / "run",
                service_user=DEFAULT_SERVICE_USER,
                service_group=DEFAULT_SERVICE_USER,
                public_host="127.0.0.1",
                public_port=11436,
                manager_socket=root / "run" / "manager.sock",
                python_root=root / "python",
                runtime_venv=root / "venv",
                cuda_root=root / "cuda",
            )
            wrapper = render_manager_wrapper(layout)
            self.assertIn("Warning: LLAMA_SERVER_BIN not found yet", wrapper)
            self.assertIn("readlink -f \"$LLAMA_SERVER_BIN\" || printf '%s' \"$LLAMA_SERVER_BIN\"", wrapper)
            self.assertIn("PYTHON_BIN is missing or not executable", wrapper)

    def test_probe_runtime_env_adds_nccl_paths(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "LLAMACPP_NCCL_ROOT": "/opt/llamacpp-superserver/nccl",
                "LD_LIBRARY_PATH": "/existing/lib",
            },
            clear=False,
        ):
            env = _probe_runtime_env()
        self.assertIsNotNone(env)
        self.assertIn("/opt/llamacpp-superserver/nccl/lib64", env["LD_LIBRARY_PATH"])
        self.assertIn("/opt/llamacpp-superserver/nccl/lib", env["LD_LIBRARY_PATH"])
        self.assertIn("/existing/lib", env["LD_LIBRARY_PATH"])

    def test_print_install_summary_invokes_help_when_command_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            help_cmd = root / CLI_COMMAND
            help_cmd.write_text("", encoding="utf-8")
            layout = InstallLayout(
                mode="user",
                state_dir=root / "state",
                bin_dir=root,
                install_root=root / "install",
                models_dir=root / "models",
                config_dir=root / "config",
                run_dir=root / "run",
                service_user="test",
                service_group="test",
                public_host="127.0.0.1",
                public_port=11436,
                manager_socket=root / "run" / "manager.sock",
                python_root=root / "python",
                runtime_venv=root / "venv",
                cuda_root=root / "cuda",
            )
            with mock.patch("llamacpp_stack.install.subprocess.run") as run_mock:
                print_install_summary(layout, install_services=True)
            run_mock.assert_called_once_with([str(help_cmd), "--help"], check=False)

    def test_wait_for_manager_socket_returns_true_when_socket_is_reachable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            socket_path = root / "run" / "manager.sock"
            socket_path.parent.mkdir(parents=True)
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                server.bind(str(socket_path))
                server.listen(1)
                layout = InstallLayout(
                    mode="system",
                    state_dir=root / "state",
                    bin_dir=root / "bin",
                    install_root=root / "install",
                    models_dir=root / "models",
                    config_dir=root / "config",
                    run_dir=root / "run",
                    service_user=DEFAULT_SERVICE_USER,
                    service_group=DEFAULT_SERVICE_USER,
                    public_host="127.0.0.1",
                    public_port=11436,
                    manager_socket=socket_path,
                    python_root=root / "python",
                    runtime_venv=root / "venv",
                    cuda_root=root / "cuda",
                )
                self.assertTrue(wait_for_manager_socket(layout, dry_run=False, timeout_seconds=1))
            finally:
                server.close()

    def test_maybe_rerun_auto_ctx_keeps_install_running_when_update_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            (models_dir / "model.gguf").write_bytes(b"gguf")
            state_dir = root / "state"
            state_dir.mkdir(parents=True)
            (state_dir / "catalog.json").write_text(
                '[{"model_id":"repo-q4","repo_id":"org/repo","filename":"model.gguf"}]\n',
                encoding="utf-8",
            )
            layout = InstallLayout(
                mode="system",
                state_dir=state_dir,
                bin_dir=root / "bin",
                install_root=root / "install",
                models_dir=models_dir,
                config_dir=root / "config",
                run_dir=root / "run",
                service_user=DEFAULT_SERVICE_USER,
                service_group=DEFAULT_SERVICE_USER,
                public_host="127.0.0.1",
                public_port=11436,
                manager_socket=root / "run" / "manager.sock",
                python_root=root / "python",
                runtime_venv=root / "venv",
                cuda_root=root / "cuda",
            )
            with (
                mock.patch("llamacpp_stack.install.prompt_bool", return_value=True),
                mock.patch("llamacpp_stack.install.wait_for_manager_socket", return_value=True),
                mock.patch(
                    "llamacpp_stack.install._run",
                    side_effect=subprocess.CalledProcessError(1, [str(layout.bin_dir / CLI_COMMAND), "update", "--auto"]),
                ) as run_mock,
            ):
                maybe_rerun_auto_ctx(layout, install_services=True, dry_run=False)
            run_mock.assert_called_once_with([str(layout.bin_dir / CLI_COMMAND), "update", "--auto"])

    def test_maybe_rerun_auto_ctx_auto_registers_local_gguf_when_catalog_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            model_path = models_dir / "qwen2.5-7b-instruct-q4_k_m.gguf"
            model_path.write_bytes(b"gguf")
            (models_dir / "mmproj-f16.gguf").write_bytes(b"gguf")
            state_dir = root / "state"
            state_dir.mkdir(parents=True)
            catalog_path = state_dir / "catalog.json"
            catalog_path.write_text("[]\n", encoding="utf-8")
            layout = InstallLayout(
                mode="system",
                state_dir=state_dir,
                bin_dir=root / "bin",
                install_root=root / "install",
                models_dir=models_dir,
                config_dir=root / "config",
                run_dir=root / "run",
                service_user=DEFAULT_SERVICE_USER,
                service_group=DEFAULT_SERVICE_USER,
                public_host="127.0.0.1",
                public_port=11436,
                manager_socket=root / "run" / "manager.sock",
                python_root=root / "python",
                runtime_venv=root / "venv",
                cuda_root=root / "cuda",
            )

            with (
                mock.patch("llamacpp_stack.install.prompt_bool", return_value=False) as prompt_mock,
                mock.patch("llamacpp_stack.install._run") as run_mock,
            ):
                maybe_rerun_auto_ctx(layout, install_services=True, dry_run=False)

            payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 1)
            self.assertEqual(payload[0]["filename"], "qwen2.5-7b-instruct-q4_k_m.gguf")
            self.assertEqual(payload[0]["quant"], "Q4_K_M")
            self.assertEqual(payload[0]["local_path"], str(model_path))
            prompt_mock.assert_called_once()
            run_mock.assert_not_called()

    def test_maybe_rerun_auto_ctx_skips_auto_register_when_catalog_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            (models_dir / "model.gguf").write_bytes(b"gguf")
            state_dir = root / "state"
            state_dir.mkdir(parents=True)
            catalog_path = state_dir / "catalog.json"
            catalog_path.write_text("{invalid-json\n", encoding="utf-8")
            layout = InstallLayout(
                mode="system",
                state_dir=state_dir,
                bin_dir=root / "bin",
                install_root=root / "install",
                models_dir=models_dir,
                config_dir=root / "config",
                run_dir=root / "run",
                service_user=DEFAULT_SERVICE_USER,
                service_group=DEFAULT_SERVICE_USER,
                public_host="127.0.0.1",
                public_port=11436,
                manager_socket=root / "run" / "manager.sock",
                python_root=root / "python",
                runtime_venv=root / "venv",
                cuda_root=root / "cuda",
            )

            with (
                mock.patch("llamacpp_stack.install.prompt_bool") as prompt_mock,
                mock.patch("llamacpp_stack.install._run") as run_mock,
                mock.patch("builtins.print") as print_mock,
            ):
                maybe_rerun_auto_ctx(layout, install_services=True, dry_run=False)

            self.assertEqual(catalog_path.read_text(encoding="utf-8"), "{invalid-json\n")
            prompt_mock.assert_not_called()
            run_mock.assert_not_called()
            printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
            self.assertIn("automatic catalog import was skipped", printed)

    def test_restart_systemd_units_uses_expected_system_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = InstallLayout(
                mode="system",
                state_dir=root / "state",
                bin_dir=root / "bin",
                install_root=root / "install",
                models_dir=root / "models",
                config_dir=root / "config",
                run_dir=root / "run",
                service_user=DEFAULT_SERVICE_USER,
                service_group=DEFAULT_SERVICE_USER,
                public_host="127.0.0.1",
                public_port=11436,
                manager_socket=root / "run" / "manager.sock",
                python_root=root / "python",
                runtime_venv=root / "venv",
                cuda_root=root / "cuda",
            )
            with mock.patch("llamacpp_stack.install._run") as run_mock:
                self.assertTrue(restart_systemd_units(layout, dry_run=False))
            run_mock.assert_called_once_with(["systemctl", "restart", MANAGER_SERVICE_NAME, SWAP_SERVICE_NAME])

    def test_stop_systemd_units_uses_expected_system_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = InstallLayout(
                mode="system",
                state_dir=root / "state",
                bin_dir=root / "bin",
                install_root=root / "install",
                models_dir=root / "models",
                config_dir=root / "config",
                run_dir=root / "run",
                service_user=DEFAULT_SERVICE_USER,
                service_group=DEFAULT_SERVICE_USER,
                public_host="127.0.0.1",
                public_port=11436,
                manager_socket=root / "run" / "manager.sock",
                python_root=root / "python",
                runtime_venv=root / "venv",
                cuda_root=root / "cuda",
            )
            with mock.patch("llamacpp_stack.install._run") as run_mock:
                self.assertTrue(stop_systemd_units(layout, dry_run=False))
            run_mock.assert_called_once_with(["systemctl", "stop", MANAGER_SERVICE_NAME, SWAP_SERVICE_NAME])

    def test_update_config_root_falls_back_to_local_when_manager_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
                        local_path=str(root / "models" / "model-q4.gguf"),
                    )
                ],
            )
            args = Namespace(
                repo=None,
                hf=None,
                model_id=None,
                file=None,
                ctx_override=4096,
                auto_ctx=False,
                catalog=catalog_path,
                config=config_path,
                llama_server=root / "llama-server",
                start_port=18080,
                public_host="127.0.0.1",
                public_port=11435,
                idle_ttl=10,
                server_config=root / SERVER_CONFIG_BASENAME,
            )

            response = mock.Mock(status_code=200)
            response.json.return_value = {"data": []}

            with (
                mock.patch("llamacpp_stack.cli.run_manager_command", side_effect=FileNotFoundError("missing socket")) as manager_mock,
                mock.patch("llamacpp_stack.cli.os.getuid", return_value=2000),
                mock.patch("llamacpp_stack.cli.os.geteuid", return_value=0),
                mock.patch("llamacpp_stack.cli.time.sleep"),
                mock.patch("llamacpp_stack.cli.requests.get", return_value=response),
            ):
                result = update_config(args)

            manager_mock.assert_called_once()
            self.assertEqual(result, "updated")

    def test_desired_models_dir_owner_uses_service_identity_for_system_mode(self) -> None:
        layout = InstallLayout(
            mode="system",
            state_dir=Path("/var/lib/llamacpp-superserver"),
            bin_dir=Path("/opt/llamacpp-superserver/bin"),
            install_root=Path("/opt/llamacpp-superserver"),
            cuda_root=Path("/opt/llamacpp-superserver/cuda"),
            models_dir=Path("/var/llamacpp_models"),
            config_dir=Path("/etc/llamacpp-superserver"),
            run_dir=Path("/run/llamacpp-superserver"),
            service_user=DEFAULT_SERVICE_USER,
            service_group=DEFAULT_SERVICE_USER,
            public_host="127.0.0.1",
            public_port=11435,
            manager_socket=Path("/run/llamacpp-superserver/manager.sock"),
            python_root=Path("/opt/llamacpp-superserver/python"),
            runtime_venv=Path("/opt/llamacpp-superserver/venv"),
        )
        self.assertEqual(desired_models_dir_owner(layout), (DEFAULT_SERVICE_USER, DEFAULT_SERVICE_USER))

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

    def test_load_catalog_normalizes_server_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "catalog.json"
            catalog_path.write_text(
                """
[
  {
    "model_id": "repo-q4",
    "repo_id": "org/repo",
    "quant": "Q4",
    "filename": "model-q4.gguf",
    "local_path": "/tmp/model-q4.gguf",
    "server_overrides": {
      "split_mode": "layer",
      "mmap": true,
      "flash-attn": "on",
      "batch_size": "1024"
    }
  }
]
""".strip()
                + "\n",
                encoding="utf-8",
            )
            model = load_catalog(catalog_path)[0]
            self.assertEqual(model.server_overrides, {"flash_attn": True, "batch_size": 1024})

    def test_render_config_applies_global_defaults_and_model_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_config = root / SERVER_CONFIG_BASENAME
            server_config.write_text(
                """
{
  "idle_ttl": 42,
  "api_port": 11436,
  "llama_server_defaults": {
    "flash_attn": true,
    "threads": 32,
    "numa": "distribute",
    "mmap": true
  }
}
""".strip()
                + "\n",
                encoding="utf-8",
            )
            with mock.patch("llamacpp_stack.cli.DEFAULT_SERVER_CONFIG_PATH", server_config):
                config_path = root / "config.yaml"
                render_llamaswap_config(
                    [
                        ManagedModel(
                            model_id="repo-q4",
                            repo_id="org/repo",
                            quant="Q4",
                            filename="model-q4.gguf",
                            local_path="/tmp/model-q4.gguf",
                            server_overrides={
                                "batch_size": 1024,
                                "mmap": False,
                                "split_mode": "row",
                            },
                        )
                    ],
                    config_path,
                    root / "llama-server",
                    18080,
                    idle_ttl=10,
                )
            rendered = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            cmd = rendered["models"]["repo-q4"]["cmd"]
            self.assertIn("--flash-attn", cmd)
            self.assertIn("--threads", cmd)
            self.assertIn("32", cmd)
            self.assertIn("--numa", cmd)
            self.assertIn("distribute", cmd)
            self.assertIn("--batch-size", cmd)
            self.assertIn("1024", cmd)
            self.assertIn("--split-mode", cmd)
            self.assertIn("row", cmd)
            self.assertIn("--no-mmap", cmd)
            self.assertNotIn("--split-mode layer", cmd)

    def test_resolve_llama_server_defaults_reads_server_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server_config = Path(tmp) / SERVER_CONFIG_BASENAME
            server_config.write_text(
                '{"llama_server_defaults":{"flash_attn":true,"mmap":true,"threads":32,"split_mode":"layer"}}\n',
                encoding="utf-8",
            )
            args = Namespace(server_config=server_config)
            self.assertEqual(resolve_llama_server_defaults(args), {"flash_attn": True, "threads": 32})

    def test_build_llama_server_command_uses_alias_and_defaults(self) -> None:
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model-q4.gguf",
            local_path="/tmp/model-q4.gguf",
            server_overrides={"gpu_layers": "all", "cache_type_k": "q8_0"},
        )
        cmd = build_llama_server_command(
            model,
            Path("/tmp/llama-server"),
            port="12345",
            server_defaults={"threads": 32, "mmap": True},
        )
        rendered = " ".join(cmd)
        self.assertIn("--threads 32", rendered)
        self.assertIn("--cache-type-k q8_0", rendered)
        self.assertNotIn("--gpu-layers all", rendered)
        self.assertNotIn("--no-mmap", rendered)


    def test_get_gpu_conflict_message_ignores_target_when_already_published(self) -> None:
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model-q4.gguf",
            local_path="/tmp/model-q4.gguf",
        )
        with (
            mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value={"repo-q4"}),
            mock.patch("llamacpp_stack.cli.get_gpu_process_map", return_value={123: "4000"}),
        ):
            self.assertIsNone(get_gpu_conflict_message("repo-q4", [model]))

    def test_get_gpu_conflict_message_reports_other_llamacpp_model(self) -> None:
        model_a = ManagedModel(
            model_id="model-a",
            repo_id="org/a",
            quant="Q4",
            filename="a.gguf",
            local_path="/tmp/a.gguf",
        )
        model_b = ManagedModel(
            model_id="model-b",
            repo_id="org/b",
            quant="Q4",
            filename="b.gguf",
            local_path="/tmp/b.gguf",
        )
        with (
            mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value=set()),
            mock.patch("llamacpp_stack.cli.get_llama_server_processes", return_value=[{"pid": 123, "model_path": "/tmp/a.gguf"}]),
            mock.patch("llamacpp_stack.cli.get_gpu_process_map", return_value={123: "4096"}),
        ):
            message = get_gpu_conflict_message("model-b", [model_a, model_b])
        self.assertIn("Cannot load model 'model-b'", message)
        self.assertIn("model-a (pid 123, 4096 MiB)", message)

    def test_get_gpu_conflict_message_reports_foreign_process(self) -> None:
        model = ManagedModel(
            model_id="model-b",
            repo_id="org/b",
            quant="Q4",
            filename="b.gguf",
            local_path="/tmp/b.gguf",
        )
        with (
            mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value=set()),
            mock.patch("llamacpp_stack.cli.get_llama_server_processes", return_value=[]),
            mock.patch("llamacpp_stack.cli.get_gpu_process_map", return_value={999: "2048"}),
            mock.patch("llamacpp_stack.cli._describe_pid", return_value="python train.py"),
        ):
            message = get_gpu_conflict_message("model-b", [model])
        self.assertIn("python train.py (pid 999, 2048 MiB)", message)

    def test_get_gpu_conflict_message_ignores_ollama_processes(self) -> None:
        model = ManagedModel(
            model_id="model-b",
            repo_id="org/b",
            quant="Q4",
            filename="b.gguf",
            local_path="/tmp/b.gguf",
        )
        with (
            mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value=set()),
            mock.patch("llamacpp_stack.cli.get_llama_server_processes", return_value=[]),
            mock.patch("llamacpp_stack.cli.get_gpu_process_map", return_value={999: "2048"}),
            mock.patch("llamacpp_stack.cli._describe_pid", return_value="ollama runner --model qwen2.5"),
        ):
            message = get_gpu_conflict_message("model-b", [model])
        self.assertIsNone(message)

    def test_resolve_idle_ttl_reads_server_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server_config = root / SERVER_CONFIG_BASENAME
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

    def test_list_models_falls_back_to_local_catalog_when_manager_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
                        local_path=str(root / "models" / "model-q4.gguf"),
                    )
                ],
            )
            args = Namespace(
                catalog=catalog_path,
                config=config_path,
                models_dir=root / "models",
                public_host="127.0.0.1",
                public_port=11437,
            )

            with (
                mock.patch("llamacpp_stack.cli.os.getuid", return_value=999999),
                mock.patch("llamacpp_stack.cli.run_manager_command", side_effect=FileNotFoundError("missing socket")),
                mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value=set()),
                mock.patch("llamacpp_stack.cli.get_llama_server_processes", return_value=[]),
                mock.patch("llamacpp_stack.cli.get_gpu_process_map", return_value={}),
                mock.patch("builtins.print") as print_mock,
            ):
                self.assertEqual(list_models(args), 0)

            printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
            self.assertIn("Manager unavailable; showing local catalog view", printed)
            self.assertIn("repo-q4", printed)

    def test_list_models_reports_gguf_without_catalog_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            (models_dir / "orphan.gguf").write_bytes(b"gguf")
            catalog_path = root / "catalog.json"
            catalog_path.write_text("[]\n", encoding="utf-8")
            args = Namespace(
                catalog=catalog_path,
                config=root / "config.yaml",
                models_dir=models_dir,
                public_host="127.0.0.1",
                public_port=11437,
            )

            with (
                mock.patch("llamacpp_stack.cli.os.getuid", return_value=999999),
                mock.patch("llamacpp_stack.cli.run_manager_command", side_effect=FileNotFoundError("missing socket")),
                mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value=set()),
                mock.patch("llamacpp_stack.cli.get_llama_server_processes", return_value=[]),
                mock.patch("llamacpp_stack.cli.get_gpu_process_map", return_value={}),
                mock.patch("builtins.print") as print_mock,
            ):
                self.assertEqual(list_models(args), 0)

            printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
            self.assertIn("Detected GGUF files in the models directory", printed)

    def test_prepare_manager_socket_path_rejects_active_manager_socket(self) -> None:
        fake_socket = mock.Mock()
        with (
            mock.patch("llamacpp_stack.cli.os.path.exists", return_value=True),
            mock.patch("llamacpp_stack.cli.socket.socket", return_value=fake_socket),
            mock.patch("llamacpp_stack.cli.os.remove") as remove_mock,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                _prepare_manager_socket_path("/tmp/manager.sock")

        self.assertIn("already in use", str(ctx.exception))
        remove_mock.assert_not_called()
        fake_socket.close.assert_called_once()

    def test_prepare_manager_socket_path_removes_stale_socket(self) -> None:
        fake_socket = mock.Mock()
        fake_socket.connect.side_effect = OSError(errno.ECONNREFUSED, "Connection refused")
        with (
            mock.patch("llamacpp_stack.cli.os.path.exists", return_value=True),
            mock.patch("llamacpp_stack.cli.socket.socket", return_value=fake_socket),
            mock.patch("llamacpp_stack.cli.os.remove") as remove_mock,
        ):
            _prepare_manager_socket_path("/tmp/manager.sock")

        remove_mock.assert_called_once_with("/tmp/manager.sock")
        fake_socket.close.assert_called_once()

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
