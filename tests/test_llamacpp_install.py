import argparse
import io
import os
import errno
import json
import socket
import ssl
import subprocess
import struct
import sys
import tempfile
import threading
import urllib.error
import urllib.request
import time
import unittest
from contextlib import redirect_stderr
from argparse import Namespace
from pathlib import Path
from unittest import mock

import yaml
import llamacpp_stack.cli as cli

from llamacpp_stack.cli import (
    ManagedModel,
    add_models,
    build_openai_model_payload,
    build_openai_model_list_payload,
    build_ollama_model_payload,
    build_info_text,
    choose_auto_ctx,
    download_hf_file,
    _prepare_manager_socket_path,
    _probe_runtime_env,
    build_llama_server_command,
    build_help_epilog,
    build_cli_parser,
    choose_gguf_file,
    ensure_model_available,
    ensure_catalog_mtp_drafters,
    ensure_replica_route_in_llamaswap_config,
    get_gpu_conflict_message,
    get_model_replica_config,
    list_models,
    list_running_ollama_models,
    load_catalog,
    normalize_server_overrides,
    normalize_server_config_payload,
    model_files_ready,
    model_name_aliases,
    migrate_server_config,
    normalize_model_id,
    resolve_catalog_model,
    persist_server_config,
    manager_hint,
    infer_install_mode,
    remove_model,
    remove_models,
    render_models_table,
    render_llamaswap_config,
    iter_catalog_with_replicas,
    resolve_request_affinity_key,
    _replica_gpu_sets,
    _raise_if_download_unsafe_while_active,
    select_replica_for_request,
    sync_replica_runtime_state,
    replica_model_id,
    REPLICA_ROUTER_STATE,
    resolve_llama_server_defaults,
    resolve_idle_ttl,
    resolve_catalog_model_name,
    restart_service_to_free_vram,
    service_commands_for_mode,
    save_catalog,
    sort_catalog_for_json,
    get_model_context_size,
    is_gemma4_sliding_window_long_context_model,
    summarize_configured_replicas,
    start_catalog_auto_update_watch,
    start_ctx_metadata_server,
    is_llamaswap_upstream_static_autoload_path,
    _api_auth_matches,
    _detect_mtp_drafter_file,
    summarize_download_state,
    request_looks_like_model_probe,
    recent_activity_blocking_model_switch,
    should_reload_after_unexpected_unload,
    stop_running_ollama_models,
    unload_models,
    update_models,
    update_config,
    main as cli_main,
)
from llamacpp_stack.install import (
    CLI_COMMAND,
    DEFAULT_SERVICE_USER,
    ELEVATED_INSTALL_ENV,
    MANAGER_SERVICE_NAME,
    LLAMA_SERVER_DEFAULTS_BASENAME,
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
    detect_existing_backend,
    locate_cuda_root_for_python,
    locate_nccl_root_for_python,
    maybe_install_nccl_via_uv,
    normalize_python_cuda_layout,
    resolve_llama_cpp_mode,
    resolve_llama_cpp_ref,
    sync_cuda_runtime,
    sync_nccl_runtime,
    detect_cuda_toolkit_package,
    maybe_install_cuda_toolkit_via_uv,
    desired_models_dir_owner,
    ensure_models_dir_ready,
    _models_dir_already_ready_for_owner,
    detect_cuda_toolkit,
    derive_models_dir,
    InstallLayout,
    _backup_existing_model_configuration,
    maybe_reexec_system_install,
    maybe_migrate_existing_install,
    maybe_refresh_runtime_package_only,
    _same_models_dir,
    _link_stable_binary,
    _is_self_referential_symlink,
    _resolve_existing_stable_target,
    _ensure_llama_server_defaults_file,
    _load_llama_server_defaults_preset,
    parse_ollama_models_from_systemctl,
    print_install_summary,
    maybe_rerun_auto_ctx,
    prompt_choice,
    render_manager_wrapper,
    render_llamaswap_wrapper,
    render_vllm_server_wrapper,
    restart_systemd_units,
    stop_systemd_units,
    wait_for_manager_socket,
    resolve_public_host,
    resolve_uv_executable,
    resolve_install_mode,
    install_stack,
    resolve_api_security_options,
    _api_cert_san_entries,
    _generate_self_signed_api_cert,
    _cert_subject_alt_names,
)


def find_free_test_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])



def _write_minimal_gguf(path: Path, entries: list[tuple[str, int, object]]) -> None:
    with path.open("wb") as fh:
        fh.write(b"GGUF")
        fh.write(struct.pack("<I", 3))
        fh.write(struct.pack("<Q", 0))
        fh.write(struct.pack("<Q", len(entries)))
        for key, value_type, value in entries:
            key_bytes = key.encode("utf-8")
            fh.write(struct.pack("<Q", len(key_bytes)))
            fh.write(key_bytes)
            fh.write(struct.pack("<I", value_type))
            if value_type == 4:
                fh.write(struct.pack("<I", int(value)))
            elif value_type == 10:
                fh.write(struct.pack("<Q", int(value)))
            elif value_type == 7:
                fh.write(b"\x01" if value else b"\x00")
            else:
                raise AssertionError(f"unsupported test GGUF value type {value_type}")


class InstallHelpersTest(unittest.TestCase):


    def test_catalog_json_sorts_newest_downloaded_first(self) -> None:
        old = ManagedModel(
            model_id="old",
            repo_id="org/old",
            quant=None,
            filename="old.gguf",
            local_path="/missing/old.gguf",
            downloaded_at="2026-01-01T00:00:00+00:00",
        )
        new = ManagedModel(
            model_id="new",
            repo_id="org/new",
            quant=None,
            filename="new.gguf",
            local_path="/missing/new.gguf",
            downloaded_at="2026-02-01T00:00:00+00:00",
        )

        self.assertEqual([m.model_id for m in sort_catalog_for_json([old, new])], ["new", "old"])

    def test_catalog_json_sort_falls_back_to_model_file_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_path = root / "old.gguf"
            new_path = root / "new.gguf"
            old_path.write_bytes(b"old")
            new_path.write_bytes(b"new")
            os.utime(old_path, (1000, 1000))
            os.utime(new_path, (2000, 2000))
            old = ManagedModel("old", "org/old", None, old_path.name, str(old_path))
            new = ManagedModel("new", "org/new", None, new_path.name, str(new_path))

            self.assertEqual([m.model_id for m in sort_catalog_for_json([old, new])], ["new", "old"])

    def test_api_auth_matches_bearer_and_x_api_key(self) -> None:
        class Headers(dict):
            def get(self, key, default=None):
                return super().get(key, default)

        self.assertTrue(_api_auth_matches(Headers({"Authorization": "Bearer secret"}), "secret"))
        self.assertTrue(_api_auth_matches(Headers({"X-API-Key": "secret"}), "secret"))
        self.assertFalse(_api_auth_matches(Headers({"Authorization": "Bearer wrong"}), "secret"))

    def test_normalize_server_config_adds_api_security_blocks(self) -> None:
        normalized, changed = normalize_server_config_payload({})
        self.assertTrue(changed)
        self.assertEqual(normalized["api_auth"], {"enabled": False, "api_key": ""})
        self.assertEqual(normalized["api_https"], {"enabled": False, "cert_file": "", "key_file": ""})

    def test_superserver_api_allows_loopback_when_key_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "catalog.json"
            catalog.write_text("[]\n", encoding="utf-8")
            config = root / "config.yaml"
            config.write_text("models: {}\n", encoding="utf-8")
            server_config = root / "conf.json"
            port = find_free_test_port()
            server_config.write_text(json.dumps({"api_port": port, "api_auth": {"enabled": True, "api_key": "secret"}}), encoding="utf-8")
            args = Namespace(
                public_host="127.0.0.1",
                public_port=port + 1,
                config=config,
                catalog=catalog,
                llama_server=root / "llama-server",
                server_config=server_config,
            )
            server = start_ctx_metadata_server(args)
            self.assertIsNotNone(server)
            try:
                url = f"http://127.0.0.1:{port}/v1/models"
                # Loopback/local browser access is intentionally trusted even
                # when the network-facing API requires an API key.
                with urllib.request.urlopen(url, timeout=3) as response:
                    self.assertEqual(response.status, 200)
                req = urllib.request.Request(url, headers={"Authorization": "Bearer secret"})
                with urllib.request.urlopen(req, timeout=3) as response:
                    self.assertEqual(response.status, 200)
            finally:
                server.shutdown()
                server.server_close()


    def test_superserver_root_shows_local_service_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "catalog.json"
            catalog.write_text("[]\n", encoding="utf-8")
            config = root / "config.yaml"
            config.write_text("models: {}\n", encoding="utf-8")
            server_config = root / "conf.json"
            port = find_free_test_port()
            server_config.write_text(json.dumps({
                "api_port": port,
                "api_auth": {"enabled": True, "api_key": "secret-local"},
            }), encoding="utf-8")
            args = Namespace(
                public_host="127.0.0.1",
                public_port=port + 1,
                config=config,
                catalog=catalog,
                llama_server=root / "llama-server",
                server_config=server_config,
            )
            server = start_ctx_metadata_server(args)
            self.assertIsNotNone(server)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as response:
                    body = response.read().decode("utf-8")
                self.assertIn("llamacpp-superserver API", body)
                self.assertIn("secret-local", body)
                self.assertIn("/v1/models", body)
                self.assertIn(str(server_config), body)
            finally:
                server.shutdown()
                server.server_close()


    def test_superserver_https_keeps_loopback_plain_http_on_same_port(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "catalog.json"
            catalog.write_text("[]\n", encoding="utf-8")
            config = root / "config.yaml"
            config.write_text("models: {}\n", encoding="utf-8")
            cert = root / "cert.pem"
            key = root / "key.pem"
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-sha256",
                "-days", "1", "-nodes", "-keyout", str(key), "-out", str(cert),
                "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            port = find_free_test_port()
            server_config = root / "conf.json"
            server_config.write_text(json.dumps({
                "api_port": port,
                "api_https": {"enabled": True, "cert_file": str(cert), "key_file": str(key)},
            }), encoding="utf-8")
            args = Namespace(
                public_host="127.0.0.1", public_port=port + 1, config=config, catalog=catalog,
                llama_server=root / "llama-server", server_config=server_config,
            )
            server = start_ctx_metadata_server(args)
            self.assertIsNotNone(server)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as response:
                    self.assertEqual(response.status, 200)
                    payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(payload, {"object": "list", "data": []})
            finally:
                server.shutdown()
                server.server_close()

    def test_superserver_https_redirect_decision_only_applies_to_remote_plain_http(self) -> None:
        from llamacpp_stack.cli import _should_redirect_plain_http_to_https

        self.assertFalse(_should_redirect_plain_http_to_https(
            https_enabled=True,
            client_host="127.0.0.1",
            connection=object(),
        ))
        self.assertFalse(_should_redirect_plain_http_to_https(
            https_enabled=False,
            client_host="192.168.1.50",
            connection=object(),
        ))
        self.assertTrue(_should_redirect_plain_http_to_https(
            https_enabled=True,
            client_host="192.168.1.50",
            connection=object(),
        ))

    def test_superserver_api_can_serve_https_with_configured_cert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "catalog.json"
            catalog.write_text("[]\n", encoding="utf-8")
            config = root / "config.yaml"
            config.write_text("models: {}\n", encoding="utf-8")
            cert = root / "cert.pem"
            key = root / "key.pem"
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-sha256",
                "-days", "1", "-nodes", "-keyout", str(key), "-out", str(cert),
                "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            port = find_free_test_port()
            server_config = root / "conf.json"
            server_config.write_text(json.dumps({
                "api_port": port,
                "api_https": {"enabled": True, "cert_file": str(cert), "key_file": str(key)},
            }), encoding="utf-8")
            args = Namespace(
                public_host="127.0.0.1", public_port=port + 1, config=config, catalog=catalog,
                llama_server=root / "llama-server", server_config=server_config,
            )
            server = start_ctx_metadata_server(args)
            self.assertIsNotNone(server)
            try:
                ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(f"https://127.0.0.1:{port}/api/version", timeout=3, context=ctx) as response:
                    self.assertEqual(response.status, 200)
            finally:
                server.shutdown()
                server.server_close()

    def test_resolve_api_security_options_generates_key_and_self_signed_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = choose_layout("user", "0.0.0.0", 11436, root / "models")
            layout = InstallLayout(
                mode=layout.mode,
                state_dir=root / "state",
                bin_dir=root / "bin",
                install_root=root / "install",
                models_dir=root / "models",
                config_dir=root / "config",
                run_dir=root / "run",
                service_user=layout.service_user,
                service_group=layout.service_group,
                public_host="0.0.0.0",
                public_port=11436,
                manager_socket=root / "run" / "manager.sock",
                python_root=root / "install" / "python",
                runtime_venv=root / "install" / "venv",
                cuda_root=root / "install" / "cuda",
                nccl_root=root / "install" / "nccl",
                backend="llama.cpp",
            )
            args = Namespace(api_auth=True, api_key="", api_https=True, api_cert_file="", api_key_file="", dry_run=True)
            auth, https = resolve_api_security_options(args, layout)
            self.assertTrue(auth["enabled"])
            self.assertTrue(str(auth["api_key"]).startswith("lcsk_"))
            self.assertTrue(https["enabled"])
            self.assertTrue(str(https["cert_file"]).endswith("superserver-api.crt"))
            self.assertTrue(str(https["key_file"]).endswith("superserver-api.key"))



    def test_resolve_api_security_options_preserves_existing_api_key_and_cert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = choose_layout("user", "0.0.0.0", 11436, root / "models")
            layout = InstallLayout(
                mode=layout.mode,
                state_dir=root / "state",
                bin_dir=root / "bin",
                install_root=root / "install",
                models_dir=root / "models",
                config_dir=root / "config",
                run_dir=root / "run",
                service_user=layout.service_user,
                service_group=layout.service_group,
                public_host="0.0.0.0",
                public_port=11436,
                manager_socket=root / "run" / "manager.sock",
                python_root=root / "install" / "python",
                runtime_venv=root / "install" / "venv",
                cuda_root=root / "install" / "cuda",
                nccl_root=root / "install" / "nccl",
                backend="llama.cpp",
            )
            layout.config_dir.mkdir(parents=True, exist_ok=True)
            cert = root / "existing.crt"
            key = root / "existing.key"
            (layout.config_dir / "conf.json").write_text(json.dumps({
                "api_auth": {"enabled": True, "api_key": "lcsk_existing"},
                "api_https": {"enabled": True, "cert_file": str(cert), "key_file": str(key)},
            }), encoding="utf-8")
            args = Namespace(
                api_auth=True,
                api_key="",
                api_https=True,
                api_cert_file="",
                api_key_file="",
                api_cert_sans="",
                regenerate_api_cert=False,
                dry_run=True,
            )
            auth, https = resolve_api_security_options(args, layout)
            self.assertEqual(auth, {"enabled": True, "api_key": "lcsk_existing"})
            self.assertEqual(https, {"enabled": True, "cert_file": str(cert), "key_file": str(key)})

    def test_api_cert_sans_include_lan_public_and_extra_addresses(self) -> None:
        with (
            mock.patch("llamacpp_stack.install._detect_lan_ip_addresses", return_value=["192.168.1.10", "10.0.0.5"]),
            mock.patch("llamacpp_stack.install._detect_public_ip_addresses", return_value=["193.147.87.218"]),
        ):
            sans = _api_cert_san_entries("0.0.0.0", extra_sans=["myhost.local", "172.16.0.2"])
        self.assertIn("DNS:localhost", sans)
        self.assertIn("IP:127.0.0.1", sans)
        self.assertIn("IP:192.168.1.10", sans)
        self.assertIn("IP:10.0.0.5", sans)
        self.assertIn("IP:193.147.87.218", sans)
        self.assertIn("DNS:myhost.local", sans)
        self.assertIn("IP:172.16.0.2", sans)

    def test_generate_self_signed_api_cert_regenerates_when_san_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = choose_layout("user", "0.0.0.0", 11436, root / "models")
            layout = InstallLayout(
                mode=layout.mode,
                state_dir=root / "state",
                bin_dir=root / "bin",
                install_root=root / "install",
                models_dir=root / "models",
                config_dir=root / "config",
                run_dir=root / "run",
                service_user=layout.service_user,
                service_group=layout.service_group,
                public_host="0.0.0.0",
                public_port=11436,
                manager_socket=root / "run" / "manager.sock",
                python_root=root / "install" / "python",
                runtime_venv=root / "install" / "venv",
                cuda_root=root / "install" / "cuda",
                nccl_root=root / "install" / "nccl",
                backend="llama.cpp",
            )
            with (
                mock.patch("llamacpp_stack.install._detect_lan_ip_addresses", return_value=[]),
                mock.patch("llamacpp_stack.install._detect_public_ip_addresses", return_value=[]),
            ):
                cert, key = _generate_self_signed_api_cert(layout, "0.0.0.0", False)
            first_sans = _cert_subject_alt_names(Path(cert))
            self.assertIn("IP:127.0.0.1", first_sans)
            self.assertNotIn("IP:193.147.87.218", first_sans)
            with (
                mock.patch("llamacpp_stack.install._detect_lan_ip_addresses", return_value=[]),
                mock.patch("llamacpp_stack.install._detect_public_ip_addresses", return_value=["193.147.87.218"]),
            ):
                cert2, key2 = _generate_self_signed_api_cert(layout, "0.0.0.0", False)
            self.assertEqual(cert, cert2)
            self.assertEqual(key, key2)
            second_sans = _cert_subject_alt_names(Path(cert2))
            self.assertIn("IP:193.147.87.218", second_sans)
            self.assertTrue(list((Path(cert).parent).glob("*.bak-*")))

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

    def test_model_files_ready_requires_expected_size_when_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models"
            model_dir.mkdir(parents=True)
            file_path = model_dir / "model.gguf"
            file_path.write_bytes(b"1234")
            self.assertFalse(model_files_ready(model_dir, ["model.gguf"], {"model.gguf": 8}))
            file_path.write_bytes(b"12345678")
            self.assertTrue(model_files_ready(model_dir, ["model.gguf"], {"model.gguf": 8}))

    def test_summarize_download_state_counts_truncated_final_as_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models"
            model_dir.mkdir(parents=True)
            file_path = model_dir / "model.gguf"
            file_path.write_bytes(b"1234")
            completed, partial, missing = summarize_download_state(model_dir, ["model.gguf", "missing.gguf"], {"model.gguf": 8})
            self.assertEqual((completed, partial, missing), (0, 1, 1))

    def test_choose_gguf_file_matches_bf16_sharded_subdirectory(self) -> None:
        sibling_names = [
            "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf",
            "BF16/Qwen3.6-35B-A3B-BF16-00002-of-00002.gguf",
        ]
        api = mock.MagicMock()
        api.model_info.return_value.siblings = [
            mock.Mock(rfilename=name) for name in sibling_names
        ]

        selected = choose_gguf_file(
            api,
            "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
            "BF16",
            None,
            None,
        )

        self.assertEqual(selected, "BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf")


    def test_choose_gguf_file_matches_ud_quant_with_mixed_separators(self) -> None:
        sibling_names = [
            "MTP/gemma-4-31B-it-BF16-MTP.gguf",
            "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
            "mmproj-F16.gguf",
        ]
        api = mock.MagicMock()
        api.model_info.return_value.siblings = [mock.Mock(rfilename=name) for name in sibling_names]
        self.assertEqual(
            choose_gguf_file(api, "unsloth/gemma-4-31B-it-qat-GGUF", "UD-Q4_K_XL", None, None),
            "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
        )

    def test_choose_gguf_file_matches_ud_q5_xl_not_default_q4(self) -> None:
        sibling_names = [
            "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
            "gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf",
        ]
        api = mock.MagicMock()
        api.model_info.return_value.siblings = [mock.Mock(rfilename=name) for name in sibling_names]
        self.assertEqual(
            choose_gguf_file(api, "unsloth/gemma-4-26B-A4B-it-GGUF", "UD-Q5_K_XL", None, None),
            "gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf",
        )

    def test_detect_mtp_drafter_prefers_root_mtp_file(self) -> None:
        api = mock.MagicMock()
        api.model_info.return_value.siblings = [
            mock.Mock(rfilename="gemma-4-31B-it-qat-UD-Q4_K_XL.gguf"),
            mock.Mock(rfilename="MTP/mtp-gemma-4-31B-it-Q8_0.gguf"),
            mock.Mock(rfilename="mtp-gemma-4-31B-it.gguf"),
        ]

        self.assertEqual(
            _detect_mtp_drafter_file(
                api,
                "unsloth/gemma-4-31B-it-qat-GGUF",
                None,
                "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
            ),
            "mtp-gemma-4-31B-it.gguf",
        )

    def test_download_guard_refuses_when_llama_server_active_unless_forced(self) -> None:
        catalog = [ManagedModel(
            model_id="loaded", repo_id="repo", quant="Q4", filename="model.gguf",
            local_path="/models/model.gguf", description="loaded"
        )]
        with mock.patch("llamacpp_stack.cli.get_llama_server_processes", return_value=[{
            "pid": 123, "model_path": "/models/model.gguf", "port": 18080, "cmdline": "llama-server"
        }]):
            with self.assertRaises(RuntimeError) as ctx:
                _raise_if_download_unsafe_while_active(catalog, force=False)
            self.assertIn("Refusing to download", str(ctx.exception))
            _raise_if_download_unsafe_while_active(catalog, force=True)

    def test_download_guard_refuses_when_api_has_in_flight_request(self) -> None:
        catalog: list[ManagedModel] = []
        with REPLICA_ROUTER_STATE.lock:
            previous = dict(REPLICA_ROUTER_STATE.base_in_flight)
            REPLICA_ROUTER_STATE.base_in_flight["busy-model"] = 1
        try:
            with mock.patch("llamacpp_stack.cli.get_llama_server_processes", return_value=[]):
                with self.assertRaises(RuntimeError) as ctx:
                    _raise_if_download_unsafe_while_active(catalog, force=False)
                self.assertIn("busy-model(in_flight=1)", str(ctx.exception))
                _raise_if_download_unsafe_while_active(catalog, force=True)
        finally:
            with REPLICA_ROUTER_STATE.lock:
                REPLICA_ROUTER_STATE.base_in_flight.clear()
                REPLICA_ROUTER_STATE.base_in_flight.update(previous)

    def test_download_hf_file_resumes_truncated_final_file_using_expected_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target_dir = root / "models"
            target_dir.mkdir(parents=True)
            dest_path = target_dir / "model.gguf"
            dest_path.write_bytes(b"ab")

            response = mock.MagicMock()
            response.status_code = 206
            response.headers = {"Content-Range": "bytes 2-3/4", "Content-Length": "2"}
            response.iter_content.return_value = [b"cd"]
            response.raise_for_status.return_value = None
            response.__enter__.return_value = response
            response.__exit__.return_value = None

            with (
                mock.patch("llamacpp_stack.cli._download_hf_file_parallel", return_value=None),
                mock.patch("llamacpp_stack.cli._download_hf_file_fast", return_value=None),
                mock.patch("llamacpp_stack.cli.requests.get", return_value=response),
            ):
                result = download_hf_file(
                    repo_id="org/repo",
                    filename="model.gguf",
                    token=None,
                    target_dir=target_dir,
                    expected_size=4,
                )

            self.assertEqual(Path(result), dest_path)
            self.assertEqual(dest_path.read_bytes(), b"abcd")

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
            real.write_text("#!/usr/bin/env bash\necho real \"$@\"\n", encoding="utf-8")
            real.chmod(0o755)
            result = _link_stable_binary(real, stable, dry_run=False)
            self.assertEqual(result, stable)
            self.assertFalse(stable.is_symlink())
            self.assertTrue(stable.exists())
            self.assertEqual((root / "llama-server.realpath").read_text(encoding="utf-8").strip(), str(real.resolve()))
            output = subprocess.check_output([str(stable), "--help"], text=True).strip()
            self.assertEqual(output, "real --help")

    def test_resolve_existing_stable_target_uses_realpath_file_for_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stable = root / "llama-server"
            real = root / "llama-b8770-bin.d" / "bin" / "llama-server"
            real.parent.mkdir(parents=True)
            real.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            real.chmod(0o755)
            _link_stable_binary(real, stable, dry_run=False)

            resolved = _resolve_existing_stable_target(root, stable, "llama-server")

            self.assertEqual(resolved, real)

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

    def test_resolve_llama_cpp_ref_prompts_for_specific_commit(self) -> None:
        with (
            mock.patch.dict("os.environ", {"LLAMACPP_LLAMA_CPP_REF_PROMPTED": ""}, clear=False),
            mock.patch("sys.stdin.isatty", return_value=True),
            mock.patch("builtins.input", side_effect=["2", "e8023568d05ffdb9d0ac65695c521ea7e72c6b75"]),
        ):
            os.environ.pop("LLAMACPP_LLAMA_CPP_REF_PROMPTED", None)
            self.assertEqual(
                resolve_llama_cpp_ref(None, "source"),
                "e8023568d05ffdb9d0ac65695c521ea7e72c6b75",
            )

    def test_resolve_llama_cpp_ref_latest_returns_empty_ref(self) -> None:
        with (
            mock.patch("sys.stdin.isatty", return_value=True),
            mock.patch("builtins.input", return_value=""),
        ):
            self.assertEqual(resolve_llama_cpp_ref(None, "source"), "")

    def test_resolve_llama_cpp_ref_uses_env_default(self) -> None:
        with mock.patch.dict("os.environ", {"LLAMACPP_LLAMA_CPP_REF": "abc123"}, clear=False):
            self.assertEqual(resolve_llama_cpp_ref(None, "source"), "abc123")

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
            self.assertIn(f"-DCMAKE_BUILD_RPATH={cuda_root / 'lib64'};{nccl_root / 'lib'}", cmake_cmd)

    def test_install_prompt_package_only_triggers_package_only_update(self) -> None:
        args = Namespace(
            mode=None,
            public_host=None,
            public_port=None,
            models_dir=None,
            install_services=False,
            update_binaries=None,
            package_only_update=None,
            llama_cpp_mode=None,
            prefer_source_cuda=False,
            prefer_binary=False,
            dry_run=True,
            enable_tls=False,
            idle_ttl=300,
            skip_venv_install=False,
        )

        dummy_layout = Namespace(
            install_root=Path("/tmp/install"),
            state_dir=Path("/tmp/state"),
            models_dir=Path("/tmp/models"),
            public_host="127.0.0.1",
            public_port=11437,
            mode="user",
            config_dir=Path("/tmp/config"),
            bin_dir=Path("/tmp/bin"),
            run_dir=Path("/tmp/run"),
            runtime_venv=Path("/tmp/install/venv"),
        )

        with (
            mock.patch("llamacpp_stack.install.resolve_install_mode", return_value="user"),
            mock.patch("llamacpp_stack.install.resolve_public_host", return_value="127.0.0.1"),
            mock.patch("llamacpp_stack.install.prompt_path", return_value=Path("/tmp/models")),
            mock.patch("llamacpp_stack.install.maybe_migrate_existing_install"),
            mock.patch("llamacpp_stack.install.choose_layout", return_value=dummy_layout),
            mock.patch("llamacpp_stack.install.is_existing_install", return_value=True),
            mock.patch("llamacpp_stack.install.prompt_bool", return_value=False),
            mock.patch("llamacpp_stack.install.maybe_reexec_system_install", return_value=None),
            mock.patch("llamacpp_stack.install.maybe_refresh_runtime_package_only") as refresh_mock,
        ):
            res = install_stack(args)
            self.assertEqual(res, 0)
            self.assertTrue(args.package_only_update)
            self.assertFalse(args.update_binaries)
            refresh_mock.assert_called_once_with(dummy_layout, True, args)


    def test_package_only_with_services_enables_units(self) -> None:
        args = Namespace(
            mode=None,
            public_host=None,
            public_port=None,
            models_dir=None,
            install_services=True,
            update_binaries=None,
            package_only_update=None,
            llama_cpp_mode=None,
            llama_cpp_ref=None,
            backend=None,
            prefer_source_cuda=False,
            prefer_binary=False,
            dry_run=True,
            enable_tls=False,
            idle_ttl=300,
            skip_venv_install=False,
            api_auth=None,
            api_key="",
            api_https=False,
            api_cert_file="",
            api_key_file="",
        )
        dummy_layout = Namespace(
            install_root=Path("/tmp/install"),
            state_dir=Path("/tmp/state"),
            models_dir=Path("/tmp/models"),
            public_host="0.0.0.0",
            public_port=11436,
            mode="user",
            config_dir=Path("/tmp/config"),
            bin_dir=Path("/tmp/bin"),
            run_dir=Path("/tmp/run"),
            runtime_venv=Path("/tmp/install/venv"),
        )
        with (
            mock.patch("llamacpp_stack.install.resolve_install_mode", return_value="user"),
            mock.patch("llamacpp_stack.install.resolve_public_host", return_value="0.0.0.0"),
            mock.patch("llamacpp_stack.install.prompt_path", return_value=Path("/tmp/models")),
            mock.patch("llamacpp_stack.install.maybe_migrate_existing_install"),
            mock.patch("llamacpp_stack.install.choose_layout", return_value=dummy_layout),
            mock.patch("llamacpp_stack.install.stop_systemd_units"),
            mock.patch("llamacpp_stack.install.is_existing_install", return_value=True),
            mock.patch("llamacpp_stack.install.prompt_bool", return_value=False),
            mock.patch("llamacpp_stack.install.detect_existing_llama_cpp_mode", return_value="source"),
            mock.patch("llamacpp_stack.install.maybe_reexec_system_install", return_value=None),
            mock.patch("llamacpp_stack.install.maybe_refresh_runtime_package_only") as refresh_mock,
            mock.patch("llamacpp_stack.install.write_api_security_config") as write_security_mock,
            mock.patch("llamacpp_stack.install.install_systemd_units") as install_units_mock,
            mock.patch("llamacpp_stack.install.restart_systemd_units") as restart_units_mock,
        ):
            res = install_stack(args)
        self.assertEqual(res, 0)
        refresh_mock.assert_called_once_with(dummy_layout, True, args)
        write_security_mock.assert_called_once()
        install_units_mock.assert_called_once_with(dummy_layout, True)
        restart_units_mock.assert_not_called()

    def test_render_llamaswap_wrapper_exports_llama_server_lib_dir(self) -> None:
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
            wrapper = render_llamaswap_wrapper(layout)
            self.assertIn('if [[ -n "${LLAMA_SERVER_BIN:-}" ]]', wrapper)
            self.assertIn('LLAMA_SERVER_LIB_DIR="$(dirname "$LLAMA_SERVER_REAL")"', wrapper)
            self.assertIn('export LD_LIBRARY_PATH="$LLAMA_SERVER_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"', wrapper)

    def test_render_vllm_server_wrapper_translates_llama_server_flags(self) -> None:
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
                backend="vllm-beta",
            )
            wrapper = render_vllm_server_wrapper(layout)
            self.assertIn("vllm.entrypoints.openai.api_server", wrapper)
            self.assertIn("--max-model-len", wrapper)
            self.assertIn("--no-enable-log-requests", wrapper)
            self.assertIn("VLLM_WORKER_MULTIPROC_METHOD", wrapper)
            self.assertIn("--n-gpu-layers", wrapper)

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

    def test_resolve_llama_cpp_mode_prompts_source_ref_before_reexec(self) -> None:
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch("sys.stdin.isatty", return_value=True),
            mock.patch("builtins.input", side_effect=["1", "2", "e8023568"]),
        ):
            self.assertEqual(resolve_llama_cpp_mode(None), "source")
            self.assertEqual(os.environ.get("LLAMACPP_LLAMA_CPP_REF"), "e8023568")

    def test_resolve_llama_cpp_ref_does_not_reprompt_after_latest_choice(self) -> None:
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch("sys.stdin.isatty", return_value=True),
            mock.patch("builtins.input", side_effect=["1", "1"]) as input_mock,
        ):
            self.assertEqual(resolve_llama_cpp_mode(None), "source")
            self.assertEqual(resolve_llama_cpp_ref(None, "source"), "")
            self.assertEqual(input_mock.call_count, 2)

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
            package_only_update=True,
            migrate_model_ids=True,
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
        self.assertIn("--package-only-update", cmd)
        self.assertIn("--migrate-model-ids", cmd)
        env = run_mock.call_args.kwargs["env"]
        self.assertEqual(env[ELEVATED_INSTALL_ENV], "1")

    def test_maybe_refresh_runtime_package_only_calls_runtime_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = InstallLayout(
                mode="user",
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
            args = Namespace(skip_venv_install=False)
            with mock.patch("llamacpp_stack.install.ensure_runtime_python") as runtime_mock:
                maybe_refresh_runtime_package_only(layout, False, args)
            runtime_mock.assert_called_once_with(layout, False, skip_pip_install=False)

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
        self.assertIn("Command guide:", help_text)
        self.assertIn("add [repo ...] [-hf HF ...]", help_text)
        self.assertIn("run [repo|-hf HF]", help_text)
        self.assertIn("remove [repo ...|-hf HF ...]", help_text)
        self.assertIn("rm [repo ...|-hf HF ...]", help_text)
        self.assertIn("--keep-files", help_text)
        self.assertIn("update [repo ...|-hf HF ...]", help_text)
        self.assertIn("validate [repo|-hf HF]", help_text)
        self.assertIn("daemon", help_text)
        self.assertIn("list", help_text)
        self.assertIn("ps", help_text)
        self.assertIn("requests [-n LINES]", help_text)
        self.assertIn("info", help_text)
        self.assertIn("Example: llamacpp-superserver add", help_text)
        self.assertIn("Example: llamacpp-superserver run", help_text)
        self.assertIn("Example: llamacpp-superserver remove", help_text)
        self.assertIn("Example: llamacpp-superserver rm", help_text)
        self.assertIn("Example: llamacpp-superserver update", help_text)
        self.assertIn("Example: llamacpp-superserver validate", help_text)
        self.assertIn("Example: llamacpp-superserver daemon", help_text)
        self.assertIn("Example: llamacpp-superserver list", help_text)
        self.assertIn("Example: llamacpp-superserver ps", help_text)
        self.assertIn("Example: llamacpp-superserver requests", help_text)
        self.assertIn("For endpoints/runtime/service/config details run", help_text)
        self.assertNotIn("Default endpoints:", help_text)
        self.assertNotIn("Installed versions:", help_text)
        self.assertNotIn("Runtime info:", help_text)
        self.assertNotIn("Service management:", help_text)
        self.assertNotIn("Config knobs:", help_text)
        self.assertNotIn("Wrapper API", help_text)

    def test_root_help_is_compact_and_without_ascii_banner(self) -> None:
        parser, _ = build_cli_parser()
        help_text = parser.format_help()
        self.assertFalse(help_text.startswith("=" * 72))
        self.assertNotIn("llama.cpp  SuperServer", help_text)
        self.assertNotRegex(help_text, r"(?m)^\s*llamacpp-superserver v\d")
        self.assertRegex(help_text, r"usage: llamacpp-superserver")
        self.assertIn("info", help_text)
        self.assertNotIn("llama-swap-guard", help_text)
        self.assertNotIn("==SUPPRESS==", help_text)

    def test_build_info_text_contains_runtime_sections(self) -> None:
        args = Namespace(
            public_host="0.0.0.0",
            public_port=11436,
            api_port=11435,
            models_dir=Path("/workvols/data3/LLAMACPP_MODELS"),
            config=Path("/var/lib/llamacpp-superserver/config.yaml"),
            catalog=Path("/var/lib/llamacpp-superserver/catalog.json"),
            server_config=Path("/etc/llamacpp-superserver/conf.json"),
            llama_server=Path("/opt/llamacpp-superserver/llama-server"),
            idle_ttl=300,
        )
        with (
            mock.patch("llamacpp_stack.cli.read_install_manifest", return_value={"llama_cpp_tag": "b8808", "llamaswap_tag": "v202"}),
            mock.patch("llamacpp_stack.cli.infer_install_mode", return_value="system"),
            mock.patch(
                "llamacpp_stack.cli.service_commands_for_mode",
                return_value=(
                    "sudo systemctl start llamacpp-superserver-manager llamacpp-superserver-swap",
                    "sudo systemctl status llamacpp-superserver-manager llamacpp-superserver-swap",
                    "sudo systemctl restart llamacpp-superserver-manager llamacpp-superserver-swap",
                ),
            ),
            mock.patch(
                "llamacpp_stack.cli.get_api_endpoint_status",
                return_value="reachable on http://0.0.0.0:11435 via 127.0.0.1 (13 catalog models listed)",
            ),
            mock.patch(
                "llamacpp_stack.cli.get_public_endpoint_status",
                return_value="reachable on http://0.0.0.0:11436 via 127.0.0.1 (97 models listed)",
            ),
        ):
            info_text = build_info_text(args)

        self.assertIn("Default endpoints:", info_text)
        self.assertIn("Superserver API:       http://0.0.0.0:11435", info_text)
        self.assertIn("Installed versions:", info_text)
        self.assertIn("llama.cpp:           b8808", info_text)
        self.assertIn("llama-swap:          v202", info_text)
        self.assertIn("Runtime info:", info_text)
        self.assertIn("Install root:        /opt/llamacpp-superserver", info_text)
        self.assertIn("Models dir:          /workvols/data3/LLAMACPP_MODELS", info_text)
        self.assertIn("Service management:", info_text)
        self.assertIn("Install mode:        system", info_text)
        self.assertIn("Config knobs:", info_text)
        self.assertIn("API_CTX factor:", info_text)
        self.assertIn("API status:          reachable on http://0.0.0.0:11435 via 127.0.0.1 (13 catalog models listed)", info_text)
        self.assertIn("UI status:           reachable on http://0.0.0.0:11436 via 127.0.0.1 (97 models listed)", info_text)

    def test_service_commands_for_mode_use_user_scope_for_user_install(self) -> None:
        start_cmd, status_cmd, restart_cmd = service_commands_for_mode("user")
        self.assertIn("systemctl --user", start_cmd)
        self.assertIn("systemctl --user", status_cmd)
        self.assertIn("systemctl --user", restart_cmd)
        self.assertNotIn("sudo systemctl", restart_cmd)

    def test_service_commands_for_mode_use_system_scope_for_system_install(self) -> None:
        start_cmd, status_cmd, restart_cmd = service_commands_for_mode("system")
        self.assertIn("sudo systemctl", start_cmd)
        self.assertIn("sudo systemctl", status_cmd)
        self.assertIn("sudo systemctl", restart_cmd)
        self.assertNotIn("systemctl --user", restart_cmd)

    def test_manager_hint_shows_only_user_commands_when_user_mode_detected(self) -> None:
        with mock.patch("llamacpp_stack.cli.infer_install_mode", return_value="user"):
            hint = manager_hint()
        self.assertIn("Detected install mode: user", hint)
        self.assertIn("systemctl --user start", hint)
        self.assertNotIn("sudo systemctl start", hint)

    def test_infer_install_mode_prefers_explicit_env(self) -> None:
        with mock.patch.dict("os.environ", {"LLAMACPP_INSTALL_MODE": "user"}, clear=False):
            self.assertEqual(infer_install_mode(), "user")

    def test_cli_parser_accepts_single_dash_auto_alias_for_run(self) -> None:
        parser, _ = build_cli_parser()
        args = parser.parse_args(["run", "-hf", "org/repo:Q4_K_M", "-auto", "--no-chat"])
        self.assertEqual(args.command, "run")
        self.assertTrue(args.auto_ctx)

    def test_cli_parser_accepts_remove_alias_rm(self) -> None:
        parser, _ = build_cli_parser()
        args = parser.parse_args(["rm", "org/a:Q4", "org/b:Q5", "--delete-files"])
        self.assertEqual(args.command, "rm")
        self.assertEqual(args.repo, ["org/a:Q4", "org/b:Q5"])
        self.assertTrue(args.delete_files)

    def test_cli_parser_remove_deletes_files_by_default(self) -> None:
        parser, _ = build_cli_parser()
        args = parser.parse_args(["remove", "org/a:Q4"])
        self.assertEqual(args.command, "remove")
        self.assertTrue(args.delete_files)

    def test_cli_parser_remove_keep_files_disables_disk_deletion(self) -> None:
        parser, _ = build_cli_parser()
        args = parser.parse_args(["remove", "org/a:Q4", "--keep-files"])
        self.assertEqual(args.command, "remove")
        self.assertFalse(args.delete_files)

    def test_cli_parser_accepts_unload_all(self) -> None:
        parser, _ = build_cli_parser()
        args = parser.parse_args(["unload", "all"])
        self.assertEqual(args.command, "unload")
        self.assertEqual(args.repo, ["all"])

    def test_cli_parser_add_accepts_hf_list_with_plain_and_hfco_formats(self) -> None:
        parser, _ = build_cli_parser()
        args = parser.parse_args(
            [
                "add",
                "-hf",
                "Qwen/Qwen2.5-32B-Instruct-GGUF:Q4_K_M",
                "hf.co/Qwen/Qwen2.5-32B-Instruct-GGUF:Q4_K_M",
                "--skip-ctx",
            ]
        )
        self.assertEqual(
            args.hf,
            [
                "Qwen/Qwen2.5-32B-Instruct-GGUF:Q4_K_M",
                "hf.co/Qwen/Qwen2.5-32B-Instruct-GGUF:Q4_K_M",
            ],
        )
        self.assertEqual(args.repo, [])

    def test_add_models_dispatches_each_reference(self) -> None:
        args = Namespace(repo=["org/a:Q4", "hf.co/org/b:Q5"], hf=None, model_id=None)
        with mock.patch("llamacpp_stack.cli.ensure_model_available", side_effect=["a", "b"]) as add_mock:
            result = add_models(args)
        self.assertEqual(result, 0)
        self.assertEqual([call.args[0].repo for call in add_mock.call_args_list], ["org/a:Q4", "hf.co/org/b:Q5"])

    def test_remove_models_dispatches_each_reference(self) -> None:
        args = Namespace(repo=["org/a:Q4", "org/b:Q5"], hf=None, model_id=None, file=None)
        with mock.patch("llamacpp_stack.cli.remove_model", side_effect=["a", "b"]) as remove_mock:
            result = remove_models(args)
        self.assertEqual(result, 0)
        self.assertEqual([call.args[0].repo for call in remove_mock.call_args_list], ["org/a:Q4", "org/b:Q5"])

    def test_remove_models_defaults_delete_files_when_missing_attribute(self) -> None:
        args = Namespace(repo=["org/a:Q4"], hf=None, model_id=None, file=None, command="remove")
        with mock.patch("llamacpp_stack.cli.remove_model", side_effect=["a"]) as remove_mock:
            result = remove_models(args)
        self.assertEqual(result, 0)
        self.assertTrue(remove_mock.call_args_list[0].args[0].delete_files)

    def test_unload_models_keeps_files_and_rewrites_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "catalog.json"
            config_path = root / "config.yaml"
            models_dir = root / "models"

            model_a = ManagedModel(
                model_id="model-a",
                repo_id="org/a",
                quant="Q4",
                filename="a.gguf",
                local_path=str(models_dir / "a.gguf"),
            )
            model_b = ManagedModel(
                model_id="model-b",
                repo_id="org/b",
                quant="Q4",
                filename="b.gguf",
                local_path=str(models_dir / "b.gguf"),
            )
            save_catalog(catalog_path, [model_a, model_b])

            args = Namespace(
                repo=["model-a"],
                hf=None,
                model_id=None,
                file=None,
                catalog=catalog_path,
                config=config_path,
                llama_server=root / "llama-server",
                start_port=18080,
                public_host="127.0.0.1",
                public_port=11435,
                server_config=root / SERVER_CONFIG_BASENAME,
            )

            with mock.patch("llamacpp_stack.cli.wait_for_models_absent", return_value=True):
                result = unload_models(args)

            self.assertEqual(result, 0)
            rendered_text = config_path.read_text(encoding="utf-8")
            self.assertNotIn("model-a", rendered_text)
            self.assertIn("model-b", rendered_text)

    def test_remove_model_deletes_selected_file_when_repo_is_shared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            repo_dir = models_dir / "unsloth" / "Kimi-Dev-72B-GGUF"
            repo_dir.mkdir(parents=True)
            q4_0 = repo_dir / "Kimi-Dev-72B-Q4_0.gguf"
            q4_1 = repo_dir / "Kimi-Dev-72B-Q4_1.gguf"
            q4_0.write_bytes(b"q4_0")
            q4_1.write_bytes(b"q4_1")

            catalog_path = root / "state" / "catalog.json"
            catalog_path.parent.mkdir(parents=True)
            save_catalog(
                catalog_path,
                [
                    ManagedModel(
                        model_id="kimi-dev-72b-q4_0",
                        repo_id="unsloth/Kimi-Dev-72B-GGUF",
                        quant="Q4_0",
                        filename=q4_0.name,
                        local_path=str(q4_0),
                    ),
                    ManagedModel(
                        model_id="kimi-dev-72b-q4_1",
                        repo_id="unsloth/Kimi-Dev-72B-GGUF",
                        quant="Q4_1",
                        filename=q4_1.name,
                        local_path=str(q4_1),
                    ),
                ],
            )

            args = Namespace(
                catalog=catalog_path,
                config=root / "state" / "config.yaml",
                llama_server=root / "llama-server",
                start_port=18080,
                public_host="127.0.0.1",
                public_port=11435,
                models_dir=models_dir,
                repo="kimi-dev-72b-q4_0",
                hf=None,
                model_id=None,
                file=None,
                delete_files=True,
                server_config=root / "conf.json",
            )

            with (
                mock.patch("llamacpp_stack.cli.apply_config_and_wait_absent"),
                mock.patch("llamacpp_stack.cli.resolve_llama_server_defaults", return_value={}),
            ):
                removed = remove_model(args)

            self.assertEqual(removed, "kimi-dev-72b-q4_0")
            self.assertFalse(q4_0.exists())
            self.assertTrue(q4_1.exists())
            self.assertTrue(repo_dir.exists())

    def test_remove_model_prunes_empty_owner_folder_when_last_model_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            repo_dir = models_dir / "unsloth" / "Kimi-Dev-72B-GGUF"
            repo_dir.mkdir(parents=True)
            q4_0 = repo_dir / "Kimi-Dev-72B-Q4_0.gguf"
            q4_0.write_bytes(b"q4_0")

            catalog_path = root / "state" / "catalog.json"
            catalog_path.parent.mkdir(parents=True)
            save_catalog(
                catalog_path,
                [
                    ManagedModel(
                        model_id="kimi-dev-72b-q4_0",
                        repo_id="unsloth/Kimi-Dev-72B-GGUF",
                        quant="Q4_0",
                        filename=q4_0.name,
                        local_path=str(q4_0),
                    )
                ],
            )

            args = Namespace(
                catalog=catalog_path,
                config=root / "state" / "config.yaml",
                llama_server=root / "llama-server",
                start_port=18080,
                public_host="127.0.0.1",
                public_port=11435,
                models_dir=models_dir,
                repo="kimi-dev-72b-q4_0",
                hf=None,
                model_id=None,
                file=None,
                delete_files=True,
                server_config=root / "conf.json",
            )

            with (
                mock.patch("llamacpp_stack.cli.apply_config_and_wait_absent"),
                mock.patch("llamacpp_stack.cli.resolve_llama_server_defaults", return_value={}),
            ):
                remove_model(args)

            self.assertFalse(q4_0.exists())
            self.assertFalse(repo_dir.exists())
            self.assertFalse((models_dir / "unsloth").exists())
            self.assertTrue(models_dir.exists())

    def test_remove_model_missing_in_catalog_deletes_matching_orphan_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            repo_dir = models_dir / "unsloth" / "Kimi-Dev-72B-GGUF"
            repo_dir.mkdir(parents=True)
            orphan = repo_dir / "Kimi-Dev-72B-Q4_0.gguf"
            keep = repo_dir / "Kimi-Dev-72B-Q4_1.gguf"
            orphan.write_bytes(b"q4_0")
            keep.write_bytes(b"q4_1")

            catalog_path = root / "state" / "catalog.json"
            catalog_path.parent.mkdir(parents=True)
            save_catalog(catalog_path, [])

            args = Namespace(
                catalog=catalog_path,
                config=root / "state" / "config.yaml",
                llama_server=root / "llama-server",
                start_port=18080,
                public_host="127.0.0.1",
                public_port=11435,
                models_dir=models_dir,
                repo="kimi-dev-72b-q4_0",
                hf=None,
                model_id=None,
                file=None,
                delete_files=True,
                server_config=root / "conf.json",
            )

            removed = remove_model(args)

            self.assertEqual(removed, "kimi-dev-72b-q4_0")
            self.assertFalse(orphan.exists())
            self.assertTrue(keep.exists())

    def test_remove_model_missing_in_catalog_still_errors_with_keep_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            repo_dir = models_dir / "unsloth" / "Kimi-Dev-72B-GGUF"
            repo_dir.mkdir(parents=True)
            orphan = repo_dir / "Kimi-Dev-72B-Q4_0.gguf"
            orphan.write_bytes(b"q4_0")

            catalog_path = root / "state" / "catalog.json"
            catalog_path.parent.mkdir(parents=True)
            save_catalog(catalog_path, [])

            args = Namespace(
                catalog=catalog_path,
                config=root / "state" / "config.yaml",
                llama_server=root / "llama-server",
                start_port=18080,
                public_host="127.0.0.1",
                public_port=11435,
                models_dir=models_dir,
                repo="kimi-dev-72b-q4_0",
                hf=None,
                model_id=None,
                file=None,
                delete_files=False,
                server_config=root / "conf.json",
            )

            with self.assertRaisesRegex(RuntimeError, "Model not found in catalog"):
                remove_model(args)

            self.assertTrue(orphan.exists())

    def test_resolve_catalog_model_accepts_separator_variant_in_model_id(self) -> None:
        catalog = [
            ManagedModel(
                model_id="kimi-dev-72b-q4_0",
                repo_id="unsloth/Kimi-Dev-72B-GGUF",
                quant="Q4_0",
                filename="Kimi-Dev-72B-Q4_0.gguf",
                local_path="/models/unsloth/Kimi-Dev-72B-GGUF/Kimi-Dev-72B-Q4_0.gguf",
            )
        ]
        resolved = resolve_catalog_model(catalog, target="kimi-dev-72b-q4-0")
        self.assertEqual(resolved.model_id, "kimi-dev-72b-q4_0")

    def test_resolve_catalog_model_accepts_case_insensitive_model_id(self) -> None:
        catalog = [
            ManagedModel(
                model_id="kimi-dev-72b-q4_0",
                repo_id="unsloth/Kimi-Dev-72B-GGUF",
                quant="Q4_0",
                filename="Kimi-Dev-72B-Q4_0.gguf",
                local_path="/models/unsloth/Kimi-Dev-72B-GGUF/Kimi-Dev-72B-Q4_0.gguf",
            )
        ]
        resolved = resolve_catalog_model(catalog, target="KIMI-DEV-72B-Q4_0")
        self.assertEqual(resolved.model_id, "kimi-dev-72b-q4_0")

    def test_cli_parser_accepts_info_command(self) -> None:
        parser, _ = build_cli_parser()
        args = parser.parse_args(["info"])
        self.assertEqual(args.command, "info")

    def test_cli_main_rejects_dash_info_with_clear_message(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(sys, "argv", [CLI_COMMAND, "-info"]),
            redirect_stderr(stderr),
        ):
            with self.assertRaises(SystemExit) as ctx:
                cli_main()

        self.assertEqual(ctx.exception.code, 2)
        rendered = stderr.getvalue()
        self.assertIn("Use the 'info' subcommand without dashes", rendered)

    def test_update_models_dispatches_each_reference(self) -> None:
        args = Namespace(repo=["org/a:Q4", "org/b:Q5"], hf=None, model_id=None, file=None)
        with mock.patch("llamacpp_stack.cli.update_config", side_effect=["updated", "updated"]) as update_mock:
            result = update_models(args)
        self.assertEqual(result, 0)
        self.assertEqual([call.args[0].repo for call in update_mock.call_args_list], ["org/a:Q4", "org/b:Q5"])

    def test_cli_main_shows_subcommand_help_on_unrecognized_argument(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(sys, "argv", [CLI_COMMAND, "run", "-hf", "org/repo:Q4_K_M", "--not-a-real-flag"]),
            redirect_stderr(stderr),
        ):
            with self.assertRaises(SystemExit) as ctx:
                cli_main()

        self.assertEqual(ctx.exception.code, 2)
        rendered = stderr.getvalue()
        self.assertIn("unrecognized arguments: --not-a-real-flag", rendered)
        self.assertIn("Options for 'run':", rendered)
        self.assertIn(f"usage: {CLI_COMMAND} run", rendered)

    def test_normalize_model_id_strips_gguf_and_shard_suffix(self) -> None:
        self.assertEqual(
            normalize_model_id(
                "unsloth/MiniMax-M2.7-GGUF",
                "UD-Q4_K_M",
                "MiniMax-M2.7-UD-Q4_K_M-00001-of-00004.gguf",
            ),
            "minimax-m2.7-ud-q4_k_m",
        )
        self.assertEqual(
            normalize_model_id(
                ".",
                "IQ1_M",
                "DeepSeek-V3-0324.IQ1_M.gguf-00001-of-00009.gguf",
            ),
            "deepseek-v3-0324.iq1_m",
        )
        self.assertEqual(
            normalize_model_id(
                "UD-Q4_K_XL",
                "UD-Q4_K_XL",
                "Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf",
            ),
            "qwen3.5-122b-a10b-ud-q4_k_xl",
        )

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
            layout = choose_layout("system", "127.0.0.1", None, None, args=None)
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

    def test_detect_existing_backend_reads_manifest_backend(self) -> None:
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
            (layout.state_dir / "install-manifest.json").write_text('{"backend":"vllm-beta"}\n', encoding="utf-8")
            self.assertEqual(detect_existing_backend(layout), "vllm-beta")

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
            self.assertEqual(
                run_mock.call_args_list,
                [
                    mock.call([str(help_cmd), "list"], check=False),
                    mock.call([str(help_cmd), "--help"], check=False),
                ],
            )

    def test_backup_existing_model_configuration_copies_current_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state"
            config_dir = root / "config"
            state_dir.mkdir(parents=True)
            config_dir.mkdir(parents=True)
            (state_dir / "catalog.json").write_text("[]\n", encoding="utf-8")
            (state_dir / "config.yaml").write_text("models: {}\n", encoding="utf-8")
            (config_dir / SERVER_CONFIG_BASENAME).write_text('{"api_port":11436}\n', encoding="utf-8")

            layout = InstallLayout(
                mode="system",
                state_dir=state_dir,
                bin_dir=root / "bin",
                install_root=root / "install",
                models_dir=root / "models",
                config_dir=config_dir,
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

            backup_dir = _backup_existing_model_configuration(layout, dry_run=False)
            self.assertIsNotNone(backup_dir)
            self.assertTrue((backup_dir / "state" / "catalog.json").exists())
            self.assertTrue((backup_dir / "state" / "config.yaml").exists())
            self.assertTrue((backup_dir / "config" / SERVER_CONFIG_BASENAME).exists())

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
            def _failing_run(cmd, **kwargs):
                if cmd == [str(layout.bin_dir / CLI_COMMAND), "update", "--auto"]:
                    raise subprocess.CalledProcessError(1, cmd)
                # allow config-migrate to pass

            with (
                mock.patch("llamacpp_stack.install.prompt_bool", return_value=True),
                mock.patch("llamacpp_stack.install.wait_for_manager_socket", return_value=True),
                mock.patch("llamacpp_stack.install.sys.stdin.isatty", return_value=True),
                mock.patch(
                    "llamacpp_stack.install._run",
                    side_effect=_failing_run,
                ) as run_mock,
            ):
                maybe_rerun_auto_ctx(layout, install_services=True, dry_run=False, args=argparse.Namespace())
            run_mock.assert_any_call([str(layout.bin_dir / CLI_COMMAND), "update", "--auto"])
            run_mock.assert_any_call([str(layout.bin_dir / CLI_COMMAND), "config-migrate"])

    def test_maybe_rerun_auto_ctx_sync_uses_preserve_ctx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            state_dir = root / "state"
            state_dir.mkdir(parents=True)
            (state_dir / "catalog.json").write_text(
                '[{"model_id":"repo-q4","repo_id":"org/repo","filename":"model.gguf"}]\n',
                encoding="utf-8",
            )
            config_dir = root / "config"
            config_dir.mkdir(parents=True)

            layout = InstallLayout(
                mode="system",
                state_dir=state_dir,
                bin_dir=root / "bin",
                install_root=root / "install",
                models_dir=models_dir,
                config_dir=config_dir,
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
                mock.patch("llamacpp_stack.install.wait_for_manager_socket", return_value=True),
                mock.patch("llamacpp_stack.install._run") as run_mock,
            ):
                maybe_rerun_auto_ctx(
                    layout,
                    install_services=True,
                    dry_run=False,
                    args=argparse.Namespace(migrate_model_ids=False, rerun_auto_ctx=False),
                )

            run_mock.assert_any_call([str(layout.bin_dir / CLI_COMMAND), "update", "--preserve-ctx"])
            run_mock.assert_any_call([str(layout.bin_dir / CLI_COMMAND), "config-migrate"])

    def test_maybe_rerun_auto_ctx_repairs_stale_server_config_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            state_dir = root / "state"
            state_dir.mkdir(parents=True)
            catalog_path = state_dir / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "qwen3-coder-next-ud-q5_k_xl",
                            "repo_id": "local",
                            "quant": "UD-Q5_K_XL",
                            "filename": "Qwen3-Coder-Next-UD-Q5_K_XL-00001-of-00003.gguf",
                            "local_path": str(models_dir / "Qwen3-Coder-Next-UD-Q5_K_XL-00001-of-00003.gguf"),
                            "ctx_size": 260096,
                            "n_gpu_layers": 128,
                        }
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            server_config = config_dir / SERVER_CONFIG_BASENAME
            server_config.write_text(
                json.dumps(
                    {
                        "models": {
                            "qwen3-coder-next-ud-q5_k_xl": {"ctx_size": 8192, "n_gpu_layers": 999}
                        },
                        "llama_server_defaults": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            layout = InstallLayout(
                mode="system",
                state_dir=state_dir,
                bin_dir=root / "bin",
                install_root=root / "install",
                models_dir=models_dir,
                config_dir=config_dir,
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

            maybe_rerun_auto_ctx(
                layout,
                install_services=False,
                dry_run=False,
                args=argparse.Namespace(migrate_model_ids=False, rerun_auto_ctx=False),
            )

            payload = json.loads(server_config.read_text(encoding="utf-8"))
            self.assertNotIn("models", payload)
            catalog_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            model_cfg = catalog_payload[0]
            self.assertEqual(model_cfg["ctx_size"], 260096)
            self.assertEqual(model_cfg["n_gpu_layers"], 128)

    def test_maybe_rerun_auto_ctx_adds_server_config_metadata_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            state_dir = root / "state"
            state_dir.mkdir(parents=True)
            catalog_path = state_dir / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "repo-q4",
                            "repo_id": "org/repo",
                            "quant": "Q4",
                            "filename": "model-q4.gguf",
                            "local_path": str(models_dir / "model-q4.gguf"),
                            "ctx_size": 32768,
                            "n_gpu_layers": 777,
                        }
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            server_config = config_dir / SERVER_CONFIG_BASENAME
            server_config.write_text('{"models": {"repo-q4": {"ctx_size": 32768, "n_gpu_layers": 777}}}\n', encoding="utf-8")

            layout = InstallLayout(
                mode="system",
                state_dir=state_dir,
                bin_dir=root / "bin",
                install_root=root / "install",
                models_dir=models_dir,
                config_dir=config_dir,
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

            maybe_rerun_auto_ctx(
                layout,
                install_services=False,
                dry_run=False,
                args=argparse.Namespace(migrate_model_ids=False, rerun_auto_ctx=False),
            )
            payload = json.loads(server_config.read_text(encoding="utf-8"))
            self.assertIn("_meta", payload)
            self.assertNotIn("example", payload["_meta"])
            self.assertIn("Active settings are top-level keys only", payload["_meta"]["note"])

    def test_maybe_rerun_auto_ctx_preserves_existing_llama_server_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            state_dir = root / "state"
            state_dir.mkdir(parents=True)
            catalog_path = state_dir / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "repo-q4",
                            "repo_id": "org/repo",
                            "quant": "Q4",
                            "filename": "model-q4.gguf",
                            "local_path": str(models_dir / "model-q4.gguf"),
                            "ctx_size": 32768,
                            "n_gpu_layers": 777,
                        }
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            server_config = config_dir / SERVER_CONFIG_BASENAME
            existing_defaults = {
                "tensor_split": "0.94,1,1,1,1,1,0.94",
                "batch-size": 3064,
                "ubatch-size": 1024,
                "threads": 32,
                "threads-batch": 32,
                "numa": "distribute",
                "fit-target": 1536,
                "flash-attn": "on",
                "keep": 512,
                "mirostat": 2,
                "mirostat_ent": 4.5,
                "mirostat_lr": 0.1,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
            }
            server_config.write_text(
                json.dumps(
                    {
                        "idle_ttl": 300,
                        "api_port": 11436,
                        "llama_server_defaults": existing_defaults,
                        "models": {"repo-q4": {"ctx_size": 32768, "n_gpu_layers": 777}},
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            layout = InstallLayout(
                mode="system",
                state_dir=state_dir,
                bin_dir=root / "bin",
                install_root=root / "install",
                models_dir=models_dir,
                config_dir=config_dir,
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

            maybe_rerun_auto_ctx(
                layout,
                install_services=False,
                dry_run=False,
                args=argparse.Namespace(migrate_model_ids=False, rerun_auto_ctx=False),
            )

            payload = json.loads(server_config.read_text(encoding="utf-8"))
            self.assertNotIn("models", payload)
            defaults = payload["llama_server_defaults"]
            self.assertEqual(defaults["tensor_split"], "0.94,1,1,1,1,1,0.94")
            self.assertEqual(defaults["batch_size"], 3064)
            self.assertEqual(defaults["ubatch_size"], 1024)
            self.assertEqual(defaults["threads_batch"], 32)
            self.assertEqual(defaults["fit_target"], 1536)
            self.assertEqual(defaults["flash_attn"], "on")
            self.assertFalse(defaults["use_fitc"])
            self.assertTrue(defaults["swa_full"])

    def test_maybe_rerun_auto_ctx_migrates_legacy_conf_models_to_catalog_and_removes_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            state_dir = root / "state"
            state_dir.mkdir(parents=True)
            catalog_path = state_dir / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "repo-q4",
                            "repo_id": "org/repo",
                            "quant": "Q4",
                            "filename": "model-q4.gguf",
                            "local_path": str(models_dir / "model-q4.gguf"),
                        }
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            server_config = config_dir / SERVER_CONFIG_BASENAME
            server_config.write_text(
                json.dumps(
                    {
                        "idle_ttl": 300,
                        "api_port": 11436,
                        "llama_server_defaults": {},
                        "models": {
                            "repo-q4": {
                                "ctx_size": 32768,
                                "n_gpu_layers": 777,
                                "tensor_split": "1,1,1",
                                "aliases": ["legacy-alias"],
                                "description": "legacy description",
                            }
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            layout = InstallLayout(
                mode="system",
                state_dir=state_dir,
                bin_dir=root / "bin",
                install_root=root / "install",
                models_dir=models_dir,
                config_dir=config_dir,
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

            maybe_rerun_auto_ctx(
                layout,
                install_services=False,
                dry_run=False,
                args=argparse.Namespace(migrate_model_ids=False, rerun_auto_ctx=False),
            )

            payload = json.loads(server_config.read_text(encoding="utf-8"))
            self.assertNotIn("models", payload)
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            model = catalog[0]
            self.assertEqual(model["ctx_size"], 32768)
            self.assertEqual(model["n_gpu_layers"], 777)
            self.assertEqual(model["tensor_split"], "1,1,1")
            self.assertIn("legacy-alias", model["aliases"])
            self.assertEqual(model["description"], "legacy description")

    def test_maybe_rerun_auto_ctx_merges_missing_llama_server_defaults_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            state_dir = root / "state"
            state_dir.mkdir(parents=True)
            catalog_path = state_dir / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "repo-q4",
                            "repo_id": "org/repo",
                            "quant": "Q4",
                            "filename": "model-q4.gguf",
                            "local_path": str(models_dir / "model-q4.gguf"),
                            "ctx_size": 32768,
                            "n_gpu_layers": 777,
                        }
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            server_config = config_dir / SERVER_CONFIG_BASENAME
            server_config.write_text(
                json.dumps(
                    {
                        "idle_ttl": 300,
                        "api_port": 11436,
                        "llama_server_defaults": {
                            "keep": 256,
                            "cache_type_k": "q4_0",
                        },
                        "models": {"repo-q4": {"ctx_size": 32768, "n_gpu_layers": 777}},
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            layout = InstallLayout(
                mode="system",
                state_dir=state_dir,
                bin_dir=root / "bin",
                install_root=root / "install",
                models_dir=models_dir,
                config_dir=config_dir,
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

            with mock.patch("llamacpp_stack.install.detect_cuda_device_count", return_value=7):
                maybe_rerun_auto_ctx(
                    layout,
                    install_services=False,
                    dry_run=False,
                    args=argparse.Namespace(migrate_model_ids=False, rerun_auto_ctx=False),
                )

            payload = json.loads(server_config.read_text(encoding="utf-8"))
            llama_defaults = payload["llama_server_defaults"]
            self.assertEqual(llama_defaults["keep"], 256)
            self.assertEqual(llama_defaults["cache_type_k"], "q4_0")
            self.assertNotIn("models", payload)
            self.assertEqual(llama_defaults["batch_size"], 4096)
            self.assertEqual(llama_defaults["ubatch_size"], 2048)
            self.assertEqual(llama_defaults["threads"], 16)
            self.assertEqual(llama_defaults["threads_batch"], 8)
            self.assertNotIn("fit_target", llama_defaults)
            self.assertEqual(llama_defaults["parallel"], 1)
            self.assertFalse(llama_defaults["use_fitc"])
            self.assertTrue(llama_defaults["swa_full"])
            self.assertTrue(llama_defaults["flash_attn"])

    def test_maybe_rerun_auto_ctx_refreshes_server_config_after_auto_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            state_dir = root / "state"
            state_dir.mkdir(parents=True)
            catalog_path = state_dir / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "qwen3-coder-next-ud-q5_k_xl",
                            "repo_id": "local",
                            "quant": "UD-Q5_K_XL",
                            "filename": "Qwen3-Coder-Next-UD-Q5_K_XL-00001-of-00003.gguf",
                            "local_path": str(models_dir / "Qwen3-Coder-Next-UD-Q5_K_XL-00001-of-00003.gguf"),
                            "ctx_size": 8192,
                            "n_gpu_layers": 999,
                        }
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            server_config = config_dir / SERVER_CONFIG_BASENAME
            server_config.write_text(
                json.dumps(
                    {
                        "models": {
                            "qwen3-coder-next-ud-q5_k_xl": {"ctx_size": 8192, "n_gpu_layers": 999}
                        },
                        "llama_server_defaults": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            layout = InstallLayout(
                mode="system",
                state_dir=state_dir,
                bin_dir=root / "bin",
                install_root=root / "install",
                models_dir=models_dir,
                config_dir=config_dir,
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

            def _fake_run(cmd, **kwargs):
                if cmd == [str(layout.bin_dir / CLI_COMMAND), "config-migrate"]:
                    return
                if cmd == [str(layout.bin_dir / CLI_COMMAND), "update", "--auto"]:
                    updated_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
                    updated_catalog[0]["ctx_size"] = 260096
                    updated_catalog[0]["n_gpu_layers"] = 128
                    catalog_path.write_text(json.dumps(updated_catalog) + "\n", encoding="utf-8")
                    return
                raise AssertionError(f"Unexpected command: {cmd}")

            with (
                mock.patch("llamacpp_stack.install.wait_for_manager_socket", return_value=True),
                mock.patch("llamacpp_stack.install._run", side_effect=_fake_run) as run_mock,
            ):
                maybe_rerun_auto_ctx(
                    layout,
                    install_services=True,
                    dry_run=False,
                    args=argparse.Namespace(migrate_model_ids=False, rerun_auto_ctx=True),
                )

            run_mock.assert_any_call([str(layout.bin_dir / CLI_COMMAND), "update", "--auto"])
            run_mock.assert_any_call([str(layout.bin_dir / CLI_COMMAND), "config-migrate"])

        def test_maybe_rerun_auto_ctx_seeds_editable_llama_server_defaults_file(self) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                models_dir = root / "models"
                models_dir.mkdir(parents=True)
                state_dir = root / "state"
                state_dir.mkdir(parents=True)
                catalog_path = state_dir / "catalog.json"
                catalog_path.write_text(
                    json.dumps(
                        [
                            {
                                "model_id": "repo-q4",
                                "repo_id": "org/repo",
                                "quant": "Q4",
                                "filename": "model-q4.gguf",
                                "local_path": str(models_dir / "model-q4.gguf"),
                                "ctx_size": 32768,
                                "n_gpu_layers": 777,
                            }
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                config_dir = root / "config"
                config_dir.mkdir(parents=True)
                server_config = config_dir / SERVER_CONFIG_BASENAME
                server_config.write_text(
                    json.dumps({"idle_ttl": 300, "api_port": 11436, "models": {"repo-q4": {"ctx_size": 32768, "n_gpu_layers": 777}}}, indent=2)
                    + "\n",
                    encoding="utf-8",
                )

                layout = InstallLayout(
                    mode="system",
                    state_dir=state_dir,
                    bin_dir=root / "bin",
                    install_root=root / "install",
                    models_dir=models_dir,
                    config_dir=config_dir,
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

                with mock.patch("llamacpp_stack.install.detect_cuda_device_count", return_value=7):
                    maybe_rerun_auto_ctx(
                        layout,
                        install_services=False,
                        dry_run=False,
                        args=argparse.Namespace(migrate_model_ids=False, rerun_auto_ctx=False),
                    )

                preset_file = config_dir / LLAMA_SERVER_DEFAULTS_BASENAME
                self.assertTrue(preset_file.exists())
                self.assertIn("presets:", preset_file.read_text(encoding="utf-8"))

                payload = json.loads(server_config.read_text(encoding="utf-8"))
                llama_defaults = payload["llama_server_defaults"]
                self.assertEqual(llama_defaults["keep"], 512)
                self.assertEqual(llama_defaults["batch-size"], 3064)
                self.assertEqual(llama_defaults["ubatch-size"], 1024)
                self.assertEqual(llama_defaults["threads-batch"], 32)
                self.assertEqual(llama_defaults["fit-target"], 1536)
                self.assertTrue(llama_defaults["flash-attn"])
                self.assertEqual(llama_defaults["tensor_split"], "0.94,1,1,1,1,1,0.94")
                self.assertEqual(llama_defaults["mirostat"], 2)
                self.assertEqual(llama_defaults["mirostat_ent"], 4.5)
                self.assertEqual(llama_defaults["mirostat_lr"], 0.1)
            payload = json.loads(server_config.read_text(encoding="utf-8"))
            self.assertNotIn("models", payload)
            catalog_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            model_cfg = catalog_payload[0]
            self.assertEqual(model_cfg["ctx_size"], 260096)
            self.assertEqual(model_cfg["n_gpu_layers"], 128)

    def test_maybe_rerun_auto_ctx_migrates_model_ids_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            state_dir = root / "state"
            state_dir.mkdir(parents=True)
            catalog_path = state_dir / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "minimax-m2.7-gguf-ud-q4_k_m",
                            "repo_id": "unsloth/MiniMax-M2.7-GGUF",
                            "quant": "UD-Q4_K_M",
                            "filename": "MiniMax-M2.7-UD-Q4_K_M-00001-of-00004.gguf",
                            "local_path": str(models_dir / "MiniMax-M2.7-UD-Q4_K_M-00001-of-00004.gguf"),
                        }
                    ]
                )
                + "\n",
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
            args = argparse.Namespace(migrate_model_ids=True, rerun_auto_ctx=False)
            maybe_rerun_auto_ctx(layout, install_services=False, dry_run=False, args=args)
            updated = json.loads(catalog_path.read_text(encoding="utf-8"))
            self.assertEqual(updated[0]["model_id"], "minimax-m2.7-ud-q4_k_m")

    def test_maybe_rerun_auto_ctx_migration_keeps_speculative_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            state_dir = root / "state"
            state_dir.mkdir(parents=True)
            catalog_path = state_dir / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "qwen3.6-35b-a3b-gguf-ud-q4_k_m",
                            "repo_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                            "quant": "UD-Q4_K_M",
                            "filename": "Qwen3.6-35B-A3B-UD-Q4_K_M-00001-of-00002.gguf",
                            "local_path": str(models_dir / "Qwen3.6-35B-A3B-UD-Q4_K_M-00001-of-00002.gguf"),
                        },
                        {
                            "model_id": "speculative-qwen3.6-35b-a3b-ud-q4_k_m",
                            "repo_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                            "quant": "UD-Q4_K_M",
                            "filename": "Qwen3.6-35B-A3B-UD-Q4_K_M-00001-of-00002.gguf",
                            "local_path": str(models_dir / "Qwen3.6-35B-A3B-UD-Q4_K_M-00001-of-00002.gguf"),
                            "speculative": True,
                            "spec_variant_of": "qwen3.6-35b-a3b-gguf-ud-q4_k_m",
                            "spec_meta": {
                                "base_model_id": "qwen3.6-35b-a3b-gguf-ud-q4_k_m",
                                "draft_model_id": "qwen3.6-35b-a3b-ud-iq1_m",
                            },
                            "server_overrides": {
                                "model_draft": str(models_dir / "Qwen3.6-35B-A3B-UD-IQ1_M.gguf"),
                            },
                        },
                    ]
                )
                + "\n",
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

            args = argparse.Namespace(migrate_model_ids=True, rerun_auto_ctx=False)
            maybe_rerun_auto_ctx(layout, install_services=False, dry_run=False, args=args)

            updated = json.loads(catalog_path.read_text(encoding="utf-8"))
            model_ids = [str(item.get("model_id") or "") for item in updated]
            self.assertIn("qwen3.6-35b-a3b-ud-q4_k_m", model_ids)
            self.assertIn("speculative-qwen3.6-35b-a3b-ud-q4_k_m", model_ids)
            self.assertNotIn("qwen3.6-35b-a3b-ud-q4_k_m-2", model_ids)

    def test_maybe_rerun_auto_ctx_migration_prefers_filename_over_repo_folder_slug(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            state_dir = root / "state"
            state_dir.mkdir(parents=True)
            catalog_path = state_dir / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "q8_0",
                            "repo_id": "Q8_0",
                            "quant": "Q8_0",
                            "filename": "Mistral-Small-4-119B-2603-Q8_0-00001-of-00004.gguf",
                            "local_path": str(models_dir / "Mistral-Small-4-119B-2603-Q8_0-00001-of-00004.gguf"),
                        },
                        {
                            "model_id": "ud-q4_k_xl",
                            "repo_id": "UD-Q4_K_XL",
                            "quant": "UD-Q4_K_XL",
                            "filename": "NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_XL-00001-of-00003.gguf",
                            "local_path": str(models_dir / "NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_XL-00001-of-00003.gguf"),
                        },
                        {
                            "model_id": "ud-q4_k_xl-2",
                            "repo_id": "UD-Q4_K_XL",
                            "quant": "UD-Q4_K_XL",
                            "filename": "Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf",
                            "local_path": str(models_dir / "Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf"),
                        },
                    ]
                )
                + "\n",
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
            args = argparse.Namespace(migrate_model_ids=True, rerun_auto_ctx=False)
            maybe_rerun_auto_ctx(layout, install_services=False, dry_run=False, args=args)
            updated = json.loads(catalog_path.read_text(encoding="utf-8"))
            renamed_ids = [item["model_id"] for item in updated]
            self.assertIn("mistral-small-4-119b-2603-q8_0", renamed_ids)
            self.assertIn("nvidia-nemotron-3-super-120b-a12b-ud-q4_k_xl", renamed_ids)
            self.assertIn("qwen3.5-122b-a10b-ud-q4_k_xl", renamed_ids)

    def test_maybe_rerun_auto_ctx_migration_prompt_shows_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            state_dir = root / "state"
            state_dir.mkdir(parents=True)
            catalog_path = state_dir / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "ud-q2_k_xl",
                            "repo_id": "UD-Q2_K_XL",
                            "quant": "UD-Q2_K_XL",
                            "filename": "MiniMax-M2.5-UD-Q2_K_XL-00001-of-00003.gguf",
                            "local_path": str(models_dir / "MiniMax-M2.5-UD-Q2_K_XL-00001-of-00003.gguf"),
                        }
                    ]
                )
                + "\n",
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
                mock.patch("llamacpp_stack.install.sys.stdin.isatty", return_value=True),
                mock.patch("llamacpp_stack.install.prompt_bool", return_value=False),
                mock.patch("builtins.print") as print_mock,
            ):
                maybe_rerun_auto_ctx(layout, install_services=False, dry_run=False, args=argparse.Namespace(rerun_auto_ctx=False))

            rendered = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
            self.assertIn("Preview of model ID renames:", rendered)
            self.assertIn("ud-q2_k_xl -> minimax-m2.5-ud-q2_k_xl", rendered)

    def test_maybe_rerun_auto_ctx_keeps_model_ids_when_migration_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            models_dir.mkdir(parents=True)
            state_dir = root / "state"
            state_dir.mkdir(parents=True)
            catalog_path = state_dir / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "minimax-m2.7-gguf-ud-q4_k_m",
                            "repo_id": "unsloth/MiniMax-M2.7-GGUF",
                            "quant": "UD-Q4_K_M",
                            "filename": "MiniMax-M2.7-UD-Q4_K_M-00001-of-00004.gguf",
                            "local_path": str(models_dir / "MiniMax-M2.7-UD-Q4_K_M-00001-of-00004.gguf"),
                        }
                    ]
                )
                + "\n",
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
            args = argparse.Namespace(migrate_model_ids=False, rerun_auto_ctx=False)
            maybe_rerun_auto_ctx(layout, install_services=False, dry_run=False, args=args)
            updated = json.loads(catalog_path.read_text(encoding="utf-8"))
            self.assertEqual(updated[0]["model_id"], "minimax-m2.7-gguf-ud-q4_k_m")

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

    def test_maybe_rerun_auto_ctx_backfills_aliases_and_prunes_default_ttl_for_existing_catalog_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models" / "org" / "repo"
            models_dir.mkdir(parents=True)
            model_path = models_dir / "example-model-q4_k_m-00001-of-00002.gguf"
            model_path.write_bytes(b"gguf")

            state_dir = root / "state"
            state_dir.mkdir(parents=True)
            catalog_path = state_dir / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "example-model-q4_k_m",
                            "repo_id": "org/repo",
                            "quant": "Q4_K_M",
                            "filename": model_path.name,
                            "local_path": str(model_path),
                            "aliases": [],
                            "ttl": 300,
                        }
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config_dir = root / "config"
            config_dir.mkdir(parents=True)
            (config_dir / SERVER_CONFIG_BASENAME).write_text(
                json.dumps({"idle_ttl": 300, "models": {}, "llama_server_defaults": {}}, indent=2) + "\n",
                encoding="utf-8",
            )

            layout = InstallLayout(
                mode="system",
                state_dir=state_dir,
                bin_dir=root / "bin",
                install_root=root / "install",
                models_dir=root / "models",
                config_dir=config_dir,
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

            maybe_rerun_auto_ctx(
                layout,
                install_services=False,
                dry_run=False,
                args=argparse.Namespace(rerun_auto_ctx=False, migrate_model_ids=False),
            )

            updated = json.loads(catalog_path.read_text(encoding="utf-8"))
            self.assertEqual(len(updated), 1)
            model = updated[0]
            aliases = model.get("aliases") or []
            self.assertIn("example-model-q4_k_m-00001-of-00002.gguf", aliases)
            self.assertIn("example-model-q4_k_m-00001-of-00002", aliases)
            self.assertIn("example-model-q4_k_m", aliases)
            self.assertIn("hf.co/org/repo", aliases)
            self.assertIn("org/repo", aliases)
            self.assertIn("hf.co/org/repo:Q4_K_M", aliases)
            self.assertIn("org/repo:Q4_K_M", aliases)
            self.assertNotIn("ttl", model)

    def test_sync_cuda_runtime_reuses_existing_runtime_when_python_env_has_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing_cuda = root / "cuda"
            (existing_cuda / "lib64").mkdir(parents=True)
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
                cuda_root=existing_cuda,
            )
            with mock.patch("llamacpp_stack.install.locate_cuda_root_for_python", return_value=None):
                self.assertEqual(sync_cuda_runtime(layout, sys.executable, dry_run=False), existing_cuda)

    def test_sync_nccl_runtime_reuses_existing_runtime_when_python_env_has_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing_nccl = root / "install" / "nccl"
            (existing_nccl / "lib").mkdir(parents=True)
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
                nccl_root=existing_nccl,
            )
            with mock.patch("llamacpp_stack.install.locate_nccl_root_for_python", return_value=None):
                self.assertEqual(sync_nccl_runtime(layout, sys.executable, dry_run=False), existing_nccl)


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
                maybe_rerun_auto_ctx(layout, install_services=True, dry_run=False, args=argparse.Namespace())

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
            with (
                mock.patch("llamacpp_stack.install._sudo_prefix", return_value=[]),
                mock.patch("llamacpp_stack.install._run") as run_mock,
            ):
                self.assertTrue(restart_systemd_units(layout, dry_run=False))
            run_mock.assert_called_once_with(["systemctl", "restart", MANAGER_SERVICE_NAME, SWAP_SERVICE_NAME])

    def test_restart_systemd_units_uses_sudo_when_not_root(self) -> None:
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
            with (
                mock.patch("llamacpp_stack.install._sudo_prefix", return_value=["sudo"]),
                mock.patch("llamacpp_stack.install._run") as run_mock,
            ):
                self.assertTrue(restart_systemd_units(layout, dry_run=False))
            run_mock.assert_called_once_with(["sudo", "systemctl", "restart", MANAGER_SERVICE_NAME, SWAP_SERVICE_NAME])

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
            with (
                mock.patch("llamacpp_stack.install._sudo_prefix", return_value=[]),
                mock.patch("llamacpp_stack.install._run") as run_mock,
            ):
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

    def test_unsupported_global_defaults_are_filtered_but_explicit_overrides_emit(self) -> None:
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model-q4.gguf",
            local_path="/tmp/model-q4.gguf",
            server_overrides={"parallel": 2},
        )
        with mock.patch("llamacpp_stack.cli.server_supports_flag", return_value=False):
            rendered = " ".join(
                build_llama_server_command(
                    model,
                    Path("/opt/llama-server"),
                    port="18080",
                    server_defaults={"mul_mat_q": True, "grp_attn_n": 1},
                )
            )
        self.assertIn("--parallel 2", rendered)
        self.assertNotIn("--mul-mat-q", rendered)
        self.assertNotIn("--grp-attn-n", rendered)

    def test_server_override_tensor_split_is_migrated_to_model_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "repo-q4",
                            "repo_id": "org/repo",
                            "quant": "Q4",
                            "filename": "model-q4.gguf",
                            "local_path": str(root / "model-q4.gguf"),
                            "tensor_split": "1,1,1,1,1,1,1",
                            "server_overrides": {"parallel": 2, "tensor_split": "1,1,1"},
                        }
                    ]
                ),
                encoding="utf-8",
            )

            model = load_catalog(catalog_path)[0]
            rendered = " ".join(build_llama_server_command(model, root / "llama-server", port="18080"))

            self.assertEqual(model.tensor_split, "1,1,1")
            self.assertEqual(model.server_overrides, {"parallel": 2})
            self.assertIn("--tensor-split 1,1,1", rendered)
            self.assertIn("--parallel 2", rendered)
            saved = json.loads(catalog_path.read_text(encoding="utf-8"))[0]
            self.assertEqual(saved["tensor_split"], "1,1,1")
            self.assertNotIn("tensor_split", saved["server_overrides"])

    def test_catalog_tensor_split_is_preserved_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "repo-q4",
                            "repo_id": "org/repo",
                            "quant": "Q4",
                            "filename": "model-q4.gguf",
                            "local_path": str(root / "model-q4.gguf"),
                            "tensor_split": "1,1,1",
                            "parallel": 2,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=7):
                model = load_catalog(catalog_path)[0]
                rendered = " ".join(build_llama_server_command(model, root / "llama-server", port="18080"))

            self.assertEqual(model.tensor_split, "1,1,1")
            self.assertIn("--tensor-split 1,1,1", rendered)
            self.assertIn("/usr/bin/env CUDA_VISIBLE_DEVICES=0,1,2", rendered)
            self.assertIn("--parallel 2", rendered)
            self.assertNotIn("--tensor-split 1,1,1,1,1,1,1", rendered)

    def test_unknown_catalog_keys_become_llama_server_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "repo-q4",
                            "repo_id": "org/repo",
                            "quant": "Q4",
                            "filename": "model-q4.gguf",
                            "local_path": str(root / "model-q4.gguf"),
                            "parallel": 2,
                            "example": None,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            model = load_catalog(catalog_path)[0]
            self.assertEqual(model.server_overrides["parallel"], 2)
            self.assertIsNone(model.server_overrides["example"])
            rendered = " ".join(build_llama_server_command(model, root / "llama-server", port="18080"))
            self.assertIn("--parallel 2", rendered)
            self.assertIn("--example", rendered)

    def test_server_config_payload_adds_replicas_and_canonicalizes_defaults(self) -> None:
        payload = {
            "idle_ttl": 600,
            "api_port": 11435,
            "llama_server_defaults": {
                "batch-size": 4048,
                "batch_size": 4096,
                "threads-batch": 16,
                "threads_batch": 8,
                "flash-attn": True,
                "flash_attn": True,
                "n-gpu-layers": -1,
                "grp_attn_n": 16,
                "use-fitc": False,
                "swa-full": True,
            },
        }
        normalized, changed = normalize_server_config_payload(payload)
        self.assertTrue(changed)
        self.assertIn("replicas", normalized)
        self.assertEqual(normalized["replicas"]["enabled"], False)
        self.assertIn("experimental", normalized)
        self.assertEqual(
            normalized["experimental"]["chat_tool_continue_repair"],
            {"enabled": False, "max_rounds": 1, "max_tokens": 2048, "stream_keepalive_seconds": 15},
        )
        self.assertNotIn("example", normalized["_meta"])
        defaults = normalized["llama_server_defaults"]
        self.assertEqual(defaults["batch_size"], 4096)
        self.assertEqual(defaults["threads_batch"], 8)
        self.assertEqual(defaults["n_gpu_layers"], -1)
        self.assertEqual(defaults["use_fitc"], False)
        self.assertEqual(defaults["swa_full"], True)
        self.assertIn("flash_attn", defaults)
        self.assertNotIn("batch-size", defaults)
        self.assertNotIn("threads-batch", defaults)
        self.assertNotIn("flash-attn", defaults)
        self.assertNotIn("n-gpu-layers", defaults)
        self.assertNotIn("use-fitc", defaults)
        self.assertNotIn("swa-full", defaults)

    def test_config_migrate_rewrites_old_metadata_and_adds_replicas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server_config = Path(tmp) / SERVER_CONFIG_BASENAME
            server_config.write_text(
                json.dumps(
                    {
                        "idle_ttl": 600,
                        "api_port": 11435,
                        "llama_server_defaults": {
                            "batch-size": 4048,
                            "batch_size": 4096,
                            "flash-attn": True,
                            "n-gpu-layers": -1,
                        },
                        "_meta": {
                            "purpose": "old",
                            "example": {
                                "idle_ttl": 300,
                                "api_port": 11436,
                                "llama_server_defaults": {"ctx_size": 65536},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = Namespace(
                server_config=server_config,
                idle_ttl=None,
                api_port=None,
                api_ctx_factor=None,
            )

            self.assertEqual(migrate_server_config(args), 0)

            payload = json.loads(server_config.read_text(encoding="utf-8"))
            self.assertIn("replicas", payload)
            self.assertEqual(payload["replicas"]["placement"], "exclusive_gpus")
            self.assertNotIn("example", payload["_meta"])
            defaults = payload["llama_server_defaults"]
            self.assertEqual(defaults["batch_size"], 4096)
            self.assertEqual(defaults["flash_attn"], True)
            self.assertEqual(defaults["n_gpu_layers"], -1)
            self.assertNotIn("batch-size", defaults)
            self.assertNotIn("flash-attn", defaults)
            self.assertNotIn("n-gpu-layers", defaults)

    def test_update_config_migrates_server_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "catalog.json"
            config_path = root / "config.yaml"
            server_config = root / SERVER_CONFIG_BASENAME
            save_catalog(catalog_path, [ManagedModel("repo-q4", "org/repo", "Q4", "model.gguf", "/tmp/model.gguf")])
            server_config.write_text(
                json.dumps(
                    {
                        "idle_ttl": 600,
                        "api_port": 11435,
                        "llama_server_defaults": {
                            "batch-size": 4048,
                            "batch_size": 4096,
                            "threads-batch": 16,
                            "threads_batch": 8,
                            "flash-attn": True,
                            "flash_attn": True,
                            "n-gpu-layers": -1,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            args = Namespace(
                repo=None,
                hf=None,
                model_id=None,
                file=None,
                ctx_override=None,
                auto_ctx=False,
                preserve_ctx=False,
                sync_gguf_ctx=False,
                catalog=catalog_path,
                config=config_path,
                llama_server=root / "llama-server",
                start_port=18080,
                public_host="127.0.0.1",
                public_port=11435,
                idle_ttl=None,
                api_port=None,
                api_ctx_factor=None,
                server_config=server_config,
            )
            response = mock.Mock(status_code=200)
            response.json.return_value = {"data": []}
            with (
                mock.patch("llamacpp_stack.cli.time.sleep"),
                mock.patch("llamacpp_stack.cli.requests.get", return_value=response),
            ):
                self.assertEqual(update_config(args), "updated")
            payload = json.loads(server_config.read_text(encoding="utf-8"))
            self.assertIn("replicas", payload)
            defaults = payload["llama_server_defaults"]
            self.assertEqual(defaults["batch_size"], 4096)
            self.assertEqual(defaults["threads_batch"], 8)
            self.assertEqual(defaults["flash_attn"], True)
            self.assertEqual(defaults["n_gpu_layers"], -1)
            self.assertNotIn("batch-size", defaults)
            self.assertNotIn("threads-batch", defaults)
            self.assertNotIn("flash-attn", defaults)
            self.assertNotIn("n-gpu-layers", defaults)

    def test_update_config_refuses_invalid_catalog_without_overwriting_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "catalog.json"
            config_path = root / "config.yaml"
            catalog_path.write_text('[{"model_id": "broken", bad json]', encoding="utf-8")
            config_path.write_text("sentinel: keep\n", encoding="utf-8")
            args = Namespace(
                repo=None,
                hf=None,
                model_id=None,
                file=None,
                ctx_override=None,
                auto_ctx=False,
                preserve_ctx=False,
                sync_gguf_ctx=False,
                catalog=catalog_path,
                config=config_path,
                llama_server=root / "llama-server",
                start_port=18080,
                public_host="127.0.0.1",
                public_port=11435,
                idle_ttl=10,
                server_config=root / SERVER_CONFIG_BASENAME,
            )

            with self.assertRaisesRegex(RuntimeError, "Refusing to update from an invalid catalog"):
                update_config(args)

            self.assertEqual(catalog_path.read_text(encoding="utf-8"), '[{"model_id": "broken", bad json]')
            self.assertEqual(config_path.read_text(encoding="utf-8"), "sentinel: keep\n")

    def test_update_config_renders_parallel_override_from_catalog(self) -> None:
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
                        server_overrides={"parallel": 2},
                    )
                ],
            )
            args = Namespace(
                repo=None,
                hf=None,
                model_id=None,
                file=None,
                ctx_override=None,
                auto_ctx=False,
                preserve_ctx=False,
                sync_gguf_ctx=False,
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
                mock.patch("llamacpp_stack.cli.time.sleep"),
                mock.patch("llamacpp_stack.cli.requests.get", return_value=response),
                mock.patch("llamacpp_stack.cli.server_supports_flag", return_value=True),
            ):
                result = update_config(args)

            self.assertEqual(result, "updated")
            rendered = config_path.read_text(encoding="utf-8")
            self.assertIn("--parallel 2", rendered)

    def test_update_config_preserves_ctx_by_default(self) -> None:
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
                        ctx_size=24576,
                    )
                ],
            )
            args = Namespace(
                repo=None,
                hf=None,
                model_id=None,
                file=None,
                ctx_override=None,
                auto_ctx=False,
                preserve_ctx=False,
                sync_gguf_ctx=False,
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
                mock.patch("llamacpp_stack.cli.sync_catalog_context_sizes") as sync_mock,
                mock.patch("llamacpp_stack.cli.time.sleep"),
                mock.patch("llamacpp_stack.cli.requests.get", return_value=response),
            ):
                result = update_config(args)

            self.assertEqual(result, "updated")
            sync_mock.assert_not_called()
            refreshed = load_catalog(catalog_path)
            self.assertEqual(refreshed[0].ctx_size, 24576)

    def test_update_config_ctx_override_refreshes_load_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "catalog.json"
            config_path = root / "config.yaml"
            model_path = root / "models" / "model-q4.gguf"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"gguf")
            save_catalog(
                catalog_path,
                [
                    ManagedModel(
                        model_id="repo-q4",
                        repo_id="org/repo",
                        quant="Q4",
                        filename="model-q4.gguf",
                        local_path=str(model_path),
                        ctx_size=24576,
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
                preserve_ctx=False,
                sync_gguf_ctx=False,
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
                mock.patch("llamacpp_stack.cli.refresh_model_load_capabilities") as refresh_mock,
                mock.patch("llamacpp_stack.cli.time.sleep"),
                mock.patch("llamacpp_stack.cli.requests.get", return_value=response),
            ):
                result = update_config(args)

            self.assertEqual(result, "updated")
            refresh_mock.assert_called()

    def test_update_config_auto_ctx_min_failed_prompts_and_deletes_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            fail_repo_dir = models_dir / "org/fail-repo"
            ok_repo_dir = models_dir / "org/ok-repo"
            fail_repo_dir.mkdir(parents=True)
            ok_repo_dir.mkdir(parents=True)

            fail_model_path = fail_repo_dir / "fail.gguf"
            ok_model_path = ok_repo_dir / "ok.gguf"
            fail_model_path.write_bytes(b"gguf")
            ok_model_path.write_bytes(b"gguf")

            catalog_path = root / "catalog.json"
            config_path = root / "config.yaml"
            save_catalog(
                catalog_path,
                [
                    ManagedModel(
                        model_id="fail-model",
                        repo_id="org/fail-repo",
                        quant="Q8_0",
                        filename="fail.gguf",
                        local_path=str(fail_model_path),
                        ctx_size=8192,
                    ),
                    ManagedModel(
                        model_id="ok-model",
                        repo_id="org/ok-repo",
                        quant="Q4_K_M",
                        filename="ok.gguf",
                        local_path=str(ok_model_path),
                        ctx_size=4096,
                    ),
                ],
            )

            args = Namespace(
                repo=None,
                hf=None,
                model_id=None,
                file=None,
                ctx_override=None,
                auto_ctx=True,
                catalog=catalog_path,
                config=config_path,
                llama_server=root / "llama-server",
                start_port=18080,
                public_host="127.0.0.1",
                public_port=11435,
                idle_ttl=10,
                server_config=root / SERVER_CONFIG_BASENAME,
                models_dir=models_dir,
            )

            response = mock.Mock(status_code=200)
            response.json.return_value = {"data": []}

            with (
                mock.patch("llamacpp_stack.cli.temporarily_unload_published_models"),
                mock.patch(
                    "llamacpp_stack.cli.choose_auto_ctx",
                    side_effect=[
                        (None, "min-failed", {"min_ctx": 8192, "reason": "timeout"}),
                        (16384, "selected", {"selected_ctx": 16384}),
                    ],
                ),
                mock.patch("llamacpp_stack.cli.resolve_llama_server_defaults", return_value={}),
                mock.patch("llamacpp_stack.cli._ask_confirmation", return_value=True) as ask_mock,
                mock.patch("llamacpp_stack.cli.time.sleep"),
                mock.patch("llamacpp_stack.cli.requests.get", return_value=response),
            ):
                result = update_config(args)

            self.assertEqual(result, "updated")
            ask_mock.assert_called_once()

            remaining = load_catalog(catalog_path)
            self.assertEqual([m.model_id for m in remaining], ["ok-model"])
            self.assertFalse(fail_repo_dir.exists())
            self.assertTrue(ok_repo_dir.exists())

    def test_update_config_auto_ctx_refreshes_load_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            repo_dir = models_dir / "org/repo"
            repo_dir.mkdir(parents=True)
            model_path = repo_dir / "ok.gguf"
            model_path.write_bytes(b"gguf")

            catalog_path = root / "catalog.json"
            config_path = root / "config.yaml"
            save_catalog(
                catalog_path,
                [
                    ManagedModel(
                        model_id="ok-model",
                        repo_id="org/repo",
                        quant="Q4_K_M",
                        filename="ok.gguf",
                        local_path=str(model_path),
                        ctx_size=4096,
                    ),
                ],
            )

            args = Namespace(
                repo=None,
                hf=None,
                model_id=None,
                file=None,
                ctx_override=None,
                auto_ctx=True,
                catalog=catalog_path,
                config=config_path,
                llama_server=root / "llama-server",
                start_port=18080,
                public_host="127.0.0.1",
                public_port=11435,
                idle_ttl=10,
                server_config=root / SERVER_CONFIG_BASENAME,
                models_dir=models_dir,
            )

            response = mock.Mock(status_code=200)
            response.json.return_value = {"data": []}

            with (
                mock.patch("llamacpp_stack.cli.temporarily_unload_published_models"),
                mock.patch(
                    "llamacpp_stack.cli.choose_auto_ctx",
                    return_value=(16384, "selected", {"selected_ctx": 16384}),
                ),
                mock.patch("llamacpp_stack.cli.resolve_llama_server_defaults", return_value={}),
                mock.patch("llamacpp_stack.cli.refresh_model_load_capabilities") as refresh_mock,
                mock.patch("llamacpp_stack.cli.time.sleep"),
                mock.patch("llamacpp_stack.cli.requests.get", return_value=response),
            ):
                result = update_config(args)

            self.assertEqual(result, "updated")
            refresh_mock.assert_called()

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


    def test_catalog_auto_update_defers_while_model_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "catalog.json"
            server_config_path = root / "conf.json"
            catalog_path.write_text("[]\n", encoding="utf-8")
            server_config_path.write_text("{}\n", encoding="utf-8")
            args = Namespace(catalog=catalog_path, server_config=server_config_path)
            stop_event = threading.Event()
            active = {"value": True}
            calls = []

            def fake_active_summary(*_args, **_kwargs):
                return "qwen(request_start)" if active["value"] else ""

            def fake_update_config(update_args):
                calls.append(update_args)

            with (
                mock.patch("llamacpp_stack.cli._active_download_blocker_summary", side_effect=fake_active_summary),
                mock.patch("llamacpp_stack.cli.update_config", side_effect=fake_update_config),
            ):
                thread = start_catalog_auto_update_watch(args, poll_s=0.05, debounce_s=0.05, stop_event=stop_event)
                time.sleep(0.1)
                catalog_path.write_text("[]\n ", encoding="utf-8")
                time.sleep(0.25)
                self.assertEqual(calls, [])
                active["value"] = False
                time.sleep(0.25)
                stop_event.set()
                thread.join(timeout=1)

            self.assertEqual(len(calls), 1)

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
                        ctx_size=65536,
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
                mock.patch("llamacpp_stack.cli.get_gpu_conflict_message", return_value=None),
                mock.patch("llamacpp_stack.cli.apply_config_and_wait", return_value=True),
            ):
                result = ensure_model_available(args)

            self.assertEqual(result, "repo-q4")
            model = load_catalog(catalog_path)[0]
            self.assertEqual(model.ttl, 300)
            self.assertEqual(model.ctx_size, 65536)
            self.assertEqual(model.n_gpu_layers, 99)
            self.assertEqual(model.tensor_split, "3,1")
            self.assertEqual(model.host, "0.0.0.0")
            self.assertFalse(model.jinja)
            self.assertEqual(model.description, "new")

    def test_existing_model_default_ctx_emits_update_hint(self) -> None:
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
                        ctx_size=8192,
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
                skip_ctx=False,
                ctx_size=8192,
                config=config_path,
                llama_server=root / "llama-server",
                start_port=18080,
                public_host="127.0.0.1",
                public_port=11435,
                n_gpu_layers=99,
                tensor_split="1",
                host="0.0.0.0",
                idle_ttl=10,
                no_jinja=False,
                hf_token=None,
                description="existing",
                service="llamaswap",
            )

            with (
                mock.patch("llamacpp_stack.cli.wait_for_model", return_value=True),
                mock.patch("llamacpp_stack.cli._emit_message") as emit_mock,
            ):
                result = ensure_model_available(args)

            self.assertEqual(result, "repo-q4")
            rendered = "\n".join(str(call.args[0]) for call in emit_mock.call_args_list if call.args)
            self.assertIn(f"{CLI_COMMAND} update --auto --model-id repo-q4", rendered)

    def test_ensure_model_available_downloads_mtp_drafter_and_enables_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            repo_dir = models_dir / "unsloth/gemma-4-31B-it-qat-GGUF"
            repo_dir.mkdir(parents=True)
            model_path = repo_dir / "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf"
            model_path.write_bytes(b"target")
            catalog_path = root / "catalog.json"
            config_path = root / "config.yaml"
            save_catalog(
                catalog_path,
                [
                    ManagedModel(
                        model_id="gemma-4-31b-it-qat-ud-q4_k_xl",
                        repo_id="unsloth/gemma-4-31B-it-qat-GGUF",
                        quant="UD-Q4_K_XL",
                        filename="gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
                        local_path=str(model_path),
                        ctx_size=8192,
                        server_overrides={"parallel": 1},
                    )
                ],
            )
            args = Namespace(
                repo=None,
                hf="unsloth/gemma-4-31B-it-qat-GGUF:UD-Q4_K_XL",
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
                tensor_split="1",
                host="127.0.0.1",
                idle_ttl=10,
                no_jinja=False,
                hf_token=None,
                description="existing",
                service="llamaswap",
            )

            api = mock.MagicMock()
            target_sibling = mock.Mock()
            target_sibling.rfilename = "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf"
            target_sibling.size = len(b"target")
            mtp_sibling = mock.Mock()
            mtp_sibling.rfilename = "mtp-gemma-4-31B-it.gguf"
            mtp_sibling.size = len(b"draft")
            api.model_info.return_value.siblings = [target_sibling, mtp_sibling]

            def fake_download(**kwargs):
                filename = kwargs["filename"]
                destination = repo_dir / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"draft")
                return str(destination)

            with (
                mock.patch("llamacpp_stack.cli.HfApi", return_value=api),
                mock.patch("llamacpp_stack.cli.get_llama_server_processes", return_value=[]),
                mock.patch("llamacpp_stack.cli.download_hf_file", side_effect=fake_download) as download_mock,
                mock.patch("llamacpp_stack.cli.get_gpu_conflict_message", return_value=None),
                mock.patch("llamacpp_stack.cli.apply_config_and_wait", return_value=True),
            ):
                result = ensure_model_available(args)

            self.assertEqual(result, "gemma-4-31b-it-qat-ud-q4_k_xl")
            downloaded = [call.kwargs["filename"] for call in download_mock.call_args_list]
            self.assertEqual(downloaded, ["mtp-gemma-4-31B-it.gguf"])
            model = load_catalog(catalog_path)[0]
            self.assertEqual(model.server_overrides["model_draft"], str(repo_dir / "mtp-gemma-4-31B-it.gguf"))
            self.assertEqual(model.server_overrides["spec_type"], "draft-mtp")
            self.assertEqual(model.server_overrides["spec_draft_n_max"], 3)
            self.assertEqual(model.server_overrides["parallel"], 1)


    def test_integrated_mtp_model_gets_draft_mtp_without_model_draft(self) -> None:
        model = ManagedModel(
            model_id="qwen-agentworld-35b-a3b-ud-q4_k_xl-mtp",
            repo_id="vadikulous-ai/Qwen-AgentWorld-35B-A3B-UD-Q4_K_XL-MTP-GGUF",
            quant=None,
            filename="Qwen-AgentWorld-35B-A3B-UD-Q4_K_XL-MTP.gguf",
            local_path="/models/Qwen-AgentWorld-35B-A3B-UD-Q4_K_XL-MTP.gguf",
            ctx_size=262144,
        )
        model.server_overrides, changed = cli._apply_mtp_server_overrides(
            model.server_overrides,
            None,
            {"mtp_defaults": {"image_min_tokens": 1024, "spec_draft_n_max": 3, "spec_draft_n_min": 0, "spec_draft_p_min": 0.75}},
        )

        self.assertTrue(changed)
        self.assertNotIn("model_draft", model.server_overrides)
        cmd = build_llama_server_command(model, Path("/bin/llama-server"), port="12345", server_defaults={})
        joined = " ".join(cmd)
        self.assertIn("--image-min-tokens 1024", joined)
        self.assertIn("--spec-type draft-mtp", joined)
        self.assertIn("--spec-draft-n-max 3", joined)
        self.assertIn("--spec-draft-p-min 0.75", joined)
        self.assertNotIn("--model-draft", joined)

    def test_mtp_model_draft_does_not_emit_duplicate_draft_max(self) -> None:
        model = ManagedModel(
            model_id="speculative-qwen",
            repo_id="org/qwen",
            quant=None,
            filename="qwen.gguf",
            local_path="/models/qwen.gguf",
            ctx_size=262144,
            server_overrides={
                "model_draft": "/models/draft.gguf",
                "spec_type": "draft-mtp",
                "spec_draft_n_max": 3,
                "draft": 16,
            },
        )

        cmd = build_llama_server_command(model, Path("/bin/llama-server"), port="12345", server_defaults={})
        joined = " ".join(cmd)
        self.assertIn("--model-draft /models/draft.gguf", joined)
        self.assertEqual(joined.count("--spec-draft-n-max"), 1)
        self.assertIn("--spec-draft-n-max 3", joined)
        self.assertNotIn("--spec-draft-n-max 16", joined)
        self.assertNotIn("--draft-max 16", joined)

    def test_update_backfills_mtp_drafter_for_existing_catalog_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models" / "org" / "Model-MTP-GGUF"
            model_dir.mkdir(parents=True)
            main_model = model_dir / "model-Q4_K_M.gguf"
            main_model.write_bytes(b"GGUF")
            model = ManagedModel(
                model_id="model-mtp-q4",
                repo_id="org/Model-MTP-GGUF",
                quant="Q4_K_M",
                filename="model-Q4_K_M.gguf",
                local_path=str(main_model),
                server_overrides={},
            )

            class Sibling:
                def __init__(self, name: str):
                    self.rfilename = name
                    self.size = 123

            class Api:
                def model_info(self, repo_id, token=None):
                    return Namespace(siblings=[Sibling("model-Q4_K_M.gguf"), Sibling("mtp-model.gguf")])

            def fake_download(**kwargs):
                destination = kwargs["target_dir"] / kwargs["filename"]
                destination.write_bytes(b"draft")
                return str(destination)

            args = Namespace(hf_token=None, models_dir=root / "models")
            with (
                mock.patch("llamacpp_stack.cli.HfApi", return_value=Api()),
                mock.patch("llamacpp_stack.cli.download_hf_file", side_effect=fake_download) as download_mock,
            ):
                changed = ensure_catalog_mtp_drafters(
                    [model],
                    args,
                    {"mtp_defaults": {"spec_draft_n_max": 3, "spec_draft_n_min": 0, "spec_draft_p_min": 0.75}},
                )

            self.assertEqual(changed, 1)
            self.assertEqual(download_mock.call_args.kwargs["filename"], "mtp-model.gguf")
            self.assertEqual(model.server_overrides["model_draft"], str(model_dir / "mtp-model.gguf"))
            self.assertEqual(model.server_overrides["spec_type"], "draft-mtp")
            self.assertEqual(model.server_overrides["spec_draft_n_max"], 3)

    def test_update_migrates_existing_mtp_override_without_mtp_repo_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models" / "org" / "Gemma-GGUF"
            model_dir.mkdir(parents=True)
            main_model = model_dir / "model-Q4_K_M.gguf"
            draft_model = model_dir / "mtp-model.gguf"
            main_model.write_bytes(b"GGUF")
            draft_model.write_bytes(b"draft")
            model = ManagedModel(
                model_id="gemma-q4",
                repo_id="org/Gemma-GGUF",
                quant="Q4_K_M",
                filename="model-Q4_K_M.gguf",
                local_path=str(main_model),
                server_overrides={
                    "model_draft": str(draft_model),
                    "spec_type": "draft-mtp",
                    "spec_draft_n_max": 4,
                },
            )

            args = Namespace(hf_token=None, models_dir=root / "models")
            changed = ensure_catalog_mtp_drafters(
                [model],
                args,
                {"mtp_defaults": {"spec_draft_n_max": 3, "spec_draft_n_min": 0, "spec_draft_p_min": 0.75}},
            )

            self.assertEqual(changed, 1)
            self.assertEqual(model.server_overrides["spec_draft_n_max"], 3)

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
            rendered_text = config_path.read_text(encoding="utf-8")
            self.assertIn("ttl: 10", rendered_text)
            self.assertIn("sendLoadingState: false", rendered_text)
            self.assertTrue(rendered_text.startswith("# llamacpp-superserver config.yaml"))


    def test_global_replica_enabled_ignores_stale_per_model_disabled_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.gguf"
            model_path.write_bytes(b"x" * 1024)
            model = ManagedModel(
                "large",
                "org/large",
                "Q4",
                "large.gguf",
                str(model_path),
                tensor_split="1,1",
                server_overrides={"replicas": {"enabled": False}},
            )
            with mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=4):
                entries = iter_catalog_with_replicas(
                    [model],
                    {"enabled": True, "max": "auto", "placement": "exclusive_gpus"},
                )
            replica_ids = [entry[0].model_id for entry in entries if entry[1] == "large"]
            self.assertEqual(replica_ids, ["large__replica_0"])

    def test_global_replica_config_generates_model_replicas_automatically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.gguf"
            model_path.write_bytes(b"x" * 1024)
            model = ManagedModel(
                model_id="repo-q4",
                repo_id="org/repo",
                quant="Q4",
                filename="model.gguf",
                local_path=str(model_path),
                tensor_split="1,1,1",
            )
            global_replicas = {"enabled": True, "max": "auto", "placement": "exclusive_gpus"}
            with mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=7):
                entries = iter_catalog_with_replicas([model], global_replicas)
            self.assertEqual([entry[0].model_id for entry in entries], ["repo-q4", "repo-q4__replica_0"])
            self.assertEqual(entries[1][2], [3, 4, 5])

    def test_render_config_uses_global_replica_defaults_does_not_precreate_replicas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.gguf"
            model_path.write_bytes(b"x" * 1024)
            config_path = root / "config.yaml"
            model = ManagedModel(
                model_id="repo-q4",
                repo_id="org/repo",
                quant="Q4",
                filename="model.gguf",
                local_path=str(model_path),
                tensor_split="1,1,1",
            )
            with mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=7):
                render_llamaswap_config(
                    [model],
                    config_path,
                    root / "llama-server",
                    18080,
                    idle_ttl=10,
                    replica_defaults={"enabled": True, "max": "auto", "placement": "exclusive_gpus"},
                )
            rendered = config_path.read_text(encoding="utf-8")
            self.assertNotIn("repo-q4__replica_0", rendered)
            self.assertNotIn("repo-q4__replica_1", rendered)

            ensure_replica_route_in_llamaswap_config(
                model,
                0,
                [0, 1, 2],
                [model],
                config_path,
                root / "llama-server",
                10,
            )
            rendered = config_path.read_text(encoding="utf-8")
            self.assertIn("repo-q4__replica_0", rendered)
            self.assertNotIn("repo-q4__replica_1", rendered)
            self.assertIn("CUDA_VISIBLE_DEVICES=0,1,2", rendered)

    def test_dynamic_replica_route_uses_cuda_visible_devices_and_matrix_vars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.gguf"
            model_path.write_bytes(b"x" * 1024)
            config_path = root / "config.yaml"
            model = ManagedModel(
                model_id="repo-q4",
                repo_id="org/repo",
                quant="Q4",
                filename="model.gguf",
                local_path=str(model_path),
                tensor_split="1,1",
                server_overrides={
                    "replicas": {"enabled": True, "max": 2, "gpus_per_replica": 2},
                },
                aliases=["public-alias"],
            )
            with mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=4):
                render_llamaswap_config([model], config_path, root / "llama-server", 18080, idle_ttl=10)

            rendered = yaml.safe_load("\n".join(line for line in config_path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")))
            models = rendered["models"]
            self.assertIn("repo-q4", models)
            self.assertNotIn("repo-q4__replica_0", models)
            self.assertNotIn("repo-q4__replica_1", models)
            self.assertEqual(models["repo-q4"]["aliases"], ["public-alias"])

            ensure_replica_route_in_llamaswap_config(model, 0, [0, 1], [model], config_path, root / "llama-server", 10)
            ensure_replica_route_in_llamaswap_config(model, 1, [2, 3], [model], config_path, root / "llama-server", 10)

            rendered = yaml.safe_load("\n".join(line for line in config_path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")))
            models = rendered["models"]
            self.assertIn("repo-q4__replica_0", models)
            self.assertIn("repo-q4__replica_1", models)
            self.assertIn("/usr/bin/env CUDA_VISIBLE_DEVICES=0,1", models["repo-q4__replica_0"]["cmd"])
            self.assertIn("/usr/bin/env CUDA_VISIBLE_DEVICES=2,3", models["repo-q4__replica_1"]["cmd"])
            self.assertIn("--tensor-split 1,1", models["repo-q4__replica_0"]["cmd"])
            self.assertIn("--device CUDA0,CUDA1", models["repo-q4__replica_0"]["cmd"])
            self.assertNotIn("--tensor-split 1,1,1,1", models["repo-q4__replica_0"]["cmd"])
            self.assertNotIn("aliases", models["repo-q4__replica_0"])
            matrix = rendered["matrix"]
            vars_map = matrix["vars"]
            replica_vars = {model_id: var_id for var_id, model_id in vars_map.items() if model_id.startswith("repo-q4__replica_")}
            self.assertIn("repo-q4__replica_0", replica_vars)
            self.assertIn("repo-q4__replica_1", replica_vars)
            joined_sets = " & ".join(matrix["sets"].values())
            self.assertIn(replica_vars["repo-q4__replica_0"], joined_sets.split(" & "))
            self.assertIn(replica_vars["repo-q4__replica_1"], joined_sets.split(" & "))
            self.assertNotIn("repo-q4__replica_0", joined_sets)
            self.assertIn(replica_vars["repo-q4__replica_0"], matrix["evict_costs"])

    def test_dynamic_replica_tensor_split_is_local_to_visible_devices_on_many_gpu_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.gguf"
            model_path.write_bytes(b"x" * 1024)
            config_path = root / "config.yaml"
            model = ManagedModel(
                model_id="seven-gpu-model",
                repo_id="org/repo",
                quant="Q4",
                filename="model.gguf",
                local_path=str(model_path),
                tensor_split="1,1,1,1,1,1,1",
                server_overrides={"replicas": {"enabled": True, "max": 1, "gpus_per_replica": 2}},
            )
            with mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=7):
                render_llamaswap_config([model], config_path, root / "llama-server", 18080, idle_ttl=10)
            rendered = yaml.safe_load("\n".join(line for line in config_path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")))
            self.assertNotIn("seven-gpu-model__replica_0", rendered["models"])
            ensure_replica_route_in_llamaswap_config(model, 0, [0, 1], [model], config_path, root / "llama-server", 10)
            rendered = yaml.safe_load("\n".join(line for line in config_path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")))
            cmd = rendered["models"]["seven-gpu-model__replica_0"]["cmd"]
            self.assertIn("/usr/bin/env CUDA_VISIBLE_DEVICES=0,1", cmd)
            self.assertIn("--tensor-split 1,1", cmd)
            self.assertIn("--device CUDA0,CUDA1", cmd)
            self.assertNotIn("--tensor-split 1,1,1,1,1,1,1", cmd)
            self.assertIn("--main-gpu 0", cmd)


    def test_base_model_with_short_tensor_split_emits_device_limit(self) -> None:
        model = ManagedModel(
            model_id="two-of-seven",
            repo_id="org/repo",
            quant="Q4",
            filename="model.gguf",
            local_path="/tmp/model.gguf",
            tensor_split="1,1",
        )
        with mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=7):
            cmd = build_llama_server_command(model, Path("/opt/llama-server"), port="18080")
        rendered = " ".join(cmd)
        self.assertIn("/usr/bin/env CUDA_VISIBLE_DEVICES=0,1", rendered)
        self.assertIn("--tensor-split 1,1", rendered)
        self.assertIn("--device CUDA0,CUDA1", rendered)

    def test_tensor_split_is_enforced_even_when_detected_gpu_count_matches_parts(self) -> None:
        model = ManagedModel(
            model_id="three-gpu-model",
            repo_id="org/repo",
            quant="Q4",
            filename="model.gguf",
            local_path="/tmp/model.gguf",
            tensor_split="1,1,1",
        )
        with mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=3):
            cmd = build_llama_server_command(model, Path("/opt/llama-server"), port="18080")
        rendered = " ".join(cmd)
        self.assertIn("/usr/bin/env CUDA_VISIBLE_DEVICES=0,1,2", rendered)
        self.assertIn("--tensor-split 1,1,1", rendered)
        self.assertIn("--device CUDA0,CUDA1,CUDA2", rendered)

    def test_server_defaults_tensor_split_does_not_override_model_placement(self) -> None:
        model = ManagedModel(
            model_id="two-of-seven",
            repo_id="org/repo",
            quant="Q4",
            filename="model.gguf",
            local_path="/tmp/model.gguf",
            tensor_split="1,1",
        )
        with mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=7):
            cmd = build_llama_server_command(
                model,
                Path("/opt/llama-server"),
                port="18080",
                server_defaults={"tensor_split": "1,1,1,1,1,1,1"},
            )
        rendered = " ".join(cmd)
        self.assertIn("/usr/bin/env CUDA_VISIBLE_DEVICES=0,1", rendered)
        self.assertIn("--tensor-split 1,1", rendered)
        self.assertIn("--device CUDA0,CUDA1", rendered)
        self.assertNotIn("--tensor-split 1,1,1,1,1,1,1", rendered)

    def test_affinity_key_uses_codex_thread_and_opencode_agent(self) -> None:
        payload = {"model": "repo-q4", "messages": [{"role": "user", "content": "hello"}]}
        codex_key = resolve_request_affinity_key("repo-q4", payload, {"thread-id": "thread-a", "session-id": "session-a"})
        agent_a = resolve_request_affinity_key("repo-q4", payload, {"Helicone-Session-Id": "sess", "x-agent-id": "agent-a"})
        agent_b = resolve_request_affinity_key("repo-q4", payload, {"Helicone-Session-Id": "sess", "x-agent-id": "agent-b"})
        self.assertIn("thread-a", codex_key)
        self.assertNotEqual(agent_a, agent_b)


    def test_select_replica_enabled_with_missing_route_falls_back_to_base(self) -> None:
        REPLICA_ROUTER_STATE.records.clear()
        REPLICA_ROUTER_STATE.affinity.clear()
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model.gguf",
            local_path="/tmp/model.gguf",
            tensor_split="1",
            server_overrides={"replicas": {"enabled": True, "max": 1, "gpus_per_replica": 1}},
        )
        with (
            mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=1),
            mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value=set()),
            mock.patch("llamacpp_stack.cli._query_gpu_memory_snapshot_cached", return_value={}),
        ):
            selected, _, is_replica = select_replica_for_request(model, {"messages": [{"role": "user", "content": "a"}]}, {"thread-id": "t"})
        self.assertFalse(is_replica)
        self.assertEqual(selected, "repo-q4")

    def test_select_replica_falls_back_to_base_when_enabled_routes_blacklisted(self) -> None:
        REPLICA_ROUTER_STATE.records.clear()
        REPLICA_ROUTER_STATE.affinity.clear()
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model.gguf",
            local_path="/tmp/model.gguf",
            tensor_split="1",
            server_overrides={"replicas": {"enabled": True, "max": 1, "gpus_per_replica": 1}},
        )
        with (
            mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=2),
            mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value={"repo-q4__replica_0"}),
            mock.patch("llamacpp_stack.cli._query_gpu_memory_snapshot_cached", return_value={}),
        ):
            select_replica_for_request(model, {"messages": [{"role": "user", "content": "prime"}]}, {"thread-id": "prime"})
            REPLICA_ROUTER_STATE.records["repo-q4__replica_0"].blacklist_until = time.monotonic() + 60
            selected, _, is_replica = select_replica_for_request(model, {"messages": [{"role": "user", "content": "next"}]}, {"thread-id": "next"})
        self.assertFalse(is_replica)
        self.assertEqual(selected, "repo-q4")

    def test_exclusive_replica_gpu_sets_returns_empty_when_base_uses_all_gpus(self) -> None:
        model = ManagedModel(
            model_id="all-gpus",
            repo_id="org/all",
            quant="Q4",
            filename="all.gguf",
            local_path="/tmp/all.gguf",
            tensor_split="1,1,1,1,1,1,1",
        )
        cfg = get_model_replica_config(
            model,
            {"enabled": True, "max": "auto", "gpus_per_replica": 7, "placement": "exclusive_gpus"},
            total_gpus=7,
        )
        self.assertEqual(_replica_gpu_sets(model, cfg, total_gpus=7), [])

    def test_exclusive_replica_gpu_sets_skip_base_visible_gpus(self) -> None:
        model = ManagedModel(
            model_id="large",
            repo_id="org/large",
            quant="Q4",
            filename="large.gguf",
            local_path="/tmp/large.gguf",
            tensor_split="1,1,1",
        )
        cfg = get_model_replica_config(
            model,
            {"enabled": True, "max": 2, "gpus_per_replica": 3, "placement": "exclusive_gpus"},
        )
        with mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=7):
            self.assertEqual(_replica_gpu_sets(model, cfg), [[3, 4, 5]])

    def test_select_replica_uses_base_when_base_is_idle(self) -> None:
        REPLICA_ROUTER_STATE.records.clear()
        REPLICA_ROUTER_STATE.affinity.clear()
        REPLICA_ROUTER_STATE.base_in_flight.clear()
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model.gguf",
            local_path="/tmp/model.gguf",
            tensor_split="1",
            server_overrides={"replicas": {"enabled": True, "max": 2, "gpus_per_replica": 1}},
        )
        with mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=2):
            selected, _, is_replica = select_replica_for_request(
                model,
                {"messages": [{"role": "user", "content": "first"}]},
                {"thread-id": "first"},
                replica_defaults={"enabled": True, "max": 2, "gpus_per_replica": 1},
                published_model_ids=set(),
            )
        self.assertFalse(is_replica)
        self.assertEqual(selected, "repo-q4")

    def test_select_replica_keeps_same_conversation_on_base_even_when_base_busy(self) -> None:
        REPLICA_ROUTER_STATE.records.clear()
        REPLICA_ROUTER_STATE.affinity.clear()
        REPLICA_ROUTER_STATE.base_in_flight.clear()
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model.gguf",
            local_path="/tmp/model.gguf",
            tensor_split="1",
            server_overrides={"replicas": {"enabled": True, "max": 2, "gpus_per_replica": 1}},
        )
        payload = {"messages": [{"role": "user", "content": "same conversation"}]}
        headers = {"thread-id": "thread-a"}
        with mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=2):
            first, affinity_key, first_is_replica = select_replica_for_request(
                model,
                payload,
                headers,
                replica_defaults={"enabled": True, "max": 2, "gpus_per_replica": 1},
                published_model_ids=set(),
            )
            REPLICA_ROUTER_STATE.request_started(first)
            second, second_affinity_key, second_is_replica = select_replica_for_request(
                model,
                {"messages": [{"role": "user", "content": "follow up"}]},
                headers,
                replica_defaults={"enabled": True, "max": 2, "gpus_per_replica": 1},
                published_model_ids={"repo-q4__replica_0"},
            )
        self.assertEqual(affinity_key, second_affinity_key)
        self.assertFalse(first_is_replica)
        self.assertFalse(second_is_replica)
        self.assertEqual(first, "repo-q4")
        self.assertEqual(second, "repo-q4")

    def test_select_replica_creates_route_on_demand(self) -> None:
        REPLICA_ROUTER_STATE.records.clear()
        REPLICA_ROUTER_STATE.affinity.clear()
        REPLICA_ROUTER_STATE.base_in_flight.clear()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.gguf"
            model_path.write_bytes(b"x" * 1024)
            config_path = root / "config.yaml"
            model = ManagedModel(
                model_id="repo-q4",
                repo_id="org/repo",
                quant="Q4",
                filename="model.gguf",
                local_path=str(model_path),
                tensor_split="1",
                server_overrides={"replicas": {"enabled": True, "max": 2, "gpus_per_replica": 1, "safety_vram_mib": 1}},
            )
            render_llamaswap_config([model], config_path, root / "llama-server", 18080, idle_ttl=10)
            self.assertNotIn("repo-q4__replica_0", config_path.read_text(encoding="utf-8"))
            REPLICA_ROUTER_STATE.base_in_flight["repo-q4"] = 1
            with (
                mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=2),
                mock.patch("llamacpp_stack.cli._query_gpu_memory_snapshot_cached", return_value={0: {"free_mib": 10000, "total_mib": 10000}, 1: {"free_mib": 10000, "total_mib": 10000}}),
                mock.patch("llamacpp_stack.cli.wait_for_published_model_id", return_value=True),
            ):
                selected, _, is_replica = select_replica_for_request(
                    model,
                    {"messages": [{"role": "user", "content": "a"}]},
                    {"thread-id": "t"},
                    replica_defaults={"enabled": True, "max": 2, "gpus_per_replica": 1, "safety_vram_mib": 1},
                    published_model_ids=set(),
                    catalog=[model],
                    config_path=config_path,
                    server_path=root / "llama-server",
                    idle_ttl=10,
                )
            self.assertTrue(is_replica)
            self.assertEqual(selected, "repo-q4__replica_0")
            rendered = config_path.read_text(encoding="utf-8")
            self.assertIn("repo-q4__replica_0", rendered)
            self.assertNotIn("repo-q4__replica_1", rendered)

    def test_select_replica_uses_ready_empty_then_sticky(self) -> None:
        REPLICA_ROUTER_STATE.records.clear()
        REPLICA_ROUTER_STATE.affinity.clear()
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model.gguf",
            local_path="/tmp/model.gguf",
            tensor_split="1",
            server_overrides={"replicas": {"enabled": True, "max": 2, "gpus_per_replica": 1}},
        )
        REPLICA_ROUTER_STATE.base_in_flight["repo-q4"] = 1
        with (
            mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=2),
            mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value={"repo-q4__replica_0"}),
        ):
            selected, affinity, is_replica = select_replica_for_request(model, {"messages": [{"role": "user", "content": "a"}]}, {"thread-id": "t1"})
            selected_again, _, _ = select_replica_for_request(model, {"messages": [{"role": "user", "content": "b"}]}, {"thread-id": "t1"})
        self.assertTrue(is_replica)
        self.assertEqual(selected, "repo-q4__replica_0")
        self.assertEqual(selected_again, selected)
        self.assertIn("t1", affinity)


    def test_select_replica_scales_new_affinity_to_cold_before_ready_empty_when_fits(self) -> None:
        REPLICA_ROUTER_STATE.records.clear()
        REPLICA_ROUTER_STATE.affinity.clear()
        with tempfile.TemporaryDirectory() as tmp:
            REPLICA_ROUTER_STATE.base_in_flight.clear()
            model_path = Path(tmp) / "small.gguf"
            model_path.write_bytes(b"x" * 1024 * 1024)
            model = ManagedModel(
                model_id="small",
                repo_id="org/small",
                quant="Q4",
                filename="small.gguf",
                local_path=str(model_path),
                tensor_split="1",
                server_overrides={"replicas": {"enabled": True, "max": 2, "gpus_per_replica": 1, "safety_vram_mib": 1}},
            )
            REPLICA_ROUTER_STATE.base_in_flight["small"] = 1
            with (
                mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=3),
                mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value={"small__replica_0", "small__replica_1"}),
                mock.patch("llamacpp_stack.cli._query_gpu_memory_snapshot_cached", return_value={0: {"free_mib": 10000, "total_mib": 10000}, 1: {"free_mib": 10000, "total_mib": 10000}, 2: {"free_mib": 10000, "total_mib": 10000}}),
            ):
                selected0, _, _ = select_replica_for_request(model, {"messages": [{"role": "user", "content": "prime"}]}, {"thread-id": "prime"})
                self.assertEqual(selected0, "small__replica_0")
                REPLICA_ROUTER_STATE.records["small__replica_0"].status = "ready"
                selected, _, is_replica = select_replica_for_request(model, {"messages": [{"role": "user", "content": "new conversation"}]}, {"thread-id": "new"})
        self.assertTrue(is_replica)
        self.assertEqual(selected, "small__replica_1")

    def test_select_replica_uses_previous_response_id_mapping(self) -> None:
        REPLICA_ROUTER_STATE.records.clear()
        REPLICA_ROUTER_STATE.affinity.clear()
        REPLICA_ROUTER_STATE.response_to_replica.clear()
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model.gguf",
            local_path="/tmp/model.gguf",
            tensor_split="1",
            server_overrides={"replicas": {"enabled": True, "max": 2, "gpus_per_replica": 1}},
        )
        REPLICA_ROUTER_STATE.bind_response("resp_123", "repo-q4__replica_1")
        with (
            mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=3),
            mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value={"repo-q4__replica_0", "repo-q4__replica_1"}),
        ):
            selected, _, is_replica = select_replica_for_request(model, {"previous_response_id": "resp_123", "messages": [{"role": "user", "content": "continue"}]}, {})
        self.assertTrue(is_replica)
        self.assertEqual(selected, "repo-q4__replica_1")

    def test_sync_replica_runtime_state_maps_ports_to_actual_gpu_memory(self) -> None:
        REPLICA_ROUTER_STATE.records.clear()
        REPLICA_ROUTER_STATE.affinity.clear()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text(
                "startPort: 18080\n"
                "models:\n"
                "  base:\n"
                "    cmd: base\n"
                "  base__replica_0:\n"
                "    cmd: r0\n"
                "  base__replica_1:\n"
                "    cmd: r1\n",
                encoding="utf-8",
            )
            model = ManagedModel(
                model_id="base",
                repo_id="org/base",
                quant="Q4",
                filename="base.gguf",
                local_path=str(root / "base.gguf"),
                tensor_split="1,1,1",
                server_overrides={"replicas": {"enabled": True, "max": 2, "gpus_per_replica": 3}},
            )
            with (
                mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=7),
                mock.patch("llamacpp_stack.cli.get_llama_server_processes", return_value=[{"pid": 1234, "port": 18081, "model_path": str(root / "base.gguf"), "cmdline": "llama-server --port 18081"}]),
                mock.patch("llamacpp_stack.cli.get_gpu_process_memory_by_pid", return_value={1234: {3: 11000.0, 4: 11200.0, 5: 10900.0}}),
            ):
                sync_replica_runtime_state([model], config_path)

            rec0 = REPLICA_ROUTER_STATE.records["base__replica_0"]
            self.assertNotIn("base__replica_1", REPLICA_ROUTER_STATE.records)
            self.assertEqual(rec0.pid, 1234)
            self.assertEqual(rec0.port, 18081)
            self.assertEqual(rec0.status, "ready")
            self.assertEqual(rec0.gpu_actual_mib, {3: 11000.0, 4: 11200.0, 5: 10900.0})
            self.assertEqual(rec0.actual_mib, 33100.0)

    def test_sync_replica_runtime_state_matches_dynamic_port_by_gpu_set(self) -> None:
        REPLICA_ROUTER_STATE.records.clear()
        REPLICA_ROUTER_STATE.affinity.clear()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text(
                "startPort: 18080\n"
                "models:\n"
                "  base:\n"
                "    cmd: base\n"
                "  filler_a:\n"
                "    cmd: filler\n"
                "  filler_b:\n"
                "    cmd: filler\n"
                "  base__replica_0:\n"
                "    cmd: r0\n",
                encoding="utf-8",
            )
            model = ManagedModel(
                model_id="base",
                repo_id="org/base",
                quant="Q4",
                filename="base.gguf",
                local_path=str(root / "base.gguf"),
                tensor_split="1,1,1",
                server_overrides={"replicas": {"enabled": True, "max": 1, "gpus_per_replica": 3}},
            )
            with (
                mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=7),
                mock.patch("llamacpp_stack.cli.get_llama_server_processes", return_value=[{"pid": 4321, "port": 18081, "model_path": str(root / "base.gguf"), "cmdline": "llama-server --port 18081"}]),
                mock.patch("llamacpp_stack.cli.get_gpu_process_memory_by_pid", return_value={4321: {3: 11000.0, 4: 11200.0, 5: 10900.0}}),
            ):
                sync_replica_runtime_state([model], config_path)

            rec0 = REPLICA_ROUTER_STATE.records["base__replica_0"]
            self.assertEqual(rec0.pid, 4321)
            self.assertEqual(rec0.port, 18081)
            self.assertEqual(rec0.status, "ready")
            self.assertEqual(rec0.gpu_actual_mib, {3: 11000.0, 4: 11200.0, 5: 10900.0})

    def test_multi_gpu_replica_placement_uses_per_gpu_memory_estimate(self) -> None:
        REPLICA_ROUTER_STATE.records.clear()
        REPLICA_ROUTER_STATE.affinity.clear()
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "large.gguf"
            # 21 GiB-like model: total process estimate is larger than a single
            # 24 GiB GPU with safety, but a 3-way tensor split should fit.
            with model_path.open("wb") as f:
                f.truncate(21 * 1024 * 1024 * 1024)
            model = ManagedModel(
                model_id="large",
                repo_id="org/large",
                quant="Q4",
                filename="large.gguf",
                local_path=str(model_path),
                tensor_split="1,1,1",
                server_overrides={"replicas": {"enabled": True, "max": 2, "gpus_per_replica": 2, "safety_vram_mib": 2048}},
            )
            REPLICA_ROUTER_STATE.base_in_flight["large"] = 1
            with (
                mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=7),
                mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value={"large__replica_0", "large__replica_1"}),
                mock.patch("llamacpp_stack.cli._query_gpu_memory_snapshot_cached", return_value={i: {"free_mib": 24000, "total_mib": 24576} for i in range(7)}),
            ):
                selected0, _, _ = select_replica_for_request(model, {"messages": [{"role": "user", "content": "prime"}]}, {"thread-id": "prime"})
                self.assertEqual(selected0, "large__replica_0")
                REPLICA_ROUTER_STATE.records["large__replica_0"].status = "ready"
                selected, _, is_replica = select_replica_for_request(model, {"messages": [{"role": "user", "content": "new"}]}, {"thread-id": "new"})
        self.assertTrue(is_replica)
        self.assertEqual(selected, "large__replica_1")

    def test_select_replica_starts_cold_when_capacity_fits_else_least_loaded(self) -> None:
        REPLICA_ROUTER_STATE.records.clear()
        REPLICA_ROUTER_STATE.affinity.clear()
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "small.gguf"
            model_path.write_bytes(b"x" * 1024 * 1024)
            model = ManagedModel(
                model_id="small",
                repo_id="org/small",
                quant="Q4",
                filename="small.gguf",
                local_path=str(model_path),
                tensor_split="1",
                server_overrides={"replicas": {"enabled": True, "max": 2, "gpus_per_replica": 1, "safety_vram_mib": 1}},
            )
            REPLICA_ROUTER_STATE.base_in_flight["small"] = 1
            with (
                mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=3),
                mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value={"small__replica_0", "small__replica_1"}),
                mock.patch("llamacpp_stack.cli._query_gpu_memory_snapshot_cached", return_value={0: {"free_mib": 1000, "total_mib": 1000}, 1: {"free_mib": 10000, "total_mib": 10000}, 2: {"free_mib": 10000, "total_mib": 10000}}),
            ):
                select_replica_for_request(model, {"messages": [{"role": "user", "content": "prime"}]}, {"thread-id": "prime"})
                REPLICA_ROUTER_STATE.records["small__replica_0"].status = "ready"
                REPLICA_ROUTER_STATE.records["small__replica_0"].in_flight = 1
                selected, _, _ = select_replica_for_request(model, {"messages": [{"role": "user", "content": "new"}]}, {"thread-id": "new"})
            self.assertEqual(selected, "small__replica_1")

            REPLICA_ROUTER_STATE.records.clear()
            REPLICA_ROUTER_STATE.affinity.clear()
            with (
                mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=3),
                mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value={"small__replica_0", "small__replica_1"}),
                mock.patch("llamacpp_stack.cli._query_gpu_memory_snapshot_cached", return_value={}),
            ):
                # Prime records as loaded and assign different loads.
                select_replica_for_request(model, {"messages": [{"role": "user", "content": "p"}]}, {"thread-id": "p"})
                REPLICA_ROUTER_STATE.records["small__replica_0"].in_flight = 5
                REPLICA_ROUTER_STATE.records["small__replica_1"].in_flight = 1
                selected2, _, _ = select_replica_for_request(model, {"messages": [{"role": "user", "content": "q"}]}, {"thread-id": "q"})
            self.assertEqual(selected2, "small__replica_1")

    def test_pack_small_models_requires_opt_in(self) -> None:
        REPLICA_ROUTER_STATE.records.clear()
        REPLICA_ROUTER_STATE.affinity.clear()
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "small.gguf"
            model_path.write_bytes(b"x" * 1024 * 1024)
            base = dict(enabled=True, max=2, gpus_per_replica=1, safety_vram_mib=1)
            exclusive = ManagedModel("m", "o/m", "Q4", "small.gguf", str(model_path), tensor_split="1", server_overrides={"replicas": base})
            packed = ManagedModel("p", "o/p", "Q4", "small.gguf", str(model_path), tensor_split="1", server_overrides={"replicas": {**base, "placement": "pack_small_models", "max_pack_fraction": 1.0}})
            REPLICA_ROUTER_STATE.base_in_flight["m"] = 1
            with (
                mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=1),
                mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value={"m__replica_0"}),
            ):
                self.assertEqual("m", select_replica_for_request(exclusive, {"messages": [{"role": "user", "content": "a"}]}, {"thread-id": "a"})[0])
                REPLICA_ROUTER_STATE.records.clear(); REPLICA_ROUTER_STATE.affinity.clear(); REPLICA_ROUTER_STATE.base_in_flight["p"] = 1
                with (
                    mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value={"p__replica_0", "p__replica_1"}),
                    mock.patch("llamacpp_stack.cli._query_gpu_memory_snapshot_cached", return_value={0: {"free_mib": 10000, "total_mib": 10000}}),
                ):
                    select_replica_for_request(packed, {"messages": [{"role": "user", "content": "prime"}]}, {"thread-id": "prime"})
                    REPLICA_ROUTER_STATE.records["p__replica_0"].status = "ready"
                    REPLICA_ROUTER_STATE.records["p__replica_0"].in_flight = 1
                    self.assertEqual(replica_model_id("p", 1), select_replica_for_request(packed, {"messages": [{"role": "user", "content": "b"}]}, {"thread-id": "b"})[0])

    def test_persist_server_config_writes_metadata_header_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server_config = Path(tmp) / SERVER_CONFIG_BASENAME
            args = Namespace(server_config=server_config, idle_ttl=123, api_port=11436)
            persist_server_config(args)
            payload = json.loads(server_config.read_text(encoding="utf-8"))
            self.assertEqual(payload["idle_ttl"], 123)
            self.assertEqual(payload["api_port"], 11436)
            self.assertIn("_meta", payload)
            self.assertIn("purpose", payload["_meta"])

    def test_load_catalog_normalizes_server_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            server_config = Path(tmp) / SERVER_CONFIG_BASENAME
            server_config.write_text('{"llama_server_defaults":{}}\n', encoding="utf-8")
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
            "reasoning-format": "none",
      "batch_size": "1024"
    }
  }
]
""".strip()
                + "\n",
                encoding="utf-8",
            )
            with mock.patch("llamacpp_stack.cli.DEFAULT_SERVER_CONFIG_PATH", server_config):
                model = load_catalog(catalog_path)[0]
            self.assertEqual(model.server_overrides, {"flash_attn": "on", "reasoning_format": "none", "batch_size": 1024})

    def test_build_llama_server_command_emits_flash_attn_with_explicit_value(self) -> None:
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model-q4.gguf",
            local_path="/tmp/model-q4.gguf",
            server_overrides={"flash_attn": "on"},
        )
        cmd = build_llama_server_command(
            model,
            Path("/tmp/llama-server"),
            port="12345",
        )
        self.assertIn("--flash-attn", cmd)
        idx = cmd.index("--flash-attn")
        self.assertEqual(cmd[idx + 1], "on")

    def test_build_llama_server_command_emits_reasoning_format_with_explicit_value(self) -> None:
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model-q4.gguf",
            local_path="/tmp/model-q4.gguf",
            server_overrides={"reasoning_format": "none"},
        )
        cmd = build_llama_server_command(
            model,
            Path("/tmp/llama-server"),
            port="12345",
        )
        self.assertIn("--reasoning-format", cmd)
        idx = cmd.index("--reasoning-format")
        self.assertEqual(cmd[idx + 1], "none")

    def test_build_llama_server_command_applies_case_insensitive_family_defaults(self) -> None:
        model = ManagedModel(
            model_id="Qwen3.6-27B-UD-Q5_K_XL",
            repo_id="unsloth/Qwen3.6-27B-MTP-GGUF",
            quant="Q5_K_XL",
            filename="Qwen3.6-27B-UD-Q5_K_XL.gguf",
            local_path="/models/qwen.gguf",
        )
        server_defaults = {
            "__family_defaults": {
                "qwen": {
                    "predict": 16384,
                    "reasoning": "on",
                    "reasoning_budget": 2048,
                    "reasoning_budget_message": "Okay, I have thought enough. I will now provide the final answer",
                    "chat_template_kwargs": {"preserve_thinking": False},
                }
            }
        }

        with mock.patch("llamacpp_stack.cli.server_supports_flag", return_value=True):
            cmd = build_llama_server_command(model, Path("/tmp/llama-server"), port="12345", server_defaults=server_defaults)

        self.assertEqual(cmd[cmd.index("--predict") + 1], "16384")
        self.assertEqual(cmd[cmd.index("--reasoning") + 1], "on")
        self.assertEqual(cmd[cmd.index("--reasoning-budget") + 1], "2048")
        self.assertEqual(
            cmd[cmd.index("--reasoning-budget-message") + 1],
            "Okay, I have thought enough. I will now provide the final answer",
        )
        self.assertEqual(cmd[cmd.index("--chat-template-kwargs") + 1], '{"preserve_thinking":false}')


    def test_build_llama_server_command_qwopus_uses_qwen_family_defaults(self) -> None:
        model = ManagedModel(
            model_id="qwopus3.6-35b-a3b-coder-mtp-q4_k_m",
            repo_id="Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF",
            quant="Q4_K_M",
            filename="Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf",
            local_path="/models/qwopus.gguf",
        )
        server_defaults = {"__family_defaults": {"qwen": {"predict": 36384, "reasoning": "off"}}}

        with mock.patch("llamacpp_stack.cli.server_supports_flag", return_value=True):
            cmd = build_llama_server_command(model, Path("/tmp/llama-server"), port="12345", server_defaults=server_defaults)

        self.assertEqual(cmd[cmd.index("--predict") + 1], "36384")
        self.assertEqual(cmd[cmd.index("--reasoning") + 1], "off")

    def test_build_llama_server_command_model_overrides_win_over_family_defaults(self) -> None:
        model = ManagedModel(
            model_id="qwen-small",
            repo_id="org/qwen",
            quant="Q4",
            filename="qwen.gguf",
            local_path="/models/qwen.gguf",
            server_overrides={"predict": 4096},
        )

        with mock.patch("llamacpp_stack.cli.server_supports_flag", return_value=True):
            cmd = build_llama_server_command(
                model,
                Path("/tmp/llama-server"),
                port="12345",
                server_defaults={"__family_defaults": {"QWEN": {"predict": 16384}}},
            )

        self.assertEqual(cmd[cmd.index("--predict") + 1], "4096")

    def test_build_llama_server_command_emits_mtp_spec_flags(self) -> None:
        model = ManagedModel(
            model_id="gemma-4-31b-it-qat-ud-q4_k_xl",
            repo_id="unsloth/gemma-4-31B-it-qat-GGUF",
            quant="UD-Q4_K_XL",
            filename="gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
            local_path="/models/gemma.gguf",
            server_overrides={
                "model_draft": "/models/mtp-gemma-4-31B-it.gguf",
                "spec_type": "draft-mtp",
                "spec_draft_n_max": 4,
            },
        )

        with mock.patch("llamacpp_stack.cli.server_supports_flag", return_value=True):
            cmd = build_llama_server_command(model, Path("/tmp/llama-server"), port="12345")

        self.assertIn("--model-draft", cmd)
        self.assertEqual(cmd[cmd.index("--model-draft") + 1], "/models/mtp-gemma-4-31B-it.gguf")
        self.assertIn("--spec-type", cmd)
        self.assertEqual(cmd[cmd.index("--spec-type") + 1], "draft-mtp")
        self.assertIn("--spec-draft-n-max", cmd)
        self.assertEqual(cmd[cmd.index("--spec-draft-n-max") + 1], "4")

    def test_build_llama_server_command_uses_ctx_size_direct_when_use_fitc_false(self) -> None:
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model-q4.gguf",
            local_path="/tmp/model-q4.gguf",
            ctx_size=65536,
        )

        with mock.patch("llamacpp_stack.cli.server_supports_flag", return_value=True):
            cmd = build_llama_server_command(
                model,
                Path("/tmp/llama-server"),
                port="12345",
                server_defaults={"use_fitc": False, "swa_full": True},
            )

        self.assertIn("--ctx-size", cmd)
        self.assertEqual(cmd[cmd.index("--ctx-size") + 1], "65536")
        self.assertNotIn("-fitc", cmd)
        self.assertNotIn("-fit", cmd)
        self.assertIn("--swa-full", cmd)

    def test_build_llama_server_command_uses_fitc_when_use_fitc_true(self) -> None:
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model-q4.gguf",
            local_path="/tmp/model-q4.gguf",
            ctx_size=65536,
        )

        cmd = build_llama_server_command(
            model,
            Path("/tmp/llama-server"),
            port="12345",
            server_defaults={"use_fitc": True},
        )

        self.assertNotIn("--ctx-size", cmd)
        self.assertIn("-fitc", cmd)
        self.assertEqual(cmd[cmd.index("-fitc") + 1], "65536")

    def test_load_catalog_reprocesses_when_server_defaults_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "catalog.json"
            server_config = root / SERVER_CONFIG_BASENAME

            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "repo-q4",
                            "repo_id": "org/repo",
                            "quant": "Q4",
                            "filename": "model-q4.gguf",
                            "local_path": "/tmp/model-q4.gguf",
                            "server_overrides": {"batch_size": 2048},
                        }
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            server_config.write_text(
                json.dumps({"llama_server_defaults": {"batch_size": 1024}}) + "\n",
                encoding="utf-8",
            )

            with mock.patch("llamacpp_stack.cli.DEFAULT_SERVER_CONFIG_PATH", server_config):
                first = load_catalog(catalog_path)
                self.assertEqual(first[0].server_overrides, {"batch_size": 2048})

                server_config.write_text(
                    json.dumps({"llama_server_defaults": {"batch_size": 2048}}) + "\n",
                    encoding="utf-8",
                )

                refreshed = load_catalog(catalog_path)
                self.assertEqual(refreshed[0].server_overrides, {"batch_size": 2048})

    def test_ensure_llama_server_defaults_file_prefers_bundle_without_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "config"
            config_dir.mkdir(parents=True)

            resolved = _ensure_llama_server_defaults_file(config_dir)

            self.assertTrue(resolved.exists())
            self.assertEqual(resolved.name, LLAMA_SERVER_DEFAULTS_BASENAME)
            self.assertFalse((config_dir / LLAMA_SERVER_DEFAULTS_BASENAME).exists())

    def test_bundled_llama_server_defaults_match_current_policy_for_7_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("llamacpp_stack.install.detect_cuda_device_count", return_value=7):
                defaults = _load_llama_server_defaults_preset(Path(tmp))

        self.assertEqual(defaults["keep"], 20000)
        self.assertEqual(defaults["batch_size"], 4096)
        self.assertEqual(defaults["ubatch_size"], 2048)
        self.assertEqual(defaults["threads"], 16)
        self.assertEqual(defaults["threads_batch"], 8)
        self.assertEqual(defaults["parallel"], 1)
        self.assertEqual(defaults["tensor_split"], "1,1,1,1,1,1,1")
        self.assertFalse(defaults["use_fitc"])
        self.assertTrue(defaults["swa_full"])
        self.assertNotIn("fit_target", defaults)
        self.assertNotIn("mirostat", defaults)

    def test_load_catalog_deduplicates_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "repo-q4",
                            "repo_id": "org/repo",
                            "quant": "Q4",
                            "filename": "model-q4.gguf",
                            "local_path": "/tmp/model-q4.gguf",
                            "aliases": ["chat", " chat ", "chat", "", "friendly"],
                        }
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            model = load_catalog(catalog_path)[0]
            self.assertEqual(model.aliases, ["chat", "friendly"])

    def test_save_catalog_prunes_ttl_equal_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text(
                """
models:
  any-model:
    ttl: 42
""".strip()
                + "\n",
                encoding="utf-8",
            )
            catalog_path = root / "catalog.json"
            model_same = ManagedModel(
                model_id="same-ttl",
                repo_id="org/repo-a",
                quant="Q4",
                filename="a.gguf",
                local_path="/tmp/a.gguf",
                ttl=42,
            )
            model_diff = ManagedModel(
                model_id="diff-ttl",
                repo_id="org/repo-b",
                quant="Q4",
                filename="b.gguf",
                local_path="/tmp/b.gguf",
                ttl=99,
            )

            with mock.patch("llamacpp_stack.cli.DEFAULT_CONFIG_PATH", config_path):
                save_catalog(catalog_path, [model_same, model_diff])

            raw = json.loads(catalog_path.read_text(encoding="utf-8"))
            same_entry = next(item for item in raw if item["model_id"] == "same-ttl")
            diff_entry = next(item for item in raw if item["model_id"] == "diff-ttl")
            self.assertNotIn("ttl", same_entry)
            self.assertEqual(diff_entry.get("ttl"), 99)

        def test_load_catalog_prunes_overrides_equal_to_global_defaults(self) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                server_config = root / SERVER_CONFIG_BASENAME
                server_config.write_text(
                    json.dumps({"llama_server_defaults": {"n_gpu_layers": 16, "batch_size": 1024}}),
                    encoding="utf-8",
                )
                with mock.patch("llamacpp_stack.cli.DEFAULT_SERVER_CONFIG_PATH", server_config):
                    catalog_path = root / "catalog.json"
                    catalog_path.write_text(
                        json.dumps(
                            [
                                {
                                    "model_id": "repo-q4",
                                    "repo_id": "org/repo",
                                    "quant": "Q4",
                                    "filename": "model-q4.gguf",
                                    "local_path": "/tmp/model-q4.gguf",
                                    "server_overrides": {"n_gpu_layers": 16, "batch_size": 1024},
                                }
                            ]
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    model = load_catalog(catalog_path)[0]
                    # per-model overrides that match global defaults should be pruned
                    self.assertEqual(model.server_overrides, {})
                    # and the effective server command should reflect the global defaults
                    cmd = build_llama_server_command(
                        model, Path("/bin/llama"), port="1234", server_defaults=resolve_llama_server_defaults()
                    )
                    self.assertIn("--n-gpu-layers", cmd)
                    idx = cmd.index("--n-gpu-layers")
                    self.assertEqual(cmd[idx + 1], "16")
                    self.assertIn("--batch-size", cmd)
                    idx2 = cmd.index("--batch-size")
                    self.assertEqual(cmd[idx2 + 1], "1024")

    def test_load_catalog_uses_mtime_cache_until_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = Path(tmp) / "catalog.json"
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "repo-q4",
                            "repo_id": "org/repo",
                            "quant": "Q4",
                            "filename": "model-q4.gguf",
                            "local_path": "/tmp/model-q4.gguf",
                        }
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            first = load_catalog(catalog_path)
            second = load_catalog(catalog_path)
            self.assertEqual(first[0].model_id, "repo-q4")
            self.assertEqual(second[0].model_id, "repo-q4")
            self.assertIsNot(first[0], second[0])

            time.sleep(0.001)
            catalog_path.write_text(
                json.dumps(
                    [
                        {
                            "model_id": "repo-q5",
                            "repo_id": "org/repo",
                            "quant": "Q5",
                            "filename": "model-q5.gguf",
                            "local_path": "/tmp/model-q5.gguf",
                        }
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            updated = load_catalog(catalog_path)
            self.assertEqual(updated[0].model_id, "repo-q5")

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
            defaults = resolve_llama_server_defaults(args)
            self.assertTrue(defaults["flash_attn"])
            self.assertEqual(defaults["threads"], 32)
            self.assertNotIn("mmap", defaults)
            self.assertNotIn("split_mode", defaults)
            self.assertEqual(defaults["cache_ram"], 32768)
            self.assertEqual(defaults["ctx_checkpoints"], 32)
            self.assertEqual(defaults["checkpoint_min_step"], 1024)

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

    def test_build_llama_server_command_omits_numa_when_none(self) -> None:
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model-q4.gguf",
            local_path="/tmp/model-q4.gguf",
            server_overrides={"numa": None},
        )
        cmd = build_llama_server_command(
            model,
            Path("/tmp/llama-server"),
            port="12345",
        )
        self.assertNotIn("--numa", cmd)

    def test_build_llama_server_command_uses_mirostat_and_keep(self) -> None:
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model-q4.gguf",
            local_path="/tmp/model-q4.gguf",
            server_overrides={
                "mirostat": 2,
                "mirostat_ent": 4.5,
                "mirostat_lr": 0.1,
                "keep": 512,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
            },
        )
        cmd = build_llama_server_command(
            model,
            Path("/tmp/llama-server"),
            port="12345",
        )
        rendered = " ".join(cmd)
        self.assertIn("--mirostat 2", rendered)
        self.assertIn("--mirostat-ent 4.5", rendered)
        self.assertIn("--mirostat-lr 0.1", rendered)
        self.assertIn("--keep 512", rendered)
        self.assertIn("--cache-type-k q8_0", rendered)
        self.assertIn("--cache-type-v q8_0", rendered)


    def test_get_gpu_conflict_message_ignores_target_when_already_loaded(self) -> None:
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model-q4.gguf",
            local_path="/tmp/model-q4.gguf",
        )
        with (
            mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value={"repo-q4"}),
            mock.patch("llamacpp_stack.cli.get_llama_server_processes", return_value=[{"pid": 123, "model_path": "/tmp/model-q4.gguf"}]),
            mock.patch("llamacpp_stack.cli.get_gpu_process_map", return_value={123: "4000"}),
        ):
            self.assertIsNone(get_gpu_conflict_message("repo-q4", [model]))

    def test_get_gpu_conflict_message_reports_other_model_even_when_target_is_published_but_unloaded(self) -> None:
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
            mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value={"model-b"}),
            mock.patch("llamacpp_stack.cli.get_llama_server_processes", return_value=[{"pid": 123, "model_path": "/tmp/a.gguf"}]),
            mock.patch("llamacpp_stack.cli.get_gpu_process_map", return_value={123: "4096"}),
        ):
            message = get_gpu_conflict_message("model-b", [model_a, model_b])
        self.assertIn("Cannot load model 'model-b'", message)
        self.assertIn("model-a (pid 123, 4096 MiB)", message)

    def test_get_gpu_conflict_message_allows_second_model_when_vram_fits(self) -> None:
        model_a = ManagedModel(
            model_id="model-a",
            repo_id="org/a",
            quant="Q4",
            filename="a.gguf",
            local_path="/tmp/a.gguf",
            tensor_split="1",
        )
        model_b = ManagedModel(
            model_id="model-b",
            repo_id="org/b",
            quant="Q4",
            filename="b.gguf",
            local_path="/tmp/b.gguf",
            tensor_split="1",
        )
        with (
            mock.patch("llamacpp_stack.cli.get_llama_server_processes", return_value=[{"pid": 123, "model_path": "/tmp/a.gguf"}]),
            mock.patch("llamacpp_stack.cli.get_gpu_process_map", return_value={123: "4096"}),
            mock.patch("llamacpp_stack.cli.estimate_model_runtime_mib", return_value=6000.0),
            mock.patch("llamacpp_stack.cli._query_gpu_memory_snapshot_cached", return_value={0: {"free_mib": 12000.0, "used_mib": 4096.0, "total_mib": 24576.0}}),
        ):
            self.assertIsNone(get_gpu_conflict_message("model-b", [model_a, model_b]))

    def test_get_gpu_conflict_message_blocks_second_model_when_vram_does_not_fit(self) -> None:
        model_a = ManagedModel(
            model_id="model-a",
            repo_id="org/a",
            quant="Q4",
            filename="a.gguf",
            local_path="/tmp/a.gguf",
            tensor_split="1",
        )
        model_b = ManagedModel(
            model_id="model-b",
            repo_id="org/b",
            quant="Q4",
            filename="b.gguf",
            local_path="/tmp/b.gguf",
            tensor_split="1",
        )
        with (
            mock.patch("llamacpp_stack.cli.get_llama_server_processes", return_value=[{"pid": 123, "model_path": "/tmp/a.gguf"}]),
            mock.patch("llamacpp_stack.cli.get_gpu_process_map", return_value={123: "4096"}),
            mock.patch("llamacpp_stack.cli.estimate_model_runtime_mib", return_value=9000.0),
            mock.patch("llamacpp_stack.cli._query_gpu_memory_snapshot_cached", return_value={0: {"free_mib": 10000.0, "used_mib": 4096.0, "total_mib": 24576.0}}),
        ):
            message = get_gpu_conflict_message("model-b", [model_a, model_b])
        self.assertIn("Cannot load model 'model-b'", message)
        self.assertIn("model-a (pid 123, 4096 MiB)", message)

    def test_responses_single_message_without_tools_looks_like_model_probe(self) -> None:
        self.assertTrue(request_looks_like_model_probe({
            "model": "gpt-5.4-mini",
            "stream": True,
            "tools": [],
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "ping"}]}],
        }))
        self.assertFalse(request_looks_like_model_probe({
            "model": "gpt-5.5",
            "stream": True,
            "tools": [{"type": "function", "name": "exec_command"}],
            "input": [{"type": "message"}, {"type": "message"}],
        }))

    def test_llamaswap_guard_blocks_upstream_static_assets_only(self) -> None:
        self.assertTrue(is_llamaswap_upstream_static_autoload_path("GET", "/upstream/gemma/sw.js"))
        self.assertTrue(is_llamaswap_upstream_static_autoload_path("HEAD", "/upstream/gemma/assets/app.js"))
        self.assertFalse(is_llamaswap_upstream_static_autoload_path("GET", "/upstream/gemma/health"))
        self.assertFalse(is_llamaswap_upstream_static_autoload_path("POST", "/v1/chat/completions"))
        self.assertFalse(is_llamaswap_upstream_static_autoload_path("POST", "/upstream/gemma/completion"))

    def test_recent_activity_blocks_model_switch_inside_grace_window(self) -> None:
        activity = {
            "qwen": {
                "last_activity_monotonic": 100.0,
                "last_phase": "request_start",
            }
        }
        blocker = recent_activity_blocking_model_switch("gemma", activity, now=125.0, grace_s=30)
        self.assertEqual(blocker, ("qwen", 25.0, "request_start"))
        self.assertIsNone(recent_activity_blocking_model_switch("gemma", activity, now=131.0, grace_s=30))
        self.assertIsNone(recent_activity_blocking_model_switch("qwen", activity, now=125.0, grace_s=30))

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
        self.assertIn("llamacpp-superserver unload", message)

    def test_get_gpu_conflict_message_ignores_foreign_process(self) -> None:
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
        self.assertIsNone(message)

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
                now=386.0,
                idle_ttl=300,
            ),
            (False, 286.0),
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

    def test_choose_auto_ctx_probes_max_second_and_short_circuits_on_success(self) -> None:
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4_K_M",
            filename="model.gguf",
            local_path="/tmp/model.gguf",
        )
        with (
            mock.patch("llamacpp_stack.cli.get_model_context_size", return_value=262144),
            mock.patch("llamacpp_stack.cli._query_gpu_free_memory_mib", return_value={0: 24576.0}),
            mock.patch("llamacpp_stack.cli._parse_probe_trace_metrics", return_value=mock.Mock()),
            mock.patch("llamacpp_stack.cli._estimate_ctx_ceiling", return_value=212992),
            mock.patch(
                "llamacpp_stack.cli.probe_model_ctx",
                side_effect=[(True, "ok"), (True, "ok")],
            ) as probe_mock,
        ):
            selected, status, info = choose_auto_ctx(model, Path("/tmp/llama-server"))

        self.assertEqual([call.args[2] for call in probe_mock.call_args_list], [8192, 262144])
        self.assertEqual(selected, 262144)
        self.assertEqual(status, "selected")
        self.assertEqual(info["selected_ctx"], 262144)

    def test_choose_auto_ctx_refresh_prefers_previous_ctx_first(self) -> None:
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4_K_M",
            filename="model.gguf",
            local_path="/tmp/model.gguf",
            ctx_size=16384,
        )
        with (
            mock.patch("llamacpp_stack.cli.get_model_context_size", return_value=65536),
            mock.patch("llamacpp_stack.cli._query_gpu_free_memory_mib", return_value={0: 24576.0}),
            mock.patch("llamacpp_stack.cli._parse_probe_trace_metrics", return_value=mock.Mock()),
            mock.patch("llamacpp_stack.cli._estimate_ctx_ceiling", return_value=24576),
            mock.patch(
                "llamacpp_stack.cli.probe_model_ctx",
                side_effect=[
                    (True, "ok", {}),
                    (True, "ok", {}),
                    (True, "ok", {}),
                ],
            ) as probe_mock,
        ):
            selected, status, info = choose_auto_ctx(model, Path("/tmp/llama-server"))

        # With a previous configured ctx present, we probe it first.
        self.assertEqual([call.args[2] for call in probe_mock.call_args_list], [16384, 65536])
        self.assertEqual(status, "selected")
        self.assertEqual(selected, 65536)
        self.assertEqual(info["selected_ctx"], 65536)

    def test_choose_auto_ctx_uses_memory_estimate_after_max_failure(self) -> None:
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4_K_M",
            filename="model.gguf",
            local_path="/tmp/model.gguf",
        )
        with (
            mock.patch("llamacpp_stack.cli.get_model_context_size", return_value=32768),
            mock.patch("llamacpp_stack.cli._query_gpu_free_memory_mib", return_value={0: 24576.0}),
            mock.patch("llamacpp_stack.cli._parse_probe_trace_metrics", return_value=mock.Mock()),
            mock.patch("llamacpp_stack.cli._estimate_ctx_ceiling", return_value=12288),
            mock.patch(
                "llamacpp_stack.cli.probe_model_ctx",
                side_effect=[
                    (True, "ok"),
                    (False, "exit--11"),
                    (True, "ok"),
                    (False, "exit--11"),
                    (True, "ok"),
                ],
            ) as probe_mock,
        ):
            selected, status, info = choose_auto_ctx(model, Path("/tmp/llama-server"))

        self.assertEqual([call.args[2] for call in probe_mock.call_args_list], [8192, 32768, 8192, 12288, 10240])
        self.assertEqual(status, "selected")
        self.assertEqual(selected, 10240)
        self.assertEqual(info["first_failure"], 12288)

    def test_choose_auto_ctx_sets_parallel_one_when_retrying_batch_memory_failure(self) -> None:
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4_K_M",
            filename="model.gguf",
            local_path="/tmp/model.gguf",
            server_overrides={"batch_size": 2048, "ubatch_size": 1024, "parallel": 8},
        )
        messages: list[str] = []

        def fake_probe(probed_model: ManagedModel, _server: Path, ctx: int):
            if ctx == 8192:
                return True, "ok", {}
            if probed_model.server_overrides.get("parallel") == 1:
                return True, "ok", {}
            return False, "slot context shift exceeds the available context size", {}

        with (
            mock.patch("llamacpp_stack.cli.get_model_context_size", return_value=262144),
            mock.patch("llamacpp_stack.cli._query_gpu_free_memory_mib", return_value={0: 24576.0}),
            mock.patch("llamacpp_stack.cli._parse_probe_trace_metrics", return_value=mock.Mock()),
            mock.patch("llamacpp_stack.cli._estimate_ctx_ceiling", return_value=8192),
            mock.patch("llamacpp_stack.cli.probe_model_ctx", side_effect=fake_probe) as probe_mock,
        ):
            selected, status, info = choose_auto_ctx(
                model,
                Path("/tmp/llama-server"),
                progress_callback=messages.append,
            )

        self.assertEqual([call.args[2] for call in probe_mock.call_args_list], [8192, 262144, 262144])
        self.assertEqual(selected, 262144)
        self.assertEqual(status, "selected")
        self.assertEqual(info["selected_ctx"], 262144)
        self.assertEqual(model.server_overrides["batch_size"], 512)
        self.assertEqual(model.server_overrides["ubatch_size"], 256)
        self.assertEqual(model.server_overrides["parallel"], 1)
        self.assertTrue(any("parallel=1" in message["message"] for message in messages))

    def test_build_openai_model_list_payload_exposes_lightweight_vision_metadata(self) -> None:
        model = ManagedModel(
            model_id="repo-vl",
            repo_id="org/repo-vl",
            quant="Q4",
            filename="model-vl.gguf",
            local_path="/tmp/model-vl.gguf",
            ctx_size=32768,
            load_capabilities=["image", "image-text-to-text"],
        )

        with mock.patch(
            "llamacpp_stack.cli.refresh_model_load_capabilities",
            side_effect=AssertionError("/v1/models must not read GGUF metadata"),
        ):
            payload = build_openai_model_list_payload(model)

        self.assertEqual(payload["metadata"]["load_capabilities"], ["image", "image-text-to-text"])
        self.assertTrue(payload["metadata"]["vision"])


    def test_gemma4_sliding_window_long_context_trusts_gguf_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gguf = Path(td) / "gemma-4-31B-it-qat-UD-Q4_K_XL.gguf"
            _write_minimal_gguf(
                gguf,
                [
                    ("gemma4.context_length", 4, 262144),
                    ("gemma4.attention.sliding_window", 4, 1024),
                ],
            )
            model = ManagedModel(
                model_id="gemma-4-31b-it-qat-ud-q4_k_xl",
                repo_id="unsloth/gemma-4-31B-it-qat-GGUF",
                quant="UD-Q4_K_XL",
                filename=gguf.name,
                local_path=str(gguf),
                ctx_size=59392,
            )

            self.assertEqual(get_model_context_size(model), 262144)
            self.assertTrue(is_gemma4_sliding_window_long_context_model(model))
            with mock.patch("llamacpp_stack.cli.probe_model_ctx") as probe:
                selected, status, info = choose_auto_ctx(model, Path("/tmp/llama-server"))

            probe.assert_not_called()
            self.assertEqual(selected, 262144)
            self.assertEqual(status, "metadata-selected")
            self.assertEqual(info["selected_ctx"], 262144)

    def test_build_openai_model_payload_exposes_ctx_probe_metrics_with_nc_defaults(self) -> None:
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model.gguf",
            local_path="/tmp/model.gguf",
            ctx_size=32768,
        )
        with mock.patch("llamacpp_stack.cli.get_model_context_size", return_value=65536):
            payload = build_openai_model_payload(model)

        metadata = payload["metadata"]
        self.assertEqual(metadata["configured_context_length"], 32768)
        self.assertEqual(metadata["api_context_length"], 16384)
        self.assertEqual(metadata["context_length"], 32768)
        self.assertEqual(metadata["gguf_context_length"], 65536)
        self.assertIsNone(metadata["ctx_probe_read_s"])
        self.assertIsNone(metadata["ctx_probe_tokens_s"])
        self.assertIsNone(metadata["ctx_probe_totals_s"])
        self.assertIsNone(metadata["ctx_probe_latency_ms"])
        self.assertIsNone(metadata["ctx_probe_speed_tps"])
        self.assertIsNone(metadata["ctx_probe_kv_gb"])
        self.assertEqual(metadata["ctx_probe_read"], "NC")
        self.assertEqual(metadata["ctx_probe_tokens"], "NC")
        self.assertEqual(metadata["ctx_probe_totals"], "NC")
        self.assertEqual(metadata["ctx_probe_latency"], "NC")
        self.assertEqual(metadata["ctx_probe_speed"], "NC")
        self.assertEqual(metadata["ctx_probe_kv"], "NC")

    def test_build_ollama_model_payload_exposes_load_capabilities_for_api_and_ps(self) -> None:
        model = ManagedModel(
            model_id="repo-vl",
            repo_id="org/repo-vl",
            quant="Q4",
            filename="model-vl.gguf",
            local_path="/tmp/model-vl.gguf",
            ctx_size=32768,
            load_capabilities=["image", "image-text-to-text"],
        )
        with mock.patch("llamacpp_stack.cli.get_model_context_size", return_value=65536):
            payload = build_ollama_model_payload(model, loaded=True, process={"pid": 1234}, gpu_process_map={})

        self.assertEqual(payload["details"]["load_capabilities"], ["image", "image-text-to-text"])
        self.assertEqual(payload["model_info"]["llamacpp.load_capabilities"], ["image", "image-text-to-text"])
        self.assertTrue(payload["details"]["vision"])

    def test_render_models_table_shows_ctx_gb_and_rate_columns(self) -> None:
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model.gguf",
            local_path="/tmp/model.gguf",
            ctx_probe_kv_gb=3.25,
            ctx_probe_read_s=118.4,
            ctx_probe_tokens_s=56.2,
            ctx_probe_totals_s=174.6,
        )
        with (
            mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value=set()),
            mock.patch("llamacpp_stack.cli.get_llama_server_processes", return_value=[]),
            mock.patch("llamacpp_stack.cli.get_gpu_process_map", return_value={}),
            mock.patch("llamacpp_stack.cli.get_model_context_size", return_value=65536),
            mock.patch("llamacpp_stack.cli.get_model_storage_info", return_value={"size": 1024, "file_count": 1, "status": "ready"}),
        ):
            table = render_models_table([model], host="127.0.0.1", port=11435, idle_ttl=10)

        self.assertIn("CTX_GB", table)
        self.assertIn("READ/S", table)
        self.assertIn("TOKEN/S", table)
        self.assertIn("TOTAL/S", table)
        self.assertIn("CFG_CTX", table)
        self.assertIn("API_CTX", table)
        self.assertNotIn("MAX_CTX", table)
        self.assertIn("3.25", table)
        self.assertIn("118.4 tok/s", table)
        self.assertIn("56.2 tok/s", table)
        self.assertIn("174.6 tok/s", table)

    def test_render_models_table_marks_error_for_failed_min_probe(self) -> None:
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4",
            filename="model.gguf",
            local_path="/tmp/model.gguf",
            auto_ctx_failed=True,
            auto_ctx_error="min-failed:timeout",
        )
        with (
            mock.patch("llamacpp_stack.cli.get_published_model_ids", return_value=set()),
            mock.patch("llamacpp_stack.cli.get_llama_server_processes", return_value=[]),
            mock.patch("llamacpp_stack.cli.get_gpu_process_map", return_value={}),
            mock.patch("llamacpp_stack.cli.get_model_storage_info", return_value={"size": 1024, "file_count": 1, "status": "ready"}),
        ):
            table = render_models_table([model], host="127.0.0.1", port=11435, idle_ttl=10)

        self.assertIn("ERROR", table)

    def test_choose_auto_ctx_returns_min_failed_when_guard_probe_fails(self) -> None:
        model = ManagedModel(
            model_id="repo-q4",
            repo_id="org/repo",
            quant="Q4_K_M",
            filename="model.gguf",
            local_path="/tmp/model.gguf",
        )
        with (
            mock.patch("llamacpp_stack.cli.get_model_context_size", return_value=32768),
            mock.patch("llamacpp_stack.cli._query_gpu_free_memory_mib", return_value={0: 24576.0}),
            mock.patch("llamacpp_stack.cli._parse_probe_trace_metrics", return_value=mock.Mock()),
            mock.patch("llamacpp_stack.cli._estimate_ctx_ceiling", return_value=12288),
            mock.patch(
                "llamacpp_stack.cli.probe_model_ctx",
                side_effect=[
                    (True, "ok"),
                    (False, "exit--11"),
                    (False, "timeout"),
                ],
            ) as probe_mock,
        ):
            selected, status, info = choose_auto_ctx(model, Path("/tmp/llama-server"))

        self.assertIsNone(selected)
        self.assertEqual(status, "min-failed")
        self.assertEqual(info["min_ctx"], 8192)
        self.assertEqual([call.args[2] for call in probe_mock.call_args_list], [8192, 32768, 8192])

    def test_model_name_aliases_include_filename_variants_and_repo_with_without_hf_prefix(self) -> None:
        model = ManagedModel(
            model_id="qwen2-5-7b-instruct-q4-k-m",
            repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
            quant="Q4_K_M",
            filename="Qwen2.5-7B-Instruct-Q4_K_M-00001-of-00025.gguf",
            local_path="/tmp/Qwen2.5-7B-Instruct-Q4_K_M-00001-of-00025.gguf",
            aliases=["custom-alias"],
        )

        aliases = model_name_aliases(model)

        self.assertIn("custom-alias", aliases)
        self.assertIn("Qwen2.5-7B-Instruct-Q4_K_M-00001-of-00025.gguf", aliases)
        self.assertIn("Qwen2.5-7B-Instruct-Q4_K_M-00001-of-00025", aliases)
        self.assertIn("Qwen2.5-7B-Instruct-Q4_K_M", aliases)
        self.assertIn("hf.co/Qwen/Qwen2.5-7B-Instruct-GGUF", aliases)
        self.assertIn("Qwen/Qwen2.5-7B-Instruct-GGUF", aliases)
        self.assertIn("hf.co/Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M", aliases)
        self.assertIn("Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M", aliases)

    def test_restart_service_to_free_vram_uses_user_systemctl_when_not_root(self) -> None:
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with (
            mock.patch("llamacpp_stack.cli.os.geteuid", return_value=1000),
            mock.patch("llamacpp_stack.cli.subprocess.run", side_effect=fake_run),
            mock.patch("llamacpp_stack.cli.time.sleep"),
        ):
            self.assertTrue(restart_service_to_free_vram("llamacpp-superserver-swap.service"))

        self.assertEqual(calls[0][:3], ["systemctl", "--user", "restart"])

    def test_restart_service_to_free_vram_uses_system_systemctl_when_root(self) -> None:
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with (
            mock.patch("llamacpp_stack.cli.os.geteuid", return_value=0),
            mock.patch("llamacpp_stack.cli.subprocess.run", side_effect=fake_run),
            mock.patch("llamacpp_stack.cli.time.sleep"),
        ):
            self.assertTrue(restart_service_to_free_vram("llamacpp-superserver-swap.service"))

        self.assertEqual(calls[0][:2], ["systemctl", "restart"])

    def test_resolve_catalog_model_name_accepts_filename_and_repo_alias_variants(self) -> None:
        model = ManagedModel(
            model_id="qwen2-5-7b-instruct-q4-k-m",
            repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
            quant="Q4_K_M",
            filename="Qwen2.5-7B-Instruct-Q4_K_M-00001-of-00025.gguf",
            local_path="/tmp/Qwen2.5-7B-Instruct-Q4_K_M-00001-of-00025.gguf",
        )
        catalog = [model]

        self.assertEqual(
            resolve_catalog_model_name("Qwen2.5-7B-Instruct-Q4_K_M-00001-of-00025.gguf", catalog),
            model.model_id,
        )
        self.assertEqual(
            resolve_catalog_model_name("Qwen2.5-7B-Instruct-Q4_K_M-00001-of-00025", catalog),
            model.model_id,
        )
        self.assertEqual(
            resolve_catalog_model_name("Qwen2.5-7B-Instruct-Q4_K_M", catalog),
            model.model_id,
        )
        self.assertEqual(
            resolve_catalog_model_name("Qwen/Qwen2.5-7B-Instruct-GGUF", catalog),
            model.model_id,
        )
        self.assertEqual(
            resolve_catalog_model_name("hf.co/Qwen/Qwen2.5-7B-Instruct-GGUF", catalog),
            model.model_id,
        )
        self.assertEqual(
            resolve_catalog_model_name("Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M", catalog),
            model.model_id,
        )
        self.assertEqual(
            resolve_catalog_model_name("hf.co/Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M", catalog),
            model.model_id,
        )

    def test_resolve_catalog_model_name_accepts_precision_suffix_aliases(self) -> None:
        model = ManagedModel(
            model_id="qwen3.6-35b-a3b-ud-q4_k_m",
            repo_id="unsloth/Qwen3.6-35B-A3B-GGUF",
            quant="UD-Q4_K_M",
            filename="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            local_path="/tmp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        )
        catalog = [model]

        for requested in (
            "qwen3.6-35b-a3b-ud-q4_k_m-bf16",
            "qwen3.6-35b-a3b-ud-q4_k_m:BF16",
            "qwen3.6-35b-a3b-ud-q4_k_m.fp16",
            "llamacpp_rig_cpp/qwen3.6-35b-a3b-ud-q4_k_m-bf16",
        ):
            self.assertEqual(resolve_catalog_model_name(requested, catalog), model.model_id)




    def test_summarize_configured_replicas_uses_detected_gpu_count(self) -> None:
        model = ManagedModel(
            model_id="qwen",
            repo_id="org/qwen",
            quant="Q4",
            filename="qwen.gguf",
            local_path="/tmp/qwen.gguf",
            tensor_split="1,1,1",
        )
        with mock.patch("llamacpp_stack.cli.detect_cuda_device_count", return_value=7):
            lines = summarize_configured_replicas(
                [model],
                {"enabled": True, "max": "auto", "gpus_per_replica": 3, "placement": "exclusive_gpus"},
            )
        self.assertEqual(len(lines), 1)
        self.assertIn("gpu_sets=[[3, 4, 5]]", lines[0])

    def test_catalog_auto_update_watch_triggers_update_on_catalog_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "catalog.json"
            server_config_path = root / "conf.json"
            catalog_path.write_text("[]", encoding="utf-8")
            server_config_path.write_text("{}", encoding="utf-8")
            args = Namespace(catalog=catalog_path, server_config=server_config_path)
            calls = []

            def fake_update(received_args):
                calls.append(received_args.catalog)

            stop_event = threading.Event()
            with (
                mock.patch("llamacpp_stack.cli.update_config", side_effect=fake_update),
                mock.patch("llamacpp_stack.cli._active_download_blocker_summary", return_value=""),
            ):
                thread = start_catalog_auto_update_watch(args, poll_s=0.05, debounce_s=0.05, stop_event=stop_event)
                try:
                    time.sleep(0.12)
                    catalog_path.write_text("[{\"model_id\": \"changed\"}]", encoding="utf-8")
                    deadline = time.time() + 2.0
                    while time.time() < deadline and not calls:
                        time.sleep(0.05)
                finally:
                    stop_event.set()
                    thread.join(timeout=1.0)

            self.assertTrue(calls)
            self.assertEqual(calls[-1], catalog_path)


    def test_catalog_auto_update_watch_does_not_loop_on_update_writing_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog_path = root / "catalog.json"
            server_config_path = root / "conf.json"
            catalog_path.write_text("[]", encoding="utf-8")
            server_config_path.write_text("{}", encoding="utf-8")
            args = Namespace(catalog=catalog_path, server_config=server_config_path)
            calls = []

            def fake_update(received_args):
                calls.append(time.time())
                catalog_path.write_text("[{\"model_id\": \"canonical\"}]", encoding="utf-8")

            stop_event = threading.Event()
            with (
                mock.patch("llamacpp_stack.cli.update_config", side_effect=fake_update),
                mock.patch("llamacpp_stack.cli._active_download_blocker_summary", return_value=""),
            ):
                thread = start_catalog_auto_update_watch(args, poll_s=0.05, debounce_s=0.05, stop_event=stop_event)
                try:
                    time.sleep(0.12)
                    catalog_path.write_text("[{\"model_id\": \"changed\"}]", encoding="utf-8")
                    deadline = time.time() + 2.0
                    while time.time() < deadline and not calls:
                        time.sleep(0.05)
                    self.assertEqual(len(calls), 1)
                    time.sleep(0.35)
                finally:
                    stop_event.set()
                    thread.join(timeout=1.0)

            self.assertEqual(len(calls), 1)

    def test_same_models_dir_matches_equivalent_existing_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root / "models" / ".." / "models"
            existing = root / "models"
            self.assertTrue(_same_models_dir(selected, existing))
            self.assertFalse(_same_models_dir(root / "other", existing))
            self.assertFalse(_same_models_dir(existing, None))

    def test_models_dir_ready_system_mode_does_not_require_invoking_user_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            models_dir.mkdir()
            models_dir.chmod(0o775)
            st = models_dir.stat()

            with mock.patch("llamacpp_stack.install.pwd.getpwnam", return_value=Namespace(pw_uid=st.st_uid)), \
                 mock.patch("llamacpp_stack.install.grp.getgrnam", return_value=Namespace(gr_gid=st.st_gid)):
                self.assertTrue(
                    _models_dir_already_ready_for_owner(
                        models_dir,
                        "llamaswap",
                        "llamaswap",
                        system_mode=True,
                    )
                )

    def test_ensure_models_dir_ready_skips_sudo_when_existing_system_dir_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp) / "models"
            models_dir.mkdir()
            models_dir.chmod(0o775)
            st = models_dir.stat()
            layout = choose_layout("system", "127.0.0.1", 8080, models_dir)

            with mock.patch("llamacpp_stack.install.pwd.getpwnam", return_value=Namespace(pw_uid=st.st_uid)), \
                 mock.patch("llamacpp_stack.install.grp.getgrnam", return_value=Namespace(gr_gid=st.st_gid)), \
                 mock.patch("llamacpp_stack.install.prompt_bool") as prompt_mock, \
                 mock.patch("llamacpp_stack.install._run") as run_mock:
                ensure_models_dir_ready(layout, dry_run=False)

            prompt_mock.assert_not_called()
            run_mock.assert_not_called()

if __name__ == "__main__":
    unittest.main()
