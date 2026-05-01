import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import llamacpp_stack.cli as cli
from llamacpp_stack.install import _merge_missing_llama_server_defaults
from llamacpp_stack.cli import build_cli_parser, build_llama_server_command, ManagedModel


class SpeculativeSupportTest(unittest.TestCase):
    def test_merge_includes_speculative_defaults_from_bundle(self) -> None:
        target = {}
        # config_dir value is not required because installer bundle contains the preset
        changed = _merge_missing_llama_server_defaults(target, Path("/nonexistent"))
        self.assertTrue(changed)
        self.assertIn("speculative_defaults", target)
        spec = target["speculative_defaults"]
        self.assertIsInstance(spec, dict)
        # Expect id_prefix and fitt to be present as authored in the bundled YAML
        self.assertIn("id_prefix", spec)
        self.assertEqual(str(spec.get("id_prefix")), "speculative-")
        self.assertIn("fitt", spec)
        self.assertEqual(int(spec.get("fitt")), 1024)

    def test_cli_parser_accepts_speculative_flag(self) -> None:
        parser, _ = build_cli_parser()
        args = parser.parse_args(["add", "org/repo", "--speculative"])
        self.assertTrue(getattr(args, "speculative", False))

    def test_build_llama_server_command_applies_spec_defaults_for_speculative_model(self) -> None:
        model = ManagedModel(
            model_id="foo",
            repo_id="org/foo",
            quant=None,
            filename="foo.gguf",
            local_path="/models/foo.gguf",
            ctx_size=8192,
        )
        model.speculative = True

        server_defaults = {
            "speculative_defaults": {
                "fit": "on",
                "fitt": 2048,
            }
        }

        cmd = build_llama_server_command(model, Path("/bin/llama-server"), port="12345", server_defaults=server_defaults)
        joined = " ".join(cmd)
        # llama-server expects short (single-dash) fit flags; assert those
        self.assertIn("-fit", joined)
        self.assertIn("-fitt 2048", joined)
        self.assertIn("-fitc 8192", joined)
        self.assertNotIn("--ctx-size", joined)

    def test_build_llama_server_command_fit_off_uses_ctx_size_and_omits_fitc(self) -> None:
        model = ManagedModel(
            model_id="foo",
            repo_id="org/foo",
            quant=None,
            filename="foo.gguf",
            local_path="/models/foo.gguf",
            ctx_size=262144,
        )
        model.speculative = True

        server_defaults = {
            "speculative_defaults": {
                "fit": "off",
                "fitt": 2048,
                "fitc": 4096,
            }
        }

        cmd = build_llama_server_command(model, Path("/bin/llama-server"), port="12345", server_defaults=server_defaults)
        joined = " ".join(cmd)
        self.assertIn("-fit off", joined)
        self.assertIn("--ctx-size 262144", joined)
        self.assertNotIn("-fitc", joined)
        self.assertNotIn("-fitt", joined)

    def test_build_llama_server_command_fit_on_respects_explicit_fitc(self) -> None:
        model = ManagedModel(
            model_id="foo",
            repo_id="org/foo",
            quant=None,
            filename="foo.gguf",
            local_path="/models/foo.gguf",
            ctx_size=262144,
        )
        model.speculative = True

        server_defaults = {
            "speculative_defaults": {
                "fit": "on",
                "fitc": 131072,
            }
        }

        cmd = build_llama_server_command(model, Path("/bin/llama-server"), port="12345", server_defaults=server_defaults)
        joined = " ".join(cmd)
        self.assertIn("-fit on", joined)
        self.assertIn("-fitc 131072", joined)
        self.assertNotIn("--ctx-size", joined)

    def test_build_llama_server_command_supports_explicit_draft_model_flags(self) -> None:
        model = ManagedModel(
            model_id="foo-spec",
            repo_id="org/foo",
            quant=None,
            filename="foo.gguf",
            local_path="/models/foo.gguf",
            ctx_size=8192,
            server_overrides={
                "model_draft": "/models/foo-draft.gguf",
                "draft": 12,
                "draft_min": 3,
                "draft_p_min": 0.85,
            },
        )
        model.speculative = True

        cmd = build_llama_server_command(model, Path("/bin/llama-server"), port="12345")
        joined = " ".join(cmd)
        self.assertIn("--model-draft /models/foo-draft.gguf", joined)
        self.assertIn("--spec-draft-n-max 12", joined)
        # draft_min and draft_p_min were removed in newer llama.cpp API
        self.assertNotIn("--draft-min", joined)
        self.assertNotIn("--draft-p-min", joined)

    def test_run_parser_accepts_repeated_hf_for_speculative_pair(self) -> None:
        parser, _ = build_cli_parser()
        args = parser.parse_args(
            [
                "run",
                "-hf",
                "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M",
                "--speculative",
                "-hf",
                "unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ1_M",
            ]
        )
        self.assertTrue(getattr(args, "speculative", False))
        self.assertIsInstance(args.hf, list)
        self.assertEqual(len(args.hf), 2)

    def test_run_command_pairs_master_and_draft_and_chats_spec_id(self) -> None:
        parser, _ = build_cli_parser()
        args = parser.parse_args(
            [
                "run",
                "-hf",
                "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M",
                "--speculative",
                "-hf",
                "unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ1_M",
            ]
        )

        with patch("llamacpp_stack.cli.ensure_model_available") as ensure_mock, patch(
            "llamacpp_stack.cli.start_chat", return_value=0
        ) as start_chat_mock:
            ensure_mock.side_effect = ["master-mid", "draft-mid", "speculative-master-mid"]

            result = cli.run_command(args)

            self.assertEqual(result, 0)
            self.assertEqual(ensure_mock.call_count, 3)

            master_call_args = ensure_mock.call_args_list[0].args[0]
            draft_call_args = ensure_mock.call_args_list[1].args[0]
            spec_call_args = ensure_mock.call_args_list[2].args[0]

            self.assertEqual(master_call_args.hf, "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M")
            self.assertFalse(bool(getattr(master_call_args, "speculative", False)))
            self.assertTrue(bool(getattr(master_call_args, "defer_publish", False)))
            self.assertFalse(bool(getattr(master_call_args, "skip_ctx", False)))

            self.assertEqual(draft_call_args.hf, "unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ1_M")
            self.assertFalse(bool(getattr(draft_call_args, "speculative", False)))
            self.assertTrue(bool(getattr(draft_call_args, "defer_publish", False)))
            self.assertTrue(bool(getattr(draft_call_args, "skip_ctx", False)))

            self.assertIsNone(getattr(spec_call_args, "hf", None))
            self.assertTrue(bool(getattr(spec_call_args, "speculative", False)))
            self.assertEqual(getattr(spec_call_args, "spec_base_model_id", None), "master-mid")
            self.assertEqual(getattr(spec_call_args, "spec_draft_model_id", None), "draft-mid")
            self.assertEqual(getattr(spec_call_args, "model_id", None), "master-mid")
            self.assertTrue(bool(getattr(spec_call_args, "skip_ctx", False)))

            start_chat_mock.assert_called_once_with("speculative-master-mid", args.public_host, args.public_port)

    def test_update_auto_ctx_paired_speculative_probes_master_and_pair_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "models"
            master_repo_dir = models_dir / "org/master"
            draft_repo_dir = models_dir / "org/draft"
            master_repo_dir.mkdir(parents=True)
            draft_repo_dir.mkdir(parents=True)

            master_path = master_repo_dir / "master.gguf"
            draft_path = draft_repo_dir / "draft.gguf"
            master_path.write_bytes(b"gguf")
            draft_path.write_bytes(b"gguf")

            catalog_path = root / "catalog.json"
            config_path = root / "config.yaml"
            cli.save_catalog(
                catalog_path,
                [
                    ManagedModel(
                        model_id="master-mid",
                        repo_id="org/master",
                        quant="Q4_K_M",
                        filename="master.gguf",
                        local_path=str(master_path),
                        ctx_size=8192,
                    ),
                    ManagedModel(
                        model_id="draft-mid",
                        repo_id="org/draft",
                        quant="IQ1_M",
                        filename="draft.gguf",
                        local_path=str(draft_path),
                        ctx_size=8192,
                    ),
                    ManagedModel(
                        model_id="spec-master-mid",
                        repo_id="org/master",
                        quant="Q4_K_M",
                        filename="master.gguf",
                        local_path=str(master_path),
                        ctx_size=8192,
                        speculative=True,
                        spec_variant_of="master-mid",
                        spec_meta={
                            "base_model_id": "master-mid",
                            "draft_model_id": "draft-mid",
                        },
                        server_overrides={"model_draft": str(draft_path)},
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
                preserve_ctx=False,
                sync_gguf_ctx=False,
                catalog=catalog_path,
                config=config_path,
                llama_server=root / "llama-server",
                start_port=18080,
                public_host="127.0.0.1",
                public_port=11435,
                idle_ttl=10,
                server_config=root / "llamacpp-superserver.json",
                models_dir=models_dir,
            )

            response = mock.Mock(status_code=200)
            response.json.return_value = {"data": []}

            with (
                mock.patch("llamacpp_stack.cli.temporarily_unload_published_models"),
                mock.patch(
                    "llamacpp_stack.cli.choose_auto_ctx",
                    return_value=(
                        24576,
                        "selected",
                        {
                            "selected_ctx": 24576,
                            "probe_read_s": 120.0,
                            "probe_tokens_s": 80.0,
                            "probe_totals_s": 200.0,
                            "probe_latency_ms": 150.0,
                            "selected_ctx_gb": 10.5,
                        },
                    ),
                ) as choose_mock,
                mock.patch(
                    "llamacpp_stack.cli.probe_model_ctx",
                    return_value=(
                        True,
                        "ok",
                        {
                            "probe_read_s": 90.0,
                            "probe_tokens_s": 60.0,
                            "probe_totals_s": 150.0,
                            "probe_latency_ms": 110.0,
                            "selected_ctx_gb": 11.0,
                        },
                    ),
                ) as pair_probe_mock,
                mock.patch("llamacpp_stack.cli.resolve_llama_server_defaults", return_value={}),
                mock.patch("llamacpp_stack.cli.time.sleep"),
                mock.patch("llamacpp_stack.cli.requests.get", return_value=response),
            ):
                result = cli.update_config(args)

            self.assertEqual(result, "updated")
            self.assertEqual(choose_mock.call_count, 1)
            self.assertEqual(pair_probe_mock.call_count, 1)

            choose_model = choose_mock.call_args.args[0]
            self.assertEqual(choose_model.model_id, "master-mid")

            probe_model = pair_probe_mock.call_args.args[0]
            probe_ctx = int(pair_probe_mock.call_args.args[2])
            self.assertEqual(probe_model.model_id, "spec-master-mid")
            self.assertEqual(probe_ctx, 24576)

            refreshed = cli.load_catalog(catalog_path)
            by_id = {m.model_id: m for m in refreshed}
            self.assertEqual(by_id["master-mid"].ctx_size, 24576)
            self.assertEqual(by_id["draft-mid"].ctx_size, 8192)
            self.assertEqual(by_id["spec-master-mid"].ctx_size, 24576)

    def test_run_command_normalizes_single_hf_before_ensure(self) -> None:
        parser, _ = build_cli_parser()
        args = parser.parse_args(["run", "-hf", "org/repo:Q4_K_M", "--no-chat"])

        with patch("llamacpp_stack.cli.ensure_model_available", return_value="org-repo-q4") as ensure_mock:
            result = cli.run_command(args)

            self.assertEqual(result, 0)
            self.assertEqual(ensure_mock.call_count, 1)
            effective_args = ensure_mock.call_args.args[0]
            self.assertEqual(effective_args.hf, "org/repo:Q4_K_M")


if __name__ == "__main__":
    unittest.main()
