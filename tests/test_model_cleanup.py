import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llamacpp_stack.cli import (
    find_orphan_model_files,
    remove_orphan_models,
    _delete_path_with_permission_fallback,
)


class ModelCleanupTests(unittest.TestCase):
    def test_orphan_scan_preserves_catalog_shards_and_finds_unregistered_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "org" / "model"
            repo.mkdir(parents=True)
            referenced = repo / "model-00001-of-00002.gguf"
            referenced.write_bytes(b"")
            (repo / "model-00002-of-00002.gguf").write_bytes(b"")
            orphan = repo / "other-q4.gguf"
            orphan.write_bytes(b"")
            (repo / "readme.md").write_bytes(b"")
            catalog = [mock.Mock(local_path=str(referenced), mmproj_path="", filename=referenced.name)]

            found = find_orphan_model_files(catalog, root)

            self.assertEqual(found, [orphan])

    def test_orphan_scan_preserves_files_when_catalog_points_to_native_model_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "org" / "native"
            native.mkdir(parents=True)
            kept = native / "model-00001-of-00002.safetensors"
            kept.write_bytes(b"")
            (native / "model-00002-of-00002.safetensors").write_bytes(b"")
            orphan = root / "org" / "unused.gguf"
            orphan.write_bytes(b"")
            catalog = [mock.Mock(local_path=str(native), mmproj_path="", filename="hf-native")]

            self.assertEqual(find_orphan_model_files(catalog, root), [orphan])

    def test_orphan_removal_dry_run_does_not_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orphan = root / "orphan.gguf"
            orphan.write_bytes(b"")
            args = argparse.Namespace(
                catalog=Path(tmp) / "catalog.json",
                models_dir=root,
                dry_run=True,
                yes=True,
            )
            with mock.patch("llamacpp_stack.cli.load_catalog", return_value=[]):
                self.assertEqual(remove_orphan_models(args), 0)
            self.assertTrue(orphan.exists())

    def test_permission_error_uses_sudo_rm_after_local_delete_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.gguf"
            target.write_bytes(b"")
            with mock.patch.object(Path, "unlink", side_effect=PermissionError("denied")), \
                 mock.patch("llamacpp_stack.cli.shutil.which", return_value="/usr/bin/sudo"), \
                 mock.patch("llamacpp_stack.cli.subprocess.run") as run:
                _delete_path_with_permission_fallback(target, Path(tmp))
            run.assert_called_once_with(
                ["/usr/bin/sudo", "rm", "-f", "--", str(target)],
                check=True,
                capture_output=True,
                text=True,
            )


if __name__ == "__main__":
    unittest.main()
