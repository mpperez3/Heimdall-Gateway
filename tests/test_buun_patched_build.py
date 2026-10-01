import pytest

from llamacpp_stack import buun_install


def test_bundled_patch_files_lists_shipped_patches():
    names = [p.name for p in buun_install._bundled_patch_files()]
    assert "buun-qwen3_5-native-mmproj.patch" in names
    assert "buun-exl3-shared-head-side-tensors.patch" in names


def test_empty_bundle_raises_instead_of_silently_skipping(monkeypatch, tmp_path):
    """A dropped bundle must fail loudly, not build an unpatched buun."""
    missing = tmp_path / "no-such-dir"
    monkeypatch.setattr(buun_install, "_bundled_patches_dir", lambda: missing)

    for call in (
        lambda: buun_install._bundled_patch_files(),
        lambda: buun_install._bundled_patches_pending(tmp_path / "buun"),
        lambda: buun_install._write_patch_stamp(tmp_path / "buun"),
        lambda: buun_install._apply_bundled_patches(tmp_path / "buun"),
    ):
        with pytest.raises(RuntimeError, match="no bundled buun patches"):
            call()


def test_empty_bundle_dir_raises(monkeypatch, tmp_path):
    empty = tmp_path / "patches"
    empty.mkdir()
    monkeypatch.setattr(buun_install, "_bundled_patches_dir", lambda: empty)
    with pytest.raises(RuntimeError, match="no bundled buun patches"):
        buun_install._bundled_patch_files()


def test_stamp_roundtrip_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.undo()
    buun = tmp_path / "buun"
    buun.mkdir()

    assert buun_install._bundled_patches_pending(buun) is True
    buun_install._write_patch_stamp(buun)
    assert buun_install._bundled_patches_pending(buun) is False

    buun_install._write_patch_stamp(buun)
    assert buun_install._bundled_patches_pending(buun) is False

    stamp = buun_install._patch_stamp_path(buun).read_text(encoding="utf-8").strip()
    assert stamp == buun_install._bundled_patch_fingerprint()


def test_stale_stamp_forces_rebuild(tmp_path):
    buun = tmp_path / "buun"
    buun.mkdir()
    buun_install._patch_stamp_path(buun).write_text("deadbeef\n", encoding="utf-8")
    assert buun_install._bundled_patches_pending(buun) is True