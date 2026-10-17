from pathlib import Path

import pytest

from runpup.domain.models import Availability, RootKind, StorageRoot
from runpup.services.paths import PathResolver


def test_available_file_and_directory(tmp_path):
    root = StorageRoot(name="Root", location=str(tmp_path))
    file = tmp_path / "adapter.bin"
    file.write_bytes(b"checkpoint")
    for reference in ("adapter.bin", "."):
        result = PathResolver().resolve(root, reference)
        assert result.state is Availability.AVAILABLE
        assert result.root_available is result.reference_available is True
    assert PathResolver().resolve(root, "adapter.bin").path == file


def test_offline_root_is_different_from_missing_file(tmp_path):
    offline = StorageRoot(name="Unmounted", location=str(tmp_path / "absent-mount"))
    result = PathResolver().resolve(offline, "adapter.bin")
    assert result.state is Availability.OFFLINE
    assert result.root_available is False
    assert result.reference_available is None
    online = StorageRoot(name="Mounted", location=str(tmp_path))
    result = PathResolver().resolve(online, "absent.bin")
    assert result.state is Availability.MISSING
    assert result.root_available is True
    assert result.reference_available is False


def test_unverified_does_not_probe_filesystem(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("Unchecked resolution must not access the root")

    monkeypatch.setattr("runpup.services.paths.os.scandir", forbidden)
    root = StorageRoot(name="Unverified", location=str(tmp_path))
    result = PathResolver().resolve(root, "adapter.bin", verify=False)
    assert result.path == tmp_path / "adapter.bin"
    assert result.state is Availability.UNVERIFIED
    assert result.root_available is result.reference_available is None


def test_portable_root_uses_explicit_base(tmp_path):
    root = StorageRoot(name="Portable", location="outputs", kind=RootKind.PORTABLE_RELATIVE)
    assert PathResolver().resolve(root, "run").state is Availability.UNVERIFIED
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "run").mkdir()
    result = PathResolver(portable_base=tmp_path).resolve(root, "run")
    assert result.path == tmp_path / "outputs" / "run"
    assert result.state is Availability.AVAILABLE


@pytest.mark.parametrize("reference", ["../outside", "a/../../outside", "/absolute", "C:/absolute", "a\\b", "", "\0bad"])
def test_unsafe_reference_is_rejected(tmp_path, reference):
    with pytest.raises(ValueError):
        PathResolver().resolve(StorageRoot(name="Root", location=str(tmp_path)), reference)


def test_symlink_escape_is_unverified(tmp_path):
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    (root_dir / "link").symlink_to(outside)
    root = StorageRoot(name="Root", location=str(root_dir))
    result = PathResolver().resolve(root, "link")
    assert result.state is Availability.UNVERIFIED
    assert "escapes" in result.reason


def test_inaccessible_root_is_offline(tmp_path, monkeypatch):
    def denied(*args):
        raise PermissionError("unavailable root")

    monkeypatch.setattr("runpup.services.paths.os.scandir", denied)
    result = PathResolver().resolve(StorageRoot(name="Root", location=str(tmp_path)), "file")
    assert result.state is Availability.OFFLINE


def test_inaccessible_reference_is_unverified(tmp_path, monkeypatch):
    original = Path.stat
    target = tmp_path / "private"

    def denied(path, *args, **kwargs):
        if path == target:
            raise PermissionError("private reference")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied)
    result = PathResolver().resolve(StorageRoot(name="Root", location=str(tmp_path)), "private")
    assert result.root_available is True
    assert result.reference_available is None
    assert result.state is Availability.UNVERIFIED


def test_reference_beneath_a_regular_file_is_missing(tmp_path):
    (tmp_path / "file").write_bytes(b"not a directory")
    result = PathResolver().resolve(StorageRoot(name="Root", location=str(tmp_path)), "file/child")
    assert result.state is Availability.MISSING
    assert result.root_available is True
    assert result.reference_available is False
