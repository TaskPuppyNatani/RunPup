import builtins
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import io
import os
from pathlib import Path
import sqlite3

import pytest

from runpup.domain.models import Artifact, Availability, Dataset, RootKind, Run, StorageRoot
from runpup.persistence.storage_roots import RootInUse
from runpup.services.paths import PathResolver
from runpup.services.storage_roots import root_status, verify_storage_root


def save_references(repository, root, paths=("present.bin", "absent.bin"), **kwargs):
    run = Run(display_name="Saved", source_storage_root_id=root.id, source_relative_directory=".", **kwargs)
    run = replace(run, artifacts=tuple(Artifact(
        id=f"{run.id}-{i}", run_id=run.id, storage_root_id=root.id,
        relative_path=path, display_name=path,
    ) for i, path in enumerate(paths)))
    repository.save_run(run)
    return run


def test_batched_counts_do_not_multiply_dependencies(repository, storage_root, monkeypatch, tmp_path):
    other = StorageRoot(name="Other", location=str(tmp_path / "offline"))
    repository.add_storage_root(other)
    datasets = tuple(Dataset(name=f"Dataset {i}", storage_root_id=storage_root.id, relative_path=f"d{i}")
                     for i in range(3))
    save_references(repository, storage_root, datasets=datasets)
    save_references(repository, storage_root, paths=("third.bin",))
    statements = []
    original = repository.database.connection

    @contextmanager
    def traced():
        with original() as connection:
            connection.set_trace_callback(statements.append)
            yield connection

    monkeypatch.setattr(repository.database, "connection", traced)
    monkeypatch.setattr(repository, "get_run", lambda *a: pytest.fail("No run aggregates"))
    summaries = {s.root.id: s for s in repository.list_storage_root_summaries()}
    assert (summaries[storage_root.id].artifact_count, summaries[storage_root.id].run_count,
            summaries[storage_root.id].dataset_count) == (3, 2, 3)
    assert not summaries[other.id].in_use
    assert sum(s.lstrip().upper().startswith("SELECT") for s in statements) == 1


def test_add_same_path_roots_and_duplicate_identity_contract(repository, tmp_path):
    a = StorageRoot(name="First", location=str(tmp_path) + "/")
    b = StorageRoot(name="Second", location=str(tmp_path))
    repository.add_storage_root(a)
    repository.add_storage_root(b)
    assert repository.get_storage_root(a.id) == a  # Absolute spellings are not canonicalized.
    assert repository.get_storage_root(b.id) == b
    with pytest.raises(sqlite3.IntegrityError):
        repository.add_storage_root(replace(a, name="Conflict"))
    assert repository.get_storage_root(a.id) == a
    with pytest.raises(ValueError):
        StorageRoot(name="Invalid", location="relative")


def test_name_edit_and_remap_preserve_every_reference_and_payload(repository, storage_root, tmp_path):
    a, b = Path(storage_root.location), tmp_path / "b"
    b.mkdir()
    for directory, data in ((a, b"source"), (b, b"destination")):
        (directory / "present.bin").write_bytes(data)
        (directory / "DO_NOT_DELETE.txt").write_bytes(data + b" sentinel")
    run = save_references(repository, storage_root)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for directory in (a, b) for path in directory.iterdir()}
    assert PathResolver().resolve(storage_root, "present.bin").path == a / "present.bin"
    renamed = repository.rename_storage_root(storage_root, "Renamed")
    mapped = repository.remap_storage_root(renamed, location=str(b), kind=renamed.kind)
    assert mapped.id == storage_root.id and mapped.name == "Renamed"
    assert repository.get_run(run.id) == run
    assert PathResolver().resolve(repository.get_storage_root(mapped.id), "present.bin").path == b / "present.bin"
    with pytest.raises(RootInUse):
        repository.remove_storage_root(mapped)
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}
    assert set(a.iterdir()) | set(b.iterdir()) == set(before)


def test_offline_remap_and_portable_mapping_preserve_identity(repository, tmp_path):
    root = StorageRoot(name="Offline", location=str(tmp_path / "old"))
    repository.add_storage_root(root)
    online = repository.remap_storage_root(root, location=str(tmp_path), kind=RootKind.ABSOLUTE)
    assert root_status(PathResolver().resolve(online, ".")) == "Online"
    offline = repository.remap_storage_root(online, location=str(tmp_path / "new"), kind=online.kind)
    assert root_status(PathResolver().resolve(offline, ".")) == "Offline"
    portable = repository.remap_storage_root(offline, location="outputs/./runs/", kind=RootKind.PORTABLE_RELATIVE)
    assert portable.id == root.id and portable.location == "outputs/runs"
    assert root_status(PathResolver().resolve(portable, ".")) == "Unverified"
    (tmp_path / "outputs" / "runs").mkdir(parents=True)
    assert root_status(PathResolver(portable_base=tmp_path).resolve(portable, ".")) == "Online"


def test_stale_dialog_definition_cannot_overwrite_or_remove(repository, storage_root):
    renamed = repository.rename_storage_root(storage_root, "New name")
    for operation in (lambda: repository.rename_storage_root(storage_root, "Stale"),
                      lambda: repository.remap_storage_root(storage_root, location="/offline", kind=storage_root.kind),
                      lambda: repository.remove_storage_root(storage_root),
                      lambda: repository.storage_root_references(storage_root)):
        with pytest.raises(ValueError, match="changed or was removed"):
            operation()
    assert repository.get_storage_root(storage_root.id) == renamed


@pytest.mark.parametrize("dependency", ["artifact", "run", "dataset"])
def test_remove_blocks_each_dependency_without_cascade(repository, storage_root, tmp_path, dependency):
    target = StorageRoot(name="Target", location=str(tmp_path))
    repository.add_storage_root(target)
    if dependency == "run":
        run = save_references(repository, target, paths=())
    elif dependency == "dataset":
        run = save_references(repository, storage_root, paths=(), datasets=(
            Dataset(name="Referenced", storage_root_id=target.id, relative_path="dataset"),))
    else:
        run = Run(display_name="Artifact owner", source_storage_root_id=storage_root.id, source_relative_directory=".")
        run = replace(run, artifacts=(Artifact(run_id=run.id, storage_root_id=target.id,
                                             relative_path="file", display_name="File"),))
        repository.save_run(run)
    with pytest.raises(RootInUse, match="Remap it instead"):
        repository.remove_storage_root(target)
    assert repository.get_storage_root(target.id) == target
    assert repository.get_run(run.id) == run


def test_forget_unused_root_leaves_external_sentinels(repository, storage_root):
    directory = Path(storage_root.location)
    (directory / "checkpoint").mkdir()
    sentinel = directory / "checkpoint" / "artifact.bin"
    sentinel.write_bytes(b"do not delete")
    before = sentinel.stat().st_mtime_ns
    repository.remove_storage_root(storage_root)
    assert repository.get_storage_root(storage_root.id) is None
    assert sentinel.read_bytes() == b"do not delete" and sentinel.stat().st_mtime_ns == before


@pytest.mark.parametrize("operation", ["add", "edit", "remap", "remove"])
def test_mutation_failures_roll_back(repository, storage_root, operation, tmp_path):
    statement = {"add": "INSERT", "edit": "UPDATE", "remap": "UPDATE", "remove": "DELETE"}[operation]
    with repository.database.transaction() as connection:
        connection.execute(f"CREATE TRIGGER fail_root BEFORE {statement} ON storage_root "
                           "BEGIN SELECT RAISE(ABORT, 'injected root failure'); END")
    new_root = StorageRoot(name="New", location=str(tmp_path))
    actions = {
        "add": lambda: repository.add_storage_root(new_root),
        "edit": lambda: repository.rename_storage_root(storage_root, "Changed"),
        "remap": lambda: repository.remap_storage_root(storage_root, location=str(tmp_path), kind=storage_root.kind),
        "remove": lambda: repository.remove_storage_root(storage_root),
    }
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        actions[operation]()
    assert repository.list_storage_roots() == [storage_root]


def test_verify_counts_references_without_payload_reads_hashing_or_crawling(repository, storage_root, monkeypatch):
    directory = Path(storage_root.location)
    (directory / "present.bin").write_bytes(b"payload")
    save_references(repository, storage_root)
    save_references(repository, storage_root, paths=("present.bin",))  # Two catalog refs, one file.
    original = os.scandir
    scanned = []

    class NoEnumeration:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.iterator.close()

        def __iter__(self):
            pytest.fail("No directory enumeration")

    def scan(path):
        assert Path(path) == directory
        scanned.append(path)
        context = NoEnumeration()
        context.iterator = original(path)
        return context

    def forbidden(*a, **kw):
        pytest.fail("No payload reads, hashing or crawling")

    with monkeypatch.context() as patch:
        for owner, name in ((builtins, "open"), (io, "open"), (Path, "open"), (os, "walk"), (hashlib, "sha256")):
            patch.setattr(owner, name, forbidden)
        patch.setattr(os, "scandir", scan)
        result = verify_storage_root(repository, storage_root)
    assert root_status(result.resolution) == "Online"
    assert (result.available, result.missing, result.offline, result.unverified) == (2, 1, 0, 0)
    assert len(scanned) == 4  # Root before/after plus two unique reference paths.
    assert all(a.verification_status.value == "unverified" for a in repository.list_runs()[0].artifacts)


@pytest.mark.parametrize("state", ["online", "offline", "portable"])
def test_verify_zero_and_unavailable_roots(repository, tmp_path, state):
    root = StorageRoot(name=state, location=("outputs" if state == "portable" else
                       str(tmp_path if state == "online" else tmp_path / "offline")),
                       kind=RootKind.PORTABLE_RELATIVE if state == "portable" else RootKind.ABSOLUTE)
    repository.add_storage_root(root)
    zero = verify_storage_root(repository, root)
    assert (zero.available, zero.missing, zero.offline, zero.unverified) == (0, 0, 0, 0)
    save_references(repository, root)
    result = verify_storage_root(repository, root)
    assert root_status(result.resolution) == {"online": "Online", "offline": "Offline", "portable": "Unverified"}[state]
    assert (result.available, result.missing, result.offline, result.unverified) == {
        "online": (0, 2, 0, 0), "offline": (0, 0, 2, 0), "portable": (0, 0, 0, 2),
    }[state]


def test_root_disappears_mid_verification(repository, storage_root, tmp_path):
    save_references(repository, storage_root)
    directory = Path(storage_root.location)
    (directory / "present.bin").write_bytes(b"payload")

    class Disappearing(PathResolver):
        def resolve(self, root, relative_path, **kwargs):
            result = super().resolve(root, relative_path, **kwargs)
            if relative_path == "present.bin":
                directory.rename(tmp_path / "unmounted")  # Test fixture simulates losing the mount.
            return result

    result = verify_storage_root(repository, storage_root, resolver=Disappearing())
    assert root_status(result.resolution) == "Offline"
    assert (result.available, result.missing, result.offline) == (1, 0, 1)
    assert (tmp_path / "unmounted" / "present.bin").read_bytes() == b"payload"
