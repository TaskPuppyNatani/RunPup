import builtins
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sqlite3
import threading
import zipfile

import pytest

from runpup.persistence.database import Database
from runpup.persistence.repository import CatalogRepository
from runpup.services.backup import BackupError, create_backup
from test_run_details import saved_run


PREFERENCES = {"import/default_directory": "/configured/outputs", "import/remember_last": True}


def unpack(result, tmp_path):
    with zipfile.ZipFile(result.path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {"runpup-backup.json", "catalog.sqlite3", "preferences.json"}
        manifest = json.loads(archive.read("runpup-backup.json"))
        preferences = json.loads(archive.read("preferences.json"))
        snapshot = tmp_path / "unpacked.sqlite3"
        snapshot.write_bytes(archive.read("catalog.sqlite3"))
    assert Database(snapshot).validate_existing() == 3
    return manifest, preferences, CatalogRepository(Database(snapshot))


def test_backup_preserves_full_aggregate_and_excludes_external_payloads(repository, saved_run, storage_root, tmp_path, monkeypatch):
    directory = Path(storage_root.location)
    (directory / "run").mkdir()
    sentinel = directory / "run" / "training_args.bin"
    sentinel.write_bytes(b"EXTERNAL_SENTINEL_MUST_NOT_BE_ARCHIVED")
    before = sentinel.read_bytes(), sentinel.stat().st_mtime_ns
    catalog_before = repository.database.path.read_bytes()
    original_open, original_io_open = builtins.open, io.open
    original_scandir = os.scandir

    def guarded_open(original):
        def open_file(path, *args, **kwargs):
            if isinstance(path, (str, Path)):
                assert not Path(path).is_relative_to(directory), "No external payload read"
            return original(path, *args, **kwargs)
        return open_file

    def scan(path):
        if isinstance(path, (str, Path)):
            assert not Path(path).is_relative_to(directory), "No external traversal"
        return original_scandir(path)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", guarded_open(original_open))
        patch.setattr(io, "open", guarded_open(original_io_open))
        patch.setattr(os, "scandir", scan)
        result = create_backup(repository.database, tmp_path / "backup.zip", PREFERENCES)
    manifest, preferences, copied = unpack(result, tmp_path)
    assert manifest["backup_format_version"] == 1 and manifest["schema_version"] == 3
    assert manifest["external_artifacts_included"] is False and manifest["created_at"] == result.created_at
    assert preferences == PREFERENCES
    assert copied.get_run(saved_run.id) == saved_run
    assert copied.get_storage_root(storage_root.id) == storage_root
    assert before == (sentinel.read_bytes(), sentinel.stat().st_mtime_ns)
    assert repository.database.path.read_bytes() == catalog_before
    with zipfile.ZipFile(result.path) as archive:
        assert all(before[0] not in archive.read(name) for name in archive.namelist())
    assert not list(tmp_path.glob(".runpup-backup-*"))


def test_empty_catalog_backup_is_valid(database, tmp_path):
    result = create_backup(database, tmp_path / "empty.zip", PREFERENCES)
    _, _, copied = unpack(result, tmp_path)
    assert copied.list_runs() == []


def test_live_wal_backup_includes_committed_data_and_excludes_open_transaction(repository, saved_run, tmp_path):
    with repository.database.connection() as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("UPDATE run SET notes = 'uncommitted' WHERE id = ?", (saved_run.id,))
        try:
            result = create_backup(repository.database, tmp_path / "live.zip", PREFERENCES)
            _, _, copied = unpack(result, tmp_path)
            assert copied.get_run(saved_run.id) == saved_run
        finally:
            writer.rollback()
    assert repository.get_run(saved_run.id) == saved_run


def test_backup_while_atomic_aggregate_changes_remains_consistent(repository, saved_run, tmp_path):
    stop, started = threading.Event(), threading.Event()
    errors = []

    def writer():
        try:
            i = 0
            while not stop.is_set():
                value = str(i % 2)
                repository.save_run(replace(saved_run, notes=value,
                    parameters=tuple(replace(p, value=value) for p in saved_run.parameters)))
                started.set()
                i += 1
                stop.wait(.001)
        except BaseException as error:
            errors.append(error)

    with repository.database.connection() as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    thread = threading.Thread(target=writer)
    thread.start()
    try:
        assert started.wait(3)
        result = create_backup(repository.database, tmp_path / "activity.zip", PREFERENCES)
    finally:
        stop.set()
        thread.join(3)
    assert not thread.is_alive() and not errors
    _, _, copied = unpack(result, tmp_path)
    run = copied.get_run(saved_run.id)
    assert run.notes in {"0", "1"} and all(p.value == run.notes for p in run.parameters)
    with copied.database.read_connection() as connection:
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()


@pytest.mark.parametrize("failure", ["unwritable", "archive", "validation", "finalization"])
def test_failed_backup_preserves_source_destination_and_cleans_staging(database, tmp_path, monkeypatch, failure):
    import runpup.services.backup as module
    destination = tmp_path / "backup.zip"
    destination.write_bytes(b"existing backup must survive")
    before = database.path.read_bytes()

    def fail(*args, **kwargs):
        raise PermissionError("injected failure")

    if failure == "unwritable":
        monkeypatch.setattr(module.tempfile, "TemporaryDirectory", fail)
    elif failure == "archive":
        monkeypatch.setattr(zipfile.ZipFile, "write", fail)
    elif failure == "validation":
        monkeypatch.setattr(zipfile.ZipFile, "testzip", lambda self: "catalog.sqlite3")
    else:
        monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(BackupError, match="source catalog was not changed"):
        create_backup(database, destination, PREFERENCES, overwrite=True)
    assert destination.read_bytes() == b"existing backup must survive" and database.path.read_bytes() == before
    assert not list(tmp_path.glob(".runpup-backup-*"))


def test_destination_conflict_and_confirmed_replacement(database, tmp_path):
    destination = tmp_path / "existing.zip"
    destination.write_bytes(b"old backup")
    with pytest.raises(BackupError, match="already exists"):
        create_backup(database, destination, PREFERENCES)
    assert destination.read_bytes() == b"old backup"
    result = create_backup(database, destination, PREFERENCES, overwrite=True)
    assert unpack(result, tmp_path)[0]["application"] == "RunPup"


def test_destination_appearing_during_backup_is_not_overwritten(database, tmp_path, monkeypatch):
    original = os.link
    destination = tmp_path / "race.zip"

    def racing(source, target):
        Path(target).write_bytes(b"appeared during backup")
        original(source, target)

    monkeypatch.setattr(os, "link", racing)
    with pytest.raises(BackupError):
        create_backup(database, destination, PREFERENCES)
    assert destination.read_bytes() == b"appeared during backup"
    assert not list(tmp_path.glob(".runpup-backup-*"))


def test_backup_rejects_active_catalog_alias_and_unlisted_settings(database, tmp_path):
    alias = tmp_path / "alias.zip"
    alias.symlink_to(database.path)
    before = database.path.read_bytes()
    with pytest.raises(BackupError, match="active catalog"):
        create_backup(database, alias, PREFERENCES, overwrite=True)
    with pytest.raises(BackupError, match="supported Import preferences"):
        create_backup(database, tmp_path / "secrets.zip", {**PREFERENCES, "api_token": "do not export"})
    assert database.path.read_bytes() == before and not (tmp_path / "secrets.zip").exists()
