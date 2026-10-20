from pathlib import Path
import sqlite3

import pytest
from PySide6.QtCore import QSettings

from runpup.domain.models import Run, StorageRoot
from runpup.persistence.database import APPLICATION_ID, Database, SchemaVersionError
from runpup.persistence.repository import CatalogRepository
from runpup.services.data_location import (
    open_startup_catalog, prepare_catalog, resolve_data_location, validate_data_directory,
)
from runpup.ui.preferences import ImportPreferences


@pytest.fixture
def preferences(tmp_path):
    return ImportPreferences(QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat), home=tmp_path)


def test_resolution_precedence_and_override_is_not_persisted(preferences, tmp_path):
    default, custom, override = (tmp_path / name for name in ("default", "custom", "override"))
    assert resolve_data_location(default).directory == default
    preferences.configured_data_directory = custom
    assert resolve_data_location(default, preferences.configured_data_directory).directory == custom
    location, database = open_startup_catalog(default, preferences.configured_data_directory, override=override)
    assert location.source == "Command-line override" and database.path == override / "runpup.sqlite3"
    assert preferences.configured_data_directory == str(custom)
    assert resolve_data_location(default, preferences.configured_data_directory).directory == custom


def test_next_launch_and_reset_preserve_previous_catalog(preferences, tmp_path):
    default, custom = tmp_path / "default", tmp_path / "custom"
    first_location, first_database = open_startup_catalog(default)
    repository = CatalogRepository(first_database)
    root = StorageRoot(name="External", location=str(tmp_path / "external"))
    repository.save_storage_root(root)
    run = Run(display_name="Keep", source_storage_root_id=root.id, source_relative_directory=".")
    repository.save_run(run)
    before = first_database.path.read_bytes()
    assert "New catalog" in validate_data_directory(custom)
    assert not custom.exists()
    preferences.configured_data_directory = custom
    assert first_database.path == first_location.database_path
    assert not custom.exists()
    second_location, second_database = open_startup_catalog(default, preferences.configured_data_directory)
    assert second_location.directory == custom and CatalogRepository(second_database).list_runs() == []
    assert first_database.path.read_bytes() == before and repository.get_run(run.id) == run
    custom_before = second_database.path.read_bytes()
    preferences.configured_data_directory = None
    restored, _ = open_startup_catalog(default, preferences.configured_data_directory)
    assert restored.directory == default
    assert second_database.path.read_bytes() == custom_before


@pytest.mark.parametrize("version", [1, 2, 3])
def test_read_only_selection_validates_legacy_then_normal_startup_migrates(tmp_path, version):
    location = resolve_data_location(tmp_path / "data")
    database = Database(location.database_path)
    database.initialize(target_version=version)
    if version < 3:
        with database.transaction() as connection:
            connection.execute("PRAGMA application_id = 0")
    before = database.path.read_bytes()
    assert database.validate_existing() == version
    assert f"schema {version}" in validate_data_directory(location.directory)
    assert database.path.read_bytes() == before
    prepare_catalog(location)
    assert database.validate_existing() == 3
    with database.read_connection() as connection:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM storage_root")


@pytest.mark.parametrize("kind", ["foreign", "unversioned", "future", "wrong_id", "empty", "garbage"])
def test_invalid_catalog_selection_and_startup_never_adopt_or_overwrite(tmp_path, kind):
    directory = tmp_path / kind
    directory.mkdir()
    path = directory / "runpup.sqlite3"
    if kind == "garbage":
        path.write_bytes(b"not SQLite")
    elif kind == "empty":
        path.touch()
    elif kind in {"future", "wrong_id"}:
        database = Database(path)
        database.initialize()
        with database.transaction() as connection:
            connection.execute("PRAGMA user_version = 99" if kind == "future" else "PRAGMA application_id = 42")
    else:
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE unrelated(value TEXT)")
            connection.execute("INSERT INTO unrelated VALUES ('preserve')")
            connection.execute(f"PRAGMA user_version = {1 if kind == 'foreign' else 0}")
    before = path.read_bytes()
    for operation in (lambda: validate_data_directory(directory),
                      lambda: open_startup_catalog(tmp_path / "default", str(directory))):
        with pytest.raises((ValueError, OSError, sqlite3.Error, SchemaVersionError)):
            operation()
    assert path.read_bytes() == before and list(directory.iterdir()) == [path]


def test_unrelated_directory_and_non_directory_are_rejected(tmp_path):
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    sentinel = unrelated / "keep.txt"
    sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="not empty"):
        validate_data_directory(unrelated)
    with pytest.raises(ValueError, match="not a directory"):
        validate_data_directory(sentinel)
    assert sentinel.read_text() == "preserve" and not (unrelated / "runpup.sqlite3").exists()


@pytest.mark.parametrize("configured", [False, 42, ["invalid"], "relative/path", "bad\0path"])
def test_bad_bootstrap_has_explicit_launch_only_recovery(tmp_path, configured):
    calls = []
    location, database = open_startup_catalog(tmp_path / "default", configured,
                                              recover=lambda error: calls.append(str(error)) or True)
    assert calls and location.source == "Default for this launch"
    assert database.validate_existing() == 3
    with pytest.raises(ValueError):
        open_startup_catalog(tmp_path / "default", configured, recover=lambda error: False)
    # A valid CLI override can recover without reading/rewriting the malformed value.
    location, _ = open_startup_catalog(tmp_path / "default", configured, override=tmp_path / "override")
    assert location.source == "Command-line override"


def test_inaccessible_configured_directory_recovers_but_bad_cli_does_not(tmp_path, monkeypatch):
    import runpup.services.data_location as module
    original = module.validate_data_directory
    inaccessible = tmp_path / "denied"

    def validate(path):
        if Path(path) == inaccessible:
            raise PermissionError("mount unavailable")
        return original(path)

    monkeypatch.setattr(module, "validate_data_directory", validate)
    location, _ = open_startup_catalog(tmp_path / "default", str(inaccessible), recover=lambda error: True)
    assert location.source == "Default for this launch"
    with pytest.raises(PermissionError):
        open_startup_catalog(tmp_path / "default", override=inaccessible,
                             recover=lambda error: pytest.fail("CLI override must fail explicitly"))
    assert not inaccessible.exists()


def test_preference_write_failure_rolls_back_unrelated_keys_and_retry_works(tmp_path):
    blocker = tmp_path / "blocking-file"
    blocker.write_text("not a directory")
    preferences = ImportPreferences(QSettings(str(blocker / "prefs.ini"), QSettings.Format.IniFormat))
    with pytest.raises(OSError, match="previous values"):
        preferences.configured_data_directory = tmp_path / "custom"
    assert preferences.configured_data_directory is None
    blocker.unlink()  # Test fixture repairs configuration access.
    blocker.mkdir()
    preferences.default_directory = "preserved import preference"
    preferences.configured_data_directory = tmp_path / "custom"
    reloaded = ImportPreferences(QSettings(preferences.settings.fileName(), QSettings.Format.IniFormat))
    assert reloaded.configured_data_directory == str(tmp_path / "custom")
    assert reloaded.default_directory == "preserved import preference"


def test_preferences_export_is_allowlisted(preferences, tmp_path):
    preferences.default_directory = tmp_path / "outputs"
    preferences.remember_last = False
    preferences.configured_data_directory = tmp_path / "data"
    preferences.settings.setValue("adjacent/api_token", "must never export")
    preferences.settings.setValue("import/last_parent", str(tmp_path / "session"))
    assert preferences.portable_export() == {
        "import/default_directory": str(tmp_path / "outputs"), "import/remember_last": False,
    }


def test_removed_configured_directory_initializes_new_catalog_without_silent_fallback(preferences, tmp_path):
    directory = tmp_path / "custom"
    _, database = open_startup_catalog(tmp_path / "default", str(directory))
    before = database.path.read_bytes()
    directory.rename(tmp_path / "preserved-old-location")  # Fixture simulates a removed/moved location.
    preferences.configured_data_directory = directory
    location, new_database = open_startup_catalog(tmp_path / "default", preferences.configured_data_directory)
    assert location.source == "Configured" and location.directory == directory
    assert new_database.validate_existing() == 3
    assert (tmp_path / "preserved-old-location" / "runpup.sqlite3").read_bytes() == before
    assert not (tmp_path / "default").exists()
