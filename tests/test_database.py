import sqlite3

import pytest

from runpup.domain.models import Run
from runpup.persistence.database import Database, SchemaVersionError
from runpup.persistence.migrations import MIGRATIONS, SCHEMA_VERSION
from runpup.persistence.repository import CatalogRepository


def test_initialization_is_explicit_idempotent_and_enables_foreign_keys(database):
    database.initialize()
    with database.connection() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 2
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert tables == {"storage_root", "run", "parameter", "metric", "artifact", "dataset",
                          "run_dataset", "tag", "run_tag", "provenance", "import_diagnostic"}
    # Enforcement must apply to new connections, not just the migration connection.
    with database.connection() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_foreign_key_rejects_unknown_root(repository):
    run = Run(display_name="Unknown storage", source_storage_root_id="absent", source_relative_directory="run")
    with pytest.raises(sqlite3.IntegrityError):
        repository.save_run(run)
    assert repository.list_runs() == []


def test_upgrade_old_database_preserves_catalog(tmp_path):
    database = Database(tmp_path / "old.sqlite3")
    database.initialize(target_version=1)
    with database.transaction() as connection:
        connection.execute("INSERT INTO storage_root VALUES (?, ?, ?, ?)", ("root", "Old root", "/offline", "absolute"))
        connection.execute(
            "INSERT INTO run(id, display_name, source_storage_root_id, source_relative_directory, imported_at) "
            "VALUES (?, ?, ?, ?, ?)", ("old", "Historical", "root", "old", "2024-01-01T00:00:00+00:00"),
        )
    database.initialize()
    run = CatalogRepository(database).get_run("old")
    assert run.display_name == "Historical"
    assert run.training_started_at is None
    assert run.diagnostics == ()
    with database.connection() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2


def test_failed_migration_rolls_back_schema_and_version(tmp_path, monkeypatch):
    database = Database(tmp_path / "old.sqlite3")
    database.initialize(target_version=1)
    monkeypatch.setitem(MIGRATIONS, 2, ("CREATE TABLE migration_probe (id TEXT)", "INVALID SQL"))
    with pytest.raises(sqlite3.OperationalError):
        database.initialize()
    with database.connection() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("SELECT name FROM sqlite_master WHERE name = 'migration_probe'").fetchone() is None


def test_newer_schema_and_downgrade_are_rejected(database):
    with pytest.raises(SchemaVersionError, match="refusing downgrade"):
        database.initialize(target_version=1)
    with database.transaction() as connection:
        connection.execute("PRAGMA user_version = 99")
    with pytest.raises(SchemaVersionError, match="99"):
        database.initialize()
    with database.connection() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 99


@pytest.mark.parametrize("existing_table", ["run", "unrelated"])
def test_unknown_unversioned_schema_is_not_silently_adopted(tmp_path, existing_table):
    database = Database(tmp_path / "unrelated.sqlite3")
    with database.transaction() as connection:
        connection.execute(f"CREATE TABLE {existing_table} (original TEXT)")
        connection.execute(f"INSERT INTO {existing_table} VALUES ('preserve me')")
    with pytest.raises(SchemaVersionError, match="unversioned"):
        database.initialize()
    with database.connection() as connection:
        assert connection.execute(f"SELECT * FROM {existing_table}").fetchone()[0] == "preserve me"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("SELECT name FROM sqlite_master WHERE name = 'storage_root'").fetchone() is None
