import sqlite3

import pytest

from runpup.domain.models import Metric, MetricSummary, Run, StorageRoot
from runpup.persistence.database import APPLICATION_ID, Database, SchemaVersionError
from runpup.persistence.migrations import MIGRATIONS, SCHEMA_VERSION
from runpup.persistence.repository import CatalogRepository


def test_initialization_is_explicit_idempotent_and_enables_foreign_keys(database):
    database.initialize()
    with database.connection() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 3
        assert connection.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
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
        # The sealed v1 baseline predates the application_id ownership marker.
        connection.execute("PRAGMA application_id = 0")
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
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert connection.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID


def test_upgrade_v2_adds_metric_unit_to_identity(tmp_path):
    database = Database(tmp_path / "version-two.sqlite3")
    database.initialize(target_version=2)
    with database.transaction() as connection:
        # Simulate the sealed v2 schema, which predates the ownership marker.
        connection.execute("PRAGMA application_id = 0")

    database.initialize()
    repository = CatalogRepository(database)
    root = StorageRoot(name="Root", location=str(tmp_path))
    repository.save_storage_root(root)
    metrics = (
        Metric(name="throughput", context="train", summary=MetricSummary.FINAL,
               value=120, unit="tokens/second"),
        Metric(name="throughput", context="train", summary=MetricSummary.FINAL,
               value=2, unit="sequences/second"),
    )
    run = Run(display_name="Units", source_storage_root_id=root.id,
              source_relative_directory="run", metrics=metrics)

    repository.save_run(run)

    assert repository.get_run(run.id).metrics == metrics


def test_failed_migration_rolls_back_schema_and_version(tmp_path, monkeypatch):
    database = Database(tmp_path / "old.sqlite3")
    database.initialize(target_version=1)
    monkeypatch.setitem(MIGRATIONS, 2, ("CREATE TABLE migration_probe (id TEXT)", "INVALID SQL"))
    with pytest.raises(sqlite3.OperationalError):
        database.initialize()
    with database.connection() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("SELECT name FROM sqlite_master WHERE name = 'migration_probe'").fetchone() is None


def test_failed_metric_identity_migration_restores_original_index(tmp_path, monkeypatch):
    database = Database(tmp_path / "failed-metric-migration.sqlite3")
    database.initialize(target_version=2)
    monkeypatch.setitem(MIGRATIONS, 3, ("DROP INDEX metric_identity", "INVALID SQL"))

    with pytest.raises(sqlite3.OperationalError):
        database.initialize()

    with database.connection() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        index_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'metric_identity'"
        ).fetchone()[0]
        assert "unit" not in index_sql


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


def test_unknown_unversioned_view_is_not_silently_adopted(tmp_path):
    database = Database(tmp_path / "unrelated-view.sqlite3")
    with database.transaction() as connection:
        connection.execute("CREATE VIEW user_view AS SELECT 'preserve me' AS value")

    with pytest.raises(SchemaVersionError, match="unversioned"):
        database.initialize()

    with database.connection() as connection:
        assert connection.execute("SELECT value FROM user_view").fetchone()[0] == "preserve me"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        assert connection.execute("SELECT name FROM sqlite_master WHERE name = 'storage_root'").fetchone() is None


def test_unknown_versioned_schema_is_not_silently_upgraded(tmp_path):
    database = Database(tmp_path / "unrelated-versioned.sqlite3")
    with database.transaction() as connection:
        connection.execute("CREATE TABLE user_data (value TEXT NOT NULL)")
        connection.execute("INSERT INTO user_data VALUES ('preserve me')")
        connection.execute("PRAGMA user_version = 1")

    with pytest.raises(SchemaVersionError, match="unrecognized"):
        database.initialize()

    with database.connection() as connection:
        assert connection.execute("SELECT value FROM user_data").fetchone()[0] == "preserve me"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0
        assert connection.execute("SELECT name FROM sqlite_master WHERE name = 'provenance'").fetchone() is None


@pytest.mark.parametrize("version", [1, 2, 3])
def test_owned_catalog_missing_required_column_is_rejected_without_mutation(tmp_path, version):
    database = Database(tmp_path / "damaged.sqlite3")
    database.initialize(target_version=version)
    with database.transaction() as connection:
        connection.execute("ALTER TABLE run DROP COLUMN notes")
    before = database.path.read_bytes()
    for operation in (database.validate_existing, database.initialize):
        with pytest.raises(SchemaVersionError, match="schema"):
            operation()
        assert database.path.read_bytes() == before


@pytest.mark.parametrize("damage", ["missing_tables", "missing_index", "wrong_metric_index", "wrong_foreign_key"])
def test_catalog_markers_do_not_substitute_for_schema_integrity(tmp_path, damage):
    database = Database(tmp_path / "marked.sqlite3")
    with database.transaction() as connection:
        if damage == "missing_tables":
            connection.execute("CREATE TABLE unrelated(value TEXT)")
        else:
            for version in range(1, 4):
                for statement in MIGRATIONS[version]:
                    if damage == "wrong_foreign_key" and statement.startswith("CREATE TABLE run ("):
                        statement = statement.replace("ON DELETE RESTRICT", "ON DELETE CASCADE")
                    connection.execute(statement)
            if damage in {"missing_index", "wrong_metric_index"}:
                connection.execute("DROP INDEX metric_identity")
            if damage == "wrong_metric_index":
                connection.execute("CREATE UNIQUE INDEX metric_identity ON metric(run_id, name)")
        connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        connection.execute("PRAGMA user_version = 3")
    before = database.path.read_bytes()
    for operation in (database.validate_existing, database.initialize):
        with pytest.raises(SchemaVersionError, match="schema"):
            operation()
        assert database.path.read_bytes() == before
