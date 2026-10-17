"""SQLite connection and transaction ownership, including schema upgrades."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator

from runpup.persistence.migrations import MIGRATIONS, SCHEMA_VERSION


# SQLite reserves this 32-bit field for identifying the application that owns a file.
APPLICATION_ID = 0x52555050  # "RUPP"

# Schema signatures for catalogs created before application_id was introduced.
# Checking required columns lets existing v1/v2 RunPup catalogs upgrade while
# refusing to treat an unrelated database's user_version as proof of ownership.
_LEGACY_BASE_COLUMNS = {
    "storage_root": {"id", "name", "location", "kind"},
    "run": {"id", "display_name", "source_storage_root_id", "source_relative_directory",
            "imported_at", "training_status"},
    "parameter": {"id", "run_id", "position", "key", "value_json", "value_type", "category"},
    "metric": {"id", "run_id", "position", "name", "context", "summary", "value", "step", "unit"},
    "artifact": {"id", "run_id", "position", "storage_root_id", "relative_path", "display_name"},
    "dataset": {"id", "name", "storage_root_id", "relative_path"},
    "run_dataset": {"run_id", "dataset_id", "position"},
    "tag": {"id", "name"},
    "run_tag": {"run_id", "tag_id", "position"},
}
_LEGACY_REQUIRED_COLUMNS = {
    1: _LEGACY_BASE_COLUMNS,
    2: {
        **_LEGACY_BASE_COLUMNS,
        "provenance": {"id", "run_id", "position", "target_entity", "target_field", "source_type",
                       "original_json", "candidate_json", "current_json", "confidence", "user_modified"},
        "import_diagnostic": {"id", "run_id", "position", "severity", "kind", "message", "detected_at"},
    },
}


class SchemaVersionError(RuntimeError):
    """The catalog cannot be safely opened at the requested schema version."""


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, isolation_level=None, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self, *, write: bool = True) -> Iterator[sqlite3.Connection]:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def initialize(self, *, target_version: int = SCHEMA_VERSION) -> None:
        """Upgrade atomically. Never downgrade, drop tables, or adopt unknown schemas."""
        if not 1 <= target_version <= SCHEMA_VERSION:
            raise SchemaVersionError(f"Unsupported target schema version: {target_version}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as connection:
            current = connection.execute("PRAGMA user_version").fetchone()[0]
            application_id = connection.execute("PRAGMA application_id").fetchone()[0]
            if current == 0:
                has_schema = connection.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type IN ('table', 'index', 'view', 'trigger') "
                    "AND name NOT GLOB 'sqlite_*' LIMIT 1"
                ).fetchone() is not None
                if application_id != 0 or has_schema:
                    raise SchemaVersionError("Refusing to initialize an unrecognized, unversioned database")
                connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            elif application_id not in (0, APPLICATION_ID):
                raise SchemaVersionError("Database belongs to a different application")
            elif application_id == 0:
                if not self._matches_legacy_schema(connection, current):
                    raise SchemaVersionError(
                        f"Refusing to upgrade an unrecognized database at schema version {current}"
                    )
                connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            if current > target_version:
                raise SchemaVersionError(
                    f"Catalog schema {current} is newer than requested {target_version}; refusing downgrade"
                )
            for version in range(current + 1, target_version + 1):
                for statement in MIGRATIONS[version]:
                    connection.execute(statement)
                connection.execute(f"PRAGMA user_version = {version}")

    @staticmethod
    def _matches_legacy_schema(connection: sqlite3.Connection, version: int) -> bool:
        required_tables = _LEGACY_REQUIRED_COLUMNS.get(version)
        if required_tables is None:
            return False
        tables = {
            row["name"] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        for table, required_columns in required_tables.items():
            if table not in tables:
                return False
            columns = {
                row["name"] for row in connection.execute(f'PRAGMA table_info("{table}")')
            }
            if not required_columns <= columns:
                return False
        return True
