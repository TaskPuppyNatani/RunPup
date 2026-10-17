"""SQLite connection and transaction ownership, including schema upgrades."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator

from runpup.persistence.migrations import MIGRATIONS, SCHEMA_VERSION


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
            if current == 0 and connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name NOT GLOB 'sqlite_*' LIMIT 1"
            ).fetchone() is not None:
                raise SchemaVersionError("Refusing to initialize an unrecognized, unversioned database")
            if current > target_version:
                raise SchemaVersionError(
                    f"Catalog schema {current} is newer than requested {target_version}; refusing downgrade"
                )
            for version in range(current + 1, target_version + 1):
                for statement in MIGRATIONS[version]:
                    connection.execute(statement)
                connection.execute(f"PRAGMA user_version = {version}")
