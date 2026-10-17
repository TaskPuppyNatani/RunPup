"""The catalog's SQL boundary. Each operation owns a complete transaction."""

from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
import json
import sqlite3
from typing import Any

from runpup.domain.models import (
    Artifact, Dataset, EntityType, ImportDiagnostic, Metric, Parameter,
    Provenance, Run, StorageRoot, Tag, validate_human_metadata,
)
from runpup.persistence.database import Database


_RUN_COLUMNS = (
    "id", "display_name", "source_storage_root_id", "source_relative_directory",
    "source_trainer", "imported_at", "training_started_at", "training_finished_at",
    "training_status", "base_model", "model_revision", "method", "notes", "rating", "favorite",
)
_TARGET_COLUMNS = {
    EntityType.PARAMETER: "parameter_id", EntityType.METRIC: "metric_id",
    EntityType.ARTIFACT: "artifact_id", EntityType.DATASET: "dataset_id",
}


def _db_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat(timespec="microseconds")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bool):
        return int(value)
    return value


def _timestamp(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _insert(connection: sqlite3.Connection, table: str, data: dict[str, Any], *, upsert: bool = False) -> None:
    # Table/column names come only from this module, never from user input.
    columns = tuple(data)
    statement = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})"
    if upsert:
        statement += " ON CONFLICT(id) DO UPDATE SET " + ", ".join(
            f"{column} = excluded.{column}" for column in columns if column != "id"
        )
    connection.execute(statement, tuple(_db_value(data[column]) for column in columns))


def _save_shared(connection: sqlite3.Connection, table: str, data: dict[str, Any]) -> None:
    existing = connection.execute(f"SELECT * FROM {table} WHERE id = ?", (data["id"],)).fetchone()
    if existing is None:
        _insert(connection, table, data)
    elif dict(existing) != {key: _db_value(value) for key, value in data.items()}:
        raise ValueError(f"Conflicting definition for shared {table} {data['id']}")


class CatalogRepository:
    def __init__(self, database: Database):
        self.database = database

    def save_storage_root(self, root: StorageRoot) -> None:
        with self.database.transaction() as connection:
            _insert(connection, "storage_root", asdict(root), upsert=True)

    def get_storage_root(self, root_id: str) -> StorageRoot | None:
        with self.database.transaction(write=False) as connection:
            row = connection.execute("SELECT * FROM storage_root WHERE id = ?", (root_id,)).fetchone()
            return StorageRoot(**dict(row)) if row is not None else None

    def list_storage_roots(self) -> list[StorageRoot]:
        with self.database.transaction(write=False) as connection:
            return [StorageRoot(**dict(row)) for row in connection.execute(
                "SELECT * FROM storage_root ORDER BY name, id"
            )]

    def save_run(self, run: Run) -> None:
        """Save a full aggregate; a failure restores its previous catalog state."""
        with self.database.transaction() as connection:
            self._save_run(connection, run)

    def save_import(self, run: Run, root: StorageRoot) -> Run:
        """Register an unchanged root, save the aggregate, and reload atomically."""
        if run.source_storage_root_id != root.id:
            raise ValueError("Import root must be the run's source storage root")
        with self.database.transaction() as connection:
            _save_shared(connection, "storage_root", asdict(root))
            self._save_run(connection, run)
            row = connection.execute("SELECT * FROM run WHERE id = ?", (run.id,)).fetchone()
            return _load_run(connection, row)

    def get_dataset(self, dataset_id: str) -> Dataset | None:
        with self.database.transaction(write=False) as connection:
            row = connection.execute("SELECT * FROM dataset WHERE id = ?", (dataset_id,)).fetchone()
            return Dataset(**dict(row)) if row is not None else None

    @staticmethod
    def _save_run(connection: sqlite3.Connection, run: Run) -> None:
        _insert(connection, "run", {key: getattr(run, key) for key in _RUN_COLUMNS}, upsert=True)
        # Remove dependent rows first. No operation here accesses external files.
        for table in ("provenance", "import_diagnostic", "run_dataset", "run_tag",
                      "parameter", "metric", "artifact"):
            connection.execute(f"DELETE FROM {table} WHERE run_id = ?", (run.id,))

        for position, parameter in enumerate(run.parameters):
            _insert(connection, "parameter", {
                "id": parameter.id, "run_id": run.id, "position": position,
                "key": parameter.key, "value_json": _json(parameter.value),
                "value_type": parameter.value_type, "category": parameter.category,
            })
        for position, metric in enumerate(run.metrics):
            _insert(connection, "metric", {**asdict(metric), "run_id": run.id, "position": position})
        for position, artifact in enumerate(run.artifacts):
            _insert(connection, "artifact", {**asdict(artifact), "position": position})
        for position, dataset in enumerate(run.datasets):
            _save_shared(connection, "dataset", asdict(dataset))
            _insert(connection, "run_dataset", {
                "run_id": run.id, "dataset_id": dataset.id, "position": position,
            })
        for position, tag in enumerate(run.tags):
            _save_shared(connection, "tag", asdict(tag))
            _insert(connection, "run_tag", {"run_id": run.id, "tag_id": tag.id, "position": position})

        for position, provenance in enumerate(run.provenance):
            targets = {column: None for column in _TARGET_COLUMNS.values()}
            if provenance.target_entity is not EntityType.RUN:
                targets[_TARGET_COLUMNS[provenance.target_entity]] = provenance.target_id
            _insert(connection, "provenance", {
                "id": provenance.id, "run_id": run.id, "position": position,
                "target_entity": provenance.target_entity, "target_field": provenance.target_field,
                **targets, "source_type": provenance.source_type,
                "source_reference": provenance.source_reference,
                "original_json": _json(provenance.original_value),
                "candidate_json": _json(provenance.candidate_value),
                "current_json": _json(provenance.current_value),
                "confidence": provenance.confidence, "user_modified": provenance.user_modified,
                "detected_at": provenance.detected_at,
            })
        for position, diagnostic in enumerate(run.diagnostics):
            _insert(connection, "import_diagnostic", {
                **asdict(diagnostic), "run_id": run.id, "position": position,
            })

    def get_run(self, run_id: str) -> Run | None:
        with self.database.transaction(write=False) as connection:
            row = connection.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
            return _load_run(connection, row) if row is not None else None

    def list_runs(self) -> list[Run]:
        with self.database.transaction(write=False) as connection:
            rows = connection.execute("SELECT * FROM run ORDER BY imported_at DESC, id").fetchall()
            return [_load_run(connection, row) for row in rows]

    def update_human_metadata(self, run_id: str, *, notes: str, rating: int | None, favorite: bool) -> None:
        validate_human_metadata(notes, rating, favorite)
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE run SET notes = ?, rating = ?, favorite = ? WHERE id = ?",
                (notes, rating, int(favorite), run_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(run_id)

    def delete_run(self, run_id: str) -> bool:
        """Delete catalog ownership/links only; external payloads are never touched."""
        with self.database.transaction() as connection:
            return connection.execute("DELETE FROM run WHERE id = ?", (run_id,)).rowcount > 0


def _children(connection: sqlite3.Connection, table: str, run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(f"SELECT * FROM {table} WHERE run_id = ? ORDER BY position", (run_id,))
    return [{key: row[key] for key in row.keys() if key not in ("run_id", "position")} for row in rows]


def _load_run(connection: sqlite3.Connection, row: sqlite3.Row) -> Run:
    run_id = row["id"]
    data = dict(row)
    for key in ("imported_at", "training_started_at", "training_finished_at"):
        data[key] = _timestamp(data[key])
    data["favorite"] = bool(data["favorite"])
    parameters = []
    for item in _children(connection, "parameter", run_id):
        value_type = item.pop("value_type")
        item["value"] = json.loads(item.pop("value_json"))
        parameter = Parameter(**item)
        if parameter.value_type != value_type:
            raise ValueError(f"Stored parameter type mismatch: {parameter.key}")
        parameters.append(parameter)
    artifacts = []
    for item in _children(connection, "artifact", run_id):
        item["verified_at"] = _timestamp(item["verified_at"])
        artifacts.append(Artifact(run_id=run_id, **item))
    provenance = []
    for item in _children(connection, "provenance", run_id):
        entity = EntityType(item["target_entity"])
        item["target_id"] = run_id if entity is EntityType.RUN else item[_TARGET_COLUMNS[entity]]
        for key in _TARGET_COLUMNS.values():
            item.pop(key)
        item["original_value"] = json.loads(item.pop("original_json"))
        item["candidate_value"] = json.loads(item.pop("candidate_json"))
        item["current_value"] = json.loads(item.pop("current_json"))
        item["user_modified"] = bool(item["user_modified"])
        item["detected_at"] = _timestamp(item["detected_at"])
        provenance.append(Provenance(**item))
    diagnostics = []
    for item in _children(connection, "import_diagnostic", run_id):
        item["detected_at"] = _timestamp(item["detected_at"])
        diagnostics.append(ImportDiagnostic(**item))
    datasets = tuple(Dataset(**dict(item)) for item in connection.execute(
        "SELECT d.* FROM dataset d JOIN run_dataset rd ON rd.dataset_id = d.id "
        "WHERE rd.run_id = ? ORDER BY rd.position", (run_id,),
    ))
    tags = tuple(Tag(**dict(item)) for item in connection.execute(
        "SELECT t.* FROM tag t JOIN run_tag rt ON rt.tag_id = t.id "
        "WHERE rt.run_id = ? ORDER BY rt.position", (run_id,),
    ))
    return Run(**data, parameters=tuple(parameters),
               metrics=tuple(Metric(**item) for item in _children(connection, "metric", run_id)),
               artifacts=tuple(artifacts), datasets=datasets, tags=tags,
               provenance=tuple(provenance), diagnostics=tuple(diagnostics))
