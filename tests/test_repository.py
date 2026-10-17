from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import pytest

from runpup.domain.models import (
    Artifact, Confidence, Dataset, EntityType, ImportDiagnostic, Metric,
    MetricSummary, Parameter, Provenance, Run, Severity, StorageRoot, Tag,
    TrainingStatus, ValueType, VerificationStatus, new_id,
)
from runpup.persistence.repository import CatalogRepository
from runpup.services.paths import PathResolver


def make_run(root, **changes):
    return Run(**{"display_name": "A useful run", "source_storage_root_id": root.id,
                  "source_relative_directory": "runs/one", **changes})


def test_complete_aggregate_round_trip(repository, storage_root):
    run_id = new_id()
    parameter = Parameter(key="rank", value=16, category="adapter")
    metrics = (
        Metric(name="loss", context="train", summary=MetricSummary.FINAL, value=0.8, step=100, unit="nats"),
        Metric(name="loss", context="validation", summary=MetricSummary.MINIMUM, value=0.7, step=80),
    )
    artifact = Artifact(run_id=run_id, storage_root_id=storage_root.id, relative_path="runs/one/adapter.bin",
                        display_name="Adapter", category="adapter", size_bytes=128,
                        hash_algorithm="sha256", hash_value="recorded-upstream-hash",
                        verification_status=VerificationStatus.VERIFIED,
                        verified_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    datasets = (Dataset(name="Named only", reference="dataset:version-1", example_count=12, split="train"),
                Dataset(name="External", storage_root_id=storage_root.id, relative_path="datasets/train.jsonl"))
    records = (
        Provenance(target_entity=EntityType.PARAMETER, target_id=parameter.id, target_field="value",
                   source_type="file", source_reference="adapter_config.json#/r", original_value=32,
                   candidate_value=32, current_value=16, confidence=Confidence.EXACT, user_modified=True),
        Provenance(target_entity=EntityType.PARAMETER, target_id=parameter.id, target_field="value",
                   source_type="file", source_reference="train.py:10", original_value=16,
                   candidate_value=16, current_value=16, confidence=Confidence.DERIVED),
        Provenance(target_entity=EntityType.RUN, target_id=run_id, target_field="base_model",
                   source_type="file", candidate_value="model/base", confidence=Confidence.INFERRED),
        Provenance(target_entity=EntityType.METRIC, target_id=metrics[0].id, target_field="value",
                   source_type="file", original_value=0.8, confidence=Confidence.EXACT),
        Provenance(target_entity=EntityType.ARTIFACT, target_id=artifact.id, target_field="size_bytes",
                   source_type="file", original_value=128),
        Provenance(target_entity=EntityType.DATASET, target_id=datasets[0].id, target_field="name",
                   source_type="user", current_value="Named only", user_modified=True),
    )
    diagnostics = (
        ImportDiagnostic(message="Malformed trainer state", severity=Severity.ERROR, kind="malformed_metadata",
                         source_reference="trainer_state.json", affected_field="training_finished_at",
                         consequence="Training finish remains unknown"),
        ImportDiagnostic(message="Artifact size could not be checked", artifact_id=artifact.id,
                         consequence="Stored size is from source metadata"),
    )
    run = make_run(storage_root, id=run_id, source_trainer="ExampleTrainer", base_model="model/base",
                   model_revision="v1", method="LoRA", training_status=TrainingStatus.COMPLETED,
                   imported_at=datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=-8))),
                   training_started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                   training_finished_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
                   notes="Worth keeping", rating=4, favorite=True, parameters=(parameter,), metrics=metrics,
                   artifacts=(artifact,), datasets=datasets, tags=(Tag(name="useful"), Tag(name="lora")),
                   provenance=records, diagnostics=diagnostics)
    repository.save_run(run)
    # A new repository/connection must recover everything, including candidate order.
    fresh = CatalogRepository(repository.database)
    assert fresh.get_run(run.id) == run
    assert fresh.list_runs() == [run]
    assert fresh.get_run("missing") is None


def test_scalar_parameter_types_survive_without_coercion(repository, storage_root):
    values = (None, False, 16, 0.25, "16")
    run = make_run(storage_root, parameters=tuple(Parameter(key=f"p{i}", value=v) for i, v in enumerate(values)))
    repository.save_run(run)
    stored = repository.get_run(run.id)
    assert tuple(p.value for p in stored.parameters) == values
    assert tuple(p.value_type for p in stored.parameters) == tuple(ValueType)
    assert tuple(type(p.value) for p in stored.parameters) == tuple(map(type, values))


def test_unknown_history_stays_unknown_after_save(repository, storage_root):
    run = make_run(storage_root)
    repository.save_run(run)
    loaded = repository.get_run(run.id)
    assert loaded.imported_at == run.imported_at
    assert loaded.training_started_at is loaded.training_finished_at is None
    assert loaded.training_status is TrainingStatus.UNKNOWN


def test_metric_context_summary_and_step_are_distinct(repository, storage_root):
    metrics = (
        Metric(name="loss", value=1.0, context="train", summary=MetricSummary.FINAL),
        Metric(name="loss", value=0.7, context="validation", summary=MetricSummary.MINIMUM),
        Metric(name="loss", value=1.2, context="train", summary=MetricSummary.VALUE_AT_STEP, step=5),
        Metric(name="loss", value=1.1, context="train", summary=MetricSummary.VALUE_AT_STEP, step=6),
    )
    run = make_run(storage_root, metrics=metrics)
    repository.save_run(run)
    assert repository.get_run(run.id).metrics == metrics


def test_metric_unit_is_part_of_metric_identity(repository, storage_root):
    metrics = (
        Metric(name="throughput", context="train", summary=MetricSummary.FINAL,
               value=120, unit="tokens/second"),
        Metric(name="throughput", context="train", summary=MetricSummary.FINAL,
               value=2, unit="sequences/second"),
    )
    run = make_run(storage_root, metrics=metrics)

    repository.save_run(run)

    assert repository.get_run(run.id).metrics == metrics


def test_duplicate_unknown_metric_identity_is_rejected_atomically(repository, storage_root):
    run = make_run(storage_root, metrics=(Metric(name="loss", value=1.0), Metric(name="loss", value=2.0)))
    with pytest.raises(sqlite3.IntegrityError):
        repository.save_run(run)
    assert repository.get_run(run.id) is None


def test_failed_aggregate_update_restores_previous_values_and_relations(repository, storage_root):
    original = make_run(storage_root, parameters=(Parameter(key="rank", value=16),), tags=(Tag(name="keep"),))
    repository.save_run(original)
    invalid = replace(original, display_name="Should roll back",
                      parameters=(Parameter(key="rank", value=32), Parameter(key="rank", value=64)))
    with pytest.raises(sqlite3.IntegrityError):
        repository.save_run(invalid)
    assert repository.get_run(original.id) == original


def test_human_update_does_not_replace_provenance_or_history(repository, storage_root):
    run_id = new_id()
    record = Provenance(target_entity=EntityType.RUN, target_id=run_id, target_field="method", source_type="user")
    run = make_run(storage_root, id=run_id, provenance=(record,))
    repository.save_run(run)
    repository.update_human_metadata(run.id, notes="Remember this", rating=5, favorite=True)
    assert repository.get_run(run.id) == replace(run, notes="Remember this", rating=5, favorite=True)
    with pytest.raises(ValueError):
        repository.update_human_metadata(run.id, notes="bad", rating=6, favorite=False)
    with pytest.raises(KeyError):
        repository.update_human_metadata("absent", notes="", rating=None, favorite=False)


def test_shared_datasets_and_tags_are_many_to_many(repository, storage_root):
    dataset, tag = Dataset(name="Reusable"), Tag(name="reusable")
    first = make_run(storage_root, datasets=(dataset,), tags=(tag,))
    second = make_run(storage_root, datasets=(dataset,), tags=(tag,))
    repository.save_run(first)
    repository.save_run(second)
    with repository.database.connection() as connection:
        for table in ("dataset", "tag"):
            assert connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 1
        for table in ("run_dataset", "run_tag"):
            assert connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 2
    assert repository.delete_run(first.id)
    assert repository.get_run(second.id) == second


def test_shared_definition_conflicts_do_not_change_other_runs(repository, storage_root):
    dataset = Dataset(name="Original")
    first = make_run(storage_root, datasets=(dataset,))
    repository.save_run(first)
    second = make_run(storage_root, datasets=(replace(dataset, name="Changed"),))
    with pytest.raises(ValueError, match="Conflicting definition"):
        repository.save_run(second)
    assert repository.get_run(first.id) == first
    assert repository.get_run(second.id) is None


def test_save_update_and_delete_never_touch_external_payloads(repository, storage_root):
    path = Path(storage_root.location) / "adapter.bin"
    path.write_bytes(b"user-owned payload")
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    run_id = new_id()
    artifact = Artifact(run_id=run_id, storage_root_id=storage_root.id,
                        relative_path="adapter.bin", display_name="Adapter")
    run = make_run(storage_root, id=run_id, artifacts=(artifact,),
                   diagnostics=(ImportDiagnostic(message="Retain diagnostic", artifact_id=artifact.id),))
    repository.save_run(run)
    repository.save_run(replace(run, notes="Update"))
    assert repository.delete_run(run.id)
    assert not repository.delete_run(run.id)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert sorted(p.name for p in path.parent.iterdir()) == ["adapter.bin"]
    with repository.database.connection() as connection:
        for table in ("run", "artifact", "import_diagnostic"):
            assert connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_root_relocation_preserves_run_and_artifact_identity(repository, storage_root, tmp_path):
    run_id = new_id()
    artifact = Artifact(run_id=run_id, storage_root_id=storage_root.id,
                        relative_path="adapter.bin", display_name="Adapter")
    run = make_run(storage_root, id=run_id, artifacts=(artifact,))
    repository.save_run(run)  # The file need not exist to register a reference.
    new_location = tmp_path / "remounted"
    new_location.mkdir()
    (new_location / "adapter.bin").write_bytes(b"existing file at new mount")
    moved_root = replace(storage_root, location=str(new_location))
    repository.save_storage_root(moved_root)
    assert repository.get_storage_root(storage_root.id) == moved_root
    assert repository.get_run(run.id) == run
    assert PathResolver().resolve(moved_root, artifact.relative_path).reference_available is True
    assert repository.list_storage_roots() == [moved_root]


def test_list_order_uses_import_time_and_stable_identity(repository, storage_root):
    earlier = datetime(2025, 1, 1, tzinfo=timezone.utc)
    runs = [make_run(storage_root, id="b", imported_at=earlier),
            make_run(storage_root, id="a", imported_at=earlier),
            make_run(storage_root, id="c", imported_at=earlier + timedelta(days=1))]
    for run in runs:
        repository.save_run(run)
    assert [run.id for run in repository.list_runs()] == ["c", "a", "b"]


def test_provenance_foreign_key_rejects_target_owned_by_another_run(repository, storage_root):
    first = make_run(storage_root, parameters=(Parameter(key="rank", value=16),))
    second = make_run(storage_root)
    repository.save_run(first)
    repository.save_run(second)
    with repository.database.transaction() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO provenance(id, run_id, position, target_entity, target_field, parameter_id, "
                "source_type, original_json, candidate_json, current_json, confidence, user_modified, detected_at) "
                "VALUES (?, ?, 0, 'parameter', 'value', ?, 'file', '16', '16', '16', 'exact', 0, ?)",
                (new_id(), second.id, first.parameters[0].id, second.imported_at.isoformat()),
            )


def test_root_references_prevent_accidental_root_deletion(repository, storage_root):
    repository.save_run(make_run(storage_root))
    with repository.database.transaction() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM storage_root WHERE id = ?", (storage_root.id,))
