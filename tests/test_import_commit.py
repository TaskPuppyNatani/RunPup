from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from runpup.domain.models import (
    Artifact, Confidence, Dataset, EntityType, ImportDiagnostic, Metric,
    MetricSummary, Provenance, Run, StorageRoot, TrainingStatus,
)
from runpup.domain.proposals import ProposedField, ProposedRun
from runpup.importers import HFTrainerImporter
from runpup.persistence.repository import CatalogRepository
from runpup.services.import_commit import ProposedRunCommitService, ProposalValidationError


def proposal(root, **fields):
    result = ProposedRun(source_storage_root=root, source_relative_directory=".")
    fields = {"display_name": "Reviewed run", **fields}
    return replace(result, fields=tuple(ProposedField(key=key, candidates=(Provenance(
        target_entity=EntityType.RUN, target_id=result.id, target_field=key,
        source_type="training_args", source_reference=f"training_args.json#/{key}",
        original_value=value, candidate_value=value, confidence=Confidence.EXACT,
    ),)) for key, value in fields.items()))


def snapshot(path):
    return {str(p.relative_to(path)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in path.rglob("*") if p.is_file()}


def test_scan_review_commit_reload(repository, storage_root, monkeypatch):
    directory = Path(storage_root.location)
    (directory / "trainer_state.json").write_text(json.dumps({
        "log_history": [{"step": 8, "loss": 0.4, "eval_loss": 0.5}],
    }))
    (directory / "training_args.json").write_text(json.dumps({
        "run_name": "Δ run", "learning_rate": 0.001, "seed": 12,
        "per_device_train_batch_size": 2, "gradient_accumulation_steps": 4,
        "world_size": 2, "dataset": {"name": "data/v1", "example_count": 40},
        "training_started_at": "2025-01-01T00:00:00Z",
    }))
    (directory / "adapter_model.safetensors").write_bytes(b"unread payload")
    before = snapshot(directory)
    scanned = HFTrainerImporter().scan(storage_root)
    reviewed = scanned.with_user_edit("base_model", "org/base").with_user_edit("notes", "keep")
    reviewed = reviewed.with_user_edit("rating", 4).with_user_edit("favorite", True)
    service = ProposedRunCommitService(repository)
    # Commit may stat references, but must never open the discovered payloads.
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path.is_relative_to(directory):
            raise AssertionError(f"Source opened during commit: {path}")
        return original_open(path, *args, **kwargs)
    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", guarded_open)
        run = service.commit(reviewed)
    assert CatalogRepository(repository.database).get_run(run.id) == run
    assert run.id == scanned.id and run.display_name == "Δ run"
    assert run.base_model == "org/base" and (run.notes, run.rating, run.favorite) == ("keep", 4, True)
    values = {p.key: p.value for p in run.parameters}
    assert values["learning_rate"] == 0.001 and type(values["seed"]) is int
    assert values["effective_batch_size"] == 16
    assert run.datasets[0].name == "data/v1" and run.datasets[0].example_count == 40
    assert len(run.metrics) == 2 and all(m.step == 8 for m in run.metrics)
    assert all(m.summary is MetricSummary.VALUE_AT_STEP for m in run.metrics)
    assert run.imported_at >= scanned.scanned_at
    assert run.training_status is TrainingStatus.UNKNOWN and run.training_finished_at is None
    assert run.artifacts == scanned.artifacts and run.diagnostics
    assert {p.id for p in reviewed.provenance} == {p.id for p in run.provenance}
    assert snapshot(directory) == before
    with repository.database.connection() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3


def test_material_resolution_preserves_all_evidence_and_edit_history(repository, storage_root):
    p = proposal(storage_root, learning_rate=0.1)
    field = p.field("learning_rate")
    losing = replace(field.candidates[0], id="losing", candidate_value=0.2, original_value="2e-1")
    p = replace(p, fields=(p.field("display_name"), replace(field, candidates=(*field.candidates, losing))))
    service = ProposedRunCommitService(repository)
    result = service.validate(p)
    assert not result.can_commit and result.blocking[0].affected_field == "learning_rate"
    with pytest.raises(ProposalValidationError):
        service.commit(p)
    assert repository.list_runs() == []
    selected = p.with_user_edit("learning_rate", 0.2)
    accepted = service.commit(selected)
    assert next(p.value for p in accepted.parameters if p.key == "learning_rate") == 0.2
    assert [r.candidate_value for r in accepted.provenance if r.target_entity is EntityType.PARAMETER] == [0.1, 0.2, 0.2]
    edited = selected.with_user_edit("learning_rate", 0.3)
    run = service.commit(edited)
    parameter = next(p for p in run.parameters if p.key == "learning_rate")
    evidence = [r for r in run.provenance if r.target_id == parameter.id]
    assert parameter.value == 0.3
    assert [r.candidate_value for r in evidence] == [0.1, 0.2, 0.2, 0.3]
    assert [r.current_value for r in evidence] == [0.3, 0.3, 0.2, 0.3]
    for old, new in zip(edited.field("learning_rate").candidates, evidence):
        assert (old.id, old.original_value, old.confidence, old.detected_at, old.source_reference) == (
            new.id, new.original_value, new.confidence, new.detected_at, new.source_reference)
    assert all(r.target_entity is EntityType.PARAMETER for r in evidence)


def test_unknown_and_nonmaterial_conflicts_remain_coherent(repository, storage_root):
    p = proposal(storage_root, base_model=None, optimizer="adam", seed=4,
                 training_started_at=None, training_finished_at=None)
    field = p.field("optimizer")
    fields = tuple(replace(f, candidates=(*f.candidates, replace(field.candidates[0], id="other",
                   candidate_value="sgd"))) if f.key == field.key else f for f in p.fields)
    run = ProposedRunCommitService(repository).commit(replace(p, fields=fields))
    assert run.base_model is run.training_started_at is run.training_finished_at is None
    assert next(p.value for p in run.parameters if p.key == "optimizer") is None
    assert any(d.kind == "unresolved_conflict" for d in run.diagnostics)


@pytest.mark.parametrize("key,value", [
    ("display_name", ""), ("base_model", 12), ("training_started_at", "yesterday"),
    ("training_started_at", "2025-01-01T00:00:00"), ("training_status", "running"),
    ("rating", 8), ("favorite", 1),
])
def test_bad_canonical_fields_block_without_writes(repository, storage_root, key, value):
    p = proposal(storage_root, **{key: value})
    result = ProposedRunCommitService(repository).validate(p)
    assert not result.can_commit and result.blocking
    assert repository.list_runs() == []


def test_missing_required_and_reversed_timestamps(repository, storage_root):
    service = ProposedRunCommitService(repository)
    p = proposal(storage_root)
    assert not service.validate(replace(p, source_relative_directory=None)).can_commit
    assert not service.validate(replace(p, fields=())).can_commit
    p = p.with_user_edit("training_started_at", "2025-01-02T00:00:00+00:00")
    p = p.with_user_edit("training_finished_at", "2025-01-01T00:00:00+00:00")
    assert not service.validate(p).can_commit


def test_history_timestamps_are_not_import_time(repository, storage_root):
    p = proposal(storage_root, training_started_at="2025-01-01T00:00:00+00:00",
                 training_finished_at="2025-01-02T00:00:00+00:00", training_status="completed")
    run = ProposedRunCommitService(repository).commit(p)
    assert run.training_started_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert run.training_finished_at < run.imported_at
    assert run.training_status is TrainingStatus.COMPLETED


def test_scan_blockers_and_artifact_diagnostics(repository, storage_root):
    p = proposal(storage_root)
    artifact = Artifact(run_id=p.id, storage_root_id=storage_root.id, relative_path="missing.bin",
                        display_name="missing", category="adapter", size_bytes=5)
    diagnostic = ImportDiagnostic(message="Optional metadata corrupt", artifact_id=artifact.id,
                                  consequence="Adapter details incomplete")
    p = replace(p, artifacts=(artifact,), diagnostics=(diagnostic,))
    service = ProposedRunCommitService(repository)
    assert not service.validate(replace(p, blocking_diagnostic_ids=(diagnostic.id,))).can_commit
    run = service.commit(p)
    assert run.diagnostics[0] == diagnostic and run.artifacts == (artifact,)


def metric_proposal(root):
    p = proposal(root)
    first = Metric(name="rate", value=1, unit="tokens/s", summary=MetricSummary.OTHER)
    second = replace(first, id="second", value=2)
    third = replace(first, id="third", value=3, unit="sequences/s")
    fields = []
    for group in ((first, second), (third,)):
        m = group[0]
        key = "metric:" + json.dumps([m.name, m.context, m.summary.value, m.step, m.unit], separators=(",", ":"))
        fields.append(ProposedField(key=key, candidates=tuple(Provenance(
            target_entity=EntityType.METRIC, target_id=m.id, target_field="value", source_type="trainer_state",
            candidate_value=m.value, original_value=m.value,
        ) for m in group)))
    return replace(p, fields=(*p.fields, *fields), metrics=(first, second, third)), fields[0].key


def test_metric_identity_selection_and_losing_evidence(repository, storage_root):
    p, key = metric_proposal(storage_root)
    service = ProposedRunCommitService(repository)
    unselected = service.validate(p).run
    assert len(unselected.metrics) == 1
    evidence = [r for r in unselected.provenance if r.target_field == key]
    assert len(evidence) == 2 and all(r.target_entity is EntityType.RUN for r in evidence)
    run = service.commit(p.with_user_edit(key, 2))
    assert [(m.value, m.unit) for m in run.metrics] == [(2, "tokens/s"), (3, "sequences/s")]
    first = run.metrics[0]
    assert [r.candidate_value for r in run.provenance if r.target_id == first.id] == [1, 2, 2]
    assert run.metrics[0].summary is MetricSummary.OTHER
    assert not service.validate(p.with_user_edit(key, "bad")).can_commit


def test_dataset_reuse_requires_explicit_equal_identity(repository, storage_root):
    service = ProposedRunCommitService(repository)
    first = service.commit(proposal(storage_root, dataset_identity="dataset", dataset_example_count=10))
    shared = first.datasets[0]
    second = service.commit(proposal(storage_root, dataset_identity="dataset", dataset_example_count=10),
                            dataset_id=shared.id)
    assert second.datasets == (shared,)
    third = service.commit(proposal(storage_root, dataset_identity="dataset", dataset_example_count=10))
    assert third.datasets[0].id != shared.id
    conflict = proposal(storage_root, dataset_identity="dataset", dataset_example_count=20)
    assert not service.validate(conflict, dataset_id=shared.id).can_commit
    with pytest.raises(ProposalValidationError, match="Conflicting definition"):
        service.commit(conflict, dataset_id=shared.id)
    assert repository.get_run(conflict.id) is None
    assert repository.get_run(first.id) == first
    assert any(r.target_entity is EntityType.DATASET and r.target_field == "name"
               for r in second.provenance)


def test_reference_safety_offline_and_missing(repository, tmp_path):
    root = StorageRoot(name="offline", location=str(tmp_path / "offline"))
    p = proposal(root)
    service = ProposedRunCommitService(repository)
    run = service.commit(p)
    assert any("offline" in d.message for d in run.diagnostics)
    Path(root.location).mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (Path(root.location) / "escape").symlink_to(outside, target_is_directory=True)
    assert not service.validate(replace(p, source_relative_directory="escape")).can_commit
    artifact = Artifact(run_id=p.id, storage_root_id="unregistered", relative_path="x", display_name="x")
    assert not service.validate(replace(p, artifacts=(artifact,))).can_commit
    changed_root = replace(root, location=str(outside))
    assert not service.validate(replace(p, source_storage_root=changed_root)).can_commit


def test_partial_dataset_facts_are_retained_without_invented_identity_or_algorithm(repository, storage_root):
    service = ProposedRunCommitService(repository)
    anonymous = service.commit(proposal(storage_root, dataset_reference="data:v1", dataset_example_count=12))
    assert anonymous.datasets == ()
    assert {p.key: p.value for p in anonymous.parameters}["dataset_example_count"] == 12
    partial = service.commit(proposal(storage_root, dataset_identity="named", dataset_hash="upstream"))
    assert partial.datasets[0].hash_value is partial.datasets[0].hash_algorithm is None
    parameter = next(p for p in partial.parameters if p.key == "dataset_hash")
    assert parameter.value == "upstream"
    assert any(r.target_id == parameter.id and r.candidate_value == "upstream" for r in partial.provenance)


def test_incoherent_metric_evidence_and_artifact_duplicates_block(repository, storage_root):
    service = ProposedRunCommitService(repository)
    p, key = metric_proposal(storage_root)
    field = p.field(key)
    incomplete = replace(field, candidates=field.candidates[:1])
    assert not service.validate(replace(p, fields=tuple(
        incomplete if f.key == key else f for f in p.fields))).can_commit
    wrong_value = replace(field, candidates=(replace(field.candidates[0], candidate_value=99),
                                            field.candidates[1]))
    assert not service.validate(replace(p, fields=tuple(
        wrong_value if f.key == key else f for f in p.fields))).can_commit
    artifact = Artifact(run_id=p.id, storage_root_id=storage_root.id,
                        relative_path="model.bin", display_name="model")
    assert not service.validate(replace(p, artifacts=(artifact, replace(artifact, id="other")))).can_commit


def test_late_failure_rolls_back_every_record_and_retries(repository, tmp_path):
    root = StorageRoot(name="new source", location=str(tmp_path / "source"))
    Path(root.location).mkdir()
    payload = Path(root.location) / "model.safetensors"
    payload.write_bytes(b"external data")
    before = snapshot(Path(root.location))
    # A pre-existing shared record and relationship must survive untouched.
    other_root = StorageRoot(name="existing", location=str(tmp_path))
    repository.save_storage_root(other_root)
    existing = Run(display_name="existing", source_storage_root_id=other_root.id,
                   source_relative_directory=".", datasets=(Dataset(name="shared"),))
    repository.save_run(existing)
    p, _ = metric_proposal(root)
    p = p.with_user_edit("dataset_identity", "new dataset").with_user_edit("learning_rate", 0.01)
    p = replace(p, artifacts=(Artifact(run_id=p.id, storage_root_id=root.id,
                relative_path=payload.name, display_name="model"),),
                diagnostics=(ImportDiagnostic(message="late diagnostic"),))
    tables = ("storage_root", "run", "parameter", "metric", "artifact", "dataset", "run_dataset",
              "tag", "run_tag", "provenance", "import_diagnostic")
    def counts():
        with repository.database.connection() as c:
            return {t: c.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}
    baseline = counts()
    with repository.database.transaction() as c:
        c.execute("""CREATE TRIGGER fail_import BEFORE INSERT ON import_diagnostic
                     WHEN NEW.message = 'late diagnostic'
                     BEGIN SELECT RAISE(ABORT, 'late failure'); END""")
    service = ProposedRunCommitService(repository)
    prepared = service.validate(p).run
    with pytest.raises(sqlite3.IntegrityError, match="late failure"):
        service.commit(p)
    assert counts() == baseline and repository.get_run(p.id) is None
    assert repository.get_storage_root(root.id) is None
    assert repository.get_run(existing.id) == existing
    assert snapshot(Path(root.location)) == before
    with repository.database.transaction() as c:
        c.execute("DROP TRIGGER fail_import")
    saved = service.commit(p)
    assert saved.id == p.id
    assert saved.parameters == prepared.parameters and saved.datasets == prepared.datasets
    repeated = service.commit(p)
    assert repeated.parameters == saved.parameters and repeated.datasets == saved.datasets
    assert counts()["run"] == 2 and snapshot(Path(root.location)) == before


def test_service_import_does_not_load_qt():
    subprocess.run([sys.executable, "-c", "import sys; import runpup.services.import_commit; "
                    "assert not any(k.startswith('PySide6') for k in sys.modules)"], check=True)
