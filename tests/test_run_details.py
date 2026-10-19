"""Focused saved-data presentation and resolver contracts (no Qt required)."""

from dataclasses import replace

import pytest

from runpup.domain.models import (
    Artifact, Availability, Confidence, Dataset, EntityType, ImportDiagnostic,
    Metric, MetricSummary, Parameter, Provenance, RootKind, Run, StorageRoot, Tag,
)
from runpup.services.paths import PathResolver
from runpup.ui.run_details import dataset_fields, evidence_groups, load_run_details, ordered_metrics, parameter_sections


@pytest.fixture
def saved_run(repository, storage_root):
    run = Run(display_name="Saved settings", source_storage_root_id=storage_root.id,
              source_relative_directory="run", notes="Keep this", rating=4, favorite=True,
              base_model="org/model", method="LoRA", tags=(Tag(name="keeper"),))
    parameters = (Parameter(key="lora_rank", value=16), Parameter(key="effective_batch_size", value=8),
                  Parameter(key="learning_rate", value=None), Parameter(key="custom.setting", value=False))
    metrics = (
        Metric(name="loss", context="train", value=.8, step=10, unit="nats", summary=MetricSummary.FINAL),
        Metric(name="loss", context="eval", value=.6, step=10, unit="nats", summary=MetricSummary.FINAL),
        Metric(name="loss", context="train", value=.9, step=5, unit="nats", summary=MetricSummary.FINAL),
        Metric(name="loss", context="train", value=.5, step=10, unit="bits", summary=MetricSummary.FINAL),
        Metric(name="loss", context="train", value=.4, step=10, unit="nats", summary=MetricSummary.MINIMUM),
        Metric(name="learning_rate", context="train", value=.00001, step=10),
    )
    artifact = Artifact(run_id=run.id, storage_root_id=storage_root.id,
                        relative_path="run/training_args.bin", display_name="Binary args", category="metadata")
    records = tuple(Provenance(target_entity=EntityType.PARAMETER, target_id=p.id, target_field="value",
                              source_type="file", source_reference="config.json#/" + p.key,
                              original_value=32 if i == 0 else p.value,
                              candidate_value=32 if i == 0 else p.value, current_value=p.value,
                              confidence=confidence, user_modified=i == 0)
                    for i, (p, confidence) in enumerate(zip(parameters, (Confidence.EXACT, Confidence.DERIVED))))
    run = replace(run, parameters=parameters, metrics=metrics, artifacts=(artifact,), provenance=records,
                  datasets=(Dataset(name="Training set", reference="dataset:v2", split="train", example_count=12),),
                  diagnostics=(ImportDiagnostic(message="Not deserialized", source_reference=artifact.relative_path,
                                                artifact_id=artifact.id),))
    repository.save_run(run)
    return run


def test_details_loads_fresh_saved_aggregate_and_only_resolves_references(repository, saved_run):
    class Resolver:
        def __init__(self):
            self.calls = []

        def resolve(self, root, path):
            self.calls.append((root.id, path))
            return PathResolver().resolve(root, path, verify=False)

    resolver = Resolver()
    details = load_run_details(repository, saved_run.id, resolver=resolver)
    assert details.run == saved_run
    assert details.datasets == (None,)
    assert resolver.calls == [(saved_run.source_storage_root_id, "run"),
                              (saved_run.source_storage_root_id, "run/training_args.bin")]
    repository.update_human_metadata(saved_run.id, notes="Changed later", rating=0, favorite=False)
    assert load_run_details(repository, saved_run.id).run.notes == "Changed later"


def test_common_unknowns_and_configured_learning_rate_do_not_use_observations(saved_run):
    sections = dict(parameter_sections(saved_run))
    training = dict(sections["Training"])
    assert training["Configured learning rate"] is None
    assert training["Planned epochs"] is None
    assert training["Effective batch size"] == 8
    assert dict(sections["LoRA / QLoRA"])["LoRA rank (r)"] == 16
    assert dict(sections["Additional saved parameters"])["custom.setting"] is False


def test_unnamed_dataset_facts_are_shown_without_inventing_identity(saved_run):
    run = replace(saved_run, datasets=(), provenance=(), parameters=(
        Parameter(key="dataset_example_count", value=12), Parameter(key="dataset_hash", value="partial-hash")))
    assert dict(dataset_fields(run)) == {"Identity": None, "Reference": None, "Split": None,
                                        "Examples": 12, "Hash": "partial-hash", "Hash algorithm": None}
    assert "Additional saved parameters" not in dict(parameter_sections(run))


def test_metric_display_preserves_every_identity_dimension_and_is_deterministic(saved_run):
    metrics = ordered_metrics(saved_run)
    assert len(metrics) == 6 and set(metrics) == set(saved_run.metrics)
    assert metrics == ordered_metrics(replace(saved_run, metrics=tuple(reversed(saved_run.metrics))))
    assert all(m.name == "loss" for m in metrics[:5])


def test_evidence_keeps_exact_derived_and_user_edit_records(saved_run):
    groups = evidence_groups(saved_run)
    assert "lora_rank" in groups[0][0] and "effective_batch_size" in groups[1][0]
    assert tuple(record for _, records in groups for record in records) == saved_run.provenance
    assert groups[0][1][0].user_modified
    assert groups[1][1][0].confidence is Confidence.DERIVED


def test_evidence_labels_all_supported_entities_without_merging_targets(saved_run):
    targets = ((EntityType.RUN, saved_run.id), (EntityType.METRIC, saved_run.metrics[0].id),
               (EntityType.ARTIFACT, saved_run.artifacts[0].id), (EntityType.DATASET, saved_run.datasets[0].id))
    records = tuple(Provenance(target_entity=entity, target_id=target, target_field="value",
                               source_type="file", source_reference="metadata.json#/value")
                    for entity, target in targets)
    groups = evidence_groups(replace(saved_run, provenance=records))
    assert len(groups) == 4
    assert "train / final / step 10 / nats" in groups[1][0]
    assert "run/training_args.bin" in groups[2][0] and "Training set" in groups[3][0]
    assert tuple(group[1][0] for group in groups) == records


@pytest.mark.parametrize("state", (Availability.AVAILABLE, Availability.OFFLINE, Availability.MISSING))
def test_artifact_availability_states_are_distinct(repository, storage_root, tmp_path, state):
    from pathlib import Path

    location = Path(storage_root.location)
    if state is Availability.OFFLINE:
        location.rmdir()
    elif state is Availability.AVAILABLE:
        (location / "payload.bin").write_bytes(b"must not be opened")
    run = Run(display_name="Availability", source_storage_root_id=storage_root.id, source_relative_directory=".")
    artifact = Artifact(run_id=run.id, storage_root_id=storage_root.id,
                        relative_path="payload.bin", display_name="Payload")
    repository.save_run(replace(run, artifacts=(artifact,)))
    details = load_run_details(repository, run.id)
    assert details.artifacts[0].resolution.state is state


def test_portable_root_without_base_stays_unverified(repository):
    root = StorageRoot(name="Portable", kind=RootKind.PORTABLE_RELATIVE, location="outputs")
    repository.save_storage_root(root)
    run = Run(display_name="Portable run", source_storage_root_id=root.id, source_relative_directory="run")
    repository.save_run(run)
    assert load_run_details(repository, run.id).source.resolution.state is Availability.UNVERIFIED


def test_deleted_run_is_reported(repository, saved_run):
    repository.delete_run(saved_run.id)
    with pytest.raises(KeyError, match="no longer"):
        load_run_details(repository, saved_run.id)
