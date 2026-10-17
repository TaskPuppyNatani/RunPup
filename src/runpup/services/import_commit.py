"""Review validation and the single ProposedRun -> catalog handoff. No Qt."""

from dataclasses import dataclass, replace
from datetime import datetime
import json
from uuid import NAMESPACE_URL, uuid5

from runpup.domain.models import (
    Availability, Dataset, EntityType, ImportDiagnostic, Parameter, Run, Severity,
    TrainingStatus, utc_now,
)
from runpup.domain.proposals import FieldState, ProposedRun
from runpup.persistence.repository import CatalogRepository
from runpup.services.paths import PathResolver


_RUN_FIELDS = frozenset({
    "display_name", "source_trainer", "base_model", "model_revision", "method",
    "training_started_at", "training_finished_at", "training_status",
    "notes", "rating", "favorite",
})
_DATASET_FIELDS = {
    "dataset_identity": "name", "dataset_reference": "reference",
    "dataset_hash": "hash_value", "dataset_hash_algorithm": "hash_algorithm",
    "dataset_example_count": "example_count", "dataset_split": "split",
}


@dataclass(frozen=True, slots=True)
class CommitValidation:
    run: Run | None
    blocking: tuple[ImportDiagnostic, ...]
    warnings: tuple[ImportDiagnostic, ...]

    @property
    def can_commit(self) -> bool:
        return self.run is not None and not self.blocking


class ProposalValidationError(ValueError):
    def __init__(self, validation: CommitValidation):
        self.validation = validation
        super().__init__("; ".join(item.message for item in validation.blocking))


def _identity(proposal: ProposedRun, kind: str, key: str) -> str:
    # Stable across validation/retry, independent of source paths and locations.
    return uuid5(NAMESPACE_URL, f"runpup:{proposal.id}:{kind}:{key}").hex


def _metric_key(metric) -> str:
    return "metric:" + json.dumps(
        [metric.name, metric.context, metric.summary.value, metric.step, metric.unit],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


class ProposedRunCommitService:
    def __init__(self, repository: CatalogRepository, *, resolver: PathResolver | None = None):
        self.repository = repository
        self.resolver = resolver or PathResolver()

    def validate(self, proposal: ProposedRun, *, dataset_id: str | None = None) -> CommitValidation:
        """Prepare without writing. Explicit dataset IDs use repository exact equality.

        Non-material conflicts retain evidence with no canonical selection. Metric
        conflicts/explicit unknowns have run-target evidence since Metric.value
        cannot be null. Historical manual edit values are preserved unchanged;
        imported candidates record the accepted current value.
        """
        blocking = list(proposal.blocking_diagnostics)
        warnings = [item for item in proposal.diagnostics
                    if item.id not in proposal.blocking_diagnostic_ids]

        def issue(message, *, key=None, block=True, kind="invalid_proposal"):
            diagnostic = ImportDiagnostic(
                id=_identity(proposal, "validation", f"{kind}:{key}:{message}"),
                message=message, affected_field=key, kind=kind,
                severity=Severity.ERROR if block else Severity.WARNING,
                detected_at=proposal.scanned_at,
            )
            (blocking if block else warnings).append(diagnostic)

        for key in proposal.blocking_fields:
            issue(f"Resolve material conflict: {key}", key=key, kind="unresolved_conflict")
        for item in proposal.fields:
            if item.state is FieldState.CONFLICT and not item.material:
                issue(f"No canonical selection for conflicting field: {item.key}",
                      key=item.key, block=False, kind="unresolved_conflict")

        try:
            root = proposal.source_storage_root
            existing = self.repository.get_storage_root(root.id)
            if existing is not None and existing != root:
                raise ValueError("Proposal storage root differs from the registered root")
            references = [(root.id, proposal.source_relative_directory)]
            references.extend((a.storage_root_id, a.relative_path) for a in proposal.artifacts)
            artifact_references = [(a.storage_root_id, a.relative_path) for a in proposal.artifacts]
            if len(set(artifact_references)) != len(artifact_references):
                raise ValueError("Duplicate artifact reference in proposal")
            roots = {root.id: root}
            for root_id, reference in dict.fromkeys(references):
                if reference is None:
                    raise ValueError("Source relative directory is required")
                if root_id not in roots:
                    roots[root_id] = self.repository.get_storage_root(root_id)
                if roots[root_id] is None:
                    raise ValueError(f"Unregistered storage root: {root_id}")
                resolution = self.resolver.resolve(roots[root_id], reference)
                if resolution.reason and "escapes its storage root" in resolution.reason:
                    raise ValueError(resolution.reason)
                if resolution.state is not Availability.AVAILABLE:
                    issue(f"Reference {root_id}/{reference} is {resolution.state.value}",
                          key=reference, block=False, kind="reference_availability")

            values = {item.key: item.value for item in proposal.fields}
            run_values = {key: value for key, value in values.items() if key in _RUN_FIELDS}
            # Domain defaults apply only to required canonical metadata, never to
            # unknown optional historical/training values.
            for key, default in (("training_status", TrainingStatus.UNKNOWN),
                                 ("notes", ""), ("favorite", False)):
                if run_values.get(key) is None:
                    run_values[key] = default
            if "source_trainer" not in values:
                run_values["source_trainer"] = proposal.source_trainer
            for key in ("training_started_at", "training_finished_at"):
                if run_values.get(key) is not None:
                    run_values[key] = datetime.fromisoformat(run_values[key])

            dataset_fields = dict(_DATASET_FIELDS)
            if (values.get("dataset_hash") is None) != (values.get("dataset_hash_algorithm") is None):
                # Dataset hashes are paired. Keep incomplete optional evidence as
                # parameters rather than guessing an algorithm or losing a hash.
                for key in ("dataset_hash", "dataset_hash_algorithm"):
                    dataset_fields.pop(key)
                issue("Incomplete dataset hash retained as parameters", key="dataset_hash",
                      block=False, kind="incomplete_metadata")
            dataset_values = {target: values[key] for key, target in dataset_fields.items()
                              if values.get(key) is not None}
            # An unnamed dataset's partial facts stay as flexible parameters.
            dataset = None
            if dataset_values.get("name") is not None:
                dataset = Dataset(id=dataset_id if dataset_id is not None else
                                  _identity(proposal, "dataset", "primary"), **dataset_values)
                existing_dataset = self.repository.get_dataset(dataset.id)
                if existing_dataset is not None and existing_dataset != dataset:
                    raise ValueError(f"Conflicting definition for shared dataset {dataset.id}")
            elif dataset_id is not None:
                raise ValueError("Dataset reuse requires a reviewed dataset identity")

            parameters, metrics, records = [], [], []
            targets = {}
            metric_groups = {}
            if len({metric.id for metric in proposal.metrics}) != len(proposal.metrics):
                raise ValueError("Duplicate metric candidate identities")
            for metric in proposal.metrics:
                metric_groups.setdefault(_metric_key(metric), []).append(metric)
            for key, group in metric_groups.items():
                field = proposal.field(key)
                if not field.candidates and not field.edits:
                    raise ValueError(f"Metric candidates lack review evidence: {key}")
                expected = {metric.id for metric in group}
                if {record.target_id for record in field.candidates} != expected or any(
                       record.target_entity is not EntityType.METRIC or
                       record.target_id not in expected or record.target_field != "value"
                       for record in field.candidates):
                    raise ValueError(f"Metric evidence has an unrelated target: {key}")
                candidate_values = {metric.id: metric.value for metric in group}
                if any(type(record.candidate_value) not in (int, float) or
                       record.candidate_value != candidate_values[record.target_id]
                       for record in field.candidates):
                    raise ValueError(f"Metric candidates disagree with their review evidence: {key}")
                value = field.value
                if value is not None:
                    canonical = replace(group[0], value=value)
                    metrics.append(canonical)
                    targets[key] = (EntityType.METRIC, canonical.id, "value")
                else:
                    targets[key] = (EntityType.RUN, proposal.id, key)

            for field in proposal.fields:
                key = field.key
                if key.startswith("metric:"):
                    if key not in targets:
                        raise ValueError(f"Review field lacks metric candidates: {key}")
                elif key in _RUN_FIELDS:
                    targets[key] = (EntityType.RUN, proposal.id, key)
                elif key in dataset_fields and dataset is not None:
                    targets[key] = (EntityType.DATASET, dataset.id, dataset_fields[key])
                else:
                    parameter = Parameter(id=_identity(proposal, "parameter", key),
                                          key=key, value=field.value)
                    parameters.append(parameter)
                    targets[key] = (EntityType.PARAMETER, parameter.id, "value")
                entity, target_id, target_field = targets[key]
                for record in (*field.candidates, *field.edits):
                    records.append(replace(
                        record, target_entity=entity, target_id=target_id, target_field=target_field,
                        current_value=record.current_value if record.user_modified else field.value,
                    ))

            run = Run(
                id=proposal.id, source_storage_root_id=root.id,
                source_relative_directory=proposal.source_relative_directory,
                imported_at=utc_now(), **run_values,
                parameters=tuple(parameters), metrics=tuple(metrics), artifacts=proposal.artifacts,
                datasets=(dataset,) if dataset else (),
                provenance=(*records, *proposal.extra_provenance),
                diagnostics=tuple(warnings),
            )
        except (ValueError, TypeError, OverflowError) as error:
            issue(str(error))
            run = None
        return CommitValidation(run if not blocking else None, tuple(blocking), tuple(warnings))

    def commit(self, proposal: ProposedRun, *, dataset_id: str | None = None) -> Run:
        validation = self.validate(proposal, dataset_id=dataset_id)
        if not validation.can_commit:
            raise ProposalValidationError(validation)
        return self.repository.save_import(validation.run, proposal.source_storage_root)
