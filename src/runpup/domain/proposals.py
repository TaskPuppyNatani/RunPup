"""Reviewable import evidence. No canonical Run creation, persistence, or Qt."""

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum

from runpup.domain.models import (
    Artifact, Confidence, EntityType, ImportDiagnostic, Metric, Provenance, Scalar,
    StorageRoot, aware_timestamp, new_id, relative_reference, required_text,
    scalar_type, utc_now,
)


class FieldState(StrEnum):
    DETECTED = "detected"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    USER_EDITED = "user_edited"


MATERIAL_FIELDS = frozenset({
    "base_model", "method", "learning_rate", "lora_rank", "lora_alpha",
    "dataset_identity", "dataset_hash", "dataset_hash_algorithm", "max_sequence_length",
})


def _same_value(left: Scalar, right: Scalar) -> bool:
    # JSON integer/real representations may agree, but True is not the number 1.
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    return left == right


@dataclass(frozen=True, slots=True, kw_only=True)
class ProposedField:
    key: str
    candidates: tuple[Provenance, ...] = ()
    edits: tuple[Provenance, ...] = ()
    material: bool = False

    def __post_init__(self) -> None:
        required_text(self.key, "proposal field key")
        if self.key in MATERIAL_FIELDS:
            object.__setattr__(self, "material", True)
        for name in ("candidates", "edits"):
            records = tuple(getattr(self, name))
            if any(not isinstance(item, Provenance) for item in records):
                raise ValueError("Field evidence must use Provenance")
            object.__setattr__(self, name, records)
        if any(not record.user_modified for record in self.edits):
            raise ValueError("Explicit edits must record user modification")

    @property
    def state(self) -> FieldState:
        if self.edits:
            return FieldState.USER_EDITED
        values = [record.candidate_value for record in self.candidates
                  if record.candidate_value is not None]
        if not values:
            return FieldState.UNKNOWN
        if any(not _same_value(value, values[0]) for value in values[1:]):
            return FieldState.CONFLICT
        return FieldState.DETECTED

    @property
    def value(self) -> Scalar:
        if self.edits:
            return self.edits[-1].current_value
        if self.state is FieldState.DETECTED:
            return next(record.candidate_value for record in self.candidates
                        if record.candidate_value is not None)
        return None


@dataclass(frozen=True, slots=True, kw_only=True)
class ProposedRun:
    source_storage_root: StorageRoot
    source_relative_directory: str | None
    id: str = field(default_factory=new_id)
    scanned_at: datetime = field(default_factory=utc_now)
    detected_importer: str | None = None
    source_trainer: str | None = None
    detection_evidence: tuple[str, ...] = ()
    fields: tuple[ProposedField, ...] = ()
    metrics: tuple[Metric, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    extra_provenance: tuple[Provenance, ...] = ()
    diagnostics: tuple[ImportDiagnostic, ...] = ()
    blocking_diagnostic_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required_text(self.id, "proposal id")
        aware_timestamp(self.scanned_at, "scanned_at", required=True)
        if self.source_relative_directory is not None:
            object.__setattr__(self, "source_relative_directory",
                               relative_reference(self.source_relative_directory))
        for name in ("detection_evidence", "fields", "metrics", "artifacts", "extra_provenance",
                     "diagnostics", "blocking_diagnostic_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len({item.key for item in self.fields}) != len(self.fields):
            raise ValueError("Proposal field keys must be unique")
        if any(item.run_id != self.id for item in self.artifacts):
            raise ValueError("Discovered artifacts must belong to this proposal")
        if not set(self.blocking_diagnostic_ids) <= {item.id for item in self.diagnostics}:
            raise ValueError("Blocking state must refer to retained diagnostics")

    def field(self, key: str) -> ProposedField:
        return next((item for item in self.fields if item.key == key),
                    ProposedField(key=key, material=key in MATERIAL_FIELDS))

    @property
    def provenance(self) -> tuple[Provenance, ...]:
        return tuple(record for item in self.fields for record in (*item.candidates, *item.edits)) + self.extra_provenance

    @property
    def blocking_fields(self) -> tuple[str, ...]:
        return tuple(item.key for item in self.fields
                     if item.material and item.state is FieldState.CONFLICT)

    @property
    def blocking_diagnostics(self) -> tuple[ImportDiagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.id in self.blocking_diagnostic_ids)

    @property
    def has_blockers(self) -> bool:
        return bool(self.blocking_fields or self.blocking_diagnostic_ids)

    def with_user_edit(self, key: str, value: Scalar) -> "ProposedRun":
        """Return a revised proposal without erasing imported or earlier manual evidence."""
        scalar_type(value)
        original = self.field(key)
        target = original.candidates[0] if original.candidates else None
        edit = Provenance(
            target_entity=target.target_entity if target else EntityType.RUN,
            target_id=target.target_id if target else self.id,
            target_field=target.target_field if target else key,
            source_type="manual", original_value=original.value, candidate_value=value,
            current_value=value, confidence=Confidence.EXACT, user_modified=True,
        )
        updated = replace(original, edits=(*original.edits, edit))
        fields = tuple(updated if item.key == key else item for item in self.fields)
        if not any(item.key == key for item in self.fields):
            fields = (*fields, updated)
        return replace(self, fields=fields)
