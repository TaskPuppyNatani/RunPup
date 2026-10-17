"""Canonical catalog values. Nothing here reads files or depends on Qt/SQL."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import uuid4


Scalar = str | int | float | bool | None


def new_id() -> str:
    return uuid4().hex


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def required_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


def relative_reference(value: str) -> str:
    """Validate and normalize a portable, root-contained lexical reference."""
    required_text(value, "relative path")
    path = PurePosixPath(value)
    if ("\\" in value or "\0" in value or path.is_absolute()
            or PureWindowsPath(value).drive or ".." in path.parts):
        raise ValueError("References must be relative paths without '..' or backslashes")
    return path.as_posix()


def _optional_text(value: str | None, name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be text or None")


def aware_timestamp(value: datetime | None, name: str, *, required: bool = False) -> None:
    if value is None and required:
        raise ValueError(f"{name} is required")
    if value is not None and (
        not isinstance(value, datetime) or value.utcoffset() is None
    ):
        raise ValueError(f"{name} must include a timezone")


def validate_human_metadata(notes: str, rating: int | None, favorite: bool) -> None:
    if not isinstance(notes, str) or type(favorite) is not bool:
        raise ValueError("Notes must be text and favorite must be boolean")
    if rating is not None and (type(rating) is not int or not 0 <= rating <= 5):
        raise ValueError("Rating must be an integer from 0 to 5, or None")


class ValueType(StrEnum):
    NULL = "null"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    REAL = "real"
    STRING = "string"


def scalar_type(value: Scalar) -> ValueType:
    types = {type(None): ValueType.NULL, bool: ValueType.BOOLEAN,
             int: ValueType.INTEGER, float: ValueType.REAL, str: ValueType.STRING}
    if type(value) not in types:
        raise ValueError("Only scalar values are supported in this milestone")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Scalar values must be finite")
    return types[type(value)]


class RootKind(StrEnum):
    ABSOLUTE = "absolute"
    PORTABLE_RELATIVE = "portable_relative"


class Availability(StrEnum):
    AVAILABLE = "available"
    OFFLINE = "offline"
    MISSING = "missing"
    UNVERIFIED = "unverified"


class TrainingStatus(StrEnum):
    UNKNOWN = "unknown"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class MetricSummary(StrEnum):
    UNKNOWN = "unknown"
    FINAL = "final"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    VALUE_AT_STEP = "value_at_step"
    OTHER = "other"


class VerificationStatus(StrEnum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    MISMATCH = "mismatch"


class Confidence(StrEnum):
    EXACT = "exact"
    DERIVED = "derived"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class EntityType(StrEnum):
    RUN = "run"
    PARAMETER = "parameter"
    METRIC = "metric"
    ARTIFACT = "artifact"
    DATASET = "dataset"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True, kw_only=True)
class StorageRoot:
    name: str
    location: str
    id: str = field(default_factory=new_id)
    kind: RootKind = RootKind.ABSOLUTE

    def __post_init__(self) -> None:
        required_text(self.id, "root id")
        required_text(self.name, "root name")
        required_text(self.location, "root location")
        if "\0" in self.location:
            raise ValueError("Root location cannot contain a null byte")
        object.__setattr__(self, "kind", RootKind(self.kind))
        if self.kind is RootKind.ABSOLUTE:
            if not Path(self.location).is_absolute():
                raise ValueError("An absolute root needs an absolute location")
        else:
            object.__setattr__(self, "location", relative_reference(self.location))


@dataclass(frozen=True, slots=True, kw_only=True)
class Parameter:
    key: str
    value: Scalar
    id: str = field(default_factory=new_id)
    category: str | None = None

    def __post_init__(self) -> None:
        required_text(self.id, "parameter id")
        required_text(self.key, "parameter key")
        if self.category is not None:
            required_text(self.category, "parameter category")
        scalar_type(self.value)

    @property
    def value_type(self) -> ValueType:
        return scalar_type(self.value)


@dataclass(frozen=True, slots=True, kw_only=True)
class Metric:
    name: str
    value: float
    id: str = field(default_factory=new_id)
    context: str = "unknown"
    summary: MetricSummary = MetricSummary.UNKNOWN
    step: int | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        required_text(self.id, "metric id")
        required_text(self.name, "metric name")
        required_text(self.context, "metric context")
        _optional_text(self.unit, "metric unit")
        object.__setattr__(self, "summary", MetricSummary(self.summary))
        if type(self.value) not in (int, float) or not math.isfinite(self.value):
            raise ValueError("Metric values must be finite numbers")
        if self.step is not None and (type(self.step) is not int or self.step < 0):
            raise ValueError("Metric step must be a non-negative integer")
        if self.summary is MetricSummary.VALUE_AT_STEP and self.step is None:
            raise ValueError("A value-at-step metric requires a step")


def _hash_pair(algorithm: str | None, value: str | None) -> None:
    if (algorithm is None) != (value is None):
        raise ValueError("Hash algorithm and value must be provided together")
    if algorithm is not None:
        required_text(algorithm, "hash algorithm")
        required_text(value, "hash value")


@dataclass(frozen=True, slots=True, kw_only=True)
class Artifact:
    run_id: str
    storage_root_id: str
    relative_path: str
    display_name: str
    id: str = field(default_factory=new_id)
    category: str = "other"
    size_bytes: int | None = None
    hash_algorithm: str | None = None
    hash_value: str | None = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    verified_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("id", "run_id", "storage_root_id", "display_name", "category"):
            required_text(getattr(self, name), name)
        object.__setattr__(self, "relative_path", relative_reference(self.relative_path))
        object.__setattr__(self, "verification_status", VerificationStatus(self.verification_status))
        if self.size_bytes is not None and (type(self.size_bytes) is not int or self.size_bytes < 0):
            raise ValueError("Artifact size must be a non-negative integer")
        _hash_pair(self.hash_algorithm, self.hash_value)
        if self.verification_status is not VerificationStatus.UNVERIFIED and self.hash_value is None:
            raise ValueError("Hash verification requires a recorded hash")
        aware_timestamp(self.verified_at, "verified_at")


@dataclass(frozen=True, slots=True, kw_only=True)
class Dataset:
    name: str
    id: str = field(default_factory=new_id)
    storage_root_id: str | None = None
    relative_path: str | None = None
    reference: str | None = None
    hash_algorithm: str | None = None
    hash_value: str | None = None
    example_count: int | None = None
    split: str | None = None

    def __post_init__(self) -> None:
        required_text(self.id, "dataset id")
        required_text(self.name, "dataset name")
        _optional_text(self.reference, "dataset reference")
        _optional_text(self.split, "dataset split")
        if (self.storage_root_id is None) != (self.relative_path is None):
            raise ValueError("Dataset root and relative path must be provided together")
        if self.relative_path is not None:
            required_text(self.storage_root_id, "dataset root id")
            object.__setattr__(self, "relative_path", relative_reference(self.relative_path))
        _hash_pair(self.hash_algorithm, self.hash_value)
        if self.example_count is not None and (
            type(self.example_count) is not int or self.example_count < 0
        ):
            raise ValueError("Example count must be a non-negative integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class Tag:
    name: str
    id: str = field(default_factory=new_id)

    def __post_init__(self) -> None:
        required_text(self.id, "tag id")
        required_text(self.name, "tag name")


@dataclass(frozen=True, slots=True, kw_only=True)
class Provenance:
    target_entity: EntityType
    target_id: str
    target_field: str
    source_type: str
    id: str = field(default_factory=new_id)
    source_reference: str | None = None
    original_value: Scalar = None
    candidate_value: Scalar = None
    current_value: Scalar = None
    confidence: Confidence = Confidence.UNKNOWN
    user_modified: bool = False
    detected_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in ("id", "target_id", "target_field", "source_type"):
            required_text(getattr(self, name), name)
        object.__setattr__(self, "target_entity", EntityType(self.target_entity))
        object.__setattr__(self, "confidence", Confidence(self.confidence))
        _optional_text(self.source_reference, "source reference")
        for value in (self.original_value, self.candidate_value, self.current_value):
            scalar_type(value)
        if type(self.user_modified) is not bool:
            raise ValueError("user_modified must be boolean")
        aware_timestamp(self.detected_at, "detected_at", required=True)


@dataclass(frozen=True, slots=True, kw_only=True)
class ImportDiagnostic:
    message: str
    id: str = field(default_factory=new_id)
    severity: Severity = Severity.WARNING
    kind: str = "incomplete_metadata"
    source_reference: str | None = None
    affected_field: str | None = None
    artifact_id: str | None = None
    consequence: str | None = None
    detected_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in ("id", "message", "kind"):
            required_text(getattr(self, name), name)
        object.__setattr__(self, "severity", Severity(self.severity))
        for name in ("source_reference", "affected_field", "artifact_id", "consequence"):
            _optional_text(getattr(self, name), name)
        aware_timestamp(self.detected_at, "detected_at", required=True)


@dataclass(frozen=True, slots=True, kw_only=True)
class Run:
    display_name: str
    source_storage_root_id: str
    source_relative_directory: str
    id: str = field(default_factory=new_id)
    source_trainer: str | None = None
    imported_at: datetime = field(default_factory=utc_now)
    training_started_at: datetime | None = None
    training_finished_at: datetime | None = None
    training_status: TrainingStatus = TrainingStatus.UNKNOWN
    base_model: str | None = None
    model_revision: str | None = None
    method: str | None = None
    notes: str = ""
    rating: int | None = None
    favorite: bool = False
    parameters: tuple[Parameter, ...] = ()
    metrics: tuple[Metric, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    datasets: tuple[Dataset, ...] = ()
    tags: tuple[Tag, ...] = ()
    provenance: tuple[Provenance, ...] = ()
    diagnostics: tuple[ImportDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        for name in ("id", "display_name", "source_storage_root_id"):
            required_text(getattr(self, name), name)
        object.__setattr__(self, "source_relative_directory",
                           relative_reference(self.source_relative_directory))
        object.__setattr__(self, "training_status", TrainingStatus(self.training_status))
        for name in ("source_trainer", "base_model", "model_revision", "method"):
            _optional_text(getattr(self, name), name)
        for name in ("imported_at", "training_started_at", "training_finished_at"):
            aware_timestamp(getattr(self, name), name)
        if self.imported_at is None:
            raise ValueError("imported_at is required")
        if (self.training_started_at is not None and self.training_finished_at is not None
                and self.training_finished_at < self.training_started_at):
            raise ValueError("Training finish precedes training start")
        validate_human_metadata(self.notes, self.rating, self.favorite)
        for name in ("parameters", "metrics", "artifacts", "datasets", "tags", "provenance", "diagnostics"):
            items = tuple(getattr(self, name))
            if len({item.id for item in items}) != len(items):
                raise ValueError(f"Duplicate identities in {name}")
            object.__setattr__(self, name, items)
        if any(artifact.run_id != self.id for artifact in self.artifacts):
            raise ValueError("Every artifact must belong to this run")
        targets = {EntityType.RUN: {self.id}, EntityType.PARAMETER: {p.id for p in self.parameters},
                   EntityType.METRIC: {m.id for m in self.metrics},
                   EntityType.ARTIFACT: {a.id for a in self.artifacts},
                   EntityType.DATASET: {d.id for d in self.datasets}}
        if any(p.target_id not in targets[p.target_entity] for p in self.provenance):
            raise ValueError("Provenance must target this run or one of its related entities")
        if any(d.artifact_id is not None and d.artifact_id not in targets[EntityType.ARTIFACT]
               for d in self.diagnostics):
            raise ValueError("Diagnostic artifact must belong to this run")
