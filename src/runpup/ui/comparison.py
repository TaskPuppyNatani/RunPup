"""Pure, saved-value comparison. No importer, filesystem, or Qt operations."""

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
import json

from runpup.domain.models import EntityType, TrainingStatus
from runpup.ui.run_details import DATASET_FIELDS, LORA_FIELDS, TRAINING_FIELDS


class ComparisonStatus(StrEnum):
    DIFFERENT = "DIFFERENT"
    ONLY_A = "ONLY A"
    ONLY_B = "ONLY B"
    BOTH_UNKNOWN = "BOTH UNKNOWN"
    SAME = "SAME"


SECTIONS = ("Identity", "Model", "Dataset", "Training", "LoRA / QLoRA",
            "Results / Metrics", "Artifacts", "Human Metadata")
_PRIORITY = {ComparisonStatus.DIFFERENT: 0, ComparisonStatus.ONLY_A: 1,
             ComparisonStatus.ONLY_B: 1, ComparisonStatus.BOTH_UNKNOWN: 2, ComparisonStatus.SAME: 3}


def value_status(a, b):
    if a is None or b is None:
        return (ComparisonStatus.BOTH_UNKNOWN if a is None and b is None else
                ComparisonStatus.ONLY_B if a is None else ComparisonStatus.ONLY_A)
    # bool is a numeric subclass in Python, but False is not a configured zero.
    equal = (type(a) is bool) == (type(b) is bool) and a == b
    return ComparisonStatus.SAME if equal else ComparisonStatus.DIFFERENT


@dataclass(frozen=True, slots=True)
class Cell:
    value: object = None
    evidence: tuple = ()


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    section: str
    key: tuple
    label: str
    a: Cell
    b: Cell

    @property
    def status(self):
        return value_status(self.a.value, self.b.value)


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    name: str
    kind: str
    size_bytes: int | None
    hash_algorithm: str | None
    hash_value: str | None
    verification: str
    verified_at: object


@dataclass(frozen=True, slots=True)
class Comparison:
    run_a: object
    run_b: object
    rows: tuple[ComparisonRow, ...]

    @property
    def counts(self):
        return Counter(row.status for row in self.rows)

    def visible_rows(self, differences_only=False):
        return tuple(row for row in self.rows if not differences_only or row.status in {
            ComparisonStatus.DIFFERENT, ComparisonStatus.ONLY_A, ComparisonStatus.ONLY_B})

    def swapped(self):
        return compare_runs(self.run_b, self.run_a)


class _RunValues:
    def __init__(self, run):
        self.run = run
        self.parameters = {(p.key, p.category): p for p in run.parameters}
        self.evidence = {}
        self.targets = {}
        for record in run.provenance:
            self.evidence.setdefault((record.target_entity, record.target_id, record.target_field), []).append(record)
            self.targets.setdefault((record.target_entity, record.target_id), []).append(record)

    def cell(self, value, entity, target, field=None):
        records = (self.evidence.get((entity, target, field), ()) if field is not None else
                   self.targets.get((entity, target), ()))
        return Cell(value, tuple(records))

    def field(self, key):
        value = getattr(self.run, key)
        if key == "training_status" and value is TrainingStatus.UNKNOWN:
            value = None
        return self.cell(value, EntityType.RUN, self.run.id, key)

    def parameter(self, key, category=None):
        parameter = self.parameters.get((key, category))
        if parameter is None:
            return self.cell(None, EntityType.RUN, self.run.id, key)
        return self.cell(parameter.value, EntityType.PARAMETER, parameter.id, "value")

    def dataset(self, key, attribute, dataset=None, *, allow_parameter=True):
        cell = Cell()
        if dataset is not None:
            cell = self.cell(getattr(dataset, attribute), EntityType.DATASET, dataset.id, attribute)
            if cell.value is not None or not allow_parameter or (key, None) not in self.parameters:
                return cell
        if not allow_parameter:
            return cell
        parameter = self.parameter(key)
        return Cell(parameter.value, cell.evidence + parameter.evidence)

    def metric(self, identity, metric=None):
        if metric is not None:
            return self.cell(metric.value, EntityType.METRIC, metric.id, "value")
        # The sealed commit service retains cleared metric evidence under this full identity.
        field = "metric:" + json.dumps(identity, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return self.cell(None, EntityType.RUN, self.run.id, field)


def metric_identity(metric):
    return (metric.name, metric.context, metric.summary.value, metric.step, metric.unit)


def compare_runs(run_a, run_b):
    if run_a.id == run_b.id:
        raise ValueError("Choose two different saved runs.")
    a, b = _RunValues(run_a), _RunValues(run_b)
    rows = []

    def add(section, key, caption, left, right):
        rows.append(ComparisonRow(section, key, caption, left, right))

    for section, fields in (
        ("Identity", (("display_name", "Run name"), ("source_trainer", "Trainer / source"),
                      ("method", "Method"), ("imported_at", "Imported"),
                      ("training_started_at", "Training started"), ("training_finished_at", "Training finished"),
                      ("training_status", "Training status"))),
        ("Model", (("base_model", "Base model"), ("model_revision", "Model revision"))),
    ):
        for key, caption in fields:
            add(section, ("run", key), caption, a.field(key), b.field(key))

    common_parameters = set()
    for section, fields in (("Training", TRAINING_FIELDS), ("LoRA / QLoRA", LORA_FIELDS)):
        for key, caption in fields:
            common_parameters.add((key, None))
            add(section, ("parameter", key, None), caption, a.parameter(key), b.parameter(key))

    dataset_fields = (*DATASET_FIELDS, ("dataset_storage_root", "Storage root ID", "storage_root_id"),
                      ("dataset_relative_path", "Relative path", "relative_path"))
    consumed_a, consumed_b = set(common_parameters), set(common_parameters)
    if len(run_a.datasets) <= 1 and len(run_b.datasets) <= 1:
        da = run_a.datasets[0] if run_a.datasets else None
        db = run_b.datasets[0] if run_b.datasets else None
        for key, caption, attribute in dataset_fields:
            add("Dataset", ("dataset", key), caption, a.dataset(key, attribute, da), b.dataset(key, attribute, db))
            for dataset, consumed in ((da, consumed_a), (db, consumed_b)):
                if dataset is None or getattr(dataset, attribute) is None:
                    consumed.add((key, None))
    else:
        # Multiple linked datasets align only by persisted shared identity, never position/name guesses.
        left, right = ({d.id: d for d in run.datasets} for run in (run_a, run_b))
        for dataset_id in sorted(left.keys() | right.keys()):
            for key, caption, attribute in dataset_fields:
                ca = a.dataset(key, attribute, left[dataset_id], allow_parameter=False) if dataset_id in left else Cell()
                cb = b.dataset(key, attribute, right[dataset_id], allow_parameter=False) if dataset_id in right else Cell()
                add("Dataset", ("dataset", dataset_id, key), f"{dataset_id} · {caption}", ca, cb)

    remaining_a, remaining_b = a.parameters.keys() - consumed_a, b.parameters.keys() - consumed_b
    for key, category in sorted(remaining_a | remaining_b,
                                key=lambda identity: (identity[0], identity[1] is not None, identity[1] or "")):
        section = ("LoRA / QLoRA" if key.startswith(("lora_", "quantization_", "load_in_")) else
                   "Dataset" if key.startswith("dataset_") else "Model" if key.startswith("model_") else "Training")
        caption = f"Saved parameter · {key}" + (f" [{category}]" if category else "")
        add(section, ("parameter", key, category), caption,
            a.parameter(key, category) if (key, category) in remaining_a else Cell(),
            b.parameter(key, category) if (key, category) in remaining_b else Cell())

    left, right = ({metric_identity(m): m for m in run.metrics} for run in (run_a, run_b))
    for identity in sorted(left.keys() | right.keys(), key=lambda k: (
            k[0], k[1], k[2], k[3] is not None, k[3] or 0, k[4] is not None, k[4] or "")):
        name, context, summary, step, unit = identity
        label = f"{name} · {context} · {summary}\nStep: {step if step is not None else 'Unknown'} · Unit: {unit if unit is not None else 'Unknown'}"
        cells = [values.metric(identity, metrics.get(identity)) for values, metrics in ((a, left), (b, right))]
        add("Results / Metrics", ("metric", *identity), label, *cells)

    left, right = ({(item.storage_root_id, item.relative_path): item for item in run.artifacts}
                   for run in (run_a, run_b))
    for identity in sorted(left.keys() | right.keys()):
        cells = []
        for values, artifacts in ((a, left), (b, right)):
            artifact = artifacts.get(identity)
            cells.append(values.cell(ArtifactMetadata(
                artifact.display_name, artifact.category, artifact.size_bytes, artifact.hash_algorithm,
                artifact.hash_value, artifact.verification_status.value, artifact.verified_at),
                EntityType.ARTIFACT, artifact.id) if artifact else Cell())
        add("Artifacts", ("artifact", *identity), f"{identity[1]}\nRoot ID: {identity[0]}", *cells)

    for key, caption in (("favorite", "Favorite"), ("rating", "Rating"), ("notes", "Notes")):
        add("Human Metadata", ("run", key), caption, a.field(key), b.field(key))
    add("Human Metadata", ("tags",), "Tags (unordered)",
        Cell(frozenset(t.name for t in run_a.tags)), Cell(frozenset(t.name for t in run_b.tags)))

    ordered = tuple(row for section in SECTIONS for row in sorted(
        (r for r in rows if r.section == section), key=lambda row: _PRIORITY[row.status]))
    return Comparison(run_a, run_b, ordered)


def load_comparison(repository, run_a_id, run_b_id):
    if run_a_id == run_b_id:
        raise ValueError("Choose two different saved runs.")
    a, b = repository.get_run(run_a_id), repository.get_run(run_b_id)
    if a is None or b is None:
        raise KeyError(f"Run {'A' if a is None else 'B'} is no longer in the catalog. Choose another saved run.")
    return compare_runs(a, b)
