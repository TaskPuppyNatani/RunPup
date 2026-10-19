"""Read-only presentation of a saved aggregate; no importer or payload access."""

from dataclasses import dataclass

from runpup.domain.models import Availability, EntityType, Run, StorageRoot
from runpup.services.paths import PathResolution, PathResolver


TRAINING_FIELDS = (
    ("epochs", "Planned epochs"), ("observed_epoch", "Observed epoch"),
    ("global_step", "Global step"), ("max_steps", "Max steps"),
    ("learning_rate", "Configured learning rate"),
    ("per_device_batch_size", "Per-device batch size"),
    ("gradient_accumulation", "Gradient accumulation"),
    ("effective_batch_size", "Effective batch size"),
    ("max_sequence_length", "Max sequence length"), ("optimizer", "Optimizer"),
    ("scheduler", "Scheduler"), ("warmup_steps", "Warmup steps"),
    ("warmup_ratio", "Warmup ratio"), ("seed", "Seed"),
    ("precision", "Precision"), ("parallelism", "Parallelism"),
)
LORA_FIELDS = (
    ("lora_rank", "LoRA rank (r)"), ("lora_alpha", "LoRA alpha"),
    ("lora_dropout", "LoRA dropout"), ("lora_target_modules", "Target modules"),
    ("qlora", "QLoRA"), ("load_in_4bit", "Load in 4-bit"),
    ("load_in_8bit", "Load in 8-bit"), ("quantization_type", "Quantization type"),
    ("quantization_compute_dtype", "Compute dtype"),
    ("double_quantization", "Double quantization"),
)
DATASET_FIELDS = (
    ("dataset_identity", "Identity", "name"), ("dataset_reference", "Reference", "reference"),
    ("dataset_split", "Split", "split"), ("dataset_example_count", "Examples", "example_count"),
    ("dataset_hash", "Hash", "hash_value"), ("dataset_hash_algorithm", "Hash algorithm", "hash_algorithm"),
)


def dataset_fields(run, dataset=None):
    # Unnamed dataset facts are intentionally persisted as parameters at import.
    parameters = {item.key: item.value for item in run.parameters}
    return tuple((caption, getattr(dataset, attribute) if dataset else parameters.get(key))
                 for key, caption, attribute in DATASET_FIELDS)


def parameter_sections(run):
    """Common unknowns stay visible; additional persisted parameters stay inspectable."""
    parameters = {item.key: item.value for item in run.parameters}
    sections = []
    common = set()
    for title, fields in (("Training", TRAINING_FIELDS), ("LoRA / QLoRA", LORA_FIELDS)):
        sections.append((title, tuple((caption, parameters.get(key)) for key, caption in fields)))
        common.update(key for key, _ in fields)
    common.update(key for key, _, _ in DATASET_FIELDS
                  if not run.datasets or parameters.get(key) is None)
    additional = tuple((key, parameters[key]) for key in sorted(parameters.keys() - common))
    if additional:
        sections.append(("Additional saved parameters", additional))
    return tuple(sections)


def ordered_metrics(run):
    important = {name: i for i, name in enumerate((
        "loss", "train_loss", "eval_loss", "train_runtime", "eval_runtime",
        "train_samples_per_second", "train_steps_per_second", "num_input_tokens_seen",
    ))}
    return tuple(sorted(run.metrics, key=lambda m: (
        important.get(m.name, len(important)), m.name, m.context, m.summary.value,
        (m.step is not None, m.step or 0), m.unit or "", m.id,
    )))


def evidence_groups(run):
    """Label every persisted target without collapsing candidates or edit history."""
    targets = {(EntityType.RUN, run.id): "Run"}
    for entity, items, caption in (
        (EntityType.PARAMETER, run.parameters, lambda p: p.key),
        (EntityType.METRIC, run.metrics, lambda m:
            f"{m.name} / {m.context} / {m.summary.value} / step {m.step} / {m.unit or 'no unit'}"),
        (EntityType.ARTIFACT, run.artifacts, lambda a: a.relative_path),
        (EntityType.DATASET, run.datasets, lambda d: d.name),
    ):
        targets.update(((entity, item.id), caption(item)) for item in items)
    groups = {}
    for record in run.provenance:
        key = (record.target_entity, record.target_id, record.target_field)
        groups.setdefault(key, []).append(record)
    return tuple((f"{entity.value.capitalize()} · {targets[entity, target]} · {field}", tuple(records))
                 for (entity, target, field), records in groups.items())


@dataclass(frozen=True, slots=True)
class ReferenceView:
    root: StorageRoot | None
    relative_path: str
    resolution: PathResolution


@dataclass(frozen=True, slots=True)
class RunDetails:
    run: Run
    source: ReferenceView
    artifacts: tuple[ReferenceView, ...]
    datasets: tuple[ReferenceView | None, ...]


def load_run_details(repository, run_id, *, resolver=None):
    """Load fresh canonical data, then check references only through PathResolver."""
    run = repository.get_run(run_id)
    if run is None:
        raise KeyError("This saved run is no longer in the catalog.")
    resolver = resolver if resolver is not None else PathResolver()
    roots = {}

    def reference(root_id, relative_path):
        if root_id not in roots:
            roots[root_id] = repository.get_storage_root(root_id)
        root = roots[root_id]
        resolution = (resolver.resolve(root, relative_path) if root is not None else
                      PathResolution(None, Availability.UNVERIFIED, None, None,
                                     "Storage root is not registered in the catalog"))
        return ReferenceView(root, relative_path, resolution)

    return RunDetails(
        run, reference(run.source_storage_root_id, run.source_relative_directory),
        tuple(reference(a.storage_root_id, a.relative_path) for a in run.artifacts),
        tuple(reference(d.storage_root_id, d.relative_path) if d.storage_root_id else None
              for d in run.datasets),
    )
