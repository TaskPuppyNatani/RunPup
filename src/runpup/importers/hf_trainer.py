"""Bounded HF Trainer/PEFT output inspection, including compatible Unsloth exports.

Only the selected directory and its immediate checkpoint-* directories are read.
Known JSON metadata is size-limited; payloads, scripts, and pickle files are not
opened. PathResolver constructs/checks references; descriptor-relative no-follow
access additionally protects reads against descendant symlink replacement.
"""

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
import stat

from runpup.domain.models import (
    Artifact, Availability, Confidence, EntityType, ImportDiagnostic, Metric,
    MetricSummary, Provenance, Severity, StorageRoot, TrainingStatus, new_id,
    relative_reference, utc_now,
)
from runpup.domain.proposals import MATERIAL_FIELDS, FieldState, ProposedField, ProposedRun
from runpup.services.paths import PathResolver


_SAFE_FD_ACCESS = hasattr(os, "O_NOFOLLOW") and os.open in os.supports_dir_fd
_JSON_SOURCES = {
    "trainer_state.json": "trainer_state", "training_args.json": "training_args",
    "adapter_config.json": "adapter_config", "config.json": "model_config",
    "dataset_info.json": "dataset_metadata", "dataset_metadata.json": "dataset_metadata",
}
_TOKENIZERS = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
    "tokenizer.model", "spiece.model",
}
_MODEL_FIELDS = {
    "base_model_name_or_path": ("base_model", "text"),
    "model_name_or_path": ("base_model", "text"), "_name_or_path": ("base_model", "text"),
    "model_revision": ("model_revision", "text"),
    "max_seq_length": ("max_sequence_length", "positive_int"),
    "max_sequence_length": ("max_sequence_length", "positive_int"),
}
_LORA_FIELDS = {
    "r": ("lora_rank", "positive_int"), "lora_r": ("lora_rank", "positive_int"),
    "lora_rank": ("lora_rank", "positive_int"), "lora_alpha": ("lora_alpha", "positive"),
    "lora_dropout": ("lora_dropout", "ratio"), "target_modules": ("lora_target_modules", "modules"),
    "bias": ("lora_bias", "text"), "peft_type": ("method", "method"),
    "unsloth_load_in_4bit": ("load_in_4bit", "boolean"),
    "unsloth_training_method": ("qlora", "qlora_flag"),
}
_TRAINING_FIELDS = {
    **_MODEL_FIELDS,
    "run_name": ("display_name", "text"), "training_method": ("method", "method"),
    "num_train_epochs": ("epochs", "nonnegative"), "learning_rate": ("learning_rate", "nonnegative"),
    "per_device_train_batch_size": ("per_device_batch_size", "positive_int"),
    "gradient_accumulation_steps": ("gradient_accumulation", "positive_int"),
    "world_size": ("parallelism", "positive_int"),
    "effective_batch_size": ("effective_batch_size", "positive_int"),
    "optim": ("optimizer", "text"), "lr_scheduler_type": ("scheduler", "text"),
    "warmup_steps": ("warmup_steps", "nonnegative_int"),
    "warmup_ratio": ("warmup_ratio", "ratio"), "seed": ("seed", "integer"),
    "precision": ("precision", "text"),
    "lora_r": ("lora_rank", "positive_int"), "lora_rank": ("lora_rank", "positive_int"),
    "lora_alpha": ("lora_alpha", "positive"), "lora_dropout": ("lora_dropout", "ratio"),
    "target_modules": ("lora_target_modules", "modules"),
    "dataset_name": ("dataset_identity", "text"), "dataset_hash": ("dataset_hash", "text"),
    "dataset_hash_algorithm": ("dataset_hash_algorithm", "text"),
}
_HISTORY_FIELDS = {
    "training_started_at": ("training_started_at", "timestamp"),
    "training_finished_at": ("training_finished_at", "timestamp"),
    "training_status": ("training_status", "status"),
}
_STATE_PLAN_FIELDS = {
    "num_train_epochs": ("epochs", "nonnegative_int"),
    "max_steps": ("max_steps", "nonnegative_int"),
}
_STATE_PROGRESS_FIELDS = {
    "epoch": ("observed_epoch", "nonnegative"),
    "global_step": ("global_step", "nonnegative_int"),
}
_QUANTIZATION_FIELDS = {
    "load_in_4bit": ("load_in_4bit", "boolean"), "load_in_8bit": ("load_in_8bit", "boolean"),
    "bnb_4bit_quant_type": ("quantization_type", "text"),
    "bnb_4bit_compute_dtype": ("quantization_compute_dtype", "text"),
    "bnb_4bit_use_double_quant": ("double_quantization", "boolean"),
}
_CORE_FIELDS = (
    "display_name", "base_model", "model_revision", "method", "learning_rate", "epochs",
    "per_device_batch_size", "gradient_accumulation", "parallelism", "effective_batch_size",
    "max_sequence_length", "optimizer", "scheduler", "warmup_steps", "warmup_ratio", "seed",
    "precision", "lora_rank", "lora_alpha", "lora_dropout", "lora_target_modules", "lora_bias",
    "load_in_4bit", "load_in_8bit", "quantization_type", "quantization_compute_dtype",
    "double_quantization", "qlora", "dataset_identity", "dataset_reference", "dataset_hash",
    "dataset_hash_algorithm", "dataset_example_count", "dataset_split",
    "training_started_at", "training_finished_at", "training_status",
)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    recognized: bool
    evidence: tuple[str, ...]
    diagnostics: tuple[ImportDiagnostic, ...]
    blocking_diagnostic_ids: tuple[str, ...]


class HFTrainerImporter:
    """One importer for the shared Trainer/PEFT file contracts; no plugin registry."""

    name = "hf_trainer"

    def __init__(self, resolver: PathResolver | None = None, *,
                 max_json_bytes: int = 4 * 1024 * 1024, max_entries: int = 5000,
                 max_total_json_bytes: int = 16 * 1024 * 1024):
        if type(max_json_bytes) is not int or max_json_bytes < 1:
            raise ValueError("max_json_bytes must be a positive integer")
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        if type(max_total_json_bytes) is not int or max_total_json_bytes < 1:
            raise ValueError("max_total_json_bytes must be a positive integer")
        self.resolver = resolver or PathResolver()
        self.max_json_bytes = max_json_bytes
        self.max_entries = max_entries
        self.max_total_json_bytes = max_total_json_bytes

    def detect(self, root: StorageRoot, relative_directory: str = ".") -> DetectionResult:
        scan = _Scan(self, root, relative_directory)
        scan.inspect()
        return scan.detection()

    def scan(self, root: StorageRoot, relative_directory: str = ".") -> ProposedRun:
        scan = _Scan(self, root, relative_directory)
        scan.inspect()
        return scan.proposal()


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("Nonfinite JSON number")
    return value


def _invalid_constant(text):
    raise ValueError("Nonfinite JSON constant")


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _pointer(reference, key):
    escaped = key.replace("~", "~0").replace("/", "~1")
    return reference + ("/" if "#" in reference else "#/") + escaped


class _Scan:
    def __init__(self, importer, root, reference):
        self.importer, self.root, self.reference = importer, root, reference
        self.id, self.scanned_at = new_id(), utc_now()
        self.documents = {}
        self.artifacts = []
        self.diagnostics = []
        self.blocking = []
        self.evidence = []
        self.candidates = {key: [] for key in _CORE_FIELDS}
        self.metrics = []
        self.extra_provenance = []
        self.entries = 0
        self.bytes_read = 0
        self.limited = False

    def diagnose(self, kind, message, *, reference=None, field=None, blocking=False,
                 consequence=None, severity=Severity.WARNING, artifact_id=None):
        if len(self.diagnostics) >= 1000:
            if self.diagnostics[-1].kind == "diagnostic_limit":
                return
            kind, message, blocking = "diagnostic_limit", "Diagnostic limit reached; review is incomplete.", True
        record = ImportDiagnostic(
            kind=kind, message=message, source_reference=reference, affected_field=field,
            consequence=consequence, severity=Severity.ERROR if blocking else severity,
            artifact_id=artifact_id, detected_at=self.scanned_at,
        )
        self.diagnostics.append(record)
        if blocking:
            self.blocking.append(record.id)

    def inspect(self):
        try:
            self.reference = relative_reference(self.reference)
        except ValueError:
            self.diagnose("unsafe_source", "Selected source is not a safe relative reference.",
                          reference=self.reference if isinstance(self.reference, str) else None, blocking=True)
            self.reference = None
            return
        selected = self.importer.resolver.resolve(self.root, self.reference)
        root = self.importer.resolver.resolve(self.root, ".")
        if selected.state is not Availability.AVAILABLE or root.state is not Availability.AVAILABLE:
            self.diagnose("source_unavailable", "Selected source cannot be safely accessed.",
                          reference=self.reference, blocking=True, consequence=selected.state.value)
            return
        if not _SAFE_FD_ACCESS:
            self.diagnose("unsupported_platform", "Safe descriptor-relative scanning is unavailable.",
                          reference=self.reference, blocking=True)
            return
        try:
            with ExitStack() as stack:
                directory = os.open(root.path, os.O_RDONLY | os.O_DIRECTORY)
                stack.callback(os.close, directory)
                if self.reference != ".":
                    for part in self.reference.split("/"):
                        directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                            dir_fd=directory)
                        stack.callback(os.close, directory)
                before = os.fstat(directory)
                self.walk(directory, self.reference, checkpoint=False)
                after = self.importer.resolver.resolve(self.root, self.reference)
                if after.state is not Availability.AVAILABLE:
                    self.diagnose("source_changed", "Selected source disappeared or became unsafe during scanning.",
                                  reference=self.reference, blocking=True)
                else:
                    current = after.path.stat()
                    if (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino):
                        self.diagnose("source_changed", "Selected source was replaced during scanning.",
                                      reference=self.reference, blocking=True)
        except OSError:
            self.diagnose("source_unreadable", "Selected source directory could not be read safely.",
                          reference=self.reference, blocking=True)
        self.recognize()

    def walk(self, directory, reference, *, checkpoint):
        try:
            with os.scandir(directory) as listing:
                entries = []
                for entry in listing:
                    self.entries += 1
                    if self.entries > self.importer.max_entries:
                        self.limited = True
                        self.diagnose("scan_limit", "Directory entry limit reached; scan is incomplete.",
                                      reference=reference, blocking=True)
                        break
                    entries.append(entry.name)
            for name in sorted(entries):
                if self.limited:
                    break
                try:
                    child = relative_reference(f"{reference}/{name}")
                except ValueError:
                    self.diagnose("unsafe_reference", "An entry has an unsupported relative path.",
                                  reference=reference)
                    continue
                try:
                    info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        self.diagnose("symlink_skipped", "Symlink was not followed.", reference=child)
                        continue
                    resolved = self.importer.resolver.resolve(self.root, child)
                    if resolved.state is not Availability.AVAILABLE:
                        self.diagnose("unavailable_entry", "An entry became unavailable or unsafe.",
                                      reference=child, consequence=resolved.state.value)
                        continue
                    current = os.stat(resolved.path, follow_symlinks=False)
                    if (info.st_dev, info.st_ino) != (current.st_dev, current.st_ino):
                        self.diagnose("entry_changed", "An entry was replaced during scanning.", reference=child)
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        if name.startswith("checkpoint-") and name != "checkpoint-":
                            self.artifact(child, name, "checkpoint", None)
                            if not checkpoint:
                                nested = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                                 dir_fd=directory)
                                try:
                                    self.walk(nested, child, checkpoint=True)
                                finally:
                                    os.close(nested)
                        continue
                    if not stat.S_ISREG(info.st_mode):
                        self.diagnose("special_file_skipped", "Non-regular entry was not opened.", reference=child)
                        continue
                    category = self.category(name)
                    if category is None:
                        continue
                    artifact = self.artifact(child, name, category, info.st_size)
                    if name == "training_args.bin":
                        self.diagnose("unsafe_format", "training_args.bin was discovered but not deserialized.",
                                      reference=child, artifact_id=artifact.id,
                                      consequence="Training arguments require safe JSON metadata.")
                    elif name in _JSON_SOURCES:
                        self.read_json(directory, name, child, info)
                except OSError:
                    self.diagnose("entry_unreadable", "An entry could not be read safely.", reference=child)
        except OSError:
            self.diagnose("directory_unreadable", "A required directory listing failed.",
                          reference=reference, blocking=not checkpoint)

    @staticmethod
    def category(name):
        lower = name.lower()
        if lower.endswith(".safetensors"):
            return "adapter" if lower.startswith("adapter") else "model"
        if lower.endswith(".gguf"):
            return "gguf"
        if name in _TOKENIZERS:
            return "tokenizer"
        if name in _JSON_SOURCES or name in {"training_args.bin", "generation_config.json"}:
            return "metadata"
        return None

    def artifact(self, reference, name, category, size):
        artifact = Artifact(run_id=self.id, storage_root_id=self.root.id, relative_path=reference,
                            display_name=name, category=category, size_bytes=size)
        self.artifacts.append(artifact)
        self.extra_provenance.append(Provenance(
            target_entity=EntityType.ARTIFACT, target_id=artifact.id, target_field="relative_path",
            source_type="filesystem", source_reference=reference, original_value=reference,
            candidate_value=reference, confidence=Confidence.EXACT, detected_at=self.scanned_at,
        ))
        return artifact

    def read_json(self, directory, name, reference, info):
        if info.st_size > self.importer.max_json_bytes:
            self.diagnose("malformed_metadata", "JSON metadata exceeded the per-file size limit.",
                          reference=reference, consequence="Other safe evidence remains available.")
            return
        remaining = self.importer.max_total_json_bytes - self.bytes_read
        if info.st_size > remaining:
            self.diagnose("metadata_limit", "Total JSON byte limit reached; review is incomplete.",
                          reference=reference, blocking=True)
            return
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            with os.fdopen(descriptor, "rb") as source:
                opened = os.fstat(source.fileno())
                if not stat.S_ISREG(opened.st_mode) or (
                    opened.st_dev, opened.st_ino
                ) != (info.st_dev, info.st_ino):
                    raise ValueError("Metadata changed before opening")
                raw = source.read(min(self.importer.max_json_bytes, remaining) + 1)
            self.bytes_read += len(raw)
            if self.bytes_read > self.importer.max_total_json_bytes:
                self.diagnose("metadata_limit", "Total JSON byte limit reached; review is incomplete.",
                              reference=reference, blocking=True)
                return
            if len(raw) > self.importer.max_json_bytes:
                raise ValueError("Metadata size limit")
            document = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_strict_object,
                                  parse_float=_finite_float, parse_constant=_invalid_constant)
            if not isinstance(document, dict):
                raise ValueError("Metadata must be an object")
            self.documents[reference] = (name, document)
        except (OSError, ValueError, UnicodeError, RecursionError):
            self.diagnose("malformed_metadata", "JSON metadata was unreadable, invalid, or exceeded the size limit.",
                          reference=reference, consequence="Other safe evidence remains available.")

    def recognize(self):
        payloads = [item for item in self.artifacts if item.category in {"adapter", "model", "gguf"}]
        for reference, (name, document) in sorted(self.documents.items()):
            if name == "trainer_state.json" and (
                isinstance(document.get("log_history"), list)
                or (type(document.get("global_step")) is int and type(document.get("max_steps")) is int)
            ):
                self.evidence.append(reference)
            elif name == "adapter_config.json" and (
                isinstance(document.get("peft_type"), str)
                and document["peft_type"] in {"LORA", "ADALORA", "IA3"}
                and any(item.relative_path.rpartition("/")[0] == reference.rpartition("/")[0]
                        for item in payloads)
            ):
                self.evidence.append(reference)
            elif name == "training_args.json" and payloads:
                signature = {"learning_rate", "num_train_epochs", "per_device_train_batch_size",
                             "gradient_accumulation_steps"} & document.keys()
                if len(signature) >= 2:
                    self.evidence.append(reference)
        if not self.evidence and not self.blocking:
            self.diagnose("unsupported_source", "No supported HF Trainer/PEFT training-output signature was found.",
                          reference=self.reference, blocking=True)

    def detection(self):
        return DetectionResult(bool(self.evidence), tuple(self.evidence),
                               tuple(self.diagnostics), tuple(self.blocking))

    def add(self, key, raw, value, source, reference, confidence=Confidence.EXACT, *,
            entity=EntityType.RUN, target_id=None, target_field=None):
        original = raw if type(raw) in (str, int, float, bool, type(None)) else _json_text(raw)
        self.candidates.setdefault(key, []).append(Provenance(
            target_entity=entity, target_id=target_id or self.id, target_field=target_field or key,
            source_type=source, source_reference=reference, original_value=original,
            candidate_value=value, confidence=confidence, detected_at=self.scanned_at,
        ))

    def extract(self, document, mapping, source, reference):
        for source_key, (key, kind) in mapping.items():
            raw = document.get(source_key)
            if raw is None:
                continue
            pointer = _pointer(reference, source_key)
            try:
                value, confidence = self.interpret(raw, kind)
                self.add(key, raw, value, source, pointer, confidence)
                if kind == "method" and str(raw).upper() == "QLORA":
                    self.add("qlora", raw, True, source, pointer, Confidence.DERIVED)
            except (ValueError, TypeError, OverflowError):
                self.diagnose("invalid_metadata", "A recognized metadata field has an unsupported value.",
                              reference=pointer, field=key)

    @staticmethod
    def interpret(raw, kind):
        confidence = Confidence.EXACT
        if kind in {"text", "method", "timestamp", "status"}:
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("Expected text")
            value = raw
            if kind == "method":
                aliases = {"LORA": "LoRA", "QLORA": "LoRA", "ADALORA": "AdaLoRA",
                           "IA3": "IA3", "FULL": "full", "FULL_FINETUNE": "full"}
                value = aliases.get(raw.upper(), raw)
                confidence = Confidence.DERIVED if value != raw else Confidence.EXACT
            elif kind == "timestamp":
                timestamp = datetime.fromisoformat(raw)
                if timestamp.utcoffset() is None:
                    raise ValueError("Timezone required")
                value = timestamp.astimezone(timezone.utc).isoformat()
            elif kind == "status":
                value = TrainingStatus(raw).value
        elif kind == "boolean":
            if type(raw) is not bool:
                raise ValueError("Expected boolean")
            value = raw
        elif kind == "qlora_flag":
            if not isinstance(raw, str) or raw.lower() not in {"lora", "qlora"}:
                raise ValueError("Expected explicit LoRA or QLoRA training method")
            value = raw.lower() == "qlora"
        elif kind == "modules":
            if isinstance(raw, str) and raw:
                value = _json_text(raw)
            elif isinstance(raw, list) and raw and all(isinstance(v, str) and v for v in raw):
                value = _json_text(sorted(set(raw)))
                confidence = Confidence.DERIVED
            else:
                raise ValueError("Expected module names or regex")
        else:
            if not _number(raw):
                raise ValueError("Expected finite number")
            if kind in {"integer", "positive_int", "nonnegative_int"} and type(raw) is not int:
                raise ValueError("Expected integer")
            if kind in {"positive", "positive_int"} and raw <= 0:
                raise ValueError("Expected positive number")
            if kind in {"nonnegative", "nonnegative_int", "ratio"} and raw < 0:
                raise ValueError("Expected nonnegative number")
            if kind == "ratio" and raw > 1:
                raise ValueError("Expected fraction")
            value = raw
        return value, confidence

    def extract_dataset(self, document, reference, *, prefix=""):
        mapping = {
            "name": ("dataset_identity", "text"), "dataset_name": ("dataset_identity", "text"),
            "reference": ("dataset_reference", "text"), "hash_value": ("dataset_hash", "text"),
            "hash": ("dataset_hash", "text"), "hash_algorithm": ("dataset_hash_algorithm", "text"),
            "example_count": ("dataset_example_count", "nonnegative_int"),
            "num_examples": ("dataset_example_count", "nonnegative_int"), "split": ("dataset_split", "text"),
        }
        source_reference = reference + prefix
        self.extract(document, mapping, "dataset_metadata", source_reference)
        splits = document.get("splits")
        if splits is not None:
            splits_reference = _pointer(source_reference, "splits")
            if not isinstance(splits, dict):
                self.diagnose("unsupported_metadata", "Dataset splits must be a structured object.",
                              reference=splits_reference)
            else:
                for split, metadata in sorted(splits.items()):
                    if isinstance(metadata, dict) and type(metadata.get("num_examples")) is int and metadata["num_examples"] >= 0:
                        pointer = _pointer(_pointer(splits_reference, split), "num_examples")
                        self.add(f"dataset_split_counts.{split}", metadata["num_examples"],
                                 metadata["num_examples"], "dataset_metadata", pointer)
                    else:
                        self.diagnose("unsupported_metadata", "Dataset split count is unavailable or invalid.",
                                      reference=_pointer(splits_reference, split))

    def extract_documents(self):
        states = {ref: doc for ref, (name, doc) in self.documents.items() if name == "trainer_state.json"}
        steps = {ref: doc["global_step"] for ref, doc in states.items()
                 if type(doc.get("global_step")) is int and doc["global_step"] >= 0}
        latest_step = max(steps.values(), default=None)
        # Filenames and mtimes do not rank state. Ties and unorderable sources
        # retain competing evidence; older snapshots are not merged as a series.
        self.selected_trainer_states = {ref for ref in states
                                        if ref not in steps or steps[ref] == latest_step}
        if len(states) > 1:
            self.diagnose("trainer_state_selected", "Trainer progress and metrics use the highest reported global step where available.",
                          reference="; ".join(sorted(self.selected_trainer_states)),
                          consequence="Tied or unorderable states retain their evidence. Older snapshots remain artifact references; all planned epoch/step evidence is retained.",
                          severity=Severity.INFO)
        for reference, (name, document) in sorted(self.documents.items()):
            source = _JSON_SOURCES[name]
            if name == "training_args.json":
                self.extract(document, _TRAINING_FIELDS, source, reference)
                self.extract(document, _HISTORY_FIELDS, source, reference)
                for flag, precision in (("bf16", "bfloat16"), ("fp16", "float16")):
                    if document.get(flag) is True:
                        self.add("precision", True, precision, source, f"{reference}#/{flag}", Confidence.DERIVED)
                dataset = document.get("dataset")
                if isinstance(dataset, str) and dataset:
                    self.add("dataset_identity", dataset, dataset, source, f"{reference}#/dataset")
                elif isinstance(dataset, dict):
                    self.extract_dataset(dataset, reference, prefix="#/dataset")
            elif name == "adapter_config.json":
                self.extract(document, {**_MODEL_FIELDS, **_LORA_FIELDS}, source, reference)
            elif name == "config.json":
                self.extract(document, {**_MODEL_FIELDS, "torch_dtype": ("model_dtype", "text"),
                                        "unsloth_version": ("unsloth_version", "text")}, source, reference)
            elif source == "dataset_metadata":
                self.extract_dataset(document, reference)
            elif name == "trainer_state.json":
                self.extract(document, _HISTORY_FIELDS, source, reference)
                self.extract(document, _STATE_PLAN_FIELDS, source, reference)
                if reference in self.selected_trainer_states:
                    self.extract(document, _STATE_PROGRESS_FIELDS, source, reference)
                    self.extract_metrics(document, reference)
            self.extract(document, _QUANTIZATION_FIELDS, source, reference)
            quantization = document.get("quantization_config")
            if quantization is not None:
                if isinstance(quantization, dict):
                    self.extract(quantization, _QUANTIZATION_FIELDS, source, reference + "#/quantization_config")
                else:
                    self.diagnose("unsupported_metadata", "Quantization metadata is not a structured object.",
                                  reference=reference + "#/quantization_config")

        four_bit = self.proposed_field("load_in_4bit")
        if four_bit.state is FieldState.DETECTED and four_bit.value is True:
            for candidate in four_bit.candidates:
                if candidate.candidate_value is True and candidate.confidence is Confidence.EXACT:
                    self.add("load_in_8bit", {"load_in_4bit": True}, False, "derived",
                             candidate.source_reference, Confidence.DERIVED)

        if not self.candidates["display_name"]:
            name = self.reference.rsplit("/", 1)[-1] if self.reference != "." else self.root.name
            self.add("display_name", name, name, "filesystem", self.reference, Confidence.INFERRED)
        factors = [self.proposed_field(key) for key in
                   ("per_device_batch_size", "gradient_accumulation", "parallelism")]
        if all(item.state is FieldState.DETECTED for item in factors):
            values = {item.key: item.value for item in factors}
            product = math.prod(values.values())
            self.add("effective_batch_size", values, product, "derived",
                     " × ".join(item.candidates[0].source_reference for item in factors), Confidence.DERIVED)
        if not self.candidates["dataset_identity"]:
            self.diagnose("incomplete_metadata", "Dataset identity was not found in safe structured metadata.",
                          field="dataset_identity", consequence="Dataset identity remains unknown.")
        self.extract_best_metrics()

    def extract_metrics(self, document, reference):
        history = document.get("log_history", [])
        if not isinstance(history, list):
            self.diagnose("unsupported_metadata", "Trainer log_history is not an array.", reference=reference)
            return
        # Only the latest observed step per metric/context/summary/unit is kept,
        # including all competing values at that step. This is not a time series.
        latest = {}
        for index, row in enumerate(history):
            if not isinstance(row, dict):
                self.diagnose("invalid_metadata", "A trainer log entry is not an object.",
                              reference=f"{reference}#/log_history/{index}")
                continue
            step = row.get("step")
            if step is not None and (type(step) is not int or step < 0):
                self.diagnose("invalid_metadata", "A trainer log step is invalid.",
                              reference=f"{reference}#/log_history/{index}/step")
                continue
            for key, value in sorted(row.items()):
                if key in {"loss", "grad_norm", "learning_rate", "num_input_tokens_seen", "epoch"}:
                    name, context = key, "train"
                elif key.startswith("eval_"):
                    name, context = key[5:], "validation"
                elif key.startswith("train_"):
                    name, context = key[6:], "train"
                else:
                    continue
                if not name.strip():
                    self.diagnose("invalid_metadata", "A logged metric has no name.",
                                  reference=f"{reference}#/log_history/{index}")
                    continue
                if not _number(value):
                    self.diagnose("invalid_metadata", "A logged metric is not a finite number.",
                                  reference=f"{reference}#/log_history/{index}/{key}")
                    continue
                summary = MetricSummary.OTHER if key == "train_loss" else (
                    MetricSummary.VALUE_AT_STEP if step is not None else MetricSummary.UNKNOWN
                )
                unit = {"runtime": "seconds", "samples_per_second": "samples/second",
                        "steps_per_second": "steps/second", "tokens_per_second": "tokens/second",
                        "num_input_tokens_seen": "tokens", "epoch": "epochs"}.get(name)
                group = (name, context, summary, unit)
                if group not in latest and len(latest) >= 256:
                    if not any(item.kind == "metric_limit" for item in self.diagnostics):
                        self.diagnose("metric_limit", "Metric identity limit reached; review is incomplete.",
                                      reference=reference, blocking=True)
                    continue
                previous = latest.get(group)
                order = step if step is not None else index
                evidence = (value, step, _pointer(f"{reference}#/log_history/{index}", key))
                if previous is None or order > previous[0]:
                    latest[group] = (order, [evidence])
                elif order == previous[0]:
                    previous[1].append(evidence)
        for (name, context, summary, unit), (_, observations) in latest.items():
            for value, step, pointer in observations:
                self.metric(name, context, summary, value, step, unit, pointer)
        if history:
            self.diagnose("history_summarized", "Log history was reduced to latest observations per metric.",
                          reference=reference, consequence="Last observations do not establish final results or completion.",
                          severity=Severity.INFO)

    def metric(self, name, context, summary, value, step, unit, pointer, confidence=Confidence.EXACT):
        if len(self.metrics) >= 256:
            if not any(item.kind == "metric_limit" for item in self.diagnostics):
                self.diagnose("metric_limit", "Metric candidate limit reached; review is incomplete.",
                              reference=pointer, blocking=True)
            return
        metric = Metric(name=name, context=context, summary=summary, value=value, step=step, unit=unit)
        self.metrics.append(metric)
        key = "metric:" + _json_text([name, context, summary.value, step, unit])
        self.add(key, value, value, "trainer_state", pointer, confidence, entity=EntityType.METRIC,
                 target_id=metric.id, target_field="value")

    def extract_best_metrics(self):
        arguments = [(reference, document) for reference, (name, document) in self.documents.items()
                     if name == "training_args.json"]
        for reference, (filename, document) in sorted(self.documents.items()):
            if (filename != "trainer_state.json" or reference not in self.selected_trainer_states
                    or document.get("best_metric") is None):
                continue
            if not _number(document["best_metric"]):
                self.diagnose("invalid_metadata", "Best metric value is not a finite number.",
                              reference=reference + "#/best_metric")
                continue
            matching = [(ref, args) for ref, args in arguments
                        if ref.rpartition("/")[0] == reference.rpartition("/")[0]]
            if not matching:
                self.add("best_metric_unidentified", document["best_metric"], document["best_metric"],
                         "trainer_state", reference + "#/best_metric")
                self.diagnose("incomplete_metadata", "Best metric value has no supported metric-name metadata.",
                              reference=reference + "#/best_metric")
                continue
            for args_ref, args in matching:
                name = args.get("metric_for_best_model")
                value, step = document["best_metric"], document.get("best_global_step")
                if not isinstance(name, str) or not name:
                    self.add("best_metric_unidentified", value, value,
                             "trainer_state", reference + "#/best_metric")
                    self.diagnose("invalid_metadata", "Best metric name or value is unavailable or invalid.",
                                  reference=reference + "#/best_metric")
                    continue
                if step is not None and (type(step) is not int or step < 0):
                    self.diagnose("invalid_metadata", "Best metric step is invalid.",
                                  reference=reference + "#/best_global_step")
                    step = None
                direction = args.get("greater_is_better")
                summary = (MetricSummary.MAXIMUM if direction is True else
                           MetricSummary.MINIMUM if direction is False else MetricSummary.OTHER)
                name = name.removeprefix("eval_")
                if not name.strip():
                    self.add("best_metric_unidentified", value, value,
                             "trainer_state", reference + "#/best_metric")
                    self.diagnose("invalid_metadata", "Best metric has no name.",
                                  reference=args_ref + "#/metric_for_best_model")
                    continue
                self.metric(name, "validation", summary, value, step, None,
                            reference + "#/best_metric; " + args_ref + "#/metric_for_best_model; "
                            + args_ref + "#/greater_is_better", Confidence.DERIVED)

    def proposed_field(self, key):
        return ProposedField(key=key, candidates=tuple(self.candidates[key]), material=key in MATERIAL_FIELDS)

    def proposal(self):
        if self.evidence:
            self.extract_documents()
        trainer = None
        if self.evidence:
            if any(self.documents[reference][0] == "trainer_state.json" for reference in self.evidence):
                trainer = "huggingface_trainer"
            unsloth_sources = [_pointer(ref, "unsloth_version") for ref, (_, doc) in self.documents.items()
                               if isinstance(doc.get("unsloth_version"), str) and doc["unsloth_version"]]
            if unsloth_sources:
                trainer = "unsloth"
            if trainer is not None:
                self.add("source_trainer", list(self.evidence), trainer, "inferred",
                         "; ".join((*self.evidence, *unsloth_sources)), Confidence.DERIVED)
        return ProposedRun(
            id=self.id, scanned_at=self.scanned_at, source_storage_root=self.root,
            source_relative_directory=self.reference,
            detected_importer=self.importer.name if self.evidence else None, source_trainer=trainer,
            detection_evidence=tuple(self.evidence),
            fields=tuple(self.proposed_field(key) for key in sorted(self.candidates)),
            metrics=tuple(self.metrics), artifacts=tuple(self.artifacts),
            extra_provenance=tuple(self.extra_provenance), diagnostics=tuple(self.diagnostics),
            blocking_diagnostic_ids=tuple(self.blocking),
        )
