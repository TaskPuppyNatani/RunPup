# RunPup

RunPup is a local-first desktop catalog for completed machine-learning training
runs. It supplies the domain model, SQLite persistence, external file references,
a minimal Run Library, and a non-UI HF Trainer/Unsloth-compatible import core.
Import review, canonical import saving, and comparison are not implemented yet.
RunPup does not train models.

## Develop and launch

Python 3.11 or later is required. From the project directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/runpup
```

The equivalent module entry point is `.venv/bin/python -m runpup`.
The default database lives in Qt's platform-specific application-local data
directory. To choose a separate catalog directory:

```sh
.venv/bin/runpup --data-dir /path/to/catalog
```

No account, network service, or sample records are needed at runtime.

## Verify

```sh
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest
```

## Boundaries

- `domain/models.py`: immutable dataclasses, scalar parameters, contextual summary
  metrics, external artifacts/datasets, provenance, and import diagnostics.
- `domain/proposals.py`: reviewable field candidates, conflicts, explicit user
  edits, and proposal blocking state, independent of persistence and Qt.
- `importers/hf_trainer.py`: read-only detection and bounded extraction into
  ProposedRun, using the shared Trainer/PEFT output contracts.
- `persistence/database.py`: SQLite connections and atomic schema migrations.
  `PRAGMA user_version` is explicit; newer schemas and downgrades are rejected.
- `persistence/repository.py`: catalog operations, with one transaction per write
  or consistent aggregate read. Widgets do not issue SQL.
- `services/paths.py`: the only external-reference path constructor. Roots may
  have absolute locations or locations relative to an explicitly supplied
  portable base. There is no remapping or portable-mode UI in this milestone.
- `ui/`: dark desktop shell, empty state, and repository-backed library rows.

Artifact and dataset records are references, not owned payloads. Saving and
deleting catalog records never copies, hashes, moves, or deletes their files.
Availability is checked on demand rather than persisted as a stale assertion.
Relative references use `/` separators and cannot escape a root, including
through symlinks. An unavailable root is OFFLINE; an absent reference within an
accessible root is MISSING; unchecked or indeterminate access is UNVERIFIED.

`CatalogRepository.save_run()` saves a complete Run aggregate, including its
parameters, metrics, artifacts, links, provenance, and diagnostics. Reuse the
same Dataset/Tag IDs across runs. Conflicting definitions of an existing shared
Dataset or Tag are rejected rather than silently changing other runs. Deleting
a run removes its owned rows and links; shared Dataset/Tag records remain.

Provenance targets a run or one of its parameters, metrics, artifacts, or linked
datasets. Multiple source candidates can coexist. No conflict-resolution
behavior is implied. Historical training timestamps and status remain unknown
unless explicitly supplied; import time is never substituted for them.

The images in `examples/ui/` are design references, not application code. The
original TaskPuppy branding asset is included as package data without alteration.

## Import core (M2A)

```python
from runpup.domain.models import StorageRoot
from runpup.importers import HFTrainerImporter

root = StorageRoot(name="Training outputs", location="/absolute/path/to/outputs")
importer = HFTrainerImporter()
detection = importer.detect(root, "my-run")
proposal = importer.scan(root, "my-run")
base_model = proposal.field("base_model")
print(base_model.state, base_model.value, proposal.blocking_fields)
reviewed = proposal.with_user_edit("base_model", "org/confirmed-model")
```

Both methods inspect the selected directory and its immediate checkpoint
directories. Detection requires recognizable Trainer state, PEFT configuration
with a colocated model payload, or training arguments with a model payload.
A generic model config or arbitrary JSON file alone is insufficient.
Compatibility with this importer does not prove a specific source trainer;
PEFT-only exports retain unknown trainer identity unless supported by explicit
framework metadata.

Safe extraction reads `trainer_state.json`, `training_args.json`,
`adapter_config.json`, `config.json`, `dataset_info.json`, and
`dataset_metadata.json`. Supported fields include explicit model/dataset
identity, training and LoRA settings, quantization metadata, and source-reported
historical timestamps/status. Target-module lists are represented as JSON text
in scalar provenance; split counts use `dataset_split_counts.<split>` fields.
Missing values remain UNKNOWN. Effective batch size is derived only when batch
size, accumulation, and an explicit world size all have coherent evidence.
LoRA configuration alone does not establish QLoRA; quantization facts remain
separate, and an explicit QLoRA method is retained as LoRA plus a QLoRA flag.

Proposed fields expose DETECTED, UNKNOWN, CONFLICT, or USER_EDITED. Conflicting
fields have no selected value. Original candidates and all manual edit records
remain in `proposal.provenance`. `has_blockers` reports unresolved material
conflicts or blocking scan diagnostics; it does not approve or save a run.
`proposal.metrics` contains original metric candidates; the associated
`metric:` review fields group their full name/context/summary/step/unit identity.
Log history is reduced to latest observed steps per series and source, retaining
competing values at the selected step. Observations are never relabeled FINAL
or used to infer training completion. Explicit best-metric metadata is retained
with its reported selection semantics.

Safetensors/GGUF payloads, tokenizer files, metadata files, and checkpoint
directories remain external Artifact references. Payloads are not read, copied,
or hashed. Pickled `training_args.bin` is discovered with a diagnostic and is
never deserialized; training scripts are ignored. Symlinks and special files
are skipped. PathResolver checks every reference, and descriptor-relative
no-follow reads protect descendant access; platforms without that capability
return a blocking diagnostic.

Default limits are 4 MiB per JSON file, 16 MiB total JSON reads, 5,000 directory
entries, 256 metric candidates, and 1,000 ordinary diagnostics. Malformed or
oversized optional files produce recoverable diagnostics. Exhausted aggregate
limits produce blocking diagnostics so incomplete evidence cannot look complete.
The source directory is never written, and this API does not access SQLite.
