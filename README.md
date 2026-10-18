# RunPup

RunPup is a local-first desktop catalog for completed machine-learning training
runs. It supplies a Run Library, an Import/Review workflow over the bounded
HF Trainer/Unsloth-compatible importer, SQLite persistence, and external file
references. Reviewed imports can also be saved programmatically. Run Details
and comparison are not implemented yet.
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
- `services/import_commit.py`: structured review validation, canonical conversion,
  and one transactional repository handoff with an aggregate reload.
- `ui/`: dark desktop shell, repository-backed Library, and an in-window
  Import/Review workspace using the existing proposal and service APIs.

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
datasets. Multiple source candidates can coexist. Historical training timestamps
and status remain unknown unless explicitly supplied; import time is never
substituted for them.

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

## Reviewed import persistence (M2B1)

```python
from runpup.services.import_commit import ProposedRunCommitService

service = ProposedRunCommitService(repository)  # initialized CatalogRepository
reviewed = proposal.with_user_edit("learning_rate", 0.0002)
validation = service.validate(reviewed)  # no catalog writes
if validation.can_commit:
    saved = service.commit(reviewed)  # returns the reloaded canonical Run
else:
    print(validation.blocking)
```

`CommitValidation` exposes blocking diagnostics, non-blocking warnings, and the
prepared Run. `commit()` validates again and raises `ProposalValidationError`
with that structured result when blocked. Database write failures propagate
after rollback. Missing required values and unresolved material conflicts block;
unknown optional values and unavailable external references do not. Known
symlink escapes and unregistered referenced roots block. No payload is opened
or hashed during commit.

Accepted values become Run fields, typed Parameters, Metrics, or a linked
Dataset. Imported candidates retain their IDs, raw/candidate values, sources,
confidence, timestamps, and modification flags; their current values reflect
the review selection. All manual edits retain their historical values and flags.
Targets are rebound to the corresponding canonical entity. Non-material
conflicts retain their evidence without a selected value. Since Metric requires
a number, unresolved or explicitly cleared metrics retain run-level provenance
under their full `metric:` identity instead of creating a numeric observation.
Resolved metrics preserve name, context, summary, step, and unit, with competing
candidates pointing to the single accepted Metric.

An identified Dataset uses a stable proposal-specific ID. To reuse a known
Dataset, pass `dataset_id=existing_dataset.id` to validation/commit; its complete
definition must match the repository's exact equality contract. Names alone do
not prove reuse. Unnamed dataset facts and incomplete hash pairs remain typed
Parameters with provenance; no name or hash algorithm is guessed. All original
diagnostics and non-blocking validation diagnostics survive reload.

`CatalogRepository.save_import()` registers the unchanged source root, reuses
the existing aggregate writer, and reloads inside one transaction. A failure
rolls back root registration, owned rows, and newly created shared records.
Existing root definitions cannot be silently remapped by an import. Run and
generated child IDs remain stable for retries of the same proposal; repeating
a successful proposal follows existing aggregate upsert behavior and records
the new import time. Duplicate-import product policy remains deferred.
Schema version remains 3.

## Desktop import and review (M2B2)

Open **Import Run**, choose a training output folder, and click **Scan Folder**.
An indeterminate progress indicator marks scan activity. Review cards show the
actual field states: DETECTED, UNKNOWN, CONFLICT, and USER EDITED. Expand
**Sources / edits** to inspect candidates, their source references, confidence,
and manual history. Conflicting evidence starts expanded.

Choose a conflicting candidate from its dropdown, or edit a value and click
**Apply** (Enter also applies text edits). The adjacent undo control discards an
unapplied edit. Empty optional inputs remain unknown. Integers and numbers stay
typed; target modules accept a JSON array of strings or a quoted pattern.
The summary reflects proposal state and the existing validation service.
**Needs attention** counts UNKNOWN fields shown for review; **Unavailable** counts
the remaining UNKNOWN metadata. Both retain complete proposal accounting, while
manual edits are counted separately. Explicit Unsloth adapter metadata preserves
QLoRA and 4-bit loading evidence; consistent explicit 4-bit loading derives an
8-bit flag of false with DERIVED provenance. Other quantization settings stay
unknown without their own evidence.
Warnings remain visible, and Blocking Items explain why Save is unavailable.
Unapplied edits also prevent saving.

**Save Run** uses the existing atomic commit service. On success, the Library
reloads persisted data, selects the saved row, and shows a success notice. The
Import page resets for the next run. A failed save keeps the reviewed proposal
and offers retry without rescanning; no partial import is committed.

Scanning, reference validation, and saving use a worker dedicated to this
workflow. Conflicting controls stay disabled while it runs. Navigating to the
Library keeps the Import page alive; closing waits for the current operation
to finish safely. No worker accesses widgets. Review cards reflow into one
column on narrower windows, and review content and diagnostics can scroll.
Source files remain external and unchanged throughout the UI flow.

## TrainerState completeness and diagnostics (M2C)

Safe `trainer_state.json` data supplies Trainer-reported planned epochs through
`epochs`, plus `observed_epoch`, `global_step`, and `max_steps` as typed
parameters. This preserves the distinction between the planned epoch count and
observed progress; differing safe configuration sources retain their evidence.
Progress and metrics use the snapshot(s) with the highest valid reported global
step. Tied or unorderable states retain competing evidence. Planned epoch/step
facts from older snapshots remain reviewable, and filenames never rank states.

Log summaries also retain observed learning rate, epoch, input-token count,
runtime, and throughput with context, step, and units where applicable. Step is
metric metadata rather than a duplicate metric. A scheduler observation never
fills the configured learning-rate parameter, and reaching an endpoint does not
assign a completed status. Unsupported configuration stays UNKNOWN;
`training_args.bin` remains discovery-only and is never deserialized.

Diagnostic text supports normal mouse/keyboard selection, Ctrl+C, and the
standard context-menu Copy action. The small **Copy** header action copies the
whole current diagnostics panel in stable plain text, with available severity,
source, field, artifact path, message, and consequence. Empty panels disable Copy
and leave the clipboard alone. Copying never changes the proposal.
