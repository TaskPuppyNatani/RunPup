# RunPup

RunPup is a local-first desktop catalog for completed machine-learning training
runs. This first foundation pass supplies the domain model, SQLite persistence,
external file references, and a minimal Run Library. Import and comparison are
not implemented yet. RunPup does not train models.

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
