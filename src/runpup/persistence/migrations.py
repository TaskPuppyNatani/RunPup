"""Ordered, additive schema migrations; applied atomically by Database."""

SCHEMA_VERSION = 3

# A tuple of statements avoids sqlite3.executescript's implicit transaction commit.
MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """CREATE TABLE storage_root (
            id TEXT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL,
            location TEXT NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('absolute', 'portable_relative'))
        )""",
        """CREATE TABLE run (
            id TEXT PRIMARY KEY NOT NULL,
            display_name TEXT NOT NULL,
            source_storage_root_id TEXT NOT NULL REFERENCES storage_root(id) ON DELETE RESTRICT,
            source_relative_directory TEXT NOT NULL,
            source_trainer TEXT,
            imported_at TEXT NOT NULL,
            training_started_at TEXT,
            training_finished_at TEXT,
            training_status TEXT NOT NULL DEFAULT 'unknown'
                CHECK(training_status IN ('unknown', 'completed', 'failed', 'interrupted')),
            base_model TEXT,
            model_revision TEXT,
            method TEXT,
            notes TEXT NOT NULL DEFAULT '',
            rating INTEGER CHECK(rating BETWEEN 0 AND 5),
            favorite INTEGER NOT NULL DEFAULT 0 CHECK(favorite IN (0, 1))
        )""",
        "CREATE INDEX run_imported ON run(imported_at DESC, id)",
        "CREATE INDEX run_source_root ON run(source_storage_root_id)",
        """CREATE TABLE parameter (
            id TEXT PRIMARY KEY NOT NULL,
            run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
            position INTEGER NOT NULL CHECK(position >= 0),
            key TEXT NOT NULL,
            value_json TEXT NOT NULL,
            value_type TEXT NOT NULL CHECK(value_type IN ('null', 'boolean', 'integer', 'real', 'string')),
            category TEXT,
            UNIQUE(run_id, id)
        )""",
        "CREATE UNIQUE INDEX parameter_identity ON parameter(run_id, key, COALESCE(category, ''))",
        """CREATE TABLE metric (
            id TEXT PRIMARY KEY NOT NULL,
            run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
            position INTEGER NOT NULL CHECK(position >= 0),
            name TEXT NOT NULL,
            context TEXT NOT NULL DEFAULT 'unknown',
            summary TEXT NOT NULL DEFAULT 'unknown'
                CHECK(summary IN ('unknown', 'final', 'minimum', 'maximum', 'value_at_step', 'other')),
            value REAL NOT NULL,
            step INTEGER CHECK(step >= 0),
            unit TEXT,
            CHECK(summary != 'value_at_step' OR step IS NOT NULL),
            UNIQUE(run_id, id)
        )""",
        "CREATE UNIQUE INDEX metric_identity ON metric(run_id, name, context, summary, COALESCE(step, -1))",
        """CREATE TABLE artifact (
            id TEXT PRIMARY KEY NOT NULL,
            run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
            position INTEGER NOT NULL CHECK(position >= 0),
            storage_root_id TEXT NOT NULL REFERENCES storage_root(id) ON DELETE RESTRICT,
            relative_path TEXT NOT NULL,
            display_name TEXT NOT NULL,
            category TEXT NOT NULL,
            size_bytes INTEGER CHECK(size_bytes >= 0),
            hash_algorithm TEXT,
            hash_value TEXT,
            verification_status TEXT NOT NULL DEFAULT 'unverified'
                CHECK(verification_status IN ('unverified', 'verified', 'mismatch')),
            verified_at TEXT,
            CHECK((hash_algorithm IS NULL) = (hash_value IS NULL)),
            CHECK(verification_status = 'unverified' OR hash_value IS NOT NULL),
            UNIQUE(run_id, id),
            UNIQUE(run_id, storage_root_id, relative_path)
        )""",
        "CREATE INDEX artifact_root ON artifact(storage_root_id)",
        """CREATE TABLE dataset (
            id TEXT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL,
            storage_root_id TEXT REFERENCES storage_root(id) ON DELETE RESTRICT,
            relative_path TEXT,
            reference TEXT,
            hash_algorithm TEXT,
            hash_value TEXT,
            example_count INTEGER CHECK(example_count >= 0),
            split TEXT,
            CHECK((storage_root_id IS NULL) = (relative_path IS NULL)),
            CHECK((hash_algorithm IS NULL) = (hash_value IS NULL))
        )""",
        "CREATE INDEX dataset_root ON dataset(storage_root_id)",
        """CREATE TABLE run_dataset (
            run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
            dataset_id TEXT NOT NULL REFERENCES dataset(id) ON DELETE RESTRICT,
            position INTEGER NOT NULL CHECK(position >= 0),
            PRIMARY KEY(run_id, dataset_id)
        )""",
        "CREATE INDEX dataset_runs ON run_dataset(dataset_id)",
        """CREATE TABLE tag (
            id TEXT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL UNIQUE
        )""",
        """CREATE TABLE run_tag (
            run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
            tag_id TEXT NOT NULL REFERENCES tag(id) ON DELETE CASCADE,
            position INTEGER NOT NULL CHECK(position >= 0),
            PRIMARY KEY(run_id, tag_id)
        )""",
        "CREATE INDEX tag_runs ON run_tag(tag_id)",
    ),
    2: (
        """CREATE TABLE provenance (
            id TEXT PRIMARY KEY NOT NULL,
            run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
            position INTEGER NOT NULL CHECK(position >= 0),
            target_entity TEXT NOT NULL CHECK(target_entity IN ('run', 'parameter', 'metric', 'artifact', 'dataset')),
            target_field TEXT NOT NULL,
            parameter_id TEXT,
            metric_id TEXT,
            artifact_id TEXT,
            dataset_id TEXT,
            source_type TEXT NOT NULL,
            source_reference TEXT,
            original_json TEXT NOT NULL,
            candidate_json TEXT NOT NULL,
            current_json TEXT NOT NULL,
            confidence TEXT NOT NULL CHECK(confidence IN ('exact', 'derived', 'inferred', 'unknown')),
            user_modified INTEGER NOT NULL CHECK(user_modified IN (0, 1)),
            detected_at TEXT NOT NULL,
            FOREIGN KEY(run_id, parameter_id) REFERENCES parameter(run_id, id) ON DELETE CASCADE,
            FOREIGN KEY(run_id, metric_id) REFERENCES metric(run_id, id) ON DELETE CASCADE,
            FOREIGN KEY(run_id, artifact_id) REFERENCES artifact(run_id, id) ON DELETE CASCADE,
            FOREIGN KEY(run_id, dataset_id) REFERENCES run_dataset(run_id, dataset_id) ON DELETE CASCADE,
            CHECK(
                (target_entity = 'run' AND parameter_id IS NULL AND metric_id IS NULL AND artifact_id IS NULL AND dataset_id IS NULL)
                OR (target_entity = 'parameter' AND parameter_id IS NOT NULL AND metric_id IS NULL AND artifact_id IS NULL AND dataset_id IS NULL)
                OR (target_entity = 'metric' AND parameter_id IS NULL AND metric_id IS NOT NULL AND artifact_id IS NULL AND dataset_id IS NULL)
                OR (target_entity = 'artifact' AND parameter_id IS NULL AND metric_id IS NULL AND artifact_id IS NOT NULL AND dataset_id IS NULL)
                OR (target_entity = 'dataset' AND parameter_id IS NULL AND metric_id IS NULL AND artifact_id IS NULL AND dataset_id IS NOT NULL)
            )
        )""",
        "CREATE INDEX provenance_run_field ON provenance(run_id, target_entity, target_field)",
        """CREATE TABLE import_diagnostic (
            id TEXT PRIMARY KEY NOT NULL,
            run_id TEXT NOT NULL REFERENCES run(id) ON DELETE CASCADE,
            position INTEGER NOT NULL CHECK(position >= 0),
            severity TEXT NOT NULL CHECK(severity IN ('info', 'warning', 'error')),
            kind TEXT NOT NULL,
            source_reference TEXT,
            affected_field TEXT,
            artifact_id TEXT,
            message TEXT NOT NULL,
            consequence TEXT,
            detected_at TEXT NOT NULL,
            FOREIGN KEY(run_id, artifact_id) REFERENCES artifact(run_id, id) ON DELETE CASCADE
        )""",
        "CREATE INDEX diagnostic_run ON import_diagnostic(run_id)",
    ),
    3: (
        "DROP INDEX metric_identity",
        """CREATE UNIQUE INDEX metric_identity ON metric(
            run_id, name, context, summary, COALESCE(step, -1), COALESCE(unit, X'')
        )""",
    ),
}
