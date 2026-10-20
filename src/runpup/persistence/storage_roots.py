"""Catalog-only storage root summaries; counts describe references, not files."""

from dataclasses import dataclass

from runpup.domain.models import StorageRoot


@dataclass(frozen=True, slots=True)
class StorageRootSummary:
    root: StorageRoot
    artifact_count: int
    run_count: int
    dataset_count: int

    @property
    def in_use(self) -> bool:
        return bool(self.artifact_count or self.run_count or self.dataset_count)

    @property
    def dependencies(self) -> str:
        return ", ".join(f"{count} {caption}{'' if count == 1 else 's'}" for count, caption in (
            (self.artifact_count, "artifact reference"), (self.run_count, "run source"),
            (self.dataset_count, "dataset"),
        ))


class RootInUse(ValueError):
    def __init__(self, summary: StorageRootSummary):
        super().__init__(f"This root is still used by {summary.dependencies}. "
                         "Remap it instead. No catalog records or external files were changed.")


# Aggregations remain independent so joining several dependency types cannot multiply counts.
ROOT_SUMMARIES_SQL = """
    SELECT r.*, COALESCE(a.n, 0) AS artifact_count,
           COALESCE(s.n, 0) AS run_count, COALESCE(d.n, 0) AS dataset_count
    FROM storage_root r
    LEFT JOIN (SELECT storage_root_id, count(*) n FROM artifact GROUP BY storage_root_id) a
        ON a.storage_root_id = r.id
    LEFT JOIN (SELECT source_storage_root_id, count(*) n FROM run GROUP BY source_storage_root_id) s
        ON s.source_storage_root_id = r.id
    LEFT JOIN (SELECT storage_root_id, count(*) n FROM dataset GROUP BY storage_root_id) d
        ON d.storage_root_id = r.id
    ORDER BY r.name, r.id
"""


def root_summary(row) -> StorageRootSummary:
    return StorageRootSummary(
        StorageRoot(**{key: row[key] for key in ("id", "name", "location", "kind")}),
        row["artifact_count"], row["run_count"], row["dataset_count"],
    )
