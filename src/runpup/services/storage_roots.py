"""Explicit existence checks through PathResolver, without payload access or writes."""

from collections import Counter
from dataclasses import dataclass

from runpup.domain.models import Availability, StorageRoot
from runpup.services.paths import PathResolution, PathResolver


def root_status(resolution: PathResolution) -> str:
    if resolution.state is Availability.AVAILABLE:
        return "Online"
    if resolution.state is Availability.OFFLINE:
        return "Offline"
    return "Unverified"


@dataclass(frozen=True, slots=True)
class RootVerification:
    root: StorageRoot
    resolution: PathResolution
    available: int = 0
    missing: int = 0
    offline: int = 0
    unverified: int = 0

    @property
    def text(self):
        return (f"Available: {self.available} · Missing: {self.missing} · "
                f"Offline: {self.offline} · Unverified: {self.unverified}")


def verify_storage_root(repository, root: StorageRoot, *, resolver=None, cancelled=lambda: False):
    resolver = resolver if resolver is not None else PathResolver()
    references = repository.storage_root_references(root)
    resolution = resolver.resolve(root, ".")
    counts = Counter()
    checked = {}
    for reference in references:
        if cancelled():
            return None
        if resolution.state is Availability.AVAILABLE:
            if reference not in checked:
                checked[reference] = resolver.resolve(root, reference)
            state = checked[reference].state
        else:
            state = (Availability.OFFLINE if resolution.state is Availability.OFFLINE
                     else Availability.UNVERIFIED)
        counts[state] += 1
    # Availability is a point-in-time observation; notice a mount disappearing mid-check.
    resolution = resolver.resolve(root, ".")
    return RootVerification(root, resolution, *(counts[state] for state in (
        Availability.AVAILABLE, Availability.MISSING, Availability.OFFLINE, Availability.UNVERIFIED,
    )))
