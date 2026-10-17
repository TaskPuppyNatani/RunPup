"""Resolve external references without copying, hashing, or modifying files."""

from dataclasses import dataclass
import os
from pathlib import Path

from runpup.domain.models import Availability, RootKind, StorageRoot, relative_reference


@dataclass(frozen=True, slots=True)
class PathResolution:
    path: Path | None
    state: Availability
    root_available: bool | None
    reference_available: bool | None
    reason: str | None = None


class PathResolver:
    def __init__(self, *, portable_base: Path | None = None):
        self.portable_base = portable_base.absolute() if portable_base is not None else None

    def resolve(self, root: StorageRoot, relative_path: str, *, verify: bool = True) -> PathResolution:
        reference = relative_reference(relative_path)
        if root.kind is RootKind.PORTABLE_RELATIVE:
            if self.portable_base is None:
                return PathResolution(None, Availability.UNVERIFIED, None, None,
                                      "A portable base directory has not been supplied")
            root_path = self.portable_base / root.location
        else:
            root_path = Path(root.location)
        path = root_path / reference
        if not verify:
            return PathResolution(path, Availability.UNVERIFIED, None, None)

        try:
            # Opening the directory checks access without enumerating its contents.
            with os.scandir(root_path):
                pass
            canonical_root = root_path.resolve()
        except OSError as error:
            return PathResolution(path, Availability.OFFLINE, False, None, str(error))
        except RuntimeError as error:  # A symlink loop on Python versions before 3.13.
            return PathResolution(path, Availability.OFFLINE, False, None, str(error))

        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(canonical_root):
                return PathResolution(path, Availability.UNVERIFIED, True, None,
                                      "Reference escapes its storage root through a symlink")
            resolved.stat()
        except (FileNotFoundError, NotADirectoryError):
            return PathResolution(path, Availability.MISSING, True, False)
        except (OSError, RuntimeError) as error:
            return PathResolution(path, Availability.UNVERIFIED, True, None, str(error))
        return PathResolution(path, Availability.AVAILABLE, True, True)
