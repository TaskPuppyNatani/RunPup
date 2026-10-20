"""Consistent SQLite snapshots and a small, explicitly allowlisted backup bundle."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
import tempfile
import zipfile

from runpup import __version__
from runpup.persistence.database import Database, SchemaVersionError
from runpup.persistence.migrations import SCHEMA_VERSION


BACKUP_FORMAT_VERSION = 1
BUNDLE_FILES = ("runpup-backup.json", "catalog.sqlite3", "preferences.json")


class BackupError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BackupResult:
    path: Path
    created_at: str


def create_backup(database: Database, destination, preferences: dict, *, overwrite=False) -> BackupResult:
    """Publish only a validated, complete archive. Never inspect a StorageRoot."""
    destination = Path(destination).expanduser().absolute()
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        if destination.suffix.lower() != ".zip":
            raise ValueError("Choose a .zip backup filename")
        if destination.resolve() == database.path.resolve() or (
            destination.exists() and database.path.exists() and destination.samefile(database.path)
        ):
            raise ValueError("The backup destination cannot be the active catalog")
        if destination.exists() and not overwrite:
            raise FileExistsError("The backup destination already exists; choose a new filename")
        if (set(preferences) != {"import/default_directory", "import/remember_last"}
                or not isinstance(preferences["import/default_directory"], str)
                or type(preferences["import/remember_last"]) is not bool):
            raise ValueError("Only the supported Import preferences can be exported")
        database.validate_existing()
        with tempfile.TemporaryDirectory(prefix=".runpup-backup-", dir=destination.parent) as staging:
            staging = Path(staging)
            snapshot = staging / "catalog.sqlite3"
            with database.read_connection() as source:
                target = sqlite3.connect(snapshot)
                try:
                    source.backup(target, pages=256, sleep=.01)
                finally:
                    target.close()
            version = Database(snapshot).validate_existing()
            if version != SCHEMA_VERSION:
                raise SchemaVersionError("The active catalog must use the current RunPup schema")
            manifest = {
                "application": "RunPup", "application_version": __version__,
                "backup_format_version": BACKUP_FORMAT_VERSION, "schema_version": version,
                "created_at": created_at, "files": list(BUNDLE_FILES),
                "external_artifacts_included": False,
                "preferences_note": "Only default import folder and remember-last toggle are included. "
                    "Paths are configuration values; they may not exist on another machine. "
                    "Bootstrap data location and session/backup history are excluded.",
            }
            bundle = staging / "bundle.zip"
            with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("runpup-backup.json", json.dumps(manifest, indent=2))
                archive.write(snapshot, "catalog.sqlite3")
                archive.writestr("preferences.json", json.dumps(preferences, indent=2))
            with zipfile.ZipFile(bundle) as archive:
                if set(archive.namelist()) != set(BUNDLE_FILES) or archive.testzip() is not None:
                    raise ValueError("The backup archive did not pass validation")
                if json.loads(archive.read("runpup-backup.json")) != manifest:
                    raise ValueError("The backup manifest did not pass validation")
            if overwrite:
                os.replace(bundle, destination)
            else:
                # An atomic no-clobber publication, even if a destination appears mid-backup.
                os.link(bundle, destination)
        return BackupResult(destination, created_at)
    except (OSError, ValueError, RuntimeError, sqlite3.Error, zipfile.BadZipFile) as error:
        logging.getLogger(__name__).debug("Backup creation failed", exc_info=True)
        reason = ("The destination already exists; choose another filename" if isinstance(error, FileExistsError) else
                  "The destination folder is unavailable" if isinstance(error, FileNotFoundError) else
                  "The destination could not be written" if isinstance(error, PermissionError) else
                  "The backup write could not be completed" if isinstance(error, OSError) else str(error))
        raise BackupError(f"Backup could not be created at {destination}. {reason}. The source catalog was not changed. "
                          "No incomplete backup was published; staging files were cleaned where accessible. "
                          "Check destination access or choose another filename and try again.") from error
