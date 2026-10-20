"""One startup location contract; selecting a location never relocates a catalog."""

from dataclasses import dataclass
import os
from pathlib import Path
import sqlite3
import stat

from runpup.persistence.database import Database, SchemaVersionError


CATALOG_FILENAME = "runpup.sqlite3"


def absolute_directory(value, *, allow_relative=False) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip() or "\0" in str(value):
        raise ValueError("A data directory must be a non-empty path without null bytes")
    path = Path(value).expanduser()
    if not allow_relative and not path.is_absolute():
        raise ValueError("The saved data directory must be an absolute path")
    return path.absolute()


@dataclass(frozen=True, slots=True)
class DataLocation:
    directory: Path
    database_path: Path
    default_directory: Path
    source: str


def resolve_data_location(default_directory, configured=None, *, override=None) -> DataLocation:
    """CLI > catalog-independent saved location > Qt's platform default."""
    default = absolute_directory(default_directory)
    if override is not None:
        directory, source = absolute_directory(override, allow_relative=True), "Command-line override"
    elif configured is not None and configured != "":
        directory, source = absolute_directory(configured), "Configured"
    else:
        directory, source = default, "Default"
    return DataLocation(directory, directory / CATALOG_FILENAME, default, source)


def validate_data_directory(value) -> str:
    """Inspect only the directory and canonical database; never create or migrate here."""
    directory = absolute_directory(value)
    try:
        mode = directory.stat().st_mode
    except FileNotFoundError:
        # No creation until launch. An inaccessible/file ancestor still raises clearly.
        for parent in directory.parents:
            try:
                if not stat.S_ISDIR(parent.stat().st_mode):
                    raise ValueError("A parent of the data directory is not a directory")
                break
            except FileNotFoundError:
                continue
        return "New catalog will be initialized on launch"
    if not stat.S_ISDIR(mode):
        raise ValueError("The selected data location is not a directory")
    catalog = directory / CATALOG_FILENAME
    try:
        catalog.lstat()
    except FileNotFoundError:
        with os.scandir(directory) as entries:
            if next(entries, None) is not None:
                raise ValueError("This directory is not empty and has no RunPup catalog. "
                                 "Choose an empty directory or an existing RunPup data directory.")
        return "New catalog will be initialized on launch"
    version = Database(catalog).validate_existing()
    return f"Existing RunPup catalog (schema {version})"


def prepare_catalog(location: DataLocation) -> Database:
    validate_data_directory(location.directory)
    database = Database(location.database_path)
    database.initialize()
    return database


def open_startup_catalog(default_directory, configured=None, *, override=None, recover=None):
    """Recovery is explicit and launch-only; this function never writes preferences."""
    try:
        location = resolve_data_location(default_directory, configured, override=override)
        return location, prepare_catalog(location)
    except (ValueError, OSError, RuntimeError, sqlite3.Error, SchemaVersionError) as error:
        if (override is not None or configured is None or configured == "" or recover is None
                or not recover(error)):
            raise
    default = resolve_data_location(default_directory)
    location = DataLocation(default.directory, default.database_path, default.default_directory,
                            "Default for this launch")
    return location, prepare_catalog(location)
