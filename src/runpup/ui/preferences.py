"""Small desktop preferences, separate from the training-run catalog."""

import logging
from pathlib import Path

from PySide6.QtCore import QSettings, QStandardPaths

from runpup.services.data_location import absolute_directory


def default_data_directory() -> Path:
    value = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    if not value:
        raise ValueError("Qt could not determine RunPup's default data directory")
    return Path(value).absolute()


class ImportPreferences:
    def __init__(self, settings: QSettings | None = None, *, home: Path | None = None):
        self.settings = settings if settings is not None else QSettings("TaskPuppy", "RunPup")
        self.home = home if home is not None else Path.home()

    def _write(self, values):
        """QSettings uses atomic sync; restore cached values if persistence fails."""
        previous = {key: (self.settings.contains(key), self.settings.value(key)) for key in values}
        for key, value in values.items():
            if value is None:
                self.settings.remove(key)
            else:
                self.settings.setValue(key, value)
        self.settings.sync()
        if self.settings.status() != QSettings.Status.NoError:
            for key, (existed, value) in previous.items():
                if existed:
                    self.settings.setValue(key, value)
                else:
                    self.settings.remove(key)
            self.settings.sync()
            # QSettings retains its first error for the object's lifetime. Reopen the
            # same store after rollback so a later retry can succeed after access is fixed.
            settings = self.settings
            if settings.organizationName():
                self.settings = QSettings(settings.format(), settings.scope(),
                                          settings.organizationName(), settings.applicationName())
            else:
                self.settings = QSettings(settings.fileName(), settings.format())
            raise OSError("RunPup could not save preferences. The previous values were retained; "
                          "check configuration-directory access and try again.")

    @property
    def configured_data_directory(self):
        # Return the raw bootstrap value so malformed settings cause visible recovery.
        return self.settings.value("startup/data_directory", None)

    @configured_data_directory.setter
    def configured_data_directory(self, value):
        self._write({"startup/data_directory": str(absolute_directory(value)) if value is not None else None})

    def portable_export(self):
        """Explicit allowlist: never archive the entire settings store."""
        return {"import/default_directory": self.default_directory, "import/remember_last": self.remember_last}

    @property
    def last_backup(self):
        return self._path("backup/last_created"), self._path("backup/last_path")

    def record_backup(self, path, created_at):
        self._write({"backup/last_path": str(path), "backup/last_created": created_at})

    def _path(self, key):
        value = self.settings.value(key, "")
        return value if isinstance(value, str) else ""

    @property
    def default_directory(self):
        return self._path("import/default_directory")

    @default_directory.setter
    def default_directory(self, value):
        self._write({"import/default_directory": str(value) if value else ""})

    @property
    def remember_last(self):
        return self.settings.value("import/remember_last", True, type=bool)

    @remember_last.setter
    def remember_last(self, enabled):
        self._write({"import/remember_last": bool(enabled)})

    def remember_selection(self, selected):
        if self.remember_last:
            path = Path(selected).absolute()
            if self._directory(path):
                try:
                    self._write({"import/last_parent": str(path.parent)})
                except OSError:
                    # Remembering a chooser convenience must not abort the sealed import flow.
                    logging.getLogger(__name__).warning("Could not remember the last import location")

    @staticmethod
    def _directory(path):
        try:
            return path.is_dir()
        except (OSError, ValueError):
            return False

    def initial_directory(self):
        candidates = []
        if self.remember_last:
            candidates.append(self._path("import/last_parent"))
        candidates.extend((self.default_directory, self.home / ".unsloth/studio/outputs", self.home))
        for candidate in candidates:
            if not candidate:
                continue
            try:
                path = Path(candidate).expanduser().absolute()
                if self._directory(path):
                    return str(path)
            except (OSError, RuntimeError, ValueError):
                continue
        return str(Path.cwd())
