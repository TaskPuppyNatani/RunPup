"""Small desktop preferences, separate from the training-run catalog."""

from pathlib import Path

from PySide6.QtCore import QSettings


class ImportPreferences:
    def __init__(self, settings: QSettings | None = None, *, home: Path | None = None):
        self.settings = settings if settings is not None else QSettings("TaskPuppy", "RunPup")
        self.home = home if home is not None else Path.home()

    def _path(self, key):
        value = self.settings.value(key, "")
        return value if isinstance(value, str) else ""

    @property
    def default_directory(self):
        return self._path("import/default_directory")

    @default_directory.setter
    def default_directory(self, value):
        self.settings.setValue("import/default_directory", str(value) if value else "")
        self.settings.sync()

    @property
    def remember_last(self):
        return self.settings.value("import/remember_last", True, type=bool)

    @remember_last.setter
    def remember_last(self, enabled):
        self.settings.setValue("import/remember_last", bool(enabled))
        self.settings.sync()

    def remember_selection(self, selected):
        if self.remember_last:
            path = Path(selected).absolute()
            if self._directory(path):
                self.settings.setValue("import/last_parent", str(path.parent))
                self.settings.sync()

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
