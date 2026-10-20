"""Catalog-independent preferences and background backup/location checks."""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from PySide6.QtCore import QStandardPaths, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit,
    QProgressBar, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from runpup.services.backup import create_backup
from runpup.services.data_location import absolute_directory, resolve_data_location, validate_data_directory
from runpup.ui.details_page import _card, _text
from runpup.ui.review_field import label


def _path_field(name):
    field = QLineEdit()
    field.setReadOnly(True)
    field.setMinimumWidth(0)
    field.setAccessibleName(name)
    return field


class _SettingsTask(QThread):
    def __init__(self, operation, parent):
        super().__init__(parent)
        self.operation = operation
        self.result, self.error = None, None

    def run(self):
        try:
            self.result = self.operation()
        except (ValueError, OSError, RuntimeError, sqlite3.Error) as error:
            self.error = str(error)


class SettingsPage(QWidget):
    idle = Signal()

    def __init__(self, preferences, *, database, location, parent=None):
        super().__init__(parent)
        self.preferences, self.database, self.location = preferences, database, location
        self.task, self.task_kind = None, None
        self.closing = False
        self.pending_directory = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 0)
        layout.addWidget(label("Settings", "heading"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(scroll, 1)
        body = QWidget()
        cards = QVBoxLayout(body)
        cards.setContentsMargins(0, 0, 0, 8)
        cards.setSpacing(14)
        scroll.setWidget(body)
        self.scroll = scroll

        card, box = _card("Data & Library")
        form = QFormLayout()
        self.active_directory = _path_field("Active data directory")
        self.database_path = _path_field("Active catalog database")
        self.next_directory = _path_field("Data directory for next ordinary launch")
        form.addRow("Active data directory", self.active_directory)
        form.addRow("Active database", self.database_path)
        form.addRow("Next ordinary launch", self.next_directory)
        box.addLayout(form)
        self.location_source = _text("", "muted")
        box.addWidget(self.location_source)
        row = QHBoxLayout()
        self.choose_data_button = QPushButton("Choose Different Directory…")
        self.choose_data_button.clicked.connect(self.choose_data_directory)
        self.default_data_button = QPushButton("Use Default")
        self.default_data_button.clicked.connect(self.use_default)
        row.addWidget(self.choose_data_button)
        row.addWidget(self.default_data_button)
        row.addStretch()
        box.addLayout(row)
        self.data_status = _text("", "muted")
        box.addWidget(self.data_status)
        box.addWidget(_text("Data-location changes require a restart. The current catalog stays open; "
                            "RunPup does not move or copy it. An explicit --data-dir always wins for that launch.", "muted"))
        cards.addWidget(card)

        card, box = _card("Backup")
        box.addWidget(_text("Backups contain RunPup’s catalog and supported Import settings. "
                            "External training artifacts, datasets and original trainer outputs are not copied.", "muted"))
        self.last_backup = _text("", "muted")
        box.addWidget(self.last_backup)
        row = QHBoxLayout()
        self.backup_button = QPushButton("Create Backup…")
        self.backup_button.setObjectName("primary")
        self.backup_button.clicked.connect(self.choose_backup)
        row.addWidget(self.backup_button)
        self.copy_backup_button = QPushButton("Copy Backup Path")
        self.copy_backup_button.clicked.connect(self.copy_backup_path)
        row.addWidget(self.copy_backup_button)
        row.addStretch()
        box.addLayout(row)
        self.backup_path = _path_field("Last backup path")
        box.addWidget(self.backup_path)
        self.backup_status = _text("", "muted")
        box.addWidget(self.backup_status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.hide()
        box.addWidget(self.progress)
        cards.addWidget(card)

        card, box = _card("Portability")
        box.addWidget(_text("The data directory holds RunPup’s catalog. Storage Roots map its references "
                            "to external files and may need remapping on another machine. Saved paths in "
                            "Import settings are not guaranteed to exist there.\n\n"
                            "--data-dir supports launching against another data directory. This desktop "
                            "launcher supplies no base directory for portable-relative roots, so they remain "
                            "Unverified. There is no automatic portable-mode detection or root rebasing.", "muted"))
        cards.addWidget(card)

        card, box = _card("Import")
        box.addWidget(_text("Default training outputs folder"))
        row = QHBoxLayout()
        self.default_directory = _path_field("Default training outputs folder")
        self.default_directory.setPlaceholderText("No default folder set")
        row.addWidget(self.default_directory, 1)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse)
        row.addWidget(self.browse_button)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_default)
        row.addWidget(self.clear_button)
        box.addLayout(row)
        self.remember_last = QCheckBox("Remember last import location")
        self.remember_last.toggled.connect(self._set_remember_last)
        box.addWidget(self.remember_last)
        box.addWidget(_text("Changes apply immediately. Import tries the valid remembered run parent, "
                            "default outputs folder, existing local Unsloth outputs, then home. "
                            "Missing folders are skipped.", "muted"))
        self.import_status = _text("", "muted")
        box.addWidget(self.import_status)
        cards.addWidget(card)
        cards.addStretch()
        self.refresh()

    @property
    def busy(self):
        return self.task is not None

    @staticmethod
    def _show_path(field, value):
        field.setText(str(value))
        field.setCursorPosition(0)

    def refresh(self):
        self._show_path(self.active_directory, self.location.directory)
        self._show_path(self.database_path, self.location.database_path)
        try:
            next_location = resolve_data_location(self.location.default_directory,
                                                 self.preferences.configured_data_directory)
            self._show_path(self.next_directory, next_location.directory)
            pending = next_location.directory != self.location.directory
            self.location_source.setText(f"Current location: {self.location.source}. "
                                         + ("Different location configured for next ordinary launch; restart required."
                                            if pending else "Next ordinary launch uses this location."))
        except (ValueError, OSError, RuntimeError):
            self.next_directory.setText("Invalid saved location — use Default to reset")
            self.location_source.setText(f"Current location: {self.location.source}. "
                                         "Saved location is invalid; reset it or launch with --data-dir.")
        self._show_path(self.default_directory, self.preferences.default_directory)
        self.clear_button.setEnabled(bool(self.preferences.default_directory))
        self.remember_last.blockSignals(True)
        self.remember_last.setChecked(self.preferences.remember_last)
        self.remember_last.blockSignals(False)
        created, path = self.preferences.last_backup
        self.last_backup.setText(f"Last backup: {created}" if created else "No backup recorded yet.")
        self._show_path(self.backup_path, path)
        self.copy_backup_button.setEnabled(bool(path))
        self._controls()

    def _controls(self):
        for button in (self.choose_data_button, self.default_data_button, self.backup_button):
            button.setEnabled(not self.busy and not self.closing)

    def choose_data_directory(self):
        if self.busy or self.closing:
            return
        selected = QFileDialog.getExistingDirectory(self, "RunPup Data Directory")
        if selected:
            self.configure_directory(selected)

    def configure_directory(self, selected):
        if self.busy or self.closing:
            return
        try:
            directory = absolute_directory(selected)
        except (ValueError, OSError, RuntimeError) as error:
            self.data_status.setText(f"Cannot use this directory: {error}. The saved location was not changed.")
            return
        self.pending_directory = directory
        self.data_status.setText("Checking the selected data directory…")
        self._start("location", lambda: validate_data_directory(directory))

    def use_default(self):
        if self.busy or self.closing:
            return
        try:
            self.preferences.configured_data_directory = None
            self.data_status.setText("Default configured for next ordinary launch. Restart required. "
                                     "The current catalog and previous data directory were not changed.")
        except OSError as error:
            self.data_status.setText(str(error))
        self.refresh()

    def choose_backup(self):
        if self.busy or self.closing:
            return
        name = datetime.now(timezone.utc).strftime("RunPup-Backup-%Y-%m-%d_%H%M%S.zip")
        directory = self.preferences.home
        if directory.absolute() == self.location.directory.absolute():
            documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
            directory = Path(documents) if documents else directory.parent
            if directory.absolute() == self.location.directory.absolute():
                directory = directory.parent
        selected, _ = QFileDialog.getSaveFileName(self, "Create RunPup Backup",
                                                  str(directory / name), "ZIP backups (*.zip)")
        if selected:
            self.start_backup(selected, overwrite=True)  # The save dialog confirms an existing destination.

    def start_backup(self, destination, *, overwrite=False):
        if self.busy or self.closing:
            return
        preferences = self.preferences.portable_export()
        self.backup_status.setText("Creating backup… Catalog data and Import settings only.")
        self._start("backup", lambda: create_backup(self.database, destination, preferences, overwrite=overwrite))

    def _start(self, kind, operation):
        self.task_kind = kind
        self.task = _SettingsTask(operation, self)
        self.task.finished.connect(self._finished)
        self.progress.setVisible(kind == "backup")
        self._controls()
        self.task.start()

    def _finished(self):
        task, self.task = self.task, None
        kind, self.task_kind = self.task_kind, None
        task.deleteLater()
        self.progress.hide()
        if kind == "backup":
            if task.error:
                self.backup_status.setText(task.error)
            else:
                result = task.result
                message = "Backup created successfully. Catalog: included · Import settings: included · External artifacts: not included."
                try:
                    self.preferences.record_backup(result.path, result.created_at)
                except OSError:
                    message += " Backup history could not be saved; the backup itself is complete."
                self.refresh()
                self._show_path(self.backup_path, result.path)
                self.copy_backup_button.setEnabled(True)
                self.backup_status.setText(message)
        elif not self.closing:
            if task.error:
                self.data_status.setText(f"Cannot use this directory: {task.error}. The saved location and "
                                         "current catalog were not changed. Choose an empty or valid RunPup directory.")
            else:
                try:
                    self.preferences.configured_data_directory = self.pending_directory
                    self.data_status.setText(f"{task.result}. Saved for next ordinary launch; restart required. "
                                             "The current catalog stays here; no files were moved.")
                except (OSError, ValueError) as error:
                    self.data_status.setText(str(error))
                self.refresh()
        self._controls()
        self.idle.emit()

    def copy_backup_path(self):
        if self.backup_path.text():
            QApplication.clipboard().setText(self.backup_path.text())

    def browse(self):
        selected = QFileDialog.getExistingDirectory(
            self, "Default Training Outputs Folder", self.preferences.initial_directory())
        if selected:
            self._save_import(lambda: setattr(self.preferences, "default_directory", selected))

    def clear_default(self):
        self._save_import(lambda: setattr(self.preferences, "default_directory", ""))

    def _set_remember_last(self, enabled):
        self._save_import(lambda: setattr(self.preferences, "remember_last", enabled))

    def _save_import(self, operation):
        try:
            operation()
            self.import_status.clear()
        except OSError as error:
            self.import_status.setText(str(error))
        self.refresh()

    def finish_before_close(self):
        self.closing = True
        self._controls()
