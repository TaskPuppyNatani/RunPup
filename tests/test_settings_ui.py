import json
from pathlib import Path
import sqlite3
import threading
import zipfile

import pytest
from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from runpup.persistence.database import Database
from runpup.services.backup import create_backup
from runpup.services.data_location import prepare_catalog, resolve_data_location
from runpup.ui.preferences import ImportPreferences
from runpup.ui.window import MainWindow
from test_details_ui import app, wait_for


@pytest.fixture
def window(app, repository, tmp_path):
    preferences = ImportPreferences(QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat), home=tmp_path)
    location = resolve_data_location(tmp_path / "default", override=repository.database.path.parent)
    window = MainWindow(repository, preferences=preferences, data_location=location)
    window.show()
    app.processEvents()
    yield window
    window.close()
    wait_for(app, lambda: not any(page.busy for page in (
        window.settings_page, window.import_page, window.details_page, window.compare_page, window.storage_roots_page)))
    window.close()


def settings(window):
    window.navigation_buttons["Settings"].click()
    return window.settings_page


def test_active_paths_and_next_launch_selection_do_not_switch_catalog(app, window, repository, tmp_path, monkeypatch):
    page = settings(window)
    assert page.active_directory.text() == str(repository.database.path.parent)
    assert page.database_path.text() == str(repository.database.path)
    assert page.database_path.isReadOnly() and "Command-line override" in page.location_source.text()
    target = tmp_path / "custom"
    target.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: "")
    page.choose_data_button.click()
    assert not page.busy and window.preferences.configured_data_directory is None
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: str(target))
    page.choose_data_button.click()
    wait_for(app, lambda: not page.busy)
    assert page.next_directory.text() == str(target) and "restart required" in page.data_status.text()
    assert window.preferences.configured_data_directory == str(target)
    assert page.database is repository.database and not (target / "runpup.sqlite3").exists()
    custom_database = prepare_catalog(resolve_data_location(target))
    before = custom_database.path.read_bytes()
    page.default_data_button.click()
    assert window.preferences.configured_data_directory is None
    assert page.next_directory.text() == str(tmp_path / "default")
    assert custom_database.path.read_bytes() == before and page.database is repository.database


def test_foreign_target_and_preference_failure_leave_location_unchanged(app, window, tmp_path, monkeypatch):
    page = settings(window)
    target = tmp_path / "foreign"
    target.mkdir()
    path = target / "runpup.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE foreign_data(value TEXT)")
        connection.execute("INSERT INTO foreign_data VALUES ('preserve')")
    before = path.read_bytes()
    page.configure_directory(target)
    wait_for(app, lambda: not page.busy)
    assert "saved location and current catalog were not changed" in page.data_status.text()
    assert window.preferences.configured_data_directory is None and path.read_bytes() == before

    def fail(*a):
        raise OSError("Cannot save preferences; previous values retained")

    with monkeypatch.context() as patch:
        patch.setattr(window.preferences, "_write", fail)
        page.configure_directory(tmp_path / "new")
        wait_for(app, lambda: not page.busy)
        assert "Cannot save preferences" in page.data_status.text()
        assert window.preferences.configured_data_directory is None
    page.configure_directory(tmp_path / "new")
    wait_for(app, lambda: not page.busy)
    assert window.preferences.configured_data_directory == str(tmp_path / "new")


def test_backup_chooser_success_failure_and_copy_path(app, window, tmp_path, monkeypatch):
    page = settings(window)
    choices = []

    def cancelled(*args):
        choices.append(args[2])
        return "", ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", cancelled)
    page.backup_button.click()
    assert not page.busy and Path(choices[0]).parent == tmp_path
    assert Path(choices[0]).parent != window.repository.database.path.parent
    destination = tmp_path / "backup.zip"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: (str(destination), "ZIP backups (*.zip)"))
    page.backup_button.click()
    wait_for(app, lambda: not page.busy)
    assert "Backup created successfully" in page.backup_status.text()
    assert "External artifacts: not included" in page.backup_status.text()
    assert page.backup_path.text() == str(destination) and window.preferences.last_backup[1] == str(destination)
    page.copy_backup_button.click()
    assert QApplication.clipboard().text() == str(destination)
    with zipfile.ZipFile(destination) as archive:
        assert json.loads(archive.read("runpup-backup.json"))["backup_format_version"] == 1
    last_backup = window.preferences.last_backup
    page.start_backup(tmp_path / "missing-parent" / "backup.zip")
    wait_for(app, lambda: not page.busy)
    assert "source catalog was not changed" in page.backup_status.text()
    assert window.preferences.last_backup == last_backup and page.backup_button.isEnabled()
    assert not page.progress.isVisible()


def test_import_preferences_remain_shared_and_keyboard_toggle_works(app, window, tmp_path, monkeypatch):
    page = settings(window)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: str(outputs))
    page.browse_button.click()
    assert window.import_page.preferences is window.preferences
    assert window.preferences.default_directory == str(outputs)
    page.scroll.ensureWidgetVisible(page.remember_last)
    page.remember_last.setFocus()
    QTest.keyClick(page.remember_last, Qt.Key.Key_Space)
    assert window.preferences.remember_last is False
    starts = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: starts.append(a[2]) or "")
    window.show_import()
    window.import_page.choose_button.click()
    assert starts == [str(outputs)]
    window.show_settings()
    page.clear_button.click()
    assert window.preferences.default_directory == "" and not page.clear_button.isEnabled()


def test_backup_chooser_does_not_default_to_active_data_directory_even_when_it_is_home(window, monkeypatch):
    page = settings(window)
    window.preferences.home = window.data_location.directory
    suggestions = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: suggestions.append(args[2]) or ("", ""))
    page.backup_button.click()
    assert Path(suggestions[0]).parent != window.data_location.directory


def test_malformed_bootstrap_can_be_reset_from_recovered_session(window):
    page = settings(window)
    window.preferences.settings.setValue("startup/data_directory", False)
    page.refresh()
    assert "Invalid saved location" in page.next_directory.text()
    page.default_data_button.click()
    assert window.preferences.configured_data_directory is None and "Default configured" in page.data_status.text()


def test_backup_duplicate_guard_snapshot_preferences_and_navigation(app, window, tmp_path, monkeypatch):
    import runpup.ui.settings_page as module
    entered, release = threading.Event(), threading.Event()
    captured = []

    def gated(database, destination, preferences, **kwargs):
        captured.append(preferences)
        entered.set()
        assert release.wait(4)
        return create_backup(database, destination, preferences, **kwargs)

    monkeypatch.setattr(module, "create_backup", gated)
    page = settings(window)
    page.start_backup(tmp_path / "snapshot.zip")
    wait_for(app, entered.is_set)
    task = page.task
    try:
        page.start_backup(tmp_path / "duplicate.zip")
        page.configure_directory(tmp_path / "must-not-be-saved")
        assert page.task is task and not page.backup_button.isEnabled()
        assert page.progress.isVisible() and not page.choose_data_button.isEnabled()
        page.remember_last.setChecked(False)
        window.show_library()
        assert window.workspace.currentWidget() is window.library_page
        window.show_settings()
        assert page.task is task and "Creating backup" in page.backup_status.text()
    finally:
        release.set()
    wait_for(app, lambda: not page.busy)
    assert len(captured) == 1 and captured[0]["import/remember_last"] is True
    assert window.preferences.remember_last is False and window.preferences.configured_data_directory is None
    assert page.backup_path.text() == str(tmp_path / "snapshot.zip")
    assert not (tmp_path / "duplicate.zip").exists()


def test_backup_close_waits_until_complete(app, window, tmp_path, monkeypatch):
    import runpup.ui.settings_page as module
    entered, release = threading.Event(), threading.Event()

    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(4)
        return create_backup(*args, **kwargs)

    monkeypatch.setattr(module, "create_backup", gated)
    page = settings(window)
    destination = tmp_path / "complete-before-close.zip"
    page.start_backup(destination)
    wait_for(app, entered.is_set)
    try:
        window.close()
        assert window.isVisible() and page.closing and page.busy
    finally:
        release.set()
    wait_for(app, lambda: not page.busy and not window.isVisible())
    assert zipfile.is_zipfile(destination) and window.preferences.last_backup[1] == str(destination)


def test_location_validation_closing_does_not_apply_pending_preference(app, window, tmp_path, monkeypatch):
    import runpup.ui.settings_page as module
    entered, release = threading.Event(), threading.Event()

    def gated(path):
        entered.set()
        assert release.wait(4)
        return "New catalog"

    monkeypatch.setattr(module, "validate_data_directory", gated)
    page = settings(window)
    page.configure_directory(tmp_path / "pending")
    wait_for(app, entered.is_set)
    try:
        page.configure_directory(tmp_path / "duplicate")
        assert page.pending_directory == tmp_path / "pending"
        window.close()
    finally:
        release.set()
    wait_for(app, lambda: not page.busy and not window.isVisible())
    assert window.preferences.configured_data_directory is None


def test_long_paths_and_sections_remain_scrollable_at_both_sizes(app, window, tmp_path):
    page = settings(window)
    long_path = tmp_path / ("long-folder-" * 18) / ("more-" * 18)
    window.preferences.configured_data_directory = long_path
    window.preferences.default_directory = long_path
    page.refresh()
    for width, height in ((1440, 900), (1040, 640)):
        window.resize(width, height)
        app.processEvents()
        assert page.next_directory.text() == str(long_path)
        assert page.default_directory.text() == str(long_path)
        assert page.scroll.horizontalScrollBar().maximum() == 0
        page.scroll.ensureWidgetVisible(page.backup_button)
        app.processEvents()
        assert page.backup_button.isVisible()
        page.scroll.ensureWidgetVisible(page.remember_last)
        app.processEvents()
        assert page.remember_last.isVisible()


def test_startup_main_uses_injected_settings_context_and_cli_errors(app, tmp_path, monkeypatch):
    import runpup.ui.app as module
    preferences = ImportPreferences(QSettings(str(tmp_path / "startup.ini"), QSettings.Format.IniFormat), home=tmp_path)
    custom, override = tmp_path / "custom", tmp_path / "override"
    preferences.configured_data_directory = custom
    monkeypatch.setattr(module, "ImportPreferences", lambda: preferences)
    monkeypatch.setattr(module, "default_data_directory", lambda: tmp_path / "default")
    contexts = []

    def window(repository, **kwargs):
        contexts.append((repository.database.path, kwargs["data_location"]))
        widget = MainWindow(repository, **kwargs)
        QTimer.singleShot(10, widget.close)
        return widget

    monkeypatch.setattr(module, "MainWindow", window)
    assert module.main(["--data-dir", str(override)]) == 0
    assert contexts[-1][0] == override / "runpup.sqlite3" and contexts[-1][1].source == "Command-line override"
    assert preferences.configured_data_directory == str(custom)
    assert module.main([]) == 0
    assert contexts[-1][0] == custom / "runpup.sqlite3"
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("preserve")
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args[2]))
    assert module.main(["--data-dir", str(blocker)]) == 1
    assert "No existing catalog was replaced" in errors[-1] and preferences.configured_data_directory == str(custom)


def test_startup_recovery_dialog_default_is_explicit_and_does_not_reset_saved_path(app, tmp_path, monkeypatch):
    import runpup.ui.app as module
    preferences = ImportPreferences(QSettings(str(tmp_path / "recovery.ini"), QSettings.Format.IniFormat), home=tmp_path)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    path = foreign / "runpup.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE preserve(value TEXT)")
    before = path.read_bytes()
    preferences.configured_data_directory = foreign
    monkeypatch.setattr(module, "ImportPreferences", lambda: preferences)
    monkeypatch.setattr(module, "default_data_directory", lambda: tmp_path / "default")
    original_exec = QMessageBox.exec
    observed = []

    def recover(dialog):
        observed.append(dialog.text())
        button = next(button for button in dialog.buttons() if button.text() == "Use Default for This Launch")
        QTimer.singleShot(0, button.click)
        return original_exec(dialog)

    monkeypatch.setattr(QMessageBox, "exec", recover)
    contexts = []

    def window(repository, **kwargs):
        contexts.append(kwargs["data_location"])
        widget = MainWindow(repository, **kwargs)
        QTimer.singleShot(10, widget.close)
        return widget

    monkeypatch.setattr(module, "MainWindow", window)
    assert module.main([]) == 0
    assert observed and contexts[0].source == "Default for this launch"
    assert preferences.configured_data_directory == str(foreign) and path.read_bytes() == before
    blocked_default = tmp_path / "blocked-default"
    blocked_default.write_text("not a directory")
    monkeypatch.setattr(module, "default_data_directory", lambda: blocked_default)
    failures = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: failures.append(args[2]))
    assert module.main([]) == 1
    assert f"Location: {blocked_default}" in failures[-1]
    assert preferences.configured_data_directory == str(foreign) and path.read_bytes() == before
