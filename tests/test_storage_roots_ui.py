from dataclasses import replace
from pathlib import Path
import sqlite3
import threading

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QFileDialog

from runpup.domain.models import RootKind, Run, StorageRoot
from runpup.services.paths import PathResolver
from runpup.ui.preferences import ImportPreferences
from runpup.ui.window import MainWindow
from test_details_ui import app, wait_for
from test_storage_roots import save_references


@pytest.fixture
def window(app, repository, tmp_path):
    prefs = ImportPreferences(QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat), home=tmp_path)
    window = MainWindow(repository, preferences=prefs)
    window.show()
    app.processEvents()
    yield window
    window.close()
    wait_for(app, lambda: not any(p.busy for p in (
        window.storage_roots_page, window.details_page, window.compare_page, window.import_page)))
    window.close()


def open_page(app, window):
    window.navigation_buttons["Storage Roots"].click()
    page = window.storage_roots_page
    wait_for(app, lambda: not page.busy)
    return page


def test_startup_does_not_probe_storage_and_page_open_checks_roots_only(app, repository, storage_root, tmp_path, monkeypatch):
    calls = []
    resolve = PathResolver.resolve

    def tracked(self, root, reference, **kwargs):
        calls.append(reference)
        return resolve(self, root, reference, **kwargs)

    monkeypatch.setattr(PathResolver, "resolve", tracked)
    monkeypatch.setattr(repository, "storage_root_references", lambda *a: pytest.fail("Verify is explicit"))
    prefs = ImportPreferences(QSettings(str(tmp_path / "startup.ini"), QSettings.Format.IniFormat), home=tmp_path)
    window = MainWindow(repository, preferences=prefs)
    try:
        assert calls == []
        window.show_storage_roots()
        wait_for(app, lambda: not window.storage_roots_page.busy)
        assert calls == ["."]
    finally:
        window.close()


def test_empty_state_and_add_validation_chooser_refresh(app, window, repository, tmp_path, monkeypatch):
    page = open_page(app, window)
    assert page.empty.isVisible() and "No storage roots" in page.empty.text()
    assert not page.inspector.isVisible()
    page.add_button.click()
    dialog = page.dialog
    dialog.save.click()
    assert "non-empty" in dialog.error.text() and not repository.list_storage_roots()
    dialog.name.setText("Test root")
    dialog.location.setText("relative")
    dialog.save.click()
    assert "absolute location" in dialog.error.text()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: "")
    dialog.browse.click()
    assert dialog.location.text() == "relative"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: str(tmp_path))
    dialog.browse.click()
    dialog.save.click()
    wait_for(app, lambda: not page.busy)
    assert page.roots.count() == 1 and not page.empty.isVisible()
    root = repository.list_storage_roots()[0]
    assert root.location == str(tmp_path) and page.selected.root.id == root.id
    assert "Online" in page.info.text() and page.location.isReadOnly()
    assert page.identity.text() == root.id


def test_edit_remap_preview_and_cancel_preserve_records(app, window, repository, storage_root, tmp_path):
    run = save_references(repository, storage_root)
    page = open_page(app, window)
    page.edit_button.click()
    dialog = page.dialog
    dialog.name.setText("Renamed")
    dialog.save.click()
    wait_for(app, lambda: not page.busy)
    renamed = repository.get_storage_root(storage_root.id)
    assert renamed == replace(storage_root, name="Renamed")
    page.remap_button.click()
    dialog = page.dialog
    dialog.location.setText(str(tmp_path / "offline destination"))
    dialog.save.click()
    assert dialog.preview.isVisible() and dialog.save.text() == "Remap"
    assert "2 artifact references" in dialog.preview_note.text()
    assert "not been verified" in dialog.preview_note.text()
    assert repository.get_storage_root(storage_root.id) == renamed
    dialog.back.click()
    assert dialog.form.isVisible() and dialog.save.text() == "Preview Remap"
    dialog.save.click()
    dialog.reject()
    assert repository.get_storage_root(storage_root.id) == renamed
    page.remap_button.click()
    dialog = page.dialog
    dialog.location.setText(str(tmp_path))
    dialog.save.click()
    dialog.save.click()
    wait_for(app, lambda: not page.busy)
    assert repository.get_storage_root(storage_root.id).location == str(tmp_path)
    assert repository.get_run(run.id) == run and page.selected.root.id == storage_root.id


def test_verify_online_missing_offline_and_portable(app, window, repository, storage_root):
    (Path(storage_root.location) / "present.bin").write_bytes(b"sentinel")
    save_references(repository, storage_root)
    page = open_page(app, window)
    assert "Online" in page.info.text() and "2 artifact references" in page.info.text()
    page.verify_button.click()
    wait_for(app, lambda: not page.busy)
    assert "Available: 1 · Missing: 1 · Offline: 0 · Unverified: 0" in page.verification_text.text()
    assert "Online" in page.info.text()
    page.remap_button.click()
    dialog = page.dialog
    dialog.kind.setCurrentIndex(dialog.kind.findData(RootKind.PORTABLE_RELATIVE))
    assert not dialog.browse.isEnabled()
    dialog.location.setText("portable/runs")
    dialog.save.click()
    dialog.save.click()
    wait_for(app, lambda: not page.busy)
    assert "Unverified" in page.info.text()
    assert "base directory" in page.reason.text()
    page.verify_button.click()
    wait_for(app, lambda: not page.busy)
    assert "Unverified: 2" in page.verification_text.text()


def test_remove_blocked_and_unused_confirmation_files_survive(app, window, repository, storage_root):
    directory = Path(storage_root.location)
    sentinel = directory / "DO_NOT_DELETE.txt"
    sentinel.write_bytes(b"keep me")
    run = save_references(repository, storage_root)
    page = open_page(app, window)
    page.remove_button.click()
    dialog = page.dialog
    assert "2 artifact references" in dialog.explanation.text()
    assert not dialog.remove.isEnabled() and "Remap" in dialog.explanation.text()
    dialog.reject()
    assert repository.get_run(run.id) == run
    repository.delete_run(run.id)  # Explicit fixture cleanup, outside the root UI.
    page.refresh_button.click()
    wait_for(app, lambda: not page.busy)
    page.remove_button.click()
    dialog = page.dialog
    assert "No external folders or files will be deleted" in dialog.explanation.text()
    dialog.reject()
    assert repository.get_storage_root(storage_root.id)
    page.remove_button.click()
    page.dialog.remove.click()
    wait_for(app, lambda: not page.busy)
    assert page.empty.isVisible() and repository.get_storage_root(storage_root.id) is None
    assert sentinel.read_bytes() == b"keep me"


def test_remove_rechecks_dependencies_at_commit(app, window, repository, storage_root):
    page = open_page(app, window)
    page.remove_button.click()
    dialog = page.dialog
    run = save_references(repository, storage_root, paths=())
    dialog.remove.click()
    assert dialog.isVisible() and "1 run source" in dialog.explanation.text()
    assert repository.get_run(run.id) == run and repository.get_storage_root(storage_root.id) == storage_root
    dialog.reject()


def test_persistence_error_is_recoverable_and_does_not_clear_existing_root(app, window, repository, storage_root, monkeypatch):
    page = open_page(app, window)

    def fail(*a, **kw):
        raise sqlite3.OperationalError("technical detail")

    with monkeypatch.context() as patch:
        patch.setattr(repository, "remap_storage_root", fail)
        page.remap_button.click()
        dialog = page.dialog
        dialog.location.setText("/offline")
        dialog.save.click()
        dialog.save.click()
        assert "No catalog records" in dialog.error.text() and "technical detail" not in dialog.error.text()
        assert repository.get_storage_root(storage_root.id) == storage_root
        dialog.reject()
    page.refresh_button.click()
    wait_for(app, lambda: not page.busy)
    assert page.remap_button.isEnabled()


def test_catalog_error_does_not_show_empty_success(app, window, repository, monkeypatch):
    page = open_page(app, window)

    def fail():
        raise sqlite3.OperationalError("unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(repository, "list_storage_root_summaries", fail)
        page.refresh_button.click()
        assert "Cannot load" in page.message.text() and not page.empty.isVisible()
    page.refresh_button.click()
    assert page.empty.isVisible()


def test_saved_change_is_reported_truthfully_when_refresh_fails(app, window, repository, storage_root, monkeypatch):
    page = open_page(app, window)
    page.edit_button.click()
    dialog = page.dialog
    dialog.name.setText("Saved name")

    def fail():
        raise sqlite3.OperationalError("read failed after commit")

    with monkeypatch.context() as patch:
        patch.setattr(repository, "list_storage_root_summaries", fail)
        dialog.save.click()
    assert "change was saved" in page.message.text()
    assert repository.get_storage_root(storage_root.id).name == "Saved name"
    assert not page.inspector.isVisible()
    page.refresh_button.click()
    wait_for(app, lambda: not page.busy)
    assert page.selected.root.name == "Saved name"


class GatedResolver(PathResolver):
    def __init__(self):
        super().__init__()
        self.enabled = False
        self.entered, self.release = threading.Event(), threading.Event()
        self.calls = []

    def resolve(self, root, reference, **kwargs):
        self.calls.append((root, reference))
        if self.enabled:
            self.entered.set()
            assert self.release.wait(4), "Test must release the worker"
        return super().resolve(root, reference, **kwargs)


def test_verify_duplicate_guard_selection_and_gui_responsiveness(app, window, repository, storage_root, tmp_path):
    save_references(repository, storage_root)
    other = StorageRoot(name="Z other", location=str(tmp_path / "offline"))
    repository.add_storage_root(other)
    resolver = GatedResolver()
    page = window.storage_roots_page
    page.resolver = resolver
    open_page(app, window)
    resolver.enabled = True
    page.verify_button.click()
    wait_for(app, resolver.entered.is_set)
    task = page.task
    try:
        page.verify_selected()
        assert page.task is task and not page.verify_button.isEnabled()
        assert not page.remap_button.isEnabled() and not page.add_button.isEnabled()
        page.roots.setCurrentRow(1)
        app.processEvents()
        assert page.selected.root == other and page.identity.text() == other.id
        assert "Offline" in page.info.text() and "Available:" not in page.verification_text.text()
        window.show_library()
        assert window.workspace.currentWidget() is window.library_page
    finally:
        resolver.release.set()
    wait_for(app, lambda: not page.busy)
    assert page.identity.text() == other.id and "Available:" not in page.verification_text.text()
    page.roots.setCurrentRow(0)
    assert "Missing: 2" in page.verification_text.text()


def test_refresh_invalidates_inflight_results_and_uses_latest_mapping(app, window, repository, storage_root, tmp_path):
    page = window.storage_roots_page
    resolver = GatedResolver()
    page.resolver = resolver
    open_page(app, window)
    resolver.enabled = True
    page.verify_button.click()
    wait_for(app, resolver.entered.is_set)
    try:
        mapped = repository.remap_storage_root(storage_root, location=str(tmp_path / "offline"), kind=storage_root.kind)
        page.refresh_button.click()
        assert page.selected.root == mapped and "Unverified" in page.info.text()
    finally:
        resolver.release.set()
    wait_for(app, lambda: not page.busy)
    assert page.selected.root == mapped and "Offline" in page.info.text()
    assert "Available:" not in page.verification_text.text()


def test_close_defers_destruction_until_worker_exits(app, window, storage_root):
    page = window.storage_roots_page
    resolver = GatedResolver()
    page.resolver = resolver
    open_page(app, window)
    resolver.enabled = True
    page.verify_button.click()
    wait_for(app, resolver.entered.is_set)
    try:
        window.close()
        assert window.isVisible() and page.closing and page.busy
    finally:
        resolver.release.set()
    wait_for(app, lambda: not page.busy and not window.isVisible())


def test_verify_error_leaves_mapping_and_allows_retry(app, window, storage_root):
    page = open_page(app, window)

    class Failed(PathResolver):
        def resolve(self, *a, **kw):
            raise OSError("probe unavailable")

    page.resolver = Failed()
    page.verify_button.click()
    wait_for(app, lambda: not page.busy)
    assert "Could not check" in page.message.text() and page.verify_button.isEnabled()
    assert page.selected.root == storage_root
    page.resolver = PathResolver()
    page.verify_button.click()
    wait_for(app, lambda: not page.busy)
    assert "complete" in page.message.text() and "Online" in page.info.text()


def test_offline_long_paths_and_navigation_preserve_selection(app, window, repository, tmp_path):
    long_path = str(tmp_path / ("long-folder-" * 18) / ("another-" * 18))
    root = StorageRoot(name="Offline root", location=long_path)
    repository.add_storage_root(root)
    for identity in ("first", "second"):
        repository.save_run(Run(id=identity, display_name=identity, source_storage_root_id=root.id, source_relative_directory="."))
    window.show_compare()
    comparison = window.compare_page
    comparison.run_a.setCurrentIndex(comparison.run_a.findData("first"))
    comparison.run_b.setCurrentIndex(comparison.run_b.findData("second"))
    wait_for(app, lambda: not comparison.busy)
    page = open_page(app, window)
    for width, height in ((1440, 900), (1040, 640)):
        window.resize(width, height)
        app.processEvents()
        assert page.location.text() == long_path and "Offline" in page.info.text()
        assert page.roots.horizontalScrollBar().maximum() == 0
        assert page.remove_button.isVisible() and page.verify_button.isEnabled()
        assert page.inspector.geometry().right() <= page.width()
    window.show_compare()
    wait_for(app, lambda: not comparison.busy)
    assert (comparison.run_a.currentData(), comparison.run_b.currentData()) == ("first", "second")
    window.show_library()
    window.library_page.open_requested.emit("first")
    wait_for(app, lambda: not window.details_page.busy)
    assert window.details_page.details.run.id == "first"
