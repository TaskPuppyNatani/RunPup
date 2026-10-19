import builtins
from dataclasses import replace
import io
import sqlite3
import threading
from time import monotonic, sleep

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

from runpup.domain.models import Artifact, ImportDiagnostic, Run, StorageRoot
from runpup.importers import HFTrainerImporter
from runpup.ui.preferences import ImportPreferences
from runpup.ui.theme import apply_theme
from runpup.ui.window import MainWindow
from test_run_details import saved_run  # Shared representative canonical aggregate fixture.


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    apply_theme(application)
    return application


def wait_for(app, condition):
    end = monotonic() + 5
    while not condition() and monotonic() < end:
        app.processEvents()
        sleep(.005)  # Yield Python execution to the catalog/reference worker.
    assert condition()
    app.processEvents()


@pytest.fixture
def window(app, repository, tmp_path):
    preferences = ImportPreferences(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat), home=tmp_path)
    window = MainWindow(repository, preferences=preferences)
    window.show()
    app.processEvents()
    yield window
    window.close()
    wait_for(app, lambda: not window.details_page.busy and not window.import_page.busy)
    window.close()


def open_run(app, window, run):
    window.show_library()
    row = next(i for i, stored in enumerate(window.runs) if stored.id == run.id)
    window.table.selectRow(row)
    window.open_details_button.click()
    wait_for(app, lambda: not window.details_page.busy)
    return window.details_page


def visible_text(page):
    return "\n".join(label.text() for label in page.scroll.widget().findChildren(QLabel))


def test_library_keyboard_activation_loads_current_identity_and_back(app, window, repository, saved_run):
    window.refresh_library()
    window.table.selectRow(0)
    repository.save_run(replace(saved_run, display_name="Changed after Library loaded"))
    window.table.setFocus()
    QTest.keyClick(window.table, Qt.Key.Key_Return)
    wait_for(app, lambda: not window.details_page.busy)
    page = window.details_page
    assert window.workspace.currentWidget() is page
    assert page.details.run.id == saved_run.id
    assert page.name_label.text() == "Changed after Library loaded"
    page.back_button.click()
    assert window.workspace.currentWidget() is window.library_page
    assert window.table.item(0, 0).text() == "Changed after Library loaded"


@pytest.mark.parametrize("failure", ("deleted", "database", "invalid"))
def test_details_load_failure_is_recoverable(app, window, repository, saved_run, monkeypatch, failure):
    window.refresh_library()
    window.table.selectRow(0)
    if failure == "deleted":
        repository.delete_run(saved_run.id)
    else:
        def fail(_):
            raise sqlite3.OperationalError("Read failed") if failure == "database" else ValueError("Invalid stored value")
        monkeypatch.setattr(repository, "get_run", fail)
    window.open_details_button.click()
    wait_for(app, lambda: not window.details_page.busy)
    page = window.details_page
    assert "Cannot open saved run" in page.status.text()
    assert page.details is None and not page.scroll.isVisible()
    page.back_button.click()
    assert window.workspace.currentWidget() is window.library_page


def test_saved_fields_metrics_provenance_and_human_metadata(app, window, saved_run):
    page = open_run(app, window, saved_run)
    text = visible_text(page)
    for expected in ("org/model", "Training set", "dataset:v2", "keeper", "Planned epochs", "Unknown",
                     "Configured learning rate", "LoRA rank (r)", "16", "8", "Not deserialized"):
        assert expected in text
    assert page.notes_editor.toPlainText() == "Keep this"
    assert page.favorite.isChecked() and page.rating.currentData() == 4
    assert page.metrics_table.rowCount() == 6
    rows = {tuple(page.metrics_table.item(row, col).text() for col in range(6)) for row in range(6)}
    assert len(rows) == 6
    assert ("loss", "0.5", "train", "10", "bits", "final") in rows
    assert not page.evidence_texts[0].isVisible()
    page.provenance_button.click()
    for button in page.evidence_buttons:
        button.click()
    assert all(text.isVisible() for text in page.evidence_texts)
    evidence = "\n".join(text.text() for text in page.evidence_texts)
    assert "EXACT · User edit" in evidence and "DERIVED · Source candidate" in evidence
    assert "config.json#/lora_rank" in evidence
    assert "Original: 32\nCandidate: 32\nCurrent: 16" in evidence


def test_persisted_diagnostics_reuse_grouping_without_source_reads(app, window, repository, saved_run, monkeypatch):
    diagnostics = (ImportDiagnostic(message="Discovered but not deserialized", source_reference="training_args.bin"),
                   ImportDiagnostic(message="Discovered but not deserialized", source_reference="checkpoint-10/training_args.bin"))
    run = replace(saved_run, diagnostics=diagnostics)
    repository.save_run(run)

    def forbidden(*args, **kwargs):
        raise AssertionError("Details must not rescan or read source files")

    monkeypatch.setattr(HFTrainerImporter, "scan", forbidden)
    monkeypatch.setattr(HFTrainerImporter, "detect", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(io, "open", forbidden)
    page = open_run(app, window, run)
    assert page.details.run.diagnostics == diagnostics
    assert "WARNING — 2 occurrences" in page.diagnostics_text.text()
    assert page.diagnostics_text.text().count("Discovered but not deserialized") == 1
    page.copy_button.click()
    copied = QApplication.clipboard().text()
    assert copied == "RunPup Import Diagnostics\n\n" + page.diagnostics_text.text()
    assert "- training_args.bin" in copied and "- checkpoint-10/training_args.bin" in copied


def test_opening_different_run_replaces_previous_state_and_empty_sections(app, window, repository, storage_root, saved_run):
    open_run(app, window, saved_run)
    empty = Run(display_name="Empty saved run", source_storage_root_id=storage_root.id, source_relative_directory=".")
    repository.save_run(empty)
    page = open_run(app, window, empty)
    assert page.details.run == empty and page.name_label.text() == empty.display_name
    assert page.metrics_table is None and page.artifacts_table is None
    assert not page.favorite.isChecked() and page.rating.currentData() is None
    assert page.notes_editor.toPlainText() == "" and page.evidence_buttons == []
    assert not page.copy_button.isEnabled()
    text = visible_text(page)
    for expected in ("No metrics were recorded", "No artifacts recorded", "No persisted provenance",
                     "No persisted import diagnostics", "No tags."):
        assert expected in text
    assert "org/model" not in text and "config.json#/lora_rank" not in text


def test_artifacts_show_resolver_states_and_full_stored_reference(app, window, repository, storage_root, saved_run, tmp_path):
    from pathlib import Path

    location = Path(storage_root.location)
    (location / "present.bin").write_bytes(b"reference only")
    offline = StorageRoot(name="Offline root", location=str(tmp_path / "disconnected"))
    repository.save_storage_root(offline)
    artifacts = tuple(Artifact(run_id=saved_run.id, storage_root_id=root.id, relative_path=path,
                               display_name=path, category="checkpoint", size_bytes=14,
                               hash_algorithm="sha256", hash_value="upstream-only-hash")
                      for root, path in ((storage_root, "present.bin"), (offline, "remote.bin"),
                                         (storage_root, "gone.bin")))
    run = replace(saved_run, artifacts=artifacts, diagnostics=())
    repository.save_run(run)
    page = open_run(app, window, run)
    assert [page.artifacts_table.item(row, 2).text() for row in range(3)] == ["Available", "Offline", "Missing"]
    assert "upstream-only-hash" in page.artifact_info.text()
    page.artifacts_table.selectRow(1)
    assert "Offline root" in page.artifact_info.text() and "Availability: Offline" in page.artifact_info.text()
    page.artifacts_table.selectRow(2)
    assert "gone.bin" in page.artifact_info.text() and "Availability: Missing" in page.artifact_info.text()
    assert "Recorded verification: unverified" in page.artifact_info.text()


def test_human_metadata_updates_only_existing_write_seam_and_refreshes_library(app, window, repository, saved_run):
    page = open_run(app, window, saved_run)
    page.notes_editor.setPlainText("Updated notes\nSecond line")
    page.favorite.setChecked(False)
    page.rating.setCurrentIndex(page.rating.findData(0))
    page.save_metadata_button.click()
    wait_for(app, lambda: not page.busy)
    stored = repository.get_run(saved_run.id)
    assert stored == replace(saved_run, notes="Updated notes\nSecond line", favorite=False, rating=0)
    assert page.details.run == stored and page.metadata_notice.text() == "Metadata saved."
    assert window.workspace.currentWidget() is page
    page.back_button.click()
    row = next(i for i, run in enumerate(window.runs) if run.id == stored.id)
    assert window.table.item(row, 5).text() == "0/5"


def test_human_update_failure_keeps_edit_for_retry(app, window, repository, saved_run, monkeypatch):
    page = open_run(app, window, saved_run)
    page.notes_editor.setPlainText("Pending notes")

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("Read-only catalog")

    monkeypatch.setattr(repository, "update_human_metadata", fail)
    page.save_metadata_button.click()
    assert "Could not save metadata" in page.metadata_notice.text()
    assert page.notes_editor.toPlainText() == "Pending notes"
    assert repository.get_run(saved_run.id) == saved_run


def test_queued_open_uses_latest_request_and_close_waits_for_worker(app, window, repository, storage_root, saved_run, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = repository.get_run

    def slow(run_id):
        entered.set()
        assert release.wait(5)
        return original(run_id)

    second = Run(display_name="Second", source_storage_root_id=storage_root.id, source_relative_directory=".")
    repository.save_run(second)
    monkeypatch.setattr(repository, "get_run", slow)
    page = window.details_page
    page.load_run(saved_run.id)
    wait_for(app, entered.is_set)
    page.load_run(second.id)
    release.set()
    wait_for(app, lambda: not page.busy)
    assert page.details.run == second
    entered.clear()
    release.clear()
    page.load_run(saved_run.id)
    wait_for(app, entered.is_set)
    window.close()
    assert window.isVisible() and window._close_requested
    release.set()
    wait_for(app, lambda: not window.isVisible())


@pytest.mark.parametrize("size", ((1440, 900), (1040, 640)))
def test_long_values_reflow_with_accessible_full_text(app, window, repository, saved_run, size):
    run = replace(saved_run, display_name="Long run " * 40, base_model="org/" + "model-" * 80)
    repository.save_run(run)
    window.resize(*size)
    page = open_run(app, window, run)
    wait_for(app, lambda: page.scroll.verticalScrollBar().maximum() > 0)
    app.processEvents()
    assert page.name_label.wordWrap() and page.name_label.text() == run.display_name
    assert page.scroll.horizontalScrollBar().maximum() == 0
    assert page.scroll.widget().width() <= page.scroll.viewport().width()
    columns = 2 if size[0] == 1440 else 1
    assert page.metadata_grid.getItemPosition(page.metadata_grid.indexOf(page.metadata_cards[1]))[1] == columns - 1
    page.scroll.ensureWidgetVisible(page.notes_editor)
    app.processEvents()
    assert page.notes_editor.isVisible() and page.save_metadata_button.isEnabled()
