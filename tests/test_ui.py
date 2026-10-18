from collections import Counter
from dataclasses import replace
import json
import threading
from time import monotonic

import pytest

from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog

from runpup.domain.models import Dataset, ImportDiagnostic, Metric, MetricSummary, Run
from runpup.domain.proposals import FieldState
from runpup.services.import_commit import ProposalValidationError
from runpup.ui.theme import apply_theme
from runpup.ui.window import MainWindow


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    apply_theme(application)
    yield application


def test_empty_library_opens_real_import_workflow(app, repository):
    window = MainWindow(repository)
    try:
        window.show()
        app.processEvents()
        assert window.content.currentWidget() is window.empty_state
        assert "0 runs" in window.run_count.text()
        window.first_import.click()
        assert window.workspace.currentWidget() is window.import_page
        assert window.import_page.proposal is None
        assert window.import_page.choose_button.isEnabled()
        assert not window.import_page.save_button.isEnabled()
        assert window.navigation_buttons["Import Run"].isChecked()
        assert repository.list_runs() == []
    finally:
        window.close()


def wait_for(app, condition, timeout=5):
    deadline = monotonic() + timeout
    while not condition() and monotonic() < deadline:
        app.processEvents()
        QTest.qWait(5)
    assert condition(), "UI operation did not finish"
    app.processEvents()


@pytest.fixture
def window(app, repository):
    widget = MainWindow(repository)
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    wait_for(app, lambda: not widget.import_page.busy)
    widget.close()


@pytest.fixture
def training_folder(tmp_path):
    directory = tmp_path / "training Δ"
    directory.mkdir()
    documents = {
        "trainer_state.json": {"log_history": [{"step": 40, "loss": 0.7, "eval_loss": 0.5}]},
        "training_args.json": {"run_name": "Useful import", "model_name_or_path": "org/base",
                               "learning_rate": 0.0002, "lora_rank": 16, "num_train_epochs": 3,
                               "per_device_train_batch_size": 2, "gradient_accumulation_steps": 4},
        "adapter_config.json": {"base_model_name_or_path": "org/base", "peft_type": "LORA",
                                "r": 32, "lora_alpha": 64, "target_modules": ["q_proj", "v_proj"]},
    }
    for name, value in documents.items():
        (directory / name).write_text(json.dumps(value))
    (directory / "adapter_model.safetensors").write_bytes(b"synthetic payload")
    (directory / "training_args.bin").write_bytes(b"never deserialize")
    (directory / "dataset_info.json").write_text("{malformed optional metadata")
    return directory


def scan(app, window, path):
    window.show_import()
    page = window.import_page
    page.select_folder(path)
    page.scan_button.click()
    wait_for(app, lambda: not page.busy)
    return page


def resolve_rank(app, page):
    row = page.field_rows["lora_rank"]
    index = row.choice.findData(32)
    row.choice.setCurrentIndex(index)
    row.choice.activated.emit(index)
    wait_for(app, lambda: not page.busy)


def test_folder_picker_cancel_and_selection(window, monkeypatch, training_folder):
    page = window.import_page
    window.navigation_buttons["Import Run"].click()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: "")
    page.choose_button.click()
    assert page.source_root is None and page.proposal is None
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(training_folder))
    page.choose_button.click()
    assert page.source_path.text() == str(training_folder) and page.scan_button.isEnabled()
    previous = page.source_root
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: "")
    page.choose_button.click()
    assert page.source_root == previous and page.proposal is None


def test_scan_review_conflict_sources_diagnostics_and_live_counts(app, window, training_folder):
    page = scan(app, window, training_folder)
    assert page.proposal is not None
    assert page.field_rows["base_model"].badge.text() == "DETECTED"
    assert page.field_rows["dataset_identity"].badge.text() == "UNKNOWN"
    rank = page.field_rows["lora_rank"]
    assert rank.badge.text() == "CONFLICT" and rank.evidence.isVisible()
    assert "32" in rank.evidence.text() and "16" in rank.evidence.text()
    assert "adapter_config.json" in rank.evidence.text() and "training_args.json" in rank.evidence.text()
    assert "lora_rank" in page.blocking_text.text() and not page.save_button.isEnabled()
    assert "not deserialized" in page.diagnostics_text.text()
    assert "JSON metadata" in page.diagnostics_text.text()
    counts = Counter(f.state.value for f in page.proposal.fields)
    for state in ("detected", "unknown", "conflict"):
        assert page.count_labels[state].text().splitlines()[0] == str(counts[state])
    evidence = page.proposal.field("lora_rank").candidates
    resolve_rank(app, page)
    assert page.proposal.field("lora_rank").state is FieldState.USER_EDITED
    assert page.proposal.field("lora_rank").value == 32
    assert page.proposal.field("lora_rank").candidates == evidence
    assert page.field_rows["lora_rank"].badge.text() == "USER EDITED"
    assert page.save_button.isEnabled()  # Optional unknown dataset does not block.
    assert page.count_labels["conflict"].text().startswith("0\n")


def test_typed_manual_edit_pending_guard_and_discard(app, window, training_folder):
    page = scan(app, window, training_folder)
    resolve_rank(app, page)
    row = page.field_rows["learning_rate"]
    original = page.proposal.field("learning_rate").candidates
    row.editor.setFocus()
    row.editor.selectAll()
    QTest.keyClicks(row.editor, "0.0003")
    assert not page.save_button.isEnabled() and "pending edits" in page.blocking_text.text()
    row.apply_button.click()
    wait_for(app, lambda: not page.busy)
    assert page.proposal.field("learning_rate").value == 0.0003
    assert type(page.proposal.field("learning_rate").value) is float
    assert page.proposal.field("learning_rate").candidates == original
    assert row.badge.text() == "USER EDITED"
    integer = page.field_rows["lora_rank"]
    integer.editor.selectAll()
    QTest.keyClicks(integer.editor, "16.5")
    integer.apply_button.click()
    assert integer.error.isVisible() and not page.save_button.isEnabled()
    assert page.proposal.field("lora_rank").value == 32
    integer.revert.click()
    assert page.save_button.isEnabled() and "No blocking" in page.blocking_text.text()
    modules = page.field_rows["lora_target_modules"]
    modules.editor.selectAll()
    QTest.keyClicks(modules.editor, '["k_proj", "o_proj"]')
    modules.apply_button.click()
    wait_for(app, lambda: not page.busy)
    assert json.loads(page.proposal.field("lora_target_modules").value) == ["k_proj", "o_proj"]


def test_successful_save_refreshes_repository_library_and_guards_repeat(app, window, repository, training_folder, monkeypatch):
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in training_folder.iterdir()}
    page = scan(app, window, training_folder)
    resolve_rank(app, page)
    proposal_id = page.proposal.id
    calls = []
    commit = page.service.commit
    def tracked_commit(proposal):
        calls.append(proposal.id)
        return commit(proposal)
    monkeypatch.setattr(page.service, "commit", tracked_commit)
    page.save_button.click()
    page.save_button.click()
    page.save_run()
    wait_for(app, lambda: not page.busy)
    assert calls == [proposal_id]
    assert repository.get_run(proposal_id).display_name == "Useful import"
    assert window.workspace.currentWidget() is window.library_page
    assert window.table.rowCount() == 1 and window.table.item(0, 0).text() == "Useful import"
    assert window.table.item(0, 0).data(Qt.ItemDataRole.UserRole) == proposal_id
    assert window.table.selectedItems() and "Saved" in window.save_notice.text()
    assert page.proposal is None and page.source_root is None
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in training_folder.iterdir()} == before


def test_save_failure_keeps_review_for_retry(app, window, repository, training_folder, monkeypatch):
    page = scan(app, window, training_folder)
    resolve_rank(app, page)
    reviewed = page.proposal
    original = page.service.commit
    def fail(_):
        raise RuntimeError("private SQL detail must not be displayed")
    monkeypatch.setattr(page.service, "commit", fail)
    page.save_button.click()
    wait_for(app, lambda: not page.busy)
    assert page.proposal is reviewed and page.save_button.isEnabled()
    assert "Nothing was saved" in page.message.text() and "try Save Run again" in page.message.text()
    assert "SQL" not in page.message.text()
    assert window.workspace.currentWidget() is page
    assert repository.list_runs() == [] and window.table.rowCount() == 0
    monkeypatch.setattr(page.service, "commit", original)
    page.save_button.click()
    wait_for(app, lambda: not page.busy)
    assert repository.get_run(reviewed.id) is not None


@pytest.mark.parametrize("kind", ["unrecognized", "disappeared", "unreadable", "exception"])
def test_scan_error_states_do_not_create_runs(app, window, repository, tmp_path, monkeypatch, kind):
    page = window.import_page
    directory = tmp_path / kind
    directory.mkdir()
    page.select_folder(directory)
    if kind == "disappeared":
        directory.rmdir()
    elif kind == "exception":
        def fail(*args):
            raise RuntimeError("internal stack information")
        monkeypatch.setattr(page.importer, "detect", fail)
    elif kind == "unreadable":
        from runpup.importers.hf_trainer import DetectionResult
        from runpup.domain.models import ImportDiagnostic
        diagnostic = ImportDiagnostic(message="Selected source cannot be safely accessed.", kind="source_unreadable")
        monkeypatch.setattr(page.importer, "detect", lambda *args: DetectionResult(
            False, (), (diagnostic,), (diagnostic.id,)))
    page.start_scan()
    wait_for(app, lambda: not page.busy)
    assert page.proposal is None and not page.save_button.isEnabled()
    assert "try again" in page.message.text()
    assert "internal stack" not in page.message.text()
    if kind == "unreadable":
        assert "cannot be safely accessed" in page.diagnostics_text.text()
    assert repository.list_runs() == []


def test_scan_worker_responsive_navigation_and_close_lifecycle(app, window, training_folder, monkeypatch):
    page = window.import_page
    original = page.importer.detect
    entered, release = threading.Event(), threading.Event()
    threads = []
    completion_threads = []
    page.idle.connect(lambda: completion_threads.append(QThread.currentThread()))
    def delayed(*args):
        threads.append(QThread.currentThread())
        entered.set()
        assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(page.importer, "detect", delayed)
    window.show_import()
    page.select_folder(training_folder)
    page.start_scan()
    wait_for(app, entered.is_set)
    try:
        assert not page.choose_button.isEnabled() and not page.scan_button.isEnabled()
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        wait_for(app, lambda: bool(ticks))
        assert threads[0] != app.thread()
        window.navigation_buttons["Library"].click()
        assert window.workspace.currentWidget() is window.library_page
        window.close()
        assert window.isVisible() and "before closing" in page.message.text()
    finally:
        release.set()
    wait_for(app, lambda: not page.busy)
    assert not window.isVisible()
    assert completion_threads == [app.thread()]


def test_commit_revalidation_failure_shows_blockers(app, window, training_folder, monkeypatch):
    page = scan(app, window, training_folder)
    resolve_rank(app, page)
    retained = page.proposal
    invalid = replace(page.validation, run=None, blocking=(ImportDiagnostic(
        message="Storage root definition changed; choose the folder again."),))
    def fail(_):
        raise ProposalValidationError(invalid)
    monkeypatch.setattr(page.service, "commit", fail)
    page.save_run()
    wait_for(app, lambda: not page.busy)
    assert page.proposal is retained and not page.save_button.isEnabled()
    assert "Storage root definition changed" in page.blocking_text.text()


def test_source_disappearing_between_detection_and_scan_is_blocked(app, window, training_folder, monkeypatch):
    page = window.import_page
    original = page.importer.scan
    def moved(root):
        training_folder.rename(training_folder.with_name("moved"))
        return original(root)
    monkeypatch.setattr(page.importer, "scan", moved)
    scan(app, window, training_folder)
    assert not page.save_button.isEnabled()
    assert page.validation.blocking
    assert "source" in page.blocking_text.text().lower()


def test_scan_completion_does_not_steal_library_navigation(app, window, training_folder):
    page = window.import_page
    page.select_folder(training_folder)
    page.start_scan()
    window.show_library()
    wait_for(app, lambda: not page.busy)
    assert page.proposal is not None
    assert window.workspace.currentWidget() is window.library_page
    window.show_import()
    assert page.field_rows["lora_rank"].badge.text() == "CONFLICT"


def test_unsuccessful_rescan_keeps_previous_review_and_requires_fresh_validation(app, window, training_folder, monkeypatch):
    from runpup.importers.hf_trainer import DetectionResult
    page = scan(app, window, training_folder)
    resolve_rank(app, page)
    retained = page.proposal
    monkeypatch.setattr(page.importer, "detect", lambda *args: DetectionResult(False, (), (), ()))
    page.start_scan()
    wait_for(app, lambda: not page.busy)
    assert page.proposal is retained
    assert "Previous review is kept" in page.message.text()
    assert not page.save_button.isEnabled()


def test_narrow_review_remains_within_viewport(app, window, training_folder):
    page = scan(app, window, training_folder)
    window.resize(1040, 640)
    app.processEvents()
    assert page.review_scroll.horizontalScrollBar().maximum() == 0
    assert all(card.width() <= page.review_scroll.viewport().width() for card in page.cards)
    assert page.card_columns[1].count() == 0
    assert all(row.editor.width() >= 60 for row in page.field_rows.values())


def test_library_refresh_displays_repository_records(app, repository, storage_root):
    window = MainWindow(repository)
    try:
        run = Run(display_name="Useful run", source_storage_root_id=storage_root.id,
                  source_relative_directory="one", base_model="model/base", method="LoRA", rating=4,
                  datasets=(Dataset(name="Training set"),), metrics=(
                      Metric(name="loss", value=0.8, context="train", summary=MetricSummary.FINAL),
                      Metric(name="loss", value=0.1, context="validation", summary=MetricSummary.MINIMUM),
                  ))
        repository.save_run(run)
        window.refresh_library()
        window.show()
        app.processEvents()
        assert window.content.currentWidget() is window.table
        assert window.table.rowCount() == 1
        assert [window.table.item(0, col).text() for col in range(7)] == [
            "Useful run", "model/base", "LoRA", "Training set", "0.8", "4/5", "Unknown",
        ]
        window.table.selectRow(0)
        assert window.inspector_name.text() == run.display_name
    finally:
        window.close()
