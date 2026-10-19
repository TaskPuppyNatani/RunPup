import builtins
from dataclasses import replace
import io
import sqlite3
import threading

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QLabel, QToolButton

from runpup.domain.models import Run, StorageRoot
from runpup.importers import HFTrainerImporter
from runpup.services.paths import PathResolver
from runpup.ui.comparison import ComparisonStatus as S
from runpup.ui.preferences import ImportPreferences
from runpup.ui.window import MainWindow
from test_comparison import pair
from test_details_ui import app, wait_for


@pytest.fixture
def window(app, repository, tmp_path):
    preferences = ImportPreferences(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat), home=tmp_path)
    window = MainWindow(repository, preferences=preferences)
    window.show()
    app.processEvents()
    yield window
    window.close()
    wait_for(app, lambda: not any(p.busy for p in (window.compare_page, window.details_page, window.import_page)))
    window.close()


@pytest.fixture
def saved_pair(repository, tmp_path, pair):
    repository.save_storage_root(StorageRoot(id="root", name="Offline outputs", location=str(tmp_path / "offline")))
    a, b = pair
    tags = {t.name: t for t in a.tags}
    b = replace(b, tags=tuple(tags[t.name] for t in b.tags))
    c = Run(id="c", display_name="Third run", source_storage_root_id="root", source_relative_directory="c")
    for run in (a, b, c):
        repository.save_run(run)
    return a, b, c


def select(app, window, a="a", b="b"):
    window.show_compare()
    page = window.compare_page
    for combo, identity in ((page.run_a, a), (page.run_b, b)):
        combo.setCurrentIndex(combo.findData(identity))
    wait_for(app, lambda: not page.busy)
    return page


def items(page):
    rows = {}
    for index in range(page.tree.topLevelItemCount()):
        section = page.tree.topLevelItem(index)
        for child in range(section.childCount()):
            item = section.child(child)
            row = item.data(0, Qt.ItemDataRole.UserRole)
            if row is not None:
                rows[row.key] = item
    return rows


def test_zero_and_one_run_states_offer_import(app, window, repository, storage_root):
    window.navigation_buttons["Compare Runs"].click()
    page = window.compare_page
    assert "At least two" in page.status.text() and not page.compare_button.isEnabled()
    repository.save_run(Run(display_name="Only run", source_storage_root_id=storage_root.id, source_relative_directory="."))
    window.show_compare()
    assert "At least two" in page.status.text()
    assert page.run_a.count() == 2 and page.comparison is None
    page.import_requested.emit()
    assert window.workspace.currentWidget() is window.import_page


def test_selectors_same_run_validation_counts_and_differences_only(app, window, saved_pair):
    window.show_compare()
    page = window.compare_page
    page.run_a.setCurrentIndex(page.run_a.findData("a"))
    assert "Run A and Run B" in page.status.text() and not page.busy
    page.run_b.setCurrentIndex(page.run_b.findData("a"))
    assert "different" in page.status.text() and page.comparison is None
    page.run_b.setCurrentIndex(page.run_b.findData("b"))
    wait_for(app, lambda: not page.busy)
    assert page.comparison.run_a.id == "a" and page.comparison.run_b.id == "b"
    assert page.run_a.itemText(page.run_a.findData("a")) != page.run_a.itemText(page.run_a.findData("b"))
    assert all(item.data(0, Qt.ItemDataRole.UserRole).status in {S.DIFFERENT, S.ONLY_A, S.ONLY_B} for item in items(page).values())
    assert "BOTH UNKNOWN" in page.summary.text() and "SAME" in page.summary.text()
    page.differences_only.setChecked(False)
    assert len(items(page)) == len(page.comparison.rows)
    assert items(page)["run", "training_status"].text(3) == "BOTH UNKNOWN"
    assert items(page)["parameter", "lora_rank", None].text(3) == "SAME"


def test_swap_and_filters_do_not_reload_catalog(app, window, saved_pair, repository, monkeypatch):
    page = select(app, window)
    before = page.comparison

    def forbidden(*args, **kwargs):
        raise AssertionError("Visual comparison controls must use loaded records")

    monkeypatch.setattr(repository, "get_run", forbidden)
    monkeypatch.setattr(repository, "list_run_summaries", forbidden)
    page.swap_button.click()
    assert page.run_a.currentData() == "b" and page.run_b.currentData() == "a"
    assert page.comparison.run_a == before.run_b
    assert items(page)["parameter", "gradient_accumulation", None].text(3) == "ONLY B"
    page.differences_only.setChecked(False)
    assert items(page)["parameter", "lora_rank", None].text(3) == "SAME"


def test_row_evidence_keeps_both_sides_and_manual_history(app, window, saved_pair):
    page = select(app, window)
    page.differences_only.setChecked(False)
    row = items(page)["parameter", "load_in_8bit", None]
    row.setExpanded(True)
    app.processEvents()
    panel = page.tree.itemWidget(row.child(0), 0)
    assert panel is not None
    for button in panel.findChildren(QToolButton):
        button.click()
    app.processEvents()
    text = "\n".join(label.text() for label in panel.findChildren(QLabel))
    assert "Run A" in text and "Run B" in text
    assert "DERIVED" in text and "EXACT" in text and "False" in text
    for evidence in panel.findChildren(QLabel):
        if "Source type:" in evidence.text():
            assert evidence.height() >= evidence.fontMetrics().height() * (evidence.text().count("\n") + 1)
    rank = items(page)["parameter", "lora_rank", None]
    rank.setExpanded(True)
    app.processEvents()
    panel = page.tree.itemWidget(rank.child(0), 0)
    for button in panel.findChildren(QToolButton):
        button.click()
    text = "\n".join(label.text() for label in panel.findChildren(QLabel))
    assert "USER EDITED" in text and "Original: 32\nCandidate: 32\nCurrent: 16" in text
    assert rank.text(3) == "SAME"


@pytest.mark.parametrize("side", ("A", "B"))
def test_details_navigation_preserves_selection_and_refreshes_human_edits(app, window, saved_pair, side):
    page = select(app, window)
    page.differences_only.setChecked(False)
    index = 0 if side == "A" else 1
    page.open_buttons[index].click()
    wait_for(app, lambda: not window.details_page.busy)
    details = window.details_page
    assert details.details.run.id == ("a" if side == "A" else "b")
    assert details.back_button.text() == "← Compare"
    details.notes_editor.setPlainText("Changed in Details")
    details.rating.setCurrentIndex(details.rating.findData(3))
    details.favorite.setChecked(True)
    details.save_metadata_button.click()
    wait_for(app, lambda: not details.busy)
    details.back_button.click()
    wait_for(app, lambda: not page.busy)
    assert window.workspace.currentWidget() is page
    assert page.run_a.currentData() == "a" and page.run_b.currentData() == "b"
    assert not page.differences_only.isChecked()
    updated = page.comparison.run_a if side == "A" else page.comparison.run_b
    assert updated.notes == "Changed in Details" and updated.rating == 3 and updated.favorite
    page.open_buttons[1 - index].click()
    wait_for(app, lambda: not details.busy)
    assert details.details.run.id == ("b" if side == "A" else "a")


@pytest.mark.parametrize("failure", ("deleted", "database"))
def test_failed_reload_clears_stale_rows_and_allows_reselection(app, window, saved_pair, repository, monkeypatch, failure):
    page = select(app, window)
    original = repository.get_run
    if failure == "deleted":
        repository.delete_run("b")
    else:
        def failed(_):
            raise sqlite3.OperationalError("Read failed")
        monkeypatch.setattr(repository, "get_run", failed)
    page.compare_button.click()
    wait_for(app, lambda: not page.busy)
    assert page.comparison is None and not page.headers.isVisible() and page.tree.topLevelItemCount() == 0
    assert "Cannot compare saved runs" in page.status.text()
    monkeypatch.setattr(repository, "get_run", original)
    page.run_b.setCurrentIndex(page.run_b.findData("c"))
    wait_for(app, lambda: not page.busy)
    assert page.comparison.run_b.id == "c"
    assert items(page)["run", "base_model"].text(3) == "ONLY A"


def test_compare_does_not_probe_roots_read_payloads_or_invoke_importer(app, window, saved_pair, repository, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Comparison must read saved catalog data only")

    monkeypatch.setattr(PathResolver, "resolve", forbidden)
    monkeypatch.setattr(repository, "list_runs", forbidden)
    monkeypatch.setattr(HFTrainerImporter, "detect", forbidden)
    monkeypatch.setattr(HFTrainerImporter, "scan", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(io, "open", forbidden)
    page = select(app, window)
    assert page.comparison is not None
    assert any(key[0] == "artifact" for key in items(page))


def test_pending_selection_replacement_and_close_lifecycle(app, window, saved_pair, repository, monkeypatch):
    page = select(app, window)
    entered, release = threading.Event(), threading.Event()
    original = repository.get_run

    def slow(run_id):
        entered.set()
        assert release.wait(5)
        return original(run_id)

    monkeypatch.setattr(repository, "get_run", slow)
    page.compare_button.click()
    wait_for(app, entered.is_set)
    page.run_b.setCurrentIndex(page.run_b.findData("c"))
    assert page.comparison is None and not page.headers.isVisible()
    release.set()
    wait_for(app, lambda: not page.busy)
    assert page.comparison.run_b.id == "c"
    entered.clear()
    release.clear()
    page.compare_button.click()
    wait_for(app, entered.is_set)
    window.close()
    assert window.isVisible() and window._close_requested
    release.set()
    wait_for(app, lambda: not window.isVisible())


@pytest.mark.parametrize("size", ((1440, 900), (1040, 640)))
def test_long_comparison_values_and_expansion_fit_target_sizes(app, window, saved_pair, repository, size):
    a = replace(saved_pair[0], display_name="Long shared run name " * 30, base_model="/model/" + "long-reference/" * 60)
    repository.save_run(a)
    window.resize(*size)
    page = select(app, window)
    row = items(page)["run", "base_model"]
    row.setExpanded(True)
    app.processEvents()
    assert page.header_names[0].text() == a.display_name
    assert a.base_model in row.toolTip(1)
    assert page.tree.horizontalScrollBar().maximum() == 0
    assert page.tree.width() <= page.width()
    assert all(combo.isVisible() and combo.width() > 0 for combo in (page.run_a, page.run_b))
    assert page.tree.itemWidget(row.child(0), 0) is not None
