from dataclasses import replace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from runpup.domain.models import Parameter, Run, Tag
from runpup.importers import HFTrainerImporter
from runpup.services.paths import PathResolver
from test_details_ui import app, window, wait_for
from test_library import summaries


@pytest.fixture
def catalog(repository, storage_root, summaries):
    tags = {}
    runs = []
    for summary in summaries:
        run_tags = tuple(tags.setdefault(name, Tag(name=name)) for name in summary.tags)
        run = Run(id=summary.id, display_name=summary.display_name, source_storage_root_id=storage_root.id,
                  source_relative_directory=".", imported_at=summary.imported_at, base_model=summary.base_model,
                  method=summary.method, source_trainer=summary.source_trainer, rating=summary.rating,
                  favorite=summary.favorite, tags=run_tags,
                  parameters=(Parameter(key="observed_epoch", value=4.0), Parameter(key="global_step", value=160),
                              Parameter(key="max_steps", value=160)))
        repository.save_run(run)
        runs.append(run)
    return runs


def choose(combo, value):
    index = combo.findData(value)
    assert index >= 0
    combo.setCurrentIndex(index)


def ids(page):
    return [run.id for run in page.model.rows]


@pytest.mark.parametrize("activation", ("button", "enter", "double_click"))
def test_filtered_sorted_activation_opens_visible_identity_and_preserves_selection(app, window, catalog, activation):
    page = window.library_page
    window.refresh_library()
    page.search.setText(" ALPHA ")
    choose(page.method, "lora")
    choose(page.sort, "name_desc")
    assert ids(page) == ["a", "d"]
    index = page.model.index(1)
    page.run_list.setCurrentIndex(index)
    if activation == "button":
        page.open_button.click()
    elif activation == "enter":
        page.run_list.setFocus()
        QTest.keyClick(page.run_list, Qt.Key.Key_Return)
    else:
        app.processEvents()
        QTest.mouseClick(page.run_list.viewport(), Qt.MouseButton.LeftButton, pos=page.run_list.visualRect(index).center())
        QTest.mouseDClick(page.run_list.viewport(), Qt.MouseButton.LeftButton, pos=page.run_list.visualRect(index).center())
    wait_for(app, lambda: not window.details_page.busy)
    assert window.workspace.currentWidget() is window.details_page
    assert window.details_page.details.run.id == "d"
    window.details_page.back_button.click()
    assert page.selected_id == "d" and page.run_list.currentIndex().row() == 1
    assert page.criteria().query == " ALPHA " and page.criteria().method == "lora" and page.criteria().sort == "name_desc"
    page.run_list.setCurrentIndex(page.model.index(0))
    page.open_button.click()
    wait_for(app, lambda: not window.details_page.busy)
    assert window.details_page.details.run.id == "a"


def test_no_matches_clear_action_and_search_preserve_filters(app, window, catalog):
    page = window.library_page
    window.refresh_library()
    choose(page.sort, "rating_desc")
    choose(page.favorites, True)
    page.search.setText("nonexistent")
    assert page.content.currentWidget() is page.no_results and "0 of 4" in page.count.text()
    page.search.clear()
    assert ids(page) == ["a", "d"] and page.favorites.currentData() is True
    page.run_list.setCurrentIndex(page.model.index(0))
    page.search.setText("beta")
    assert page.selected_id is None and not page.open_button.isEnabled()
    page.reset_button.click()
    assert ids(page) == ["a", "d", "b", "c"] and page.sort.currentData() == "rating_desc"
    assert page.content.currentWidget() is page.run_list and page.selected_id is None


@pytest.mark.parametrize("edit", ("rating", "favorite"))
def test_details_edits_refresh_without_losing_library_criteria(app, window, catalog, edit):
    page = window.library_page
    window.refresh_library()
    page.search.setText("alpha")
    choose(page.favorites, True)
    choose(page.rating, "rated")
    choose(page.method, "lora")
    choose(page.tag, "adapter")
    choose(page.sort, "rating_desc")
    criteria = page.criteria()
    page.run_list.setCurrentIndex(page.model.index(0))
    page.open_button.click()
    wait_for(app, lambda: not window.details_page.busy)
    details = window.details_page
    if edit == "rating":
        details.rating.setCurrentIndex(details.rating.findData(0))
    else:
        details.favorite.setChecked(False)
    details.save_metadata_button.click()
    wait_for(app, lambda: not details.busy)
    details.back_button.click()
    assert page.criteria() == criteria
    if edit == "rating":
        assert ids(page) == ["d", "a"] and page.selected_id == "a"
        assert page.model.rows[1].rating == 0
    else:
        assert ids(page) == ["d"] and page.selected_id is None
        assert not page.open_button.isEnabled()


def test_refresh_retains_a_filter_whose_last_option_disappears(app, window, repository, catalog):
    page = window.library_page
    window.refresh_library()
    choose(page.method, "full ft")
    repository.save_run(replace(catalog[1], method=None))
    window.refresh_library()
    assert page.method.currentData() == "full ft" and page.content.currentWidget() is page.no_results
    page.reset_filters()
    assert len(page.model.rows) == 4


def test_library_browsing_never_loads_aggregates_or_probes_external_files(app, window, repository, storage_root, catalog, monkeypatch):
    from pathlib import Path

    Path(storage_root.location).rmdir()

    def forbidden(*args, **kwargs):
        raise AssertionError("Library must use saved summaries only")

    monkeypatch.setattr(repository, "list_runs", forbidden)
    monkeypatch.setattr(repository, "get_run", forbidden)
    monkeypatch.setattr(PathResolver, "resolve", forbidden)
    monkeypatch.setattr(HFTrainerImporter, "scan", forbidden)
    monkeypatch.setattr(HFTrainerImporter, "detect", forbidden)
    window.refresh_library()
    page = window.library_page
    assert len(page.model.rows) == 4
    text = page.model.index(0).data()
    assert "Observed epoch 4.0" in text and "Step 160" in text and "Status unknown" in text
    monkeypatch.setattr(repository, "list_run_summaries", forbidden)
    page.search.setText("model-one")
    choose(page.tag, "good")
    choose(page.sort, "name_asc")
    assert ids(page) == ["a"]


def test_empty_catalog_has_import_action_separate_from_no_matches(app, window):
    page = window.library_page
    assert page.content.currentWidget() is page.empty_state
    page.search.setText("anything")
    assert page.content.currentWidget() is page.empty_state
    page.first_import.click()
    assert window.workspace.currentWidget() is window.import_page


@pytest.mark.parametrize("size", ((1440, 900), (1040, 640)))
def test_long_rich_rows_and_filter_controls_fit_compact_sizes(app, window, repository, catalog, size):
    long_run = replace(catalog[0], display_name="A very long saved run name " * 30, base_model="/models/" + "long-path/" * 70,
                       tags=tuple(Tag(name=f"tag-{i}-" + "long" * 50) for i in range(8)))
    repository.save_run(long_run)
    window.resize(*size)
    window.refresh_library()
    page = window.library_page
    app.processEvents()
    for control in (page.search, page.favorites, page.rating, page.method, page.tag, page.sort, page.run_list):
        assert control.isVisible() and control.isEnabled()
        assert control.geometry().right() <= page.width()
    assert page.run_list.horizontalScrollBar().maximum() == 0
    assert page.run_list.sizeHintForRow(0) <= 150
    row = next(i for i, run in enumerate(page.model.rows) if run.id == "a")
    index = page.model.index(row)
    assert long_run.display_name in index.data(Qt.ItemDataRole.ToolTipRole)
    assert long_run.tags[-1].name in index.data(Qt.ItemDataRole.ToolTipRole)
    page.search.setFocus()
    QTest.keyClick(page.search, Qt.Key.Key_Tab)
    assert page.favorites.hasFocus()
