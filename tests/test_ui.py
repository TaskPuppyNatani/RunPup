import pytest

from PySide6.QtWidgets import QApplication, QMessageBox

from runpup.domain.models import Dataset, Metric, MetricSummary, Run
from runpup.ui.theme import apply_theme
from runpup.ui.window import MainWindow


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    apply_theme(application)
    yield application


def test_empty_library_shows_honest_import_placeholder(app, repository, monkeypatch):
    notices = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: notices.append(args[2]))
    window = MainWindow(repository)
    try:
        window.show()
        app.processEvents()
        assert window.content.currentWidget() is window.empty_state
        assert "0 runs" in window.run_count.text()
        window.first_import.click()
        assert "later milestone" in notices[0]
        assert repository.list_runs() == []
    finally:
        window.close()


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
