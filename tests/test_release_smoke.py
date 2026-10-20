"""One release workflow across real widgets/services and a fresh reader process."""

import json
from pathlib import Path
import subprocess
import sys
import zipfile

from PySide6.QtCore import QSettings

from runpup.persistence.database import Database
from runpup.persistence.repository import CatalogRepository
from runpup.ui.preferences import ImportPreferences
from runpup.ui.window import MainWindow
from test_details_ui import app, wait_for


def test_release_workflow_survives_restart_without_touching_sources(app, repository, tmp_path):
    preferences = ImportPreferences(QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat), home=tmp_path)
    directories = []
    before = {}
    for name, loss in (("Run A", .8), ("Run B", .5)):
        directory = tmp_path / name
        directory.mkdir()
        documents = {
            "trainer_state.json": {"global_step": 10, "max_steps": 10, "log_history": [{"step": 10, "loss": loss}]},
            "training_args.json": {"run_name": name, "model_name_or_path": "org/base", "learning_rate": .0002,
                                   "num_train_epochs": 2, "per_device_train_batch_size": 1},
            "adapter_config.json": {"peft_type": "LORA", "r": 16, "base_model_name_or_path": "org/base"},
        }
        for filename, data in documents.items():
            (directory / filename).write_text(json.dumps(data))
        (directory / "adapter_model.safetensors").write_bytes(b"external model sentinel")
        (directory / "training_args.bin").write_bytes(b"opaque; never deserialize")
        directories.append(directory)
        before.update({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()})
    window = MainWindow(repository, preferences=preferences)
    try:
        window.show()
        assert not repository.list_run_summaries()
        identities = []
        for directory in directories:
            window.show_import()
            page = window.import_page
            page.select_folder(directory)
            page.scan_button.click()
            wait_for(app, lambda: not page.busy)
            assert page.proposal and not page.proposal.has_blockers
            assert "not deserialized" in page.diagnostics_text.text()
            identities.append(page.proposal.id)
            page.save_button.click()
            wait_for(app, lambda: not page.busy)
            assert window.workspace.currentWidget() is window.library_page
        first = repository.get_run(identities[0])
        assert first.provenance and first.diagnostics and first.metrics and first.artifacts
        window._open_run(first.id)
        wait_for(app, lambda: not window.details_page.busy)
        details = window.details_page
        details.notes_editor.setPlainText("Release note retained")
        details.rating.setCurrentIndex(details.rating.findData(4))
        details.favorite.setChecked(True)
        details.save_metadata_button.click()
        wait_for(app, lambda: not details.busy)
        window.show_library()
        library = window.library_page
        library.search.setText("run a")
        library.favorites.setCurrentIndex(library.favorites.findData(True))
        assert [row.id for row in library.model.rows] == [first.id]
        window.show_compare()
        comparison = window.compare_page
        comparison.run_a.setCurrentIndex(comparison.run_a.findData(identities[0]))
        comparison.run_b.setCurrentIndex(comparison.run_b.findData(identities[1]))
        wait_for(app, lambda: not comparison.busy)
        assert comparison.comparison.run_a.id == first.id
        assert comparison.comparison.run_b.id == identities[1]
        assert any(row.status.value == "BOTH UNKNOWN" for row in comparison.comparison.rows)
        window.show_storage_roots()
        roots = window.storage_roots_page
        wait_for(app, lambda: not roots.busy)
        for i in range(roots.roots.count()):
            roots.roots.setCurrentRow(i)
            if roots.selected.root.id == first.source_storage_root_id:
                break
        destination = tmp_path / "remapped"
        destination.mkdir()
        sentinel = destination / "adapter_model.safetensors"
        sentinel.write_bytes(b"destination sentinel")
        before[sentinel] = sentinel.read_bytes(), sentinel.stat().st_mtime_ns
        roots.remap_button.click()
        roots.dialog.location.setText(str(destination))
        roots.dialog.save.click()  # Preview, then confirm.
        roots.dialog.save.click()
        wait_for(app, lambda: not roots.busy)
        roots.verify_button.click()
        wait_for(app, lambda: not roots.busy)
        assert "Online" in roots.info.text() and "Missing:" in roots.verification_text.text()
        assert repository.get_run(first.id).artifacts == first.artifacts
        roots.remove_button.click()
        assert not roots.dialog.remove.isEnabled()
        roots.dialog.reject()
        window.show_settings()
        settings = window.settings_page
        settings._save_import(lambda: setattr(preferences, "default_directory", str(directories[0].parent)))
        backup = tmp_path / "release.zip"
        settings.start_backup(backup)
        wait_for(app, lambda: not settings.busy)
        assert "successfully" in settings.backup_status.text()
        with zipfile.ZipFile(backup) as archive:
            assert set(archive.namelist()) == {"runpup-backup.json", "catalog.sqlite3", "preferences.json"}
            snapshot = tmp_path / "snapshot.sqlite3"
            snapshot.write_bytes(archive.read("catalog.sqlite3"))
        assert Database(snapshot).validate_existing() == 3
        assert CatalogRepository(Database(snapshot)).get_run(first.id).notes == "Release note retained"
    finally:
        window.close()
        wait_for(app, lambda: not any(p.busy for p in (
            window.import_page, window.details_page, window.compare_page, window.storage_roots_page, window.settings_page)))
        window.close()
    reader = """
import json, sys
from runpup.persistence.database import Database
from runpup.persistence.repository import CatalogRepository
database = Database(sys.argv[1]); database.initialize()
repository = CatalogRepository(database)
print(json.dumps([(r.id, r.notes, r.rating, r.favorite, len(r.provenance), len(r.diagnostics)) for r in repository.list_runs()]))
"""
    result = subprocess.run([sys.executable, "-c", reader, str(repository.database.path)],
                            check=True, capture_output=True, text=True)
    rows = {row[0]: row[1:] for row in json.loads(result.stdout)}
    assert set(rows) == set(identities)
    assert rows[first.id][:3] == ["Release note retained", 4, True]
    assert rows[first.id][3] and rows[first.id][4]
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == state for path, state in before.items())
