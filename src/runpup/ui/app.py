"""Application composition and developer entry point."""

import argparse
from pathlib import Path
import sqlite3
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from runpup.persistence.database import SchemaVersionError
from runpup.persistence.repository import CatalogRepository
from runpup.services.data_location import open_startup_catalog
from runpup.ui.preferences import ImportPreferences, default_data_directory
from runpup.ui.theme import apply_theme
from runpup.ui.window import MainWindow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RunPup local training-run catalog")
    parser.add_argument("--data-dir", type=Path,
                        help="Data directory for this launch; overrides saved Settings without changing them")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName("RunPup")
    app.setOrganizationName("TaskPuppy")
    apply_theme(app)
    preferences = ImportPreferences()
    configured = preferences.configured_data_directory
    default_directory = None
    attempted = args.data_dir if args.data_dir is not None else configured or "platform default"

    def recover(error):
        nonlocal attempted
        dialog = QMessageBox(QMessageBox.Icon.Warning, "Cannot open configured RunPup catalog",
                             f"Configured location: {configured}\n\n{error}\n\n"
                             "The saved location was not changed. Use the default for this launch, "
                             "or exit and launch with --data-dir pointing to another catalog.")
        dialog.setTextFormat(Qt.TextFormat.PlainText)
        use_default = dialog.addButton("Use Default for This Launch", QMessageBox.ButtonRole.AcceptRole)
        dialog.addButton("Exit", QMessageBox.ButtonRole.RejectRole)
        dialog.exec()
        accepted = dialog.clickedButton() is use_default
        if accepted:
            attempted = default_directory
        return accepted

    try:
        default_directory = default_data_directory()
        if args.data_dir is None and (configured is None or configured == ""):
            attempted = default_directory
        location, database = open_startup_catalog(default_directory, configured,
                                                 override=args.data_dir, recover=recover)
    except (ValueError, OSError, RuntimeError, sqlite3.Error, SchemaVersionError) as error:
        QMessageBox.critical(None, "Cannot open RunPup catalog",
                             f"Location: {attempted}\n\n"
                             f"{error}\n\nNo existing catalog was replaced. Check access or launch with "
                             "--data-dir pointing to an empty or valid RunPup data directory. "
                             "The saved location was not changed.")
        return 1
    window = MainWindow(CatalogRepository(database), preferences=preferences, data_location=location)
    window.show()
    return app.exec()
