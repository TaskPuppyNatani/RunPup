"""Application composition and developer entry point."""

import argparse
from pathlib import Path
import sqlite3
import sys

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QApplication, QMessageBox

from runpup.persistence.database import Database, SchemaVersionError
from runpup.persistence.repository import CatalogRepository
from runpup.ui.theme import apply_theme
from runpup.ui.window import MainWindow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RunPup local training-run catalog")
    parser.add_argument("--data-dir", type=Path, help="Directory for the local catalog database")
    args = parser.parse_args(argv)
    app = QApplication([sys.argv[0]])
    app.setApplicationName("RunPup")
    app.setOrganizationName("TaskPuppy")
    apply_theme(app)
    data_dir = args.data_dir or Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppLocalDataLocation
    ))
    database = Database(data_dir / "runpup.sqlite3")
    try:
        database.initialize()
    except (OSError, sqlite3.Error, SchemaVersionError) as error:
        QMessageBox.critical(None, "Cannot open RunPup catalog", str(error))
        return 1
    window = MainWindow(CatalogRepository(database))
    window.show()
    return app.exec()
