"""A repository-backed Library with deliberately limited milestone actions."""

from importlib.resources import files

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from runpup.domain.models import Run
from runpup.persistence.repository import CatalogRepository
from runpup.ui.import_page import ImportPage
from runpup.ui.details_page import DetailsPage
from runpup.ui.library_page import LibraryPage
from runpup.ui.preferences import ImportPreferences
from runpup.ui.settings_page import SettingsPage


def _label(text: str, name: str = "muted", *, wrap: bool = False) -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setObjectName(name)
    label.setWordWrap(wrap)
    return label


class MainWindow(QMainWindow):
    def __init__(self, repository: CatalogRepository, *, preferences=None):
        super().__init__()
        self.repository = repository
        self.preferences = preferences if preferences is not None else ImportPreferences()
        self._close_requested = False
        self.setWindowTitle("RunPup — Run Library")
        self.resize(1440, 860)
        self.setMinimumSize(1040, 640)
        branding = str(files("runpup").joinpath("assets/branding/taskpuppy-kreations.png"))
        self.setWindowIcon(QIcon(branding))

        shell = QWidget()
        shell.setObjectName("shell")
        layout = QHBoxLayout(shell)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)
        self.setCentralWidget(shell)
        navigation = QFrame()
        navigation.setObjectName("navigation")
        navigation.setFixedWidth(210)
        nav = QVBoxLayout(navigation)
        nav.setContentsMargins(16, 24, 16, 24)
        nav.setSpacing(8)
        brand_image = QLabel()
        brand_image.setPixmap(QPixmap(branding).scaled(
            92, 112, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))
        brand_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav.addWidget(brand_image)
        brand_name = _label("RunPup", "brand")
        brand_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav.addWidget(brand_name)
        tagline = _label("Your training run catalog", wrap=True)
        tagline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav.addWidget(tagline)
        nav.addSpacing(24)
        self.navigation_buttons = {}
        for title in ("Library", "Import Run", "Compare Runs", "Storage Roots", "Settings"):
            button = QPushButton(title)
            button.setObjectName("nav")
            button.setCheckable(title in {"Library", "Import Run", "Settings"})
            button.setAutoExclusive(title in {"Library", "Import Run", "Settings"})
            button.setChecked(title == "Library")
            if title == "Library":
                button.clicked.connect(self.show_library)
            elif title == "Import Run":
                button.clicked.connect(self.show_import)
            elif title == "Settings":
                button.clicked.connect(self.show_settings)
            else:
                button.setEnabled(False)
                button.setToolTip("Available in a later milestone")
            nav.addWidget(button)
            self.navigation_buttons[title] = button
        nav.addStretch()
        nav.addWidget(_label("●  Local catalog", "local"))
        nav.addWidget(_label("Your training files stay where they are.", wrap=True))
        layout.addWidget(navigation)

        self.workspace = QStackedWidget()
        layout.addWidget(self.workspace, 1)
        self.library_page = LibraryPage(repository)
        self.library_page.open_requested.connect(self._open_run)
        self.library_page.import_requested.connect(self.show_import)
        self.workspace.addWidget(self.library_page)
        self.import_page = ImportPage(repository, preferences=self.preferences)
        self.import_page.run_saved.connect(self._import_saved)
        self.import_page.idle.connect(self._import_idle)
        self.workspace.addWidget(self.import_page)
        self.settings_page = SettingsPage(self.preferences)
        self.workspace.addWidget(self.settings_page)
        self.details_page = DetailsPage(repository)
        self.details_page.back_requested.connect(self.show_library)
        self.details_page.metadata_saved.connect(self.refresh_library)
        self.details_page.idle.connect(self._import_idle)
        self.workspace.addWidget(self.details_page)
        self.refresh_library()

    def _open_run(self, run_id):
        self.details_page.load_run(run_id)
        self.workspace.setCurrentWidget(self.details_page)
        self.navigation_buttons["Library"].setChecked(True)
        self.setWindowTitle("RunPup — Run Details")

    def show_settings(self) -> None:
        self.settings_page.refresh()
        self.workspace.setCurrentWidget(self.settings_page)
        self.navigation_buttons["Settings"].setChecked(True)
        self.setWindowTitle("RunPup — Settings")

    def show_import(self) -> None:
        self.workspace.setCurrentWidget(self.import_page)
        self.navigation_buttons["Import Run"].setChecked(True)
        self.setWindowTitle("RunPup — Import Run")

    def show_library(self) -> None:
        self.refresh_library()
        self.workspace.setCurrentWidget(self.library_page)
        self.navigation_buttons["Library"].setChecked(True)
        self.setWindowTitle("RunPup — Run Library")

    def _import_saved(self, run: Run) -> None:
        self.show_library()
        self.library_page.mark_saved(run)

    def closeEvent(self, event) -> None:
        if self.import_page.busy or self.details_page.busy:
            self._close_requested = True
            self.import_page.finish_before_close()
            self.details_page.finish_before_close()
            event.ignore()
        else:
            super().closeEvent(event)

    def _import_idle(self) -> None:
        if self._close_requested:
            self.close()

    def refresh_library(self) -> None:
        self.library_page.refresh()
