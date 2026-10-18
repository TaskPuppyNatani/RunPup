"""A repository-backed Library with deliberately limited milestone actions."""

from importlib.resources import files
import sqlite3

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QFrame, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QMessageBox, QPushButton, QSplitter, QStackedWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from runpup.domain.models import MetricSummary, Run
from runpup.persistence.repository import CatalogRepository
from runpup.ui.import_page import ImportPage
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
        self.runs: list[Run] = []
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
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.library_page = splitter
        self.workspace.addWidget(splitter)
        splitter.setChildrenCollapsible(False)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(8, 8, 8, 0)
        center_layout.setSpacing(18)
        heading_row = QHBoxLayout()
        headings = QVBoxLayout()
        headings.addWidget(_label("Run Library", "heading"))
        self.run_count = _label("0 runs · local training history")
        headings.addWidget(self.run_count)
        heading_row.addLayout(headings)
        heading_row.addStretch()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_library)
        heading_row.addWidget(refresh)
        import_button = QPushButton("Import Run")
        import_button.setObjectName("primary")
        import_button.clicked.connect(self.show_import)
        heading_row.addWidget(import_button)
        center_layout.addLayout(heading_row)
        self.save_notice = _label("", "local", wrap=True)
        self.save_notice.hide()
        center_layout.addWidget(self.save_notice)

        self.content = QStackedWidget()
        self.empty_state = QWidget()
        self.empty_state.setObjectName("emptyState")
        empty = QVBoxLayout(self.empty_state)
        empty.setSpacing(14)
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_heading = _label("No training runs yet.", "emptyHeading")
        empty_heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty.addWidget(empty_heading)
        explanation = _label("Keep the settings and results worth coming back to.", wrap=True)
        explanation.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty.addWidget(explanation)
        self.first_import = QPushButton("Import Your First Run")
        self.first_import.setObjectName("primary")
        self.first_import.clicked.connect(self.show_import)
        empty.addWidget(self.first_import, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.content.addWidget(self.empty_state)

        self.table = QTableWidget(0, 8)
        self.table.setObjectName("runTable")
        self.table.setHorizontalHeaderLabels([
            "Run name", "Base model", "Method", "Dataset", "Final train loss", "Rating", "Status", "Imported",
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(44)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.content.addWidget(self.table)
        center_layout.addWidget(self.content, 1)
        splitter.addWidget(center)

        inspector = QFrame()
        inspector.setObjectName("inspector")
        inspector.setMinimumWidth(230)
        inspector.setMaximumWidth(320)
        right = QVBoxLayout(inspector)
        right.setContentsMargins(22, 24, 22, 24)
        right.setSpacing(18)
        right.addWidget(_label("Run inspector", "inspectorHeading"))
        self.inspector_name = _label("Select a run", wrap=True)
        right.addWidget(self.inspector_name)
        right.addWidget(_label("Detailed settings and artifacts will appear here in a later milestone.", wrap=True))
        right.addStretch()
        right.addWidget(_label("Settings. Results. Sources.", wrap=True))
        splitter.addWidget(inspector)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 250])
        self.import_page = ImportPage(repository, preferences=self.preferences)
        self.import_page.run_saved.connect(self._import_saved)
        self.import_page.idle.connect(self._import_idle)
        self.workspace.addWidget(self.import_page)
        self.settings_page = SettingsPage(self.preferences)
        self.workspace.addWidget(self.settings_page)
        self.refresh_library()

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
        self.save_notice.setText(f"Saved ‘{run.display_name}’ to your library.")
        self.save_notice.show()
        for row, stored in enumerate(self.runs):
            if stored.id == run.id:
                self.table.selectRow(row)
                self.table.scrollToItem(self.table.item(row, 0))
                break

    def closeEvent(self, event) -> None:
        if self.import_page.busy:
            self._close_requested = True
            self.import_page.finish_before_close()
            event.ignore()
        else:
            super().closeEvent(event)

    def _import_idle(self) -> None:
        if self._close_requested:
            self.close()

    def refresh_library(self) -> None:
        try:
            runs = self.repository.list_runs()
        except (sqlite3.Error, ValueError) as error:
            QMessageBox.critical(self, "Cannot read catalog", str(error))
            return
        self.runs = runs
        self.table.clearSelection()
        self.inspector_name.setText("Select a run")
        count = len(runs)
        self.run_count.setText(f"{count} {'run' if count == 1 else 'runs'} · local training history")
        self.table.setRowCount(count)
        for row, run in enumerate(runs):
            loss = [metric for metric in run.metrics if metric.name == "loss"
                    and metric.context == "train" and metric.summary is MetricSummary.FINAL]
            values = (
                run.display_name, run.base_model or "—", run.method or "—",
                ", ".join(dataset.name for dataset in run.datasets) or "—",
                f"{loss[0].value:.5g}" if len(loss) == 1 else "—",
                f"{run.rating}/5" if run.rating is not None else "—",
                run.training_status.value.capitalize(), run.imported_at.astimezone().strftime("%Y-%m-%d"),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.table.setItem(row, column, item)
            self.table.item(row, 0).setData(Qt.ItemDataRole.UserRole, run.id)
            self.table.item(row, 7).setToolTip(run.imported_at.isoformat())
        self.content.setCurrentWidget(self.table if runs else self.empty_state)

    def _selection_changed(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self.runs) and self.table.selectedItems():
            self.inspector_name.setText(self.runs[row].display_name)
