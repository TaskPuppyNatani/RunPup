"""Saved Run Details, backed exclusively by canonical catalog data."""

import sqlite3

from PySide6.QtCore import QEvent, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QComboBox, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QPushButton, QScrollArea, QSizePolicy,
    QTableWidget, QTableWidgetItem, QTextEdit, QToolButton, QVBoxLayout, QWidget,
)

from runpup.ui.diagnostics import format_diagnostics
from runpup.ui.evidence import format_evidence
from runpup.ui.review_field import display, label
from runpup.ui.run_details import dataset_fields, evidence_groups, load_run_details, ordered_metrics, parameter_sections


def _text(value, name="", *, selectable=True):
    widget = label(str(value), name, wrap=True)
    widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    if selectable:
        widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse |
                                       Qt.TextInteractionFlag.TextSelectableByKeyboard)
    return widget


def _card(title):
    card = QFrame()
    card.setObjectName("reviewCard")
    card.setMinimumWidth(0)
    box = QVBoxLayout(card)
    box.setContentsMargins(16, 14, 16, 14)
    box.setSpacing(8)
    box.addWidget(_text(title, "sectionHeading", selectable=False))
    return card, box


def _fields(box, fields):
    grid = QGridLayout()
    grid.setColumnStretch(1, 1)
    for row, (caption, value) in enumerate(fields):
        heading = _text(caption, "muted", selectable=False)
        heading.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        heading.setFixedWidth(160)
        grid.addWidget(heading, row, 0, Qt.AlignmentFlag.AlignTop)
        grid.addWidget(_text(display(value)), row, 1)
    box.addLayout(grid)


def _fold(box, title, body):
    button = QToolButton()
    button.setObjectName("detailDisclosure")
    button.setText(title)
    button.setToolTip(title)
    button.setCheckable(True)
    button.setArrowType(Qt.ArrowType.RightArrow)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    button.toggled.connect(body.setVisible)
    button.toggled.connect(lambda checked: button.setArrowType(
        Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow))
    box.addWidget(button)
    body.hide()
    box.addWidget(body)
    return button


def _table(headers, rows):
    table = QTableWidget(len(rows), len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setAlternatingRowColors(True)
    table.setShowGrid(False)
    table.verticalHeader().hide()
    table.verticalHeader().setDefaultSectionSize(36)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.setMinimumWidth(0)
    table.setMinimumHeight(120)
    table.setMaximumHeight(300)
    table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
    for row, values in enumerate(rows):
        for column, value in enumerate(values):
            item = QTableWidgetItem(display(value))
            item.setToolTip(display(value))
            table.setItem(row, column, item)
    return table


def _reference_text(reference):
    root = reference.root
    resolution = reference.resolution
    return (f"Storage root: {root.name if root else 'Unregistered'}\n"
            f"Root reference: {root.location if root else 'Unknown'}\n"
            f"Relative path: {reference.relative_path}\n"
            f"Availability: {resolution.state.value.capitalize()}\n"
            f"Resolved path: {resolution.path if resolution.path is not None else 'Unverified'}" +
            (f"\n{resolution.reason}" if resolution.reason else ""))


class _DetailsTask(QThread):
    def __init__(self, repository, run_id, resolver, parent):
        super().__init__(parent)
        self.repository, self.run_id, self.resolver = repository, run_id, resolver
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = load_run_details(self.repository, self.run_id, resolver=self.resolver)
        except (KeyError, ValueError, OSError, sqlite3.Error) as error:
            self.error = str(error)


class DetailsPage(QWidget):
    back_requested = Signal()
    metadata_saved = Signal()
    idle = Signal()

    def __init__(self, repository, *, resolver=None):
        super().__init__()
        self.repository, self.resolver = repository, resolver
        self.details = None
        self.task = None
        self.requested_id = None
        self.request_number = 0
        self.metadata_save_pending = False
        self.closing = False
        layout = QVBoxLayout(self)
        heading = QHBoxLayout()
        self.back_button = QPushButton("← Library")
        self.back_button.clicked.connect(self.back_requested)
        heading.addWidget(self.back_button)
        heading.addWidget(label("Run Details", "heading"))
        heading.addStretch()
        layout.addLayout(heading)
        self.status = _text("", "muted")
        layout.addWidget(self.status)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.viewport().installEventFilter(self)
        layout.addWidget(self.scroll, 1)
        self.metadata_cards = []

    @property
    def busy(self):
        return self.task is not None

    def load_run(self, run_id):
        self.requested_id = run_id
        self.request_number += 1
        self.metadata_save_pending = False
        self.details = None
        self.status.setText("Loading saved run…")
        self.scroll.hide()
        if not self.busy:
            self._start_load()

    def _start_load(self):
        self.task = _DetailsTask(self.repository, self.requested_id, self.resolver, self)
        self.task.request_number = self.request_number
        self.task.finished.connect(self._loaded)
        self.task.start()

    def _loaded(self):
        task = self.task
        self.task = None
        task.deleteLater()
        if not self.closing:
            if task.request_number != self.request_number:
                self._start_load()
                return
            if task.error:
                self.status.setText(f"Cannot open saved run. {task.error}")
            else:
                self.details = task.result
                self._render()
                self.status.setText("Saved catalog data · external availability checked on open")
                self.scroll.show()
        self.idle.emit()

    def finish_before_close(self):
        self.closing = True

    def _render(self):
        run = self.details.run
        previous = self.scroll.takeWidget()
        if previous:
            previous.deleteLater()
        content = QWidget()
        content.setMinimumWidth(0)
        box = QVBoxLayout(content)
        box.setContentsMargins(0, 0, 8, 8)
        box.setSpacing(14)
        self.name_label = _text(run.display_name, "emptyHeading")
        box.addWidget(self.name_label)
        box.addWidget(_text(f"{run.source_trainer or 'Unknown trainer'} · {run.method or 'Unknown method'}\n"
                           f"Imported {run.imported_at.isoformat()}", "muted"))
        card, identity = _card("Identity")
        _fields(identity, (("Run ID", run.id), ("Training status", run.training_status.value.capitalize()),
                          ("Training started", run.training_started_at.isoformat() if run.training_started_at else None),
                          ("Training finished", run.training_finished_at.isoformat() if run.training_finished_at else None),
                          ("Favorite", "Yes" if run.favorite else "No"),
                          ("Rating", f"{run.rating}/5" if run.rating is not None else "Not rated"),
                          ("Tags", ", ".join(t.name for t in run.tags) or "No tags.")))
        source = _text(_reference_text(self.details.source), "muted")
        _fold(identity, "Source reference", source)
        box.addWidget(card)

        grid_widget = QWidget()
        self.metadata_grid = QGridLayout(grid_widget)
        self.metadata_grid.setContentsMargins(0, 0, 0, 0)
        self.metadata_grid.setSpacing(14)
        self.metadata_cards = []
        card, fields = _card("Model")
        _fields(fields, (("Base model", run.base_model), ("Revision", run.model_revision), ("Method", run.method)))
        self.metadata_cards.append(card)
        card, fields = _card("Dataset")
        if not run.datasets:
            _fields(fields, dataset_fields(run))
        for dataset, reference in zip(run.datasets, self.details.datasets):
            _fields(fields, dataset_fields(run, dataset))
            if reference:
                _fold(fields, "Dataset file reference", _text(_reference_text(reference), "muted"))
        self.metadata_cards.append(card)
        for title, values in parameter_sections(run):
            card, fields = _card(title)
            _fields(fields, values)
            self.metadata_cards.append(card)
        box.addWidget(grid_widget)
        self._arrange_cards()

        card, results = _card("Results / Metrics")
        metrics = ordered_metrics(run)
        self.metrics_table = None
        if metrics:
            self.metrics_table = _table(("Metric", "Value", "Context", "Step", "Unit", "Summary"), [
                (m.name, f"{m.value:.8g}", m.context, m.step, m.unit, m.summary.value) for m in metrics])
            header = self.metrics_table.horizontalHeader()
            for column, width in enumerate((120, 95, 55, 75, 135), start=1):
                header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
                self.metrics_table.setColumnWidth(column, width)
            for row, metric in enumerate(metrics):
                self.metrics_table.item(row, 1).setToolTip(str(metric.value))
            self.metrics_table.setAccessibleName("Saved metrics with context, step, unit and summary")
            results.addWidget(self.metrics_table)
        else:
            results.addWidget(_text("No metrics were recorded for this run.", "muted"))
        box.addWidget(card)

        card, artifacts = _card(f"Artifacts · {len(run.artifacts)} external references")
        self.artifacts_table = None
        if run.artifacts:
            self.artifacts_table = _table(("Relative path", "Kind", "Availability"), [
                (a.relative_path, a.category, r.resolution.state.value.capitalize())
                for a, r in zip(run.artifacts, self.details.artifacts)])
            self.artifacts_table.setAccessibleName("Saved artifact references")
            for column, width in ((1, 105), (2, 115)):
                self.artifacts_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
                self.artifacts_table.setColumnWidth(column, width)
            artifacts.addWidget(self.artifacts_table)
            self.artifact_info = _text("")
            artifacts.addWidget(self.artifact_info)
            self.artifacts_table.itemSelectionChanged.connect(self._select_artifact)
            self.artifacts_table.selectRow(0)
        else:
            artifacts.addWidget(_text("No artifacts recorded.", "muted"))
        box.addWidget(card)

        card, evidence = _card("Provenance / Evidence")
        evidence_body = QWidget()
        evidence_box = QVBoxLayout(evidence_body)
        evidence_box.setContentsMargins(0, 0, 0, 0)
        self.evidence_buttons = []
        self.evidence_texts = []
        for title, records in evidence_groups(run):
            text = _text(format_evidence(records))
            self.evidence_texts.append(text)
            self.evidence_buttons.append(_fold(evidence_box, f"{title} · {len(records)} records", text))
        if not run.provenance:
            evidence_box.addWidget(_text("No persisted provenance.", "muted"))
        self.provenance_button = _fold(evidence, f"Inspect {len(run.provenance)} saved evidence records", evidence_body)
        box.addWidget(card)

        card, diagnostics = _card("Import Diagnostics")
        self.copy_button = QPushButton("Copy Diagnostics")
        self.copy_button.setObjectName("compact")
        self.copy_button.setEnabled(bool(run.diagnostics))
        self.copy_button.clicked.connect(self.copy_diagnostics)
        diagnostics.addWidget(self.copy_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.diagnostics_text = _text(format_diagnostics(
            run.diagnostics, artifact_paths={a.id: a.relative_path for a in run.artifacts},
            root=self.details.source.root) if run.diagnostics else "No persisted import diagnostics.")
        diagnostics.addWidget(self.diagnostics_text)
        box.addWidget(card)

        card, human = _card("Notes / Human metadata")
        human.addWidget(_text("Notes, favorite and rating can be saved here. Tags are shown above.", "muted"))
        self.notes_editor = QTextEdit()
        self.notes_editor.setAcceptRichText(False)
        self.notes_editor.setAccessibleName("Run notes")
        self.notes_editor.setPlaceholderText("No notes.")
        self.notes_editor.setPlainText(run.notes)
        self.notes_editor.setFixedHeight(120)
        human.addWidget(self.notes_editor)
        controls = QHBoxLayout()
        self.favorite = QCheckBox("Favorite")
        self.favorite.setChecked(run.favorite)
        controls.addWidget(self.favorite)
        self.rating = QComboBox()
        self.rating.setAccessibleName("Run rating")
        self.rating.addItem("Not rated", None)
        for value in range(6):
            self.rating.addItem(f"{value}/5", value)
        self.rating.setCurrentIndex(self.rating.findData(run.rating))
        controls.addWidget(self.rating)
        controls.addStretch()
        self.save_metadata_button = QPushButton("Save metadata")
        self.save_metadata_button.clicked.connect(self._save_metadata)
        controls.addWidget(self.save_metadata_button)
        human.addLayout(controls)
        self.metadata_notice = _text("Metadata saved." if self.metadata_save_pending else "", "local")
        self.metadata_save_pending = False
        human.addWidget(self.metadata_notice)
        box.addWidget(card)
        box.addStretch()
        self.scroll.setWidget(content)
        self.scroll.verticalScrollBar().setValue(0)

    def _arrange_cards(self):
        columns = 2 if self.scroll.viewport().width() >= 1000 else 1
        for index, card in enumerate(self.metadata_cards):
            self.metadata_grid.addWidget(card, index // columns, index % columns, Qt.AlignmentFlag.AlignTop)
        self.metadata_grid.setColumnStretch(0, 1)
        self.metadata_grid.setColumnStretch(1, 1 if columns == 2 else 0)

    def eventFilter(self, watched, event):
        if watched is self.scroll.viewport() and event.type() == QEvent.Type.Resize and self.metadata_cards:
            self._arrange_cards()
        return super().eventFilter(watched, event)

    def _select_artifact(self):
        if self.sender() is not self.artifacts_table:
            return
        row = self.artifacts_table.currentRow()
        if row < 0:
            return
        artifact = self.details.run.artifacts[row]
        reference = self.details.artifacts[row]
        self.artifact_info.setText(
            f"{artifact.display_name}\n{_reference_text(reference)}\n"
            f"Stored size: {display(artifact.size_bytes)} bytes\n"
            f"Stored hash: {display(artifact.hash_algorithm)} · {display(artifact.hash_value)}\n"
            f"Recorded verification: {artifact.verification_status.value}\n"
            f"Verified at: {artifact.verified_at.isoformat() if artifact.verified_at else 'Unknown'}")

    def copy_diagnostics(self):
        if self.details and self.details.run.diagnostics:
            QApplication.clipboard().setText("RunPup Import Diagnostics\n\n" + self.diagnostics_text.text())

    def _save_metadata(self):
        try:
            self.repository.update_human_metadata(
                self.details.run.id, notes=self.notes_editor.toPlainText(),
                rating=self.rating.currentData(), favorite=self.favorite.isChecked())
        except (KeyError, ValueError, sqlite3.Error) as error:
            self.metadata_notice.setText(f"Could not save metadata: {error}")
            return
        run_id = self.details.run.id
        self.metadata_saved.emit()
        self.load_run(run_id)
        self.metadata_save_pending = True
