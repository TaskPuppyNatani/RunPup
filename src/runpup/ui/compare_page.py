"""Two selected saved aggregates; difference-first, offline comparison UI."""

from datetime import datetime
from html import escape
import sqlite3

from PySide6.QtCore import QEvent, QSize, Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QHeaderView,
    QLineEdit, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from runpup.ui.comparison import ArtifactMetadata, ComparisonStatus, SECTIONS, load_comparison
from runpup.ui.details_page import _card, _fold, _text
from runpup.ui.evidence import format_evidence
from runpup.ui.review_field import display, label


def format_value(value):
    if isinstance(value, ArtifactMetadata):
        return (f"{value.kind} · {value.name}\nStored size: {display(value.size_bytes)} bytes\n"
                f"Stored hash: {display(value.hash_algorithm)} · {display(value.hash_value)}\n"
                f"Recorded verification: {value.verification}\nVerified at: {display(value.verified_at)}")
    if isinstance(value, frozenset):
        return ", ".join(sorted(value)) or "No tags"
    if isinstance(value, datetime):
        return value.isoformat()
    return display(value) if value != "" else "(Empty)"


def _tooltip(text):
    return "<qt>" + escape(text).replace("\n", "<br>") + "</qt>"


class _CompareTask(QThread):
    def __init__(self, repository, identities, request, parent):
        super().__init__(parent)
        self.repository, self.identities, self.request = repository, identities, request
        self.result = self.error = None

    def run(self):
        try:
            self.result = load_comparison(self.repository, *self.identities)
        except (KeyError, ValueError, sqlite3.Error, OSError) as error:
            self.error = str(error)


class ComparePage(QWidget):
    open_requested = Signal(str)
    import_requested = Signal()
    idle = Signal()

    def __init__(self, repository):
        super().__init__()
        self.repository = repository
        self.comparison = None
        self.task = None
        self.request = 0
        self.requested = None
        self.closing = False
        self.panels = []
        layout = QVBoxLayout(self)
        title_row = QHBoxLayout()
        title_row.addWidget(label("Compare Runs", "heading"))
        title_row.addStretch()
        import_button = QPushButton("Import Run")
        import_button.clicked.connect(self.import_requested)
        title_row.addWidget(import_button)
        layout.addLayout(title_row)
        controls = QGridLayout()
        self.run_a, self.run_b = QComboBox(), QComboBox()
        for column, (caption, combo) in enumerate((("Run A", self.run_a), ("Run B", self.run_b))):
            combo.setMinimumWidth(0)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setAccessibleName(caption)
            combo.currentIndexChanged.connect(self._selection_changed)
            controls.addWidget(label(caption, "fieldTitle"), 0, column)
            controls.addWidget(combo, 1, column)
            controls.setColumnStretch(column, 1)
        self.swap_button = QPushButton("⇄ Swap A/B")
        self.swap_button.clicked.connect(self.swap)
        controls.addWidget(self.swap_button, 1, 2)
        self.compare_button = QPushButton("Compare / Refresh")
        self.compare_button.clicked.connect(self._selection_changed)
        controls.addWidget(self.compare_button, 1, 3)
        layout.addLayout(controls)
        self.status = _text("Choose two saved runs.", "muted")
        layout.addWidget(self.status)

        self.headers = QWidget()
        header_grid = QHBoxLayout(self.headers)
        header_grid.setContentsMargins(0, 0, 0, 0)
        self.header_names, self.header_info, self.open_buttons = [], [], []
        for side in ("A", "B"):
            card, box = _card(f"RUN {side}")
            name = QLineEdit()
            name.setReadOnly(True)
            name.setAccessibleName(f"Run {side} name")
            name.setMinimumWidth(0)
            box.addWidget(name)
            info = _text("", "muted")
            box.addWidget(info)
            button = QPushButton(f"Open Run {side} Details")
            button.setObjectName("compact")
            button.clicked.connect(lambda _checked=False, side=side: self._open_details(side))
            box.addWidget(button)
            header_grid.addWidget(card, 1)
            self.header_names.append(name)
            self.header_info.append(info)
            self.open_buttons.append(button)
        self.headers.hide()
        layout.addWidget(self.headers)
        filter_row = QHBoxLayout()
        self.differences_only = QCheckBox("Show differences only")
        self.differences_only.setChecked(True)
        self.differences_only.setToolTip("Hide SAME and BOTH UNKNOWN. Full view retains every comparison row.")
        self.differences_only.toggled.connect(self._render_rows)
        filter_row.addWidget(self.differences_only)
        filter_row.addWidget(label("Expand a row for full values and evidence.", "muted"))
        filter_row.addStretch()
        layout.addLayout(filter_row)
        self.summary = _text("", "muted")
        layout.addWidget(self.summary)
        self.tree = QTreeWidget()
        self.tree.setObjectName("comparisonTree")
        self.tree.setAccessibleName("Saved run comparison")
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(("Field / identity", "Run A", "Run B", "Status"))
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tree.setWordWrap(False)
        self.tree.setIndentation(14)
        header = self.tree.header()
        header.setStretchLastSection(False)
        for column in range(3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.tree.setColumnWidth(3, 125)
        self.tree.itemExpanded.connect(self._expand_row)
        self.tree.viewport().installEventFilter(self)
        layout.addWidget(self.tree, 1)
        self.scope_note = _text("Artifacts compare stored root/path and metadata; file availability and contents are unchecked.", "muted")
        layout.addWidget(self.scope_note)

    @property
    def busy(self):
        return self.task is not None

    def refresh(self):
        previous = (self.run_a.currentData(), self.run_b.currentData())
        try:
            summaries = self.repository.list_run_summaries()
        except (sqlite3.Error, ValueError) as error:
            self.request += 1
            self.requested = None
            self._clear(f"Cannot read saved runs: {error}")
            return
        for combo, identity in zip((self.run_a, self.run_b), previous):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Choose a saved run…", None)
            for run in summaries:
                caption = (f"{run.id[:8]} · {run.display_name} · {run.method or 'Unknown method'} · "
                           f"{run.imported_at.strftime('%Y-%m-%d %H:%M')}")
                combo.addItem(caption, run.id)
                combo.setItemData(combo.count() - 1, _tooltip(caption + "\nRun ID: " + run.id), Qt.ItemDataRole.ToolTipRole)
            if identity is not None and combo.findData(identity) < 0:
                combo.addItem(f"Unavailable run · {identity}", identity)
            combo.setCurrentIndex(max(0, combo.findData(identity)))
            combo.blockSignals(False)
        if len(summaries) < 2:
            self.request += 1
            self.requested = None
            self._clear("At least two saved runs are needed. Import a run or return to Library.")
            self.compare_button.setEnabled(False)
            self.swap_button.setEnabled(False)
        else:
            self._selection_changed()

    def _clear(self, message):
        self.comparison = None
        self.headers.hide()
        self.tree.clear()
        self.panels.clear()
        self.summary.clear()
        self.status.setText(message)

    def _selection_changed(self, *_):
        self.request += 1
        identities = (self.run_a.currentData(), self.run_b.currentData())
        for combo in (self.run_a, self.run_b):
            combo.setToolTip(_tooltip(combo.currentText()))
        self.requested = identities if all(identities) and identities[0] != identities[1] else None
        self.compare_button.setEnabled(self.requested is not None)
        self.swap_button.setEnabled(self.requested is not None)
        if self.requested is None:
            self._clear("Choose two different saved runs." if all(identities) else "Choose Run A and Run B from the saved catalog.")
        else:
            self._clear("Loading two saved runs…")
            if not self.busy:
                self._start()

    def _start(self):
        self.task = _CompareTask(self.repository, self.requested, self.request, self)
        self.task.finished.connect(self._loaded)
        self.task.start()

    def _loaded(self):
        task, self.task = self.task, None
        task.deleteLater()
        if not self.closing:
            if task.request != self.request:
                if self.requested is not None:
                    self._start()
                    return
            elif task.error:
                self._clear(f"Cannot compare saved runs. {task.error}")
            else:
                self.comparison = task.result
                self._render()
        self.idle.emit()

    def finish_before_close(self):
        self.closing = True

    def swap(self):
        if self.comparison is None:
            # A pending read is superseded through the normal request path.
            identities = self.run_b.currentData(), self.run_a.currentData()
        else:
            identities = self.comparison.run_b.id, self.comparison.run_a.id
        for combo, identity in zip((self.run_a, self.run_b), identities):
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(identity))
            combo.setToolTip(_tooltip(combo.currentText()))
            combo.blockSignals(False)
        if self.comparison is not None:
            self.comparison = self.comparison.swapped()
            self.requested = identities
            self._render()
        else:
            self._selection_changed()

    def _render(self):
        for index, run in enumerate((self.comparison.run_a, self.comparison.run_b)):
            self.header_names[index].setText(run.display_name)
            self.header_names[index].setCursorPosition(0)
            self.header_names[index].setToolTip(_tooltip(run.display_name + "\nRun ID: " + run.id))
            rating = f"{run.rating}/5" if run.rating is not None else "Unrated"
            self.header_info[index].setText(f"{run.method or 'Unknown method'} · {run.source_trainer or 'Unknown trainer'}\n"
                                           f"{rating} · {'Favorite' if run.favorite else 'Not favorite'} · Imported {run.imported_at.strftime('%Y-%m-%d')}\n"
                                           f"Run ID: {run.id}")
        self.headers.show()
        self.status.setText("Saved canonical values · Unknown on both sides does not establish equality.")
        self.summary.setText(" · ".join(f"{self.comparison.counts[status]} {status.value}" for status in ComparisonStatus))
        self._render_rows()

    def _render_rows(self, *_):
        self.tree.clear()
        self.panels.clear()
        if self.comparison is None:
            return
        rows = self.comparison.visible_rows(self.differences_only.isChecked())
        for section in SECTIONS:
            section_rows = [row for row in rows if row.section == section]
            if not section_rows:
                continue
            parent = QTreeWidgetItem((section, "", "", f"{len(section_rows)} {'row' if len(section_rows) == 1 else 'rows'}"))
            self.tree.addTopLevelItem(parent)
            for row in section_rows:
                item = QTreeWidgetItem((row.label, format_value(row.a.value).replace("\n", " · "),
                                        format_value(row.b.value).replace("\n", " · "), row.status.value))
                item.setData(0, Qt.ItemDataRole.UserRole, row)
                item.setSizeHint(0, QSize(0, 48 if "\n" in row.label else 34))
                for column in range(4):
                    item.setToolTip(column, _tooltip((row.label, format_value(row.a.value),
                                                     format_value(row.b.value), row.status.value)[column]))
                color = ("#d6b8f4" if row.status is ComparisonStatus.DIFFERENT else
                         "#a8dce7" if row.status in {ComparisonStatus.ONLY_A, ComparisonStatus.ONLY_B} else "#aeb7ca")
                item.setForeground(3, QColor(color))
                item.addChild(QTreeWidgetItem(("Full values / Evidence",)))
                parent.addChild(item)
            parent.setExpanded(True)
        if not rows:
            self.tree.addTopLevelItem(QTreeWidgetItem(("No value differences in this view. Uncheck differences only for the full comparison.",)))
        self.tree.scrollToTop()

    def _expand_row(self, item):
        row = item.data(0, Qt.ItemDataRole.UserRole)
        if row is None or self.tree.itemWidget(item.child(0), 0) is not None:
            return
        child = item.child(0)
        child.setText(0, "")
        child.setFirstColumnSpanned(True)
        panel = QWidget()
        grid = QGridLayout(panel)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.addWidget(_text(row.label, "sectionHeading"), 0, 0, 1, 2)
        for column, cell in enumerate((row.a, row.b)):
            side = QWidget()
            box = QVBoxLayout(side)
            box.addWidget(label(f"Run {'A' if column == 0 else 'B'}", "sectionHeading"))
            box.addWidget(_text(format_value(cell.value)))
            confidence = sorted({record.confidence.value.upper() for record in cell.evidence})
            cue = ("USER EDITED · " if any(r.user_modified for r in cell.evidence) else "") + ", ".join(confidence)
            if cue:
                box.addWidget(_text(cue, "sourcePath"))
            evidence = _text(format_evidence(cell.evidence))
            button = _fold(box, f"Evidence · {len(cell.evidence)} {'record' if len(cell.evidence) == 1 else 'records'}", evidence)
            button.toggled.connect(self._size_panels)
            box.addStretch()
            grid.addWidget(side, 1, column)
        self.tree.setItemWidget(child, 0, panel)
        self.panels.append((child, panel))
        self._size_panels()

    def _size_panels(self, *_):
        width = max(200, self.tree.viewport().width() - 50)
        for child, panel in self.panels:
            panel.layout().invalidate()
            height = panel.layout().heightForWidth(width)
            panel.setFixedHeight(max(100, height))
            child.setSizeHint(0, QSize(0, panel.height()))
        self.tree.doItemsLayout()

    def eventFilter(self, watched, event):
        if watched is self.tree.viewport() and event.type() == QEvent.Type.Resize:
            self._size_panels()
        return super().eventFilter(watched, event)

    def _open_details(self, side):
        if self.comparison is not None:
            self.open_requested.emit(self.comparison.run_a.id if side == "A" else self.comparison.run_b.id)
