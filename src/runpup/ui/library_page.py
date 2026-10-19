"""Compact Library rows over cached, lightweight repository summaries."""

from html import escape
import sqlite3

from PySide6.QtCore import QAbstractListModel, QModelIndex, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPen
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QGridLayout, QHBoxLayout, QLineEdit, QListView,
    QPushButton, QStackedWidget, QStyledItemDelegate, QStyle, QVBoxLayout, QWidget,
)

from runpup.ui.library_view import LibraryCriteria, filter_options, summary_lines, summary_text, visible_runs
from runpup.ui.review_field import label


class RunListModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = ()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.rows):
            return None
        run = self.rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.AccessibleTextRole:
            return summary_text(run)
        if role == Qt.ItemDataRole.ToolTipRole:
            return "<qt>" + escape(summary_text(run)).replace("\n", "<br>") + "</qt>"
        if role == Qt.ItemDataRole.UserRole:
            return run.id

    def replace_rows(self, rows):
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()


class RunDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        return QSize(0, 138)

    def paint(self, painter, option, index):
        run = index.model().rows[index.row()]
        painter.save()
        painter.setRenderHint(painter.RenderHint.Antialiasing)
        rect = QRectF(option.rect).adjusted(3, 3, -3, -3)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        focused = bool(option.state & QStyle.StateFlag.State_HasFocus)
        painter.setBrush(QColor("#342b47" if selected else "#20242f"))
        painter.setPen(QPen(QColor("#8ecdd6" if focused else "#9972c5" if selected else "#3b465a"), 2 if focused else 1))
        painter.drawRoundedRect(rect, 8, 8)
        x, y, width = int(rect.x()) + 14, int(rect.y()), int(rect.width()) - 28
        font = QFont(option.font)

        def text(value, top, color, *, bold=False, size=13, available=width, elide=Qt.TextElideMode.ElideRight):
            font.setPixelSize(size)
            font.setBold(bold)
            painter.setFont(font)
            painter.setPen(QColor(color))
            shown = painter.fontMetrics().elidedText(value, elide, max(0, available))
            painter.drawText(x, y + top, shown)

        rating = f"{run.rating}/5" if run.rating is not None else "Unrated"
        badge = ("★ Favorite · " if run.favorite else "") + rating
        font.setPixelSize(13)
        painter.setFont(font)
        badge_width = painter.fontMetrics().horizontalAdvance(badge)
        # The primary name and human metadata share one line without colliding.
        text(run.display_name, 23, "#ebe9f2", bold=True, size=15, available=width - badge_width - 22)
        painter.setFont(QFont(option.font))
        font.setPixelSize(13)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor("#d4b9f6"))
        painter.drawText(x + width - badge_width, y + 23, badge)
        text("Model · " + (run.base_model or "Unknown"), 46, "#abdae3", elide=Qt.TextElideMode.ElideMiddle)
        metadata, progress = summary_lines(run)
        text(metadata, 67, "#bec6d8")
        text(progress, 88, "#aeb7ca")
        if not run.tags:
            text("No tags", 116, "#929db3")
        else:
            offset = 0
            for tag in run.tags[:3]:
                caption = painter.fontMetrics().elidedText(tag, Qt.TextElideMode.ElideRight, 150)
                chip_width = painter.fontMetrics().horizontalAdvance(caption) + 16
                painter.setBrush(QColor("#302a40"))
                painter.setPen(QPen(QColor("#5b496e")))
                chip = QRectF(x + offset, y + 99, chip_width, 23)
                painter.drawRoundedRect(chip, 4, 4)
                painter.setPen(QColor("#d8c9ee"))
                painter.drawText(x + offset + 8, y + 115, caption)
                offset += chip_width + 6
            if len(run.tags) > 3:
                painter.drawText(x + offset + 4, y + 115, f"+{len(run.tags) - 3} tags")
        painter.restore()


class LibraryPage(QWidget):
    open_requested = Signal(str)
    import_requested = Signal()

    def __init__(self, repository):
        super().__init__()
        self.repository = repository
        self.runs = ()
        self.selected_id = None
        self.refresh_failed = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 0)
        layout.setSpacing(12)
        heading = QHBoxLayout()
        heading.addWidget(label("Run Library", "heading"))
        heading.addStretch()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        heading.addWidget(refresh)
        self.open_button = QPushButton("Open Run Details")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._activate)
        heading.addWidget(self.open_button)
        import_button = QPushButton("Import Run")
        import_button.setObjectName("primary")
        import_button.clicked.connect(self.import_requested)
        heading.addWidget(import_button)
        layout.addLayout(heading)

        search_row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search runs, models, trainers, methods, tags…")
        self.search.setAccessibleName("Search saved runs")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.apply_criteria)
        search_row.addWidget(self.search, 1)
        self.reset_button = QPushButton("Reset filters")
        self.reset_button.clicked.connect(self.reset_filters)
        search_row.addWidget(self.reset_button)
        layout.addLayout(search_row)

        filters = QGridLayout()
        self.favorites = self._combo("Favorite filter", (("All runs", False), ("Favorites only", True)))
        self.rating = self._combo("Rating filter", (("All ratings", "all"), ("Rated", "rated"), ("Unrated", "unrated")))
        self.method = self._combo("Method filter", (("All methods", None),))
        self.tag = self._combo("Tag filter", (("All tags", None),))
        self.sort = self._combo("Sort saved runs", (
            ("Imported: newest first", "newest"), ("Imported: oldest first", "oldest"),
            ("Name: A–Z", "name_asc"), ("Name: Z–A", "name_desc"), ("Rating: high to low", "rating_desc")))
        for column, combo in enumerate((self.favorites, self.rating, self.method, self.tag)):
            filters.addWidget(combo, 0, column)
            filters.setColumnStretch(column, 1)
        filters.addWidget(self.sort, 1, 2, 1, 2)
        self.count = label("0 runs · local training history")
        filters.addWidget(self.count, 1, 0, 1, 2)
        layout.addLayout(filters)
        self.notice = label("", "local", wrap=True)
        self.notice.hide()
        layout.addWidget(self.notice)

        self.content = QStackedWidget()
        self.empty_state, empty_box = self._empty("No training runs yet.", "Keep the settings and results worth coming back to.")
        self.first_import = QPushButton("Import Your First Run")
        self.first_import.setObjectName("primary")
        self.first_import.clicked.connect(self.import_requested)
        empty_box.addWidget(self.first_import, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.content.addWidget(self.empty_state)
        self.no_results, no_results_box = self._empty("No matching runs", "No runs match the current search and filters.")
        clear = QPushButton("Clear search and filters")
        clear.clicked.connect(self.reset_filters)
        no_results_box.addWidget(clear, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.content.addWidget(self.no_results)
        self.run_list = QListView()
        self.run_list.setObjectName("runList")
        self.run_list.setAccessibleName("Saved runs")
        self.run_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.run_list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.run_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.run_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.run_list.setUniformItemSizes(True)
        self.model = RunListModel(self)
        self.run_list.setModel(self.model)
        self.run_list.setItemDelegate(RunDelegate(self.run_list))
        self.run_list.selectionModel().selectionChanged.connect(self._selection_changed)
        self.run_list.activated.connect(self._activate)
        self.content.addWidget(self.run_list)
        layout.addWidget(self.content, 1)
        QWidget.setTabOrder(self.search, self.favorites)
        QWidget.setTabOrder(self.favorites, self.rating)
        QWidget.setTabOrder(self.rating, self.method)
        QWidget.setTabOrder(self.method, self.tag)
        QWidget.setTabOrder(self.tag, self.sort)
        QWidget.setTabOrder(self.sort, self.run_list)

    def _combo(self, name, options):
        combo = QComboBox()
        combo.setMinimumWidth(0)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setAccessibleName(name)
        combo.setToolTip(name)
        for title, value in options:
            combo.addItem(title, value)
        combo.currentIndexChanged.connect(self.apply_criteria)
        return combo

    @staticmethod
    def _empty(title, explanation):
        widget = QWidget()
        box = QVBoxLayout(widget)
        box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for text, name in ((title, "emptyHeading"), (explanation, "muted")):
            caption = label(text, name, wrap=True)
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box.addWidget(caption)
        return widget, box

    def criteria(self):
        return LibraryCriteria(self.search.text(), self.favorites.currentData(), self.rating.currentData(),
                               self.method.currentData(), self.tag.currentData(), self.sort.currentData())

    def refresh(self):
        try:
            runs = self.repository.list_run_summaries()
        except (sqlite3.Error, ValueError) as error:
            self.refresh_failed = True
            self.notice.setText(f"Cannot refresh Library: {error}")
            self.notice.show()
            return
        if self.refresh_failed:
            self.notice.hide()
            self.refresh_failed = False
        self.runs = tuple(runs)
        for combo, attribute, all_label in ((self.method, "method", "All methods"), (self.tag, "tags", "All tags")):
            current, caption = combo.currentData(), combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(all_label, None)
            for value, title in filter_options(self.runs, attribute):
                combo.addItem(title, value)
            if current is not None and combo.findData(current) < 0:
                combo.addItem(caption, current)
            combo.setCurrentIndex(max(0, combo.findData(current)))
            combo.blockSignals(False)
        self.apply_criteria()

    def apply_criteria(self, *_):
        if not hasattr(self, "model"):
            return
        rows = visible_runs(self.runs, self.criteria())
        selected = self.selected_id
        selection = self.run_list.selectionModel()
        selection.blockSignals(True)
        self.model.replace_rows(rows)
        self.run_list.setCurrentIndex(QModelIndex())
        for row, run in enumerate(rows):
            if run.id == selected:
                self.run_list.setCurrentIndex(self.model.index(row))
                break
        selection.blockSignals(False)
        self._selection_changed()
        total, count = len(self.runs), len(rows)
        self.count.setText(f"{count} of {total} runs" if count != total else f"{count} {'run' if count == 1 else 'runs'} · local training history")
        self.content.setCurrentWidget(self.empty_state if not total else self.run_list if count else self.no_results)

    def _selection_changed(self, *_):
        selected = self.run_list.selectionModel().selectedRows()
        self.selected_id = selected[0].data(Qt.ItemDataRole.UserRole) if selected else None
        self.open_button.setEnabled(self.selected_id is not None)

    def _activate(self, *_):
        if self.selected_id is not None:
            self.open_requested.emit(self.selected_id)

    def reset_filters(self):
        for control in (self.search, self.favorites, self.rating, self.method, self.tag):
            control.blockSignals(True)
            if control is self.search:
                control.clear()
            else:
                control.setCurrentIndex(0)
            control.blockSignals(False)
        self.apply_criteria()

    def mark_saved(self, run):
        self.notice.setText(f"Saved ‘{run.display_name}’ to your library.")
        self.notice.show()
        for row, summary in enumerate(self.model.rows):
            if summary.id == run.id:
                self.run_list.setCurrentIndex(self.model.index(row))
                self.run_list.scrollTo(self.model.index(row))
                return
        self.notice.setText(self.notice.text() + " Your current search or filters hide this run.")
