"""Manage catalog mappings while external storage remains user-owned."""

from dataclasses import replace
from html import escape
import sqlite3

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit,
    QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)

from runpup.domain.models import RootKind, StorageRoot
from runpup.services.paths import PathResolver
from runpup.services.storage_roots import root_status, verify_storage_root
from runpup.ui.details_page import _card, _text
from runpup.ui.review_field import label


def _readonly(value=""):
    field = QLineEdit(value)
    field.setReadOnly(True)
    field.setCursorPosition(0)
    return field


class RootDialog(QDialog):
    """Add/name edit or an explicit two-stage mapping preview."""

    def __init__(self, page, mode, summary=None):
        super().__init__(page)
        self.page, self.mode, self.summary = page, mode, summary
        self.root = summary.root if summary else None
        self.candidate = None
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle({"add": "Add Storage Root", "edit": "Edit Storage Root",
                             "remap": "Remap Storage Root"}[mode])
        self.setMinimumWidth(480)
        self.resize(600, 330)
        layout = QVBoxLayout(self)
        layout.addWidget(_text(self.windowTitle(), "sectionHeading"))
        self.form = QWidget()
        form = QFormLayout(self.form)
        form.setContentsMargins(0, 0, 0, 0)
        self.name = QLineEdit(self.root.name if self.root else "")
        self.name.setAccessibleName("Root name")
        form.addRow("Name", self.name)
        self.name.setReadOnly(mode == "remap")
        self.kind = QComboBox()
        self.kind.addItem("Absolute", RootKind.ABSOLUTE)
        self.kind.addItem("Portable relative", RootKind.PORTABLE_RELATIVE)
        if self.root:
            self.kind.setCurrentIndex(self.kind.findData(self.root.kind))
        self.location = QLineEdit(self.root.location if self.root else "")
        self.location.setAccessibleName("Mapped location")
        self.browse = QPushButton("Choose Folder…")
        self.browse.clicked.connect(self._browse)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        if mode != "edit":
            form.addRow("Path type", self.kind)
            form.addRow("Location", self.location)
            form.addRow("", self.browse)
            form.addRow(_text("Offline mappings are allowed. Same-path roots remain separate identities. "
                               "Portable relative paths need an existing resolver base; without one, "
                               "their status is Unverified.", "muted"))
        layout.addWidget(self.form)
        self.preview = QWidget()
        preview = QFormLayout(self.preview)
        preview.setContentsMargins(0, 0, 0, 0)
        if self.root:
            preview.addRow("Name", _readonly(self.root.name))
            preview.addRow("Root ID", _readonly(self.root.id))
            preview.addRow("Current location", _readonly(self.root.location))
        self.new_location = _readonly()
        preview.addRow("New location", self.new_location)
        self.preview_type = _text("")
        preview.addRow("New path type", self.preview_type)
        self.preview_note = _text("")
        preview.addRow(self.preview_note)
        self.preview.hide()
        layout.addWidget(self.preview)
        self.error = _text("", "errorText")
        layout.addWidget(self.error)
        layout.addStretch()
        actions = QHBoxLayout()
        self.back = QPushButton("Back")
        self.back.clicked.connect(self._back)
        self.back.hide()
        actions.addWidget(self.back)
        actions.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        actions.addWidget(cancel)
        self.save = QPushButton("Preview Remap" if mode == "remap" else "Save")
        self.save.setObjectName("primary")
        self.save.setDefault(True)
        self.save.clicked.connect(self._save)
        actions.addWidget(self.save)
        layout.addLayout(actions)
        self._kind_changed()
        self.setModal(True)

    def _kind_changed(self):
        self.browse.setEnabled(self.kind.currentData() == RootKind.ABSOLUTE)

    def _browse(self):
        # Do not probe the mapped location on the UI thread just to seed a chooser.
        location = QFileDialog.getExistingDirectory(self, "Choose Storage Root")
        if location:
            self.location.setText(location)

    def _back(self):
        self.candidate = None
        self.preview.hide()
        self.form.show()
        self.back.hide()
        self.save.setText("Preview Remap")

    def _save(self):
        self.error.clear()
        try:
            if self.mode == "edit":
                self.candidate = replace(self.root, name=self.name.text())
            elif self.candidate is None:
                self.candidate = (replace(self.root, location=self.location.text(), kind=self.kind.currentData())
                                  if self.root else StorageRoot(name=self.name.text(),
                                      location=self.location.text(), kind=self.kind.currentData()))
                if self.mode == "remap":
                    self.new_location.setText(self.candidate.location)
                    self.new_location.setCursorPosition(0)
                    self.preview_type.setText(self.candidate.kind.value)
                    self.preview_note.setText(
                        f"{self.summary.dependencies} will keep this root identity.\n\n"
                        "The new mapping has not been verified. RunPup will not move or modify any files.")
                    self.form.hide()
                    self.preview.show()
                    self.back.show()
                    self.save.setText("Remap")
                    return
            if self.page.mutate(self.mode, self.root, self.candidate):
                self.accept()
            else:
                self.error.setText(self.page.message.text())
                self.candidate = None
                if self.mode == "remap":
                    self._back()
        except ValueError as error:
            self.candidate = None
            self.error.setText(str(error))


class RemoveRootDialog(QDialog):
    def __init__(self, page, summary):
        super().__init__(page)
        self.page, self.summary = page, summary
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("Remove Storage Root Reference")
        self.setMinimumWidth(480)
        self.resize(600, 280)
        layout = QVBoxLayout(self)
        layout.addWidget(_text(summary.root.name, "sectionHeading"))
        layout.addWidget(_readonly(summary.root.location))
        self.explanation = _text(
            f"This root cannot be removed: {summary.dependencies} still depend on it. "
            "Remap it instead. No catalog records or external files were changed."
            if summary.in_use else
            "RunPup will forget this unused storage root reference.\n"
            "No external folders or files will be deleted.")
        layout.addWidget(self.explanation)
        layout.addStretch()
        actions = QHBoxLayout()
        actions.addStretch()
        cancel = QPushButton("Close" if summary.in_use else "Cancel")
        cancel.clicked.connect(self.reject)
        actions.addWidget(cancel)
        self.remove = QPushButton("Remove Reference")
        self.remove.setEnabled(not summary.in_use)
        self.remove.clicked.connect(self._remove)
        actions.addWidget(self.remove)
        layout.addLayout(actions)
        self.setModal(True)

    def _remove(self):
        if self.page.mutate("remove", self.summary.root):
            self.accept()
        else:
            self.explanation.setText(self.page.message.text())


class _RootTask(QThread):
    def __init__(self, repository, resolver, roots, verify, generation, parent):
        super().__init__(parent)
        self.repository, self.resolver, self.roots = repository, resolver, roots
        self.verify, self.generation = verify, generation
        self.result, self.error = {}, None

    def run(self):
        try:
            for root in self.roots:
                if self.isInterruptionRequested():
                    break
                self.result[root.id] = (
                    verify_storage_root(self.repository, root, resolver=self.resolver,
                                        cancelled=self.isInterruptionRequested)
                    if self.verify else self.resolver.resolve(root, "."))
        except (ValueError, OSError, sqlite3.Error) as error:
            self.error = str(error)


class StorageRootsPage(QWidget):
    idle = Signal()

    def __init__(self, repository, *, resolver=None):
        super().__init__()
        self.repository = repository
        self.resolver = resolver if resolver is not None else PathResolver()
        self.summaries, self.states, self.verifications = {}, {}, {}
        self.task, self.dialog = None, None
        self.generation, self.pending = 0, False
        self.closing = False
        layout = QVBoxLayout(self)
        heading = QHBoxLayout()
        heading.addWidget(label("Storage Roots", "heading"))
        heading.addStretch()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        heading.addWidget(self.refresh_button)
        self.add_button = QPushButton("Add Root")
        self.add_button.setObjectName("primary")
        self.add_button.clicked.connect(lambda: self.open_dialog("add"))
        heading.addWidget(self.add_button)
        layout.addLayout(heading)
        layout.addWidget(_text("Stable mappings to external training files. Counts describe catalog references.", "muted"))
        self.message = _text("", "muted")
        layout.addWidget(self.message)
        self.empty = _text("No storage roots configured.\nAdd a storage root to keep external model and "
                           "checkpoint references relocatable without copying files.", "muted")
        layout.addWidget(self.empty, 1)
        self.empty.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.roots = QListWidget()
        self.roots.setObjectName("storageRootList")
        self.roots.setAccessibleName("Storage roots")
        self.roots.setMinimumHeight(110)
        self.roots.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.roots.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.roots.currentItemChanged.connect(self._selection_changed)
        layout.addWidget(self.roots, 1)
        self.inspector, box = _card("Selected root")
        self.name = _text("", "sectionHeading")
        box.addWidget(self.name)
        form = QFormLayout()
        self.location, self.identity = _readonly(), _readonly()
        form.addRow("Mapped location", self.location)
        form.addRow("Stable ID", self.identity)
        box.addLayout(form)
        self.info = _text("", "muted")
        box.addWidget(self.info)
        actions = QHBoxLayout()
        self.edit_button, self.remap_button = QPushButton("Edit Name"), QPushButton("Remap")
        self.verify_button, self.remove_button = QPushButton("Verify"), QPushButton("Remove Reference")
        for button, callback in (
            (self.edit_button, lambda: self.open_dialog("edit")),
            (self.remap_button, lambda: self.open_dialog("remap")),
            (self.verify_button, self.verify_selected),
            (self.remove_button, lambda: self.open_dialog("remove")),
        ):
            button.clicked.connect(callback)
            actions.addWidget(button)
        actions.addStretch()
        box.addLayout(actions)
        self.verification_text = _text("", "muted")
        box.addWidget(self.verification_text)
        self.reason = _readonly()
        self.reason.setAccessibleName("Availability detail")
        self.reason.hide()
        box.addWidget(self.reason)
        layout.addWidget(self.inspector)
        self._selection_changed()

    @property
    def busy(self):
        return self.task is not None

    @property
    def selected(self):
        item = self.roots.currentItem()
        return self.summaries.get(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def refresh(self):
        if self.closing:
            return False
        identity = self.selected.root.id if self.selected else None
        self.generation += 1
        self.states.clear()
        self.verifications.clear()
        self.pending = False
        try:
            summaries = self.repository.list_storage_root_summaries()
        except (ValueError, OSError, sqlite3.Error):
            self.summaries = {}
            self.roots.clear()
            self.empty.hide()
            self.message.setText("Cannot load storage roots. No mappings were changed. Refresh to try again.")
            if self.task:
                self.task.requestInterruption()
            self._selection_changed()
            return False
        self.summaries = {s.root.id: s for s in summaries}
        self.roots.blockSignals(True)
        self.roots.clear()
        selected_item = None
        for summary in summaries:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, summary.root.id)
            self.roots.addItem(item)
            self._render_item(item)
            if summary.root.id == identity:
                selected_item = item
        self.roots.setCurrentItem(selected_item or self.roots.item(0))
        self.roots.blockSignals(False)
        self.empty.setVisible(not summaries)
        self.roots.setVisible(bool(summaries))
        self.message.setText("Checking root availability…" if summaries else "")
        if self.busy:
            self.pending = bool(summaries)
            self.task.requestInterruption()
        elif summaries:
            self._start(False, tuple(s.root for s in summaries))
        self._selection_changed()
        return True

    def _render_item(self, item):
        summary = self.summaries[item.data(Qt.ItemDataRole.UserRole)]
        root = summary.root
        state = self.states.get(root.id)
        status = root_status(state) if state else "Unverified"
        item.setText(f"{root.name}  ·  {status}  ·  {summary.artifact_count} artifact references\n"
                     f"{root.location}\n{root.kind.value}")
        item.setToolTip("<qt>" + escape(
            f"{root.name}\n{root.location}\n{root.id}\n{summary.dependencies}"
        ).replace("\n", "<br>") + "</qt>")

    def _selection_changed(self, *args):
        summary = self.selected
        self.inspector.setVisible(summary is not None)
        self.add_button.setEnabled(not self.busy and not self.closing)
        for button in (self.edit_button, self.remap_button, self.verify_button, self.remove_button):
            button.setEnabled(summary is not None and not self.busy and not self.closing)
        if summary is None:
            return
        root = summary.root
        self.name.setText(root.name)
        for field, value in ((self.location, root.location), (self.identity, root.id)):
            field.setText(value)
            field.setCursorPosition(0)
        state = self.states.get(root.id)
        status = root_status(state) if state else "Unverified"
        self.info.setText(f"{status} · {root.kind.value}\n{summary.dependencies}")
        verification = self.verifications.get(root.id)
        text = ("Last existence check · " + verification.text if verification else
                "Verify checks known artifact paths without reading their contents.")
        self.verification_text.setText(text)
        self.reason.setVisible(bool(state and state.reason))
        self.reason.setText(state.reason if state and state.reason else "")
        self.reason.setCursorPosition(0)

    def open_dialog(self, mode):
        if self.busy or self.closing or (self.dialog is not None and self.dialog.isVisible()):
            return
        if mode != "add" and self.selected is None:
            return
        self.dialog = (RemoveRootDialog(self, self.selected) if mode == "remove" else
                       RootDialog(self, mode, self.selected))
        self.dialog.finished.connect(lambda: setattr(self, "dialog", None))
        self.dialog.open()

    def mutate(self, mode, root, candidate=None):
        if self.busy or self.closing:
            return False
        try:
            if mode == "add":
                self.repository.add_storage_root(candidate)
            elif mode == "edit":
                self.repository.rename_storage_root(root, candidate.name)
            elif mode == "remap":
                self.repository.remap_storage_root(root, location=candidate.location, kind=candidate.kind)
            elif mode == "remove":
                self.repository.remove_storage_root(root)
            else:
                raise ValueError("Unknown root action")
        except ValueError as error:
            self.message.setText(str(error))
            return False
        except (OSError, sqlite3.Error):
            self.message.setText("Cannot save this storage root change. No catalog records or external "
                                 "files were changed. Refresh the catalog and try again.")
            return False
        if not self.refresh():
            self.message.setText("The storage root change was saved, but the list could not refresh. "
                                 "No external files were changed. Refresh to try again.")
        if candidate and candidate.id in self.summaries:
            for index in range(self.roots.count()):
                item = self.roots.item(index)
                if item.data(Qt.ItemDataRole.UserRole) == candidate.id:
                    self.roots.setCurrentItem(item)
                    break
        return True

    def verify_selected(self):
        if self.busy or self.closing or not self.selected:
            return
        self.message.setText(f"Verifying {self.selected.root.name}…")
        self._start(True, (self.selected.root,))

    def _start(self, verify, roots):
        self.task = _RootTask(self.repository, self.resolver, roots, verify, self.generation, self)
        self.task.finished.connect(self._finished)
        self.task.start()
        self._selection_changed()

    def _finished(self):
        task, self.task = self.task, None
        task.deleteLater()
        if not self.closing:
            if self.pending:
                self.pending = False
                self._start(False, tuple(s.root for s in self.summaries.values()))
                return
            if task.generation == self.generation:
                if task.error:
                    self.message.setText("Could not check this storage root. No mappings or external files "
                                         "were changed. Refresh, check the drive or mount, and try again.")
                else:
                    for root in task.roots:
                        summary = self.summaries.get(root.id)
                        result = task.result.get(root.id)
                        if summary and summary.root == root and result:
                            self.states[root.id] = result.resolution if task.verify else result
                            if task.verify:
                                self.verifications[root.id] = result
                    for index in range(self.roots.count()):
                        self._render_item(self.roots.item(index))
                    self.message.setText("Existence check complete. Results reflect the time checked." if task.verify
                                         else "Root availability checked. Use Verify to check known artifact paths.")
            self._selection_changed()
        self.idle.emit()

    def finish_before_close(self):
        self.closing = True
        self.pending = False
        if self.task:
            self.task.requestInterruption()
        if self.dialog:
            self.dialog.reject()
        self._selection_changed()
