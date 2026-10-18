"""Import location preferences only."""

from PySide6.QtWidgets import QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget

from runpup.ui.preferences import ImportPreferences
from runpup.ui.review_field import label


class SettingsPage(QWidget):
    def __init__(self, preferences: ImportPreferences, parent=None):
        super().__init__(parent)
        self.preferences = preferences
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 0)
        layout.setSpacing(16)
        layout.addWidget(label("Settings", "heading"))
        layout.addWidget(label("Choose where Import Run starts looking for training outputs.", wrap=True))
        card = QFrame()
        card.setObjectName("reviewCard")
        box = QVBoxLayout(card)
        box.setContentsMargins(16, 16, 16, 16)
        box.setSpacing(12)
        box.addWidget(label("Import", "sectionHeading"))
        box.addWidget(label("Default training outputs folder"))
        row = QHBoxLayout()
        self.default_directory = QLineEdit()
        self.default_directory.setReadOnly(True)
        self.default_directory.setMinimumWidth(0)
        self.default_directory.setPlaceholderText("No default folder set")
        self.default_directory.setAccessibleName("Default training outputs folder")
        row.addWidget(self.default_directory, 1)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse)
        row.addWidget(self.browse_button)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_default)
        row.addWidget(self.clear_button)
        box.addLayout(row)
        self.remember_last = QCheckBox("Remember last import location")
        self.remember_last.toggled.connect(self._set_remember_last)
        box.addWidget(self.remember_last)
        box.addWidget(label("When enabled, Import Run starts at the parent of your last selected run. Otherwise it uses your default folder, local Unsloth outputs when present, or your home folder.", wrap=True))
        box.addWidget(label("Changes take effect immediately. Missing folders are skipped.", wrap=True))
        layout.addWidget(card)
        layout.addStretch()
        self.refresh()

    def refresh(self):
        self.default_directory.setText(self.preferences.default_directory)
        self.default_directory.setToolTip(self.preferences.default_directory)
        self.clear_button.setEnabled(bool(self.preferences.default_directory))
        self.remember_last.blockSignals(True)
        self.remember_last.setChecked(self.preferences.remember_last)
        self.remember_last.blockSignals(False)

    def browse(self):
        selected = QFileDialog.getExistingDirectory(
            self, "Default Training Outputs Folder", self.preferences.initial_directory())
        if selected:
            self.preferences.default_directory = selected
            self.refresh()

    def clear_default(self):
        self.preferences.default_directory = ""
        self.refresh()

    def _set_remember_last(self, enabled):
        self.preferences.remember_last = enabled
