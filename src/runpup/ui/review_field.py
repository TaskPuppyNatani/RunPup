"""Compact proposal-field controls; all accepted edits go through ProposedRun."""

import json
import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QToolButton, QVBoxLayout,
)

from runpup.domain.proposals import FieldState, ProposedField


INTEGERS = frozenset({
    "lora_rank", "seed", "per_device_batch_size", "gradient_accumulation",
    "parallelism", "effective_batch_size", "max_sequence_length", "warmup_steps",
    "dataset_example_count", "rating",
})
REALS = frozenset({"learning_rate", "epochs", "lora_alpha", "lora_dropout", "warmup_ratio", "best_metric_unidentified"})
BOOLEANS = frozenset({"load_in_4bit", "load_in_8bit", "double_quantization", "qlora", "favorite"})


def label(text, name="muted", *, wrap=False):
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setObjectName(name)
    widget.setWordWrap(wrap)
    widget.setMinimumWidth(0)
    return widget


def display(value):
    if value is None:
        return "Unknown"
    if type(value) is bool:
        return "True" if value else "False"
    return str(value)


class ReviewField(QFrame):
    edited = Signal(str, object)
    pending = Signal(str, bool)

    def __init__(self, field: ProposedField, title: str, parent=None):
        super().__init__(parent)
        self.field = field
        self.setObjectName("reviewField")
        self.setMinimumWidth(0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(2)
        heading = QHBoxLayout()
        heading.setSpacing(5)
        caption = label(title, "fieldTitle", wrap=True)
        caption.setFixedWidth(96)
        heading.addWidget(caption)
        self.badge = label("", "stateBadge")
        layout.addLayout(heading)

        self.choice = QComboBox()
        self.choice.setAccessibleName(f"Select candidate for {title}")
        self.choice.activated.connect(self._select_candidate)
        editing = QHBoxLayout()
        editing.setSpacing(5)
        boolean = field.key in BOOLEANS or type(field.value) is bool
        if boolean or field.key == "training_status":
            self.editor = QComboBox()
            options = (("Unknown", None), ("True", True), ("False", False)) if boolean else (
                ("Unknown", None), ("Completed", "completed"),
                ("Failed", "failed"), ("Interrupted", "interrupted"),
            )
            for caption, value in options:
                self.editor.addItem(caption, value)
            self.editor.activated.connect(self._pending)
        else:
            self.editor = QLineEdit()
            self.editor.setPlaceholderText('JSON array or quoted pattern' if field.key == "lora_target_modules"
                                           else "Unknown · enter a value")
            self.editor.textEdited.connect(self._pending)
            self.editor.returnPressed.connect(self.apply_edit)
        self.editor.setMinimumWidth(0)
        self.editor.setAccessibleName(title)
        heading.addWidget(self.editor, 1)
        heading.addWidget(self.badge)
        layout.addWidget(self.choice)
        self.apply_button = QPushButton("Apply")
        self.apply_button.setObjectName("compact")
        self.apply_button.setFixedWidth(50)
        self.apply_button.setAccessibleName(f"Apply {title}")
        self.apply_button.clicked.connect(self.apply_edit)
        editing.addWidget(self.apply_button)
        self.revert = QToolButton()
        self.revert.setText("↶")
        self.revert.setToolTip("Discard unapplied edit")
        self.revert.setAccessibleName(f"Discard unapplied {title} edit")
        self.revert.clicked.connect(self.discard_edit)
        editing.addWidget(self.revert)
        self.error = label("", "errorText", wrap=True)
        self.error.hide()
        layout.addWidget(self.error)
        self.evidence_button = QToolButton()
        self.evidence_button.setObjectName("evidence")
        self.evidence_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.evidence_button.setCheckable(True)
        self.evidence_button.toggled.connect(lambda checked: self.evidence.setVisible(checked))
        editing.insertWidget(0, self.evidence_button, 1)
        layout.addLayout(editing)
        self.evidence = label("", "evidenceText", wrap=True)
        self.evidence.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.evidence.hide()
        layout.addWidget(self.evidence)
        self.set_field(field)

    def set_field(self, field):
        self.field = field
        self.badge.setText(field.state.value.upper().replace("_", " "))
        self.badge.setProperty("state", field.state.value)
        self.badge.style().unpolish(self.badge)
        self.badge.style().polish(self.badge)
        self.choice.clear()
        self.choice.addItem("Choose a candidate…", None)
        for record in field.candidates:
            source = record.source_reference or record.source_type
            self.choice.addItem(f"{display(record.candidate_value)}  ·  {source}", record.candidate_value)
        self.choice.setVisible(field.state is FieldState.CONFLICT)
        evidence = []
        for record in (*field.candidates, *field.edits):
            source = record.source_reference or record.source_type
            value = record.current_value if record.user_modified else record.candidate_value
            text = f"{'User edit' if record.user_modified else 'Candidate'}: {display(value)}\n{source} · {record.confidence.value}"
            if record.original_value != value:
                text += f"\nOriginal: {display(record.original_value)}"
            evidence.append(text)
        self.evidence.setText("\n\n".join(evidence) or "No source value was found. You can leave this unknown.")
        sources = len(field.candidates)
        cue = "No detected source · optional edit"
        if sources:
            first = field.candidates[0]
            source = (first.source_reference or first.source_type).split("#")[0].rsplit("/", 1)[-1]
            cue = f"{source[:32]} · {first.confidence.value}"
            if sources > 1:
                cue += f" · +{sources - 1}"
        self.evidence_button.setText(f"Sources / edits · {cue} ▾")
        if field.state is FieldState.CONFLICT:
            self.evidence_button.setChecked(True)
        self.discard_edit()

    def _pending(self, *_):
        self.apply_button.show()
        self.revert.show()
        self.apply_button.setEnabled(True)
        self.revert.setEnabled(True)
        self.pending.emit(self.field.key, True)

    def discard_edit(self):
        value = self.field.value
        if isinstance(self.editor, QComboBox):
            self.editor.setCurrentIndex(max(0, self.editor.findData(value)))
        else:
            self.editor.setText("" if value is None else str(value))
        self.error.hide()
        self.apply_button.setEnabled(False)
        self.revert.setEnabled(False)
        self.apply_button.hide()
        self.revert.hide()
        self.pending.emit(self.field.key, False)

    def _select_candidate(self, index):
        if index > 0:
            self.edited.emit(self.field.key, self.choice.itemData(index))

    def apply_edit(self):
        try:
            if isinstance(self.editor, QComboBox):
                value = self.editor.currentData()
            else:
                text = self.editor.text().strip()
                value = text or None
                key = self.field.key
                if text and (key in INTEGERS or key.startswith("dataset_split_counts.")):
                    value = int(text)
                elif text and (key in REALS or key.startswith("metric:")):
                    value = float(text)
                    if not math.isfinite(value):
                        raise ValueError("Enter a finite number")
                elif text and key == "lora_target_modules":
                    modules = json.loads(text)
                    if not (isinstance(modules, str) or isinstance(modules, list) and
                            all(isinstance(item, str) for item in modules)):
                        raise ValueError("Use a JSON array of module names or a quoted pattern")
                    value = json.dumps(modules, ensure_ascii=False)
            self.edited.emit(self.field.key, value)
        except (ValueError, OverflowError):
            self.error.setText("Invalid value. Use a whole number." if self.field.key in INTEGERS else
                               "Invalid value. Use a finite number or valid JSON for target modules.")
            self.error.show()
            self._pending()
