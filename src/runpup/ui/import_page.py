"""The bounded Import/Review workspace over the sealed importer and commit service."""

from collections import Counter
import logging
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QProgressBar,
    QListWidget, QListWidgetItem, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from runpup.domain.models import StorageRoot
from runpup.domain.proposals import FieldState
from runpup.importers import HFTrainerImporter
from runpup.services.import_commit import ProposedRunCommitService, ProposalValidationError
from runpup.ui.diagnostics import format_diagnostics
from runpup.ui.preferences import ImportPreferences
from runpup.ui.review_field import ReviewField, label


_CORE_REVIEW_FIELDS = frozenset({
    "display_name", "source_trainer", "base_model", "method", "dataset_identity",
    "learning_rate", "epochs", "per_device_batch_size", "gradient_accumulation", "max_sequence_length",
})
_SUMMARY_LABELS = {
    "detected": "Detected", "needs_attention": "Needs attention", "unavailable": "Unavailable",
    "conflict": "Conflicts", "artifacts": "Artifacts",
}


def _review_visible(field):
    return field.key in _CORE_REVIEW_FIELDS or bool(field.candidates or field.edits)


def _review_fields(proposal):
    fields = list(proposal.fields)
    if not any(field.key == "source_trainer" for field in fields):
        fields.append(proposal.field("source_trainer"))
    return fields


class _ImportTask(QThread):
    """Only this workflow's blocking operations; never touches widgets."""

    def __init__(self, action, importer, service, *, root=None, proposal=None):
        # App ownership keeps a running thread alive even if a receiver is deleted.
        super().__init__(QApplication.instance())
        self.action, self.importer, self.service = action, importer, service
        self.root, self.proposal = root, proposal
        self.value = self.error = None

    def run(self):
        try:
            if self.action == "scan":
                detection = self.importer.detect(self.root)
                if self.isInterruptionRequested():
                    return
                proposed = self.importer.scan(self.root) if detection.recognized else None
                validation = self.service.validate(proposed) if proposed is not None else None
                self.value = (detection, proposed, validation)
            elif self.action == "validate":
                self.value = self.service.validate(self.proposal)
            elif self.action == "save":
                self.value = self.service.commit(self.proposal)
        except Exception as error:
            logging.getLogger(__name__).exception("Import %s failed", self.action)
            self.error = error


class ImportPage(QWidget):
    run_saved = Signal(object)
    idle = Signal()

    def __init__(self, repository, parent=None, *, importer=None, service=None, preferences=None):
        super().__init__(parent)
        self.preferences = preferences if preferences is not None else ImportPreferences()
        self.importer = importer or HFTrainerImporter()
        self.service = service or ProposedRunCommitService(repository)
        self.proposal = self.validation = self.source_root = None
        self._worker = None
        self._closing = False
        self._pending_fields = set()
        self._displayed_diagnostics = ()
        self.field_rows = {}
        self.cards = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(label("Import Run", "heading"))
        layout.addWidget(label("Scan a training folder. Review the evidence. Save to your library."))
        stages = QHBoxLayout()
        self.stages = []
        for number, title in enumerate(("Choose Folder", "Scan", "Review ProposedRun", "Save"), 1):
            badge = label(f"{number}   {title}", "stage")
            stages.addWidget(badge, 1)
            self.stages.append(badge)
        layout.addLayout(stages)
        source = QFrame()
        source.setObjectName("sourcePanel")
        source_layout = QHBoxLayout(source)
        source_labels = QVBoxLayout()
        source_labels.addWidget(label("SELECTED TRAINING FOLDER", "eyebrow"))
        self.source_path = label("No folder selected", "sourcePath", wrap=True)
        source_labels.addWidget(self.source_path)
        self.scan_info = label("HF Trainer / PEFT / Unsloth-compatible metadata", wrap=True)
        source_labels.addWidget(self.scan_info)
        source_layout.addLayout(source_labels, 1)
        self.choose_button = QPushButton("Choose Training Folder")
        self.choose_button.setObjectName("primary")
        self.choose_button.clicked.connect(self.choose_folder)
        source_layout.addWidget(self.choose_button)
        self.scan_button = QPushButton("Scan Folder")
        self.scan_button.setEnabled(False)
        self.scan_button.clicked.connect(self.start_scan)
        source_layout.addWidget(self.scan_button)
        layout.addWidget(source)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.message = label("Choose a folder to begin. Your training files stay where they are.", wrap=True)
        layout.addWidget(self.message)

        columns = QHBoxLayout()
        columns.setSpacing(12)
        self.review_scroll = QScrollArea()
        self.review_scroll.setWidgetResizable(True)
        self.review_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.review_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.review_body = QWidget()
        self.grid = QGridLayout(self.review_body)
        self.grid.setContentsMargins(0, 0, 4, 0)
        self.grid.setSpacing(10)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.card_columns = [QVBoxLayout(), QVBoxLayout()]
        for i, column in enumerate(self.card_columns):
            column.setSpacing(10)
            column.setAlignment(Qt.AlignmentFlag.AlignTop)
            self.grid.addLayout(column, 0, i)
        self._review_viewport = self.review_scroll.viewport()
        self._review_viewport.installEventFilter(self)
        self.review_scroll.setWidget(self.review_body)
        columns.addWidget(self.review_scroll, 1)

        summary_scroll = QScrollArea()
        summary_scroll.setWidgetResizable(True)
        summary_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        summary_scroll.setFrameShape(QFrame.Shape.NoFrame)
        summary_scroll.setFixedWidth(280)
        summary = QWidget()
        right = QVBoxLayout(summary)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)
        counts_card, counts_layout = self._card("Import Summary")
        self.count_labels = {}
        counts_grid = QGridLayout()
        for i, (key, caption) in enumerate(_SUMMARY_LABELS.items()):
            counter = label(f"0\n{caption}", "counter", wrap=True)
            counter.setProperty("state", key)
            counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
            counts_grid.addWidget(counter, i // 2, i % 2, 1, 2 if key == "artifacts" else 1)
            self.count_labels[key] = counter
        self.count_labels["needs_attention"].setToolTip("Unknown review fields you may want to fill in manually.")
        self.count_labels["unavailable"].setToolTip("Metadata RunPup could not safely recover from this training output.")
        counts_grid.setSpacing(4)
        counts_layout.addLayout(counts_grid)
        self.edited_count = label("0 user edited")
        counts_layout.addWidget(self.edited_count)
        right.addWidget(counts_card)
        warning_card, warning_layout = self._card("Warnings / Diagnostics", "warningCard")
        warning_heading = warning_layout.takeAt(0).widget()
        warning_heading.setWordWrap(True)
        warning_header = QHBoxLayout()
        warning_header.addWidget(warning_heading, 1)
        self.copy_diagnostics_button = QPushButton("Copy")
        self.copy_diagnostics_button.setObjectName("compact")
        self.copy_diagnostics_button.setToolTip("Copy diagnostics")
        self.copy_diagnostics_button.setAccessibleName("Copy diagnostics")
        self.copy_diagnostics_button.clicked.connect(self.copy_diagnostics)
        warning_header.addWidget(self.copy_diagnostics_button)
        warning_layout.addLayout(warning_header)
        self.diagnostics_text = label("Scan a folder to see diagnostics.", wrap=True)
        self.diagnostics_text.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.diagnostics_text.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        diagnostics_scroll = QScrollArea()
        diagnostics_scroll.setWidgetResizable(True)
        diagnostics_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        diagnostics_scroll.setFrameShape(QFrame.Shape.NoFrame)
        diagnostics_scroll.setMaximumHeight(180)
        diagnostics_scroll.setWidget(self.diagnostics_text)
        warning_layout.addWidget(diagnostics_scroll)
        right.addWidget(warning_card)
        blocking_card, blocking_layout = self._card("Blocking Items", "blockingCard")
        self.blocking_text = label("No proposal to save yet.", wrap=True)
        blocking_layout.addWidget(self.blocking_text)
        right.addWidget(blocking_card)
        note, note_layout = self._card("Provenance Note", "noteCard")
        note_layout.addWidget(label("Every candidate keeps its source. Expand Sources / edit history to inspect the evidence.", wrap=True))
        right.addWidget(note)
        right.addStretch()
        summary_scroll.setWidget(summary)
        columns.addWidget(summary_scroll)
        layout.addLayout(columns, 1)
        footer = QHBoxLayout()
        footer.addWidget(label("Local catalog  ·  External files remain in place", wrap=True), 1)
        self.save_button = QPushButton("Save Run")
        self.save_button.setObjectName("primary")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_run)
        footer.addWidget(self.save_button)
        layout.addLayout(footer)
        self.reset()

    @staticmethod
    def _card(title, name="reviewCard"):
        card = QFrame()
        card.setObjectName(name)
        card.setMinimumWidth(0)
        card.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        box = QVBoxLayout(card)
        box.setContentsMargins(12, 10, 12, 10)
        box.setSpacing(5)
        box.addWidget(label(title, "sectionHeading"))
        return card, box

    @property
    def busy(self):
        return self._worker is not None

    def choose_folder(self):
        if self.busy:
            return
        selected = QFileDialog.getExistingDirectory(
            self, "Choose Training Folder", self.preferences.initial_directory())
        if selected:
            self.select_folder(selected)

    def select_folder(self, path):
        if self.busy:
            return
        absolute = str(Path(path).absolute())
        self.preferences.remember_selection(absolute)
        if self.source_root is None or self.source_root.location != absolute:
            self.reset()
            self.source_root = StorageRoot(name=Path(absolute).name or "Training folder", location=absolute)
        self.source_path.setText(absolute)
        self.source_path.setToolTip(absolute)
        self.message.setText("Folder selected. Scan to detect training metadata and external artifacts.")
        self.scan_button.setEnabled(True)
        self._set_stage(1)

    def start_scan(self):
        if self.source_root is not None and not self.busy:
            self._start("scan")

    def _start(self, action):
        self._worker = _ImportTask(action, self.importer, self.service,
                                   root=self.source_root, proposal=self.proposal)
        self._worker.finished.connect(self._task_finished)
        self._worker.finished.connect(self._worker.deleteLater)
        self._set_busy(True)
        self.message.setText({"scan": "Scanning safely… reading supported metadata and discovering references.",
                              "validate": "Checking reviewed values and references…",
                              "save": "Saving reviewed run to the local catalog…"}[action])
        self._set_stage(2 if action == "scan" else 4 if action == "save" else 3)
        self._worker.start()

    def _set_busy(self, busy):
        self.progress.setVisible(busy)
        self.choose_button.setEnabled(not busy)
        self.scan_button.setEnabled(not busy and self.source_root is not None)
        self.review_body.setEnabled(not busy)
        self._update_save()

    @Slot()
    def _task_finished(self):
        worker = self._worker
        self._worker = None
        if self._closing:
            self.idle.emit()
            return
        self._set_busy(False)
        if worker.error is not None:
            if isinstance(worker.error, ProposalValidationError):
                self.validation = worker.error.validation
                self._update_summary()
            if worker.action == "save":
                self.message.setText("Save failed. Nothing was saved by this attempt. Your review is kept. Check the catalog location and try Save Run again.")
            else:
                self.validation = None
                action = "Scan" if worker.action == "scan" else "Review validation"
                retained = "Your existing review is kept. " if self.proposal is not None else ""
                self.message.setText(f"{action} failed. {retained}Check folder access and try again with Scan Folder.")
            self._set_stage(3 if self.proposal is not None else 1)
            self._update_save()
        elif worker.action == "scan":
            detection, proposed, validation = worker.value
            if proposed is None:
                self.validation = None
                retained = " Previous review is kept." if self.proposal is not None else ""
                self.message.setText(f"No supported training output could be scanned.{retained} Choose a readable HF Trainer / PEFT output folder and try again.")
                self._set_diagnostics(detection.diagnostics)
                self._update_save()
                self.blocking_text.setText("A recognized, safely scanned training folder is required.")
                self._set_stage(1)
            else:
                self.proposal, self.validation = proposed, validation
                self._render_review()
                self.message.setText("Scan complete. Review the detected values and resolve any blocking items.")
                self._set_stage(3)
        elif worker.action == "validate":
            self.validation = worker.value
            self._update_summary()
            self.message.setText("Review updated. Imported candidates and your edit history are retained.")
            self._set_stage(3)
        else:
            saved = worker.value
            self.reset()
            self.run_saved.emit(saved)
        self.idle.emit()

    def _clear_review(self):
        for column in self.card_columns:
            while column.count():
                item = column.takeAt(0)
                if item.widget():
                    item.widget().hide()
                    item.widget().deleteLater()
        self.cards.clear()
        self.field_rows.clear()
        self._pending_fields.clear()

    def reset(self):
        self.proposal = self.validation = self.source_root = None
        self._clear_review()
        self.source_path.setText("No folder selected")
        self.scan_info.setText("HF Trainer / PEFT / Unsloth-compatible metadata")
        self.message.setText("Choose a folder to begin. Your training files stay where they are.")
        card, box = self._card("Bring your training history into focus")
        box.addWidget(label("Choose an existing training output folder, then scan. Review settings, sources, warnings, and artifact references before saving.", wrap=True))
        self.cards.append(card)
        self._arrange_cards()
        self._set_stage(1)
        self._set_busy(False)
        self._update_summary()

    def _render_review(self):
        self._clear_review()
        p = self.proposal
        groups = {title: [] for title in ("Identity", "Model", "Dataset", "Training", "LoRA / QLoRA", "Results / Metrics", "Other metadata")}
        fields = _review_fields(p)
        order = {key: i for i, key in enumerate((
            "display_name", "source_trainer", "base_model", "model_revision", "method",
            "dataset_identity", "dataset_hash", "dataset_example_count", "learning_rate",
            "epochs", "observed_epoch", "global_step", "max_steps", "per_device_batch_size", "gradient_accumulation", "max_sequence_length",
            "optimizer", "scheduler", "lora_rank", "lora_alpha", "lora_dropout", "lora_target_modules",
        ))}
        fields.sort(key=lambda field: (order.get(field.key, len(order)), field.key))
        for field in fields:
            key = field.key
            if not _review_visible(field):
                continue
            if key.startswith("metric:"):
                group = "Results / Metrics"
            elif key.startswith(("lora_", "quantization_", "load_in_")) or key in {"qlora", "double_quantization"}:
                group = "LoRA / QLoRA"
            elif key.startswith("dataset_"):
                group = "Dataset"
            elif key in {"display_name", "source_trainer", "training_started_at", "training_finished_at", "training_status", "notes", "rating", "favorite"}:
                group = "Identity"
            elif key in {"base_model", "model_revision", "method", "model_dtype"}:
                group = "Model"
            elif key in {"learning_rate", "epochs", "observed_epoch", "global_step", "max_steps", "seed", "precision", "optimizer", "scheduler", "warmup_steps", "warmup_ratio", "parallelism", "per_device_batch_size", "gradient_accumulation", "effective_batch_size", "max_sequence_length"}:
                group = "Training"
            else:
                group = "Other metadata"
            groups[group].append(field)
        metrics = {metric.id: metric for metric in p.metrics}
        for title, fields in groups.items():
            if not fields:
                continue
            card, box = self._card(title)
            for field in fields:
                caption = {"display_name": "Run name", "source_trainer": "Source trainer", "base_model": "Base model",
                           "dataset_identity": "Dataset name", "lora_rank": "LoRA rank (r)",
                           "lora_alpha": "LoRA alpha", "lora_target_modules": "Target modules"}.get(
                               field.key, field.key.replace("_", " ").capitalize())
                if field.key.startswith("metric:"):
                    metric = next((metrics[r.target_id] for r in field.candidates if r.target_id in metrics), None)
                    if metric:
                        caption = f"{metric.name} · {metric.context} · {metric.summary.value}"
                        if metric.step is not None:
                            caption += f" · step {metric.step}"
                        if metric.unit:
                            caption += f" · {metric.unit}"
                row = ReviewField(field, caption)
                row.edited.connect(self._edit_field)
                row.pending.connect(self._pending_changed)
                self.field_rows[field.key] = row
                box.addWidget(row)
            self.cards.append(card)
        if p.artifacts:
            card, box = self._card(f"Artifacts · {len(p.artifacts)} external references")
            artifacts = QListWidget()
            artifacts.setObjectName("artifactList")
            artifacts.setAccessibleName("External artifact references")
            artifacts.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            artifacts.setMinimumWidth(0)
            artifacts.setMaximumHeight(250)
            artifacts.setWordWrap(True)
            for artifact in p.artifacts:
                size = f" · {artifact.size_bytes:,} bytes" if artifact.size_bytes is not None else ""
                item = QListWidgetItem(f"{artifact.display_name}  ·  {artifact.category}{size}\n{artifact.relative_path}")
                item.setToolTip(f"Storage root: {artifact.storage_root_id}\n{artifact.relative_path}")
                artifacts.addItem(item)
            box.addWidget(artifacts)
            self.cards.append(card)
        trainer = {"huggingface_trainer": "Hugging Face Trainer", "unsloth": "Unsloth"}.get(
            p.source_trainer, p.source_trainer or "Unknown")
        self.scan_info.setText(f"HF Trainer / PEFT importer · Detected trainer: {trainer}\n"
                               f"{len(p.detection_evidence)} metadata sources recognized · {len(p.artifacts)} artifacts")
        self.scan_info.setToolTip("\n".join(p.detection_evidence))
        self._arrange_cards()
        self._update_summary()

    def _arrange_cards(self):
        columns = 2 if self.review_scroll.viewport().width() >= 690 else 1
        for column in self.card_columns:
            while column.count():
                column.takeAt(0)
        for index, card in enumerate(self.cards):
            self.card_columns[index % columns].addWidget(card)
        for column in range(2):
            self.grid.setColumnStretch(column, 1 if column < columns else 0)

    def eventFilter(self, watched, event):
        if watched is getattr(self, "_review_viewport", None) and event.type() == QEvent.Type.Resize:
            self._arrange_cards()
        return super().eventFilter(watched, event)

    @Slot(str, bool)
    def _pending_changed(self, key, pending):
        if pending:
            self._pending_fields.add(key)
        else:
            self._pending_fields.discard(key)
        self._update_save()

    @Slot(str, object)
    def _edit_field(self, key, value):
        if self.busy or self.proposal is None:
            return
        self.proposal = self.proposal.with_user_edit(key, value)
        self.field_rows[key].set_field(self.proposal.field(key))
        self.validation = None
        self._update_summary()
        self._start("validate")

    def _diagnostics(self, items):
        artifacts = {a.id: a.relative_path for a in self.proposal.artifacts} if self.proposal else {}
        root = self.proposal.source_storage_root if self.proposal else None
        return format_diagnostics(items, artifact_paths=artifacts, root=root)

    def _set_diagnostics(self, items, *, empty="No diagnostics."):
        self._displayed_diagnostics = tuple(items)
        self.diagnostics_text.setText(self._diagnostics(self._displayed_diagnostics) if self._displayed_diagnostics else empty)
        self.copy_diagnostics_button.setEnabled(bool(self._displayed_diagnostics))

    def copy_diagnostics(self):
        if self._displayed_diagnostics:
            QApplication.clipboard().setText("RunPup Import Diagnostics\n\n" + self.diagnostics_text.text())

    def _update_summary(self):
        fields = _review_fields(self.proposal) if self.proposal else ()
        counts = Counter(field.state.value for field in fields)
        for field in fields:
            if field.state is FieldState.UNKNOWN:
                counts["needs_attention" if _review_visible(field) else "unavailable"] += 1
        counts["artifacts"] = len(self.proposal.artifacts) if self.proposal else 0
        for key, widget in self.count_labels.items():
            widget.setText(f"{counts[key]}\n{_SUMMARY_LABELS[key]}")
        self.edited_count.setText(f"{counts['user_edited']} user edited")
        for card in self.cards:
            conflict = any(row.field.state is FieldState.CONFLICT for row in card.findChildren(ReviewField))
            card.setProperty("conflict", conflict)
            card.style().unpolish(card)
            card.style().polish(card)
        if self.validation is not None:
            self._set_diagnostics(self.validation.warnings)
            self.blocking_text.setText(self._diagnostics(self.validation.blocking) if self.validation.blocking else
                                       "No blocking items. Ready to save.")
        elif self.proposal is None:
            self._set_diagnostics((), empty="Scan a folder to see diagnostics.")
            self.blocking_text.setText("No proposal to save yet.")
        self._update_save()

    def _update_save(self):
        ready = self.validation is not None and self.validation.can_commit
        self.save_button.setEnabled(not self.busy and not self._pending_fields and ready)
        if self._pending_fields:
            self.blocking_text.setText("Apply or discard your pending edits before saving.")
        elif self.validation is None and self.proposal is not None:
            self.blocking_text.setText("Review validation is pending. Save will become available when checks pass.")
        elif self.validation is not None:
            self.blocking_text.setText(self._diagnostics(self.validation.blocking) if self.validation.blocking else
                                       "No blocking items. Ready to save.")

    def _set_stage(self, current):
        for number, stage in enumerate(self.stages, 1):
            stage.setProperty("active", number == current)
            stage.setProperty("complete", number < current)
            stage.style().unpolish(stage)
            stage.style().polish(stage)

    def save_run(self):
        if not self.busy and self.save_button.isEnabled() and self.proposal is not None:
            self._start("save")

    def finish_before_close(self):
        self._closing = True
        self.message.setText("Finishing the current operation before closing…")
        if self._worker is not None:
            self._worker.requestInterruption()
