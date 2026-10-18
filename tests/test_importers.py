from dataclasses import replace
from datetime import datetime, timezone
import builtins
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from runpup.domain.models import Confidence, MetricSummary, RootKind, StorageRoot
from runpup.domain.proposals import FieldState
from runpup.importers import HFTrainerImporter
from runpup.importers.hf_trainer import _Scan
from runpup.services.paths import PathResolver


def write_json(directory, name, value):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "run outputs café"
    directory.mkdir()
    root = StorageRoot(name="Training outputs", location=str(tmp_path))
    return root, directory, directory.name


def minimal(directory):
    write_json(directory, "trainer_state.json", {"global_step": 4, "log_history": []})


def snapshot(directory):
    return {p.relative_to(directory).as_posix(): (
        p.read_bytes(), p.stat().st_size, p.stat().st_mtime_ns
    ) for p in directory.rglob("*") if p.is_file() and not p.is_symlink()}


def test_minimal_trainer_detection_unknowns_and_timestamp_semantics(source):
    root, directory, reference = source
    minimal(directory)
    os.utime(directory / "trainer_state.json", (946684800, 946684800))
    importer = HFTrainerImporter()
    detection = importer.detect(root, reference)
    proposal = importer.scan(root, reference)
    assert detection.recognized
    assert detection.evidence == (reference + "/trainer_state.json",)
    assert proposal.detected_importer == "hf_trainer"
    assert proposal.source_storage_root == root
    assert proposal.source_relative_directory == reference
    assert not proposal.has_blockers
    for key in ("base_model", "training_started_at", "training_finished_at", "training_status",
                "dataset_identity", "learning_rate"):
        assert proposal.field(key).state is FieldState.UNKNOWN
        assert proposal.field(key).value is None
    assert proposal.scanned_at.year != 2000
    assert proposal.scanned_at.utcoffset() is not None
    assert proposal.field("display_name").candidates[0].confidence is Confidence.INFERRED


def test_unsloth_compatible_adapter_and_quantization_extraction(source):
    root, directory, reference = source
    write_json(directory, "adapter_config.json", {
        "peft_type": "LORA", "base_model_name_or_path": "org/base", "r": 16,
        "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["v_proj", "q_proj"],
        "bias": "none",
    })
    write_json(directory, "config.json", {
        "_name_or_path": "org/base", "unsloth_version": "2026.1",
        "quantization_config": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_use_double_quant": True},
    })
    (directory / "adapter_model.safetensors").write_bytes(b"not inspected")
    proposal = HFTrainerImporter().scan(root, reference)
    assert proposal.source_trainer == "unsloth"
    assert not proposal.has_blockers
    assert proposal.field("base_model").state is FieldState.DETECTED
    assert len(proposal.field("base_model").candidates) == 2
    assert proposal.field("base_model").value == "org/base"
    assert proposal.field("method").value == "LoRA"
    assert proposal.field("lora_rank").value == 16
    assert proposal.field("lora_alpha").value == 32
    assert proposal.field("lora_dropout").value == 0.05
    assert json.loads(proposal.field("lora_target_modules").value) == ["q_proj", "v_proj"]
    assert proposal.field("lora_bias").value == "none"
    assert proposal.field("load_in_4bit").value is True
    assert proposal.field("quantization_type").value == "nf4"
    assert proposal.field("quantization_compute_dtype").candidates[0].source_reference.endswith(
        "config.json#/quantization_config/bnb_4bit_compute_dtype")
    assert proposal.field("qlora").state is FieldState.UNKNOWN
    write_json(directory, "training_args.json", {"training_method": "QLORA"})
    explicit = HFTrainerImporter().scan(root, reference)
    assert explicit.field("method").state is FieldState.DETECTED
    assert explicit.field("method").value == "LoRA"
    assert explicit.field("qlora").value is True
    assert any(p.original_value == "QLORA" for p in explicit.field("method").candidates)


def test_generic_model_export_and_unrelated_json_are_not_training_signatures(source):
    root, directory, reference = source
    write_json(directory, "notes.json", {"learning_rate": 0.1})
    write_json(directory, "config.json", {"model_type": "llama"})
    (directory / "model.safetensors").write_bytes(b"model")
    proposal = HFTrainerImporter().scan(root, reference)
    assert not HFTrainerImporter().detect(root, reference).recognized
    assert proposal.detected_importer is None
    assert proposal.has_blockers
    assert [d.kind for d in proposal.blocking_diagnostics] == ["unsupported_source"]


def test_unsloth_markers_preserve_exact_and_derived_quantization_evidence(source):
    root, directory, reference = source
    write_json(directory, "adapter_config.json", {
        "peft_type": "LORA", "unsloth_load_in_4bit": True, "unsloth_training_method": "qlora",
    })
    (directory / "adapter_model.safetensors").write_bytes(b"not inspected")
    proposal = HFTrainerImporter().scan(root, reference)
    for key, source_key, original in (("load_in_4bit", "unsloth_load_in_4bit", True),
                                      ("qlora", "unsloth_training_method", "qlora")):
        field = proposal.field(key)
        assert field.state is FieldState.DETECTED and field.value is True
        record, = field.candidates
        assert record.original_value == original
        assert record.source_type == "adapter_config"
        assert record.source_reference == f"{reference}/adapter_config.json#/{source_key}"
        assert record.confidence is Confidence.EXACT
    eight_bit = proposal.field("load_in_8bit")
    assert eight_bit.state is FieldState.DETECTED and eight_bit.value is False
    derived, = eight_bit.candidates
    assert derived.source_type == "derived" and derived.confidence is Confidence.DERIVED
    assert json.loads(derived.original_value) == {"load_in_4bit": True}
    assert derived.source_reference == proposal.field("load_in_4bit").candidates[0].source_reference
    for key in ("quantization_type", "quantization_compute_dtype", "double_quantization"):
        assert proposal.field(key).state is FieldState.UNKNOWN
    edited = proposal.with_user_edit("qlora", False)
    assert edited.field("qlora").value is False
    assert all(record in edited.provenance for record in proposal.provenance)


def test_plain_lora_does_not_guess_unsloth_quantization_flags(source):
    root, directory, reference = source
    write_json(directory, "adapter_config.json", {"peft_type": "LORA"})
    (directory / "adapter_model.safetensors").write_bytes(b"not inspected")
    proposal = HFTrainerImporter().scan(root, reference)
    for key in ("load_in_4bit", "qlora", "load_in_8bit"):
        assert proposal.field(key).state is FieldState.UNKNOWN
        assert not proposal.field(key).candidates


def test_unsloth_conflicts_retain_explicit_and_derived_candidates(source):
    root, directory, reference = source
    write_json(directory, "adapter_config.json", {
        "peft_type": "LORA", "unsloth_load_in_4bit": True, "unsloth_training_method": "qlora",
    })
    (directory / "adapter_model.safetensors").write_bytes(b"not inspected")
    checkpoint = "checkpoint-4/adapter_config.json"
    write_json(directory, checkpoint, {
        "unsloth_load_in_4bit": False, "unsloth_training_method": "lora",
    })
    proposal = HFTrainerImporter().scan(root, reference)
    for key in ("load_in_4bit", "qlora"):
        field = proposal.field(key)
        assert field.state is FieldState.CONFLICT
        assert {record.candidate_value for record in field.candidates} == {True, False}
        assert all(record.confidence is Confidence.EXACT for record in field.candidates)
        assert len({record.source_reference for record in field.candidates}) == 2
    # A conflicting premise cannot establish the opposite loading mode.
    assert proposal.field("load_in_8bit").state is FieldState.UNKNOWN
    write_json(directory, checkpoint, {"unsloth_training_method": "lora", "load_in_8bit": True})
    proposal = HFTrainerImporter().scan(root, reference)
    eight_bit = proposal.field("load_in_8bit")
    assert eight_bit.state is FieldState.CONFLICT
    assert {(record.candidate_value, record.confidence, record.source_type)
            for record in eight_bit.candidates} == {
                (True, Confidence.EXACT, "adapter_config"), (False, Confidence.DERIVED, "derived"),
            }
    edited = proposal.with_user_edit("load_in_8bit", False)
    assert edited.field("load_in_8bit").candidates == eight_bit.candidates


def test_safe_training_arguments_with_payload_are_a_supported_signature(source):
    root, directory, reference = source
    write_json(directory, "training_args.json", {"learning_rate": 0.0002, "num_train_epochs": 3})
    (directory / "model.safetensors").write_bytes(b"model")
    proposal = HFTrainerImporter().scan(root, reference)
    assert proposal.detected_importer == "hf_trainer"
    assert proposal.field("learning_rate").value == 0.0002
    assert proposal.field("epochs").value == 3


def test_adapter_export_does_not_invent_a_specific_source_trainer(source):
    root, directory, reference = source
    write_json(directory, "adapter_config.json", {"peft_type": "LORA", "r": 8})
    (directory / "adapter_model.safetensors").write_bytes(b"payload")
    proposal = HFTrainerImporter().scan(root, reference)
    assert proposal.detected_importer == "hf_trainer"
    assert proposal.source_trainer is None
    assert proposal.field("source_trainer").state is FieldState.UNKNOWN


def test_material_conflicts_and_manual_edits_preserve_all_candidates(source):
    root, directory, reference = source
    minimal(directory)
    write_json(directory, "adapter_config.json", {
        "peft_type": "LORA", "base_model_name_or_path": "base/a", "r": 8, "lora_alpha": 16,
    })
    write_json(directory, "config.json", {"_name_or_path": "base/b", "max_seq_length": 4096})
    write_json(directory, "training_args.json", {
        "model_name_or_path": "base/c", "training_method": "FULL", "learning_rate": 0.001,
        "lora_r": 16, "lora_alpha": 32, "max_seq_length": 2048,
        "dataset_name": "dataset/a", "dataset_hash": "aaa",
    })
    write_json(directory, "checkpoint-4/training_args.json", {
        "learning_rate": 0.002, "dataset_name": "dataset/b", "dataset_hash": "bbb",
    })
    proposal = HFTrainerImporter().scan(root, reference)
    conflicts = {"base_model", "method", "learning_rate", "lora_rank", "lora_alpha",
                 "max_sequence_length", "dataset_identity", "dataset_hash"}
    assert set(proposal.blocking_fields) == conflicts
    assert proposal.has_blockers
    for key in conflicts:
        assert proposal.field(key).state is FieldState.CONFLICT
        assert proposal.field(key).value is None
    evidence = proposal.provenance
    edited = proposal
    for key in sorted(conflicts):
        edited = edited.with_user_edit(key, proposal.field(key).candidates[0].candidate_value)
    assert not edited.has_blockers
    assert all(edited.field(key).state is FieldState.USER_EDITED for key in conflicts)
    assert all(record in edited.provenance for record in evidence)
    assert proposal.field("base_model").state is FieldState.CONFLICT
    edited_again = edited.with_user_edit("base_model", "manual/model")
    assert edited_again.field("base_model").value == "manual/model"
    assert len(edited_again.field("base_model").edits) == 2
    assert all(item.user_modified and item.source_type == "manual"
               for item in edited_again.field("base_model").edits)
    assert edited_again.field("base_model").candidates == proposal.field("base_model").candidates
    unknown_edit = proposal.with_user_edit("seed", None)
    assert unknown_edit.field("seed").state is FieldState.USER_EDITED


def test_training_parameters_dataset_and_explicit_history_are_preserved(source):
    root, directory, reference = source
    minimal(directory)
    write_json(directory, "training_args.json", {
        "run_name": "Useful run", "num_train_epochs": 2.5, "learning_rate": 0.0001,
        "per_device_train_batch_size": 2, "gradient_accumulation_steps": 4, "world_size": 3,
        "max_seq_length": 2048, "optim": "adamw_torch", "lr_scheduler_type": "cosine",
        "warmup_steps": 10, "warmup_ratio": 0.1, "seed": 42, "bf16": True,
        "training_started_at": "2025-01-01T01:00:00+01:00",
        "training_finished_at": "2025-01-01T02:00:00+01:00", "training_status": "completed",
        "dataset": {"name": "org/data", "reference": "org/data@v1", "hash": "abc",
                    "hash_algorithm": "sha256", "example_count": 100, "split": "train",
                    "splits": {"custom/split": {"num_examples": 12}}},
    })
    write_json(directory, "dataset_info.json", {"dataset_name": "org/data", "splits": {
        "train": {"num_examples": 100}, "validation": {"num_examples": 20},
    }})
    proposal = HFTrainerImporter().scan(root, reference)
    expected = {"display_name": "Useful run", "epochs": 2.5, "per_device_batch_size": 2,
                "gradient_accumulation": 4, "parallelism": 3, "effective_batch_size": 24,
                "max_sequence_length": 2048, "optimizer": "adamw_torch", "scheduler": "cosine",
                "warmup_steps": 10, "warmup_ratio": 0.1, "seed": 42, "precision": "bfloat16",
                "dataset_identity": "org/data", "dataset_reference": "org/data@v1",
                "dataset_hash": "abc", "dataset_hash_algorithm": "sha256",
                "dataset_example_count": 100, "dataset_split": "train",
                "dataset_split_counts.validation": 20, "training_status": "completed"}
    for key, value in expected.items():
        assert proposal.field(key).value == value, key
    assert proposal.field("effective_batch_size").candidates[0].confidence is Confidence.DERIVED
    assert any(p.source_reference.endswith("#/dataset/name")
               for p in proposal.field("dataset_identity").candidates)
    assert proposal.field("dataset_split_counts.custom/split").value == 12
    assert proposal.field("dataset_split_counts.custom/split").candidates[0].source_reference.endswith(
        "#/dataset/splits/custom~1split/num_examples")
    assert datetime.fromisoformat(proposal.field("training_started_at").value) == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert proposal.scanned_at > datetime.fromisoformat(proposal.field("training_finished_at").value)


def test_effective_batch_does_not_assume_single_process(source):
    root, directory, reference = source
    minimal(directory)
    write_json(directory, "training_args.json", {
        "per_device_train_batch_size": 2, "gradient_accumulation_steps": 4,
    })
    proposal = HFTrainerImporter().scan(root, reference)
    assert proposal.field("parallelism").state is FieldState.UNKNOWN
    assert proposal.field("effective_batch_size").state is FieldState.UNKNOWN


def test_latest_metrics_preserve_context_step_unit_and_competing_values(source):
    root, directory, reference = source
    write_json(directory, "trainer_state.json", {"global_step": 20, "max_steps": 20, "log_history": [
        {"loss": 1.2, "step": 10}, {"eval_loss": 0.9, "step": 10},
        {"loss": 0.8, "step": 20}, {"loss": 0.7, "step": 20},
        {"eval_loss": 0.6, "eval_runtime": 2.5, "step": 20},
        {"loss": 1.1, "step": 5},  # Array order is not a completion signal.
        {"train_loss": 0.95, "train_runtime": 10.0, "step": 20},
    ]})
    proposal = HFTrainerImporter().scan(root, reference)
    losses = [m for m in proposal.metrics if m.name == "loss"]
    train = [m for m in losses if m.context == "train" and m.summary is MetricSummary.VALUE_AT_STEP]
    validation = [m for m in losses if m.context == "validation"]
    assert [m.value for m in train] == [0.8, 0.7]
    assert [(m.value, m.step) for m in validation] == [(0.6, 20)]
    assert all(m.step == 20 for m in proposal.metrics)
    assert all(m.summary is not MetricSummary.FINAL for m in proposal.metrics)
    assert any(m.unit == "seconds" and m.context == "validation" for m in proposal.metrics)
    assert proposal.field("training_status").state is FieldState.UNKNOWN
    metric_fields = [f for f in proposal.fields if f.key.startswith("metric:")]
    conflict = next(f for f in metric_fields if f.state is FieldState.CONFLICT)
    assert [p.candidate_value for p in conflict.candidates] == [0.8, 0.7]
    assert all(p.target_id in {m.id for m in proposal.metrics} for f in metric_fields for p in f.candidates)
    assert not proposal.has_blockers


def test_trainer_state_facts_and_scheduler_observations_do_not_guess_configuration(source):
    root, directory, reference = source
    write_json(directory, "trainer_state.json", {
        "num_train_epochs": 4, "epoch": 4.0, "global_step": 160, "max_steps": 160,
        "log_history": [
            {"loss": 0.8, "learning_rate": 0.00001, "grad_norm": 0.2,
             "num_input_tokens_seen": 1000, "epoch": 3.9, "step": 150},
            {"loss": 0.7, "learning_rate": 0.0, "grad_norm": 0.1,
             "num_input_tokens_seen": 1200, "epoch": 4.0, "step": 160},
            {"eval_loss": 0.5, "eval_runtime": 2.0, "eval_samples_per_second": 4.0,
             "train_runtime": 20.0, "train_tokens_per_second": 60.0, "epoch": 4.0, "step": 160},
        ],
    })
    (directory / "training_args.bin").write_bytes(b"never deserialize")
    before = snapshot(directory)
    proposal = HFTrainerImporter().scan(root, reference)
    for key, value in {"epochs": 4, "observed_epoch": 4.0, "global_step": 160, "max_steps": 160}.items():
        field = proposal.field(key)
        assert field.value == value and field.state is FieldState.DETECTED
        assert field.candidates[0].confidence is Confidence.EXACT
        assert "trainer_state.json#/" in field.candidates[0].source_reference
    for key in ("learning_rate", "per_device_batch_size", "gradient_accumulation", "max_sequence_length",
                "optimizer", "scheduler", "warmup_steps", "seed", "precision", "training_status"):
        assert proposal.field(key).state is FieldState.UNKNOWN
    rates = [m for m in proposal.metrics if m.name == "learning_rate"]
    assert [(m.value, m.context, m.summary, m.step) for m in rates] == [
        (0.0, "train", MetricSummary.VALUE_AT_STEP, 160)]
    assert {(m.context, m.value) for m in proposal.metrics if m.name == "loss"} == {("train", 0.7), ("validation", 0.5)}
    assert any(m.name == "num_input_tokens_seen" and m.value == 1200 and m.unit == "tokens" for m in proposal.metrics)
    assert any(m.name == "epoch" and m.value == 4.0 and m.step == 160 for m in proposal.metrics)
    assert any(m.name == "tokens_per_second" and m.unit == "tokens/second" for m in proposal.metrics)
    assert any(d.kind == "unsafe_format" for d in proposal.diagnostics)
    assert snapshot(directory) == before


def test_checkpoint_state_selection_uses_reported_steps_and_retains_tied_disagreements(source):
    root, directory, reference = source
    for name, step, epoch, loss in (("checkpoint-999", 30, 0.75, 2.0),
                                    ("checkpoint-9", 160, 4.0, 0.7),
                                    ("checkpoint-other", 160, 3.9, 0.9)):
        write_json(directory, f"{name}/trainer_state.json", {
            "global_step": step, "epoch": epoch, "max_steps": 160, "num_train_epochs": 4,
            "log_history": [{"loss": loss, "step": step}],
        })
    write_json(directory, "training_args.json", {"num_train_epochs": 3.5})
    first = HFTrainerImporter().scan(root, reference)
    second = HFTrainerImporter().scan(root, reference)
    assert first.field("global_step").value == 160
    assert first.field("observed_epoch").state is FieldState.CONFLICT
    assert first.field("epochs").state is FieldState.CONFLICT
    assert {r.candidate_value for r in first.field("epochs").candidates} == {3.5, 4}
    assert {(m.value, m.step) for m in first.metrics if m.name == "loss"} == {(0.7, 160), (0.9, 160)}
    assert [(m.value, m.step) for m in first.metrics] == [(m.value, m.step) for m in second.metrics]
    assert any(d.kind == "trainer_state_selected" for d in first.diagnostics)


def test_invalid_trainer_state_facts_are_not_coerced(source):
    root, directory, reference = source
    write_json(directory, "trainer_state.json", {
        "num_train_epochs": True, "epoch": -1, "global_step": 3.5, "max_steps": -2, "log_history": [],
    })
    proposal = HFTrainerImporter().scan(root, reference)
    for key in ("epochs", "observed_epoch", "global_step", "max_steps"):
        assert proposal.field(key).state is FieldState.UNKNOWN
        assert any(d.affected_field == key for d in proposal.diagnostics)


def test_explicit_best_metric_semantics_do_not_claim_completion(source):
    root, directory, reference = source
    write_json(directory, "trainer_state.json", {
        "log_history": [], "best_metric": 0.45, "best_global_step": 6,
    })
    write_json(directory, "training_args.json", {
        "metric_for_best_model": "eval_loss", "greater_is_better": False,
    })
    proposal = HFTrainerImporter().scan(root, reference)
    assert [(m.name, m.context, m.summary, m.step, m.value) for m in proposal.metrics] == [
        ("loss", "validation", MetricSummary.MINIMUM, 6, 0.45)]
    assert proposal.field("training_status").value is None


@pytest.mark.parametrize("name", [None, "eval_"])
def test_unnamed_best_metric_preserves_value_when_arguments_exist(source, name):
    root, directory, reference = source
    write_json(directory, "trainer_state.json", {"log_history": [], "best_metric": 0.45})
    write_json(directory, "training_args.json", {} if name is None else {"metric_for_best_model": name})
    proposal = HFTrainerImporter().scan(root, reference)
    assert proposal.field("best_metric_unidentified").value == 0.45
    assert proposal.field("best_metric_unidentified").candidates[0].source_reference.endswith("#/best_metric")
    assert proposal.metrics == ()
    suffix = "#/best_metric" if name is None else "#/metric_for_best_model"
    assert any(d.source_reference.endswith(suffix) for d in proposal.diagnostics if d.source_reference)


@pytest.mark.parametrize("bad_json", [
    "{", "[]", '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"r":1,"r":2}',
])
def test_optional_malformed_metadata_is_recoverable_and_retained(source, bad_json):
    root, directory, reference = source
    minimal(directory)
    (directory / "adapter_config.json").write_text(bad_json)
    proposal = HFTrainerImporter().scan(root, reference)
    assert proposal.detected_importer == "hf_trainer"
    assert not proposal.has_blockers
    diagnostic = next(d for d in proposal.diagnostics if d.kind == "malformed_metadata")
    assert diagnostic.source_reference.endswith("/adapter_config.json")
    assert diagnostic.consequence
    assert any(a.display_name == "adapter_config.json" for a in proposal.artifacts)


def test_metadata_size_and_metric_limits_are_explicit(source):
    root, directory, reference = source
    minimal(directory)
    (directory / "adapter_config.json").write_bytes(b" " * 1000)
    proposal = HFTrainerImporter(max_json_bytes=128).scan(root, reference)
    assert not proposal.has_blockers
    assert any(d.kind == "malformed_metadata" for d in proposal.diagnostics)
    limited = HFTrainerImporter(max_entries=1).scan(root, reference)
    assert limited.has_blockers
    assert any(d.kind == "scan_limit" for d in limited.blocking_diagnostics)
    total_limit = HFTrainerImporter(max_total_json_bytes=20).scan(root, reference)
    assert any(d.kind == "metadata_limit" for d in total_limit.blocking_diagnostics)
    with (directory / "config.json").open("wb") as oversized:
        oversized.truncate(32 * 1024**2)
    oversized_optional = HFTrainerImporter().scan(root, reference)
    assert not oversized_optional.has_blockers
    assert any(d.kind == "malformed_metadata" and d.source_reference.endswith("config.json")
               for d in oversized_optional.diagnostics)
    write_json(directory, "trainer_state.json", {"log_history": [
        {**{f"eval_metric_{index}": index for index in range(260)}, "step": 1},
    ]})
    metrics_limited = HFTrainerImporter().scan(root, reference)
    assert len(metrics_limited.metrics) == 256
    assert any(d.kind == "metric_limit" for d in metrics_limited.blocking_diagnostics)
    write_json(directory, "trainer_state.json", {"log_history": [None] * 1100})
    diagnostics_limited = HFTrainerImporter().scan(root, reference)
    assert len(diagnostics_limited.diagnostics) == 1001
    assert any(d.kind == "diagnostic_limit" for d in diagnostics_limited.blocking_diagnostics)


def test_unsafe_formats_scripts_and_model_payloads_are_never_opened(source, monkeypatch):
    root, directory, reference = source
    minimal(directory)
    write_json(directory, "adapter_config.json", {
        "peft_type": "LORA", "unsloth_load_in_4bit": True, "unsloth_training_method": "qlora",
    })
    (directory / "training_args.bin").write_bytes(b"cos\nsystem\n(S'false'\ntR.")
    (directory / "training.py").write_text("raise AssertionError('must never execute')")
    for name in ("adapter_model.safetensors", "model.gguf"):
        with (directory / name).open("wb") as payload:
            payload.truncate(8 * 1024**3)
    (directory / "tokenizer.json").write_text("{}")
    write_json(directory, "checkpoint-4/trainer_state.json", {"global_step": 4, "log_history": []})
    (directory / "checkpoint-4/model.safetensors").write_bytes(b"payload")
    original_open, original_import = os.open, builtins.__import__
    original_builtin_open, original_io_open = builtins.open, io.open

    def guarded_open(path, *args, **kwargs):
        if str(path).endswith((".bin", ".py", ".safetensors", ".gguf")):
            raise AssertionError("Payload must not be opened")
        return original_open(path, *args, **kwargs)

    def no_deserialize(*args, **kwargs):
        raise AssertionError("Unsafe deserialization/hashing must not occur")

    def guarded_file_open(original):
        def checked(path, *args, **kwargs):
            if str(path).endswith((".bin", ".py", ".safetensors", ".gguf")):
                raise AssertionError("Payload must not be opened")
            return original(path, *args, **kwargs)
        return checked

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("Torch must not be imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)
    monkeypatch.setattr(builtins, "open", guarded_file_open(original_builtin_open))
    monkeypatch.setattr(io, "open", guarded_file_open(original_io_open))
    monkeypatch.setattr(pickle, "load", no_deserialize)
    monkeypatch.setattr(pickle, "loads", no_deserialize)
    monkeypatch.setattr(hashlib, "sha256", no_deserialize)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    proposal = HFTrainerImporter().scan(root, reference)
    assert not proposal.has_blockers
    assert proposal.field("load_in_4bit").value is True
    assert proposal.field("qlora").value is True
    assert proposal.field("load_in_8bit").value is False
    assert {a.category for a in proposal.artifacts} >= {"metadata", "adapter", "model", "gguf", "checkpoint", "tokenizer"}
    assert next(a for a in proposal.artifacts if a.display_name == "model.gguf").size_bytes == 8 * 1024**3
    assert all(a.hash_value is None and a.hash_algorithm is None for a in proposal.artifacts)
    assert not any(a.display_name.endswith(".py") for a in proposal.artifacts)
    assert any(d.kind == "unsafe_format" and d.artifact_id for d in proposal.diagnostics)


def test_source_files_and_layout_are_unchanged(source):
    root, directory, reference = source
    minimal(directory)
    write_json(directory, "training_args.json", {"learning_rate": 0.001, "num_train_epochs": 2})
    (directory / "adapter_model.safetensors").write_bytes(b"source payload")
    (directory / "training_args.bin").write_bytes(b"opaque")
    before = snapshot(directory)
    HFTrainerImporter().detect(root, reference)
    HFTrainerImporter().scan(root, reference)
    assert snapshot(directory) == before
    assert sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*")) == sorted(before)


def test_special_metadata_file_is_not_opened(source):
    root, directory, reference = source
    minimal(directory)
    os.mkfifo(directory / "training_args.json")
    proposal = HFTrainerImporter().scan(root, reference)
    assert not proposal.has_blockers
    assert any(d.kind == "special_file_skipped" for d in proposal.diagnostics)


@pytest.mark.parametrize("reference", ["../outside", "/absolute", "C:/absolute", "a/../../outside", "a\\b"])
def test_selected_source_traversal_is_blocking(source, reference):
    root, _, _ = source
    proposal = HFTrainerImporter().scan(root, reference)
    assert proposal.has_blockers
    assert proposal.source_relative_directory is None
    assert proposal.blocking_diagnostics[0].kind == "unsafe_source"


def test_symlinks_outside_root_or_selected_source_are_not_followed(source, tmp_path):
    root, directory, reference = source
    minimal(directory)
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    write_json(outside, "training_args.json", {"learning_rate": 99})
    (directory / "training_args.json").symlink_to(outside / "training_args.json")
    (directory / "checkpoint-escape").symlink_to(outside, target_is_directory=True)
    # Even symlinks to a sibling within the StorageRoot are not traversed.
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    (directory / "checkpoint-sibling").symlink_to(sibling, target_is_directory=True)
    proposal = HFTrainerImporter().scan(root, reference)
    assert proposal.field("learning_rate").state is FieldState.UNKNOWN
    assert sum(d.kind == "symlink_skipped" for d in proposal.diagnostics) == 3
    assert not any("escape" in a.relative_path or "sibling" in a.relative_path for a in proposal.artifacts)
    assert not HFTrainerImporter().scan(root, reference + "/checkpoint-escape").detected_importer


def test_symlink_replacement_between_stat_and_open_is_not_read(source, tmp_path, monkeypatch):
    root, directory, reference = source
    minimal(directory)
    write_json(directory, "adapter_config.json", {"peft_type": "LORA", "r": 8})
    (directory / "adapter_model.safetensors").write_bytes(b"payload")
    external = tmp_path.parent / (tmp_path.name + "-secret.json")
    external.write_text('{"learning_rate":99}')
    original_open = os.open

    def swap(path, *args, **kwargs):
        if path == "trainer_state.json":
            target = directory / path
            target.unlink()
            target.symlink_to(external)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap)
    proposal = HFTrainerImporter().scan(root, reference)
    assert proposal.field("learning_rate").state is FieldState.UNKNOWN
    assert any(d.kind == "malformed_metadata" for d in proposal.diagnostics)


def test_portable_root_and_selected_source_symlink(source):
    root, directory, reference = source
    minimal(directory)
    portable = replace(root, location=".", kind=RootKind.PORTABLE_RELATIVE)
    proposal = HFTrainerImporter(PathResolver(portable_base=Path(root.location))).scan(portable, reference)
    assert proposal.detected_importer == "hf_trainer"
    assert all(a.relative_path.startswith(reference + "/") for a in proposal.artifacts)
    unavailable = HFTrainerImporter().scan(portable, reference)
    assert unavailable.has_blockers
    link = Path(root.location) / "selected-link"
    link.symlink_to(directory, target_is_directory=True)
    assert HFTrainerImporter().scan(root, "selected-link").has_blockers


def test_source_unreadable_and_disappearance_are_blocking(source, monkeypatch):
    root, directory, reference = source
    minimal(directory)
    original = os.scandir

    def denied(path):
        if isinstance(path, int):
            raise PermissionError("simulated unreadable selected directory")
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", denied)
        proposal = HFTrainerImporter().scan(root, reference)
    assert any(d.kind == "directory_unreadable" for d in proposal.blocking_diagnostics)
    original_walk = _Scan.walk

    def disappear(scan, *args, **kwargs):
        original_walk(scan, *args, **kwargs)
        directory.rename(directory.with_name("moved"))

    monkeypatch.setattr(_Scan, "walk", disappear)
    proposal = HFTrainerImporter().scan(root, reference)
    assert any(d.kind == "source_changed" for d in proposal.blocking_diagnostics)


def test_invalid_recognized_values_remain_diagnostics_not_crashes(source):
    root, directory, reference = source
    write_json(directory, "trainer_state.json", {"log_history": [
        None, {"loss": "bad", "step": 1}, {"loss": 10**400, "step": 2},
        {"eval_loss": 0.3, "step": -1},
    ], "best_metric": {"unsupported": "object"}})
    write_json(directory, "adapter_config.json", {
        "peft_type": ["LORA"], "r": True, "lora_alpha": -1,
        "unsloth_load_in_4bit": "true", "unsloth_training_method": True,
    })
    write_json(directory, "training_args.json", {"training_started_at": "2025-01-01", "learning_rate": "guess"})
    proposal = HFTrainerImporter().scan(root, reference)
    assert not proposal.has_blockers
    assert proposal.field("lora_rank").state is FieldState.UNKNOWN
    assert proposal.field("learning_rate").state is FieldState.UNKNOWN
    assert proposal.field("training_started_at").state is FieldState.UNKNOWN
    for key in ("load_in_4bit", "qlora", "load_in_8bit"):
        assert proposal.field(key).state is FieldState.UNKNOWN
    assert len([d for d in proposal.diagnostics if d.kind == "invalid_metadata"]) >= 6


def test_importer_and_proposals_do_not_import_qt_or_sqlite():
    result = subprocess.run([
        sys.executable, "-c",
        "import runpup.domain.proposals, runpup.importers, sys; "
        "assert 'sqlite3' not in sys.modules; "
        "assert not any(n.startswith(('PySide6', 'torch')) for n in sys.modules)",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
