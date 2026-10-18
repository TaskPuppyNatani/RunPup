from dataclasses import replace

import pytest

from runpup.domain.models import ImportDiagnostic, Severity
from runpup.ui.diagnostics import format_diagnostics, group_diagnostics


def unsafe_warning(path, **kwargs):
    return ImportDiagnostic(
        kind="unsafe_format", source_reference=path,
        message="training_args.bin was discovered but not deserialized.",
        consequence="Training arguments require safe JSON metadata.", **kwargs,
    )


def test_repeated_warnings_group_all_sources_naturally_without_mutation():
    paths = ["training_args.bin", *(f"checkpoint-{step}/training_args.bin" for step in (30, 60, 90, 120, 150, 160))]
    originals = tuple(unsafe_warning(path, artifact_id=f"artifact-{index}") for index, path in enumerate(reversed(paths)))
    artifacts = {item.artifact_id: item.source_reference for item in originals}
    groups = group_diagnostics(originals)
    assert len(groups) == 1 and groups[0].occurrences == originals
    assert all(record is original for record, original in zip(groups[0].occurrences, originals))
    text = format_diagnostics(originals, artifact_paths=artifacts)
    assert text == (
        "WARNING — 7 occurrences\ntraining_args.bin was discovered but not deserialized.\n"
        "Training arguments require safe JSON metadata.\n\nSources:\n"
        + "\n".join("- " + path for path in paths)
    )
    assert format_diagnostics(tuple(reversed(originals)), artifact_paths=artifacts) == text
    assert text.count("was discovered but not deserialized.") == 1
    assert text.count("\n- ") == 7
    assert tuple(item.source_reference for item in originals) == tuple(reversed(paths))


@pytest.mark.parametrize("change", [
    {"severity": Severity.INFO}, {"kind": "different_code"},
    {"message": "Dataset identity is unavailable."},
    {"consequence": "Another remediation."}, {"affected_field": "dataset_identity"},
])
def test_different_diagnostic_semantics_do_not_merge(change):
    original = unsafe_warning("training_args.bin", affected_field="learning_rate")
    other = replace(original, source_reference="checkpoint-30/training_args.bin", **change)
    assert len(group_diagnostics((original, other))) == 2
    assert "occurrences" not in format_diagnostics((original, other))


def test_grouped_artifact_paths_survive_without_duplicating_sources():
    items = (unsafe_warning("first.bin", artifact_id="first"),
             unsafe_warning("second.bin", artifact_id="second"))
    text = format_diagnostics(items, artifact_paths={"first": "first.bin", "second": "extra/artifact.bin"})
    assert "WARNING — 2 occurrences" in text
    for path in ("first.bin", "second.bin", "extra/artifact.bin"):
        assert text.count(path) == 1
    assert "Artifacts:\n- extra/artifact.bin" in text


def test_duplicate_source_records_retain_occurrences_without_duplicate_paragraphs():
    items = (unsafe_warning("training_args.bin"), unsafe_warning("training_args.bin"))
    group, = group_diagnostics(items)
    assert len(group.occurrences) == 2
    text = format_diagnostics(items)
    assert "2 occurrences" in text and text.count("\n- training_args.bin") == 1
