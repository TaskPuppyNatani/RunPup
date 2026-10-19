from dataclasses import replace
from datetime import datetime, timezone

import pytest

from runpup.domain.models import (
    Artifact, Confidence, Dataset, EntityType, Metric, MetricSummary, Parameter,
    Provenance, Run, Tag,
)
from runpup.ui.comparison import ComparisonStatus as S, compare_runs, load_comparison, value_status


@pytest.mark.parametrize("a,b,status", (
    ("model", "model", S.SAME), ("a", "b", S.DIFFERENT), (None, None, S.BOTH_UNKNOWN),
    (16, None, S.ONLY_A), (None, 16, S.ONLY_B), (False, False, S.SAME),
    (False, 0, S.DIFFERENT), (16, "16", S.DIFFERENT), (1, 1.0, S.SAME),
    (.000200000001, .0002, S.DIFFERENT), (frozenset({"a", "b"}), frozenset({"b", "a"}), S.SAME),
))
def test_value_status_uses_typed_values_without_display_rounding(a, b, status):
    assert value_status(a, b) is status


@pytest.fixture
def pair():
    moment = datetime(2026, 1, 1, tzinfo=timezone.utc)
    a = Run(id="a", display_name="Shared name", source_storage_root_id="root", source_relative_directory="a",
            imported_at=moment, base_model="model/a", favorite=True, tags=(Tag(name="good"), Tag(name="lora")))
    b = Run(id="b", display_name="Shared name", source_storage_root_id="root", source_relative_directory="b",
            imported_at=moment, base_model="model/b", rating=0, notes="Long notes " * 100,
            tags=(Tag(name="lora"), Tag(name="good")))
    pa = (Parameter(key="epochs", value=4), Parameter(key="learning_rate", value=.000200000001),
          Parameter(key="gradient_accumulation", value=4), Parameter(key="load_in_8bit", value=False),
          Parameter(key="lora_rank", value=16), Parameter(key="custom", value="a", category="scope-a"))
    pb = (Parameter(key="epochs", value=3), Parameter(key="learning_rate", value=.0002),
          Parameter(key="max_steps", value=160), Parameter(key="load_in_8bit", value=False),
          Parameter(key="lora_rank", value=16), Parameter(key="custom", value="b", category="scope-b"))
    evidence_a = (
        Provenance(target_entity=EntityType.PARAMETER, target_id=pa[1].id, target_field="value",
                   source_type="user", original_value=.0001, candidate_value=.0001, current_value=pa[1].value,
                   confidence=Confidence.EXACT, user_modified=True),
        Provenance(target_entity=EntityType.PARAMETER, target_id=pa[3].id, target_field="value",
                   source_type="derived", candidate_value=False, current_value=False, confidence=Confidence.DERIVED),
        Provenance(target_entity=EntityType.PARAMETER, target_id=pa[4].id, target_field="value",
                   source_type="user", original_value=32, candidate_value=32, current_value=16,
                   confidence=Confidence.EXACT, user_modified=True),
    )
    evidence_b = (Provenance(target_entity=EntityType.PARAMETER, target_id=pb[3].id, target_field="value",
                             source_type="file", source_reference="config.json#/load_in_8bit", candidate_value=False,
                             current_value=False, confidence=Confidence.EXACT),)
    metrics_a = (
        Metric(name="loss", context="train", step=10, unit="nats", summary=MetricSummary.VALUE_AT_STEP, value=.8),
        Metric(name="loss", context="validation", step=10, unit="nats", summary=MetricSummary.VALUE_AT_STEP, value=.7),
        Metric(name="runtime", context="train", unit="seconds", summary=MetricSummary.FINAL, value=100),
    )
    metrics_b = (
        replace(metrics_a[0], id="b-loss", value=.6),
        Metric(name="loss", context="train", step=20, unit="nats", summary=MetricSummary.VALUE_AT_STEP, value=.5),
        Metric(name="loss", context="train", step=10, unit="bits", summary=MetricSummary.VALUE_AT_STEP, value=.4),
        Metric(name="loss", context="train", step=10, unit="nats", summary=MetricSummary.FINAL, value=.3),
        replace(metrics_a[2], id="b-runtime"),
    )
    artifacts_a = (
        Artifact(run_id="a", storage_root_id="root", relative_path="adapter.safetensors", display_name="Adapter", size_bytes=64),
        Artifact(run_id="a", storage_root_id="root", relative_path="config.json", display_name="Config"),
        Artifact(run_id="a", storage_root_id="root", relative_path="checkpoint-10/model.bin", display_name="Checkpoint"),
    )
    artifacts_b = (
        replace(artifacts_a[0], id="b-adapter", run_id="b", size_bytes=128),
        replace(artifacts_a[1], id="b-config", run_id="b"),
        Artifact(run_id="b", storage_root_id="root", relative_path="checkpoint-20/model.bin", display_name="Checkpoint"),
    )
    return (replace(a, parameters=pa, metrics=metrics_a, artifacts=artifacts_a, provenance=evidence_a),
            replace(b, parameters=pb, metrics=metrics_b, artifacts=artifacts_b, provenance=evidence_b))


def row_map(comparison):
    return {row.key: row for row in comparison.rows}


def test_common_values_unknowns_and_current_human_edits(pair):
    rows = row_map(compare_runs(*pair))
    assert rows["run", "base_model"].status is S.DIFFERENT
    assert rows["run", "training_status"].status is S.BOTH_UNKNOWN
    assert rows["parameter", "gradient_accumulation", None].status is S.ONLY_A
    assert rows["parameter", "max_steps", None].status is S.ONLY_B
    assert rows["parameter", "load_in_8bit", None].status is S.SAME
    assert rows["parameter", "lora_rank", None].status is S.SAME
    lr = rows["parameter", "learning_rate", None]
    assert lr.a.value == pair[0].parameters[1].value and lr.a.evidence[0].candidate_value == .0001
    assert lr.status is S.DIFFERENT and lr.a.evidence[0].user_modified
    assert rows["parameter", "custom", "scope-a"].status is S.ONLY_A
    assert rows["parameter", "custom", "scope-b"].status is S.ONLY_B


def test_difference_order_filter_and_counts_are_deterministic(pair):
    result = compare_runs(*pair)
    priority = {S.DIFFERENT: 0, S.ONLY_A: 1, S.ONLY_B: 1, S.BOTH_UNKNOWN: 2, S.SAME: 3}
    for section in {row.section for row in result.rows}:
        statuses = [priority[row.status] for row in result.rows if row.section == section]
        assert statuses == sorted(statuses)
    assert all(row.status in {S.DIFFERENT, S.ONLY_A, S.ONLY_B} for row in result.visible_rows(True))
    assert result.visible_rows(False) == result.rows
    assert sum(result.counts.values()) == len(result.rows)
    reversed_a = replace(pair[0], parameters=tuple(reversed(pair[0].parameters)), metrics=tuple(reversed(pair[0].metrics)),
                         artifacts=tuple(reversed(pair[0].artifacts)))
    assert compare_runs(reversed_a, pair[1]).rows == result.rows


def test_swap_reverses_only_side_statuses_and_preserves_all_identities(pair):
    comparison = compare_runs(*pair)
    swapped = comparison.swapped()
    assert swapped.run_a is pair[1] and swapped.run_b is pair[0]
    flipped = {S.ONLY_A: S.ONLY_B, S.ONLY_B: S.ONLY_A}
    for key, before in row_map(comparison).items():
        after = row_map(swapped)[key]
        assert after.a == before.b and after.b == before.a
        assert after.status is flipped.get(before.status, before.status)


def test_metric_union_preserves_context_step_unit_and_summary(pair):
    rows = [r for r in compare_runs(*pair).rows if r.section == "Results / Metrics"]
    assert len(rows) == 6
    assert sorted(row.status.value for row in rows) == sorted([S.DIFFERENT, S.ONLY_A, S.ONLY_B, S.ONLY_B, S.ONLY_B, S.SAME])
    rows = {row.key: row for row in rows}
    assert rows["metric", "loss", "train", "value_at_step", 10, "nats"].status is S.DIFFERENT
    assert rows["metric", "loss", "validation", "value_at_step", 10, "nats"].status is S.ONLY_A
    assert rows["metric", "loss", "train", "final", 10, "nats"].status is S.ONLY_B


def test_null_and_empty_metric_units_remain_distinct(pair):
    metric = Metric(name="metric", value=1, unit=None)
    a, b = (replace(run, metrics=(replace(metric, unit=unit),)) for run, unit in zip(pair, (None, "")))
    rows = [r for r in compare_runs(a, b).rows if r.section == "Results / Metrics"]
    assert {row.status for row in rows} == {S.ONLY_A, S.ONLY_B} and len(rows) == 2


def test_cleared_metric_keeps_saved_run_level_evidence_on_unknown_side(pair):
    record = Provenance(target_entity=EntityType.RUN, target_id="a",
                        target_field='metric:["loss","train","value_at_step",10,"nats"]',
                        source_type="user", original_value=.8, candidate_value=.8, current_value=None,
                        user_modified=True, confidence=Confidence.EXACT)
    a = replace(pair[0], metrics=pair[0].metrics[1:], provenance=pair[0].provenance + (record,))
    row = row_map(compare_runs(a, pair[1]))["metric", "loss", "train", "value_at_step", 10, "nats"]
    assert row.status is S.ONLY_B and row.a.value is None
    assert row.a.evidence == (record,)


def test_artifacts_align_by_root_and_relative_reference_without_touching_files(pair):
    rows = {r.key: r for r in compare_runs(*pair).rows if r.section == "Artifacts"}
    assert len(rows) == 4
    assert rows["artifact", "root", "adapter.safetensors"].status is S.DIFFERENT
    assert rows["artifact", "root", "config.json"].status is S.SAME
    assert rows["artifact", "root", "checkpoint-10/model.bin"].status is S.ONLY_A
    assert rows["artifact", "root", "checkpoint-20/model.bin"].status is S.ONLY_B
    b = replace(pair[1], artifacts=(replace(pair[1].artifacts[1], storage_root_id="other-root"),))
    changed = row_map(compare_runs(pair[0], b))
    assert changed["artifact", "root", "config.json"].status is S.ONLY_A
    assert changed["artifact", "other-root", "config.json"].status is S.ONLY_B


def test_tags_human_metadata_and_notes_are_not_training_quality(pair):
    rows = row_map(compare_runs(*pair))
    assert rows["tags",].status is S.SAME
    assert rows["run", "rating"].status is S.ONLY_B and rows["run", "rating"].b.value == 0
    assert rows["run", "favorite"].status is S.DIFFERENT
    assert rows["run", "notes"].a.value == "" and rows["run", "notes"].b.value == pair[1].notes
    assert rows["run", "notes"].status is S.DIFFERENT


def test_single_dataset_fields_and_unnamed_facts_compare_without_name_guesses(pair):
    a, b = (replace(run, datasets=(Dataset(name=name, example_count=count),))
            for run, name, count in zip(pair, ("Train set", "Other set"), (12, None)))
    rows = row_map(compare_runs(a, b))
    assert rows["dataset", "dataset_identity"].status is S.DIFFERENT
    assert rows["dataset", "dataset_example_count"].status is S.ONLY_A
    unknown = row_map(compare_runs(*pair))
    assert unknown["dataset", "dataset_identity"].status is S.BOTH_UNKNOWN
    a = replace(pair[0], parameters=pair[0].parameters + (Parameter(key="dataset_example_count", value=12),))
    assert row_map(compare_runs(a, pair[1]))["dataset", "dataset_example_count"].status is S.ONLY_A


def test_multiple_datasets_use_shared_ids_and_ignore_link_order(pair):
    one, two = Dataset(id="one", name="Shared"), Dataset(id="two", name="Also shared")
    a, b = replace(pair[0], datasets=(one, two)), replace(pair[1], datasets=(two, one))
    rows = [r for r in compare_runs(a, b).rows if r.section == "Dataset"]
    assert len(rows) == 16 and all(row.status in {S.SAME, S.BOTH_UNKNOWN} for row in rows)


def test_dataset_facts_compare_once_across_structured_and_parameter_storage(pair):
    count = Parameter(key="dataset_example_count", value=12)
    partial_hash = Parameter(key="dataset_hash", value="partial-hash")
    a = replace(pair[0], parameters=pair[0].parameters + (count, partial_hash))
    b = replace(pair[1], datasets=(Dataset(name="Named", example_count=12),),
                parameters=pair[1].parameters + (replace(partial_hash, id="b-hash"),))
    rows = row_map(compare_runs(a, b))
    assert rows["dataset", "dataset_example_count"].status is S.SAME
    assert rows["dataset", "dataset_hash"].status is S.SAME
    assert ("parameter", "dataset_example_count", None) not in rows
    assert ("parameter", "dataset_hash", None) not in rows


def test_saved_pair_read_uses_only_two_selected_aggregate_reads(pair):
    class Repository:
        def __init__(self):
            self.ids = []

        def get_run(self, run_id):
            self.ids.append(run_id)
            return next((run for run in pair if run.id == run_id), None)

    repository = Repository()
    assert load_comparison(repository, "a", "b").run_a is pair[0]
    assert repository.ids == ["a", "b"]
    with pytest.raises(ValueError, match="different"):
        load_comparison(repository, "a", "a")
    with pytest.raises(KeyError, match="no longer"):
        load_comparison(repository, "a", "missing")
