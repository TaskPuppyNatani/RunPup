from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from runpup.domain.models import ImportDiagnostic, Parameter, Run, Tag, TrainingStatus
from runpup.persistence.library import RunSummary
from runpup.ui.library_view import LibraryCriteria, filter_options, summary_lines, summary_text, visible_runs


@pytest.fixture
def summaries():
    moment = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return (
        RunSummary(id="a", display_name="Alpha keeper", imported_at=moment, base_model="org/Model-One",
                   source_trainer="Unsloth", method="LoRA", favorite=True, rating=5, tags=("Good", "adapter")),
        RunSummary(id="b", display_name="beta", imported_at=moment + timedelta(days=1),
                   method="Full FT", rating=0, tags=("test",)),
        RunSummary(id="c", display_name="Zulu", imported_at=moment + timedelta(days=1), tags=("good",)),
        RunSummary(id="d", display_name="alpha keeper", imported_at=moment, method="lora", favorite=True,
                   rating=4, tags=("adapter",)),
    )


@pytest.mark.parametrize("query", (" ALPHA KEEP ", "model-one", "uNSLoTH", "LoRa", "good"))
def test_search_matches_intended_fields_case_insensitively(summaries, query):
    assert "a" in [run.id for run in visible_runs(summaries, LibraryCriteria(query=query))]


@pytest.mark.parametrize("criteria, expected", (
    (LibraryCriteria(query="  "), {"a", "b", "c", "d"}),
    (LibraryCriteria(favorites_only=True), {"a", "d"}),
    (LibraryCriteria(rating="rated"), {"a", "b", "d"}),
    (LibraryCriteria(rating="unrated"), {"c"}),
    (LibraryCriteria(method="lora"), {"a", "d"}),
    (LibraryCriteria(tag="good"), {"a", "c"}),
    (LibraryCriteria(query="model", favorites_only=True, rating="rated", method="lora", tag="good"), {"a"}),
    (LibraryCriteria(query="missing"), set()),
))
def test_filters_and_search_compose_without_treating_zero_as_unrated(summaries, criteria, expected):
    assert {run.id for run in visible_runs(summaries, criteria)} == expected


@pytest.mark.parametrize("sort, expected", (
    ("newest", ["b", "c", "a", "d"]), ("oldest", ["a", "d", "b", "c"]),
    ("name_asc", ["a", "d", "b", "c"]), ("name_desc", ["c", "b", "a", "d"]),
    ("rating_desc", ["a", "d", "b", "c"]),
))
def test_sorts_have_stable_identity_ties(summaries, sort, expected):
    assert [run.id for run in visible_runs(tuple(reversed(summaries)), LibraryCriteria(sort=sort))] == expected


def test_options_are_catalog_derived_normalized_and_deterministic(summaries):
    assert filter_options(summaries, "method") == (("full ft", "Full FT"), ("lora", "LoRA"))
    assert filter_options(summaries, "tags") == (("adapter", "adapter"), ("good", "Good"), ("test", "test"))


def test_progress_is_persisted_and_never_claims_completion(summaries):
    run = replace(summaries[0], parameters=tuple(Parameter(key=key, value=value) for key, value in (
        ("epochs", 4), ("observed_epoch", 4.0), ("global_step", 160), ("max_steps", 160), ("qlora", True))))
    metadata, progress = summary_lines(run)
    assert "QLoRA" in metadata
    assert "Planned epochs 4" in progress and "Observed epoch 4.0" in progress
    assert "Step 160" in progress and "Max steps 160" in progress and "Status unknown" in progress
    assert run.training_status is TrainingStatus.UNKNOWN
    competing = replace(run, parameters=run.parameters + (Parameter(key="epochs", value=8, category="other"),))
    assert "Planned epochs" not in summary_lines(competing)[1]
    assert "Favorite · 5/5" in summary_text(run)


def test_summary_read_is_three_batched_queries_and_excludes_heavy_relations(repository, storage_root, monkeypatch):
    tag = Tag(name="Findable")
    run = Run(display_name="Saved summary", source_storage_root_id=storage_root.id, source_relative_directory=".",
              notes="Secret notes", rating=0, favorite=True, tags=(tag,), diagnostics=(ImportDiagnostic(message="Secret warning"),),
              parameters=(Parameter(key="epochs", value=3), Parameter(key="qlora", value=True),
                          Parameter(key="large.unneeded", value="Secret parameter")))
    repository.save_run(run)
    statements = []
    original = repository.database.connection

    @contextmanager
    def tracked():
        with original() as connection:
            connection.set_trace_callback(statements.append)
            connection.set_authorizer(lambda action, table, *rest: sqlite3.SQLITE_DENY
                                     if action == sqlite3.SQLITE_READ and table in {
                                         "artifact", "provenance", "import_diagnostic", "metric", "storage_root", "dataset"
                                     } else sqlite3.SQLITE_OK)
            yield connection

    monkeypatch.setattr(repository.database, "connection", tracked)
    rows = repository.list_run_summaries()
    assert len([sql for sql in statements if sql.startswith("SELECT")]) == 3
    assert len(rows) == 1 and rows[0].id == run.id
    assert rows[0].rating == 0 and rows[0].favorite and rows[0].tags == ("Findable",)
    assert [(p.key, p.value) for p in rows[0].parameters] == [("epochs", 3), ("qlora", True)]
    for query in ("Secret warning", "Secret parameter", "Secret notes"):
        assert visible_runs(rows, LibraryCriteria(query=query)) == ()
    assert visible_runs(rows, LibraryCriteria(query="findable")) == tuple(rows)


def test_summary_query_count_is_constant_for_many_runs(repository, storage_root, monkeypatch):
    for index in range(120):
        repository.save_run(Run(display_name=f"Synthetic {index:03}", source_storage_root_id=storage_root.id,
                                source_relative_directory=".", parameters=(Parameter(key="global_step", value=index),)))
    statements = []
    original = repository.database.connection

    @contextmanager
    def tracked():
        with original() as connection:
            connection.set_trace_callback(statements.append)
            yield connection

    monkeypatch.setattr(repository.database, "connection", tracked)
    assert len(repository.list_run_summaries()) == 120
    assert len([sql for sql in statements if sql.startswith("SELECT")]) == 3
