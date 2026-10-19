"""Pure Library criteria and compact persisted-summary presentation."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LibraryCriteria:
    query: str = ""
    favorites_only: bool = False
    rating: str = "all"
    method: str | None = None
    tag: str | None = None
    sort: str = "newest"


def normalized(value):
    return value.strip().casefold()


def filter_options(runs, attribute):
    values = (tag for run in runs for tag in run.tags) if attribute == "tags" else (
        getattr(run, attribute) for run in runs)
    options = {}
    for value in sorted((v for v in values if v and v.strip()), key=lambda v: (normalized(v), v)):
        options.setdefault(normalized(value), value.strip())
    return tuple(options.items())


def visible_runs(runs, criteria):
    query = normalized(criteria.query)
    matches = []
    for run in runs:
        fields = (run.display_name, run.base_model, run.source_trainer, run.method, *run.tags)
        if query and not any(query in (value or "").casefold() for value in fields):
            continue
        if criteria.favorites_only and not run.favorite:
            continue
        if ((criteria.rating == "rated" and run.rating is None) or
                (criteria.rating == "unrated" and run.rating is not None)):
            continue
        if criteria.method is not None and normalized(run.method or "") != criteria.method:
            continue
        if criteria.tag is not None and not any(normalized(tag) == criteria.tag for tag in run.tags):
            continue
        matches.append(run)
    # Stable IDs always break ties ascending, including descending primary sorts.
    matches.sort(key=lambda run: run.id)
    if criteria.sort in {"newest", "oldest"}:
        matches.sort(key=lambda run: run.imported_at, reverse=criteria.sort == "newest")
    elif criteria.sort in {"name_asc", "name_desc"}:
        matches.sort(key=lambda run: run.display_name.casefold(), reverse=criteria.sort == "name_desc")
    elif criteria.sort == "rating_desc":
        matches.sort(key=lambda run: (run.rating is None, -(run.rating if run.rating is not None else 0)))
    return tuple(matches)


def parameter_value(run, key):
    values = [p.value for p in run.parameters if p.key == key]
    # Category-specific competing values do not select an arbitrary summary.
    return values[0] if values and all(type(v) is type(values[0]) and v == values[0] for v in values) else None


def summary_lines(run):
    trainer = {"huggingface_trainer": "Hugging Face Trainer", "unsloth": "Unsloth"}.get(
        run.source_trainer, run.source_trainer or "Unknown trainer")
    method = run.method or "Unknown method"
    if parameter_value(run, "qlora") is True and "qlora" not in method.casefold():
        method += " · QLoRA"
    metadata = f"{method} · {trainer} · Imported {run.imported_at.astimezone().strftime('%Y-%m-%d %H:%M')}"
    progress = []
    for key, caption in (("epochs", "Planned epochs"), ("observed_epoch", "Observed epoch"),
                         ("global_step", "Step"), ("max_steps", "Max steps")):
        value = parameter_value(run, key)
        if value is not None:
            progress.append(f"{caption} {value}")
    progress.append(f"Status {run.training_status.value}")
    return (metadata, " · ".join(progress))


def summary_text(run):
    metadata, progress = summary_lines(run)
    rating = f"{run.rating}/5" if run.rating is not None else "Unrated"
    return (f"{run.display_name}\nModel: {run.base_model or 'Unknown'}\n{metadata}\n{progress}\n"
            f"{'Favorite' if run.favorite else 'Not favorite'} · {rating}\n"
            f"Tags: {', '.join(run.tags) or 'No tags'}\nImported: {run.imported_at.isoformat()}")
