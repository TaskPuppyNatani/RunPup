"""Presentation grouping only; original diagnostics are retained unchanged."""

from dataclasses import dataclass
import re

from runpup.domain.models import ImportDiagnostic


@dataclass(frozen=True, slots=True)
class DiagnosticGroup:
    occurrences: tuple[ImportDiagnostic, ...]


def group_diagnostics(items):
    groups = {}
    for item in items:
        identity = (item.severity, item.kind, item.message, item.consequence, item.affected_field)
        groups.setdefault(identity, []).append(item)
    return tuple(DiagnosticGroup(tuple(occurrences)) for occurrences in groups.values())


def _path_order(path):
    return (path.count("/"), tuple((1, int(part)) if part.isdecimal() else (0, part)
                                    for part in re.split(r"([0-9]+)", path)))


def format_diagnostics(items, *, artifact_paths=None, root=None):
    artifact_paths = artifact_paths or {}
    paragraphs = []
    for group in group_diagnostics(items):
        item = group.occurrences[0]
        count = len(group.occurrences)
        lines = [item.severity.value.upper() + (f" — {count} occurrences" if count > 1 else "")]
        if count == 1:
            for caption, value in (("Source", item.source_reference), ("Field", item.affected_field),
                                   ("Artifact", artifact_paths.get(item.artifact_id))):
                if value:
                    lines.append(f"{caption}: {value}")
        elif item.affected_field:
            lines.append(f"Field: {item.affected_field}")
        message = item.message.replace(root.id, root.name) if root is not None else item.message
        lines.append(message)
        if item.consequence:
            lines.append(item.consequence)
        if count > 1:
            sources = {record.source_reference for record in group.occurrences if record.source_reference}
            artifacts = {artifact_paths[record.artifact_id] for record in group.occurrences
                         if record.artifact_id in artifact_paths} - sources
            for caption, paths in (("Sources", sources), ("Artifacts", artifacts)):
                if paths:
                    lines.extend(("", caption + ":"))
                    lines.extend("- " + path for path in sorted(paths, key=_path_order))
        paragraphs.append("\n".join(lines))
    return "\n\n".join(paragraphs) or "No diagnostics."
