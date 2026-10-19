"""Shared plain-text rendering of persisted candidates and human edit history."""

from runpup.ui.review_field import display


def format_evidence(records):
    return "\n\n".join(
        f"{record.confidence.value.upper()} · {'User edit' if record.user_modified else 'Source candidate'}\n"
        f"Source type: {record.source_type}\nSource: {record.source_reference or 'Unknown'}\n"
        f"Original: {display(record.original_value)}\nCandidate: {display(record.candidate_value)}\n"
        f"Current: {display(record.current_value)}\nRecorded: {record.detected_at.isoformat()}\n"
        f"Target ID: {record.target_id}" for record in records
    ) or "No persisted evidence for this value."
