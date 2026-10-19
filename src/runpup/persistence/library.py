"""Lightweight catalog summaries, without owned evidence or external references."""

from dataclasses import dataclass
from datetime import datetime

from runpup.domain.models import Parameter, TrainingStatus


SUMMARY_PARAMETERS = ("epochs", "observed_epoch", "global_step", "max_steps", "qlora")


@dataclass(frozen=True, slots=True)
class RunSummary:
    id: str
    display_name: str
    imported_at: datetime
    base_model: str | None = None
    source_trainer: str | None = None
    method: str | None = None
    rating: int | None = None
    favorite: bool = False
    training_status: TrainingStatus = TrainingStatus.UNKNOWN
    tags: tuple[str, ...] = ()
    parameters: tuple[Parameter, ...] = ()
