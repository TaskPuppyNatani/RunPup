from datetime import datetime
import subprocess
import sys

import pytest

from runpup.domain.models import EntityType, ImportDiagnostic, Metric, MetricSummary, Parameter, Provenance, Run
from runpup.domain.proposals import FieldState, ProposedField


def test_domain_and_persistence_import_without_qt():
    result = subprocess.run([
        sys.executable, "-c",
        "import runpup.domain.models, runpup.persistence.repository, runpup.services.paths, sys; "
        "assert not any(name.startswith('PySide6') for name in sys.modules)",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("value", [{"nested": 1}, [1, 2], float("nan"), float("inf")])
def test_parameters_reject_unsupported_or_nonfinite_values(value):
    with pytest.raises(ValueError):
        Parameter(key="config", value=value)


def test_value_at_step_requires_step():
    with pytest.raises(ValueError, match="requires a step"):
        Metric(name="loss", value=1.0, summary=MetricSummary.VALUE_AT_STEP)


def test_naive_training_timestamp_is_rejected():
    with pytest.raises(ValueError, match="timezone"):
        Run(display_name="History", source_storage_root_id="root", source_relative_directory="run",
            training_started_at=datetime(2025, 1, 1))


def test_detection_timestamps_are_required():
    with pytest.raises(ValueError, match="required"):
        ImportDiagnostic(message="Incomplete", detected_at=None)
    with pytest.raises(ValueError, match="required"):
        Provenance(target_entity=EntityType.RUN, target_id="run", target_field="method",
                   source_type="file", detected_at=None)


def test_proposal_scalar_conflicts_preserve_boolean_numeric_distinction():
    def evidence(value):
        return Provenance(target_entity=EntityType.RUN, target_id="proposal", target_field="learning_rate",
                          source_type="training_args", candidate_value=value)

    numeric = ProposedField(key="learning_rate", candidates=(evidence(1), evidence(1.0)))
    assert numeric.state is FieldState.DETECTED
    mixed = ProposedField(key="learning_rate", candidates=(evidence(True), evidence(1)))
    assert mixed.state is FieldState.CONFLICT
    assert mixed.material
