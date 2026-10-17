"""Read-only importers that stop at reviewable domain proposals."""

from runpup.importers.hf_trainer import DetectionResult, HFTrainerImporter

__all__ = ["DetectionResult", "HFTrainerImporter"]
