__all__ = ["DeviceExposure"]

from execution_engine.omop.criterion.continuous import ContinuousCriterion


class DeviceExposure(ContinuousCriterion):
    """A device_exposure criterion in a recommendation."""
