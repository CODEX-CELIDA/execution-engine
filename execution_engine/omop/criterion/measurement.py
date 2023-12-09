__all__ = ["Measurement"]

from execution_engine.omop.criterion.point_in_time import PointInTimeCriterion


class Measurement(PointInTimeCriterion):
    """A measurement criterion in a cohort definition."""
