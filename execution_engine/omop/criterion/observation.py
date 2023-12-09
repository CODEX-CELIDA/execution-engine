__all__ = ["Observation"]

from execution_engine.omop.criterion.point_in_time import PointInTimeCriterion


class Observation(PointInTimeCriterion):
    """An observation criterion in a cohort definition."""
