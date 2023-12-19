__all__ = ["ConditionOccurrence"]

from execution_engine.omop.criterion.continuous import ContinuousCriterion


class ConditionOccurrence(ContinuousCriterion):
    """A condition occurrence criterion in a recommendation."""
