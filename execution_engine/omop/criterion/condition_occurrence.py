from .concept import ConceptCriterion


class ConditionOccurrence(ConceptCriterion):
    """A condition occurrence criterion in a cohort definition."""

    _OMOP_TABLE = "condition_occurrence"
    _OMOP_COLUMN_PREFIX = "condition"
