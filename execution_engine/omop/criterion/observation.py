from .concept import ConceptCriterion


class Observation(ConceptCriterion):
    """An observation criterion in a cohort definition."""

    _OMOP_TABLE = "observation"
    _OMOP_COLUMN_PREFIX = "observation"
