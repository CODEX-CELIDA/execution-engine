from .concept import ConceptCriterion


class ProcedureOccurrence(ConceptCriterion):
    """A procedure occurrence criterion in a cohort definition."""

    _OMOP_TABLE = "procedure_occurrence"
    _OMOP_COLUMN_PREFIX = "procedure"
