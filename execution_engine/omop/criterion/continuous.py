from sqlalchemy import Select

from execution_engine.omop.criterion.concept import ConceptCriterion


class ContinuousCriterion(ConceptCriterion):
    """A point-in-time criterion in a cohort definition."""

    def _create_query(self) -> Select:
        """
        Get the SQL representation of the criterion.
        """

        query = self._sql_header()
        query = self._sql_filter_concept(query)

        if self._OMOP_VALUE_REQUIRED:
            raise NotImplementedError("ContinuousCriterion does not support value")

        return query
