from sqlalchemy import Select

from execution_engine.omop.criterion.abstract import column_interval_type
from execution_engine.omop.criterion.concept import ConceptCriterion
from execution_engine.util.interval import IntervalType


class ContinuousCriterion(ConceptCriterion):
    """A continuous criterion (having a start and end datetime) in a recommendation."""

    def _create_query(self) -> Select:
        """
        Get the SQL representation of the criterion.
        """
        if self._OMOP_VALUE_REQUIRED:
            raise NotImplementedError("ContinuousCriterion does not support value")

        query = self._sql_header()
        query = self._sql_filter_concept(query)
        query = query.add_columns(column_interval_type(IntervalType.POSITIVE))
        query = self._filter_datetime(query)
        query = self._filter_base_persons(query)

        return query
