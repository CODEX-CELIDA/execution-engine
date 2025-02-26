from typing import Any, Dict, Self

from sqlalchemy import Select, select

from execution_engine.omop.criterion.abstract import (
    Criterion,
    column_interval_type,
    observation_end_datetime,
    observation_start_datetime,
)
from execution_engine.util.interval import IntervalType


class NoopCriterion(Criterion):
    """
    Select patients who are post-surgical in the timeframe between the day of the surgery and 6 days after the surgery.
    """

    _static = True

    def _create_query(self) -> Select:
        """
        Get the SQL Select query for data required by this criterion.
        """
        subquery = self.base_query().subquery()

        query = select(
            subquery.c.person_id,
            column_interval_type(IntervalType.POSITIVE),
            observation_start_datetime.label("interval_start"),
            observation_end_datetime.label("interval_end"),
        )

        query = self._filter_base_persons(query, c_person_id=subquery.c.person_id)
        query = self._filter_datetime(query)

        return query

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Create an object from a dictionary.
        """
        return cls(**data)

    def description(self) -> str:
        """
        Get a description of the criterion.
        """
        return self.__class__.__name__

    def dict(self) -> dict:
        """
        Get a dictionary representation of the object.
        """
        return {
            "class": self.__class__.__name__,
        }
