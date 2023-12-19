import logging
from typing import Any, Dict

from sqlalchemy import NUMERIC, DateTime, and_, bindparam, case, func, select
from sqlalchemy.dialects.postgresql import INTERVAL
from sqlalchemy.sql import Select
from sqlalchemy.sql.functions import concat

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import (
    Criterion,
    create_conditional_interval_column,
)
from execution_engine.util import Interval, ValueNumber, value_factory
from execution_engine.util.sql import SelectInto

__all__ = ["DrugExposure"]


class DrugExposure(Criterion):
    """A drug exposure criterion in a recommendation."""

    def __init__(
        self,
        name: str,
        exclude: bool,
        category: CohortCategory,
        drug_concepts: list[str],
        ingredient_concept: Concept,
        dose: ValueNumber | None,
        frequency: int | None,
        interval: Interval | str | None,
        route: Concept | None,
    ) -> None:
        """
        Initialize the drug administration action.
        """
        super().__init__(name=name, exclude=exclude, category=category)
        self._set_omop_variables_from_domain("drug")
        self._drug_concepts = drug_concepts
        self._ingredient_concept = ingredient_concept
        self._dose = dose
        self._frequency = frequency

        if interval is not None:
            if isinstance(interval, str):
                interval = Interval(interval)
            assert isinstance(interval, Interval), "interval must be an Interval or str"

        self._interval = interval
        self._route = route

    @property
    def concept(self) -> Concept:
        """Get the concept of the ingredient associated with this DrugExposure"""
        return self._ingredient_concept

    def _sql_filter_concept(self, query: Select) -> Select:
        """
        Return the SQL to filter the data for the criterion.
        """
        drug_exposure = self._table

        query = query.where(drug_exposure.c.drug_concept_id.in_(self._drug_concepts))

        return query

    def _create_query(self) -> Select:
        query = select(
            self._table.c.person_id,
        ).select_from(self._table)

        query = self._sql_generate(query)
        return query

    def _sql_generate(self, query: Select) -> Select:
        drug_exposure = self._table

        query = self._sql_filter_concept(query)

        if self._dose is not None:
            if self._route is not None:
                # route is not implemented yet because it uses HemOnc codes in the standard vocabulary
                # (cf concept_class_id = 'Route') but these are not standard codes and HemOnc might not be
                # addressable in FHIR
                logging.warning("Route specified, but not implemented yet")

            # todo: this won't work if no interval is specified (e.g. when just looking for a single dose)
            # todo: change INTERVAL (postgres dialect) to Interval (sqlalchemy.types.Interval)
            interval = func.cast(concat(1, self._interval.name), INTERVAL)  # type: ignore
            interval_length_seconds = func.cast(
                func.extract("EPOCH", interval), NUMERIC
            ).label("interval_length_seconds")
            one_second = func.cast(concat(1, "second"), INTERVAL)

            # Filter only drug_exposures that are inbetween the start and end date of the cohort
            # todo: move this into another function (duplicate code with super()._insert_datetime()
            c_start = self._get_datetime_column(self._table)
            c_end = self._get_datetime_column(self._table, "end")
            query = query.filter(
                and_(
                    c_start <= bindparam("observation_end_datetime"),
                    c_end >= bindparam("observation_start_datetime"),
                )
            )

            interval_starts = query.add_columns(
                (
                    func.date_trunc(
                        "day", bindparam("observation_start_datetime", type_=DateTime)
                    )
                    + interval_length_seconds
                    * (
                        func.floor(
                            func.extract(
                                "EPOCH",
                                (
                                    drug_exposure.c.drug_exposure_start_datetime
                                    - func.date_trunc(
                                        "day",
                                        bindparam(
                                            "observation_start_datetime", type_=DateTime
                                        ),
                                    )
                                ),
                            )
                            / interval_length_seconds
                        )
                        * one_second
                    )
                ).label("interval_start"),
                drug_exposure.c.drug_exposure_start_datetime.label("start_datetime"),
                drug_exposure.c.drug_exposure_end_datetime.label("end_datetime"),
                drug_exposure.c.quantity.label("quantity"),
            ).cte("interval_starts")

            date_ranges = select(
                interval_starts.c.person_id,
                func.generate_series(
                    interval_starts.c.interval_start,
                    interval_starts.c.end_datetime,
                    interval,
                ).label("interval_start"),
                interval_starts.c.start_datetime,
                interval_starts.c.end_datetime,
                interval_starts.c.quantity,
            ).cte("date_ranges")

            interval_quantities = (
                select(
                    date_ranges.c.person_id,
                    date_ranges.c.interval_start,
                    (
                        func.least(
                            date_ranges.c.end_datetime,
                            date_ranges.c.interval_start + interval,
                        )
                        - func.greatest(
                            date_ranges.c.start_datetime,
                            date_ranges.c.interval_start,
                        )
                    ).label("time_diff"),
                    date_ranges.c.start_datetime,
                    date_ranges.c.end_datetime,
                    date_ranges.c.quantity,
                )
                .select_from(date_ranges)
                .cte("interval_quantities")
            )

            # Calculate the ratio of the interval that the drug was taken and handle the case where the
            # interval is 0 (set ratio to 1 explicitly, this is a "bolus" dose)
            ir_ratio_num = func.extract("EPOCH", interval_quantities.c.time_diff)
            ir_ratio_denom = func.extract(
                "EPOCH", interval_quantities.c.end_datetime
            ) - func.extract("EPOCH", interval_quantities.c.start_datetime)
            ir_ratio = case(
                (ir_ratio_denom == 0, 1), else_=ir_ratio_num / ir_ratio_denom
            ).label("ratio")

            interval_ratios = (
                select(
                    interval_quantities.c.person_id,
                    interval_quantities.c.interval_start,
                    ir_ratio,
                    interval_quantities.c.quantity,
                )
                .select_from(interval_quantities)
                .cte("interval_ratios")
            )

            c_interval_quantity = func.sum(
                interval_ratios.c.quantity * interval_ratios.c.ratio
            ).label("interval_quantity")
            c_interval_count = func.count().label("interval_count")

            conditional_interval_column = create_conditional_interval_column(
                condition=and_(
                    self._dose.to_sql(column_name=c_interval_quantity, with_unit=False),
                    c_interval_count >= self._frequency,
                )
            )

            query = (
                select(
                    interval_ratios.c.person_id,
                    interval_ratios.c.interval_start.label("interval_start"),
                    (interval_ratios.c.interval_start + interval - one_second).label(
                        "interval_end"
                    ),
                    conditional_interval_column.label("interval_type"),
                    # c_interval_quantity,
                    # c_interval_count,
                )
                .select_from(interval_ratios)
                .where(interval_ratios.c.ratio > 0)
                .group_by(interval_ratios.c.person_id, interval_ratios.c.interval_start)
            )

        return query

    def _insert_datetime(self, query: SelectInto) -> SelectInto:
        """
        Return the SQL to insert the datetime for the criterion.

        Nothing to do here, because the datetime is already inserted by the
        _sql_generate method.
        """
        return query

    def _sql_select_data(self, query: Select) -> Select:
        """
        Return the SQL to select the data for the criterion.
        """

        drug_exposure = self._table

        query = query.add_columns(
            drug_exposure.c.drug_concept_id.label("parameter_concept_id"),
            drug_exposure.c.drug_exposure_start_datetime.label("start_datetime"),
            drug_exposure.c.drug_exposure_end_datetime.label("end_datetime"),
            drug_exposure.c.quantity.label("drug_dose_as_number"),
        )

        return query

    def description(self) -> str:
        """
        Get a human-readable description of the criterion.
        """
        return (
            f"{self.__class__.__name__}['{self._name}']("
            f"ingredient={self._ingredient_concept.concept_name}, "
            f"dose={str(self._dose)}, frequency={self._frequency}/{self._interval}, "
            f"route={self._route.concept_name if self._route is not None else None} "
            f")"
        )

    def dict(self) -> dict[str, Any]:
        """
        Return a dictionary representation of the criterion.
        """
        return {
            "name": self._name,
            "exclude": self._exclude,
            "category": self._category.value,
            "drug_concepts": self._drug_concepts,
            "ingredient_concept": self._ingredient_concept.dict(),
            "dose": self._dose.dict() if self._dose is not None else None,
            "frequency": self._frequency,
            "interval": self._interval,
            "route": self._route.dict() if self._route is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DrugExposure":
        """
        Create a drug exposure criterion from a dictionary representation.
        """

        dose = value_factory(**data["dose"]) if data["dose"] is not None else None

        assert dose is None or isinstance(dose, ValueNumber), "Dose must be a number"

        return cls(
            name=data["name"],
            exclude=data["exclude"],
            category=CohortCategory(data["category"]),
            drug_concepts=data["drug_concepts"],
            ingredient_concept=Concept(**data["ingredient_concept"]),
            dose=dose,
            frequency=data["frequency"],
            interval=data["interval"],
            route=Concept(**data["route"]) if data["route"] is not None else None,
        )
