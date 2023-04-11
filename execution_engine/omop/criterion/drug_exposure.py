import logging
from typing import Any, Dict

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import INTERVAL
from sqlalchemy.sql import Select
from sqlalchemy.sql.functions import concat

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.util import ValueNumber, value_factory
from execution_engine.util.sql import SelectInto

__all__ = ["DrugExposure"]


class DrugExposure(Criterion):
    """A drug exposure criterion in a cohort definition."""

    def __init__(
        self,
        name: str,
        exclude: bool,
        category: CohortCategory,
        drug_concepts: list[str],
        ingredient_concept: Concept,
        dose: ValueNumber | None,
        frequency: int | None,
        interval: str | None,
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

    def _sql_generate(self, query: Select) -> Select:

        drug_exposure = self._table

        query = self._sql_filter_concept(query)

        if self._dose is not None:

            if self._route is not None:
                # route is not implemented yet because it uses HemOnc codes in the standard vocabulary
                # (cf concept_class_id = 'Route') but these are not standard codes and HemOnc might not be
                # addressable in FHIR
                logging.warning("Route specified, but not implemented yet")

            if self._frequency != 1:
                raise NotImplementedError("Frequency != 1 not implemented yet")

            interval = func.cast(concat(1, f" {self._interval}"), INTERVAL)

            # Filter only drug_exposures that are inbetween the start and end date of the cohort
            query = super()._insert_datetime(query)

            date_ranges = query.add_columns(
                func.generate_series(
                    func.date_trunc(
                        "day", drug_exposure.c.drug_exposure_start_datetime
                    ),
                    drug_exposure.c.drug_exposure_end_datetime,
                    interval,
                ).label("interval_start"),
                drug_exposure.c.drug_exposure_start_datetime.label("start_datetime"),
                drug_exposure.c.drug_exposure_end_datetime.label("end_datetime"),
                drug_exposure.c.quantity.label("quantity"),
            ).cte("date_ranges")

            interval_quantities = (
                select(
                    [
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
                    ]
                )
                .select_from(date_ranges)
                .cte("interval_quantities")
            )

            interval_ratios = (
                select(
                    [
                        interval_quantities.c.person_id,
                        interval_quantities.c.interval_start,
                        (
                            func.extract("EPOCH", interval_quantities.c.time_diff)
                            / (
                                func.extract(
                                    "EPOCH", interval_quantities.c.end_datetime
                                )
                                - func.extract(
                                    "EPOCH", interval_quantities.c.start_datetime
                                )
                            )
                        ).label("ratio"),
                        interval_quantities.c.quantity,
                    ]
                )
                .select_from(interval_quantities)
                .cte("interval_ratios")
            )

            c_interval_quantity = func.sum(
                interval_ratios.c.quantity * interval_ratios.c.ratio
            ).label("interval_quantity")
            query = (
                select(
                    [
                        interval_ratios.c.person_id,
                        interval_ratios.c.interval_start.label("valid_from"),
                        (interval_ratios.c.interval_start + interval).label("valid_to"),
                        c_interval_quantity,
                    ]
                )
                .select_from(interval_ratios)
                .where(interval_ratios.c.ratio > 0)
                .group_by(interval_ratios.c.person_id, interval_ratios.c.interval_start)
                .having(
                    self._dose.to_sql(column_name=c_interval_quantity, with_unit=False)
                )
                .order_by(interval_ratios.c.person_id, interval_ratios.c.interval_start)
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
