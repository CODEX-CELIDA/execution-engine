import logging
from typing import Any, Dict

from sqlalchemy import func, literal_column
from sqlalchemy.dialects.postgresql import INTERVAL
from sqlalchemy.sql import Select
from sqlalchemy.sql.functions import concat

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.util import ValueNumber, value_factory

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

            dose_sql = self._dose.to_sql(
                table_name=None, column_name="sum(de.quantity)", with_unit=False
            )

            if self._route is not None:
                # route is not implemented yet because it uses HemOnc codes in the standard vocabulary
                # (cf concept_class_id = 'Route') but these are not standard codes and HemOnc might not be
                # addressable in FHIR
                logging.warning("Route specified, but not implemented yet")

            # TODO: The following logic selects solely based on drug_exposure_start_datetime, not end_datetime
            c_valid_from = func.date_trunc(
                self._interval, drug_exposure.c.drug_exposure_start_datetime
            ).label("valid_from")
            c_valid_to = (
                func.date_trunc(
                    self._interval, drug_exposure.c.drug_exposure_start_datetime
                )
                + func.cast(concat(1, f" {self._interval}"), INTERVAL)
            ).label("valid_to")

            query = query.add_columns(
                c_valid_from,
                c_valid_to,
                func.count(literal_column("de.*")).label("cnt"),
                func.sum(drug_exposure.c.quantity).label("dose"),
            )

            query = (
                query.group_by(
                    drug_exposure.c.person_id, c_valid_from.name, c_valid_to.name
                )
                .having(dose_sql)
                .having(func.count(literal_column("de.*")) == self._frequency)
            )

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
