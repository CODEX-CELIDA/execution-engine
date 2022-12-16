import logging
from typing import Any, Dict

from sqlalchemy import func, literal_column, select
from sqlalchemy.sql import Select

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
        self._dose = dose
        self._frequency = frequency
        self._interval = interval
        self._route = route

    def _sql_generate(self, sql: Select) -> Select:

        drug_exposure = self._table

        if self._dose is not None:

            dose_sql = self._dose.to_sql(
                table_name=None, column_name="sum(de.quantity)", with_unit=False
            )

            if self._route is not None:
                # route is not implemented yet because it uses HemOnc codes in the standard vocabulary
                # (cf concept_class_id = 'Route') but these are not standard codes and HemOnc might not be
                # addressable in FHIR
                logging.warning("Route specified, but not implemented yet")

            query = (
                select(
                    drug_exposure.c.person_id,
                    func.date_trunc(
                        "day", drug_exposure.c.drug_exposure_start_datetime
                    ).label("interval"),
                    func.count(literal_column("de.*")).label("cnt"),
                    func.sum(drug_exposure.c.quantity).label("dose"),
                )
                .select_from(drug_exposure)
                .join(
                    self._base_table,
                    self._base_table.c.person_id == drug_exposure.c.person_id,
                )
                .where(drug_exposure.c.drug_concept_id.in_(self._drug_concepts))
                .group_by(drug_exposure.c.person_id, literal_column("interval"))
                .having(dose_sql)
                .having(func.count(literal_column("de.*")) == self._frequency)
            )
        else:
            # no dose specified, so we just count the number of drug exposures
            query = (
                select(drug_exposure.c.person_id)
                .select_from(drug_exposure)
                .join(
                    self._base_table,
                    self._base_table.c.person_id == drug_exposure.c.person_id,
                )
                .where(drug_exposure.c.drug_concept_id.in_(self._drug_concepts))
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
            dose=dose,
            frequency=data["frequency"],
            interval=data["interval"],
            route=Concept(**data["route"]) if data["route"] is not None else None,
        )
