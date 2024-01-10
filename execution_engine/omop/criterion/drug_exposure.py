import logging
from typing import Any, Dict

from sqlalchemy import and_, case, func, select
from sqlalchemy.sql import Select

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import (
    SQL_ONE_SECOND,
    Criterion,
    column_interval_type,
    create_conditional_interval_column,
)
from execution_engine.omop.db.omop import tables as omop
from execution_engine.util.enum import TimeUnit
from execution_engine.util.interval import IntervalType
from execution_engine.util.sql import SelectInto
from execution_engine.util.types import Dosage
from execution_engine.util.value import ValueNumber
from execution_engine.util.value.factory import value_factory

__all__ = ["DrugExposure"]


class DrugExposure(Criterion):
    """A drug exposure criterion in a recommendation."""

    def __init__(
        self,
        name: str,
        exclude: bool,
        category: CohortCategory,
        ingredient_concept: Concept,
        dose: Dosage | None,
        route: Concept | None,
    ) -> None:
        """
        Initialize the drug administration action.
        """
        super().__init__(name=name, exclude=exclude, category=category)
        self._set_omop_variables_from_domain("drug")
        self._ingredient_concept = ingredient_concept

        if dose is None:
            dose = Dosage(
                dose=None,
                count=None,
                duration=None,
                frequency=1,
                interval=1 * TimeUnit.DAY,
            )
        elif dose.frequency is None:
            logging.warning(  # type: ignore # (statement is reachable)
                f"No frequency specified in {self.description()}, using default 'Any / day'"
            )
            # set period first, otherwise validation error is triggered
            dose.period = 1 * TimeUnit.DAY
            dose.frequency = 1

        self._dose = dose
        self._route = route

    @property
    def concept(self) -> Concept:
        """Get the concept of the ingredient associated with this DrugExposure"""
        return self._ingredient_concept

    def _sql_filter_concept(self, query: Select) -> Select:
        """
        Return the SQL to filter the data for the criterion.

        We here filter the drug_exposure table by all concepts that are descendants of the ingredient concept that
        is associated with this DrugExposure Criterion.

        Warning: We only take drugs that have the same amount_unit_concept_id as the ingredient!
        If drug mixtures are specified in the OMOP CDM database, these WILL NOT be taken into account!

        :param query: The query to filter
        :return: The filtered query
        """
        drug_exposure = self._table

        c = omop.Concept.__table__.alias("c")
        cd = omop.Concept.__table__.alias("cd")
        ca = omop.t_concept_ancestor.alias("ca")
        ds = omop.t_drug_strength.alias("ds")
        ds_ingr = omop.t_drug_strength.alias("ds_ingr")

        drugs_by_ingredient = (
            select(cd.c.concept_id.label("descendant_concept_id"))
            .join(
                ca,
                c.c.concept_id == ca.c.ancestor_concept_id,
            )
            .join(
                cd,
                cd.c.concept_id == ca.c.descendant_concept_id,
            )
            .join(
                ds,
                and_(
                    ds.c.drug_concept_id == cd.c.concept_id,
                    ds.c.ingredient_concept_id == c.c.concept_id,
                ),
            )
            .join(
                ds_ingr,
                ds_ingr.c.drug_concept_id == c.c.concept_id,
            )
            .where(
                and_(
                    c.c.concept_id == self._ingredient_concept.concept_id,
                    c.c.domain_id == "Drug",
                    ds.c.amount_unit_concept_id == ds_ingr.c.amount_unit_concept_id,
                    func.now().between(c.c.valid_start_date, c.c.valid_end_date),
                    func.now().between(cd.c.valid_start_date, cd.c.valid_end_date),
                    func.now().between(ds.c.valid_start_date, ds.c.valid_end_date),
                )
            )
        )

        query = query.where(drug_exposure.c.drug_concept_id.in_(drugs_by_ingredient))

        return query

    def _create_query(self) -> Select:
        query = select(
            self._table.c.person_id,
        ).select_from(self._table)
        query = self._sql_filter_concept(query)
        query = self._filter_datetime(query)

        if self._route is not None:
            # route is not implemented yet because it uses HemOnc codes in the standard vocabulary
            # (cf concept_class_id = 'Route') but these are not standard codes and HemOnc might not be
            # addressable in FHIR
            logging.warning("Route specified, but not implemented yet")

        # todo: this won't work if no interval is specified (e.g. when just looking for a single dose)
        interval = self._dose.interval.to_sql_interval()

        interval_starts = self.cte_interval_starts(
            query, interval, add_columns=[self._table.c.quantity.label("quantity")]
        )
        date_ranges = self.cte_date_ranges(
            interval_starts, interval, add_columns=[interval_starts.c.quantity]
        )

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

        if self._dose.dose is not None:
            conditional_interval_column = create_conditional_interval_column(
                condition=and_(
                    self._dose.dose.to_sql(column_name=c_interval_quantity),
                    self._dose.frequency.to_sql(column_name=c_interval_count),
                )
            )
        elif self._dose.frequency is not None:
            conditional_interval_column = create_conditional_interval_column(
                condition=self._dose.frequency.to_sql(column_name=c_interval_count)
            )
        else:
            # any dose and any frequency is fine
            conditional_interval_column = column_interval_type(IntervalType.POSITIVE)  # type: ignore # (statement is reachable)

        query = (
            select(
                interval_ratios.c.person_id,
                interval_ratios.c.interval_start.label("interval_start"),
                (interval_ratios.c.interval_start + interval - SQL_ONE_SECOND).label(
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
        route = self._route if hasattr(self, "_route") else None

        parts = [f"ingredient={self._ingredient_concept.concept_name}"]
        if self._dose is not None:
            parts.append(f"dose={str(self._dose)}")

        if route is not None:
            parts.append(f"route={route.concept_name}")

        return f"{self.__class__.__name__}[" + ", ".join(parts) + "]"

    def dict(self) -> dict[str, Any]:
        """
        Return a dictionary representation of the criterion.
        """
        return {
            "name": self._name,
            "exclude": self._exclude,
            "category": self._category.value,
            "ingredient_concept": self._ingredient_concept.dict(),
            "dose": self._dose.dict(include_meta=True)
            if self._dose is not None
            else None,
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
            ingredient_concept=Concept(**data["ingredient_concept"]),
            dose=dose,
            route=Concept(**data["route"]) if data["route"] is not None else None,
        )
