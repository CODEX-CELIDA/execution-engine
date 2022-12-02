import logging

import pandas as pd
from sqlalchemy import func, literal_column, select, table, text

from ...util import ValueNumber
from ..concepts import Concept
from ..vocabulary import standard_vocabulary
from .abstract import Criterion


class DrugExposure(Criterion):
    """A drug exposure criterion in a cohort definition."""

    _OMOP_TABLE = "drug_exposure"
    _OMOP_COLUMN_PREFIX = "drug"

    def __init__(
        self,
        name: str,
        exclude: bool,
        drug_concepts: pd.DataFrame,
        dose: ValueNumber | None,
        frequency: int | None,
        interval: str | None,
        route: Concept | None,
    ) -> None:
        """
        Initialize the drug administration action.
        """
        super().__init__(name, exclude)
        self._drug_concepts = drug_concepts
        self._dose = dose
        self._frequency = frequency
        self._interval = interval
        self._route = route

    @classmethod
    def filter_same_unit(cls, df: pd.DataFrame, unit: Concept) -> pd.DataFrame:
        """
        Filters the given dataframe to only include rows with the given unit.

        If the unit is "international unit" or "unit" then the respective other unit is also included.
        This is because RxNorm (upon which OMOP is based) seems to use unit (UNT) for international units.
        """
        logging.warning(
            "Selecting only drug entries with unit matching to that of the recommendation"
        )

        df_filtered = df.query("amount_unit_concept_id==@unit.id")

        other_unit = None
        if unit.name == "unit":
            other_unit = standard_vocabulary.get_standard_unit_concept("[iU]")
        elif unit.name == "international unit":
            other_unit = standard_vocabulary.get_standard_unit_concept("[U]")

        if other_unit is not None:
            logging.info(
                f'Detected unit "{unit.name}", also selecting "{other_unit.name}"'
            )
            df_filtered = pd.concat(
                [df_filtered, df.query("amount_unit_concept_id==@other_unit.id")]
            )

        return df_filtered

    def _sql_generate(self, sql_select: str) -> str:

        # fixme : implement "AND drug_exposure_start_datetime BETWEEN (.., ...)"

        if self._dose is not None:
            df_drugs = self.filter_same_unit(
                self._drug_concepts, self._dose.unit
            )  # todo: we should not filter here, but perform conversions of values instead (if possible)
            dose_sql = self._dose.to_sql(
                table_name=None, column_name="sum(de.quantity)", with_unit=False
            )

            if self._route is not None:
                # route is not implemented yet because it uses HemOnc codes in the standard vocabulary
                # (cf concept_class_id = 'Route') but these are not standard codes and HemOnc might not be
                # addressable in FHIR
                logging.warning("Route specified, but not implemented yet")

            drug_concept_ids = df_drugs["drug_concept_id"].tolist()

            query = (
                select(
                    literal_column("de.person_id"),
                    func.date_trunc(
                        "day", literal_column("de.drug_exposure_start_datetime")
                    ).label("interval"),
                    func.count(literal_column("de.*")).label("cnt"),
                    func.sum(literal_column("de.quantity")).label("dose"),
                )
                .select_from(table("drug_exposure").alias("de"))
                .where(literal_column("de.drug_concept_id").in_(drug_concept_ids))
                .group_by(literal_column("de.person_id"), literal_column("interval"))
                .having(dose_sql)
                .having(func.count(literal_column("de.*")) == self._frequency)
            )
        else:
            # no dose specified, so we just count the number of drug exposures
            drug_concept_ids = self._drug_concepts["drug_concept_id"].tolist()
            query = (
                select(literal_column("de.person_id"))
                .select_from(table("drug_exposure").alias("de"))
                .where(literal_column("de.drug_concept_id").in_(drug_concept_ids))
            )

        return query
