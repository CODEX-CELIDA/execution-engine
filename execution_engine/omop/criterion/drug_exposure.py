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
        dose: ValueNumber,
        frequency: int,
        interval: str,
        drug_concepts: pd.DataFrame,
    ) -> None:
        """
        Initialize the drug administration action.
        """
        super().__init__(name, exclude)
        self._dose = dose
        self._frequency = frequency
        self._interval = interval
        self._drug_concepts = drug_concepts

    @classmethod
    def filter_same_unit(cls, df: pd.DataFrame, unit: Concept) -> pd.DataFrame:
        """
        Filters the given dataframe to only include rows with the given unit.

        If the unit is "international unit" or "unit" then the respective other unit is also included.
        This is because RxNorm (upon which OMOP is based) seems to use unit (UNT) for international units.
        """
        logging.warning(
            "selecting only drug entries with unit matching to that of the recommendation"
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
        df_drugs = self.filter_same_unit(
            self._drug_concepts, self._dose.unit
        )  # todo: we should not filter here, but perform conversions of values instead (if possible)
        dose_sql = self._dose.to_sql(
            table_name=None, column_name="sum(de.quantity)", with_unit=False
        )

        # fmt: off
        query = text(  # nosec
            f"""SELECT
            de.person_id,
            date_trunc(:intervals, de.drug_exposure_start_datetime) as interval,
            count(de.*) as cnt,
            sum(de.quantity) as dose
        FROM drug_exposure de
        WHERE de.drug_concept_id IN (:drug_concept_ids)
         -- AND drug_exposure_start_datetime BETWEEN (.., ...)
        GROUP BY de.person_id, interval
        HAVING
            {dose_sql}
            AND count(de.*) = :frequency
        """)
        # fmt: on

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

        return query
