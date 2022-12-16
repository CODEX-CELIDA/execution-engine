from sqlalchemy import Float, Integer, literal_column, text
from sqlalchemy.sql import Select

from execution_engine.constants import (
    LOINC_TIDAL_VOLUME,
    OMOP_BODY_WEIGHT,
    OMOP_GENDER_FEMALE,
    OMOP_GENDER_MALE,
)
from execution_engine.omop.criterion.concept import ConceptCriterion
from execution_engine.omop.vocabulary import LOINC, standard_vocabulary
from execution_engine.util import ValueNumber

pass

__all__ = ["TidalVolumePerIdealBodyWeight"]


class TidalVolumePerIdealBodyWeight(ConceptCriterion):
    """
    Tidal volume per ideal body weight

    This class explicitly calculates the ideal body weight for a patient and then
    calculates the tidal volume per ideal body weight. The ideal body weight is
    calculated using the following formula from ARDS net
    (Ref: https://www.nejm.org/doi/10.1056/NEJM200005043421801 ).

    This class is required because no standard OMOP concept exists for tidal volume
    per ideal body weight.
    """

    _value: ValueNumber

    def _sql_generate(self, query: Select) -> Select:
        import warnings

        warnings.warn("Make sure that base table is joined for subselects")

        sql_ibw = (
            text(  # nosec
                f"""
        SELECT DISTINCT ON (m.person_id) m.person_id,
        (CASE
        WHEN p.gender_concept_id = :omop_gender_male  THEN 50.0 + 0.91 * (m.value_as_number - 152.4)
        WHEN p.gender_concept_id = :omop_gender_female THEN 45.5 + 0.91 * (m.value_as_number - 152.4)
        ELSE 47.75 + 0.91 * (m.value_as_number - 152.4)
        END ) AS ideal_body_weight
        FROM measurement m
        INNER JOIN "person" p ON m.person_id = p.person_id
        INNER JOIN "{self._base_table.name}" t ON m.person_id = t.person_id
        WHERE m.measurement_concept_id = :omop_body_weight
        """
            )
            .columns(person_id=Integer, ideal_body_weight=Float)
            .bindparams(
                omop_gender_male=OMOP_GENDER_MALE,
                omop_gender_female=OMOP_GENDER_FEMALE,
                omop_body_weight=OMOP_BODY_WEIGHT,
            )
        )

        sql_ibw = sql_ibw.alias("ibw")

        concept_tv = standard_vocabulary.get_standard_concept(
            system_uri=LOINC.system_uri, concept=LOINC_TIDAL_VOLUME
        )
        sql_value = self._value.to_sql(
            table_name=None,
            column_name=literal_column("m.value_as_number / ibw.ideal_body_weight"),
            with_unit=False,
        )
        # fixme: remove this when it's clear that the above statement works
        # sql_tv = text(f'''
        # INNER JOIN meausurement m ON (table_out.person_id = m.person_id)
        # INNER JOIN ({sql_ibw}) i ON m.person_id = i.person_id)
        # ''')

        tbl_meas = self._table

        query = query.join(
            tbl_meas, tbl_meas.c.person_id == self._base_table.c.person_id
        )
        query = query.join(sql_ibw, sql_ibw.c.person_id == tbl_meas.c.person_id)
        query = query.filter(tbl_meas.c.measurement_concept_id == concept_tv.concept_id)
        query = query.filter(sql_value)

        return query

    @staticmethod
    def predicted_body_weight_ardsnet(gender: int, height_in_cm: float) -> float:
        """
        Predicted body weight according to ARDSNet

        Ref: https://www.nejm.org/doi/10.1056/NEJM200005043421801 (ARDSnet)
        coding: female = 0, male = 1
        height must be in cm
        """
        return 45.5 + (4.5 * gender) + 0.91 * (height_in_cm - 152.4)
