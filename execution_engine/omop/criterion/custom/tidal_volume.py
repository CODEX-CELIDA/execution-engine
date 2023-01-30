from sqlalchemy import case, literal_column, select, union_all
from sqlalchemy.sql import Select

from execution_engine.clients import omopdb
from execution_engine.constants import LOINC_TIDAL_VOLUME, OMOPConcepts
from execution_engine.omop.criterion.concept import ConceptCriterion
from execution_engine.omop.vocabulary import LOINC, standard_vocabulary
from execution_engine.util import ValueNumber

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

    def _sql_filter_tidal_volume(self, query: Select) -> Select:
        """
        Return the SQL to filter the data for the criterion.

        Filtering for TIDAL VOLUME LOINC.
        """

        # todo: cache concept
        concept_tv = standard_vocabulary.get_standard_concept(
            system_uri=LOINC.system_uri, concept=LOINC_TIDAL_VOLUME
        )

        query = query.filter(
            self._table.c.measurement_concept_id == concept_tv.concept_id
        )
        return query

    def _sql_select_ideal_body_weight(
        self, label: str = "ideal_body_weight", person_id: int | None = None
    ) -> Select:
        """
        Return the SQL to calculate the ideal body weight for a patient.
        """
        person = omopdb.tables["cds_cdm.person"].alias("p")
        measurement = self._table

        query = self._sql_header(person_id=person_id)
        query = (
            query.add_columns(
                case(
                    (
                        person.c.gender_concept_id == OMOPConcepts.GENDER_MALE.value,
                        50.5 + 0.91 * (measurement.c.value_as_number - 152.4),
                    ),
                    (
                        person.c.gender_concept_id == OMOPConcepts.GENDER_FEMALE.value,
                        45.5 + 0.91 * (measurement.c.value_as_number - 152.4),
                    ),
                    else_=(47.75 + 0.91 * (measurement.c.value_as_number - 152.4)),
                ).label(label),
            )
            .join(person, person.c.person_id == measurement.c.person_id)
            .where(
                measurement.c.measurement_concept_id == OMOPConcepts.BODY_WEIGHT.value
            )
        )

        return query

    def _sql_generate(self, query: Select) -> Select:
        sql_ibw = self._sql_select_ideal_body_weight().alias("ibw")

        sql_value = self._value.to_sql(
            table_name=None,
            column_name=literal_column("m.value_as_number / ibw.ideal_body_weight"),
            with_unit=False,
        )

        query = query.join(sql_ibw, sql_ibw.c.person_id == self._table.c.person_id)
        query = self._sql_filter_concept(
            query, override_concept_id=OMOPConcepts.TIDAL_VOLUME_ON_VENTILATOR.value
        )
        query = query.filter(sql_value)

        return query

    def sql_select_data(self, person_id: int | None = None) -> Select:
        """
        Get patient data for this criterion
        """
        measurement = self._table

        ibw = self._sql_select_ideal_body_weight(
            label="value_as_number", person_id=person_id
        )
        ibw = ibw.add_columns(
            measurement.c.measurement_concept_id.label("parameter_concept_id"),
            measurement.c.measurement_datetime.label("start_datetime"),
            measurement.c.unit_concept_id.label("unit_concept_id"),
        )
        ibw = self._insert_datetime(ibw)

        tv = self._sql_header(person_id=person_id)
        tv = self._sql_filter_concept(tv)
        tv = tv.add_columns(
            measurement.c.value_as_number.label("value_as_number"),
            measurement.c.measurement_concept_id.label("parameter_concept_id"),
            measurement.c.measurement_datetime.label("start_datetime"),
            measurement.c.unit_concept_id.label("unit_concept_id"),
        )
        tv = self._insert_datetime(tv)

        query = union_all(ibw, tv).alias("data")

        # need to return Select, not CompoundSelect
        return select(query.columns).select_from(query)

    def _sql_select_data(self, query: Select) -> Select:
        # sql_select_data() is overridden to return the union of the ideal body weight
        # and tidal volume data. The _sql_generate() method is not used.
        raise NotImplementedError()

    @staticmethod
    def predicted_body_weight_ardsnet(gender: int, height_in_cm: float) -> float:
        """
        Predicted body weight according to ARDSNet

        Ref: https://www.nejm.org/doi/10.1056/NEJM200005043421801 (ARDSnet)
        coding: female = 0, male = 1
        height must be in cm
        """
        return 45.5 + (4.5 * gender) + 0.91 * (height_in_cm - 152.4)
