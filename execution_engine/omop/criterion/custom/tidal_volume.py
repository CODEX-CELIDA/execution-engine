from typing import Any

import sqlalchemy
from sqlalchemy import (
    CTE,
    Alias,
    ColumnElement,
    TableClause,
    and_,
    case,
    func,
    literal,
    select,
    true,
)
from sqlalchemy.sql import Select

import execution_engine.omop.db.omop.tables as omop_tables
from execution_engine.constants import OMOPConcepts
from execution_engine.omop.criterion.abstract import (
    observation_end_datetime,
    observation_start_datetime,
)
from execution_engine.omop.criterion.point_in_time import PointInTimeCriterion
from execution_engine.util.value import ValueNumber

__all__ = ["TidalVolumePerIdealBodyWeight"]

MIN_BODY_WEIGHT_KG = 0.0001  # Minimum body weight in kg, used to avoid division by zero or negative values when deriving ideal body weight


class TidalVolumePerIdealBodyWeight(PointInTimeCriterion):
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

    __GENDER_TO_INT = {"female": 0, "male": 1, "unknown": 0.5}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._table = self._cte()

    def _cte(self) -> CTE:
        """
        Get the CTE for this criterion.
        """
        m = omop_tables.Measurement.__table__.alias("m")

        sql_ibw = self._sql_select_ideal_body_weight(tbl_measurement=m).alias("ibw")
        value_as_number = (m.c.value_as_number / sql_ibw.c.ideal_body_weight).label(
            "value_as_number"
        )
        value_as_concept_id = literal(None).label("value_as_concept_id")
        unit_concept_id = literal(OMOPConcepts.UNIT_ML_PER_KG.value).label(
            "unit_concept_id"
        )

        query = (
            select(
                m.c.person_id,
                m.c.measurement_datetime,
                m.c.measurement_concept_id,
                value_as_number,
                value_as_concept_id,
                unit_concept_id,
                sql_ibw.c.measurement_datetime.label(
                    "measurement_body_height_datetime"
                ),
            )
            .distinct(m.c.person_id, m.c.measurement_datetime)
            .select_from(m)
            .join(sql_ibw, onclause=true(), isouter=True)
            .where(
                and_(
                    m.c.measurement_concept_id
                    == OMOPConcepts.TIDAL_VOLUME_ON_VENTILATOR.value,
                    m.c.measurement_datetime.between(
                        observation_start_datetime, observation_end_datetime
                    ),
                )
            )
        )

        return query.cte("measurement_tvibw")

    @staticmethod
    def _sql_select_ideal_body_weight(
        tbl_measurement: TableClause,
        label: str = "ideal_body_weight",
        person_id: int | None = None,
    ) -> Select:
        """
        Return the SQL to calculate the ideal body weight for a patient.
        """
        person = omop_tables.Person.__table__.alias("p")
        measurement_ibw = omop_tables.Measurement.__table__.alias("m_ibw")

        query = (
            select(
                measurement_ibw.c.person_id,
                measurement_ibw.c.measurement_datetime,
                func.greatest(
                    MIN_BODY_WEIGHT_KG,
                    case(
                        (
                            person.c.gender_concept_id
                            == OMOPConcepts.GENDER_MALE.value,
                            50.0 + 0.91 * (measurement_ibw.c.value_as_number - 152.4),
                        ),
                        (
                            person.c.gender_concept_id
                            == OMOPConcepts.GENDER_FEMALE.value,
                            45.5 + 0.91 * (measurement_ibw.c.value_as_number - 152.4),
                        ),
                        else_=(
                            47.75 + 0.91 * (measurement_ibw.c.value_as_number - 152.4)
                        ),
                    ),
                ).label(label),
            )
            .join(person, person.c.person_id == measurement_ibw.c.person_id)
            .where(
                and_(
                    measurement_ibw.c.measurement_concept_id
                    == OMOPConcepts.BODY_HEIGHT.value,
                    measurement_ibw.c.measurement_datetime
                    <= tbl_measurement.c.measurement_datetime,
                    measurement_ibw.c.person_id == tbl_measurement.c.person_id,
                )
            )
            .order_by(measurement_ibw.c.measurement_datetime.desc())
            .limit(1)
            .subquery()
            .lateral()
        )

        return query

    def _get_datetime_column(
        self, table: TableClause | CTE | Alias, type_: str = "start"
    ) -> sqlalchemy.Column:
        return table.c.measurement_datetime

    def _sql_filter_concept(
        self, query: Select, override_concept_id: int | None = None
    ) -> Select:
        # OMOP Standard Vocabulary does not have a concept for "tidal volume per ideal body weight",
        # so we are using a custom concept for this criterion (in a custom code system).
        # In the query however, we are using the standard concept for "tidal volume on ventilator"
        # and dividing by the ideal body weight to get the tidal volume per ideal body weight.
        # So we need to override the concept_id in the query (which otherwise would not exist in OMOP).

        return super()._sql_filter_concept(
            query, override_concept_id=OMOPConcepts.TIDAL_VOLUME_ON_VENTILATOR.value
        )

    def _filter_days_with_all_values_valid(
        self, query: Select, sql_value: ColumnElement = None
    ) -> Select:
        return super()._filter_days_with_all_values_valid(query, sql_value)

    def sql_select_data(self, person_id: int | None = None) -> Select:
        """
        Get patient data for this criterion
        """
        """measurement = self._table

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
        return select(query.columns).select_from(query)"""
        # todo implement this
        raise NotImplementedError()

    def _sql_select_data(self, query: Select) -> Select:
        # sql_select_data() is overridden to return the union of the ideal body weight
        # and tidal volume data. The _sql_generate() method is not used.
        raise NotImplementedError()

    @classmethod
    def predicted_body_weight_ardsnet(self, gender: str, height_in_cm: float) -> float:
        """
        Predicted body weight according to ARDSNet

        Ref: https://www.nejm.org/doi/10.1056/NEJM200005043421801 (ARDSnet)
        coding: female = 0, male = 1
        height must be in cm
        """

        if gender not in self.__GENDER_TO_INT:
            raise ValueError(
                f"Unrecognized gender {gender}, must be one of {self.__GENDER_TO_INT.keys()}"
            )

        return (
            45.5 + (4.5 * self.__GENDER_TO_INT[gender]) + 0.91 * (height_in_cm - 152.4)
        )

    @classmethod
    def height_for_predicted_body_weight_ardsnet(
        self, gender: str, predicted_weight: float
    ) -> float:
        """
        Height for predicted body weight according to ARDSNet
        """
        if gender not in self.__GENDER_TO_INT:
            raise ValueError(
                f"Unrecognized gender {gender}, must be one of {self.__GENDER_TO_INT.keys()}"
            )

        return (
            predicted_weight - (45.5 + (4.5 * self.__GENDER_TO_INT[gender]))
        ) / 0.91 + 152.4
