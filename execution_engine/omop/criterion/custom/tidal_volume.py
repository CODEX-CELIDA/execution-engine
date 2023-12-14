from sqlalchemy import ColumnElement, Interval, bindparam, case, func, select, union_all
from sqlalchemy.sql import Select

from execution_engine.clients import omopdb
from execution_engine.constants import OMOPConcepts
from execution_engine.omop.criterion.abstract import create_conditional_interval_column
from execution_engine.omop.criterion.point_in_time import PointInTimeCriterion
from execution_engine.util import ValueNumber

__all__ = ["TidalVolumePerIdealBodyWeight"]


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
                        person.c.gender_concept_id
                        == int(OMOPConcepts.GENDER_MALE.value),
                        50.0 + 0.91 * (measurement.c.value_as_number - 152.4),
                    ),
                    (
                        person.c.gender_concept_id
                        == int(OMOPConcepts.GENDER_FEMALE.value),
                        45.5 + 0.91 * (measurement.c.value_as_number - 152.4),
                    ),
                    else_=(47.75 + 0.91 * (measurement.c.value_as_number - 152.4)),
                ).label(label),
            )
            .join(person, person.c.person_id == measurement.c.person_id)
            .where(
                measurement.c.measurement_concept_id == OMOPConcepts.BODY_HEIGHT.value
            )
            # .group_by(measurement.c.person_id)
        )

        return query

    def _create_query(self) -> Select:
        """
        Create the SQL query to calculate the tidal volume per ideal body weight.
        """
        # todo: refactor to not duplicate code from PointInTimeCriterion._create_query()
        if self._OMOP_VALUE_REQUIRED:
            assert self._value is not None, "Value is required for this criterion"

        interval_hours_param = bindparam(
            "validity_threshold_hours", value=12
        )  # todo make dynamic
        datetime_col = self._get_datetime_column(self._table, "start")
        time_threshold_param = func.cast(
            func.concat(interval_hours_param, "hours"), Interval
        )

        sql_ibw = self._sql_select_ideal_body_weight().alias("ibw")
        c_value = (self._table.c.value_as_number / sql_ibw.c.ideal_body_weight).label(
            "value_as_number"
        )

        cte = select(
            self._table.c.person_id,
            datetime_col.label("datetime"),
            self._table.c.value_as_concept_id,
            c_value,
            self._table.c.unit_concept_id,
            func.lead(datetime_col)
            .over(partition_by=self._table.c.person_id, order_by=datetime_col)
            .label("next_datetime"),
        )

        # todo: this creates multiple rows if multiple "height" values exist -- need to use
        #       the latest height value instead (but we have the datetime information, should be easy)
        cte = cte.join(sql_ibw, sql_ibw.c.person_id == self._table.c.person_id)
        cte = self._sql_filter_concept(
            cte, override_concept_id=OMOPConcepts.TIDAL_VOLUME_ON_VENTILATOR.value
        ).cte("RankedMeasurements")

        sql_value = self._value.to_sql(
            table=cte,
            column_name=c_value.name,
            with_unit=False,
        )

        conditional_column = create_conditional_interval_column(sql_value)

        query = select(
            cte.c.person_id,
            conditional_column.label("interval_type"),
            cte.c.datetime.label("interval_start"),
            func.least(
                cte.c.datetime + time_threshold_param,
                func.coalesce(
                    cte.c.next_datetime, cte.c.datetime + time_threshold_param
                ),
            ).label("interval_end"),
        )

        return query

    def _filter_days_with_all_values_valid(
        self, query: Select, sql_value: ColumnElement = None
    ) -> Select:
        return super()._filter_days_with_all_values_valid(query, sql_value)

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
