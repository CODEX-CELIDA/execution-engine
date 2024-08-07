import json
from datetime import datetime, timedelta

import pendulum

from execution_engine.omop.db.omop.tables import (
    ConditionOccurrence,
    DrugExposure,
    Measurement,
    Observation,
    ProcedureOccurrence,
    VisitDetail,
    VisitOccurrence,
)
from execution_engine.util.interval import IntervalType
from execution_engine.util.value import ValueConcept, ValueNumber
from tests.functions import (
    create_condition,
    create_drug_exposure,
    create_measurement,
    create_observation,
    create_procedure,
    create_visit,
)


class BaseDataGenerator:
    """
    A base class for generating test data for various entities in the software.
    Intended to be subclassed for generating specific types of data.
    Contains common attributes and methods useful for generating data.
    """

    concept_id: int
    static: bool = False
    start_datetime: datetime
    end_datetime: datetime | None = None
    value: ValueNumber | ValueConcept | None = None
    comparator: str | None = None
    missing_data_type: IntervalType

    def _get_value(self, value: ValueNumber | ValueConcept, valid: bool):
        if isinstance(value, ValueNumber):
            value_as_number = value.value

            if valid:
                if self.comparator == "<":
                    value_as_number -= 1
                elif self.comparator == ">":
                    value_as_number += 1
            else:
                if self.comparator in ["<", "<="]:
                    value_as_number += 1
                elif self.comparator in [">", ">=", "="]:
                    value_as_number -= 1

            return value_as_number

        elif isinstance(value, ValueConcept):
            if valid:
                return value.concept_id
            else:
                return value.concept_id + 100000000000
        else:
            raise ValueError(f"Invalid value type: {self.value}")

    def generate_data(
        self, vo: VisitOccurrence, valid=True
    ) -> list[
        ConditionOccurrence
        | ProcedureOccurrence
        | DrugExposure
        | Measurement
        | Observation
        | VisitDetail
    ]:
        raise NotImplementedError("Must be implemented by subclass.")

    def to_dict(self, vo: VisitOccurrence, valid=True) -> list[dict]:
        raise NotImplementedError("Must be implemented by subclass.")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return hash(self) == hash(other)

    def __hash__(self):
        all_properties = dir(self)
        attributes = {
            prop: str(getattr(self, prop))
            for prop in all_properties
            if not prop.startswith("__") and not callable(getattr(self, prop))
        }
        return hash((self.__class__, json.dumps(attributes)))

    def __and__(self, other):
        from tests._testdata.generator.composite import AndGenerator

        return AndGenerator(self, other)

    def __or__(self, other):
        from tests._testdata.generator.composite import OrGenerator

        return OrGenerator(self, other)

    def __invert__(self):
        from tests._testdata.generator.composite import NotGenerator

        return NotGenerator(self)

    @property
    def name(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"{self.name}"


class ProcedureGenerator(BaseDataGenerator):
    start_datetime = pendulum.parse("2023-03-02 12:00:00+01:00")
    end_datetime = pendulum.parse("2023-03-03 12:00:00+01:00")

    duration_threshold_hours: ValueNumber | None = None
    missing_data_type = IntervalType.NEGATIVE

    def generate_data(
        self, vo: VisitOccurrence, valid=True
    ) -> list[ProcedureOccurrence]:
        if self.duration_threshold_hours is not None:
            add = 1 if valid else -1

            start_datetime = self.start_datetime
            end_datetime = self.start_datetime + timedelta(
                hours=self.duration_threshold_hours + add
            )

        else:
            if valid:
                start_datetime = self.start_datetime
                end_datetime = self.end_datetime
            else:
                start_datetime = vo.visit_end_datetime + timedelta(days=1)
                end_datetime = start_datetime + timedelta(days=1)

        procedure = create_procedure(
            vo=vo,
            procedure_concept_id=self.concept_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )

        return [procedure]

    def to_dict(self, vo: VisitOccurrence, valid=True) -> list[dict]:
        procedures = self.generate_data(vo, valid=valid)

        return [
            {
                "person_id": procedure.person_id,
                "concept_id": procedure.procedure_concept_id,
                "start_datetime": procedure.procedure_datetime,
                "end_datetime": procedure.procedure_end_datetime,
                "valid": valid,
                "static": self.static,
                "missing_data_type": self.missing_data_type,
                "type": "procedure",
            }
            for procedure in procedures
        ]


class ConditionGenerator(BaseDataGenerator):
    missing_data_type = IntervalType.NEGATIVE

    def generate_data(
        self, vo: VisitOccurrence, valid: bool = True
    ) -> list[ConditionOccurrence]:
        if valid:
            start_datetime = vo.visit_start_datetime
            end_datetime = vo.visit_end_datetime
        else:
            start_datetime = vo.visit_end_datetime + timedelta(days=1)
            end_datetime = start_datetime + timedelta(days=1)

        condition = create_condition(
            vo=vo,
            condition_concept_id=self.concept_id,
            condition_start_datetime=start_datetime,
            condition_end_datetime=end_datetime,
        )
        return [condition]

    def to_dict(self, vo: VisitOccurrence, valid=True) -> list[dict]:
        conditions = self.generate_data(vo, valid=valid)

        return [
            {
                "person_id": condition.person_id,
                "concept_id": condition.condition_concept_id,
                "start_datetime": condition.condition_start_datetime,
                "end_datetime": condition.condition_end_datetime,
                "valid": valid,
                "static": self.static,
                "missing_data_type": self.missing_data_type,
                "type": "condition",
            }
            for condition in conditions
        ]


class MeasurementGenerator(BaseDataGenerator):
    missing_data_type = IntervalType.NO_DATA

    start_datetime = pendulum.parse("2023-03-02 12:00:00+01:00")

    def __init__(
        self,
        value: ValueNumber | ValueConcept | None = None,
        comparator: str | None = None,
    ):
        if value is None:
            assert self.value is not None, "Value must be provided"
        else:
            self.value = value

        if comparator is None:
            assert self.comparator is not None, "Comparator must be provided"
        else:
            self.comparator = comparator

        if isinstance(self.value, ValueNumber):
            if self.comparator not in ["<", "<=", ">", ">=", "="]:
                raise ValueError(f"Invalid comparator: {comparator}")
        elif isinstance(self.value, ValueConcept):
            if self.comparator is not None:
                raise ValueError("Comparator must be None for ValueConcept")
        else:
            raise ValueError(f"Invalid value type: {value}")

    def generate_data(
        self, vo: VisitOccurrence, valid: bool = True
    ) -> list[Measurement]:
        value = self._get_value(self.value, valid)

        start_datetimes = self.start_datetime

        if not isinstance(start_datetimes, (list, tuple, set)):
            start_datetimes = [start_datetimes]

        measurement = [
            create_measurement(
                vo=vo,
                measurement_concept_id=self.concept_id,
                measurement_datetime=start_datetime,
                value_as_number=value,
                unit_concept_id=self.value.unit.concept_id,
            )
            for start_datetime in start_datetimes
        ]

        return measurement

    def to_dict(self, vo: VisitOccurrence, valid=True) -> list[dict]:
        measurements = self.generate_data(vo, valid=valid)

        return [
            {
                "person_id": measurement.person_id,
                "concept_id": measurement.measurement_concept_id,
                "start_datetime": measurement.measurement_datetime,
                "end_datetime": measurement.measurement_datetime,
                "valid": valid,
                "static": self.static,
                "missing_data_type": self.missing_data_type,
                "type": "measurement",
            }
            for measurement in measurements
        ]

    def __repr__(self) -> str:
        return f"{self.name}({self.value}, {self.comparator})"


class ObservationGenerator(MeasurementGenerator):
    missing_data_type = IntervalType.NO_DATA

    start_datetime = pendulum.parse("2023-03-15 12:00:00+01:00")

    def __init__(
        self,
        value: ValueNumber | str | ValueConcept | None = None,
        comparator: str | None = None,
    ):
        if value is not None:
            self.value = value

        if comparator is not None:
            self.comparator = comparator

        if isinstance(self.value, ValueNumber):
            if self.comparator not in ["<", "<=", ">", ">=", "="]:
                raise ValueError(f"Invalid comparator: {comparator}")
        elif isinstance(self.value, (ValueConcept, str)):
            if self.comparator is not None:
                raise ValueError("Comparator must be None for ValueConcept")

    def generate_data(
        self, vo: VisitOccurrence, valid: bool = True
    ) -> list[Observation]:
        if valid:
            value = self.value
        else:
            if self.value is None:
                # this is just a concept, so we can't generate an invalid value
                return []
            else:
                value = self._get_value(self.value, valid)

        unit_concept_id = (
            value.unit.concept_id if isinstance(value, ValueNumber) else None
        )

        observation = create_observation(
            vo=vo,
            observation_concept_id=self.concept_id,
            observation_datetime=self.start_datetime,
            value_as_number=value,
            unit_concept_id=unit_concept_id,
        )

        return [observation]

    def to_dict(self, vo: VisitOccurrence, valid=True) -> list[dict]:
        observations = self.generate_data(vo, valid=valid)

        return [
            {
                "person_id": observation.person_id,
                "concept_id": observation.observation_concept_id,
                "start_datetime": observation.observation_datetime,
                "end_datetime": observation.observation_datetime,
                "valid": valid,
                "static": self.static,
                "missing_data_type": self.missing_data_type,
                "type": "observation",
            }
            for observation in observations
        ]

    def __repr__(self) -> str:
        return f"{self.name}({self.value}, {self.comparator})"


class DrugExposureGenerator(BaseDataGenerator):
    start_datetime = pendulum.parse("2023-03-02 12:00:00+01:00")
    end_datetime = pendulum.parse("2023-03-03 12:00:00+01:00")
    quantity: ValueNumber
    route_concept_id: int | None = None
    doses_per_day: int

    missing_data_type = IntervalType.NEGATIVE

    def __init__(self, quantity: ValueNumber, comparator: str, doses_per_day: int = 1):
        self.quantity = quantity
        self.comparator = comparator
        self.doses_per_day = doses_per_day

    def generate_data(
        self, vo: VisitOccurrence, valid: bool = True
    ) -> list[DrugExposure]:
        quantity = self._get_value(self.quantity, valid) * 2  # over two days
        quantity /= self.doses_per_day

        drug_exposures = []

        for _ in range(self.doses_per_day):
            drug_exposures.append(
                create_drug_exposure(
                    vo=vo,
                    drug_concept_id=self.concept_id,
                    start_datetime=self.start_datetime,
                    end_datetime=self.end_datetime,
                    quantity=quantity,
                    route_concept_id=self.route_concept_id,
                )
            )

        return drug_exposures

    def to_dict(self, vo: VisitOccurrence, valid=True) -> list[dict]:
        drug_exposures = self.generate_data(vo, valid=valid)

        return [
            {
                "person_id": drug_exposure.person_id,
                "concept_id": drug_exposure.drug_concept_id,
                "start_datetime": drug_exposure.drug_exposure_start_datetime,
                "end_datetime": drug_exposure.drug_exposure_end_datetime,
                "valid": valid,
                "static": self.static,
                "missing_data_type": self.missing_data_type,
                "type": "drug_exposure",
            }
            for drug_exposure in drug_exposures
        ]

    def __repr__(self) -> str:
        return f"{self.name}({self.quantity}, {self.comparator}, {self.doses_per_day} per day)"


class VisitGenerator(BaseDataGenerator):
    start_datetime = pendulum.parse("2023-03-02 12:00:00+01:00")
    end_datetime = pendulum.parse("2023-03-18 12:00:00+01:00")
    missing_data_type = IntervalType.NEGATIVE
    invalid_concept_id: int

    def generate_data(self, person_id: int, valid=True) -> list[VisitOccurrence]:
        if valid:
            concept_id = self.concept_id
        else:
            concept_id = self.invalid_concept_id

        visit = create_visit(
            person_id=person_id,
            visit_start_datetime=self.start_datetime,
            visit_end_datetime=self.end_datetime,
            visit_concept_id=concept_id,
        )
        return [visit]

    def to_dict(self, person_id: int, valid=True) -> list[dict]:
        visits = self.generate_data(person_id, valid=valid)

        return [
            {
                "person_id": visit.person_id,
                "concept_id": visit.visit_concept_id,
                "start_datetime": visit.visit_start_datetime,
                "end_datetime": visit.visit_end_datetime,
                "valid": valid,
                "static": self.static,
                "missing_data_type": self.missing_data_type,
                "type": "visit",
            }
            for visit in visits
        ]

    def __repr__(self) -> str:
        return f"{self.name}"
