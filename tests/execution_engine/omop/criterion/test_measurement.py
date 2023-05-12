import pandas as pd
import pendulum
import pytest

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.measurement import Measurement
from execution_engine.util import ValueNumber
from tests.execution_engine.omop.criterion.test_criterion import TestCriterion
from tests.functions import create_measurement


class TestMeasurement(TestCriterion):

    concept_tv = Concept(
        concept_id=21490854,
        concept_name="Tidal volume Ventilator --on ventilator",
        domain_id="Measurement",
        vocabulary_id="LOINC",
        concept_class_id="Clinical Observation",
        standard_concept="S",
        concept_code="76222-9",
        invalid_reason=None,
    )

    concept_appt = Concept(
        concept_id=3013466,
        concept_name="aPTT in Blood by Coagulation assay",
        domain_id="Measurement",
        vocabulary_id="LOINC",
        concept_class_id="Clinical Observation",
        standard_concept="S",
        concept_code="3173-2",
        invalid_reason=None,
    )

    concept_unit_ml = Concept(
        concept_id=8587,
        concept_name="milliliter",
        domain_id="Unit",
        vocabulary_id="UCUM",
        concept_class_id="Unit",
        standard_concept="S",
        concept_code="mL",
        invalid_reason=None,
    )

    concept_unit_kg = Concept(
        concept_id=9529,
        concept_name="kilogram",
        domain_id="Unit",
        vocabulary_id="UCUM",
        concept_class_id="Unit",
        standard_concept="S",
        concept_code="kg",
        invalid_reason=None,
    )

    CONCEPT = concept_tv  # concept used in the database entry
    UNIT_CONCEPT = concept_unit_kg  # unit concept used in the database entry
    VALUE = 20.5  # value used in the database entries

    @pytest.fixture
    def measurement_criterion(self, base_table, db_session, person_visit):
        def _create_measurement(
            concept: Concept, value: ValueNumber, exclude: bool
        ) -> pd.DataFrame:
            p, vo = person_visit

            criterion = Measurement(
                name="test",
                exclude=exclude,
                category=CohortCategory.POPULATION,
                concept=concept,
                value=value,
                static=None,
            )

            query = criterion.sql_generate(base_table=base_table)

            df = pd.read_sql(
                query,
                db_session.connection(),
                params={
                    "observation_start_datetime": vo.visit_start_datetime,
                    "observation_end_datetime": vo.visit_end_datetime,
                },
            )
            df["valid_date"] = pd.to_datetime(df["valid_date"])

            return df

        return _create_measurement

    @pytest.mark.parametrize(
        "times",
        [
            ["2023-03-04 18:00:00"],
            [  # multiple per one day
                "2023-03-04 00:00:00",
                "2023-03-04 03:00:00",
                "2023-03-04 06:00:00",
            ],
            [  # multiple days
                "2023-03-04 00:00:00",
                "2023-03-06 01:00:00",
                "2023-03-19 03:00:00",
            ],
        ],
    )  # time ranges used in the database entry
    @pytest.mark.parametrize("exclude", [True, False])  # exclude used in the criterion
    @pytest.mark.parametrize(
        "concept_criterion", [CONCEPT]
    )  # concepts used in the criterion
    @pytest.mark.parametrize(
        "unit_concept_criterion", [UNIT_CONCEPT]
    )  # units used in the criterion
    @pytest.mark.parametrize(
        "value_criterion",
        [
            {"value": f">={VALUE}", "match": True},
            {"value": f"{VALUE}", "match": True},
            {"value": f"{VALUE-0.1}", "match": False},
            {"value": f"{VALUE}-30.5", "match": True},
            {"value": f"{VALUE+1}-{VALUE+10}", "match": False},
            {"value": f"-{VALUE}--0.1", "match": False},
        ],
    )  # values used in the criterion
    def test_measurement(
        self,
        person_visit,
        db_session,
        measurement_criterion,
        times,
        exclude,
        concept_criterion,
        unit_concept_criterion,
        value_criterion,
    ):
        self.perform_test(
            person_visit,
            db_session,
            measurement_criterion,
            times,
            exclude,
            concept_criterion,
            unit_concept_criterion,
            value_criterion,
        )

    @pytest.mark.parametrize(
        "times", [["2023-03-04 18:00:00"]]
    )  # time ranges used in the database entry
    @pytest.mark.parametrize("exclude", [True, False])  # exclude used in the criterion
    @pytest.mark.parametrize(
        "concept_criterion", [concept_appt]
    )  # concepts used in the criterion
    @pytest.mark.parametrize(
        "unit_concept_criterion", [concept_unit_kg]
    )  # units used in the criterion
    @pytest.mark.parametrize(
        "value_criterion",
        [
            {"value": f"{VALUE}", "match": True},
        ],
    )  # values used in the criterion
    def test_measurement_no_match(
        self,
        person_visit,
        db_session,
        measurement_criterion,
        times,
        exclude,
        concept_criterion,
        unit_concept_criterion,
        value_criterion,
    ):
        self.perform_test(
            person_visit,
            db_session,
            measurement_criterion,
            times,
            exclude,
            concept_criterion,
            unit_concept_criterion,
            value_criterion,
        )

    @pytest.mark.parametrize(
        "times", [["2023-03-04 18:00:00"]]
    )  # time ranges used in the database entry
    @pytest.mark.parametrize("exclude", [True])  # exclude used in the criterion
    @pytest.mark.parametrize(
        "concept_criterion", [concept_tv]
    )  # concepts used in the criterion
    @pytest.mark.parametrize(
        "unit_concept_criterion", [concept_unit_ml]
    )  # units used in the criterion
    @pytest.mark.parametrize(
        "value_criterion",
        [
            {"value": f">{VALUE}", "match": "error"},
            {"value": f"<{VALUE}", "match": "error"},
        ],
    )  # values used in the criterion
    def test_measurement_value_error(
        self,
        person_visit,
        db_session,
        measurement_criterion,
        times,
        exclude,
        concept_criterion,
        unit_concept_criterion,
        value_criterion,
    ):
        with pytest.raises(ValueError):
            self.perform_test(
                person_visit,
                db_session,
                measurement_criterion,
                times,
                exclude,
                concept_criterion,
                unit_concept_criterion,
                value_criterion,
            )

    def perform_test(
        self,
        person_visit,
        db_session,
        measurement_criterion,
        times,
        exclude,
        concept_criterion,
        unit_concept_criterion,
        value_criterion,
    ):
        _, vo = person_visit

        times = [pendulum.parse(time) for time in times]

        for time in times:
            c = create_measurement(
                vo=vo,
                measurement_concept_id=self.CONCEPT.concept_id,
                datetime=time,
                value_as_number=self.VALUE,
                unit_concept_id=self.UNIT_CONCEPT.concept_id,
            )
            db_session.add(c)

        db_session.commit()

        value = ValueNumber.parse(value_criterion["value"], unit=unit_concept_criterion)

        # run criterion against db
        df = measurement_criterion(
            concept=concept_criterion, value=value, exclude=exclude
        )

        criterion_matches = (
            value_criterion["match"]
            & (concept_criterion == self.CONCEPT)
            & (unit_concept_criterion == self.UNIT_CONCEPT)
        )

        if criterion_matches:
            valid_dates = self.date_points(times)
        else:
            valid_dates = set()

        if exclude:
            valid_dates = self.invert_date_points(
                start_datetime=vo.visit_start_datetime,
                end_datetime=vo.visit_end_datetime,
                subtract=valid_dates,
            )

        assert set(df["valid_date"].dt.date) == valid_dates
