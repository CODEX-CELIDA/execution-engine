from abc import ABC, abstractmethod

import pandas as pd
import pendulum
import pytest

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.util import ValueConcept, ValueNumber
from tests.execution_engine.omop.criterion.test_criterion import TestCriterion


class ValueCriterion(TestCriterion, ABC):
    VALUE_NUMERIC = 20.5

    @pytest.fixture
    def concept(self):
        raise NotImplementedError(
            "Subclasses should override this method to provide their own fixture"
        )

    @pytest.fixture
    def concept_no_match(self):
        raise NotImplementedError(
            "Subclasses should override this method to provide their own fixture"
        )

    @pytest.fixture
    def unit_concept(self):
        raise NotImplementedError(
            "Subclasses should override this method to provide their own fixture"
        )

    @pytest.fixture
    def unit_concept_no_match(self):
        raise NotImplementedError(
            "Subclasses should override this method to provide their own fixture"
        )

    @pytest.fixture
    def criterion_class(self):
        raise NotImplementedError(
            "Subclasses should override this method to provide their own fixture"
        )

    @pytest.fixture
    def value_concept(self):
        return Concept(
            concept_id=46237106,
            concept_name="OK",
            domain_id="Meas Value",
            vocabulary_id="LOINC",
            concept_class_id="Answer",
            standard_concept="S",
            concept_code="LA22024-6",
            invalid_reason=None,
        )

    @pytest.fixture
    def value_concept_no_match(self):
        return Concept(
            concept_id=45878591,
            concept_name="Out of range",
            domain_id="Meas Value",
            vocabulary_id="LOINC",
            concept_class_id="Answer",
            standard_concept="S",
            concept_code="LA18593-6",
            invalid_reason=None,
        )

    @abstractmethod
    def create_value(
        self, visit_occurrence, concept_id, datetime, value, unit_concept_id
    ):
        raise NotImplementedError(
            "Subclasses should override this method to provide their own fixture"
        )

    @pytest.fixture
    def criterion_fixture(
        self,
        concept,
        concept_no_match,
        unit_concept,
        unit_concept_no_match,
        criterion_class,
        value_concept,
        value_concept_no_match,
    ) -> dict:
        return {
            "concept": concept,
            "concept_no_match": concept_no_match,
            "unit_concept": unit_concept,
            "unit_concept_no_match": unit_concept_no_match,
            "criterion_class": criterion_class,
            "value_concept": value_concept,
            "value_concept_no_match": value_concept_no_match,
        }

    @pytest.fixture
    def criterion_execute_func(
        self, base_table, db_session, criterion_class, observation_window
    ):
        def _create_value(
            concept: Concept, value: ValueNumber | ValueConcept, exclude: bool
        ) -> pd.DataFrame:
            criterion = criterion_class(
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
                params=observation_window.dict(),
            )
            df["valid_date"] = pd.to_datetime(df["valid_date"])

            return df

        return _create_value

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
        "criterion_value",
        [
            {"value": f">={VALUE_NUMERIC}", "match": True},
            {"value": f"{VALUE_NUMERIC}", "match": True},
            {"value": f"{VALUE_NUMERIC-0.1}", "match": False},
            {"value": f"{VALUE_NUMERIC}-30.5", "match": True},
            {"value": f"{VALUE_NUMERIC+1}-{VALUE_NUMERIC+10}", "match": False},
            {"value": f"-{VALUE_NUMERIC}--0.1", "match": False},
        ],
    )  # values used in the criterion
    def test_value_numeric(
        self,
        person_visit,
        db_session,
        criterion_execute_func,
        times,
        exclude,
        criterion_value,
        criterion_fixture,
    ):
        self.perform_test(
            value_db=self.VALUE_NUMERIC,
            person_visit=person_visit,
            db_session=db_session,
            criterion_execute_func=criterion_execute_func,
            times=times,
            exclude=exclude,
            criterion_concept=criterion_fixture["concept"],
            criterion_unit_concept=criterion_fixture["unit_concept"],
            criterion_value=criterion_value,
            criterion_fixture=criterion_fixture,
        )

    @pytest.mark.parametrize(
        "times", [["2023-03-04 18:00:00"]]
    )  # time ranges used in the database entry
    @pytest.mark.parametrize("exclude", [True, False])  # exclude used in the criterion
    @pytest.mark.parametrize("criterion_concept_name", ["concept", "concept_no_match"])
    @pytest.mark.parametrize(
        "criterion_unit_concept_name", ["unit_concept", "unit_concept_no_match"]
    )
    @pytest.mark.parametrize(
        "criterion_value",
        [
            {"value": f"{VALUE_NUMERIC}", "match": True},
        ],
    )  # values used in the criterion
    def test_value_numeric_concept_no_match(
        self,
        person_visit,
        db_session,
        criterion_execute_func,
        times,
        exclude,
        criterion_concept_name,
        criterion_unit_concept_name,
        criterion_value,
        criterion_fixture,
    ):
        self.perform_test(
            value_db=self.VALUE_NUMERIC,
            person_visit=person_visit,
            db_session=db_session,
            criterion_execute_func=criterion_execute_func,
            times=times,
            exclude=exclude,
            criterion_concept=criterion_fixture[criterion_concept_name],
            criterion_unit_concept=criterion_fixture[criterion_unit_concept_name],
            criterion_value=criterion_value,
            criterion_fixture=criterion_fixture,
        )

    @pytest.mark.parametrize(
        "times", [["2023-03-04 18:00:00"]]
    )  # time ranges used in the database entry
    @pytest.mark.parametrize("exclude", [True])  # exclude used in the criterion
    @pytest.mark.parametrize(
        "criterion_value",
        [
            {"value": f">{VALUE_NUMERIC}", "match": "error"},
            {"value": f"<{VALUE_NUMERIC}", "match": "error"},
        ],
    )  # values used in the criterion
    def test_value_numeric_error(
        self,
        person_visit,
        db_session,
        criterion_execute_func,
        times,
        exclude,
        criterion_value,
        criterion_fixture,
    ):
        with pytest.raises(ValueError):
            self.perform_test(
                value_db=self.VALUE_NUMERIC,
                person_visit=person_visit,
                db_session=db_session,
                criterion_execute_func=criterion_execute_func,
                times=times,
                exclude=exclude,
                criterion_concept=criterion_fixture["concept"],
                criterion_unit_concept=criterion_fixture["unit_concept"],
                criterion_value=criterion_value,
                criterion_fixture=criterion_fixture,
            )

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
        "criterion_value",
        [  # value used in the criterion
            {"fixture": "value_concept", "match": True},
            {"fixture": "value_concept_no_match", "match": False},
        ],
    )
    def test_value_concept(
        self,
        person_visit,
        db_session,
        value_concept,
        criterion_execute_func,
        times,
        exclude,
        criterion_value,
        criterion_fixture,
    ):
        self.perform_test(
            value_db=value_concept,
            person_visit=person_visit,
            db_session=db_session,
            criterion_execute_func=criterion_execute_func,
            times=times,
            exclude=exclude,
            criterion_concept=criterion_fixture["concept"],
            criterion_unit_concept=criterion_fixture["unit_concept"],
            criterion_value={
                "value": criterion_fixture[criterion_value["fixture"]],
                "match": criterion_value["match"],
            },
            criterion_fixture=criterion_fixture,
        )

    def perform_test(
        self,
        value_db,
        person_visit,
        db_session,
        criterion_execute_func,
        times,
        exclude,
        criterion_concept,
        criterion_unit_concept,
        criterion_value,
        criterion_fixture,
    ):
        _, vo = person_visit

        times = [pendulum.parse(time) for time in times]

        for time in times:
            c = self.create_value(
                visit_occurrence=vo,
                concept_id=criterion_fixture["concept"].concept_id,
                datetime=time,
                value=value_db,
                unit_concept_id=criterion_fixture["unit_concept"].concept_id,
            )
            db_session.add(c)

        db_session.commit()

        if isinstance(criterion_value["value"], str):
            value = ValueNumber.parse(
                criterion_value["value"], unit=criterion_unit_concept
            )
        elif isinstance(criterion_value["value"], Concept):
            value = ValueConcept(value=criterion_value["value"])
        else:
            raise ValueError(f"Unknown value type: {type(criterion_value['value'])}")

        # run criterion against db
        df = criterion_execute_func(
            concept=criterion_concept, value=value, exclude=exclude
        )

        criterion_matches = (
            criterion_value["match"]
            & (criterion_concept == criterion_fixture["concept"])
            & (criterion_unit_concept == criterion_fixture["unit_concept"])
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
