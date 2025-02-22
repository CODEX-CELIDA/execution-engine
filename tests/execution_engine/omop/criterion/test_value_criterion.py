from abc import ABC, abstractmethod

import pendulum
import pytest

from execution_engine.omop.concepts import Concept
from execution_engine.util.value import ValueConcept, ValueNumber
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

    @pytest.mark.parametrize(
        "times",
        [
            ["2023-03-04 06:00:00"],
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
        observation_window,
        times,
        criterion_value,
        criterion_fixture,
    ):
        self.perform_test(
            values_db=self.VALUE_NUMERIC,
            person_visit=person_visit,
            db_session=db_session,
            criterion_execute_func=criterion_execute_func,
            observation_window=observation_window,
            times=times,
            criterion_concept=criterion_fixture["concept"],
            criterion_unit_concept=criterion_fixture["unit_concept"],
            criterion_value=criterion_value,
            criterion_fixture=criterion_fixture,
        )

    @pytest.mark.parametrize(
        "times,values_db",
        [
            (  # same time
                (
                    ("2023-03-04 12:00:00+01:00"),
                    ("2023-03-04 12:00:00+01:00"),
                    ("2023-03-04 12:00:00+01:00"),
                ),
                (VALUE_NUMERIC + 1, VALUE_NUMERIC, VALUE_NUMERIC - 1),
            ),
        ],
    )  # time ranges used in the database entry
    @pytest.mark.parametrize(
        "criterion_value",
        [
            {"value": f">={VALUE_NUMERIC-1}", "match": True},
            {"value": f">={VALUE_NUMERIC}", "match": False},
            {"value": f"<={VALUE_NUMERIC +1}", "match": True},
            {"value": f"{VALUE_NUMERIC}", "match": False},
            {"value": f"{VALUE_NUMERIC-0.1}", "match": False},
            {"value": f"{VALUE_NUMERIC-1}-{VALUE_NUMERIC+1}", "match": True},
            {"value": f"{VALUE_NUMERIC+1}-{VALUE_NUMERIC+10}", "match": False},
            {"value": f"-{VALUE_NUMERIC}--0.1", "match": False},
        ],
    )  # values used in the criterion
    def test_multiple_values_same_time(
        self,
        person_visit,
        db_session,
        criterion_execute_func,
        observation_window,
        times,
        values_db,
        criterion_value,
        criterion_fixture,
    ):
        self.perform_test(
            values_db=values_db,
            person_visit=person_visit,
            db_session=db_session,
            criterion_execute_func=criterion_execute_func,
            observation_window=observation_window,
            times=times,
            criterion_concept=criterion_fixture["concept"],
            criterion_unit_concept=criterion_fixture["unit_concept"],
            criterion_value=criterion_value,
            criterion_fixture=criterion_fixture,
        )

    @pytest.mark.parametrize(
        "times", [["2023-03-04 18:00:00"]]
    )  # time ranges used in the database entry
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
        observation_window,
        times,
        criterion_value,
        criterion_fixture,
    ):
        with pytest.raises(ValueError):
            self.perform_test(
                values_db=self.VALUE_NUMERIC,
                person_visit=person_visit,
                db_session=db_session,
                criterion_execute_func=criterion_execute_func,
                observation_window=observation_window,
                times=times,
                criterion_concept=criterion_fixture["concept"],
                criterion_unit_concept=criterion_fixture["unit_concept"],
                criterion_value=criterion_value,
                criterion_fixture=criterion_fixture,
            )

    @pytest.mark.parametrize(
        "times",
        [
            ["2023-03-04 06:00:00"],
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
        observation_window,
        times,
        criterion_value,
        criterion_fixture,
    ):
        self.perform_test(
            values_db=value_concept,
            person_visit=person_visit,
            db_session=db_session,
            criterion_execute_func=criterion_execute_func,
            observation_window=observation_window,
            times=times,
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
        values_db,
        person_visit,
        db_session,
        criterion_execute_func,
        observation_window,
        times,
        criterion_concept,
        criterion_unit_concept,
        criterion_value,
        criterion_fixture,
    ):
        p, vo = person_visit[0]

        if not isinstance(values_db, (list, tuple)):
            values_db = [values_db] * len(times)

        assert len(values_db) == len(times)

        times = [pendulum.parse(time) for time in times]

        for value_db, time in zip(values_db, times):
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
        df = criterion_execute_func(concept=criterion_concept, value=value).query(
            f"{p.person_id} == person_id"
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

        assert set(df["valid_date"].dt.date) == valid_dates

    @pytest.mark.parametrize(
        "test_cases",
        [
            [
                {
                    "times": ["2023-03-04 06:00:00"],
                    "criterion_value": f">={VALUE_NUMERIC}",
                    "value_db": [VALUE_NUMERIC],
                    "expected": {"2023-03-04"},
                },
                {
                    "times": [  # multiple per one day (first value smaller than threshold)
                        "2023-03-04 00:00:00",
                        "2023-03-04 03:00:00",
                        "2023-03-04 06:00:00",
                    ],
                    "criterion_value": f">={VALUE_NUMERIC}",
                    "value_db": [VALUE_NUMERIC - 1, VALUE_NUMERIC, VALUE_NUMERIC + 1],
                    "expected": {},  # because we test "all values valid per day"
                },
                {
                    "times": [  # multiple days
                        "2023-03-04 00:00:00",
                        "2023-03-15 01:00:00",
                        "2023-03-19 03:00:00",
                    ],
                    "criterion_value": f">={VALUE_NUMERIC}",
                    "value_db": [VALUE_NUMERIC + 1, VALUE_NUMERIC, VALUE_NUMERIC - 1],
                    "expected": {"2023-03-04", "2023-03-15"},
                },
            ],
        ],
    )
    def test_value_multiple_persons(
        self,
        person_visit,
        db_session,
        criterion_execute_func,
        observation_window,
        concept,
        unit_concept,
        test_cases,
    ):
        vos = [pv[1] for pv in person_visit]

        for vo, tc in zip(vos, test_cases):
            times = [pendulum.parse(time) for time in tc["times"]]

            for value, time in zip(tc["value_db"], times):
                c = self.create_value(
                    visit_occurrence=vo,
                    concept_id=concept.concept_id,
                    datetime=time,
                    value=value,
                    unit_concept_id=unit_concept.concept_id,
                )
                db_session.add(c)

            db_session.commit()

        value = ValueNumber.parse(tc["criterion_value"], unit=unit_concept)

        # run criterion against db
        df = criterion_execute_func(concept=concept, value=value)

        for vo, tc in zip(vos, test_cases):
            df_person = df.query(f"{vo.person_id} == person_id")
            valid_dates = self.date_points(tc["expected"])

            assert set(df_person["valid_date"].dt.date) == valid_dates
