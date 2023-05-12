from abc import ABC

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
    def create_value_func(self):
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
        create_value_func,
    ) -> dict:
        return {
            "concept": concept,
            "concept_no_match": concept_no_match,
            "unit_concept": unit_concept,
            "unit_concept_no_match": unit_concept_no_match,
            "criterion_class": criterion_class,
            "create_value_func": create_value_func,
        }

    @pytest.fixture
    def criterion_execute_func(
        self, base_table, db_session, person_visit, criterion_class
    ):
        def _create_value(
            concept: Concept, value: ValueNumber | ValueConcept, exclude: bool
        ) -> pd.DataFrame:
            p, vo = person_visit

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
                params={
                    "observation_start_datetime": vo.visit_start_datetime,
                    "observation_end_datetime": vo.visit_end_datetime,
                },
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
    def test_concept_no_match(
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
    def test_value_error(
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
            c = criterion_fixture["create_value_func"](
                vo=vo,
                measurement_concept_id=criterion_fixture["concept"].concept_id,
                datetime=time,
                value_as_number=value_db,
                unit_concept_id=criterion_fixture["unit_concept"].concept_id,
            )
            db_session.add(c)

        db_session.commit()

        value = ValueNumber.parse(criterion_value["value"], unit=criterion_unit_concept)

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
