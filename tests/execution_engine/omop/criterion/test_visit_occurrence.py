import pendulum
import pytest

from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.visit_occurrence import VisitOccurrence
from tests._fixtures.omop_fixture import disable_postgres_trigger
from tests._testdata.concepts import INPATIENT_VISIT, INTENSIVE_CARE
from tests.execution_engine.omop.criterion.test_criterion import TestCriterion, date_set
from tests.functions import create_visit


class TestVisitOccurrence(TestCriterion):
    @pytest.fixture
    def concept(self):
        return Concept(
            concept_id=32037,
            concept_name="Intensive Care",
            domain_id="Visit",
            vocabulary_id="Visit",
            concept_class_id="Visit",
            standard_concept="S",
            concept_code="OMOP4822460",
            invalid_reason=None,
        )

    @pytest.fixture
    def concept_no_match(self):
        return Concept(
            concept_id=581478,
            concept_name="Ambulance Visit",
            domain_id="Visit",
            vocabulary_id="Visit",
            concept_class_id="Visit",
            standard_concept="S",
            concept_code="OMOP4822457",
            invalid_reason=None,
        )

    @pytest.fixture
    def criterion_class(self):
        return VisitOccurrence

    @pytest.fixture
    def base_table(
        self,
        person,
        db_session,
        base_criterion,
        observation_window,
    ):
        """
        Execute the base criterion

        The TestCriterion.base_table uses person_visit as fixture, which inserts a visit for each person.
        This is not suitable for testing VisitOccurrence, because we need insert specific visits for each test case,
        and we must not have other visits in the table.
        Therefor, the base_table fixture is overwritten here, and uses only person, not person_visit fixture.
        """
        with self.execute_base_criterion(
            base_criterion, db_session, observation_window
        ):
            yield

    @pytest.mark.parametrize(
        "visit_datetimes,expected",
        [
            (  # non overlapping
                [
                    (("2023-03-04 00:00:00", "2023-03-04 02:00:00"), INTENSIVE_CARE),
                    (("2023-03-04 03:00:00", "2023-03-04 05:00:00"), INPATIENT_VISIT),
                    (("2023-03-04 06:00:00", "2023-03-04 08:00:00"), INPATIENT_VISIT),
                ],
                {"2023-03-04"},
            ),
            (  # no intensive care
                [
                    (("2023-03-04 00:00:00", "2023-03-04 02:00:00"), INPATIENT_VISIT),
                    (("2023-03-04 03:00:00", "2023-03-04 05:00:00"), INPATIENT_VISIT),
                    (("2023-03-04 06:00:00", "2023-03-04 08:00:00"), INPATIENT_VISIT),
                ],
                {},
            ),
            (  # exact overlap
                [
                    (("2023-03-04 00:00:00", "2023-03-04 02:00:00"), INTENSIVE_CARE),
                    (("2023-03-04 02:00:00", "2023-03-04 04:00:00"), INTENSIVE_CARE),
                    (("2023-03-04 04:00:00", "2023-03-04 06:00:00"), INTENSIVE_CARE),
                ],
                {"2023-03-04"},
            ),
            (  # partial overlap
                [
                    (("2023-03-04 00:00:00", "2023-03-04 02:00:00"), INTENSIVE_CARE),
                    (("2023-03-04 01:30:00", "2023-03-04 03:30:00"), INTENSIVE_CARE),
                    (("2023-03-04 04:00:00", "2023-03-04 06:00:00"), INTENSIVE_CARE),
                ],
                {"2023-03-04"},
            ),
            (  # contained overlap
                [
                    (("2023-03-04 00:00:00", "2023-03-04 06:00:00"), INTENSIVE_CARE),
                    (("2023-03-04 01:00:00", "2023-03-04 02:00:00"), INTENSIVE_CARE),
                    (("2023-03-04 03:00:00", "2023-03-04 04:00:00"), INTENSIVE_CARE),
                ],
                {"2023-03-04"},
            ),
            (  # non overlapping
                [
                    (
                        ("2023-03-01 08:00:00", "2023-03-03 16:00:00"),
                        INTENSIVE_CARE,
                    ),  # 3
                    (
                        ("2023-03-04 09:00:00", "2023-03-06 15:00:00"),
                        INTENSIVE_CARE,
                    ),  # +3
                    (
                        ("2023-03-07 10:00:00", "2023-03-09 18:00:00"),
                        INTENSIVE_CARE,
                    ),  # +3
                ],
                {
                    "2023-03-01",
                    "2023-03-02",
                    "2023-03-03",
                    "2023-03-04",
                    "2023-03-05",
                    "2023-03-06",
                    "2023-03-07",
                    "2023-03-08",
                    "2023-03-09",
                },
            ),
            (  # non overlapping - not all intensive care
                [
                    (
                        ("2023-03-01 08:00:00", "2023-03-03 16:00:00"),
                        INTENSIVE_CARE,
                    ),  # 3
                    (
                        ("2023-03-04 09:00:00", "2023-03-06 15:00:00"),
                        INPATIENT_VISIT,
                    ),  # +3
                    (
                        ("2023-03-07 10:00:00", "2023-03-09 18:00:00"),
                        INTENSIVE_CARE,
                    ),  # +3
                ],
                {
                    "2023-03-01",
                    "2023-03-02",
                    "2023-03-03",
                    "2023-03-07",
                    "2023-03-08",
                    "2023-03-09",
                },
            ),
            (  # exaxt overlap
                [
                    (
                        ("2023-03-01 08:00:00", "2023-03-03 16:00:00"),
                        INTENSIVE_CARE,
                    ),  # 3
                    (
                        ("2023-03-03 16:00:00", "2023-03-05 23:59:00"),
                        INTENSIVE_CARE,
                    ),  # +2
                    (
                        ("2023-03-05 23:59:00", "2023-03-08 11:00:00"),
                        INTENSIVE_CARE,
                    ),  # +2
                ],
                {
                    "2023-03-01",
                    "2023-03-02",
                    "2023-03-03",
                    "2023-03-04",
                    "2023-03-05",
                    "2023-03-06",
                    "2023-03-07",
                    "2023-03-08",
                },
            ),
            (  # partial overlap
                [
                    (
                        ("2023-03-01 08:00:00", "2023-03-03 16:00:00"),
                        INTENSIVE_CARE,
                    ),  # 3
                    (
                        ("2023-03-03 12:00:00", "2023-03-05 20:00:00"),
                        INTENSIVE_CARE,
                    ),  # +2
                    (
                        ("2023-03-06 10:00:00", "2023-03-08 18:00:00"),
                        INTENSIVE_CARE,
                    ),  # +3
                ],
                {
                    "2023-03-01",
                    "2023-03-02",
                    "2023-03-03",
                    "2023-03-04",
                    "2023-03-05",
                    "2023-03-06",
                    "2023-03-07",
                    "2023-03-08",
                },
            ),
            (  # partial overlap - not all intensive care
                [
                    (
                        ("2023-03-01 08:00:00", "2023-03-03 16:00:00"),
                        INTENSIVE_CARE,
                    ),  # 3
                    (
                        ("2023-03-03 12:00:00", "2023-03-05 20:00:00"),
                        INPATIENT_VISIT,
                    ),  # +2
                    (
                        ("2023-03-06 10:00:00", "2023-03-08 18:00:00"),
                        INTENSIVE_CARE,
                    ),  # +3
                ],
                {
                    "2023-03-01",
                    "2023-03-02",
                    "2023-03-03",
                    "2023-03-06",
                    "2023-03-07",
                    "2023-03-08",
                },
            ),
            (  # contained overlap
                [
                    (
                        ("2023-03-01 08:00:00", "2023-03-09 18:00:00"),
                        INTENSIVE_CARE,
                    ),  # 9
                    (
                        ("2023-03-03 10:00:00", "2023-03-05 12:00:00"),
                        INTENSIVE_CARE,
                    ),  # +0
                    (
                        ("2023-03-06 14:00:00", "2023-03-08 16:00:00"),
                        INTENSIVE_CARE,
                    ),  # +0
                ],
                {
                    "2023-03-01",
                    "2023-03-02",
                    "2023-03-03",
                    "2023-03-04",
                    "2023-03-05",
                    "2023-03-06",
                    "2023-03-07",
                    "2023-03-08",
                    "2023-03-09",
                },
            ),
            (  # contained overlap
                [
                    (
                        ("2023-03-01 08:00:00", "2023-03-09 18:00:00"),
                        INPATIENT_VISIT,
                    ),  # 9
                    (
                        ("2023-03-03 10:00:00", "2023-03-05 12:00:00"),
                        INPATIENT_VISIT,
                    ),  # +0
                    (
                        ("2023-03-06 14:00:00", "2023-03-08 16:00:00"),
                        INTENSIVE_CARE,
                    ),  # +0
                ],
                {"2023-03-06", "2023-03-07", "2023-03-08"},
            ),
            (  # before observation window (same day)
                [
                    (("2023-02-01 00:00:00", "2023-03-01 02:00:00"), INTENSIVE_CARE),
                ],
                {},
            ),
            (  # before observation window (previous day)
                [
                    (("2023-02-01 00:00:00", "2023-02-27 02:00:00"), INTENSIVE_CARE),
                ],
                {},
            ),
            (  # after observation window (same day)
                # note that proper time zone is required, otherwise (if given in utc), it will be the next day (see next example)
                [
                    (
                        ("2023-03-31 23:59:00+02:00", "2023-04-01 02:00:00"),
                        INTENSIVE_CARE,
                    ),
                ],
                {},
            ),
            (  # after observation window (next day)
                [
                    (
                        (
                            "2023-03-31 23:59:00+00:00",  # note: this is 2023-04-01 01:59:00+02:00 i.e. next day in Europe/Berlin
                            "2023-04-01 02:00:00",
                        ),
                        INTENSIVE_CARE,
                    ),
                ],
                {},
            ),
            (  # during begin observation window
                [
                    (("2023-02-01 00:00:00", "2023-03-04 02:00:00"), INTENSIVE_CARE),
                ],
                {"2023-03-01", "2023-03-02", "2023-03-03", "2023-03-04"},
            ),
            (  # during end observation window
                [
                    (
                        ("2023-03-29 23:59:00+02:00", "2023-04-01 02:00:00"),
                        INTENSIVE_CARE,
                    ),
                ],
                {"2023-03-29", "2023-03-30", "2023-03-31"},
            ),
        ],
    )
    def test_visit_occurrence(
        self,
        person,
        concept,
        db_session,
        base_criterion,
        base_table,
        criterion_execute_func,
        visit_datetimes,
        expected,
        observation_window,
    ):
        for (
            visit_start_datetime,
            visit_end_datetime,
        ), visit_concept_id in visit_datetimes:
            vo = create_visit(
                person_id=person[0].person_id,
                visit_start_datetime=pendulum.parse(visit_start_datetime),
                visit_end_datetime=pendulum.parse(visit_end_datetime),
                visit_concept_id=visit_concept_id,
            )
            db_session.add(vo)

        db_session.commit()

        # need to disable postgres trigger to avoid constraint violation due to overlapping intervals in testdata
        with disable_postgres_trigger(db_session):
            # execute base criterion because its results (the BASE person_ids) are used as filter in the criterion
            with self.execute_base_criterion(
                base_criterion, db_session, observation_window
            ):
                df = criterion_execute_func(concept=concept)

        assert set(df["valid_date"].dt.date) == date_set(expected)

    @pytest.mark.parametrize(
        "test_cases",
        [
            (
                [
                    {
                        "time_range": [  # non-overlapping
                            ("2023-03-03 08:00:00", "2023-03-03 16:00:00"),
                            ("2023-03-04 09:00:00", "2023-03-06 15:00:00"),
                            ("2023-03-08 10:00:00", "2023-03-09 18:00:00"),
                        ],
                        "expected": {
                            "2023-03-03",
                            "2023-03-04",
                            "2023-03-05",
                            "2023-03-06",
                            "2023-03-08",
                            "2023-03-09",
                        },
                    },
                    {
                        "time_range": [  # exact overlap
                            ("2023-03-01 08:00:00", "2023-03-02 16:00:00"),
                            ("2023-03-02 16:00:00", "2023-03-03 23:59:00"),
                            ("2023-03-03 23:59:00", "2023-03-04 11:00:00"),
                        ],
                        "expected": {
                            "2023-03-01",
                            "2023-03-02",
                            "2023-03-03",
                            "2023-03-04",
                        },
                    },
                    {
                        "time_range": [  # overlap by some margin
                            ("2023-03-01 08:00:00", "2023-03-03 16:00:00"),
                            ("2023-03-03 12:00:00", "2023-03-05 20:00:00"),
                            ("2023-03-06 10:00:00", "2023-03-08 18:00:00"),
                        ],
                        "expected": {
                            "2023-03-01",
                            "2023-03-02",
                            "2023-03-03",
                            "2023-03-04",
                            "2023-03-05",
                            "2023-03-06",
                            "2023-03-07",
                            "2023-03-08",
                        },
                    },
                ]
            )
        ],
    )
    def test_visit_occurrence_multiple_patients(
        self,
        person,
        db_session,
        concept,
        base_criterion,
        observation_window,
        criterion_execute_func,
        test_cases,
    ):
        # need to disable postgres trigger to avoid constraint violation due to overlapping intervals in testdata
        with disable_postgres_trigger(db_session):
            for tc, p in zip(test_cases, person):
                for (
                    visit_start_datetime,
                    visit_end_datetime,
                ) in tc["time_range"]:
                    vo = create_visit(
                        person_id=p.person_id,
                        visit_start_datetime=pendulum.parse(visit_start_datetime),
                        visit_end_datetime=pendulum.parse(visit_end_datetime),
                        visit_concept_id=INTENSIVE_CARE,
                    )
                    db_session.add(vo)

            db_session.commit()

            with self.execute_base_criterion(
                base_criterion, db_session, observation_window
            ):
                # run criterion against db
                df = criterion_execute_func(concept=concept)

        for tc, p in zip(test_cases, person):
            assert set(
                df.query(f"person_id=={p.person_id}")["valid_date"].dt.date
            ) == date_set(tc["expected"])
