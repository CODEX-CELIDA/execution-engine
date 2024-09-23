import pandas as pd
import pendulum
import pytest
import sqlalchemy.exc
from sqlalchemy import func, select

from execution_engine.omop.db.celida.views import partial_day_coverage
from execution_engine.omop.db.omop.tables import VisitDetail, VisitOccurrence
from execution_engine.settings import get_config, update_config
from tests._fixtures.omop_fixture import disable_postgres_trigger
from tests._testdata import concepts
from tests.execution_engine.omop.criterion.test_criterion import TestCriterion, date_set
from tests.functions import create_visit, create_visit_detail


@pytest.mark.parametrize("use_visit_details", [True, False])
class TestActivePatientsDuringPeriod(TestCriterion):

    @pytest.fixture(autouse=True)
    def configure_visit_details(self, use_visit_details):
        """
        This fixture sets up the configuration before each test.
        It will automatically be used before any other fixture, including base_criterion.
        """
        config = get_config()
        config.episode_of_care_visit_detail = use_visit_details
        update_config(**config.model_dump())

    @staticmethod
    def insert_visits(db_session, person, visit_datetimes, use_visit_details):
        for visit_start_datetime, visit_end_datetime in visit_datetimes:

            # we use a 0 sec duration for visit_occurrence if using for visit details to make sure the visit details
            # table is indeed used
            vo = create_visit(
                person_id=person.person_id,
                visit_start_datetime=pendulum.parse(visit_start_datetime),
                visit_end_datetime=pendulum.parse(
                    visit_end_datetime
                    if not use_visit_details
                    else visit_start_datetime
                ),
                visit_concept_id=concepts.INTENSIVE_CARE,
            )
            db_session.add(vo)
            db_session.commit()

            if use_visit_details:
                vd = create_visit_detail(
                    vo=vo,
                    visit_detail_concept_id=concepts.INTENSIVE_CARE,
                    visit_detail_start_datetime=pendulum.parse(visit_start_datetime),
                    visit_detail_end_datetime=pendulum.parse(visit_end_datetime),
                )
                db_session.add(vd)
                db_session.commit()

    def test_active_patients_during_period(
        self, person, db_session, base_criterion, observation_window, use_visit_details
    ):
        person = person[0]

        from execution_engine.clients import omopdb

        if use_visit_details:
            assert base_criterion._table.original == VisitDetail.__table__
        else:
            assert base_criterion._table.original == VisitOccurrence.__table__

        # used to check for overlapping intervals
        omopdb.enable_interval_check_trigger()

        def count_day_entries(visit_datetimes: list[tuple[str, str]]) -> int:
            """
            Insert visit_occurrence entries into the database for a given person, then apply the base_criterion to
            count the number of days that the person was active during the observation period.
            For each entry in visit_datetimes, the first element is the visit start datetime and the
            second element is the visit end datetime.
            """

            self.insert_visits(db_session, person, visit_datetimes, use_visit_details)
            try:
                with self.execute_base_criterion(
                    base_criterion,
                    db_session,
                    observation_window,
                ):
                    stmt = (
                        select(
                            partial_day_coverage.c.person_id,
                            func.count("*").label("count"),
                        )
                        .where(partial_day_coverage.c.run_id == self.run_id)
                        .group_by(partial_day_coverage.c.person_id)
                    )
                    result = db_session.execute(stmt)
                    count = [dict(row._mapping) for row in result]
            finally:
                # cleanup
                db_session.rollback()  # rollback to avoid constraint violation

                if use_visit_details:
                    db_session.query(VisitDetail).delete()

                db_session.query(VisitOccurrence).delete()
                db_session.commit()

            return count[0]["count"] if len(count) > 0 else 0

        non_overlapping = [
            ("2023-03-04 00:00:00", "2023-03-04 02:00:00"),
            ("2023-03-04 03:00:00", "2023-03-04 05:00:00"),
            ("2023-03-04 06:00:00", "2023-03-04 08:00:00"),
        ]
        assert count_day_entries(non_overlapping) == 1

        exact_overlap = [
            ("2023-03-04 00:00:00", "2023-03-04 02:00:00"),
            ("2023-03-04 02:00:00", "2023-03-04 04:00:00"),
            ("2023-03-04 04:00:00", "2023-03-04 06:00:00"),
        ]
        assert count_day_entries(exact_overlap) == 1

        partial_overlap = [
            ("2023-03-04 00:00:00", "2023-03-04 02:00:00"),
            ("2023-03-04 01:30:00", "2023-03-04 03:30:00"),
            ("2023-03-04 04:00:00", "2023-03-04 06:00:00"),
        ]
        with pytest.raises(
            sqlalchemy.exc.ProgrammingError, match="Overlapping intervals detected"
        ):
            assert count_day_entries(partial_overlap) == 1

        contained_overlap = [
            ("2023-03-04 00:00:00", "2023-03-04 06:00:00"),
            ("2023-03-04 01:00:00", "2023-03-04 02:00:00"),
            ("2023-03-04 03:00:00", "2023-03-04 04:00:00"),
        ]
        with pytest.raises(
            sqlalchemy.exc.ProgrammingError, match="Overlapping intervals detected"
        ):
            assert count_day_entries(contained_overlap) == 1

        non_overlapping = [
            ("2023-03-01 08:00:00", "2023-03-03 16:00:00"),  # 3
            ("2023-03-04 09:00:00", "2023-03-06 15:00:00"),  # +3
            ("2023-03-07 10:00:00", "2023-03-09 18:00:00"),  # +3
        ]
        assert count_day_entries(non_overlapping) == 9

        exact_overlap = [
            ("2023-03-01 08:00:00", "2023-03-03 16:00:00"),  # 3
            ("2023-03-03 16:00:00", "2023-03-05 23:59:00"),  # +2
            ("2023-03-05 23:59:00", "2023-03-08 11:00:00"),  # +2
        ]
        assert count_day_entries(exact_overlap) == 8

        partial_overlap = [
            ("2023-03-01 08:00:00", "2023-03-03 16:00:00"),  # 3
            ("2023-03-03 12:00:00", "2023-03-05 20:00:00"),  # +2
            ("2023-03-06 10:00:00", "2023-03-08 18:00:00"),  # +3
        ]
        with pytest.raises(
            sqlalchemy.exc.ProgrammingError, match="Overlapping intervals detected"
        ):
            assert count_day_entries(partial_overlap) == 8

        contained_overlap = [
            ("2023-03-01 08:00:00", "2023-03-09 18:00:00"),  # 9
            ("2023-03-03 10:00:00", "2023-03-05 12:00:00"),  # +0
            ("2023-03-06 14:00:00", "2023-03-08 16:00:00"),  # +0
        ]
        with pytest.raises(
            sqlalchemy.exc.ProgrammingError, match="Overlapping intervals detected"
        ):
            assert count_day_entries(contained_overlap) == 9

        # in the next two cases, the visit either ends on the day the observation window starts or the day before.
        before_observation_window_same_day = [
            ("2023-02-01 00:00:00", "2023-03-01 02:00:00"),
        ]
        assert count_day_entries(before_observation_window_same_day) == 0

        before_observation_window_previous_day = [
            ("2023-02-01 00:00:00", "2023-02-27 02:00:00"),
        ]
        assert count_day_entries(before_observation_window_previous_day) == 0

        # note that proper time zone is required, otherwise (if given in utc), it will be the next day (see next example)
        after_observation_window_same_day = [
            ("2023-03-31 23:59:00+02:00", "2023-04-01 02:00:00"),
        ]
        assert count_day_entries(after_observation_window_same_day) == 0

        after_observation_window_next_day = [
            (
                "2023-03-31 23:59:00+00:00",  # note: this is 2023-04-01 01:59:00+02:00 i.e. next day in Europe/Berlin
                "2023-04-01 02:00:00",
            ),
        ]
        assert count_day_entries(after_observation_window_next_day) == 0

        during_begin_observation_window = [
            ("2023-02-01 00:00:00", "2023-03-04 02:00:00"),
        ]
        assert count_day_entries(during_begin_observation_window) == 4

        during_end_observation_window = [
            ("2023-03-29 23:59:00+02:00", "2023-04-01 02:00:00"),
        ]
        assert count_day_entries(during_end_observation_window) == 3

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
    def test_multiple_patients_active_during_period(
        self,
        person,
        db_session,
        base_criterion,
        observation_window,
        test_cases,
        use_visit_details,
    ):
        # need to disable postgres trigger to avoid constraint violation due to overlapping intervals in testdata
        with disable_postgres_trigger(db_session):
            for p, tc in zip(person, test_cases):
                self.insert_visits(
                    db_session, p, tc["time_range"], use_visit_details=use_visit_details
                )

            with self.execute_base_criterion(
                base_criterion,
                db_session,
                observation_window,
            ):
                stmt = select(
                    partial_day_coverage.c.person_id,
                    partial_day_coverage.c.valid_date,
                ).where(partial_day_coverage.c.run_id == self.run_id)
                df = pd.read_sql(stmt, db_session.connection())
                df["valid_date"] = pd.to_datetime(df["valid_date"])

            db_session.query(VisitOccurrence).delete()
            db_session.commit()

            for tc, p in zip(test_cases, person):
                assert set(
                    df.query(f"person_id=={p.person_id}")["valid_date"].dt.date
                ) == date_set(tc["expected"])
