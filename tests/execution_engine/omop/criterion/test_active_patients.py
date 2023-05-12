import datetime

import pendulum
import pytest
from sqlalchemy import func, text

from execution_engine.omop.criterion.visit_occurrence import PatientsActiveDuringPeriod
from tests.execution_engine.omop.criterion.test_criterion import TestCriterion
from tests.functions import create_visit


class TestActivePatientsDuringPeriod(TestCriterion):
    @pytest.fixture
    def observation_start_datetime(self) -> datetime.datetime:
        return pendulum.parse("2023-03-01 09:36:24")

    @pytest.fixture
    def observation_end_datetime(self) -> datetime.datetime:
        return pendulum.parse("2023-03-31 14:21:11")

    @pytest.fixture
    def base_criterion(self):
        return PatientsActiveDuringPeriod("TestActivePatients")

    def test_active_patients_during_period(
        self,
        person,
        db_session,
        base_criterion,
        observation_start_datetime,
        observation_end_datetime,
    ):
        def count_day_entries(visit_datetimes: list[tuple[str, str]]) -> int:
            """
            Insert visit_occurrence entries into the database for a given person, then apply the base_criterion to
            count the number of days that the person was active during the observation period.
            For each entry in visit_datetimes, the first element is the visit start datetime and the
            second element is the visit end datetime.
            """

            for visit_start_datetime, visit_end_datetime in visit_datetimes:
                vo = create_visit(
                    person,
                    pendulum.parse(visit_start_datetime),
                    pendulum.parse(visit_end_datetime),
                )
                db_session.add(vo)

            db_session.commit()

            base_table = self.create_base_table(
                base_criterion,
                db_session,
                observation_start_datetime,
                observation_end_datetime,
            )

            count = db_session.query(func.count(base_table.c.person_id)).scalar()

            base_table.drop(db_session.connection())
            db_session.execute(text('DELETE FROM "cds_cdm"."visit_occurrence";'))
            db_session.commit()

            return count

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
        assert count_day_entries(partial_overlap) == 1

        contained_overlap = [
            ("2023-03-04 00:00:00", "2023-03-04 06:00:00"),
            ("2023-03-04 01:00:00", "2023-03-04 02:00:00"),
            ("2023-03-04 03:00:00", "2023-03-04 04:00:00"),
        ]
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
        assert count_day_entries(partial_overlap) == 8

        contained_overlap = [
            ("2023-03-01 08:00:00", "2023-03-09 18:00:00"),  # 9
            ("2023-03-03 10:00:00", "2023-03-05 12:00:00"),  # +0
            ("2023-03-06 14:00:00", "2023-03-08 16:00:00"),  # +0
        ]
        assert count_day_entries(contained_overlap) == 9

        before_observation_window = [
            ("2023-02-01 00:00:00", "2023-03-01 02:00:00"),
        ]
        assert count_day_entries(before_observation_window) == 0

        after_observation_window = [
            ("2023-03-31 23:59:00", "2023-04-01 02:00:00"),
        ]
        assert count_day_entries(after_observation_window) == 0

        during_begin_observation_window = [
            ("2023-02-01 00:00:00", "2023-03-04 02:00:00"),
        ]
        assert count_day_entries(during_begin_observation_window) == 4

        during_end_observation_window = [
            ("2023-03-29 23:59:00", "2023-04-01 02:00:00"),
        ]
        assert count_day_entries(during_end_observation_window) == 3
