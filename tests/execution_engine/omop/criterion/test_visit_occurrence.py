import pendulum
import pytest
from sqlalchemy import func, select

from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.visit_occurrence import VisitOccurrence
from execution_engine.omop.db.cdm import Person
from execution_engine.util.sql import select_into
from tests._testdata.concepts import INPATIENT_VISIT, INTENSIVE_CARE
from tests.execution_engine.omop.criterion.test_criterion import TestCriterion, to_table
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
    ):
        base_table = to_table("base_table")

        query = select_into(select(Person.person_id), base_table, temporary=True)
        db_session.execute(query)
        db_session.commit()

        count = db_session.query(func.count(base_table.c.person_id)).scalar()
        assert (
            count > 0
        ), "Base table (active patients in period) should have at least one row."

        yield base_table

        base_table.drop(db_session.connection())

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
            (  # before observation window
                [
                    (("2023-02-01 00:00:00", "2023-03-01 02:00:00"), INTENSIVE_CARE),
                ],
                {},
            ),
            (  # after observation window
                [
                    (("2023-03-31 23:59:00", "2023-04-01 02:00:00"), INTENSIVE_CARE),
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
                    (("2023-03-29 23:59:00", "2023-04-01 02:00:00"), INTENSIVE_CARE),
                ],
                {"2023-03-29", "2023-03-30", "2023-03-31"},
            ),
        ],
    )
    def test_visit_occurrence(
        self,
        person,
        db_session,
        base_criterion,
        occurrence_criterion,
        visit_datetimes,
        expected,
    ):
        for (
            visit_start_datetime,
            visit_end_datetime,
        ), visit_concept_id in visit_datetimes:
            vo = create_visit(
                person_id=person.person_id,
                visit_start_datetime=pendulum.parse(visit_start_datetime),
                visit_end_datetime=pendulum.parse(visit_end_datetime),
                visit_concept_id=visit_concept_id,
            )
            db_session.add(vo)

        db_session.commit()

        # run criterion against db
        df = occurrence_criterion(exclude=False)

        valid_dates = set(pendulum.parse(e).date() for e in expected)
        assert set(df["valid_date"].dt.date) == valid_dates
