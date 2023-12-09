import pandas as pd
import pytest
from sqlalchemy import and_, bindparam, select

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.procedure_occurrence import ProcedureOccurrence
from execution_engine.omop.db.celida.tables import RecommendationResultInterval
from execution_engine.util import ValueNumber
from tests._fixtures.concept import concept_unit_hour
from tests.execution_engine.omop.criterion.test_criterion import TestCriterion
from tests.execution_engine.omop.criterion.test_occurrence_criterion import Occurrence
from tests.functions import create_procedure


class TestProcedureOccurrence(Occurrence):
    @pytest.fixture
    def concept(self):
        return Concept(
            concept_id=4230167,
            concept_name="Artificial respiration",
            domain_id="Procedure",
            vocabulary_id="SNOMED",
            concept_class_id="Procedure",
            standard_concept="S",
            concept_code="40617009",
            invalid_reason=None,
        )

    @pytest.fixture
    def criterion_class(self):
        return ProcedureOccurrence

    def create_occurrence(
        self, visit_occurrence, concept_id, start_datetime, end_datetime
    ):
        return create_procedure(
            vo=visit_occurrence,
            procedure_concept_id=concept_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )

    def test_duration(
        self,
        person_visit,
        db_session,
        concept,
        criterion_execute_func,
        observation_window,
        base_table,
    ):
        _, vo = person_visit[0]

        time_ranges = [
            ("2023-03-04 18:00:00Z", "2023-03-04 19:30:00Z", False),
            ("2023-03-04 20:00:00Z", "2023-03-04 21:30:00Z", False),
            ("2023-03-05 19:30:00Z", "2023-03-05 21:30:00Z", True),
        ]

        def criterion_execute_func_timing(
            concept: Concept,
            exclude: bool,
            value: ValueNumber | None = None,
        ):
            timing = ValueNumber(value_min=2, unit=concept_unit_hour)

            criterion = ProcedureOccurrence(
                name="test",
                exclude=exclude,
                category=CohortCategory.POPULATION,
                concept=concept,
                value=value,
                timing=timing,
                static=None,
            )
            # todo: deduplicate code (from TestCriterion)
            query = criterion.create_query()

            query = query.add_columns(
                bindparam("recommendation_run_id", self.recommendation_run_id).label(
                    "recommendation_run_id"
                ),
                bindparam("plan_id", None).label("plan_id"),
                bindparam("criterion_id", self.criterion_id).label("criterion_id"),
                bindparam("cohort_category", CohortCategory.POPULATION).label(
                    "cohort_category"
                ),
            )

            t_result = RecommendationResultInterval.__table__
            query_insert = t_result.insert().from_select(query.selected_columns, query)
            db_session.execute(query_insert, params=observation_window.dict())

            stmt = select(
                self.result_view.c.person_id, self.result_view.c.valid_date
            ).where(
                and_(
                    self.result_view.c.recommendation_run_id
                    == self.recommendation_run_id,
                    self.result_view.c.criterion_id == self.criterion_id,
                )
            )
            df = pd.read_sql(
                stmt,
                db_session.connection(),
                params=observation_window.dict()
                | {"run_id": TestCriterion.recommendation_run_id},
            )
            df["valid_date"] = pd.to_datetime(df["valid_date"])

            return df

        self.perform_test(
            person_visit,
            concept,
            db_session,
            criterion_execute_func_timing,
            observation_window,
            time_ranges,
        )
