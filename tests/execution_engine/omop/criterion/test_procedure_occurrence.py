import pytest

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.procedure_occurrence import ProcedureOccurrence
from execution_engine.util import ValueNumber
from tests._fixtures.concept import concept_unit_hour
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
            self.insert_criterion(db_session, criterion, observation_window)
            df = self.fetch_filtered_results(
                db_session,
                pi_pair_id=self.pi_pair_id,
                criterion_id=self.criterion_id,
                category=criterion.category,
            )

            return df

        self.perform_test(
            person_visit,
            concept,
            db_session,
            criterion_execute_func_timing,
            observation_window,
            time_ranges,
        )
