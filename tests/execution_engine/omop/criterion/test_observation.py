import pytest

from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.observation import Observation
from tests.execution_engine.omop.criterion.test_value_criterion import ValueCriterion
from tests.functions import create_observation


class TestObservation(ValueCriterion):
    @pytest.fixture
    def concept(self):
        return Concept(
            concept_id=4169185,
            concept_name="Allergy to heparin",
            domain_id="Observation",
            vocabulary_id="SNOMED",
            concept_class_id="Clinical Finding",
            standard_concept="S",
            concept_code="294872001",
            invalid_reason=None,
        )

    @pytest.fixture
    def concept_no_match(self):
        return Concept(
            concept_id=4170358,
            concept_name="Allergy to heparinoid",
            domain_id="Observation",
            vocabulary_id="SNOMED",
            concept_class_id="Clinical Finding",
            standard_concept="S",
            concept_code="294876003",
            invalid_reason=None,
        )

    @pytest.fixture
    def unit_concept(self):
        return Concept(
            concept_id=8587,
            concept_name="milliliter",
            domain_id="Unit",
            vocabulary_id="UCUM",
            concept_class_id="Unit",
            standard_concept="S",
            concept_code="mL",
            invalid_reason=None,
        )

    @pytest.fixture
    def unit_concept_no_match(self):
        return Concept(
            concept_id=9529,
            concept_name="kilogram",
            domain_id="Unit",
            vocabulary_id="UCUM",
            concept_class_id="Unit",
            standard_concept="S",
            concept_code="kg",
            invalid_reason=None,
        )

    @pytest.fixture
    def criterion_class(self):
        return Observation

    def create_value(
        self, visit_occurrence, concept_id, datetime, value, unit_concept_id
    ):

        value_as_concept_id = value.concept_id if isinstance(value, Concept) else None
        value_as_number = value if isinstance(value, float | int) else None
        value_as_string = value if isinstance(value, str) else None

        return create_observation(
            vo=visit_occurrence,
            observation_concept_id=concept_id,
            observation_datetime=datetime,
            value_as_concept_id=value_as_concept_id,
            value_as_number=value_as_number,
            value_as_string=value_as_string,
            unit_concept_id=unit_concept_id,
        )
