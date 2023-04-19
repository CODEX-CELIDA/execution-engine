import pytest

from execution_engine.omop.concepts import Concept


@pytest.fixture
def test_concept():
    return Concept(
        concept_id=1,
        concept_name="Test Concept",
        concept_code="unit",
        domain_id="units",
        vocabulary_id="test",
        concept_class_id="test",
    )


@pytest.fixture
def unit_concept():
    return Concept(
        concept_id=1,
        concept_name="Test Unit",
        concept_code="unit",
        domain_id="units",
        vocabulary_id="test",
        concept_class_id="test",
    )
