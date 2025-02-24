import pandas as pd
import pytest

from execution_engine.omop.concepts import Concept, CustomConcept


class TestConcept:
    def test_from_series(self):
        data = {
            "concept_id": 1,
            "concept_name": "Test Concept",
            "concept_code": "C123",
            "domain_id": "Test Domain",
            "vocabulary_id": "Test Vocabulary",
            "concept_class_id": "Test Class",
            "standard_concept": "S",
            "invalid_reason": None,
        }
        series = pd.Series(data)
        concept = Concept.from_series(series)

        for key, value in data.items():
            assert getattr(concept, key) == value

    def test_str(self):
        concept = Concept(
            concept_id=1,
            concept_name="Test Concept",
            concept_code="C123",
            domain_id="Test Domain",
            vocabulary_id="Test Vocabulary",
            concept_class_id="Test Class",
            standard_concept="S",
            invalid_reason=None,
        )
        assert str(concept) == "Test Concept"

        concept = Concept(
            concept_id=1,
            concept_name="Test Concept",
            concept_code="C123",
            domain_id="Test Domain",
            vocabulary_id="Test Vocabulary",
            concept_class_id="Test Class",
            standard_concept="N",
            invalid_reason=None,
        )
        assert str(concept) == "Test Concept"

    def test_repr(self):
        concept = Concept(
            concept_id=1,
            concept_name="Test Concept",
            concept_code="C123",
            domain_id="Test Domain",
            vocabulary_id="Test Vocabulary",
            concept_class_id="Test Class",
            standard_concept="S",
            invalid_reason=None,
        )
        assert (
            repr(concept)
            == "Concept(concept_id=1, concept_name='Test Concept', concept_code='C123', domain_id='Test Domain', "
            "vocabulary_id='Test Vocabulary', concept_class_id='Test Class', standard_concept='S', invalid_reason=None)"
        )

    def test_is_custom(self):
        concept = Concept(
            concept_id=-1,
            concept_name="Test Concept",
            concept_code="C123",
            domain_id="Test Domain",
            vocabulary_id="Test Vocabulary",
            concept_class_id="Test Class",
            standard_concept="S",
            invalid_reason=None,
        )
        assert concept.is_custom()


class TestCustomConcept:
    def test_init(self):
        custom_concept = CustomConcept(
            name="Test Custom Concept",
            concept_code="CC123",
            domain_id="Test Domain",
            vocabulary_id="Test Vocabulary",
        )

        assert custom_concept.concept_name == "Test Custom Concept"
        assert custom_concept.concept_code == "CC123"
        assert custom_concept.domain_id == "Test Domain"
        assert custom_concept.vocabulary_id == "Test Vocabulary"
        assert custom_concept.concept_class_id == "Custom"
        assert custom_concept.standard_concept is None
        assert custom_concept.invalid_reason is None

    def test_id_property(self):
        custom_concept = CustomConcept(
            name="Test Custom Concept",
            concept_code="CC123",
            domain_id="Test Domain",
            vocabulary_id="Test Vocabulary",
        )
        with pytest.raises(ValueError, match="Custom concepts do not have an id"):
            custom_concept.id

    def test_str(self):
        custom_concept = CustomConcept(
            name="Test Custom Concept",
            concept_code="CC123",
            domain_id="Test Domain",
            vocabulary_id="Test Vocabulary",
        )
        assert (
            str(custom_concept)
            == 'Custom Concept: "Test Custom Concept" [Test Vocabulary#CC123]'
        )
