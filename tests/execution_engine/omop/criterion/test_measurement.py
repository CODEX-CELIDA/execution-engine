import pytest

from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.measurement import Measurement
from tests.execution_engine.omop.criterion.test_value_criterion import ValueCriterion
from tests.functions import create_measurement


class TestMeasurement(ValueCriterion):
    @pytest.fixture
    def concept(self):
        return Concept(
            concept_id=21490854,
            concept_name="Tidal volume Ventilator --on ventilator",
            domain_id="Measurement",
            vocabulary_id="LOINC",
            concept_class_id="Clinical Observation",
            standard_concept="S",
            concept_code="76222-9",
            invalid_reason=None,
        )

    @pytest.fixture
    def concept_no_match(self):
        return Concept(
            concept_id=3013466,
            concept_name="aPTT in Blood by Coagulation assay",
            domain_id="Measurement",
            vocabulary_id="LOINC",
            concept_class_id="Clinical Observation",
            standard_concept="S",
            concept_code="3173-2",
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
        return Measurement

    def create_value(
        self, visit_occurrence, concept_id, datetime, value, unit_concept_id
    ):
        value_as_concept_id = value.concept_id if isinstance(value, Concept) else None
        value_as_number = value if isinstance(value, float | int) else None

        return create_measurement(
            vo=visit_occurrence,
            measurement_concept_id=concept_id,
            measurement_datetime=datetime,
            value_as_number=value_as_number,
            value_as_concept_id=value_as_concept_id,
            unit_concept_id=unit_concept_id,
        )
