from unittest.mock import MagicMock

import pytest
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.element import Element
from fhir.resources.extension import Extension
from fhir.resources.quantity import Quantity
from fhir.resources.range import Range

from execution_engine.converter.converter import (
    CriterionConverter,
    CriterionConverterFactory,
    code_display,
    parse_code,
    parse_value,
    select_value,
)
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.util import ValueConcept, ValueNumber


class TestSelectValue:
    def test_select_value_codeable_concept(self):
        cc = CodeableConcept.construct()
        elem = Extension.construct(valueCodeableConcept=cc)
        value = select_value(elem, "value")
        assert value == cc

    def test_select_value_quantity(self):
        q = Quantity.construct()
        elem = Extension.construct(valueQuantity=q)
        value = select_value(elem, "value")
        assert value == q

    def test_select_value_range(self):
        r = Range.construct()
        elem = Extension.construct(valueRange=r)
        value = select_value(elem, "value")
        assert value == r

    def test_select_value_boolean(self):
        b = True
        elem = Extension.construct(valueBoolean=b)
        value = select_value(elem, "value")
        assert value == b

    def test_select_value_not_found(self):
        elem = Extension.construct()
        with pytest.raises(ValueError):
            select_value(elem, "value")


class TestParseCode:
    def test_parse_code(self, monkeypatch):
        mock_get_standard_concept = MagicMock()
        monkeypatch.setattr(
            "execution_engine.omop.vocabulary.standard_vocabulary.get_standard_concept",
            mock_get_standard_concept,
        )

        coding = Coding.construct(system="http://example.com", code="123")
        codeable_concept = CodeableConcept.construct(coding=[coding])
        parse_code(codeable_concept)

        mock_get_standard_concept.assert_called_once_with("http://example.com", "123")


class TestCodeDisplay:
    def test_code_display_with_display(self):
        coding = Coding.construct(
            system="http://example.com", code="123", display="Example display"
        )
        codeable_concept = CodeableConcept.construct(coding=[coding])
        display = code_display(codeable_concept)
        assert display == "Example display"

    def test_code_display_without_display(self):
        coding = Coding.construct(system="http://example.com", code="123")
        codeable_concept = CodeableConcept.construct(coding=[coding])
        display = code_display(codeable_concept)
        assert display == "123"


class TestParseValue:
    def test_parse_value_codeable_concept(self, monkeypatch, test_concept):
        mock_get_standard_concept = MagicMock()
        mock_get_standard_concept.return_value = test_concept
        monkeypatch.setattr(
            "execution_engine.omop.vocabulary.standard_vocabulary.get_standard_concept",
            mock_get_standard_concept,
        )

        coding = Coding.construct(system="http://example.com", code="123")
        codeable_concept = CodeableConcept.construct(coding=[coding])
        elem = Extension.construct(valueCodeableConcept=codeable_concept)

        value = parse_value(elem, "value")

        assert isinstance(value, ValueConcept)
        mock_get_standard_concept.assert_called_once_with(
            system_uri="http://example.com", concept="123"
        )

    def test_parse_value_quantity(self, monkeypatch, unit_concept):
        mock_get_standard_unit_concept = MagicMock()
        mock_get_standard_unit_concept.return_value = unit_concept
        monkeypatch.setattr(
            "execution_engine.omop.vocabulary.standard_vocabulary.get_standard_unit_concept",
            mock_get_standard_unit_concept,
        )

        quantity = Quantity.construct(value=42, unit="kg")
        elem = Extension.construct(valueQuantity=quantity)

        value = parse_value(elem, "value")

        assert isinstance(value, ValueNumber)
        assert value.value == 42
        mock_get_standard_unit_concept.assert_called_once_with("kg")

    def test_parse_value_range(self, monkeypatch, unit_concept):
        mock_get_standard_unit_concept = MagicMock()
        mock_get_standard_unit_concept.return_value = unit_concept
        monkeypatch.setattr(
            "execution_engine.omop.vocabulary.standard_vocabulary.get_standard_unit_concept",
            mock_get_standard_unit_concept,
        )

        low = Quantity.construct(value=10, code="kg")
        high = Quantity.construct(value=20, code="kg")
        range_obj = Range.construct(low=low, high=high)
        elem = Extension.construct(valueRange=range_obj)

        value = parse_value(elem, "value")

        assert isinstance(value, ValueNumber)
        assert value.value_min == 10
        assert value.value_max == 20
        mock_get_standard_unit_concept.assert_called_once_with("kg")


class TestCriterionConverter:
    # Continuing the test classes for CriterionConverter and CriterionConverterFactory

    class MockCriterionConverter(CriterionConverter):
        @classmethod
        def from_fhir(cls, fhir_definition: Element) -> "CriterionConverter":
            return cls(name="MockConverter", exclude=False)

        @classmethod
        def valid(cls, fhir_definition: Element) -> bool:
            return fhir_definition.id == "valid"

        def to_criterion(self) -> Criterion | CriterionCombination:
            raise NotImplementedError()

    def test_criterion_converter_factory_register(self):
        factory = CriterionConverterFactory()
        factory.register(TestCriterionConverter.MockCriterionConverter)

        assert (
            TestCriterionConverter.MockCriterionConverter in factory._converters
        ), "Converter was not registered"

    def test_criterion_converter_factory_get(self):
        factory = CriterionConverterFactory()
        factory.register(TestCriterionConverter.MockCriterionConverter)

        element = Element.construct(id="valid")
        converter = factory.get(element)

        assert isinstance(
            converter, TestCriterionConverter.MockCriterionConverter
        ), "Converter factory returned wrong converter type"

    def test_criterion_converter_factory_no_matching_converter(self):
        factory = CriterionConverterFactory()
        factory.register(TestCriterionConverter.MockCriterionConverter)

        element = Element.construct(id="not-valid")

        with pytest.raises(ValueError, match="Cannot find a converter"):
            factory.get(element)
