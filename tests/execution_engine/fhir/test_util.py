import pytest
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.element import Element
from fhir.resources.extension import Extension

from execution_engine.fhir.util import get_coding, get_extension


class TestUtils:
    @pytest.fixture
    def codeable_concept(self):
        cc = CodeableConcept.model_construct(
            coding=[
                Coding.model_construct(
                    system="http://example.com/system",
                    code="example_code",
                    display="Example Code",
                )
            ]
        )
        return cc

    def test_get_coding(self, codeable_concept):
        coding = get_coding(codeable_concept)
        assert coding.system == "http://example.com/system"
        assert coding.code == "example_code"
        assert coding.display == "Example Code"

    def test_get_coding_with_system_uri(self, codeable_concept):
        coding = get_coding(codeable_concept, "http://example.com/system")
        assert coding.system == "http://example.com/system"
        assert coding.code == "example_code"
        assert coding.display == "Example Code"

    def test_get_coding_fail(self):
        cc = CodeableConcept.model_construct()
        with pytest.raises(ValueError):
            get_coding(cc)

    @pytest.fixture
    def element_with_extension(self):
        ext = Extension.model_construct(
            url="http://example.com/extension", valueString="test"
        )
        elem = Element.model_construct(extension=[ext])
        return elem

    def test_get_extension(self, element_with_extension):
        extension = get_extension(
            element_with_extension, "http://example.com/extension"
        )
        assert extension.url == "http://example.com/extension"
        assert extension.valueString == "test"

    def test_get_extension_not_found(self, element_with_extension):
        extension = get_extension(
            element_with_extension, "http://example.com/nonexistent"
        )
        assert extension is None

    def test_get_extension_no_extensions(self):
        elem = Element.model_construct()
        extension = get_extension(elem, "http://example.com/extension")
        assert extension is None
