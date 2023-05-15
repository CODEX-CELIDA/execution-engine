from unittest.mock import MagicMock, patch

import pytest

from execution_engine.omop.vocabulary import (
    LOINC,
    SNOMEDCT,
    UCUM,
    AbstractStandardVocabulary,
    KontaktartDE,
    StandardVocabulary,
    VocabularyFactory,
    VocabularyNotFoundError,
)


class TestAbstractStandardVocabulary:
    def test_omop_concept(self):
        with patch("execution_engine.clients.omopdb.get_concept") as mock_get_concept:
            concept_code = "12345"
            LOINC.omop_concept(concept_code)
            mock_get_concept.assert_called_once_with(
                "LOINC", concept_code, standard=False
            )

            mock_get_concept.reset_mock()

            SNOMEDCT.omop_concept(concept_code)
            mock_get_concept.assert_called_once_with(
                "SNOMED", concept_code, standard=False
            )


class TestAbstractMappedVocabulary:
    def test_omop_concept(self):
        with patch(
            "execution_engine.clients.omopdb.get_concept_info"
        ) as mock_get_concept_info:
            concept_code = "intensivstationaer"
            KontaktartDE.omop_concept(concept_code)
            mock_get_concept_info.assert_called_once_with(32037)

            with pytest.raises(
                KeyError, match="Concept not_found not found in .* vocabulary"
            ):
                KontaktartDE.omop_concept("not_found")


class TestVocabularyFactory:
    def test_init_and_register(self):
        vf = VocabularyFactory()
        assert isinstance(vf._vocabulary[LOINC.system_uri], LOINC)
        assert isinstance(vf._vocabulary[SNOMEDCT.system_uri], SNOMEDCT)

    def test_get(self):
        vf = VocabularyFactory()

        with pytest.raises(
            VocabularyNotFoundError, match="Vocabulary not_found not found"
        ):
            vf.get("not_found")

        assert isinstance(vf.get(LOINC.system_uri), LOINC)
        assert isinstance(vf.get(SNOMEDCT.system_uri), SNOMEDCT)


class TestStandardVocabulary:
    def test_get_standard_concept(self):
        with patch(
            "execution_engine.omop.vocabulary.VocabularyFactory.get"
        ) as mock_get:
            mock_vocabulary = MagicMock(spec=AbstractStandardVocabulary)
            mock_get.return_value = mock_vocabulary
            standard_vocabulary = StandardVocabulary()

            standard_vocabulary.get_standard_concept(LOINC.system_uri, "12345")
            mock_vocabulary.omop_standard_concept.assert_called_once_with("12345")

    def test_get_concept(self):
        with patch(
            "execution_engine.omop.vocabulary.VocabularyFactory.get"
        ) as mock_get:
            mock_vocabulary = MagicMock(spec=AbstractStandardVocabulary)
            mock_get.return_value = mock_vocabulary
            standard_vocabulary = StandardVocabulary()

            standard_vocabulary.get_concept(LOINC.system_uri, "12345")
            mock_vocabulary.omop_concept.assert_called_once_with("12345")

    def test_get_standard_unit_concept(self):
        with patch(
            "execution_engine.omop.vocabulary.StandardVocabulary.get_standard_concept"
        ) as mock_get_standard_concept:
            standard_vocabulary = StandardVocabulary()

            standard_vocabulary.get_standard_unit_concept("kg")
            mock_get_standard_concept.assert_called_once_with(UCUM.system_uri, "kg")

    def test_related_to(self):
        with patch(
            "execution_engine.clients.omopdb.concept_related_to"
        ) as mock_related_to:
            standard_vocabulary = StandardVocabulary()

            standard_vocabulary.related_to(1, 2, "relationship")
            mock_related_to.assert_called_once_with(1, 2, "relationship")
