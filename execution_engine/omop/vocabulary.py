import logging
from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, Type

from ..clients import webapi
from .concepts import Concept


class AbstractVocabulary(ABC):
    """
    Abstract vocabulary class.
    """

    """ The vocabulary code system identifier. """
    system_uri: str

    @classmethod
    def name(cls) -> str:
        """
        Get the name of the vocabulary.
        """
        return cls.__name__

    @classmethod
    def omop_standard_concept(cls, concept: str) -> Concept:
        """
        Get the OMOP Standard Vocabulary standard concept for the given code in the given vocabulary.
        """
        pass

    @classmethod
    def is_system(cls, system: str) -> bool:
        """
        Check if the given system is the same as this vocabulary.
        """
        return cls.system_uri == system


class AbstractStandardVocabulary(AbstractVocabulary):
    """
    Abstract class for vocabularies that are included in the OMOP Standard Vocabulary (e.g. SNOMED CT, ICD10, etc.).
    """

    """ The vocabulary name in the OMOP Standard Vocabulary. """
    omop_vocab_name: str

    @classmethod
    def omop_standard_concept(cls, concept: str) -> Concept:
        """
        Get the OMOP Standard Vocabulary standard concept for the given code in the given vocabulary.
        """
        return webapi.get_standard_concept(cls.omop_vocab_name, concept)


class LOINC(AbstractStandardVocabulary):
    """
    LOINC vocabulary.
    """

    system_uri = "http://loinc.org"
    omop_vocab_name = "LOINC"


class SNOMEDCT(AbstractStandardVocabulary):
    """
    SNOMEDCT vocabulary.
    """

    system_uri = "http://snomed.info/sct"
    omop_vocab_name = "SNOMED"


class UCUM(AbstractStandardVocabulary):
    """
    UCUM vocabulary.
    """

    system_uri = "http://unitsofmeasure.org"
    omop_vocab_name = "UCUM"


class AbstractMappedVocabulary(AbstractVocabulary):
    """
    Base class for vocabularies that are not included in the OMOP Standard Vocabulary.

    This class defines a mapping from the vocabulary to the OMOP Standard Vocabulary.
    """

    map: Dict[str, str]

    @classmethod
    def omop_standard_concept(cls, concept: str) -> Concept:
        """
        Get the OMOP Standard Vocabulary standard concept for the given code in the given vocabulary.
        """
        if concept not in cls.map:
            raise KeyError(
                f"Concept {concept} not found in {cls.system_uri} vocabulary"
            )

        return webapi.get_concept_info(cls.map[concept])


class KontaktartDE(AbstractMappedVocabulary):
    """
    KontaktartDE vocabulary.
    """

    system_uri = "http://fhir.de/CodeSystem/kontaktart-de"
    map = {
        "intensivstationaer": "32037",
    }


class VocabularyFactory:
    """
    Vocabulary factory.
    """

    def __init__(self) -> None:
        self._vocabulary: Dict[str, AbstractVocabulary] = {}
        self.init()

    def init(self) -> None:
        """
        Initialize the vocabulary factory.
        """
        self.register(LOINC)
        self.register(SNOMEDCT)
        self.register(KontaktartDE)
        self.register(UCUM)

    def register(self, vocabulary: Type[AbstractVocabulary]) -> None:
        """
        Register a vocabulary.
        """
        voc = vocabulary()
        self._vocabulary[voc.system_uri] = voc

    def get(self, system_uri: str) -> AbstractVocabulary:
        """
        Get the vocabulary class for the given system.
        """
        if system_uri in self._vocabulary:
            return self._vocabulary[system_uri]
        else:
            raise KeyError(f"Vocabulary {system_uri} not found")


class StandardVocabulary:
    """
    OMOP Standard Vocabulary utilities
    """

    def __init__(self) -> None:
        self._vf = VocabularyFactory()

    def get_standard_concept(self, system_uri: str, concept: str) -> Concept:
        """
        Get the OMOP Standard Vocabulary standard concept for the given code in the given vocabulary.
        """
        return self._vf.get(system_uri).omop_standard_concept(concept)


standard_vocabulary = StandardVocabulary()
