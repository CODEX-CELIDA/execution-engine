from abc import ABC
from typing import Type

from execution_engine.clients import omopdb
from execution_engine.omop.concepts import Concept, CustomConcept

OMOP_INTENSIVE_CARE = 32037
OMOP_INPATIENT_VISIT = 9201
OMOP_OUTPATIENT_VISIT = 9202
OMOP_SURGICAL_PROCEDURE = 4301351  # OMOP surgical procedure


class VocabularyNotFoundError(Exception):
    """
    Exception raised when a vocabulary is not found.
    """


class VocabularyNotStandardError(Exception):
    """
    Exception raised when a vocabulary is not included in the OMOP standard vocabulary.
    """


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
    def omop_concept(cls, concept: str, standard: bool = False) -> Concept:
        """
        Get the OMOP Standard Vocabulary standard concept for the given code in the given vocabulary.
        """
        raise NotImplementedError()

    @classmethod
    def omop_standard_concept(cls, concept: str) -> Concept:
        """
        Get the OMOP Standard Vocabulary standard concept for the given code in the given vocabulary.
        """
        return cls.omop_concept(concept, standard=True)

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
    def omop_concept(cls, concept: str, standard: bool = False) -> Concept:
        """
        Get the OMOP Standard Vocabulary standard concept for the given code in the given vocabulary.
        """
        return omopdb.get_concept(cls.omop_vocab_name, concept, standard=standard)


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


class ATCDE(AbstractStandardVocabulary):
    """
    ATC DE vocabulary.
    """

    system_uri = "http://fhir.de/CodeSystem/bfarm/atc"
    omop_vocab_name = "ATC"


class ICD10GM(AbstractStandardVocabulary):
    """
    ICD10 German Modification
    """

    system_uri = "http://fhir.de/CodeSystem/dimdi/icd-10-gm"
    omop_vocab_name = "ICD10GM"


class ICD10CM(AbstractStandardVocabulary):
    """
    ICD10 Clinical Modification
    """

    system_uri = "http://hl7.org/fhir/sid/icd-10-cm"
    omop_vocab_name = "ICD10CM"


class UCUM(AbstractStandardVocabulary):
    """
    UCUM vocabulary.
    """

    system_uri = "http://unitsofmeasure.org"
    omop_vocab_name = "UCUM"

    # Explicit ucum unit code to name mapping to avoid disambiguity (e.g. s could be seconds or Siemens)
    UCUM_UNITS = {
        "s": "second",
        "kg": "kilogram",
        "m": "meter",
        "mg": "milligram",
        "L": "liter",
        "ml": "milliliter",
        "mmHg": "millimeter of mercury",
        "mL/kg": "milliliter per kilogram",
        "[iU]": "international unit",
        "[U]": "unit",
        "h": "hour",
        "H": "Henry",
        "S": "Siemens",
    }

    @classmethod
    def omop_concept(cls, concept: str, standard: bool = False) -> Concept:
        """
        Get the OMOP Standard Vocabulary standard concept for the given code in the given vocabulary.
        """
        return omopdb.get_concept(
            cls.omop_vocab_name,
            concept,
            standard=standard,
            name=UCUM.UCUM_UNITS.get(concept),
        )


class AbstractMappedVocabulary(AbstractVocabulary):
    """
    Base class for vocabularies that are not included in the OMOP Standard Vocabulary, but have a mapping to it.

    This class defines a mapping from the vocabulary to the OMOP Standard Vocabulary.
    """

    map: dict[str, int]

    @classmethod
    def omop_concept(cls, concept: str, standard: bool = False) -> Concept:
        """
        Get the OMOP Standard Vocabulary standard concept for the given code in the given vocabulary.
        """
        if concept not in cls.map:
            raise KeyError(
                f"Concept {concept} not found in {cls.system_uri} vocabulary"
            )

        return omopdb.get_concept_info(cls.map[concept])


class KontaktartDE(AbstractMappedVocabulary):
    """
    KontaktartDE vocabulary.
    """

    system_uri = "http://fhir.de/CodeSystem/kontaktart-de"
    map = {
        "intensivstationaer": OMOP_INTENSIVE_CARE,
        "vorstationaer": OMOP_OUTPATIENT_VISIT,
        "normalstationaer": OMOP_INPATIENT_VISIT,
    }


class CODEXCELIDA(AbstractVocabulary):
    """
    CODEXCELIDA vocabulary.
    """

    system_uri = "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/CodeSystem/codex-celida"
    vocab_id = "CODEX_CELIDA"
    map = {
        "tvpibw": CustomConcept(
            concept_name="Tidal volume / ideal body weight (ARDSnet)",
            concept_code="tvpibw",
            domain_id="Measurement",
            vocabulary_id=vocab_id,
        )
    }

    @classmethod
    def omop_concept(cls, concept: str, standard: bool = False) -> Concept:
        """
        Get the OMOP Standard Vocabulary standard concept for the given code in the given vocabulary.
        """

        if concept not in cls.map:
            raise KeyError(
                f"Concept {concept} not found in {cls.system_uri} vocabulary"
            )

        return cls.map[concept]


class VocabularyFactory:
    """
    Vocabulary factory.
    """

    def __init__(self) -> None:
        self._vocabulary: dict[str, AbstractVocabulary] = {}
        self.init()

    def init(self) -> None:
        """
        Initialize the vocabulary factory.
        """
        self.register(LOINC)
        self.register(SNOMEDCT)
        self.register(KontaktartDE)
        self.register(UCUM)
        self.register(ATCDE)
        self.register(ICD10GM)
        self.register(ICD10CM)
        self.register(CODEXCELIDA)

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
            raise VocabularyNotFoundError(f'Vocabulary "{system_uri}" not found')


class StandardVocabulary:
    """
    OMOP Standard Vocabulary utilities
    """

    def __init__(self) -> None:
        self._vf = VocabularyFactory()

    def register(self, vocabulary: Type[AbstractVocabulary]) -> None:
        """
        Register a vocabulary.
        """
        self._vf.register(vocabulary)

    def get_standard_concept(self, system_uri: str, concept: str) -> Concept:
        """
        Get the OMOP Standard Vocabulary standard concept for the given code in the given vocabulary.
        """
        return self._vf.get(system_uri).omop_standard_concept(concept)

    def get_concept(
        self, system_uri: str, concept: str, standard: bool = True
    ) -> Concept:
        """
        Get the OMOP Standard Vocabulary concept for the given code in the given vocabulary.
        """
        return self._vf.get(system_uri).omop_concept(concept, standard=standard)

    def get_standard_unit_concept(self, code: str) -> Concept:
        """
        Get the OMOP Standard Vocabulary standard unit concept for the given code.
        """
        return self.get_standard_concept(UCUM.system_uri, code)

    def related_to(self, ancestor: int, descendant: int, relationship_id: str) -> bool:
        """
        Check if descendant is related to ancestor by the given relationship.
        """
        return omopdb.concept_related_to(ancestor, descendant, relationship_id)

    def get_vocabulary(self, system_uri: str) -> AbstractVocabulary:
        """
        Retrieve the vocabulary for the given system URI.
        """
        return self._vf.get(system_uri)


standard_vocabulary = StandardVocabulary()
