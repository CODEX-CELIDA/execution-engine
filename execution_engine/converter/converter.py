from abc import ABC, abstractmethod

from fhir.resources.element import Element

from execution_engine.omop.criterion.abstract import Criterion


class CriterionConverter(ABC):
    """
    An abstract criterion converter (interface).

    An instance of this class performs the conversion of some FHIR element to an OMOP criterion.
    """

    @classmethod
    @abstractmethod
    def from_fhir(cls, fhir_definition: Element) -> "CriterionConverter":
        """Creates a new Converter from a FHIR element/resource."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def valid(cls, fhir_definition: Element) -> bool:
        """
        Checks if the given FHIR definition is valid in the context of CPG-on-EBM-on-FHIR.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_criterion(self) -> Criterion:
        """Converts this characteristic to a Criterion."""
        raise NotImplementedError()
