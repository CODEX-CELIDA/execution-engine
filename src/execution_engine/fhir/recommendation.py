import logging
from enum import Enum
from typing import List

from fhir.resources.activitydefinition import ActivityDefinition
from fhir.resources.evidencevariable import (
    EvidenceVariable,
    EvidenceVariableCharacteristic,
    EvidenceVariableCharacteristicDefinitionByTypeAndValue,
)
from fhir.resources.plandefinition import PlanDefinition

from .client import FHIRClient


class Recommendation:
    """CPG-on-EBM-on-FHIR Recommendation."""

    def __init__(self, canonical_url: str, fhir_connector: FHIRClient):
        self.canonical_url = canonical_url
        self.fhir = fhir_connector

        self._recommendation = None
        self._population = None
        self._actions: List[ActivityDefinition] = []

        self.load()

    def load(self) -> None:
        """
        Load the recommendation from the FHIR server.
        """

        plan_def = self.get_recommendation(self.canonical_url)
        ev = self.fhir.get_resource("EvidenceVariable", plan_def.subjectCanonical)

        self._recommendation = plan_def
        self._population = ev

        logging.info("Recommendation loaded.")

    def get_recommendation(self, canonical_url: str) -> PlanDefinition:
        """Read the PlanDefinition resource from the FHIR server."""
        return self.fhir.get_resource("PlanDefinition", canonical_url)

    @property
    def population(self) -> EvidenceVariable:
        """
        The population for the recommendation.
        """
        return self._population

    @property
    def actions(self) -> List[ActivityDefinition]:
        """
        The actions for the recommendation.
        """
        return self._actions

    @staticmethod
    def is_combination_definition(
        characteristic: EvidenceVariableCharacteristic,
    ) -> bool:
        """
        Determine if the characteristic is a combination of other characteristics.

        EvidenceVariable.characteristic are either a single characteristic defined using
        EvidenceVariableCharacteristic.definitionByTypeAndValue or a combination of other
        characteristics defined using EvidenceVariableCharacteristic.definitionByCombination.
        """
        return characteristic.definitionByCombination is not None
