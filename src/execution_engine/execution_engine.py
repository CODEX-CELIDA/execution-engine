import logging
import os
from enum import Enum
from typing import Dict, Iterator, List, Optional, Tuple, Union

from fhir.resources.activitydefinition import ActivityDefinition
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.evidencevariable import (
    EvidenceVariable,
    EvidenceVariableCharacteristic,
)
from fhir.resources.plandefinition import PlanDefinition

from . import LOINC, SNOMEDCT
from .characteristic import (
    AbstractCharacteristic,
    AllergyCharacteristic,
    CharacteristicCombination,
    CharacteristicFactory,
    ConditionCharacteristic,
)
from .fhir.client import FHIRClient
from .fhir.recommendation import Recommendation
from .fhir.terminology import FHIRTerminologyClient
from .omop import webapi
from .omop.client import WebAPIClient
from .omop.cohort_definition import (
    CohortDefinition,
    InclusionCriterion,
    InclusionRule,
    ObservationWindow,
    Occurrence,
    PrimaryCriteria,
    PrimaryCriteriaLimit,
    StartWindow,
)
from .omop.concepts import ConceptSet, ConceptSetManager
from .omop.criterion import ConditionOccurrence
from .omop.vocabulary import Vocabulary


class ExecutionEngine:
    """The Execution Engine is responsible for reading recommendations in CPG-on-EBM-on-FHIR format and creating an OMOP Cohort Definition from them."""

    def __init__(self) -> None:

        self.setup_logging()

        if os.environ.get("FHIR_BASE_URL") is None:
            raise Exception("FHIR_BASE_URL environment variable not set.")

        fhir_base_url: str = os.environ["FHIR_BASE_URL"]

        if os.environ.get("OMOP_WEBAPI_URL") is None:
            raise Exception("OMOP_WEBAPI_URL environment variable not set.")
        omop_webapi_url: str = os.environ["OMOP_WEBAPI_URL"]

        self._fhir = FHIRClient(fhir_base_url)
        self._omop = WebAPIClient(omop_webapi_url)

    def setup_logging(self) -> None:
        """Sets up logging."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

    def process_recommendation(self, recommendation_url: str) -> CohortDefinition:
        """Processes a single recommendation and creates an OMOP Cohort Definition from it."""
        rec = Recommendation(recommendation_url, self._fhir)

        cd = self.generate_population_cohort_definition(rec.population)

        return cd

    def _init_characteristics_factory(self) -> CharacteristicFactory:
        cf = CharacteristicFactory()
        cf.register_characteristic_type(ConditionCharacteristic())
        cf.register_characteristic_type(AllergyCharacteristic())
        # cf.register_characteristic_type(RadiologyCharacteristic()) #fixme: implement
        # cf.register_characteristic_type(ProcedureCharacteristic()) #fixme: implement
        # cf.register_characteristic_type(EpisodeOfCareCharacteristic()) #fixme: implement
        # cf.register_characteristic_type(VentilationObservableCharacteristic()) #fixme: implement
        # cf.register_characteristic_type(LaboratoryCharacteristic()) #fixme: implement
        return cf

    def _parse_characteristics(self, ev: EvidenceVariable) -> CharacteristicCombination:
        """Parses the characteristics of an EvidenceVariable and returns a CharacteristicCombination."""
        cf = self._init_characteristics_factory()

        def get_characteristic_combination(
            characteristic: EvidenceVariableCharacteristic,
        ) -> Tuple[CharacteristicCombination, EvidenceVariableCharacteristic]:
            comb = CharacteristicCombination(
                CharacteristicCombination.Code(
                    characteristic.definitionByCombination.code
                )
            )
            characteristics = characteristic.definitionByCombination.characteristic
            return comb, characteristics

        def get_characteristics(
            comb: CharacteristicCombination,
            characteristics: EvidenceVariableCharacteristic,
        ) -> CharacteristicCombination:
            sub: Union[AbstractCharacteristic, CharacteristicCombination]
            for c in characteristics:
                if c.definitionByCombination is not None:
                    sub = get_characteristics(*get_characteristic_combination(c))
                else:
                    sub = cf.get_characteristic(c.definitionByTypeAndValue)
                comb.add(sub)

            return comb

        if len(ev.characteristic) == 1 and Recommendation.is_combination_definition(
            ev.characteristic[0]
        ):
            comb, characteristics = get_characteristic_combination(ev.characteristic[0])
        else:
            comb = CharacteristicCombination(CharacteristicCombination.Code.ALL_OF)
            characteristics = ev.characteristic

        get_characteristics(comb, characteristics)

        return comb

    @staticmethod
    def _split_characteristics(
        comb: CharacteristicCombination,
    ) -> Tuple[
        AbstractCharacteristic,
        List[Union[AbstractCharacteristic, CharacteristicCombination]],
    ]:
        """Splits a combination of characteristics into omop primary criterion and inclusion criteria."""

        primary: Optional[AbstractCharacteristic] = None
        inclusion: List[Union[AbstractCharacteristic, CharacteristicCombination]] = []

        for c in comb:
            if primary is None and isinstance(c, AbstractCharacteristic):
                primary = c
            else:
                inclusion.append(c)

        if primary is None:
            raise Exception(
                "No primary characteristic found."
            )  # fixme: better error message

        return primary, inclusion

    @staticmethod
    def _generate_primary_criterion(
        cm: ConceptSetManager, primary: AbstractCharacteristic
    ) -> PrimaryCriteria:
        # generate primary criterion
        c, primary_criterion = primary.to_omop()
        cm.add(c)

        return PrimaryCriteria(
            [primary_criterion],
            window=ObservationWindow(),
            limit=PrimaryCriteriaLimit(),
        )

    @staticmethod
    def _generate_inclusion_criteria(
        cm: ConceptSetManager,
        inclusion: List[Union[AbstractCharacteristic, CharacteristicCombination]],
    ) -> List[InclusionRule]:
        def to_inclusion_criterion(inc: AbstractCharacteristic) -> InclusionCriterion:
            c, inclusionCriterion = inc.to_omop()
            cm.add(c)
            # fixme : if negative this must be "at most 0"
            criterion = InclusionCriterion(
                inclusionCriterion,
                startWindow=StartWindow(),
                occurrence=Occurrence(Occurrence.Type.AT_LEAST, count=1),
            )
            return criterion

        def get_inclusion_rule_from_combination(
            name: str,
            comb: CharacteristicCombination,
            criteria: List[InclusionCriterion],
        ) -> InclusionRule:
            return InclusionRule(
                name,
                type=InclusionRule.InclusionRuleType.AT_LEAST,  # fixme: adapt to actual case
                count=1,
                criteria=criteria,
            )

        rules: List[InclusionRule] = []

        for i, inc in enumerate(inclusion):
            if isinstance(inc, AbstractCharacteristic):
                criterion = to_inclusion_criterion(inc)
                rule = InclusionRule(
                    f"inclusion-rule-{i}",
                    type=InclusionRule.InclusionRuleType.AT_LEAST,
                    count=1,
                    criteria=[criterion],
                )
            elif isinstance(inc, CharacteristicCombination):
                criteria = []
                criteria = [to_inclusion_criterion(sub_inc) for sub_inc in inc]
                rule = get_inclusion_rule_from_combination(
                    f"inclusion-rule-{i}", inc, criteria
                )
            else:
                raise ValueError("Invalid inclusion type")

            rules.append(rule)

        return rules

    def generate_population_cohort_definition(
        self, population: EvidenceVariable
    ) -> CohortDefinition:
        """Generates a population cohort definition from a recommendation."""
        logging.info("Generating population cohort definition.")

        comb = self._parse_characteristics(population)
        primary, inclusion = self._split_characteristics(comb)

        cm = ConceptSetManager()

        primary_criterion = self._generate_primary_criterion(cm, primary)
        inclusion_rules = self._generate_inclusion_criteria(cm, inclusion)

        cd = CohortDefinition(cm, primary_criterion, inclusion_rules)

        return cd

    def create_cohort(
        self, name: str, description: str, definition: CohortDefinition
    ) -> Dict:
        """Creates a cohort in the OMOP CDM."""
        return webapi.create_cohort(
            name=name, description=description, definition=definition.json()
        )

    def execute(self) -> List[int]:
        """Executes the Cohort Definition and returns a list of Person IDs."""
        pass
