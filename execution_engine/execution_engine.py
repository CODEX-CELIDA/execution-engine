import logging
import os
from typing import Dict, List, Optional, Tuple, Union

from fhir.resources.evidencevariable import (
    EvidenceVariable,
    EvidenceVariableCharacteristic,
)
from fhir.resources.plandefinition import PlanDefinitionAction, PlanDefinitionGoal

from .action import (
    AbstractAction,
    ActionFactory,
    ActionSelectionBehavior,
    BodyPositioningAction,
    DrugAdministrationAction,
    VentilatorManagementAction,
)
from .characteristic import (
    AbstractCharacteristic,
    AllergyCharacteristic,
    CharacteristicCombination,
    CharacteristicFactory,
    ConditionCharacteristic,
    EpisodeOfCareCharacteristic,
    LaboratoryCharacteristic,
    ProcedureCharacteristic,
    RadiologyCharacteristic,
)
from .clients import webapi
from .fhir.client import FHIRClient
from .fhir.recommendation import Recommendation
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
from .omop.concepts import ConceptSetManager
from .omop.webapi import WebAPIClient


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

    @staticmethod
    def setup_logging() -> None:
        """Sets up logging."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

    def process_recommendation(self, recommendation_url: str) -> CohortDefinition:
        """Processes a single recommendation and creates an OMOP Cohort Definition from it."""
        rec = Recommendation(recommendation_url, self._fhir)

        cohort_def = self.generate_population_cohort_definition(rec.population)

        intervention_rules = self.generate_intervention_rules(rec.actions, rec.goals)
        cohort_def.inclusion_rules.extend(intervention_rules)

        return cohort_def

    @staticmethod
    def _init_characteristics_factory() -> CharacteristicFactory:
        cf = CharacteristicFactory()
        cf.register_characteristic_type(ConditionCharacteristic)
        cf.register_characteristic_type(AllergyCharacteristic)
        cf.register_characteristic_type(RadiologyCharacteristic)
        cf.register_characteristic_type(ProcedureCharacteristic)
        cf.register_characteristic_type(EpisodeOfCareCharacteristic)
        # cf.register_characteristic_type(VentilationObservableCharacteristic) # fixme: implement
        cf.register_characteristic_type(LaboratoryCharacteristic)

        return cf

    @staticmethod
    def _init_action_factory() -> ActionFactory:
        af = ActionFactory()
        af.register_action_type(DrugAdministrationAction)
        af.register_action_type(VentilatorManagementAction)
        af.register_action_type(BodyPositioningAction)

        return af

    def _parse_characteristics(self, ev: EvidenceVariable) -> CharacteristicCombination:
        """Parses the characteristics of an EvidenceVariable and returns a CharacteristicCombination."""
        cf = self._init_characteristics_factory()

        def get_characteristic_combination(
            characteristic: EvidenceVariableCharacteristic,
        ) -> Tuple[CharacteristicCombination, EvidenceVariableCharacteristic]:
            comb = CharacteristicCombination(
                CharacteristicCombination.Code(
                    characteristic.definitionByCombination.code
                ),
                exclude=characteristic.exclude,
            )
            characteristics = characteristic.definitionByCombination.characteristic
            return comb, characteristics

        def get_characteristics(
            comb: CharacteristicCombination,
            characteristics: List[EvidenceVariableCharacteristic],
        ) -> CharacteristicCombination:
            sub: Union[AbstractCharacteristic, CharacteristicCombination]
            for c in characteristics:
                if c.definitionByCombination is not None:
                    sub = get_characteristics(*get_characteristic_combination(c))
                else:
                    sub = cf.get_characteristic(c)
                comb.add(sub)

            return comb

        if len(ev.characteristic) == 1 and Recommendation.is_combination_definition(
            ev.characteristic[0]
        ):
            comb, characteristics = get_characteristic_combination(ev.characteristic[0])
        else:
            comb = CharacteristicCombination(
                CharacteristicCombination.Code.ALL_OF, exclude=False
            )
            characteristics = ev.characteristic

        get_characteristics(comb, characteristics)

        return comb

    def _parse_actions(
        self, actions_def: List[Recommendation.Action], goals: List[PlanDefinitionGoal]
    ) -> Tuple[List[AbstractAction], ActionSelectionBehavior]:

        af = self._init_action_factory()

        assert (
            len(set([a.action.selectionBehavior for a in actions_def])) == 1
        ), "All actions must have the same selection behaviour."
        selection_behavior = ActionSelectionBehavior(
            actions_def[0].action.selectionBehavior
        )

        actions = []
        for action_def in actions_def:
            actions.append(af.get_action(action_def, goals))

        return actions, selection_behavior

    @staticmethod
    def _split_characteristics(
        comb: CharacteristicCombination,
    ) -> Tuple[
        AbstractCharacteristic,
        List[Union[AbstractCharacteristic, CharacteristicCombination]],
    ]:
        """Splits a combination of characteristics into OMOP primary criterion and inclusion criteria."""

        primary: Optional[AbstractCharacteristic] = None
        inclusion: List[Union[AbstractCharacteristic, CharacteristicCombination]] = []

        # Find a characteristic that can be used as primary criterion for OMOP Cohort Definition
        # The primary criterion must be a single (not combined) characteristic that is NOT excluded
        for c in comb:
            if (
                primary is None
                and isinstance(c, AbstractCharacteristic)
                and not c.exclude
            ):
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
        def to_inclusion_criterion(
            inc: AbstractCharacteristic, excluded_by_combination: bool
        ) -> InclusionCriterion:
            c, inclusion_criterion = inc.to_omop()
            cm.add(c)

            if inc.exclude ^ excluded_by_combination:
                # excluded by characteristic or excluded by combination
                occurrence = Occurrence(Occurrence.Type.AT_MOST, 0)
            else:
                occurrence = Occurrence(Occurrence.Type.AT_LEAST, 1)

            criterion = InclusionCriterion(
                inclusion_criterion,
                startWindow=StartWindow(),
                occurrence=occurrence,
            )
            return criterion

        def combination_code_to_inclusion_type(
            code: CharacteristicCombination.Code,
        ) -> InclusionRule.Type:
            if code == CharacteristicCombination.Code.ALL_OF:
                ir_type = InclusionRule.Type.ALL
            elif code == CharacteristicCombination.Code.ANY_OF:
                ir_type = InclusionRule.Type.ANY
            elif code == CharacteristicCombination.Code.AT_MOST:
                ir_type = InclusionRule.Type.AT_MOST
            elif code == CharacteristicCombination.Code.AT_LEAST:
                ir_type = InclusionRule.Type.AT_LEAST
            else:
                raise Exception(f"Invalid combination code: {code}.")
            return ir_type

        def get_inclusion_rule_from_combination(
            name: str,
            comb: CharacteristicCombination,
            criteria: List[InclusionCriterion],
        ) -> InclusionRule:

            return InclusionRule(
                name,
                type=combination_code_to_inclusion_type(comb.code),
                count=comb.threshold,
                criteria=criteria,
            )

        rules: List[InclusionRule] = []

        for i, inc in enumerate(inclusion):
            if isinstance(inc, AbstractCharacteristic):
                criterion = to_inclusion_criterion(inc, False)
                rule = InclusionRule(
                    f"inclusion-rule-{i}",
                    type=InclusionRule.Type.AT_LEAST,
                    count=1,
                    criteria=[criterion],
                )
            elif isinstance(inc, CharacteristicCombination):
                criteria = [
                    to_inclusion_criterion(sub_inc, inc.exclude) for sub_inc in inc
                ]
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

        cd = CohortDefinition()

        cd.primary_criteria = self._generate_primary_criterion(
            cd.concept_set_manager, primary
        )
        cd.inclusion_rules.extend(
            self._generate_inclusion_criteria(cd.concept_set_manager, inclusion)
        )

        return cd

    def _generate_intervention_inclusion_criteria(
        cm: ConceptSetManager,
        actions: List[AbstractAction],
        selection_behavior: ActionSelectionBehavior,
    ) -> List[InclusionRule]:

        criteria = []

        for action in actions:
            c, inclusion_criterion = action.to_omop()
            cm.add(c)

            criterion = InclusionCriterion(
                inclusion_criterion,
                startWindow=StartWindow(),
                occurrence=Occurrence(Occurrence.Type.AT_LEAST, 1),
            )

            criteria.append(criterion)

        ir_type, ir_count = ActionSelectionBehavior(
            selection_behavior
        ).to_inclusion_rule_type()

        rule = InclusionRule(
            "intervention",
            type=ir_type,
            count=ir_count,
            criteria=criteria,
        )

        return rule

    def generate_intervention_rules(
        self, actions: List[Recommendation.Action], goals: List[PlanDefinitionGoal]
    ) -> List[InclusionRule]:
        """Generates intervention rules from a recommendation."""
        logging.info("Generating intervention rules.")

        actions, selection_behavior = self._parse_actions(actions, goals)

        return self._generate_intervention_inclusion_criteria(
            self, actions, selection_behavior
        )

    @staticmethod
    def create_cohort(
        name: str, description: str, definition: CohortDefinition
    ) -> Union[List, Dict]:
        """Creates a cohort in the OMOP CDM."""
        return webapi.create_cohort(
            name=name, description=description, definition=definition.json()
        )

    def execute(self) -> List[int]:
        """Executes the Cohort Definition and returns a list of Person IDs."""
        pass
