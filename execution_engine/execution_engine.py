import logging
import os
import warnings
from typing import Tuple, Union

from fhir.resources.evidencevariable import (
    EvidenceVariable,
    EvidenceVariableCharacteristic,
)

from .converter.action import (
    AbstractAction,
    ActionFactory,
    BodyPositioningAction,
    DrugAdministrationAction,
    VentilatorManagementAction,
)
from .converter.characteristic.abstract import AbstractCharacteristic
from .converter.characteristic.allergy import AllergyCharacteristic
from .converter.characteristic.combination import CharacteristicCombination
from .converter.characteristic.condition import ConditionCharacteristic
from .converter.characteristic.episode_of_care import EpisodeOfCareCharacteristic
from .converter.characteristic.factory import CharacteristicFactory
from .converter.characteristic.laboratory import LaboratoryCharacteristic
from .converter.characteristic.procedure import ProcedureCharacteristic
from .converter.characteristic.radiology import RadiologyCharacteristic
from .fhir.client import FHIRClient
from .fhir.recommendation import Recommendation
from .fhir_omop_mapping import ActionSelectionBehavior, characteristic_to_criterion
from .omop.cohort_definition import CohortDefinition
from .omop.criterion.combination import CriterionCombination
from .omop.criterion.visit_occurrence import ActivePatients


class ExecutionEngine:
    """
    The Execution Engine is responsible for reading recommendations in CPG-on-EBM-on-FHIR format
    and creating an OMOP Cohort Definition from them."""

    def __init__(self) -> None:

        self.setup_logging()

        if os.environ.get("FHIR_BASE_URL") is None:
            raise Exception("FHIR_BASE_URL environment variable not set.")

        fhir_base_url: str = os.environ["FHIR_BASE_URL"]

        # todo: init clients here?
        self._fhir = FHIRClient(fhir_base_url)

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

        cd = CohortDefinition(ActivePatients(name="active_patients"))

        characteristics = self._parse_characteristics(rec.population)
        for characteristic in characteristics:
            cd.add(characteristic_to_criterion(characteristic))

        actions, selection_behavior = self._parse_actions(rec.actions)
        comb_actions = self.action_combination(selection_behavior)

        for action in actions:
            if action is None:
                warnings.warn("Action is None")  # type: ignore
                continue
            comb_actions.add(action.to_criterion())

        cd.add(comb_actions)

        # (create execution plan)
        # execute single sqls
        # execute combination sql

        return cd

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
            characteristics: list[EvidenceVariableCharacteristic],
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
        self, actions_def: list[Recommendation.Action]
    ) -> Tuple[list[AbstractAction], ActionSelectionBehavior]:
        """
        Parses the actions of a Recommendation (PlanDefinition) and returns a list of Action objects and the
        corresponding action selection behavior.
        """

        af = self._init_action_factory()

        assert (
            len(set([a.action.selectionBehavior for a in actions_def])) == 1
        ), "All actions must have the same selection behaviour."

        selection_behavior = ActionSelectionBehavior(
            actions_def[0].action.selectionBehavior
        )

        # loop through PlanDefinition.action elements and find the corresponding Action object (by action.code)
        actions = []
        for action_def in actions_def:
            actions.append(af.get_action(action_def))

        return actions, selection_behavior

    def execute(self) -> list[int]:
        """Executes the Cohort Definition and returns a list of Person IDs."""
        pass

    def action_combination(
        self, selection_behavior: ActionSelectionBehavior
    ) -> CriterionCombination:
        """
        Get the correct action combination based on the action selection behavior.
        """

        if selection_behavior.code == CharacteristicCombination.Code.ANY_OF:
            operator = CriterionCombination.Operator("OR")
        elif selection_behavior.code == CharacteristicCombination.Code.ALL_OF:
            operator = CriterionCombination.Operator("AND")
        elif selection_behavior.code == CharacteristicCombination.Code.AT_LEAST:
            if selection_behavior.threshold == 1:
                operator = CriterionCombination.Operator("OR")
            else:
                raise NotImplementedError(
                    f"AT_LEAST with threshold {selection_behavior.threshold} not implemented."
                )
        elif selection_behavior.code == CharacteristicCombination.Code.AT_MOST:
            raise NotImplementedError("AT_MOST not implemented.")
        else:
            raise NotImplementedError(
                f"Selection behavior {str(selection_behavior.code)} not implemented."
            )
        return CriterionCombination(
            name="intervention_actions", exclude=False, operator=operator
        )
