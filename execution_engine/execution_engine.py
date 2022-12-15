import hashlib
import logging
import os
import pickle  # nosec
from datetime import datetime
from typing import Tuple, Union, cast

from fhir.resources.evidencevariable import (
    EvidenceVariable,
    EvidenceVariableCharacteristic,
)

from execution_engine.clients import omopdb
from execution_engine.constants import CohortCategory
from execution_engine.converter.action.abstract import AbstractAction
from execution_engine.converter.action.body_positioning import BodyPositioningAction
from execution_engine.converter.action.drug_administration import (
    DrugAdministrationAction,
)
from execution_engine.converter.action.ventilator_management import (
    VentilatorManagementAction,
)
from execution_engine.converter.characteristic.abstract import AbstractCharacteristic
from execution_engine.converter.characteristic.allergy import AllergyCharacteristic
from execution_engine.converter.characteristic.combination import (
    CharacteristicCombination,
)
from execution_engine.converter.characteristic.condition import ConditionCharacteristic
from execution_engine.converter.characteristic.episode_of_care import (
    EpisodeOfCareCharacteristic,
)
from execution_engine.converter.characteristic.laboratory import (
    LaboratoryCharacteristic,
)
from execution_engine.converter.characteristic.procedure import ProcedureCharacteristic
from execution_engine.converter.characteristic.radiology import RadiologyCharacteristic
from execution_engine.converter.converter import (
    CriterionConverter,
    CriterionConverterFactory,
)
from execution_engine.converter.goal.abstract import Goal
from execution_engine.converter.goal.laboratory_value import LaboratoryValueGoal
from execution_engine.converter.goal.ventilator_management import (
    VentilatorManagementGoal,
)
from execution_engine.fhir.client import FHIRClient
from execution_engine.fhir.recommendation import Recommendation, RecommendationPlan
from execution_engine.fhir_omop_mapping import (
    ActionSelectionBehavior,
    characteristic_to_criterion,
)
from execution_engine.omop.cohort_definition import (
    CohortDefinition,
    CohortDefinitionCombination,
)
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.omop.criterion.visit_occurrence import ActivePatients
from execution_engine.omop.db import result as result_db


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
        self._db = omopdb

    @staticmethod
    def setup_logging() -> None:
        """Sets up logging."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

    @staticmethod
    def _init_characteristics_factory() -> CriterionConverterFactory:
        cf = CriterionConverterFactory()
        cf.register(ConditionCharacteristic)
        cf.register(AllergyCharacteristic)
        cf.register(RadiologyCharacteristic)
        cf.register(ProcedureCharacteristic)
        cf.register(EpisodeOfCareCharacteristic)
        # cf.register(VentilationObservableCharacteristic) # fixme: implement (valueset retrieval / caching)
        cf.register(LaboratoryCharacteristic)

        return cf

    @staticmethod
    def _init_action_factory() -> CriterionConverterFactory:
        af = CriterionConverterFactory()
        af.register(DrugAdministrationAction)
        af.register(VentilatorManagementAction)
        af.register(BodyPositioningAction)

        return af

    @staticmethod
    def _init_goal_factory() -> CriterionConverterFactory:
        gf = CriterionConverterFactory()
        gf.register(LaboratoryValueGoal)
        gf.register(VentilatorManagementGoal)

        return gf

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
            sub: Union[CriterionConverter, CharacteristicCombination]
            for c in characteristics:
                if c.definitionByCombination is not None:
                    sub = get_characteristics(*get_characteristic_combination(c))
                else:
                    sub = cf.get(c)
                    sub = cast(
                        AbstractCharacteristic, sub
                    )  # only for mypy, doesn't do anything at runtime
                comb.add(sub)

            return comb

        if len(ev.characteristic) == 1 and RecommendationPlan.is_combination_definition(
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
        self, actions_def: list[RecommendationPlan.Action]
    ) -> Tuple[list[AbstractAction], ActionSelectionBehavior]:
        """
        Parses the actions of a Recommendation (PlanDefinition) and returns a list of Action objects and the
        corresponding action selection behavior.
        """

        af = self._init_action_factory()
        gf = self._init_goal_factory()

        assert (
            len(set([a.action.selectionBehavior for a in actions_def])) == 1
        ), "All actions must have the same selection behaviour."

        selection_behavior = ActionSelectionBehavior(
            actions_def[0].action.selectionBehavior
        )

        # loop through PlanDefinition.action elements and find the corresponding Action object (by action.code)
        actions: list[AbstractAction] = []
        for action_def in actions_def:
            action = af.get(action_def)
            action = cast(
                AbstractAction, action
            )  # only for mypy, doesn't do anything at runtime

            for goal_def in action_def.goals:
                goal = gf.get(goal_def)
                goal = cast(Goal, goal)
                action.goals.append(goal)

            actions.append(action)

        return actions, selection_behavior

    def _action_combination(
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
            name="intervention_actions",
            category=CohortCategory.INTERVENTION,
            exclude=False,
            operator=operator,
        )

    def load_recommendation(
        self, recommendation_url: str, force_reload: bool = False
    ) -> CohortDefinitionCombination:
        """Processes a single recommendation and creates an OMOP Cohort Definition from it."""
        rec = Recommendation(recommendation_url, self._fhir)

        rec_plan_cohorts: list[CohortDefinition] = []

        for rec_plan in rec.plans():
            cd = CohortDefinition(ActivePatients(name="active_patients"))

            characteristics = self._parse_characteristics(rec_plan.population)
            for characteristic in characteristics:
                cd.add(characteristic_to_criterion(characteristic))

            actions, selection_behavior = self._parse_actions(rec_plan.actions)
            comb_actions = self._action_combination(selection_behavior)

            for action in actions:
                if action is None:
                    raise ValueError("Action is None.")
                comb_actions.add(action.to_criterion())

            cd.add(comb_actions)

            rec_plan_cohorts.append(cd)

        return CohortDefinitionCombination(
            rec_plan_cohorts, url=rec.url, version=rec.version
        )

    def execute(self, cd: CohortDefinitionCombination) -> None:
        """Executes the Cohort Definition and returns a list of Person IDs."""

        with self._db.session.begin():
            cd_id = self.register_cohort_definition(cd)
            self.register_run(cd, cd_id)
            self.execute_cohort_definition(cd, cd_id)

    def register_cohort_definition(self, cd: CohortDefinitionCombination) -> int:
        """Registers the Cohort Definition in the result database."""

        cd_table = result_db.CohortDefinition.__table__

        cd_pickle = pickle.dumps(cd)
        cd_hash = hashlib.sha256(cd_pickle).hexdigest()

        cd_db = (
            cd_table.select()
            .where(cd_table.c.cohort_definition_hash == cd_hash)
            .execute()
            .fetchone()
        )

        if cd_db is not None:
            return cd_db.cohort_definition_id

        sql = (
            result_db.CohortDefinition.__table__.insert()
            .values(
                recommendation_url=cd.url,
                recommendation_version=cd.version,
                cohort_definition_hash=cd_hash,
                cohort_definition_pickle=cd_pickle,
                create_datetime=datetime.now(),
            )
            .returning(result_db.CohortDefinition.cohort_definition_id)
        )

        result = self._db.execute(sql)
        return result.fetchone()[0]

    def register_run(self, cd: CohortDefinitionCombination, cd_id: int) -> None:
        """Registers the run in the result database."""
        raise NotImplementedError()

    def execute_cohort_definition(
        self, cd: CohortDefinitionCombination, cd_id: int
    ) -> None:
        """Executes the Cohort Definition"""
        raise NotImplementedError()
