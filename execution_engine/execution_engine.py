import hashlib
import logging
from datetime import datetime
from typing import Tuple, Union, cast

import pandas as pd
import sqlalchemy
from fhir.resources.evidencevariable import (
    EvidenceVariable,
    EvidenceVariableCharacteristic,
)
from sqlalchemy import and_, insert, select

from execution_engine.clients import fhir_client, omopdb
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
from execution_engine.fhir.recommendation import Recommendation, RecommendationPlan
from execution_engine.fhir_omop_mapping import (
    ActionSelectionBehavior,
    characteristic_to_criterion,
)
from execution_engine.omop.cohort.cohort_definition import CohortDefinition
from execution_engine.omop.cohort.cohort_definition_combination import (
    CohortDefinitionCombination,
)
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.omop.criterion.visit_occurrence import PatientsActiveDuringPeriod
from execution_engine.omop.db.celida import tables as result_db
from execution_engine.omop.db.celida.tables import (
    CohortDefinition as CohortDefinitionTable,
)
from execution_engine.omop.db.celida.tables import (
    RecommendationCriterion,
    RecommendationResult,
    RecommendationResultInterval,
    RecommendationRun,
)
from execution_engine.omop.serializable import Serializable
from execution_engine.task import runner


class ExecutionEngine:
    """
    The Execution Engine is responsible for reading recommendations in CPG-on-EBM-on-FHIR format
    and creating an OMOP Cohort Definition from them."""

    def __init__(self, verbose: bool = False) -> None:
        self.setup_logging(verbose)
        self._fhir = fhir_client
        self._db = omopdb

    @staticmethod
    def setup_logging(verbose: bool = False) -> None:
        """Sets up logging."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.DEBUG if verbose else logging.INFO,
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
        """
        Processes a single recommendation and creates an OMOP Cohort Definition from it.

        Given the canonical url, the recommendation is retrieved from the FHIR server and parsed.
        A collection of CohortDefinition objects, each consisting of a set of criteria and criteria combinations,
        is created from the recommendation. The CohortDefinition objects are combined into a single
        CohortDefinitionCombination object. A JSON representation of the complete Cohort Definition is stored in the
        result database (standard schema "celida"), if it is not already stored.

        The mapping between FHIR resources / profiles and objects is as follows:

        * Recommendation -> CohortDefinitionCombination
          * RecommendationPlan 1..* -> CohortDefinition
            * EligibilityCriteria 1..* -> CriterionCombination / Criterion
            * InterventionActivity 1..* -> CriterionCombination / Criterion
            * Goal 1..* -> CriterionCombination / Criterion

        :param recommendation_url: The canonical URL of the recommendation.
        :param force_reload: If True, the recommendation is recreated from the FHIR source even if it is already
                             stored in the database.
        :return: The Cohort Definition.
        """

        if not force_reload:
            cdd = self.load_recommendation_from_database(recommendation_url)
            if cdd is not None:
                logging.info(
                    f"Loaded recommendation {recommendation_url} (version={cdd.version}) from database."
                )
                return cdd

        rec = Recommendation(recommendation_url, self._fhir)

        rec_plan_cohorts: list[CohortDefinition] = []

        base_criterion = PatientsActiveDuringPeriod(name="active_patients")

        for rec_plan in rec.plans():
            cd = CohortDefinition(
                name=rec_plan.name,
                url=rec_plan.url,
                base_criterion=base_criterion,
            )

            # parse population and create criteria
            characteristics = self._parse_characteristics(rec_plan.population)

            for characteristic in characteristics:
                cd.add_population(characteristic_to_criterion(characteristic))

            # parse intervention and create criteria
            actions, selection_behavior = self._parse_actions(rec_plan.actions)
            comb_actions = self._action_combination(selection_behavior)

            for action in actions:
                if action is None:
                    raise ValueError("Action is None.")
                comb_actions.add(action.to_criterion())

            cd.add_intervention(comb_actions)

            rec_plan_cohorts.append(cd)

        cdd = CohortDefinitionCombination(
            rec_plan_cohorts,
            base_criterion=base_criterion,
            url=rec.url,
            name=rec.name,
            title=rec.title,
            version=rec.version,
            description=rec.description,
        )

        self.register_cohort_definition(cdd)

        return cdd

    def execute(
        self,
        cdd: CohortDefinitionCombination,
        start_datetime: datetime,
        end_datetime: datetime | None,
    ) -> int:
        """Executes the Cohort Definition and returns a list of Person IDs."""

        if end_datetime is None:
            end_datetime = datetime.now()
        # fixme: set start_datetime and end_datetime as class variables
        # fixme: potentially also register run_id as class variable

        with self._db.begin():
            self.register_cohort_definition(cdd)
            run_id = self.register_run(
                cdd, start_datetime=start_datetime, end_datetime=end_datetime
            )

        self.execute_cohort_definition(
            cdd,
            run_id=run_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )

        return run_id

    def load_recommendation_from_database(
        self, url: str, version: str | None = None
    ) -> CohortDefinitionCombination | None:
        """
        Loads a recommendation from the database. If version is None, the latest created recommendation is returned.
        """
        cd_table = result_db.CohortDefinition

        query = (
            select(cd_table)
            .where(cd_table.recommendation_url == url)
            .order_by(cd_table.create_datetime.desc())
        )
        if version is not None:
            query.where(cd_table.recommendation_version == version)

        with self._db.connect() as con:
            cd_db = con.execute(query).fetchone()

        if cd_db is not None:
            cd = CohortDefinitionCombination.from_json(
                cd_db.cohort_definition_json.decode()
            )
            cd.id = cd_db.cohort_definition_id
            return cd

        return None

    @staticmethod
    def _hash(obj: Serializable) -> tuple[bytes, str]:
        json = obj.json()
        return json, hashlib.sha256(json).hexdigest()

    def register_cohort_definition(self, cd: CohortDefinitionCombination) -> None:
        """Registers the Cohort Definition in the result database."""

        cd_table = result_db.CohortDefinition

        cd_json, cd_hash = self._hash(cd)
        query = select(cd_table).where(cd_table.cohort_definition_hash == cd_hash)

        with self._db.begin() as con:
            cd_db = con.execute(query).fetchone()

            if cd_db is not None:
                cd.id = cd_db.cohort_definition_id
            else:
                query = (
                    insert(result_db.CohortDefinition)
                    .values(
                        recommendation_name=cd.name,
                        recommendation_title=cd.title,
                        recommendation_url=cd.url,
                        recommendation_version=cd.version,
                        cohort_definition_hash=cd_hash,
                        cohort_definition_json=cd_json,
                        create_datetime=datetime.now(),
                    )
                    .returning(result_db.CohortDefinition.cohort_definition_id)
                )

                result = con.execute(query)
                cd.id = result.fetchone().cohort_definition_id

            for cd_plan in cd.cohort_definitions():
                self.register_plan(cd_plan, cohort_definition_id=cd.id)

                for criterion in cd_plan.flatten():
                    self.register_criterion(criterion)

    def register_plan(
        self, cd_plan: CohortDefinition, cohort_definition_id: int
    ) -> None:
        """
        Registers the Cohort Definition Plan in the result database.

        :param cd_plan: The Cohort Definition Plan.
        :param cohort_definition_id: The ID of the Cohort Definition.
        """
        _, cd_plan_hash = self._hash(cd_plan)
        query = select(result_db.RecommendationPlan).where(
            result_db.RecommendationPlan.recommendation_plan_hash == cd_plan_hash
        )
        with self._db.begin() as con:
            cd_plan_db = con.execute(query).fetchone()

            if cd_plan_db is not None:
                cd_plan.id = cd_plan_db.plan_id
            else:
                query = (
                    insert(result_db.RecommendationPlan)
                    .values(
                        cohort_definition_id=cohort_definition_id,
                        recommendation_plan_url=cd_plan.url,
                        recommendation_plan_name=cd_plan.name,
                        recommendation_plan_hash=cd_plan_hash,
                    )
                    .returning(result_db.RecommendationPlan.plan_id)
                )

                result = con.execute(query)
                cd_plan.id = result.fetchone().plan_id

    def register_criterion(self, criterion: Criterion) -> None:
        """
        Registers the Cohort Definition Criterion in the result database.

        :param criterion: The Cohort Definition Criterion.
        """
        _, crit_hash = self._hash(criterion)

        query = select(result_db.RecommendationCriterion).where(
            result_db.RecommendationCriterion.criterion_hash == crit_hash
        )
        with self._db.begin() as con:
            criterion_db = con.execute(query).fetchone()

            if criterion_db is not None:
                criterion.id = criterion_db.criterion_id
            else:
                query = (
                    insert(result_db.RecommendationCriterion)
                    .values(
                        criterion_hash=crit_hash,
                        criterion_name=criterion.unique_name(),
                        criterion_description=criterion.description(),
                    )
                    .returning(result_db.RecommendationCriterion.criterion_id)
                )

                result = con.execute(query)
                criterion.id = result.fetchone().criterion_id

    def register_run(
        self,
        cd: CohortDefinitionCombination,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> int:
        """Registers the run in the result database."""
        with self._db.begin() as con:
            query = (
                result_db.RecommendationRun.__table__.insert()
                .values(
                    cohort_definition_id=cd.id,
                    observation_start_datetime=start_datetime,
                    observation_end_datetime=end_datetime,
                    run_datetime=datetime.now(),
                )
                .returning(result_db.RecommendationRun.recommendation_run_id)
            )

            result = con.execute(query).fetchone()

        return result.recommendation_run_id

    def execute_cohort_definition(
        self,
        cd: CohortDefinitionCombination,
        run_id: int,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> None:
        """
        Executes the Cohort Definition and stores the results in the result tables.
        """
        assert isinstance(
            start_datetime, datetime
        ), "start_datetime must be of type datetime"
        assert isinstance(
            end_datetime, datetime
        ), "end_datetime must be of type datetime"

        assert start_datetime.tzinfo, "start_datetime must be timezone-aware"
        assert end_datetime.tzinfo, "end_datetime must be timezone-aware"

        date_format = "%Y-%m-%d %H:%M:%S %z"

        logging.info(
            f"Observation window from {start_datetime.strftime(date_format)} to {end_datetime.strftime(date_format)}"
        )

        params = {
            "run_id": run_id,
            "observation_start_datetime": start_datetime,
            "observation_end_datetime": end_datetime,
        }

        # observation_window = TimeRange(start=start_datetime, end=end_datetime)

        # todo: warning: current implementation might run into memory problems ->
        #  then splitting by person might be necessary
        #  otherwise not needed intermediate results may be deleted after processing

        # todo determine runner class
        # runner_class = runner.ParallelTaskRunner if use_multiprocessing else runner.SequentialTaskRunner

        execution_map = cd.execution_map()
        task_runner = runner.SequentialTaskRunner(execution_map)
        task_runner.run(params)

    def insert_intervals(self, data: pd.DataFrame, con: sqlalchemy.Connection) -> None:
        """Inserts the intervals into the database."""
        if data.empty:
            return

        (
            data.to_sql(
                name=RecommendationResultInterval.__tablename__,
                con=con,
                if_exists="append",
                index=False,
            )
        )

    def fetch_patients(self, run_id: int) -> dict:
        """Retrieve list of patients associated with a single run."""
        assert isinstance(run_id, int)

        t = RecommendationResult.__table__.alias("result")
        t_criterion = RecommendationCriterion.__table__.alias("criteria")
        query = (
            select(
                t.c.cohort_category,
                t_criterion.c.name,
                t.c.person_id,
            )
            .select_from(t)
            .join(t_criterion)
            .filter(
                and_(
                    t.c.recommendation_run_id == run_id,
                    t.c.plan_id.is_(None),
                )
            )
        )

        return self._db.query(query)

    def fetch_criteria(self, run_id: int) -> dict:
        """Retrieve individual criteria associated with a single run."""
        assert isinstance(run_id, int)

        t_cd = CohortDefinitionTable.__table__.alias("cd")
        t_run = RecommendationRun.__table__.alias("run")

        query = (
            select(t_cd.c.cohort_definition_hash, t_cd.c.cohort_definition_json)
            .join(t_run)
            .filter(t_run.c.recommendation_run_id == run_id)
        )

        return self._db.query(query)

    def fetch_run(self, run_id: int) -> dict:
        """
        Retrieve information about a single run.
        """
        t_cd = CohortDefinitionTable.__table__.alias("cd")
        t_run = RecommendationRun.__table__.alias("run")

        query = (
            select(
                t_run.c.recommendation_run_id,
                t_run.c.observation_start_datetime,
                t_run.c.observation_end_datetime,
                t_cd.c.cohort_definition_id,
                t_cd.c.recommendation_url,
                t_cd.c.recommendation_version,
                t_cd.c.cohort_definition_hash,
            )
            .select_from(t_run)
            .join(t_cd)
            .filter(t_run.c.recommendation_run_id == run_id)
        )

        return self._db.query(query).iloc[0].to_dict()

    # TODO: Should be based on run id
    def fetch_patient_data(
        self,
        person_id: int,
        criterion_name: str,
        cdd: CohortDefinitionCombination,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> pd.DataFrame:
        """Retrieve patient data for a person and single criterion."""
        criterion = cdd.get_criterion(criterion_name)

        statement = criterion.sql_select_data(person_id)
        params = {
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
        }

        self._db.log_query(statement, params)

        return self._db.query(statement, **params)
