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

from execution_engine import __version__, fhir
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
from execution_engine.fhir_omop_mapping import (
    ActionSelectionBehavior,
    characteristic_to_criterion,
)
from execution_engine.omop import cohort
from execution_engine.omop.cohort import PopulationInterventionPair
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.omop.criterion.visit_occurrence import PatientsActiveDuringPeriod
from execution_engine.omop.db.celida import tables as result_db
from execution_engine.omop.serializable import Serializable
from execution_engine.task import runner


class ExecutionEngine:
    """
    The Execution Engine is responsible for reading recommendations in CPG-on-EBM-on-FHIR format
    and creating an Recommendation object from them."""

    # todo: improve documentation

    def __init__(self, verbose: bool = False) -> None:
        # late import to allow pytest to set env variables first
        from execution_engine.settings import config

        self.setup_logging(verbose)
        self._fhir = fhir_client
        self._db = omopdb
        self._db.init()

        self._config = config

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

        if len(
            ev.characteristic
        ) == 1 and fhir.RecommendationPlan.is_combination_definition(
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
        self, actions_def: list[fhir.RecommendationPlan.Action]
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
    ) -> cohort.Recommendation:
        """
        Processes a single recommendation and creates a Recommendation object from it.

        Given the canonical url, the recommendation is retrieved from the FHIR server and parsed.
        A collection of PopulationInterventionPair objects, each consisting of a set of criteria and criteria
        combinations,is created from the recommendation. The PopulationInterventionPair objects are combined into a
        single Recommendation object. A JSON representation of the complete Recommendation is stored in the
        result database (standard schema "celida"), if it is not already stored.

        The mapping between FHIR resources / profiles and objects is as follows:

        * Recommendation -> Recommendation
          * RecommendationPlan 1..* -> PopulationInterventionPair
            * EligibilityCriteria 1..* -> CriterionCombination / Criterion
            * InterventionActivity 1..* -> CriterionCombination / Criterion
            * Goal 1..* -> CriterionCombination / Criterion

        :param recommendation_url: The canonical URL of the recommendation.
        :param force_reload: If True, the recommendation is recreated from the FHIR source even if it is already
                             stored in the database.
        :return: The Recommendation object.
        """

        if not force_reload:
            recommendation = self.load_recommendation_from_database(recommendation_url)
            if recommendation is not None:
                logging.info(
                    f"Loaded recommendation {recommendation_url} (version={recommendation.version}) from database."
                )
                return recommendation

        rec = fhir.Recommendation(recommendation_url, self._fhir)

        pi_pairs: list[PopulationInterventionPair] = []

        base_criterion = PatientsActiveDuringPeriod(name="active_patients")

        for rec_plan in rec.plans():
            pi_pair = PopulationInterventionPair(
                name=rec_plan.name,
                url=rec_plan.url,
                base_criterion=base_criterion,
            )

            # parse population and create criteria
            characteristics = self._parse_characteristics(rec_plan.population)

            for characteristic in characteristics:
                pi_pair.add_population(characteristic_to_criterion(characteristic))

            # parse intervention and create criteria
            actions, selection_behavior = self._parse_actions(rec_plan.actions)
            comb_actions = self._action_combination(selection_behavior)

            for action in actions:
                if action is None:
                    raise ValueError("Action is None.")
                comb_actions.add(action.to_criterion())

            pi_pair.add_intervention(comb_actions)

            pi_pairs.append(pi_pair)

        recommendation = cohort.Recommendation(
            pi_pairs,
            base_criterion=base_criterion,
            url=rec.url,
            name=rec.name,
            title=rec.title,
            version=rec.version,
            description=rec.description,
        )

        self.register_recommendation(recommendation)

        return recommendation

    def execute(
        self,
        recommendation: cohort.Recommendation,
        start_datetime: datetime,
        end_datetime: datetime | None,
    ) -> int:
        """Executes the Recommendation"""
        # todo: improve documentation

        if end_datetime is None:
            end_datetime = datetime.now()
        # fixme: set start_datetime and end_datetime as class variables
        # fixme: potentially also register run_id as class variable

        with self._db.begin():
            self.register_recommendation(recommendation)
            run_id = self.register_run(
                recommendation, start_datetime=start_datetime, end_datetime=end_datetime
            )

        self.execute_recommendation(
            recommendation,
            run_id=run_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            use_multiprocessing=self._config.celida_ee_multiprocessing_use,
            multiprocessing_pool_size=self._config.celida_ee_multiprocessing_pool_size,
        )

        return run_id

    def load_recommendation_from_database(
        self, url: str, version: str | None = None
    ) -> cohort.Recommendation | None:
        """
        Loads a recommendation from the database. If version is None, the latest created recommendation is returned.
        """
        rec_table = result_db.Recommendation

        query = (
            select(rec_table)
            .where(rec_table.recommendation_url == url)
            .order_by(rec_table.create_datetime.desc())
        )
        if version is not None:
            query.where(rec_table.recommendation_version == version)

        with self._db.connect() as con:
            rec_db = con.execute(query).fetchone()

        if rec_db is not None:
            recommendation = cohort.Recommendation.from_json(
                rec_db.recommendation_json.decode()
            )
            recommendation.id = rec_db.recommendation_id
            return recommendation

        return None

    @staticmethod
    def _hash(obj: Serializable) -> tuple[bytes, str]:
        json = obj.json()
        return json, hashlib.sha256(json).hexdigest()

    def register_recommendation(self, recommendation: cohort.Recommendation) -> None:
        """Registers the Recommendation in the result database."""

        recommendation_table = result_db.Recommendation

        rec_json, rec_hash = self._hash(recommendation)
        query = select(recommendation_table).where(
            recommendation_table.recommendation_hash == rec_hash
        )

        with self._db.begin() as con:
            rec_db = con.execute(query).fetchone()

            if rec_db is not None:
                recommendation.id = rec_db.recommendation_id
            else:
                query = (
                    insert(recommendation_table)
                    .values(
                        recommendation_name=recommendation.name,
                        recommendation_title=recommendation.title,
                        recommendation_url=recommendation.url,
                        recommendation_version=recommendation.version,
                        recommendation_hash=rec_hash,
                        recommendation_json=rec_json,
                        create_datetime=datetime.now(),
                    )
                    .returning(recommendation_table.recommendation_id)
                )

                result = con.execute(query)
                recommendation.id = result.fetchone().recommendation_id

                con.commit()

            for pi_pair in recommendation.population_intervention_pairs():
                self.register_population_intervention_pair(
                    pi_pair, recommendation_id=recommendation.id
                )

                for criterion in pi_pair.flatten():
                    self.register_criterion(criterion)

    def register_population_intervention_pair(
        self, pi_pair: PopulationInterventionPair, recommendation_id: int
    ) -> None:
        """
        Registers the Population/Intervention Pair in the result database.

        :param pi_pair: The Population/Intervention Pair.
        :param recommendation_id: The ID of the Population/Intervention Pair.
        """
        _, pi_pair_hash = self._hash(pi_pair)
        query = select(result_db.PopulationInterventionPair).where(
            result_db.PopulationInterventionPair.pi_pair_hash == pi_pair_hash
        )
        with self._db.begin() as con:
            pi_pair_db = con.execute(query).fetchone()

            if pi_pair_db is not None:
                pi_pair.id = pi_pair_db.pi_pair_id
            else:
                query = (
                    insert(result_db.PopulationInterventionPair)
                    .values(
                        recommendation_id=recommendation_id,
                        pi_pair_url=pi_pair.url,
                        pi_pair_name=pi_pair.name,
                        pi_pair_hash=pi_pair_hash,
                    )
                    .returning(result_db.PopulationInterventionPair.pi_pair_id)
                )

                result = con.execute(query)
                pi_pair.id = result.fetchone().pi_pair_id

                con.commit()

    def register_criterion(self, criterion: Criterion) -> None:
        """
        Registers the Criterion in the result database.

        :param criterion: The Criterion.
        """
        _, crit_hash = self._hash(criterion)

        query = select(result_db.Criterion).where(
            result_db.Criterion.criterion_hash == crit_hash
        )
        with self._db.begin() as con:
            criterion_db = con.execute(query).fetchone()

            if criterion_db is not None:
                criterion.id = criterion_db.criterion_id
            else:
                query = (
                    insert(result_db.Criterion)
                    .values(
                        criterion_hash=crit_hash,
                        criterion_name=criterion.unique_name(),
                        criterion_description=criterion.description(),
                    )
                    .returning(result_db.Criterion.criterion_id)
                )

                result = con.execute(query)
                criterion.id = result.fetchone().criterion_id

    def register_run(
        self,
        recommendation: cohort.Recommendation,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> int:
        """Registers the run in the result database."""
        with self._db.begin() as con:
            query = (
                result_db.ExecutionRun.__table__.insert()
                .values(
                    recommendation_id=recommendation.id,
                    observation_start_datetime=start_datetime,
                    observation_end_datetime=end_datetime,
                    run_datetime=datetime.now(),
                    engine_version=__version__,
                )
                .returning(result_db.ExecutionRun.run_id)
            )

            result = con.execute(query).fetchone()

        return result.run_id

    def execute_recommendation(
        self,
        recommendation: cohort.Recommendation,
        run_id: int,
        start_datetime: datetime,
        end_datetime: datetime,
        use_multiprocessing: bool = False,
        multiprocessing_pool_size: int = 2,
    ) -> None:
        """
        Executes the Recommendation and stores the results in the result tables.
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

        bind_params = {
            "run_id": run_id,
            "observation_start_datetime": start_datetime,
            "observation_end_datetime": end_datetime,
        }

        # todo: warning: current implementation might run into memory problems ->
        #  then splitting by person might be necessary
        #  otherwise not needed intermediate results may be deleted after processing

        execution_graph = recommendation.execution_graph()
        task_runner: runner.TaskRunner

        if use_multiprocessing:
            task_runner = runner.ParallelTaskRunner(
                execution_graph, num_workers=multiprocessing_pool_size
            )
        else:
            task_runner = runner.SequentialTaskRunner(execution_graph)

        task_runner.run(bind_params)

    def insert_intervals(self, data: pd.DataFrame, con: sqlalchemy.Connection) -> None:
        """Inserts the intervals into the database."""
        if data.empty:
            return

        (
            data.to_sql(
                name=result_db.ResultInterval.__tablename__,
                con=con,
                if_exists="append",
                index=False,
            )
        )

    def fetch_patients(self, run_id: int) -> dict:
        """Retrieve list of patients associated with a single run."""
        assert isinstance(run_id, int)
        # todo: test / fix me
        t = result_db.ResultInterval.__table__.alias("result")
        t_criterion = result_db.Criterion.__table__.alias("criteria")
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
                    t.c.run_id == run_id,
                    t.c.pi_pair_id.is_(None),
                )
            )
        )

        return self._db.query(query)

    def fetch_criteria(self, run_id: int) -> dict:
        """Retrieve individual criteria associated with a single run."""
        assert isinstance(run_id, int)

        t_rec = result_db.Recommendation.__table__.alias("rec")
        t_run = result_db.ExecutionRun.__table__.alias("run")

        query = (
            select(t_rec.c.recommendation_hash, t_rec.c.recommendation_json)
            .join(t_run)
            .filter(t_run.c.run_id == run_id)
        )

        return self._db.query(query)

    def fetch_run(self, run_id: int) -> dict:
        """
        Retrieve information about a single run.
        """
        t_rec = result_db.Recommendation.__table__.alias("rec")
        t_run = result_db.ExecutionRun.__table__.alias("run")

        query = (
            select(
                t_run.c.run_id,
                t_run.c.observation_start_datetime,
                t_run.c.observation_end_datetime,
                t_rec.c.recommendation_id,
                t_rec.c.recommendation_url,
                t_rec.c.recommendation_version,
                t_rec.c.recommendation_hash,
            )
            .select_from(t_run)
            .join(t_rec)
            .filter(t_run.c.run_id == run_id)
        )

        return self._db.query(query).iloc[0].to_dict()

    # TODO: Should be based on run id
    def fetch_patient_data(
        self,
        person_id: int,
        criterion_name: str,
        recommendation: cohort.Recommendation,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> pd.DataFrame:
        """Retrieve patient data for a person and single criterion."""
        criterion = recommendation.get_criterion(criterion_name)

        statement = criterion.sql_select_data(person_id)
        bind_params = {
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
        }

        self._db.log_query(statement, bind_params)

        return self._db.query(statement, params=bind_params)
