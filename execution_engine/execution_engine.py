import hashlib
import json
import logging
from datetime import datetime

import pandas as pd
import sqlalchemy
from sqlalchemy import and_, insert, select, update

from execution_engine import __version__
from execution_engine.builder import ExecutionEngineBuilder
from execution_engine.clients import fhir_client, omopdb
from execution_engine.converter.recommendation_factory import (
    FhirToRecommendationFactory,
)
from execution_engine.omop import cohort
from execution_engine.omop.cohort import PopulationInterventionPair
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.db.celida import tables as result_db
from execution_engine.omop.serializable import Serializable
from execution_engine.task import runner


class ExecutionEngine:
    """
    The Execution Engine is responsible for reading recommendations in CPG-on-EBM-on-FHIR format, creating
    corresponding Criterion objects from then and executing them on the OMOP database to yield a cohort of patients
    that match the criteria (and the combination of criteria).
    """

    # todo: improve documentation

    def __init__(
        self,
        builder: ExecutionEngineBuilder | None = None,
        verbose: bool = False,
    ) -> None:
        # late import to allow pytest to set env variables first
        from execution_engine.settings import get_config

        self.setup_logging(verbose)
        self._fhir = fhir_client
        self._db = omopdb
        self._db.init()

        self._config = get_config()
        self.fhir_parser = FhirToRecommendationFactory(builder)

    @staticmethod
    def setup_logging(verbose: bool = False) -> None:
        """Sets up logging."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.DEBUG if verbose else logging.INFO,
        )

    def load_recommendation(
        self,
        recommendation_url: str,
        recommendation_package_version: str = "latest",
        force_reload: bool = False,
        parser_version: int = 2,
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
        :param recommendation_package_version: The version of the recommendation.
        :param force_reload: If True, the recommendation is recreated from the FHIR source even if it is already
                             stored in the database.
        :param parser_version: The version of the FHIR parser to use. Currently, only version 1 and 2 (default) are
            supported.
        :return: The Recommendation object.
        """

        if not force_reload:
            recommendation = self.load_recommendation_from_database(
                recommendation_url, recommendation_package_version
            )
            if recommendation is not None:
                logging.info(
                    f"Loaded recommendation {recommendation_url} (version={recommendation.version}, "
                    f"package version={recommendation_package_version}) from database."
                )
                return recommendation

        # recommendation could not be loaded from database, fetch it from FHIR server

        recommendation = self.fhir_parser.parse_recommendation_from_url(
            url=recommendation_url,
            package_version=recommendation_package_version,
            parser_version=parser_version,
            fhir_client=self._fhir,
        )

        self.register_recommendation(recommendation)

        return recommendation

    def execute(
        self,
        recommendation: cohort.Recommendation,
        start_datetime: datetime,
        end_datetime: datetime | None,
    ) -> int:
        """
        Executes a recommendation and stores the results in the result database.

        :param recommendation: The Recommendation object (loaded from ExecutionEngine.load_recommendation).
        :param start_datetime: The start of the observation window.
        :param end_datetime: The end of the observation window. If None, the current time is used.
        :return: The ID of the run.
        """
        # todo: improve documentation

        if end_datetime is None:
            end_datetime = datetime.now()
        # fixme: set start_datetime and end_datetime as class variables
        # fixme: potentially also register run_id as class variable

        logging.info(
            f"Executing recommendation {recommendation.url} (execution engine version={__version__})."
        )

        with self._db.begin():
            # If the recommendation has been loaded from the
            # database, its _id slot is not None. Otherwise, register
            # the recommendation to store it into the database and
            # assign an id.
            if recommendation._id is None:
                self.register_recommendation(recommendation)
            run_id = self.register_run(
                recommendation, start_datetime=start_datetime, end_datetime=end_datetime
            )

        self.execute_recommendation(
            recommendation,
            run_id=run_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            use_multiprocessing=self._config.multiprocessing_use,
            multiprocessing_pool_size=self._config.multiprocessing_pool_size,
        )

        return run_id

    def load_recommendation_from_database(
        self, url: str, version: str | None = None, package_version: str | None = None
    ) -> cohort.Recommendation | None:
        """
        Loads a recommendation from the database.

        :param url: The canonical URL of the recommendation.
        :param version: The version of the recommendation. If version is None, the latest created recommendation is
            returned.
        :param package_version: The version of the recommendation package.
        :return: The Recommendation object or None if the recommendation is not found.
        """

        if package_version is not None and version is not None:
            raise ValueError("Only one of version and package_version can be set.")

        rec_table = result_db.Recommendation

        query = (
            select(rec_table)
            .where(rec_table.recommendation_url == url)
            .order_by(rec_table.create_datetime.desc())
        )
        if version is not None:
            query.where(rec_table.recommendation_version == version)
        elif package_version is not None:
            query.where(rec_table.recommendation_package_version == package_version)

        with self._db.connect() as con:
            rec_db = con.execute(query).fetchone()

        if rec_db is not None:
            recommendation = cohort.Recommendation.from_json(
                rec_db.recommendation_json.decode()
            )
            # All objects in the deserialized object graph must have
            # an id.
            assert recommendation._id is not None
            assert recommendation._base_criterion._id is not None
            for pi_pair in recommendation._pi_pairs:
                assert pi_pair._id is not None
                for criterion in pi_pair.flatten():
                    assert criterion._id is not None
            return recommendation

        return None

    @staticmethod
    def _hash(obj: Serializable) -> tuple[bytes, str]:
        json = obj.json()
        return json, hashlib.sha256(json).hexdigest()

    def register_recommendation(self, recommendation: cohort.Recommendation) -> None:
        """Registers the Recommendation in the result database."""
        # Get the hash but ignore the JSON representation for now
        # since we will compute and insert a complete JSON
        # representation later when we know all ids.
        _, rec_hash = self._hash(recommendation)
        recommendation_table = result_db.Recommendation
        # We look for a recommendation with the computed hash in the
        # database. If there is one, set the id of our recommendation
        # to the stored id. Otherwise, store our recommendation
        # (without the JSON representation) in the database and
        # receive the fresh id.
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
                        recommendation_package_version=recommendation.package_version,
                        recommendation_hash=rec_hash,
                        recommendation_json=bytes(),  # updated later
                        create_datetime=datetime.now(),
                    )
                    .returning(recommendation_table.recommendation_id)
                )

                result = con.execute(query)
                recommendation.id = result.fetchone().recommendation_id
        # Register all child objects. After that, the recommendation
        # and all child objects have valid ids (either restored or
        # fresh).
        self.register_criterion(recommendation._base_criterion)
        for pi_pair in recommendation.population_intervention_pairs():
            self.register_population_intervention_pair(
                pi_pair, recommendation_id=recommendation.id
            )
            for criterion in pi_pair.flatten():
                self.register_criterion(criterion)

        assert recommendation.id is not None
        # TODO(jmoringe): mypy doesn't like this one. Not sure why.
        # assert recommendation._base_criterion._id is not None
        for pi_pair in recommendation._pi_pairs:
            assert pi_pair._id is not None
            for criterion in pi_pair.flatten():
                assert criterion._id is not None

        # Update the recommendation in the database with the final
        # JSON representation and execution graph (now that
        # recommendation id, criteria ids and pi pair is are known)
        # TODO(jmoringe): only when necessary
        with self._db.begin() as con:
            rec_graph: bytes = json.dumps(
                recommendation.execution_graph().to_cytoscape_dict(), sort_keys=True
            ).encode()

            rec_json: bytes = recommendation.json()
            logging.info(f"Storing recommendation {recommendation}")
            update_query = (
                update(recommendation_table)
                .where(recommendation_table.recommendation_id == recommendation.id)
                .values(
                    recommendation_json=rec_json,
                    recommendation_execution_graph=rec_graph,
                )
            )

            con.execute(update_query)

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
                        criterion_description=criterion.description(),
                    )
                    .returning(result_db.Criterion.criterion_id)
                )

                result = con.execute(query)
                criterion.id = result.fetchone().criterion_id

        assert criterion.id is not None

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
