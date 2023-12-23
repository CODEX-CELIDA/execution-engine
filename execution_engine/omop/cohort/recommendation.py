import itertools
import re
from typing import Any, Dict, Iterator, Self

from sqlalchemy import (
    Column,
    Date,
    Integer,
    MetaData,
    Select,
    Table,
    and_,
    bindparam,
    select,
)

import execution_engine.util.cohort_logic as logic
from execution_engine.constants import CohortCategory
from execution_engine.execution_graph import ExecutionGraph
from execution_engine.execution_map import ExecutionMap
from execution_engine.omop import cohort

# )
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.omop.criterion.factory import criterion_factory
from execution_engine.omop.db.celida.tables import RecommendationResultInterval
from execution_engine.omop.serializable import Serializable


class Recommendation(Serializable):
    """
    A recommendation OMOP as a collection of separate population/intervention pairs.

    A population/intervention pair represents an individual recommendation plan (i.e. one part of a single
    recommendation), whereas a Recommendation represents the whole recommendation, consisting of one or multiple
    recommendation plans = population/intervention pairs.
    """

    base_table: Table | None = None

    def __init__(
        self,
        pi_pairs: list[cohort.PopulationInterventionPair],
        base_criterion: Criterion,
        name: str,
        title: str,
        url: str,
        version: str,
        description: str,
        recommendation_id: int | None = None,
    ) -> None:
        self._pi_pairs: list[cohort.PopulationInterventionPair] = pi_pairs
        self._base_criterion: Criterion = base_criterion
        self._name: str = name
        self._title: str = title
        self._url: str = url
        self._version: str = version
        self._description: str = description
        # The id is used in the recommendation_id field in the result tables.
        self._id = recommendation_id

    @property
    def name(self) -> str:
        """
        Get the name of the recommendation.
        """
        return self._name

    @property
    def title(self) -> str:
        """
        Get the title of the recommendation.
        """
        return self._title

    @property
    def url(self) -> str:
        """
        Get the url of the recommendation.
        """
        return self._url

    @property
    def version(self) -> str:
        """
        Get the version of the recommendation.
        """
        return self._version

    @property
    def description(self) -> str:
        """
        Get the description of the recommendation.
        """
        return self._description

    def execution_graph(self) -> ExecutionGraph:
        """
        Get the execution maps of the full recommendation.

        The execution map of the full recommendation is constructed from combining the population and intervention
        execution maps of the individual population/intervention pairs of the recommendation.
        """

        p_maps = []
        i_maps = []
        pi_maps = []

        for pi_pair in self._pi_pairs:
            emap = pi_pair.execution_map()
            p, i = emap[CohortCategory.POPULATION], emap[CohortCategory.INTERVENTION]
            pi = p >> i
            pi.set_params(p.params)

            p_maps.append(p)
            i_maps.append(i)
            pi_maps.append(pi)

        p_map = ExecutionMap.combine_from(*p_maps, operator=logic.NoDataPreservingOr)
        i_map = ExecutionMap.combine_from(*i_maps, operator=logic.NoDataPreservingOr)
        pi_map = ExecutionMap.combine_from(*pi_maps, operator=logic.NoDataPreservingAnd)

        common_graph = (
            p_map.to_graph(CohortCategory.POPULATION)
            + i_map.to_graph(CohortCategory.INTERVENTION)
            + pi_map.to_graph(CohortCategory.POPULATION_INTERVENTION)
        )
        common_graph.set_sink_nodes_store()

        return common_graph

    def criteria(self) -> CriterionCombination:
        """
        Get the criteria of the recommendation.
        """
        criteria = CriterionCombination(
            name="root",
            exclude=False,
            category=CohortCategory.BASE,
            operator=CriterionCombination.Operator("OR"),
        )

        for pi_pair in self._pi_pairs:
            criteria.add(pi_pair.criteria())

        return criteria

    def flatten(self) -> list[Criterion]:
        """
        Retrieve all criteria in a flat list
        """
        return list(itertools.chain(*[pi_pair.flatten() for pi_pair in self._pi_pairs]))

    def population_intervention_pairs(
        self,
    ) -> Iterator[cohort.PopulationInterventionPair]:
        """
        Iterate over the population/intervention pairs.
        """
        yield from self._pi_pairs

    def __len__(self) -> int:
        """
        Get the number of population/intervention pairs.
        """
        return len(self._pi_pairs)

    def __getitem__(self, index: int) -> cohort.PopulationInterventionPair:
        """
        Get the population/intervention pair at the given index.
        """
        return self._pi_pairs[index]

    @staticmethod
    def to_table(name: str) -> Table:
        """
        Convert a name to a valid SQL table name.
        """
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        metadata = MetaData()
        return Table(
            name,
            metadata,
            Column("person_id", Integer, primary_key=True),
            Column("valid_date", Date),
        )

    @staticmethod
    def select_patients(category: CohortCategory) -> Select:
        """
        Select the patients in the given cohort category.
        """
        # todo: update me
        table = RecommendationResultInterval.__table__

        return select(table.c.person_id).where(
            and_(
                table.c.recommendation_run_id == bindparam("run_id"),
                table.c.pi_pair_id.is_(None),
                table.c.criterion_id.is_(None),
                table.c.cohort_category == category.name,
            )
        )

    def get_criterion(self, criterion_unique_name: str) -> Criterion:
        """
        Retrieve a criterion object by its unique name.
        """

        """        for pi_pair in self._pi_pairs:
            criterion = pi_pair.get_criterion(criterion_unique_name)
            if criterion is not None:
                return criterion

        raise ValueError(f"Could not find criterion '{criterion_unique_name}'")"""

        raise NotImplementedError()

    def dict(self) -> dict:
        """
        Get the combination as a dictionary.
        """
        return {
            "id": self._id,
            "population_intervention_pairs": [c.dict() for c in self._pi_pairs],
            "base_criterion": self._base_criterion.dict(),
            "recommendation_name": self._name,
            "recommendation_title": self._title,
            "recommendation_url": self._url,
            "recommendation_version": self._version,
            "recommendation_description": self._description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Create a combination from a dictionary.
        """
        base_criterion = criterion_factory(**data["base_criterion"])
        assert isinstance(
            base_criterion, Criterion
        ), "Base criterion must be a Criterion"

        return cls(
            pi_pairs=[
                cohort.PopulationInterventionPair.from_dict(c)
                for c in data["population_intervention_pairs"]
            ],
            base_criterion=base_criterion,
            name=data["recommendation_name"],
            title=data["recommendation_title"],
            url=data["recommendation_url"],
            version=data["recommendation_version"],
            description=data["recommendation_description"],
            recommendation_id=data["id"] if "id" in data else None,
        )
