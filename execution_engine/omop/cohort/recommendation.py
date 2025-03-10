import re
from typing import Any, Dict, Iterator, Self, cast

import networkx as nx
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

import execution_engine.util.logic as logic
from execution_engine.constants import CohortCategory
from execution_engine.execution_graph import ExecutionGraph
from execution_engine.omop.cohort.population_intervention_pair import (
    PopulationInterventionPairExpr,
)
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.factory import criterion_factory
from execution_engine.omop.db.celida.tables import ResultInterval
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
        expr: logic.BooleanFunction,
        base_criterion: Criterion,
        name: str,
        title: str,
        url: str,
        version: str,
        description: str,
        package_version: str | None = None,
    ) -> None:
        self._expr: logic.BooleanFunction = expr
        self._base_criterion: Criterion = base_criterion
        self._name: str = name
        self._title: str = title
        self._url: str = url
        self._version: str = version
        self._description: str = description
        self._package_version: str | None = package_version

    def __repr__(self) -> str:
        """
        Get the string representation of the recommendation.
        """
        pi_repr = "\n".join(
            [("    " + line) for line in repr(self._expr).split("\n")]
        ).strip()
        pi_repr = (
            pi_repr[0] + "\n    " + pi_repr[1:-2] + pi_repr[-2] + "\n  " + pi_repr[-1]
        )
        return (
            f"{self.__class__.__name__}(\n"
            f"  pi_pairs={pi_repr},\n"
            f"  base_criterion={repr(self._base_criterion)},\n"
            f"  name={repr(self._name)},\n"
            f"  title={repr(self._title)},\n"
            f"  url={repr(self._url)},\n"
            f"  version={repr(self._version)},\n"
            f"  description={repr(self._description)},\n"
            f"  package_version={repr(self._package_version)},\n"
            f")"
        )

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
    def package_version(self) -> str | None:
        """
        Get the version of the recommendation package.
        """
        return self._package_version

    @property
    def description(self) -> str:
        """
        Get the description of the recommendation.
        """
        return self._description

    @property
    def base_criterion(self) -> Criterion | None:
        """
        Get the base criterion of the recommendation.
        """
        return self._base_criterion

    def execution_graph(self) -> ExecutionGraph:
        """
        Get the execution maps of the full recommendation.

        The execution map of the full recommendation is constructed from combining the population and intervention
        execution maps of the individual population/intervention pairs of the recommendation.
        """

        # p_sink_nodes = []
        # pi_sink_nodes = []

        common_graph = ExecutionGraph.from_expression(
            self._expr,
            base_criterion=self._base_criterion,
            category=CohortCategory.POPULATION_INTERVENTION,
        )

        # for pi_pair in self.population_intervention_pairs():
        #     subgraph: ExecutionGraph = cast(ExecutionGraph, common_graph.subgraph(nx.ancestors(common_graph, pi_pair) | {pi_pair}))
        #     p_sink_nodes.append(subgraph.sink_node(CohortCategory.POPULATION))
        #     pi_sink_nodes.append(subgraph.sink_node(CohortCategory.POPULATION_INTERVENTION))

        p_sink_nodes = common_graph.sink_nodes(CohortCategory.POPULATION)

        p_combination_node = logic.NoDataPreservingOr(
            *common_graph.sink_nodes(CohortCategory.POPULATION)
        )
        # pi_combination_node = logic.NoDataPreservingAnd(
        #     *pi_sink_nodes,
        # )

        common_graph.add_node(
            p_combination_node, store_result=True, category=CohortCategory.POPULATION
        )

        # common_graph.add_node(
        #     pi_combination_node,
        #     store_result=True,
        #     category=CohortCategory.POPULATION_INTERVENTION,
        # )

        common_graph.add_edges_from((src, p_combination_node) for src in p_sink_nodes)
        # common_graph.add_edges_from((src, pi_combination_node) for src in pi_sink_nodes)

        import json

        with open("/home/glichtner/cyto.json", "w") as f:
            json.dump({"elements": common_graph.to_cytoscape_dict()}, f, indent=4)

        if not nx.is_directed_acyclic_graph(common_graph):
            raise ValueError("The recommendation execution graph is not a DAG.")

        return common_graph

    def flatten(self) -> list[Criterion]:
        """
        Retrieve all criteria in a flat list
        """

        def traverse(expr: logic.Expr) -> list[Criterion]:
            if expr.is_Atom:
                assert isinstance(expr, logic.Symbol), f"Expected Symbol, got {expr}"
                return [expr.criterion]

            gathered = []
            for sub_expr in expr.args:
                gathered.extend(traverse(sub_expr))
            return gathered

        return [self._base_criterion] + traverse(self._expr)

    def population_intervention_pairs(self) -> Iterator[PopulationInterventionPairExpr]:
        """
        Iterate over all PopulationInterventionPairExpr in the expression tree.
        """

        def traverse(expr: logic.Expr) -> Iterator[PopulationInterventionPairExpr]:
            if isinstance(expr, PopulationInterventionPairExpr):
                yield expr
            else:
                for sub_expr in expr.args:
                    yield from traverse(sub_expr)

        yield from traverse(self._expr)

    def __str__(self) -> str:
        """
        Get the string representation of the recommendation.
        """
        return f"Recommendation(name='{self._name}', description='{self.description}')"

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
        table = ResultInterval.__table__

        return select(table.c.person_id).where(
            and_(
                table.c.run_id == bindparam("run_id"),
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

    def reset_state(self) -> None:
        """
        Reset the state of the recommendation.

        Sets all _id attributes to None in the recommendation and all its population/intervention pairs and criteria.
        """
        self._id = None

        for pi_pair in self.population_intervention_pairs():
            pi_pair._id = None

        for criterion in self.flatten():
            criterion._id = None

    def dict(self) -> dict:
        """
        Get the combination as a dictionary.
        """
        base_criterion = self._base_criterion
        return {
            "expr": self._expr.dict(),
            "base_criterion": {
                "class_name": base_criterion.__class__.__name__,
                "data": base_criterion.dict(),
            },
            "recommendation_name": self._name,
            "recommendation_title": self._title,
            "recommendation_url": self._url,
            "recommendation_version": self._version,
            "recommendation_package_version": self._package_version,
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
            expr=cast(
                logic.BooleanFunction, logic.BooleanFunction.from_dict(data["expr"])
            ),
            base_criterion=base_criterion,
            name=data["recommendation_name"],
            title=data["recommendation_title"],
            url=data["recommendation_url"],
            version=data["recommendation_version"],
            description=data["recommendation_description"],
            package_version=data["recommendation_package_version"],
        )
