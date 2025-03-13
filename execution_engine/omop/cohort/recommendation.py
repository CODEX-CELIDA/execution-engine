import copy
import re
from typing import Iterator

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
from execution_engine.omop.db.celida.tables import ResultInterval
from execution_engine.util.serializable import SerializableDataClass


class Recommendation(SerializableDataClass):
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

    @classmethod
    def filter_symbols(cls, node: logic.Expr, filter_: logic.Expr) -> logic.Expr:
        """
        Filter (=AND-combine) all symbols by the applied filter function

        Used to filter all intervention criteria (symbols) by the population output in order to exclude
        all intervention events outside the population intervals, which may otherwise interfere with corrected
        determination of temporal combination, i.e. the presence of an intervention event during some time window.
        """

        if isinstance(node, logic.Symbol):
            return logic.LeftDependentToggle(left=filter_, right=node)
        elif isinstance(node, logic.Expr):
            converted_args = [cls.filter_symbols(a, filter_) for a in node.args]

            if any(a is not b for a, b in zip(node.args, converted_args)):
                node.update_args(*converted_args)

        return node

    @classmethod
    def filter_intervention_criteria_by_population(cls, expr: logic.Expr) -> logic.Expr:
        """
        Filter all intervention criteria by the output of the population.
        """

        from execution_engine.omop.cohort import PopulationInterventionPairExpr

        # we might make changes to the expression (e.g. filtering), so we must preserve
        # the original expression from the caller
        expr = copy.deepcopy(expr)

        def traverse(
            expr: logic.Expr,
        ) -> None:
            if isinstance(expr, PopulationInterventionPairExpr):
                p, i = expr.left, expr.right

                # filter all intervention criteria by the output of the population - this is performed to filter out
                # intervention events that outside of the population intervals (i.e. the time windows during which
                # patients are part of the population) as otherwise events outside of the population time may be picked up
                # by Temporal criteria that determine the presence of some event or condition during a specific time window.
                i = cls.filter_symbols(i, filter_=p)

                traverse(i)
                traverse(p)

            elif not expr.is_Atom:
                for child in expr.args:
                    traverse(child)

        traverse(expr)

        return expr

    def execution_graph(self) -> ExecutionGraph:
        """
        Get the execution maps of the full recommendation.

        The execution map of the full recommendation is constructed from combining the population and intervention
        execution maps of the individual population/intervention pairs of the recommendation.
        """

        expr_filtered = self.filter_intervention_criteria_by_population(self._expr)

        common_graph = ExecutionGraph.from_expression(
            expr_filtered,
            base_criterion=self._base_criterion,
            category=CohortCategory.POPULATION_INTERVENTION,
        )

        p_sink_nodes = common_graph.sink_nodes(CohortCategory.POPULATION)

        p_combination_node = logic.NoDataPreservingOr(
            *common_graph.sink_nodes(CohortCategory.POPULATION)
        )

        common_graph.add_node(
            p_combination_node, store_result=True, category=CohortCategory.POPULATION
        )

        common_graph.add_edges_from((src, p_combination_node) for src in p_sink_nodes)

        return common_graph

    def flatten(self) -> list[Criterion]:
        """
        Retrieve all criteria in a flat list
        """

        def traverse(expr: logic.BaseExpr) -> list[Criterion]:
            if expr.is_Atom:
                assert isinstance(expr, Criterion), f"Expected Symbol, got {expr}"
                return [expr]

            gathered = []
            for sub_expr in expr.args:
                gathered.extend(traverse(sub_expr))
            return gathered

        return [self._base_criterion] + traverse(self._expr)

    def population_intervention_pairs(self) -> Iterator[PopulationInterventionPairExpr]:
        """
        Iterate over all PopulationInterventionPairExpr in the expression tree.
        """

        def traverse(expr: logic.BaseExpr) -> Iterator[PopulationInterventionPairExpr]:
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
