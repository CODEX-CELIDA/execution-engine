from typing import cast

import execution_engine.util.logic as logic
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.serializable import Serializable


class PopulationInterventionPairExpr(logic.LeftDependentToggle, Serializable):
    """
    A logical expression that ties together a population expression and an intervention expression,
    plus any extra info like name, url, base_criterion, etc.

    A population/intervention pair in OMOP as a collection of separate criteria.

    A population/intervention pair represents an individual recommendation plan (i.e. one part of a single recommendation),
    whereas a population/intervention pair combination represents the whole recommendation, consisting of one or multiple
    recommendation plans = population/intervention pairs.
    In turn, a population/intervention pair is a collection of criteria, which can be either a single criterion or a combination
    of criteria (i.e. a criterion combination).
    These single criteria can be either a single criterion (e.g. "has condition X") or a combination of criteria
    (e.g. "has condition X and lab value Y >= Z").
    """

    _name: str
    _url: str
    _base_criterion: Criterion

    def __new__(
        cls,
        population_expr: logic.BaseExpr,
        intervention_expr: logic.BaseExpr,
        *,
        name: str,
        url: str,
        base_criterion: Criterion,
        **kwargs: dict,
    ) -> "PopulationInterventionPairExpr":
        """
        Create a new PopulationInterventionPairExpr object.
        """
        self = cast(
            PopulationInterventionPairExpr,
            super().__new__(
                cls, left=population_expr, right=intervention_expr, **kwargs
            ),
        )
        self._name = name
        self._url = url
        self._base_criterion = base_criterion

        return self

    @property
    def name(self) -> str:
        """
        The name of the population/intervention pair.
        """
        return self._name

    @property
    def url(self) -> str:
        """
        The URL of the population/intervention pair.
        """
        return self._url

    @property
    def base_criterion(self) -> Criterion:
        """
        The base criterion of the population/intervention pair.
        """
        return self._base_criterion

    def __repr__(self) -> str:
        """
        Return a string representation of the object.
        """
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"url={self.url!r}, "
            f"base_criterion={self.base_criterion!r}, "
            f"left={self.left!r}, right={self.right!r}, "
        )

    def get_instance_variables(self, immutable: bool = False) -> dict | tuple:
        """
        Get the instance variables of the object.
        """
        # Include the base class instance variables plus the ones we added:
        base_vars = cast(dict, super().get_instance_variables(immutable=False))
        base_vars["name"] = self.name
        base_vars["url"] = self.url
        base_vars["base_criterion"] = self.base_criterion

        # Convert to tuple if needed
        if immutable:
            return tuple(sorted(base_vars.items()))
        return base_vars
