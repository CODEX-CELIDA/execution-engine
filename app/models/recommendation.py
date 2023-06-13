from typing import TypedDict

from execution_engine.omop.cohort_definition import CohortDefinitionCombination


class Recommendation(TypedDict):
    """
    Recommendation for execution engine (for type hinting).
    """

    name: str
    title: str
    description: str
    cohort_definition: CohortDefinitionCombination