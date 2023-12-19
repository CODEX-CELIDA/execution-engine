from typing import TypedDict

from execution_engine.omop.cohort import Recommendation as RecommendationTable


class Recommendation(TypedDict):
    """
    Recommendation for execution engine (for type hinting).
    """

    name: str
    title: str
    description: str
    recommendation: RecommendationTable
