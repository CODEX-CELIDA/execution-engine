import sys

from app.dependencies import get_recommendations

from execution_engine.omop.cohort import Recommendation

sys.path.append("..")
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter()


@router.get("/recommendation/list")
async def recommendation_list(
    recommendations: dict = Depends(get_recommendations),
) -> list[dict[str, str]]:
    """
    Get available recommendations by URL
    """
    return [
        {
            "recommendation_name": rec["name"],
            "recommendation_title": rec["title"],
            "recommendation_description": rec["description"],
            "recommendation_url": rec_url,
        }
        for rec_url, rec in recommendations.items()
    ]


@router.get("/recommendation/criteria")
async def recommendation_criteria(
    recommendation_url: str,
    recommendations: dict = Depends(get_recommendations),
) -> dict:
    """
    Get criteria names by recommendation URL
    """

    if recommendation_url not in recommendations:
        raise HTTPException(status_code=404, detail="recommendation not found")

    recommendation: Recommendation = recommendations[recommendation_url][
        "recommendation"
    ]

    data = []

    for c in recommendation.flatten():
        data.append(
            {
                "unique_name": c.unique_name(),
                "description": c.description(),
                "cohort_category": c.category,
                "concept_name": c.concept.concept_name.title(),
            }
        )

    return {"criterion": data}
