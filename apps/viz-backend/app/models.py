from datetime import date, datetime

from pydantic import BaseModel


class Recommendation(BaseModel):
    """
    Represents a single recommendation.
    """

    recommendation_id: int
    recommendation_name: str
    recommendation_title: str
    recommendation_url: str
    recommendation_version: str | None = None
    recommendation_package_version: str | None = None
    create_datetime: datetime


class RecommendationRun(BaseModel):
    """
    Represents a single recommendation run.
    """

    run_id: int
    observation_start_datetime: datetime
    observation_end_datetime: datetime
    run_datetime: datetime
    recommendation_name: str
    recommendation_title: str
    recommendation_url: str
    recommendation_description: str


class Interval(BaseModel):
    """
    Represents a single interval in a recommendation run.
    """

    person_id: int
    pi_pair_id: int | None = None
    criterion_id: int | None = None
    pi_pair_name: str | None = None
    criterion_description: str | None = None
    interval_type: str
    interval_start: datetime
    interval_end: datetime
    cohort_category: str


class DayCoverage(BaseModel):
    """
    Represents a single day of coverage in a recommendation run.
    """

    person_id: int
    cohort_category: str
    valid_date: date  # Add this to match `view_full_day_coverage`
