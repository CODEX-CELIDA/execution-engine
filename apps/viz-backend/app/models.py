from datetime import datetime

from pydantic import BaseModel


class RecommendationRun(BaseModel):
    """
    Represents a single recommendation run.
    """

    run_id: int
    observation_start_datetime: datetime
    observation_end_datetime: datetime
    run_datetime: datetime


class Interval(BaseModel):
    """
    Represents a single interval in a recommendation run.
    """

    person_id: int
    pi_pair_id: int | None = None
    criterion_id: int | None = None
    pi_pair_name: str | None = None
    criterion_name: str | None = None
    interval_type: str
    interval_start: datetime
    interval_end: datetime
    cohort_category: str
