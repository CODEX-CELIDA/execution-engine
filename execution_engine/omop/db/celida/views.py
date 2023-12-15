from sqlalchemy import Date, Select, func, select
from sqlalchemy.dialects.postgresql import INTERVAL
from sqlalchemy.sql.functions import concat

from execution_engine.constants import IntervalType
from execution_engine.omop.db.base import Base
from execution_engine.omop.db.celida.tables import (
    RecommendationResultInterval,
    RecommendationRun,
)
from execution_engine.omop.db.view import view


def view_full_day_coverage() -> Select:
    """
    Return Select statement for view to calculate the full day coverage of recommendation results.
    """

    one_day = func.cast(concat(1, "day"), INTERVAL)
    one_second = func.cast(concat(1, "second"), INTERVAL)
    rr = RecommendationRun.__table__.alias("rr")
    rri = RecommendationResultInterval.__table__.alias("rri")

    date_series = select(
        RecommendationRun.recommendation_run_id,
        func.generate_series(
            func.date_trunc("day", RecommendationRun.observation_start_datetime),
            func.date_trunc("day", RecommendationRun.observation_end_datetime),
            one_day,
        )
        .cast(Date)
        .label("day"),
        RecommendationRun.observation_start_datetime,
        RecommendationRun.observation_end_datetime,
    ).cte("date_series")

    interval_coverage = (
        select(
            date_series.c.recommendation_run_id,
            rri.c.person_id,
            rri.c.cohort_category,
            rri.c.plan_id,
            rri.c.criterion_id,
            date_series.c.day,
            func.sum(
                func.least(
                    date_series.c.day + one_day,
                    rri.c.interval_end,
                    date_series.c.observation_end_datetime,
                )
                - func.greatest(
                    date_series.c.day,
                    rri.c.interval_start,
                    date_series.c.observation_start_datetime,
                )
            ).label("covered_time"),
        )
        .select_from(date_series)
        .outerjoin(
            rri,
            (date_series.c.recommendation_run_id == rri.c.recommendation_run_id)
            & (rri.c.interval_type.in_(["POSITIVE", "NO_DATA"])),
        )
        .filter(
            rri.c.interval_start < date_series.c.day + one_day,
            rri.c.interval_end > date_series.c.day,
        )
        .group_by(
            date_series.c.recommendation_run_id,
            date_series.c.day,
            rri.c.person_id,
            rri.c.cohort_category,
            rri.c.plan_id,
            rri.c.criterion_id,
        )
        .cte("interval_coverage")
    )

    observation_start_date = func.date_trunc(
        "day", rr.c.observation_start_datetime
    ).cast(Date)
    observation_end_date = func.date_trunc("day", rr.c.observation_end_datetime).cast(
        Date
    )
    covered_time = interval_coverage.c.covered_time
    day = interval_coverage.c.day.label("valid_date")

    # subtract one second from the covered time because 00:00:00 - 23:59:59 is considered to be a full day
    covered_dates = (
        select(
            interval_coverage.c.recommendation_run_id,
            interval_coverage.c.person_id,
            interval_coverage.c.cohort_category,
            interval_coverage.c.plan_id,
            interval_coverage.c.criterion_id,
            day,
        )
        .join(
            rr, rr.c.recommendation_run_id == interval_coverage.c.recommendation_run_id
        )
        .filter(
            (
                (day == observation_start_date)
                & (day == observation_end_date)
                & (
                    covered_time
                    >= rr.c.observation_end_datetime
                    - rr.c.observation_start_datetime
                    - one_second
                )
            )
            | (
                (day == observation_start_date)
                & (
                    covered_time
                    >= rr.c.observation_start_datetime
                    - observation_start_date
                    - one_second
                )
            )
            | (
                (day == observation_end_date)
                & (
                    covered_time
                    >= rr.c.observation_end_datetime
                    - func.date_trunc("day", rr.c.observation_end_datetime)
                    - one_second
                )
            )
            | (
                (day != observation_start_date)
                & (day != observation_end_date)
                & (covered_time >= one_day - one_second)
            )
        )
    )

    return covered_dates


def view_partial_day_coverage() -> Select:
    """
    Return Select statement for view to calculate the partial day coverage of recommendation results.
    """
    subquery = (
        select(
            func.generate_series(
                func.greatest(
                    RecommendationResultInterval.interval_start.cast(Date),
                    RecommendationRun.observation_start_datetime.cast(Date),
                ),
                func.least(
                    RecommendationResultInterval.interval_end.cast(Date),
                    RecommendationRun.observation_end_datetime.cast(Date),
                ),
                func.cast(concat(1, " day"), INTERVAL),
            )
            .cast(Date)
            .label("date_series")
        )
        .lateral()
        .alias("t_date_series")
    )

    # Creating the main query
    stmt = (
        select(
            RecommendationResultInterval.recommendation_run_id,
            RecommendationResultInterval.plan_id,
            RecommendationResultInterval.criterion_id,
            RecommendationResultInterval.cohort_category,
            RecommendationResultInterval.person_id,
            subquery.c.date_series.label("valid_date"),
        )
        .distinct()
        .where(
            RecommendationResultInterval.interval_type.in_(
                [IntervalType.POSITIVE, IntervalType.NO_DATA]
            )
        )
    )

    return stmt


full_day_coverage = view(
    "full_day_coverage",
    Base.metadata,
    view_full_day_coverage(),
    schema="celida",
)

partial_day_coverage = view(
    "partial_day_coverage",
    Base.metadata,
    view_partial_day_coverage(),
    schema="celida",
)
