from sqlalchemy import Date, Select, func, select, text
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
    initial_cte = select(
        RecommendationResultInterval.recommendation_run_id,
        RecommendationResultInterval.plan_id,
        RecommendationResultInterval.criterion_id,
        RecommendationResultInterval.cohort_category,
        RecommendationResultInterval.person_id,
        RecommendationResultInterval.interval_type,
        RecommendationResultInterval.interval_start,
        RecommendationResultInterval.interval_end,
        func.least(
            RecommendationResultInterval.interval_end,
            func.date_trunc("day", RecommendationResultInterval.interval_start)
            + text("interval '1 day'"),
        ).label("adjusted_interval_end"),
    ).where(
        RecommendationResultInterval.interval_type.in_(
            [IntervalType.POSITIVE, IntervalType.NO_DATA]
        )
    )

    initial_subquery = initial_cte.subquery()

    recursive_cte = select(
        initial_subquery.c.recommendation_run_id,
        initial_subquery.c.plan_id,
        initial_subquery.c.criterion_id,
        initial_subquery.c.cohort_category,
        initial_subquery.c.person_id,
        initial_subquery.c.interval_type,
        func.date_trunc("day", initial_subquery.c.adjusted_interval_end).label(
            "interval_start"
        ),
        initial_subquery.c.interval_end,
        func.least(
            initial_subquery.c.interval_end,
            func.date_trunc("day", initial_subquery.c.adjusted_interval_end)
            + text("interval '1 day'"),
        ).label("adjusted_interval_end"),
    ).where(initial_subquery.c.adjusted_interval_end < initial_subquery.c.interval_end)

    # Combine initial and recursive parts using text
    full_cte = initial_cte.union_all(recursive_cte).cte(
        recursive=True, name="date_intervals"
    )
    # todo: make sure that day grenzen are based on the timezone that is set ! - implement a test
    # todo: implement test for view (in general)
    # todo: implement test for trigger (non-overlapping intervals)
    # Final select statement
    final_query = (
        select(
            full_cte.c.recommendation_run_id,
            full_cte.c.plan_id,
            full_cte.c.criterion_id,
            full_cte.c.cohort_category,
            full_cte.c.person_id,
            func.cast(func.date_trunc("day", full_cte.c.interval_start), Date).label(
                "valid_date"
            ),
            func.date_trunc("day", full_cte.c.interval_start).label("valid_datetime"),
        )
        .group_by(
            full_cte.c.recommendation_run_id,
            full_cte.c.plan_id,
            full_cte.c.criterion_id,
            full_cte.c.cohort_category,
            full_cte.c.person_id,
            func.date_trunc("day", full_cte.c.interval_start),
        )
        .having(
            func.sum(
                func.extract("epoch", full_cte.c.adjusted_interval_end)
                - func.extract("epoch", full_cte.c.interval_start)
            )
            == 24 * 60 * 60
        )
    )

    return final_query


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