from sqlalchemy import Date, Select, and_, cast, func, not_, or_, select, text, true
from sqlalchemy.dialects.postgresql import INTERVAL
from sqlalchemy.sql.functions import concat

from execution_engine.omop.db.base import Base
from execution_engine.omop.db.celida.schema import SCHEMA_NAME
from execution_engine.omop.db.celida.tables import (
    Criterion,
    ExecutionRun,
    PopulationInterventionPair,
    ResultInterval,
)
from execution_engine.omop.db.view import view
from execution_engine.util.interval import IntervalType


def view_interval_coverage_slow() -> Select:
    """
    Return Select statement for view to calculate the full day coverage of recommendation results.
    """

    one_day = func.cast(concat(1, "day"), INTERVAL)
    rri = ResultInterval.__table__.alias("rri")

    date_series = select(
        ExecutionRun.run_id,
        func.generate_series(
            func.date_trunc("day", ExecutionRun.observation_start_datetime),
            func.date_trunc("day", ExecutionRun.observation_end_datetime),
            one_day,
        )
        .cast(Date)
        .label("day"),
        ExecutionRun.observation_start_datetime,
        ExecutionRun.observation_end_datetime,
    ).cte("date_series")

    interval_coverage = (
        select(
            date_series.c.run_id,
            rri.c.person_id,
            rri.c.cohort_category,
            rri.c.pi_pair_id,
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
            func.bool_or(rri.c.interval_type == IntervalType.POSITIVE).label(
                "has_positive"
            ),
            func.bool_or(rri.c.interval_type == IntervalType.NO_DATA).label(
                "has_no_data"
            ),
            func.bool_or(rri.c.interval_type == IntervalType.NOT_APPLICABLE).label(
                "has_not_applicable"
            ),
            func.bool_or(rri.c.interval_type == IntervalType.NEGATIVE).label(
                "has_negative"
            ),
        )
        .select_from(date_series)
        .outerjoin(rri, (date_series.c.run_id == rri.c.run_id))
        .filter(
            rri.c.interval_start < date_series.c.day + one_day,
            rri.c.interval_end > date_series.c.day,
        )
        .group_by(
            date_series.c.run_id,
            date_series.c.day,
            rri.c.person_id,
            rri.c.cohort_category,
            rri.c.pi_pair_id,
            rri.c.criterion_id,
        )
    )

    return interval_coverage


interval_coverage_slow = view(
    "interval_coverage_slow",
    Base.metadata,
    view_interval_coverage_slow(),
    schema=SCHEMA_NAME,
)


def view_interval_coverage() -> Select:
    """
    Return Select statement for view to calculate the full day coverage of recommendation results.
    """

    # Aliases for tables
    ri = ResultInterval.__table__.alias("ri")
    er = ExecutionRun.__table__.alias("er")

    # Define the person_intervals CTE
    person_intervals = (
        select(
            ri.c.run_id,
            ri.c.person_id,
            ri.c.cohort_category,
            ri.c.pi_pair_id,
            ri.c.criterion_id,
            ri.c.interval_type,
            cast(
                func.greatest(ri.c.interval_start, er.c.observation_start_datetime),
                Date,
            ).label("interval_start_date"),
            cast(
                func.least(ri.c.interval_end, er.c.observation_end_datetime)
                - text("INTERVAL '1 SECOND'"),
                Date,
            ).label("interval_end_date"),
        )
        .select_from(ri.join(er, ri.c.run_id == er.c.run_id))
        .where(
            ri.c.interval_end >= er.c.observation_start_datetime,
            ri.c.interval_start <= er.c.observation_end_datetime,
        )
        .cte("person_intervals")
    )

    pi = person_intervals.alias("pi")

    # Define the generate_series function as a lateral table and alias it as 'd'
    generate_series = func.generate_series(
        pi.c.interval_start_date, pi.c.interval_end_date, text("INTERVAL '1 day'")
    )

    # Use lateral join with the generate_series function
    d = generate_series.table_valued("date").lateral("d")

    # Build the main query
    interval_coverage = (
        select(
            pi.c.run_id,
            pi.c.person_id,
            pi.c.cohort_category,
            pi.c.pi_pair_id,
            pi.c.criterion_id,
            d.c.date.label("day"),
            func.bool_or(pi.c.interval_type == IntervalType.POSITIVE).label(
                "has_positive"
            ),
            func.bool_or(pi.c.interval_type == IntervalType.NEGATIVE).label(
                "has_negative"
            ),
            func.bool_or(pi.c.interval_type == IntervalType.NOT_APPLICABLE).label(
                "has_not_applicable"
            ),
            func.bool_or(pi.c.interval_type == IntervalType.NO_DATA).label(
                "has_no_data"
            ),
        )
        .select_from(pi.join(d, true()))
        .group_by(
            pi.c.run_id,
            pi.c.person_id,
            pi.c.cohort_category,
            pi.c.pi_pair_id,
            pi.c.criterion_id,
            d.c.date,
        )
        .order_by(
            pi.c.person_id,
            pi.c.cohort_category,
            pi.c.pi_pair_id,
            pi.c.criterion_id,
            d.c.date,
            pi.c.run_id,
        )
    )

    return interval_coverage


interval_coverage = view(
    "interval_coverage", Base.metadata, view_interval_coverage(), schema=SCHEMA_NAME
)


def view_full_day_coverage() -> Select:
    """
    Return Select statement for view to calculate the full day coverage of recommendation results.
    """
    day = interval_coverage.c.day.label("valid_date")

    # subtract one second from the covered time because 00:00:00 - 23:59:59 is considered to be a full day
    covered_dates = select(
        interval_coverage.c.run_id,
        interval_coverage.c.person_id,
        interval_coverage.c.cohort_category,
        interval_coverage.c.pi_pair_id,
        interval_coverage.c.criterion_id,
        day,
    ).filter(
        and_(
            interval_coverage.c.has_positive,
            not_(interval_coverage.c.has_negative),
        )
    )

    return covered_dates


def view_full_day_coverage_with_not_applicable() -> Select:
    """
    Return Select statement for view to calculate the full day coverage of recommendation results.
    In this view, NO_DATA is considered as positive.
    """
    day = interval_coverage.c.day.label("valid_date")

    # subtract one second from the covered time because 00:00:00 - 23:59:59 is considered to be a full day
    covered_dates = select(
        interval_coverage.c.run_id,
        interval_coverage.c.person_id,
        interval_coverage.c.cohort_category,
        interval_coverage.c.pi_pair_id,
        interval_coverage.c.criterion_id,
        day,
    ).filter(
        and_(
            or_(
                interval_coverage.c.has_positive, interval_coverage.c.has_not_applicable
            ),
            not_(interval_coverage.c.has_negative),
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
                    ResultInterval.interval_start.cast(Date),
                    ExecutionRun.observation_start_datetime.cast(Date),
                ),
                func.least(
                    ResultInterval.interval_end.cast(Date),
                    ExecutionRun.observation_end_datetime.cast(Date),
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
            ResultInterval.run_id,
            ResultInterval.pi_pair_id,
            ResultInterval.criterion_id,
            ResultInterval.cohort_category,
            ResultInterval.person_id,
            subquery.c.date_series.label("valid_date"),
        )
        .distinct()
        .where(
            ResultInterval.interval_type.in_(
                [
                    IntervalType.POSITIVE,
                    # IntervalType.NO_DATA # Jan-13: removed because NO_DATA should not count as positive anymore
                ]
            )
        )
    )

    return stmt


def interval_result_view() -> Select:
    """
    Return Select statement for view to calculate the interval results.
    """

    rri = ResultInterval.__table__.alias("rri")
    pip = PopulationInterventionPair.__table__.alias("pip")
    rc = Criterion.__table__.alias("rc")

    stmt = (
        select(
            rri.c.run_id,
            rri.c.person_id,
            rri.c.pi_pair_id,
            pip.c.pi_pair_name,
            rri.c.criterion_id,
            rc.c.criterion_description,
            rri.c.cohort_category,
            rri.c.interval_type,
            rri.c.interval_start,
            rri.c.interval_end,
        )
        .select_from(rri)
        .outerjoin(pip, (rri.c.pi_pair_id == pip.c.pi_pair_id))
        .outerjoin(rc, (rri.c.criterion_id == rc.c.criterion_id))
    )

    return stmt


full_day_coverage = view(
    "full_day_coverage", Base.metadata, view_full_day_coverage(), schema=SCHEMA_NAME
)

full_day_coverage_with_not_applicable = view(
    "full_day_coverage_with_na",
    Base.metadata,
    view_full_day_coverage_with_not_applicable(),
    schema=SCHEMA_NAME,
)

partial_day_coverage = view(
    "partial_day_coverage",
    Base.metadata,
    view_partial_day_coverage(),
    schema=SCHEMA_NAME,
)

interval_result = view(
    "interval_result", Base.metadata, interval_result_view(), schema=SCHEMA_NAME
)
