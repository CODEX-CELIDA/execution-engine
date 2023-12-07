from sqlalchemy import (
    CompoundSelect,
    Enum,
    Insert,
    Integer,
    Select,
    TypeDecorator,
    bindparam,
    cast,
    select,
)
from sqlalchemy.dialects.postgresql import ENUM

from execution_engine.constants import CohortCategory, IntervalType
from execution_engine.omop.db.celida import (
    RecommendationResult,
    RecommendationResultInterval,
)


def add_result_insert(
    query: Select | CompoundSelect,
    plan_id: int | None,
    criterion_id: int | None,
    cohort_category: CohortCategory,
) -> Insert:
    """
    Insert the result of the query into the result table.
    """
    if not isinstance(query, Select) and not isinstance(query, CompoundSelect):
        raise ValueError("sql must be a Select or CompoundSelect")

    query_select: Select

    # Always surround the original query by a select () query, as
    # otherwise problems arise when using CompoundSelect, multiple columns or DISTINCT person_id
    description = query.description
    query = query.alias("base_select")
    query_select = select(
        query.c.person_id,
        query.c.interval_start,
        query.c.interval_end,
        cast(
            query.c.interval_type, RecommendationResultInterval.interval_type.type
        ).label("interval_type"),
    ).select_from(query)

    query_select = query_select.add_columns(
        bindparam("run_id", type_=Integer()).label("recommendation_run_id"),
        bindparam("plan_id", plan_id).label("plan_id"),
        bindparam("cohort_category", cohort_category, type_=Enum(CohortCategory)).label(
            "cohort_category"
        ),
        bindparam("criterion_id", criterion_id).label("criterion_id"),
    )

    t_result = RecommendationResultInterval.__table__
    query_insert = t_result.insert().from_select(
        query_select.selected_columns, query_select
    )

    query_select.description = description
    query_insert.description = description

    return query_insert
