from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Connection,
    Enum,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Table,
    event,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from execution_engine.constants import CohortCategory, IntervalType
from execution_engine.omop.db.base import Base
from execution_engine.omop.db.celida import SCHEMA_NAME
from execution_engine.omop.db.celida.triggers import (
    create_trigger_interval_overlap_check_sql,
    trigger_interval_overlap_check_function_sql,
)

IntervalTypeEnum = Enum(IntervalType, name="interval_type", schema=SCHEMA_NAME)
CohortCategoryEnum = Enum(CohortCategory, name="cohort_category", schema=SCHEMA_NAME)


class Recommendation(Base):  # noqa: D101
    __tablename__ = "recommendation"
    __table_args__ = {"schema": SCHEMA_NAME}

    recommendation_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )
    recommendation_name: Mapped[str]
    recommendation_title: Mapped[str]
    recommendation_url: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True
    )
    recommendation_version: Mapped[str]
    recommendation_hash: Mapped[str] = mapped_column(
        String(64), index=True, unique=True
    )
    recommendation_json = mapped_column(LargeBinary)
    create_datetime: Mapped[datetime]


class RecommendationPlan(Base):  # noqa: D101
    __tablename__ = "recommendation_plan"
    __table_args__ = {"schema": SCHEMA_NAME}

    plan_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )
    recommendation_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.recommendation.recommendation_id"),
        index=True,
    )
    recommendation_plan_url: Mapped[str]
    recommendation_plan_name: Mapped[str]
    recommendation_plan_hash: Mapped[str] = mapped_column(
        String(64), index=True, unique=True
    )


class RecommendationCriterion(Base):  # noqa: D101
    __tablename__ = "recommendation_criterion"
    __table_args__ = {"schema": SCHEMA_NAME}

    criterion_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )
    # todo: add link to recommendation or 1:n to population/intervention pair?
    criterion_name: Mapped[str]
    criterion_description: Mapped[str]
    criterion_hash: Mapped[str] = mapped_column(String(64), index=True, unique=True)


class RecommendationRun(Base):  # noqa: D101
    __tablename__ = "recommendation_run"
    __table_args__ = {"schema": SCHEMA_NAME}

    recommendation_run_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )
    recommendation_id = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.recommendation.recommendation_id"),
        index=True,
    )
    observation_start_datetime: Mapped[datetime]
    observation_end_datetime: Mapped[datetime]
    run_datetime: Mapped[datetime]

    recommendation: Mapped["Recommendation"] = relationship(
        primaryjoin="RecommendationRun.recommendation_id == Recommendation.recommendation_id",
    )


class RecommendationResult(Base):  # noqa: D101
    __tablename__ = "recommendation_result"
    __table_args__ = (
        Index(
            "ix_rec_result_run_id_cohort_category_person_id_valid_date",
            "recommendation_run_id",
            "cohort_category",
            "person_id",
            "valid_date",
        ),
        Index(
            "ix_rec_result__run_id_plan_id_criterion_id_valid_date",
            "recommendation_run_id",
            "plan_id",
            "criterion_id",
            "valid_date",
        ),
        {"schema": SCHEMA_NAME},
    )

    recommendation_result_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True, autoincrement=True
    )
    recommendation_run_id = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.recommendation_run.recommendation_run_id"),
        index=True,
    )
    plan_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.recommendation_plan.plan_id"),
        index=True,
        nullable=True,
    )
    criterion_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.recommendation_criterion.criterion_id"),
        index=True,
        nullable=True,
    )
    cohort_category = mapped_column(CohortCategoryEnum)
    person_id: Mapped[int] = mapped_column(
        ForeignKey("cds_cdm.person.person_id"), index=True
    )
    valid_date: Mapped[date]

    recommendation_run: Mapped["RecommendationRun"] = relationship(
        primaryjoin="RecommendationResult.recommendation_run_id == RecommendationRun.recommendation_run_id",
    )

    recommendation_plan: Mapped["RecommendationPlan"] = relationship(
        primaryjoin="RecommendationResult.plan_id == RecommendationPlan.plan_id",
    )

    recommendation_criterion: Mapped["RecommendationCriterion"] = relationship(
        primaryjoin="RecommendationResult.criterion_id == RecommendationCriterion.criterion_id",
    )

    # person = relationship(
    #    "cds_cdm.person",
    #    primaryjoin="RecommendationResult.person_id == cds_cdm.person.person_id",
    # )


class RecommendationResultInterval(Base):  # noqa: D101
    __tablename__ = "recommendation_result_interval"
    __table_args__ = (
        Index(
            "ix_rec_result_int_run_id_cohort_category_person_id_valid_date",
            "recommendation_run_id",
            "cohort_category",
            "person_id",
            "interval_start",
            "interval_end",
        ),
        Index(
            "ix_rec_result_int_run_id_plan_id_criterion_id_valid_date",
            "recommendation_run_id",
            "plan_id",
            "criterion_id",
            "person_id",
            "interval_start",
            "interval_end",
        ),
        {"schema": SCHEMA_NAME},
    )

    recommendation_result_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True, autoincrement=True
    )
    recommendation_run_id = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.recommendation_run.recommendation_run_id"),
        index=True,
    )
    plan_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.recommendation_plan.plan_id"),
        index=True,
        nullable=True,
    )
    criterion_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.recommendation_criterion.criterion_id"),
        index=True,
        nullable=True,
    )
    cohort_category = mapped_column(CohortCategoryEnum)
    person_id: Mapped[int] = mapped_column(
        ForeignKey("cds_cdm.person.person_id"), index=True
    )
    interval_start: Mapped[datetime]
    interval_end: Mapped[datetime]
    interval_type = mapped_column(IntervalTypeEnum)

    recommendation_run: Mapped["RecommendationRun"] = relationship(
        primaryjoin="RecommendationResultInterval.recommendation_run_id == RecommendationRun.recommendation_run_id",
    )

    recommendation_plan: Mapped["RecommendationPlan"] = relationship(
        primaryjoin="RecommendationResultInterval.plan_id == RecommendationPlan.plan_id",
    )

    recommendation_criterion: Mapped["RecommendationCriterion"] = relationship(
        primaryjoin="RecommendationResultInterval.criterion_id == RecommendationCriterion.criterion_id",
    )


@event.listens_for(RecommendationResultInterval.__table__, "after_create")
def create_interval_overlap_check_triggers(
    target: Table, connection: Connection, **kw: Any
) -> None:
    """
    Create triggers for the recommendation_result_interval table.
    """
    connection.execute(
        text(
            trigger_interval_overlap_check_function_sql.format(
                schema=target.schema, table=target.name
            )
        )
    )
    connection.execute(
        text(
            create_trigger_interval_overlap_check_sql.format(
                schema=target.schema, table=target.name
            )
        )
    )


class Comment(Base):  # noqa: D101
    __tablename__ = "comment"
    __table_args__ = {"schema": SCHEMA_NAME}

    comment_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    recommendation_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.recommendation.recommendation_id"),
        index=True,
        nullable=True,
    )

    person_id: Mapped[int] = mapped_column(
        ForeignKey("cds_cdm.person.person_id"), index=True
    )

    text: Mapped[str]
    datetime: Mapped[datetime]
