from datetime import date, datetime

from sqlalchemy import Enum, ForeignKey, Index, Integer, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from execution_engine.constants import CohortCategory, IntervalType
from execution_engine.omop.db.base import Base


class CohortDefinition(Base):  # noqa: D101
    __tablename__ = "cohort_definition"
    __table_args__ = {"schema": "celida"}

    cohort_definition_id: Mapped[int] = mapped_column(
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
    cohort_definition_hash: Mapped[str] = mapped_column(
        String(64), index=True, unique=True
    )
    cohort_definition_json = mapped_column(LargeBinary)
    create_datetime: Mapped[datetime]


class RecommendationPlan(Base):  # noqa: D101
    __tablename__ = "recommendation_plan"
    __table_args__ = {"schema": "celida"}

    plan_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )
    cohort_definition_id: Mapped[int] = mapped_column(
        ForeignKey("celida.cohort_definition.cohort_definition_id"),
        index=True,
    )
    recommendation_plan_url: Mapped[str]
    recommendation_plan_name: Mapped[str]
    recommendation_plan_hash: Mapped[str] = mapped_column(
        String(64), index=True, unique=True
    )


class RecommendationCriterion(Base):  # noqa: D101
    __tablename__ = "recommendation_criterion"
    __table_args__ = {"schema": "celida"}

    criterion_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )
    # todo: add link to cohort definition or 1:n to recommendation plan?
    criterion_name: Mapped[str]
    criterion_description: Mapped[str]
    criterion_hash: Mapped[str] = mapped_column(String(64), index=True, unique=True)


class RecommendationRun(Base):  # noqa: D101
    __tablename__ = "recommendation_run"
    __table_args__ = {"schema": "celida"}

    recommendation_run_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )
    cohort_definition_id = mapped_column(
        ForeignKey("celida.cohort_definition.cohort_definition_id"),
        index=True,
    )
    observation_start_datetime: Mapped[datetime]
    observation_end_datetime: Mapped[datetime]
    run_datetime: Mapped[datetime]

    cohort_definition: Mapped["CohortDefinition"] = relationship(
        primaryjoin="RecommendationRun.cohort_definition_id == CohortDefinition.cohort_definition_id",
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
        {"schema": "celida"},
    )

    recommendation_result_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True, autoincrement=True
    )
    recommendation_run_id = mapped_column(
        ForeignKey("celida.recommendation_run.recommendation_run_id"),
        index=True,
    )
    plan_id: Mapped[int] = mapped_column(
        ForeignKey("celida.recommendation_plan.plan_id"), index=True, nullable=True
    )
    criterion_id: Mapped[int] = mapped_column(
        ForeignKey("celida.recommendation_criterion.criterion_id"),
        index=True,
        nullable=True,
    )
    cohort_category = mapped_column(Enum(CohortCategory, schema="celida"))
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
        {"schema": "celida"},
    )

    recommendation_result_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True, autoincrement=True
    )
    recommendation_run_id = mapped_column(
        ForeignKey("celida.recommendation_run.recommendation_run_id"),
        index=True,
    )
    plan_id: Mapped[int] = mapped_column(
        ForeignKey("celida.recommendation_plan.plan_id"), index=True, nullable=True
    )
    criterion_id: Mapped[int] = mapped_column(
        ForeignKey("celida.recommendation_criterion.criterion_id"),
        index=True,
        nullable=True,
    )
    cohort_category = mapped_column(Enum(CohortCategory, schema="celida"))
    person_id: Mapped[int] = mapped_column(
        ForeignKey("cds_cdm.person.person_id"), index=True
    )
    interval_start: Mapped[date]
    interval_end: Mapped[date]
    interval_type = mapped_column(Enum(IntervalType, schema="celida"))

    recommendation_run: Mapped["RecommendationRun"] = relationship(
        primaryjoin="RecommendationResult.recommendation_run_id == RecommendationRun.recommendation_run_id",
    )

    recommendation_plan: Mapped["RecommendationPlan"] = relationship(
        primaryjoin="RecommendationResult.plan_id == RecommendationPlan.plan_id",
    )

    recommendation_criterion: Mapped["RecommendationCriterion"] = relationship(
        primaryjoin="RecommendationResult.criterion_id == RecommendationCriterion.criterion_id",
    )


class Comment(Base):  # noqa: D101
    __tablename__ = "comment"
    __table_args__ = {"schema": "celida"}

    comment_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    cohort_definition_id: Mapped[int] = mapped_column(
        ForeignKey("celida.cohort_definition.cohort_definition_id"),
        index=True,
        nullable=True,
    )

    person_id: Mapped[int] = mapped_column(
        ForeignKey("cds_cdm.person.person_id"), index=True
    )

    text: Mapped[str]
    datetime: Mapped[datetime]
