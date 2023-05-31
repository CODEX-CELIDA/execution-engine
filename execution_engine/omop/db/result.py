from datetime import date, datetime
from typing import Optional

from sqlalchemy import Enum, ForeignKey, Index, Integer, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from execution_engine.constants import CohortCategory
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
    recommendation_plan_name: Mapped[str]


class RecommendationCriterion(Base):  # noqa: D101
    __tablename__ = "recommendation_criterion"
    __table_args__ = {"schema": "celida"}

    criterion_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )
    cohort_definition_id: Mapped[int] = mapped_column(
        ForeignKey("celida.cohort_definition.cohort_definition_id"),
        index=True,
    )
    plan_id: Mapped[int] = mapped_column(
        ForeignKey("celida.recommendation_plan.plan_id"),
        index=True,
    )
    criterion_name: Mapped[str]


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
            "ix_run_id_criterion_name_valid_date",
            "recommendation_run_id",
            "criterion_name",
            "valid_date",
        ),
        {"schema": "celida"},
    )

    recommendation_results_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )
    recommendation_run_id = mapped_column(
        ForeignKey("celida.recommendation_run.recommendation_run_id"),
        index=True,
    )
    cohort_category = mapped_column(Enum(CohortCategory, schema="celida"))
    recommendation_plan_name: Mapped[Optional[str]]
    criterion_name: Mapped[Optional[str]]
    valid_date: Mapped[date]
    person_id = mapped_column(ForeignKey("cds_cdm.person.person_id"), index=True)

    recommendation_run: Mapped["RecommendationRun"] = relationship(
        primaryjoin="RecommendationResult.recommendation_run_id == RecommendationRun.recommendation_run_id",
    )

    # person = relationship(
    #    "cds_cdm.person",
    #    primaryjoin="RecommendationResult.person_id == cds_cdm.person.person_id",
    # )
