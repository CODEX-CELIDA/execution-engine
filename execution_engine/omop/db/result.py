from datetime import date, datetime

from sqlalchemy import Date, DateTime, Enum, ForeignKey, Integer, LargeBinary, String
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
    recommendation_name: Mapped[str] = mapped_column(String(255), nullable=False)
    recommendation_title: Mapped[str] = mapped_column(String(255), nullable=False)
    recommendation_url: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True
    )
    recommendation_version: Mapped[str] = mapped_column(String(255), nullable=False)
    cohort_definition_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True, unique=True
    )
    cohort_definition_json = mapped_column(LargeBinary, nullable=False)
    create_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False)


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
        nullable=False,
        index=True,
    )
    observation_start_datetime: Mapped[datetime] = mapped_column(
        DateTime, nullable=False
    )
    observation_end_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    run_datetime: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    cohort_definition = relationship(
        "CohortDefinition",
        primaryjoin="RecommendationRun.cohort_definition_id == CohortDefinition.cohort_definition_id",
    )


class RecommendationResult(Base):  # noqa: D101
    __tablename__ = "recommendation_result"
    __table_args__ = {"schema": "celida"}

    recommendation_results_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )
    recommendation_run_id = mapped_column(
        ForeignKey("celida.recommendation_run.recommendation_run_id"),
        nullable=False,
        index=True,
    )
    cohort_category = mapped_column(
        Enum(CohortCategory, schema="celida"), nullable=False
    )
    recommendation_plan_name: Mapped[str] = mapped_column(String(255), nullable=True)
    criterion_name: Mapped[str] = mapped_column(String(255), nullable=True)
    valid_date: Mapped[date] = mapped_column(Date())
    person_id = mapped_column(
        ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    )

    recommendation_run = relationship(
        "RecommendationRun",
        primaryjoin="RecommendationResult.recommendation_run_id == RecommendationRun.recommendation_run_id",
    )

    # person = relationship(
    #    "cds_cdm.person",
    #    primaryjoin="RecommendationResult.person_id == cds_cdm.person.person_id",
    # )
