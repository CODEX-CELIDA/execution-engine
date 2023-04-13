from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
)
from sqlalchemy.orm import relationship

from execution_engine.constants import CohortCategory
from execution_engine.omop.db.base import Base


class CohortDefinition(Base):  # noqa: D101
    __tablename__ = "cohort_definition"
    __table_args__ = {"schema": "celida"}

    cohort_definition_id = Column(
        Integer,
        primary_key=True,
        index=True,
    )
    recommendation_name = Column(String(255), nullable=False)
    recommendation_title = Column(String(255), nullable=False)
    recommendation_url = Column(String(255), nullable=False, index=True)
    recommendation_version = Column(String(255), nullable=False)
    cohort_definition_hash = Column(String(64), nullable=False, index=True, unique=True)
    cohort_definition_json = Column(LargeBinary, nullable=False)
    create_datetime = Column(DateTime, nullable=False)


class RecommendationRun(Base):  # noqa: D101
    __tablename__ = "recommendation_run"
    __table_args__ = {"schema": "celida"}

    recommendation_run_id = Column(
        Integer,
        primary_key=True,
        index=True,
    )
    cohort_definition_id = Column(
        ForeignKey("celida.cohort_definition.cohort_definition_id"),
        nullable=False,
        index=True,
    )
    observation_start_datetime = Column(DateTime, nullable=False)
    observation_end_datetime = Column(DateTime, nullable=False)
    run_datetime = Column(DateTime, nullable=False)

    cohort_definition = relationship(
        "CohortDefinition",
        primaryjoin="RecommendationRun.cohort_definition_id == CohortDefinition.cohort_definition_id",
    )


class RecommendationResult(Base):  # noqa: D101
    __tablename__ = "recommendation_result"
    __table_args__ = {"schema": "celida"}

    recommendation_results_id = Column(
        Integer,
        primary_key=True,
        index=True,
    )
    recommendation_run_id = Column(
        ForeignKey("celida.recommendation_run.recommendation_run_id"),
        nullable=False,
        index=True,
    )
    cohort_category = Column(Enum(CohortCategory, schema="celida"), nullable=False)
    recommendation_plan_name = Column(String(255))
    criterion_name = Column(String(255))
    valid_date = Column(Date())
    person_id = Column(
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
