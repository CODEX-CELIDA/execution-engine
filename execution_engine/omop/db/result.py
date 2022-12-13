from sqlalchemy import Column, Integer, LargeBinary, Numeric, String

from .base import Base, DateTime


class CohortDefinition(Base):  # noqa: D101
    __tablename__ = "cohort_definition"
    __table_args__ = {"schema": "celida"}

    cohort_definition_id = Column(
        Integer,
        primary_key=True,
        index=True,
    )
    recommendation_canonical_url = Column(String(255), nullable=False, index=True)
    recommendation_version = Column(String(255), nullable=False)
    cohort_definition_pickle = Column(LargeBinary, nullable=False)
    create_datetime = Column(DateTime, nullable=False)


class RecommendationRun(Base):  # noqa: D101
    __tablename__ = "recommendation_run"
    __table_args__ = {"schema": "celida"}

    recommendation_run_id = Column(
        Integer,
        primary_key=True,
        index=True,
    )
    cohort_definition_id = Column(Integer, nullable=False, index=True)
    observation_start_datetime = Column(DateTime, nullable=False)
    observation_end_datetime = Column(DateTime, nullable=False)
    run_datetime = Column(DateTime, nullable=False)


class RecommendationPersonDatum(Base):  # noqa: D101
    __tablename__ = "recommendation_person_data"
    __table_args__ = {"schema": "celida"}

    recommendation_person_data_id = Column(
        Integer,
        primary_key=True,
        index=True,
    )
    recommendation_run_id = Column(Integer, nullable=False, index=True)
    person_id = Column(Integer, nullable=False, index=True)
    cohort_type = Column(String(20), nullable=False)
    criterion_name = Column(String(255), nullable=False)
    domain_id = Column(String(20), nullable=False)
    parameter_concept_id = Column(Integer, nullable=False, index=True)
    value_as_number = Column(Numeric)
    value_as_concept_id = Column(Integer, index=True)
    unit_concept_id = Column(Integer, index=True)
    start_datetime = Column(DateTime, nullable=False)
    end_datetime = Column(DateTime)


class RecommendationResult(Base):  # noqa: D101
    __tablename__ = "recommendation_result"
    __table_args__ = {"schema": "celida"}

    recommendation_results_id = Column(
        Integer,
        primary_key=True,
        index=True,
    )
    recommendation_run_id = Column(Integer, nullable=False, index=True)
    cohort_type = Column(String(20), nullable=False)
    criterion_name = Column(String(255), nullable=False)
    person_id = Column(Integer, nullable=False, index=True)
