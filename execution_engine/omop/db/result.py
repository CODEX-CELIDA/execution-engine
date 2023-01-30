from sqlalchemy import Column, Enum, ForeignKey, Integer, LargeBinary, Numeric, String
from sqlalchemy.orm import relationship

from execution_engine.constants import CohortCategory
from execution_engine.omop.db.base import Base, DateTime


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
        primaryjoin="RecommmendationRun.cohort_definition_id == CohortDefinition.cohort_definition_id",
    )


class RecommendationPersonDatum(Base):  # noqa: D101
    __tablename__ = "recommendation_person_data"
    __table_args__ = {"schema": "celida"}

    recommendation_person_data_id = Column(
        Integer,
        primary_key=True,
        index=True,
    )
    recommendation_run_id = Column(Integer, nullable=False, index=True)
    person_id = Column(
        ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    )
    cohort_category = Column(Enum(CohortCategory, schema="celida"), nullable=False)
    criterion_name = Column(String(255), nullable=False)
    domain_id = Column(String(20), nullable=False)
    parameter_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False, index=True
    )
    start_datetime = Column(DateTime, nullable=False)
    end_datetime = Column(DateTime)
    value_as_number = Column(Numeric)
    value_as_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"), index=True)
    unit_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"), index=True)
    drug_dose_as_number = Column(Numeric)
    drug_dose_unit_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), index=True
    )

    recommendation_run = relationship(
        "RecommendationRun",
        primaryjoin="RecommendationPersonDatum.recommendation_run_id == RecommendationRun.recommendation_run_id",
    )

    person = relationship(
        "cds_cdm.Person",
        primaryjoin="RecommendationPersonDatum.person_id == cds_cdm.Person.person_id",
    )

    parameter_concept = relationship(
        "cds_cdm.Concept",
        primaryjoin="RecommendationPersonDatum.parameter_concept_id == cds_cdm.Concept.concept_id",
    )

    value_as_concept = relationship(
        "cds_cdm.Concept",
        primaryjoin="RecommendationPersonDatum.value_as_concept_id == cds_cdm.Concept.concept_id",
    )

    unit_concept = relationship(
        "cds_cdm.Concept",
        primaryjoin="RecommendationPersonDatum.unit_concept_id == cds_cdm.Concept.concept_id",
    )

    drug_dose_unit_concept = relationship(
        "cds_cdm.Concept",
        primaryjoin="RecommendationPersonDatum.drug_dose_unit_concept_id == cds_cdm.Concept.concept_id",
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
    criterion_combination_name = Column(String(255))
    criterion_name = Column(String(255))
    person_id = Column(
        ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    )

    recommendation_run = relationship(
        "RecommendationRun",
        primaryjoin="RecommendationResult.recommendation_run_id == RecommendationRun.recommendation_run_id",
    )

    person = relationship(
        "cds_cdm.Person",
        primaryjoin="RecommendationResult.person_id == cds_cdm.Person.person_id",
    )
