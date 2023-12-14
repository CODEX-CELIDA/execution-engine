from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Table,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from execution_engine.omop.db.base import Base, metadata
from execution_engine.omop.db.omop import SCHEMA_NAME

t_cohort = Table(
    "cohort",
    metadata,
    Column("cohort_definition_id", Integer, nullable=False),
    Column("subject_id", Integer, nullable=False),
    Column("cohort_start_date", Date, nullable=False),
    Column("cohort_end_date", Date, nullable=False),
    schema=SCHEMA_NAME,
)


class Concept(Base):  # noqa: D101 # noqa: D101
    __tablename__ = "concept"
    __table_args__ = {"schema": SCHEMA_NAME}

    concept_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    concept_name: Mapped[str] = mapped_column(String(255))
    domain_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.domain.domain_id"), index=True
    )
    vocabulary_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.vocabulary.vocabulary_id"), index=True
    )
    concept_class_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept_class.concept_class_id"), index=True
    )
    standard_concept: Mapped[Optional[str]] = mapped_column(String(1))
    concept_code: Mapped[str] = mapped_column(String(50), index=True)
    valid_start_date: Mapped[date]
    valid_end_date: Mapped[date]
    invalid_reason: Mapped[Optional[str]] = mapped_column(String(1))

    concept_class: Mapped["ConceptClas"] = relationship(
        primaryjoin="Concept.concept_class_id == ConceptClas.concept_class_id",
    )
    domain: Mapped["Domain"] = relationship(
        primaryjoin="Concept.domain_id == Domain.domain_id"
    )
    vocabulary: Mapped["Vocabulary"] = relationship(
        primaryjoin="Concept.vocabulary_id == Vocabulary.vocabulary_id"
    )


class ConceptClas(Base):  # noqa: D101 # noqa: D101
    __tablename__ = "concept_class"
    __table_args__ = {"schema": SCHEMA_NAME}

    concept_class_id: Mapped[str] = mapped_column(
        String(20), primary_key=True, index=True
    )
    concept_class_name: Mapped[str] = mapped_column(String(255))
    concept_class_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )

    concept_class_concept: Mapped["Concept"] = relationship(
        primaryjoin="ConceptClas.concept_class_concept_id == Concept.concept_id",
    )


class Domain(Base):  # noqa: D101
    __tablename__ = "domain"
    __table_args__ = {"schema": SCHEMA_NAME}

    domain_id: Mapped[str] = mapped_column(String(20), primary_key=True, index=True)
    domain_name: Mapped[str] = mapped_column(String(255))
    domain_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )

    domain_concept: Mapped["Concept"] = relationship(
        primaryjoin="Domain.domain_concept_id == Concept.concept_id"
    )


class Vocabulary(Base):  # noqa: D101
    __tablename__ = "vocabulary"
    __table_args__ = {"schema": SCHEMA_NAME}

    vocabulary_id: Mapped[str] = mapped_column(String(20), primary_key=True, index=True)
    vocabulary_name: Mapped[str] = mapped_column(String(255))
    vocabulary_reference: Mapped[Optional[str]] = mapped_column(String(255))
    vocabulary_version: Mapped[Optional[str]] = mapped_column(String(255))
    vocabulary_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )

    vocabulary_concept: Mapped["Concept"] = relationship(
        primaryjoin="Vocabulary.vocabulary_concept_id == Concept.concept_id"
    )


t_cdm_source = Table(
    "cdm_source",
    metadata,
    Column("cdm_source_name", String(255), nullable=False),
    Column("cdm_source_abbreviation", String(25), nullable=False),
    Column("cdm_holder", String(255), nullable=False),
    Column("source_description", Text),
    Column("source_documentation_reference", String(255)),
    Column("cdm_etl_reference", String(255)),
    Column("source_release_date", Date, nullable=False),
    Column("cdm_release_date", Date, nullable=False),
    Column("cdm_version", String(10)),
    Column(
        "cdm_version_concept_id",
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"),
        nullable=False,
    ),
    Column("vocabulary_version", String(20), nullable=False),
    schema=SCHEMA_NAME,
)


t_cohort_definition = Table(
    "cohort_definition",
    metadata,
    Column("cohort_definition_id", Integer, nullable=False),
    Column("cohort_definition_name", String(255), nullable=False),
    Column("cohort_definition_description", Text),
    Column(
        "definition_type_concept_id",
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"),
        nullable=False,
    ),
    Column("cohort_definition_syntax", Text),
    Column(
        "subject_concept_id",
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"),
        nullable=False,
    ),
    Column("cohort_initiation_date", Date),
    schema=SCHEMA_NAME,
)


t_concept_ancestor = Table(
    "concept_ancestor",
    metadata,
    Column(
        "ancestor_concept_id",
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column(
        "descendant_concept_id",
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column("min_levels_of_separation", Integer, nullable=False),
    Column("max_levels_of_separation", Integer, nullable=False),
    schema=SCHEMA_NAME,
)


t_concept_synonym = Table(
    "concept_synonym",
    metadata,
    Column(
        "concept_id",
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column("concept_synonym_name", String(1000), nullable=False),
    Column(
        "language_concept_id",
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"),
        nullable=False,
    ),
    schema=SCHEMA_NAME,
)


class Cost(Base):  # noqa: D101
    __tablename__ = "cost"
    __table_args__ = {"schema": SCHEMA_NAME}

    cost_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    cost_event_id: Mapped[int] = mapped_column(Integer, index=True)
    cost_domain_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.domain.domain_id")
    )
    cost_type_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    currency_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    total_charge = mapped_column(Numeric, nullable=True)
    total_cost = mapped_column(Numeric, nullable=True)
    total_paid = mapped_column(Numeric, nullable=True)
    paid_by_payer = mapped_column(Numeric, nullable=True)
    paid_by_patient = mapped_column(Numeric, nullable=True)
    paid_patient_copay = mapped_column(Numeric, nullable=True)
    paid_patient_coinsurance = mapped_column(Numeric, nullable=True)
    paid_patient_deductible = mapped_column(Numeric, nullable=True)
    paid_by_primary = mapped_column(Numeric, nullable=True)
    paid_ingredient_cost = mapped_column(Numeric, nullable=True)
    paid_dispensing_fee = mapped_column(Numeric, nullable=True)
    payer_plan_period_id: Mapped[Optional[int]]
    amount_allowed = mapped_column(Numeric, nullable=True)
    revenue_code_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    revenue_code_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    drg_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    drg_source_value: Mapped[Optional[str]] = mapped_column(String(3))

    cost_domain: Mapped["Domain"] = relationship()
    cost_type_concept: Mapped["Concept"] = relationship(
        primaryjoin="Cost.cost_type_concept_id == Concept.concept_id"
    )
    currency_concept: Mapped["Concept"] = relationship(
        primaryjoin="Cost.currency_concept_id == Concept.concept_id"
    )
    drg_concept: Mapped["Concept"] = relationship(
        primaryjoin="Cost.drg_concept_id == Concept.concept_id"
    )
    revenue_code_concept: Mapped["Concept"] = relationship(
        primaryjoin="Cost.revenue_code_concept_id == Concept.concept_id"
    )


t_drug_strength = Table(
    "drug_strength",
    metadata,
    Column(
        "drug_concept_id",
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column(
        "ingredient_concept_id",
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column("amount_value", Numeric),
    Column("amount_unit_concept_id", ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")),
    Column("numerator_value", Numeric),
    Column(
        "numerator_unit_concept_id", ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    ),
    Column("denominator_value", Numeric),
    Column(
        "denominator_unit_concept_id", ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    ),
    Column("box_size", Integer),
    Column("valid_start_date", Date, nullable=False),
    Column("valid_end_date", Date, nullable=False),
    Column("invalid_reason", String(1)),
    schema=SCHEMA_NAME,
)


t_fact_relationship = Table(
    "fact_relationship",
    metadata,
    Column(
        "domain_concept_id_1",
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column("fact_id_1", Integer, nullable=False),
    Column(
        "domain_concept_id_2",
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column("fact_id_2", Integer, nullable=False),
    Column(
        "relationship_concept_id",
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    schema=SCHEMA_NAME,
)


class Location(Base):  # noqa: D101
    __tablename__ = "location"
    __table_args__ = {"schema": SCHEMA_NAME}

    location_id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )
    address_1: Mapped[Optional[str]] = mapped_column(String(50))
    address_2: Mapped[Optional[str]] = mapped_column(String(50))
    city: Mapped[Optional[str]] = mapped_column(String(50))
    state: Mapped[Optional[str]] = mapped_column(String(2))
    zip: Mapped[Optional[str]] = mapped_column(String(9))
    county: Mapped[Optional[str]] = mapped_column(String(20))
    location_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    country_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    country_source_value: Mapped[Optional[str]] = mapped_column(String(80))
    latitude = mapped_column(Numeric, nullable=True)
    longitude = mapped_column(Numeric, nullable=True)

    country_concept: Mapped["Concept"] = relationship()


class Metadatum(Base):  # noqa: D101
    __tablename__ = "metadata"
    __table_args__ = {"schema": SCHEMA_NAME}

    metadata_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    metadata_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"), index=True
    )
    metadata_type_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    name: Mapped[str] = mapped_column(String(250))
    value_as_string: Mapped[Optional[str]] = mapped_column(String(250))
    value_as_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    value_as_number = mapped_column(Numeric, nullable=True)
    metadata_date: Mapped[Optional[date]]
    metadata_datetime: Mapped[Optional[datetime]]

    metadata_concept: Mapped["Concept"] = relationship(
        primaryjoin="Metadatum.metadata_concept_id == Concept.concept_id"
    )
    metadata_type_concept: Mapped["Concept"] = relationship(
        primaryjoin="Metadatum.metadata_type_concept_id == Concept.concept_id",
    )
    value_as_concept: Mapped["Concept"] = relationship(
        primaryjoin="Metadatum.value_as_concept_id == Concept.concept_id"
    )


class NoteNlp(Base):  # noqa: D101
    __tablename__ = "note_nlp"
    __table_args__ = {"schema": SCHEMA_NAME}

    note_nlp_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    note_id: Mapped[int] = mapped_column(Integer, index=True)
    section_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    snippet: Mapped[Optional[str]] = mapped_column(String(250))
    offset: Mapped[Optional[str]] = mapped_column(String(50))
    lexical_variant: Mapped[str] = mapped_column(String(250))
    note_nlp_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"), index=True
    )
    note_nlp_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    nlp_system: Mapped[Optional[str]] = mapped_column(String(250))
    nlp_date: Mapped[date]
    nlp_datetime: Mapped[Optional[datetime]]
    term_exists: Mapped[Optional[str]] = mapped_column(String(1))
    term_temporal: Mapped[Optional[str]] = mapped_column(String(50))
    term_modifiers: Mapped[Optional[str]] = mapped_column(String(2000))

    note_nlp_concept: Mapped["Concept"] = relationship(
        primaryjoin="NoteNlp.note_nlp_concept_id == Concept.concept_id"
    )
    note_nlp_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="NoteNlp.note_nlp_source_concept_id == Concept.concept_id",
    )
    section_concept: Mapped["Concept"] = relationship(
        primaryjoin="NoteNlp.section_concept_id == Concept.concept_id"
    )


class Relationship(Base):  # noqa: D101
    __tablename__ = "relationship"
    __table_args__ = {"schema": SCHEMA_NAME}

    relationship_id: Mapped[str] = mapped_column(
        String(20), primary_key=True, index=True
    )
    relationship_name: Mapped[str] = mapped_column(String(255))
    is_hierarchical: Mapped[str] = mapped_column(String(1))
    defines_ancestry: Mapped[str] = mapped_column(String(1))
    reverse_relationship_id: Mapped[str] = mapped_column(String(20))
    relationship_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )

    relationship_concept: Mapped["Concept"] = relationship()


t_source_to_concept_map = Table(
    "source_to_concept_map",
    metadata,
    Column("source_code", String(50), nullable=False, index=True),
    Column(
        "source_concept_id",
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"),
        nullable=False,
    ),
    Column("source_vocabulary_id", String(20), nullable=False, index=True),
    Column("source_code_description", String(255)),
    Column(
        "target_concept_id",
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column(
        "target_vocabulary_id",
        ForeignKey(f"{SCHEMA_NAME}.vocabulary.vocabulary_id"),
        nullable=False,
        index=True,
    ),
    Column("valid_start_date", Date, nullable=False),
    Column("valid_end_date", Date, nullable=False),
    Column("invalid_reason", String(1)),
    schema=SCHEMA_NAME,
)


class CareSite(Base):  # noqa: D101
    __tablename__ = "care_site"
    __table_args__ = {"schema": SCHEMA_NAME}

    care_site_id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )
    care_site_name: Mapped[Optional[str]] = mapped_column(String(255))
    place_of_service_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    location_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.location.location_id")
    )
    care_site_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    place_of_service_source_value: Mapped[Optional[str]] = mapped_column(String(50))

    location: Mapped["Location"] = relationship()
    place_of_service_concept: Mapped["Concept"] = relationship()


t_concept_relationship = Table(
    "concept_relationship",
    metadata,
    Column(
        "concept_id_1",
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column(
        "concept_id_2",
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column(
        "relationship_id",
        ForeignKey(f"{SCHEMA_NAME}.relationship.relationship_id"),
        nullable=False,
        index=True,
    ),
    Column("valid_start_date", Date, nullable=False),
    Column("valid_end_date", Date, nullable=False),
    Column("invalid_reason", String(1)),
    schema=SCHEMA_NAME,
)


class Provider(Base):  # noqa: D101
    __tablename__ = "provider"
    __table_args__ = {"schema": SCHEMA_NAME}

    provider_id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )
    provider_name: Mapped[Optional[str]] = mapped_column(String(255))
    npi: Mapped[Optional[str]] = mapped_column(String(20))
    dea: Mapped[Optional[str]] = mapped_column(String(20))
    specialty_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    care_site_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.care_site.care_site_id")
    )
    year_of_birth: Mapped[Optional[int]]
    gender_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    provider_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    specialty_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    specialty_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    gender_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    gender_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )

    care_site: Mapped["CareSite"] = relationship()
    gender_concept: Mapped["Concept"] = relationship(
        primaryjoin="Provider.gender_concept_id == Concept.concept_id"
    )
    gender_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="Provider.gender_source_concept_id == Concept.concept_id"
    )
    specialty_concept: Mapped["Concept"] = relationship(
        primaryjoin="Provider.specialty_concept_id == Concept.concept_id"
    )
    specialty_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="Provider.specialty_source_concept_id == Concept.concept_id",
    )


class Person(Base):  # noqa: D101
    __tablename__ = "person"
    __table_args__ = {"schema": SCHEMA_NAME}

    person_id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )
    gender_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"), index=True
    )
    year_of_birth: Mapped[int]
    month_of_birth: Mapped[Optional[int]]
    day_of_birth: Mapped[Optional[int]]
    birth_datetime: Mapped[Optional[datetime]]
    race_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    ethnicity_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    location_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.location.location_id")
    )
    provider_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.provider.provider_id")
    )
    care_site_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.care_site.care_site_id")
    )
    person_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    gender_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    gender_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    race_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    race_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    ethnicity_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    ethnicity_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )

    care_site: Mapped["CareSite"] = relationship()
    ethnicity_concept: Mapped["Concept"] = relationship(
        primaryjoin="Person.ethnicity_concept_id == Concept.concept_id"
    )
    ethnicity_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="Person.ethnicity_source_concept_id == Concept.concept_id",
    )
    gender_concept: Mapped["Concept"] = relationship(
        primaryjoin="Person.gender_concept_id == Concept.concept_id"
    )
    gender_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="Person.gender_source_concept_id == Concept.concept_id"
    )
    location: Mapped["Location"] = relationship()
    provider: Mapped["Provider"] = relationship()
    race_concept: Mapped["Concept"] = relationship(
        primaryjoin="Person.race_concept_id == Concept.concept_id"
    )
    race_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="Person.race_source_concept_id == Concept.concept_id"
    )


class ConditionEra(Base):  # noqa: D101
    __tablename__ = "condition_era"
    __table_args__ = {"schema": SCHEMA_NAME}

    condition_era_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.person.person_id"), index=True
    )
    condition_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"), index=True
    )
    condition_era_start_date: Mapped[date]
    condition_era_end_date: Mapped[date]
    condition_occurrence_count: Mapped[Optional[int]]

    condition_concept: Mapped["Concept"] = relationship()
    person: Mapped["Person"] = relationship()


t_death = Table(
    "death",
    metadata,
    Column(
        "person_id",
        ForeignKey(f"{SCHEMA_NAME}.person.person_id"),
        nullable=False,
        index=True,
    ),
    Column("death_date", Date, nullable=False),
    Column("death_datetime", DateTime),
    Column("death_type_concept_id", ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")),
    Column("cause_concept_id", ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")),
    Column("cause_source_value", String(50)),
    Column("cause_source_concept_id", ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")),
    schema=SCHEMA_NAME,
)


class DoseEra(Base):  # noqa: D101
    __tablename__ = "dose_era"
    __table_args__ = {"schema": SCHEMA_NAME}

    dose_era_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.person.person_id"), index=True
    )
    drug_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"), index=True
    )
    unit_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    dose_value = Column(Numeric, nullable=False)
    dose_era_start_date: Mapped[date]
    dose_era_end_date: Mapped[date]

    drug_concept: Mapped["Concept"] = relationship(
        primaryjoin="DoseEra.drug_concept_id == Concept.concept_id"
    )
    person: Mapped["Person"] = relationship()
    unit_concept: Mapped["Concept"] = relationship(
        primaryjoin="DoseEra.unit_concept_id == Concept.concept_id"
    )


class DrugEra(Base):  # noqa: D101
    __tablename__ = "drug_era"
    __table_args__ = {"schema": SCHEMA_NAME}

    drug_era_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.person.person_id"), index=True
    )
    drug_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"), index=True
    )
    drug_era_start_date: Mapped[date]
    drug_era_end_date: Mapped[date]
    drug_exposure_count: Mapped[Optional[int]]
    gap_days: Mapped[Optional[int]]

    drug_concept: Mapped["Concept"] = relationship()
    person: Mapped["Person"] = relationship()


class Episode(Base):  # noqa: D101
    __tablename__ = "episode"
    __table_args__ = {"schema": SCHEMA_NAME}

    episode_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.person.person_id")
    )
    episode_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    episode_start_date: Mapped[date]
    episode_start_datetime: Mapped[Optional[datetime]]
    episode_end_date: Mapped[Optional[date]]
    episode_end_datetime: Mapped[Optional[datetime]]
    episode_parent_id: Mapped[Optional[int]]
    episode_number: Mapped[Optional[int]]
    episode_object_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    episode_type_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    episode_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    episode_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )

    episode_concept: Mapped["Concept"] = relationship(
        primaryjoin="Episode.episode_concept_id == Concept.concept_id"
    )
    episode_object_concept: Mapped["Concept"] = relationship(
        primaryjoin="Episode.episode_object_concept_id == Concept.concept_id"
    )
    episode_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="Episode.episode_source_concept_id == Concept.concept_id"
    )
    episode_type_concept: Mapped["Concept"] = relationship(
        primaryjoin="Episode.episode_type_concept_id == Concept.concept_id"
    )
    person: Mapped["Person"] = relationship()


class ObservationPeriod(Base):  # noqa: D101
    __tablename__ = "observation_period"
    __table_args__ = {"schema": SCHEMA_NAME}

    observation_period_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.person.person_id"), index=True
    )
    observation_period_start_date: Mapped[date]
    observation_period_end_date: Mapped[date]
    period_type_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )

    period_type_concept: Mapped["Concept"] = relationship()
    person: Mapped["Person"] = relationship()


class PayerPlanPeriod(Base):  # noqa: D101
    __tablename__ = "payer_plan_period"
    __table_args__ = {"schema": SCHEMA_NAME}

    payer_plan_period_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.person.person_id"), index=True
    )
    payer_plan_period_start_date: Mapped[date]
    payer_plan_period_end_date: Mapped[date]
    payer_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    payer_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    payer_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    plan_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    plan_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    plan_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    sponsor_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    sponsor_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    sponsor_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    family_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    stop_reason_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    stop_reason_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    stop_reason_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )

    payer_concept: Mapped["Concept"] = relationship(
        primaryjoin="PayerPlanPeriod.payer_concept_id == Concept.concept_id"
    )
    payer_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="PayerPlanPeriod.payer_source_concept_id == Concept.concept_id",
    )
    person: Mapped["Person"] = relationship()
    plan_concept: Mapped["Concept"] = relationship(
        primaryjoin="PayerPlanPeriod.plan_concept_id == Concept.concept_id"
    )
    plan_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="PayerPlanPeriod.plan_source_concept_id == Concept.concept_id",
    )
    sponsor_concept: Mapped["Concept"] = relationship(
        primaryjoin="PayerPlanPeriod.sponsor_concept_id == Concept.concept_id",
    )
    sponsor_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="PayerPlanPeriod.sponsor_source_concept_id == Concept.concept_id",
    )
    stop_reason_concept: Mapped["Concept"] = relationship(
        primaryjoin="PayerPlanPeriod.stop_reason_concept_id == Concept.concept_id",
    )
    stop_reason_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="PayerPlanPeriod.stop_reason_source_concept_id == Concept.concept_id",
    )


class Speciman(Base):  # noqa: D101
    __tablename__ = "specimen"
    __table_args__ = {"schema": SCHEMA_NAME}

    specimen_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.person.person_id"), index=True
    )
    specimen_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"), index=True
    )
    specimen_type_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    specimen_date: Mapped[date]
    specimen_datetime: Mapped[Optional[datetime]]
    quantity = mapped_column(Numeric, nullable=True)
    unit_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    anatomic_site_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    disease_status_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    specimen_source_id: Mapped[Optional[str]] = mapped_column(String(50))
    specimen_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    unit_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    anatomic_site_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    disease_status_source_value: Mapped[Optional[str]] = mapped_column(String(50))

    anatomic_site_concept: Mapped["Concept"] = relationship(
        primaryjoin="Speciman.anatomic_site_concept_id == Concept.concept_id"
    )
    disease_status_concept: Mapped["Concept"] = relationship(
        primaryjoin="Speciman.disease_status_concept_id == Concept.concept_id",
    )
    person: Mapped["Person"] = relationship()
    specimen_concept: Mapped["Concept"] = relationship(
        primaryjoin="Speciman.specimen_concept_id == Concept.concept_id"
    )
    specimen_type_concept: Mapped["Concept"] = relationship(
        primaryjoin="Speciman.specimen_type_concept_id == Concept.concept_id"
    )
    unit_concept: Mapped["Concept"] = relationship(
        primaryjoin="Speciman.unit_concept_id == Concept.concept_id"
    )


class VisitOccurrence(Base):  # noqa: D101
    __tablename__ = "visit_occurrence"
    __table_args__ = {"schema": SCHEMA_NAME}

    visit_occurrence_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.person.person_id"), index=True
    )
    visit_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"), index=True
    )
    visit_start_date: Mapped[date]
    visit_start_datetime: Mapped[Optional[datetime]]
    visit_end_date: Mapped[date]
    visit_end_datetime: Mapped[Optional[datetime]]
    visit_type_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    provider_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.provider.provider_id")
    )
    care_site_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.care_site.care_site_id")
    )
    visit_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    visit_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    admitted_from_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    admitted_from_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    discharged_to_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    discharged_to_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    preceding_visit_occurrence_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.visit_occurrence.visit_occurrence_id")
    )

    admitted_from_concept: Mapped["Concept"] = relationship(
        primaryjoin="VisitOccurrence.admitted_from_concept_id == Concept.concept_id",
    )
    care_site: Mapped["CareSite"] = relationship()
    discharged_to_concept: Mapped["Concept"] = relationship(
        primaryjoin="VisitOccurrence.discharged_to_concept_id == Concept.concept_id",
    )
    person: Mapped["Person"] = relationship()
    preceding_visit_occurrence: Mapped["VisitOccurrence"] = relationship(
        remote_side=[visit_occurrence_id]
    )
    provider: Mapped["Provider"] = relationship()
    visit_concept: Mapped["Concept"] = relationship(
        primaryjoin="VisitOccurrence.visit_concept_id == Concept.concept_id"
    )
    visit_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="VisitOccurrence.visit_source_concept_id == Concept.concept_id",
    )
    visit_type_concept: Mapped["Concept"] = relationship(
        primaryjoin="VisitOccurrence.visit_type_concept_id == Concept.concept_id",
    )


t_episode_event = Table(
    "episode_event",
    metadata,
    Column(
        "episode_id", ForeignKey(f"{SCHEMA_NAME}.episode.episode_id"), nullable=False
    ),
    Column("event_id", Integer, nullable=False),
    Column(
        "episode_event_field_concept_id",
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"),
        nullable=False,
    ),
    schema=SCHEMA_NAME,
)


class VisitDetail(Base):  # noqa: D101
    __tablename__ = "visit_detail"
    __table_args__ = {"schema": SCHEMA_NAME}

    visit_detail_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.person.person_id"), index=True
    )
    visit_detail_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"), index=True
    )
    visit_detail_start_date: Mapped[date]
    visit_detail_start_datetime: Mapped[Optional[datetime]]
    visit_detail_end_date: Mapped[date]
    visit_detail_end_datetime: Mapped[Optional[datetime]]
    visit_detail_type_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    provider_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.provider.provider_id")
    )
    care_site_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.care_site.care_site_id")
    )
    visit_detail_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    visit_detail_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    admitted_from_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    admitted_from_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    discharged_to_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    discharged_to_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    preceding_visit_detail_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.visit_detail.visit_detail_id")
    )
    parent_visit_detail_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.visit_detail.visit_detail_id")
    )
    visit_occurrence_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.visit_occurrence.visit_occurrence_id"),
        index=True,
    )

    admitted_from_concept: Mapped["Concept"] = relationship(
        primaryjoin="VisitDetail.admitted_from_concept_id == Concept.concept_id",
    )
    care_site: Mapped["CareSite"] = relationship()
    discharged_to_concept: Mapped["Concept"] = relationship(
        primaryjoin="VisitDetail.discharged_to_concept_id == Concept.concept_id",
    )
    parent_visit_detail: Mapped["VisitDetail"] = relationship(
        remote_side=[visit_detail_id],
        primaryjoin="VisitDetail.parent_visit_detail_id == VisitDetail.visit_detail_id",
    )
    person: Mapped["Person"] = relationship()
    preceding_visit_detail: Mapped["VisitDetail"] = relationship(
        remote_side=[visit_detail_id],
        primaryjoin="VisitDetail.preceding_visit_detail_id == VisitDetail.visit_detail_id",
    )
    provider: Mapped["Provider"] = relationship()
    visit_detail_concept: Mapped["Concept"] = relationship(
        primaryjoin="VisitDetail.visit_detail_concept_id == Concept.concept_id",
    )
    visit_detail_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="VisitDetail.visit_detail_source_concept_id == Concept.concept_id",
    )
    visit_detail_type_concept: Mapped["Concept"] = relationship(
        primaryjoin="VisitDetail.visit_detail_type_concept_id == Concept.concept_id",
    )
    visit_occurrence: Mapped["VisitOccurrence"] = relationship()


class ConditionOccurrence(Base):  # noqa: D101
    __tablename__ = "condition_occurrence"
    __table_args__ = {"schema": SCHEMA_NAME}

    condition_occurrence_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.person.person_id"), index=True
    )
    condition_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"), index=True
    )
    condition_start_date: Mapped[date]
    condition_start_datetime: Mapped[Optional[datetime]]
    condition_end_date: Mapped[Optional[date]]
    condition_end_datetime: Mapped[Optional[datetime]]
    condition_type_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    condition_status_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    stop_reason: Mapped[Optional[str]] = mapped_column(String(20))
    provider_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.provider.provider_id")
    )
    visit_occurrence_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.visit_detail.visit_detail_id")
    )
    condition_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    condition_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    condition_status_source_value: Mapped[Optional[str]] = mapped_column(String(50))

    condition_concept: Mapped["Concept"] = relationship(
        primaryjoin="ConditionOccurrence.condition_concept_id == Concept.concept_id",
    )
    condition_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="ConditionOccurrence.condition_source_concept_id == Concept.concept_id",
    )
    condition_status_concept: Mapped["Concept"] = relationship(
        primaryjoin="ConditionOccurrence.condition_status_concept_id == Concept.concept_id",
    )
    condition_type_concept: Mapped["Concept"] = relationship(
        primaryjoin="ConditionOccurrence.condition_type_concept_id == Concept.concept_id",
    )
    person: Mapped["Person"] = relationship()
    provider: Mapped["Provider"] = relationship()
    visit_detail: Mapped["VisitDetail"] = relationship()
    visit_occurrence: Mapped["VisitOccurrence"] = relationship()


class DeviceExposure(Base):  # noqa: D101
    __tablename__ = "device_exposure"
    __table_args__ = {"schema": SCHEMA_NAME}

    device_exposure_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.person.person_id"), index=True
    )
    device_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"), index=True
    )
    device_exposure_start_date: Mapped[date]
    device_exposure_start_datetime: Mapped[Optional[datetime]]
    device_exposure_end_date: Mapped[Optional[date]]
    device_exposure_end_datetime: Mapped[Optional[datetime]]
    device_type_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    unique_device_id: Mapped[Optional[str]] = mapped_column(String(255))
    production_id: Mapped[Optional[str]] = mapped_column(String(255))
    quantity: Mapped[int]
    provider_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.provider.provider_id")
    )
    visit_occurrence_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.visit_detail.visit_detail_id")
    )
    device_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    device_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    unit_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    unit_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    unit_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )

    device_concept: Mapped["Concept"] = relationship(
        primaryjoin="DeviceExposure.device_concept_id == Concept.concept_id"
    )
    device_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="DeviceExposure.device_source_concept_id == Concept.concept_id",
    )
    device_type_concept: Mapped["Concept"] = relationship(
        primaryjoin="DeviceExposure.device_type_concept_id == Concept.concept_id",
    )
    person: Mapped["Person"] = relationship()
    provider: Mapped["Provider"] = relationship()
    unit_concept: Mapped["Concept"] = relationship(
        primaryjoin="DeviceExposure.unit_concept_id == Concept.concept_id"
    )
    unit_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="DeviceExposure.unit_source_concept_id == Concept.concept_id",
    )
    visit_detail: Mapped["VisitDetail"] = relationship()
    visit_occurrence: Mapped["VisitOccurrence"] = relationship()


class DrugExposure(Base):  # noqa: D101
    __tablename__ = "drug_exposure"
    __table_args__ = {"schema": SCHEMA_NAME}

    drug_exposure_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.person.person_id"), index=True
    )
    drug_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"), index=True
    )
    drug_exposure_start_date: Mapped[date]
    drug_exposure_start_datetime: Mapped[Optional[datetime]]
    drug_exposure_end_date: Mapped[date]
    drug_exposure_end_datetime: Mapped[Optional[datetime]]
    verbatim_end_date: Mapped[Optional[date]]
    drug_type_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    stop_reason: Mapped[Optional[str]] = mapped_column(String(20))
    refills: Mapped[Optional[int]]
    quantity = mapped_column(Numeric, nullable=True)
    days_supply: Mapped[Optional[int]]
    sig = Column(Text)  # ToDo: map to mapped_column
    route_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    lot_number: Mapped[Optional[str]] = mapped_column(String(50))
    provider_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.provider.provider_id")
    )
    visit_occurrence_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.visit_detail.visit_detail_id")
    )
    drug_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    drug_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    route_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    dose_unit_source_value: Mapped[Optional[str]] = mapped_column(String(50))

    drug_concept: Mapped["Concept"] = relationship(
        primaryjoin="DrugExposure.drug_concept_id == Concept.concept_id"
    )
    drug_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="DrugExposure.drug_source_concept_id == Concept.concept_id",
    )
    drug_type_concept: Mapped["Concept"] = relationship(
        primaryjoin="DrugExposure.drug_type_concept_id == Concept.concept_id"
    )
    person: Mapped["Person"] = relationship()
    provider: Mapped["Provider"] = relationship()
    route_concept: Mapped["Concept"] = relationship(
        primaryjoin="DrugExposure.route_concept_id == Concept.concept_id"
    )
    visit_detail: Mapped["VisitDetail"] = relationship()
    visit_occurrence: Mapped["VisitOccurrence"] = relationship()


class Measurement(Base):  # noqa: D101
    __tablename__ = "measurement"
    __table_args__ = {"schema": SCHEMA_NAME}

    measurement_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.person.person_id"), index=True
    )
    measurement_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"), index=True
    )
    measurement_date: Mapped[date]
    measurement_datetime: Mapped[Optional[datetime]]
    measurement_time: Mapped[Optional[str]] = mapped_column(String(10))
    measurement_type_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    operator_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    value_as_number = mapped_column(Numeric, nullable=True)
    value_as_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    unit_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    range_low = mapped_column(Numeric, nullable=True)
    range_high = mapped_column(Numeric, nullable=True)
    provider_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.provider.provider_id")
    )
    visit_occurrence_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.visit_detail.visit_detail_id")
    )
    measurement_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    measurement_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    unit_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    unit_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    value_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    measurement_event_id: Mapped[Optional[int]]
    meas_event_field_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )

    meas_event_field_concept: Mapped["Concept"] = relationship(
        primaryjoin="Measurement.meas_event_field_concept_id == Concept.concept_id",
    )
    measurement_concept: Mapped["Concept"] = relationship(
        primaryjoin="Measurement.measurement_concept_id == Concept.concept_id",
    )
    measurement_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="Measurement.measurement_source_concept_id == Concept.concept_id",
    )
    measurement_type_concept: Mapped["Concept"] = relationship(
        primaryjoin="Measurement.measurement_type_concept_id == Concept.concept_id",
    )
    operator_concept: Mapped["Concept"] = relationship(
        primaryjoin="Measurement.operator_concept_id == Concept.concept_id"
    )
    person: Mapped["Person"] = relationship()
    provider: Mapped["Provider"] = relationship()
    unit_concept: Mapped["Concept"] = relationship(
        primaryjoin="Measurement.unit_concept_id == Concept.concept_id"
    )
    unit_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="Measurement.unit_source_concept_id == Concept.concept_id",
    )
    value_as_concept: Mapped["Concept"] = relationship(
        primaryjoin="Measurement.value_as_concept_id == Concept.concept_id"
    )
    visit_detail: Mapped["VisitDetail"] = relationship()
    visit_occurrence: Mapped["VisitOccurrence"] = relationship()


class Note(Base):  # noqa: D101
    __tablename__ = "note"
    __table_args__ = {"schema": SCHEMA_NAME}

    note_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.person.person_id"), index=True
    )
    note_date: Mapped[date]
    note_datetime: Mapped[Optional[datetime]]
    note_type_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"), index=True
    )
    note_class_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    note_title: Mapped[Optional[str]] = mapped_column(String(250))
    note_text = Column(Text)  # todo: map to mapped_column
    encoding_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    language_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    provider_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.provider.provider_id")
    )
    visit_occurrence_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.visit_detail.visit_detail_id")
    )
    note_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    note_event_id: Mapped[Optional[int]]
    note_event_field_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )

    encoding_concept: Mapped["Concept"] = relationship(
        primaryjoin="Note.encoding_concept_id == Concept.concept_id"
    )
    language_concept: Mapped["Concept"] = relationship(
        primaryjoin="Note.language_concept_id == Concept.concept_id"
    )
    note_class_concept: Mapped["Concept"] = relationship(
        primaryjoin="Note.note_class_concept_id == Concept.concept_id"
    )
    note_event_field_concept: Mapped["Concept"] = relationship(
        primaryjoin="Note.note_event_field_concept_id == Concept.concept_id"
    )
    note_type_concept: Mapped["Concept"] = relationship(
        primaryjoin="Note.note_type_concept_id == Concept.concept_id"
    )
    person: Mapped["Person"] = relationship()
    provider: Mapped["Provider"] = relationship()
    visit_detail: Mapped["VisitDetail"] = relationship()
    visit_occurrence: Mapped["VisitOccurrence"] = relationship()


class Observation(Base):  # noqa: D101
    __tablename__ = "observation"
    __table_args__ = {"schema": SCHEMA_NAME}

    observation_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.person.person_id"), index=True
    )
    observation_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"), index=True
    )
    observation_date: Mapped[date]
    observation_datetime: Mapped[Optional[datetime]]
    observation_type_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    value_as_number = mapped_column(Numeric, nullable=True)
    value_as_string: Mapped[Optional[str]] = mapped_column(String(60))
    value_as_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    qualifier_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    unit_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    provider_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.provider.provider_id")
    )
    visit_occurrence_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.visit_detail.visit_detail_id")
    )
    observation_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    observation_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    unit_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    qualifier_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    value_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    observation_event_id: Mapped[Optional[int]]
    obs_event_field_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )

    obs_event_field_concept: Mapped["Concept"] = relationship(
        primaryjoin="Observation.obs_event_field_concept_id == Concept.concept_id",
    )
    observation_concept: Mapped["Concept"] = relationship(
        primaryjoin="Observation.observation_concept_id == Concept.concept_id",
    )
    observation_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="Observation.observation_source_concept_id == Concept.concept_id",
    )
    observation_type_concept: Mapped["Concept"] = relationship(
        primaryjoin="Observation.observation_type_concept_id == Concept.concept_id",
    )
    person: Mapped["Person"] = relationship()
    provider: Mapped["Provider"] = relationship()
    qualifier_concept: Mapped["Concept"] = relationship(
        primaryjoin="Observation.qualifier_concept_id == Concept.concept_id"
    )
    unit_concept: Mapped["Concept"] = relationship(
        primaryjoin="Observation.unit_concept_id == Concept.concept_id"
    )
    value_as_concept: Mapped["Concept"] = relationship(
        primaryjoin="Observation.value_as_concept_id == Concept.concept_id"
    )
    visit_detail: Mapped["VisitDetail"] = relationship()
    visit_occurrence: Mapped["VisitOccurrence"] = relationship()


class ProcedureOccurrence(Base):  # noqa: D101
    __tablename__ = "procedure_occurrence"
    __table_args__ = {"schema": SCHEMA_NAME}

    procedure_occurrence_id: Mapped[int] = mapped_column(
        primary_key=True,
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.person.person_id"), index=True
    )
    procedure_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id"), index=True
    )
    procedure_date: Mapped[date]
    procedure_datetime: Mapped[Optional[datetime]]
    procedure_end_date: Mapped[Optional[date]]
    procedure_end_datetime: Mapped[Optional[datetime]]
    procedure_type_concept_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    modifier_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    quantity: Mapped[Optional[int]]
    provider_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.provider.provider_id")
    )
    visit_occurrence_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.visit_detail.visit_detail_id")
    )
    procedure_source_value: Mapped[Optional[str]] = mapped_column(String(50))
    procedure_source_concept_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(f"{SCHEMA_NAME}.concept.concept_id")
    )
    modifier_source_value: Mapped[Optional[str]] = mapped_column(String(50))

    modifier_concept: Mapped["Concept"] = relationship(
        primaryjoin="ProcedureOccurrence.modifier_concept_id == Concept.concept_id",
    )
    person: Mapped["Person"] = relationship()
    procedure_concept: Mapped["Concept"] = relationship(
        primaryjoin="ProcedureOccurrence.procedure_concept_id == Concept.concept_id",
    )
    procedure_source_concept: Mapped["Concept"] = relationship(
        primaryjoin="ProcedureOccurrence.procedure_source_concept_id == Concept.concept_id",
    )
    procedure_type_concept: Mapped["Concept"] = relationship(
        primaryjoin="ProcedureOccurrence.procedure_type_concept_id == Concept.concept_id",
    )
    provider: Mapped["Provider"] = relationship()
    visit_detail: Mapped["VisitDetail"] = relationship()
    visit_occurrence: Mapped["VisitOccurrence"] = relationship()
