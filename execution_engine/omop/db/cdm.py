from sqlalchemy import (
    Column,
    Date,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Table,
    Text,
    text,
)
from sqlalchemy.orm import relationship

from .base import Base, DateTime, metadata

t_cohort = Table(
    "cohort",
    metadata,
    Column("cohort_definition_id", Integer, nullable=False),
    Column("subject_id", Integer, nullable=False),
    Column("cohort_start_date", Date, nullable=False),
    Column("cohort_end_date", Date, nullable=False),
    schema="cds_cdm",
)


class Concept(Base):  # noqa: D101 # noqa: D101
    __tablename__ = "concept"
    __table_args__ = {"schema": "cds_cdm"}

    concept_id = Column(Integer, primary_key=True, index=True)
    concept_name = Column(String(255), nullable=False)
    domain_id = Column(
        ForeignKey("cds_cdm.domain.domain_id"), nullable=False, index=True
    )
    vocabulary_id = Column(
        ForeignKey("cds_cdm.vocabulary.vocabulary_id"), nullable=False, index=True
    )
    concept_class_id = Column(
        ForeignKey("cds_cdm.concept_class.concept_class_id"), nullable=False, index=True
    )
    standard_concept = Column(String(1))
    concept_code = Column(String(50), nullable=False, index=True)
    valid_start_date = Column(Date, nullable=False)
    valid_end_date = Column(Date, nullable=False)
    invalid_reason = Column(String(1))

    concept_class = relationship(
        "ConceptClas",
        primaryjoin="Concept.concept_class_id == ConceptClas.concept_class_id",
    )
    domain = relationship("Domain", primaryjoin="Concept.domain_id == Domain.domain_id")
    vocabulary = relationship(
        "Vocabulary", primaryjoin="Concept.vocabulary_id == Vocabulary.vocabulary_id"
    )


class ConceptClas(Base):  # noqa: D101 # noqa: D101
    __tablename__ = "concept_class"
    __table_args__ = {"schema": "cds_cdm"}

    concept_class_id = Column(String(20), primary_key=True, index=True)
    concept_class_name = Column(String(255), nullable=False)
    concept_class_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )

    concept_class_concept = relationship(
        "Concept",
        primaryjoin="ConceptClas.concept_class_concept_id == Concept.concept_id",
    )


class Domain(Base):  # noqa: D101
    __tablename__ = "domain"
    __table_args__ = {"schema": "cds_cdm"}

    domain_id = Column(String(20), primary_key=True, index=True)
    domain_name = Column(String(255), nullable=False)
    domain_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"), nullable=False)

    domain_concept = relationship(
        "Concept", primaryjoin="Domain.domain_concept_id == Concept.concept_id"
    )


class Vocabulary(Base):  # noqa: D101
    __tablename__ = "vocabulary"
    __table_args__ = {"schema": "cds_cdm"}

    vocabulary_id = Column(String(20), primary_key=True, index=True)
    vocabulary_name = Column(String(255), nullable=False)
    vocabulary_reference = Column(String(255))
    vocabulary_version = Column(String(255))
    vocabulary_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )

    vocabulary_concept = relationship(
        "Concept", primaryjoin="Vocabulary.vocabulary_concept_id == Concept.concept_id"
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
        ForeignKey("cds_cdm.concept.concept_id"),
        nullable=False,
    ),
    Column("vocabulary_version", String(20), nullable=False),
    schema="cds_cdm",
)


t_cohort_definition = Table(
    "cohort_definition",
    metadata,
    Column("cohort_definition_id", Integer, nullable=False),
    Column("cohort_definition_name", String(255), nullable=False),
    Column("cohort_definition_description", Text),
    Column(
        "definition_type_concept_id",
        ForeignKey("cds_cdm.concept.concept_id"),
        nullable=False,
    ),
    Column("cohort_definition_syntax", Text),
    Column(
        "subject_concept_id", ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    ),
    Column("cohort_initiation_date", Date),
    schema="cds_cdm",
)


t_concept_ancestor = Table(
    "concept_ancestor",
    metadata,
    Column(
        "ancestor_concept_id",
        ForeignKey("cds_cdm.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column(
        "descendant_concept_id",
        ForeignKey("cds_cdm.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column("min_levels_of_separation", Integer, nullable=False),
    Column("max_levels_of_separation", Integer, nullable=False),
    schema="cds_cdm",
)


t_concept_synonym = Table(
    "concept_synonym",
    metadata,
    Column(
        "concept_id",
        ForeignKey("cds_cdm.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column("concept_synonym_name", String(1000), nullable=False),
    Column(
        "language_concept_id", ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    ),
    schema="cds_cdm",
)


class Cost(Base):  # noqa: D101
    __tablename__ = "cost"
    __table_args__ = {"schema": "cds_cdm"}

    cost_id = Column(
        Integer,
        primary_key=True,
    )
    cost_event_id = Column(Integer, nullable=False, index=True)
    cost_domain_id = Column(ForeignKey("cds_cdm.domain.domain_id"), nullable=False)
    cost_type_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )
    currency_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    total_charge = Column(Numeric)
    total_cost = Column(Numeric)
    total_paid = Column(Numeric)
    paid_by_payer = Column(Numeric)
    paid_by_patient = Column(Numeric)
    paid_patient_copay = Column(Numeric)
    paid_patient_coinsurance = Column(Numeric)
    paid_patient_deductible = Column(Numeric)
    paid_by_primary = Column(Numeric)
    paid_ingredient_cost = Column(Numeric)
    paid_dispensing_fee = Column(Numeric)
    payer_plan_period_id = Column(Integer)
    amount_allowed = Column(Numeric)
    revenue_code_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    revenue_code_source_value = Column(String(50))
    drg_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    drg_source_value = Column(String(3))

    cost_domain = relationship("Domain")
    cost_type_concept = relationship(
        "Concept", primaryjoin="Cost.cost_type_concept_id == Concept.concept_id"
    )
    currency_concept = relationship(
        "Concept", primaryjoin="Cost.currency_concept_id == Concept.concept_id"
    )
    drg_concept = relationship(
        "Concept", primaryjoin="Cost.drg_concept_id == Concept.concept_id"
    )
    revenue_code_concept = relationship(
        "Concept", primaryjoin="Cost.revenue_code_concept_id == Concept.concept_id"
    )


t_drug_strength = Table(
    "drug_strength",
    metadata,
    Column(
        "drug_concept_id",
        ForeignKey("cds_cdm.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column(
        "ingredient_concept_id",
        ForeignKey("cds_cdm.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column("amount_value", Numeric),
    Column("amount_unit_concept_id", ForeignKey("cds_cdm.concept.concept_id")),
    Column("numerator_value", Numeric),
    Column("numerator_unit_concept_id", ForeignKey("cds_cdm.concept.concept_id")),
    Column("denominator_value", Numeric),
    Column("denominator_unit_concept_id", ForeignKey("cds_cdm.concept.concept_id")),
    Column("box_size", Integer),
    Column("valid_start_date", Date, nullable=False),
    Column("valid_end_date", Date, nullable=False),
    Column("invalid_reason", String(1)),
    schema="cds_cdm",
)


t_fact_relationship = Table(
    "fact_relationship",
    metadata,
    Column(
        "domain_concept_id_1",
        ForeignKey("cds_cdm.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column("fact_id_1", Integer, nullable=False),
    Column(
        "domain_concept_id_2",
        ForeignKey("cds_cdm.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column("fact_id_2", Integer, nullable=False),
    Column(
        "relationship_concept_id",
        ForeignKey("cds_cdm.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    schema="cds_cdm",
)


class Location(Base):  # noqa: D101
    __tablename__ = "location"
    __table_args__ = {"schema": "cds_cdm"}

    location_id = Column(
        Integer,
        primary_key=True,
        index=True,
    )
    address_1 = Column(String(50))
    address_2 = Column(String(50))
    city = Column(String(50))
    state = Column(String(2))
    zip = Column(String(9))
    county = Column(String(20))
    location_source_value = Column(String(50))
    country_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    country_source_value = Column(String(80))
    latitude = Column(Numeric)
    longitude = Column(Numeric)

    country_concept = relationship("Concept")


class Metadatum(Base):  # noqa: D101
    __tablename__ = "metadata"
    __table_args__ = {"schema": "cds_cdm"}

    metadata_id = Column(
        Integer,
        primary_key=True,
    )
    metadata_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False, index=True
    )
    metadata_type_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )
    name = Column(String(250), nullable=False)
    value_as_string = Column(String(250))
    value_as_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    value_as_number = Column(Numeric)
    metadata_date = Column(Date)
    metadata_datetime = Column(DateTime)

    metadata_concept = relationship(
        "Concept", primaryjoin="Metadatum.metadata_concept_id == Concept.concept_id"
    )
    metadata_type_concept = relationship(
        "Concept",
        primaryjoin="Metadatum.metadata_type_concept_id == Concept.concept_id",
    )
    value_as_concept = relationship(
        "Concept", primaryjoin="Metadatum.value_as_concept_id == Concept.concept_id"
    )


class NoteNlp(Base):  # noqa: D101
    __tablename__ = "note_nlp"
    __table_args__ = {"schema": "cds_cdm"}

    note_nlp_id = Column(
        Integer,
        primary_key=True,
    )
    note_id = Column(Integer, nullable=False, index=True)
    section_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    snippet = Column(String(250))
    offset = Column(String(50))
    lexical_variant = Column(String(250), nullable=False)
    note_nlp_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"), index=True)
    note_nlp_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    nlp_system = Column(String(250))
    nlp_date = Column(Date, nullable=False)
    nlp_datetime = Column(DateTime)
    term_exists = Column(String(1))
    term_temporal = Column(String(50))
    term_modifiers = Column(String(2000))

    note_nlp_concept = relationship(
        "Concept", primaryjoin="NoteNlp.note_nlp_concept_id == Concept.concept_id"
    )
    note_nlp_source_concept = relationship(
        "Concept",
        primaryjoin="NoteNlp.note_nlp_source_concept_id == Concept.concept_id",
    )
    section_concept = relationship(
        "Concept", primaryjoin="NoteNlp.section_concept_id == Concept.concept_id"
    )


class Relationship(Base):  # noqa: D101
    __tablename__ = "relationship"
    __table_args__ = {"schema": "cds_cdm"}

    relationship_id = Column(String(20), primary_key=True, index=True)
    relationship_name = Column(String(255), nullable=False)
    is_hierarchical = Column(String(1), nullable=False)
    defines_ancestry = Column(String(1), nullable=False)
    reverse_relationship_id = Column(String(20), nullable=False)
    relationship_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )

    relationship_concept = relationship("Concept")


t_source_to_concept_map = Table(
    "source_to_concept_map",
    metadata,
    Column("source_code", String(50), nullable=False, index=True),
    Column(
        "source_concept_id", ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    ),
    Column("source_vocabulary_id", String(20), nullable=False, index=True),
    Column("source_code_description", String(255)),
    Column(
        "target_concept_id",
        ForeignKey("cds_cdm.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column(
        "target_vocabulary_id",
        ForeignKey("cds_cdm.vocabulary.vocabulary_id"),
        nullable=False,
        index=True,
    ),
    Column("valid_start_date", Date, nullable=False),
    Column("valid_end_date", Date, nullable=False),
    Column("invalid_reason", String(1)),
    schema="cds_cdm",
)


class CareSite(Base):  # noqa: D101
    __tablename__ = "care_site"
    __table_args__ = {"schema": "cds_cdm"}

    care_site_id = Column(
        Integer,
        primary_key=True,
        index=True,
    )
    care_site_name = Column(String(255))
    place_of_service_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    location_id = Column(ForeignKey("cds_cdm.location.location_id"))
    care_site_source_value = Column(String(50))
    place_of_service_source_value = Column(String(50))

    location = relationship("Location")
    place_of_service_concept = relationship("Concept")


t_concept_relationship = Table(
    "concept_relationship",
    metadata,
    Column(
        "concept_id_1",
        ForeignKey("cds_cdm.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column(
        "concept_id_2",
        ForeignKey("cds_cdm.concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column(
        "relationship_id",
        ForeignKey("cds_cdm.relationship.relationship_id"),
        nullable=False,
        index=True,
    ),
    Column("valid_start_date", Date, nullable=False),
    Column("valid_end_date", Date, nullable=False),
    Column("invalid_reason", String(1)),
    schema="cds_cdm",
)


class Provider(Base):  # noqa: D101
    __tablename__ = "provider"
    __table_args__ = {"schema": "cds_cdm"}

    provider_id = Column(
        Integer,
        primary_key=True,
        index=True,
    )
    provider_name = Column(String(255))
    npi = Column(String(20))
    dea = Column(String(20))
    specialty_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    care_site_id = Column(ForeignKey("cds_cdm.care_site.care_site_id"))
    year_of_birth = Column(Integer)
    gender_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    provider_source_value = Column(String(50))
    specialty_source_value = Column(String(50))
    specialty_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    gender_source_value = Column(String(50))
    gender_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))

    care_site = relationship("CareSite")
    gender_concept = relationship(
        "Concept", primaryjoin="Provider.gender_concept_id == Concept.concept_id"
    )
    gender_source_concept = relationship(
        "Concept", primaryjoin="Provider.gender_source_concept_id == Concept.concept_id"
    )
    specialty_concept = relationship(
        "Concept", primaryjoin="Provider.specialty_concept_id == Concept.concept_id"
    )
    specialty_source_concept = relationship(
        "Concept",
        primaryjoin="Provider.specialty_source_concept_id == Concept.concept_id",
    )


class Person(Base):  # noqa: D101
    __tablename__ = "person"
    __table_args__ = {"schema": "cds_cdm"}

    person_id = Column(
        Integer,
        primary_key=True,
        index=True,
    )
    gender_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False, index=True
    )
    year_of_birth = Column(Integer, nullable=False)
    month_of_birth = Column(Integer)
    day_of_birth = Column(Integer)
    birth_datetime = Column(DateTime)
    race_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"), nullable=False)
    ethnicity_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )
    location_id = Column(ForeignKey("cds_cdm.location.location_id"))
    provider_id = Column(ForeignKey("cds_cdm.provider.provider_id"))
    care_site_id = Column(ForeignKey("cds_cdm.care_site.care_site_id"))
    person_source_value = Column(String(50))
    gender_source_value = Column(String(50))
    gender_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    race_source_value = Column(String(50))
    race_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    ethnicity_source_value = Column(String(50))
    ethnicity_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))

    care_site = relationship("CareSite")
    ethnicity_concept = relationship(
        "Concept", primaryjoin="Person.ethnicity_concept_id == Concept.concept_id"
    )
    ethnicity_source_concept = relationship(
        "Concept",
        primaryjoin="Person.ethnicity_source_concept_id == Concept.concept_id",
    )
    gender_concept = relationship(
        "Concept", primaryjoin="Person.gender_concept_id == Concept.concept_id"
    )
    gender_source_concept = relationship(
        "Concept", primaryjoin="Person.gender_source_concept_id == Concept.concept_id"
    )
    location = relationship("Location")
    provider = relationship("Provider")
    race_concept = relationship(
        "Concept", primaryjoin="Person.race_concept_id == Concept.concept_id"
    )
    race_source_concept = relationship(
        "Concept", primaryjoin="Person.race_source_concept_id == Concept.concept_id"
    )


class ConditionEra(Base):  # noqa: D101
    __tablename__ = "condition_era"
    __table_args__ = {"schema": "cds_cdm"}

    condition_era_id = Column(
        Integer,
        primary_key=True,
        server_default=text(
            "nextval('cds_cdm.condition_era_condition_era_id_seq'::regclass)"
        ),
    )
    person_id = Column(
        ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    )
    condition_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False, index=True
    )
    condition_era_start_date = Column(Date, nullable=False)
    condition_era_end_date = Column(Date, nullable=False)
    condition_occurrence_count = Column(Integer)

    condition_concept = relationship("Concept")
    person = relationship("Person")


t_death = Table(
    "death",
    metadata,
    Column(
        "person_id", ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    ),
    Column("death_date", Date, nullable=False),
    Column("death_datetime", DateTime),
    Column("death_type_concept_id", ForeignKey("cds_cdm.concept.concept_id")),
    Column("cause_concept_id", ForeignKey("cds_cdm.concept.concept_id")),
    Column("cause_source_value", String(50)),
    Column("cause_source_concept_id", ForeignKey("cds_cdm.concept.concept_id")),
    schema="cds_cdm",
)


class DoseEra(Base):  # noqa: D101
    __tablename__ = "dose_era"
    __table_args__ = {"schema": "cds_cdm"}

    dose_era_id = Column(
        Integer,
        primary_key=True,
    )
    person_id = Column(
        ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    )
    drug_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False, index=True
    )
    unit_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"), nullable=False)
    dose_value = Column(Numeric, nullable=False)
    dose_era_start_date = Column(Date, nullable=False)
    dose_era_end_date = Column(Date, nullable=False)

    drug_concept = relationship(
        "Concept", primaryjoin="DoseEra.drug_concept_id == Concept.concept_id"
    )
    person = relationship("Person")
    unit_concept = relationship(
        "Concept", primaryjoin="DoseEra.unit_concept_id == Concept.concept_id"
    )


class DrugEra(Base):  # noqa: D101
    __tablename__ = "drug_era"
    __table_args__ = {"schema": "cds_cdm"}

    drug_era_id = Column(
        Integer,
        primary_key=True,
    )
    person_id = Column(
        ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    )
    drug_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False, index=True
    )
    drug_era_start_date = Column(Date, nullable=False)
    drug_era_end_date = Column(Date, nullable=False)
    drug_exposure_count = Column(Integer)
    gap_days = Column(Integer)

    drug_concept = relationship("Concept")
    person = relationship("Person")


class Episode(Base):  # noqa: D101
    __tablename__ = "episode"
    __table_args__ = {"schema": "cds_cdm"}

    episode_id = Column(
        Integer,
        primary_key=True,
    )
    person_id = Column(ForeignKey("cds_cdm.person.person_id"), nullable=False)
    episode_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )
    episode_start_date = Column(Date, nullable=False)
    episode_start_datetime = Column(DateTime)
    episode_end_date = Column(Date)
    episode_end_datetime = Column(DateTime)
    episode_parent_id = Column(Integer)
    episode_number = Column(Integer)
    episode_object_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )
    episode_type_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )
    episode_source_value = Column(String(50))
    episode_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))

    episode_concept = relationship(
        "Concept", primaryjoin="Episode.episode_concept_id == Concept.concept_id"
    )
    episode_object_concept = relationship(
        "Concept", primaryjoin="Episode.episode_object_concept_id == Concept.concept_id"
    )
    episode_source_concept = relationship(
        "Concept", primaryjoin="Episode.episode_source_concept_id == Concept.concept_id"
    )
    episode_type_concept = relationship(
        "Concept", primaryjoin="Episode.episode_type_concept_id == Concept.concept_id"
    )
    person = relationship("Person")


class ObservationPeriod(Base):  # noqa: D101
    __tablename__ = "observation_period"
    __table_args__ = {"schema": "cds_cdm"}

    observation_period_id = Column(
        Integer,
        primary_key=True,
        server_default=text(
            "nextval('cds_cdm.observation_period_observation_period_id_seq'::regclass)"
        ),
    )
    person_id = Column(
        ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    )
    observation_period_start_date = Column(Date, nullable=False)
    observation_period_end_date = Column(Date, nullable=False)
    period_type_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )

    period_type_concept = relationship("Concept")
    person = relationship("Person")


class PayerPlanPeriod(Base):  # noqa: D101
    __tablename__ = "payer_plan_period"
    __table_args__ = {"schema": "cds_cdm"}

    payer_plan_period_id = Column(
        Integer,
        primary_key=True,
        server_default=text(
            "nextval('cds_cdm.payer_plan_period_payer_plan_period_id_seq'::regclass)"
        ),
    )
    person_id = Column(
        ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    )
    payer_plan_period_start_date = Column(Date, nullable=False)
    payer_plan_period_end_date = Column(Date, nullable=False)
    payer_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    payer_source_value = Column(String(50))
    payer_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    plan_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    plan_source_value = Column(String(50))
    plan_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    sponsor_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    sponsor_source_value = Column(String(50))
    sponsor_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    family_source_value = Column(String(50))
    stop_reason_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    stop_reason_source_value = Column(String(50))
    stop_reason_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))

    payer_concept = relationship(
        "Concept", primaryjoin="PayerPlanPeriod.payer_concept_id == Concept.concept_id"
    )
    payer_source_concept = relationship(
        "Concept",
        primaryjoin="PayerPlanPeriod.payer_source_concept_id == Concept.concept_id",
    )
    person = relationship("Person")
    plan_concept = relationship(
        "Concept", primaryjoin="PayerPlanPeriod.plan_concept_id == Concept.concept_id"
    )
    plan_source_concept = relationship(
        "Concept",
        primaryjoin="PayerPlanPeriod.plan_source_concept_id == Concept.concept_id",
    )
    sponsor_concept = relationship(
        "Concept",
        primaryjoin="PayerPlanPeriod.sponsor_concept_id == Concept.concept_id",
    )
    sponsor_source_concept = relationship(
        "Concept",
        primaryjoin="PayerPlanPeriod.sponsor_source_concept_id == Concept.concept_id",
    )
    stop_reason_concept = relationship(
        "Concept",
        primaryjoin="PayerPlanPeriod.stop_reason_concept_id == Concept.concept_id",
    )
    stop_reason_source_concept = relationship(
        "Concept",
        primaryjoin="PayerPlanPeriod.stop_reason_source_concept_id == Concept.concept_id",
    )


class Speciman(Base):  # noqa: D101
    __tablename__ = "specimen"
    __table_args__ = {"schema": "cds_cdm"}

    specimen_id = Column(
        Integer,
        primary_key=True,
    )
    person_id = Column(
        ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    )
    specimen_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False, index=True
    )
    specimen_type_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )
    specimen_date = Column(Date, nullable=False)
    specimen_datetime = Column(DateTime)
    quantity = Column(Numeric)
    unit_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    anatomic_site_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    disease_status_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    specimen_source_id = Column(String(50))
    specimen_source_value = Column(String(50))
    unit_source_value = Column(String(50))
    anatomic_site_source_value = Column(String(50))
    disease_status_source_value = Column(String(50))

    anatomic_site_concept = relationship(
        "Concept", primaryjoin="Speciman.anatomic_site_concept_id == Concept.concept_id"
    )
    disease_status_concept = relationship(
        "Concept",
        primaryjoin="Speciman.disease_status_concept_id == Concept.concept_id",
    )
    person = relationship("Person")
    specimen_concept = relationship(
        "Concept", primaryjoin="Speciman.specimen_concept_id == Concept.concept_id"
    )
    specimen_type_concept = relationship(
        "Concept", primaryjoin="Speciman.specimen_type_concept_id == Concept.concept_id"
    )
    unit_concept = relationship(
        "Concept", primaryjoin="Speciman.unit_concept_id == Concept.concept_id"
    )


class VisitOccurrence(Base):  # noqa: D101
    __tablename__ = "visit_occurrence"
    __table_args__ = {"schema": "cds_cdm"}

    visit_occurrence_id = Column(
        Integer,
        primary_key=True,
        server_default=text(
            "nextval('cds_cdm.visit_occurrence_visit_occurrence_id_seq'::regclass)"
        ),
    )
    person_id = Column(
        ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    )
    visit_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False, index=True
    )
    visit_start_date = Column(Date, nullable=False)
    visit_start_datetime = Column(DateTime)
    visit_end_date = Column(Date, nullable=False)
    visit_end_datetime = Column(DateTime)
    visit_type_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )
    provider_id = Column(ForeignKey("cds_cdm.provider.provider_id"))
    care_site_id = Column(ForeignKey("cds_cdm.care_site.care_site_id"))
    visit_source_value = Column(String(50))
    visit_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    admitted_from_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    admitted_from_source_value = Column(String(50))
    discharged_to_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    discharged_to_source_value = Column(String(50))
    preceding_visit_occurrence_id = Column(
        ForeignKey("cds_cdm.visit_occurrence.visit_occurrence_id")
    )

    admitted_from_concept = relationship(
        "Concept",
        primaryjoin="VisitOccurrence.admitted_from_concept_id == Concept.concept_id",
    )
    care_site = relationship("CareSite")
    discharged_to_concept = relationship(
        "Concept",
        primaryjoin="VisitOccurrence.discharged_to_concept_id == Concept.concept_id",
    )
    person = relationship("Person")
    preceding_visit_occurrence = relationship(
        "VisitOccurrence", remote_side=[visit_occurrence_id]
    )
    provider = relationship("Provider")
    visit_concept = relationship(
        "Concept", primaryjoin="VisitOccurrence.visit_concept_id == Concept.concept_id"
    )
    visit_source_concept = relationship(
        "Concept",
        primaryjoin="VisitOccurrence.visit_source_concept_id == Concept.concept_id",
    )
    visit_type_concept = relationship(
        "Concept",
        primaryjoin="VisitOccurrence.visit_type_concept_id == Concept.concept_id",
    )


t_episode_event = Table(
    "episode_event",
    metadata,
    Column("episode_id", ForeignKey("cds_cdm.episode.episode_id"), nullable=False),
    Column("event_id", Integer, nullable=False),
    Column(
        "episode_event_field_concept_id",
        ForeignKey("cds_cdm.concept.concept_id"),
        nullable=False,
    ),
    schema="cds_cdm",
)


class VisitDetail(Base):  # noqa: D101
    __tablename__ = "visit_detail"
    __table_args__ = {"schema": "cds_cdm"}

    visit_detail_id = Column(
        Integer,
        primary_key=True,
        server_default=text(
            "nextval('cds_cdm.visit_detail_visit_detail_id_seq'::regclass)"
        ),
    )
    person_id = Column(
        ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    )
    visit_detail_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False, index=True
    )
    visit_detail_start_date = Column(Date, nullable=False)
    visit_detail_start_datetime = Column(DateTime)
    visit_detail_end_date = Column(Date, nullable=False)
    visit_detail_end_datetime = Column(DateTime)
    visit_detail_type_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )
    provider_id = Column(ForeignKey("cds_cdm.provider.provider_id"))
    care_site_id = Column(ForeignKey("cds_cdm.care_site.care_site_id"))
    visit_detail_source_value = Column(String(50))
    visit_detail_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    admitted_from_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    admitted_from_source_value = Column(String(50))
    discharged_to_source_value = Column(String(50))
    discharged_to_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    preceding_visit_detail_id = Column(
        ForeignKey("cds_cdm.visit_detail.visit_detail_id")
    )
    parent_visit_detail_id = Column(ForeignKey("cds_cdm.visit_detail.visit_detail_id"))
    visit_occurrence_id = Column(
        ForeignKey("cds_cdm.visit_occurrence.visit_occurrence_id"),
        nullable=False,
        index=True,
    )

    admitted_from_concept = relationship(
        "Concept",
        primaryjoin="VisitDetail.admitted_from_concept_id == Concept.concept_id",
    )
    care_site = relationship("CareSite")
    discharged_to_concept = relationship(
        "Concept",
        primaryjoin="VisitDetail.discharged_to_concept_id == Concept.concept_id",
    )
    parent_visit_detail = relationship(
        "VisitDetail",
        remote_side=[visit_detail_id],
        primaryjoin="VisitDetail.parent_visit_detail_id == VisitDetail.visit_detail_id",
    )
    person = relationship("Person")
    preceding_visit_detail = relationship(
        "VisitDetail",
        remote_side=[visit_detail_id],
        primaryjoin="VisitDetail.preceding_visit_detail_id == VisitDetail.visit_detail_id",
    )
    provider = relationship("Provider")
    visit_detail_concept = relationship(
        "Concept",
        primaryjoin="VisitDetail.visit_detail_concept_id == Concept.concept_id",
    )
    visit_detail_source_concept = relationship(
        "Concept",
        primaryjoin="VisitDetail.visit_detail_source_concept_id == Concept.concept_id",
    )
    visit_detail_type_concept = relationship(
        "Concept",
        primaryjoin="VisitDetail.visit_detail_type_concept_id == Concept.concept_id",
    )
    visit_occurrence = relationship("VisitOccurrence")


class ConditionOccurrence(Base):  # noqa: D101
    __tablename__ = "condition_occurrence"
    __table_args__ = {"schema": "cds_cdm"}

    condition_occurrence_id = Column(
        Integer,
        primary_key=True,
        server_default=text(
            "nextval('cds_cdm.condition_occurrence_condition_occurrence_id_seq'::regclass)"
        ),
    )
    person_id = Column(
        ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    )
    condition_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False, index=True
    )
    condition_start_date = Column(Date, nullable=False)
    condition_start_datetime = Column(DateTime)
    condition_end_date = Column(Date)
    condition_end_datetime = Column(DateTime)
    condition_type_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )
    condition_status_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    stop_reason = Column(String(20))
    provider_id = Column(ForeignKey("cds_cdm.provider.provider_id"))
    visit_occurrence_id = Column(
        ForeignKey("cds_cdm.visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id = Column(ForeignKey("cds_cdm.visit_detail.visit_detail_id"))
    condition_source_value = Column(String(50))
    condition_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    condition_status_source_value = Column(String(50))

    condition_concept = relationship(
        "Concept",
        primaryjoin="ConditionOccurrence.condition_concept_id == Concept.concept_id",
    )
    condition_source_concept = relationship(
        "Concept",
        primaryjoin="ConditionOccurrence.condition_source_concept_id == Concept.concept_id",
    )
    condition_status_concept = relationship(
        "Concept",
        primaryjoin="ConditionOccurrence.condition_status_concept_id == Concept.concept_id",
    )
    condition_type_concept = relationship(
        "Concept",
        primaryjoin="ConditionOccurrence.condition_type_concept_id == Concept.concept_id",
    )
    person = relationship("Person")
    provider = relationship("Provider")
    visit_detail = relationship("VisitDetail")
    visit_occurrence = relationship("VisitOccurrence")


class DeviceExposure(Base):  # noqa: D101
    __tablename__ = "device_exposure"
    __table_args__ = {"schema": "cds_cdm"}

    device_exposure_id = Column(
        Integer,
        primary_key=True,
        server_default=text(
            "nextval('cds_cdm.device_exposure_device_exposure_id_seq'::regclass)"
        ),
    )
    person_id = Column(
        ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    )
    device_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False, index=True
    )
    device_exposure_start_date = Column(Date, nullable=False)
    device_exposure_start_datetime = Column(DateTime)
    device_exposure_end_date = Column(Date)
    device_exposure_end_datetime = Column(DateTime)
    device_type_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )
    unique_device_id = Column(String(255))
    production_id = Column(String(255))
    quantity = Column(Integer)
    provider_id = Column(ForeignKey("cds_cdm.provider.provider_id"))
    visit_occurrence_id = Column(
        ForeignKey("cds_cdm.visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id = Column(ForeignKey("cds_cdm.visit_detail.visit_detail_id"))
    device_source_value = Column(String(50))
    device_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    unit_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    unit_source_value = Column(String(50))
    unit_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))

    device_concept = relationship(
        "Concept", primaryjoin="DeviceExposure.device_concept_id == Concept.concept_id"
    )
    device_source_concept = relationship(
        "Concept",
        primaryjoin="DeviceExposure.device_source_concept_id == Concept.concept_id",
    )
    device_type_concept = relationship(
        "Concept",
        primaryjoin="DeviceExposure.device_type_concept_id == Concept.concept_id",
    )
    person = relationship("Person")
    provider = relationship("Provider")
    unit_concept = relationship(
        "Concept", primaryjoin="DeviceExposure.unit_concept_id == Concept.concept_id"
    )
    unit_source_concept = relationship(
        "Concept",
        primaryjoin="DeviceExposure.unit_source_concept_id == Concept.concept_id",
    )
    visit_detail = relationship("VisitDetail")
    visit_occurrence = relationship("VisitOccurrence")


class DrugExposure(Base):  # noqa: D101
    __tablename__ = "drug_exposure"
    __table_args__ = {"schema": "cds_cdm"}

    drug_exposure_id = Column(
        Integer,
        primary_key=True,
        server_default=text(
            "nextval('cds_cdm.drug_exposure_drug_exposure_id_seq'::regclass)"
        ),
    )
    person_id = Column(
        ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    )
    drug_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False, index=True
    )
    drug_exposure_start_date = Column(Date, nullable=False)
    drug_exposure_start_datetime = Column(DateTime)
    drug_exposure_end_date = Column(Date, nullable=False)
    drug_exposure_end_datetime = Column(DateTime)
    verbatim_end_date = Column(Date)
    drug_type_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )
    stop_reason = Column(String(20))
    refills = Column(Integer)
    quantity = Column(Numeric)
    days_supply = Column(Integer)
    sig = Column(Text)
    route_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    lot_number = Column(String(50))
    provider_id = Column(ForeignKey("cds_cdm.provider.provider_id"))
    visit_occurrence_id = Column(
        ForeignKey("cds_cdm.visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id = Column(ForeignKey("cds_cdm.visit_detail.visit_detail_id"))
    drug_source_value = Column(String(50))
    drug_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    route_source_value = Column(String(50))
    dose_unit_source_value = Column(String(50))

    drug_concept = relationship(
        "Concept", primaryjoin="DrugExposure.drug_concept_id == Concept.concept_id"
    )
    drug_source_concept = relationship(
        "Concept",
        primaryjoin="DrugExposure.drug_source_concept_id == Concept.concept_id",
    )
    drug_type_concept = relationship(
        "Concept", primaryjoin="DrugExposure.drug_type_concept_id == Concept.concept_id"
    )
    person = relationship("Person")
    provider = relationship("Provider")
    route_concept = relationship(
        "Concept", primaryjoin="DrugExposure.route_concept_id == Concept.concept_id"
    )
    visit_detail = relationship("VisitDetail")
    visit_occurrence = relationship("VisitOccurrence")


class Measurement(Base):  # noqa: D101
    __tablename__ = "measurement"
    __table_args__ = {"schema": "cds_cdm"}

    measurement_id = Column(
        Integer,
        primary_key=True,
        server_default=text(
            "nextval('cds_cdm.measurement_measurement_id_seq'::regclass)"
        ),
    )
    person_id = Column(
        ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    )
    measurement_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False, index=True
    )
    measurement_date = Column(Date, nullable=False)
    measurement_datetime = Column(DateTime)
    measurement_time = Column(String(10))
    measurement_type_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )
    operator_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    value_as_number = Column(Numeric)
    value_as_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    unit_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    range_low = Column(Numeric)
    range_high = Column(Numeric)
    provider_id = Column(ForeignKey("cds_cdm.provider.provider_id"))
    visit_occurrence_id = Column(
        ForeignKey("cds_cdm.visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id = Column(ForeignKey("cds_cdm.visit_detail.visit_detail_id"))
    measurement_source_value = Column(String(50))
    measurement_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    unit_source_value = Column(String(50))
    unit_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    value_source_value = Column(String(50))
    measurement_event_id = Column(Integer)
    meas_event_field_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))

    meas_event_field_concept = relationship(
        "Concept",
        primaryjoin="Measurement.meas_event_field_concept_id == Concept.concept_id",
    )
    measurement_concept = relationship(
        "Concept",
        primaryjoin="Measurement.measurement_concept_id == Concept.concept_id",
    )
    measurement_source_concept = relationship(
        "Concept",
        primaryjoin="Measurement.measurement_source_concept_id == Concept.concept_id",
    )
    measurement_type_concept = relationship(
        "Concept",
        primaryjoin="Measurement.measurement_type_concept_id == Concept.concept_id",
    )
    operator_concept = relationship(
        "Concept", primaryjoin="Measurement.operator_concept_id == Concept.concept_id"
    )
    person = relationship("Person")
    provider = relationship("Provider")
    unit_concept = relationship(
        "Concept", primaryjoin="Measurement.unit_concept_id == Concept.concept_id"
    )
    unit_source_concept = relationship(
        "Concept",
        primaryjoin="Measurement.unit_source_concept_id == Concept.concept_id",
    )
    value_as_concept = relationship(
        "Concept", primaryjoin="Measurement.value_as_concept_id == Concept.concept_id"
    )
    visit_detail = relationship("VisitDetail")
    visit_occurrence = relationship("VisitOccurrence")


class Note(Base):  # noqa: D101
    __tablename__ = "note"
    __table_args__ = {"schema": "cds_cdm"}

    note_id = Column(
        Integer,
        primary_key=True,
    )
    person_id = Column(
        ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    )
    note_date = Column(Date, nullable=False)
    note_datetime = Column(DateTime)
    note_type_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False, index=True
    )
    note_class_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )
    note_title = Column(String(250))
    note_text = Column(Text, nullable=False)
    encoding_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )
    language_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )
    provider_id = Column(ForeignKey("cds_cdm.provider.provider_id"))
    visit_occurrence_id = Column(
        ForeignKey("cds_cdm.visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id = Column(ForeignKey("cds_cdm.visit_detail.visit_detail_id"))
    note_source_value = Column(String(50))
    note_event_id = Column(Integer)
    note_event_field_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))

    encoding_concept = relationship(
        "Concept", primaryjoin="Note.encoding_concept_id == Concept.concept_id"
    )
    language_concept = relationship(
        "Concept", primaryjoin="Note.language_concept_id == Concept.concept_id"
    )
    note_class_concept = relationship(
        "Concept", primaryjoin="Note.note_class_concept_id == Concept.concept_id"
    )
    note_event_field_concept = relationship(
        "Concept", primaryjoin="Note.note_event_field_concept_id == Concept.concept_id"
    )
    note_type_concept = relationship(
        "Concept", primaryjoin="Note.note_type_concept_id == Concept.concept_id"
    )
    person = relationship("Person")
    provider = relationship("Provider")
    visit_detail = relationship("VisitDetail")
    visit_occurrence = relationship("VisitOccurrence")


class Observation(Base):  # noqa: D101
    __tablename__ = "observation"
    __table_args__ = {"schema": "cds_cdm"}

    observation_id = Column(
        Integer,
        primary_key=True,
        server_default=text(
            "nextval('cds_cdm.observation_observation_id_seq'::regclass)"
        ),
    )
    person_id = Column(
        ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    )
    observation_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False, index=True
    )
    observation_date = Column(Date, nullable=False)
    observation_datetime = Column(DateTime)
    observation_type_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )
    value_as_number = Column(Numeric)
    value_as_string = Column(String(60))
    value_as_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    qualifier_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    unit_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    provider_id = Column(ForeignKey("cds_cdm.provider.provider_id"))
    visit_occurrence_id = Column(
        ForeignKey("cds_cdm.visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id = Column(ForeignKey("cds_cdm.visit_detail.visit_detail_id"))
    observation_source_value = Column(String(50))
    observation_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    unit_source_value = Column(String(50))
    qualifier_source_value = Column(String(50))
    value_source_value = Column(String(50))
    observation_event_id = Column(Integer)
    obs_event_field_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))

    obs_event_field_concept = relationship(
        "Concept",
        primaryjoin="Observation.obs_event_field_concept_id == Concept.concept_id",
    )
    observation_concept = relationship(
        "Concept",
        primaryjoin="Observation.observation_concept_id == Concept.concept_id",
    )
    observation_source_concept = relationship(
        "Concept",
        primaryjoin="Observation.observation_source_concept_id == Concept.concept_id",
    )
    observation_type_concept = relationship(
        "Concept",
        primaryjoin="Observation.observation_type_concept_id == Concept.concept_id",
    )
    person = relationship("Person")
    provider = relationship("Provider")
    qualifier_concept = relationship(
        "Concept", primaryjoin="Observation.qualifier_concept_id == Concept.concept_id"
    )
    unit_concept = relationship(
        "Concept", primaryjoin="Observation.unit_concept_id == Concept.concept_id"
    )
    value_as_concept = relationship(
        "Concept", primaryjoin="Observation.value_as_concept_id == Concept.concept_id"
    )
    visit_detail = relationship("VisitDetail")
    visit_occurrence = relationship("VisitOccurrence")


class ProcedureOccurrence(Base):  # noqa: D101
    __tablename__ = "procedure_occurrence"
    __table_args__ = {"schema": "cds_cdm"}

    procedure_occurrence_id = Column(
        Integer,
        primary_key=True,
        server_default=text(
            "nextval('cds_cdm.procedure_occurrence_procedure_occurrence_id_seq'::regclass)"
        ),
    )
    person_id = Column(
        ForeignKey("cds_cdm.person.person_id"), nullable=False, index=True
    )
    procedure_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False, index=True
    )
    procedure_date = Column(Date, nullable=False)
    procedure_datetime = Column(DateTime)
    procedure_end_date = Column(Date)
    procedure_end_datetime = Column(DateTime)
    procedure_type_concept_id = Column(
        ForeignKey("cds_cdm.concept.concept_id"), nullable=False
    )
    modifier_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    quantity = Column(Integer)
    provider_id = Column(ForeignKey("cds_cdm.provider.provider_id"))
    visit_occurrence_id = Column(
        ForeignKey("cds_cdm.visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id = Column(ForeignKey("cds_cdm.visit_detail.visit_detail_id"))
    procedure_source_value = Column(String(50))
    procedure_source_concept_id = Column(ForeignKey("cds_cdm.concept.concept_id"))
    modifier_source_value = Column(String(50))

    modifier_concept = relationship(
        "Concept",
        primaryjoin="ProcedureOccurrence.modifier_concept_id == Concept.concept_id",
    )
    person = relationship("Person")
    procedure_concept = relationship(
        "Concept",
        primaryjoin="ProcedureOccurrence.procedure_concept_id == Concept.concept_id",
    )
    procedure_source_concept = relationship(
        "Concept",
        primaryjoin="ProcedureOccurrence.procedure_source_concept_id == Concept.concept_id",
    )
    procedure_type_concept = relationship(
        "Concept",
        primaryjoin="ProcedureOccurrence.procedure_type_concept_id == Concept.concept_id",
    )
    provider = relationship("Provider")
    visit_detail = relationship("VisitDetail")
    visit_occurrence = relationship("VisitOccurrence")
