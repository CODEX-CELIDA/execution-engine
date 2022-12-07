# coding: utf-8
from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm.decl_api import DeclarativeMeta

Base: DeclarativeMeta = declarative_base()
metadata = Base.metadata


t_cdm_source = Table(
    "cdm_source",
    metadata,
    Column("cdm_source_name", String(255), nullable=False),
    Column("cdm_source_abbreviation", String(25)),
    Column("cdm_holder", String(255)),
    Column("source_description", Text),
    Column("source_documentation_reference", String(255)),
    Column("cdm_etl_reference", String(255)),
    Column("source_release_date", Date),
    Column("cdm_release_date", Date),
    Column("cdm_version", String(10)),
    Column("vocabulary_version", String(20)),
)


class Cohort(Base):
    """
    OMOP CDM Table: Cohort

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#COHORT
    """

    __tablename__ = "cohort"

    cohort_definition_id = Column(Integer, primary_key=True, nullable=False, index=True)
    subject_id = Column(Integer, primary_key=True, nullable=False, index=True)
    cohort_start_date = Column(Date, primary_key=True, nullable=False)
    cohort_end_date = Column(Date, primary_key=True, nullable=False)


class Concept(Base):
    """
    OMOP CDM Table: Concept

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#CONCEPT
    """

    __tablename__ = "concept"

    concept_id = Column(Integer, primary_key=True, unique=True)
    concept_name = Column(String(255), nullable=False)
    domain_id = Column(ForeignKey("domain.domain_id"), nullable=False, index=True)
    vocabulary_id = Column(
        ForeignKey("vocabulary.vocabulary_id"), nullable=False, index=True
    )
    concept_class_id = Column(
        ForeignKey("concept_class.concept_class_id"), nullable=False, index=True
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


class ConceptClas(Base):
    """
    OMOP CDM Table: Concept Class

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#CONCEPT_CLASS
    """

    __tablename__ = "concept_class"

    concept_class_id = Column(String(20), primary_key=True, unique=True)
    concept_class_name = Column(String(255), nullable=False)
    concept_class_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)

    concept_class_concept = relationship(
        "Concept",
        primaryjoin="ConceptClas.concept_class_concept_id == Concept.concept_id",
    )


class Domain(Base):
    """
    OMOP CDM Table: Domain

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#DOMAIN
    """

    __tablename__ = "domain"

    domain_id = Column(String(20), primary_key=True, unique=True)
    domain_name = Column(String(255), nullable=False)
    domain_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)

    domain_concept = relationship(
        "Concept", primaryjoin="Domain.domain_concept_id == Concept.concept_id"
    )


class Location(Base):
    """
    OMOP CDM Table: Location

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#LOCATION
    """

    __tablename__ = "location"

    location_id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('location_location_id_seq'::regclass)"),
    )
    address_1 = Column(String(50))
    address_2 = Column(String(50))
    city = Column(String(50))
    state = Column(String(2))
    zip = Column(String(9))
    county = Column(String(20))
    location_source_value = Column(String(50))


t_metadata_ = Table(
    "metadata",
    metadata,
    Column("metadata_concept_id", Integer, nullable=False),
    Column("metadata_type_concept_id", Integer, nullable=False),
    Column("name", String(250), nullable=False),
    Column("value_as_string", Text),
    Column("value_as_concept_id", Integer),
    Column("metadata_date", Date),
    Column("metadata_datetime", DateTime),
)


class SourceToConceptMap(Base):
    """
    OMOP CDM Table: Source to Concept Map

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#SOURCE_TO_CONCEPT_MAP
    """

    __tablename__ = "source_to_concept_map"

    source_code = Column(String(50), primary_key=True, nullable=False, index=True)
    source_concept_id = Column(Integer, nullable=False)
    source_vocabulary_id = Column(
        String(20), primary_key=True, nullable=False, index=True
    )
    source_code_description = Column(String(255))
    target_concept_id = Column(Integer, primary_key=True, nullable=False, index=True)
    target_vocabulary_id = Column(String(20), nullable=False, index=True)
    valid_start_date = Column(Date, nullable=False)
    valid_end_date = Column(Date, primary_key=True, nullable=False)
    invalid_reason = Column(String(1))


t_test = Table(
    "test",
    metadata,
    Column("index", BigInteger, index=True),
    Column("drug_concept_id", BigInteger),
)


class Vocabulary(Base):
    """
    OMOP CDM Table: Vocabulary

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#VOCABULARY
    """

    __tablename__ = "vocabulary"

    vocabulary_id = Column(String(20), primary_key=True, unique=True)
    vocabulary_name = Column(String(255), nullable=False)
    vocabulary_reference = Column(String(255))
    vocabulary_version = Column(String(255))
    vocabulary_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)

    vocabulary_concept = relationship(
        "Concept", primaryjoin="Vocabulary.vocabulary_concept_id == Concept.concept_id"
    )


class AttributeDefinition(Base):
    """
    OMOP CDM Table: Attribute Definition

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#ATTRIBUTE_DEFINITION
    """

    __tablename__ = "attribute_definition"

    attribute_definition_id = Column(Integer, primary_key=True, index=True)
    attribute_name = Column(String(255), nullable=False)
    attribute_description = Column(Text)
    attribute_type_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)
    attribute_syntax = Column(Text)

    attribute_type_concept = relationship("Concept")


class CareSite(Base):
    """
    OMOP CDM Table: Care Site

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#CARE_SITE
    """

    __tablename__ = "care_site"

    care_site_id = Column(Integer, primary_key=True)
    care_site_name = Column(String(255))
    place_of_service_concept_id = Column(ForeignKey("concept.concept_id"))
    location_id = Column(ForeignKey("location.location_id"))
    care_site_source_value = Column(String(50))
    place_of_service_source_value = Column(String(50))

    location = relationship("Location")
    place_of_service_concept = relationship("Concept")


class CohortDefinition(Base):
    """
    OMOP CDM Table: Cohort Definition

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#COHORT_DEFINITION
    """

    __tablename__ = "cohort_definition"

    cohort_definition_id = Column(Integer, primary_key=True, index=True)
    cohort_definition_name = Column(String(255), nullable=False)
    cohort_definition_description = Column(Text)
    definition_type_concept_id = Column(
        ForeignKey("concept.concept_id"), nullable=False
    )
    cohort_definition_syntax = Column(Text)
    subject_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)
    cohort_initiation_date = Column(Date)

    definition_type_concept = relationship(
        "Concept",
        primaryjoin="CohortDefinition.definition_type_concept_id == Concept.concept_id",
    )
    subject_concept = relationship(
        "Concept",
        primaryjoin="CohortDefinition.subject_concept_id == Concept.concept_id",
    )


class ConceptAncestor(Base):
    """
    OMOP CDM Table: Concept Ancestor

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#CONCEPT_ANCESTOR
    """

    __tablename__ = "concept_ancestor"

    ancestor_concept_id = Column(
        ForeignKey("concept.concept_id"), primary_key=True, nullable=False, index=True
    )
    descendant_concept_id = Column(
        ForeignKey("concept.concept_id"), primary_key=True, nullable=False, index=True
    )
    min_levels_of_separation = Column(Integer, nullable=False)
    max_levels_of_separation = Column(Integer, nullable=False)

    ancestor_concept = relationship(
        "Concept",
        primaryjoin="ConceptAncestor.ancestor_concept_id == Concept.concept_id",
    )
    descendant_concept = relationship(
        "Concept",
        primaryjoin="ConceptAncestor.descendant_concept_id == Concept.concept_id",
    )


t_concept_synonym = Table(
    "concept_synonym",
    metadata,
    Column("concept_id", ForeignKey("concept.concept_id"), nullable=False, index=True),
    Column("concept_synonym_name", String(1000), nullable=False),
    Column("language_concept_id", ForeignKey("concept.concept_id"), nullable=False),
    UniqueConstraint("concept_id", "concept_synonym_name", "language_concept_id"),
)


class DrugStrength(Base):
    """
    OMOP CDM Table: Drug Strength

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#DRUG_STRENGTH
    """

    __tablename__ = "drug_strength"

    drug_concept_id = Column(
        ForeignKey("concept.concept_id"), primary_key=True, nullable=False, index=True
    )
    ingredient_concept_id = Column(
        ForeignKey("concept.concept_id"), primary_key=True, nullable=False, index=True
    )
    amount_value = Column(Numeric)
    amount_unit_concept_id = Column(ForeignKey("concept.concept_id"))
    numerator_value = Column(Numeric)
    numerator_unit_concept_id = Column(ForeignKey("concept.concept_id"))
    denominator_value = Column(Numeric)
    denominator_unit_concept_id = Column(ForeignKey("concept.concept_id"))
    box_size = Column(Integer)
    valid_start_date = Column(Date, nullable=False)
    valid_end_date = Column(Date, nullable=False)
    invalid_reason = Column(String(1))

    amount_unit_concept = relationship(
        "Concept",
        primaryjoin="DrugStrength.amount_unit_concept_id == Concept.concept_id",
    )
    denominator_unit_concept = relationship(
        "Concept",
        primaryjoin="DrugStrength.denominator_unit_concept_id == Concept.concept_id",
    )
    drug_concept = relationship(
        "Concept", primaryjoin="DrugStrength.drug_concept_id == Concept.concept_id"
    )
    ingredient_concept = relationship(
        "Concept",
        primaryjoin="DrugStrength.ingredient_concept_id == Concept.concept_id",
    )
    numerator_unit_concept = relationship(
        "Concept",
        primaryjoin="DrugStrength.numerator_unit_concept_id == Concept.concept_id",
    )


t_fact_relationship = Table(
    "fact_relationship",
    metadata,
    Column(
        "domain_concept_id_1",
        ForeignKey("concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column("fact_id_1", Integer, nullable=False),
    Column(
        "domain_concept_id_2",
        ForeignKey("concept.concept_id"),
        nullable=False,
        index=True,
    ),
    Column("fact_id_2", Integer, nullable=False),
    Column(
        "relationship_concept_id",
        ForeignKey("concept.concept_id"),
        nullable=False,
        index=True,
    ),
)


class Relationship(Base):
    """
    OMOP CDM Table: Relationship

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#RELATIONSHIP
    """

    __tablename__ = "relationship"

    relationship_id = Column(String(20), primary_key=True, unique=True)
    relationship_name = Column(String(255), nullable=False)
    is_hierarchical = Column(String(1), nullable=False)
    defines_ancestry = Column(String(1), nullable=False)
    reverse_relationship_id = Column(
        ForeignKey("relationship.relationship_id"), nullable=False
    )
    relationship_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)

    relationship_concept = relationship("Concept")
    reverse_relationship = relationship("Relationship", remote_side=[relationship_id])


class CohortAttribute(Base):
    """
    OMOP CDM Table: Cohort Attribute

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#COHORT_ATTRIBUTE
    """

    __tablename__ = "cohort_attribute"

    cohort_definition_id = Column(
        ForeignKey("cohort_definition.cohort_definition_id"),
        primary_key=True,
        nullable=False,
        index=True,
    )
    subject_id = Column(Integer, primary_key=True, nullable=False, index=True)
    cohort_start_date = Column(Date, primary_key=True, nullable=False)
    cohort_end_date = Column(Date, primary_key=True, nullable=False)
    attribute_definition_id = Column(
        ForeignKey("attribute_definition.attribute_definition_id"),
        primary_key=True,
        nullable=False,
    )
    value_as_number = Column(Numeric)
    value_as_concept_id = Column(ForeignKey("concept.concept_id"))

    attribute_definition = relationship("AttributeDefinition")
    cohort_definition = relationship("CohortDefinition")
    value_as_concept = relationship("Concept")


class ConceptRelationship(Base):
    """
    OMOP CDM Table: Concept Relationship

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#CONCEPT_RELATIONSHIP
    """

    __tablename__ = "concept_relationship"

    concept_id_1 = Column(
        ForeignKey("concept.concept_id"), primary_key=True, nullable=False, index=True
    )
    concept_id_2 = Column(
        ForeignKey("concept.concept_id"), primary_key=True, nullable=False, index=True
    )
    relationship_id = Column(
        ForeignKey("relationship.relationship_id"),
        primary_key=True,
        nullable=False,
        index=True,
    )
    valid_start_date = Column(Date, nullable=False)
    valid_end_date = Column(Date, nullable=False)
    invalid_reason = Column(String(1))

    concept = relationship(
        "Concept", primaryjoin="ConceptRelationship.concept_id_1 == Concept.concept_id"
    )
    concept1 = relationship(
        "Concept", primaryjoin="ConceptRelationship.concept_id_2 == Concept.concept_id"
    )
    relationship = relationship("Relationship")


class Provider(Base):
    """
    OMOP CDM Table: Provider

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#PROVIDER
    """

    __tablename__ = "provider"

    provider_id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('provider_provider_id_seq'::regclass)"),
    )
    provider_name = Column(String(255))
    npi = Column(String(20))
    dea = Column(String(20))
    specialty_concept_id = Column(ForeignKey("concept.concept_id"))
    care_site_id = Column(ForeignKey("care_site.care_site_id"))
    year_of_birth = Column(Integer)
    gender_concept_id = Column(ForeignKey("concept.concept_id"))
    provider_source_value = Column(String(50))
    specialty_source_value = Column(String(50))
    specialty_source_concept_id = Column(ForeignKey("concept.concept_id"))
    gender_source_value = Column(String(50))
    gender_source_concept_id = Column(ForeignKey("concept.concept_id"))

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


class Person(Base):
    """
    OMOP CDM Table: Person

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#PERSON
    """

    __tablename__ = "person"

    person_id = Column(
        Integer,
        primary_key=True,
        unique=True,
        server_default=text("nextval('person_person_id_seq'::regclass)"),
    )
    gender_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)
    year_of_birth = Column(Integer, nullable=False)
    month_of_birth = Column(Integer)
    day_of_birth = Column(Integer)
    birth_datetime = Column(DateTime)
    race_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)
    ethnicity_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)
    location_id = Column(ForeignKey("location.location_id"))
    provider_id = Column(ForeignKey("provider.provider_id"))
    care_site_id = Column(ForeignKey("care_site.care_site_id"))
    person_source_value = Column(String(50))
    gender_source_value = Column(String(50))
    gender_source_concept_id = Column(ForeignKey("concept.concept_id"))
    race_source_value = Column(String(50))
    race_source_concept_id = Column(ForeignKey("concept.concept_id"))
    ethnicity_source_value = Column(String(50))
    ethnicity_source_concept_id = Column(ForeignKey("concept.concept_id"))

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


class Death(Person):
    """
    OMOP CDM Table: Death

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#DEATH
    """

    __tablename__ = "death"

    person_id = Column(ForeignKey("person.person_id"), primary_key=True, index=True)
    death_date = Column(Date, nullable=False)
    death_datetime = Column(DateTime)
    death_type_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)
    cause_concept_id = Column(ForeignKey("concept.concept_id"))
    cause_source_value = Column(String(50))
    cause_source_concept_id = Column(ForeignKey("concept.concept_id"))

    cause_concept = relationship(
        "Concept", primaryjoin="Death.cause_concept_id == Concept.concept_id"
    )
    cause_source_concept = relationship(
        "Concept", primaryjoin="Death.cause_source_concept_id == Concept.concept_id"
    )
    death_type_concept = relationship(
        "Concept", primaryjoin="Death.death_type_concept_id == Concept.concept_id"
    )


class ConditionEra(Base):
    """
    OMOP CDM Table: Condition Era

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#CONDITION_ERA
    """

    __tablename__ = "condition_era"

    condition_era_id = Column(Integer, primary_key=True)
    person_id = Column(ForeignKey("person.person_id"), nullable=False, index=True)
    condition_concept_id = Column(
        ForeignKey("concept.concept_id"), nullable=False, index=True
    )
    condition_era_start_date = Column(Date, nullable=False)
    condition_era_end_date = Column(Date, nullable=False)
    condition_occurrence_count = Column(Integer)

    condition_concept = relationship("Concept")
    person = relationship("Person")


class DoseEra(Base):
    """
    OMOP CDM Table: Dose Era

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#DOSE_ERA
    """

    __tablename__ = "dose_era"

    dose_era_id = Column(Integer, primary_key=True)
    person_id = Column(ForeignKey("person.person_id"), nullable=False, index=True)
    drug_concept_id = Column(
        ForeignKey("concept.concept_id"), nullable=False, index=True
    )
    unit_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)
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


class DrugEra(Base):
    """
    OMOP CDM Table: Drug Era

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#DRUG_ERA
    """

    __tablename__ = "drug_era"

    drug_era_id = Column(Integer, primary_key=True)
    person_id = Column(ForeignKey("person.person_id"), nullable=False, index=True)
    drug_concept_id = Column(
        ForeignKey("concept.concept_id"), nullable=False, index=True
    )
    drug_era_start_date = Column(Date, nullable=False)
    drug_era_end_date = Column(Date, nullable=False)
    drug_exposure_count = Column(Integer)
    gap_days = Column(Integer)

    drug_concept = relationship("Concept")
    person = relationship("Person")


class ObservationPeriod(Base):
    """
    OMOP CDM Table: Observation Period

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#OBSERVATION_PERIOD
    """

    __tablename__ = "observation_period"

    observation_period_id = Column(
        Integer,
        primary_key=True,
        server_default=text(
            "nextval('observation_period_observation_period_id_seq'::regclass)"
        ),
    )
    person_id = Column(ForeignKey("person.person_id"), nullable=False, index=True)
    observation_period_start_date = Column(Date, nullable=False)
    observation_period_end_date = Column(Date, nullable=False)
    period_type_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)

    period_type_concept = relationship("Concept")
    person = relationship("Person")


class PayerPlanPeriod(Base):
    """
    OMOP CDM Table: Payer Plan Period

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#PAYER_PLAN_PERIOD
    """

    __tablename__ = "payer_plan_period"

    payer_plan_period_id = Column(Integer, primary_key=True)
    person_id = Column(ForeignKey("person.person_id"), nullable=False, index=True)
    payer_plan_period_start_date = Column(Date, nullable=False)
    payer_plan_period_end_date = Column(Date, nullable=False)
    payer_concept_id = Column(Integer)
    payer_source_value = Column(String(50))
    payer_source_concept_id = Column(Integer)
    plan_concept_id = Column(Integer)
    plan_source_value = Column(String(50))
    plan_source_concept_id = Column(Integer)
    sponsor_concept_id = Column(Integer)
    sponsor_source_value = Column(String(50))
    sponsor_source_concept_id = Column(Integer)
    family_source_value = Column(String(50))
    stop_reason_concept_id = Column(Integer)
    stop_reason_source_value = Column(String(50))
    stop_reason_source_concept_id = Column(Integer)

    person = relationship("Person")


class Speciman(Base):
    """
    OMOP CDM Table: Specimen

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#SPECIMEN
    """

    __tablename__ = "specimen"

    specimen_id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('specimen_specimen_id_seq'::regclass)"),
    )
    person_id = Column(ForeignKey("person.person_id"), nullable=False, index=True)
    specimen_concept_id = Column(
        ForeignKey("concept.concept_id"), nullable=False, index=True
    )
    specimen_type_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)
    specimen_date = Column(Date, nullable=False)
    specimen_datetime = Column(DateTime)
    quantity = Column(Numeric)
    unit_concept_id = Column(ForeignKey("concept.concept_id"))
    anatomic_site_concept_id = Column(ForeignKey("concept.concept_id"))
    disease_status_concept_id = Column(ForeignKey("concept.concept_id"))
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


class VisitOccurrence(Base):
    """
    OMOP CDM Table: Visit Occurrence

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#VISIT_OCCURRENCE
    """

    __tablename__ = "visit_occurrence"

    visit_occurrence_id = Column(
        Integer,
        primary_key=True,
        server_default=text(
            "nextval('visit_occurrence_visit_occurrence_id_seq'::regclass)"
        ),
    )
    person_id = Column(ForeignKey("person.person_id"), nullable=False, index=True)
    visit_concept_id = Column(Integer, nullable=False, index=True)
    visit_start_date = Column(Date, nullable=False)
    visit_start_datetime = Column(DateTime)
    visit_end_date = Column(Date, nullable=False)
    visit_end_datetime = Column(DateTime)
    visit_type_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)
    provider_id = Column(ForeignKey("provider.provider_id"))
    care_site_id = Column(ForeignKey("care_site.care_site_id"))
    visit_source_value = Column(String(50))
    visit_source_concept_id = Column(ForeignKey("concept.concept_id"))
    admitting_source_concept_id = Column(ForeignKey("concept.concept_id"))
    admitting_source_value = Column(String(50))
    discharge_to_concept_id = Column(ForeignKey("concept.concept_id"))
    discharge_to_source_value = Column(String(50))
    preceding_visit_occurrence_id = Column(
        ForeignKey("visit_occurrence.visit_occurrence_id")
    )

    admitting_source_concept = relationship(
        "Concept",
        primaryjoin="VisitOccurrence.admitting_source_concept_id == Concept.concept_id",
    )
    care_site = relationship("CareSite")
    discharge_to_concept = relationship(
        "Concept",
        primaryjoin="VisitOccurrence.discharge_to_concept_id == Concept.concept_id",
    )
    person = relationship("Person")
    preceding_visit_occurrence = relationship(
        "VisitOccurrence", remote_side=[visit_occurrence_id]
    )
    provider = relationship("Provider")
    visit_source_concept = relationship(
        "Concept",
        primaryjoin="VisitOccurrence.visit_source_concept_id == Concept.concept_id",
    )
    visit_type_concept = relationship(
        "Concept",
        primaryjoin="VisitOccurrence.visit_type_concept_id == Concept.concept_id",
    )


class ConditionOccurrence(Base):
    """
    OMOP CDM Table: Condition Occurrence

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#CONDITION_OCCURRENCE
    """

    __tablename__ = "condition_occurrence"

    condition_occurrence_id = Column(
        Integer,
        primary_key=True,
        server_default=text(
            "nextval('condition_occurrence_condition_occurrence_id_seq'::regclass)"
        ),
    )
    person_id = Column(ForeignKey("person.person_id"), nullable=False, index=True)
    condition_concept_id = Column(
        ForeignKey("concept.concept_id"), nullable=False, index=True
    )
    condition_start_date = Column(Date, nullable=False)
    condition_start_datetime = Column(DateTime)
    condition_end_date = Column(Date)
    condition_end_datetime = Column(DateTime)
    condition_type_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)
    stop_reason = Column(String(20))
    provider_id = Column(ForeignKey("provider.provider_id"))
    visit_occurrence_id = Column(
        ForeignKey("visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id = Column(Integer)
    condition_source_value = Column(String(50))
    condition_source_concept_id = Column(ForeignKey("concept.concept_id"))
    condition_status_source_value = Column(String(50))
    condition_status_concept_id = Column(ForeignKey("concept.concept_id"))

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
    visit_occurrence = relationship("VisitOccurrence")


class Cost(Base):
    """
    OMOP CDM Table: Cost

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#COST
    """

    __tablename__ = "cost"

    cost_id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('cost_cost_id_seq'::regclass)"),
    )
    cost_event_id = Column(Integer, nullable=False)
    cost_domain_id = Column(String(20), nullable=False)
    cost_type_concept_id = Column(Integer, nullable=False)
    currency_concept_id = Column(ForeignKey("concept.concept_id"))
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
    payer_plan_period_id = Column(ForeignKey("payer_plan_period.payer_plan_period_id"))
    amount_allowed = Column(Numeric)
    revenue_code_concept_id = Column(Integer)
    reveue_code_source_value = Column(String(50))
    drg_concept_id = Column(ForeignKey("concept.concept_id"))
    drg_source_value = Column(String(3))

    currency_concept = relationship(
        "Concept", primaryjoin="Cost.currency_concept_id == Concept.concept_id"
    )
    drg_concept = relationship(
        "Concept", primaryjoin="Cost.drg_concept_id == Concept.concept_id"
    )
    payer_plan_period = relationship("PayerPlanPeriod")


class DeviceExposure(Base):
    """
    OMOP CDM Table: Device Exposure

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#DEVICE_EXPOSURE
    """

    __tablename__ = "device_exposure"

    device_exposure_id = Column(Integer, primary_key=True)
    person_id = Column(ForeignKey("person.person_id"), nullable=False, index=True)
    device_concept_id = Column(
        ForeignKey("concept.concept_id"), nullable=False, index=True
    )
    device_exposure_start_date = Column(Date, nullable=False)
    device_exposure_start_datetime = Column(DateTime)
    device_exposure_end_date = Column(Date)
    device_exposure_end_datetime = Column(DateTime)
    device_type_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)
    unique_device_id = Column(String(50))
    quantity = Column(Integer)
    provider_id = Column(ForeignKey("provider.provider_id"))
    visit_occurrence_id = Column(
        ForeignKey("visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id = Column(Integer)
    device_source_value = Column(String(100))
    device_source_concept_id = Column(ForeignKey("concept.concept_id"))

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
    visit_occurrence = relationship("VisitOccurrence")


class DrugExposure(Base):
    """
    OMOP CDM Table: Drug Exposure

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#DRUG_EXPOSURE
    """

    __tablename__ = "drug_exposure"

    drug_exposure_id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('drug_exposure_drug_exposure_id_seq'::regclass)"),
    )
    person_id = Column(ForeignKey("person.person_id"), nullable=False, index=True)
    drug_concept_id = Column(
        ForeignKey("concept.concept_id"), nullable=False, index=True
    )
    drug_exposure_start_date = Column(Date, nullable=False)
    drug_exposure_start_datetime = Column(DateTime)
    drug_exposure_end_date = Column(Date, nullable=False)
    drug_exposure_end_datetime = Column(DateTime)
    verbatim_end_date = Column(Date)
    drug_type_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)
    stop_reason = Column(String(20))
    refills = Column(Integer)
    quantity = Column(Numeric)
    days_supply = Column(Integer)
    sig = Column(Text)
    route_concept_id = Column(ForeignKey("concept.concept_id"))
    lot_number = Column(String(50))
    provider_id = Column(ForeignKey("provider.provider_id"))
    visit_occurrence_id = Column(
        ForeignKey("visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id = Column(Integer)
    drug_source_value = Column(String(50))
    drug_source_concept_id = Column(ForeignKey("concept.concept_id"))
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
    visit_occurrence = relationship("VisitOccurrence")


class Measurement(Base):
    """
    OMOP CDM Table: Measurement

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#MEASUREMENT
    """

    __tablename__ = "measurement"

    measurement_id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('measurement_measurement_id_seq'::regclass)"),
    )
    person_id = Column(ForeignKey("person.person_id"), nullable=False, index=True)
    measurement_concept_id = Column(
        ForeignKey("concept.concept_id"), nullable=False, index=True
    )
    measurement_date = Column(Date, nullable=False)
    measurement_datetime = Column(DateTime)
    measurement_time = Column(String(10))
    measurement_type_concept_id = Column(
        ForeignKey("concept.concept_id"), nullable=False
    )
    operator_concept_id = Column(ForeignKey("concept.concept_id"))
    value_as_number = Column(Numeric)
    value_as_concept_id = Column(ForeignKey("concept.concept_id"))
    unit_concept_id = Column(ForeignKey("concept.concept_id"))
    range_low = Column(Numeric)
    range_high = Column(Numeric)
    provider_id = Column(ForeignKey("provider.provider_id"))
    visit_occurrence_id = Column(
        ForeignKey("visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id = Column(Integer)
    measurement_source_value = Column(String(50))
    measurement_source_concept_id = Column(ForeignKey("concept.concept_id"))
    unit_source_value = Column(String(50))
    value_source_value = Column(String(50))

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
    value_as_concept = relationship(
        "Concept", primaryjoin="Measurement.value_as_concept_id == Concept.concept_id"
    )
    visit_occurrence = relationship("VisitOccurrence")


class Note(Base):
    """
    OMOP CDM Table: Note

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#NOTE
    """

    __tablename__ = "note"

    note_id = Column(Integer, primary_key=True)
    person_id = Column(ForeignKey("person.person_id"), nullable=False, index=True)
    note_date = Column(Date, nullable=False)
    note_datetime = Column(DateTime)
    note_type_concept_id = Column(
        ForeignKey("concept.concept_id"), nullable=False, index=True
    )
    note_class_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)
    note_title = Column(String(250))
    note_text = Column(Text)
    encoding_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)
    language_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)
    provider_id = Column(ForeignKey("provider.provider_id"))
    visit_occurrence_id = Column(
        ForeignKey("visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id = Column(Integer)
    note_source_value = Column(String(50))

    encoding_concept = relationship(
        "Concept", primaryjoin="Note.encoding_concept_id == Concept.concept_id"
    )
    language_concept = relationship(
        "Concept", primaryjoin="Note.language_concept_id == Concept.concept_id"
    )
    note_class_concept = relationship(
        "Concept", primaryjoin="Note.note_class_concept_id == Concept.concept_id"
    )
    note_type_concept = relationship(
        "Concept", primaryjoin="Note.note_type_concept_id == Concept.concept_id"
    )
    person = relationship("Person")
    provider = relationship("Provider")
    visit_occurrence = relationship("VisitOccurrence")


class Observation(Base):
    """
    OMOP CDM Table: Observation

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#OBSERVATION
    """

    __tablename__ = "observation"

    observation_id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('observation_observation_id_seq'::regclass)"),
    )
    person_id = Column(ForeignKey("person.person_id"), nullable=False, index=True)
    observation_concept_id = Column(
        ForeignKey("concept.concept_id"), nullable=False, index=True
    )
    observation_date = Column(Date, nullable=False)
    observation_datetime = Column(DateTime)
    observation_type_concept_id = Column(
        ForeignKey("concept.concept_id"), nullable=False
    )
    value_as_number = Column(Numeric)
    value_as_string = Column(String(60))
    value_as_concept_id = Column(ForeignKey("concept.concept_id"))
    qualifier_concept_id = Column(ForeignKey("concept.concept_id"))
    unit_concept_id = Column(ForeignKey("concept.concept_id"))
    provider_id = Column(ForeignKey("provider.provider_id"))
    visit_occurrence_id = Column(
        ForeignKey("visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id = Column(Integer)
    observation_source_value = Column(String(50))
    observation_source_concept_id = Column(ForeignKey("concept.concept_id"))
    unit_source_value = Column(String(50))
    qualifier_source_value = Column(String(50))

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
    visit_occurrence = relationship("VisitOccurrence")


class ProcedureOccurrence(Base):
    """
    OMOP CDM Table: Procedure Occurrence

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#PROCEDURE_OCCURRENCE
    """

    __tablename__ = "procedure_occurrence"

    procedure_occurrence_id = Column(
        Integer,
        primary_key=True,
        server_default=text(
            "nextval('procedure_occurrence_procedure_occurrence_id_seq'::regclass)"
        ),
    )
    person_id = Column(ForeignKey("person.person_id"), nullable=False, index=True)
    procedure_concept_id = Column(
        ForeignKey("concept.concept_id"), nullable=False, index=True
    )
    procedure_date = Column(Date, nullable=False)
    procedure_datetime = Column(DateTime)
    procedure_type_concept_id = Column(ForeignKey("concept.concept_id"), nullable=False)
    modifier_concept_id = Column(ForeignKey("concept.concept_id"))
    quantity = Column(Integer)
    provider_id = Column(ForeignKey("provider.provider_id"))
    visit_occurrence_id = Column(
        ForeignKey("visit_occurrence.visit_occurrence_id"), index=True
    )
    visit_detail_id = Column(Integer)
    procedure_source_value = Column(String(50))
    procedure_source_concept_id = Column(ForeignKey("concept.concept_id"))
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
    visit_occurrence = relationship("VisitOccurrence")


class VisitDetail(Base):
    """
    OMOP CDM Table: Visit Detail

    Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html#VISIT_DETAIL
    """

    __tablename__ = "visit_detail"

    visit_detail_id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('visit_detail_visit_detail_id_seq'::regclass)"),
    )
    person_id = Column(ForeignKey("person.person_id"), nullable=False, index=True)
    visit_detail_concept_id = Column(Integer, nullable=False, index=True)
    visit_detail_start_date = Column(Date, nullable=False)
    visit_detail_start_datetime = Column(DateTime)
    visit_detail_end_date = Column(Date, nullable=False)
    visit_detail_end_datetime = Column(DateTime)
    visit_detail_type_concept_id = Column(
        ForeignKey("concept.concept_id"), nullable=False
    )
    provider_id = Column(ForeignKey("provider.provider_id"))
    care_site_id = Column(ForeignKey("care_site.care_site_id"))
    admitting_source_concept_id = Column(ForeignKey("concept.concept_id"))
    discharge_to_concept_id = Column(ForeignKey("concept.concept_id"))
    preceding_visit_detail_id = Column(ForeignKey("visit_detail.visit_detail_id"))
    visit_detail_source_value = Column(String(50))
    visit_detail_source_concept_id = Column(ForeignKey("concept.concept_id"))
    admitting_source_value = Column(String(50))
    discharge_to_source_value = Column(String(50))
    visit_detail_parent_id = Column(ForeignKey("visit_detail.visit_detail_id"))
    visit_occurrence_id = Column(
        ForeignKey("visit_occurrence.visit_occurrence_id"), nullable=False
    )

    admitting_source_concept = relationship(
        "Concept",
        primaryjoin="VisitDetail.admitting_source_concept_id == Concept.concept_id",
    )
    care_site = relationship("CareSite")
    discharge_to_concept = relationship(
        "Concept",
        primaryjoin="VisitDetail.discharge_to_concept_id == Concept.concept_id",
    )
    person = relationship("Person")
    preceding_visit_detail = relationship(
        "VisitDetail",
        remote_side=[visit_detail_id],
        primaryjoin="VisitDetail.preceding_visit_detail_id == VisitDetail.visit_detail_id",
    )
    provider = relationship("Provider")
    visit_detail_parent = relationship(
        "VisitDetail",
        remote_side=[visit_detail_id],
        primaryjoin="VisitDetail.visit_detail_parent_id == VisitDetail.visit_detail_id",
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


class NoteNlp(Base):
    """
    Note NLP table

    Ref: https://ohdsi.github.io/CommonDataModel/cdm54.html#NOTE_NLP
    """

    __tablename__ = "note_nlp"

    note_nlp_id = Column(Integer, primary_key=True)
    note_id = Column(ForeignKey("note.note_id"), nullable=False, index=True)
    section_concept_id = Column(ForeignKey("concept.concept_id"))
    snippet = Column(String(250))
    offset = Column(String(250))
    lexical_variant = Column(String(250), nullable=False)
    note_nlp_concept_id = Column(ForeignKey("concept.concept_id"), index=True)
    note_nlp_source_concept_id = Column(Integer)
    nlp_system = Column(String(250))
    nlp_date = Column(Date, nullable=False)
    nlp_datetime = Column(DateTime)
    term_exists = Column(String(1))
    term_temporal = Column(String(50))
    term_modifiers = Column(String(2000))

    note = relationship("Note")
    note_nlp_concept = relationship(
        "Concept", primaryjoin="NoteNlp.note_nlp_concept_id == Concept.concept_id"
    )
    section_concept = relationship(
        "Concept", primaryjoin="NoteNlp.section_concept_id == Concept.concept_id"
    )
