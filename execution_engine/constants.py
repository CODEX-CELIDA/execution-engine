from enum import Enum

SCT_CLINICAL_FINDING = "404684003"  # Clinical finding (finding)
SCT_ALLERGIC_DISPOSITION = "609328004"  # Allergic disposition (finding)
SCT_RADIOLOGIC_FINDING = "118247008"  # Radiologic finding (finding)
LOINC_EPISODE_OF_CARE_TYPE = "78030-4"  # Episode of care Type
SCT_PROCEDURE = "71388002"  # Procedure (procedure)
SCT_VENTILATOR_OBSERVABLE = "364698001"  # Ventilator observable (observable entity)

VS_VENTILATOR_OBSERVATIONS = "https://medizininformatik-initiative.de/fhir/ext/modul-icu/ValueSet/Code-Observation-Beatmung-LOINC"
VS_LABORATORY_OBSERVATIONS = "https://www.netzwerk-universitaetsmedizin.de/fhir/cpg-on-ebm-on-fhir/ValueSet/vs-laboratory-observations"

SCT_VENTILATOR_CARE_AND_ADJUSTMENT = (
    "385857005"  # Ventilator care and adjustment (regime/therapy)
)
SCT_LAB_FINDINGS_SURVEILLANCE = (
    "410394004"  # Lab findings surveillance (regime/therapy)
)

CS_PLAN_DEFINITION_TYPE = "http://terminology.hl7.org/CodeSystem/plan-definition-type"
EXT_DOSAGE_CONDITION = "https://www.netzwerk-universitaetsmedizin.de/fhir/cpg-on-ebm-on-fhir/StructureDefinition/ext-dosage-condition"
EXT_CPG_PARTOF = "http://hl7.org/fhir/uv/cpg/StructureDefinition/cpg-partOf"

LOINC_TIDAL_VOLUME = "76222-9"  # Tidal volume ^on ventilator


class CohortCategory(Enum):
    """
    The category of a cohort.
    """

    BASE = "base"
    POPULATION = "population"
    INTERVENTION = "intervention"
    POPULATION_INTERVENTION = "population_intervention"

    def __str__(self) -> str:
        """
        Get the string representation of the category.
        """
        return self.name


class IntervalType(Enum):
    """
    The type of interval
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NODATA = "nodata"


class OMOPConcepts(Enum):
    """
    Collection of standard concepts in the OMOP CDM.
    """

    VISIT_TYPE_STILL_PATIENT = "32220"
    BODY_HEIGHT = 3036277  # Body height (observation)
    BODY_WEIGHT = 3025315  # Body weight (observation)
    GENDER_FEMALE = 8532
    GENDER_MALE = 8507
    TIDAL_VOLUME_ON_VENTILATOR = 21490854
    ALLERGY = 43021170
