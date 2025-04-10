from enum import Enum, StrEnum

SCT_CLINICAL_FINDING = "404684003"  # Clinical finding (finding)
SCT_ALLERGIC_DISPOSITION = "609328004"  # Allergic disposition (finding)
SCT_RADIOLOGIC_FINDING = "118247008"  # Radiologic finding (finding)
LOINC_EPISODE_OF_CARE_TYPE = "78030-4"  # Episode of care Type
SCT_PROCEDURE = "71388002"  # Procedure (procedure)
SCT_VENTILATOR_OBSERVABLE = "364698001"  # Ventilator observable (observable entity)

VS_LABORATORY_OBSERVATIONS = "https://www.netzwerk-universitaetsmedizin.de/fhir/cpg-on-ebm-on-fhir/ValueSet/vs-observations"
VS_LABORATORY_OBSERVATIONS_DOWNLOAD_URL = (
    "https://ceosys.github.io/cpg-on-ebm-on-fhir/ValueSet-vs-observations.json"
)


SCT_VENTILATOR_CARE_AND_ADJUSTMENT = (
    "385857005"  # Ventilator care and adjustment (regime/therapy)
)
SCT_LAB_FINDINGS_SURVEILLANCE = (
    "410394004"  # Lab findings surveillance (regime/therapy)
)

SCT_ASSESSMENT_SCALE = "273249006"  # Assessment scales (assessment scale)

CS_PLAN_DEFINITION_TYPE = "http://terminology.hl7.org/CodeSystem/plan-definition-type"
EXT_DOSAGE_CONDITION = "https://www.netzwerk-universitaetsmedizin.de/fhir/cpg-on-ebm-on-fhir/StructureDefinition/ext-dosage-condition"
EXT_ACTION_COMBINATION_METHOD = "https://www.netzwerk-universitaetsmedizin.de/fhir/cpg-on-ebm-on-fhir/StructureDefinition/ext-action-combination-method"
EXT_CPG_PARTOF = "http://hl7.org/fhir/uv/cpg/StructureDefinition/cpg-partOf"
EXT_RELATIVE_TIME = "https://www.netzwerk-universitaetsmedizin.de/fhir/cpg-on-ebm-on-fhir/StructureDefinition/relative-time"

CS_ACTION_COMBINATION_METHOD = "https://www.netzwerk-universitaetsmedizin.de/fhir/cpg-on-ebm-on-fhir/CodeSystem/cs-action-combination-method"

LOINC_TIDAL_VOLUME = "76222-9"  # Tidal volume ^on ventilator


class CohortCategory(StrEnum):
    """
    The category of a cohort.
    """

    BASE = "BASE"
    POPULATION = "POPULATION"
    INTERVENTION = "INTERVENTION"
    POPULATION_INTERVENTION = "POPULATION_INTERVENTION"

    def __repr__(self) -> str:
        """
        Get the string representation of the category.
        """
        return f"{self.__class__.__name__}.{self.name}"

    def __str__(self) -> str:
        """
        Get the string representation of the category.
        """
        return self.name


class OMOPConcepts(Enum):
    """
    Collection of standard concepts in the OMOP CDM.
    """

    VISIT_TYPE_STILL_PATIENT = 32220
    BODY_HEIGHT = 3036277  # Body height (observation; from LOINC)
    BODY_WEIGHT_LOINC = 3025315  # Body weight (maps to LOINC code)
    BODY_WEIGHT_SNOMED = 4099154  # Body weight (maps to SNOMED code)
    GENDER_FEMALE = 8532
    GENDER_MALE = 8507
    TIDAL_VOLUME_ON_VENTILATOR = 21490854
    ALLERGY = 43021170
    UNIT_ML_PER_KG = 9571
    UNIT_KG = 9529
    UNIT_IU_PER_KG = 9335
    UNIT_MG_PER_KG = 9562

    def __str__(self) -> str:
        """
        Get the string representation of the concept.
        """
        return str(self.value)
