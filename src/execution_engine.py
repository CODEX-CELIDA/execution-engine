from typing import List
from enum import Enum
from fhir_tx import get_descendents, get_value_set
from omop.concepts import ConceptSet, ConceptSetManager
from omop.vocabulary import Vocabulary, get_concept_info, get_standard_concept
from fhir.resources.codeableconcept import CodeableConcept

SCT_CLINICAL_FINDING = "404684003"  # Clinical finding (finding)
SCT_ALLERGIC_DISPOSITION = "609328004"  # Allergic disposition (finding)
SCT_RADIOLOGIC_FINDING = "118247008"  # Radiologic finding (finding)
SCT_EPISODE_OF_CARE_TYPE = "78030-4"  # Episode of care Type
SCT_PROCEDURE = "71388002"  # Procedure (procedure)
SCT_VENTILATOR_OBSERVABLE = "364698001"  # Ventilator observable (observable entity)

VS_VENTILATOR_OBSERVABLE = "https://medizininformatik-initiative.de/fhir/ext/modul-icu/ValueSet/Code-Observation-Beatmung-LOINC"


SNOMEDCT = "http://snomed.info/sct"
LOINC = "http://loinc.org"


def isSNOMEDCT(cc: CodeableConcept) -> bool:
    """Checks if the given CodeableConcept is a SNOMED CT concept."""
    return cc.system == SNOMEDCT


def isLOINC(cc: CodeableConcept) -> bool:
    """Checks if the given CodeableConcept is a LOINC concept."""
    return cc.system == LOINC


def isCondition(cc: CodeableConcept) -> bool:
    """Checks if the given CodeableConcept is a Condition."""
    return isSNOMEDCT(cc) and cc.code == SCT_CLINICAL_FINDING


def isAllergy(cc: CodeableConcept) -> bool:
    """Checks if the given CodeableConcept is an Allergy."""
    return isSNOMEDCT(cc) and cc.code == SCT_ALLERGIC_DISPOSITION


def isRadiology(cc: CodeableConcept) -> bool:
    """Checks if the given CodeableConcept is a Radiology finding."""
    return isSNOMEDCT(cc) and cc.code == SCT_RADIOLOGIC_FINDING


def isProcedure(cc: CodeableConcept) -> bool:
    """Checks if the given CodeableConcept is a Procedure."""
    return isSNOMEDCT(cc) and cc.code == SCT_PROCEDURE


def isEpisodeOfCare(cc: CodeableConcept) -> bool:
    """Checks if the given CodeableConcept is an Episode of Care."""
    return isLOINC(cc) and cc.code == SCT_EPISODE_OF_CARE_TYPE


def isVentilationObservable(cc: CodeableConcept) -> bool:
    """Checks if the given CodeableConcept is a Ventilation Observable."""
    ventilationObservablesSCT = get_descendents(SNOMEDCT, SCT_VENTILATOR_OBSERVABLE)
    ventilationObservablesLOINC = get_value_set(VS_VENTILATOR_OBSERVABLE)

    return (isSNOMEDCT(cc) and cc.code in ventilationObservablesSCT) or (
        isLOINC(cc) and cc.code in ventilationObservablesLOINC
    )


def isLaboratory(cc: CodeableConcept) -> bool:
    """Checks if the given CodeableConcept is a Laboratory finding."""
    # TODO: Don't just use all LOINC codes, but restrict to subset of important ones (or actually used ones)
    return isLOINC(cc)


def conceptToOMOP(cc: CodeableConcept) -> str:
    """Converts a CodeableConcept to an OMOP Cohort Definition ID."""
    if isCondition(cc):
        pass  # return conditionToOMOP(cc)
    elif isAllergy(cc):
        pass  # return allergyToOMOP(cc)
    elif isRadiology(cc):
        pass  # return radiologyToOMOP(cc)
    elif isProcedure(cc):
        pass  # return procedureToOMOP(cc)
    elif isEpisodeOfCare(cc):
        pass  # return episodeOfCareToOMOP(cc)
    elif isVentilationObservable(cc):
        pass  # return ventilationObservableToOMOP(cc)
    elif isLaboratory(cc):
        pass  # return laboratoryToOMOP(cc)
    else:
        raise Exception(f"Unknown concept: {cc}")

    return "NotImplemented"
