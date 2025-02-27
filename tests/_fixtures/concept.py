import pytest

from execution_engine.constants import OMOPConcepts
from execution_engine.omop.concepts import Concept
from tests._testdata import concepts


@pytest.fixture
def test_concept():
    return Concept(
        concept_id=1,
        concept_name="Test Concept",
        concept_code="unit",
        domain_id="units",
        vocabulary_id="test",
        concept_class_id="test",
    )


@pytest.fixture
def unit_concept():
    return Concept(
        concept_id=1,
        concept_name="Test Unit",
        concept_code="unit",
        domain_id="units",
        vocabulary_id="test",
        concept_class_id="test",
    )


concept_artificial_respiration = Concept(
    concept_id=4230167,
    concept_name="Artificial respiration",
    domain_id="Procedure",
    vocabulary_id="SNOMED",
    concept_class_id="Procedure",
    standard_concept="S",
    concept_code="40617009",
    invalid_reason=None,
)

concept_surgical_procedure = Concept(
    concept_id=4301351,
    concept_name="Surgical procedure",
    domain_id="Procedure",
    vocabulary_id="SNOMED",
    concept_class_id="Procedure",
    standard_concept="S",
    concept_code="387713003",
    invalid_reason=None,
)

concept_delir_screening = Concept( # TODO(jmoringe): copied from above (concept_artificial_respiration)
    concept_id=4230167,
    concept_name="Delir Screening",
    domain_id="Procedure",
    vocabulary_id="SNOMED",
    concept_class_id="Procedure",
    standard_concept="S",
    concept_code="40617009",
    invalid_reason=None,
)

concept_covid19 = Concept(
    concept_id=37311061,
    concept_name="COVID-19",
    domain_id="Condition",
    vocabulary_id="SNOMED",
    concept_class_id="Clinical Finding",
    standard_concept="S",
    concept_code="840539006",
    invalid_reason=None,
)

concept_unit_mg = Concept(
    concept_id=8576,
    concept_name="milligram",
    domain_id="Unit",
    vocabulary_id="UCUM",
    concept_class_id="Unit",
    standard_concept="S",
    concept_code="mg",
    invalid_reason=None,
)

concept_unit_mg_kg = Concept(
    concept_id=OMOPConcepts.UNIT_MG_PER_KG.value,
    concept_name="milligram per kilogram",
    domain_id="Unit",
    vocabulary_id="UCUM",
    concept_class_id="Unit",
    standard_concept="S",
    concept_code="mg/kg",
    invalid_reason=None,
)

concept_unit_hour = Concept(
    concept_id=8505,
    concept_name="hour",
    domain_id="Unit",
    vocabulary_id="UCUM",
    concept_class_id="Unit",
    standard_concept="S",
    concept_code="h",
    invalid_reason=None,
)

concept_unit_sec = Concept(
    concept_id=concepts.UNIT_SECOND,
    concept_name="second",
    domain_id="Unit",
    vocabulary_id="UCUM",
    concept_class_id="Unit",
    standard_concept="S",
    concept_code="s",
    invalid_reason=None,
)

concept_unit_percent = Concept(
    concept_id=concepts.UNIT_PERCENT,
    concept_name="percent",
    domain_id="Unit",
    vocabulary_id="UCUM",
    concept_class_id="Unit",
    standard_concept="S",
    concept_code="%",
    invalid_reason=None,
)

concept_unit_mg_l = Concept(
    concept_id=concepts.UNIT_MG_PER_L,
    concept_name="milligram per liter",
    domain_id="Unit",
    vocabulary_id="UCUM",
    concept_class_id="Unit",
    standard_concept="S",
    concept_code="mg/L",
    invalid_reason=None,
)

concept_unit_ug_l = Concept(
    concept_id=concepts.UNIT_UG_PER_L,
    concept_name="microgram per liter",
    domain_id="Unit",
    vocabulary_id="UCUM",
    concept_class_id="Unit",
    standard_concept="S",
    concept_code="ug/L",
    invalid_reason=None,
)

concept_unit_cm = Concept(
    concept_id=concepts.UNIT_CM,
    concept_name="centimeter",
    domain_id="Unit",
    vocabulary_id="UCUM",
    concept_class_id="Unit",
    standard_concept="S",
    concept_code="cm",
    invalid_reason=None,
)

concept_unit_ml = Concept(
    concept_id=concepts.UNIT_ML,
    concept_name="milliliter",
    domain_id="Unit",
    vocabulary_id="UCUM",
    concept_class_id="Unit",
    standard_concept="S",
    concept_code="mL",
    invalid_reason=None,
)

concept_unit_cm_h2o = Concept(
    concept_id=concepts.UNIT_CM_H2O,
    concept_name="centimeter of water",
    domain_id="Unit",
    vocabulary_id="UCUM",
    concept_class_id="Unit",
    standard_concept="S",
    concept_code="cm[H2O]",
    invalid_reason=None,
)

concept_unit_kg = Concept(
    concept_id=concepts.UNIT_KG,
    concept_name="kilogram",
    domain_id="Unit",
    vocabulary_id="UCUM",
    concept_class_id="Unit",
    standard_concept="S",
    concept_code="kg",
    invalid_reason=None,
)

concept_unit_ie = Concept(
    concept_id=concepts.UNIT_IE,
    concept_name="international unit",
    domain_id="Unit",
    vocabulary_id="UCUM",
    concept_class_id="Unit",
    standard_concept="S",
    concept_code="IU",
    invalid_reason=None,
)

concept_unit_mm_hg = Concept(
    concept_id=concepts.UNIT_MM_HG,
    concept_name="millimeter of mercury",
    domain_id="Unit",
    vocabulary_id="UCUM",
    concept_class_id="Unit",
    standard_concept="S",
    concept_code="mm[Hg]",
    invalid_reason=None,
)

concept_heparin_ingredient = Concept(
    concept_id=1367571,
    concept_name="heparin",
    domain_id="Drug",
    vocabulary_id="RxNorm",
    concept_class_id="Ingredient",
    standard_concept="S",
    concept_code="5224",
    invalid_reason=None,
)


concept_route_subcutaneous = Concept(
    concept_id=4142048,
    concept_name="Subcutaneous route",
    domain_id="Route",
    vocabulary_id="SNOMED",
    concept_class_id="Qualifier Value",
    standard_concept="S",
    concept_code="34206005",
    invalid_reason=None,
)

concept_route_intravenous = Concept(
    concept_id=4171047,
    concept_name="Intravenous route",
    domain_id="Route",
    vocabulary_id="SNOMED",
    concept_class_id="Qualifier Value",
    standard_concept="S",
    concept_code="47625008",
    invalid_reason=None,
)

concept_heparin_allergy = Concept(
    concept_id=4169185,
    concept_name="Allergy to heparin",
    domain_id="Observation",
    vocabulary_id="SNOMED",
    concept_class_id="Clinical Finding",
    standard_concept="S",
    concept_code="294872001",
    invalid_reason=None,
)

concept_tidal_volume = Concept(
    concept_id=21490854,
    concept_name="Tidal volume Ventilator --on ventilator",
    domain_id="Measurement",
    vocabulary_id="LOINC",
    concept_class_id="Clinical Observation",
    standard_concept="S",
    concept_code="76222-9",
    invalid_reason=None,
)

concept_body_weight = Concept(
    concept_id=3025315,
    concept_name="Body weight",
    domain_id="Measurement",
    vocabulary_id="LOINC",
    concept_class_id="Clinical Observation",
    standard_concept="S",
    concept_code="29463-7",
    invalid_reason=None,
)

"""
The following list of concepts are heparin drugs and all of them directly map to heparin as ingredient (via ancestor,
not relationship !)
"""
concepts_heparin_other = [
    Concept(
        concept_id=40894655,
        concept_name="heparin Topical Gel [Thrombocutan]",
        domain_id="Drug",
        vocabulary_id="RxNorm Extension",
        concept_class_id="Branded Drug Form",
        standard_concept="S",
        concept_code="OMOP2092617",
        invalid_reason=None,
    ),
    Concept(
        concept_id=40988159,
        concept_name="heparin Injectable Solution [Vetren]",
        domain_id="Drug",
        vocabulary_id="RxNorm Extension",
        concept_class_id="Branded Drug Form",
        standard_concept="S",
        concept_code="OMOP2186121",
        invalid_reason=None,
    ),
    Concept(
        concept_id=41269112,
        concept_name="heparin Stick [Vetren]",
        domain_id="Drug",
        vocabulary_id="RxNorm Extension",
        concept_class_id="Branded Drug Form",
        standard_concept="S",
        concept_code="OMOP2467074",
        invalid_reason=None,
    ),
    Concept(
        concept_id=44188826,
        concept_name="Dihydroergotamine 0.5 MG / heparin 5000 UNT Prefilled Syringe [Heparin Dihydergot] Box of 20",
        domain_id="Drug",
        vocabulary_id="RxNorm Extension",
        concept_class_id="Branded Drug Box",
        standard_concept="S",
        concept_code="OMOP3065549",
        invalid_reason=None,
    ),
]


"""
The following list of concepts are heparin drugs, but not all of them have a "maps to" ingredient.
"""
concepts_heparin_other_including_non_ingredient_related = [
    Concept(
        concept_id=995426,
        concept_name="101000 MG heparin 0.6 UNT/MG Topical Gel by Axicorp",
        domain_id="Drug",
        vocabulary_id="RxNorm Extension",
        concept_class_id="Marketed Product",
        standard_concept="S",
        concept_code="OMOP4821932",
        invalid_reason=None,
    ),
    Concept(
        concept_id=1367697,
        concept_name="heparin calcium 25000 UNT/ML",
        domain_id="Drug",
        vocabulary_id="RxNorm",
        concept_class_id="Clinical Drug Comp",
        standard_concept="S",
        concept_code="849698",
        invalid_reason=None,
    ),
    Concept(
        concept_id=44216409,
        concept_name="200000 MG heparin 1.8 UNT/MG Topical Gel [Heparin Ratiopharm]",
        domain_id="Drug",
        vocabulary_id="RxNorm Extension",
        concept_class_id="Quant Branded Drug",
        standard_concept="S",
        concept_code="OMOP3093132",
        invalid_reason=None,
    ),
    Concept(
        concept_id=44507578,
        concept_name="6 ML heparin sodium, porcine 100 UNT/ML Prefilled Syringe",
        domain_id="Drug",
        vocabulary_id="RxNorm",
        concept_class_id="Quant Clinical Drug",
        standard_concept="S",
        concept_code="1442414",
        invalid_reason=None,
    ),
    Concept(
        concept_id=44215905,
        concept_name="101000 MG Arnica extract 0.1 MG/MG / guaiazulene 0.00005 MG/MG / heparin 0.04 UNT/MG / Lecithin 0.01 MG/MG / Matricaria chamomilla flowering top oil 0.00005 MG/MG Topical Gel [Arnica Kneipp]",
        domain_id="Drug",
        vocabulary_id="RxNorm Extension",
        concept_class_id="Quant Branded Drug",
        standard_concept="S",
        concept_code="OMOP3092628",
        invalid_reason=None,
    ),
]

concept_enoxparin = Concept(
    concept_id=995271,
    concept_name="0.4 ML Enoxaparin 100 MG/ML Injectable Solution [Inhixa] by Emra-Med",
    domain_id="Drug",
    vocabulary_id="RxNorm Extension",
    concept_class_id="Marketed Product",
    standard_concept="S",
    concept_code="OMOP4821780",
    invalid_reason=None,
)

concept_enoxparin_ingredient = Concept(
    concept_id=1301025,
    concept_name="enoxaparin",
    domain_id="Drug",
    vocabulary_id="RxNorm",
    concept_class_id="Ingredient",
    standard_concept="S",
    concept_code="67108",
    invalid_reason=None,
)
