from pydantic import BaseModel

from tests._testdata import concepts


class CriterionDefinition(BaseModel):
    name: str
    type: str
    static: bool
    concept_id: int
    threshold: float | None = None
    dosage: float | None = None
    dosage_threshold: float | None = None
    doses_per_day: int | None = None
    unit_concept_id: int | None = None
    n_occurrences: list[int] | None = None
    occurrences_per_day: list[int] | None = None


COVID19 = CriterionDefinition(
    name="COVID19", type="condition", static=True, concept_id=concepts.COVID19
)
VENOUS_THROMBOSIS = CriterionDefinition(
    name="VENOUS_THROMBOSIS",
    type="condition",
    static=True,
    concept_id=concepts.VENOUS_THROMBOSIS,
)
HIT2 = CriterionDefinition(
    name="HIT2",
    type="condition",
    static=True,
    concept_id=concepts.HEPARIN_INDUCED_THROMBOCYTOPENIA_WITH_THROMBOSIS,
)
HEPARIN_ALLERGY = CriterionDefinition(
    name="HEPARIN_ALLERGY",
    type="observation",
    static=True,
    concept_id=concepts.ALLERGY_HEPARIN,
)
HEPARINOID_ALLERGY = CriterionDefinition(
    name="HEPARINOID_ALLERGY",
    type="observation",
    static=True,
    concept_id=concepts.ALLERGY_HEPARINOID,
)
THROMBOCYTOPENIA = CriterionDefinition(
    name="THROMBOCYTOPENIA",
    type="condition",
    static=True,
    concept_id=concepts.THROMBOCYTOPENIA,
)
PULMONARY_EMBOLISM = CriterionDefinition(
    name="PULMONARY_EMBOLISM",
    type="condition",
    static=True,
    concept_id=concepts.PULMONARY_EMBOLISM,
)
ARDS = CriterionDefinition(
    name="ARDS", type="condition", static=True, concept_id=concepts.ARDS
)

WEIGHT = CriterionDefinition(
    name="WEIGHT",
    type="observation",
    static=True,
    threshold=70,
    unit_concept_id=concepts.UNIT_KG,
    concept_id=concepts.WEIGHT,
)
DALTEPARIN = CriterionDefinition(
    name="DALTEPARIN",
    type="drug",
    dosage_threshold=5000,
    doses_per_day=1,
    static=False,
    concept_id=concepts.DALTEPARIN,
)
ENOXAPARIN = CriterionDefinition(
    name="ENOXAPARIN",
    type="drug",
    dosage_threshold=40,
    doses_per_day=1,
    static=False,
    concept_id=concepts.ENOXAPARIN,
)
NADROPARIN_LOW_WEIGHT = CriterionDefinition(
    name="NADROPARIN_LOW_WEIGHT",
    type="drug",
    dosage_threshold=3800,
    doses_per_day=1,
    static=False,
    concept_id=concepts.NADROPARIN,
)
NADROPARIN_HIGH_WEIGHT = CriterionDefinition(
    name="NADROPARIN_HIGH_WEIGHT",
    type="drug",
    dosage_threshold=5700,
    doses_per_day=1,
    static=False,
    concept_id=concepts.NADROPARIN,
)
CERTOPARIN = CriterionDefinition(
    name="CERTOPARIN",
    type="drug",
    dosage_threshold=3000,
    doses_per_day=1,
    static=False,
    concept_id=concepts.CERTOPARIN,
)
FONDAPARINUX_PROPHYLACTIC = CriterionDefinition(
    name="FONDAPARINUX_PROPHYLACTIC",
    type="drug",
    dosage=2.5,
    doses_per_day=1,
    static=False,
    concept_id=concepts.FONDAPARINUX,
)
FONDAPARINUX_THERAPEUTIC = CriterionDefinition(
    name="FONDAPARINUX_THERAPEUTIC",
    type="drug",
    dosage=2.5,
    doses_per_day=2,
    static=False,
    concept_id=concepts.FONDAPARINUX,
)
HEPARIN = CriterionDefinition(
    name="HEPARIN",
    type="drug",
    dosage=1,
    doses_per_day=1,
    static=False,
    concept_id=concepts.HEPARIN,
)
ARGATROBAN = CriterionDefinition(
    name="ARGATROBAN",
    type="drug",
    dosage=1,
    doses_per_day=1,
    static=False,
    concept_id=concepts.ARGATROBAN,
)
ICU = CriterionDefinition(
    name="ICU",
    type="visit",
    n_occurrences=[0, 2],
    static=False,
    concept_id=concepts.INTENSIVE_CARE,
)
PRONING = CriterionDefinition(
    name="PRONING",
    type="procedure",
    n_occurrences=[0, 20],
    static=False,
    concept_id=concepts.PRONE_POSITIONING,
)
VENTILATED = CriterionDefinition(
    name="VENTILATED",
    type="procedure",
    n_occurrences=[0, 20],
    static=False,
    concept_id=concepts.ARTIFICIAL_RESPIRATION,
)
D_DIMER = CriterionDefinition(
    name="D_DIMER",
    type="measurement",
    threshold=2,
    static=False,
    unit_concept_id=concepts.UNIT_MG_PER_L,
    concept_id=concepts.LAB_DDIMER,
)
APTT = CriterionDefinition(
    name="APTT",
    type="measurement",
    threshold=50,
    static=False,
    unit_concept_id=concepts.UNIT_SECOND,
    concept_id=concepts.LAB_APTT,
)
TIDAL_VOLUME = CriterionDefinition(
    name="TIDAL_VOLUME",
    type="measurement",
    threshold=6,
    static=False,
    occurrences_per_day=[6, 30],
    unit_concept_id=concepts.UNIT_ML,
    concept_id=concepts.TIDAL_VOLUME,
)
PMAX = CriterionDefinition(
    name="PMAX",
    type="measurement",
    threshold=30,
    occurrences_per_day=[6, 30],
    static=False,
    unit_concept_id=concepts.UNIT_CM_H2O,
    concept_id=concepts.PRESSURE_MAX,
)
FiO2 = CriterionDefinition(
    name="FiO2",
    type="measurement",
    occurrences_per_day=[6, 30],
    static=False,
    unit_concept_id=concepts.UNIT_PERCENT,
    concept_id=concepts.INHALED_OXYGEN_CONCENTRATION,
)
PEEP = CriterionDefinition(
    name="PEEP",
    type="measurement",
    occurrences_per_day=[6, 30],
    static=False,
    unit_concept_id=concepts.UNIT_CM_H2O,
    concept_id=concepts.PEEP,
)
OXYGENATION_INDEX = CriterionDefinition(
    name="OXYGENATION_INDEX",
    type="measurement",
    threshold=150,
    occurrences_per_day=[6, 30],
    static=False,
    unit_concept_id=concepts.UNIT_MM_HG,
    concept_id=concepts.LAB_HOROWITZ,
)
