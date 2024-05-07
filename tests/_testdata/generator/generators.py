from datetime import timedelta

from execution_engine.omop.criterion.custom import (
    TidalVolumePerIdealBodyWeight as TVPIBW,
)
from execution_engine.omop.db.omop.tables import Measurement, VisitOccurrence
from execution_engine.util.interval import IntervalType
from execution_engine.util.value import ValueNumber
from tests._fixtures import concept
from tests._testdata import concepts
from tests._testdata.generator.data_generator import (
    ConditionGenerator,
    DrugExposureGenerator,
    MeasurementGenerator,
    ObservationGenerator,
    ProcedureGenerator,
    VisitGenerator,
)


class COVID19(ConditionGenerator):
    name = "COVID19"
    concept_id = concepts.COVID19
    static = True


class VenousThrombosis(ConditionGenerator):
    name = "VENOUS_THROMBOSIS"
    concept_id = concepts.VENOUS_THROMBOSIS
    static = True


class HIT2(ConditionGenerator):
    name = "HIT2"
    concept_id = concepts.HEPARIN_INDUCED_THROMBOCYTOPENIA_WITH_THROMBOSIS
    static = True


class HeparinAllergy(ObservationGenerator):
    name = "HEPARIN_ALLERGY"
    concept_id = concepts.ALLERGY_HEPARIN
    static = True
    missing_data_type = (
        IntervalType.NEGATIVE
    )  # is stored in observation_occurrence, but resembles condition (see AllergyCharacteristic)


class HeparinoidAllergy(ObservationGenerator):
    name = "HEPARINOID_ALLERGY"
    concept_id = concepts.ALLERGY_HEPARINOID
    static = True
    missing_data_type = (
        IntervalType.NEGATIVE
    )  # is stored in observation_occurrence, but resembles condition (see AllergyCharacteristic)


class Thrombocytopenia(ConditionGenerator):
    name = "THROMBOCYTOPENIA"
    concept_id = concepts.THROMBOCYTOPENIA
    static = True


class PulmonaryEmbolism(ConditionGenerator):
    name = "PULMONARY_EMBOLISM"
    concept_id = concepts.PULMONARY_EMBOLISM
    static = True


class ARDS(ConditionGenerator):
    name = "ARDS"
    concept_id = concepts.ARDS
    static = True


class AtrialFibrillation(ConditionGenerator):
    name = "ATRIAL_FIBRILLATION"
    concept_id = concepts.ATRIAL_FIBRILLATION
    static = True


class Weight(MeasurementGenerator):
    name = "WEIGHT"
    concept_id = concepts.BODY_WEIGHT
    static = True
    value = ValueNumber(value=70, unit=concept.concept_unit_kg)


class HeightByIdealBodyWeight(MeasurementGenerator):
    name = "HEIGHT"
    concept_id = concepts.BODY_HEIGHT
    comparator = "="
    static = True

    def __init__(self, ideal_body_weight: float):
        self.ideal_body_weight = ideal_body_weight
        self.value = ValueNumber(
            value=TVPIBW.height_for_predicted_body_weight_ardsnet("female", 70),
            unit=concept.concept_unit_cm,
        )

        super().__init__()


class Dalteparin(DrugExposureGenerator):
    name = "DALTEPARIN"
    concept_id = concepts.DALTEPARIN
    route_concept_id = concepts.ROUTE_SUBCUTANEOUS


class Enoxaparin(DrugExposureGenerator):
    name = "ENOXAPARIN"
    concept_id = concepts.ENOXAPARIN
    route_concept_id = concepts.ROUTE_SUBCUTANEOUS


class Nadroparin(DrugExposureGenerator):
    name = "NADROPARIN"
    concept_id = concepts.NADROPARIN
    route_concept_id = concepts.ROUTE_SUBCUTANEOUS


class Certoparin(DrugExposureGenerator):
    name = "CERTOPARIN"
    concept_id = concepts.CERTOPARIN
    route_concept_id = concepts.ROUTE_SUBCUTANEOUS


class Tinzaparin(DrugExposureGenerator):
    name = "TINZAPARIN"
    concept_id = concepts.TINZAPARIN
    route_concept_id = concepts.ROUTE_SUBCUTANEOUS


class HeparinSubcutaneous(DrugExposureGenerator):
    name = "HEPARIN_SUBCUTANEOUS"
    concept_id = concepts.HEPARIN
    route_concept_id = concepts.ROUTE_SUBCUTANEOUS


class Fondaparinux(DrugExposureGenerator):
    name = "FONDAPARINUX"
    concept_id = concepts.FONDAPARINUX
    route_concept_id = concepts.ROUTE_SUBCUTANEOUS


class Argatroban(DrugExposureGenerator):
    name = "ARGATROBAN"
    concept_id = concepts.ARGATROBAN
    route_concept_id = concepts.ROUTE_INTRAVENOUS


class HeparinIntravenous(DrugExposureGenerator):
    name = "HEPARIN"
    concept_id = concepts.HEPARIN
    route_concept_id = concepts.ROUTE_INTRAVENOUS


class EnoxaparinTherapeutic(DrugExposureGenerator):
    name = "ENOXAPARIN_THERAPEUTIC"
    concept_id = concepts.ENOXAPARIN
    route_concept_id = concepts.ROUTE_SUBCUTANEOUS


class IntensiveCare(VisitGenerator):
    name = "INTENSIVE_CARE"
    concept_id = concepts.INTENSIVE_CARE
    invalid_concept_id = concepts.INPATIENT_VISIT


class Proning(ProcedureGenerator):
    name = "PRONING"
    concept_id = concepts.PRONE_POSITIONING
    duration_threshold_hours = 16


class Ventilated(ProcedureGenerator):
    name = "VENTILATED"
    concept_id = concepts.ARTIFICIAL_RESPIRATION


class DDimer(MeasurementGenerator):
    name = "D_DIMER"
    concept_id = concepts.LAB_DDIMER
    value = ValueNumber(value=2, unit=concept.concept_unit_ug_l)
    comparator = ">="


class APTT(MeasurementGenerator):
    name = "APTT"
    concept_id = concepts.LAB_APTT
    value = ValueNumber(value=50, unit=concept.concept_unit_sec)
    comparator = ">"


class TidalVolume(MeasurementGenerator):
    name = "TIDAL_VOLUME"
    concept_id = concepts.TIDAL_VOLUME
    comparator = "<"

    def __init__(self, weight: float):
        self.weight = weight
        self.value = ValueNumber(value=6 * weight, unit=concept.concept_unit_ml)

        super().__init__()


class PPlateau(MeasurementGenerator):
    name = "P_PLATEAU"
    concept_id = concepts.PRESSURE_PLATEAU
    value = ValueNumber(value=30, unit=concept.concept_unit_cm_h2o)
    comparator = "<="


class FiO2Base(MeasurementGenerator):
    concept_id = concepts.INHALED_OXYGEN_CONCENTRATION
    # value = ValueNumber(value=0.6, unit=concept.concept_unit_percent)
    comparator = "="

    def generate_data(
        self, vo: VisitOccurrence, valid: bool = True
    ) -> list[Measurement]:
        """
        Generate FiO2 measurements for the given visit occurrence.

        If valid == False, we must not generate any data, as invalid data for one FiO2 category may be a valid
        value for another FiO2 category. E.g. the invalid value for FiO2_30 = 30-39% is 29%, and this is the valid
        value for FiO2_20 = 20-29%. This messes up the expected results in the test.
        """
        if valid:
            return super().generate_data(vo, valid)
        else:
            return []


class FiO2_30(FiO2Base):
    value = ValueNumber(value=30, unit=concept.concept_unit_percent)
    start_datetime = FiO2Base.start_datetime + timedelta(hours=1)


class FiO2_40(FiO2Base):
    value = ValueNumber(value=40, unit=concept.concept_unit_percent)
    start_datetime = FiO2Base.start_datetime + timedelta(hours=2)


class FiO2_50(FiO2Base):
    value = ValueNumber(value=50, unit=concept.concept_unit_percent)
    start_datetime = FiO2Base.start_datetime + timedelta(hours=3)


class FiO2_60(FiO2Base):
    value = ValueNumber(value=60, unit=concept.concept_unit_percent)
    start_datetime = FiO2Base.start_datetime + timedelta(hours=4)


class FiO2_70(FiO2Base):
    value = ValueNumber(value=70, unit=concept.concept_unit_percent)
    start_datetime = FiO2Base.start_datetime + timedelta(hours=5)


class FiO2_80(FiO2Base):
    value = ValueNumber(value=80, unit=concept.concept_unit_percent)
    start_datetime = FiO2Base.start_datetime + timedelta(hours=6)


class FiO2_90(FiO2Base):
    value = ValueNumber(value=90, unit=concept.concept_unit_percent)
    start_datetime = FiO2Base.start_datetime + timedelta(hours=7)


class FiO2_100(FiO2Base):
    value = ValueNumber(value=100, unit=concept.concept_unit_percent)
    start_datetime = FiO2Base.start_datetime + timedelta(hours=8)


class PEEPBase(MeasurementGenerator):
    concept_id = concepts.PEEP
    comparator = ">"

    def generate_data(
        self, vo: VisitOccurrence, valid: bool = True
    ) -> list[Measurement]:
        """
        Generate PEEP measurements for the given visit occurrence.

        If valid == False, we must not generate any data, as invalid data for one PEEP category may be a valid
        value for another PEEP category. This messes up the expected results in the test.
        """

        if valid:
            return super().generate_data(vo, valid)
        else:
            return []


class PEEP_5(PEEPBase):
    value = ValueNumber(value=5, unit=concept.concept_unit_cm_h2o)
    start_datetime = [FiO2_30.start_datetime, FiO2_40.start_datetime]


class PEEP_8(PEEPBase):
    value = ValueNumber(value=8, unit=concept.concept_unit_cm_h2o)
    start_datetime = FiO2_50.start_datetime


class PEEP_10(PEEPBase):
    value = ValueNumber(value=10, unit=concept.concept_unit_cm_h2o)
    start_datetime = [FiO2_60.start_datetime, FiO2_70.start_datetime]


class PEEP_14(PEEPBase):
    value = ValueNumber(value=14, unit=concept.concept_unit_cm_h2o)
    start_datetime = [FiO2_80.start_datetime, FiO2_90.start_datetime]


class PEEP_18(PEEPBase):
    value = ValueNumber(value=18, unit=concept.concept_unit_cm_h2o)
    start_datetime = FiO2_100.start_datetime


class OxygenationIndex(MeasurementGenerator):
    concept_id = concepts.LAB_HOROWITZ
    value = ValueNumber(value=150, unit=concept.concept_unit_mm_hg)
    comparator = "<"


Weight40kg = Weight(
    value=ValueNumber(value=40, unit=concept.concept_unit_kg), comparator="="
)


Weight50kg = Weight(
    value=ValueNumber(value=50, unit=concept.concept_unit_kg), comparator="="
)

Weight57kg = Weight(
    value=ValueNumber(value=57, unit=concept.concept_unit_kg), comparator="="
)

Weight65kg = Weight(
    value=ValueNumber(value=65, unit=concept.concept_unit_kg), comparator="="
)

Weight70kg = Weight(
    value=ValueNumber(value=70, unit=concept.concept_unit_kg), comparator="="
)

Weight82kg = Weight(
    value=ValueNumber(value=82, unit=concept.concept_unit_kg), comparator="="
)

Weight83kg = Weight(
    value=ValueNumber(value=83, unit=concept.concept_unit_kg), comparator="="
)

Weight92kg = Weight(
    value=ValueNumber(value=92, unit=concept.concept_unit_kg), comparator="="
)

Weight110kg = Weight(
    value=ValueNumber(value=110, unit=concept.concept_unit_kg), comparator="="
)

# Prophylactic dosages

Dalteparin2500ie1xd = Dalteparin(
    quantity=ValueNumber(value=2500, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=1,
)

Dalteparin5000ie1xd = Dalteparin(
    quantity=ValueNumber(value=5000, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=1,
)


NadroparinProphylactic3800ie1xd = Nadroparin(
    quantity=ValueNumber(value=3800, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=1,
)

NadroparinProphylactic5700ie1xd = Nadroparin(
    quantity=ValueNumber(value=5700, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=1,
)

Certoparin3000ie1xd = Certoparin(
    quantity=ValueNumber(value=3000, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=1,
)

Tinzaparin3500ie1xd = Tinzaparin(
    quantity=ValueNumber(value=3500, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=1,
)

HeparinSubcutaneous5000ie2xd = HeparinSubcutaneous(
    quantity=ValueNumber(value=5000, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=2,
)

HeparinSubcutaneous5000ie3xd = HeparinSubcutaneous(
    quantity=ValueNumber(value=5000, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=3,
)

HeparinSubcutaneous75002xd = HeparinSubcutaneous(
    quantity=ValueNumber(value=7500, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=2,
)

Fondaparinux2_5mg1xd = Fondaparinux(
    quantity=ValueNumber(value=2.5, unit=concept.concept_unit_mg),
    comparator="=",
    doses_per_day=1,
)


# Therapeutic dosages

Dalteparin200ie70kg1xd = Dalteparin(
    quantity=ValueNumber(
        value=Weight70kg.value.value * 200, unit=concept.concept_unit_ie
    ),
    comparator="=",
    doses_per_day=1,
)
Dalteparin100ie70kg2xd = Dalteparin(
    quantity=ValueNumber(
        value=Weight70kg.value.value * 100, unit=concept.concept_unit_ie
    ),
    comparator="=",
    doses_per_day=2,
)

Dalteparin10000ie50kg1xd = Dalteparin(
    quantity=ValueNumber(value=10000, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=1,
)

Dalteparin12500ie57kg1xd = Dalteparin(
    quantity=ValueNumber(value=12500, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=1,
)

Dalteparin15000ie82kg1xd = Dalteparin(
    quantity=ValueNumber(value=15000, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=1,
)

Dalteparin18000ie83kg1xd = Dalteparin(
    quantity=ValueNumber(value=18000, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=1,
)

Enoxaparin70kg1xd = Enoxaparin(
    quantity=ValueNumber(
        value=Weight70kg.value.value * 1.5, unit=concept.concept_unit_mg
    ),
    comparator="=",
    doses_per_day=1,
)
Enoxaparin70kg2xd = Enoxaparin(
    quantity=ValueNumber(
        value=Weight70kg.value.value * 1, unit=concept.concept_unit_mg
    ),
    comparator="=",
    doses_per_day=2,
)

Nadroparinlt50kg2xd = Nadroparin(
    quantity=ValueNumber(value=3800, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=2,
)

Nadroparin50_59kg2xd = Nadroparin(
    quantity=ValueNumber(value=4750, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=2,
)

Nadroparin60_69kg2xd = Nadroparin(
    quantity=ValueNumber(value=5700, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=2,
)

Nadroparin70_79kg2xd = Nadroparin(
    quantity=ValueNumber(value=6650, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=2,
)

Nadroparin80_89kg2xd = Nadroparin(
    quantity=ValueNumber(value=7600, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=2,
)

Nadroparingt90kg2xd = Nadroparin(
    quantity=ValueNumber(value=8550, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=2,
)

CertoparinTherapeutic = Certoparin(
    quantity=ValueNumber(value=8000, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=2,
)

TinzaparinTherapeutic70kg1xd = Tinzaparin(
    quantity=ValueNumber(
        value=Weight70kg.value.value * 175, unit=concept.concept_unit_ie
    ),
    comparator="=",
    doses_per_day=1,
)

Fondaparinux_lt50kg1xd = Fondaparinux(
    quantity=ValueNumber(value=5, unit=concept.concept_unit_mg),
    comparator="=",
    doses_per_day=1,
)

Fondaparinux50_100kg1xd = Fondaparinux(
    quantity=ValueNumber(value=7.5, unit=concept.concept_unit_mg),
    comparator="=",
    doses_per_day=1,
)

Fondaparinuxgt100kg1xd = Fondaparinux(
    quantity=ValueNumber(value=10, unit=concept.concept_unit_mg),
    comparator="=",
    doses_per_day=1,
)

HeparinIV100ie = HeparinIntravenous(
    quantity=ValueNumber(value=100, unit=concept.concept_unit_ie),
    comparator="=",
    doses_per_day=1,
)

Argatroban100mg = Argatroban(
    quantity=ValueNumber(value=100, unit=concept.concept_unit_mg),
    comparator="=",
    doses_per_day=1,
)
