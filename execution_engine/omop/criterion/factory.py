from typing import Type

from execution_engine.omop.criterion.condition_occurrence import ConditionOccurrence
from execution_engine.omop.criterion.custom import TidalVolumePerIdealBodyWeight
from execution_engine.omop.criterion.drug_exposure import DrugExposure
from execution_engine.omop.criterion.measurement import Measurement
from execution_engine.omop.criterion.observation import Observation
from execution_engine.omop.criterion.point_in_time import PointInTimeCriterion
from execution_engine.omop.criterion.procedure_occurrence import ProcedureOccurrence
from execution_engine.omop.criterion.visit_detail import VisitDetail
from execution_engine.omop.criterion.visit_occurrence import (
    ActivePatients,
    PatientsActiveDuringPeriod,
    VisitOccurrence,
)

__all__ = ["criterion_factory", "register_criterion_class"]

from execution_engine.util import logic

class_map: dict[str, Type[logic.BaseExpr]] = {
    # "ConceptCriterion": ConceptCriterion, # is an Abstract class
    "LogicalCriterionCombination": logic.BooleanFunction,
    "TemporalCriterionCombination": logic.BooleanFunction,
    "NonCommutativeLogicalCriterionCombination": logic.BooleanFunction,
    "ConditionOccurrence": ConditionOccurrence,
    "DrugExposure": DrugExposure,
    "Measurement": Measurement,
    "Observation": Observation,
    "ProcedureOccurrence": ProcedureOccurrence,
    "VisitOccurrence": VisitOccurrence,
    "ActivePatients": ActivePatients,
    "PatientsActiveDuringPeriod": PatientsActiveDuringPeriod,
    "TidalVolumePerIdealBodyWeight": TidalVolumePerIdealBodyWeight,
    "VisitDetail": VisitDetail,
    "PointInTimeCriterion": PointInTimeCriterion,
    "PersonalWindowTemporalIndicatorCombination": logic.BooleanFunction,
}


def register_criterion_class(
    class_name: str,
    criterion_class: Type[logic.BaseExpr],
) -> None:
    """
    Register a criterion class.

    :param class_name: The name of the criterion class.
    :param criterion_class: The criterion class.
    """
    class_map[class_name] = criterion_class


def criterion_factory(class_name: str, data: dict) -> logic.BaseExpr:
    """
    Create a criterion from a dictionary representation.

    :param class_name: The name of the criterion class.
    :param data: The dictionary representation of the criterion.
    :return: The criterion.
    :raises ValueError: If the class name is not recognized.
    """

    if class_name not in class_map:
        raise ValueError(f"Unknown criterion class {class_name}")

    return class_map[class_name].from_dict(data)
