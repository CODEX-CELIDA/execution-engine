from typing import Type

from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.omop.criterion.concept import ConceptCriterion
from execution_engine.omop.criterion.condition_occurrence import ConditionOccurrence
from execution_engine.omop.criterion.custom import TidalVolumePerIdealBodyWeight
from execution_engine.omop.criterion.drug_exposure import DrugExposure
from execution_engine.omop.criterion.measurement import Measurement
from execution_engine.omop.criterion.observation import Observation
from execution_engine.omop.criterion.procedure_occurrence import ProcedureOccurrence
from execution_engine.omop.criterion.visit_occurrence import (
    ActivePatients,
    VisitOccurrence,
)

__all__ = ["criterion_factory"]


def criterion_factory(class_name: str, data: dict) -> Criterion | CriterionCombination:
    """
    Create a criterion from a dictionary representation.

    Parameters
    ----------
    class_name : str
        The name of the criterion class.
    data : dict
        The dictionary representation of the criterion.

    Returns
    -------
    Criterion | CriterionCombination
        The criterion.

    Raises
    ------
    ValueError
        If the class name is not recognized.
    """

    class_map: dict[str, Type[Criterion] | Type[CriterionCombination]] = {
        "ConceptCriterion": ConceptCriterion,
        "CriterionCombination": CriterionCombination,
        "ConditionOccurrence": ConditionOccurrence,
        "DrugExposure": DrugExposure,
        "Measurement": Measurement,
        "Observation": Observation,
        "ProcedureOccurrence": ProcedureOccurrence,
        "VisitOccurrence": VisitOccurrence,
        "ActivePatients": ActivePatients,
        "TidalVolumePerIdealBodyWeight": TidalVolumePerIdealBodyWeight,
    }

    """Create a criterion from a dictionary."""
    if class_name not in class_map:
        raise ValueError(f"Unknown criterion class {class_name}")

    return class_map[class_name].from_dict(data)