from execution_engine.characteristic import CharacteristicCombination
from execution_engine.omop.criterion import CriterionCombination


def characteristic_code_to_criterion_combination_operator(
    code: CharacteristicCombination.Code, threshold: int | None = None
) -> CriterionCombination.Operator:
    """
    Convert a characteristic combination code (from FHIR) to a criterion combination operator (for OMOP).
    """
    mapping = {
        CharacteristicCombination.Code.ALL_OF: CriterionCombination.Operator.AND,
        CharacteristicCombination.Code.ANY_OF: CriterionCombination.Operator.OR,
        CharacteristicCombination.Code.AT_LEAST: CriterionCombination.Operator.AT_LEAST,
        CharacteristicCombination.Code.AT_MOST: CriterionCombination.Operator.AT_MOST,
    }

    if code not in mapping:
        raise NotImplementedError(f"Unknown CharacteristicCombination.Code: {code}")
    return CriterionCombination.Operator(operator=mapping[code], threshold=threshold)
