from execution_engine.characteristic import (
    AbstractCharacteristic,
    CharacteristicCombination,
)

from .omop.criterion.abstract import AbstractCriterion
from .omop.criterion.combination import CriterionCombination


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


def characteristic_to_criterion(
    characteristic: AbstractCharacteristic | CharacteristicCombination,
) -> AbstractCriterion:
    """
    Convert a characteristic to a criterion.
    """
    if isinstance(characteristic, CharacteristicCombination):
        operator = characteristic_code_to_criterion_combination_operator(
            characteristic.code, characteristic.threshold
        )
        comb = CriterionCombination(
            name="...", exclude=characteristic.exclude, operator=operator
        )
        for c in characteristic:
            comb.add(characteristic_to_criterion(c))
        return comb
    else:
        return characteristic.to_criterion()


class ActionSelectionBehavior:
    """Mapping from FHIR PlanDefinition.action.selectionBehavior to OMOP InclusionRule Type/Count."""

    _map = {
        "any": CharacteristicCombination.Code.ANY_OF,
        "all": CharacteristicCombination.Code.ALL_OF,
        "all-or-none": None,
        "exactly-one": None,
        "at-most-one": CharacteristicCombination.Code.AT_MOST,
        "one-or-more": CharacteristicCombination.Code.AT_LEAST,
    }

    def __init__(self, behavior: str) -> None:
        if behavior not in self._map:
            raise ValueError(f"Invalid action selection behavior: {behavior}")
        elif self._map[behavior] is None:
            raise ValueError(f"Unsupported action selection behavior: {behavior}")

        self._behavior = behavior

    @property
    def code(self) -> CharacteristicCombination.Code:
        """
        Get the characteristic combination code.
        """

        if self._map[self._behavior] is None:
            raise ValueError(f"Unsupported action selection behavior: {self._behavior}")

        return self._map[self._behavior]  # type: ignore
