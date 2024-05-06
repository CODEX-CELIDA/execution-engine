from .constants import CohortCategory
from .converter.characteristic.abstract import AbstractCharacteristic
from .converter.characteristic.combination import CharacteristicCombination
from .omop.criterion.abstract import Criterion
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
        CharacteristicCombination.Code.EXACTLY: CriterionCombination.Operator.EXACTLY,
    }

    if code not in mapping:
        raise NotImplementedError(f"Unknown CharacteristicCombination.Code: {code}")
    return CriterionCombination.Operator(operator=mapping[code], threshold=threshold)


def characteristic_to_criterion(
    characteristic: AbstractCharacteristic | CharacteristicCombination,
) -> Criterion | CriterionCombination:
    """
    Convert a characteristic to a criterion.
    """
    if isinstance(characteristic, CharacteristicCombination):
        operator = characteristic_code_to_criterion_combination_operator(
            characteristic.code, characteristic.threshold
        )
        comb = CriterionCombination(
            category=CohortCategory.POPULATION,
            exclude=characteristic.exclude,
            operator=operator,
        )
        for c in characteristic:
            comb.add(characteristic_to_criterion(c))
        return comb
    else:
        return characteristic.to_criterion()


class ActionSelectionBehavior:
    """Mapping from FHIR PlanDefinition.action.selectionBehavior to OMOP InclusionRule Type/Count."""

    """ If no selection behavior is specified, use this one. """
    _default_behavior = "any"

    _map = {
        "any": {"code": CharacteristicCombination.Code.ANY_OF, "threshold": None},
        "all": {"code": CharacteristicCombination.Code.ALL_OF, "threshold": None},
        "all-or-none": {
            "code": CharacteristicCombination.Code.ALL_OR_NONE,
            "threshold": None,
        },
        "exactly-one": {"code": CharacteristicCombination.Code.EXACTLY, "threshold": 1},
        "at-most-one": {"code": CharacteristicCombination.Code.AT_MOST, "threshold": 1},
        "one-or-more": {
            "code": CharacteristicCombination.Code.AT_LEAST,
            "threshold": 1,
        },
    }

    def __init__(self, behavior: str | None) -> None:
        if behavior is None:
            behavior = self._default_behavior

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

        return self._map[self._behavior]["code"]  # type: ignore

    @property
    def threshold(self) -> int | None:
        """
        Get the threshold for the characteristic combination.
        """

        if self._map[self._behavior] is None:
            raise ValueError(f"Unsupported action selection behavior: {self._behavior}")

        return self._map[self._behavior]["threshold"]  # type: ignore
