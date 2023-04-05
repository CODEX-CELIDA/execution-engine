from enum import Enum
from typing import Iterator, Union

from execution_engine.converter.characteristic.abstract import AbstractCharacteristic


class CharacteristicCombination:
    """Combination of Characteristics"""

    class Code(Enum):
        """
        The code for the combination of characteristics.
        """

        ALL_OF = "all-of"  # all characteristics must be true
        ANY_OF = "any-of"  # at least one characteristic must be true
        AT_LEAST = "at-least"  # at least n characteristics must be true
        AT_MOST = "at-most"  # at most n characteristics must be true
        STATISTICAL = "statistical"  # statistical combination of characteristics
        NET_EFFECT = "net-effect"  # net effect of characteristics
        DATASET = "dataset"  # dataset of characteristics

    def __init__(self, code: Code, exclude: bool, threshold: int | None = None) -> None:
        """
        Creates a new characteristic combination.
        """
        self.code: CharacteristicCombination.Code = code
        self.characteristics: list[
            Union["AbstractCharacteristic", "CharacteristicCombination"]
        ] = []
        self._exclude: bool = exclude
        self.threshold: int | None = threshold

    def add(
        self,
        characteristic: Union["AbstractCharacteristic", "CharacteristicCombination"],
    ) -> None:
        """Adds a characteristic to this combination."""
        self.characteristics.append(characteristic)

    @property
    def exclude(self) -> bool:
        """Returns whether to exclude the combination."""
        return self._exclude

    def __iter__(self) -> Iterator:
        """Return an iterator for the characteristics of this combination."""
        return iter(self.characteristics)
