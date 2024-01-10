import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Self


class Serializable(ABC):
    """
    Base class for serializable objects.
    """

    _id: int | None = None

    @property
    def id(self) -> int:
        """
        Get the id of the object (used in the database).
        """
        if self._id is None:
            raise ValueError("Id not set")
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        """
        Set the id of the object (used in the database).
        """
        self._id = value

    @abstractmethod
    def dict(self) -> dict:
        """
        Get a dictionary representation of the object.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Create an object from a dictionary.
        """
        raise NotImplementedError()

    def json(self) -> bytes:
        """
        Get a JSON representation of the object.

        The json excludes the id, as this is auto-inserted by the database
        and not known during the creation of the object.
        """

        s_json = self.dict()

        if "id" in s_json:
            del s_json["id"]

        return json.dumps(s_json, sort_keys=True).encode()

    @classmethod
    def from_json(cls, data: str) -> Self:
        """
        Create a combination from a JSON string.
        """
        return cls.from_dict(json.loads(data))

    def __eq__(self, other: Any) -> bool:
        """
        Check if two objects are equal.
        """
        if not isinstance(other, self.__class__):
            return False

        return self.dict() == other.dict()

    def __hash__(self) -> int:
        """
        Get the hash of the object.
        """
        return hash(self.json())
