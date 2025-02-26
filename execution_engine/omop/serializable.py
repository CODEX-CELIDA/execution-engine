import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Self


class Serializable(ABC):
    """
    Base class for serializable objects.
    """

    _id: int | None = None
    """
    The id is used in the database tables.
    """

    def set_id(self, value: int) -> None:
        """
        Assigns the database ID to the object. This can only be done once.

        This ID corresponds to the primary key in the database and is set
        when the object is persisted.

        :param value: The database ID assigned to the object.
        :raises ValueError: If the ID has already been set.
        """
        if self._id is not None:
            raise ValueError("Database ID has already been set!")
        self._id = value

    @property
    def id(self) -> int:
        """
        Retrieves the database ID of the object.

        This ID is only available after the object has been stored in the database.

        :return: The database ID, or None if the object has not been stored yet.
        """
        if self._id is None:
            raise ValueError("Database ID has not been set yet!")
        return self._id

    def is_persisted(self) -> bool:
        """
        Returns True if the object has been stored in the database.
        """
        return self._id is not None

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

        return json.dumps(s_json, sort_keys=True).encode()

    @classmethod
    def from_json(cls, data: str | bytes) -> Self:
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
