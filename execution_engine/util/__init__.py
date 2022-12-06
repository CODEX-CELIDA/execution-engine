from abc import ABC, ABCMeta, abstractmethod
from typing import Any

from pydantic import BaseModel, root_validator
from sqlalchemy import and_, literal_column
from sqlalchemy.sql.elements import ClauseList

from execution_engine.omop.concepts import Concept

ucum_to_postgres = {
    "s": "second",
    "min": "minute",
    "h": "hour",
    "d": "day",
    "wk": "week",
    "mo": "month",
    "a": "year",
}


class Value(BaseModel, ABC):
    """A value in a criterion."""

    @abstractmethod
    def to_sql(self, table_name: str) -> str:
        """
        Get the SQL representation of the value.
        """
        pass


class ValueNumber(Value):
    """
    A value of type number.
    """

    unit: Concept
    value: float | None = None
    value_min: float | None = None
    value_max: float | None = None

    @root_validator
    def validate_value(cls, values: dict) -> dict:
        """
        Validate that value or value_min/value_max is set.
        """

        if values.get("value") is None:
            if values.get("value_min") is None and values.get("value_max") is None:
                raise ValueError("Either value or value_min and value_max must be set.")
            if (
                values.get("value_min") is not None
                and values.get("value_max") is not None
            ):
                if values.get("value_min") > values.get("value_max"):  # type: ignore
                    raise ValueError(
                        "value_min must be less than or equal to value_max."
                    )
        else:
            if (
                values.get("value_min") is not None
                or values.get("value_max") is not None
            ):
                raise ValueError(
                    "value and value_min/value_max are mutually exclusive."
                )
        return values

    def __str__(self) -> str:
        """
        Get the string representation of the value.
        """
        if self.value is not None:
            return f"Value == {self.value} {self.unit.name}"
        elif self.value_min is not None and self.value_max is not None:
            return f"{self.value_min} <= Value <= {self.value_max} {self.unit.name}"
        elif self.value_min is not None:
            return f"Value >= {self.value_min} {self.unit.name}"
        elif self.value_max is not None:
            return f"Value <= {self.value_max} {self.unit.name}"

        return "Value (undefined)"

    def __repr__(self) -> str:
        """
        Get the string representation of the value.
        """
        return str(self)

    def to_sql(
        self,
        table_name: str | None,
        column_name: str = "value_as_number",
        with_unit: bool = True,
    ) -> ClauseList:
        """
        Get the sqlalchemy representation of the value.
        """

        clauses = []

        if table_name is not None:
            table_name = f"{table_name}."
        else:
            table_name = ""

        c = literal_column(f"{table_name}{column_name}")

        if with_unit:
            c_unit = literal_column(f"{table_name}unit_concept_id")
            clauses.append(c_unit == self.unit.id)

        if self.value is not None:
            clauses.append(c == self.value)

        else:
            if self.value_min is not None:
                clauses.append(c >= self.value_min)
            if self.value_max is not None:
                clauses.append(c <= self.value_max)

        return and_(*clauses)


class ValueConcept(Value):
    """
    A value of type concept.
    """

    value: Concept

    def to_sql(self, table_name: str) -> str:
        """
        Get the SQL representation of the value.
        """
        return f"{table_name}.value_as_concept_id = {self.value.id}"

    def __str__(self) -> str:
        """
        Get the string representation of the value.
        """
        return f"Value == {str(self.value)}"

    def __repr__(self) -> str:
        """
        Get the string representation of the value.
        """
        return str(self)


class AbstractPrivateMethods(ABCMeta):
    """
    A metaclass that prevents overriding of methods decorated with @typing.final.
    """

    def __new__(mcs, name: str, bases: tuple, class_dict: dict) -> Any:
        """
        Instantiate a new class.

        Checks for __final__ attribute set on methods of parent classes (via @typing.final decorator)
        and raises an error if a child class tries to override them.
        """
        private = {
            key: base.__qualname__
            for base in bases
            for key, value in vars(base).items()
            if callable(value) and getattr(value, "__final__", False)
        }

        if any(key in private for key in class_dict):
            message = ", ".join([f"{v}.{k}" for k, v in private.items()])
            raise RuntimeError(f"Methods {message} may not be overriden")
        return super().__new__(mcs, name, bases, class_dict)
