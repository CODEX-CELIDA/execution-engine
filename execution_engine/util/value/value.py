from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel, root_validator
from pydantic.generics import GenericModel
from sqlalchemy import (
    ColumnClause,
    ColumnElement,
    TableClause,
    and_,
    func,
    literal_column,
)

from execution_engine.omop.concepts import Concept

__all__ = [
    "Value",
    "ValueNumeric",
    "ValueNumber",
    "ValueConcept",
    "check_int",
]

ValueT = TypeVar("ValueT")
UnitT = TypeVar("UnitT")


def check_int(cls: BaseModel, v: int | float | None) -> int | None:
    """
    Check that a value is an integer. Used as a validator for pydantic.
    """
    if v is None:
        return v

    if isinstance(v, float) and not v.is_integer():
        raise ValueError(f"Float value {v} not allowed for integer field")

    if isinstance(v, int):
        return v

    try:
        v = float(v)
    except ValueError:
        raise ValueError(f"Value must be an integer, not {type(v)}")

    if not v.is_integer():
        raise ValueError(f"Float value {v} not allowed for integer field")

    return int(v)


def check_unit_none(cls: BaseModel, v: None) -> None:
    """
    Check that a value is not supplied. Used as a validator for pydantic.
    """
    if v is not None:
        raise ValueError("unit must be None.")

    return v


def check_value_min_max_none(cls: BaseModel, v: None) -> None:
    """
    Check that a value is not supplied. Used as a validator for pydantic.
    """
    if v is not None:
        raise ValueError("value_min/value_max must be None.")

    return v


def get_precision(value: float | int) -> int:
    """
    Get the precision (i.e. the number of decimal places) of a float or int.
    """
    if not isinstance(value, (int, float)):
        raise TypeError("value must be a float or an int.")

    n = str(value)

    if "." not in n:
        if "e" in n:
            return int(
                n.split("e")[1][1:]
            )  # In case of scientific notation, return the magnitude as precision.
        return 0  # the number is an integer, so its precision is 0
    return len(
        n.split(".")[1]
    )  # the number of digits after the decimal point is the precision


class Value(GenericModel, ABC):
    """A value in a criterion."""

    @staticmethod
    def _get_column(
        table: TableClause | None, column_name: str | ColumnElement
    ) -> ColumnClause:
        if table is not None and isinstance(column_name, ColumnElement):
            raise ValueError(
                "If table is set, column_name must be a string, not a ColumnElement."
            )

        if table is not None:
            return table.c[column_name]

        if isinstance(column_name, ColumnElement):
            return column_name

        return literal_column(column_name)

    @abstractmethod
    def to_sql(
        self,
        table: TableClause | None,
        column_name: str = "value_as_number",
        with_unit: bool = True,
    ) -> ColumnElement:
        """
        Get the SQL representation of the value.
        """

    def dict(
        self, *args: Any, include_meta: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Get the dictionary representation of the value.

        :param include_meta: Whether to include the class name in the dictionary.
        :return: The dictionary representation of the value.
        """
        data = super().dict(*args, **kwargs)
        if include_meta:
            return {
                "class_name": self.__class__.__name__,
                "data": data,
            }
        return data

    def supports_units(self) -> bool:
        """
        Returns true if this type of value supports units.
        """
        return hasattr(self, "unit")


class ValueNumeric(Value, Generic[ValueT, UnitT]):
    """
    A value of type number.
    """

    unit: UnitT
    value: ValueT | None = None
    value_min: ValueT | None = None
    value_max: ValueT | None = None

    # todo: disabled for because I need the TimeUnit enum in ValueDuration to call a function on it
    #  (to_sql_interval_length_seconds). However it was previously enabled for export to dict() I believe. Is this
    #  needed?
    # class Config:
    #     """
    #     Pydantic configuration.
    #     """
    #
    #     use_enum_values = True
    #     """ Use enum values instead of names (of e.g. TimeUnit, when populating). """

    @root_validator  # type: ignore
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
            s = f"={self.value}"
        elif self.value_min is not None and self.value_max is not None:
            s = f"=between({self.value_min}, {self.value_max})"
        elif self.value_min is not None:
            s = f">={self.value_min}"
        elif self.value_max is not None:
            s = f"<={self.value_max}"
        else:
            raise ValueError("Value is not set.")

        if self.unit is not None:
            s += f" {self.unit}"

        return s

    @classmethod
    def parse(cls, s: str, unit: UnitT | None = None) -> "ValueNumber":
        """
        Parse a string representation of a value.
        """

        value_min = None
        value_max = None
        value = None

        if s.startswith(">="):
            value_min = float(s[2:])
        elif s.startswith("<="):
            value_max = float(s[2:])
        elif s.startswith(">"):
            raise ValueError(f"{cls.__name__} does not support '>' (only '>=').")
        elif s.startswith("<"):
            raise ValueError(f"{cls.__name__} does not support '<' (only '<=').")
        elif "--" in s:
            parts = s.split("--")
            value_min = float(parts[0])
            value_max = -float(parts[1])
        elif "-" in s and s.count("-") > 1:  # Check for more than one '-' sign.
            parts = [
                part for part in s.split("-") if part
            ]  # Split and ignore empty strings
            value_min = -float(parts[0])
            value_max = float(parts[1])
        elif "-" in s and s.count("-") == 1 and not s.startswith("-"):
            parts = s.split("-")
            value_min = float(parts[0])
            value_max = float(parts[1])
        else:
            value = float(s)

        return cls(value_min=value_min, value_max=value_max, value=value, unit=unit)

    def _get_unit_clause(self, table: TableClause | None) -> ColumnClause:
        """
        Get the clause for the unit.
        """
        raise NotImplementedError(
            "ValueNumeric does not support units. "
            "It needs to be subclassed with a specific unit type."
        )

    def to_sql(
        self,
        table: TableClause | None = None,
        column_name: str | ColumnClause = "value_as_number",
        with_unit: bool = False,  # with_unit is required as drug doses from FHIR have a unit,
        # but the drug_exposure table not
    ) -> ColumnElement:
        """
        Get the sqlalchemy representation of the value.

        :param table: The table to get the column from.
        :param column_name: The name of the column to get.
        :param with_unit: Whether to include the unit in the clause (as an additional WHERE AND clause).
        :return: The sqlalchemy clause.
        """

        clauses = []

        c = self._get_column(table, column_name)

        if with_unit:
            clauses.append(self._get_unit_clause(table))

        def eps(number: ValueT) -> float:
            return min(0.001, 10 ** (-(get_precision(cast(float, number)) + 1)))

        if self.value is not None:
            clauses.append(func.abs(c - self.value) < eps(self.value))
        else:
            if self.value_min is not None:
                clauses.append((c - self.value_min) >= -eps(self.value_min))
            if self.value_max is not None:
                clauses.append((c - self.value_max) <= eps(self.value_max))

        return and_(*clauses)


class ValueNumber(ValueNumeric[float, Concept]):
    """
    A float value with a unit of type Concept.
    """

    def _get_unit_clause(self, table: TableClause | None) -> ColumnClause:
        """
        Get the clause for the unit.
        """
        c_unit = self._get_column(table, "unit_concept_id")
        return c_unit == self.unit.concept_id


class ValueConcept(Value):
    """
    A value of type concept.
    """

    value: Concept

    def to_sql(
        self,
        table: TableClause | None,
        column_name: str | ColumnClause = "value_as_concept_id",
        with_unit: bool = False,
    ) -> ColumnElement:
        """
        Get the SQL representation of the value.
        """
        if with_unit:
            raise ValueError("ValueConcept does not support units.")

        c = self._get_column(table, column_name)

        clause = c == self.value.concept_id

        return clause

    def __repr__(self) -> str:
        """
        Get the string representation of the value.
        """
        return f"Value == {repr(self.value)}"
