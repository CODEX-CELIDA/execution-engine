from abc import ABC, abstractmethod

from pydantic import BaseModel, root_validator

from execution_engine.omop.concepts import Concept


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

    def to_sql(
        self,
        table_name: str | None,
        column_name: str = "value_as_number",
        with_unit: bool = True,
    ) -> str:
        """
        Get the SQL representation of the value.
        """

        sql = []

        if table_name is not None:
            table_name = f"{table_name}."
        else:
            table_name = ""

        if with_unit:
            sql.append(f"{table_name}unit_concept_id = {self.unit.id}")

        if self.value is not None:
            sql.append(f"{table_name}{column_name} = {self.value}")
        else:
            if self.value_min is not None:
                sql.append(f"{table_name}{column_name} >= {self.value_min}")
            if self.value_max is not None:
                sql.append(f"{table_name}{column_name} <= {self.value_max}")

        return " AND ".join(sql)


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
