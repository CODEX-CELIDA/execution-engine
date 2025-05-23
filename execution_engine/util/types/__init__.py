from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from execution_engine.task.process import AnyInterval
from execution_engine.util import serializable
from execution_engine.util.enum import TimeUnit
from execution_engine.util.value import ValueNumber, ValueNumeric
from execution_engine.util.value.time import ValueCount, ValueDuration, ValuePeriod

PersonIntervals = dict[int, AnyInterval]


@serializable.register_class
class Timing(BaseModel):
    """
    The timing of a criterion.

    The criterion is satisfied _once_ if it is satisfied 'frequency' times in a certain period,
    where each time should last 'duration'. The criterion is satisfied if it is satisfied 'count' times.

    The timing of a criterion is defined by the following parameters:
    - count: The number of times that the criterion should be satisfied according to the other parameters.
    - duration: The duration of each time that the criterion should be satisfied.
    - frequency: Number of repetitions of the criterion per period (= the interval in which the criterion should be
        satisfied 'frequency' times).
    """

    count: ValueCount | None = None
    duration: ValueDuration | None = None  # from duration OR boundsRange
    frequency: ValueCount | None = None
    interval: ValuePeriod | None = None
    model_config = ConfigDict(validate_assignment=True, use_enum_values=True)

    @field_validator("count", "frequency", mode="before")
    def convert_to_value_count(cls, v: ValueCount | int | str) -> ValueCount:
        """
        Convert the count and frequency to ValueFrequency when they are integers.

        Possible because ValueFrequency has no other parameters.
        """
        if isinstance(v, int):
            return ValueCount(value=v)
        if isinstance(v, str):
            return ValueCount.parse(v)
        return v

    @field_validator("interval", mode="before")
    def convert_to_value_period(cls, v: Any) -> ValuePeriod:
        """
        Convert the interval to ValuePeriod when it is an integer or ValueNumeric.

        Possible under the assumption that a single value is value, not value_min or value_max.
        """
        if isinstance(v, ValuePeriod):
            return v
        if isinstance(v, (TimeUnit, str)):
            return ValuePeriod(value=1, unit=v)
        if isinstance(v, ValueNumeric):
            return ValuePeriod(value=v.value, unit=v.unit)

        return v

    @field_validator("duration", mode="before")
    def convert_to_value_duration(cls, v: Any) -> ValueDuration:
        """
        Convert the duration to ValueDuration when it is an integer or ValueNumeric.

        Possible under the assumption that a single value is value, not value_min or value_max.
        """
        if isinstance(v, ValueDuration):
            return v
        if isinstance(v, (TimeUnit, str)):
            return ValueDuration(value=1, unit=v)
        if isinstance(v, ValueNumeric):
            return ValueDuration(value=v.value, unit=v.unit)

        return v

    @model_validator(mode="after")
    def validate_value(cls, values: Any) -> dict:
        """
        Validate interval is set when frequency is set; set frequency when interval is set.
        """

        if values.interval is None:
            if values.frequency is not None:
                raise ValueError("interval must be set when frequency is set.")
        else:
            if values.frequency is None:
                values.frequency = 1

        return values

    def __str_components__(self) -> list[str]:
        """
        Get the string representation of the timing.
        """
        params = []

        if self.count is not None:
            params.append(f"count{self.count}")
        if self.duration is not None:
            params.append(f"duration{self.duration}")
        if self.frequency is not None:
            params.append(f"frequency{self.frequency} per {self.interval}")

        return params

    def __str__(self) -> str:
        """
        Get the string representation of the timing.
        """
        return (
            f"{self.__class__.__name__}(" + ", ".join(self.__str_components__()) + ")"
        )

    def model_dump(
        self, *args: Any, include_meta: bool = False, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Get the dictionary representation of the timing.

        :param include_meta: Whether to include the class name in the dictionary.
        :return: The dictionary representation of the value.
        """
        data = super().model_dump(*args, **kwargs)
        if include_meta:
            return {
                "class_name": self.__class__.__name__,
                "data": data,
            }
        return data


@serializable.register_class
class Dosage(Timing):
    """
    A dosage consisting of a dose in addition to the Timing fields.
    """

    dose: ValueNumber | None = None
    model_config = ConfigDict(use_enum_values=True)

    def __str_components__(self) -> list[str]:
        """
        Get the string representation of the dosage.
        """
        params = []

        if self.dose is not None:
            params.append(f"dose{self.dose}")

        return params + super().__str_components__()
