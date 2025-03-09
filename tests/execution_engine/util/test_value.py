import json

import pytest
from pydantic import ValidationError
from pydantic.functional_validators import model_validator
from sqlalchemy import Column, ColumnClause, MetaData, Table

from execution_engine.omop.concepts import Concept
from execution_engine.util.enum import TimeUnit
from execution_engine.util.value import ValueConcept, ValueNumber
from execution_engine.util.value.factory import value_factory
from execution_engine.util.value.time import ValueDuration, ValuePeriod
from execution_engine.util.value.value import get_precision
from tests._fixtures.concept import concept_unit_mg


def test_get_precision():
    # Test the function for a few basic cases to make sure it's working
    assert get_precision(20) == 0
    assert get_precision(0.003) == 3
    assert get_precision(2e-5) == 5

    # Test the function for some edge cases
    assert get_precision(0) == 0
    assert get_precision(0.0) == 1
    assert get_precision(1e-0) == 1

    # Test the function for some invalid inputs
    with pytest.raises(TypeError):
        get_precision("2e-abc")

    # Test the function for empty input
    with pytest.raises(TypeError):
        get_precision("")


def assert_serialization(value):
    json_str = json.dumps(value.model_dump())
    dict_from_str = json.loads(json_str)

    assert value.__class__(**dict_from_str) == value

    json_str = json.dumps(value.model_dump(include_meta=True))
    dict_from_str = json.loads(json_str)

    assert value_factory(**dict_from_str) == value


@pytest.fixture
def test_table():
    metadata = MetaData()
    return Table(
        "test_table",
        metadata,
        Column("value_as_number"),
        Column("unit_concept_id"),
        Column("value_as_concept_id"),
    )


class TestValueNumber:
    @pytest.fixture
    def value_number_root_validator_disabled(self):
        class ValueNumberValidatorDisabled(ValueNumber):
            @model_validator(mode="after")
            def validate_value(cls, values: "ValueNumber") -> "ValueNumber":
                return values  # Override with no validation logic

        yield ValueNumberValidatorDisabled

    def test_validate_value(self, unit_concept):
        with pytest.raises(
            ValueError, match="Either value or value_min and value_max must be set."
        ):
            ValueNumber(unit=unit_concept)

        with pytest.raises(
            ValueError, match="value and value_min/value_max are mutually exclusive."
        ):
            ValueNumber(unit=unit_concept, value=5, value_min=1, value_max=10)

        with pytest.raises(
            ValueError, match="value and value_min/value_max are mutually exclusive."
        ):
            ValueNumber(unit=unit_concept, value=5, value_max=10)

        with pytest.raises(
            ValueError, match="value and value_min/value_max are mutually exclusive."
        ):
            ValueNumber(unit=unit_concept, value=5, value_min=1)

        with pytest.raises(
            ValueError, match="value_min must be less than or equal to value_max."
        ):
            ValueNumber(unit=unit_concept, value_min=10, value_max=5)

        vn = ValueNumber(unit=unit_concept, value=5)
        assert vn.value == 5
        assert vn.value_min is None
        assert vn.value_max is None
        assert vn.unit == unit_concept

        vn = ValueNumber(unit=unit_concept, value_min=1, value_max=10)
        assert vn.value is None
        assert vn.value_min == 1
        assert vn.value_max == 10

        with pytest.raises(
            ValidationError, match="1 validation error for ValueNumber\nunit"
        ):
            vn = ValueNumber(value_min=1)

    def test_str(self, unit_concept):
        vn = ValueNumber(unit=unit_concept, value=5)
        assert str(vn) == "=5.0 Test Unit"

        vn = ValueNumber(unit=unit_concept, value_min=1, value_max=10)
        assert str(vn) == "=between(1.0, 10.0) Test Unit"

        vn = ValueNumber(unit=unit_concept, value_min=1)
        assert str(vn) == ">=1.0 Test Unit"

        vn = ValueNumber(unit=unit_concept, value_max=10)
        assert str(vn) == "<=10.0 Test Unit"

    def test_str_unreachable(self, unit_concept, value_number_root_validator_disabled):
        vn = value_number_root_validator_disabled(unit=unit_concept)
        with pytest.raises(ValueError, match="Value is not set."):
            str(vn)

    def test_repr(self, unit_concept):
        vn = ValueNumber(unit=unit_concept, value=5)
        assert (
            repr(vn)
            == """ValueNumber(unit=Concept(concept_id=1, concept_name='Test Unit', concept_code='unit', domain_id='units', vocabulary_id='test', concept_class_id='test', standard_concept=None, invalid_reason=None), value=5.0, value_min=None, value_max=None)"""
        )

    def test_to_sql(self, unit_concept, test_table):
        vn = ValueNumber(unit=unit_concept, value=5)

        clauses = vn.to_sql(with_unit=True)
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "unit_concept_id = :unit_concept_id_1 AND abs(value_as_number - :value_as_number_1) < :abs_1"
        )

        vn = ValueNumber(unit=unit_concept, value_min=1, value_max=10)
        clauses = vn.to_sql(with_unit=True)
        assert len(clauses.clauses) == 3
        assert (
            str(clauses)
            == "unit_concept_id = :unit_concept_id_1 AND value_as_number - :value_as_number_1 >= :param_1 AND value_as_number - :value_as_number_2 <= :param_2"
        )

        vn = ValueNumber(unit=unit_concept, value_min=1)
        clauses = vn.to_sql(with_unit=True)
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "unit_concept_id = :unit_concept_id_1 AND value_as_number - :value_as_number_1 >= :param_1"
        )

        vn = ValueNumber(unit=unit_concept, value_max=10)
        clauses = vn.to_sql(with_unit=True)
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "unit_concept_id = :unit_concept_id_1 AND value_as_number - :value_as_number_1 <= :param_1"
        )

        with pytest.raises(
            ValueError,
            match="If table is set, column_name must be a string, not a ColumnElement.",
        ):
            vn.to_sql(table=test_table, column_name=ColumnClause("value_as_number"))

        vn = ValueNumber(unit=unit_concept, value=5)
        clauses = vn.to_sql(table=test_table, with_unit=True)
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "test_table.unit_concept_id = :unit_concept_id_1 AND abs(test_table.value_as_number - :value_as_number_1) < :abs_1"
        )

        vn = ValueNumber(unit=unit_concept, value_min=1, value_max=10)
        clauses = vn.to_sql(table=test_table, with_unit=True)
        assert len(clauses.clauses) == 3
        assert (
            str(clauses)
            == "test_table.unit_concept_id = :unit_concept_id_1 AND test_table.value_as_number - :value_as_number_1 >= :param_1 AND test_table.value_as_number - :value_as_number_2 <= :param_2"
        )

        vn = ValueNumber(unit=unit_concept, value_min=1)
        clauses = vn.to_sql(table=test_table, with_unit=True)
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "test_table.unit_concept_id = :unit_concept_id_1 AND test_table.value_as_number - :value_as_number_1 >= :param_1"
        )

        vn = ValueNumber(unit=unit_concept, value_max=10)
        clauses = vn.to_sql(table=test_table, with_unit=True)
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "test_table.unit_concept_id = :unit_concept_id_1 AND test_table.value_as_number - :value_as_number_1 <= :param_1"
        )

        vn = ValueNumber(unit=unit_concept, value=5)
        custom_column = ColumnClause("custom_column")
        clauses = vn.to_sql(column_name=custom_column, with_unit=True)
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "unit_concept_id = :unit_concept_id_1 AND abs(custom_column - :custom_column_1) < :abs_1"
        )

    def test_value_number_parse_value(self, unit_concept):
        result = ValueNumber.parse("3.2", unit_concept)
        assert result.value == 3.2
        assert result.unit == unit_concept
        assert result.value_min is None
        assert result.value_max is None

    def test_value_number_parse_value_min(self, unit_concept):
        result = ValueNumber.parse(">=2.5", unit_concept)
        assert result.value_min == 2.5
        assert result.unit == unit_concept
        assert result.value is None
        assert result.value_max is None

    def test_value_number_parse_value_max(self, unit_concept):
        result = ValueNumber.parse("<=10.0", unit_concept)
        assert result.value_max == 10.0
        assert result.unit == unit_concept
        assert result.value is None
        assert result.value_min is None

    def test_value_number_parse_range(self, unit_concept):
        result = ValueNumber.parse("0.5-6.5", unit_concept)
        assert result.value_min == 0.5
        assert result.value_max == 6.5
        assert result.unit == unit_concept
        assert result.value is None

    def test_value_number_parse_range_with_negative_min_max(self, unit_concept):
        result = ValueNumber.parse("-6.5--0.5", unit_concept)
        assert result.value_min == -6.5
        assert result.value_max == -0.5
        assert result.unit == unit_concept
        assert result.value is None

    def test_value_number_parse_range_with_negative_min(self, unit_concept):
        result = ValueNumber.parse("-0.5-6.5", unit_concept)
        assert result.value_min == -0.5
        assert result.value_max == 6.5
        assert result.unit == unit_concept
        assert result.value is None

    def test_value_number_parse_range_with_negative_max(self, unit_concept):
        with pytest.raises(ValidationError):
            ValueNumber.parse("0.5--6.5", unit_concept)

    def test_value_number_parse_negative_value(self, unit_concept):
        result = ValueNumber.parse("-3.2", unit_concept)
        assert result.value == -3.2
        assert result.unit == unit_concept
        assert result.value_min is None
        assert result.value_max is None

    def test_value_number_parse_non_number(self, unit_concept):
        with pytest.raises(ValueError):
            ValueNumber.parse("non_number", unit_concept)

    def test_value_number_parse_greater_than_not_supported(self, unit_concept):
        with pytest.raises(ValueError) as e:
            ValueNumber.parse(">2.5", unit_concept)
        assert str(e.value) == "ValueNumber does not support '>' (only '>=')."

    def test_value_number_parse_less_than_not_supported(self, unit_concept):
        with pytest.raises(ValueError) as e:
            ValueNumber.parse("<10.0", unit_concept)
        assert str(e.value) == "ValueNumber does not support '<' (only '<=')."

    def test_serialization(self):
        assert_serialization(
            ValueNumber(value_min=1, value_max=10, unit=concept_unit_mg)
        )
        assert_serialization(ValueNumber(value_max=10, unit=concept_unit_mg))
        assert_serialization(ValueNumber(value_min=1, unit=concept_unit_mg))
        assert_serialization(ValueNumber(value=5, unit=concept_unit_mg))


class TestValueConcept:
    def test_to_sql(self, test_concept, test_table):
        value_concept = ValueConcept(value=test_concept)

        sql = value_concept.to_sql(table=test_table)
        assert str(sql) == "test_table.value_as_concept_id = :value_as_concept_id_1"

        sql = value_concept.to_sql(table=None)
        assert str(sql) == "value_as_concept_id = :value_as_concept_id_1"

        with pytest.raises(ValueError, match="ValueConcept does not support units."):
            value_concept.to_sql(table=test_table, with_unit=True)

    def test_str(self, test_concept):
        value_concept = ValueConcept(value=test_concept)
        assert (
            str(value_concept)
            == "Value == Concept(concept_id=1, concept_name='Test Concept', concept_code='unit', domain_id='units', vocabulary_id='test', concept_class_id='test', standard_concept=None, invalid_reason=None)"
        )

    def test_repr(self, test_concept):
        value_concept = ValueConcept(value=test_concept)
        assert (
            repr(value_concept)
            == "ValueConcept(value=Concept(concept_id=1, concept_name='Test Concept', concept_code='unit', domain_id='units', vocabulary_id='test', concept_class_id='test', standard_concept=None, invalid_reason=None))"
        )

    def test_dict(self, test_concept):
        value_concept = ValueConcept(value=test_concept)

        json_representation = value_concept.model_dump()
        assert json_representation == {
            "value": {
                "concept_class_id": "test",
                "concept_code": "unit",
                "concept_id": 1,
                "concept_name": "Test Concept",
                "domain_id": "units",
                "invalid_reason": None,
                "standard_concept": None,
                "vocabulary_id": "test",
            }
        }

    def test_dict_meta(self, test_concept):
        value_concept = ValueConcept(value=test_concept)

        json_representation = value_concept.model_dump(include_meta=True)
        assert json_representation == {
            "class_name": "ValueConcept",
            "data": {
                "value": {
                    "concept_class_id": "test",
                    "concept_code": "unit",
                    "concept_id": 1,
                    "concept_name": "Test Concept",
                    "domain_id": "units",
                    "invalid_reason": None,
                    "standard_concept": None,
                    "vocabulary_id": "test",
                }
            },
        }

    def test_serialization(self, test_concept):
        assert_serialization(ValueConcept(value=test_concept))


class TestValueFactory:
    def test_value_number(self):
        data = {
            "unit": Concept(
                concept_id=1,
                concept_name="Test Unit",
                concept_code="unit",
                domain_id="units",
                vocabulary_id="test",
                concept_class_id="test",
            ),
            "value": 42.0,
        }

        value = value_factory("ValueNumber", data)
        assert isinstance(value, ValueNumber)
        assert value.unit == data["unit"]
        assert value.value == data["value"]

    def test_value_concept(self):
        data = {
            "value": Concept(
                concept_id=1,
                concept_name="Test Concept",
                concept_code="test_concept",
                domain_id="test",
                vocabulary_id="test",
                concept_class_id="test",
            ),
        }

        value = value_factory("ValueConcept", data)
        assert isinstance(value, ValueConcept)
        assert value.value == data["value"]

    def test_unknown_class(self):
        data = {
            "value": 42.0,
        }

        with pytest.raises(ValueError, match="Unknown value class UnknownClass"):
            value_factory("UnknownClass", data)


class TestValuePeriod:
    def test_non_negative_value(self):
        vp = ValuePeriod(unit=TimeUnit.DAY, value=4)
        assert vp.value == 4

    def test_negative_value_raises_error(self):
        with pytest.raises(ValueError):
            ValuePeriod(unit=TimeUnit.DAY, value=-1)

    def test_non_int_value_raises_error(self):
        with pytest.raises(ValidationError):
            ValuePeriod(unit=TimeUnit.DAY, value=1.5)

    def test_validator_not_allows_none(self):
        with pytest.raises(ValidationError):
            ValuePeriod(unit=TimeUnit.HOUR)

    def test_validator_disallows_value_min(self):
        with pytest.raises(ValidationError):
            ValuePeriod(unit=TimeUnit.HOUR, value_min=5)

    def test_validator_disallows_value_max(self):
        with pytest.raises(ValidationError):
            ValuePeriod(unit=TimeUnit.HOUR, value_max=10)

    def test_value_min_max_validation(self):
        with pytest.raises(ValueError):
            ValuePeriod(unit=TimeUnit.DAY, value_min=2, value_max=5)

    def test_normal_value(self):
        vtf = ValuePeriod(unit=TimeUnit.DAY, value=3)
        assert vtf.value == 3
        assert vtf.unit == TimeUnit.DAY

    def test_serialization(self):
        assert_serialization(ValuePeriod(unit=TimeUnit.DAY, value=3))
        assert_serialization(ValuePeriod(unit=TimeUnit.WEEK, value=3.0))


class TestValueDuration:
    def test_float_value_handling(self):
        vd = ValueDuration(unit=TimeUnit.HOUR, value=2.5)
        assert vd.value == 2.5
        assert vd.unit == TimeUnit.HOUR

    def test_str_with_value(self):
        # Assuming TimeUnit is defined and has a member 'HOUR'
        vt = ValueDuration(unit=TimeUnit.HOUR, value=10)
        assert str(vt) == "=10.0 HOUR"

    def test_str_with_value_min_max(self):
        vt = ValueDuration(unit=TimeUnit.HOUR, value_min=5, value_max=15)
        assert str(vt) == "=between(5.0, 15.0) HOUR"

    def test_str_with_value_min(self):
        vt = ValueDuration(unit=TimeUnit.HOUR, value_min=5)
        assert str(vt) == ">=5.0 HOUR"

    def test_str_with_value_max(self):
        vt = ValueDuration(unit=TimeUnit.HOUR, value_max=15)
        assert str(vt) == "<=15.0 HOUR"

    def test_str_value_not_set(self):
        with pytest.raises(ValidationError):
            ValueDuration(unit=TimeUnit.HOUR)

    def test_parse_greater_than_or_equal(self):
        vt = ValueDuration.parse(">=5", TimeUnit.DAY)
        assert vt.value_min == 5
        assert vt.unit == TimeUnit.DAY

    def test_serialization(self):
        assert_serialization(ValueDuration(unit=TimeUnit.HOUR, value_max=15))
        assert_serialization(ValueDuration(unit=TimeUnit.MINUTE, value_min=5))
        assert_serialization(
            ValueDuration(unit=TimeUnit.DAY, value_min=5, value_max=15)
        )
        assert_serialization(ValueDuration(unit=TimeUnit.WEEK, value_max=15))
        assert_serialization(ValueDuration(unit=TimeUnit.SECOND, value_min=5))
