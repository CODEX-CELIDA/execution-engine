import pytest
from pydantic import ValidationError
from sqlalchemy import Column, ColumnClause, MetaData, Table

from execution_engine.omop.concepts import Concept
from execution_engine.util import (
    ValueConcept,
    ValueNumber,
    get_precision,
    value_factory,
)


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
        vn_class = type(
            "ValueNumberValidatorDisabled",
            ValueNumber.__bases__,
            dict(ValueNumber.__dict__),
        )
        post_validators = ValueNumber.__post_root_validators__
        vn_class.__post_root_validators__ = []
        yield vn_class
        assert ValueNumber.__post_root_validators__ == post_validators

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
        assert str(vn) == "Value == 5.0 Test Unit"

        vn = ValueNumber(unit=unit_concept, value_min=1, value_max=10)
        assert str(vn) == "1.0 <= Value <= 10.0 Test Unit"

        vn = ValueNumber(unit=unit_concept, value_min=1)
        assert str(vn) == "Value >= 1.0 Test Unit"

        vn = ValueNumber(unit=unit_concept, value_max=10)
        assert str(vn) == "Value <= 10.0 Test Unit"

    def test_str_unreachable(self, unit_concept, value_number_root_validator_disabled):
        vn = value_number_root_validator_disabled(unit=unit_concept)
        with pytest.raises(ValueError, match="Value is not set."):
            str(vn)

    def test_repr(self, unit_concept):
        vn = ValueNumber(unit=unit_concept, value=5)
        assert repr(vn) == "Value == 5.0 Test Unit"

    def test_to_sql(self, unit_concept, test_table):
        vn = ValueNumber(unit=unit_concept, value=5)

        clauses = vn.to_sql()
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "unit_concept_id = :unit_concept_id_1 AND abs(value_as_number - :value_as_number_1) < :abs_1"
        )

        vn = ValueNumber(unit=unit_concept, value_min=1, value_max=10)
        clauses = vn.to_sql()
        assert len(clauses.clauses) == 3
        assert (
            str(clauses)
            == "unit_concept_id = :unit_concept_id_1 AND value_as_number - :value_as_number_1 >= :param_1 AND value_as_number - :value_as_number_2 <= :param_2"
        )

        vn = ValueNumber(unit=unit_concept, value_min=1)
        clauses = vn.to_sql()
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "unit_concept_id = :unit_concept_id_1 AND value_as_number - :value_as_number_1 >= :param_1"
        )

        vn = ValueNumber(unit=unit_concept, value_max=10)
        clauses = vn.to_sql()
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "unit_concept_id = :unit_concept_id_1 AND value_as_number - :value_as_number_1 <= :param_1"
        )

        with pytest.raises(
            ValueError,
            match="If table is set, column_name must be a string, not a ColumnClause.",
        ):
            vn.to_sql(table=test_table, column_name=ColumnClause("value_as_number"))

        vn = ValueNumber(unit=unit_concept, value=5)
        clauses = vn.to_sql(table=test_table)
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "test_table.unit_concept_id = :unit_concept_id_1 AND abs(test_table.value_as_number - :value_as_number_1) < :abs_1"
        )

        vn = ValueNumber(unit=unit_concept, value_min=1, value_max=10)
        clauses = vn.to_sql(table=test_table)
        assert len(clauses.clauses) == 3
        assert (
            str(clauses)
            == "test_table.unit_concept_id = :unit_concept_id_1 AND test_table.value_as_number - :value_as_number_1 >= :param_1 AND test_table.value_as_number - :value_as_number_2 <= :param_2"
        )

        vn = ValueNumber(unit=unit_concept, value_min=1)
        clauses = vn.to_sql(table=test_table)
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "test_table.unit_concept_id = :unit_concept_id_1 AND test_table.value_as_number - :value_as_number_1 >= :param_1"
        )

        vn = ValueNumber(unit=unit_concept, value_max=10)
        clauses = vn.to_sql(table=test_table)
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "test_table.unit_concept_id = :unit_concept_id_1 AND test_table.value_as_number - :value_as_number_1 <= :param_1"
        )

        vn = ValueNumber(unit=unit_concept, value=5)
        custom_column = ColumnClause("custom_column")
        clauses = vn.to_sql(column_name=custom_column)
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
        assert str(e.value) == "ValueNumber does not support >."

    def test_value_number_parse_less_than_not_supported(self, unit_concept):
        with pytest.raises(ValueError) as e:
            ValueNumber.parse("<10.0", unit_concept)
        assert str(e.value) == "ValueNumber does not support <."


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
            == 'Value == OMOP Concept: "Test Concept" (1) [test#unit]'
        )

    def test_repr(self, test_concept):
        value_concept = ValueConcept(value=test_concept)
        assert (
            repr(value_concept)
            == 'Value == OMOP Concept: "Test Concept" (1) [test#unit]'
        )

    def test_dict(self, test_concept):
        value_concept = ValueConcept(value=test_concept)

        json_representation = value_concept.dict()
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
