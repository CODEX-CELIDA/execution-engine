import pytest
from pydantic import ValidationError
from sqlalchemy import ColumnClause

from execution_engine.omop.concepts import Concept
from execution_engine.util import ValueConcept, ValueNumber, value_factory


class TestValueNumber:
    @pytest.fixture
    def unit(self):
        return Concept(
            concept_id=1,
            concept_name="Test Unit",
            concept_code="unit",
            domain_id="units",
            vocabulary_id="test",
            concept_class_id="test",
        )

    def test_validate_value(self, unit):
        with pytest.raises(
            ValueError, match="Either value or value_min and value_max must be set."
        ):
            ValueNumber(unit=unit)

        with pytest.raises(
            ValueError, match="value and value_min/value_max are mutually exclusive."
        ):
            ValueNumber(unit=unit, value=5, value_min=1, value_max=10)

        with pytest.raises(
            ValueError, match="value and value_min/value_max are mutually exclusive."
        ):
            ValueNumber(unit=unit, value=5, value_max=10)

        with pytest.raises(
            ValueError, match="value and value_min/value_max are mutually exclusive."
        ):
            ValueNumber(unit=unit, value=5, value_min=1)

        with pytest.raises(
            ValueError, match="value_min must be less than or equal to value_max."
        ):
            ValueNumber(unit=unit, value_min=10, value_max=5)

        vn = ValueNumber(unit=unit, value=5)
        assert vn.value == 5
        assert vn.value_min is None
        assert vn.value_max is None
        assert vn.unit == unit

        vn = ValueNumber(unit=unit, value_min=1, value_max=10)
        assert vn.value is None
        assert vn.value_min == 1
        assert vn.value_max == 10

        with pytest.raises(
            ValidationError, match="1 validation error for ValueNumber\nunit"
        ):
            vn = ValueNumber(value_min=1)

    def test_str(self, unit):
        vn = ValueNumber(unit=unit, value=5)
        assert str(vn) == "Value == 5.0 Test Unit"

        vn = ValueNumber(unit=unit, value_min=1, value_max=10)
        assert str(vn) == "1.0 <= Value <= 10.0 Test Unit"

        vn = ValueNumber(unit=unit, value_min=1)
        assert str(vn) == "Value >= 1.0 Test Unit"

        vn = ValueNumber(unit=unit, value_max=10)
        assert str(vn) == "Value <= 10.0 Test Unit"

    def test_repr(self, unit):
        vn = ValueNumber(unit=unit, value=5)
        assert repr(vn) == "Value == 5.0 Test Unit"

    def test_to_sql(self, unit):
        vn = ValueNumber(unit=unit, value=5)

        clauses = vn.to_sql()
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "unit_concept_id = :unit_concept_id_1 AND value_as_number = :value_as_number_1"
        )

        vn = ValueNumber(unit=unit, value_min=1, value_max=10)
        clauses = vn.to_sql()
        assert len(clauses.clauses) == 3
        assert (
            str(clauses)
            == "unit_concept_id = :unit_concept_id_1 AND value_as_number >= :value_as_number_1 AND value_as_number <= :value_as_number_2"
        )

        vn = ValueNumber(unit=unit, value_min=1)
        clauses = vn.to_sql()
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "unit_concept_id = :unit_concept_id_1 AND value_as_number >= :value_as_number_1"
        )

        vn = ValueNumber(unit=unit, value_max=10)
        clauses = vn.to_sql()
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "unit_concept_id = :unit_concept_id_1 AND value_as_number <= :value_as_number_1"
        )

        with pytest.raises(
            ValueError,
            match="If table_name is set, column_name must be a string, not a ColumnClause.",
        ):
            vn.to_sql(
                table_name="test_table", column_name=ColumnClause("value_as_number")
            )

        vn = ValueNumber(unit=unit, value=5)
        clauses = vn.to_sql(table_name="test_table")
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "test_table.unit_concept_id = :test_table_unit_concept_id_1 AND test_table.value_as_number = :test_table_value_as_number_1"
        )

        vn = ValueNumber(unit=unit, value_min=1, value_max=10)
        clauses = vn.to_sql(table_name="test_table")
        assert len(clauses.clauses) == 3
        assert (
            str(clauses)
            == "test_table.unit_concept_id = :test_table_unit_concept_id_1 AND test_table.value_as_number >= :test_table_value_as_number_1 AND test_table.value_as_number <= :test_table_value_as_number_2"
        )

        vn = ValueNumber(unit=unit, value_min=1)
        clauses = vn.to_sql(table_name="test_table")
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "test_table.unit_concept_id = :test_table_unit_concept_id_1 AND test_table.value_as_number >= :test_table_value_as_number_1"
        )

        vn = ValueNumber(unit=unit, value_max=10)
        clauses = vn.to_sql(table_name="test_table")
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "test_table.unit_concept_id = :test_table_unit_concept_id_1 AND test_table.value_as_number <= :test_table_value_as_number_1"
        )

        vn = ValueNumber(unit=unit, value=5)
        custom_column = ColumnClause("custom_column")
        clauses = vn.to_sql(column_name=custom_column)
        assert len(clauses.clauses) == 2
        assert (
            str(clauses)
            == "unit_concept_id = :unit_concept_id_1 AND custom_column = :custom_column_1"
        )


class TestValueConcept:
    @pytest.fixture
    def test_concept(self):
        return Concept(
            concept_id=1,
            concept_name="Test Concept",
            concept_code="unit",
            domain_id="units",
            vocabulary_id="test",
            concept_class_id="test",
        )

    def test_to_sql(self, test_concept):
        value_concept = ValueConcept(value=test_concept)

        sql = value_concept.to_sql(table_name="test_table")
        assert sql == "test_table.value_as_concept_id = 1"

        sql = value_concept.to_sql(table_name=None)
        assert sql == "value_as_concept_id = 1"

        with pytest.raises(ValueError, match="ValueConcept does not support units."):
            value_concept.to_sql(table_name="test_table", with_unit=True)

    def test_str(self, test_concept):
        value_concept = ValueConcept(value=test_concept)
        assert (
            str(value_concept)
            == 'Value == OMOP Concept: "Test Concept" (1) [test#unit]'
        )

    def test_repr(self, test_concept):
        value_concept = ValueConcept(value=test_concept)
        assert (
            str(value_concept)
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
