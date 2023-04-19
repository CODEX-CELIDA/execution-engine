import pytest
from sqlalchemy import Column, Integer, MetaData, Table, union
from sqlalchemy.sql import select

from execution_engine.util.sql import SelectInto, select_into


class TestSelectInto:
    @pytest.fixture
    def metadata(self):
        return MetaData()

    @pytest.fixture
    def source_table(self, metadata):
        return Table("source", metadata, Column("id", Integer))

    @pytest.fixture
    def union_source_table(self, metadata):
        return Table("source_union", metadata, Column("id", Integer))

    @pytest.fixture
    def target_table(self, metadata):
        return Table("target", metadata, Column("id", Integer))

    def test_select_into(self, source_table, target_table):
        # Create a Select object
        query = select(source_table)

        # Create an alias for the target table
        target_alias = target_table.alias()

        # Create a SelectInto object
        select_into_obj = select_into(query, target_alias, temporary=True)

        assert isinstance(select_into_obj, SelectInto)
        assert select_into_obj.select is query
        assert select_into_obj.into is target_alias
        assert select_into_obj.temporary is True

    def test_select_into_compiles(self, source_table, target_table, union_source_table):
        query = select(source_table)
        target_alias = target_table.alias()
        select_into_obj = select_into(query, target_alias, temporary=True)

        compiled_query = str(select_into_obj.compile())

        expected_query = (
            "SELECT source.id INTO TEMPORARY TABLE target AS target_1 FROM source"
        )
        assert compiled_query.replace("\n", "") == expected_query

        # Test with temporary=False
        select_into_obj = select_into(query, target_alias, temporary=False)
        compiled_query = str(select_into_obj.compile())

        expected_query = "SELECT source.id INTO TABLE target AS target_1 FROM source"
        assert compiled_query.replace("\n", "") == expected_query

        # test compound select
        query = union(select(source_table), select(union_source_table))
        select_into_obj = select_into(query, target_alias, temporary=True)

        compiled_query = str(select_into_obj.compile())
        expected_query = "SELECT source.id INTO TEMPORARY TABLE target AS target_1 FROM source UNION SELECT source_union.id FROM source_union"
        assert compiled_query.replace("\n", "") == expected_query
