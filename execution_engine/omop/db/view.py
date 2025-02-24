from typing import Any

import sqlalchemy as sa
from sqlalchemy import Connection, MetaData, Selectable, Table
from sqlalchemy.ext import compiler
from sqlalchemy.schema import DDLElement
from sqlalchemy.sql import table
from sqlalchemy.sql.compiler import SQLCompiler

# from https://github.com/sqlalchemy/sqlalchemy/wiki/Views


class CreateView(DDLElement):
    """
    A CREATE VIEW statement.
    """

    def __init__(self, name: str, selectable: Selectable, schema: str | None = None):
        self.name = name
        self.selectable = selectable
        self.schema = schema


class DropView(DDLElement):
    """
    A DROP VIEW statement.
    """

    def __init__(self, name: str, schema: str | None = None):
        self.name = name
        self.schema = schema


@compiler.compiles(CreateView)
def _create_view(element: DDLElement, compiler: SQLCompiler, **kw: Any) -> str:
    """
    Compile a CREATE VIEW statement.
    """
    return "CREATE VIEW %s AS %s" % (
        compiler.preparer.format_table(element),
        compiler.sql_compiler.process(element.selectable, literal_binds=True),
    )


@compiler.compiles(DropView)
def _drop_view(element: DDLElement, compiler: SQLCompiler, **kw: Any) -> str:
    """
    Compile a DROP VIEW statement.
    """
    return "DROP VIEW %s" % (compiler.preparer.format_table(element))


def view(
    name: str, metadata: MetaData, selectable: Selectable, schema: str | None = None
) -> Table:
    """
    Create a view in the database.

    :param name: The name of the view.
    :param metadata: The metadata object.
    :param selectable: The selectable object.
    :param schema: The schema name.
    :return: The table object.
    """

    def view_exists(
        ddl: DDLElement, target: MetaData, connection: Connection, **kw: Any
    ) -> bool:
        """
        Check if a view exists.
        """
        return ddl.name in sa.inspect(connection).get_view_names(schema=schema)

    def view_doesnt_exist(
        ddl: DDLElement, target: MetaData, connection: Connection, **kw: Any
    ) -> bool:
        """
        Check if a view does not exist.

        """
        return not view_exists(ddl, target, connection, **kw)

    t = table(
        name,
        *(
            sa.Column(c.name, c.type, primary_key=c.primary_key)
            for c in selectable.selected_columns
        ),
        schema=schema
    )
    t.primary_key.update(c for c in t.c if c.primary_key)

    sa.event.listen(
        metadata,
        "after_create",
        CreateView(name, selectable, schema).execute_if(callable_=view_doesnt_exist),
    )
    sa.event.listen(
        metadata,
        "before_drop",
        DropView(name, schema).execute_if(callable_=view_exists),
    )

    return t
