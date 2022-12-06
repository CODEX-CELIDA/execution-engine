from typing import Any

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql import Alias, Select, TableClause
from sqlalchemy.sql.compiler import SQLCompiler


class SelectInto(Select):
    """
    A Select statement that is used to create a temporary table.
    """

    def __init__(
        self, into: TableClause, temporary: bool, *expr: list, **kwargs: dict
    ) -> None:
        super().__init__(*expr, **kwargs)
        self.into = into
        self.temporary = temporary


@compiles(SelectInto)
def s_into(element: SelectInto, compiler: SQLCompiler, **kwargs: dict) -> str:
    """
    Compile a SelectInto object to a SELECT INTO TEMPORARY statement.
    """
    text = compiler.visit_select(element, **kwargs)
    table = element.into
    if isinstance(table, Alias):
        into = compiler.preparer.format_table(table.original)
    elif isinstance(table, TableClause):
        into = compiler.preparer.format_table(table)
    else:
        raise ValueError(f"Unexpected type for into: {type(table)}")

    text = text.replace(
        "FROM", f"INTO {'TEMPORARY' if element.temporary else ''} TABLE {into} FROM"
    )

    return text


def select_into(*expr: list, into: TableClause, temporary: bool = False) -> SelectInto:
    """
    Create a SelectInto object that compiles to a SELECT INTO TEMPORARY statement.
    """
    cls = SelectInto._create_future_select(*expr)
    cls.into = into
    cls.temporary = temporary

    return cls
