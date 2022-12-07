from typing import Any

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql import Alias, Select, TableClause
from sqlalchemy.sql.compiler import SQLCompiler
from sqlalchemy.sql.expression import ClauseElement, Executable


class SelectInto(Executable, ClauseElement):
    """
    A Select statement that is used to create a temporary table.
    """

    inherit_cache = True

    def __init__(self, select: Select, into: Alias, temporary: bool) -> None:
        self.select = select
        self.into = into
        self.temporary = temporary


@compiles(SelectInto)
def s_into(element: SelectInto, compiler: SQLCompiler, **kwargs: dict) -> str:
    """
    Compile a SelectInto object to a SELECT INTO TEMPORARY statement.
    """

    table_into = compiler.process(element.into, asfrom=True, **kwargs)
    select = compiler.process(element.select, **kwargs)

    select = select.replace(
        "FROM",
        f"INTO {'TEMPORARY' if element.temporary else ''} TABLE {table_into} FROM",
        1,
    )  # replace only the first occurrence, as CompoundSelects may contain multiple FROMs (e.g. UNION, EXCEPT)

    return select


select_into = SelectInto


def Xselect_into(*expr: list, into: TableClause, temporary: bool = False) -> SelectInto:
    """
    Create a SelectInto object that compiles to a SELECT INTO TEMPORARY statement.
    """
    cls = SelectInto._create_future_select(*expr)
    cls.into = into
    cls.temporary = temporary

    return cls
