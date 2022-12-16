from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql import Alias, CompoundSelect, Select
from sqlalchemy.sql.compiler import SQLCompiler
from sqlalchemy.sql.expression import ClauseElement, Executable


class SelectInto(Executable, ClauseElement):
    """
    A Select statement that is used to create a temporary table.
    """

    inherit_cache = True

    def __init__(self, query: Select, into: Alias, temporary: bool) -> None:
        self.select = query
        self.into = into
        self.temporary = temporary


@compiles(SelectInto)
def s_into(element: SelectInto, compiler: SQLCompiler, **kwargs: dict) -> str:
    """
    Compile a SelectInto object to a SELECT INTO TEMPORARY statement.
    """

    table_into = compiler.process(element.into, asfrom=True, **kwargs)
    select = compiler.process(element.select, **kwargs)

    if isinstance(element.select, CompoundSelect):
        # take first select from compound selects (EXCEPT, INTERSECT, UNION)
        froms = element.select.selects[0].froms
    else:
        froms = element.select.froms

    assert (
        len(froms) == 1
    ), "SelectInto only supports single table selects (in from clause)"
    from_clause = "FROM " + compiler.process(froms[0], asfrom=True)
    assert from_clause in select, "FROM clause not found in select"
    idx_from = select.index(from_clause)

    replace = f"INTO {'TEMPORARY' if element.temporary else ''} TABLE {table_into} FROM"

    select = select[:idx_from] + replace + select[idx_from + 4 :]

    return select


select_into = SelectInto
