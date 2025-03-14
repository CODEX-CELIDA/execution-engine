import pickle  # nosec: need to test pickling of objects
from multiprocessing import Process, Queue

import pytest

from execution_engine.util.enum import TimeIntervalType
from execution_engine.util.logic import (
    AllOrNone,
    And,
    BooleanFunction,
    ExactCount,
    Expr,
    LeftDependentToggle,
    MaxCount,
    MinCount,
    NoDataPreservingAnd,
    NoDataPreservingOr,
    NonSimplifiableAnd,
    Not,
    Or,
    Symbol,
    TemporalExactCount,
    TemporalMaxCount,
    TemporalMinCount,
)
from tests.mocks.criterion import MockCriterion

dummy_criterion = MockCriterion(
    name="dummy_criterion",
)


x, y, z = (
    MockCriterion("x"),
    MockCriterion("y"),
    MockCriterion("z"),
)


# Tests for Expr
class TestExpr:
    def test_is_Atom_false(self):
        expr = Expr()
        assert not expr.is_Atom


# Tests for Symbol
class TestSymbol:
    def test_symbol_creation(self):
        symbol = dummy_criterion
        assert repr(symbol) == "MockCriterion(\n  name='dummy_criterion'\n)"
        assert symbol == dummy_criterion

    def test_is_Atom_true(self):
        symbol = x
        assert symbol.is_Atom

    def test_is_Not_false(self):
        symbol = x
        assert not symbol.is_Not


class TestBooleanFunction:
    def test_or_creation(self):
        or_expr = Or(x, y)
        assert isinstance(or_expr, Or)

    def test_or_is_Atom_false(self):
        or_expr = Or(x, y)
        assert not or_expr.is_Atom


class TestAnd:
    def test_and_creation_with_multiple_args(self):
        and_expr = And(
            x,
            y,
            Not(z),
        )
        assert isinstance(and_expr, And)

        assert (
            repr(and_expr) == "And(\n"
            "  MockCriterion(\n"
            "      name='x'\n"
            "    ),\n"
            "  MockCriterion(\n"
            "      name='y'\n"
            "    ),\n"
            "  Not(\n"
            "      MockCriterion(\n"
            "          name='z'\n"
            "        )\n"
            "    )\n"
            ")"
        )
        assert and_expr.args[0] == x
        assert and_expr.args[1] == y
        assert and_expr.args[2] == Not(z)

        assert not and_expr.is_Not
        assert not and_expr.is_Atom

    def test_and_creation_with_single_arg(self):
        single_arg = x
        and_expr = And(single_arg)
        assert and_expr is single_arg


class TestOr:
    def test_or_creation_with_multiple_args(self):
        or_expr = Or(
            x,
            y,
            Not(z),
        )
        assert isinstance(or_expr, Or)

        assert (
            repr(or_expr) == "Or(\n"
            "  MockCriterion(\n"
            "      name='x'\n"
            "    ),\n"
            "  MockCriterion(\n"
            "      name='y'\n"
            "    ),\n"
            "  Not(\n"
            "      MockCriterion(\n"
            "          name='z'\n"
            "        )\n"
            "    )\n"
            ")"
        )
        assert or_expr.args[0] == x
        assert or_expr.args[1] == y
        assert or_expr.args[2] == Not(z)

        assert not or_expr.is_Not
        assert not or_expr.is_Atom

    def test_or_creation_with_single_arg(self):
        single_arg = x
        or_expr = Or(single_arg)
        assert or_expr is single_arg


class TestNot:
    def test_not_creation(self):
        not_expr = Not(x)
        assert isinstance(not_expr, Not)
        assert str(not_expr) == "~MockCriterion[x]"
        assert not_expr.args[0] == x

        assert not_expr.is_Not
        assert not not_expr.is_Atom

    def test_not_creation_with_multiple_args(self):
        with pytest.raises(ValueError):
            Not(x, y)


class TestNonSimplifiableAnd:
    def test_non_simplifiable_and_creation(self):
        non_simp_and = NonSimplifiableAnd(x, y)
        assert isinstance(non_simp_and, NonSimplifiableAnd)
        assert not non_simp_and.is_Not
        assert not non_simp_and.is_Atom

    def test_non_simplifiable_and_single_arg(self):
        single_arg = x
        non_simp_and = NonSimplifiableAnd(single_arg)
        assert isinstance(non_simp_and, NonSimplifiableAnd)
        assert non_simp_and.args[0] is single_arg

    def test_non_simplifiable_and_equality(self):
        non_simp_and1 = NonSimplifiableAnd(x)
        non_simp_and2 = NonSimplifiableAnd(x)
        assert non_simp_and1 == non_simp_and2


class TestNoDataPreservingAnd:
    def test_no_data_preserving_and_creation(self):
        no_data_and = NoDataPreservingAnd(x, y)
        assert isinstance(no_data_and, NoDataPreservingAnd)
        assert no_data_and.args[0] == x
        assert no_data_and.args[1] == y

        assert not no_data_and.is_Not
        assert not no_data_and.is_Atom


class TestNoDataPreservingOr:
    def test_no_data_preserving_or_creation(self):
        no_data_or = NoDataPreservingOr(x, y)
        assert isinstance(no_data_or, NoDataPreservingOr)
        assert no_data_or.args[0] == x
        assert no_data_or.args[1] == y

        assert not no_data_or.is_Not
        assert not no_data_or.is_Atom


class TestLeftDependentToggle:
    def test_left_dependent_toggle_creation(self):
        left_expr = x
        right_expr = y
        left_toggle = LeftDependentToggle(left=left_expr, right=right_expr)
        assert isinstance(left_toggle, LeftDependentToggle)
        assert left_toggle.left == left_expr == left_toggle.args[0]
        assert left_toggle.right == right_expr == left_toggle.args[1]

        assert not left_toggle.is_Not
        assert not left_toggle.is_Atom


def worker(queue: Queue, symbol: Symbol):
    """
    Worker function to test pickling and unpickling of Symbol objects.
    This function tries to put a Symbol object into a multiprocessing Queue,
    which implicitly tests the pickling process.
    """
    queue.put(symbol)


class TestSymbolMultiprocessing:
    @pytest.fixture(
        params=[
            Expr(1, 2, 3),
            Symbol(dummy_criterion),
            BooleanFunction(1, 2, 3),
            Or(1, 2, 3),
            And(1, 2, 3),
            Not(1),
            MinCount(1, 2, 3, threshold=2),
            MaxCount(1, 2, 3, threshold=2),
            ExactCount(1, 2, 3, threshold=2),
            AllOrNone(1, 2, 3),
            NonSimplifiableAnd(1, 2, 3),
            NoDataPreservingAnd(1, 2, 3),
            NoDataPreservingOr(1, 2, 3),
            LeftDependentToggle(left=1, right=2),
            TemporalMinCount(
                1,
                2,
                3,
                threshold=2,
                start_time=None,
                end_time=None,
                interval_type=TimeIntervalType.DAY,
                interval_criterion=None,
            ),
            TemporalMaxCount(
                1,
                2,
                3,
                threshold=3,
                start_time=None,
                end_time=None,
                interval_type=TimeIntervalType.MORNING_SHIFT,
                interval_criterion=None,
            ),
            TemporalExactCount(
                1,
                2,
                3,
                threshold=2,
                start_time=None,
                end_time=None,
                interval_type=TimeIntervalType.NIGHT_SHIFT,
                interval_criterion=None,
            ),
        ],
        ids=lambda expr: expr.__class__.__name__,
    )
    def expr(self, request):
        return request.param

    def test_symbol_pickle_unpickle(self, expr):
        """
        Test if a Symbol object can be pickled and unpickled directly.
        """

        pickled_expr = pickle.dumps(expr)  # nosec: need to test pickling of objects
        unpickled_expr = pickle.loads(
            pickled_expr
        )  # nosec: need to test pickling of objects

        assert isinstance(
            unpickled_expr, expr.__class__
        ), f"Unpickled object is not an instance of {expr.__class__.__name__}"
        for arg in expr.args:
            assert (
                arg in unpickled_expr.args
            ), "The args of the unpickled Expression do not match the original"

    def test_symbol_multiprocessing_transfer(self, expr):
        """
        Test if a Symbol object can be transferred to another process.
        """
        queue = Queue()

        process = Process(target=worker, args=(queue, expr))
        process.start()
        process.join()

        received_expr = queue.get()

        assert isinstance(
            received_expr, expr.__class__
        ), f"Received object is not an instance of {expr.__class__.__name__}"
        for arg in expr.args:
            assert (
                arg in received_expr.args
            ), "The args of the received Expression do not match the original"
