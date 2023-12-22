import pytest
import sympy

import execution_engine.util.cohort_logic as logic
from execution_engine.constants import CohortCategory
from execution_engine.execution_map import ExecutionMap
from execution_engine.omop.criterion.combination import CriterionCombination
from tests._fixtures.mock import MockCriterion


def sort_numbers_in_string(query: str) -> str:
    import re

    # Extract all numbers from the string
    numbers = [int(num) for num in re.findall(r"\d+", query)]
    sorted_numbers = sorted(numbers)

    # Create a mapping of old to new numbers
    mapping = {old: new for old, new in zip(numbers, sorted_numbers)}

    # Replace the old numbers with the new numbers
    result = re.sub(r"c(\d+)", lambda match: f"c{mapping[int(match.group(1))]}", query)

    return result


class TestExecutionMap:
    @pytest.fixture
    def base_criterion(self):
        return MockCriterion("base_criterion", category=CohortCategory.BASE)

    @pytest.fixture
    def expressions(self):
        combinations_str = [
            "~((c1 | (~c2 & ~c3 & (c4 | c5 | (c5 & c6)))) & c3)",
            " b1 | b2 & (c1 | c2) & ~c3 & ~(a1 | a2)",
            "a1 & b1 & c1",
        ]

        return combinations_str

    @pytest.fixture
    def execution_maps(self, base_criterion, expressions):
        emaps = [
            ExecutionMap(self.build_combination(expr), base_criterion, params={})
            for expr in expressions
        ]
        return emaps

    @staticmethod
    def build_combination(expr_str: str):
        op_and = CriterionCombination.Operator(CriterionCombination.Operator.AND)
        op_or = CriterionCombination.Operator(CriterionCombination.Operator.OR)

        # Parse the expression
        expr = sympy.sympify(expr_str)

        # Recursively build the combination
        def build(expr):
            if isinstance(expr, sympy.Symbol):
                category = {
                    "p": CohortCategory.POPULATION,
                    "i": CohortCategory.INTERVENTION,
                    "pi": CohortCategory.POPULATION_INTERVENTION,
                }.get(expr.name[:-1], CohortCategory.BASE)
                return MockCriterion(str(expr), category=category)
            elif isinstance(expr, sympy.Not):
                inner = build(expr.args[0])
                inner.exclude = True
                return inner
            elif isinstance(expr, sympy.And) or isinstance(expr, sympy.Or):
                criteria = [build(arg) for arg in expr.args]

                if all(c.category == CohortCategory.BASE for c in criteria):
                    category = CohortCategory.BASE
                elif all(c.category == CohortCategory.POPULATION for c in criteria):
                    category = CohortCategory.POPULATION
                elif all(c.category == CohortCategory.INTERVENTION for c in criteria):
                    category = CohortCategory.INTERVENTION
                else:
                    category = CohortCategory.POPULATION_INTERVENTION

                comb = CriterionCombination(
                    name=str(expr),
                    operator=op_and if isinstance(expr, sympy.And) else op_or,
                    exclude=False,
                    category=category,
                    criteria=criteria,
                )
                print(f"Created combination {str(expr)} {category}")
                return comb
            else:
                raise ValueError("Unsupported expression type")

        c = build(expr)

        if isinstance(c, CriterionCombination):
            return c
        else:
            return CriterionCombination(
                name=str(expr),
                operator=op_and,
                exclude=False,
                category=c.category,
                criteria=[c],
            )

    def test_execution_map_parse(self, base_criterion):
        params = {
            "observation_start_datetime": "2020-01-01 00:00:00Z",
            "observation_end_datetime": "2020-01-03 23:59:59Z",
        }

        combinations_str = [
            "~((c1 | (~c2 & ~c3 & (c4 | c5 | (c5 & c6)))) & c3)",
            " b1 | b2 & (c1 | c2) & ~c3 & ~(a1 | a2)",
            "a1 & b1 & c1",
        ]

        criteria = CriterionCombination(
            name="root",
            exclude=False,
            category=CohortCategory.POPULATION_INTERVENTION,
            operator=CriterionCombination.Operator("OR"),
        )

        for s in combinations_str:
            comb = self.build_combination(s)

            emap = ExecutionMap(comb, base_criterion, params=params)

            assert str(emap.expression) == str(sympy.sympify(s))

            criteria.add(comb)

        emap = ExecutionMap(criteria, base_criterion, params=params)

        assert (
            str(emap.expression)
            == "(a1 & b1 & c1) | (b1 | (b2 & ~c3 & (c1 | c2) & ~(a1 | a2))) | ~(c3 & (c1 | (~c2 & ~c3 & (c4 | c5 | (c5 & c6)))))"
        )

    def test_get_combined_category(self, base_criterion):
        with pytest.raises(
            AssertionError, match="all args must be instance of ExecutionMap"
        ):
            ExecutionMap.get_combined_category(
                ExecutionMap(self.build_combination("c1"), base_criterion, params={}),
                self.build_combination("c2"),
            )

        def combined_category(exprs):
            emaps = [
                ExecutionMap(self.build_combination(expr), base_criterion, params={})
                for expr in exprs
            ]
            return ExecutionMap.get_combined_category(*emaps)

        assert combined_category(["b1"]) == CohortCategory.BASE
        assert combined_category(["b1", "b2"]) == CohortCategory.BASE
        assert combined_category(["b1 & b3", "b2 | ~b4"]) == CohortCategory.BASE

        assert combined_category(["p1"]) == CohortCategory.POPULATION
        assert combined_category(["p1", "p2"]) == CohortCategory.POPULATION
        assert (
            combined_category(["p1 & p4 | p5", "p2 & b1", "p3 | b2"])
            == CohortCategory.POPULATION
        )

        assert combined_category(["i1"]) == CohortCategory.INTERVENTION
        assert combined_category(["i1", "i2"]) == CohortCategory.INTERVENTION
        assert (
            combined_category(["b1", "b2", "i1", "b3"]) == CohortCategory.INTERVENTION
        )
        assert (
            combined_category(["i1 & i4 | i5", "i2 & b1", "i3 | b2"])
            == CohortCategory.INTERVENTION
        )

        assert combined_category(["pi1"]) == CohortCategory.POPULATION_INTERVENTION
        assert (
            combined_category(["pi1 & i1 & p1"])
            == CohortCategory.POPULATION_INTERVENTION
        )
        assert combined_category(["p1", "i1"]) == CohortCategory.POPULATION_INTERVENTION
        assert (
            combined_category(["pi1", "i1", "b1"])
            == CohortCategory.POPULATION_INTERVENTION
        )
        assert (
            combined_category(["p1", "b2", "i1"])
            == CohortCategory.POPULATION_INTERVENTION
        )
        assert (
            combined_category(["p1 | p2", "i1 & i2"])
            == CohortCategory.POPULATION_INTERVENTION
        )
        assert (
            combined_category(["p1", "p2", "i1", "p1"])
            == CohortCategory.POPULATION_INTERVENTION
        )

    def test_category(self, base_criterion):
        def category(expr):
            emap = ExecutionMap(self.build_combination(expr), base_criterion, params={})
            return emap.category

        assert category("b1") == CohortCategory.BASE
        assert category("b1 & b2") == CohortCategory.BASE
        assert category("b1 & b3 & b2 | ~b4") == CohortCategory.BASE
        assert category("p1") == CohortCategory.POPULATION
        assert category("p1 & p2") == CohortCategory.POPULATION
        assert category("p1 & p4 | p5 & p2 & b1 & p3 | b2") == CohortCategory.POPULATION
        assert category("i1") == CohortCategory.INTERVENTION
        assert category("i1 & i2") == CohortCategory.INTERVENTION
        assert category("b1 & b2 & i1 & b3") == CohortCategory.INTERVENTION
        assert (
            category("i1 & i4 | i5 & i2 & b1 & i3 | b2") == CohortCategory.INTERVENTION
        )
        assert category("pi1") == CohortCategory.POPULATION_INTERVENTION
        assert category("pi1 & i1 & p1") == CohortCategory.POPULATION_INTERVENTION
        assert category("p1 & i1") == CohortCategory.POPULATION_INTERVENTION
        assert category("pi1 & i1 & b1") == CohortCategory.POPULATION_INTERVENTION
        assert category("p1 & b2 & i1") == CohortCategory.POPULATION_INTERVENTION
        assert category("p1 | p2 & i1 & i2") == CohortCategory.POPULATION_INTERVENTION
        assert category("p1 & p2 & i1 & p1") == CohortCategory.POPULATION_INTERVENTION

    def test_and_combination(self, base_criterion, execution_maps, expressions):
        def emap(expr):
            return ExecutionMap(self.build_combination(expr), base_criterion, params={})

        combined_map = emap("c1") & emap("c2") & emap("c3")
        assert isinstance(combined_map, ExecutionMap)
        assert str(combined_map.expression) == "c3 & (c1 & c2)"

        combined_map = emap("c1 & c2 & ~c3") & emap("c4 & c5") & emap("c6 & (c7 | c8)")
        assert isinstance(combined_map, ExecutionMap)
        assert (
            str(combined_map.expression)
            == "(c6 & (c7 | c8)) & ((c4 & c5) & (c1 & c2 & ~c3))"
        )

        combined_map = execution_maps[0] & execution_maps[1] & execution_maps[2]
        assert isinstance(combined_map, ExecutionMap)
        expected_expression = (
            f"({expressions[2]}) & (({expressions[1]}) & ({expressions[0]}))"
        )

        assert sympy.sympify(str(combined_map.expression)) == sympy.sympify(
            expected_expression
        )

    def test_or_combination(self, base_criterion, execution_maps, expressions):
        def emap(expr):
            return ExecutionMap(self.build_combination(expr), base_criterion, params={})

        combined_map = emap("c1") | emap("c2") | emap("c3")
        assert isinstance(combined_map, ExecutionMap)
        assert str(combined_map.expression) == "c3 | (c1 | c2)"

        combined_map = emap("c1 & c2 & ~c3") | emap("c4 & c5") | emap("c6 & (c7 | c8)")
        assert isinstance(combined_map, ExecutionMap)
        assert (
            str(combined_map.expression)
            == "(c6 & (c7 | c8)) | ((c4 & c5) | (c1 & c2 & ~c3))"
        )

        combined_map = execution_maps[0] | execution_maps[1] | execution_maps[2]
        assert isinstance(combined_map, ExecutionMap)
        expected_expression = (
            f"({expressions[2]}) | (({expressions[1]}) | ({expressions[0]}))"
        )

        assert sympy.sympify(str(combined_map.expression)) == sympy.sympify(
            expected_expression
        )

    def test_invert_operation(self, base_criterion, execution_maps, expressions):
        def build_emap(expr):
            return ExecutionMap(self.build_combination(expr), base_criterion, params={})

        emap = build_emap("c1")
        inverted_map = ~emap
        assert isinstance(inverted_map, ExecutionMap)
        for criterion in inverted_map.flatten():
            assert criterion.exclude is False
        assert str(inverted_map.expression) == "~c1"

        emap = build_emap("a1 & b1 | c2")
        inverted_map = ~emap
        assert isinstance(inverted_map, ExecutionMap)
        for criterion in inverted_map.flatten():
            assert criterion.exclude is False
        assert str(inverted_map.expression) == "~(c2 | (a1 & b1))"

        for emap, expr in zip(execution_maps, expressions):
            inverted_map = ~emap
            assert isinstance(inverted_map, ExecutionMap)
            assert sympy.sympify(str(inverted_map.expression)) == sympy.sympify(
                f"~({expr})"
            )

    def test_union_operation(self, base_criterion, execution_maps, expressions):
        def emap(expr):
            return ExecutionMap(self.build_combination(expr), base_criterion, params={})

        with pytest.raises(
            AssertionError, match="all args must be instance of ExecutionMap"
        ):
            ExecutionMap.combine_from(
                execution_maps[0], self.build_combination("c2"), operator=logic.And
            )

        with pytest.raises(AssertionError, match="base criteria must be equal"):
            ExecutionMap.combine_from(
                execution_maps[0],
                ExecutionMap(
                    self.build_combination("c2"),
                    MockCriterion("base_criterion2", category=CohortCategory.BASE),
                    params={},
                ),
                operator=logic.And,
            )

        with pytest.raises(AssertionError, match="params must be equal"):
            ExecutionMap.combine_from(
                execution_maps[0],
                ExecutionMap(
                    self.build_combination("c2"), base_criterion, params={"a": 1}
                ),
                operator=logic.And,
            )

        combined_map = ExecutionMap.combine_from(
            emap("c1"), emap("c2"), emap("c3"), operator=logic.Or
        )
        assert isinstance(combined_map, ExecutionMap)
        assert str(combined_map.expression) == "c1 | c2 | c3"

        combined_map = ExecutionMap.combine_from(
            emap("c1 & c2 & ~c3"),
            emap("c4 & c5"),
            emap("c6 & (c7 | c8)"),
            operator=logic.Or,
        )
        assert isinstance(combined_map, ExecutionMap)
        assert (
            str(combined_map.expression)
            == "(c4 & c5) | (c6 & (c7 | c8)) | (c1 & c2 & ~c3)"
        )

        combined_map = ExecutionMap.combine_from(*execution_maps, operator=logic.Or)
        assert isinstance(combined_map, ExecutionMap)
        expected_expression = (
            f"({expressions[2]}) | (({expressions[1]}) | ({expressions[0]}))"
        )

        assert sympy.sympify(str(combined_map.expression)) == sympy.sympify(
            expected_expression
        )
