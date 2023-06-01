import re

import pytest

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
    # Test the `_to_nnf` method

    @pytest.fixture
    def comb(self):
        c1 = MockCriterion("c1")
        c2 = MockCriterion("c2")
        c3 = MockCriterion("c3", exclude=True)
        c4 = MockCriterion("c4")
        c5 = MockCriterion("c5")
        c6 = MockCriterion("c6")
        c7 = MockCriterion("c7")
        c1.id, c2.id, c3.id, c4.id, c5.id, c6.id, c7.id = range(1, 8)

        op_and = CriterionCombination.Operator(CriterionCombination.Operator.AND)
        op_or = CriterionCombination.Operator(CriterionCombination.Operator.OR)
        comb1_and_incl = CriterionCombination(
            "comb1_and_incl",
            exclude=False,
            operator=op_and,
            category=CohortCategory.POPULATION_INTERVENTION,
        )
        comb2_or_incl = CriterionCombination(
            "comb2_or_incl",
            exclude=False,
            operator=op_or,
            category=CohortCategory.POPULATION_INTERVENTION,
        )
        comb3_and_excl = CriterionCombination(
            "comb3_and_excl",
            exclude=True,
            operator=op_and,
            category=CohortCategory.POPULATION_INTERVENTION,
        )
        comb4_or_excl = CriterionCombination(
            "comb4_or_excl",
            exclude=True,
            operator=op_or,
            category=CohortCategory.POPULATION_INTERVENTION,
        )
        # ~(c6 & c7)
        comb3_and_excl.add(c6)
        comb3_and_excl.add(c7)

        # ~(c4 | c5 | ~(c6 & c7))
        comb4_or_excl.add(c4)
        comb4_or_excl.add(c5)
        comb4_or_excl.add(comb3_and_excl)

        # (c2 | ~c3)
        comb2_or_incl.add(c2)
        comb2_or_incl.add(c3)

        # (c1 & (c2 | ~c3) & ~(c4 | c5 | ~(c6 & c7)))
        comb1_and_incl.add(c1)
        comb1_and_incl.add(comb2_or_incl)
        comb1_and_incl.add(comb4_or_excl)

        return (
            comb1_and_incl,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            comb2_or_incl,
            comb3_and_excl,
            comb4_or_excl,
        )

    def test_to_nnf(self, comb):
        comb1_and_incl, c1, c2, c3, c4, c5, c6, c7, *_ = comb

        em = ExecutionMap(comb1_and_incl)
        _, hashmap = em._to_nnf(comb1_and_incl)

        # Check if the hashmap contains the correct criteria
        assert set(hashmap.values()) == {c1, c2, c3, c4, c5, c6, c7}

        # Check if the NNF expression is correct
        nnf_expr = em.nnf()
        assert str(nnf_expr) == "c1 & c6 & c7 & ~c4 & ~c5 & (c2 | ~c3)"

    def test_invalid_nnf(self, comb):
        comb1_and_incl, *_ = comb

        em = ExecutionMap(comb1_and_incl)

        # Test with an invalid NNF expression
        with pytest.raises(ValueError, match="Unknown type"):
            em._nnf = comb
            em.combine(CohortCategory.POPULATION_INTERVENTION)

    def test_cohort_category_matching(self):
        c1 = MockCriterion("c1", category=CohortCategory.POPULATION)
        c2 = MockCriterion("c2", category=CohortCategory.INTERVENTION)
        c3 = MockCriterion("c3", category=CohortCategory.POPULATION_INTERVENTION)
        c1.id, c2.id, c3.id = 1, 2, 3

        op_and = CriterionCombination.Operator(CriterionCombination.Operator.AND)
        comb_p = CriterionCombination(
            "comb_p", exclude=False, operator=op_and, category=CohortCategory.POPULATION
        )
        comb_i = CriterionCombination(
            "comb_i",
            exclude=False,
            operator=op_and,
            category=CohortCategory.INTERVENTION,
        )
        comb_pi = CriterionCombination(
            "comb_pi",
            exclude=False,
            operator=op_and,
            category=CohortCategory.POPULATION_INTERVENTION,
        )
        comb_p.add_all([c1, c2, c3])
        comb_i.add_all([c1, c2, c3])
        comb_pi.add_all([c1, c2, c3])

        nnf_p = ExecutionMap(comb_p).combine(CohortCategory.POPULATION)
        assert len(nnf_p.selects) == 1
        assert nnf_p.compile().params["criterion_id_1"] == 1

        nnf_i = ExecutionMap(comb_i).combine(CohortCategory.INTERVENTION)
        assert len(nnf_i.selects) == 1
        assert nnf_i.compile().params["criterion_id_1"] == 2

        nnf_pi = ExecutionMap(comb_pi).combine(CohortCategory.POPULATION_INTERVENTION)
        assert len(nnf_pi.selects) == 3
        assert nnf_pi.compile().params["criterion_id_1"] == 1
        assert nnf_pi.compile().params["criterion_id_2"] == 2
        assert nnf_pi.compile().params["criterion_id_3"] == 3

    def test_sequential(self, comb):
        comb1_and_incl, c1, c2, c3, c4, c5, c6, c7, *_ = comb

        em = ExecutionMap(comb1_and_incl)
        seq = list(em.sequential())

        # nnf pushes negations into the criteria
        c4.exclude = True
        c5.exclude = True

        # Check if the sequential method returns the correct criteria
        assert set([str(s) for s in seq]) == {str(c) for c in comb[1:8]}

    # todo: reinstanstiate
    def DISABLED_test_combine(self, comb):
        comb1_and_incl, *_ = comb

        em = ExecutionMap(comb1_and_incl)

        # Test with a specific cohort_category
        cohort_category = CohortCategory.POPULATION_INTERVENTION
        combined_sql = em.combine(cohort_category)

        # Check if the combined SQL query contains the correct criteria

        query = "c1 & c6 & c7 & ~c4 & ~c5 & (c2 | ~c3)"
        query = sort_numbers_in_string(query)
        query = query.replace("&", "INTERSECT")
        query = query.replace("|", "UNION")
        query = query.replace(
            "~", ""
        )  # this is handled by the actual selection SQL of the criteria
        query = query.replace(
            "c",
            "SELECT celida.recommendation_result.person_id, celida.recommendation_result.valid_date \nFROM celida.recommendation_result \nWHERE celida.recommendation_result.recommendation_run_id = :run_id AND celida.recommendation_result.criterion_name = :criterion_name_",
        )
        assert query == str(combined_sql)

    def test_invalid_operator(self):
        c1 = MockCriterion("c1")
        c2 = MockCriterion("c2")

        op_atleast = CriterionCombination.Operator(
            CriterionCombination.Operator.AT_LEAST, threshold=1
        )
        comb = CriterionCombination(
            "comb1_and_incl",
            exclude=False,
            operator=op_atleast,
            category=CohortCategory.POPULATION_INTERVENTION,
        )
        comb.add(c1)
        comb.add(c2)

        with pytest.raises(
            NotImplementedError,
            match=re.escape(
                'Operator "Operator(operator=AT_LEAST, threshold=1)" not implemented'
            ),
        ):
            ExecutionMap(comb)
