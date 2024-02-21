import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tests.recommendation.utils.expression_parsing import criteria_combination_str_to_df


class TestCriteriaCombinationStrToDf:
    def test_valid_input_with_mixed_criteria(self):
        criteria_str = "Criterion1> ?Criterion2< !Criterion3="
        expected_columns = pd.MultiIndex.from_tuples(
            [("Criterion1", ">"), ("Criterion2", "<"), ("Criterion3", "=")],
            names=["criterion", "comparator"],
        )
        expected_df = pd.DataFrame(
            {
                ("Criterion1", ">"): [True, True],
                ("Criterion2", "<"): [True, False],
                ("Criterion3", "="): [False, False],
            },
            columns=expected_columns,
        )

        result_df = criteria_combination_str_to_df(criteria_str)
        assert_frame_equal(result_df, expected_df)

    def test_valid_input_with_only_always_present_criteria(self):
        criteria_str = "Criterion1> Criterion2< Criterion3="
        expected_columns = pd.MultiIndex.from_tuples(
            [("Criterion1", ">"), ("Criterion2", "<"), ("Criterion3", "=")],
            names=["criterion", "comparator"],
        )
        expected_df = pd.DataFrame(
            {
                ("Criterion1", ">"): [True],
                ("Criterion2", "<"): [True],
                ("Criterion3", "="): [True],
            },
            columns=expected_columns,
        )

        result_df = criteria_combination_str_to_df(criteria_str)
        assert_frame_equal(result_df, expected_df)

    def test_valid_input_with_only_optional_criteria(self):
        criteria_str = "?Criterion1> ?Criterion2< ?Criterion3="
        expected_columns = pd.MultiIndex.from_tuples(
            [("Criterion1", ">"), ("Criterion2", "<"), ("Criterion3", "=")],
            names=["criterion", "comparator"],
        )
        expected_df = pd.DataFrame(
            {
                ("Criterion1", ">"): [
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                ],
                ("Criterion2", "<"): [
                    True,
                    True,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                ],
                ("Criterion3", "="): [
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                ],
            },
            columns=expected_columns,
        )

        result_df = criteria_combination_str_to_df(criteria_str)
        assert_frame_equal(result_df, expected_df)

    def test_invalid_criterion_name(self):
        criteria_str = "InvalidCriterion?"
        with pytest.raises(ValueError) as exc_info:
            criteria_combination_str_to_df(criteria_str)
        assert "Invalid criterion name" in str(exc_info.value)

    def test_invalid_condition(self):
        criteria_str = "*Criterion1>"
        with pytest.raises(ValueError) as exc_info:
            criteria_combination_str_to_df(criteria_str)
        assert "Invalid c" in str(exc_info.value)

    def test_missing_comparator(self):
        criteria_str = "Criterion"
        with pytest.raises(ValueError) as exc_info:
            criteria_combination_str_to_df(criteria_str)
        assert "Invalid criterion name (missing comparator)" in str(exc_info.value)

    def test_invalid_string(self):
        criteria_str = "?COVID19 ?HEPARINOID_ALLERGY?DALTEPARIN="
        with pytest.raises(ValueError) as exc_info:
            criteria_combination_str_to_df(criteria_str)
        assert "Invalid criterion name" in str(exc_info.value)
