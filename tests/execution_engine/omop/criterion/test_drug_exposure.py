import pandas as pd
import pendulum
import pytest

from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.combination.logical import (
    LogicalCriterionCombination,
)
from execution_engine.omop.criterion.drug_exposure import DrugExposure
from execution_engine.util.enum import TimeUnit
from execution_engine.util.types import Dosage
from execution_engine.util.value import ValueNumber
from execution_engine.util.value.time import ValueCount
from tests._fixtures.concept import (
    concept_enoxparin,
    concept_heparin_ingredient,
    concept_route_intravenous,
    concept_route_subcutaneous,
    concept_unit_mg,
    concept_unit_mg_kg,
    concepts_heparin_other,
)
from tests._testdata import concepts
from tests.execution_engine.omop.criterion.test_criterion import TestCriterion, date_set
from tests.functions import create_drug_exposure, create_measurement


class TestDrugExposure(TestCriterion):
    @pytest.fixture
    def execute_drug_exposure_criterion(
        self, base_table, db_session, observation_window
    ):
        def _run_drug_exposure(
            ingredient_concept: Concept,
            exclude: bool,
            dose: Dosage | None,
            route: Concept | None,
        ) -> pd.DataFrame:
            criterion = DrugExposure(
                category=CohortCategory.POPULATION,
                ingredient_concept=ingredient_concept,
                dose=dose,
                route=route,
            )
            if exclude:
                criterion = LogicalCriterionCombination.Not(
                    criterion, category=criterion.category
                )

            self.insert_criterion(db_session, criterion, observation_window)

            df = self.fetch_full_day_result(
                db_session,
                pi_pair_id=self.pi_pair_id,
                criterion_id=self.criterion_id,
                category=criterion.category,
            )

            return df

        return _run_drug_exposure

    @staticmethod
    def create_drug_exposure(
        visit_occurrence,
        drug_concept_id,
        start_datetime,
        end_datetime,
        quantity,
        route_concept_id=None,
    ):
        return create_drug_exposure(
            vo=visit_occurrence,
            drug_concept_id=drug_concept_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            quantity=quantity,
            route_concept_id=route_concept_id,
        )

    def perform_test(
        self,
        db_session,
        person_visit,
        execute_drug_exposure_criterion,
        drug_exposures,
        dosage,
        expected,
        route=None,
    ):
        _, vo = person_visit[0]

        # Create drug exposure entries
        for exposure in drug_exposures:
            c = self.create_drug_exposure(
                visit_occurrence=vo,
                drug_concept_id=exposure["drug_concept_id"],
                start_datetime=pendulum.parse(exposure["start_datetime"]),
                end_datetime=pendulum.parse(exposure["end_datetime"]),
                quantity=exposure["quantity"],
                route_concept_id=exposure.get("route_concept_id", None),
            )
            db_session.add(c)

        db_session.commit()

        # Execute the criterion and compare the result with the expected output
        result = execute_drug_exposure_criterion(
            ingredient_concept=concept_heparin_ingredient,
            exclude=False,
            dose=dosage,
            route=route,
        )

        assert set(result["valid_date"].dt.date) == date_set(expected)

    # Test case 1: Single day drug exposure, dose as a single quantity, frequency once per day
    @pytest.mark.parametrize(
        "drug_exposures",
        [
            (
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-04 09:36:24",
                    "end_datetime": "2023-03-04 10:36:24",
                    "quantity": 100,
                },
                # date         qty
                # 2023-03-04 100.0
            )
        ],
    )
    @pytest.mark.parametrize(
        "dosage,expected",
        [
            (
                Dosage(
                    dose=ValueNumber(value=100, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-04"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value=101, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {},
            ),
            (
                Dosage(
                    dose=ValueNumber(value=99, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=99, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-04"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=100, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-04"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=101, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_max=99, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_max=100, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-04"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_max=101, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-04"},
            ),
            (
                Dosage(
                    dose=ValueNumber(
                        value_min=100, value_max=100, unit=concept_unit_mg
                    ),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-04"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=99, value_max=101, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-04"},
            ),
            (
                Dosage(
                    dose=ValueNumber(
                        value_min=0, value_max=10000, unit=concept_unit_mg
                    ),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-04"},
            ),
        ],
    )
    def test_single_day_drug_dose(
        self,
        db_session,
        person_visit,
        execute_drug_exposure_criterion,
        drug_exposures,
        dosage,
        expected,
    ):
        self.perform_test(
            db_session,
            person_visit,
            execute_drug_exposure_criterion,
            drug_exposures,
            dosage,
            expected,
        )

    # Test case 2a: Multi-day drug exposure, dose as a range, frequency once per week (only first week due to
    # quantity of 100 spend of 7 days, making less than 50 per week in the second week)
    @pytest.mark.parametrize(
        "drug_exposures",
        [
            (
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-01 09:36:24+01:00",
                    "end_datetime": "2023-03-08 23:36:24+01:00",
                    "quantity": 100,
                },
                # date         qty
                # 2023-03-01   7.9
                # 2023-03-02  13.2
                # 2023-03-03  13.2
                # 2023-03-04  13.2
                # 2023-03-05  13.2
                # 2023-03-06  13.2
                # 2023-03-07  13.2 --> 87.1 (for week 1)
                # 2023-03-08  13.0 --> 13.0 (for week 2)
            )
        ],
    )
    @pytest.mark.parametrize(
        "dosage,expected",
        [
            (
                Dosage(
                    dose=ValueNumber(
                        value_min=87.0, value_max=150, unit=concept_unit_mg
                    ),
                    frequency=1,
                    interval=TimeUnit.WEEK,
                ),
                {
                    "2023-03-01",
                    "2023-03-02",
                    "2023-03-03",
                    "2023-03-04",
                    "2023-03-05",
                    "2023-03-06",
                    "2023-03-07",
                },
            ),
            (
                Dosage(
                    dose=ValueNumber(
                        value_min=12.9, value_max=86.9, unit=concept_unit_mg
                    ),
                    frequency=1,
                    interval=TimeUnit.WEEK,
                ),
                {
                    "2023-03-08",
                    "2023-03-09",
                    "2023-03-10",
                    "2023-03-11",
                    "2023-03-12",
                    "2023-03-13",
                    "2023-03-14",
                },
            ),
            (
                Dosage(
                    dose=ValueNumber(
                        value_min=12.9, value_max=86.9, unit=concept_unit_mg
                    ),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {
                    "2023-03-02",
                    "2023-03-03",
                    "2023-03-04",
                    "2023-03-05",
                    "2023-03-06",
                    "2023-03-07",
                    "2023-03-08",
                },
            ),
            (
                Dosage(
                    dose=ValueNumber(
                        value_min=12.9, value_max=86.9, unit=concept_unit_mg
                    ),
                    frequency=2,
                    interval=TimeUnit.DAY,
                ),
                {},
            ),
        ],
    )
    def test_multi_day_drug_exposure(
        self,
        db_session,
        person_visit,
        execute_drug_exposure_criterion,
        drug_exposures,
        dosage,
        expected,
    ):
        self.perform_test(
            db_session,
            person_visit,
            execute_drug_exposure_criterion,
            drug_exposures,
            dosage,
            expected,
        )

    # Test case 3: Multiple drug exposures with different drug_concept_ids, quantities, and time ranges
    @pytest.mark.parametrize(
        "drug_exposures",
        [
            (
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-01 09:36:24",
                    "end_datetime": "2023-03-01 10:36:24",
                    "quantity": 100,
                },
                # date         qty
                # 2023-03-01 100.0
                {
                    "drug_concept_id": concept_enoxparin.concept_id,
                    "start_datetime": "2023-03-02 09:36:24",
                    "end_datetime": "2023-03-02 10:36:24",
                    "quantity": 200,
                },
                # date         qty
                # 2023-03-02 200.0
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-03 09:36:24",
                    "end_datetime": "2023-03-03 10:36:24",
                    "quantity": 100,
                },
                # date         qty
                # 2023-03-03 100.0
            )
        ],
    )
    @pytest.mark.parametrize(
        "dosage,expected",
        [
            (
                Dosage(
                    dose=ValueNumber(value_min=100, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-01", "2023-03-03"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_max=99, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {},
            ),
        ],
    )
    def test_multiple_drug_exposures_different_drugs(
        self,
        db_session,
        person_visit,
        execute_drug_exposure_criterion,
        drug_exposures,
        dosage,
        expected,
    ):
        self.perform_test(
            db_session,
            person_visit,
            execute_drug_exposure_criterion,
            drug_exposures,
            dosage,
            expected,
        )

    # Test case 4: Overlapping drug exposure entries, dose as a range
    @pytest.mark.parametrize(
        "drug_exposures",
        [
            (
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-01 09:36:24+01:00",
                    "end_datetime": "2023-03-02 10:36:24+01:00",
                    "quantity": 100,
                },
                # date         qty total
                # 2023-03-01  57.6 57.6
                # 2023-03-02  42.4 128.8
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-02 09:36:24+01:00",
                    "end_datetime": "2023-03-03 10:36:24+01:00",
                    "quantity": 150,
                },
                # date         qty total
                # 2023-03-02  86.4 128.8
                # 2023-03-03  63.6 63.6
            )
        ],
    )
    @pytest.mark.parametrize(
        "dosage,expected",
        [
            (
                Dosage(
                    dose=ValueNumber(value_min=50, value_max=200, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-01", "2023-03-03"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=50, value_max=200, unit=concept_unit_mg),
                    frequency=">=1",
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-01", "2023-03-02", "2023-03-03"},
            ),
            (
                Dosage(
                    dose=ValueNumber(
                        value_min=128.7, value_max=200, unit=concept_unit_mg
                    ),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {},
            ),
            (
                Dosage(
                    dose=ValueNumber(
                        value_min=128.7, value_max=200, unit=concept_unit_mg
                    ),
                    frequency=">=1",
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-02"},
            ),
            (
                Dosage(
                    dose=ValueNumber(
                        value_min=128.7, value_max=200, unit=concept_unit_mg
                    ),
                    frequency=2,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-02"},
            ),
            (
                Dosage(
                    dose=ValueNumber(
                        value_min=128.7, value_max=200, unit=concept_unit_mg
                    ),
                    frequency=3,
                    interval=TimeUnit.DAY,
                ),
                {},
            ),
            (
                Dosage(
                    dose=ValueNumber(
                        value_min=128.8, value_max=200, unit=concept_unit_mg
                    ),
                    frequency=2,
                    interval=TimeUnit.DAY,
                ),
                {},
            ),
        ],
    )
    def test_overlapping_drug_exposures(
        self,
        db_session,
        person_visit,
        execute_drug_exposure_criterion,
        drug_exposures,
        dosage,
        expected,
    ):
        self.perform_test(
            db_session,
            person_visit,
            execute_drug_exposure_criterion,
            drug_exposures,
            dosage,
            expected,
        )

    # Test case 5: Non-overlapping drug exposure entries with different quantities
    @pytest.mark.parametrize(
        "drug_exposures",
        [
            (
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-01 09:36:24+01:00",
                    "end_datetime": "2023-03-01 10:36:24+01:00",
                    "quantity": 100,
                },
                # date         qty
                # 2023-03-01 100.0
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-02 09:36:24+01:00",
                    "end_datetime": "2023-03-02 10:36:24+01:00",
                    "quantity": 200,
                },
                # date         qty
                # 2023-03-02 200.0
            )
        ],
    )
    @pytest.mark.parametrize(
        "dosage,expected",
        [
            (
                Dosage(
                    dose=ValueNumber(value_min=100, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-01", "2023-03-02"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=200, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-02"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=200.1, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {},
            ),
        ],
    )
    def test_non_overlapping_drug_exposures(
        self,
        db_session,
        person_visit,
        execute_drug_exposure_criterion,
        drug_exposures,
        dosage,
        expected,
    ):
        self.perform_test(
            db_session,
            person_visit,
            execute_drug_exposure_criterion,
            drug_exposures,
            dosage,
            expected,
        )

    # Test case 6: Multiple drug exposure entries with different drug_concept_ids
    @pytest.mark.parametrize(
        "drug_exposures",
        [
            (
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-11 09:36:24+01:00",
                    "end_datetime": "2023-03-11 10:36:24+01:00",
                    "quantity": 100,
                },
                # date         qty
                # 2023-03-11 100.0
                {
                    "drug_concept_id": concept_enoxparin.concept_id,
                    "start_datetime": "2023-03-02 09:36:24+01:00",
                    "end_datetime": "2023-03-02 10:36:24+01:00",
                    "quantity": 200,
                },
                # date         qty
                # 2023-03-02 200.0
            )
        ],
    )
    @pytest.mark.parametrize(
        "dosage,expected",
        [
            (
                Dosage(
                    dose=ValueNumber(value_max=200, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-11"},
            ),
        ],
    )
    def test_multiple_exposures_different_drugs(
        self,
        db_session,
        person_visit,
        execute_drug_exposure_criterion,
        drug_exposures,
        dosage,
        expected,
    ):
        self.perform_test(
            db_session,
            person_visit,
            execute_drug_exposure_criterion,
            drug_exposures,
            dosage,
            expected,
        )

    # Test case 7: Drug exposure entries with partial overlap
    @pytest.mark.parametrize(
        "drug_exposures",
        [
            (
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-01 09:36:24+01:00",
                    "end_datetime": "2023-03-03 10:36:24+01:00",
                    "quantity": 100,
                },
                # date         qty total
                # 2023-03-01  29.4  29.4
                # 2023-03-02  49.0  93.1
                # 2023-03-03  21.6  95.1
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-02 09:36:24+01:00",
                    "end_datetime": "2023-03-04 10:36:24+01:00",
                    "quantity": 150,
                },
                # date         qty total
                # 2023-03-02  44.1  93.1
                # 2023-03-03  73.5  95.1
                # 2023-03-04  32.5  32.5
            )
        ],
    )
    @pytest.mark.parametrize(
        "dosage,expected",
        [
            (
                Dosage(
                    dose=ValueNumber(value_min=29.3, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-01", "2023-03-04"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=29.3, unit=concept_unit_mg),
                    frequency=">=1",
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-01", "2023-03-02", "2023-03-03", "2023-03-04"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=29.3, unit=concept_unit_mg),
                    frequency=2,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-02", "2023-03-03"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=93.0, unit=concept_unit_mg),
                    frequency=2,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-02", "2023-03-03"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=94.0, unit=concept_unit_mg),
                    frequency=2,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-03"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=96.0, unit=concept_unit_mg),
                    frequency=2,
                    interval=TimeUnit.DAY,
                ),
                {},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=1.0, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.WEEK,
                ),
                {},  # None because of interval WEEK
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=1.0, unit=concept_unit_mg),
                    frequency=">=1",
                    interval=TimeUnit.WEEK,
                ),
                {
                    "2023-03-01",
                    "2023-03-02",
                    "2023-03-03",
                    "2023-03-04",
                    "2023-03-05",
                    "2023-03-06",
                    "2023-03-07",
                },  # all days in that week because of interval WEEK
            ),
        ],
    )
    def test_drug_exposure_partial_overlap(
        self,
        db_session,
        person_visit,
        execute_drug_exposure_criterion,
        drug_exposures,
        dosage,
        expected,
    ):
        self.perform_test(
            db_session,
            person_visit,
            execute_drug_exposure_criterion,
            drug_exposures,
            dosage,
            expected,
        )

    # Test case 8: Drug exposure entries exactly next to each other
    @pytest.mark.parametrize(
        "drug_exposures",
        [
            (
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-01 09:36:24+01:00",
                    "end_datetime": "2023-03-02 09:36:24+01:00",
                    "quantity": 100,
                },
                # date         qty
                # 2023-03-01  60.0
                # 2023-03-02  40.0
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-02 09:36:24+01:00",
                    "end_datetime": "2023-03-03 09:36:24+01:00",
                    "quantity": 150,
                },
                # date         qty total
                # 2023-03-02  90.0 130.0
                # 2023-03-03  60.0  60.0
            )
        ],
    )
    @pytest.mark.parametrize(
        "dosage,expected",
        [
            (
                Dosage(
                    dose=ValueNumber(value_min=1, value_max=200, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-01", "2023-03-03"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=1, value_max=200, unit=concept_unit_mg),
                    frequency=">=1",
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-01", "2023-03-02", "2023-03-03"},
            ),
            (
                Dosage(
                    dose=ValueNumber(
                        value_min=129.9, value_max=200, unit=concept_unit_mg
                    ),
                    frequency=2,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-02"},
            ),
        ],
    )
    def test_drug_exposure_exactly_next_to_each_other(
        self,
        db_session,
        person_visit,
        execute_drug_exposure_criterion,
        drug_exposures,
        dosage,
        expected,
    ):
        self.perform_test(
            db_session,
            person_visit,
            execute_drug_exposure_criterion,
            drug_exposures,
            dosage,
            expected,
        )

    # Test case 9: Drug exposure entries out of observation window
    @pytest.mark.parametrize(
        "drug_exposures",
        [
            (
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-04-01 09:36:24+01:00",
                    "end_datetime": "2023-04-02 09:36:24+01:00",
                    "quantity": 100,
                },
                # date         qty
                # 2023-04-01  60.0
                # 2023-04-02  40.0
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-04-02 09:36:24+01:00",
                    "end_datetime": "2023-04-03 09:36:24+01:00",
                    "quantity": 150,
                },
                # date         qty total
                # 2023-04-02  90.0 130.0
                # 2023-04-03  60.0  60.0
            )
        ],
    )
    @pytest.mark.parametrize(
        "dosage,expected",
        [
            (
                Dosage(
                    dose=ValueNumber(value_min=0, value_max=200, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {},
            ),
        ],
    )
    def test_drug_exposure_out_of_observation_window(
        self,
        db_session,
        person_visit,
        execute_drug_exposure_criterion,
        drug_exposures,
        dosage,
        expected,
    ):
        self.perform_test(
            db_session,
            person_visit,
            execute_drug_exposure_criterion,
            drug_exposures,
            dosage,
            expected,
        )

    # Test case 10: Drug exposure entries starting before or ending after observation window
    # question:
    # - the drug entry part that is within the DAY of the observation window should count, or really only
    #   the part that is within the observation window? currently the former is implemented
    #   what if the interval is not day but week: should the drug entry part that is within the WEEK of the
    #   observation window count? what is the reference for the first week day?
    @pytest.mark.parametrize(
        "drug_exposures",
        [
            (
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-02-28 12:00:00+01:00",
                    "end_datetime": "2023-03-02 12:00:00+01:00",
                    "quantity": 200,
                },
                # date         qty
                # 2023-02-28  50.0
                # 2023-03-01  40.0 (before start)
                #             60.0 (after start)
                # 2023-03-02  50.0
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-30 12:00:00+02:00",
                    "end_datetime": "2023-04-01 12:00:00+02:00",
                    "quantity": 200,
                },
                # date         qty
                # 2023-03-30  50.0
                # 2023-03-31  59.8 (before end)
                #             40.2 (after end)
                # 2023-04-01  50
            )
        ],
    )
    @pytest.mark.parametrize(
        "dosage,expected",
        [
            (
                Dosage(
                    dose=ValueNumber(value_min=100, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-01", "2023-03-31"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=100, unit=concept_unit_mg),
                    frequency=">=1",
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-01", "2023-03-31"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=50, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-01", "2023-03-02", "2023-03-30", "2023-03-31"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=101, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {},
            ),
        ],
    )
    def test_drug_exposure_overlapping_start_and_end_of_observation_window(
        self,
        db_session,
        person_visit,
        execute_drug_exposure_criterion,
        drug_exposures,
        dosage,
        expected,
    ):
        self.perform_test(
            db_session,
            person_visit,
            execute_drug_exposure_criterion,
            drug_exposures,
            dosage,
            expected,
        )

    # Test case 11: Multiple exposures adding up to an exact number
    @pytest.mark.parametrize(
        "drug_exposures",
        [
            (
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-03 18:51:00+01:00",
                    "end_datetime": "2023-03-12 23:59:00+01:00",
                    "quantity": 796.08,
                },
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-05 00:00:00+01:00",
                    "end_datetime": "2023-03-09 09:36:24+01:00",
                    "quantity": 38018.4,
                },
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-06 12:34:56+01:00",
                    "end_datetime": "2023-03-08 09:36:25+01:00",
                    "quantity": 1620.89,
                },
            )
        ],
    )
    # date            total  n
    # 2023-03-03 	  18.54  1
    # 2023-03-04 	  86.40  1
    # 2023-03-05 	8726.40  2
    # 2023-03-06 	9137.44  3
    # 2023-03-07 	9590.40  3
    # 2023-03-08 	9072.25  3
    # 2023-03-09 	3544.80  2
    # 2023-03-10 	  86.40  1
    # 2023-03-11 	  86.40  1
    # 2023-03-12 	  86.34  1
    @pytest.mark.parametrize(
        "dosage,expected",
        [
            (
                Dosage(
                    dose=ValueNumber(value=18.54, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-03"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value=86.40, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-04", "2023-03-10", "2023-03-11"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value=8726.40, unit=concept_unit_mg),
                    frequency=">=1",
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-05"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value=9137.44, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {},
            ),
            (
                Dosage(
                    dose=ValueNumber(value=9137.44, unit=concept_unit_mg),
                    frequency=">=1",
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-06"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value=9137.44, unit=concept_unit_mg),
                    frequency=3,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-06"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value=9590.40, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {},
            ),
            (
                Dosage(
                    dose=ValueNumber(value=9590.40, unit=concept_unit_mg),
                    frequency=">=1",
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-07"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value=9590.40, unit=concept_unit_mg),
                    frequency=3,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-07"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value=9072.25, unit=concept_unit_mg),
                    frequency=3,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-08"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value=3544.80, unit=concept_unit_mg),
                    frequency=2,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-09"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value=86.34, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-12"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value=10000, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=10.40, unit=concept_unit_mg),
                    frequency=4,
                    interval=TimeUnit.DAY,
                ),
                {},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=10.40, unit=concept_unit_mg),
                    frequency=3,
                    interval=TimeUnit.DAY,
                ),
                {
                    "2023-03-06",
                    "2023-03-07",
                    "2023-03-08",
                },
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=10.40, unit=concept_unit_mg),
                    frequency=2,
                    interval=TimeUnit.DAY,
                ),
                {
                    "2023-03-05",
                    "2023-03-09",
                },
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=10.40, unit=concept_unit_mg),
                    frequency=">=2",
                    interval=TimeUnit.DAY,
                ),
                {
                    "2023-03-05",
                    "2023-03-06",
                    "2023-03-07",
                    "2023-03-08",
                    "2023-03-09",
                },
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=10.40, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {
                    "2023-03-03",
                    "2023-03-04",
                    "2023-03-10",
                    "2023-03-11",
                    "2023-03-12",
                },
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=10.40, unit=concept_unit_mg),
                    frequency=">=1",
                    interval=TimeUnit.DAY,
                ),
                {
                    "2023-03-03",
                    "2023-03-04",
                    "2023-03-05",
                    "2023-03-06",
                    "2023-03-07",
                    "2023-03-08",
                    "2023-03-09",
                    "2023-03-10",
                    "2023-03-11",
                    "2023-03-12",
                },
            ),
        ],
    )
    def test_multiple_exposures_adding_up_to_exact_number(
        self,
        db_session,
        person_visit,
        execute_drug_exposure_criterion,
        drug_exposures,
        dosage,
        expected,
    ):
        self.perform_test(
            db_session,
            person_visit,
            execute_drug_exposure_criterion,
            drug_exposures,
            dosage,
            expected,
        )

    # Test case 12: non-ingredient drug concepts
    @pytest.mark.parametrize(
        "drug_exposures",
        [
            (
                {
                    "drug_concept_id": concept1.concept_id,
                    "start_datetime": "2023-03-01 00:00:00+01:00",
                    "end_datetime": "2023-03-03 12:00:00+01:00",
                    "quantity": 200,
                },
                {
                    "drug_concept_id": concept2.concept_id,
                    "start_datetime": "2023-03-02 00:00:00+01:00",
                    "end_datetime": "2023-03-05 00:00:00+01:00",
                    "quantity": 201,
                },
            )
            for concept1, concept2 in list(
                zip(concepts_heparin_other, concepts_heparin_other[1:])
            )
        ],
    )
    # date          qty     n
    # 2023-03-01 	 80.0 	1
    # 2023-03-02 	147.0 	2
    # 2023-03-03 	107.0 	2
    # 2023-03-04 	 67.0 	1
    # 2023-03-05 	  0.0 	1
    @pytest.mark.parametrize(
        "dosage,expected",
        [
            (
                Dosage(
                    dose=ValueNumber(value=80, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-01"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value=147, unit=concept_unit_mg),
                    frequency=2,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-02"},
            ),
            (
                Dosage(
                    dose=ValueNumber(value_min=67, unit=concept_unit_mg),
                    frequency=2,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-02", "2023-03-03"},
            ),
        ],
    )
    def test_non_ingredient_drug_concepts(
        self,
        db_session,
        person_visit,
        execute_drug_exposure_criterion,
        drug_exposures,
        dosage,
        expected,
    ):
        self.perform_test(
            db_session,
            person_visit,
            execute_drug_exposure_criterion,
            drug_exposures,
            dosage,
            expected,
        )

    @pytest.mark.parametrize(
        "drug_exposures",
        [
            (
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-04 09:36:24+01:00",
                    "end_datetime": "2023-03-04 09:36:24+01:00",
                    "quantity": 100,
                },
                # date         qty
                # 2023-03-04 100.0
            )
        ],
    )
    @pytest.mark.parametrize(
        "dosage,expected",
        [
            (
                Dosage(
                    dose=ValueNumber(value=100, unit=concept_unit_mg),
                    frequency=1,
                    interval=TimeUnit.DAY,
                ),
                {"2023-03-04"},
            ),
        ],
    )
    def test_zero_second_dosage(
        self,
        db_session,
        person_visit,
        execute_drug_exposure_criterion,
        drug_exposures,
        dosage,
        expected,
    ):
        self.perform_test(
            db_session,
            person_visit,
            execute_drug_exposure_criterion,
            drug_exposures,
            dosage,
            expected,
        )

    @pytest.mark.parametrize(
        "test_cases",
        [
            [
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-01 09:36:24+01:00",
                    "end_datetime": "2023-03-03 10:36:24+01:00",
                    "quantity": 1000,
                    "expected": {"2023-03-01", "2023-03-02", "2023-03-03"},
                },
                # date         qty
                # 2023-03-01 100.0
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-02 09:36:24+01:00",
                    "end_datetime": "2023-03-04 10:36:24+01:00",
                    "quantity": 2000,
                    "expected": {"2023-03-02", "2023-03-03", "2023-03-04"},
                },
                # date         qty
                # 2023-03-02 200.0
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-04 09:36:24+01:00",
                    "end_datetime": "2023-03-06 10:36:24+01:00",
                    "quantity": 1000,
                    "expected": {"2023-03-04", "2023-03-05", "2023-03-06"},
                },
                # date         qty
                # 2023-03-03 100.0
            ]
        ],
    )
    @pytest.mark.parametrize(
        "dosage",
        [
            Dosage(
                dose=ValueNumber(value_min=100, unit=concept_unit_mg),
                frequency=1,
                interval=TimeUnit.DAY,
            )
        ],
    )
    def test_drug_exposure_multiple_person(
        self,
        person_visit,
        db_session,
        execute_drug_exposure_criterion,
        test_cases,
        dosage,
    ):
        vos = [pv[1] for pv in person_visit]

        for vo, exposure in zip(vos, test_cases):
            c = self.create_drug_exposure(
                visit_occurrence=vo,
                drug_concept_id=exposure["drug_concept_id"],
                start_datetime=pendulum.parse(exposure["start_datetime"]),
                end_datetime=pendulum.parse(exposure["end_datetime"]),
                quantity=exposure["quantity"],
            )
            db_session.add(c)

        db_session.commit()

        # Execute the criterion and compare the result with the expected output
        result = execute_drug_exposure_criterion(
            ingredient_concept=concept_heparin_ingredient,
            exclude=False,
            dose=dosage,
            route=None,
        )

        for vo, exposure in zip(vos, test_cases):
            expected = date_set(exposure["expected"])
            actual = set(
                result.query(f"person_id == {vo.person_id}")["valid_date"].dt.date
            )

            assert actual == expected

    @pytest.mark.parametrize(
        "drug_exposures",
        [
            [
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-01 09:36:24+01:00",
                    "end_datetime": "2023-03-03 10:36:24+01:00",
                    "quantity": 1000,
                    "route_concept_id": concepts.ROUTE_SUBCUTANEOUS,
                },
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-02 09:36:24+01:00",
                    "end_datetime": "2023-03-04 10:36:24+01:00",
                    "quantity": 2000,
                    "route_concept_id": concepts.ROUTE_INTRAVENOUS,
                },
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-04 09:36:24+01:00",
                    "end_datetime": "2023-03-06 10:36:24+01:00",
                    "quantity": 1000,
                    "route_concept_id": concepts.ROUTE_INTRAVENOUS,
                },
            ]
        ],
    )
    @pytest.mark.parametrize(
        "dosage",
        [
            Dosage(
                dose=ValueNumber(value_min=100, unit=concept_unit_mg),
                frequency=ValueCount(value_min=1),
                interval=TimeUnit.DAY,
            )
        ],
    )
    @pytest.mark.parametrize(
        "route,expected",
        [
            (concept_route_subcutaneous, {"2023-03-01", "2023-03-02", "2023-03-03"}),
            (
                concept_route_intravenous,
                {"2023-03-02", "2023-03-03", "2023-03-04", "2023-03-05", "2023-03-06"},
            ),
        ],
    )
    def test_drug_exposure_routes(
        self,
        person_visit,
        db_session,
        execute_drug_exposure_criterion,
        drug_exposures,
        dosage,
        route,
        expected,
    ):
        self.perform_test(
            db_session,
            person_visit,
            execute_drug_exposure_criterion,
            drug_exposures,
            dosage,
            expected,
            route,
        )

    @pytest.mark.parametrize(
        "drug_exposures",
        [
            [
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-01 09:36:24+01:00",
                    "end_datetime": "2023-03-01 10:36:24+01:00",
                    "quantity": 63 * 100,
                    "route_concept_id": concepts.ROUTE_SUBCUTANEOUS,
                },
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-02 09:36:24+01:00",
                    "end_datetime": "2023-03-02 10:36:24+01:00",
                    "quantity": 70 * 100,
                    "route_concept_id": concepts.ROUTE_INTRAVENOUS,
                },
                {
                    "drug_concept_id": concept_heparin_ingredient.concept_id,
                    "start_datetime": "2023-03-04 09:36:24+01:00",
                    "end_datetime": "2023-03-04 10:36:24+01:00",
                    "quantity": 110 * 100,
                    "route_concept_id": concepts.ROUTE_INTRAVENOUS,
                },
            ]
        ],
    )
    @pytest.mark.parametrize(
        "dosage",
        [
            Dosage(
                dose=ValueNumber(value=100, unit=concept_unit_mg_kg),
                frequency=ValueCount(value_min=1),
                interval=TimeUnit.DAY,
            )
        ],
    )
    @pytest.mark.parametrize(
        "weight_kg,expected",
        [
            (62, {}),
            (
                63,
                {
                    "2023-03-01",
                },
            ),
            (70, {"2023-03-02"}),
            (
                110,
                {
                    "2023-03-04",
                },
            ),
        ],
    )
    def test_drug_exposure_weight_related(
        self,
        person_visit,
        db_session,
        execute_drug_exposure_criterion,
        drug_exposures,
        dosage,
        weight_kg,
        expected,
    ):
        _, vo = person_visit[0]
        weight_measurement = create_measurement(
            vo=vo,
            measurement_concept_id=concepts.BODY_WEIGHT,
            measurement_datetime=pendulum.parse("2023-03-01 09:36:24+01:00"),
            value_as_number=weight_kg,
        )

        db_session.add(weight_measurement)

        self.perform_test(
            db_session,
            person_visit,
            execute_drug_exposure_criterion,
            drug_exposures,
            dosage,
            expected,
        )
