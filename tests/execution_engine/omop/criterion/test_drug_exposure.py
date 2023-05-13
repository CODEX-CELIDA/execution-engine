import pandas as pd
import pendulum
import pytest

from execution_engine.clients import omopdb
from execution_engine.constants import CohortCategory
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.drug_exposure import DrugExposure
from execution_engine.util import ValueNumber
from tests.execution_engine.omop.criterion.test_criterion import TestCriterion
from tests.functions import create_drug_exposure


class TestDrugExposure(TestCriterion):

    concept_unit_mg = Concept(
        concept_id=8576,
        concept_name="milligram",
        domain_id="Unit",
        vocabulary_id="UCUM",
        concept_class_id="Unit",
        standard_concept="S",
        concept_code="mg",
        invalid_reason=None,
    )

    concept_heparin_ingredient = Concept(
        concept_id=1367571,
        concept_name="heparin",
        domain_id="Drug",
        vocabulary_id="RxNorm",
        concept_class_id="Ingredient",
        standard_concept="S",
        concept_code="5224",
        invalid_reason=None,
    )

    concepts_heparin_other = [
        Concept(
            concept_id=995426,
            concept_name="101000 MG heparin 0.6 UNT/MG Topical Gel by Axicorp",
            domain_id="Drug",
            vocabulary_id="RxNorm Extension",
            concept_class_id="Marketed Product",
            standard_concept="S",
            concept_code="OMOP4821932",
            invalid_reason=None,
        ),
        Concept(
            concept_id=1367697,
            concept_name="heparin calcium 25000 UNT/ML",
            domain_id="Drug",
            vocabulary_id="RxNorm",
            concept_class_id="Clinical Drug Comp",
            standard_concept="S",
            concept_code="849698",
            invalid_reason=None,
        ),
        Concept(
            concept_id=44216409,
            concept_name="200000 MG heparin 1.8 UNT/MG Topical Gel [Heparin Ratiopharm]",
            domain_id="Drug",
            vocabulary_id="RxNorm Extension",
            concept_class_id="Quant Branded Drug",
            standard_concept="S",
            concept_code="OMOP3093132",
            invalid_reason=None,
        ),
        Concept(
            concept_id=44507578,
            concept_name="6 ML heparin sodium, porcine 100 UNT/ML Prefilled Syringe",
            domain_id="Drug",
            vocabulary_id="RxNorm",
            concept_class_id="Quant Clinical Drug",
            standard_concept="S",
            concept_code="1442414",
            invalid_reason=None,
        ),
        Concept(
            concept_id=44215905,
            concept_name="101000 MG Arnica extract 0.1 MG/MG / guaiazulene 0.00005 MG/MG / heparin 0.04 UNT/MG / Lecithin 0.01 MG/MG / Matricaria chamomilla flowering top oil 0.00005 MG/MG Topical Gel [Arnica Kneipp]",
            domain_id="Drug",
            vocabulary_id="RxNorm Extension",
            concept_class_id="Quant Branded Drug",
            standard_concept="S",
            concept_code="OMOP3092628",
            invalid_reason=None,
        ),
    ]

    concept_enoxparin_ingredient = Concept(
        concept_id=995271,
        concept_name="0.4 ML Enoxaparin 100 MG/ML Injectable Solution [Inhixa] by Emra-Med",
        domain_id="Drug",
        vocabulary_id="RxNorm Extension",
        concept_class_id="Marketed Product",
        standard_concept="S",
        concept_code="OMOP4821780",
        invalid_reason=None,
    )

    @staticmethod
    def drug_concepts(concept: Concept):
        """
        Given a drug concept, find the ingredient of that drug and return all drugs that contain that ingredient.
        """

        ingredient = omopdb.drug_vocabulary_to_ingredient(
            concept.vocabulary_id, concept.concept_code  # type: ignore
        )

        drugs = omopdb.drugs_by_ingredient(ingredient.concept_id, with_unit=True)  # type: ignore

        return {"ingredient": ingredient, "drugs": drugs}

    @pytest.fixture
    def execute_drug_exposure_criterion(self, base_table, db_session, person_visit):
        def _run_drug_exposure(
            ingredient_concept: Concept,
            exclude: bool,
            dose: ValueNumber | None,
            frequency: int | None,
            interval: str | None,
            route: Concept | None,
        ) -> pd.DataFrame:
            p, vo = person_visit

            criterion = DrugExposure(
                name="test",
                exclude=exclude,
                category=CohortCategory.POPULATION,
                drug_concepts=omopdb.drugs_by_ingredient(
                    ingredient_concept.concept_id, with_unit=True
                )["drug_concept_id"].tolist(),
                ingredient_concept=ingredient_concept,
                dose=dose,
                frequency=frequency,
                interval=interval,
                route=route,
            )

            query = criterion.sql_generate(base_table=base_table)

            df = pd.read_sql(
                query,
                db_session.connection(),
                params={
                    "observation_start_datetime": vo.visit_start_datetime,
                    "observation_end_datetime": vo.visit_end_datetime,
                },
            )
            df["valid_date"] = pd.to_datetime(df["valid_date"])

            return df

        return _run_drug_exposure

    @staticmethod
    def create_drug_exposure(
        visit_occurrence, drug_concept_id, start_datetime, end_datetime, quantity
    ):
        return create_drug_exposure(
            vo=visit_occurrence,
            drug_concept_id=drug_concept_id,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            quantity=quantity,
        )

    def Xtest_drug_exposure(
        self,
        person_visit,
        db_session,
        db_drug_exposures,
        execute_drug_exposure_criterion,
        criterion_concept,
        criterion_dose,
        criterion_interval,
        criterion_exclude,
    ):
        _, vo = person_visit

        times = [
            (pendulum.parse(start), pendulum.parse(end))
            for start, end, _, _ in db_drug_exposures
        ]
        drugs = [drug for _, _, drug, _ in db_drug_exposures]
        quantities = [quantity for _, _, _, quantity in db_drug_exposures]
        # drug_ingredients = [self.drug_concepts(drug) for drug in drugs]

        for (start, end), drug, quantity in zip(times, drugs, quantities):
            c = self.create_drug_exposure(
                visit_occurrence=vo,
                drug_concept_id=drug.concept_id,
                start_datetime=start,
                end_datetime=end,
                quantity=quantity,
            )
            db_session.add(c)

        db_session.commit()

        # run criterion against db
        df = execute_drug_exposure_criterion(
            ingredient_concept=criterion_concept,
            exclude=criterion_exclude,
            dose=criterion_dose,
            frequency=criterion_interval,
            interval=criterion_interval,
        )

        # todo : must find out on which days the criterion matches (given the drug dosing, interval, route, ...)
        criterion_matches = criterion_concept == self.concept_heparin_ingredient

        if criterion_matches:
            valid_dates = self.date_points(times)
        else:
            valid_dates = set()

        if criterion_exclude:
            valid_dates = self.invert_date_points(
                start_datetime=vo.visit_start_datetime,
                end_datetime=vo.visit_end_datetime,
                subtract=valid_dates,
            )

        assert set(df["valid_date"].dt.date) == valid_dates

    @pytest.mark.parametrize(
        "drug_exposures,dose,frequency,interval,expected",
        [
            # Test case 1: Single day drug exposure, dose as a single quantity, frequency once per day
            (
                [
                    {
                        "drug_concept_id": concept_heparin_ingredient.concept_id,
                        "start_datetime": "2023-03-04 09:36:24",
                        "end_datetime": "2023-03-04 10:36:24",
                        "quantity": 100,
                    },
                ],
                ValueNumber(value="100", unit=concept_unit_mg),
                1,
                "day",
                {"2023-03-04"},
            ),
            # Test case 2a: Multi-day drug exposure, dose as a range, frequency once per week (only first week due to
            # quantity of 100 spend of 7 days, making less than 50 per week in the second week)
            (
                [
                    {
                        "drug_concept_id": concept_heparin_ingredient.concept_id,
                        "start_datetime": "2023-03-01 09:36:24",
                        "end_datetime": "2023-03-08 23:36:24",
                        "quantity": 100,
                    },
                ],
                ValueNumber(value_min=50, value_max=150, unit=concept_unit_mg),
                1,
                "week",
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
            # Test case 2b: Multi-day drug exposure, dose as a range, frequency once per week (only second week due to
            # quantity of 600 spend on 7 days, making more than 50 in the first week)
            (
                [
                    {
                        "drug_concept_id": concept_heparin_ingredient.concept_id,
                        "start_datetime": "2023-03-01 09:36:24",
                        "end_datetime": "2023-03-08 23:36:24",
                        "quantity": 600,
                    },
                ],
                ValueNumber(value_min=50, value_max=150, unit=concept_unit_mg),
                1,
                "week",
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
            # Test case 3: Multiple drug exposures with different drug_concept_ids, quantities, and time ranges
            (
                [
                    {
                        "drug_concept_id": concept_heparin_ingredient.concept_id,
                        "start_datetime": "2023-03-01 09:36:24",
                        "end_datetime": "2023-03-01 10:36:24",
                        "quantity": 100,
                    },
                    {
                        "drug_concept_id": concept_enoxparin_ingredient.concept_id,
                        "start_datetime": "2023-03-02 09:36:24",
                        "end_datetime": "2023-03-02 10:36:24",
                        "quantity": 200,
                    },
                    {
                        "drug_concept_id": concept_heparin_ingredient.concept_id,
                        "start_datetime": "2023-03-03 09:36:24",
                        "end_datetime": "2023-03-03 10:36:24",
                        "quantity": 100,
                    },
                ],
                ValueNumber(value=100, unit=concept_unit_mg),
                1,
                "day",
                {"2023-03-01", "2023-03-03"},
            ),
            # Test case 4: Overlapping drug exposure entries, dose as a range
            (
                [
                    {
                        "drug_concept_id": concept_heparin_ingredient.concept_id,
                        "start_datetime": "2023-03-01 09:36:24",
                        "end_datetime": "2023-03-02 10:36:24",
                        "quantity": 100,
                    },
                    {
                        "drug_concept_id": concept_heparin_ingredient.concept_id,
                        "start_datetime": "2023-03-02 09:36:24",
                        "end_datetime": "2023-03-03 10:36:24",
                        "quantity": 150,
                    },
                ],
                ValueNumber(value_min=50, value_max=200, unit=concept_unit_mg),
                2,
                "day",
                {"2023-03-02"},
            ),
            # Test case 5: Non-overlapping drug exposure entries with different quantities
            (
                [
                    {
                        "drug_concept_id": concept_heparin_ingredient.concept_id,
                        "start_datetime": "2023-03-01 09:36:24",
                        "end_datetime": "2023-03-01 10:36:24",
                        "quantity": 100,
                    },
                    {
                        "drug_concept_id": concept_heparin_ingredient.concept_id,
                        "start_datetime": "2023-03-02 09:36:24",
                        "end_datetime": "2023-03-02 10:36:24",
                        "quantity": 200,
                    },
                ],
                ValueNumber(value_min=100, unit=concept_unit_mg),
                1,
                "day",
                {"2023-03-01", "2023-03-02"},
            ),
            # Test case 6: Multiple drug exposure entries with different drug_concept_ids
            (
                [
                    {
                        "drug_concept_id": concept_heparin_ingredient.concept_id,
                        "start_datetime": "2023-03-11 09:36:24",
                        "end_datetime": "2023-03-11 10:36:24",
                        "quantity": 100,
                    },
                    {
                        "drug_concept_id": concept_enoxparin_ingredient.concept_id,
                        "start_datetime": "2023-03-02 09:36:24",
                        "end_datetime": "2023-03-02 10:36:24",
                        "quantity": 200,
                    },
                ],
                ValueNumber(value_max=100, unit=concept_unit_mg),
                1,
                "day",
                {"2023-03-11"},
            ),
            # Test case 7: Drug exposure entries with partial overlap
            (
                [
                    {
                        "drug_concept_id": concept_heparin_ingredient.concept_id,
                        "start_datetime": "2023-03-01 09:36:24",
                        "end_datetime": "2023-03-03 10:36:24",
                        "quantity": 100,
                    },
                    {
                        "drug_concept_id": concept_heparin_ingredient.concept_id,
                        "start_datetime": "2023-03-02 09:36:24",
                        "end_datetime": "2023-03-04 10:36:24",
                        "quantity": 150,
                    },
                ],
                ValueNumber(value_min=30, unit=concept_unit_mg),
                2,
                "day",
                {"2023-03-02", "2023-03-03"},
            ),
            # Test case 8: Drug exposure entries exactly next to each other
            (
                [
                    {
                        "drug_concept_id": concept_heparin_ingredient.concept_id,
                        "start_datetime": "2023-03-01 09:36:24",
                        "end_datetime": "2023-03-02 09:36:24",
                        "quantity": 100,
                    },
                    {
                        "drug_concept_id": concept_heparin_ingredient.concept_id,
                        "start_datetime": "2023-03-02 09:36:24",
                        "end_datetime": "2023-03-03 09:36:24",
                        "quantity": 150,
                    },
                ],
                ValueNumber(value_min=100, value_max=200, unit=concept_unit_mg),
                2,
                "day",
                {"2023-03-02"},
            ),
            # Test case 9: Drug exposure entries out of observation window
            (
                [
                    {
                        "drug_concept_id": concept_heparin_ingredient.concept_id,
                        "start_datetime": "2023-04-01 09:36:24",
                        "end_datetime": "2023-04-02 09:36:24",
                        "quantity": 100,
                    },
                    {
                        "drug_concept_id": concept_heparin_ingredient.concept_id,
                        "start_datetime": "2023-04-02 09:36:24",
                        "end_datetime": "2023-04-03 09:36:24",
                        "quantity": 150,
                    },
                ],
                ValueNumber(value_min=100, value_max=200, unit=concept_unit_mg),
                1,
                "day",
                set(),
            ),
        ],
    )
    def test_execute_drug_exposure_criterion(
        self,
        db_session,
        person_visit,
        execute_drug_exposure_criterion,
        drug_exposures,
        dose,
        frequency,
        interval,
        expected,
    ):
        _, vo = person_visit
        # Create drug exposure entries
        for exposure in drug_exposures:
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
            ingredient_concept=self.concept_heparin_ingredient,
            exclude=False,
            dose=dose,
            frequency=frequency,
            interval=interval,
            route=None,
        )

        assert set(result["valid_date"].dt.date) == set(
            pendulum.parse(e).date() for e in expected
        )
