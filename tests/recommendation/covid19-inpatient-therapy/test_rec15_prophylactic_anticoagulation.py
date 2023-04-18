import datetime

import pandas as pd
import pytest
from sqlalchemy.orm import sessionmaker

from execution_engine.omop.db.base import (  # noqa: F401 -- do not remove - needed for sqlalchemy to work
    Base,
    metadata,
)
from tests import concepts
from tests.functions import generate_dataframe, to_extended


@pytest.mark.recommendation
class TestRecommendation15ProphylacticAnticoagulation:
    @pytest.fixture
    def visit_start_date(self) -> datetime.datetime:
        visit_start_date = datetime.datetime(2023, 3, 1)
        return visit_start_date

    @pytest.fixture
    def visit_end_date(self) -> datetime.datetime:
        visit_end_date = datetime.datetime(2023, 3, 31)
        return visit_end_date

    @pytest.fixture
    def population_intervention(self) -> dict:
        population = {
            "COVID19": concepts.COVID19,
            "VENOUS_THROMBOSIS": concepts.VENOUS_THROMBOSIS,
            "HIT2": concepts.HEPARIN_INDUCED_THROMBOCYTOPENIA_WITH_THROMBOSIS,
            "HEPARIN_ALLERGY": concepts.ALLERGY_HEPARIN,
            "HEPARINOID_ALLERGY": concepts.ALLERGY_HEPARINOID,
            "THROMBOCYTOPENIA": concepts.THROMBOCYTOPENIA,
        }

        interventions = {
            "DALTEPARIN": concepts.DALTEPARIN,
            "ENOXAPARIN": concepts.ENOXAPARIN,
            "NADROPARIN_LOW_WEIGHT": concepts.NADROPARIN,
            "NADROPARIN_HIGH_WEIGHT": concepts.NADROPARIN,
            "CERTOPARIN": concepts.CERTOPARIN,
            "FONDAPARINUX": concepts.FONDAPARINUX,
        }

        return population | interventions

    @pytest.fixture
    def person_combinations(
        self,
        visit_start_date: datetime.datetime,
        visit_end_date: datetime.datetime,
        population_intervention: dict,
        run_slow_tests: bool,
    ) -> pd.DataFrame:

        df = generate_dataframe(population_intervention)

        # Remove invalid combinations
        idx_invalid = df["NADROPARIN_HIGH_WEIGHT"] & df["NADROPARIN_LOW_WEIGHT"]
        df = df[~idx_invalid].copy()

        if not run_slow_tests:
            df = df.iloc[:20]

        return df

    @pytest.fixture
    def criteria_extended(
        self,
        insert_criteria: dict,
        criteria: pd.DataFrame,
        population_intervention: dict,
        visit_start_date: datetime.datetime,
        visit_end_date: datetime.datetime,
    ) -> pd.DataFrame:

        idx_static = criteria["static"]
        criteria.loc[idx_static, "start_datetime"] = pd.to_datetime(visit_start_date)
        criteria.loc[idx_static, "end_datetime"] = pd.to_datetime(visit_end_date)
        df = to_extended(
            criteria[["person_id", "concept", "start_datetime", "end_datetime"]],
            observation_start_date=pd.to_datetime(visit_start_date),
            observation_end_date=pd.to_datetime(visit_end_date),
        )
        df.loc[
            :, [c for c in population_intervention.keys() if c not in df.columns]
        ] = False

        df["p_AntithromboticProphylaxisWithLWMH"] = (
            df["COVID19"]
            & ~df["VENOUS_THROMBOSIS"]
            & ~(
                df["HIT2"]
                | df["HEPARIN_ALLERGY"]
                | df["HEPARINOID_ALLERGY"]
                | df["THROMBOCYTOPENIA"]
            )
        )
        df["p_AntithromboticProphylaxisWithFondaparinux"] = (
            df["COVID19"]
            & ~df["VENOUS_THROMBOSIS"]
            & (
                df["HIT2"]
                | df["HEPARIN_ALLERGY"]
                | df["HEPARINOID_ALLERGY"]
                | df["THROMBOCYTOPENIA"]
            )
        )
        df["p_NoAntithromboticProphylaxis"] = df["COVID19"] & df["VENOUS_THROMBOSIS"]

        df["i_AntithromboticProphylaxisWithLWMH"] = (
            df["DALTEPARIN"]
            | df["ENOXAPARIN"]
            | df["NADROPARIN_LOW_WEIGHT"]
            | df["NADROPARIN_HIGH_WEIGHT"]
            | df["CERTOPARIN"]
        )
        df["i_AntithromboticProphylaxisWithFondaparinux"] = df["FONDAPARINUX"]
        df["i_NoAntithromboticProphylaxis"] = ~(
            df["DALTEPARIN"]
            | df["ENOXAPARIN"]
            | df["NADROPARIN_LOW_WEIGHT"]
            | df["NADROPARIN_HIGH_WEIGHT"]
            | df["CERTOPARIN"]
            | df["FONDAPARINUX"]
        )

        df["p_i_AntithromboticProphylaxisWithLWMH"] = (
            df["p_AntithromboticProphylaxisWithLWMH"]
            & df["i_AntithromboticProphylaxisWithLWMH"]
        )
        df["p_i_AntithromboticProphylaxisWithFondaparinux"] = (
            df["p_AntithromboticProphylaxisWithFondaparinux"]
            & df["i_AntithromboticProphylaxisWithFondaparinux"]
        )
        df["p_i_NoAntithromboticProphylaxis"] = (
            df["p_NoAntithromboticProphylaxis"] & df["i_NoAntithromboticProphylaxis"]
        )

        df["p"] = (
            df["p_AntithromboticProphylaxisWithLWMH"]
            | df["p_AntithromboticProphylaxisWithFondaparinux"]
            | df["p_NoAntithromboticProphylaxis"]
        )
        df["i"] = (
            df["i_AntithromboticProphylaxisWithLWMH"]
            | df["i_AntithromboticProphylaxisWithFondaparinux"]
            | df["i_NoAntithromboticProphylaxis"]
        )

        df["p_i"] = (
            df["p_i_AntithromboticProphylaxisWithLWMH"]
            | df["p_i_AntithromboticProphylaxisWithFondaparinux"]
            | df["p_i_NoAntithromboticProphylaxis"]
        )

        return df

    def test_recommendation_15_prophylactic_anticoagulation(
        self,
        db_session: sessionmaker,
        criteria_extended: pd.DataFrame,
        visit_start_date: datetime.datetime,
        visit_end_date: datetime.datetime,
    ) -> None:
        import itertools

        from execution_engine.clients import omopdb
        from execution_engine.execution_engine import ExecutionEngine

        base_url = (
            "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"
        )
        recommendation_url = (
            "covid19-inpatient-therapy/recommendation/prophylactic-anticoagulation"
        )

        start_datetime = visit_start_date - datetime.timedelta(days=3)
        end_datetime = visit_end_date + datetime.timedelta(days=3)

        e = ExecutionEngine(verbose=False)

        print(recommendation_url)
        cdd = e.load_recommendation(base_url + recommendation_url, force_reload=False)

        e.execute(cdd, start_datetime=start_datetime, end_datetime=end_datetime)

        df_result = omopdb.query(
            """
        SELECT * FROM celida.recommendation_result
        WHERE

             criterion_name is null
        """
        )
        df_result["valid_date"] = pd.to_datetime(df_result["valid_date"])
        df_result["name"] = df_result["cohort_category"].map(
            {
                "INTERVENTION": "db_i_",
                "POPULATION": "db_p_",
                "POPULATION_INTERVENTION": "db_p_i_",
            }
        ) + df_result["recommendation_plan_name"].fillna("")

        df_result = df_result.rename(columns={"valid_date": "date"})
        df_result = df_result.pivot_table(
            columns="name",
            index=["person_id", "date"],
            values="recommendation_results_id",
            aggfunc=len,
            fill_value=0,
        ).astype(bool)

        plan_names = [
            "AntithromboticProphylaxisWithLWMH",
            "AntithromboticProphylaxisWithFondaparinux",
            "NoAntithromboticProphylaxis",
        ]

        cols = ["_".join(i) for i in itertools.product(["p", "i", "p_i"], plan_names)]
        cols_db = [
            "_".join(i)
            for i in itertools.product(["db_p", "db_i", "db_p_i"], plan_names)
        ]

        m = criteria_extended.set_index(["person_id", "date"])[cols]
        m = m.join(df_result)

        m.loc[:, [c for c in cols_db if c not in m.columns]] = False

        for plan in plan_names:
            m[f"p_{plan}_eq"] = m[f"p_{plan}"] == m[f"db_p_{plan}"]
            m[f"i_{plan}_eq"] = m[f"i_{plan}"] == m[f"db_i_{plan}"]
            m[f"p_i_{plan}_eq"] = m[f"p_i_{plan}"] == m[f"db_p_i_{plan}"]
            print(plan)
            print("p", (m[f"p_{plan}_eq"]).all(), m[f"p_{plan}"].sum())
            print("i", (m[f"i_{plan}_eq"]).all(), m[f"i_{plan}"].sum())
            print("pi", (m[f"p_i_{plan}_eq"]).all(), m[f"p_i_{plan}"].sum())

        eq = m[[c for c in m.columns if c.endswith("_eq")]]

        # assert eq.all(axis=1).all()

        peq = eq.groupby("person_id").all()
        assert peq.all(axis=1).all()
