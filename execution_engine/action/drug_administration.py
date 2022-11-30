import warnings
from typing import Tuple

import pandas as pd
from fhir.resources.activitydefinition import ActivityDefinition
from fhir.resources.dosage import Dosage
from fhir.resources.quantity import Quantity

from ..action import AbstractAction
from ..clients import omopdb
from ..fhir.recommendation import Recommendation
from ..goal import LaboratoryValueGoal
from ..omop.vocabulary import SNOMEDCT, VocabularyFactory, standard_vocabulary
from ..util import ValueNumber


class DrugAdministrationAction(AbstractAction):
    """
    A drug administration action.
    """

    _concept_code = "432102000"  # Administration of substance (procedure)
    _concept_vocabulary = SNOMEDCT
    _goal_type = LaboratoryValueGoal  # todo: Most likely there is no 1:1 relationship between action and goal types

    @classmethod
    def from_fhir(cls, action_def: Recommendation.Action) -> "AbstractAction":
        """Creates a new action from a FHIR PlanDefinition."""

        if action_def.activity is None:
            raise NotImplementedError("No activity defined for action")

        if action_def.activity.dosage is None:
            assert (
                action_def.goals
            ), "DrugAdministrationAction without a dosage must have a goal"
            raise NotImplementedError(
                "DrugAdministrationAction without a dosage is not yet implemented"
            )

        # has dosage
        dosage = action_def.activity.dosage[
            0
        ]  # dosage is bound to 0..1 by drug-administration-action profile

        df_drugs = cls.get_drug_concepts(action_def.activity)
        dose = cls.process_dosage(dosage)
        frequency, interval = cls.process_timing(dosage)

        warnings.warn(
            "selecting only drug entries with unit matching to that of the recommendation"
        )
        df_drugs = df_drugs.query("amount_unit_concept_id==@dose.unit.id")
        dose_sql = dose.to_sql(table_name=None, column_name="dose", with_unit=False)

        # fmt: off
        query = (  # nosec
            f"""SELECT
            person_id,
            date_trunc(drug_exposure_start_datetime, %(interval)s) as interval,
            count(*) as cnt
            sum(quantity) as dose
        FROM drug_exposure de
        WHERE drug_concept_id IN (%(drug_concept_ids)s)
         -- AND drug_exposure_start_datetime BETWEEN (.., ...)
        HAVING
            {dose_sql}
            AND cnt = %(frequency)
        """)
        # fmt: on

        return omopdb.query(
            query,
            interval=interval,
            frequency=frequency,
            drug_concept_ids=df_drugs["drug_concept_id"].tolist(),
        )

    @classmethod
    def get_drug_concepts(cls, activity: ActivityDefinition) -> pd.DataFrame:
        """
        Gets all drug concepts that match the given definition in the productCodeableConcept.
        """
        vf = VocabularyFactory()

        # find ingredient concept
        cc = activity.productCodeableConcept

        ingredients = []

        for coding in cc.coding:
            vocab = vf.get(coding.system)
            ingredient = omopdb.drug_vocabulary_to_ingredient(
                vocab.omop_vocab_name, coding.code  # type: ignore
            )
            ingredients.append(ingredient)
        assert len(set(ingredients)) == 1

        print(ingredient)

        # get all drug concept ids for that ingredient
        df = omopdb.drugs_by_ingredient(ingredient.id, with_unit=True)

        # assert drug units are mutually exclusive
        assert (
            df.loc[df["amount_unit_concept_id"].notnull(), "numerator_unit_concept_id"]
            .isnull()
            .all()
        )
        assert (
            df.loc[df["numerator_unit_concept_id"].notnull(), "amount_unit_concept_id"]
            .isnull()
            .all()
        )

        df["amount_unit_concept_id"].fillna(
            df["numerator_unit_concept_id"], inplace=True
        )

        return df

    @classmethod
    def process_dosage(cls, dosage: Dosage) -> ValueNumber:
        """
        Processes the dosage of a drug administration action into a ValueNumber.
        """
        dose_and_rate = dosage.doseAndRate[
            0
        ]  # todo: should iterate over doseAndRate (could have multiple)

        if dose_and_rate.doseQuantity is not None:
            value = dose_and_rate.doseQuantity
            ValueNumber(
                value=value.value,
                unit=standard_vocabulary.get_standard_unit_concept(value.unit),
            )
        elif dose_and_rate.doseRange is not None:
            value = dose_and_rate.doseRange
            if value.low is not None and value.high is not None:
                assert (
                    value.low.code == value.high.code
                ), "Range low and high unit must be the same"

            unit_code = value.low.code if value.low is not None else value.high.code

            def value_or_none(x: Quantity) -> float | None:
                if x is None:
                    return None
                return x.value

            value = ValueNumber(
                unit=standard_vocabulary.get_standard_unit_concept(unit_code),
                value_min=value_or_none(value.low),
                value_max=value_or_none(value.high),
            )

        if (
            dose_and_rate.rateRatio is not None
            or dose_and_rate.rateRange is not None
            or dose_and_rate.rateQuantity is not None
        ):
            raise NotImplementedError("dosage rates have not been implemented yet")

        return value

    @classmethod
    def process_timing(cls, dosage: Dosage) -> Tuple[int, str]:
        """
        Returns the frequency and interval of the dosage.
        """
        ucum_to_postgres = {
            "s": "second",
            "min": "minute",
            "h": "hour",
            "d": "day",
            "wk": "week",
            "mo": "month",
            "a": "year",
        }
        timing = dosage.timing
        frequency = timing.repeat.frequency
        period = timing.repeat.period
        interval = ucum_to_postgres[timing.repeat.periodUnit]

        if period != 1:
            raise NotImplementedError(
                "Periods other than 1 have not yet been implemented"
            )

        return frequency, interval
