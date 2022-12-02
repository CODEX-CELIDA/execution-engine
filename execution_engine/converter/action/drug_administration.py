import logging
import warnings
from typing import Tuple

import pandas as pd
from fhir.resources.activitydefinition import ActivityDefinition
from fhir.resources.dosage import Dosage
from fhir.resources.plandefinition import PlanDefinitionGoal

from execution_engine.clients import omopdb
from execution_engine.fhir.recommendation import Recommendation
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.drug_exposure import DrugExposure
from execution_engine.omop.vocabulary import SNOMEDCT, VocabularyFactory
from execution_engine.util import ValueNumber

from ...omop.criterion.combination import CriterionCombination
from ..converter import parse_value
from .abstract import AbstractAction


class DrugAdministrationAction(AbstractAction):
    """
    A drug administration action.
    """

    _concept_code = "432102000"  # Administration of substance (procedure)
    _concept_vocabulary = SNOMEDCT

    def __init__(
        self,
        name: str,
        exclude: bool,
        drug_concepts: pd.DataFrame,
        dose: ValueNumber | None = None,
        frequency: int | None = None,
        interval: str | None = None,
    ) -> None:
        """
        Initialize the drug administration action.
        """
        super().__init__(name=name, exclude=exclude)
        self._drug_concepts = drug_concepts
        self._dose = dose
        self._frequency = frequency
        self._interval = interval

    @classmethod
    def from_fhir(cls, action_def: Recommendation.Action) -> "AbstractAction":
        """Creates a new action from a FHIR PlanDefinition."""

        if action_def.activity is None:
            raise NotImplementedError("No activity defined for action")

        df_drugs, ingredient = cls.get_drug_concepts(action_def.activity)

        name = f"drug_{ingredient.name}"
        exclude = (
            action_def.activity.doNotPerform
            if action_def.activity.doNotPerform is not None
            else False
        )

        if action_def.activity.dosage is None:
            assert (
                action_def.goals
            ), "DrugAdministrationAction without a dosage must have a goal"

            # must return criterion according to goal
            # return combination of drug criterion (any application !) and goal criterion
            action = cls(name, exclude, drug_concepts=df_drugs)

        else:
            # has dosage
            dosage = action_def.activity.dosage[
                0
            ]  # dosage is bound to 0..1 by drug-administration-action profile
            dose = cls.process_dosage(dosage)
            frequency, interval = cls.process_timing(dosage)

            action = cls(
                name=name,
                exclude=exclude,
                drug_concepts=df_drugs,
                dose=dose,
                frequency=frequency,
                interval=interval,
            )

        return action

    @classmethod
    def get_drug_concepts(
        cls, activity: ActivityDefinition
    ) -> Tuple[pd.DataFrame, Concept]:
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
            if ingredient is None:
                if vocab.name() == "SNOMEDCT":
                    raise ValueError(
                        f"Could not find ingredient for SNOMEDCT code {coding.code}, but SNOMEDCT is required"
                    )
                logging.warning(
                    f"Could not find ingredient for {vocab.name()} code {coding.code}, skipping"
                )
                continue

            ingredients.append(ingredient)

        if len(set(ingredients)) > 1:
            raise NotImplementedError(
                "Multiple ingredients found for productCodeableConcept"
            )
        elif len(set(ingredients)) == 0:
            raise ValueError("No ingredient found for productCodeableConcept")

        ingredient = ingredients[0]

        logging.info(f"Found ingredient {ingredient.name} id={ingredient.id}")  # type: ignore

        # get all drug concept ids for that ingredient
        df = omopdb.drugs_by_ingredient(ingredient.id, with_unit=True)  # type: ignore

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

        return df, ingredient

    @classmethod
    def process_dosage(cls, dosage: Dosage) -> ValueNumber:
        """
        Processes the dosage of a drug administration action into a ValueNumber.
        """
        dose_and_rate = dosage.doseAndRate[
            0
        ]  # todo: should iterate over doseAndRate (could have multiple)

        value = parse_value(dose_and_rate, value_prefix="dose")

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

    def _to_criterion(self) -> Criterion | CriterionCombination | None:
        """
        Returns a criterion that represents this action.
        """

        return DrugExposure(
            name=self._name,
            exclude=self._exclude,
            dose=self._dose,
            frequency=self._frequency,
            interval=self._interval,
            drug_concepts=self._drug_concepts,
        )
