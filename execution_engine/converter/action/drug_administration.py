import logging
from typing import Tuple, TypedDict, cast

import pandas as pd
from fhir.resources.activitydefinition import ActivityDefinition
from fhir.resources.dosage import Dosage
from fhir.resources.extension import Extension

from execution_engine.clients import omopdb
from execution_engine.constants import EXT_DOSAGE_CONDITION, CohortCategory
from execution_engine.converter.action.abstract import AbstractAction
from execution_engine.converter.converter import parse_code, parse_value
from execution_engine.fhir.recommendation import RecommendationPlan
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.omop.criterion.concept import ConceptCriterion
from execution_engine.omop.criterion.drug_exposure import DrugExposure
from execution_engine.omop.vocabulary import (
    SNOMEDCT,
    VocabularyFactory,
    standard_vocabulary,
)
from execution_engine.util import Value, ValueNumber, ucum_to_postgres


class ExtensionType(TypedDict):
    """
    A type for the extension dictionary.
    """

    code: Concept
    value: Value


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
        ingredient_concept: Concept,
        dose: ValueNumber | None = None,
        frequency: int | None = None,
        interval: str | None = None,
        route: Concept | None = None,
        extensions: list[ExtensionType] | None = None,
    ) -> None:
        """
        Initialize the drug administration action.
        """
        super().__init__(name=name, exclude=exclude)
        self._drug_concepts = drug_concepts
        self._ingredient_concept = ingredient_concept
        self._dose = dose
        self._frequency = frequency
        self._interval = interval
        self._route = route
        self._extensions = extensions

    @classmethod
    def from_fhir(cls, action_def: RecommendationPlan.Action) -> "AbstractAction":
        """Creates a new action from a FHIR PlanDefinition."""

        if action_def.activity is None:
            raise NotImplementedError("No activity defined for action")

        df_drugs, ingredient = cls.get_drug_concepts(action_def.activity)

        name = f"drug_{ingredient.concept_name}"
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
            action = cls(
                name,
                exclude,
                drug_concepts=cls.drug_concept_ids(df_drugs),
                ingredient_concept=ingredient,
            )

        else:
            # has dosage
            dosage = action_def.activity.dosage[
                0
            ]  # dosage is bound to 0..1 by drug-administration-action profile
            dose = cls.process_dosage(dosage)
            route = cls.process_route(dosage)
            frequency, interval = cls.process_timing(dosage)

            extensions = cls.process_dosage_extensions(dosage)

            df_drugs = cls.filter_same_unit(
                df_drugs, dose.unit
            )  # todo: we should not filter here, but perform conversions of values instead (if possible)

            action = cls(
                name=name,
                exclude=exclude,
                drug_concepts=cls.drug_concept_ids(df_drugs),
                ingredient_concept=ingredient,
                dose=dose,
                frequency=frequency,
                interval=interval,
                route=route,
                extensions=extensions,
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

        logging.info(f"Found ingredient {ingredient.concept_name} id={ingredient.concept_id}")  # type: ignore

        # get all drug concept ids for that ingredient
        df = omopdb.drugs_by_ingredient(ingredient.concept_id, with_unit=True)  # type: ignore

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
    def drug_concept_ids(cls, df_drugs: pd.DataFrame) -> list[int]:
        """
        Returns a list of drug concept ids.
        """
        return df_drugs["drug_concept_id"].sort_values().tolist()

    @classmethod
    def process_dosage(cls, dosage: Dosage) -> ValueNumber:
        """
        Processes the dosage of a drug administration action into a ValueNumber.
        """
        dose_and_rate = dosage.doseAndRate[
            0
        ]  # todo: should iterate over doseAndRate (could have multiple)

        value = cast(ValueNumber, parse_value(dose_and_rate, value_prefix="dose"))

        if (
            dose_and_rate.rateRatio is not None
            or dose_and_rate.rateRange is not None
            or dose_and_rate.rateQuantity is not None
        ):
            raise NotImplementedError("dosage rates have not been implemented yet")

        return value

    @classmethod
    def process_dosage_extensions(cls, dosage: Dosage) -> list[ExtensionType]:
        """
        Processes extensions of dosage

        Currently, the following extensions are supported:
        - dosage-condition: Condition that must be met for the dosage to be applied (e.g. body weight < 70 kg)
        """

        extensions: list[ExtensionType] = []

        if dosage.extension is None:
            return extensions

        def extension_field(extensions: list[Extension], url: str) -> Extension:
            cmpr = [ext for ext in extensions if ext.url == url]
            assert len(cmpr) == 1

            return cmpr[0]

        for extension in dosage.extension:
            if extension.url == EXT_DOSAGE_CONDITION:
                assert (
                    len(extension.extension) == 2
                ), "Dosage condition must have 2 sub-extensions (code, value)"
                condition_value = extension_field(extension.extension, "value")
                condition_type = extension_field(extension.extension, "type")

                code = parse_code(condition_type.valueCodeableConcept)
                value = parse_value(condition_value, "value")

                extensions.append({"code": code, "value": value})
            else:
                raise NotImplementedError(f"Unknown dosage extension {extension.url}")
        return extensions

    @classmethod
    def process_route(cls, dosage: Dosage) -> Concept | None:
        """
        Processes the route of a drug administration action into a ValueNumber.
        """
        if dosage.route is None:
            return None

        return parse_code(dosage.route)

    @classmethod
    def process_timing(cls, dosage: Dosage) -> Tuple[int, str]:
        """
        Returns the frequency and interval of the dosage.
        """
        timing = dosage.timing
        frequency = timing.repeat.frequency
        period = timing.repeat.period
        interval = ucum_to_postgres[timing.repeat.periodUnit]

        if period != 1:
            raise NotImplementedError(
                "Periods other than 1 have not yet been implemented"
            )

        return frequency, interval

    @classmethod
    def filter_same_unit(cls, df: pd.DataFrame, unit: Concept) -> pd.DataFrame:
        """
        Filters the given dataframe to only include rows with the given unit.

        If the unit is "international unit" or "unit" then the respective other unit is also included.
        This is because RxNorm (upon which OMOP is based) seems to use unit (UNT) for international units.
        """
        logging.warning(
            "Selecting only drug entries with unit matching to that of the recommendation"
        )

        df_filtered = df.query("amount_unit_concept_id==@unit.concept_id")

        other_unit = None
        if unit.concept_name == "unit":
            other_unit = standard_vocabulary.get_standard_unit_concept("[iU]")
        elif unit.concept_name == "international unit":
            other_unit = standard_vocabulary.get_standard_unit_concept("[U]")

        if other_unit is not None:
            logging.info(
                f'Detected unit "{unit.concept_name}", also selecting "{other_unit.concept_name}"'
            )
            df_filtered = pd.concat(
                [
                    df_filtered,
                    df.query("amount_unit_concept_id==@other_unit.concept_id"),
                ]
            )

        return df_filtered

    def _to_criterion(self) -> Criterion | CriterionCombination | None:
        """
        Returns a criterion that represents this action.
        """

        drug_action = DrugExposure(
            name=self._name,
            exclude=self._exclude,
            category=CohortCategory.INTERVENTION,
            drug_concepts=self._drug_concepts,
            ingredient_concept=self._ingredient_concept,
            dose=self._dose,
            frequency=self._frequency,
            interval=self._interval,
            route=self._route,
        )

        if self._extensions:
            comb = CriterionCombination(
                name=f"{self._name}_extensions",
                exclude=drug_action.exclude,  # need to pull up the exclude flag
                category=CohortCategory.INTERVENTION,
                operator=CriterionCombination.Operator("AND"),
            )
            drug_action.exclude = (
                False  # reset the exclude flag, as it is now part of the combination
            )

            comb.add(drug_action)

            for extension in self._extensions:
                comb.add(
                    ConceptCriterion(
                        name=f"{self._name}_ext_{extension['code'].concept_name}",
                        exclude=False,  # extensions are always included (at least for now)
                        category=CohortCategory.INTERVENTION,
                        concept=extension["code"],
                        value=extension["value"],
                    )
                )

            return comb

        return drug_action
