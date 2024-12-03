import logging
from typing import Self, TypedDict, cast

import pandas as pd
from fhir.resources.activitydefinition import ActivityDefinition
from fhir.resources.dosage import Dosage as FHIRDosage

from execution_engine.clients import omopdb
from execution_engine.constants import EXT_DOSAGE_CONDITION, CohortCategory
from execution_engine.converter.action.abstract import AbstractAction
from execution_engine.converter.converter import (
    get_extension_by_url,
    parse_code,
    parse_value,
)
from execution_engine.fhir.recommendation import RecommendationPlan
from execution_engine.omop.concepts import Concept
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination.logical import (
    LogicalCriterionCombination,
    NonCommutativeLogicalCriterionCombination,
)
from execution_engine.omop.criterion.drug_exposure import DrugExposure
from execution_engine.omop.criterion.point_in_time import PointInTimeCriterion
from execution_engine.omop.vocabulary import (
    SNOMEDCT,
    VocabularyFactory,
    standard_vocabulary,
)
from execution_engine.util.types import Dosage
from execution_engine.util.value import Value, ValueNumber


class ExtensionType(TypedDict):
    """
    A type for the extension dictionary.
    """

    code: Concept
    value: Value
    type: str


class DrugAdministrationAction(AbstractAction):
    """
    A drug administration action.
    """

    _concept_code = "432102000"  # Administration of substance (procedure)
    _concept_vocabulary = SNOMEDCT

    class DosageDefinition(TypedDict):
        """
        A type for the dosage definition dictionary.
        """

        dose: Dosage
        route: Concept | None = None
        extensions: list[ExtensionType] | None = None

    def __init__(
        self,
        exclude: bool,
        ingredient_concept: Concept,
        dosages: list[DosageDefinition] | None = None,
    ) -> None:
        """
        Initialize the drug administration action.
        """
        super().__init__(exclude=exclude)
        self._ingredient_concept = ingredient_concept
        self._dosages = dosages

    @classmethod
    def from_fhir(cls, action_def: RecommendationPlan.Action) -> Self:
        """Creates a new action from a FHIR PlanDefinition."""

        if action_def.activity_definition_fhir is None:
            raise NotImplementedError("No activity defined for action")

        ingredient = cls.get_ingredient_concept(action_def.activity_definition_fhir)

        exclude = (
            action_def.activity_definition_fhir.doNotPerform
            if action_def.activity_definition_fhir.doNotPerform is not None
            else False
        )

        if action_def.activity_definition_fhir.dosage is None:
            assert (
                action_def.goals_fhir
            ), "DrugAdministrationAction without a dosage must have a goal"

            # must return criterion according to goal
            # return combination of drug criterion (any application !) and goal criterion
            action = cls(
                exclude,
                ingredient_concept=ingredient,
            )

            return action

        else:
            # has dosage

            dosages = []

            for dosage in action_def.activity_definition_fhir.dosage:
                if dosage.doseAndRate is None:
                    raise NotImplementedError(
                        "Dosage without doseAndRate is not supported"
                    )
                dose = cls.process_dosage(dosage)
                route = cls.process_route(dosage)

                extensions = cls.process_dosage_extensions(dosage)

                dosages.append(
                    cls.DosageDefinition(dose=dose, route=route, extensions=extensions)
                )

            action = cls(
                exclude=exclude,
                ingredient_concept=ingredient,
                dosages=dosages,
            )

        return action

    @classmethod
    def get_ingredient_concept(cls, activity: ActivityDefinition) -> Concept:
        """
        Gets all drug concepts that match the given definition in the productCodeableConcept.
        """
        vf = VocabularyFactory()

        # find ingredient concept
        cc = activity.productCodeableConcept

        ingredients = []

        for coding in cc.coding:
            vocab = vf.get(coding.system)
            try:
                ingredient = omopdb.drug_vocabulary_to_ingredient(
                    vocab.omop_vocab_name, coding.code  # type: ignore
                )
            except AssertionError:
                if vocab.name() == "SNOMEDCT":
                    raise ValueError(
                        f"Could not find ingredient for SNOMEDCT code {coding.code}, but SNOMEDCT is required"
                    )
                logging.warning(
                    f"Could not find ingredient for {vocab.name()} code {coding.code}, skipping"
                )
                continue

            if ingredient is not None:
                ingredients.append(ingredient)

        if len(set(ingredients)) > 1:
            raise NotImplementedError(
                "Multiple ingredients found for productCodeableConcept"
            )
        elif len(set(ingredients)) == 0:
            raise ValueError("No ingredient found for productCodeableConcept")

        ingredient = ingredients[0]

        logging.info(f"Found ingredient {ingredient.concept_name} id={ingredient.concept_id}")  # type: ignore

        return ingredient

    @classmethod
    def drug_concept_ids(cls, df_drugs: pd.DataFrame) -> list[int]:
        """
        Returns a list of drug concept ids.
        """
        return df_drugs["drug_concept_id"].sort_values().tolist()

    @classmethod
    def process_dosage(cls, dosage: FHIRDosage) -> Dosage:
        """
        Processes the dosage of a drug administration action into a ValueNumber.
        """
        dose_and_rate = dosage.doseAndRate[
            0
        ]  # todo: should iterate over doseAndRate (could have multiple)

        dose = cast(ValueNumber, parse_value(dose_and_rate, value_prefix="dose"))

        if (
            dose_and_rate.rateRatio is not None
            or dose_and_rate.rateRange is not None
            or dose_and_rate.rateQuantity is not None
        ):
            raise NotImplementedError("dosage rates have not been implemented yet")

        timing = cls.process_timing(dosage.timing)

        return Dosage(dose=dose, **timing.model_dump())

    @classmethod
    def process_dosage_extensions(cls, dosage: FHIRDosage) -> list[ExtensionType]:
        """
        Processes extensions of dosage

        Currently, the following extensions are supported:
        - dosage-condition: Condition that must be met for the dosage to be applied (e.g. body weight < 70 kg)
        """

        extensions: list[ExtensionType] = []

        if dosage.extension is None:
            return extensions

        for extension in dosage.extension:
            if extension.url == EXT_DOSAGE_CONDITION:
                assert (
                    len(extension.extension) == 2
                ), "Dosage condition must have 2 sub-extensions (code, value)"
                condition_value = get_extension_by_url(extension, "value")
                condition_type = get_extension_by_url(extension, "type")

                code = parse_code(condition_type.valueCodeableConcept)
                value = parse_value(condition_value, "value")

                extensions.append({"code": code, "value": value, "type": "conditional"})
            else:
                raise NotImplementedError(f"Unknown dosage extension {extension.url}")

        return extensions

    @classmethod
    def process_route(cls, dosage: FHIRDosage) -> Concept | None:
        """
        Processes the route of a drug administration action into a ValueNumber.
        """
        if dosage.route is None:
            return None

        return parse_code(dosage.route)

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

    def _to_criterion(self) -> Criterion | LogicalCriterionCombination | None:
        """
        Returns a criterion that represents this action.
        """

        drug_actions: list[Criterion | LogicalCriterionCombination] = []

        if not self._dosages:
            # no dosages, just return the drug exposure
            return DrugExposure(
                exclude=False,
                category=CohortCategory.INTERVENTION,
                ingredient_concept=self._ingredient_concept,
                dose=None,
                route=None,
            )

        for dosage in self._dosages:
            drug_action = DrugExposure(
                exclude=False,  # first set to False, as the exclude flag is pulled up into the combination
                category=CohortCategory.INTERVENTION,
                ingredient_concept=self._ingredient_concept,
                dose=dosage["dose"],
                route=dosage.get("route", None),
            )

            extensions = dosage.get("extensions", None)
            if extensions is None or len(extensions) == 0:
                drug_actions.append(drug_action)
            else:
                if len(extensions) > 1:
                    raise NotImplementedError(
                        "Multiple extensions in dosage not supported yet"
                    )

                extension = extensions[0]

                if extension["type"] != "conditional":
                    raise NotImplementedError(
                        f"Extension type {extension['type']} not supported yet"
                    )

                # drug_action.exclude = False  # reset the exclude flag, as it is now part of the combination

                ext_criterion = PointInTimeCriterion(
                    exclude=False,  # extensions are always included (at least for now)
                    category=CohortCategory.INTERVENTION,
                    concept=extension["code"],
                    value=extension["value"],
                )

                comb = NonCommutativeLogicalCriterionCombination.ConditionalFilter(
                    exclude=False,  # drug_action.exclude,  # need to pull up the exclude flag from the criterion into the combination
                    category=CohortCategory.INTERVENTION,
                    left=ext_criterion,
                    right=drug_action,
                )

                drug_actions.append(comb)

        if len(drug_actions) == 1:
            # set the exclude flag to the value of the action, as this is the only action
            # drug_actions[0].exclude = self._exclude
            assert not drug_actions[0].exclude
            return drug_actions[0]
        else:
            comb = LogicalCriterionCombination(
                exclude=False,  # self._exclude,
                category=CohortCategory.INTERVENTION,
                operator=LogicalCriterionCombination.Operator("OR"),
            )
            comb.add_all(drug_actions)
            return comb
