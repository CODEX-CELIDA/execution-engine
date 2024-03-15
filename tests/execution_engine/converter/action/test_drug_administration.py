from execution_engine.converter.action.drug_administration import (
    DrugAdministrationAction,
)
from execution_engine.omop.criterion.drug_exposure import DrugExposure
from execution_engine.util.enum import TimeUnit
from execution_engine.util.types import Dosage
from execution_engine.util.value import ValueNumber
from execution_engine.util.value.time import ValueCount
from tests._fixtures import concept


class TestDrugAdministration:
    def test_single_dose(self):
        dosage = Dosage(
            dose=ValueNumber(value=1, unit=concept.concept_unit_mg),
            count=1,
            duration=None,
            frequency=ValueCount(value_min=1),
            interval=1 * TimeUnit.DAY,
        )
        dosage_def = DrugAdministrationAction.DosageDefinition(dose=dosage)

        action = DrugAdministrationAction(
            name="test",
            exclude=False,
            ingredient_concept=concept.concept_heparin_ingredient,
            dosages=[dosage_def],
        )

        criterion = action.to_criterion()

        assert isinstance(criterion, DrugExposure)
        assert criterion._dose == dosage

    def test_multiple_doses(self):
        dosages = [
            Dosage(
                dose=ValueNumber(value=1, unit=concept.concept_unit_mg),
                count=20,
                duration=10 * TimeUnit.HOUR,
                frequency=ValueCount(value=10),
                interval=2 * TimeUnit.DAY,
            ),
            Dosage(
                dose=ValueNumber(value=2, unit=concept.concept_unit_mg),
                count=3,
                duration=None,
                frequency=None,
                interval=None,
            ),
            Dosage(
                dose=ValueNumber(value=3, unit=concept.concept_unit_mg),
                count=1,
                duration=None,
                frequency=ValueCount(value_min=1),
                interval=1 * TimeUnit.DAY,
            ),
        ]

        action = DrugAdministrationAction(
            name="test",
            exclude=False,
            ingredient_concept=concept.concept_heparin_ingredient,
            dosages=[
                DrugAdministrationAction.DosageDefinition(dose=d) for d in dosages
            ],
        )

        comb = action.to_criterion()
        criteria = list(comb)

        assert len(criteria) == 3
        assert all(isinstance(c, DrugExposure) for c in criteria)

        assert criteria[0]._dose == dosages[0]

        # The second dose has no frequency/interval, so it get's the default frequency of 1 per day
        assert criteria[1]._dose == Dosage(
            dose=ValueNumber(value=2, unit=concept.concept_unit_mg),
            count=dosages[1].count,
            duration=dosages[1].duration,
            frequency=ValueCount(value_min=1),
            interval=1 * TimeUnit.DAY,
        )
        assert criteria[2]._dose == dosages[2]
