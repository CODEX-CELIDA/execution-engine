from abc import abstractmethod
from typing import Self, Type, final

from fhir.resources.timing import Timing as FHIRTiming

from execution_engine.constants import CohortCategory
from execution_engine.converter.converter import CriterionConverter, parse_code
from execution_engine.converter.goal.abstract import Goal
from execution_engine.fhir.recommendation import RecommendationPlan
from execution_engine.fhir.util import get_coding
from execution_engine.omop.criterion.abstract import Criterion
from execution_engine.omop.criterion.combination import CriterionCombination
from execution_engine.omop.vocabulary import AbstractVocabulary
from execution_engine.util import AbstractPrivateMethods
from execution_engine.util.types import Timing
from execution_engine.util.value.time import ValueCount, ValueDuration, ValuePeriod


class AbstractAction(CriterionConverter, metaclass=AbstractPrivateMethods):
    """
    An abstract action.

    An instance of this class represents an action entry of the PlanDefinition resource in the context of
    CPG-on-EBM-on-FHIR. In the Implementation Guide (specifically, the InterventionPlan profile),
    several types of actions are defined, including:
    - Drug Administration
    - Ventilator Management
    - Body Positioning
    Each of these slices from the Implementation Guide is represented by a subclass of this class.

    Subclasses must define the following methods:
    - valid: returns True if the supplied action  falls within the scope of the subclass
    - to_omop: converts the action to an OMOP criterion
    - from_fhir: creates a new instance of the subclass from a FHIR PlanDefinition.action element

    """

    _criterion_class: Type[Criterion]
    _concept_code: str
    _concept_vocabulary: Type[AbstractVocabulary]
    _goals: list[Goal]

    def __init__(self, name: str, exclude: bool):
        super().__init__(name=name, exclude=exclude)
        self._goals = []

    @classmethod
    @abstractmethod
    def from_fhir(cls, action_def: RecommendationPlan.Action) -> Self:
        """Creates a new action from a FHIR PlanDefinition."""
        raise NotImplementedError()

    @classmethod
    def valid(
        cls,
        action_def: RecommendationPlan.Action,
    ) -> bool:
        """Checks if the given FHIR definition is a valid action in the context of CPG-on-EBM-on-FHIR."""
        cc = get_coding(action_def.fhir().code)
        return (
            cls._concept_vocabulary.is_system(cc.system)
            and cc.code == cls._concept_code
        )

    @classmethod
    def process_timing(cls, timing: FHIRTiming) -> Timing:
        """
        Returns the frequency and interval of the dosage.
        """

        rep = timing.repeat

        if timing.event is not None:
            raise NotImplementedError("event has not been implemented")

        if rep is None:
            code = parse_code(timing.code)
            raise NotImplementedError(
                f"Timing without repeat has not been implemented (code={code})"
            )

        # Process BOUND
        if (
            rep.boundsPeriod is not None
            or rep.boundsDuration is not None
            or rep.boundsRange is not None
        ):
            raise NotImplementedError("Timing with bounds has not been implemented")

        # Process COUNT
        if rep.count is not None or rep.countMax is not None:
            count = ValueCount(
                value_min=rep.count,
                value_max=rep.countMax if rep.countMax is not None else None,
            )
        else:
            count = None

        # Process DURATION
        if rep.duration is not None or rep.durationMax is not None:
            duration = ValueDuration(
                value_min=rep.duration, value_max=rep.durationMax, unit=rep.durationUnit
            )
        else:
            duration = None

        # Process FREQUENCY
        if rep.frequency is not None or rep.frequencyMax is not None:
            frequency = ValueCount(value_min=rep.frequency, value_max=rep.frequencyMax)
        else:
            frequency = None

        # Process INTERVAL
        if rep.periodMax is not None:
            raise NotImplementedError("periodMax has not been implemented")

        if rep.period is not None:
            interval = ValuePeriod(value=rep.period, unit=rep.periodUnit)
        else:
            interval = None

        if rep.dayOfWeek is not None:
            raise NotImplementedError("dayOfWeek has not been implemented")

        if rep.timeOfDay is not None:
            raise NotImplementedError("timeOfDay has not been implemented")

        if rep.when is not None:
            raise NotImplementedError("when has not been implemented")

        if rep.offset is not None:
            raise NotImplementedError("offset has not been implemented")

        return Timing(
            count=count, duration=duration, frequency=frequency, interval=interval
        )

    @abstractmethod
    def _to_criterion(self) -> Criterion | CriterionCombination | None:
        """Converts this characteristic to a Criterion."""
        raise NotImplementedError()

    @final
    def to_criterion(self) -> Criterion | CriterionCombination:
        """
        Converts this action to a criterion.
        """
        action = self._to_criterion()

        if action is None:
            assert (
                self.goals
            ), "Action without explicit criterion must have at least one goal"

        if self.goals:
            combination = CriterionCombination(
                name=f"{self._name}_plus_goals",
                exclude=self._exclude,  # need to pull up the exclude flag from the criterion into the combination
                category=CohortCategory.INTERVENTION,
                operator=CriterionCombination.Operator("AND"),
            )
            if action is not None:
                action.exclude = False  # reset the exclude flag, as it is now part of the combination
                combination.add(action)

            for goal in self.goals:
                combination.add(goal.to_criterion())

            return combination

        return action  # type: ignore

    @property
    def goals(self) -> list[Goal]:
        """
        Returns the goals as defined in the FHIR PlanDefinition.
        """
        return self._goals
