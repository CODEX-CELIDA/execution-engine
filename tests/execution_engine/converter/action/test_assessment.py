import pytest

from execution_engine.converter.action.assessment import AssessmentAction
from execution_engine.omop.criterion.measurement import Measurement
from execution_engine.omop.criterion.observation import Observation
from execution_engine.omop.criterion.procedure_occurrence import ProcedureOccurrence
from execution_engine.util.enum import TimeUnit
from execution_engine.util.types import Timing
from execution_engine.util.value.time import ValueCount
from tests._fixtures import concept


class TestAssessmentAction:
    @pytest.mark.parametrize(
        "code, criterion_class",
        [
            (concept.concept_artificial_respiration, ProcedureOccurrence),
            (concept.concept_heparin_allergy, Observation),
            (concept.concept_tidal_volume, Measurement),
        ],
    )
    @pytest.mark.parametrize(
        "timing",
        [
            Timing(
                count=1,
                duration=10 * TimeUnit.HOUR,
                frequency=ValueCount(value=1),
                interval=1 * TimeUnit.DAY,
            )
        ],
    )
    def test_assessment_action(self, code, timing, criterion_class):
        action = AssessmentAction(exclude=False, code=code, timing=timing)

        criterion = action.to_positive_criterion()

        assert isinstance(criterion, criterion_class)
        assert criterion._concept == code
        assert criterion._timing == timing
        assert criterion._OMOP_VALUE_REQUIRED is False
