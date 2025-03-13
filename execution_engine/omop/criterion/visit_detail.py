from execution_engine.omop.concepts import Concept
from execution_engine.util.types import Timing
from execution_engine.util.value import Value

__all__ = ["VisitDetail"]

from execution_engine.omop.criterion.continuous import ContinuousCriterion


class VisitDetail(ContinuousCriterion):
    """
    A visit detail criterion in a recommendation.

    This criterion is used to filter the visit_detail table.
    visit details may be transfers between units of a hospital or a change of bed. We d
    """

    def __init__(
        self,
        concept: Concept,
        value: Value | None = None,
        static: bool | None = None,
        timing: Timing | None = None,
        value_required: bool | None = None,
    ):
        super().__init__(
            concept=concept,
            value=value,
            static=static,
            timing=timing,
            value_required=value_required,
        )

        # visit concepts are mapped automatically to the visit_occurrence table, so map visit_detail explicitly here
        self._set_omop_variables_from_domain("visit_detail")
