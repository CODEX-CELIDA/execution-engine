from typing import Any

from execution_engine.omop.criterion.concept import ConceptCriterion

__all__ = ["VisitDetail"]


class VisitDetail(ConceptCriterion):
    """
    A visit detail criterion  in a cohort definition.

    This criterion is used to filter the visit_detail table.
    visit details may be transfers between units of a hospital or a change of bed. We d
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # visit concepts are mapped automatically to the visit_occurrence table, so map visit_detail explicitly here
        self._set_omop_variables_from_domain("visit_detail")
