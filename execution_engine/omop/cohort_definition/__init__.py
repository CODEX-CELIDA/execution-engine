from enum import Enum
from typing import Any, Dict, List, Optional

from ..concepts import ConceptSetManager
from ..criterion import Criterion


class ObservationWindow:
    """An observation window in a cohort definition."""

    def __init__(self, prior_days: int = 0, post_days: int = 0):
        self.prior_days = prior_days
        self.post_days = post_days

    def json(self) -> Dict:
        """Return the JSON representation of the observation window."""
        return {"PriorDays": self.prior_days, "PostDays": self.post_days}


class PrimaryCriteriaLimit:
    """A primary criteria limit in a cohort definition."""

    def __init__(self, type: str = "First"):
        self.type = type

    def json(self) -> Dict:
        """Return the JSON representation of the primary criteria limit."""
        return {"Type": self.type}


class PrimaryCriteria:
    """A primary criteria in a cohort definition."""

    def __init__(
        self,
        criteria: List[Criterion],
        window: ObservationWindow,
        limit: PrimaryCriteriaLimit,
    ):
        self.criteria = criteria
        self.window = window
        self.limit = limit

    def json(self) -> Dict:
        """Return the JSON representation of the primary criteria."""
        return {
            "CriteriaList": [c.json() for c in self.criteria],
            "ObservationWindow": self.window.json(),
            "PrimaryCriteriaLimit": self.limit.json(),
        }


class StartWindow:
    """A start window in a cohort definition."""

    def __init__(
        self,
        start_days: int = 0,
        start_coeff: int = -1,
        end_days: int = 0,
        end_coef: int = -1,
        use_event_end: bool = False,
    ):
        self.start = {"Days": start_days, "Coeff": start_coeff}
        self.end = {"Days": end_days, "Coeff": end_coef}
        self.use_event_end = use_event_end

    def json(self) -> Dict:
        """Return the JSON representation of the start window."""
        return {"Start": self.start, "End": self.end, "UseEventEnd": self.use_event_end}


class Occurrence:
    """An occurrence in a cohort definition."""

    class Type(Enum):
        """The type of occurrence."""

        AT_MOST = 1
        AT_LEAST = 2
        EXACTLY = 0

    class CountColumn(Enum):
        """The count column of an occurrence."""

        VISIT_ID = "VISIT_ID"
        DOMAIN_CONCEPT = "DOMAIN_CONCEPT"
        START_DATE = "START_DATE"

    def __init__(
        self,
        type: Type,
        count: int,
        count_column: CountColumn = CountColumn.DOMAIN_CONCEPT,
    ):
        self.type = type
        self.count = count
        self.countColumn = count_column

    def json(self) -> Dict:
        """Return the JSON representation of the occurrence."""
        return {
            "Type": self.type.value,
            "Count": self.count,
            "CountColumn": self.countColumn.value,
        }


class InclusionCriterion:
    """An inclusion criterion in a cohort definition."""

    def __init__(
        self, criterion: Criterion, start_window: StartWindow, occurrence: Occurrence
    ):
        self.criterion = criterion
        self.start_window = start_window
        self.occurrence = occurrence

    def json(self) -> Dict:
        """Return the JSON representation of the inclusion criterion."""
        return {
            "Criteria": self.criterion.json(),
            "StartWindow": self.start_window.json(),
            "Occurrence": self.occurrence.json(),
        }


class InclusionRule:
    """An inclusion rule in a cohort definition."""

    class Type(Enum):
        """The type of inclusion rule."""

        ALL = "ALL"
        ANY = "ANY"
        AT_MOST = "AT_MOST"
        AT_LEAST = "AT_LEAST"

    def __init__(
        self,
        name: str,
        type: Type,
        count: Optional[int],
        criteria: List[InclusionCriterion],
    ):
        self.name = name
        self.type = type
        self.count = count
        self.criteria = criteria
        self.demographicCriteria: List = []
        self.groups: List = []

    def json(self) -> Dict:
        """Return the JSON representation of the inclusion rule."""

        expression: Dict[str, Any] = {
            "Type": self.type.value,
            "CriteriaList": [c.json() for c in self.criteria],
        }

        if self.type in [self.Type.AT_MOST, self.Type.AT_LEAST]:
            expression["Count"] = self.count

        return {
            "name": self.name,
            "expression": expression,
            "DemographicCriteriaList": self.demographicCriteria,
            "Groups": self.groups,
        }


class CohortDefinition:
    """A cohort definition."""

    _concept_set_manager: ConceptSetManager = ConceptSetManager()
    _primary_criteria: PrimaryCriteria
    _inclusion_rules: List[InclusionRule]

    def __init__(
        self
    ):
        self._concept_set_manager: ConceptSetManager = ConceptSetManager()
        self._inclusion_rules = []

    @property
    def concept_set_manager(self) -> ConceptSetManager:
        """Return the concept set manager."""
        return self._concept_set_manager

    @property
    def primary_criteria(self) -> PrimaryCriteria:
        """Return the primary criteria of the cohort definition."""
        return self._primary_criteria

    @primary_criteria.setter
    def primary_criteria(self, criteria: PrimaryCriteria) -> None:
        """Set the primary criteria of the cohort definition."""
        self._primary_criteria = criteria

    @property
    def inclusion_rules(self) -> List[InclusionRule]:
        """Return the inclusion rules of the cohort definition."""
        return self._inclusion_rules

    def json(self) -> Dict:
        """Return the JSON representation of the cohort definition."""
        return {
            "ConceptSets": self.concept_set_manager.json(),
            "PrimaryCriteria": self.primary_criteria.json(),
            "QualifiedLimit": {"Type": "First"},
            "ExpressionLimit": {"Type": "First"},
            "InclusionRules": [r.json() for r in self.inclusion_rules],
            "CensoringCriteria": [],
            "CollapseSettings": {"CollapseType": "ERA", "EraPad": 0},
            "CensorWindow": {},
            "cdmVersionRange": ">=5.0.0",
        }
