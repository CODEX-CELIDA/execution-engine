from enum import Enum
from typing import Dict, List

from .concepts import ConceptSetManager
from .criterion import Criterion


class ObservationWindow:
    """An observation window in a cohort definition."""

    def __init__(self, priorDays: int = 0, postDays: int = 0):
        self.priorDays = priorDays
        self.postDays = postDays

    def json(self) -> Dict:
        """Return the JSON representation of the observation window."""
        return {"PriorDays": self.priorDays, "PostDays": self.postDays}


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
        useEventEnd: bool = False,
    ):
        self.start = {"Days": start_days, "Coeff": start_coeff}
        self.end = {"Days": end_days, "Coeff": end_coef}
        self.useEventEnd = useEventEnd

    def json(self) -> Dict:
        """Return the JSON representation of the start window."""
        return {"Start": self.start, "End": self.end, "UseEventEnd": self.useEventEnd}


class Occurrence:
    """An occurrence in a cohort definition."""

    class Type(Enum):
        """The type of an occurrence."""

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
        countColumn: CountColumn = CountColumn.DOMAIN_CONCEPT,
    ):
        self.type = type
        self.count = count
        self.countColumn = countColumn

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
        self, criterion: Criterion, startWindow: StartWindow, occurrence: Occurrence
    ):
        self.criterion = criterion
        self.startWindow = startWindow
        self.occurrence = occurrence

    def json(self) -> Dict:
        """Return the JSON representation of the inclusion criterion."""
        return {
            "Criteria": self.criterion.json(),
            "StartWindow": self.startWindow.json(),
            "Occurrence": self.occurrence.json(),
        }


class InclusionRule:
    """An inclusion rule in a cohort definition."""

    class InclusionRuleType(Enum):
        """The type of an inclusion rule."""

        AT_MOST = "AT_MOST"
        AT_LEAST = "AT_LEAST"
        EXACTLY = "EXACTLY"

    def __init__(
        self,
        name: str,
        type: InclusionRuleType,
        count: int,
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
        return {
            "name": self.name,
            "expression": {
                "Type": self.type.value,
                "Count": self.count,
                "CriteriaList": [c.json() for c in self.criteria],
            },
            "DemographicCriteriaList": self.demographicCriteria,
            "Groups": self.groups,
        }


class CohortDefinition:
    """A cohort definition."""

    def __init__(
        self,
        conceptSetManager: ConceptSetManager,
        primaryCriteria: PrimaryCriteria,
        inclusionRules: List[InclusionRule],
    ):
        self.conceptSetManager = conceptSetManager
        self.primaryCriteria = primaryCriteria
        self.inclusionRules = inclusionRules

    def json(self) -> Dict:
        """Return the JSON representation of the cohort definition."""
        return {
            "ConceptSets": self.conceptSetManager.json(),
            "PrimaryCriteria": self.primaryCriteria.json(),
            "QualifiedLimit": {"Type": "First"},
            "ExpressionLimit": {"Type": "First"},
            "InclusionRules": [r.json() for r in self.inclusionRules],
            "CensoringCriteria": [],
            "CollapseSettings": {"CollapseType": "ERA", "EraPad": 0},
            "CensorWindow": {},
            "cdmVersionRange": ">=5.0.0",
        }
