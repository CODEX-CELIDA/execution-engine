from execution_engine import fhir
from execution_engine.builder import ExecutionEngineBuilder
from execution_engine.converter.parser.base import FhirRecommendationParserInterface
from execution_engine.converter.parser.factory import FhirRecommendationParserFactory
from execution_engine.fhir.client import FHIRClient
from execution_engine.fhir.recommendation import (
    RecommendationPlan,
    RecommendationPlanCollection,
)
from execution_engine.omop import cohort
from execution_engine.omop.cohort.population_intervention_pair import (
    PopulationInterventionPairExpr,
)
from execution_engine.omop.criterion.visit_occurrence import PatientsActiveDuringPeriod
from execution_engine.util import logic as logic


class FhirToRecommendationFactory:
    """
    A class for parsing FHIR resources into OMOP cohort objects.

    This class provides methods for parsing FHIR resources, such as Recommendation (PlanDefinition) and EvidenceVariable,
    into OMOP cohort objects, such as PopulationInterventionPair and Recommendation. The parsed objects include
    characteristics, actions, goals, and other relevant metadata.
    """

    def __init__(self, builder: ExecutionEngineBuilder | None = None):
        self.builder = builder

    def parse_recommendation_from_url(
        self,
        url: str,
        package_version: str,
        fhir_client: FHIRClient,
        parser_version: int = 2,
    ) -> cohort.Recommendation:
        """
        Creates a Recommendation object by fetching and parsing recommendation data from a given URL.

        This function utilizes a FHIR connector to access recommendation data, constructs population and intervention
        pairs based on the recommendation plans, and aggregates them into a Recommendation object which includes
        metadata like name, version, and description.

        Args:
            url (str): The URL from which to fetch the recommendation data.
            package_version (str): The version of the recommendation package to be used.
            fhir_client (FHIRClient): An instance of FHIRClient used to connect to and fetch data from a FHIR server.
            parser_version (int): The version of the FHIR parser to be used. Defaults to 2.

        Returns:
            cohort.Recommendation: An instance of the Recommendation class populated with the parsed recommendation data,
            including population intervention pairs and other relevant metadata.

        Raises:
            ValueError: If an action within a recommendation plan is None, indicating incomplete or invalid data.
        """

        parser = FhirRecommendationParserFactory(builder=self.builder).get_parser(
            parser_version
        )

        rec = fhir.Recommendation(
            url,
            package_version=package_version,
            fhir_connector=fhir_client,
        )

        # Recursively build a single expression from the nested plans/collections:
        expr = self._parse_collection(rec.plans(), parser)

        recommendation = cohort.Recommendation(
            expr=expr,
            base_criterion=PatientsActiveDuringPeriod(),
            url=rec.url,
            name=rec.name,
            title=rec.title,
            version=rec.version,
            description=rec.description,
            package_version=rec.package_version,
        )

        return recommendation

    def _parse_collection(
        self,
        plan_or_collection: RecommendationPlanCollection | RecommendationPlan,
        parser: FhirRecommendationParserInterface,
    ) -> logic.Expr:
        """
        Recursively parse a single RecommendationPlan or a nested RecommendationPlanCollection
        into a logic.Expr. If it's a collection, gather each sub-item's expression and combine
        them using parser.parse_action_combination_method(...).
        """
        if isinstance(plan_or_collection, RecommendationPlan):
            # Parse a leaf plan: build a population + interventions expression (e.g. PopulationInterventionPairExpr)
            population_criteria = parser.parse_characteristics(
                plan_or_collection.population
            )
            intervention_criteria = parser.parse_actions(
                plan_or_collection.actions, plan_or_collection
            )
            return PopulationInterventionPairExpr(
                population_expr=population_criteria,
                intervention_expr=intervention_criteria,
                name=plan_or_collection.name,
                url=plan_or_collection.url,
                base_criterion=PatientsActiveDuringPeriod(),
            )

        elif isinstance(plan_or_collection, RecommendationPlanCollection):
            # Recursively parse all sub-items
            sub_exprs = [
                self._parse_collection(sub_item, parser)
                for sub_item in plan_or_collection.plans
            ]

            combination_op = parser.parse_action_combination_method(
                plan_or_collection.fhir
            )

            # Combine all sub-expressions with the appropriate operator
            return combination_op(*sub_exprs)

        else:
            raise TypeError("Unknown plan_or_collection type.")
