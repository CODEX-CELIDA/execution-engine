from execution_engine import fhir
from execution_engine.builder import ExecutionEngineBuilder
from execution_engine.converter.parser.factory import FhirRecommendationParserFactory
from execution_engine.fhir.client import FHIRClient
from execution_engine.omop import cohort
from execution_engine.omop.cohort import PopulationInterventionPair
from execution_engine.omop.criterion.visit_occurrence import PatientsActiveDuringPeriod


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

        pi_pairs: list[PopulationInterventionPair] = []

        base_criterion = PatientsActiveDuringPeriod()

        for rec_plan in rec.plans():
            pi_pair = PopulationInterventionPair(
                name=rec_plan.name,
                url=rec_plan.url,
                base_criterion=base_criterion,
            )

            # parse population and create criteria
            population_criteria = parser.parse_characteristics(rec_plan.population)
            pi_pair.set_population(population_criteria)

            # parse intervention and create criteria
            actions = parser.parse_actions(rec_plan.actions, rec_plan)
            pi_pair.add_intervention(actions)

            pi_pairs.append(pi_pair)

        recommendation = cohort.Recommendation(
            pi_pairs,
            base_criterion=base_criterion,
            url=rec.url,
            name=rec.name,
            title=rec.title,
            version=rec.version,
            description=rec.description,
            package_version=rec.package_version,
        )

        return recommendation
