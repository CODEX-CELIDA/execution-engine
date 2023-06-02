# app_state.py
import logging

from execution_engine.execution_engine import ExecutionEngine


class AppState:
    """
    Application state
    """

    _execution_engine: ExecutionEngine | None = None
    _recommendations: dict = {}
    _initialized: bool = False

    @classmethod
    def initialize(cls) -> None:
        """
        Initialize the execution engine and load all recommendations.
        """
        if cls._initialized:
            return

        base_url = (
            "https://www.netzwerk-universitaetsmedizin.de/fhir/codex-celida/guideline/"
        )

        urls = [
            "covid19-inpatient-therapy/recommendation/no-therapeutic-anticoagulation",
            # "sepsis/recommendation/ventilation-plan-ards-tidal-volume",
            # "covid19-inpatient-therapy/recommendation/ventilation-plan-ards-tidal-volume",
            # "covid19-inpatient-therapy/recommendation/covid19-ventilation-plan-ards-peep",
            # "covid19-inpatient-therapy/recommendation/prophylactic-anticoagulation",
            # "covid19-inpatient-therapy/recommendation/therapeutic-anticoagulation",
            # "covid19-inpatient-therapy/recommendation/covid19-abdominal-positioning-ards",
        ]

        e = ExecutionEngine(verbose=False)

        for recommendation_url in urls:
            url = base_url + recommendation_url
            logging.info(f"Loading {url}")
            rec = e.load_recommendation(url, force_reload=False)
            cls._recommendations[url] = {
                "name": rec.name,
                "title": rec.title,
                "description": rec.description,
                "cohort_definition": rec,
            }

        cls._initialized = True

    @classmethod
    def get_execution_engine(cls) -> ExecutionEngine:
        """
        Get execution engine.
        """
        if cls._execution_engine is None:
            raise Exception("ExecutionEngine not initialized")

        return cls._execution_engine

    @classmethod
    def get_recommendations(cls) -> dict:
        """
        Get available recommendations by URL.
        """
        if cls._recommendations is None:
            raise Exception("Recommendations not initialized")

        return cls._recommendations
