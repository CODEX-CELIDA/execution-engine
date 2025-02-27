from execution_engine.converter.action.procedure import ProcedureAction
from execution_engine.omop.vocabulary import SNOMEDCT


class AssessmentAction(ProcedureAction):
    """
    An AssessmentAction is an action that is used to assess a patient's condition.

    This action just tests whether the assessment has been performed by determining whether any value
    is present in the respective OMOP CDM table.
    """

    _concept_code = "386053000"  # Evaluation procedure (procedure)
    _concept_vocabulary = SNOMEDCT
