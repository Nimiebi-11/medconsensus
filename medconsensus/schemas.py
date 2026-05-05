from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


DISCLAIMER = "For clinician decision support only. Not a substitute for professional medical judgment."


class SyntheticCaseRequest(BaseModel):
    """Input accepted by the A2A invoke endpoint."""

    case_id: str | None = Field(default=None, description="Optional synthetic case identifier.")
    patient_case: str = Field(..., min_length=20, description="Synthetic, de-identified patient case text.")
    synthetic: bool = Field(default=True, description="Must be true; real patient cases are rejected.")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("patient_case")
    @classmethod
    def no_obvious_phi(cls, value: str) -> str:
        lowered = value.lower()
        blocked_terms = [" ssn ", "social security", "medical record number", " mrn ", "date of birth", " dob "]
        if any(term in f" {lowered} " for term in blocked_terms):
            raise ValueError("Input appears to contain PHI-like identifiers. Submit synthetic/de-identified cases only.")
        return value


class DiagnosisCandidate(BaseModel):
    name: str
    confidence: float = Field(..., ge=0, le=100)
    supporting_evidence: list[str]
    confirming_test: str
    icd10_code: str


class SpecialistOutput(BaseModel):
    agent: Literal["cardiology", "pulmonology", "general_medicine"]
    diagnoses: list[DiagnosisCandidate] = Field(..., min_length=1, max_length=2)
    challenge_to_rivals: str


class DebateRound(BaseModel):
    agent: Literal["cardiology", "pulmonology", "general_medicine"]
    maintains_position: bool
    revised_diagnosis: str | None = None
    revision_reason: str | None = None
    challenge_target: Literal["cardiology", "pulmonology", "general_medicine"]
    challenge_argument: str
    underweighted_finding: str


class FinalVerdictDiagnosis(BaseModel):
    name: str
    icd10_code: str
    confidence: float = Field(..., ge=0, le=100)
    reasoning: str


class RunnerUpDiagnosis(BaseModel):
    name: str
    icd10_code: str
    brief_reason: str


class FinalVerdict(BaseModel):
    primary_diagnosis: FinalVerdictDiagnosis
    runner_up_diagnoses: list[RunnerUpDiagnosis]
    dissenting_view_worth_noting: str
    next_steps: list[str]
    red_flags: list[str]
    confidence_below_70_explanation: str | None = None


class PublicSpecialistAssessment(BaseModel):
    agent: str
    top_diagnosis: str
    differential_diagnoses: list[str]
    supporting_evidence: list[str]
    concerns_or_missing_data: list[str]


class PublicDebateSummary(BaseModel):
    from_agent: str
    challenge: str
    response_or_revision: str


class ICD10Code(BaseModel):
    code: str
    label: str


class Consensus(BaseModel):
    most_likely_diagnosis: str
    must_not_miss_diagnoses: list[str]
    icd10_codes: list[ICD10Code]
    recommended_next_questions: list[str]
    recommended_next_tests: list[str]
    confidence: float = Field(..., ge=0, le=100)
    safety_notes: list[str]
    disclaimer: str = DISCLAIMER


class ConsensusReport(BaseModel):
    model_config = ConfigDict(json_schema_extra={"description": "MedConsensus structured output."})

    metadata: dict[str, Any]
    case_summary: str
    specialist_assessments: list[PublicSpecialistAssessment]
    debate_summary: list[PublicDebateSummary]
    consensus: Consensus
