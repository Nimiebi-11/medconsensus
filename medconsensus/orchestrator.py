from __future__ import annotations

from medconsensus.agents import CardiologyAgent, ConsensusAgent, GeneralMedicineAgent, PulmonologyAgent
from medconsensus.llm import build_llm_client, configured_provider, llm_enabled
from medconsensus.schemas import (
    Consensus,
    ConsensusReport,
    DISCLAIMER,
    ICD10Code,
    PublicDebateSummary,
    PublicSpecialistAssessment,
    SpecialistOutput,
    SyntheticCaseRequest,
)


AGENT_LABELS = {
    "cardiology": "Cardiology",
    "pulmonology": "Pulmonology",
    "general_medicine": "General Medicine",
}


class MedConsensusOrchestrator:
    """Runs the MedConsensus synthetic debate flow."""

    def __init__(self, llm_client=None) -> None:
        self.llm_client = build_llm_client() if llm_client is None else llm_client
        self.specialists = [
            CardiologyAgent(self.llm_client),
            PulmonologyAgent(self.llm_client),
            GeneralMedicineAgent(self.llm_client),
        ]
        self.consensus_agent = ConsensusAgent(self.llm_client)

    def invoke(self, request: SyntheticCaseRequest) -> ConsensusReport:
        if not request.synthetic:
            raise ValueError("MedConsensus accepts synthetic/de-identified demonstration cases only.")

        case_text = request.patient_case.strip()
        assessments = [agent.assess(case_text) for agent in self.specialists]
        debates = [agent.debate(case_text, assessments) for agent in self.specialists]
        verdict = self.consensus_agent.synthesize(case_text, assessments, debates)
        mode = (
            "llm_multi_agent"
            if self.llm_client is not None
            and all(agent.last_assessment_used_llm and agent.last_debate_used_llm for agent in self.specialists)
            and self.consensus_agent.last_used_llm
            else "deterministic_fallback"
        )

        return ConsensusReport(
            metadata={
                "mode": mode,
                "llm_provider": configured_provider() if self.llm_client is not None else None,
                "llm_requested": llm_enabled(),
                "phi_stored": False,
            },
            case_summary=_summarize_case(case_text),
            specialist_assessments=[_public_assessment(item) for item in assessments],
            debate_summary=[
                PublicDebateSummary(
                    from_agent=AGENT_LABELS[item.agent],
                    challenge=item.challenge_argument,
                    response_or_revision=(
                        f"Revised to {item.revised_diagnosis}: {item.revision_reason}"
                        if item.revised_diagnosis
                        else f"Maintained position; underweighted finding: {item.underweighted_finding}"
                    ),
                )
                for item in debates
            ],
            consensus=Consensus(
                most_likely_diagnosis=verdict.primary_diagnosis.name,
                must_not_miss_diagnoses=[item.name for item in verdict.runner_up_diagnoses],
                icd10_codes=_icd10_codes(verdict),
                recommended_next_questions=[
                    "Clarify chest pain, exertional limitation, orthopnea/PND, syncope, palpitations, medication adherence, salt intake, infection symptoms, and thromboembolic risk factors.",
                    "Ask about prior echocardiogram results, prior renal function, baseline functional status, and recent medication changes.",
                ],
                recommended_next_tests=verdict.next_steps[1:],
                confidence=verdict.primary_diagnosis.confidence,
                safety_notes=[
                    "Use only synthetic or de-identified cases; do not submit PHI.",
                    "MedConsensus does not diagnose patients, recommend treatment, or replace clinician judgment.",
                    "This system recommends next questions/tests and red flags, not treatment orders.",
                    *verdict.red_flags,
                ],
                disclaimer=DISCLAIMER,
            ),
        )


def _public_assessment(output: SpecialistOutput) -> PublicSpecialistAssessment:
    top = output.diagnoses[0]
    return PublicSpecialistAssessment(
        agent=AGENT_LABELS[output.agent],
        top_diagnosis=top.name,
        differential_diagnoses=[diagnosis.name for diagnosis in output.diagnoses],
        supporting_evidence=top.supporting_evidence,
        concerns_or_missing_data=[
            output.challenge_to_rivals,
            f"Key confirming test: {top.confirming_test}",
        ],
    )


def _icd10_codes(verdict) -> list[ICD10Code]:
    seen: set[str] = set()
    codes = [ICD10Code(code=verdict.primary_diagnosis.icd10_code, label=verdict.primary_diagnosis.name)]
    seen.add(verdict.primary_diagnosis.icd10_code)
    for item in verdict.runner_up_diagnoses:
        if item.icd10_code not in seen:
            codes.append(ICD10Code(code=item.icd10_code, label=item.name))
            seen.add(item.icd10_code)
    return codes


def _summarize_case(case_text: str) -> str:
    normalized = " ".join(case_text.split())
    return normalized[:420] + ("..." if len(normalized) > 420 else "")
