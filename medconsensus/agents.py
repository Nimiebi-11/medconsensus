from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from medconsensus.schemas import DebateRound, DiagnosisCandidate, FinalVerdict, FinalVerdictDiagnosis, RunnerUpDiagnosis, SpecialistOutput


SAFETY_PROMPT = """All cases are synthetic/de-identified clinical decision-support simulations.
Do not diagnose real patients. Do not recommend treatment. Recommend next questions/tests and red flags only.
Use only the supplied case facts. Return valid JSON matching the requested schema."""

SPECIALIST_PROMPTS = {
    "cardiology": """You are a board-certified cardiologist AI with deep expertise in cardiovascular and vascular disease.
Identify the top 2 cardiology-perspective differential diagnoses. For each, provide diagnosis name, supporting case evidence, confidence 0-100, key confirming test, and ICD-10 code.
Briefly note what makes you doubt other specialties' likely conclusions.""",
    "pulmonology": """You are a board-certified pulmonologist AI with deep expertise in respiratory, airway, and pulmonary vascular disease.
Identify the top 2 pulmonary-perspective differential diagnoses. For each, provide diagnosis name, supporting case evidence, confidence 0-100, key confirming test, and ICD-10 code.
Explicitly challenge the most likely cardiology diagnosis if your pulmonary diagnosis is superior.""",
    "general_medicine": """You are a board-certified internist AI representing systemic, infectious, metabolic, renal, endocrine, and rare-disease perspectives that subspecialists may miss.
Identify the top 2 broad differential diagnoses. For each, provide diagnosis name, supporting case evidence, confidence 0-100, key confirming test, and ICD-10 code.
Flag zebra diagnoses when relevant and note if the presentation is atypical for the obvious answer.""",
}

DEBATE_PROMPT = """You are defending or revising your specialist assessment after reading colleagues' generated outputs.
Say whether you maintain your top diagnosis, identify the weakest rival argument with specific case evidence, and name any finding all agents may be underweighting."""

CONSENSUS_PROMPT = """You are the MedConsensus Consensus Agent, a chief-of-medicine style synthesizer.
Render a final diagnostic judgment for a synthetic clinical reasoning exercise. You are not a tiebreaker; weigh evidence, dissent, missing data, next questions/tests, ICD-10 codes, and red flags.
Output recommendations for clinician decision support only, not treatment."""


@dataclass(frozen=True)
class CaseFeatures:
    text: str
    lower: str

    def has_any(self, *terms: str) -> bool:
        return any(term.lower() in self.lower for term in terms)


def _features(case_text: str) -> CaseFeatures:
    return CaseFeatures(text=case_text, lower=case_text.lower())


def _evidence(features: CaseFeatures, candidates: list[tuple[str, str]]) -> list[str]:
    matches = [label for term, label in candidates if term.lower() in features.lower]
    return matches[:5] or ["No specialty-specific finding dominates the supplied synthetic case."]


class SpecialistAgent:
    name: str

    def __init__(self, llm_client: Any | None = None) -> None:
        self.llm_client = llm_client
        self.last_assessment_used_llm = False
        self.last_debate_used_llm = False

    def assess(self, case_text: str) -> SpecialistOutput:
        raise NotImplementedError

    def debate(self, case_text: str, assessments: list[SpecialistOutput]) -> DebateRound:
        self.last_debate_used_llm = False
        if self.llm_client is not None:
            try:
                result = self.llm_client.complete_json(
                    f"{SAFETY_PROMPT}\n\n{DEBATE_PROMPT}",
                    {
                        "agent": self.name,
                        "case": case_text,
                        "colleague_assessments": [item.model_dump() for item in assessments],
                    },
                    DebateRound,
                )
                if result.agent == self.name:
                    self.last_debate_used_llm = True
                    return result
            except Exception:
                self.last_debate_used_llm = False
        target = "cardiology" if self.name != "cardiology" else "pulmonology"
        return DebateRound(
            agent=self.name,  # type: ignore[arg-type]
            maintains_position=True,
            challenge_target=target,  # type: ignore[arg-type]
            challenge_argument=self._challenge(case_text, assessments),
            underweighted_finding=self._underweighted(case_text),
        )

    def _challenge(self, case_text: str, assessments: list[SpecialistOutput]) -> str:
        return "The competing assessment should separate primary pathology from secondary findings using the objective case data."

    def _underweighted(self, case_text: str) -> str:
        return "Time course and objective vital/lab abnormalities should be weighed together before narrowing the differential."

    def _llm_assess(self, case_text: str) -> SpecialistOutput | None:
        self.last_assessment_used_llm = False
        if self.llm_client is None:
            return None
        try:
            result = self.llm_client.complete_json(
                f"{SAFETY_PROMPT}\n\n{SPECIALIST_PROMPTS[self.name]}",
                {"agent": self.name, "case": case_text},
                SpecialistOutput,
            )
            if result.agent == self.name:
                self.last_assessment_used_llm = True
                return result
        except Exception:
            self.last_assessment_used_llm = False
        return None


class CardiologyAgent(SpecialistAgent):
    name = "cardiology"

    def assess(self, case_text: str) -> SpecialistOutput:
        llm_result = self._llm_assess(case_text)
        if llm_result is not None:
            return llm_result
        f = _features(case_text)
        low_cardiac_signal = f.has_any("bnp 60", "bnp 90", "no jvd", "no s3", "normal cardiac silhouette")
        if low_cardiac_signal:
            primary = DiagnosisCandidate(
                name="Primary cardiac cause less likely",
                confidence=62,
                supporting_evidence=_evidence(
                    f,
                    [
                        ("bnp 60", "Low BNP argues against congestive heart failure"),
                        ("bnp 90", "Low BNP argues against congestive heart failure"),
                        ("no jvd", "No jugular venous distension"),
                        ("no s3", "No S3 gallop"),
                        ("normal cardiac silhouette", "Normal cardiac silhouette"),
                    ],
                ),
                confirming_test="ECG and serial troponins to screen for cardiac mimics",
                icd10_code="R06.00",
            )
            secondary = DiagnosisCandidate(
                name="Demand ischemia or occult acute coronary syndrome",
                confidence=24,
                supporting_evidence=_evidence(
                    f,
                    [
                        ("tachycardia", "Tachycardia can raise myocardial oxygen demand"),
                        ("troponin", "Troponin testing is relevant when dyspnea overlaps with cardiac symptoms"),
                        ("chest", "Chest symptoms require cardiac screening"),
                    ],
                ),
                confirming_test="Serial ECGs and serial high-sensitivity troponins",
                icd10_code="I24.9",
            )
            return SpecialistOutput(
                agent="cardiology",
                diagnoses=[primary, secondary],
                challenge_to_rivals="Cardiac screening remains appropriate, but the supplied case lacks the core congestion markers for heart failure.",
            )
        heart_failure_evidence = _evidence(
            f,
            [
                ("orthopnea", "Orthopnea and 2-pillow sleep pattern"),
                ("jvd", "Jugular venous distension"),
                ("pitting edema", "Bilateral pitting ankle edema"),
                ("s3", "S3 gallop"),
                ("bnp", "Elevated BNP"),
                ("cardiomegaly", "Cardiomegaly on chest radiograph"),
                ("kerley", "Kerley B lines suggesting interstitial edema"),
                ("pleural effusions", "Bilateral pleural effusions"),
            ],
        )
        confidence = 88 if f.has_any("bnp", "s3", "jvd", "cardiomegaly") else 62
        return SpecialistOutput(
            agent="cardiology",
            diagnoses=[
                DiagnosisCandidate(
                    name="Acute decompensated heart failure",
                    confidence=confidence,
                    supporting_evidence=heart_failure_evidence,
                    confirming_test="Transthoracic echocardiogram with ECG and repeat cardiac biomarkers",
                    icd10_code="I50.9",
                ),
                DiagnosisCandidate(
                    name="Acute coronary syndrome with heart failure presentation",
                    confidence=45 if f.has_any("troponin", "chest pain") else 28,
                    supporting_evidence=_evidence(
                        f,
                        [
                            ("troponin", "Borderline troponin elevation"),
                            ("diabetes", "Diabetes increases atypical ACS risk"),
                            ("htn", "Longstanding hypertension is a coronary risk factor"),
                        ],
                    ),
                    confirming_test="Serial ECGs and serial high-sensitivity troponins",
                    icd10_code="I24.9",
                ),
            ],
            challenge_to_rivals="Pulmonary explanations must account for JVD, S3, edema, BNP elevation, and cardiomegaly together.",
        )

    def _challenge(self, case_text: str, assessments: list[SpecialistOutput]) -> str:
        return "Pulmonology overweights pleural effusions unless it explains the cardiac congestion pattern: S3, JVD, edema, BNP elevation, cardiomegaly, and Kerley B lines."

    def _underweighted(self, case_text: str) -> str:
        return "Creatinine rise and hyponatremia may reflect clinically important congestion and reduced effective arterial volume."


class PulmonologyAgent(SpecialistAgent):
    name = "pulmonology"

    def assess(self, case_text: str) -> SpecialistOutput:
        llm_result = self._llm_assess(case_text)
        if llm_result is not None:
            return llm_result
        f = _features(case_text)
        if f.has_any("d-dimer", "long flight", "unilateral calf", "clear lungs"):
            return SpecialistOutput(
                agent="pulmonology",
                diagnoses=[
                    DiagnosisCandidate(
                        name="Pulmonary embolism",
                        confidence=84,
                        supporting_evidence=_evidence(
                            f,
                            [
                                ("sudden dyspnea", "Sudden dyspnea"),
                                ("pleuritic", "Pleuritic chest discomfort"),
                                ("long flight", "Recent prolonged immobility"),
                                ("hr 118", "Tachycardia"),
                                ("spo2 91", "Hypoxemia"),
                                ("unilateral calf", "Unilateral calf tenderness"),
                                ("d-dimer", "Elevated D-dimer"),
                                ("clear lungs", "Clear lung exam despite dyspnea"),
                            ],
                        ),
                        confirming_test="CT pulmonary angiography if pretest probability supports imaging",
                        icd10_code="I26.99",
                    ),
                    DiagnosisCandidate(
                        name="Pleurisy or viral pleurodynia",
                        confidence=22,
                        supporting_evidence=["Pleuritic discomfort is present, but thromboembolic risk factors dominate."],
                        confirming_test="Clinical exam, ECG, chest imaging review, and inflammatory/infectious testing as indicated",
                        icd10_code="R07.1",
                    ),
                ],
                challenge_to_rivals="Cardiac anchoring is weak when BNP is low, cardiac silhouette is normal, and thromboembolic risk factors are explicit.",
            )
        if f.has_any("fever", "productive cough", "infiltrate", "wheezes", "copd"):
            return SpecialistOutput(
                agent="pulmonology",
                diagnoses=[
                    DiagnosisCandidate(
                        name="COPD exacerbation with community-acquired pneumonia",
                        confidence=86,
                        supporting_evidence=_evidence(
                            f,
                            [
                                ("fever", "Fever"),
                                ("productive cough", "Productive cough"),
                                ("wheezes", "Diffuse wheezes"),
                                ("right lower lobe crackles", "Focal right lower lobe crackles"),
                                ("wbc 15.8", "Leukocytosis"),
                                ("right lower lobe infiltrate", "Right lower lobe infiltrate"),
                                ("hyperinflation", "Hyperinflation"),
                                ("copd", "Known COPD history"),
                            ],
                        ),
                        confirming_test="Pulse oximetry/ABG if severe, sputum testing if indicated, and repeat chest imaging follow-up",
                        icd10_code="J44.0",
                    ),
                    DiagnosisCandidate(
                        name="Acute heart failure mimic",
                        confidence=16,
                        supporting_evidence=["Dyspnea overlaps with heart failure, but BNP is low and edema/JVD are absent."],
                        confirming_test="BNP trend and bedside ultrasound/echocardiography if clinical concern persists",
                        icd10_code="I50.9",
                    ),
                ],
                challenge_to_rivals="The focal infiltrate, fever, leukocytosis, wheezing, and low BNP make primary cardiogenic edema less defensible.",
            )
        effusion_evidence = _evidence(
            f,
            [
                ("dyspnea", "Progressive dyspnea"),
                ("crackles", "Bilateral basal crackles"),
                ("pleural effusions", "Bilateral pleural effusions on chest radiograph"),
                ("spo2 94", "Mild oxygenation abnormality"),
                ("smoking", "Prior smoking history"),
            ],
        )
        return SpecialistOutput(
            agent="pulmonology",
            diagnoses=[
                DiagnosisCandidate(
                    name="Cardiogenic pulmonary edema with pleural effusions",
                    confidence=76,
                    supporting_evidence=effusion_evidence,
                    confirming_test="Chest ultrasound and echocardiogram to correlate effusions with filling pressures",
                    icd10_code="J81.0",
                ),
                DiagnosisCandidate(
                    name="Pulmonary embolism",
                    confidence=34 if f.has_any("tachycardia", "hr 102", "spo2 94") else 20,
                    supporting_evidence=_evidence(
                        f,
                        [
                            ("hr 102", "Tachycardia"),
                            ("spo2 94", "Mildly reduced oxygen saturation"),
                            ("dyspnea", "Dyspnea is a core presenting symptom"),
                        ],
                    ),
                    confirming_test="D-dimer if low/intermediate pretest probability, or CT pulmonary angiography if indicated",
                    icd10_code="I26.99",
                ),
            ],
            challenge_to_rivals="Primary lung disease is less favored because the imaging pattern is more hydrostatic than focal infectious or obstructive.",
        )

    def _challenge(self, case_text: str, assessments: list[SpecialistOutput]) -> str:
        return "Cardiology should still account for pulmonary vascular mimics because tachycardia and dyspnea can overlap with pulmonary embolism."

    def _underweighted(self, case_text: str) -> str:
        return "Oxygen saturation is only mildly reduced, which supports congestion but does not eliminate pulmonary vascular disease."


class GeneralMedicineAgent(SpecialistAgent):
    name = "general_medicine"

    def assess(self, case_text: str) -> SpecialistOutput:
        llm_result = self._llm_assess(case_text)
        if llm_result is not None:
            return llm_result
        f = _features(case_text)
        if f.has_any("fever", "productive cough", "wbc 15.8", "infiltrate"):
            return SpecialistOutput(
                agent="general_medicine",
                diagnoses=[
                    DiagnosisCandidate(
                        name="Community-acquired pneumonia with COPD exacerbation",
                        confidence=82,
                        supporting_evidence=_evidence(
                            f,
                            [
                                ("fever", "Fever"),
                                ("productive cough", "Productive cough"),
                                ("wbc 15.8", "Leukocytosis"),
                                ("right lower lobe infiltrate", "Right lower lobe infiltrate"),
                                ("copd", "COPD history"),
                                ("spo2 90", "Hypoxemia"),
                            ],
                        ),
                        confirming_test="CBC/CMP trend, respiratory viral testing, sputum testing when indicated, and oxygenation assessment",
                        icd10_code="J18.9",
                    ),
                    DiagnosisCandidate(
                        name="Sepsis physiology from pulmonary source",
                        confidence=38,
                        supporting_evidence=["Fever, tachycardia, tachypnea, and leukocytosis raise systemic-risk concern."],
                        confirming_test="Lactate and blood cultures if clinically unstable or sepsis criteria are met",
                        icd10_code="A41.9",
                    ),
                ],
                challenge_to_rivals="A cardiac explanation is atypical because BNP is low and infectious findings are explicit.",
            )
        if f.has_any("long flight", "estrogen", "unilateral calf", "d-dimer"):
            return SpecialistOutput(
                agent="general_medicine",
                diagnoses=[
                    DiagnosisCandidate(
                        name="Pulmonary embolism",
                        confidence=80,
                        supporting_evidence=_evidence(
                            f,
                            [
                                ("sudden dyspnea", "Sudden dyspnea"),
                                ("pleuritic", "Pleuritic chest discomfort"),
                                ("long flight", "Recent prolonged immobility"),
                                ("estrogen", "Estrogen therapy"),
                                ("unilateral calf", "Unilateral calf tenderness"),
                                ("d-dimer", "Elevated D-dimer"),
                            ],
                        ),
                        confirming_test="Formal pretest probability scoring followed by CT pulmonary angiography or V/Q scan when indicated",
                        icd10_code="I26.99",
                    ),
                    DiagnosisCandidate(
                        name="Pneumothorax or pleural process",
                        confidence=18,
                        supporting_evidence=["Sudden dyspnea and pleuritic symptoms require imaging review even with no focal infiltrate."],
                        confirming_test="Chest radiograph review or point-of-care lung ultrasound",
                        icd10_code="R06.02",
                    ),
                ],
                challenge_to_rivals="The case is atypical for heart failure because the cardiac silhouette and BNP do not support congestion.",
            )
        return SpecialistOutput(
            agent="general_medicine",
            diagnoses=[
                DiagnosisCandidate(
                    name="Volume overload from new heart failure syndrome",
                    confidence=82 if f.has_any("edema", "bnp", "orthopnea") else 58,
                    supporting_evidence=_evidence(
                        f,
                        [
                            ("orthopnea", "Orthopnea"),
                            ("edema", "Peripheral edema"),
                            ("bnp", "Elevated BNP"),
                            ("cr 1.4", "Creatinine above baseline"),
                            ("na 133", "Mild hyponatremia"),
                            ("htn", "Longstanding hypertension"),
                            ("t2dm", "Diabetes as a cardiovascular risk factor"),
                        ],
                    ),
                    confirming_test="Medication, diet, renal, thyroid, ECG, and echocardiographic review",
                    icd10_code="E87.70",
                ),
                DiagnosisCandidate(
                    name="Renal or endocrine contributor to fluid retention",
                    confidence=30,
                    supporting_evidence=_evidence(
                        f,
                        [
                            ("cr 1.4", "Creatinine increased from baseline"),
                            ("na 133", "Hyponatremia"),
                            ("edema", "Peripheral edema"),
                        ],
                    ),
                    confirming_test="CMP, urinalysis with protein quantification, TSH, medication review",
                    icd10_code="R60.9",
                ),
            ],
            challenge_to_rivals="The obvious cardiac answer is strong, but renal decline, sodium abnormality, and systemic triggers need active review.",
        )

    def _challenge(self, case_text: str, assessments: list[SpecialistOutput]) -> str:
        return "Both subspecialty views should look for the precipitating cause instead of stopping at the syndrome label."

    def _underweighted(self, case_text: str) -> str:
        return "No prior cardiac history is documented, so new structural disease, ischemia, medication effects, and renal triggers all need clarification."


class ConsensusAgent:
    def __init__(self, llm_client: Any | None = None) -> None:
        self.llm_client = llm_client
        self.last_used_llm = False

    def synthesize(self, case_text: str, assessments: list[SpecialistOutput], debates: list[DebateRound]) -> FinalVerdict:
        llm_verdict = self._llm_synthesize(case_text, assessments, debates)
        if llm_verdict is not None:
            return llm_verdict
        return self._deterministic_synthesize(case_text)

    def _llm_synthesize(
        self,
        case_text: str,
        assessments: list[SpecialistOutput],
        debates: list[DebateRound],
    ) -> FinalVerdict | None:
        self.last_used_llm = False
        if self.llm_client is None:
            return None
        try:
            result = self.llm_client.complete_json(
                f"{SAFETY_PROMPT}\n\n{CONSENSUS_PROMPT}",
                {
                    "case": case_text,
                    "specialist_assessments": [item.model_dump() for item in assessments],
                    "debate_round": [item.model_dump() for item in debates],
                },
                FinalVerdict,
            )
            self.last_used_llm = True
            return result
        except Exception:
            self.last_used_llm = False
            return None

    def _deterministic_synthesize(self, case_text: str) -> FinalVerdict:
        f = _features(case_text)
        if f.has_any("d-dimer", "long flight", "unilateral calf", "estrogen"):
            return FinalVerdict(
                primary_diagnosis=FinalVerdictDiagnosis(
                    name="Pulmonary embolism",
                    icd10_code="I26.99",
                    confidence=84,
                    reasoning="Sudden dyspnea, pleuritic discomfort, tachycardia, hypoxemia, recent flight, estrogen exposure, unilateral calf tenderness, elevated D-dimer, clear lungs, low BNP, and normal cardiac silhouette align most strongly with pulmonary embolism.",
                ),
                runner_up_diagnoses=[
                    RunnerUpDiagnosis(name="Acute coronary syndrome", icd10_code="I24.9", brief_reason="Chest discomfort and dyspnea still justify ECG/troponin screening."),
                    RunnerUpDiagnosis(name="Pneumothorax or pleural process", icd10_code="R06.02", brief_reason="Pleuritic symptoms require imaging review even when PE is favored."),
                ],
                dissenting_view_worth_noting="Cardiology's screening recommendation remains important because dyspnea and chest symptoms can hide cardiac disease.",
                next_steps=[
                    "Ask about hemoptysis, syncope, prior VTE, malignancy, recent surgery, pregnancy status when relevant, and bleeding risk.",
                    "Apply a PE pretest probability tool, obtain ECG/troponin as indicated, and use D-dimer only in the proper risk stratum.",
                    "Consider CT pulmonary angiography or V/Q scan when pretest probability and contraindications support imaging.",
                ],
                red_flags=["Hypotension, syncope, worsening hypoxemia, right heart strain, or severe respiratory distress."],
            )
        if f.has_any("fever", "productive cough", "wbc 15.8", "right lower lobe infiltrate"):
            return FinalVerdict(
                primary_diagnosis=FinalVerdictDiagnosis(
                    name="COPD exacerbation with community-acquired pneumonia",
                    icd10_code="J44.0",
                    confidence=85,
                    reasoning="Fever, productive cough, wheezing, focal crackles, leukocytosis, right lower lobe infiltrate, hyperinflation, COPD history, low BNP, and absent JVD/edema favor a pulmonary infectious process over cardiogenic edema.",
                ),
                runner_up_diagnoses=[
                    RunnerUpDiagnosis(name="Sepsis physiology from pulmonary source", icd10_code="A41.9", brief_reason="Fever, tachycardia, tachypnea, and leukocytosis require severity assessment."),
                    RunnerUpDiagnosis(name="Heart failure mimic", icd10_code="I50.9", brief_reason="Dyspnea overlaps, but low BNP and absent congestion signs argue against it."),
                ],
                dissenting_view_worth_noting="Cardiology's recommendation to screen cardiac mimics is reasonable if symptoms, ECG, or biomarkers shift.",
                next_steps=[
                    "Ask about sputum volume/color, baseline oxygen use, recent antibiotics/steroids, sick contacts, vaccination status, and prior exacerbations.",
                    "Check oxygenation trend, CBC/CMP, respiratory viral testing, and sputum studies when clinically indicated.",
                    "Review chest imaging and consider follow-up imaging based on clinical course and risk factors.",
                ],
                red_flags=["Increasing oxygen requirement, exhaustion, altered mental status, hypotension, lactate elevation, or inability to protect airway."],
            )
        primary = "Acute decompensated heart failure"
        confidence = 86 if f.has_any("bnp", "jvd", "s3", "kerley") else 68
        explanation = None if confidence >= 70 else "Additional objective cardiac and pulmonary testing is needed to separate congestion from mimics."
        return FinalVerdict(
            primary_diagnosis=FinalVerdictDiagnosis(
                name=primary,
                icd10_code="I50.9",
                confidence=confidence,
                reasoning="The most coherent synthesis is a heart failure syndrome because dyspnea, orthopnea, JVD, edema, S3, elevated BNP, cardiomegaly, pleural effusions, and Kerley B lines align as one physiology.",
            ),
            runner_up_diagnoses=[
                RunnerUpDiagnosis(name="Pulmonary embolism", icd10_code="I26.99", brief_reason="Dyspnea and tachycardia justify consideration even though the congestion pattern is stronger."),
                RunnerUpDiagnosis(name="Acute coronary syndrome", icd10_code="I24.9", brief_reason="Borderline troponin and cardiometabolic risk require serial ECG/troponin evaluation."),
            ],
            dissenting_view_worth_noting="Pulmonology's pulmonary vascular caution deserves attention if hypoxemia, pleuritic symptoms, unilateral leg findings, or high-risk history is uncovered.",
            next_steps=[
                "Ask about chest pain, exertional symptoms, medication adherence, diet/salt load, recent infection, and prior echocardiogram history.",
                "Obtain ECG, repeat troponin, BNP trend, CMP, CBC, urinalysis, and thyroid testing as clinically appropriate.",
                "Obtain transthoracic echocardiogram and review chest imaging; consider PE workup only if pretest probability supports it.",
            ],
            red_flags=[
                "Worsening hypoxemia, respiratory distress, syncope, hypotension, ischemic ECG changes, or rising troponin.",
                "Rapid renal deterioration, severe electrolyte abnormality, or altered mental status.",
            ],
            confidence_below_70_explanation=explanation,
        )
