from __future__ import annotations

from medconsensus.schemas import ConsensusReport


def render_markdown(report: ConsensusReport) -> str:
    """Render a concise demo-friendly report from the structured JSON."""

    lines = [
        "# MedConsensus Report",
        "",
        f"**Case summary:** {report.case_summary}",
        "",
        "## Specialist Assessments",
    ]
    for item in report.specialist_assessments:
        lines.extend(
            [
                f"### {item.agent}",
                f"- Top diagnosis: {item.top_diagnosis}",
                f"- Differential: {', '.join(item.differential_diagnoses)}",
                f"- Evidence: {'; '.join(item.supporting_evidence)}",
                f"- Concerns/missing data: {'; '.join(item.concerns_or_missing_data)}",
            ]
        )

    lines.extend(["", "## Debate Summary"])
    for item in report.debate_summary:
        lines.extend(
            [
                f"### {item.from_agent}",
                f"- Challenge: {item.challenge}",
                f"- Response/revision: {item.response_or_revision}",
            ]
        )

    consensus = report.consensus
    lines.extend(
        [
            "",
            "## Consensus",
            f"- Most likely diagnosis: {consensus.most_likely_diagnosis}",
            f"- Must-not-miss diagnoses: {', '.join(consensus.must_not_miss_diagnoses)}",
            f"- ICD-10 codes: {', '.join(f'{item.code} ({item.label})' for item in consensus.icd10_codes)}",
            f"- Confidence: {consensus.confidence}/100",
            f"- Next questions: {'; '.join(consensus.recommended_next_questions)}",
            f"- Next tests: {'; '.join(consensus.recommended_next_tests)}",
            f"- Safety notes: {'; '.join(consensus.safety_notes)}",
            "",
            f"**Disclaimer:** {consensus.disclaimer}",
        ]
    )
    return "\n".join(lines)
