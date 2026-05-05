from pathlib import Path

from fastapi.testclient import TestClient

from medconsensus.app import app
from medconsensus.schemas import DISCLAIMER


client = TestClient(app)


DEMO_CASE = {
    "case_id": "pytest-synthetic-heart-failure",
    "synthetic": True,
    "patient_case": (
        "Patient: 58F synthetic case. Progressively worsening dyspnea x 3 weeks, orthopnea, "
        "2-pillow sleep. HR 102, BP 148/92, SpO2 94% on room air. Bilateral crackles, JVD, "
        "2+ pitting edema, S3 gallop. BNP 890, borderline troponin, Cr 1.4, Na 133. "
        "CXR with cardiomegaly, bilateral pleural effusions, and Kerley B lines."
    ),
}


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_homepage_demo_controls() -> None:
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "API Docs" in html
    assert "id=\"healthButton\"" in html
    assert "id=\"agentCardButton\"" in html
    assert "function resetCase()" in html
    assert "function showEndpoint(kind, path)" in html
    assert "Show raw JSON" in html
    assert "apiPanel" in html
    assert "Health Check" in html
    assert "Agent Card Summary" in html
    assert "Fallback mode is active because no paid LLM credits are available" in html
    assert "LLM Multi-Agent Mode" in html
    assert "Deterministic Fallback Mode" in html
    assert "1. Specialist Assessments" in html
    assert "2. Debate Summary" in html
    assert "3. Final Consensus" in html


def test_synthetic_case_invocation() -> None:
    response = client.post("/invoke", json=DEMO_CASE)
    assert response.status_code == 200
    payload = response.json()
    assert payload["consensus"]["disclaimer"] == DISCLAIMER
    assert "does not diagnose patients" in " ".join(payload["consensus"]["safety_notes"]).lower()
    assert payload["consensus"]["most_likely_diagnosis"]
    assert isinstance(payload["consensus"]["confidence"], (int, float))
    assert payload["consensus"]["icd10_codes"]


def test_json_output_schema() -> None:
    response = client.post("/tasks", json=DEMO_CASE)
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"metadata", "case_summary", "specialist_assessments", "debate_summary", "consensus"}
    assert payload["metadata"]["mode"] in {"deterministic_fallback", "llm_multi_agent"}
    assert len(payload["specialist_assessments"]) == 3
    assert {"agent", "top_diagnosis", "differential_diagnoses", "supporting_evidence", "concerns_or_missing_data"} <= set(
        payload["specialist_assessments"][0].keys()
    )
    assert 0 <= payload["consensus"]["confidence"] <= 100
    assert {"code", "label"} <= set(payload["consensus"]["icd10_codes"][0].keys())
    assert payload["consensus"]["recommended_next_questions"]
    assert payload["consensus"]["recommended_next_tests"]


def test_mcp_discovery_and_tool_invocation() -> None:
    discovery = client.get("/mcp")
    assert discovery.status_code == 200
    assert discovery.json()["transport"] == "streamable_http"
    assert "run_consensus" in discovery.json()["tools"]

    tools = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert tools.status_code == 200
    assert tools.json()["result"]["tools"][0]["name"] == "run_consensus"

    call = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "run_consensus",
                "arguments": {
                    "synthetic": True,
                    "patient_case": DEMO_CASE["patient_case"],
                },
            },
        },
    )
    assert call.status_code == 200
    result = call.json()["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["consensus"]["disclaimer"] == DISCLAIMER
    assert result["structuredContent"]["metadata"]["mode"] in {"deterministic_fallback", "llm_multi_agent"}


def test_no_phi_storage_behavior() -> None:
    storage_dir = Path("storage")
    before = set(storage_dir.rglob("*")) if storage_dir.exists() else set()
    response = client.post("/invoke", json=DEMO_CASE)
    assert response.status_code == 200
    after = set(storage_dir.rglob("*")) if storage_dir.exists() else set()
    assert after == before

    rejected = client.post(
        "/invoke",
        json={
            "synthetic": True,
            "patient_case": "Synthetic-looking case with MRN 12345 and date of birth 01/01/1970 included.",
        },
    )
    assert rejected.status_code == 422
