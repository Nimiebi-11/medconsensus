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


def test_no_phi_storage_behavior(tmp_path) -> None:
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
