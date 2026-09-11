"""Exercise the running local Compose review flow using the synthetic fixture."""

import json
from pathlib import Path

import httpx

root = Path(__file__).resolve().parents[1]
plan = {
    "portfolio": "corporate",
    "jurisdiction": "SAMA",
    "obligor_id": "OBL-0008",
    "date_from": "2025-01-31",
    "date_to": "2025-12-31",
    "as_of_date": "2026-01-15",
    "metrics": ["pit_pd"],
}
headers = {"Authorization": "Bearer " + (root / ".reviewer-token").read_text().strip()}
with httpx.Client(base_url="http://127.0.0.1:8080", headers=headers, timeout=30) as client:
    assert client.get("/review").status_code == 200

    def prepare():
        response = client.post(
            "/v1/query/validate", json={"user_text": "Synthetic smoke test", "query_plan": plan}
        )
        response.raise_for_status()
        return response.json()["sql_review"]["revision_id"]

    rid = prepare()
    assert (
        client.post(
            "/v1/query/review", json={"revision_id": rid, "decision": "approved"}
        ).status_code
        == 200
    )
    body = {"revision_id": rid, "user_text": "Synthetic smoke test", "query_plan": plan}
    changed = {**body, "query_plan": {**plan, "obligor_id": "SYNTH-OTHER"}}
    assert client.post("/v1/factsheet", json=changed).status_code == 409
    result = client.post("/v1/factsheet", json=body)
    assert result.status_code == 200, result.text
    assert client.post("/v1/factsheet", json=body).status_code == 409
    rid = prepare()
    invalid = client.post(
        "/v1/query/review",
        json={
            "revision_id": rid,
            "decision": "rejected",
            "comment": "Reject executable SQL",
            "corrected_query_plan": {**plan, "sql": "SELECT 1"},
        },
    )
    assert invalid.status_code == 422
    rid = prepare()
    corrected_plan = {**plan, "metrics": ["pit_pd", "stage"]}
    response = client.post(
        "/v1/query/review",
        json={
            "revision_id": rid,
            "decision": "rejected",
            "comment": "Include stage",
            "corrected_query_plan": corrected_plan,
        },
    )
    assert response.status_code == 200, response.text
    new = response.json()["corrected_sql_review"]["revision_id"]
    new_body = {**body, "revision_id": new, "query_plan": corrected_plan}
    assert client.post("/v1/factsheet", json=new_body).status_code == 409
    assert (
        client.post(
            "/v1/query/review", json={"revision_id": new, "decision": "approved"}
        ).status_code
        == 200
    )
    assert client.post("/v1/factsheet", json=new_body).status_code == 200
assert httpx.post("http://127.0.0.1:8080/v1/factsheet", json=body).status_code == 401
print(
    json.dumps(
        {
            "review_page": "passed",
            "approved_execution": "passed",
            "changed_identity": "blocked",
            "replay": "blocked",
            "edited_sql": "blocked",
            "correction_reapproval": "passed",
            "unauthenticated": "blocked",
        }
    )
)
