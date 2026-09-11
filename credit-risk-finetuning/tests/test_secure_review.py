import hashlib
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from credit_risk import api
from credit_risk import data_service as ds
from credit_risk.settings import settings

PLAN = {
    "portfolio": "corporate",
    "jurisdiction": "SAMA",
    "obligor_id": "OBL-0008",
    "date_from": "2025-01-31",
    "date_to": "2025-12-31",
    "as_of_date": "2026-01-15",
    "metrics": ["pit_pd"],
}


@pytest.fixture
def clients(tmp_path):
    from test_api_factsheet import ensure_fixture

    ensure_fixture()
    with (
        patch.object(settings, "service_token", "synthetic-internal"),
        patch.object(settings, "review_store_path", tmp_path / "reviews.db"),
        patch.object(
            settings,
            "reviewers",
            {
                "alice": {
                    "role": "credit_analyst",
                    "token_sha256": hashlib.sha256(b"synthetic-reviewer").hexdigest(),
                }
            },
        ),
    ):
        dc = TestClient(ds.app)

        def call(operation, payload):
            with patch.object(settings, "service_role", "data_service"):
                r = dc.post(
                    "/internal/v1/query/" + operation,
                    json={**payload, "reviewer_id": "alice"},
                    headers={"x-service-token": settings.service_token},
                )
            if r.status_code >= 400:
                from fastapi import HTTPException

                raise HTTPException(r.status_code, r.json()["detail"])
            return r.json()

        with patch.object(api, "data_call", call):
            yield TestClient(api.app, headers={"Authorization": "Bearer synthetic-reviewer"}), dc


def prepare(c, plan=PLAN):
    r = c.post("/v1/query/validate", json={"user_text": "Synthetic review", "query_plan": plan})
    assert r.status_code == 200, r.text
    return r.json()["sql_review"]


def test_auth_and_spoofing(clients):
    c, _ = clients
    assert TestClient(api.app).post("/v1/query/fetch", json={}).status_code == 401
    assert (
        c.post(
            "/v1/analyse",
            json={"query_plan": PLAN, "user_text": "hi", "reviewer_role": "regulator_liaison"},
        ).status_code
        == 422
    )
    assert (
        c.post("/v1/query/fetch", json={"query_plan": PLAN, "user_text": "hi"}).status_code == 409
    )


def test_approval_bound_and_single_use(clients):
    c, _ = clients
    p = prepare(c)
    assert (
        c.post(
            "/v1/query/review", json={"revision_id": p["revision_id"], "decision": "approved"}
        ).status_code
        == 200
    )
    body = {
        "user_text": "hi",
        "query_plan": {**PLAN, "obligor_id": "SYNTH-OTHER"},
        "revision_id": p["revision_id"],
    }
    assert c.post("/v1/query/fetch", json=body).status_code == 409
    body["query_plan"] = PLAN
    r = c.post("/v1/query/fetch", json=body)
    assert r.status_code == 200, r.text
    assert c.post("/v1/query/fetch", json=body).status_code == 409


def test_invalid_correction_audited(clients):
    c, _ = clients
    p = prepare(c)
    r = c.post(
        "/v1/query/review",
        json={
            "revision_id": p["revision_id"],
            "decision": "rejected",
            "comment": "bad column",
            "corrected_query_plan": {**PLAN, "sql": "DROP TABLE x"},
        },
    )
    assert r.status_code == 422
    with ds.store().connect() as con:
        assert (
            con.execute(
                "SELECT count(*) FROM events WHERE event='correction_validation_failed'"
            ).fetchone()[0]
            == 1
        )
    assert ds.store().get(p["revision_id"], "alice")["status"] == "rejected"


def test_correction_requires_fresh_approval(clients):
    c, _ = clients
    p = prepare(c)
    r = c.post(
        "/v1/query/review",
        json={
            "revision_id": p["revision_id"],
            "decision": "rejected",
            "comment": "add stage",
            "corrected_query_plan": {**PLAN, "metrics": ["pit_pd", "stage"]},
        },
    )
    assert r.status_code == 200, r.text
    new = r.json()["corrected_sql_review"]
    assert new["status"] == "pending"
    assert new["request_digest"] != p["request_digest"]


def test_internal_auth(clients):
    _, dc = clients
    assert (
        dc.post(
            "/internal/v1/query/execute", json={"query_plan": PLAN, "reviewer_id": "alice"}
        ).status_code
        == 401
    )


def test_correction_revokes_already_approved_revision(clients):
    c, _ = clients
    packet = prepare(c)
    rid = packet["revision_id"]
    assert (
        c.post("/v1/query/review", json={"revision_id": rid, "decision": "approved"}).status_code
        == 200
    )
    corrected = {**PLAN, "metrics": ["pit_pd", "stage"]}
    response = c.post(
        "/v1/query/review",
        json={
            "revision_id": rid,
            "decision": "rejected",
            "comment": "Include stage",
            "corrected_query_plan": corrected,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["corrected_sql_review"]["status"] == "pending"
    assert (
        c.post(
            "/v1/query/fetch", json={"revision_id": rid, "user_text": "execute", "query_plan": PLAN}
        ).status_code
        == 409
    )


@pytest.mark.parametrize("change", ["snapshot", "schema", "policy"])
def test_stale_approval_fails_closed(clients, change):
    c, _ = clients
    packet = prepare(c)
    rid = packet["revision_id"]
    assert (
        c.post("/v1/query/review", json={"revision_id": rid, "decision": "approved"}).status_code
        == 200
    )
    context = (
        patch.object(ds, "snapshot_hash", return_value="changed")
        if change == "snapshot"
        else patch.dict(
            (ds.registry if change == "schema" else ds.policy).data, {"revision_marker": "changed"}
        )
    )
    with context:
        assert (
            c.post(
                "/v1/query/fetch",
                json={"revision_id": rid, "user_text": "execute", "query_plan": PLAN},
            ).status_code
            == 409
        )


def test_valid_correction_is_typed_feedback(clients):
    import json

    c, _ = clients
    rid = prepare(c)["revision_id"]
    corrected = {**PLAN, "metrics": ["pit_pd", "stage"]}
    response = c.post(
        "/v1/query/review",
        json={
            "revision_id": rid,
            "decision": "rejected",
            "comment": "Include stage",
            "corrected_query_plan": corrected,
        },
    )
    assert response.status_code == 200
    with ds.store().connect() as con:
        row = con.execute(
            "SELECT payload FROM records WHERE identity=? ORDER BY seq DESC LIMIT 1", (rid,)
        ).fetchone()
    record = json.loads(row["payload"])
    assert record["corrected_query_plan"]["metrics"] == ["pit_pd", "stage"]
    assert record["sql_review_packet"]["parameter_values"]
    assert set(record["sql_review_packet"]["parameter_values"]) == {"***MASKED***"}
