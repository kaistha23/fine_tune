"""Explicit authenticated clients and real approval flow for integration tests."""

import hashlib

from fastapi.testclient import TestClient as Client

from credit_risk.settings import settings


def TestClient(app, **kwargs):
    settings.reviewers = {
        "test-reviewer": {
            "role": "credit_analyst",
            "token_sha256": hashlib.sha256(b"test-reviewer-credential").hexdigest(),
        }
    }
    settings.service_token = "test-internal-credential"
    return Client(app, headers={"Authorization": "Bearer test-reviewer-credential"}, **kwargs)


def execute_reviewed(client, plan):
    headers = {"x-service-token": settings.service_token}
    body = {"query_plan": plan, "reviewer_id": "test-reviewer"}
    r = client.post("/internal/v1/query/prepare", json=body, headers=headers)
    if r.status_code != 200:
        return r
    rid = r.json()["revision_id"]
    r = client.post(
        "/internal/v1/query/review",
        json={"reviewer_id": "test-reviewer", "revision_id": rid, "decision": "approved"},
        headers=headers,
    )
    if r.status_code != 200:
        return r
    return client.post(
        "/internal/v1/query/execute", json={**body, "revision_id": rid}, headers=headers
    )
