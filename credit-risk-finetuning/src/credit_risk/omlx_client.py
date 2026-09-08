from __future__ import annotations

import json
from typing import Any

import httpx

from credit_risk.schemas import CreditResponse, Evidence


class OMLXClient:
    def __init__(self, base_url: str, model: str, api_key: str = "",
                 timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        # oMLX refuses unauthenticated requests when a key is configured. Without this the
        # whole guarded path failed at the last step with "Model service unavailable" -
        # correct behaviour, since the raw 401 must not reach the caller, but it made an
        # auth problem look like an outage.
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def generate_credit_response(self, question: str, factsheet: dict[str, Any],
                                 evidence: list[Evidence]) -> CreditResponse:
        context = {
            "factsheet": factsheet,
            "evidence": [item.model_dump(mode="json") for item in evidence],
        }
        response_schema = CreditResponse.model_json_schema()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a credit-risk advisory copilot. Use only supplied facts and evidence. "
                    "Separate facts from inference. Every fact must cite: use an evidence_id from "
                    "the evidence list for anything drawn from policy or regulation, and the "
                    "factsheet's case_id for anything drawn from the obligor's own data. Return "
                    "JSON matching the supplied schema. Set human_approval_required=true for "
                    "recommendations."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"question": question, "context": context,
                                       "response_schema": response_schema}),
            },
        ]
        with httpx.Client(timeout=self.timeout) as client:
            result = client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 2500,
                    "response_format": {"type": "json_object"},
                },
            )
            result.raise_for_status()
        content = result.json()["choices"][0]["message"]["content"]
        return CreditResponse.model_validate_json(content)
