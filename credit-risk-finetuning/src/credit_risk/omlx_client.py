from __future__ import annotations

import json
from typing import Any

import httpx

from credit_risk.schemas import CreditResponse, Evidence


class OMLXClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

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
                    "Separate facts from inference. Cite evidence_id values. Return JSON matching "
                    "the supplied schema. Set human_approval_required=true for recommendations."
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
