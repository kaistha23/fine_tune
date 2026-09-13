"""The one definition of what the model is shown.

There were three, and they disagreed:

  dataset.py     SYSTEM_PROMPT + json.dumps(case)
  omlx_client.py a different system prompt + {question, context, response_schema}
  feedback.py    no system prompt at all + "Case reference: FB-123"

Fine-tuning on one shape and serving another wastes the run: with mask_prompt the loss
sits entirely on the assistant turn, so the adapter learns to emit a target conditioned on
an input structure it never sees in production. The feedback variant was worse than
useless - it taught the model to produce a full credit assessment from an identifier
string containing none of the case data, which is training it to hallucinate.

Every path now builds its turns here. A change to the prompt is a change to the training
distribution, so it belongs in one file with a version on it: bump PROMPT_VERSION when the
wording changes, because an adapter trained under one version is not comparable with one
trained under another.
"""

from __future__ import annotations

import json
from typing import Any

# Bump whenever SYSTEM_PROMPT or the user-message shape changes. Recorded in dataset
# provenance so a checkpoint can be traced to the prompt it was trained under.
PROMPT_VERSION = "v5.0.0"

SYSTEM_PROMPT = (
    "You are a credit-risk advisory copilot. Use only the supplied factsheet and evidence. "
    "Separate facts, model outputs, inference and recommendations. Do not invent evidence, "
    "thresholds or customer facts. Every fact must cite: use an evidence_id from the "
    "evidence list for anything drawn from policy or regulation, and the factsheet's "
    "case_id for anything drawn from the obligor's own data. Identify missing information "
    "and abstain when evidence is insufficient. For a numeric threshold comparison, include "
    "a derivation using a valid calculated metric, its exact unit, the cited threshold, "
    "operator and comparison result. Treat supplied rule_evaluations as deterministic; "
    "do not contradict their status or substitute another rule. Return JSON matching the "
    "supplied schema. "
    "Set human_approval_required=true for recommendations."
)


def build_user_content(
    question: str,
    factsheet: dict[str, Any],
    evidence: list[dict[str, Any]],
    response_schema: dict[str, Any] | None = None,
    rule_evaluations: list[dict[str, Any]] | None = None,
) -> str:
    """The user turn, identical at training and inference time.

    Key order is fixed rather than left to dict insertion order: the serialised text is
    what the model sees, and a reordering would be a silent change to the training
    distribution.
    """
    payload: dict[str, Any] = {
        "question": question,
        "context": {
            "factsheet": factsheet,
            "evidence": evidence,
            "rule_evaluations": rule_evaluations or [],
        },
    }
    if response_schema is None:
        from credit_risk.schemas import CreditResponse, compact_json_schema

        response_schema = compact_json_schema(CreditResponse.model_json_schema())
    payload["response_schema"] = response_schema
    return json.dumps(payload, ensure_ascii=False, sort_keys=False)


def build_messages(
    question: str,
    factsheet: dict[str, Any],
    evidence: list[dict[str, Any]],
    response_schema: dict[str, Any] | None = None,
    rule_evaluations: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_content(
                question, factsheet, evidence, response_schema, rule_evaluations
            ),
        },
    ]
