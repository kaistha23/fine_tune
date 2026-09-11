from datetime import date

import pytest

from credit_risk import data_service as ds
from credit_risk.evaluation.gates import ReleaseGates, evaluate_gates
from credit_risk.evaluation.metrics import score_cases
from credit_risk.guardrails import validate_output
from credit_risk.rag.filters import RetrievalPolicy, chunk_is_visible
from credit_risk.rag.index import InMemoryPolicyIndex
from credit_risk.rag.retriever import PolicyRetriever
from credit_risk.rag.schemas import AccessContext, PolicyChunk
from credit_risk.schemas import CreditResponse, Evidence, QueryPlan


def test_false_claim_and_empty_eval_fail():
    e = Evidence(
        evidence_id="E",
        jurisdiction="SAMA",
        document_id="D",
        document_version="1",
        section="1",
        text="PD is 2 percent.",
        score=1,
    )
    r = CreditResponse(
        answer_status="ANSWERED",
        executive_summary="PD is 99 percent.",
        facts=[{"statement": "PD is 99 percent.", "evidence_ids": ["E"]}],
    )
    assert not validate_output(r, [e]).passed
    assert not evaluate_gates(score_cases([]), ReleaseGates("configs/evaluation_thresholds.yaml"))[
        "promotable"
    ]


def test_role_and_irrelevant_evidence():
    p = RetrievalPolicy("configs/retrieval.yaml")
    ctx = AccessContext(jurisdiction="SAMA", role="credit_analyst", as_of_date=date(2025, 1, 1))
    c = PolicyChunk(
        chunk_id="c",
        document_id="d",
        document_version="1",
        jurisdiction="SAMA",
        approval_status="approved",
        allowed_roles=["regulator_liaison"],
        text="Roses need water.",
    )
    assert not chunk_is_visible(c, p.build_predicate(ctx))
    c.allowed_roles = []
    i = InMemoryPolicyIndex()
    i.upsert([c], namespaces={"SAMA": "policy_sama"})
    assert (
        PolicyRetriever(p, i).retrieve("capital adequacy covenants", ctx)["answer_status"]
        == "INSUFFICIENT_EVIDENCE"
    )


@pytest.mark.parametrize(
    "update",
    [
        {"pit_pd": 7},
        {"pit_pd": float("nan")},
        {"obligor_id": "wrong"},
        {"observation_date": "bad"},
        {"data_cutoff_date": None},
    ],
)
def test_result_contract(update):
    from test_secure_review import PLAN

    p = QueryPlan(**PLAN)
    c = ds.compiler.compile(p)
    row = {
        "obligor_id": p.obligor_id,
        "portfolio": "corporate",
        "jurisdiction": "SAMA",
        "observation_date": "2025-06-30",
        "data_cutoff_date": "2025-06-30",
        "model_run_date": "2025-06-30",
        "pit_pd": 0.02,
    }
    row.update(update)
    with pytest.raises(ds.ResultValidationError):
        ds.validate_result([row], c, p, ds.registry.data["query_controls"])


def test_malformed_model_response_counts_as_failure():
    from credit_risk.evaluation.metrics import GoldCase, score_case

    case = GoldCase(
        case_id="g", portfolio="retail", task_type="factsheet", jurisdiction="SAMA", question="?"
    )
    assert not score_case(case, "not json").schema_valid


def test_negated_statement_is_not_supported():
    from credit_risk.guardrails import supported_text

    assert not supported_text("PD is 2 percent.", "It is not true that PD is 2 percent.")


def test_mismatched_benchmarks_are_rejected():
    from credit_risk.evaluation.metrics import score_cases
    from credit_risk.evaluation.runner import compare_adapters

    c = score_cases([])
    d = {**c, "case_set_hash": "different"}
    report = compare_adapters(c, d, ReleaseGates("configs/evaluation_thresholds.yaml"))
    assert not report["promote"]
    assert any("Benchmark" in s for s in report["portfolio_regressions"])
