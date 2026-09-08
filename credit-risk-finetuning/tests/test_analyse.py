"""The guarded inference path and action control (findings H8, and Plan dimension 4).

omlx_client.py existed but had no caller, so validate_output, the CreditResponse contract
and the abstention logic were never exercised end to end.
"""
import subprocess
import sys
import unittest
from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient

from credit_risk import api, data_service
from credit_risk.rag.filters import RetrievalPolicy
from credit_risk.rag.index import InMemoryPolicyIndex
from credit_risk.rag.retriever import PolicyRetriever
from credit_risk.rag.schemas import PolicyChunk
from credit_risk.risk_tiers import classify, gate
from credit_risk.schemas import (
    AnswerStatus,
    CreditResponse,
    InferenceClaim,
    Jurisdiction,
    Portfolio,
    QueryPlan,
    SupportedClaim,
)

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "data" / "curated" / "credit_risk.duckdb"
POLICY = ROOT / "configs" / "retrieval.yaml"


def ensure_fixture() -> None:
    if not FIXTURE.is_file():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fixture.py")],
                       cwd=ROOT, check=True, capture_output=True)


def plan(**overrides) -> QueryPlan:
    base = dict(
        portfolio=Portfolio.CORPORATE, jurisdiction=Jurisdiction.SAMA,
        obligor_id="OBL-0008", date_from=date(2025, 1, 31), date_to=date(2025, 12, 31),
        as_of_date=date(2026, 1, 15), metrics=["current_ratio", "dscr", "pit_pd", "stage"],
    )
    base.update(overrides)
    return QueryPlan(**base)


def sama_chunk() -> PolicyChunk:
    return PolicyChunk(
        chunk_id="s1", jurisdiction=Jurisdiction.SAMA, document_id="SAMA-DOC-4",
        document_version="1.0", section_id="7.2", approval_status="approved",
        confidentiality_level="internal",
        text="A significant increase in credit risk requires stage 2 classification.",
    )


def response(**overrides) -> CreditResponse:
    base = dict(
        answer_status=AnswerStatus.ANSWERED,
        executive_summary="Leverage has risen and utilisation is elevated.",
        facts=[SupportedClaim(statement="Stage 2 applies on SICR.",
                              evidence_ids=["SAMA-DOC-4#7.2"])],
        inferences=[InferenceClaim(statement="Deterioration is likely to continue.",
                                   basis="utilisation trend", confidence=0.6)],
        recommendation="",
        human_approval_required=True,
    )
    base.update(overrides)
    return CreditResponse(**base)


class StubModel:
    def __init__(self, reply: CreditResponse | None = None, fail: bool = False):
        self.reply = reply or response()
        self.fail = fail
        self.seen: dict | None = None

    def generate_credit_response(self, question, factsheet, evidence):
        if self.fail:
            raise RuntimeError("connection refused to 127.0.0.1:9905")
        self.seen = {"question": question, "factsheet": factsheet, "evidence": evidence}
        return self.reply


class AnalyseRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_fixture()

    def setUp(self) -> None:
        ds_role = data_service.settings.service_role
        data_service.settings.service_role = "data_service"
        ds_client = TestClient(data_service.app)

        def fake_rows(request):
            saved = data_service.settings.service_role
            data_service.settings.service_role = "data_service"
            try:
                return ds_client.post(
                    "/internal/v1/query/execute",
                    headers={"x-service-token": data_service.settings.service_token},
                    json=request.query_plan.model_dump(mode="json")).json()
            finally:
                data_service.settings.service_role = saved

        data_service.settings.service_role = ds_role
        self._rows, api._request_rows = api._request_rows, fake_rows
        self._model = api.model_client
        self._retriever = api.retriever

        index = InMemoryPolicyIndex()
        index.upsert([sama_chunk()], collection="policy_sama")
        api.retriever = PolicyRetriever(RetrievalPolicy(POLICY), index)
        self.client = TestClient(api.app)

    def tearDown(self) -> None:
        api._request_rows = self._rows
        api.model_client = self._model
        api.retriever = self._retriever

    def post(self, **overrides):
        body = {"user_text": "significant increase in credit risk",
                "query_plan": plan().model_dump(mode="json")}
        body.update(overrides)
        return self.client.post("/v1/analyse", json=body)

    def test_full_path_returns_factsheet_evidence_and_a_checked_answer(self) -> None:
        api.model_client = StubModel()
        r = self.post()
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["answer_status"], "ANSWERED")
        self.assertEqual(body["factsheet"]["obligor_id"], "OBL-0008")
        self.assertTrue(body["evidence"])
        self.assertTrue(body["output_guardrail"]["passed"])

    def test_the_model_sees_the_factsheet_not_the_rows(self) -> None:
        stub = StubModel()
        api.model_client = stub
        self.post()
        self.assertIn("calculated_metrics", stub.seen["factsheet"])
        self.assertNotIn("rows", stub.seen["factsheet"])

    def test_uncited_claim_is_blocked_even_though_the_model_answered(self) -> None:
        api.model_client = StubModel(response(
            facts=[SupportedClaim(statement="SICR threshold is 30 days.",
                                  evidence_ids=["SAMA-DOC-99#1.1"])]))
        body = self.post().json()
        self.assertFalse(body["output_guardrail"]["passed"])
        self.assertFalse(body["action_control"]["released"])
        self.assertEqual(body["action_control"]["release"], "blocked")

    def test_no_evidence_means_abstention_not_an_answer(self) -> None:
        api.retriever = PolicyRetriever(RetrievalPolicy(POLICY), InMemoryPolicyIndex())
        api.model_client = StubModel()
        body = self.post().json()
        self.assertEqual(body["answer_status"], "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(body["response"])
        self.assertFalse(body["action_control"]["released"])

    def test_model_failure_does_not_leak_the_host(self) -> None:
        api.model_client = StubModel(fail=True)
        r = self.post()
        self.assertEqual(r.status_code, 502)
        self.assertNotIn("9905", r.json()["detail"])
        self.assertNotIn("127.0.0.1", r.json()["detail"])

    def test_unknown_role_cannot_retrieve(self) -> None:
        api.model_client = StubModel()
        r = self.post(reviewer_role="intern")
        self.assertEqual(r.status_code, 403)


class ActionControlTests(unittest.TestCase):
    def test_autonomous_decision_language_is_prohibited(self) -> None:
        tier, reasons = classify(
            response(recommendation="We hereby approve the facility at the requested limit."),
            "credit_deterioration")
        self.assertEqual(tier, "prohibited")
        self.assertTrue(any("prohibited_action" in r for r in reasons))
        self.assertEqual(gate(response(
            recommendation="We hereby approve the facility."), "credit_deterioration"
        )["release"], "blocked")

    def test_staging_conclusions_need_senior_approval(self) -> None:
        tier, _ = classify(
            response(executive_summary="We recommend stage migration to stage 3."),
            "credit_deterioration")
        self.assertEqual(tier, "high")
        self.assertEqual(gate(response(
            executive_summary="Recommend stage migration."), "credit_deterioration"
        )["release"], "senior_credit_approval_required")

    def test_a_plain_factsheet_can_release_automatically(self) -> None:
        # No recommendation and no material conclusion. Note the default fixture mentions
        # SICR, which correctly forces high tier, so this case states its own facts.
        control = gate(response(
            executive_summary="Utilisation rose from 62% to 80% over twelve months.",
            facts=[SupportedClaim(statement="Utilisation is 80%.",
                                  evidence_ids=["SAMA-DOC-4#7.2"])],
            inferences=[], recommendation="", human_approval_required=False), "factsheet")
        self.assertEqual(control["risk_tier"], "low")
        self.assertTrue(control["released"])

    def test_any_recommendation_lifts_a_low_tier_output(self) -> None:
        control = gate(response(
            executive_summary="Utilisation rose over twelve months.",
            facts=[SupportedClaim(statement="Utilisation is 80%.",
                                  evidence_ids=["SAMA-DOC-4#7.2"])],
            inferences=[], recommendation="Increase monitoring frequency.",
            human_approval_required=False), "factsheet")
        self.assertEqual(control["risk_tier"], "medium")
        self.assertFalse(control["released"])

    def test_mentioning_sicr_anywhere_forces_senior_approval(self) -> None:
        # The default fixture cites a SICR clause, which is exactly the kind of material
        # conclusion a human must own.
        self.assertEqual(classify(response(), "credit_deterioration")[0], "high")

    def test_escalation_requested_by_the_model_is_honoured(self) -> None:
        tier, reasons = classify(
            response(answer_status=AnswerStatus.ESCALATE), "credit_deterioration")
        self.assertEqual(tier, "high")
        self.assertIn("model_requested_escalation", reasons)


if __name__ == "__main__":
    unittest.main()
