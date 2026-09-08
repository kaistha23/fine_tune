import unittest
from pathlib import Path

from credit_risk.architecture_policy import ArchitecturePolicy, ArchitecturePolicyError

POLICY = Path(__file__).parents[1] / "configs" / "architecture_policy.yaml"


class ArchitecturePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ArchitecturePolicy(POLICY, "1.2.0")

    def test_api_cannot_execute_sql(self) -> None:
        with self.assertRaises(ArchitecturePolicyError):
            self.policy.require("api_gateway", "execute_sql")

    def test_api_can_view_but_cannot_edit_and_execute_sql(self) -> None:
        self.policy.require("api_gateway", "view_parameterised_sql")
        with self.assertRaises(ArchitecturePolicyError):
            self.policy.require("api_gateway", "edit_and_execute_sql")

    def test_data_service_can_execute_only_read_only_sql(self) -> None:
        self.policy.require("data_service", "execute_read_only_sql")
        with self.assertRaises(ArchitecturePolicyError):
            self.policy.require("data_service", "mutate_database")

    def test_unknown_capability_is_denied(self) -> None:
        with self.assertRaises(ArchitecturePolicyError):
            self.policy.require("feedback_worker", "call_model")


if __name__ == "__main__":
    unittest.main()
