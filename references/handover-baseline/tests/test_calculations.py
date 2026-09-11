import unittest

from credit_risk.calculations import current_ratio, percentage_point_change, relative_change_pct


class CalculationTests(unittest.TestCase):
    def test_current_ratio(self) -> None:
        result = current_ratio({"current_assets": 120.0, "current_liabilities": 100.0})
        self.assertEqual(result.value, 1.2)
        self.assertEqual(result.validation_status, "valid")

    def test_zero_denominator_is_invalid(self) -> None:
        result = current_ratio({"current_assets": 120.0, "current_liabilities": 0.0})
        self.assertIsNone(result.value)
        self.assertEqual(result.validation_status, "invalid")

    def test_pd_changes_are_not_confused(self) -> None:
        self.assertEqual(percentage_point_change(0.05, 0.02), 3.0)
        self.assertEqual(relative_change_pct(0.05, 0.02), 150.0)


if __name__ == "__main__":
    unittest.main()

