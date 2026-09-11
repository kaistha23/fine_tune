from pathlib import Path
import unittest

import yaml


COMPOSE = Path(__file__).parents[1] / "compose.yaml"


class ComposeArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))

    def test_api_has_no_database_volume(self) -> None:
        api = self.compose["services"]["api"]
        self.assertFalse(api.get("volumes"))

    def test_only_data_service_mounts_curated_data_read_only(self) -> None:
        data_service = self.compose["services"]["data-service"]
        self.assertIn("./data/curated:/data:ro", data_service["volumes"])
        for name, service in self.compose["services"].items():
            if name != "data-service":
                self.assertNotIn("./data/curated:/data:ro", service.get("volumes", []))

    def test_restricted_data_network_is_internal(self) -> None:
        self.assertTrue(self.compose["networks"]["restricted-data"]["internal"])

    def test_model_training_and_omlx_are_not_containerised(self) -> None:
        services = set(self.compose["services"])
        self.assertNotIn("omlx", services)
        self.assertNotIn("training", services)


if __name__ == "__main__":
    unittest.main()


