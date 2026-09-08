import re
import tomllib
import unittest
from pathlib import Path

import yaml

COMPOSE = Path(__file__).parents[1] / "compose.yaml"
PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


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

    def test_every_service_drops_all_capabilities(self) -> None:
        # Qdrant was the one service without cap_drop or init (finding M6).
        for name, service in self.compose["services"].items():
            with self.subTest(service=name):
                self.assertEqual(service.get("cap_drop"), ["ALL"])
                self.assertIn("no-new-privileges:true", service.get("security_opt", []))

    def test_only_the_api_is_published_to_the_host(self) -> None:
        published = {
            name for name, service in self.compose["services"].items()
            if service.get("ports")
        }
        self.assertEqual(published, {"api"})

    def test_data_service_is_only_on_the_internal_network(self) -> None:
        self.assertEqual(self.compose["services"]["data-service"]["networks"],
                         ["restricted-data"])


if __name__ == "__main__":
    unittest.main()


class QdrantVersionPinTests(unittest.TestCase):
    """The client warns, and can misread new payload index types, when the server is on a
    different major/minor line. The two pins are declared in different files, so nothing
    stopped them drifting apart until this test."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
        cls.pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    def _image_line(self) -> tuple[int, int]:
        image = self.compose["services"]["qdrant"]["image"]
        match = re.search(r":v(\d+)\.(\d+)\.", image)
        self.assertIsNotNone(match, f"qdrant image must pin an exact version, got {image}")
        return int(match.group(1)), int(match.group(2))

    def _client_specifiers(self) -> list[str]:
        rag = self.pyproject["project"]["optional-dependencies"]["rag"]
        return [d for d in rag if d.startswith("qdrant-client")]

    def test_the_qdrant_image_is_pinned_to_an_exact_patch(self) -> None:
        self._image_line()

    def test_the_client_is_bounded_to_the_server_line(self) -> None:
        specifiers = self._client_specifiers()
        self.assertEqual(len(specifiers), 1)
        spec = specifiers[0]
        major, minor = self._image_line()
        self.assertIn(f">={major}.{minor}", spec)
        self.assertIn(f"<{major}.{minor + 1}", spec)
