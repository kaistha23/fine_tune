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
        self.assertTrue(all("curated" not in v for v in api.get("volumes", [])))

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
            name for name, service in self.compose["services"].items() if service.get("ports")
        }
        self.assertEqual(published, {"api"})

    def test_data_service_is_only_on_the_internal_network(self) -> None:
        self.assertEqual(self.compose["services"]["data-service"]["networks"], ["restricted-data"])


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


class PinnedVersionTests(unittest.TestCase):
    """compose.yaml and settings.py both pin the config versions, in different files.

    Bumping the registry without bumping compose makes every container fail to start,
    because SchemaRegistry refuses a version it was not pinned to. That is the control
    working, but it should fail here rather than at deploy time.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))

    def _environment(self) -> dict:
        merged = {}
        for service in self.compose["services"].values():
            merged.update(service.get("environment") or {})
        return merged

    def test_the_compose_registry_pin_matches_the_registry_file(self) -> None:
        registry = yaml.safe_load(
            (Path(__file__).parents[1] / "configs" / "schema_registry.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            str(self._environment()["CR_SCHEMA_REGISTRY_VERSION"]), str(registry["version"])
        )

    def test_the_compose_policy_pin_matches_the_policy_file(self) -> None:
        policy = yaml.safe_load(
            (Path(__file__).parents[1] / "configs" / "architecture_policy.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            str(self._environment()["CR_ARCHITECTURE_POLICY_VERSION"]), str(policy["version"])
        )

    def test_settings_pin_the_same_versions_as_compose(self) -> None:
        from credit_risk.settings import Settings

        defaults = Settings()
        environment = self._environment()
        self.assertEqual(
            defaults.schema_registry_version, str(environment["CR_SCHEMA_REGISTRY_VERSION"])
        )
        self.assertEqual(
            defaults.architecture_policy_version, str(environment["CR_ARCHITECTURE_POLICY_VERSION"])
        )

    def test_no_credential_is_a_literal_in_compose(self) -> None:
        # Tokens and keys must come from the host environment, never from the file.
        for name in ("CR_SERVICE_TOKEN", "CR_OMLX_API_KEY"):
            with self.subTest(variable=name):
                value = str(self._environment().get(name, ""))
                if value:
                    self.assertTrue(value.startswith("${"), f"{name} is a literal")

    def test_missing_embedding_identity_does_not_block_compose_lifecycle_commands(self) -> None:
        api_environment = self.compose["services"]["api"]["environment"]
        self.assertEqual(api_environment["CR_EMBEDDING_MODEL"], "${CR_EMBEDDING_MODEL:-}")
        self.assertEqual(api_environment["CR_EMBEDDING_REVISION"], "${CR_EMBEDDING_REVISION:-}")
