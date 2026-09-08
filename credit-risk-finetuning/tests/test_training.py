"""Training and fusing commands (findings H2, H9)."""
from pathlib import Path
import unittest

import yaml

from credit_risk.training import build_fuse_command, build_train_command


CONFIG = Path(__file__).parents[1] / "configs" / "training.yaml"


class TrainingConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_lora_parameters_match_the_plans(self) -> None:
        # Both plans specify rank 16 / alpha 32 / dropout 0.05. The original config had
        # no lora_parameters block at all.
        lora = self.config["lora_parameters"]
        self.assertEqual(lora["rank"], 16)
        self.assertEqual(lora["scale"], 2.0)  # alpha 32 / rank 16
        self.assertEqual(lora["dropout"], 0.05)

    def test_prompt_masking_is_enabled(self) -> None:
        self.assertTrue(self.config["mask_prompt"])

    def test_memory_controls_are_set_for_a_64gb_mac(self) -> None:
        self.assertTrue(self.config["grad_checkpoint"])
        self.assertEqual(self.config["batch_size"], 1)
        self.assertEqual(self.config["max_seq_length"], 2048)


class CommandTests(unittest.TestCase):
    def test_train_command(self) -> None:
        command = build_train_command(CONFIG)
        self.assertEqual(command[:4], ["python", "-m", "mlx_lm.lora", "--config"])

    def test_fuse_command_targets_a_servable_directory(self) -> None:
        # oMLX cannot load adapters, so an unfused adapter can never be served.
        command = build_fuse_command(CONFIG, Path("/tmp/fused"))
        self.assertIn("mlx_lm.fuse", command)
        self.assertIn("--adapter-path", command)
        self.assertIn("--save-path", command)


if __name__ == "__main__":
    unittest.main()
