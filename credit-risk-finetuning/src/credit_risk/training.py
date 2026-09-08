"""Drive MLX-LM training and fusing from a reviewed config (findings H2, H9).

Two commands, because a trained adapter is not yet servable:

    train   run mlx_lm.lora against configs/training.yaml
    fuse    merge the adapter into a standalone MLX model directory

The fuse step is not optional. oMLX discovers whole model directories and has no
adapter-loading flag, so a bare adapter cannot be served and therefore cannot be
evaluated. mlx_lm.server can hot-load an adapter, which is faster while iterating; fuse
once the adapter is worth serving.

Both commands print by default and only run with --execute, so a config is always read
before it is trusted.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("configs/training.yaml")
# oMLX's default model directory. The directory name becomes the served model id.
DEFAULT_MODEL_DIR = Path.home() / ".omlx" / "models"


def build_train_command(config: Path) -> list[str]:
    return ["python", "-m", "mlx_lm.lora", "--config", str(config)]


def build_fuse_command(config: Path, save_path: Path) -> list[str]:
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    return [
        "python", "-m", "mlx_lm.fuse",
        "--model", str(settings["model"]),
        "--adapter-path", str(settings["adapter_path"]),
        "--save-path", str(save_path),
    ]


def _run(command: list[str], execute: bool) -> None:
    if execute:
        subprocess.run(command, check=True)
    else:
        print(" ".join(command))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="train", choices=["train", "fuse"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--save-path", type=Path, default=None,
        help="Fused model directory. Defaults to ~/.omlx/models/<adapter name>.")
    parser.add_argument("--execute", action="store_true",
                        help="Execute instead of printing the command")
    args = parser.parse_args()

    if args.command == "train":
        _run(build_train_command(args.config), args.execute)
        return

    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    save_path = args.save_path or DEFAULT_MODEL_DIR / Path(settings["adapter_path"]).name
    _run(build_fuse_command(args.config, save_path), args.execute)


if __name__ == "__main__":
    main()
