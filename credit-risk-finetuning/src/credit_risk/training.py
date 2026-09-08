from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def build_command(config: Path) -> list[str]:
    return ["python", "-m", "mlx_lm.lora", "--config", str(config)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLX-LM QLoRA from a reviewed config")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--execute", action="store_true", help="Execute instead of printing command")
    args = parser.parse_args()
    command = build_command(args.config)
    if args.execute:
        subprocess.run(command, check=True)
    else:
        print(" ".join(command))


if __name__ == "__main__":
    main()
