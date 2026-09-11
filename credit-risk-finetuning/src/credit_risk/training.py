"""Preflight, versioned candidate training and protected adapter export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from credit_risk.dataset import verify_manifest
from credit_risk.review_store import digest
from credit_risk.tokenization import CHAT_TEMPLATE_MODE, configure_non_thinking

DEFAULT_CONFIG = Path("configs/training.yaml")
DEFAULT_MODEL_DIR = Path("models/candidates")


def build_train_command(config):
    return [
        sys.executable,
        "-m",
        "credit_risk.training",
        "train",
        "--config",
        str(config),
        "--execute",
    ]


def build_fuse_command(config, save_path):
    cfg = yaml.safe_load(Path(config).read_text())
    return [
        sys.executable,
        "-m",
        "mlx_lm.fuse",
        "--model",
        str(cfg["model"]),
        "--adapter-path",
        str(cfg["adapter_path"]),
        "--save-path",
        str(save_path),
    ]


def iterations_for(examples, batch, accumulation, epochs=2):
    if min(examples, batch, accumulation, epochs) <= 0:
        raise ValueError("Positive training sizes required")
    return math.ceil(math.ceil(examples * epochs / batch) / accumulation) * accumulation


def preflight(config, tokenizer=None):
    cfg = yaml.safe_load(Path(config).read_text())
    manifest = verify_manifest(Path(cfg["data"]))
    if not manifest["counts"]["train"] or not manifest["counts"]["valid"]:
        raise ValueError("Nonempty training and validation splits required")
    if cfg.get("fine_tune_type") != "lora" or not cfg.get("mask_prompt"):
        raise ValueError("Quantized-base LoRA and assistant-only loss required")
    adapter = Path(cfg["adapter_path"])
    if adapter.exists():
        raise ValueError("Candidate adapter directory already exists")
    if not adapter.resolve().is_relative_to(Path("adapters/candidates").resolve()):
        raise ValueError("Adapter must be inside candidates directory")
    model = Path(cfg["model"])
    if not model.is_dir():
        from huggingface_hub import snapshot_download

        model = Path(snapshot_download(cfg["model"], local_files_only=True))
    model_config = json.loads((model / "config.json").read_text())
    quant = model_config.get("quantization") or model_config.get("quantization_config")
    if not quant or quant.get("bits") != 4:
        raise ValueError("A verified 4-bit base is required")
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model, local_files_only=True, trust_remote_code=False
        )
    configure_non_thinking(tokenizer)
    maximum = cfg["max_seq_length"]
    lengths = []
    assistant = []
    for split in ("train", "valid"):
        for line in (Path(cfg["data"]) / (split + ".jsonl")).read_text().splitlines():
            turns = json.loads(line)["messages"]
            full = tokenizer.apply_chat_template(turns, tokenize=True, return_dict=False)
            prefix = tokenizer.apply_chat_template(
                turns[:-1], tokenize=True, add_generation_prompt=True, return_dict=False
            )
            if len(full) > maximum or len(full) <= len(prefix):
                raise ValueError("Truncated or empty assistant target")
            lengths.append(len(full))
            assistant.append(len(full) - len(prefix))
    cfg["model"] = str(model.resolve())
    cfg["iters"] = iterations_for(
        manifest["counts"]["train"],
        cfg["batch_size"],
        cfg["grad_accumulation_steps"],
        cfg.pop("epochs", 2),
    )
    spike = cfg.pop("spike_iters", None)
    if spike:
        if spike < 20 or spike > 50 or spike % cfg["grad_accumulation_steps"]:
            raise ValueError("Spike requires 20–50 aligned micro-batches")
        cfg["iters"] = spike
    metadata = {
        "dataset_manifest": manifest["manifest_hash"],
        "base_snapshot": model.name,
        "model_config_hash": digest(model_config),
        "quantization": quant,
        "chat_template_hash": digest(tokenizer.chat_template),
        "chat_template_mode": CHAT_TEMPLATE_MODE,
        "max_tokens": max(lengths),
        "min_assistant_tokens": min(assistant),
        "micro_batches": cfg["iters"],
        "optimizer_updates": cfg["iters"] // cfg["grad_accumulation_steps"],
        "promotable": False,
    }
    return cfg, metadata


class EarlyStop(Exception):
    pass


def execute_training(cfg, metadata):
    import types

    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import lora
    from mlx_lm.tuner.datasets import load_dataset
    from mlx_lm.utils import load

    output = Path(cfg["adapter_path"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_manifest.json").write_text(json.dumps(metadata, indent=2))
    args = vars(lora.build_parser().parse_args([]))
    args.update(cfg)
    for key, value in lora.CONFIG_DEFAULTS.items():
        if args.get(key) is None:
            args[key] = value
    args = types.SimpleNamespace(**args)
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": False})
    configure_non_thinking(tokenizer)
    train, valid, _ = load_dataset(args, tokenizer)

    class Callback:
        best = float("inf")
        bad = 0
        last_iteration = 0
        best_iteration = None

        def on_train_loss_report(self, info):
            if not math.isfinite(info["train_loss"]):
                raise FloatingPointError("Non-finite training loss")
            self.last_iteration = info["iteration"]
            info = {
                **info,
                "optimizer_updates": self.last_iteration // cfg["grad_accumulation_steps"],
            }
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps({"train": info}) + "\n")

        def on_val_loss_report(self, info):
            if not math.isfinite(info["val_loss"]):
                raise FloatingPointError("Non-finite validation loss")
            self.last_iteration = max(self.last_iteration, info["iteration"])
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps({"validation": info}) + "\n")
            min_delta = cfg.get("early_stopping_min_delta", 0.0)
            if info["val_loss"] < self.best - min_delta:
                self.best = info["val_loss"]
                self.bad = 0
                self.best_iteration = info["iteration"]
                if info["iteration"] >= 0:
                    mx.save_safetensors(
                        str(output / "best_adapters.safetensors"),
                        dict(tree_flatten(model.trainable_parameters())),
                    )
            else:
                self.bad += 1
                if self.bad >= cfg.get("early_stopping_patience", 2):
                    raise EarlyStop()

    callback = Callback()
    status = "failed"
    try:
        lora.train_model(args, model, train, valid, callback)
        from mlx_lm.tuner.datasets import CacheDataset
        from mlx_lm.tuner.trainer import evaluate

        final_loss = evaluate(
            model,
            CacheDataset(valid),
            args.batch_size,
            args.val_batches,
            max_seq_length=args.max_seq_length,
        )
        try:
            callback.on_val_loss_report(
                {"iteration": cfg["iters"], "val_loss": final_loss, "phase": "final_checkpoint"}
            )
        except EarlyStop:
            pass  # The complete budget has already run; final validation is still recorded.
        status = "completed"
    except EarlyStop:
        status = "early_stopped"
    finally:
        final_checkpoint = output / "adapters.safetensors"
        best_checkpoint = output / "best_adapters.safetensors"
        mx.save_safetensors(
            str(final_checkpoint), dict(tree_flatten(model.trainable_parameters()))
        )
        if status in {"completed", "early_stopped"} and not best_checkpoint.exists():
            shutil.copyfile(final_checkpoint, best_checkpoint)
            callback.best_iteration = callback.last_iteration
        (output / "completion.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "peak_mlx_bytes": mx.get_peak_memory(),
                    "promotable": False,
                    "completed_micro_batches": cfg["iters"]
                    if status == "completed"
                    else callback.last_iteration,
                    "optimizer_updates": (
                        cfg["iters"] if status == "completed" else callback.last_iteration
                    )
                    // cfg["grad_accumulation_steps"],
                    "best_iteration": callback.best_iteration,
                    "selected_checkpoint": "best_adapters.safetensors"
                    if best_checkpoint.exists()
                    else None,
                }
            )
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", nargs="?", default="train", choices=["train", "fuse"])
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--save-path", type=Path)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--checkpoint", choices=["best", "final"], default="best")
    a = p.parse_args()
    if a.command == "train":
        cfg, meta = preflight(a.config)
        if a.execute:
            execute_training(cfg, meta)
        else:
            print(json.dumps(meta, indent=2))
    else:
        cfg = yaml.safe_load(a.config.read_text())
        dest = a.save_path or DEFAULT_MODEL_DIR / Path(cfg["adapter_path"]).name
        if dest.exists() or not dest.resolve().is_relative_to(DEFAULT_MODEL_DIR.resolve()):
            raise ValueError("New candidate export directory required")
        adapter = Path(cfg["adapter_path"])
        if not (adapter / "completion.json").is_file():
            raise ValueError("Completed candidate required")
        completion = json.loads((adapter / "completion.json").read_text())
        if completion["status"] not in {"completed", "early_stopped"}:
            raise ValueError("Successful training required")
        pinned = json.loads((adapter / "adapter_config.json").read_text())
        if not Path(pinned["model"]).is_dir():
            raise ValueError("Pinned local base required")
        command = build_fuse_command(a.config, dest)
        command[command.index("--model") + 1] = pinned["model"]
        name = "best_adapters.safetensors" if a.checkpoint == "best" else "adapters.safetensors"
        if not (adapter / name).is_file():
            raise ValueError("Selected checkpoint unavailable")
        if a.execute:
            import tempfile

            with tempfile.TemporaryDirectory(prefix="credit-checkpoint-") as staging:
                selected = Path(staging)
                shutil.copyfile(adapter / "adapter_config.json", selected / "adapter_config.json")
                shutil.copyfile(adapter / name, selected / "adapters.safetensors")
                command[command.index("--adapter-path") + 1] = str(selected)
                subprocess.run(command, check=True)
            (dest / "export_manifest.json").write_text(
                json.dumps(
                    {
                        "base": pinned["model"],
                        "checkpoint": name,
                        "checkpoint_sha256": hashlib.sha256(
                            (adapter / name).read_bytes()
                        ).hexdigest(),
                    }
                )
            )
        else:
            print(" ".join(command))


if __name__ == "__main__":
    main()
