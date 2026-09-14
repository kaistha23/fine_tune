"""Preflight, versioned candidate training and protected adapter export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from credit_risk.dataset import verify_manifest
from credit_risk.review_store import digest
from credit_risk.tokenization import (
    CHAT_TEMPLATE_MODE,
    configure_non_thinking,
    ensure_non_thinking,
    verify_training_tokens,
)

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
            full, prefix = verify_training_tokens(tokenizer, turns)
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
    output = Path(cfg["adapter_path"])
    if output.exists():
        raise ValueError("Candidate adapter directory already exists")
    try:
        _execute_training(cfg, metadata)
    except Exception as exc:
        if output.is_dir() and not (output / "completion.json").exists():
            (output / "completion.json").write_text(json.dumps({
                "status": "failed", "manifest_version": 2,
                "error": type(exc).__name__, "selected_checkpoint": None, "promotable": False,
            }))
        raise


def _execute_training(cfg, metadata):
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
    ensure_non_thinking(configure_non_thinking(tokenizer))
    train, valid, _ = load_dataset(args, tokenizer)

    class Callback:
        best = float("inf")
        bad = 0
        last_iteration = 0
        best_iteration = None
        baseline_loss = None
        selected_loss = None

        def on_train_loss_report(self, info):
            if not math.isfinite(info["train_loss"]):
                raise FloatingPointError("Non-finite training loss")
            self.last_iteration = info["iteration"]
            info = {
                **info,
                "optimizer_updates": self.last_iteration // cfg["grad_accumulation_steps"],
                "reported_at": datetime.now(UTC).isoformat(),
            }
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps({"train": info}) + "\n")

        def on_val_loss_report(self, info):
            if not math.isfinite(info["val_loss"]):
                raise FloatingPointError("Non-finite validation loss")
            self.last_iteration = max(self.last_iteration, info["iteration"])
            with (output / "metrics.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {"validation": {**info, "reported_at": datetime.now(UTC).isoformat()}}
                    )
                    + "\n"
                )
            iteration = info["iteration"]
            loss = info["val_loss"]
            if iteration == 0:
                if self.baseline_loss is not None:
                    raise ValueError("Duplicate baseline validation")
                self.baseline_loss = loss
                self.best = loss
                return
            if self.baseline_loss is None:
                raise ValueError("Baseline validation required before checkpoint selection")
            if iteration < cfg["grad_accumulation_steps"]:
                return
            min_delta = cfg.get("early_stopping_min_delta", 0.0)
            if loss < self.best - min_delta:
                self.best = loss
                self.selected_loss = loss
                self.bad = 0
                self.best_iteration = iteration
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
                    "manifest_version": 2,
                    "baseline_loss": callback.baseline_loss,
                    "selected_loss": callback.selected_loss,
                    "best_iteration": callback.best_iteration,
                    "best_optimizer_updates": (callback.best_iteration // cfg["grad_accumulation_steps"])
                    if callback.best_iteration is not None else None,
                    "checkpoint_sha256": hashlib.sha256(best_checkpoint.read_bytes()).hexdigest()
                    if best_checkpoint.exists() else None,
                    "final_checkpoint_sha256": hashlib.sha256(final_checkpoint.read_bytes()).hexdigest(),
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
        if a.checkpoint == "best" and (
            completion.get("manifest_version") != 2
            or not completion.get("best_optimizer_updates")
            or completion.get("selected_loss") is None
            or completion.get("baseline_loss") is None
            or completion["selected_loss"] >= completion["baseline_loss"]
        ):
            raise ValueError("No verified best checkpoint improved over baseline; select final explicitly")
        if not (adapter / name).is_file():
            raise ValueError("Selected checkpoint unavailable")
        expected_hash = completion.get(
            "checkpoint_sha256" if a.checkpoint == "best" else "final_checkpoint_sha256"
        )
        if completion.get("manifest_version") == 2 and not expected_hash:
            raise ValueError("Selected checkpoint has no recorded hash")
        if expected_hash and hashlib.sha256((adapter / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Checkpoint hash mismatch")
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
