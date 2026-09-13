"""Exercise execute_training with a simulated MLX boundary, including saved weights."""
import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from credit_risk.training import execute_training
from credit_risk.tokenization import verify_training_tokens


@pytest.fixture
def backend(monkeypatch):
    state = {"weight": 0, "events": [(0, 1.0)], "final_loss": 1.1}
    model = types.SimpleNamespace(trainable_parameters=lambda: {"weight": state["weight"]})
    tok = types.SimpleNamespace(apply_chat_template=lambda *a, **kw: [1])
    def train(args, model, train, valid, callback):
        for iteration, loss in state["events"]:
            state["weight"] = iteration
            callback.on_val_loss_report({"iteration": iteration, "val_loss": loss})
        if state.get("error"):
            raise RuntimeError("backend failed")
        state["weight"] = args.iters
        callback.on_train_loss_report({"iteration": args.iters, "train_loss": state.get("train_loss", .8)})
    def module(name, **attrs):
        item = types.ModuleType(name)
        item.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, item)
        return item
    core = module("mlx.core", save_safetensors=lambda path, weights: Path(path).write_text(json.dumps(weights)),
                  get_peak_memory=lambda: 100)
    module("mlx", core=core)
    module("mlx.utils", tree_flatten=lambda d: list(d.items()))
    lora = module("mlx_lm.lora", build_parser=argparse.ArgumentParser, CONFIG_DEFAULTS={}, train_model=train)
    module("mlx_lm", lora=lora)
    module("mlx_lm.tuner")
    module("mlx_lm.tuner.datasets", load_dataset=lambda *a: ([1], [2], []), CacheDataset=lambda x: x)
    module("mlx_lm.tuner.trainer", evaluate=lambda *a, **k: state["final_loss"])
    module("mlx_lm.utils", load=lambda *a, **k: (model, tok))
    return state


def run(tmp_path):
    cfg = {"adapter_path": str(tmp_path / "candidate"), "model": "fixture", "iters": 8,
           "grad_accumulation_steps": 2, "batch_size": 1, "val_batches": 1,
           "max_seq_length": 100, "early_stopping_patience": 2, "early_stopping_min_delta": .05}
    execute_training(cfg, {})
    return tmp_path / "candidate", json.loads((tmp_path / "candidate/completion.json").read_text())


def test_no_improvement_never_manufactures_best(backend, tmp_path):
    path, meta = run(tmp_path)
    assert not (path / "best_adapters.safetensors").exists()
    assert (path / "adapters.safetensors").is_file()
    assert meta["selected_checkpoint"] is None and meta["baseline_loss"] == 1
    assert meta["best_iteration"] is None


def test_improved_weights_are_pinned_not_final_weights(backend, tmp_path):
    backend.update(events=[(0, 1.), (2, .8)], final_loss=.9)
    path, meta = run(tmp_path)
    assert json.loads((path / "best_adapters.safetensors").read_text())["weight"] == 2
    assert json.loads((path / "adapters.safetensors").read_text())["weight"] == 8
    assert meta["best_optimizer_updates"] == 1
    assert meta["selected_loss"] == .8
    assert meta["checkpoint_sha256"] == hashlib.sha256((path / "best_adapters.safetensors").read_bytes()).hexdigest()


def test_no_optimizer_update_cannot_save_best(backend, tmp_path):
    backend.update(events=[(0, 1.), (1, .2)], final_loss=1.1)
    path, meta = run(tmp_path)
    assert not (path / "best_adapters.safetensors").exists()
    assert meta["best_iteration"] is None


def test_early_stop_without_improvement_has_no_best(backend, tmp_path):
    backend.update(events=[(0, 1.), (2, 1.2), (4, 1.1)])
    path, meta = run(tmp_path)
    assert meta["status"] == "early_stopped"
    assert meta["optimizer_updates"] == 2
    assert not (path / "best_adapters.safetensors").exists()


@pytest.mark.parametrize("updates,error", [({"events": [(0, float("nan"))]}, FloatingPointError),
                                          ({"train_loss": float("inf")}, FloatingPointError),
                                          ({"error": True}, RuntimeError)])
def test_failed_run_cannot_be_exported(backend, tmp_path, updates, error):
    backend.update(updates)
    with pytest.raises(error):
        run(tmp_path)
    meta = json.loads((tmp_path / "candidate/completion.json").read_text())
    assert meta["status"] == "failed"


def test_mask_boundary_checks_actual_dataset():
    class Tokenizer:
        def apply_chat_template(self, turns, **kwargs):
            return [1, 2] if kwargs.get("add_generation_prompt") else [1, 2, 3]
    class Dataset:
        def __init__(self, *args, **kwargs):
            pass
        def process(self, record):
            return [1, 2, 3], 1
    with pytest.raises(ValueError, match="mask mismatch"):
        verify_training_tokens(Tokenizer(), [{"role": "user"}, {"role": "assistant"}], Dataset)


def test_forwarding_tokenizer_wrapper_does_not_recurse():
    from credit_risk.tokenization import configure_non_thinking
    class Inner:
        def apply_chat_template(self, *args, **kwargs):
            return kwargs["enable_thinking"]
    class Wrapper:
        def __init__(self):
            object.__setattr__(self, "_tokenizer", Inner())
        def __getattr__(self, key):
            return getattr(self._tokenizer, key)
        def __setattr__(self, key, value):
            setattr(self._tokenizer, key, value)
        def apply_chat_template(self, *args, **kwargs):
            return self._tokenizer.apply_chat_template(*args, **kwargs)
    wrapped = Wrapper()
    assert configure_non_thinking(wrapped) is wrapped
    assert configure_non_thinking(wrapped).apply_chat_template([]) is False
