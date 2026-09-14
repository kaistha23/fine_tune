"""Opt-in Metal smoke: tiny random quantized model, no downloads or production data."""
import json
import os
from dataclasses import asdict

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_MLX_TRAINING_SMOKE") != "1",
                                reason="Opt-in native Metal training smoke")


def test_native_quantized_lora_and_actual_mask(tmp_path):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.llama import Model, ModelArgs
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from credit_risk.tokenization import configure_non_thinking, verify_training_tokens
    from credit_risk.training import execute_training

    mx.random.seed(42)
    base = tmp_path / "tiny-base"
    base.mkdir()
    args = ModelArgs(model_type="llama", hidden_size=64, num_hidden_layers=1,
                     intermediate_size=128, num_attention_heads=2, rms_norm_eps=1e-5,
                     vocab_size=32, max_position_embeddings=128, head_dim=32)
    model = Model(args)
    nn.quantize(model, group_size=32, bits=4)
    mx.save_safetensors(str(base / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (base / "config.json").write_text(json.dumps({**{k: v for k, v in asdict(args).items() if v is not None},
        "quantization": {"group_size": 32, "bits": 4}, "eos_token_id": 1}))
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "user": 2, "assistant": 3,
                                    "stage": 4, "one": 5, "system": 6}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>")
    tok.chat_template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% else %}{{ eos_token }}{% endif %}"
    tok.save_pretrained(base)
    turns = [{"role": "user", "content": "stage"}, {"role": "assistant", "content": "one"}]
    full, prefix = verify_training_tokens(configure_non_thinking(tok), turns)
    assert len(full) > len(prefix) > 0
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "valid"):
        (data / (split + ".jsonl")).write_text((json.dumps({"messages": turns}) + "\n") * 4)
    output = tmp_path / "candidate"
    execute_training({"model": str(base), "adapter_path": str(output), "data": str(data),
                      "iters": 8, "batch_size": 1, "grad_accumulation_steps": 2,
                      "num_layers": 1, "fine_tune_type": "lora", "mask_prompt": True,
                      "steps_per_eval": 2, "steps_per_report": 2, "save_every": 100,
                      "val_batches": 1, "max_seq_length": 32, "learning_rate": .01,
                      "early_stopping_patience": 10,
                      "lora_parameters": {"rank": 2, "scale": 2., "dropout": 0.}},
                     {"purpose": "test-only random tiny model"})
    meta = json.loads((output / "completion.json").read_text())
    assert meta["status"] == "completed" and meta["optimizer_updates"] == 4
    assert meta["baseline_loss"] is not None
    if meta["selected_checkpoint"]:
        assert meta["selected_loss"] < meta["baseline_loss"] and meta["best_optimizer_updates"] > 0
    weights = mx.load(str(output / "adapters.safetensors"))
    assert any(bool(mx.any(w != 0)) for name, w in weights.items() if name.endswith("lora_b"))
