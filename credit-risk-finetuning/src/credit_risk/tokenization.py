"""One recorded chat-template policy for training and local inference."""

from __future__ import annotations

from functools import wraps

CHAT_TEMPLATE_MODE = {"enable_thinking": False}


def configure_non_thinking(tokenizer):
    """Make Qwen's direct-response mode the tokenizer default.

    MLX-LM's ``ChatDataset`` calls ``apply_chat_template`` itself and does not expose
    arbitrary template arguments. Wrapping the tokenizer keeps its internal training
    path, preflight token counts and evaluation prompts on the same explicit policy.
    """

    # MLX's TokenizerWrapper forwards setattr to the underlying HF tokenizer.
    # Capturing a wrapper-bound method and assigning it through that wrapper creates
    # recursion: wrapper -> patched HF method -> wrapper. Patch the actual owner.
    target = tokenizer
    while callable(getattr(getattr(target, "_tokenizer", None), "apply_chat_template", None)):
        target = target._tokenizer
    if getattr(target, "_credit_risk_non_thinking", False):
        return tokenizer
    original = target.apply_chat_template

    @wraps(original)
    def apply_chat_template(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return original(*args, **kwargs)

    target.apply_chat_template = apply_chat_template
    target._credit_risk_non_thinking = True
    return tokenizer


def verify_training_tokens(tokenizer, turns, dataset_class=None):
    """Verify the actual trainer mask, not merely a template length estimate."""
    if dataset_class is None:
        from mlx_lm.tuner.datasets import ChatDataset
        dataset_class = ChatDataset
    full = tokenizer.apply_chat_template(turns, tokenize=True, return_dict=False)
    prefix = tokenizer.apply_chat_template(
        turns[:-1], tokenize=True, add_generation_prompt=True, return_dict=False
    )
    processed = dataset_class(
        [{"messages": turns}], tokenizer, chat_key="messages", mask_prompt=True
    ).process({"messages": turns})
    if list(processed[0]) != list(full) or processed[1] != len(prefix):
        raise ValueError("Trainer tokenization or mask mismatch")
    if list(full[:len(prefix)]) != list(prefix):
        raise ValueError("Assistant prefix differs from full training sequence")
    return full, prefix
