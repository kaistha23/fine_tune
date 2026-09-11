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

    if getattr(tokenizer, "_credit_risk_non_thinking", False):
        return tokenizer
    original = tokenizer.apply_chat_template

    @wraps(original)
    def apply_chat_template(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return original(*args, **kwargs)

    tokenizer.apply_chat_template = apply_chat_template
    tokenizer._credit_risk_non_thinking = True
    return tokenizer
