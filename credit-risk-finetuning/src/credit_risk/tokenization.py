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

    # Every layer needs the default. MLX-LM's TokenizerWrapper fills in
    # enable_thinking=has_thinking (True for Qwen) before delegating, so patching only the
    # inner HF tokenizer is silently overridden and prompts end in an open <think> block.
    # The wrapper forwards setattr to the HF tokenizer, so each patch is written straight
    # into its own owner with object.__setattr__; the captured original is that owner's
    # own bound method, so wrapper -> inner never loops back to the wrapper.
    layers = [tokenizer]
    while callable(getattr(getattr(layers[-1], "_tokenizer", None), "apply_chat_template", None)):
        layers.append(layers[-1]._tokenizer)
    for target in layers:
        if getattr(target, "_credit_risk_non_thinking", False):
            continue
        original = target.apply_chat_template

        def apply_chat_template(*args, _original=original, **kwargs):
            kwargs.setdefault("enable_thinking", False)
            return _original(*args, **kwargs)

        object.__setattr__(target, "apply_chat_template", wraps(original)(apply_chat_template))
        object.__setattr__(target, "_credit_risk_non_thinking", True)
    return tokenizer


def ensure_non_thinking(tokenizer):
    """Fail before generation or training if the template still opens a thinking block.

    An open block makes Qwen reason in prose until max_tokens, so every structured
    answer is invalid; that must stop the job, not become a 0% scorecard.
    """
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "ping"}], tokenize=False, add_generation_prompt=True
    )
    if isinstance(probe, str) and "<think>" in probe:
        if "</think>" not in probe[probe.rfind("<think>") :]:
            raise ValueError("Chat template left a thinking block open; enable_thinking=False was not applied")
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
